"""Owned A/B servers over an unchanged resident checkpoint; no inference."""

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import select
import signal
import statistics
import subprocess
import sys
import time


@contextmanager
def owned_server(root, output, job):
    from cuattest.client import Client

    ready = output / f"{job['label']}-ready.json"
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        CUATTEST_CACHE=str(output / "cache" / job["label"]),
    )
    env.update(job.get("env", {}))
    command = [
        sys.executable,
        str(Path(__file__).with_name("server.py")),
        str(root),
        str(ready),
    ]
    if job.get("registered"):
        command.append("--registered")
    with (output / f"{job['label']}-server.log").open("x") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        pidfd = None
        try:
            deadline = time.monotonic() + 1800
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server failed: {job['label']} {process.returncode}"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup")
                if ready.exists():
                    try:
                        setup = json.loads(ready.read_text())
                        break
                    except json.JSONDecodeError:
                        # The child may still be writing its small readiness
                        # document. Existence alone is not a complete record.
                        pass
                time.sleep(0.1)
            pidfd = os.pidfd_open(setup["pid"])
            client = Client(f"http://127.0.0.1:{setup['port']}")
            info = client.info()
            assert all(
                d["pid"] == setup["pid"] and d["host_backend"] == "C++"
                for d in info["devices"]
            )
            yield client, info
        finally:
            if pidfd is not None:
                try:
                    signal.pidfd_send_signal(pidfd, signal.SIGINT)
                except ProcessLookupError:
                    pass
                if not select.select([pidfd], [], [], 30)[0]:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                    if not select.select([pidfd], [], [], 30)[0]:
                        raise RuntimeError(
                            "owned CUDA consumer death remains unconfirmed"
                        )
                os.close(pidfd)
            elif process.poll() is None:
                process.send_signal(signal.SIGINT)
            process.wait(timeout=60)


def pack_plan(specs, placement, alignment=256):
    """Explicit alignment holes are not model weights; hash only logical spans."""
    sizes = dict.fromkeys(placement.values(), 0)
    offsets = {}
    for spec in sorted(specs, key=lambda t: t.name):
        device = placement[spec.name]
        offset = (sizes[device] + alignment - 1) // alignment * alignment
        offsets[spec.name] = offset
        sizes[device] = offset + spec.nbytes
    return offsets, sizes


def load_packed(large, specs, files, placement, hash_threads):
    import torch
    from blake3 import blake3
    from safetensors import safe_open

    offsets, sizes = pack_plan(specs, placement)
    # Allocate at load time. Never hold an ordinary full model AND its packed
    # copy: that would exceed the near-capacity benchmark's VRAM budget.
    arenas = {
        d: torch.empty(n, dtype=torch.uint8, device=f"cuda:{d}")
        for d, n in sizes.items()
    }
    model = large.ResidentCheckpoint({})
    digests = {}
    loaded = 0
    last = time.monotonic()
    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(
                safe_open(str(path), framework="pt", device="cpu")
            )
            for name, path in files.items()
        }
        for spec in sorted(
            specs, key=lambda t: (-large.allocated_size(t.nbytes), t.name)
        ):
            cpu = readers[spec.shard].get_tensor(spec.name)
            digests[spec.name] = blake3(
                cpu.reshape(-1).view(torch.uint8).numpy(), max_threads=hash_threads
            ).digest()
            offset = offsets[spec.name]
            view = (
                arenas[placement[spec.name]][offset : offset + spec.nbytes]
                .view(cpu.dtype)
                .view(cpu.shape)
            )
            view.copy_(cpu)
            model.tensors[spec.name] = view
            loaded += spec.nbytes
            if time.monotonic() - last > 5:
                print(
                    f"packed and independently hashed {loaded / large.GIB:.1f} GiB",
                    flush=True,
                )
                last = time.monotonic()
    for device in sizes:
        torch.cuda.synchronize(device)
    # Check exact storage identity, shape/dtype and disjoint logical coverage;
    # sharing an arena is not permission to count aliases or holes as weights.
    for spec in specs:
        value = model.tensors[spec.name]
        arena = arenas[placement[spec.name]]
        assert value.is_contiguous() and tuple(value.shape) == spec.shape
        assert value.numel() * value.element_size() == spec.nbytes
        assert value.data_ptr() == arena.data_ptr() + offsets[spec.name]
        assert value.untyped_storage().data_ptr() == arena.data_ptr()
    root = blake3(
        len(specs).to_bytes(4, "little")
        + b"".join(digests[t.name] for t in sorted(specs, key=lambda t: t.name))
    ).hexdigest()
    return (
        model,
        root,
        {
            "arena_bytes": sizes,
            "alignment_padding_bytes": sum(sizes.values()) - loaded,
            "offsets": offsets,
        },
    )


def run_job(large, root, output, job, model, expected_root, specs, placement):
    from cuattest.ipc import (
        share_model,
        claim_ipc_refs,
        wire_refs,
        quarantine_ipc_refs,
        _complete_ipc_refs,
        assert_ipc_refs_immutable,
    )
    from cuattest.expect import verify_evidence

    refs = None
    keepalive = None
    released = False
    try:
        with owned_server(root, output, job) as (client, info):
            keys = large.describe_notary(info)
            result = dict(
                job,
                notary=info,
                placement=placement,
                tensor_count=len(specs),
                attested_bytes=sum(t.nbytes for t in specs),
                independent_model_root=expected_root,
            )
            if not job.get("registered"):
                samples = large.sample_model(
                    client,
                    model,
                    large.DEFAULT_MODEL.model_id,
                    job.get("runs", 30),
                    job.get("warmup_runs", 5),
                    keys,
                    expected_root=expected_root,
                )
                result["samples"] = [asdict(s) for s in samples]
            else:
                _, refs, keepalive = share_model(model)
                refs = claim_ipc_refs(refs)
                started = time.perf_counter()
                registration = client._rpc(
                    "/experiment/register", {"tensors": wire_refs(refs)}
                )
                result["register_seconds"] = time.perf_counter() - started
                result["unique_handles"] = registration["unique_handles"]
                samples = []
                for iteration in range(job.get("warmup_runs", 5) + job.get("runs", 30)):
                    assert_ipc_refs_immutable(refs)
                    started = time.perf_counter()
                    signed = client._rpc(
                        "/experiment/sign",
                        {
                            "token": registration["token"],
                            "model": large.DEFAULT_MODEL.model_id,
                        },
                    )
                    seconds = time.perf_counter() - started
                    assert_ipc_refs_immutable(refs)
                    verified = verify_evidence(signed, trusted_pubkeys=keys)
                    assert (
                        verified.model_root == expected_root
                        and verified.tensor_count == len(specs)
                    )
                    if iteration >= job.get("warmup_runs", 5):
                        samples.append(
                            dict(
                                seconds=seconds,
                                model_root=verified.model_root,
                                tensor_count=len(specs),
                                attested_bytes=result["attested_bytes"],
                            )
                        )
                result["samples"] = samples
                started = time.perf_counter()
                ack = client._rpc(
                    "/experiment/unregister", {"token": registration["token"]}
                )
                if ack != {"released": True}:
                    raise RuntimeError("unregister did not confirm storage release")
                result["unregister_seconds"] = time.perf_counter() - started
                released = True
            result["mean_seconds"] = statistics.fmean(
                s["seconds"] for s in result["samples"]
            )
            with (output / f"{job['label']}-results.json").open("x") as stream:
                json.dump(result, stream, indent=2)
            print(
                f"COMPLETE {job['label']} {result['mean_seconds'] * 1000:.3f} ms",
                flush=True,
            )
        # owned_server returns only after exact PID death, even under errors.
        released = True
    finally:
        if refs is not None:
            if released:
                _complete_ipc_refs(refs)
                keepalive.release()
            else:
                # A failed death check must leak/quarantine, not guess safety.
                quarantine_ipc_refs(refs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--cubins", required=True)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--placement", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "src"))
    sys.path.insert(0, str(args.root / "tests" / "performance"))
    import benchmark_large_model as large
    import torch

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "jobs").mkdir()
    os.environ["CUATTEST_KERNEL_DIR"] = args.cubins
    model = None
    try:
        with owned_server(args.root, args.output, {"label": "setup"}) as (client, info):
            devices = large.inspect_devices("all", info)
            large.warm_notaries(client, devices, large.describe_notary(info))
            devices = large.inspect_devices("all", info)
            all_specs, files = large.local_specs(args.checkpoint)
            specs = large.select_tensors(all_specs, ["mtp."])
            placement = (
                json.loads(args.placement.read_text())["placement"]
                if args.placement
                else large.plan_placement(specs, devices, large.GIB, 0.90)
            )
            start = time.perf_counter()
            layout = {}
            if args.packed:
                model, expected_root, layout = load_packed(
                    large, specs, files, placement, 8
                )
            else:
                model, expected_root = large.load_resident(specs, files, placement, 8)
                layout["memory"] = large.memory_report(model, specs, devices, placement)
            layout.update(
                placement=placement,
                independent_model_root=expected_root,
                load_seconds=time.perf_counter() - start,
                packed=args.packed,
                free_after_load={
                    d: torch.cuda.mem_get_info(d)[0] for d in set(placement.values())
                },
            )
            assert all(free > large.GIB for free in layout["free_after_load"].values())
            assert (
                expected_root
                == "077a5b303b1d6ed829a9c30fdb0102cf61e63fd5bcc562f8ebe153e10fdbe805"
            )
            (args.output / "resident-layout.json").write_text(
                json.dumps(layout, indent=2)
            )
        completed = set()
        deadline = time.monotonic() + 10800
        print("READY; checkpoint remains resident", flush=True)
        while time.monotonic() < deadline and not (args.output / "STOP").exists():
            for path in sorted((args.output / "jobs").glob("*.json")):
                if path.name in completed:
                    continue
                try:
                    job = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                # Fail the campaign on any safety/correctness error; never
                # continue with a quarantined lease or a mismatched digest.
                run_job(
                    large,
                    args.root,
                    args.output,
                    job,
                    model,
                    expected_root,
                    specs,
                    placement,
                )
                completed.add(path.name)
                print("READY", flush=True)
            time.sleep(0.25)
    finally:
        if model is not None:
            model.tensors.clear()
        gc.collect()
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        print("Experiment ended; checkpoint references released", flush=True)


if __name__ == "__main__":
    main()

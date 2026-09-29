#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Same resident allocation layout, owned services, production IPC/hash A/B.

No packing or synthetic padding. Without --checkpoint, load a pinned GPT-2;
with it, use the near-capacity benchmark's safetensors loader and CPU oracle.
Registration/retirement are reported separately; time the full public sign
call, including producer readiness and metadata checks, never verification.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def serve(args):
    sys.path.insert(0, str(args.source / "src"))
    from cuattest.multigpu import MultiGpuNotary
    from cuattest.server import _NotaryHTTPServer, make_handler

    notary = MultiGpuNotary(max_request_bytes=1 << 40, max_request_tiles=8388608)
    server = None
    try:
        server = _NotaryHTTPServer(("127.0.0.1", 0), make_handler(notary))
        with args.ready.open("x") as output:
            json.dump(
                {"port": server.server_port, "info": notary.info.as_dict()}, output
            )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        notary.close()


@contextmanager
def owned_server(args, label, mode, *, baseline=False):
    from cuattest.client import Client

    source = args.baseline_source if baseline else args.source
    cubins = args.baseline_cubins if baseline else args.cubins
    ready = args.output / (label + "-ready.json")
    environment = dict(
        os.environ,
        PYTHONPATH=str(source / "src"),
        CUATTEST_KERNEL_DIR=str(cubins),
        CUATTEST_CACHE=str(args.output / "cache"),
        CUATTEST_HASH_MODE=mode,
        CUATTEST_DISABLE_NATIVE_HOST="0",
    )
    with (args.output / (label + "-server.log")).open("x") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--serve",
                "--source",
                str(source),
                "--ready",
                str(ready),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 300
            while not ready.exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"owned server startup failed: {label}")
                time.sleep(0.1)
            while True:
                try:
                    setup = json.loads(ready.read_text())
                    break
                except json.JSONDecodeError:
                    if child.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("incomplete readiness document")
                    time.sleep(0.05)
            yield Client(f"http://127.0.0.1:{setup['port']}"), setup["info"]
        finally:
            # Popen keeps this exact, unreaped child identity. Only after wait
            # confirms its exit may an ambiguous lease be forcibly retired.
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
            child.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--ready", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cubins", type=Path)
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--baseline-cubins", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--model",
        default="openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e",
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    if args.serve:
        return serve(args)
    if (
        args.output is None
        or args.cubins is None
        or args.runs < 1
        or args.cycles < 1
        or args.warmup_runs < 0
    ):
        parser.error(
            "require --output, --cubins, positive runs/cycles and nonnegative warmups"
        )
    if bool(args.baseline_source) != bool(args.baseline_cubins):
        parser.error("baseline source and CUBIN directory must be supplied together")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source / "src"))
    import torch
    from blake3 import blake3
    from cuattest.expect import verify_evidence
    import benchmark_large_model as large
    import benchmark_torch_models as small

    model = None
    results = []
    try:
        with owned_server(args, "setup", "auto") as (client, info):
            keys = small.describe_notary(info)
            if args.checkpoint:
                devices = large.inspect_devices("all", info)
                large.warm_notaries(client, devices, keys)
                devices = large.inspect_devices("all", info)
                all_specs, files = large.local_specs(args.checkpoint)
                specs = large.select_tensors(all_specs, ["mtp."])
                placement = large.plan_placement(specs, devices, large.GIB, 0.90)
                model, expected_root = large.load_resident(specs, files, placement, 8)
                layout = large.memory_report(model, specs, devices, placement)
                model_id = large.DEFAULT_MODEL.model_id
            else:
                from transformers import AutoModelForCausalLM

                spec = small.ModelSpec.parse(args.model)
                model = (
                    AutoModelForCausalLM.from_pretrained(
                        spec.model_id, revision=spec.revision, local_files_only=True
                    )
                    .eval()
                    .to("cuda:0")
                )
                roots = [
                    blake3(t.cpu().contiguous().view(torch.uint8).numpy()).digest()
                    for _, t in sorted(model.state_dict().items())
                    if t.is_cuda and t.numel()
                ]
                expected_root = blake3(
                    len(roots).to_bytes(4, "little") + b"".join(roots)
                ).hexdigest()
                layout, placement, model_id = None, None, spec.model_id
            with (args.output / "layout.json").open("x") as output:
                json.dump(
                    dict(
                        memory=layout,
                        placement=placement,
                        independent_model_root=expected_root,
                    ),
                    output,
                    indent=2,
                )
        jobs = [
            (mode, registered, False)
            for mode in ("standard", "auto")
            for registered in (False, True)
        ]
        if args.baseline_source:
            jobs.insert(0, ("standard", False, True))
        for cycle in range(args.cycles):
            for mode, registered, baseline in (
                jobs if cycle % 2 == 0 else list(reversed(jobs))
            ):
                label = f"{cycle}-{'baseline' if baseline else mode}-{'registered' if registered else 'oneshot'}"
                handle = None
                death_confirmed = False
                try:
                    with owned_server(args, label, mode, baseline=baseline) as (
                        client,
                        info,
                    ):
                        keys = small.describe_notary(info)
                        record = dict(
                            label=label,
                            mode=mode,
                            registered=registered,
                            baseline=baseline,
                            notary=info,
                            samples=[],
                            independent_model_root=expected_root,
                        )
                        if registered:
                            started = time.perf_counter()
                            handle = client.register_model(model)
                            record["register_seconds"] = time.perf_counter() - started
                            record["unique_handles"] = len(
                                {r["handle"] for r in handle._refs}
                            )
                            for i in range(args.runs + args.warmup_runs):
                                started = time.perf_counter()
                                signed = handle.sign(model_id)
                                seconds = time.perf_counter() - started
                                verified = verify_evidence(signed, trusted_pubkeys=keys)
                                assert (
                                    verified.model_root == expected_root
                                    and verified.tensor_count == len(handle.names)
                                )
                                if i >= args.warmup_runs:
                                    record["samples"].append(
                                        dict(
                                            seconds=seconds,
                                            tensor_count=verified.tensor_count,
                                            attested_bytes=handle.attested_bytes,
                                            model_root=verified.model_root,
                                        )
                                    )
                            started = time.perf_counter()
                            handle.close()
                            record["close_seconds"] = time.perf_counter() - started
                        else:
                            record["samples"] = [
                                asdict(s)
                                for s in small.sample_model(
                                    client,
                                    model,
                                    model_id,
                                    args.runs,
                                    args.warmup_runs,
                                    keys,
                                    expected_root=expected_root,
                                )
                            ]
                        record["mean_seconds"] = statistics.fmean(
                            s["seconds"] for s in record["samples"]
                        )
                        results.append(record)
                        with (args.output / (label + "-results.json")).open(
                            "x"
                        ) as output:
                            json.dump(record, output, indent=2)
                        print(
                            f"COMPLETE {label}: {record['mean_seconds'] * 1000:.3f} ms",
                            flush=True,
                        )
                    death_confirmed = True
                finally:
                    # Normal code uses close ACKs. This recovery path is only
                    # legal because our owned_server has confirmed child death.
                    if handle is not None and death_confirmed:
                        handle.close(server_completed=True)
        return 0
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())

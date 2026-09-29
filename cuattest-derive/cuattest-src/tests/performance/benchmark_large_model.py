#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Near-capacity, multi-GPU attestation of pretrained model weights.

The default is the BF16 main-model state of Qwen3.5-397B-A17B, pinned to an
immutable revision. This benchmarks resident weights, not inference: it loads
the named safetensors directly, without executing model code, allocating KV
caches, quantizing weights, offloading, or making synthetic padding tensors.
See docs/performance.md for the separate notary command and memory requirements.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cuattest import safetensors as checkpoint_reader
from cuattest._cuda import Cuda
from cuattest._hosthash import blake3_digest
from cuattest.client import Client
from cuattest.multigpu import MAX_AGGREGATE_TENSORS
from cuattest.notary import _CHUNK, _TILE_CHUNKS, _validated_model

if __package__:
    from .benchmark_torch_models import (
        DEFAULT_URL,
        ModelResult,
        ModelSpec,
        attest_once,
        describe_notary,
        nonnegative_int,
        positive_int,
        print_summary,
        sample_model,
    )
else:
    from benchmark_torch_models import (
        DEFAULT_URL,
        ModelResult,
        ModelSpec,
        attest_once,
        describe_notary,
        nonnegative_int,
        positive_int,
        print_summary,
        sample_model,
    )

GIB = 1024**3
DEFAULT_MODEL = ModelSpec(
    "Qwen/Qwen3.5-397B-A17B", "8472618112abcbd45acbcdc58436aff4233c23f7"
)
# The auxiliary MTP module is not part of the stock Transformers inference
# model. Keeping it would exceed available VRAM on probqa.com after CUDA
# context initialization. Explicitly report this selection, never silently
# drop arbitrary weights until a too-large checkpoint happens to fit.
DEFAULT_EXCLUDE_PREFIXES = ("mtp.",)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    shard: str


@dataclass(frozen=True)
class DeviceMemory:
    ordinal: int
    uuid: str
    total: int
    free: int


@dataclass
class ResidentCheckpoint:
    tensors: dict

    def state_dict(self):
        return self.tensors.copy()


def fraction(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result < 1:
        raise argparse.ArgumentTypeError("must be finite and in [0, 1)")
    return result


def reserve_gib(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.25:
        raise argparse.ArgumentTypeError("must reserve at least 0.25 GiB per GPU")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=ModelSpec.parse, default=DEFAULT_MODEL)
    parser.add_argument(
        "--checkpoint", type=Path, help="use a local safetensors checkpoint"
    )
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face download cache")
    parser.add_argument(
        "--devices", default="all", help="all or comma-separated producer CUDA ordinals"
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--runs", type=positive_int, default=5)
    parser.add_argument("--warmup-runs", type=nonnegative_int, default=1)
    parser.add_argument("--registered", action="store_true", help="reuse IPC mappings and benchmark the full RegisteredModel.sign call")
    parser.add_argument("--reserve-gib", type=reserve_gib, default=1.0)
    parser.add_argument(
        "--min-utilization",
        type=fraction,
        default=0.90,
        help="minimum weight bytes / total VRAM on EACH GPU (default: 0.90)",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        help="explicit tensor-name prefix to omit; repeatable (Qwen preset: mtp.)",
    )
    parser.add_argument("--hash-threads", type=positive_int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check placement without downloading weights or allocating them",
    )
    parser.add_argument(
        "--json-output", type=Path, help="write results to a NEW JSON file"
    )
    args = parser.parse_args(argv)
    if args.exclude_prefix is None:
        args.exclude_prefix = (
            list(DEFAULT_EXCLUDE_PREFIXES)
            if args.model == DEFAULT_MODEL and args.checkpoint is None
            else []
        )
    if any(not prefix for prefix in args.exclude_prefix):
        parser.error("--exclude-prefix cannot be empty")
    if args.json_output is not None and args.json_output.exists():
        parser.error("--json-output must not already exist")
    return args


def select_devices(selection: str, count: int) -> list[int]:
    try:
        devices = (
            list(range(count))
            if selection == "all"
            else [int(x) for x in selection.split(",")]
        )
    except ValueError as error:
        raise ValueError(
            "--devices must be all or comma-separated CUDA ordinals"
        ) from error
    if (
        not devices
        or len(set(devices)) != len(devices)
        or any(not 0 <= d < count for d in devices)
    ):
        raise ValueError("--devices must select distinct visible CUDA devices")
    return devices


def inspect_devices(selection: str, info: dict) -> list[DeviceMemory]:
    import torch

    devices = select_devices(selection, torch.cuda.device_count())
    cuda = Cuda()
    cuda.init()
    entries = info["devices"] if info.get("multi_gpu") else [info]
    served = {entry["device_uuid"] for entry in entries}
    result = []
    for device in devices:
        uid = cuda.device_uuid(cuda.device(device))
        if uid not in served:
            raise ValueError(f"cuda:{device} ({uid}) is not served by this notary")
        # Query AFTER creating the producer context and with the separate
        # notary already running. Advertised board capacity overestimates the
        # space usable for weights, especially near 100% occupancy.
        with torch.cuda.device(device):
            torch.cuda.init()
            free, total = torch.cuda.mem_get_info()
        result.append(DeviceMemory(device, uid, total, free))
    return result


def warm_notaries(client, devices, trusted_keys):
    """Materialize CUDA's lazy signer resources before budgeting model storage."""
    import torch

    probe = ResidentCheckpoint(
        {
            f"probe.{d.ordinal}": torch.zeros(
                1, dtype=torch.uint8, device=f"cuda:{d.ordinal}"
            )
            for d in devices
        }
    )
    try:
        print("initializing notary launch memory (excluded from timings)", flush=True)
        sample = attest_once(client, probe, "cuattest/benchmark-prepare", trusted_keys)
        expected = blake3_digest(
            len(devices).to_bytes(4, "little") + blake3_digest(b"\0") * len(devices)
        ).hex()
        if sample.model_root != expected:
            raise RuntimeError("notary initialization probe returned incorrect bytes")
    finally:
        probe.tensors.clear()
        gc.collect()
        for d in devices:
            with torch.cuda.device(d.ordinal):
                torch.cuda.empty_cache()


def select_tensors(specs: list[TensorSpec], prefixes: list[str]) -> list[TensorSpec]:
    selected = [
        t for t in specs if t.nbytes and not any(t.name.startswith(p) for p in prefixes)
    ]
    selected.sort(key=lambda t: t.name)
    if not selected or len(selected) > MAX_AGGREGATE_TENSORS:
        raise ValueError(
            f"checkpoint must select 1..{MAX_AGGREGATE_TENSORS} nonempty tensors"
        )
    if len({t.name for t in selected}) != len(selected):
        raise ValueError("checkpoint contains duplicate tensor names")
    return selected


def allocated_size(nbytes: int) -> int:
    # PyTorch rounds large CUDA allocations to 2 MiB segments. Small weights
    # share allocator pools; the explicit reserve covers pool slack, IPC map
    # overhead, and the notary's retained scratch, not extra model copies.
    alignment = 2 * 1024**2 if nbytes >= 10 * 1024**2 else 512
    return (nbytes + alignment - 1) // alignment * alignment


def plan_placement(
    specs: list[TensorSpec], devices: list[DeviceMemory], reserve: int, minimum: float
) -> dict[str, int]:
    capacities = {d.ordinal: d.free - reserve for d in devices}
    used = dict.fromkeys(capacities, 0)
    logical = dict.fromkeys(capacities, 0)
    assignment = {}
    if any(capacity <= 0 for capacity in capacities.values()):
        raise ValueError("not enough free VRAM after reserving notary headroom")
    # Largest-first placement avoids stranding an 8-GiB expert tensor after
    # small tensors have filled every GPU. Never slice a tensor or offload it;
    # changing placement must not change the canonical named-span fold.
    for tensor in sorted(specs, key=lambda t: (-allocated_size(t.nbytes), t.name)):
        size = allocated_size(tensor.nbytes)
        fits = [d for d in capacities if used[d] + size <= capacities[d]]
        if not fits:
            raise ValueError(
                f"checkpoint does not fit with reserved headroom; cannot place {tensor.name} ({tensor.nbytes / GIB:.3f} GiB)"
            )
        device = min(fits, key=lambda d: (used[d] / capacities[d], d))
        assignment[tensor.name] = device
        used[device] += size
        logical[device] += tensor.nbytes
    for device in devices:
        if logical[device.ordinal] / device.total < minimum:
            raise ValueError(
                f"cuda:{device.ordinal} would not reach --min-utilization {minimum:.1%}; use a smaller threshold only for smoke tests"
            )
    return assignment


def local_specs(path: Path) -> tuple[list[TensorSpec], dict[str, Path]]:
    tensors = checkpoint_reader.load(path)
    specs = [
        TensorSpec(t.name, t.shape, t.dtype, t.nbytes, str(t.path)) for t in tensors
    ]
    return specs, {str(t.path): t.path for t in tensors}


def hub_specs(model: ModelSpec, cache_dir: Path | None):
    from huggingface_hub import HfApi

    api = HfApi()
    # Resolve unpinned user choices ONCE. Metadata and all weight downloads
    # must use that immutable commit, even if the upstream branch advances.
    revision = api.model_info(model.model_id, revision=model.revision).sha
    meta = api.get_safetensors_metadata(model.model_id, revision=revision, timeout=60)
    specs = []
    for name, shard in meta.weight_map.items():
        tensor = meta.files_metadata[shard].tensors[name]
        nbytes = tensor.data_offsets[1] - tensor.data_offsets[0]
        shape = tuple(tensor.shape)
        if any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f"invalid checkpoint shape for {name}")
        bits = checkpoint_reader.DTYPE_BITS[tensor.dtype] * math.prod(shape)
        if nbytes < 0 or bits != nbytes * 8:
            raise ValueError(f"invalid checkpoint metadata for {name}")
        specs.append(TensorSpec(name, shape, tensor.dtype, nbytes, shard))
    return specs, revision


def download_checkpoint(
    model: ModelSpec, revision: str, specs: list[TensorSpec], cache_dir: Path | None
) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    shards = sorted({t.shard for t in specs})
    root = Path(
        snapshot_download(
            model.model_id,
            revision=revision,
            allow_patterns=shards,
            cache_dir=cache_dir,
            max_workers=4,
        )
    )
    files = {name: root / name for name in shards}
    # Validate the actual complete shard headers/ranges, not just the Hub's
    # remote metadata. Extra explicitly excluded tensors may share a shard.
    disk = {
        t.name: t
        for path in files.values()
        for t in checkpoint_reader.tensors_in_file(path)
    }
    for spec in specs:
        tensor = disk.get(spec.name)
        if tensor is None or (
            tensor.path,
            tensor.shape,
            tensor.dtype,
            tensor.nbytes,
        ) != (files[spec.shard], spec.shape, spec.dtype, spec.nbytes):
            raise ValueError(
                f"downloaded tensor contradicts pinned metadata: {spec.name}"
            )
    return files


def load_resident(
    specs: list[TensorSpec],
    files: dict[str, Path],
    placement: dict[str, int],
    hash_threads: int,
) -> tuple[ResidentCheckpoint, str]:
    import torch
    from blake3 import blake3
    from safetensors import safe_open

    model = ResidentCheckpoint({})
    digests = {}
    total = sum(t.nbytes for t in specs)
    loaded = 0
    last_report = time.monotonic()
    try:
        with ExitStack() as stack:
            readers = {
                name: stack.enter_context(
                    safe_open(str(path), framework="pt", device="cpu")
                )
                for name, path in files.items()
            }
            for spec in sorted(
                specs, key=lambda t: (-allocated_size(t.nbytes), t.name)
            ):
                cpu = readers[spec.shard].get_tensor(spec.name)
                # Hash the mapped checkpoint bytes independently while loading,
                # outside the timed interval. No full checkpoint-sized CPU
                # copy is required, and uninitialized CUDA memory is never a
                # substitute for real pretrained weights.
                cpu_bytes = cpu.reshape(-1).view(torch.uint8).numpy()
                digests[spec.name] = blake3(
                    cpu_bytes, max_threads=hash_threads
                ).digest()
                model.tensors[spec.name] = cpu.to(f"cuda:{placement[spec.name]}")
                del cpu_bytes, cpu
                loaded += spec.nbytes
                if time.monotonic() - last_report >= 5 or loaded == total:
                    print(
                        f"loaded and independently hashed {loaded / GIB:.1f}/{total / GIB:.1f} GiB",
                        flush=True,
                    )
                    last_report = time.monotonic()
        for device in set(placement.values()):
            torch.cuda.synchronize(device)
        ordered = b"".join(digests[t.name] for t in sorted(specs, key=lambda t: t.name))
        return model, blake3_digest(len(specs).to_bytes(4, "little") + ordered).hex()
    except BaseException:
        model.tensors.clear()
        raise


def resident_bytes(model, specs, placement):
    """Measure unique CUDA storage, not aliases, CPU offloads, or allocator cache."""
    if set(model.tensors) != {t.name for t in specs}:
        raise RuntimeError("resident checkpoint tensor set does not match the plan")
    seen = set()
    result = dict.fromkeys(placement.values(), 0)
    for spec in specs:
        tensor = model.tensors[spec.name]
        device = placement[spec.name]
        if not tensor.is_cuda or tensor.device.index != device:
            raise RuntimeError(f"{spec.name} is not resident on planned cuda:{device}")
        if (
            not tensor.is_contiguous()
            or tensor.numel() * tensor.element_size() != spec.nbytes
        ):
            raise RuntimeError(f"{spec.name} is not the planned contiguous byte span")
        storage = tensor.untyped_storage()
        key = (device, storage.data_ptr())
        if key in seen or storage.nbytes() != spec.nbytes:
            raise RuntimeError(
                "capacity benchmark requires unique, unpadded weight storage"
            )
        seen.add(key)
        result[device] += storage.nbytes()
    return result


def memory_report(model, specs, devices, placement):
    import torch

    result = []
    actual = resident_bytes(model, specs, placement)
    for device in devices:
        resident = actual.get(device.ordinal, 0)
        free, _ = torch.cuda.mem_get_info(device.ordinal)
        result.append(
            {
                **asdict(device),
                "resident_weight_bytes": resident,
                "weight_utilization": resident / device.total,
                "free_after_load": free,
                "torch_allocated": torch.cuda.memory_allocated(device.ordinal),
                "torch_reserved": torch.cuda.memory_reserved(device.ordinal),
            }
        )
        print(
            f"cuda:{device.ordinal}: {resident / GIB:.3f} GiB weights / {device.total / GIB:.3f} GiB ({resident / device.total:.2%}); {free / GIB:.3f} GiB free",
            flush=True,
        )
    return result


def main(argv=None) -> int:
    import torch

    args = parse_args(argv)
    from cuattest.ipc import _native_export_cuda_allocation

    if _native_export_cuda_allocation is None:
        raise RuntimeError(
            "safe CUDA IPC export requires the native extension; install cuattest with pip install -e ."
        )
    model_id = (
        "local/checkpoint" if args.checkpoint is not None else args.model.model_id
    )
    _validated_model(model_id)
    client = Client(args.url)
    info = client.info()
    trusted_keys = describe_notary(info)
    devices = inspect_devices(args.devices, info)
    # CUDA can allocate gigabytes of local-memory backing on the first signer
    # launch. A context-only memory check misses that cost. Even --dry-run
    # performs this tiny verified probe; it still allocates no model weights.
    warm_notaries(client, devices, trusted_keys)
    devices = inspect_devices(args.devices, info)
    if args.checkpoint is not None:
        all_specs, files = local_specs(args.checkpoint)
        revision = "local-checkpoint"
        model_id = "local/checkpoint"
    else:
        all_specs, revision = hub_specs(args.model, args.cache_dir)
        model_id = args.model.model_id
        files = None
    specs = select_tensors(all_specs, args.exclude_prefix)
    reserve = int(args.reserve_gib * GIB)
    placement = plan_placement(specs, devices, reserve, args.min_utilization)
    nbytes = sum(t.nbytes for t in specs)
    tile_bytes = _CHUNK * _TILE_CHUNKS
    tiles = sum((t.nbytes + tile_bytes - 1) // tile_bytes for t in specs)
    print(
        f"\n{model_id}@{revision}: {len(specs):,} tensors, {nbytes / GIB:.3f} GiB; excluded prefixes: {args.exclude_prefix}",
        flush=True,
    )
    print(
        f"notary must allow at least {nbytes} request bytes and {tiles} scheduling tiles",
        flush=True,
    )
    for device in devices:
        amount = sum(t.nbytes for t in specs if placement[t.name] == device.ordinal)
        print(
            f"planned cuda:{device.ordinal}: {amount / GIB:.3f} GiB ({amount / device.total:.2%}), reserve {args.reserve_gib:.2f} GiB",
            flush=True,
        )
    if args.dry_run:
        return 0
    if files is None:
        files = download_checkpoint(args.model, revision, specs, args.cache_dir)
    # Downloads can be lengthy. Recheck current free memory before allocating,
    # rather than trusting a fit computed before another job started.
    devices = inspect_devices(args.devices, info)
    placement = plan_placement(specs, devices, reserve, args.min_utilization)
    model = None
    try:
        model, root = load_resident(specs, files, placement, args.hash_threads)
        occupancy = memory_report(model, specs, devices, placement)
        if any(d["free_after_load"] < reserve for d in occupancy):
            raise RuntimeError(
                "allocator overhead consumed reserved notary headroom; increase --reserve-gib"
            )
        samples = sample_model(
            client,
            model,
            model_id,
            args.runs,
            args.warmup_runs,
            trusted_keys,
            expected_root=root,
            registered=args.registered,
        )
        result = ModelResult(
            model_id,
            revision,
            sum(math.prod(t.shape) for t in specs),
            ",".join(sorted({t.dtype for t in specs})),
            samples,
        )
        print_summary([result])
        print(
            f"verified input throughput: {nbytes / GIB / result.average_seconds:.2f} GiB/s",
            flush=True,
        )
        if args.json_output:
            report = {
                "benchmark": "pretrained-checkpoint-residency-v1",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "hostname": platform.node(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "notary": info,
                "excluded_prefixes": args.exclude_prefix,
                "devices": occupancy,
                "warmup_runs": args.warmup_runs,
                "registered": args.registered,
                "reserve_bytes": reserve,
                "independent_model_root": root,
                "result": asdict(result),
                "mean_seconds": result.average_seconds,
                "stdev_seconds": result.stdev_seconds,
            }
            with args.json_output.open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        return 0
    finally:
        if model is not None:
            model.tensors.clear()
        gc.collect()
        # A failed/uncertain Client call retains its process-owned IPC leases.
        # Do not force-release those leases to make cleanup or a retry succeed.
        for device in devices:
            with torch.cuda.device(device.ordinal):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())

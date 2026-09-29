#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark repeated attestations of CUDA-resident Hugging Face models.

Start the notary in another process before running this script::

    .venv/bin/cuattest serve
    .venv/bin/python tests/performance/benchmark_torch_models.py

By default the benchmark measures pinned revisions of GPT-2 and GPT-2 Large.
Pass ``--model MODEL_ID[@REVISION]`` more than once to benchmark a different
set of causal language models.
Model loading, CUDA IPC export, and evidence verification are deliberately
outside the timed interval so each sample matches the elapsed time reported by
``examples/measure_torch_model.py``.
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cuattest.client import Client, NotaryClientError
from cuattest.expect import verify_evidence
from cuattest.ipc import share_model

DEFAULT_URL = os.environ.get("CUATTEST_URL", "http://127.0.0.1:8077")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    revision: str | None = None

    @classmethod
    def parse(cls, value: str) -> ModelSpec:
        model_id, separator, revision = value.rpartition("@")
        if not separator:
            return cls(value)
        if not model_id or not revision:
            raise argparse.ArgumentTypeError(
                "model must be MODEL_ID or MODEL_ID@REVISION"
            )
        return cls(model_id, revision)


DEFAULT_MODELS = (
    ModelSpec(
        "openai-community/gpt2",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e",
    ),
    ModelSpec(
        "openai-community/gpt2-large",
        "32b71b12589c2f8d625668d2335a01cac3249519",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL_ID[@REVISION]",
        type=ModelSpec.parse,
        help=(
            "Hugging Face causal language model to benchmark; repeat for multiple "
            "models (default: pinned GPT-2 and GPT-2 Large revisions)"
        ),
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=5,
        help="number of measured attestations per model (default: 5)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=nonnegative_int,
        default=1,
        help="warm-up attestations excluded from each average (default: 1)",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"notary base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="CUDA device on which to load models (default: cuda)",
    )
    parser.add_argument("--registered", action="store_true", help="reuse IPC mappings; time the complete RegisteredModel.sign call")
    args = parser.parse_args(argv)
    if args.models is None:
        args.models = list(DEFAULT_MODELS)
    return args


@dataclass(frozen=True)
class AttestationSample:
    seconds: float
    tensor_count: int
    attested_bytes: int
    model_root: str


@dataclass(frozen=True)
class ModelResult:
    model_id: str
    revision: str
    parameter_count: int
    dtypes: str
    samples: tuple[AttestationSample, ...]

    @property
    def average_seconds(self) -> float:
        return statistics.fmean(sample.seconds for sample in self.samples)

    @property
    def stdev_seconds(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return statistics.stdev(sample.seconds for sample in self.samples)


def attest_once(
    client: Client, model, model_id: str, trusted_pubkey: str | dict[str, str]
) -> AttestationSample:
    names, refs, keepalive = share_model(model)
    attested_bytes = sum(ref["nbytes"] for ref in refs)
    try:
        started = time.perf_counter()
        signed = client.sign(refs, model_id)
        seconds = time.perf_counter() - started
        # A multi-GPU service always returns an aggregate, including for GPT-2
        # on one GPU. Pin each participating session; never take keys from the
        # receipt being timed or silently skip aggregate verification.
        keys = (
            {"trusted_pubkeys": trusted_pubkey}
            if isinstance(trusted_pubkey, dict)
            else {"trusted_pubkey": trusted_pubkey}
        )
        verified = verify_evidence(signed, **keys)
    finally:
        # Client.sign retires the acknowledged process-owned storage leases.
        # Keep the tensors alive until it has returned and verification has run.
        del keepalive

    if verified.tensor_count != len(names):
        raise RuntimeError(
            f"signed tensor count {verified.tensor_count} does not match the "
            f"{len(names)} shared tensors"
        )
    return AttestationSample(
        seconds=seconds,
        tensor_count=verified.tensor_count,
        attested_bytes=attested_bytes,
        model_root=verified.model_root,
    )


def sample_model(
    client: Client,
    model,
    model_id: str,
    runs: int,
    warmup_runs: int,
    trusted_pubkey: str | dict[str, str],
    *,
    expected_root: str | None = None,
    registered: bool = False,
) -> tuple[AttestationSample, ...]:
    """Shared timing contract for small models and the near-capacity workload."""
    samples = []
    # Register/close are setup/retirement, reported outside steady-state
    # samples. The timed public call includes metadata checks, producer-stream
    # readiness, HTTP, all fresh hashing/signing, and JSON response parsing.
    with client.register_model(model) if registered else nullcontext(None) as handle:
        for index in range(warmup_runs + runs):
            if handle is None:
                sample = attest_once(client, model, model_id, trusted_pubkey)
            else:
                started = time.perf_counter()
                signed = handle.sign(model_id)
                seconds = time.perf_counter() - started
                keys = ({"trusted_pubkeys": trusted_pubkey} if isinstance(trusted_pubkey, dict)
                        else {"trusted_pubkey": trusted_pubkey})
                verified = verify_evidence(signed, **keys)
                if verified.tensor_count != len(handle.names):
                    raise RuntimeError("registered tensor count changed")
                sample = AttestationSample(seconds, verified.tensor_count, handle.attested_bytes, verified.model_root)
            if expected_root is not None and sample.model_root != expected_root:
                raise RuntimeError("model root differs from the expected or previous root")
            expected_root = sample.model_root
            if index < warmup_runs:
                label = f"warm-up {index + 1}/{warmup_runs}"
            else:
                samples.append(sample)
                label = f"run {index - warmup_runs + 1}/{runs}"
            print(
                f"{label}: measured {sample.tensor_count} tensors in {sample.seconds:.4f}s",
                flush=True,
            )
    if len({(s.tensor_count, s.attested_bytes) for s in samples}) != 1:
        raise RuntimeError("shared tensor set changed between attestations")
    return tuple(samples)


def describe_notary(info: dict) -> str | dict[str, str]:
    """Show the service identity and capture pins through trusted local setup."""
    devices = info["devices"] if info.get("multi_gpu") else [info]
    for device in devices:
        print(f"notary  {device['gpu_did']}")
        print(
            f"        {device['device']} ({device['arch']}), pid {device['pid']}; "
            f"cuda:{device['device_ordinal']} {device.get('device_uuid', '')}"
        )
        print(f"        host   {device.get('host_backend', 'unknown')}")
        print(f"        kernel {device['kernel_cid']}")
        print(f"        cubin  {device['cubin_cid']}  [{device['compiler']}]")
    if info.get("multi_gpu"):
        return {d["device_uuid"]: d["gpu_pubkey_uncompressed"] for d in devices}
    return info["gpu_pubkey_uncompressed"]


def benchmark_model(
    client: Client,
    model_spec: ModelSpec,
    device: str,
    runs: int,
    warmup_runs: int,
    trusted_pubkey: str | dict[str, str],
    *,
    registered: bool = False,
) -> ModelResult:
    import torch
    from transformers import AutoModelForCausalLM

    requested_revision = model_spec.revision or "the repository default revision"
    print(
        f"\nloading {model_spec.model_id} at {requested_revision} into this process..."
    )
    # Prefer the non-executable checkpoint format. The default benchmark models
    # both publish safetensors weights, and arbitrary --model values fail closed
    # instead of falling back to a pickle checkpoint.
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_spec.model_id,
            revision=model_spec.revision,
            use_safetensors=True,
        )
        .to(device)
        .eval()
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    dtypes = ",".join(
        sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for parameter in model.parameters()
            }
        )
    )
    revision = getattr(model.config, "_commit_hash", None) or "unknown"
    print(
        f"loaded {parameter_count:,} parameters ({dtypes}); "
        f"resolved revision {revision}"
    )

    try:
        samples = sample_model(
            client, model, model_spec.model_id, runs, warmup_runs, trusted_pubkey,
            registered=registered,
        )
        result = ModelResult(
            model_id=model_spec.model_id,
            revision=revision,
            parameter_count=parameter_count,
            dtypes=dtypes,
            samples=samples,
        )
        sample = result.samples[0]
        print(
            f"average measured {sample.tensor_count} tensors "
            f"({sample.attested_bytes / (1024**2):,.1f} MiB) in "
            f"{result.average_seconds:.4f}s over {runs} runs "
            f"(sample stdev {result.stdev_seconds:.4f}s)"
        )
        return result
    finally:
        # Run models sequentially so the large-model measurement does not
        # inherit allocations retained by the small-model caching allocator.
        del model
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


def print_summary(results: list[ModelResult]) -> None:
    print("\naverages")
    print(
        f"{'model':<36} {'tensors':>7} {'attested MiB':>12} "
        f"{'runs':>5} {'mean (s)':>10} {'stdev (s)':>11}"
    )
    for result in results:
        sample = result.samples[0]
        print(
            f"{result.model_id:<36} {sample.tensor_count:>7} "
            f"{sample.attested_bytes / (1024**2):>12.1f} "
            f"{len(result.samples):>5} {result.average_seconds:>10.4f} "
            f"{result.stdev_seconds:>11.4f}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    client = Client(args.url)
    try:
        info = client.info()
    except NotaryClientError as error:
        print(error, file=sys.stderr)
        return 1

    trusted_pubkey = describe_notary(info)
    print(
        f"benchmarking {len(args.models)} model(s), {args.warmup_runs} warm-up "
        f"and {args.runs} measured attestations each"
    )

    results = [
        benchmark_model(
            client,
            model_spec,
            args.device,
            args.runs,
            args.warmup_runs,
            trusted_pubkey,
            registered=args.registered,
        )
        for model_spec in args.models
    ]
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

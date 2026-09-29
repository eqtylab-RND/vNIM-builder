# Production registration and asynchronous hashing validation — 2026-09-09

This qualifies the implementation described in the
[performance report](../performance.md#production-registration-and-asynchronous-hashing-2026-09-09),
not just the earlier experimental server. Raw timings, identities, placement,
size-sweep samples and sanitizer records are in the
[machine-readable report](../performance/registration-2026-09-09.json).

## Functional and host validation

| Check | Result |
| --- | --- |
| Final default CPU suite | **595 passed, 93 opt-in/hardware skips** |
| RTX 2080 GPU/IPC/stream regressions, both backends | **49 passed**; final 14 registration cases also rerun after the local-retirement fix |
| Blackwell registered IPC and existing multi-GPU regressions, all eight GPUs, forced async | **22 passed**, both backends, reversed device ordinals |
| Independent cryptographic differential tests | **32 passed** on RTX 2080; **256 passed** across all eight Blackwell GPUs with async forced |
| Real pretrained models | GPT-2/GPT-2 Large on the laptop, full 739.102-GiB Qwen on Blackwell; every warmup/timed receipt verified against independent CPU hashes |
| Clang 21 ASan + LSan + UBSan + bounds | Exact native core: **8 concurrent sessions passed**, no diagnostic |
| Clang 21 TSan | Exact native core: **8 concurrent sessions passed**, no diagnostic |
| Ruff `F,E9`, compileall, `git diff --check` | Pass |

The new regressions cover zero-copy registration, all declared producer streams,
partial blocks/tiles, unaligned spans, changed values between observations,
storage replacement even without version increments, lazy-view refusal,
overlapping one-shot/registered imports, sequence replay, bounded retained
work, expired/closed tickets and delayed imports. Lost import/sign/close
responses retain the producer lease; matching close retries retire it once.
Fault injection covers failed import/bounds queries/unmap/context destruction,
context-entry/unwinding errors, exception-allocation failure, and interruption
after acknowledged close but before removing the local recovery entry.

The host harness exercises the added native dispatch threshold on either side
of its inclusive boundary, aggregation across spans, the async kernel's own
grid limit, and overflow rejection **before** allocating workspaces. It retains
the existing deferred-DMA, failed-cleanup and independent-session concurrency
checks. This is a targeted ASan/UBSan/TSan rerun, not a new full MSan/CFI/etc.
audit or an ASan build of every Python dependency; the earlier
[whole-codebase audit](sanitizers-2026-09-08.md) remains separate.

## Compute Sanitizer

Driver **610.57.04** on both hosts; Compute Sanitizer **2026.2.1.0**. The laptop's
post-reboot CUDA/NVML path works. No driver replacement or unrelated-process
termination was performed.

| Workload | memcheck | initcheck | racecheck | synccheck |
| --- | --- | --- | --- | --- |
| RTX 2080 selftest, native + fallback | Clean | Clean | Clean | Clean |
| Blackwell forced-async driver-only hash probe, native + fallback | Clean | Clean, global + shared | Clean | Clean |
| Blackwell registered IPC, fresh producer per backend | Clean, duplicate-import case excluded as below | Clean, shared | Clean | Clean |

The driver-only probe checks **181** combinations of complete/partial/unaligned
inputs and mixed-span folding, plus repeated 64-MiB hashes, against Python's
`blake3` package. Every tool/backend invocation passed. Logs report zero
errors, zero race hazards/warnings, and (for memcheck) zero leaked bytes or
allocations. Strict driver API reporting, module-unload checks, allocation
padding, stream-ordered races, bulk-copy and tensor-operation checks are enabled
where supported. Registered IPC runs cover seven cases per backend; memcheck
runs six and explicitly deselects one. IPC initcheck uses shared memory because
cross-process initialization tracking is unsupported; the driver-only probe
checks both address spaces. Exit codes **and diagnostic text** were inspected.

### Raw memcheck finding: duplicate-import tracking

The unfiltered registered memcheck run **failed**, reporting out-of-bounds
reads after another reference to the same imported allocation was closed.
The [CUDA-only control](../../tests/sanitizers/ipc_refcount_probe.py) reproduces
the same result with a one-byte PTX copy and no cuAttest/PyTorch imports:

| Control | Result |
| --- | --- |
| One open, copy, one close, instrumented | Pass; zero errors/leaks |
| Two opens, one close, copy, final close, uninstrumented | Pass; correct byte |
| Two opens, one close, kernel copy, instrumented | False out-of-bounds read in `copy_byte`, followed by CUDA error 719 |
| Two opens/one close with only a host DtoH copy, instrumented | Pass; the kernel shadow allocation table is needed to reproduce it |

CUDA explicitly [reference-counts repeated opens in the same context](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html),
and unmaps only after the corresponding final close. The instrumented control
loses its shadow allocation at the first close. This establishes a memcheck
limitation independently of the application's lease logic. The raw finding
and control logs are retained; they are **not** called clean. The ordinary GPU,
initcheck, racecheck and synccheck suites still exercise overlapping imports.
The default sanitizer runner does not silently exclude this case.

### Initial mixed-backend racecheck run: inconclusive

An initial run using one producer process for both backends stopped making
progress after six cases. Python/native stack capture placed it in the second
producer's `torch.cuda.Stream.synchronize()` during export, before that
registration's handles were submitted. The GPU was busy; the service was idle.
After more than 20 minutes it was terminated (exit 15), **not counted as a
pass**. No race diagnostic was emitted. A separate Torch-only two-allocation
control passed, so the precise cause of this combined instrumented transition
is not established.

The final runner uses a fresh producer for each backend. Both complete all
seven cases cleanly under racecheck (38.07/37.46 seconds), including the blocked
unrelated-stream test. Ordinary live GPU tests continue to cover the combined
backend transition. This scoped result does not claim to resolve the initial
instrumented stall.

## Reproduction and artifacts

Rebuild the native extension and architecture-matched CUBIN from this checkout.
The final CUDA source SHA-256 is
`99696c70a2fc8cad461cd9b43e2d1b2aed525d4d2ee61a99f2b8e4eb2d398c2d`.
CUBIN SHA-256:

- `sm_75`, NVRTC 12.9: `b530beba62ee72b036c35048f3e3a0af123319573b852186386e53a368717cdb`.
- `sm_120`, NVRTC 13.3: `1ad989b55026705cd135c34552efd00aa3918ba931f55a7b5b94133d7f303e5e`.

Replacing duplicated old-GPU fallback code with a call to the original helper,
and formatting/comments, reproduced **bit-identical CUBINs** on both targets.
The final source identity is nevertheless different and was rebuilt/advertised
correctly; source/CUBIN allowlists must be updated.

```bash
pytest -q
CUATTEST_TEST_GPU=1 CUATTEST_TEST_MULTIGPU=1 CUATTEST_HASH_MODE=async \
pytest -q tests/test_registered_integration.py tests/test_multigpu_integration.py

# Driver-only probe: all four tools, both backends, isolated writable cache.
CUATTEST_HASH_MODE=async CUATTEST_KERNEL_DIR=/path/to/cubins \
python tests/performance/experiments/ipc_hash/sanitize.py "$PWD" /tmp/new-hash-audit

CUATTEST_HASH_MODE=async CUATTEST_KERNEL_DIR=/path/to/cubins \
python tests/sanitizers/run_cuda.py /tmp/new-registered-audit --suite registered \
  --tool initcheck --tool racecheck --tool synccheck

# Explicit exclusion of the independently reproduced memcheck limitation.
PYTEST_ADDOPTS='-k "not duplicate_imports"' CUATTEST_HASH_MODE=async \
CUATTEST_KERNEL_DIR=/path/to/cubins \
python tests/sanitizers/run_cuda.py /tmp/new-registered-memcheck \
  --suite registered --tool memcheck

compute-sanitizer --tool memcheck --target-processes all --error-exitcode 86 \
  python tests/sanitizers/ipc_refcount_probe.py --opens 2
```

Use `auto`/`standard`, not forced async, on the RTX 2080. The real multi-GPU
tests use every visible device; constrain visibility if other jobs are running.
Audit caches are isolated from the normal CUBIN cache. Local logs are under
`/tmp/cuattest-register.TO93hk`; remote originals under
`probqa.com:/tmp/cuattest-register.jpxkPu`, with a local `remote/` mirror.
The final full-model trials are `qwen-final/`; final sanitizer logs are
`final-hash-sanitizers/`, `final-registered-tools/`, and `final-split-memcheck/`.
Raw failed/inconclusive runs remain in `registered-sanitizers/`, with
`refcount-*`, `racecheck-*`, and `torch-racecheck-control.log` controls/traces.

All owned benchmark/audit processes exited and the remote GPUs were released.
The unrelated laptop desktop workload was left running. The attempted
Blackwell GPT-2 Large run had no offline checkpoint cache and produced no
timing samples; it is not included in the performance claims.

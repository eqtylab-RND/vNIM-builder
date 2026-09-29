# Build configurations and invariant checks

cuAttest uses setuptools for its C++ host extension and NVRTC for its CUDA
kernels. Both implement the same three configurations:

| Configuration | Native C++ | CUDA | Internal Python checks |
| --- | --- | --- | --- |
| **Release** (default) | `-O3 -DNDEBUG` | optimized, `NDEBUG` defined | disabled |
| **Debug** | `-O0 -g3 -UNDEBUG` | `--device-debug`, optimization off, `NDEBUG` undefined | enabled |
| **AssertedRelease** | `-O3 -g -UNDEBUG` | optimized with line information, `NDEBUG` undefined | enabled |

AssertedRelease is for finding bugs under optimized execution; it is not a
different cryptographic algorithm. Debug and AssertedRelease can be much
slower and use more registers, stack, and shared memory. Cooperative launch
occupancy is queried from the **actual loaded kernel**, not a Release estimate.

## Build and install

From the repository root, with a C++17 compiler and the dependencies installed:

```bash
# Normal production installation (also the default when the variable is absent).
CUATTEST_BUILD_TYPE=Release python -m pip install --no-cache-dir --force-reinstall --no-deps .

# Optimized code, with internal assertions enabled.
CUATTEST_BUILD_TYPE=AssertedRelease python -m pip install --no-cache-dir --force-reinstall --no-deps .

# Unoptimized native/device code with debug information and assertions.
CUATTEST_BUILD_TYPE=Debug python -m pip install --no-cache-dir --force-reinstall --no-deps .
```

Use separate virtual environments for configurations you want to keep side by
side. `--no-deps` assumes dependencies are already installed; see the
[installation instructions](../README.md#install). Do not reuse a cached wheel
built in another configuration: wheel filenames/version numbers do not encode
this local build choice.

For an editable checkout, either use the same environment variable with
`pip install --no-cache-dir -e .`, or rebuild just the extension:

```bash
CUATTEST_BUILD_TYPE=AssertedRelease python setup.py build_ext --inplace
```

The latter is a development command, not the recommended package installation
interface. Switching configuration forces both extension and object-file
recompilation, even if source timestamps have not changed. The final compiler
options override CPython's usual `-DNDEBUG` and inherited optimization flags.

The compiled extension and wheel record their configuration. You do **not**
need to keep the environment variable set after installation. Restart existing
Python/notary processes after rebuilding; changing an environment variable
cannot rewrite a loaded extension's assertions. A conflicting
`CUATTEST_BUILD_TYPE` is rejected instead of creating a mixed build.

Check the installed configuration without loading CUDA:

```bash
python -c 'from cuattest._build_config import BUILD_TYPE, ASSERTIONS_ENABLED; print(BUILD_TYPE, ASSERTIONS_ENABLED)'
python -c 'from cuattest import _native; print(_native.BUILD_TYPE, _native.ASSERTIONS_ENABLED)'
```

Without a native extension, a wheel's stamped configuration still governs
Python/CUDA. An unbuilt source tree defaults to Release, or accepts the
explicit environment selection. Debug/AssertedRelease reject Python `-O`,
`-OO`, and `PYTHONOPTIMIZE`, which would otherwise remove Python assertions.
Release does **not** require `python -O`; its internal assertion guards are off
even under the ordinary interpreter, and pytest's own assertions remain on.

## CUDA artifacts and deployment

By default `cuattest build-kernel` and automatic compilation use the installed
configuration. NVRTC is required for a cache miss, but not for loading a
matching prebuilt CUBIN. A build machine can explicitly compile another mode:

```bash
cuattest build-kernel --arch sm_120 --build-type AssertedRelease --out /tmp/asserted-cubins
```

An explicit `--build-type` here selects **only that compilation**; it does not
change the installed host extension or the notary's runtime configuration.
Release retains `p256_cuda_notary_b3.sm_120.cubin`; the other modes use
`p256_cuda_notary_b3.sm_120.Debug.cubin` and
`p256_cuda_notary_b3.sm_120.AssertedRelease.cubin`. They can coexist in one
directory. Each digest-bound sidecar also records `build_type` and the exact
mode-specific `build_options`. A missing/mismatched policy is a cache miss,
including for older sidecars: there is no fallback to another mode's CUBIN.
The test-oracle cache is mode-specific too.

Assertions change the trusted source identity, and asserted/debug machine code
has its own CUBIN identity. Rebuild artifacts and update source/CUBIN allowlists
deliberately. Do not accept any artifact merely because its configuration
label says Release; the existing independent content-hash checks still apply.

## What is checked

The assertions target internal contracts, especially:

- Native descriptor packing, reduction-plan sizes/termination, workspace
  capacity, aligned metadata, launch limits, result extents, and IPC ownership
  and drain obligations.
- BLAKE3's block/chunk bounds, domain flags, vector alignment, streaming-stack
  population, tensor/tile prefixes, reduction rounds and child indices.
- SHA-256 buffering and length accounting; limb widths, output alias rules,
  borrow/carry bounds, canonical modular results, sparse-prime normalization,
  and Montgomery reduction cancellation.
- CUDA launch geometry, shared-scratch ownership, fixed-base table readiness,
  receipt buffer counting, base encodings and credential schema capacities.
- Python planner postconditions, multi-GPU partition/completion invariants,
  persistent mapping ledgers, and producer-registration retirement states.

Checks have no required side effects; Release does not evaluate C++/CUDA
assertion expressions. Additional diagnostic scans/helpers are also excluded
with `#ifndef NDEBUG`. Python uses the matching `ASSERTIONS_ENABLED` guard.
Not every possible bug is an assertable invariant, so these supplement—not
replace—differential tests, sanitizers, or review.

Input parsing, bounds/security validation, signature verification, and
fail-closed CUDA/IPC cleanup remain ordinary **unconditional** checks in every
configuration. An overflowing counting writer or empty BLAKE3 message is not
an internal bug. Zero is also a valid field element; the point-at-infinity
path intentionally permits the internal inverse-of-zero convention.

A host assertion terminates its process. A failed CUDA assertion poisons the
context; the existing cleanup/quarantine path must run, and that notary cannot
continue serving normally. Assertions print expressions/sites, not operand or
key values. Nevertheless, debug code changes secret-dependent control flow
and core dumps can contain secrets: use test keys/data, disable core dumps
where appropriate (`ulimit -c 0`), and deploy Release for production.

CUDA's [assertion behavior](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)
and [NVRTC options](https://docs.nvidia.com/cuda/nvrtc/#supported-compile-options)
define the device behavior. NVRTC supplies `__assertfail` but not a host libc
`assert.h`; the self-contained source provides the same NDEBUG-controlled
macro using that builtin, while offline NVCC uses the standard header.

## Test both configurations

The matrix runner builds separate packages and caches, verifies the imported
extension's real mode/path, runs the entire pytest suite, and preserves logs
and exit codes. It neither reinstalls packages nor overwrites the checkout's
in-place extension. Supply a **new** output directory:

```bash
# CPU matrix: Release AND AssertedRelease, with GPU tests explicitly skipped.
python tests/build_matrix.py /tmp/cuattest-cpu-matrix

# GPU/IPC regressions, independent CPU/GPU crypto, and both backend selftests.
python tests/build_matrix.py /tmp/cuattest-gpu-matrix --gpu --crypto

# Dedicated idle multi-GPU machine; near-capacity tests reserve almost all
# free VRAM on cuda:0. Do not use --large-capacity on a shared/busy GPU.
python tests/build_matrix.py /tmp/cuattest-multigpu-matrix \
  --gpu --crypto --crypto-devices all --multigpu --large-capacity

# Optional additional Debug qualification; slower and not a latency benchmark.
python tests/build_matrix.py /tmp/cuattest-debug-matrix --configuration Debug --gpu
```

GPU matrices prebuild each visible architecture before starting HTTP fixtures,
so slow asserted NVRTC compilation does not consume service-startup deadlines.
For a repeated matrix, `--reuse-artifacts /tmp/previous-matrix` copies only its
kernel/oracle caches into the new run; normal source/mode/digest checks still
apply. The Python package and native extension are always rebuilt.

The suite includes real native/device fault probes in disposable subprocesses,
mode-switch rebuild checks, cross-mode cache rejection, and verification that
the Python internal assertions follow the same policy. Deliberate assertion
failures are expected **only in those child probes**, never in the normal
test process or model-observation workload. Dependency setup is in [TEST.md](../TEST.md).

See the [2026-09-09 validation report](testing/assertions-2026-09-09.md) for
executed configurations, hardware coverage, artifact identities, and qualifications.

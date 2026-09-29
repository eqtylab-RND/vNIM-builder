# cuAttest sanitizer audit — 2026-09-08

## Outcome

No new cuAttest defect was confirmed in the exercised paths. All **32 primary
Compute Sanitizer invocations** passed, as did a separate near-capacity memcheck
run, the native sanitizer matrix, and the ASan/UBSan-instrumented Python extension
suite (**594 passed, 12 skipped**). A final ordinary six-GPU suite passed
**726 tests with no skips**.

This is not an unqualified "every tool was clean" result: optional pointer-pair
checking, raw Helgrind, whole-interpreter LSan, and CUDA unused-capacity checking
produced diagnostics. The evidence and qualifications are recorded below; those
runs are not counted as clean passes.

Revision: `b1863098e6b375ed13f97f4bcc70ec860758543e`.
The original worktree was clean before and after the sanitizer runs. Those runs
modified no tracked source, tests, documentation, or installed extension. Builds and audit
controls were made in isolated snapshots; cuAttest's normal writable CUBIN cache
was not used. No software installation, system-security change, or termination
of an existing user workload was needed.

## Environment and artifacts

| Item | Laptop | `probqa.com` |
| --- | --- | --- |
| GPU | RTX 2080, `sm_75`, 8 GiB | RTX PRO 6000 Blackwell Server Edition, `sm_120`, 97,887 MiB reported per GPU |
| Devices exercised | One GPU | Physical ordinals `0,1,2,3,4,6`; GPUs 5 and 7 excluded because they were busy at audit start |
| Driver | 610.43.02 | 610.57.04 |
| Compute Sanitizer | 2026.2.1.0, build 38334959 | Same |
| Python | 3.13.13, Conda | 3.10.12 |
| PyTorch | 2.14.0+cu130 | Same |
| Production CUBIN compiler | NVRTC 12.9 | NVRTC 13.3 |

Host instrumentation used Clang 21.1.8 and Valgrind 3.26.0. The MSan run reused
the previously built instrumented libc++/libc++abi at
`/tmp/cuattest-sanitizers.4cWFgp/msan-libcxx`; its build configuration still records
`LLVM_USE_SANITIZER=MemoryWithOrigins`. All native harnesses were freshly built
from the current source snapshot.

Local artifacts: `/tmp/cuattest-audit.PmCCep`.
Remote originals: `/tmp/cuattest-audit.X8RZd0`.
Remote CUDA, capacity, and unused-memory logs were copied locally into
`remote-cuda/`, `remote-capacity/`, and `remote-unused/`.
This report is versioned; the raw logs, controls, and build artifacts remain in
**temporary storage** and are not included in the repository. Unless otherwise
stated, relative artifact paths below are relative to the local artifact
directory above, not to `docs/testing/`.

The laptop's old prebuilt search location contained an older kernel. The cache
validator rejected it and compiled a fresh CUBIN in this audit's `cuda/cubins/`.
Both tested architectures' sidecars identify source BLAKE3
`6ee00b19f5541382285d3d1354a9752ddc4ebed905b80ce4f6a01c1cc1089d75`.
The source snapshot and original CUDA/C++ source SHA-256 values were also compared
and matched. The snapshot's native extension was rebuilt on each host, with
explicit `PYTHONPATH` selection to avoid an older editable installation.

## GPU results

| Workload | memcheck | initcheck | racecheck | synccheck |
| --- | --- | --- | --- | --- |
| RTX 2080: native and fallback selftests | Both pass | Both pass | Both pass | Both pass |
| RTX 2080: cryptographic differential suite | 24 pass | 24 pass | 24 pass | 24 pass |
| Blackwell GPU 0: native and fallback selftests | Both pass | Both pass | Both pass | Both pass |
| Blackwell GPU 0: cryptographic differential suite | 24 pass | 24 pass | 24 pass | 24 pass |
| Six Blackwell GPUs: real cross-process IPC/export suite | 22 pass | 22 pass, shared only | 22 pass | 22 pass |
| Six Blackwell GPUs: concurrent/repeated multi-GPU requests | 4 pass | 4 pass, shared only | 4 pass | 4 pass |

Every primary invocation returned zero. Memcheck reported zero errors and zero
leaked bytes/allocations. Racecheck reported zero hazards, errors, and warnings.
Initcheck and synccheck reported zero errors. Exact per-process results are in
`cuda/results.json` (local) and `remote-cuda/results.json` (remote).
The four selected multi-GPU cases deselect four deliberate error-path cases;
the complete suite was subsequently run without CUDA instrumentation.

The cryptographic corpus exercises 4,096 identical host/device 1-KiB blocks,
partial chunks, all 16 input alignments, tree boundaries, BLAKE3, SHA-256, HMAC,
P-256 arithmetic, RFC 6979, ECDSA, key generation, and signed receipts, comparing
against independent Python/library results. Each CUDA-instrumented differential
suite uses GPU 0; it is not 24 cases multiplied by six devices. The later ordinary
six-GPU suite explicitly selects `CUATTEST_CRYPTO_DEVICES=all`, giving 144
cryptographic cases within its 726 total tests.

Additional allocation-pressure check on idle Blackwell GPU 0:
near-capacity memcheck (`remote-capacity/both.log`), **2 passed**, zero errors and
zero leaked bytes. Both backends create a cold signer while a filler allocation
leaves only 1 GiB free. The hashed payload is tiny: this validates near-capacity
allocation/teardown, **not hashing the full large-model benchmark under a
sanitizer**.

The primary runner enables child-process instrumentation, allocation padding,
full leaks, cache-control checks, stream-ordered allocation races, informational
race hazards, deadlock detection, bulk-copy and tensor-operation checks. Strict
module-unload checking and all API errors are enabled for driver-only workloads.
PyTorch IPC runs report explicit API failures, retaining the checked-in runner's
qualification for handled implicit CUDA Runtime initialization probes. There
are no forced blocking launches, kernel filters, or new project suppressions.

Selftests and cryptographic tests check **both global and shared initialization**.
IPC workloads check shared memory only: NVIDIA explicitly documents that
initcheck does not support IPC allocations and can produce false positives.
[NVIDIA known limitations](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations)
The kernels do not use OptiX, device-side allocation, or tensor-core instruction
families; enabling related options does not create coverage of unused features.

## Native and Python results

Each standalone native run exercises the **verbatim current native core** with
an instrumented CUDA-driver double, not a second implementation. It includes
eight concurrent sessions, checked size arithmetic, allocation bounds, deduplication,
workspace reuse/growth, and injected allocation, launch, DMA, drain, output, and
IPC-close failures. Deferred mock DMA exposes premature host/device frees.
The harness also verifies explicit stream use and that the import gate is
released before completion synchronization. It does not emulate GPU cryptography.

| Check | Result / evidence |
| --- | --- |
| AddressSanitizer + UBSan + bounds + LeakSanitizer | Pass; stack use-after-scope/return checks enabled; `run-address.log` |
| UBSan + bounds/local-bounds + implicit conversions | Pass; `run-undefined.log` |
| Standalone LeakSanitizer | Pass; `run-leak.log` |
| ThreadSanitizer | Pass; `run-thread.log` |
| MemorySanitizer + origin tracking level 2 + instrumented libc++/libc++abi | Pass; `run-memory.log` |
| Integer sanitizer | Pass with the existing exclusion for unsigned wrap in system C++ headers only; `run-integer.log` |
| Cross-DSO Control Flow Integrity, LTO/LLD | Pass; `run-cfi.log` |
| TypeSanitizer, `-fsanitize=type` | Pass; both harness and driver double instrumented; `run-type.log` |
| HWAddressSanitizer | Pass; experimental x86 page-aliasing heap mode only; `run-hwaddress.log` |
| SafeStack | Hardening smoke pass; `run-safe-stack.log` |
| Scudo + GWP-ASan, sample rate 1 | Sampled allocator-hardening smoke pass; `scudo-gwp-asan.log` |
| libstdc++ debug/pedantic containers + FORTIFY 3 + strong stack protector | Pass; `debug-stl.log` |
| Valgrind Memcheck, full leaks and origins | Zero errors; all 12,651 allocations freed, zero bytes at exit; `valgrind-memcheck.log` |
| Valgrind DRD | Zero reported errors; standard system suppressions active; `valgrind-drd.log` |
| libFuzzer + ASan/UBSan, actual planner | 1,060,825 executions in 61 seconds, no finding; `fuzzer.log` |
| ASan/UBSan/bounds-instrumented Python extension plus CPU/real CUDA/IPC/crypto tests | **594 passed, 12 skipped**; `asan-extension-tests.log` |
| CPython debug allocator and `-X dev` | **535 passed, 71 opt-in skips**; `python-debug.log` |
| Full ordinary six-GPU suite, all crypto devices and near-capacity enabled | **726 passed, no skips**; `remote-ordinary-all-six.log` |
| Clang static analyzer, full `_native.cpp` including bindings | No diagnostics; `native.plist`, `clang-analyzer.log` |
| Ruff F,E9 and original `git diff --check` | Pass |

The ASan extension run uses `PYTHONMALLOC=malloc`, strict string checking, and
the process-local CUDA compatibility setting `protect_shadow_gap=0`. The loaded
extension path was verified to be the new instrumented copy. Interpreter-wide
LSan is disabled **only for that run**, not for the standalone native leak checks.

Standalone TSan/MSan/CFI/TypeSanitizer results do not claim instrumentation of
CPython, PyTorch, or proprietary NVIDIA driver internals. In particular, a normal
Python process with an MSan-only extension is not a meaningful full MSan audit;
instrumented dependencies/interceptors are required.
[LLVM MemorySanitizer documentation](https://clang.llvm.org/docs/MemorySanitizer.html#handling-external-code)
TypeSanitizer is an experimental strict-aliasing check, not another general
memory/race detector.
[LLVM TypeSanitizer documentation](https://clang.llvm.org/docs/TypeSanitizer.html)
NumericalStabilitySanitizer, RealtimeSanitizer and DataFlowSanitizer are not counted
as runs: this native integer/metadata code has no relevant floating-point or
annotated realtime/taint contract to validate with those tools.

## Diagnostics retained, not counted as clean passes

1. **Raw Helgrind: 127 errors in 20 contexts, exit 86.** The stacks and addresses
   point to the system `libgcc_s` exception unwinder's initialization state.
   An independent eight-thread throw/catch program containing no cuAttest code
   reproduced 352 errors in 12 contexts at the same runtime locations.
   A constructor preload that initializes only the system unwinder before the
   unchanged harness starts its threads gives zero reports. This narrows the
   diagnosis to runtime initialization/tool interaction; it does not establish
   whether libgcc or Helgrind is ultimately at fault. No project warm-up or
   suppression was added. Logs: `valgrind-helgrind.log`,
   `helgrind-control-throw.log`, `helgrind-warm-constructor.log`.

2. **Optional ASan invalid-pointer-pair checking: exit 1.**
   `detect_invalid_pointer_pairs=2` reports libstdc++ 15's
   `std::less<const char*>` in valid `std::string::assign` overlap handling.
   An independent two-byte vector-to-string assignment reproduces the same
   diagnostic without cuAttest. Ordinary ASan remains clean. Logs:
   `run-pointer-pairs.log`, `asan-control-pointer-pair-small.log`;
   control source: `control-pointer-pair.cpp`.

3. **Whole-interpreter LSan: exit 1, not a clean leak audit.**
   Empty Conda Python reports 153,983 bytes / 2,864 allocations in retained
   Python Unicode objects. With the native module imported, both zero and
   4,096 successful calls to `native.plan` report exactly 610,681 bytes /
   11,289 allocations with the same Python allocation stacks. The control shows
   no growth from those planner calls, not that every interpreter allocation is
   correctly retired. Logs: `python-lsan-empty.log`, `python-lsan-plan-0.log`,
   `python-lsan-plan-4096.log`. Native standalone LSan and Valgrind stay enabled
   and clean. Earlier exploratory `python-lsan-4096.log` used a nonexistent
   method; only the corrected `python-lsan-plan-*` probes support this result.

4. **CUDA `initcheck --track-unused-memory --unused-memory-threshold 0`: exit 86
   for each backend on each host.** Each selftest reports three unused one-byte
   global allocations: two empty-input sentinels and a secondary-reduction
   sentinel. These are accurately reported unused capacity, not uninitialized
   reads; both allocation helpers already explain why zero-byte requests use
   an unread/unwritten sentinel. The normal global/shared initialization checks
   are clean. Raw logs and exit codes: `unused/` and `remote-unused/`.

The previously documented Compute Sanitizer blocked-stream timing limitation
and deliberate failed-pinned-free diagnostic were not rerun as separate CUDA
probes in this audit. Their regressions were exercised by the ordinary/host-ASan
suites. CUDA instrumentation timings are not used to judge performance or
stream independence.

## Reproduction and scope

See the checked-in [sanitizer guide](../sanitizers.md) for reproduction instructions
and the [native](../../tests/sanitizers/build_native.py) and
[CUDA](../../tests/sanitizers/run_cuda.py) runners.
All `build-*.log` files retain the native compiler commands. CUDA command lines
are in `cuda-runner.log` and `remote-cuda-runner.log`. `run-extra.py` records the
supplementary unused-memory and capacity commands in their `results.json` files
and preserves nonzero exits. It is an audit-only file outside the repository.

The ASan extension suite was run from the source snapshot with:

```bash
DEBUGINFOD_URLS= \
ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0:strict_string_checks=1 \
UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
LD_PRELOAD=/usr/lib/llvm-21/lib/clang/21/lib/linux/libclang_rt.asan-x86_64.so \
PYTHONMALLOC=malloc \
PYTHONPATH=/tmp/cuattest-audit.PmCCep/asan-extension:/tmp/cuattest-audit.PmCCep/source/tests/sanitizers \
CUATTEST_KERNEL_DIR=/tmp/cuattest-audit.PmCCep/cuda/cubins \
CUATTEST_CACHE=/tmp/cuattest-audit.PmCCep/asan-cache \
CUATTEST_SANITIZER_CACHE=/tmp/cuattest-audit.PmCCep/cuda/oracle-cache \
CUATTEST_TEST_GPU=1 CUATTEST_TEST_CRYPTO=1 \
/home/serge/work/Upwork/Clients/JonathanDotan/cuattest/src/cuAttest/.venv/bin/python \
  -m pytest -q -p cache_oracle
```

Do not deploy sanitized extensions or the fake CUDA driver. Passing finite test
corpora is evidence for the exercised code paths, not proof of absence of
memory-safety, concurrency, or cryptographic defects.

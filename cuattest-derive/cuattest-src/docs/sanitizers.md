# Sanitizer audit and regression checks

For the latest full audit, see the
[2026-09-08 sanitizer report](testing/sanitizers-2026-09-08.md).
This guide retains earlier findings and reproduction instructions.

Audit date: **2026-09-07**. Starting revision:
`f717032d3486cbc611597545be1d8c48457241c5`, plus the fixes described below.
All results are for exercised paths, not a proof of absence of defects.

## Findings fixed

1. **Uninitialized shared-memory descriptor padding.** Global-only initcheck
   was clean, but `--initcheck-address-space all` reported 18 shared-memory
   reads per selftest in credential N-Quad sorting. A pointer plus a 32-bit
   length left four padding bytes that NVRTC copied with a 64-bit load.
   `CredentialNQuad.length` is now full-width, with a compile-time assertion
   that the structure contains no padding. No blanket scratch-memory clearing
   or sanitizer suppression was added. Hashes and signatures still match the
   independent CPU implementations.
2. **Missing explicit module retirement.** The optional strict module-lifetime
   check reported a module reclaimed only by context destruction. Healthy
   notaries now call `cuModuleUnload` before destroying their context. This
   was not a persistent VRAM leak: context destruction already reclaimed it.
   Unconfirmed GPU work still uses fail-closed context teardown, without
   unloading a possibly active module. Regression tests cover unload failure,
   context-stack preservation, uncertain work, and idempotent close. The
   cryptographic test fixture also explicitly retires its separate module.
   Confirmed context destruction is tracked independently of unload errors:
   an IPC failure can be acknowledged after successful destruction even when
   unloading raised. Failed or unattempted destruction remains fail-closed,
   including after a repeated, no-op `close()`.
   Native cleanup uncertainty is also latched on the Notary before its runner
   is closed or detached, and cleared only after successful context destruction.
   Fault-injection regressions ensure that failed destruction followed by an
   unwinding context-query error or exception-allocation failure still withholds
   HTTP headers, keeping producer storage quarantined.
3. **Cross-library CFI type identity.** The by-value CUDA IPC handle was an
   anonymous-namespace record, giving it a different C++ type identity in each
   instrumented translation unit. Cross-DSO CFI rejected the driver call even
   though the byte layout matched. The declaration now uses CUDA's external
   `CUipcMemHandle_st` tag, retaining the same 64-byte ABI. The instrumented
   driver double and native harness exercise that boundary directly.

Inline comments and tests explain these invariants. The sanitizer runner also
has CPU-only tests ensuring shared-memory checking remains enabled, child
processes are instrumented, failures cannot produce a successful final exit,
and a missing native extension cannot silently count as native coverage.
Cold-cache regressions exercise real CUBIN publication with a test compiler,
including an inherited cache override and a read-only normal cache.

## Environment and coverage

The laptop has an RTX 2080 (`sm_75`, 8 GiB), Python 3.13.13, Clang 21.1.8,
NVRTC 12.9, Compute Sanitizer 2026.2.1, and Valgrind 3.26.0. Remote validation
used `probqa.com`: eight RTX PRO 6000 Blackwell GPUs (`sm_120`, approximately
96 GiB each), Python 3.10.12, and NVRTC 13.3. Both hosts used NVIDIA driver
610.43.02 and PyTorch 2.14.0+cu130. Existing GPU workloads were left running.

LLVM utilities, LLD, libc++ development packages, and Valgrind were installed
on the laptop. A separate MemorySanitizer-instrumented libc++/libc++abi was
built from LLVM's `llvmorg-21.1.8` tag under the audit scratch directory. No
system security setting or production CUBIN cache was changed.

| Check | Workload | Result |
| --- | --- | --- |
| CUDA memcheck | Both backend selftests; 24 cryptographic differential cases on RTX 2080 | Pass; zero errors/leaks |
| CUDA initcheck, **global and shared** | Same selftests and all 24 differential cases | Pass; zero uninitialized reads |
| CUDA racecheck, including informational hazards | Same selftests and all 24 differential cases | Pass; zero hazards/warnings/errors |
| CUDA synccheck | Same selftests and all 24 differential cases | Pass; zero errors |
| All four CUDA tools on Blackwell | Both backend selftests on remote GPU 7 | All eight runs pass |
| CUDA memcheck across processes | All 22 PyTorch export cases on remote GPUs 4 and 5; both backends | Pass; zero errors/leaks after explicit test cleanup |
| CUDA racecheck, synccheck, shared initcheck across processes | Same 22 export cases, both backends | Pass; zero reports; global IPC initcheck is unsupported, as detailed below |
| ASan + UBSan + bounds | Instrumented native extension, full CPU and opt-in single-GPU tests | **430 passed, 32 skipped** |
| ASan + LSan | Native-core/driver-double harness, eight concurrent sessions | Pass; no leaks or invalid accesses |
| UBSan, bounds, implicit conversions | Same native harness | Pass |
| Integer sanitizer | Same harness; system-header unsigned-wrap exclusion below | Pass with that narrow exclusion |
| ThreadSanitizer | Same harness, independent concurrent notary sessions | Pass |
| MemorySanitizer + origin tracking | Same harness plus instrumented libc++ and libc++abi | Pass |
| Cross-DSO CFI | Native core and driver double, LTO enabled | Pass after ABI identity fix |
| SafeStack | Native harness | Pass |
| HWAddressSanitizer | Native harness, x86 page-aliasing heap checks | Pass |
| Valgrind Memcheck | Native harness | Zero errors; all 12,175 allocations freed |
| Valgrind Helgrind and DRD | Native harness | Zero reported races; standard system suppressions active |
| Clang static analyzer | Complete `_native.cpp`, including Python bindings | No diagnostics |
| libFuzzer + ASan + UBSan | Actual native planner, arbitrary encoded spans | 1,093,962 executions in 31 seconds; no finding |
| CPython debug allocator / development mode | CPU suite | 404 passed, 58 skipped |
| Ordinary eight-GPU integration | Full CPU/GPU/multi-GPU suite | **436 passed, 26 skipped** |
| Final combined laptop suite | CPU, real single-GPU integration, and cryptographic comparisons together | **454 passed, 8 skipped** |

The final laptop skips are six multi-GPU cases and two opt-in large-capacity
cases. Remote ordinary tests skip the 24 separately exercised crypto cases and
the two large-capacity cases. Near-VRAM-capacity model workloads were not run
under sanitizers; large size arithmetic is covered by the native planner
harness, but that is not equivalent to instrumenting a near-capacity allocation.

The CUDA differential suite covers 4,096 identical host/device 1-KiB blocks,
partial chunks, all 16 input alignments, tree boundaries, SHA-256, HMAC,
P-256 arithmetic, RFC 6979, ECDSA, key generation, and signed receipts. See
[the testing guide](../TEST.md#4a-independent-cpugpu-cryptographic-comparisons)
for its independent Python reference implementations.

The host harness extracts the **verbatim native planner and runner** from
`_native.cpp`; it does not maintain a second implementation. Its instrumented
driver double uses real host allocations/copies and injects allocation,
launch, synchronization, copy, kernel-status, output-length, and IPC-close
failures. Eight independent sessions exercise concurrent driver initialization,
planner overflow rejection, IPC deduplication, bounds, workspace growth/reuse,
and cleanup/quarantine. It does not execute or emulate GPU cryptography.

The standalone TSan/MSan/CFI runs do **not** cover the CPython binding layer,
real NVIDIA driver internals, or PyTorch internals. Bindings and real client
exports are instead exercised by the ASan/UBSan extension run and real CUDA
integration. MSan requires instrumented dependencies, which is why an ordinary
Python process with an instrumented extension alone is not a meaningful MSan
pass. [LLVM MemorySanitizer documentation](https://clang.llvm.org/docs/MemorySanitizer.html)

## Multi-GPU dispatch follow-up (2026-09-08)

The [eight-GPU performance change](performance.md#concurrent-eight-gpu-attestation-2026-09-08)
was checked with the following additional runs:

| Check | Scope | Result |
| --- | --- | --- |
| CUDA memcheck, racecheck, synccheck, shared initcheck | Four selected cross-process multi-GPU cases per tool; both backends; all eight Blackwell GPUs | Zero errors; zero race hazards; final memcheck zero leaked bytes |
| ASan/LSan, UBSan, TSan, MSan with instrumented libc++ | Current verbatim native core and driver double; eight concurrent sessions | Pass, including a barrier proving the import gate is released before GPU synchronization |
| ASan/UBSan extension | Final CPU suite plus real RTX 2080 IPC and crypto cases | 503 passed, 12 multi-GPU/capacity skips |
| Ordinary eight-GPU suite | CPU, all GPU/IPC cases, 192 crypto cases, near-capacity checks | 679 passed, no skips; four later-added runner CPU cases also pass |

The first multi-GPU memcheck run reported 16 MiB retained in eight PyTorch
allocator-cache blocks, with allocation stacks originating in test tensors.
Module teardown now synchronizes producer streams and releases only unused
framework cache; it never force-retires live or quarantined IPC leases. The
repeat passed with zero leaked bytes. Initial diagnostics remain available,
not suppressed or overwritten.

Remote CUDA logs are under
`/tmp/cuattest-blackwell-opt.2U5S32/multigpu-sanitizers`; the clean memcheck repeat
is under `/tmp/cuattest-blackwell-opt.2U5S32/multigpu-memcheck-final`.
Local instrumented builds are under `/tmp/cuattest-opt-work.aVFIyY`.
These CUDA sanitizer cases use small tensors, not the full 739 GiB benchmark.
Global-memory IPC initcheck remains unsupported as described below; shared
initcheck, kernel memory access, synchronization, and race checks are covered.

## Private-stream follow-up (2026-09-08)

The [private-stream change](performance.md#private-cuda-streams-2026-09-08)
adds pinned asynchronous transfers and reusable fallback workspaces. It was
checked with the following updated suites:

| Check | Scope | Result |
| --- | --- | --- |
| Ordinary CPU suite | Stream ABI, DMA/drain failures, keygen scrubbing, failed growth, cleanup interruption and existing regressions | 488 passed, 66 hardware/opt-in skips |
| Ordinary eight-GPU suite | Both backends, all GPU/IPC cases, 192 crypto cases, stream isolation and near-capacity cases | 722 passed, no skips |
| ASan/UBSan extension | CPU suite plus real RTX 2080 IPC, crypto and stream-isolation tests | 542 passed, 12 multi-GPU/capacity skips |
| ASan/LSan, UBSan with implicit conversions, TSan, MSan with instrumented libc++ | Verbatim native core; eight concurrent sessions; deferred DMA driver double | All passed |
| CUDA memcheck, global/shared initcheck, racecheck, synccheck | Each tool: both backend selftests plus four selected eight-GPU IPC cases | All 12 invocations passed; zero errors/hazards; memcheck zero leaked bytes. IPC initcheck uses shared memory only |

The driver double rejects context-wide synchronization and verifies the
explicit stream on both kernel launches and copies. It now defers transfers
until a successful stream drain or simulated context destruction, exposing
premature host/device frees to sanitizers. Both backends preserve quarantine
on failed completion; keygen seed scrubbing also runs in the owning stream.
`cuMemAllocHost` allocations are context-owned, so teardown abandons unsafe
staging to context destruction instead of freeing it during unwinding or
freeing it again afterwards. This follows NVIDIA's
[context resource-lifetime contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html).

The live isolation tests use a host-released stream memory wait with a watchdog,
not a competing spin kernel. They prove signed direct-span requests complete
while a different stream remains blocked, including legacy stream 0, and
verify the resulting receipts against independent CPU hashes.

Local instrumented builds are retained under
`/tmp/cuattest-stream-work.rlxaae`. The extension run disables interpreter-wide
LSan as qualified above; the standalone native ASan/LSan harness does not.
Remote CUDA logs are under `/tmp/cuattest-streams.nGAaw2/cuda-sanitizers`.
Reproduce those CUDA suites with the existing runner:

```bash
CUATTEST_KERNEL_DIR=/path/to/validated/cubins \
  .venv/bin/python tests/sanitizers/run_cuda.py /path/to/new/audit-output \
  --suite selftest --suite multigpu
```

An additional **instrumented isolation assertion failed**, distinct from the
clean memory/race checks: memcheck made all four deliberate blocked-stream
tests wait for their watchdogs (zero reported memory errors or leaked bytes).
The standalone [CUDA-only reproducer](../tests/sanitizers/stream_isolation_probe.cu)
contains no cuAttest, Python, PyTorch, IPC, or cooperative kernel. On the RTX
2080 (driver 610.43.02, Compute Sanitizer 2026.2.1), its independent stream
completed in 8 microseconds normally, leaving the
other stream blocked; under memcheck it took 3.000080 s until the watchdog
released the other stream. This reproduces instrumentation-dependent loss of
isolation independently of the application. Do not use memcheck to judge
concurrency or benchmark performance; run the isolation regressions normally
(also passed with host ASan/UBSan), and use the sanitizer suites above for
memory/race correctness. The failing assertion and tool logs are retained,
not suppressed or counted as passing.

```bash
# sm_75 is the tested RTX 2080; select the appropriate architecture elsewhere.
nvcc -std=c++17 -arch=sm_75 tests/sanitizers/stream_isolation_probe.cu -lcuda \
  -o /path/to/audit-output/stream-isolation-probe
/path/to/audit-output/stream-isolation-probe
compute-sanitizer --tool memcheck --error-exitcode 86 \
  /path/to/audit-output/stream-isolation-probe
```

The probe returns 0 if isolation holds, 3 if the other stream was drained.
On the tested tool/driver combination memcheck reported zero memory errors
but the instrumented probe returned 3.

## Producer ordering and teardown review follow-up (2026-09-08)

Two regressions after `004f48e` were fixed. Public `hash_dptr()` now queues a
producer-event dependency before either backend reads direct input, defaulting
to the legacy stream and accepting an explicit same-context producer stream.
IPC/export-ready internal calls retain their private-stream isolation. A failed
pinned-buffer free no longer skips stream/module/context destruction; remaining
wrappers are detached before CUDA reclaims their allocations. Destruction
success, not an earlier free/unload exception, controls IPC acknowledgement.

Validation of these fixes:

- CPU default suite: **535 passed, 71 opt-in skips**.
- RTX 2080 CPU/GPU/crypto suite: **594 passed, 12 multi-GPU/capacity skips**,
  both normally and with an ASan/UBSan/bounds-instrumented native extension.
  Interpreter-wide LSan remains disabled for that extension run as qualified
  below; the standalone ASan/LSan harness keeps leak checking enabled.
- `probqa.com`, all eight Blackwell GPUs, including multi-GPU and near-capacity
  cases: **774 passed, no skips**.
- Both backend selftests under memcheck, global/shared initcheck, racecheck and
  synccheck on both hosts: **16 successful invocations**, zero memory errors,
  leaks or race hazards. The selftests exercise the public direct-pointer path.
- Verbatim native-core harness: ASan/LSan/UBSan/bounds, TSan, and MSan with
  instrumented libc++ passed, including eight concurrent sessions. Clang static
  analysis and Ruff passed.

The new live ordering tests gate producer writes and inspect the pending
private-stream dependency before releasing the writer; they do not rely on a
timing threshold. They pass normally and under host ASan/UBSan. Replacing only
`hash_dptr()` with its original implementation makes both legacy-write cases
fail with `missing producer dependency`. The new CPU cleanup regression also
fails against the original `close()`.

One **deliberate error-path** memcheck run is not counted as clean:
`test_real_fallback_context_is_destroyed_after_pinned_free_error` passed its
assertions, but `--leak-check full` reported the two intentionally unfreed
staging buffers (69 + 68 bytes), returning 86. A raw `ctypes`/CUDA control with
no cuAttest code (`cuCtxCreate_v2`, `cuMemAllocHost_v2(137)`, successful
`cuCtxDestroy_v2`, no explicit free) likewise reported 137 bytes and returned 86.
Memcheck reports allocations not explicitly freed **before** context
destruction, as described in NVIDIA's
[leak-check definition](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#leak-checking).
The driver's [context-destruction contract](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html)
reclaims `cuMemAllocHost` allocations; this diagnostic does not establish a
surviving allocation after successful destruction. Retrying an ambiguous free
or freeing those pointers after destruction would risk a double free. These
diagnostics remain unsuppressed and separate from healthy-path leak checks.

Artifacts: `/tmp/cuattest-stream-fixes.NuOyXo` on the laptop and
`/tmp/cuattest-stream-fixes.e76fFL` on `probqa.com`; selftest logs are under
`cuda/`, and the two deliberate omitted-free reports are
`pinned-cleanup-memcheck.log` and `raw-context-host-memcheck.log` locally.
These are temporary paths. The stream-isolation instrumentation limitation
described above still applies; CUDA-instrumented isolation assertions were not
rerun as part of this follow-up.

## Reports that were not cuAttest defects

- `initcheck --track-unused-memory` reports three one-byte allocations in the
  selftest: two empty-input sentinels and an unused secondary-reduction buffer.
  CUDA rejects zero-byte allocations; these sentinels are deliberately never
  read or written. Comments at both allocation helpers record this distinction.
  The unused-capacity scan exits 86; ordinary global/shared initcheck is clean.
- LSan on this uninstrumented Conda Python build reports retained Unicode
  allocations even without importing cuAttest: 154,447 bytes / 2,873 allocations.
  With the native module imported, **zero and 4,096 planner calls both reported
  exactly 601,534 bytes / 11,118 allocations**, with the same Python stacks.
  This is not a clean whole-interpreter LSan result. Extension ASan tests use
  `detect_leaks=0`; the standalone native LSan and Valgrind runs keep leak checks
  enabled and free everything.
- Optional ASan pointer-pair checks report libstdc++ 15's `std::less<const char*>`
  during a valid `std::string::assign`. A standalone program containing only
  `std::vector<char>` and `std::string` reproduces it without any cuAttest code.
  These optional checks are not counted as a clean pass; normal ASan is clean.
- The integer sanitizer initially reported deliberate unsigned loop/hash wrap
  in libstdc++ headers. `tests/sanitizers/integer.ignore` excludes **only**
  `unsigned-integer-overflow` in system C++ headers. All project integer checks
  and other undefined-behavior checks remain enabled.
- `--report-api-errors all` also reports CUDA Runtime's handled
  `cuCtxGetDevice_v2` / `CUDA_ERROR_INVALID_CONTEXT` initialization probes in
  PyTorch. A tiny PyTorch-only program reproduces seven reports without
  importing cuAttest. IPC runs use `--report-api-errors explicit`, retaining
  every explicit API failure; raw diagnostics remain available with
  `--api-errors all`. That API-reporting choice does not change kernel memory,
  race, or synchronization checking.
- **Global initcheck does not support CUDA IPC allocations.** NVIDIA documents
  this limitation. A standalone driver-only producer initialized and synchronized
  4,096 bytes; its child copied every byte back correctly, yet initcheck reported
  128 uninitialized reads. The two-GPU IPC suite likewise passed all 22 tests but
  produced 1,728 global-initcheck reports. The IPC runner therefore checks shared
  memory by default; `--ipc-initcheck-address-space all` reproduces the unsupported
  global check. Full global/shared initialization checking remains enabled for
  the selftests and differential suite. No producer memory is modified to hide
  the diagnostic.
  [NVIDIA's known limitations](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations)
- PyTorch export tests originally retained allocator caches until process exit.
  Explicitly dropping test tensors and clearing unused GPU cache blocks reduced
  the two-GPU leak report from 260,046,984 bytes to 136 bytes. The remaining
  pinned scalar-return cache also appeared in a PyTorch-only `torch.equal`
  reproduction. The export-only regression now compares every downloaded value
  on CPU, avoiding that unrelated GPU scalar operation; allocation leak checks
  stay fully enabled. The final 22-case IPC memcheck run reported **zero errors
  and zero leaked allocations**. Neither cleanup changes the production exporter.
- ASan's default shadow-gap protection conflicts with CUDA virtual-address
  initialization on this laptop (`cuInit` returned out-of-memory). The
  **process-local** option `protect_shadow_gap=0` allowed CUDA to initialize.
  Ordinary ASan shadow/redzone instrumentation remained enabled.
- HWASan on this x86 host checks the heap using experimental page aliasing,
  not AArch64-style stack/global tagging. The standalone harness does not
  fork, as required for this mode.
  [LLVM HWASan design](https://clang.llvm.org/docs/HardwareAssistedAddressSanitizerDesign.html)

## Reproduce CUDA checks

Run from the repository root, with the native extension installed/built and
the `test-crypto`, `client`, and `verify` dependencies available:

```bash
sanitize_dir=$(mktemp -d -t cuattest-sanitizers.XXXXXX)
# Build this checkout's extension, even if the venv has another installed copy.
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python tests/sanitizers/run_cuda.py "$sanitize_dir" \
  --suite selftest --suite crypto

# Real PyTorch producer, separate HTTP notary, both host backends.
# Every visible GPU is exercised, with reversed consumer device visibility.
CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python tests/sanitizers/run_cuda.py \
  "$sanitize_dir/ipc" --suite ipc

# Concurrent shards, repeated fresh partial tiles, every visible GPU.
.venv/bin/python tests/sanitizers/run_cuda.py \
  "$sanitize_dir/multigpu" --suite multigpu
```

The runner serializes all four tools, saves individual logs and `results.json`,
continues collecting results after a failed tool, and exits nonzero if any run
failed. Select a smaller run with, for example, `--tool initcheck --suite selftest`.
Selftests explicitly require the native extension for their native iteration;
the crypto and IPC fixtures verify which backend actually ran. The multi-GPU
suite requires at least two visible GPUs and covers both backends; it excludes
intentional invalid-driver-call cases that memcheck would correctly report as
API errors. Run the complete uninstrumented multi-GPU suite for those cases.

Child processes put this checkout's `src` first on `PYTHONPATH`, preventing a
shared venv's stale editable installation from silently supplying another
implementation. Build the current native extension in place before an audit.

The runner always sets its child processes' writable `CUATTEST_CACHE` to the
output directory, overriding any inherited value; new CUBINs and sidecars go
in `output/cubins`. An explicit `CUATTEST_KERNEL_DIR` is preserved as a read-only
prebuilt search location, but a cache miss still writes under the output
directory. The parent process's environment and normal cache are unchanged.
The test-only oracle cache includes source/adapters, architecture, compiler
version, and an artifact checksum; it never supplies a production module. Use
separate output directories for concurrent invocations. The first oracle
compilation can take several minutes. Set `CUATTEST_CRYPTO_DEVICES=all` to run
the differential corpus on every visible GPU, rather than the default GPU 0.

Memcheck enables allocation padding, full allocation leak checks, cache-control
access checks, and stream-ordered allocation race checks. Global and shared
initcheck are both enabled for non-IPC workloads; IPC uses shared-memory checking
because of the documented tool limitation above. Racecheck includes informational hazards and a
30-second deadlock timeout. Bulk-copy and tensor-operation checks are enabled;
this code does not use OptiX or tensor-core instructions, so enabling those
checks does not imply exercising those instruction families.

Strict module-unload checking is enabled only for driver-only selftest/crypto
processes: NVIDIA says not to enable it for CUDA Runtime clients such as
PyTorch. No forced blocking launches or kernel filters are used.
[NVIDIA Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)

## Reproduce native checks

With Clang and its sanitizer runtimes installed, build and run each harness
separately. `DEBUGINFOD_URLS=` avoids a symbolizer waiting on network debuginfo:

```bash
set -e
for check in address undefined leak thread integer cfi safe-stack hwaddress none; do
  .venv/bin/python tests/sanitizers/build_native.py "$check" "$sanitize_dir/$check"
  DEBUGINFOD_URLS= ASAN_OPTIONS=detect_leaks=1 \
    UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
    TSAN_OPTIONS=halt_on_error=1 HWASAN_OPTIONS=halt_on_error=1 \
    "$sanitize_dir/$check/harness"
done
```

CFI needs LLD and LTO. `--cxx clang++-21 --linker lld-21` selects versioned
tools explicitly. SafeStack is a hardening smoke test, not a race or leak
detector. Sanitized runtimes and the mock `libcuda.so.1` are **test-only**;
do not deploy them with the notary.

For MSan, build instrumented libc++ and libc++abi first:

```bash
git clone --depth 1 --branch llvmorg-21.1.8 --filter=blob:none --sparse \
  https://github.com/llvm/llvm-project.git "$sanitize_dir/llvm-project"
git -C "$sanitize_dir/llvm-project" sparse-checkout set \
  runtimes libcxx libcxxabi cmake llvm/cmake llvm/utils/llvm-lit llvm/utils/lit libc
cmake -G Ninja -S "$sanitize_dir/llvm-project/runtimes" \
  -B "$sanitize_dir/msan-libcxx-build" \
  -DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi' -DLLVM_USE_SANITIZER=MemoryWithOrigins \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="$sanitize_dir/msan-libcxx" \
  -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF -DLIBCXX_INCLUDE_TESTS=OFF \
  -DLIBCXXABI_INCLUDE_TESTS=OFF -DLIBCXXABI_USE_LLVM_UNWINDER=OFF
cmake --build "$sanitize_dir/msan-libcxx-build" --parallel 8
cmake --install "$sanitize_dir/msan-libcxx-build"
.venv/bin/python tests/sanitizers/build_native.py memory "$sanitize_dir/memory" \
  --msan-libcxx "$sanitize_dir/msan-libcxx"
DEBUGINFOD_URLS= MSAN_OPTIONS=halt_on_error=1 "$sanitize_dir/memory/harness"
```

Valgrind checks use the uninstrumented standalone harness:

```bash
valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all \
  --track-origins=yes --error-exitcode=86 "$sanitize_dir/none/harness"
valgrind --tool=helgrind --error-exitcode=86 "$sanitize_dir/none/harness"
valgrind --tool=drd --error-exitcode=86 "$sanitize_dir/none/harness"
```

For planner fuzzing, use the generated core and the current Python headers:

```bash
python_include=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
clang++ -std=c++17 -O1 -g -fno-omit-frame-pointer \
  -fsanitize=fuzzer,address,undefined -I"$python_include" -I"$sanitize_dir/none" \
  tests/sanitizers/planner_fuzz.cpp -ldl -pthread -o "$sanitize_dir/planner_fuzz"
DEBUGINFOD_URLS= ASAN_OPTIONS=detect_leaks=1 "$sanitize_dir/planner_fuzz" \
  -max_total_time=30 -max_len=2048 -timeout=5 -artifact_prefix="$sanitize_dir/"
```

## Reproduce extension ASan/UBSan checks

Build into an isolated package copy, not over a shared live extension:

```bash
mkdir -p "$sanitize_dir/asan"
cp -a src/cuattest "$sanitize_dir/asan/cuattest"
python_include=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
extension_suffix=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
clang++ -std=c++17 -O1 -g -fPIC -shared -fno-omit-frame-pointer \
  -fno-optimize-sibling-calls -fsanitize=address,undefined,bounds \
  -fno-sanitize-recover=all -fsanitize-address-use-after-return=always \
  -fsanitize-address-use-after-scope -shared-libasan -I"$python_include" \
  src/cuattest/_native.cpp -ldl \
  -o "$sanitize_dir/asan/cuattest/_native$extension_suffix"
asan_runtime=$(clang++ -print-file-name=libclang_rt.asan-x86_64.so)
DEBUGINFOD_URLS= ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0:strict_string_checks=1 \
  UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 LD_PRELOAD="$asan_runtime" \
  PYTHONMALLOC=malloc PYTHONPATH="$sanitize_dir/asan" \
  CUATTEST_KERNEL_DIR="$sanitize_dir/cubins" CUATTEST_TEST_GPU=1 \
  .venv/bin/python -m pytest -q
```

The runtime filename above is x86-64-specific. Keep the interpreter-wide LSan
qualification described earlier; do not describe this `detect_leaks=0` run as
a leak check. Run static analysis and Python's allocator diagnostics separately:

```bash
clang++ --analyze -std=c++17 -I"$python_include" src/cuattest/_native.cpp \
  -o "$sanitize_dir/native.plist"
PYTHONMALLOC=debug .venv/bin/python -X dev -m pytest -q
```

Audit logs and build artifacts were retained at
`/tmp/cuattest-sanitizers.4cWFgp` on the laptop and
`/tmp/cuattest-sanitizers.UDdB3y` on `probqa.com`. These scratch paths are not
durable storage; the checked-in runners and this report are the reproducible
record. Optional pointer-pair, interpreter-leak, unused-capacity, and raw
PyTorch API reports are retained alongside the clean scoped runs.

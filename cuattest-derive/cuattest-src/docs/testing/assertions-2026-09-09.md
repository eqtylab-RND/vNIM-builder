# Build-configuration and assertion validation — 2026-09-09

The [build guide](../build-configurations.md) describes the implementation and
reproduction commands. Validation uses separately compiled native extensions,
mode-specific CUBINs, and matching Python assertion policies; this is not the
same Release binary run twice with different labels.

## Local laptop

RTX 2080 (`sm_75`), driver **610.57.04**, NVRTC **12.9**, Python **3.13.13**.
The post-reboot CUDA path is healthy. The normal desktop workload was retained.

| Check | Release | AssertedRelease | Debug |
| --- | --- | --- | --- |
| CPU suite | 619 passed, 94 opt-in skips | 619 passed, 94 opt-in skips | 619 passed, 94 opt-in skips |
| Full suite with GPU/IPC enabled | 669 passed, 44 skips | 669 passed, 44 skips | Not run as a full GPU suite |
| Native GPU selftest | Pass | Pass | Pass |
| Python-fallback GPU selftest | Pass | Pass | Pass |
| Native invalid-invariant subprocess | Returns normally, assertion removed | Expected SIGABRT | Expected SIGABRT |
| Device invalid-chunk subprocess | Returns normally, assertion removed | Expected device assertion | Debug selftest uses enabled assertions, no invalid-device probe |

The remaining 44 local GPU-suite skips are the independent crypto opt-in,
multi-GPU and near-capacity suites; they are covered separately on probqa.com.
The default CPU run explicitly disables all GPU opt-ins. Intentional faults
run only in disposable children, with native core dumps disabled, and are
checked for the expected assertion expression/exit status.

Additional checks:

- Release and AssertedRelease wheels were built, installed into separate
  temporary targets, and imported without a build-mode environment variable.
  Both preserved the correct native and Python policy.
- Reusing the same native object/output directory for
  Release → AssertedRelease → Debug → Release rebuilt the extension correctly.
  Both setuptools and its compiler instance must have `force=True`.
- Cross-mode CUBINs, missing/altered policy sidecars, inconsistent native/wheel
  configuration, and optimized Python interpreters in asserted modes are
  rejected. All three explicit `build-kernel --build-type` choices are tested.
- The exact assertion-enabled native core passed Clang 21 ASan/LSan/UBSan/bounds
  and TSan harnesses, eight concurrent sessions each, without diagnostics.
- AssertedRelease's native GPU selftest passed Compute Sanitizer **synccheck**
  with `ERROR SUMMARY: 0 errors`. This is a targeted synchronization check,
  not a rerun or blanket clean claim for every sanitizer/dependency.
- Ruff `F,E9`, compileall and `git diff --check` passed.

## Artifact identity and Release cost

CUDA source SHA-256:
`31b54a6615d2ff0f40e9ccf704f52ee80809eaf0ef34f63a06c963c3075c15bf`.

| NVRTC 12.9 / sm_75 | CUBIN bytes | SHA-256 |
| --- | ---: | --- |
| Release | 6,101,576 | `b530beba62ee72b036c35048f3e3a0af123319573b852186386e53a368717cdb` |
| AssertedRelease | 32,083,640 | `4b994aaeb889a80c2108e8dcd222d06df866a59a26478342f7fd50518efdf427` |
| Debug | 16,159,616 | `e4f11a2c40cb9d8cbefb438bcbeeec904c7b68f57b94b4d7c7afe57735263770` |

Release's CUDA binaries on **both architectures** are byte-for-byte identical
to the pre-assertion optimized binaries recorded in the
[previous validation](registered-ipc-2026-09-09.md).
Neither contains an `__assertfail` reference. This establishes no device-code cost
for the new Release assertions, not an unmeasured end-to-end latency claim.
The source identity changed, and old sidecars without build-policy metadata
are intentionally cache misses. Asserted/debug builds are substantially
larger and are not recommended for production secrets or latency comparison.

## Extended eight-GPU matrix

probqa.com: eight RTX PRO 6000 Blackwell Server Edition GPUs (`sm_120`),
driver **610.57.04**, NVRTC **13.3**, Python **3.10.12**.

| Final configuration | Full pytest suite | Native selftest | Fallback selftest |
| --- | --- | --- | --- |
| Release | **937 passed, zero skips** (73.46 s) | Pass | Pass |
| AssertedRelease | **937 passed, zero skips** (115.08 s) | Pass | Pass |

Both runs enable real GPU/IPC, all-device independent cryptographic tests,
multi-GPU routing and near-capacity tests, with async hashing forced. This
includes **256 cryptographic differential cases per configuration** across
all eight GPUs, repeated registrations and reversed consumer ordinals, and
both cold-signer backends with only 1 GiB free VRAM. The full 739-GiB pretrained
model performance benchmark was **not** rerun; these are correctness checks.

| NVRTC 13.3 / sm_120 | CUBIN bytes | SHA-256 |
| --- | ---: | --- |
| Release | 9,247,216 | `1ad989b55026705cd135c34552efd00aa3918ba931f55a7b5b94133d7f303e5e` |
| AssertedRelease | 43,621,104 | `abf85aec1070b205fbe1615f5b508ed42926c59723c346968c55b869b390562c` |

AssertedRelease contains the expected `__assertfail` reference. Debug was
qualified on the laptop, not as an eight-GPU configuration.

The initial runs each completed **929 checks** but failed one CPU fixture:
the fixture emulated `sm_75` while inheriting the deliberately forced async
mode. The fixture now explicitly selects standard hashing; it continues to
check independent source/CUBIN hashing without depending on the physical
GPU's test mode. Those initial runs are retained as failed, not relabeled clean.
The final matrices rebuild the host packages and reuse only the earlier
digest-verified, mode-specific kernel/oracle caches. Their CUDA source and
compiler policies are unchanged. Prebuilding kernels also keeps expensive
NVRTC compilation outside HTTP fixture startup deadlines. The final cache,
ordering and failure-propagation runner regressions pass in both modes.

## Reproduction and logs

```bash
python tests/build_matrix.py /tmp/new-local-matrix --gpu

# Dedicated idle Blackwell host only; all visible GPUs are exercised.
python tests/build_matrix.py /tmp/new-blackwell-matrix \
  --gpu --crypto --crypto-devices all --multigpu --large-capacity --hash-mode async
```

Add `--reuse-artifacts /tmp/previous-matrix` when matching CUDA artifacts have
already been compiled; cache misses still compile and the host always rebuilds.

Local artifacts/logs: `/tmp/cuattest-asserts.zSeq0J`; remote:
`probqa.com:/tmp/cuattest-asserts.JxMJgE`. Each host's `final-matrix/` contains
both build/identity/kernel/pytest/selftest logs and machine-readable phase exit
codes. Local `cpu-*-final.log` files contain the final three-mode CPU reruns.
The independent forced-async CPU reruns exercise the fixture-isolation fix.
The checkout's in-place extension remains **Release**; matrix builds, wheels,
and caches are isolated from it.
All owned remote test/service processes exited and their GPU memory was
released. No driver changes or unrelated-process termination were required.

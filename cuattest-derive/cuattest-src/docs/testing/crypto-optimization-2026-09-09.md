# Cryptographic optimization validation — 2026-09-09

This report qualifies the arithmetic/table-scan change described in
[the performance results](../performance.md#arithmetic-and-critical-path-experiments-2026-09-09).
The baseline was `4708698`; final CUDA source SHA-256 is
`638ee4bf4209a262c304eb6f9798c3be054c1613f034295a85c794b344648a5b`.
Both host implementations use the same new CUDA primitives. Native host code,
stream/IPC lifecycle behavior, BLAKE3 and receipt serialization are unchanged.

## Functional results

| Check | Result |
| --- | --- |
| Ordinary CPU suite | **546 passed, 79 opt-in/hardware skips** |
| RTX 2080 full suite, GPU + crypto enabled | **613 passed, 12 multi-GPU/capacity skips** |
| Eight Blackwell GPUs, GPU + crypto + multi-GPU + capacity enabled | **849 passed, no skips** |
| Separate cryptographic suite, both backends | **32 passed on each architecture** |
| Ruff `F,E9`, compileall, whitespace/diff checks | Pass |
| Matched model A/B | Every accepted receipt verified; every root matched independent CPU BLAKE3 |

The eight-GPU run selects `CUATTEST_CRYPTO_DEVICES=all`: its total includes
256 cryptographic test cases, both host backends, reversed producer/notary
device ordinals, fresh partial-tile data, concurrent shard requests, and the
near-capacity cold-signer tests. The real 739.102-GiB pretrained benchmark
was separately verified while resident on all eight GPUs; it is not the tiny
payload used by the capacity regression.

New deterministic coverage includes:

- Symbolic execution of the **actual CUDA inversion schedules**, proving
  `p-2` and `n-2`, including the field chain's final multiply and the order
  chain's exact Montgomery conversion constant.
- Bounds derived from the **actual sparse-reducer coefficients**, covering
  every 512-bit input and proving three normalization passes sufficient.
- Device reduction of all 65,536 zero/max assignments of sixteen 32-bit
  words, plus 2,048 random wide values, against Python `%`; output aliasing
  the input is tested too.
- Montgomery multiplication modulo both field prime and group order,
  compared with Python `a*b*pow(2**256, -1, modulus)`, with disjoint output
  and aliases of either operand.
- 201 exact RFC 6979/ECDSA/public-point vectors, tested with independent
  signing lanes and cooperative eight-lane groups. Adjacent groups have
  different inputs; a partial final warp catches overbroad shuffle masks.
- CPU checks for table partitions and a source-hash guard preventing
  archived experiments from silently benchmarking edited baseline code.

Existing BLAKE3/SHA-256/HMAC vectors, 4,096 identical host/device 1-KiB
blocks, partial inputs, tree boundaries, aliasing, signed receipts, and
stream/IPC failure-path coverage remain enabled. Known private scalars and
nonces are confined to the test adapters; production exports no new entry
points or timing globals.

## Sanitizers

On the RTX 2080 and Blackwell GPU 0, **all 24 invocations passed** (12 per host):
native and fallback selftests plus the 32-case differential suite under each
of memcheck, initcheck, racecheck and synccheck. Memcheck reported zero errors
and zero leaked bytes/allocations; initcheck and synccheck zero errors;
racecheck zero hazards, errors and warnings. Both exit codes and diagnostic
summaries were checked.

| Differential suite, both backends | RTX 2080 | Blackwell GPU 0 |
| --- | --- | --- |
| memcheck | 32 passed; 0 errors/leaks | 32 passed; 0 errors/leaks |
| initcheck, global + shared | 32 passed; 0 errors | 32 passed; 0 errors |
| racecheck, informational hazards enabled | 32 passed; 0 hazards | 32 passed; 0 hazards |
| synccheck | 32 passed; 0 errors | 32 passed; 0 errors |

The runner enables strict driver API errors, module-unload checking, full
leaks, allocation padding, stream-ordered race checks, informational race
hazards, deadlock detection, and both global/shared initialization checks.
No blocking-launch mode, kernel filters, or new suppressions were added.
These instrumented suites use **GPU 0 on each host**, not all eight GPUs at
once; all-device coverage above is the ordinary, uninstrumented suite.

A separate Clang 21.1.8 **ASan + UBSan** harness compiled the experimental
three-/four-pass sparse reducers as C++, compared them with Boost `cpp_int`,
and passed all 65,536 limb corners plus **1,000,000 random 512-bit values**.
Sanitizer recovery was disabled. Its proven initial quotient interval was
`[-4,4]`, with signed intermediate bound `38,654,705,655`. The final reducer
uses the tested three-pass form; its production source is independently
covered by the CPU bound proof and device differential tests.

This is not a new whole-codebase TSan/MSan/CFI audit. No native host C++ was
changed; see the [earlier broad sanitizer audit](sanitizers-2026-09-08.md)
for those tools and their qualifications. CPU sanitizers cannot establish
CUDA warp correctness, and neither these tests nor constant-work source
structure constitute a formal GPU side-channel proof.

## Reproduce and inspect

Install the existing test-crypto dependencies, NVRTC/CUDA headers, and Compute
Sanitizer as described in [TEST.md](../../TEST.md) and the
[sanitizer guide](../sanitizers.md). Use isolated writable cache directories:

```bash
audit_dir=$(mktemp -d /tmp/cuattest-crypto-audit.XXXXXX)
.venv/bin/python tests/sanitizers/run_cuda.py "$audit_dir" \
  --suite selftest --suite crypto

# Run only on idle GPUs: the last flag temporarily fills nearly all VRAM.
CUATTEST_TEST_GPU=1 CUATTEST_TEST_CRYPTO=1 \
CUATTEST_CRYPTO_DEVICES=all CUATTEST_TEST_MULTIGPU=1 \
CUATTEST_TEST_LARGE_CAPACITY=1 .venv/bin/python -m pytest -q
```

The runner independently isolates the writable production cache and test-only
oracle cache. The first oracle compilation took several minutes (about five
on the laptop and ten including setup on Blackwell); subsequent sanitizer
runs reused its source/compiler/architecture/checksum-validated test CUBIN.

Original logs and binaries are temporary, not versioned:

- Laptop: `/tmp/cuattest-perf-experiments.br62IP/final-sanitizers/`,
  `final-cpu.log`, `final-local-all.log`, `final-crypto.log`, and
  `reduction-check-archived/` with `reduction-check-archived.log`.
- Remote: `/tmp/cuattest-perf-20260909.jRCVUz/final-sanitizers/`,
  `final-all-tests.log` and `final-crypto.log`.
  These logs and the runner's result manifest were also copied into the
  laptop artifact directory's `remote-validation/` subdirectory.
- [Frozen arithmetic/host experiment generators](../../tests/performance/experiments/README.md)
  and [raw performance evidence](../performance/crypto-2026-09-09.json) are
  versioned; exact source and CUBIN identities accompany the measurements.

Rejected work is not counted as clean: a receipt-writer experiment failed
real RTX 2080 receipt validation despite passing on Blackwell, and was not
integrated. An experimental host's first context creation hit OOM; a later
successful retry is separately labeled. Initial unprivileged Nsight Compute
counter access was denied and was repeated with authorized sudo. No GPU
clock changes, driver-security changes or new software installations were
needed. Only experiment-owned producers/services were stopped.

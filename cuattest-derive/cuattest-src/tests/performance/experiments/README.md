# Frozen performance experiments, 2026-09-09

The later **1ce0fbe** hashing/IPC campaign lives in [ipc_hash/](ipc_hash/README.md).
It separately tests packed allocations, persistent registration and async staging.

These are **test-only, partly rejected experiments**, not runtime backends.
They reconstruct the arithmetic, hashing, table-scan, and host-scheduling
candidates in [the performance report](../../../docs/performance.md).
Never deploy their CUBINs or mix them with the normal cache. `stages` exports
profiling counters, and `writer_cached` failed receipt validation on an RTX
2080 even though it passed on Blackwell. Neither is in production.

The generators require the exact **4708698** baseline source (SHA-256 checked).
Using only an old `CUATTEST_KERNEL_DIR` is insufficient: the loader correctly
rebuilds when that CUBIN does not match the selected source. Always specify the
baseline source when comparing against an edited checkout.

From the repository root, with the benchmark/test dependencies installed:

```bash
trial_dir=$(mktemp -d /tmp/cuattest-crypto-trials.XXXXXX)
git archive 4708698 | tar -x -C "$trial_dir"
export CUATTEST_KERNEL_SRC="$trial_dir/src/cuattest/kernel/p256_cuda_notary_b3.cu"
export CUATTEST_CACHE="$trial_dir/cache"
export PYTHONPATH=src

# CPU-only compilation; choose the target architecture explicitly.
.venv/bin/python tests/performance/experiments/compile_variants.py \
  "$trial_dir/variants" --arch sm_120 \
  --variant baseline --variant karatsuba --variant prime_fast_mont+table8

# Local GPU microbenchmarks: four active lanes, dependent operation chains,
# Python-integer differential checks before and after every measured kernel.
.venv/bin/python tests/performance/experiments/bench_arithmetic.py \
  "$trial_dir/arithmetic" --variant baseline --variant karatsuba \
  --variant prime_fast_mont --runs 11 --device 0

# Exact experimental reducer, Boost cpp_int oracle, Clang 21 ASan + UBSan.
.venv/bin/python tests/performance/experiments/check_prime_reduction.py \
  "$trial_dir/reduction-check"
```

Compilation writes a source/CUBIN checksum manifest per candidate and fails
if a candidate fails compilation. Use a fresh output directory each time.
The optional `make_host_variants.py BASELINE_SNAPSHOT OUTPUT` builds four
isolated native host copies with different IPC import batching; no ownership,
range validation, launch ordering, or cleanup checks are intentionally removed.
It requires the exact baseline native source too.

For end-to-end comparison, start each owned service with its explicit
`CUATTEST_KERNEL_SRC` and `CUATTEST_KERNEL_DIR`, then run the existing
`benchmark_torch_models.py` or `benchmark_large_model.py`. Keep checkpoint
revision, dtype, placement, warmups and sample counts identical; alternate
baseline/candidate batches. Time only `Client.sign`, independently verify the
CPU root and every receipt, retain outliers, and never pool profiler samples
with ordinary timings. The September 9 campaign kept the entire pretrained
739.102 GiB Qwen checkpoint resident across service restarts.

Microkernel timings are not end-to-end speedups. In particular, specialized
squaring and Karatsuba sometimes improved a narrow operation without improving
the actual request. Full Nsight Compute replay (`--clock-control none`) and
Nsight Systems request traces were collected separately. The performance
report identifies the retained production combination, raw result data, and
the locations of the original campaign's full logs/traces.

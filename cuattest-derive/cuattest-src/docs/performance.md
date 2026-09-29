# Performance benchmarks

The original GPT-2 workloads remain unchanged. A third, explicit
[near-capacity eight-GPU benchmark](#near-capacity-eight-gpu-benchmark) uses
the pretrained Qwen3.5-397B main-model weights on `probqa.com`.

The latest [arithmetic and critical-path experiments](#arithmetic-and-critical-path-experiments-2026-09-09)
improve the small-model workloads and receipt kernel, but do **not** establish
another end-to-end speedup for the near-capacity model.

> **Baseline preserved:** the first measurements below were captured before
> optimizing cuAttest. They remain the comparison point for subsequent work,
> not optimized performance claims.

This baseline measures end-to-end attestation latency for two resident
PyTorch models. GPT-2 preserves the existing example's small-model workload.
[GPT-2 Large](https://huggingface.co/openai-community/gpt2-large) is the larger
workload: its model card describes 774 million parameters and its
[safetensors checkpoint](https://huggingface.co/openai-community/gpt2-large/blob/main/model.safetensors)
is 3.25 GB. It was selected because it leaves practical headroom on the 8 GB
RTX 2080 used here while remaining substantially larger than GPT-2.

## Reproduce it

Start the notary and benchmark client in separate shells:

```bash
.venv/bin/pip install '.[client,verify]' transformers
.venv/bin/cuattest serve
.venv/bin/python tests/performance/benchmark_torch_models.py
```

The script defaults to one warm-up and five measured attestations for each of
`openai-community/gpt2` and `openai-community/gpt2-large`. Those defaults pin
the exact model revisions in the results table. It also accepts any set of
Hugging Face causal-language-model IDs, with an optional revision after `@`:

```bash
.venv/bin/python tests/performance/benchmark_torch_models.py \
  --model openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --model openai-community/gpt2-large@32b71b12589c2f8d625668d2335a01cac3249519 \
  --warmup-runs 1 --runs 5
```

Each sample times the complete `Client.sign(...)` call with
`time.perf_counter()`, matching `examples/measure_torch_model.py`. The interval
includes request preparation, the local HTTP round trip, CUDA IPC import,
measurement, GPU receipt signing, response parsing, the immutability check, and
acknowledged IPC-lease retirement. Model download/loading, `share_model(...)`,
and receipt verification are outside the interval. A fresh IPC lease is
created for every attestation, and every returned receipt is verified against
the notary's public key before the sample is accepted.

The two baseline models run sequentially in FP32. The small model is deleted
and the PyTorch CUDA cache is emptied before the large model is loaded.
Checkpoints are loaded only through safetensors; downloading and model
inference are not benchmarked. The recorded run used locally cached
checkpoints.

## Pre-optimization results

Captured on 2026-09-04 from source revision
`88ae46cf05e392bbe9e7738d1457ad315803fa4e`:

| Model | Resolved model revision | Parameters | Tensors | Attested data | Timed samples (s) | Mean (s) | Sample stdev (s) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `openai-community/gpt2` | `607a30d783dfa663caf39e06633721c8d4cfcd7e` | 124,439,808 | 149 | 621.9 MiB | 0.185, 0.190, 0.189, 0.189, 0.187 | **0.188** | 0.002 |
| `openai-community/gpt2-large` | `32b71b12589c2f8d625668d2335a01cac3249519` | 774,030,080 | 437 | 3,198.1 MiB | 0.332, 0.333, 0.330, 0.333, 0.335 | **0.333** | 0.002 |

The excluded warm-up samples were 0.212 s for GPT-2 and 0.334 s for GPT-2
Large. "Attested data" is the sum of the byte ranges named by the model's
runtime `state_dict`; it can exceed unique resident weight storage because a
tied weight exposed under two names is intentionally measured twice.

## Fused cooperative-kernel results

Captured on 2026-09-04 on the same machine, with the same model revisions,
dtype, warm-up count, sample count, and benchmark script. This is the first
performance result after replacing the per-tensor chunk launches, host-driven
reduction launches, model-root launch, and signing launch with one cooperative
request kernel:

| Model | Timed samples (s) | Mean (s) | Sample stdev (s) | Change from baseline | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| `openai-community/gpt2` | 0.148, 0.146, 0.149, 0.146, 0.148 | **0.147** | 0.001 | **-21.8%** | **1.28x** |
| `openai-community/gpt2-large` | 0.237, 0.244, 0.237, 0.241, 0.238 | **0.239** | 0.003 | **-28.2%** | **1.39x** |

The excluded warm-up samples were 0.208 s for GPT-2 and 0.239 s for GPT-2
Large. A second complete run produced means of 0.147 s and 0.239 s,
respectively. The fused build used kernel CID
`bafkr4ihcalhbcx4u4be3mt3ufqldueblxamorju3glwj3wrvj7hmmsgtuu` and CUBIN CID
`bafkr4ibo5555wqx3u5ajhvyh5uyynjunn6w5g5ett6t5fr2vxs2ylubf7q`.

These are end-to-end results, so they include work unaffected by kernel
fusion, such as CUDA IPC setup/teardown, HTTP, response processing, and
producer lease retirement. On this 46-SM GPU the occupancy query admits 92
resident blocks of 128 threads; grid-stride loops cover larger models without
exceeding that cooperative-launch limit.

## Native C++ host-path results

Captured on 2026-09-04 with the same machine, pinned model revisions, FP32
weights, and unchanged fused kernel/CUBIN. Each backend was measured in two
independent batches, each with three excluded warm-ups and 20 measured
attestations per model. Both servers were built from the same working tree;
only `CUATTEST_DISABLE_NATIVE_HOST=1` selected the Python fallback.
`/v1/info` and the benchmark banner reported the selected backend, preventing
an accidental comparison with a stale installation.

| Model | Host backend | Runs | Mean (s) | Sample stdev (s) | Change vs Python | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `openai-community/gpt2` | Python fallback | 40 | **0.1555** | 0.0043 | — | — |
| `openai-community/gpt2` | C++ | 40 | **0.1526** | 0.0016 | **-1.9%** | **1.02x** |
| `openai-community/gpt2-large` | Python fallback | 40 | **0.2525** | 0.0042 | — | — |
| `openai-community/gpt2-large` | C++ | 40 | **0.2433** | 0.0019 | **-3.6%** | **1.04x** |

The individual Python batch means were 0.1567/0.1544 s for GPT-2 and
0.2521/0.2529 s for GPT-2 Large; C++ produced 0.1532/0.1521 s and
0.2428/0.2438 s. The native call covers unique CUDA IPC import,
driver-authoritative device/range validation, reduction-plan construction,
CUDA argument marshaling, reusable scratch allocation, launch, result
transfer, and guaranteed IPC cleanup. The GIL is released while that native
request executes. The Python implementation remains available as a portable
diagnostic fallback, not the production default.

These percentages are observed differences on a non-exclusive desktop GPU,
not universal speedup guarantees; in particular, the small-model delta is of
the same order as occasional host/GPU scheduling noise.

For a narrower host-only check, 10,000 constructions of a synthetic 149-span
plan had median latency of 283.4 microseconds in Python and 34.4 microseconds
through the extension, including Python-to-C++ argument conversion. Reusing
the native CUDA workspaces also removes allocation/free calls from steady-state
requests.

### Pre-device-optimization bottleneck

Nsight Systems profiling of two warm-ups plus five measured GPT-2 attestations
on the C++ backend found a median `measure_attest_fused_kernel` duration of
147.44 ms (146.03 ms minimum). The seven requests made 147 IPC opens and 147
closes—21 unique allocator segments per request—which consumed 19.05 ms in
total, or 2.72 ms per request including one close outlier. All recorded GPU
host-to-device and device-to-host transfers together consumed less than 0.05
ms. The measured benchmark mean under the profiler was 0.154 s.

This machine therefore does not reproduce a 64 ms GPT-2 result: its GPU kernel
alone takes approximately 147 ms in steady state. More importantly, the
profile shows why changing the host language produces a real but small 3%
end-to-end gain rather than eliminating tens of milliseconds. Once the C++
path is active, further material acceleration must target the device hashing
work (or reduce bytes hashed while preserving the attestation contract), not
Python orchestration.

The tensor count is also a misleading intuition for this workload: GPT-2's
149 entries name 621.9 MiB, including its tied weight under both state-dict
names. A 64 ms attestation is already hashing that logical byte stream at about
9.5 GiB/s before accounting for tree reduction, model-root folding, and P-256
receipt construction/signing.

## Device-kernel optimization results

Captured on 2026-09-04 after profiling the preceding build with Nsight Compute
`--set full`. The optimized build has kernel CID
`bafkr4ice37xl7kzw6xonqyuczrfkmamid5lv6udvapk2u5lu5qmbvsw2my` and CUBIN CID
`bafkr4ihymzvdvltnembmnhhjdebhoysb5xnjbntehapog53y6gw375ia2u`.

### Full-profile method

The notary was started under Nsight Compute in one shell, and one GPT-2 request
was sent with the benchmark client from another. `--set full` replays the
selected kernel to collect the complete metric set; these profiler results are
not included in the ordinary timing averages below.

```bash
ncu --set full --target-processes all --kernel-name-base function \
  --kernel-name 'regex:^measure_model_fused_kernel$' \
  --launch-count 1 --force-overwrite \
  --export /tmp/cuattest-ncu-full-measure .venv/bin/cuattest serve

.venv/bin/python tests/performance/benchmark_torch_models.py \
  --model openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --warmup-runs 0 --runs 1
```

The pre-optimization build was captured the same way with
`measure_attest_fused_kernel`; the final receipt kernel was captured separately
with `attest_measured_kernel`. NCU replay timings are compared as device-kernel
metrics, while the model table below comes from ordinary, unprofiled service
runs.

### What the full profile found

The original “fused” entry made the P-256 signer and statement builder reachable
from every hashing thread. Even though only grid rank zero used them, ptxas had
to provision their 255-register, 11,264-byte frame for every lane. Occupancy
fell to 22.10%, 93.03% of scheduler cycles had no eligible warp, and
runtime-indexed BLAKE3 state/message arrays spilled through the load/store unit.

The optimized measurement entry is isolated from receipt construction. BLAKE3
uses a compile-time scalar message schedule, four 16-byte vector loads for each
aligned 64-byte block, vector chaining-value transfers, and a byte-safe path for
unaligned or partial input. Each lane interleaves two independent BLAKE3 leaves,
issuing both loads before the dependent compression chains. Every tensor tree
level and the model fold remain in one cooperative launch. A small kernel queued
on the same stream privately consumes the folded root and computes the four
independent P-256 signatures in four lanes. There is still only one host
synchronization and one result copy.

| NCU metric, GPT-2 | Before: combined measurement/signing | After: measurement only |
| --- | ---: | ---: |
| Kernel duration | 167.22 ms | **3.51 ms** |
| Registers per thread | 255 | **80** |
| Stack frame per thread | 11,264 B | **2,832 B** |
| Theoretical occupancy | 25.00% | **75.00%** |
| Achieved occupancy | 22.10% | **70.92%** |
| DRAM throughput vs peak | 1.86% | **43.91%** |
| DRAM traffic rate | 8.32 GB/s | **211.29 GB/s** |
| Useful bytes per 32-byte global-load sector | 1.06 B | **15.76 B** |
| Theoretical excessive global-load sectors | 649.61 MB | **22.93 MB** |
| Scheduler cycles with no eligible warp | 93.03% | **74.41%** |
| Eligible warps per scheduler | 0.08 | **0.38** |
| LG-throttle stall per issued instruction | 19.62 cycles | **2.29 cycles** |
| Long-scoreboard stall per issued instruction | 1.58 cycles | **15.40 cycles** |
| ALU utilization | 1.17% | **37.14%** |

That measurement profile read 700.65 MB and wrote 39.96 MB of DRAM while
hashing 621.9 MiB of logical model data and its BLAKE3 reduction workspaces. Its
useful model-input rate is about 186 GB/s (173 GiB/s). The separate receipt
kernel took 10.64 ms, used 255 registers and a 7,840-byte stack, and executed
3.877 million instructions; its DRAM utilization was only 0.93%. Thus the two
then-current GPU kernels totalled 14.15 ms, an **11.8x** reduction from the
original 167.22 ms combined kernel.

### End-to-end model results

Two independent batches used ten excluded warm-ups and 50 timed attestations
per model. Every timed receipt was verified before its sample was accepted.

| Model | Runs | Batch means (s) | Pooled mean (s) | Sample stdev (s) | Change vs C++ pre-device result | Speedup |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `openai-community/gpt2` | 100 | 0.0184, 0.0189 | **0.0187** | 0.0012 | **-87.8%** | **8.18x** |
| `openai-community/gpt2-large` | 100 | 0.0482, 0.0483 | **0.0483** | 0.0024 | **-80.2%** | **5.04x** |

Against the original baseline at the top of this page, these means are 10.1x
faster for GPT-2 and 6.90x faster for GPT-2 Large. GPT-2 benefits especially
from the fixed-cost signing optimizations; GPT-2 Large still hashes 3,198.1 MiB
while the four-signature receipt cost remains constant.

### Why this is not 100% memory bandwidth

The requested near-100% DRAM target was **not** reached: the current tiled
measurement kernel, documented below, uses 57.53% of peak DRAM bandwidth. It is
nevertheless the fastest correct variant measured. Forcing the bandwidth
counter upward would require extra traffic and would make the attestation
slower; maximizing a utilization counter is not the same objective as
minimizing latency.

The proposed four-thread interleaving was implemented and profiled. Four lanes
jointly loaded each 64-byte message block, raising useful bytes per 32-byte
sector to 28.4 and reducing theoretical excessive sectors to about 4%.
Nevertheless, its kernel took **7.04 ms**, versus 3.51 ms for the retained
one-thread/two-leaf path at that stage and 2.64 ms after the scratch tiling
below. It used 92 registers, achieved only 62.3% occupancy, executed 441 million
instructions, and generated 2.6-way shared-memory bank conflicts. The
coalescing improvement did not repay the shuffles, shared-memory traffic, and
lost parallelism.

Other controlled trials reached the same conclusion:

| Trial | Result | Decision |
| --- | --- | --- |
| One leaf per lane | 4.34 ms, 64 registers, 99.39% achieved occupancy | rejected; less ILP and 23% slower than two leaves |
| Two leaves per lane | **3.51 ms**, 80 registers, 70.92% achieved occupancy | retained as the leaf compressor; tiled reduction later reached 2.64 ms |
| Three leaves per lane | 4.63 ms median in the isolated benchmark, 128 registers | rejected; register pressure dominates |
| Prefetch a full 64-byte block before compression | 4.40 ms, 80 registers, 75% occupancy | rejected; slower than paired ILP |
| 256 threads per block | 21.1 ms end to end vs 20.1 ms at 128 threads | rejected |
| Four single-thread signing blocks | 11.82 ms vs 11.95 ms for four lanes, but 15.2M vs 4.43M instructions | rejected |
| P-256 fixed window, 4/5/6/7 bits | 11.95/11.06/**10.64**/10.81 ms | retained six bits |

BLAKE3's cryptographic chunk is fixed at 1,024 bytes by the
[official specification](https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.tex).
A 128 KiB unit can only be a scheduling tile containing 128 correctly countered
BLAKE3 leaves; it cannot replace the leaves without changing every digest. The
scratch optimization below uses exactly that construction: independent leaves
retain their counters, seven tree levels are reduced within the tile, and only
the resulting subtree CV enters global scratch.

The remaining full-profile signature is correspondingly different from the
original one: spill traffic and very low occupancy are gone, while memory
latency and BLAKE3's dependent integer rounds share the limit. NVIDIA's
[Nsight Compute triage guidance](https://docs.nvidia.com/cuda/developer-preview/13.4/nsight-compute/ComputeTriage/index.html)
recommends validating both duration and effective bandwidth, and treating load/
store throttling with coalescing, vector access, tiling, or instruction-level
parallelism according to the measured limiter. Paired-leaf ILP and block-local
tiling improved both bandwidth and duration; variants that further raised
coalescing or raw-bandwidth counters also raised duration. The retained result
is therefore the fastest measured attestation variant, rather than a
manufactured near-100% bandwidth figure produced by redundant reads.

## 128 KiB scratch-space tiling

Captured on 2026-09-04 after replacing per-semantic-chunk global CV storage
with block-local 128 KiB scheduling tiles. The exact build used for the full
profile and benchmark in this section has kernel CID
`bafkr4ibmjxlf2jo6nth54ywizn2fzi765ht6xk6st46vtbejpkycole45m` and CUBIN CID
`bafkr4ibypfypfygk3al2yqgdoog5cvxmclb7hezqwmntwfw6jrqgggfafm`.

A subsequent readability-only refactor has kernel CID
`bafkr4ibmlwfntmgdkfaiwnm2w3ydcuoqlhniplo4mwxwswpxaqb44uw3ti` and CUBIN CID
`bafkr4ieozksb435nrn4rxkq52oeh7i3llmko6mb47zfjnnisouxuyhykae`.
On the same RTX 2080, an NCU basic-profile verification measured 2.67 ms and
57.71% of peak DRAM throughput. Its compiled measurement kernel still uses 96
registers per thread, 2,832 bytes of stack per thread, and 8,384 bytes of
static shared memory per block. The detailed table below remains tied to the
earlier exact CUBIN so that historical measurements are not silently mixed.

Each 64-thread half-block hashes two 1 KiB BLAKE3 leaves per lane and reduces
the resulting 128 CVs through seven tree levels in a padded, transposed shared
array. Only the final 32-byte tile CV reaches global scratch. The ordinary
cross-tile reduction is unchanged, including odd-node carry and final `ROOT`
placement, so this is only a scheduling transformation and every digest remains
standard BLAKE3.

For the motivating single-input case:

| 96 GiB tensor | Previous per-chunk scratch | 128 KiB tiled scratch |
| --- | ---: | ---: |
| Semantic 1 KiB chunks | 100,663,296 | 100,663,296 |
| Scheduling tiles | — | 786,432 |
| Primary global CV workspace | 3,072 MiB | 24 MiB |
| Secondary global CV workspace | 1,536 MiB | 12 MiB |
| **Total persistent scratch** | **4,608 MiB (4.5 GiB)** | **36 MiB** |

This is a **128x reduction**, from 4.69% to about 0.037% of the input size.
The additional 8,384-byte shared array is per resident block, is automatically
reused for every tile, and is not a persistent allocation.

An exact-CUBIN `ncu --set full` capture shows that avoiding global CV traffic
also improves the measurement kernel rather than trading memory capacity for
latency:

| NCU metric, GPT-2 | Per-chunk global scratch | 128 KiB tiled scratch |
| --- | ---: | ---: |
| Kernel duration | 3.51 ms | **2.64 ms** |
| DRAM throughput vs peak | 43.91% | **57.53%** |
| DRAM traffic rate | 211.29 GB/s | **255.25 GB/s** |
| DRAM reads | 700.65 MB | **672.75 MB** |
| DRAM writes | 39.96 MB | **1.91 MB** |
| Theoretical excessive global sectors | 22.93 MB | **20.39 MB** |
| Registers per thread | 80 | 96 |
| Static shared memory per block | 0 | 8,384 B |
| Achieved occupancy | 70.92% | 63.53% |
| Excessive shared-memory wavefronts | — | **0** |

Despite the lower occupancy, removing almost all global reduction writes cuts
measurement time another 24.8%. The logical 621.9 MiB GPT-2 input is consumed
at about 247 GB/s (230 GiB/s).

Two ordinary benchmark batches again used ten excluded warm-ups and 50 timed,
verified attestations per model:

| Model | Runs | Batch means (s) | Pooled mean (s) | Sample stdev (s) | Change vs pre-tiling | Speedup |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `openai-community/gpt2` | 100 | 0.0172, 0.0177 | **0.0175** | 0.0014 | **-6.4%** | **1.07x** |
| `openai-community/gpt2-large` | 100 | 0.0395, 0.0409 | **0.0402** | 0.0091 | **-16.8%** | **1.20x** |

The second GPT-2 Large batch includes one 0.1270 s desktop scheduling outlier;
it is retained in both its mean and the pooled deviation. Against the original
baseline, the tiled means are 10.7x faster for GPT-2 and 8.28x faster for GPT-2
Large.

## Machine and software

| Component | Baseline configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080, 8,192 MiB, `sm_75` |
| GPU availability | 7,048 MiB free before starting the notary; non-exclusive desktop GPU |
| Driver | 610.43.02 |
| CPU | Intel Core i9-9980HK, 8 cores / 16 threads |
| OS | Linux 7.0.0-30-generic, x86-64 |
| Python | 3.13.13 |
| PyTorch | 2.14.0 |
| Transformers | 5.16.1 |
| safetensors | 0.8.0 |
| cuAttest | 0.1.0 |
| Kernel compiler | CUDA 12.9 prebuilt CUBIN |
| Kernel CID | `bafkr4ifrxpcslaeedbj2j7bbkyz5g7oos3oudrnxq4mxmh4a6mmbrgpise` |
| CUBIN CID | `bafkr4ictdlsklwdvupidrtszqolr37rsb2s4ldjsitokojzxzlbntdx4du` |

Future comparisons should use the same model revisions, dtype,
warm-up count, measured-run count, hardware, and comparable GPU load; an
otherwise idle GPU is preferred.

## Near-capacity eight-GPU benchmark

`tests/performance/benchmark_large_model.py` adds a pretrained workload sized
for the eight RTX PRO 6000 Blackwell Server Edition GPUs on `probqa.com`.
It is **opt-in**: running the original script still benchmarks only GPT-2 and
GPT-2 Large and does not start an enormous download.

The preset is [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B),
revision `8472618112abcbd45acbcdc58436aff4233c23f7`. It selects the 1,371 stored
main-model tensors, including the vision encoder and output head, in their
original BF16/F32 dtypes: 396,802,360,816 elements and 793,604,738,912 bytes
(739.102 GiB). The selection's names, shapes, and parameter count match the
stock Transformers model; the separately stored `mtp.*` auxiliary predictor
is explicitly excluded. The full checkpoint including that predictor exceeds
usable VRAM after producer and notary contexts are created. No arbitrary
layers, experts, or main-model tensors are dropped to force a fit.

This is a **resident-weight attestation benchmark, not an inference benchmark**.
The script streams safetensors into a named PyTorch tensor state rather than
constructing execution graphs or KV caches. It executes no downloaded model
code, changes no dtype, performs no quantization or CPU offload, and adds no
padding allocations. Each complete tensor lives on one GPU; largest-first
placement balances storage without splitting tensors or changing the sorted
name order used for the global root.

### Run it

Use separate shells on the GPU host, as with the small benchmarks:

```bash
.venv/bin/pip install '.[benchmark]'

# Shell 1: explicitly raise both aggregate request caps for this benchmark.
.venv/bin/cuattest serve --devices all \
  --max-request-bytes 1099511627776 --max-request-tiles 8388608

# Shell 2: first check the pinned headers and the live GPU memory budgets.
.venv/bin/python tests/performance/benchmark_large_model.py --dry-run

# Then download/cache the selected shards, load, verify, and time the model.
.venv/bin/python tests/performance/benchmark_large_model.py \
  --warmup-runs 1 --runs 5 --json-output qwen397b-results.json
```

The default service caps (128 GiB / 1,048,576 tiles) deliberately remain
unchanged. This workload requires 793,604,738,912 logical bytes and 6,055,330
tiles; raising only the byte cap is insufficient. The larger caps above apply
only to the explicitly started benchmark service. The original GPT-2 benchmark
client also works against this multi-GPU endpoint and verifies its aggregate
receipts.

Allow at least **850 GB of free download/cache space**; the checkpoint files
themselves total approximately 807 GB including the auxiliary predictor.
Download time is not a timed sample, but can be around two hours on a
1-Gbit/s connection. Hugging Face's
cache is reusable across runs; `--cache-dir PATH` changes the weight download
location. `--dry-run` reads remote headers and sends a one-byte-per-GPU
initialization probe, but does not download weight payloads or allocate the
model. The probe materializes CUDA's lazy signing resources before checking
free VRAM. A local `.safetensors` file, index, or directory can be
used with `--checkpoint PATH`; explicitly add `--exclude-prefix mtp.` when
using a local copy of this Qwen checkpoint. Custom checkpoints have no implicit
architectural exclusions.

### Pretrained checkpoint results

Captured on 2026-09-07 at 05:20:30 UTC on `probqa.com` (`Sarge-SRV2025`),
using all eight RTX PRO 6000 Blackwell Server Edition GPUs, the native C++
host backend, PyTorch `2.14.0+cu130` (CUDA runtime 13.0), and an `sm_120`
CUBIN compiled with NVRTC 13.3. This completed run used the actual downloaded
pretrained weights at the pinned revision above, with only `mtp.*` excluded;
it is separate from the earlier initialized-weight capacity test below.

| Model | Tensors | Resident / attested data | Measured runs | Mean (s) | Sample stdev (s) | Aggregate input throughput (GiB/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen3.5-397B-A17B` | 1,371 | 739.102 GiB | 5 | **1.5730** | 0.0025 | **469.87** |

The five timed samples were 1.574039, 1.573655, 1.570636, 1.570316, and
1.576329 seconds. One model warm-up and the tiny initialization probe were
excluded. Each sample times only `Client.sign(...)`; downloading, loading,
independent CPU hashing, IPC export, and receipt verification are outside the
interval. GPU shards were measured serially: throughput is total attested
bytes divided by mean end-to-end latency, not concurrent eight-GPU bandwidth.

Actual unique weight storage was 92.386–92.388 GiB per GPU, approximately
97.24% of each GPU's 95.010 GiB CUDA-reported capacity. Free VRAM after loading
was 1.228–1.234 GiB per GPU, preserving the requested 1 GiB reserve without
quantization, offload, or padding allocations.

Every warm-up and measured receipt verified against the pinned GPU UUID/key
map and matched the independent CPU-computed model root:
`077a5b303b1d6ed829a9c30fdb0102cf61e63fd5bcc562f8ebe153e10fdbe805`.
The kernel and CUBIN CIDs were the same as those recorded for the capacity
validation below. The notary shut down cleanly after the run and released its
GPU contexts.

Source report on `probqa.com`:
`/tmp/cuattest-large.9uP7wI/pretrained-results.json`.

### Concurrent eight-GPU attestation (2026-09-08)

The optimized service reduces full-checkpoint latency from **1.5743 s to
0.2721 s: 5.79x faster, or 82.7% less latency**. This measures the entire
739.102 GiB checkpoint across eight GPUs, not a single GPU's 92.388 GiB shard.

| Native host implementation | Timed requests | Mean (s) | Sample stdev (s) | Median (s) | Aggregate input throughput (GiB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before: serial GPU shards (`38dbbf0`) | 40 | 1.5743 | 0.0161 | 1.5693 | 469.48 |
| After: concurrent shards, batched IPC imports | 40 | **0.2721** | 0.0192 | **0.2663** | **2716.51** |

Each row combines two batches of 20 requests with three excluded warm-ups per
batch. Baseline batch means were 1.572851 and 1.575751 s; final batch means were
0.268364 and 0.275791 s. All outliers are retained, including the final run's
0.357004 s sample. [Raw samples, occupancy, source hashes, and profile
summaries](performance/qwen397b-multigpu-2026-09-08.json) are checked in.

The unchanged pretrained checkpoint stayed resident with identical placement
through the A/B experiments. It occupied 97.24% of each GPU's CUDA-reported
capacity and left 1.241–1.247 GiB free after loading. Each owned service started
with fresh session keys, pinned through the trusted local setup channel.
Every warm-up and measured receipt matched the same independently CPU-hashed
root recorded above. Timing remains exclusively `Client.sign(...)`: no loading,
IPC export, CPU hashing, or verification is included. There is no quantization,
offload, digest reuse, or cross-request IPC mapping cache.

The host was `Sarge-SRV2025`, with eight RTX PRO 6000 Blackwell Server Edition
GPUs, driver `610.57.04`, PyTorch `2.14.0+cu130`, and NVRTC 13.3. Both builds
loaded the same `sm_120` CUBIN. This change does not alter the device kernels:
kernel CID `bafkr4ido4afrt5kucobcqxj5cnkks5jn3rhl5wifxagoj5vadqomcce5ou`,
CUBIN CID `bafkr4icgixawe242xqa4uaprbg6tqp4i5zhbgnqf73kqfqns34s5vtdve4`.

#### What profiling changed

Nsight Systems 2026.1.3 captured one warm-up and three steady requests per
variant, separately from the table's ordinary timing samples. Before the
change, the eight shards' kernels ran sequentially. Dedicated per-GPU workers
now overlap them while keeping each session's mutable workspaces exclusive.
The authenticated shard order and global root are independent of completion
order. HTTP requests still queue, and even error responses wait for all
dispatched GPU work and confirmed IPC cleanup.

The intermediate parallel-only build averaged 0.2889 s over 80 development
samples. Its trace exposed driver IPC contention: the median open call rose
from 62.0 microseconds in the baseline to 978.2 microseconds. A short native
import-batch gate brought that back to 65.4 microseconds. The gate is released
before launch/synchronization; eight-GPU hashing still overlapped for
133.6–136.3 ms in each final steady trace. It does not replace per-allocation
owner/bounds validation or cleanup on failure.

Per-GPU median kernel times remained essentially unchanged: **176.26 ms** for
hashing and **9.40 ms** for receipt construction/signing. The first-kernel-start
to last-kernel-end interval across all GPUs fell from approximately 1549 ms
to 225–228 ms. These are device-timeline intervals, not HTTP latency or peak
DRAM bandwidth; the throughput table uses useful attested bytes divided by
complete request time.

To reproduce ordinary timings, use the service and large-model commands above
with `--warmup-runs 3 --runs 20`, in an idle GPU window. For request-only tracing:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=qwen397b-trace \
  .venv/bin/python tests/performance/profile_multigpu_server.py

# In another shell, then interrupt only the owned profiling server when done:
.venv/bin/python tests/performance/benchmark_large_model.py \
  --warmup-runs 1 --runs 3 --json-output qwen397b-profiled.json
```

The helper starts capture after all notary sessions initialize. Tracing their
initialization from process launch caused an allocator abort in the
instrumented process on this driver/profiler combination, with both host
backends. Deferred capture completed successfully; it avoids the observed
startup failure without claiming to establish its cause. See NVIDIA's
[capture-range documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
Never include profiled samples in an ordinary latency comparison.

Full experiment reports, logs, and `.nsys-rep` files are retained on `probqa.com`
under `/tmp/cuattest-blackwell-opt.2U5S32/results2`. The experiment-owned
producer and services were stopped, and all eight GPUs returned to zero
reported allocated VRAM before subsequent validation.

Validation included **679 passing tests on all eight GPUs**, with both host
backends, 192 device-crypto differential cases, reversed IPC device ordinals,
and near-capacity cases enabled. Four subsequently added sanitizer-runner CPU
cases also passed. The final laptop suite passed 503 cases with an
ASan/UBSan-instrumented extension (12 hardware/opt-in skips). All four CUDA
sanitizer tools passed the selected multi-GPU success paths; IPC initcheck uses
shared-memory checking because global IPC initialization tracking is unsupported.
The native eight-thread harness passed ASan/LSan, UBSan, TSan, and MSan with
instrumented libc++. See [sanitizer scope and reproduction](sanitizers.md).

### Private CUDA streams (2026-09-08)

Both host backends now use one **nonblocking CUDA stream per notary session**.
Metadata upload, measurement, receipt signing, and output download execute in
that stream, followed by one `cuStreamSynchronize`, not `cuCtxSynchronize`.
Pinned host staging and device workspaces are retained between requests in
both backends: allocating/freeing them on every fallback request was itself
an implicit synchronization point. Cold allocation or workspace growth can
still block. Session requests remain serialized because signing consumes the
measurement kernel's private, one-shot root.

This follows NVIDIA's [nonblocking stream semantics](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html):
an ordinary stream without `CU_STREAM_NON_BLOCKING` still implicitly joins
the legacy default stream. No kernel or receipt-format changes were needed.

The matched full-checkpoint experiment on the same eight Blackwell GPUs found
**no meaningful idle-server latency improvement** from streams alone:

| Native host implementation | Timed requests | Mean (s) | Sample stdev (s) | Median (s) |
| --- | ---: | ---: | ---: | ---: |
| Concurrent shards, context waits (`fc1900e`) | 60 | 0.2688 | 0.0129 | 0.2651 |
| Concurrent shards, private streams | 40 | 0.2682 | 0.0161 | 0.2648 |

The approximately 0.5 ms mean difference is much smaller than observed
variation; it is not evidence of a speedup. Each batch contains 20 requests
and excludes three warm-ups. All timed outliers remain, including 0.363319 s
in the final build. Baseline batch means were 0.272994, 0.268045, and 0.265239 s;
final means were 0.264085 and 0.272374 s. The same 739.102 GiB pretrained
checkpoint stayed resident throughout, with the same placement and independent
CPU root. Only `Client.sign(...)` is timed. [Raw samples, source hashes,
occupancy, development runs, and profiler counts](performance/qwen397b-streams-2026-09-08.json)
are retained separately from the earlier sequential-to-concurrent comparison.

Nsight Systems captured one warm-up plus three requests. Across eight GPUs it
recorded 64 kernel launches, 32 asynchronous uploads, 32 asynchronous downloads,
and 32 stream waits, with **zero context waits inside requests**. Each GPU's
copies and kernels used the same non-default stream. A single context wait
after `cuProfilerStop` is outside request execution. Steady per-GPU kernel
medians were 176.36 ms for hashing and 9.40 ms for signing, explaining why
narrowing host synchronization does not materially change the isolated run.

The stronger regression is isolation: both backends finish independently
verified signed requests while another stream, including legacy stream 0,
remains deliberately blocked. Tests also cover partial inputs, shrinking
workspace reuse, failed growth, and DMA/launch/drain/teardown failures. Failed
completion keeps device and pinned buffers under context-owned quarantine;
stream destruction alone is never an IPC acknowledgement.

The producer exporter still calls `torch.cuda.synchronize(device)` before
exporting each device's tensors. Sources can have outstanding writes on
arbitrary PyTorch streams, so waiting on only the current stream would be
incorrect without an explicit producer-event readiness contract. This export
step is outside the benchmark timing. Setup and direct `hash_bytes` uploads
also remain blocking convenience operations; the change concerns notary
request execution, not a claim that every CUDA API call is nonblocking.

Full logs and reports are on `probqa.com` under
`/tmp/cuattest-streams.nGAaw2/results`. The owned benchmark producer and servers
were stopped, and all eight GPUs returned to zero reported VRAM allocation.

## Arithmetic and critical-path experiments (2026-09-09)

The retained change combines sparse P-256 field reduction, fixed public
inversion schedules, Montgomery-domain group-order inversion, and cooperative
eight-lane generator-table scans. Schoolbook multiplication remains in use.
BLAKE3, receipt contents/canonicalization, producer readiness, private streams,
IPC validation/cleanup, and the native import-batch gate are unchanged.

### Matched final measurements

Baseline **4708698** and the final build each received two batches of 60 timed
requests per model, with five excluded warmups per batch, in baseline/final/
final/baseline order. Models stayed resident between service restarts. Every
receipt was verified and its root compared with independently CPU-hashed
weights. Only `Client.sign(...)` was timed; all outliers remain. Profiler runs
are separate and are not included in these statistics.

| Workload | GPU(s) | Baseline mean ± sample stdev (ms) | Final mean ± sample stdev (ms) | Mean change |
| --- | --- | ---: | ---: | ---: |
| GPT-2, FP32, 621.9 MiB | RTX 2080 | 22.634 ± 2.575 | **18.003 ± 2.251** | **−20.5%** |
| GPT-2 Large, FP32, 3,198.1 MiB | RTX 2080 | 48.188 ± 3.728 | **43.983 ± 3.992** | **−8.7%** |
| Qwen3.5-397B, original BF16/F32, 739.102 GiB | Eight RTX PRO 6000 Blackwell GPUs | 269.222 ± 11.040 | **270.342 ± 18.375** | +0.4%; no demonstrated change |

Qwen's medians were 268.455 and 265.460 ms, respectively. The candidate's
0.382708-second outlier is retained. Neither the smaller median nor the
1.121-ms larger mean establishes an end-to-end improvement or regression
against the observed variation. This comparison starts from the already
parallel, private-stream implementation, **not** the historical 1.573-second
sequential baseline.

The Qwen checkpoint, revision, 1,371-tensor selection and exclusions are
unchanged from the earlier pretrained run. Its 793,604,738,912 bytes occupied
about 97.24% of each GPU's usable capacity, leaving 1.241–1.247 GiB free after
the producer and warmed baseline notaries were present. All batches used
identical placement and CPU root
`077a5b303b1d6ed829a9c30fdb0102cf61e63fd5bcc562f8ebe153e10fdbe805`.
The laptop was a non-exclusive desktop GPU; the remote GPUs had no unrelated
compute workloads during the measured requests. GPU clocks were not changed.

Final Nsight Systems captures used one warmup plus three requests, giving
24 steady kernel samples per build across eight GPUs:

| Blackwell per-GPU median | Baseline | Final |
| --- | ---: | ---: |
| Model hashing | 176.308 ms | 176.430 ms |
| Receipt construction/signing | 9.147 ms | **6.721 ms (−26.5%)** |
| Measurement registers/thread | 94 | 94 |
| Receipt registers/thread | 255 | 254 |

The first-hash-start to last-receipt-end windows were 228.21–228.82 ms for
baseline and 224.64–227.42 ms for final. These device windows do not include
all request preparation, IPC import/close, HTTP and response processing, and
are not substitutes for the ordinary request-time table.

### What was tried, including rejected candidates

The campaign compiled 30 isolated CUDA variants, built four native scheduling
variants, and ran independent Python-integer arithmetic comparisons before
timing public test inputs. Development batches generally had 20 requests and
three warmups; later controls/candidates had 40. They are exploratory results,
not interchangeable with the matched final batches above.

| Experiment | Observation / disposition |
| --- | --- |
| One-level 256-bit Karatsuba | RTX 2080 field multiplication worsened from 1.665 to 1.737 µs; inverse chains from 816 to 934 µs. Blackwell receipt time changed only from about 9.41 to 9.32 ms. **Not retained.** |
| Unrolled schoolbook, specialized Comba squaring, 32-bit schoolbook limbs | Blackwell receipt medians about 9.22, 9.25 and 9.46 ms. No reliable full-request win; retained the existing schoolbook product. |
| Sparse prime reduction, fixed exponent schedules, Montgomery inversion | Best arithmetic-only combination reduced the Blackwell receipt to about 7.85 ms. Field inversion uses sparse reduction; order inversion amortizes Montgomery conversion across the whole chain. **Retained together.** |
| Cooperative constant-work table scans, 2/4/8/16 lanes | Standalone receipt medians about 8.84/8.56/8.42/8.31 ms. The eight-lane arithmetic combination measured 6.86 ms before integration and 6.72 ms in the final build. **Eight lanes retained and validated on both architectures.** |
| Combined arithmetic with 16-lane scans | About 6.65 ms, only 0.07 ms below the final eight-lane kernel, with two signing warps instead of one. Insufficient benefit to select another configuration. |
| Warp-shared tensor lookup; read-only-cache loads | Hash medians 176.20/176.39 ms versus 176.27 ms baseline: no meaningful gain. |
| Sequential rather than interleaved paired BLAKE3 leaves | Hash median 174.89 ms, approximately 0.8% lower, but no established full-request win. Not selected. |
| Higher-occupancy register caps, 8/12 blocks per SM | Hashing slowed to 190.67/205.35 ms. Not retained. |
| Ungated IPC imports; batches of 1/4/16 allocations | Ordinary means 286.80/302.87/301.29/292.67 ms. Ungated imports reduced launch skew but increased other host overhead. Kept the existing import-batch gate. |
| Cached receipt-writer bookkeeping | Passed Blackwell trials but failed a real RTX 2080 GPT-2 request: the signed model CID contradicted the measurement. **Rejected for correctness; no writer change ships.** |

The initial ungated-host attempt failed during context creation with an
out-of-memory error; it produced no timed samples. Its later 40-request retry
is recorded separately. A draft small-model comparison was deliberately
interrupted when its baseline would have selected the edited source; the
replacement explicitly selected pristine source and CUBINs. Neither failure
is counted as a successful measurement.

### What now limits latency

In a separate test-only, tiny-input stage probe, the best development
combination reduced signing from 9.84 to 4.27 million GPU cycles. The point
walk fell from 6.41 to 2.54 million cycles, field inversion/affine conversion
from 1.39 to 0.31 million, and order inversion from 1.37 to 0.74 million.
Receipt preparation and assembly stayed near 1.77 and 8.45 million cycles:
**serialization now outweighs the cryptographic signing stage** in this probe.
These `clock64` counters exist only in experimental builds, never production.

For the large model, approximately 176 ms of hashing per GPU and 42–45 ms of
hash-start skew dominate the approximately 7-ms receipt. Accelerating
four-limb multiplication alone cannot remove those costs. The stronger next
targets are bulk-hash memory/dependency latency and safe IPC scheduling, while
preserving standard BLAKE3 and the producer-ownership contract.

A separate 256-MiB hashing diagnostic under `ncu --set full` showed 41.39%
achieved occupancy, 30.50% of peak DRAM throughput, and 42.2% of warp cycles
stalled on L1TEX scoreboard dependencies. Only about half the bytes in global
memory sectors were useful. This supports a memory-access/dependency limit,
**not a claim that VRAM bandwidth is saturated**. It was a diagnostic probe,
not a replay of the entire near-capacity model. The full receipt profile also
showed substantial, mostly L1-cached local-memory traffic. Nsight's generic
"add more blocks" advice for a one-block serial receipt is not itself a
validated optimization.

### Reproduction and evidence

[Raw ordinary samples, variant identities, microbenchmarks and trace summaries](performance/crypto-2026-09-09.json)
are versioned. [Frozen experiment generators and instructions](../tests/performance/experiments/README.md)
require the exact baseline source hash to prevent silently relabeling a new
build as baseline. The final source SHA-256 is
`638ee4bf4209a262c304eb6f9798c3be054c1613f034295a85c794b344648a5b`;
kernel CID is `bafkr4icusplkefze4qmfkzbujfdrrbfhgxjlb35yx2uusdmuemkoz43rbq`.
Rebuild CUBINs for the selected architecture; source/CUBIN identity checks
remain enabled.

Full logs, compiler outputs and Nsight reports remain in temporary storage:
`/tmp/cuattest-perf-experiments.br62IP` locally and
`/tmp/cuattest-perf-20260909.jRCVUz` on `probqa.com`. The owned large-model
producer and servers were stopped and the checkpoint's VRAM was released.
See the [cryptographic regression and sanitizer report](testing/crypto-optimization-2026-09-09.md)
for validation details and limitations.

## Hashing and IPC experiments (2026-09-09)

**All four follow-up ideas were implemented and tested as opt-in experiments.**
The best measured combination was persistent registration plus double-buffered
asynchronous hashing, keeping the original allocation layout: **161.36 ms**
versus **266.87 ms** for current production, a **39.5% latency reduction**.
These are benchmark prototypes, **not new production defaults or a supported
registration API**. Production stream ordering, one-shot IPC acknowledgements,
and storage-release behavior are unchanged.

This comparison starts at **`1ce0fbe`**, already containing the arithmetic
optimizations above, not the older 1.5-second sequential implementation.
It used all eight idle RTX PRO 6000 Blackwell GPUs on `probqa.com`, the same
actual pretrained Qwen checkpoint/revision, all **1,371 selected tensors /
739.102 GiB**, unchanged BF16/F32 bytes, and identical per-tensor GPU placement.
Every warmup and sample rehashed live VRAM and verified every receipt against
pinned session keys and the independently computed CPU root. No cached hashes,
reduced models, dummy allocations, or disabled verification were used.

### End-to-end measurements

| Allocation / IPC / hashing | Samples | Mean (ms) | Median (ms) | Change in mean vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Original / one-shot / production | 220 | 266.87 | 262.53 | baseline |
| Packed / one-shot / production | 120 | 229.85 | 228.47 | −13.9% |
| Original / registered / production | 120 | 199.31 | 198.97 | −25.3% |
| Original / one-shot / full-tile specialization | 80 | 277.49 | 273.29 | **+4.0%, rejected** |
| Original / one-shot / double-buffered async | 90 | 224.83 | 223.72 | −15.8% |
| Original / one-shot / single-buffer async control | 30 | 244.93 | 239.05 | −8.2% |
| Packed / one-shot / double-buffered async | 120 | 204.24 | 202.47 | −23.5% |
| Packed / registered / production | 40 | 203.57 | 203.55 | −23.7% |
| Original / registered / double-buffered async | 60 | **161.36** | **161.25** | **−39.5%** |
| Packed / registered / double-buffered async | 120 | 177.60 | 177.72 | −33.5% |

One-shot timings cover the full existing `Client.sign`, excluding export,
loading and receipt verification, as before. Registered timings cover the
complete token-based HTTP sign request/response; registration and unregister
are separate. Thus registration improves both CUDA mapping work and request
metadata handling; this is not a claim that every saved millisecond was spent
inside `cuIpcOpenMemHandle`. The producer retains its allocation leases across
*all* registered signs and releases them only after confirmed unregister or
the exact owned consumer process's death. Sign response headers alone are not
a release acknowledgement in this experimental protocol.

Each ordinary batch has five warmups. Baseline/registered and packed
baseline/async batches include A/B/B/A ordering; other candidates were screened
and repeated as recorded in the raw data. Layouts were loaded in separate
phases, never together in VRAM. The fastest registered+async configuration has
one 60-sample batch, not a cross-machine performance guarantee. No outliers
were removed (baseline maximum: 459.69 ms); profiler samples are excluded from
the table. Pooling the baseline's four batches is explicit here; its individual
means were 266.96, 271.09, 265.34 and 263.57 ms.

Registration costs for original storage were **302–337 ms**, unregister
**41–43 ms**. Packed storage required **246–275 ms** and **6.9–7.4 ms**,
respectively. This conservative prototype validates registration with an
ordinary unsigned measurement before retaining imports, so these setup costs
include an extra full hash. Both layouts took about a minute to load and
independently CPU-hash from the already-cached checkpoint, outside attestation
timings. One-shot packed measurements do not require the registration protocol.

### Why the candidates differ

- **Packing:** one arena per GPU reduces unique IPC allocations from **454 to
  8**. There are only **8,352 alignment bytes** across the entire model; holes
  are never hashed or counted as weights. Device placement, offsets, storage
  identity and disjoint tensor spans are audited. Importing a 92-GiB arena is
  not constant-cost: trace medians were roughly 1.9–2.5 ms per arena import,
  versus 71–82 µs for ordinary handles. Fewer API calls do not imply a 57×
  speedup. Original and packed layouts both retained over 1 GiB free per GPU.
- **Registration:** the warm trace has **zero IPC opens/closes** between
  registered signs. Eight-GPU hash-start spread falls from roughly **43–50 ms
  to 3.7–4.0 ms**. Kernels still read every byte. A live mutation test produced
  three different CPU-matching signed roots without re-registering, and proved
  that sign completion could not release the producer lease. Stale tokens
  were rejected.
- **Full-tile specialization:** moving uncommon tail/byte-safe work out of
  line reduced Blackwell register usage **94 → 72**, permitting 1,316 instead
  of 940 cooperative blocks. Nevertheless, full-model hash time worsened
  **176.38 → 187.77 ms**. More occupancy alone did not improve latency.
- **Async staging:** a lane-owned ping-pong buffer stages the next 64-byte
  compression blocks using `cp.async` while the current blocks are compressed.
  Copy completion precedes every read/reuse; no divergent block barrier is
  introduced. Tails and unaligned inputs retain their original byte-safe path.
  Shared memory rises from **11,152 to 43,920 bytes/block**, reducing the grid
  from 940 to 376 blocks, yet original-layout full-model hashing improves
  **176.38 → 138.74 ms**. The one-buffer control is slower than double buffering.
  This candidate is *not* a universal win: the cached 64-MiB probe regresses,
  and the packed layout's async hash takes **153.49 ms**. The measured layout
  sensitivity explains why stacking all three winning ideas is not optimal;
  its precise memory-system cause has not been isolated.

The implementation follows NVIDIA's documented
[asynchronous-copy completion rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)
and [IPC allocation lifetime requirement](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html).
The prototype's copy instructions are guarded for `sm_80+`; performance and
sanitizer qualification here are specifically on `sm_120`. A production
rollout needs workload-aware kernel selection and a separately reviewed
registration/producer-readiness API, not a hidden mapping cache under `/v1/sign`.

See [raw timings and trace summaries](performance/ipc-hash-2026-09-09.json),
[reproducible experiment tools](../tests/performance/experiments/ipc_hash/README.md)
and [validation results](testing/ipc-hash-2026-09-09.md). Full original logs and
Nsight files are in `/tmp/cuattest-ipc-hash.nqZjYM` on `probqa.com`; the local
campaign directory is `/tmp/cuattest-ipc-hash.WjsXJI`. Owned benchmark processes
were stopped and their large-model VRAM released. The local RTX 2080 follow-up
was interrupted by a driver/library mismatch (loaded kernel module 610.43.02,
installed userspace 610.57.04); no local model speedup is claimed.

## Earlier capacity and correctness checks

On 2026-09-07, a full-layout capacity test on `probqa.com` held all 739.102 GiB
at once: approximately 92.388 GiB per GPU, or 97.24% of each GPU's 95.010 GiB
CUDA-reported capacity (lower than the board's advertised capacity). It left
1.228–1.234 GiB free per GPU and passed three independently CPU-checked signed
attestations. This used initialized weights with the pinned model's exact
names, shapes, and stored dtypes while the checkpoint was downloading; it is
**not a pretrained timing result**. Every benchmark run records actual unique
CUDA storage, free VRAM after loading, and PyTorch allocated/reserved memory
separately.

That capacity validation used kernel CID
`bafkr4iaowp7shbwcdpx4tljkegxyad2uxxbm2w7iie4bxunl5gg7bux25m` and the `sm_120`
CUBIN CID `bafkr4ieflyskm7ncp7jncevzufn5uus33gxfmdjpbwipwqagtq6veloile`.

The default reserves **1 GiB per GPU** after both processes' CUDA contexts
and the signer's lazy launch resources exist, including space for notary
scratch and allocator slack. It requires
weight storage to occupy at least 90% of **each** selected GPU, rejects models
that cannot fit, and rechecks free memory after downloading. It never silently
shrinks the model or counts tied aliases / allocator cache as extra resident
weights. If another workload is using the machine, stop this benchmark and
choose an idle window; it does not evict other GPU processes.

Each tensor is independently BLAKE3-hashed on the CPU while it is loaded from
the mapped checkpoint. Every warm-up and timed receipt must verify under the
running notaries' pinned UUID/key map and match that independent global root.
As in the original benchmarks, only `Client.sign(...)` is timed: downloads,
loading, IPC export, and cryptographic verification are excluded. Current
multi-GPU services execute shards concurrently; reported throughput remains
aggregate end-to-end input throughput, not a peak memory-bandwidth measurement.

The JSON output includes the exact model revision, excluded prefixes, model
root, raw samples, mean/deviation, per-device occupancy, and notary identities
with kernel/CUBIN CIDs. The output file must be new to avoid overwriting an
earlier result. Keys obtained through `/v1/info` assume a trusted local setup
channel, as in the existing benchmark; an untrusted endpoint cannot establish
its own identity.

Regression coverage does not fill VRAM during ordinary test runs:

```bash
pytest -q tests/test_benchmarks.py
CUATTEST_TEST_MULTIGPU=1 pytest -q tests/test_benchmark_large_integration.py
```

The integration tests send a small real BF16 checkpoint through the exact
loading, placement, HTTP, verification, and JSON-report path on every visible
GPU, with reversed producer/notary ordinals and both host backends. Only these
smoke tests lower `--min-utilization` to zero.

An additional resource regression temporarily reserves nearly all free VRAM
and should run only on an idle GPU:

```bash
CUATTEST_TEST_LARGE_CAPACITY=1 pytest -q tests/test_large_capacity.py
```

It leaves 1 GiB free, then exercises the first receipt launch through both
backends. This caught CUDA's original local-memory backing allocation: large
receipt serialization buffers were allocated per thread even though only
thread zero used them. The public buffers now use block-shared scratch;
the final model-root fold's streaming state is also shared after parallel
hashing finishes. Private signing-lane and parallel hash state remain separate.
In that earlier `sm_75` build, the measurement kernel used a 64-byte stack and
11,152 bytes of shared memory (96 registers); the receipt kernel used a 2,240-byte stack
and 15,192 bytes of shared memory. This test's unused reservation is not a
model or a timing sample.

## Production registration and asynchronous hashing (2026-09-09)

The selected combination is now implemented: **persistent IPC registration +
double-buffered asynchronous hashing, without repacking weights**. Registration
is opt-in because it changes storage lifetime; the existing one-shot API stays
available. Every observation still hashes every logical byte and produces fresh
signed evidence. No tensor/model digest is cached.

The final public API includes producer-stream readiness and **both pre- and
post-observation layout/storage checks** inside the timed call. Its full-model
result is **172.01 ms vs 264.73 ms before integration: 35.0% lower latency,
1.54× faster**. The earlier ~161-ms experimental/provisional result did not
include all these production checks; it is not the final public-API latency.
[Raw samples, identities, placement and size sweep](performance/registration-2026-09-09.json).

### Full pretrained checkpoint on eight Blackwell GPUs

Same pinned Qwen3.5-397B-A17B main-model state as above: **1,371 tensors,
793,604,738,912 logical bytes (739.102 GiB)**, excluding only the documented
auxiliary `mtp.*` weights. The original allocation layout has **454 unique IPC
handles**, occupies about **97.2%** of reported device memory per GPU, and leaves
about 1.23 GiB free per device after loading with warmed notaries. There is no
packing, quantization, offload, synthetic padding, inference or duplicate
full-model allocation.

| Implementation / API | Samples | Mean ± sample SD (ms) | Median (ms) | Mean change vs previous |
| --- | --- | --- | --- | --- |
| Previous `1ce0fbe`, one-shot | 60 | 264.73 ± 12.35 | 263.53 | — |
| Integrated code, forced standard hash, one-shot | 60 | 265.93 ± 8.89 | 263.94 | +0.5% |
| Integrated code, automatic async hash, one-shot | 60 | 230.29 ± 13.87 | 226.78 | −13.0% |
| Registered IPC, forced standard hash | 60 | 210.14 ± 7.65 | 210.47 | −20.6% |
| **Registered IPC + automatic async hash** | **60** | **172.01 ± 9.82** | **171.69** | **−35.0%** |

Two cycles ran each configuration over the **same resident tensors**, reversing
configuration order in the second cycle. Each trial used five excluded warmups
and 30 recorded observations. All outliers remain, including the 224.03-ms
maximum for the selected combination. Every warmup and sample was verified
against pinned session public keys and the independently CPU-computed root:

```text
077a5b303b1d6ed829a9c30fdb0102cf61e63fd5bcc562f8ebe153e10fdbe805
```

Setup/retirement are separate from steady-state observations: registration
took **98.80–148.75 ms** and acknowledged close **44.60–51.18 ms**, across the
four registered trials. The timed operation is the **complete public
`RegisteredModel.sign`** or `Client.sign`, including HTTP/JSON and relevant
per-call ownership checks. Loading, independent CPU hashing, export/register,
close, receipt verification and profiler instrumentation are outside it.

### Laptop and automatic-dispatch qualification

The reboot restored the RTX 2080's CUDA/NVML path; its driver is **610.57.04**.
The same pinned GPT-2 and GPT-2 Large benchmarks ran with the normal desktop
workload left running. Each cell below includes 40 observations across two
reversed-order trials, with five warmups excluded per trial:

| RTX 2080, automatic mode | One-shot mean (ms) | Registered mean (ms) | Reduction |
| --- | --- | --- | --- |
| GPT-2 | 15.95 | 14.60 | 8.4% |
| GPT-2 Large | 38.27 | 30.03 | 21.6% |

Automatic mode uses the **original kernel on `sm_75`**. Its small differences
from forced-standard trials are run-to-run variation, not another algorithm.
The attempted Blackwell GPT-2 Large run lacked an offline checkpoint cache and
produced no timings; no result for that combination is claimed.

The Blackwell size sweep supports a conservative default. Direct-pointer hash
medians, including stream handoff/launch/completion, were:

| Input | Standard (ms) | Async (ms) |
| --- | --- | --- |
| 64 MiB | 0.098 | 0.111 |
| 128 MiB | 0.135 | 0.140 |
| 256 MiB | 0.565 | 0.530 |
| 1 GiB | 1.956 | 1.881 |
| 4 GiB | 7.571 | 6.160 |

Therefore `CUATTEST_HASH_MODE=auto` selects async only for qualified **`sm_120`
with at least 1 GiB aggregate logical input per GPU**. Other architectures and
small requests retain standard hashing. `standard` is the A/B escape hatch;
`async` forces the path on `sm_80+` for qualification, not as an unmeasured
universal recommendation.

The two entry points share BLAKE3 tail handling, reduction and the private
signing handoff, but require different cooperative launch limits. On this
Blackwell build, standard uses **94 registers / 11,152 shared bytes / 940
blocks**, async **95 registers / 43,920 shared bytes / 376 blocks**, with zero
local bytes in either measurement kernel. Lane-owned double buffers overlap
16-byte copies with compression; each copy group is waited before reads/reuse.
Incomplete chunks use the existing length-aware BLAKE3 path, never additional
logical padding. The standard entry point does not pay for async shared memory.

### Use and reproduce

```python
from cuattest.client import Client

with Client().register_model(model) as registered:
    receipt = registered.sign("my-model")
    # Values may change BETWEEN completed observations; storage must stay fixed.
    receipt = registered.sign("my-model")
```

The producer drains only the declared CUDA streams (default: current per GPU),
not the whole device. Join other writers or supply
`streams={ordinal: [writer_stream, ...]}`. The consumer retains private
nonblocking streams and concurrent GPU shards. Storage stays pinned until
matching close acknowledgement; ambiguous completion is quarantined and
recoverable, not GC-released. See the [API/lifetime contract](api.md#repeated-observations).

Both existing benchmark CLIs accept `--registered`. For controlled same-layout
A/B testing with owned services:

```bash
python tests/performance/benchmark_registered.py \
  --output /tmp/new-registration-benchmark --cubins /path/to/current/cubins \
  --checkpoint /path/to/pinned/Qwen/checkpoint \
  --baseline-source /path/to/1ce0fbe --baseline-cubins /path/to/baseline/cubins \
  --runs 30 --warmup-runs 5 --cycles 2
```

Omit `--checkpoint` to use the cached pinned GPT-2, or select the existing
GPT-2 Large revision with `--model`. These controlled A/B runs are offline;
prepare checkpoints first. `benchmark_hash_sizes.py` reproduces the cutoff
sweep. The new CUDA source changes code identity: rebuild CUBINs and refresh
source/CUBIN allowlists before deployment.

Validation includes **595 CPU tests, both host backends, real all-eight-GPU IPC,
independent cryptographic comparisons, ASan/UBSan/TSan and all four Compute
Sanitizer tools**. The [validation report](testing/registered-ipc-2026-09-09.md)
explicitly retains the raw duplicate-import memcheck limitation and the
inconclusive initial mixed-backend racecheck run, alongside the successful
scoped reruns. All owned remote processes exited and their VRAM was released;
GPU clocks and unrelated workloads were not changed.

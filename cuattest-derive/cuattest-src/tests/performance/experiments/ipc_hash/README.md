# Hashing and IPC latency experiments — 2026-09-09

Test-only prototypes of all four follow-up suggestions. The production API,
CUDA source, and native extension are not modified. Read the
[measurements and limitations](../../../../docs/performance.md#hashing-and-ipc-experiments-2026-09-09)
before using these tools. Do not deploy this single-client loopback server.

The source generator requires the exact **1ce0fbe** CUDA source, SHA-256
`638ee4bf4209a262c304eb6f9798c3be054c1613f034295a85c794b344648a5b`.
Its candidates are `fulltile` (rejected), `async` (double-buffered), and
`async_single` (one-buffer control). No forced register cap is applied.
Each compilation records source/CUBIN SHA-256, architecture and NVRTC version.
An `sm_80+` guard keeps unsupported copy instructions off older architectures;
only Blackwell received complete GPU qualification in this campaign.

## Compile and check

Use a fresh snapshot and an isolated writable cache. Build its native extension
with the Python used to run the server. Never rely on an editable installation
pointing at another checkout or on an old CUBIN directory alone.

```bash
trial_dir=$(mktemp -d /tmp/cuattest-ipc-hash.XXXXXX)
git archive 1ce0fbe | tar -x -C "$trial_dir"
export PYTHONPATH="$trial_dir/src"
export CUATTEST_KERNEL_SRC="$trial_dir/src/cuattest/kernel/p256_cuda_notary_b3.cu"
export CUATTEST_CACHE="$trial_dir/cache"

.venv/bin/python tests/performance/experiments/ipc_hash/variants.py \
  "$trial_dir/variants" --arch sm_120

export CUATTEST_KERNEL_SRC="$trial_dir/variants/async/kernel.cu"
export CUATTEST_KERNEL_DIR="$trial_dir/variants/async/cubins"
.venv/bin/python tests/performance/experiments/ipc_hash/probe.py
.venv/bin/python tests/performance/experiments/ipc_hash/sanitize.py \
  "$trial_dir" "$trial_dir/async-sanitizers"
```

`probe.py` independently checks 181 complete/partial/unaligned/tree cases and
repeated 64-MiB hashes using Python `blake3`. `sanitize.py` runs memcheck,
initcheck (global and shared), racecheck and synccheck, with both native and
fallback hosts, checking exit codes and diagnostic summaries. Results under
instrumentation are **not** benchmark samples.

## Resident full-model A/B

`resident.py SNAPSHOT OUTPUT CHECKPOINT --cubins BASELINE_CUBINS` loads the
actual pinned Qwen checkpoint, independently hashes it on the CPU, warms
notaries, and retains it for queued owned-server jobs (maximum three hours).
The output directory must be new. It requires the eight-GPU capacity used in
the report; it does not evict other workloads. Run on idle hardware only.

For the packed phase add `--packed --placement ORDINARY_OUTPUT/resident-layout.json`.
**Stop the first producer before loading the second phase.** Packing happens
during checkpoint loading, not by allocating another full GPU model. It keeps
logical names, dtypes, shapes and placement unchanged, aligns tensor starts to
256 bytes and reports padding separately from model bytes.

Queue jobs from another terminal after `READY` appears:

```bash
.venv/bin/python tests/performance/experiments/ipc_hash/enqueue.py \
  OUTPUT 01-baseline --runs 60
.venv/bin/python tests/performance/experiments/ipc_hash/enqueue.py \
  OUTPUT 02-async --variant VARIANTS/async --runs 60
.venv/bin/python tests/performance/experiments/ipc_hash/enqueue.py \
  OUTPUT 03-registered-async --registered --variant VARIANTS/async --runs 60
```

Jobs sort by filename; enqueue reversed repetitions to counter ordering effects.
Omitting `--variant` uses the baseline CUBIN selected when starting the producer.
Do not inherit a candidate `CUATTEST_KERNEL_SRC` when starting a baseline run.
`--trace --runs 3 --warmup-runs 1` starts Nsight capture after initialization.
`traces.py OUTPUT` exports SQLite and distinguishes sign kernels from the
extra registration-validation hash. Write an `OUTPUT/STOP` sentinel after all
queued jobs complete to release the checkpoint and exit the producer.

`report.py CAMPAIGN OUTPUT_JSON` combines `ordinary/` and `packed/` raw results,
keeps every non-warmup timing, deduplicates repeated placement/root metadata,
and excludes profiler samples from ordinary aggregates. Full logs and raw
per-job results remain authoritative.

Summary keys have the form `LAYOUT/JOB`, for example `ordinary/01-baseline`
and `packed/01-baseline`. Repetitions ending in `-a`, `-b`, `-c` or `-d` are
combined only within the same layout and job; reusing a job label in the two
layout directories never pools their A/B samples. Raw trial labels and timings
remain unchanged.

## Registration contract — intentionally not `/v1/sign`

`server.py --registered` accepts only `/experiment/register`,
`/experiment/sign`, and `/experiment/unregister` POSTs. Each server has one
opaque, generation-specific registration and serializes requests. It retains
validated **mappings, not digests**; sign still launches the real hashing and
receipt kernels and uses the normal multi-GPU manifest and verification.

The producer claims its process-owned allocation leases once and holds them
for the entire registration. **Neither sign headers nor a generic HTTP error
release those leases.** Only confirmed unregister or confirmed death of the
exact owned consumer PID permits release; ambiguity remains quarantined.
The harness owns that process through a pidfd, never kills unrelated work,
and fails closed on registration/sign/unregister errors. This is a bounded
experiment, not an authenticated multi-client registration service.

The timed checkpoint remains immutable throughout. `check_registration.py
SNAPSHOT NEW_OUTPUT` separately proves fresh hashing after synchronized
between-request mutations, refusal to release an active producer lease, and
stale-token rejection. A public reusable API would need an explicit producer
stream/event handoff, per-observation mutation guards and comprehensive
disconnect/teardown fault qualification. Do not simply retain maps behind the
existing one-shot `Client.sign` completion semantics.

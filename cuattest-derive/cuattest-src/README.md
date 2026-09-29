# cuAttest

**cu**DA **attest**ation — a P-256 notary whose signing state is retained
**inside the GPU**.

The trusted notary supplies an operating-system CSPRNG seed to a CUDA kernel,
which derives a private scalar and retains it in device globals. The scalar is
not copied back by cuAttest, although the trusted host can reproduce it from
the seed. The notary hashes another process's live VRAM over CUDA IPC and signs
a measurement document — plus a set of EQTY statements — entirely on the
device. The key dies with the process, so a notary's identity spans exactly one
run.

Two processes with separate state: the notary is trusted, while the model
workload need not be:

```
process A: the notary            process B: the model
─────────────────────            ────────────────────
holds the P-256 key              holds the weights
never sees the model             never sees the key
                 ← IPC handles ──
hashes B's live VRAM in place
signs what it hashed
                 ── document ──→  verifies the fold and the signature
```

The client chooses the IPC spans. A signature authenticates the submitted,
bounds-checked VRAM bytes; it does not prove that an inference runtime used
them. A compromised workload can submit decoy buffers unless a separate
trusted integration binds the spans to the executing graph.

## Documentation

**[docs/index.html](docs/index.html)** — how it works, illustrated. Open it in a
browser; it is self-contained, with no build step and nothing to fetch.

[Quickstart](docs/quickstart.md) · [How it works](docs/how-it-works.md) ·
[Expected CID](docs/expected-cid.md) · [The kernel](docs/kernel.md) ·
[API](docs/api.md) ·
[Trust model](docs/trust-model.md) ·
[Operations](docs/operations.md) ·
[Performance](docs/performance.md) ·
[Testing](TEST.md)

[Build configurations and assertions](docs/build-configurations.md) documents
Release (default), Debug, and optimized AssertedRelease builds and their test matrix.

## Install

Create a virtual environment once:
```bash
python3 -m venv .venv
. .venv/bin/activate
```

Rebuild the program upon a change:
```bash
pip install .                 # core + independent host-side BLAKE3
pip install '.[client]'       # + torch, to share your own tensors
pip install '.[build]'        # + NVRTC and cooperative-groups headers
pip install '.[verify]'       # + cryptography, to verify signed receipts
```

On Linux, a source install compiles a small C++17 extension for the
latency-sensitive CUDA host path, so it needs a C++ compiler. Other platforms
retain the Python fallback for offline tooling. `libcuda` and `libnvrtc` are
still opened at runtime rather than linked, so importing the package needs no
GPU or CUDA toolkit. The small host BLAKE3 dependency computes code identities
outside the CUBIN being identified.

## Install the kernel

"Installing the kernel" means producing machine code for a GPU architecture.
The CUBIN is what actually runs, so it is what gets hashed and registered — the
`.cu` travels with the package but is never substituted for the compiled form.

```bash
cuattest build-kernel                       # this GPU's architecture
cuattest build-kernel --arch sm_90 --arch sm_100 --out ./cubins
```

NVRTC needs no GPU, so CUBINs can be built on a build machine and shipped.
At startup the notary looks for one in `$CUATTEST_KERNEL_DIR`, then the
package's `cubins/`, then `~/.cache/cuattest/cubins`, and compiles on demand
if it finds none. Every CUBIN has a digest sidecar binding it to the exact
source; a package upgrade or `CUATTEST_KERNEL_SRC` change invalidates stale
architecture-named cache entries automatically.

## Use

```bash
cuattest selftest      # keygen, hashing, measurement and signing, end to end
cuattest info          # this session's identity and the code it is running
cuattest serve         # HTTP on 127.0.0.1:8077
```

```
GET  /v1/info      identity: did:key, public key, kernel and CUBIN CIDs
POST /v1/measure   {"tensors": [{"handle", "nbytes", "seg_off", "t_off", "device"}, ...]}
POST /v1/sign      {"tensors": [...], "model": "..."}  # atomic measure + sign
GET  /healthz
```

As a library:

```python
from cuattest import Notary
n = Notary()
print(n.info.gpu_did)                    # did:key:z...
m = n.measure(refs)                      # unsigned diagnostic measurement
receipt = n.sign(refs, "my-model")       # atomic signed measurement
n.close()
```

From the process that owns a model:

```python
from cuattest.ipc import share_model
from cuattest.client import Client

names, refs, keepalive = share_model(model)   # zero copy for ordinary contiguous tensors
receipt = Client().sign(refs, "my-model")
del keepalive        # only after the atomic request returns
```

For repeated observations of stable storage, register once:

```python
with Client().register_model(model) as registered:
    receipt = registered.sign("my-model")
    # Update weights only BETWEEN completed observations, without reallocating.
    receipt = registered.sign("my-model")  # hashes all bytes again
```

Registration retains IPC mappings and pins producer storage until explicit
close. Each observation checks the tensor layout/storage and waits only for
the current producer stream on each GPU; join other writers first or pass
`streams={ordinal: [writer_stream, ...]}`. Do not mutate during a call.
See the [registration API and recovery contract](docs/api.md#repeated-observations)
and [performance results](docs/performance.md#production-registration-and-asynchronous-hashing-2026-09-09).

`keepalive` matters: quiesce writers before sharing, then keep tensors alive and
immutable until the request is acknowledged. Freeing one could recycle its
segment; mutating one could yield a
torn cross-tensor state while the notary hashes. The client rejects PyTorch
version-counter changes and retires its process-owned storage leases after a
successful or definitive response. A timeout or disconnect is different: the
server may still be hashing, so the client keeps the original storage pinned
in its lease registry and raises `NotaryRequestUncertainError`. Keep `keepalive`
referenced, keep the tensors immutable, and call
`keepalive.release(server_completed=True)` only after an independent signal
proves that request completed, was cancelled, or its notary process exited.
Each `refs` list is one-shot: after any client request starts, call
`share_model()` again before measuring the same live model a second time. This
prevents a surviving handle from outliving its storage lease or mutation guard.
One-shot/in-flight state belongs to the process lease registry, so copying or
JSON-round-tripping the refs cannot reset it. Calling `keepalive.release()`
while its `Client` request is active is rejected and leaves the storage pinned.
Packed `torch.quint4x2` and `torch.quint2x4` tensors are rejected because the
current byte-span protocol cannot represent their physical bit-packed extent.
Lazy conjugate/negative views and non-contiguous layouts are materialized,
synchronized, and retained before sharing so hashes cover logical tensor values.
The client requires legacy-IPC-compatible allocations: configure PyTorch with
`backend:native,expandable_segments:False` **before allocating the model**.
Expandable-segment/VMM and `cudaMallocAsync` storage receive an explicit error;
see [allocator setup](docs/operations.md#client-allocator-compatibility).

See `examples/measure_torch_model.py` for the whole flow.

## What it signs

One `/v1/sign` measures and signs before another request can intervene. Its
receipt contains the exact document bytes as hex; decode those bytes rather
than re-serialising the document, because the signature covers them:

- a **measurement document** with the model root, device ordinal, source CID,
  CUBIN CID, server timestamp, and a detached P-256 signature
- a versioned **manifest** of statements: one notary-issued `StateAttestation`
  credential whose subject is the GPU itself, carrying a `state` with what the
  bytes are (`modelRoot`), which resident copy they came from (`instanceID`)
  and which credential preceded it for that copy (`previousStateCredential`).

The notary service adds up to two more statements before returning, both
signed by its own durable key:

- an `IdentityAttestation` binding this session's GPU DID to the CUBIN, kernel
  source and device it loaded. The GPU cannot check those values — it signs
  digests the service hands it — so issuing them under a pinnable service
  identity makes the claim attributable rather than anonymous.
- a `gpuModelTensorsV1` state report, when the host has declared what it
  loaded (`Notary.declare_loaded_model`). Its `modelCID` is the caller's
  Model asset CID: use the same raw-file or Iroh collection CID as the Model
  input to the signed SDK computation. `modelRoot` independently identifies
  GPU tensor bytes, alongside `instanceID` and `reportedBy`. These identifiers
  need not match; the host binds them but does not prove a faithful load.
  Raw source tensor CIDs from `safetensors.model_cid` remain supported for
  callers explicitly comparing identical tensor bytes and ordering.

`verify_evidence()` reconstructs the credential's detached-JWS signing input
and verifies its ES256 proof with the same pinned GPU public key as the
measurement document. The manifest is mandatory: removing it is a verification
error, not a measurement-only downgrade. Statements issued by anyone other than
the GPU are reported in `unverified_statements` rather than trusted;
authenticating them needs that issuer's key pinned separately.

Each resident copy keeps its own chain, so repeated observations of one model
link together while two byte-identical copies in separate allocations stay
apart — `modelRoot` cannot tell them apart, `instanceID` can.

CIDs are CIDv1 raw/blake3-256, so a tensor's digest *is* its content id. The
`did:key` is multicodec `0x1200` over the compressed public point.

## Running against a container

The notary listens on loopback and CUDA IPC handles only resolve inside a
shared IPC namespace, so a containerised client needs both:

```bash
docker run --gpus all --network host --ipc host ...
```

## Requirements

- C++17 compiler for a Linux source install.
- NVIDIA driver (`libcuda.so.1`). CUDA toolkit only for `build-kernel`.
- Compute capability 7.5 or newer, with cooperative kernel launch support.
- On an H100/H200 in confidential-compute mode the GPU must be unlocked
  (`nvidia-smi conf-compute -srs 1`) or `cuInit` fails with 802; the error
  message says so.

## Multi-GPU

For a near-capacity pretrained workload across all eight GPUs on `probqa.com`,
see the [739-GiB Qwen3.5 benchmark](docs/performance.md#near-capacity-eight-gpu-benchmark).
The existing GPT-2 and GPT-2 Large benchmark defaults are unchanged.

For models spread across GPUs, run `cuattest serve --devices all` and use the
same `share_model()` / `Client.sign()` calls. GPU UUIDs route each span to its
owner; the aggregate receipt authenticates every GPU's contribution. See
[multi-GPU setup and verification](docs/multi-gpu.md) for key pinning and usage.

## Tests

```bash
pytest tests         # CPU regressions; no GPU needed
cuattest selftest     # the real path, on a GPU
CUATTEST_TEST_GPU=1 pytest tests/test_gpu_regressions.py  # both host backends
CUATTEST_TEST_MULTIGPU=1 pytest tests/test_multigpu_integration.py  # 2+ GPUs
```

Independent RDF normalization tests use optional `pyld`; saved-model integration
tests use optional `torch`, `transformers`, and `safetensors`. Install these and
the `verify` extra to run the complete suite. Frozen RDF canonicalization vectors
also run without PyLD. GPU regressions require CUDA and the native host extension.

## Layout

```
kernel/p256_cuda_notary_b3.cu   the CUDA kernel: BLAKE3, P-256, statements
src/cuattest/
  _native.cpp   C++ CUDA IPC, planning, launch, and reusable workspaces
  _cuda.py      Python CUDA driver binding for setup and the fallback path
  _nvrtc.py     NVRTC, for compiling the kernel
  kernel.py     locating, compiling and caching the CUBIN
  notary.py     the session: keygen, hash, measure, sign
  ids.py        CID and did:key encoding
  ipc.py        client side — share torch tensors, zero copy
  client.py     client side — talk to a running notary
  server.py     HTTP front end
  cli.py        the cuattest command
```

## Scope

This is the GPU notary and nothing else. It does not do TEE or TPM attestation,
does not talk to a key broker, and has no opinion about how a manifest gets
stored. It answers who it is, what is in the memory you point it at, and signs
the answer.

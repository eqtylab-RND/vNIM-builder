# How it works

> The illustrated version of this page is [`index.html`](index.html) — same
> mechanism, with the split, the hash tree and the statement graph drawn out.
> This one is the text companion, for reading in a terminal or on GitHub.

## The problem

You can check a model file's hash on disk. That tells you nothing about the
contents of specific VRAM spans ten minutes later. cuAttest measures spans a
client submits; it does not determine whether an inference runtime used them.

## The split

cuAttest splits the problem across two processes.

Two processes, one secret each.

```
notary process                          model process
──────────────                          ─────────────
P-256 key, generated inside the GPU     the weights
never sees the model                    never sees the key
        ◄────── IPC handles ────────────
hashes the other process's VRAM
signs what it hashed
        ─────── document ──────────────►  verifies fold + signature
```

The notary is trusted; the model workload is not. Separating them prevents a
compromised workload from directly using the signing state. The caller
re-computes the document's internal fold and checks its signature against a
public key obtained through a trusted channel. Those checks detect corrupted
evidence, but they cannot make a dishonest notary truthful.

## The five steps

**1. Keygen, once per process.** The trusted notary obtains a 32-byte seed from
the operating-system CSPRNG. A CUDA kernel hashes that seed, maps it to a P-256
private scalar, and retains the scalar in device globals. CUDA timing and
scheduling effects are not credited as entropy. The host knows the seed and
can reproduce the scalar; device-only storage separates signing state from the
untrusted workload, not from the trusted host. The scalar dies with the
process, so a notary's identity spans exactly one run.

**2. Load from independently hashed bytes.** A host BLAKE3 implementation
hashes the source and CUBIN before CUDA loads the module. The registered
artefact is the machine code that ran, not a digest reported by that same
executable or a file that happened to sit nearby. A source-and-binary sidecar
invalidates stale cache entries.

**3. Share, don't copy.** The model process asks the CUDA driver for an IPC
handle to the allocator *segment* holding each tensor. This deliberately avoids
torch's private `_share_cuda_()`, whose storage-ownership transition is not
transactional when its later event or Python-object construction fails.
The notary maps each unique segment once and reads the tensor at
`segment_base + seg_off + t_off`. CUDA ordinals are process-local, so producer
metadata is validated but not compared with the notary's ordinal. The imported
allocation's driver-reported owner in the notary namespace is authoritative;
a different owner is rejected. The complete span is checked against the
driver-reported allocation before any hash kernel launches.
These are memory-safety and device-identity checks, not a semantic binding.
The client chooses every handle and offset and can submit a pristine decoy
buffer unrelated to the graph it actually executes.
Zero-element tensors are omitted on both the runtime and checkpoint sides: no
non-empty byte span exists to authenticate. Non-contiguous tensors are copied
and all affected devices are synchronized before their handles are shared.
Packed `quint4x2`/`quint2x4` tensors are rejected rather than deriving a false
byte length from PyTorch's rounded `element_size()` value.
Shared tensors must then remain immutable until acknowledgement; the client
uses PyTorch version counters as a best-effort mutation tripwire. Response
headers are sent only after every mapping closes or successful context
destruction releases them all. If neither cleanup can be confirmed, the server
closes without a response and exits, so the producer keeps its process-owned
storage leases quarantined. Contiguous tensors remain zero-copy, and the
notary never allocates a model-sized copy.

The latency-sensitive host portion is one compiled C++ call. It imports and
bounds-checks all unique IPC allocations, builds the fused reduction schedule,
marshals and launches the kernel, copies back only the final output, and closes
every mapping before returning to Python. Its small metadata and reduction
workspaces are retained and grown between requests; model weight bytes are never
copied into them. Python remains responsible for request validation, service
policy, evidence checking, and HTTP.

**4. Hash in place.** BLAKE3, on the GPU: threads preserve its semantic
1024-byte chunks, but schedule them in 128 KiB tiles. Each tile's first seven
binary-tree levels stay in block-local shared memory, and only one 32-byte CV
per tile enters the global reduction workspaces.

```
all tensor bytes ─► 1 KiB chunk compression (128 KiB scheduling tiles)
                 ─► seven block-local tree levels (shared memory)
                 ─► one CV per tile ─► grid.sync()
                 ─► cross-tile tree level ─► grid.sync() ─► …
                 ─► ordered 32-byte digest per tensor
                 ─► model-root fold

model root = BLAKE3( LE32(count) ‖ digest₁ ‖ digest₂ ‖ … )
```

This entire measurement pipeline is one cooperative kernel launch per request.
Its grid is capped at the occupancy-safe number of concurrently resident
blocks, as required for a grid-wide barrier; grid-stride loops make that fixed
resident grid cover models of any supported size. The digest *is* the content
id: CIDv1, raw codec, blake3-256.

**5. Measure and sign atomically.** A single serialized request maps the
tensors and captures a server timestamp. The cooperative measurement kernel
folds the model root and publishes it to a module-private, one-shot handoff. A
small finalization kernel queued immediately behind it on the same CUDA stream
atomically consumes that root, assembles the receipt, and computes its two
signatures in parallel. There is one host synchronization and no intermediate
digest copy or host-supplied signable root. The pending root is invalidated at
measurement entry and wiped after consumption, so there is no mutable "last
measurement" that a second client can replace. The finalization kernel builds:

- a **measurement document** — model root/count, timestamp, model name, GPU
  ordinal, signer DID, source CID and CUBIN CID — with a detached P-256 signature
- a **manifest** (version 3) holding one registered `StateAttestation`
  credential.

That credential is the interesting one. Its subject is the GPU itself, and the
`state` nested in that subject carries four things: `modelRoot`, the BLAKE3
fold over the submitted spans; `instanceID`, a fold over the hashed IPC
handles and extents that identifies *which resident copy* was measured;
`stateType`, versioning the payload; and `previousStateCredential`, linking to
that copy's previous credential by id, or null for the first in a session.

The GPU's `stateType` is `gpuModelTensorsStateV1` — what the device found.
The host issues a separate `gpuModelTensorsV1` report about the same copy,
signed by its own key. `modelCID` identifies the caller's Model asset and must
match that asset's input CID in a signed SDK computation. It can name a raw
file or an Iroh collection. `modelRoot` identifies the GPU tensor measurement;
`instanceID` identifies the resident copy and `reportedBy` names the host.
The asset CID and GPU root need not match and their signed association does
not prove faithful loading. Only callers choosing the raw tensor-root helper
with matching bytes and span ordering can compare the two directly.
The report is issued after the first measurement exists; the declared asset
identity is fixed at load.

The state travels inside the signed document rather than beside it, so the
GPU's JWS covers those values directly. The credential's own `id` is a UUID
derived from the canonicalized preimage of everything it asserts, so a receipt
whose `state` was edited after signing no longer hashes to the id inside its
own signed document.

Content addressing cannot distinguish two identical models, because identical
bytes hash identically — that is the point of it. `instanceID` is keyed on
residency instead, so it stays constant across repeated observations of one
copy and differs between two copies holding the same weights. Each copy
therefore gets its own chain.

The narrow claim is "this code hashed these submitted spans"; it is not an
inference-execution claim.

The service then adds an `IdentityAttestation` signed by its own durable key,
binding the session's GPU DID to the CUBIN, kernel source and device. That
covers the one thing the GPU cannot attest about itself: the code identity
values are computed on the host and handed to the kernel, which signs them
without being able to check them.

## Continuous measurement

Nothing stops you re-measuring. Obtain fresh one-shot refs for the still-live
spans with `share_model()` and a later cycle is just the hashing again. If the
root changes, bytes in those submitted spans changed; comparing the
independently verified per-position digests says which submitted entry moved.
An unchanged root still does not establish that the runtime used those spans
for inference.

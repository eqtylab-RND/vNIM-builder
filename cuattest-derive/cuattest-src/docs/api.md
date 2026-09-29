# API

## HTTP

The sections below describe single-GPU receipts. `serve --devices all` uses the
same routes with [multi-GPU info and aggregate receipts](multi-gpu.md).
References exported by the current client also include `device_uuid`, required
for multi-GPU routing and optional for a single-GPU service.

Requests queue by design; a multi-GPU request runs independent GPU shards
concurrently and joins all of them before acknowledging completion. Each
request line and complete header block has one 10-second monotonic wall-clock
deadline that trickled bytes do not reset; POST bodies then have a separate
30-second total deadline. Before importing CUDA IPC handles, the service also
enforces operator-configured ceilings on span count, aggregate logical bytes,
and 128 KiB scheduling tiles; repeated references count repeatedly.

### `GET /v1/info`

```json
{
  "gpu_did": "did:key:zDnae...",
  "gpu_pubkey_uncompressed": "04df1c...",
  "kernel_cid": "bafkr4idcw7...",
  "cubin_cid": "bafkr4ifxwk...",
  "compiler": "13.0 (prebuilt)",
  "arch": "sm_90",
  "device": "NVIDIA H100 NVL",
  "device_ordinal": 0,
  "pid": 53,
  "host_backend": "C++"
}
```

### `POST /v1/measure`

```json
{"tensors": [{"handle": "<hex>", "nbytes": 16384, "seg_off": 0, "t_off": 0, "device": 0}]}
```

`handle` is a CUDA IPC handle, hex encoded; torch's two-byte header is stripped
if present. `device` is the producer's process-local CUDA ordinal and is kept
for protocol diagnostics, but it is not compared with the notary's ordinal:
`CUDA_VISIBLE_DEVICES` can give the same physical GPU different ordinals in the
two processes. After importing the handle, the notary asks the driver for its
owner in the notary namespace and requires that owner to be the selected
device. Returns:

```json
{"digests": "<hex, 32 bytes per tensor>",
 "model_root": "<hex>",
 "vram_cid": "bafkr4i...",
 "tensor_count": 291,
 "measured_at": "2026-08-27T20:04:52Z",
 "seconds": 0.13}
```

This endpoint is diagnostic and unsigned. It is never retained as mutable
"last measurement" state and cannot later be signed.

### `POST /v1/sign`

```json
{"model": "Qwen/Qwen2.5-0.5B-Instruct",
 "tensors": [{"handle": "<hex>", "nbytes": 16384, "seg_off": 0, "t_off": 0, "device": 0}]}
```

Measurement and signing happen atomically in one serialized request, and the
server captures `measuredAt`; clients cannot backdate it. One cooperative
kernel performs chunk hashing, all reduction levels, and the model-root fold.
A small finalization kernel queued behind it on the same stream atomically
consumes a private one-shot root and signs the receipt. Both complete before
the request's single host synchronization. This prevents one client's
measurement from being signed with another client's metadata.
Before returning, the host decodes the signed document and requires its
`model` field to equal the exact model string from this request, alongside the
host-measured root, count, device, signer and code identities.
It also reconstructs the state statement and its credential wrapper from those
trusted values, requiring closed schemas, a canonical statement CID and exactly
one GPU-issued credential over it — including that `instanceID` is the value
this service supplied, so a self-consistent substitution is still caught.
Unknown statement fields are never published as part of a successful receipt.

The `model` string and every IPC span are client-supplied. Bounds and device
validation make the memory access safe but do not prove that the spans belong
to that named model or were used for inference. A stronger claim requires a
trusted policy or runtime integration that selects the tensors independently
of the requesting workload.

The receipt wraps the exact JSON document the kernel signed. Decode
`measurementDocument` from hex rather than re-encoding its JSON: the detached
signature covers those exact bytes.

For `/v1/sign`, `seconds` covers IPC mapping, the fused measurement launch,
same-stream receipt finalization, result transfer, and IPC unmapping. For
`/v1/measure`, the same field covers the unsigned form of that path.

```json
{"modelRoot": "...",
 "model_root": "...",
 "vram_cid": "bafkr4i...",
 "tensor_count": 291,
 "digests": "<hex, 32 bytes per tensor>",
 "gpu_pubkey_uncompressed": "04...",
 "kernel_cid": "bafkr4i...",
 "cubin_cid": "bafkr4i...",
 "device": "cuda:0",
 "measurementDocument": "<hex — decode, don't re-serialise>",
 "measurementSignature": "<128 hex chars, r‖s>",
 "manifest": {"version": "3",
              "statements": {"urn:cid:...": {"@type": "CredentialRegistration", ...}}}}
```

### `GET /healthz`

`{"status": "ok"}`.

### Persistent IPC registration

Use the Python client below to manage producer ownership. Direct HTTP users
must pin the exact exported storage through **matching close acknowledgement**,
not just through a successful sign response. These POST endpoints share the
service's serialized request processing:

| Endpoint | Request | Response |
| --- | --- | --- |
| `/v1/registrations/open` | `{}` (no memory handles) | `{session, token, pid}` |
| `/v1/registrations/import` | `{session, token, tensors: [...]}` | `{session, token, registered: true, tensor_count}` |
| `/v1/registrations/sign` | `{session, token, sequence, model}` | `{session, token, sequence, receipt}` |
| `/v1/registrations/close` | `{session, token}` | `{session, token, released: true}` |

Tickets are generated by the service, scoped to its random session, and never
reused. Import consumes a pending ticket exactly once. Sequence numbers start
at 1 and increase exactly once per observation. Sign uses the same fresh
hashing, private one-shot signing handoff, signed facts and multi-GPU manifest
as `/v1/sign`; no digest is cached. Its server `seconds` excludes registration.

At most 16 tickets/registrations are retained per service. Unimported tickets
expire after 60 seconds. Aggregate registered tensor/byte/tile counts also
respect the operator's request limits. Close is idempotent within the exact
session, including after a lost close reply; a closed or expired ticket cannot
be resurrected by a delayed import. Active registrations do **not** expire:
retiring producer storage while a request may be running would be unsafe.

Response headers, a successful observation, and HTTP errors do not release
persistent producer leases. Only a complete close body identifying the exact
session/ticket does. A server restart rejects the old session; it is not an
automatic release acknowledgement from the old process.

## Python

```python
from cuattest import Notary, TensorRef

n = Notary(
    device=0,
    artifact_dir="./artifacts",
    max_request_tensors=16_384,
    max_request_bytes=128 * 1024**3,
    max_request_tiles=1_048_576,
)
n.info.gpu_did                       # did:key:z...
m = n.measure([TensorRef(handle=h, nbytes=n_bytes, device=0)])  # unsigned
receipt = n.sign([TensorRef(handle=h, nbytes=n_bytes, device=0)], "my-model")
n.close()
```

Each `Notary` operation pushes that instance's CUDA context and pops the same
entry before returning. Construction likewise pops the entry created by
`cuCtxCreate`, preserving the caller's complete context stack rather than only
restoring its top value. Multiple instances may therefore coexist on one
thread, although an individual instance remains non-thread-safe.

`hash_dptr(ptr, nbytes, *, producer_stream=None)` orders its private CUDA stream
after writes already submitted to the notary context's legacy default stream.
For a different producer stream, pass its raw CUDA stream handle (in the same
context) as `producer_stream`. The handoff uses a queued event rather than a
device-wide barrier. Join any other writers into that
producer stream before calling, keep the allocation alive, and do not modify
it until the hash returns. Internal fused calls require already-ordered input;
the IPC exporter and `hash_bytes()` establish readiness separately.

If CUDA accepts direct-span work but both synchronization attempts fail,
`hash_dptr()`/`hash_bytes()` destroy the notary context before returning
`GpuSessionAbortedError`. If destruction cannot be confirmed they instead
raise `GpuCleanupUncertainError`; caller-owned `hash_dptr()` memory must remain
allocated because the kernel may still reference it. `hash_bytes()`
automatically quarantines its internally allocated source.

`close()` still attempts stream, module, and context teardown if freeing a
staging buffer fails. The cleanup error is reported, but only confirmed context
destruction ends IPC quarantine; an error from a prior free/unload does not
negate successful destruction. Unvisited buffer wrappers are detached before
context destruction to prevent later double frees.

Client side, in the process that owns the model:

```python
from cuattest.ipc import share_model, share_tensors
from cuattest.client import Client

names, refs, keepalive = share_model(model)
client = Client("http://127.0.0.1:8077")
info = client.info()
receipt = client.sign(refs, "my-model")
keepalive.release()
```

Quiesce all writers before calling `share_model`/`share_tensors`; from that call
until acknowledged completion, the producer must not write the shared tensors
on any stream or through any raw CUDA API.
`Client` rejects a result if PyTorch's version counters detect an in-place
write, but inference tensors and raw device writes can bypass that tripwire;
the immutability requirement is the authority. After an ambiguous failure it
continues until completion, cancellation, or notary termination is confirmed.
Direct-HTTP users call `keepalive.assert_unchanged()` before accepting a result.
Non-contiguous layouts and lazy conjugate/negative views are materialized so
the measured bytes represent logical tensor values. Even a contiguous view
may need a copy if a lazy flag is set. All such copies are synchronized on
their owning GPUs before any export and retained alongside the original
tensor under the same mutation and acknowledgement rules.
Packed quantized dtypes `torch.quint4x2` and `torch.quint2x4` are rejected:
their logical element counts do not describe a byte-aligned physical span, and
the current IPC protocol deliberately has no bit-offset representation.

The client requires **legacy CUDA-IPC-compatible allocations**. PyTorch
`expandable_segments:True` uses VMM-backed storage, whose IPC format is not
supported by cuAttest's current 64-byte-handle protocol; `cudaMallocAsync`
allocations are also unsupported. Configure `backend:native,expandable_segments:False`
before importing torch and allocating the model. The exporter checks actual
allocation capability, not the current environment string, and raises a
diagnostic error for incompatible storage without publishing any refs.
See [allocator setup and recovery](operations.md#client-allocator-compatibility).

`Client.measure` and `Client.sign` release cuAttest's process-owned storage
leases after a successful or definitive server response. The lease registry
pins the original allocator storage even if the model object is dropped.
`keepalive.release()` is idempotent and remains useful when posting the refs
with another HTTP client.
References are one-shot: once a `Client` request claims a refs list, concurrent
or later reuse raises `IpcReferenceReuseError`. Call `share_model()` or
`share_tensors()` again to obtain a new lease and mutation guard.
The authoritative state is attached to the process-global allocation lease,
not just its mutable refs dictionaries. A pre-request copy or JSON round trip
therefore observes the original's in-flight state and becomes stale when that
monotonic lease ID is retired. `keepalive.release()` rejects an active Client
claim—even with `server_completed=True`; only the Client that received an
acknowledgement or proved the request unsent can perform that transition.

If a client call raises `NotaryRequestUncertainError`, the connection failed
before an acknowledgement and the server may still be reading the allocation.
The refs are marked as quarantined: retain `keepalive` and do not release or
unload the tensors. Only after independently confirming completion,
cancellation, or termination of that notary process may the producer retire
the leases with `keepalive.release(server_completed=True)`. The explicit
flag is intentionally required so an exception handler cannot accidentally
trade retained VRAM for use-after-free in live CUDA work.

Connection refusal, DNS failure, and invalid destinations prove that no
connection was established and release immediately. Route errors such as
`EHOSTUNREACH`/`ENETUNREACH`, timeouts, resets, and malformed HTTP status lines
before response headers have an unknown phase and use
`NotaryRequestUncertainError`. Once headers arrive, cleanup is acknowledged:
even a truncated or malformed response body releases the lease. If CUDA IPC
unmapping fails, the notary destroys its context before sending a fatal
response; if destruction itself cannot be confirmed, it sends no headers and
stops the server.

`Client` deliberately ignores `http_proxy`/`https_proxy` and related
environment settings. CUDA IPC is a same-host protocol, and a proxy-generated
HTTP error cannot acknowledge that the actual notary closed its GPU mapping.

Verify a receipt before treating its convenience fields as evidence:

```python
from cuattest.expect import verify_evidence

verified = verify_evidence(receipt, trusted_pubkey=info["gpu_pubkey_uncompressed"])
print(verified.vram_cid, verified.tensor_count)
```

Verification extracts the root/count from the signed document, validates the
P-256 signature and signer DID, cryptographically verifies the credential JWS
proof, checks code/device metadata, and re-folds any supplied per-tensor
digests. Since a kernel signing receipt always contains the manifest, its
omission is rejected rather than treated as optional evidence.

Statements issued by anyone other than the GPU — the service's own
`IdentityAttestation` — are returned in `unverified_statements` rather than
counted as evidence. Authenticating them needs the issuer's key pinned
independently of the GPU's.

`cuattest.expect.compare` likewise requires `trusted_pubkey`; it never accepts
the receipt's self-supplied key as its trust anchor. Malformed documents,
contradictory fields, and invalid signatures raise `EvidenceError`. A returned
`Comparison(matches=False, ...)` therefore means valid authenticated evidence
was compared and its model content differed.

Encoders, if you need to check ids yourself:

```python
from cuattest.ids import raw_cid, did_key_p256
raw_cid(digest32)          # bafkr4i...  CIDv1 raw/blake3-256
did_key_p256(x32, y32)     # did:key:z...
```

### Repeated observations

```python
from cuattest.client import Client
from cuattest.expect import verify_evidence

client = Client()
# Example: a single-GPU service/model on producer cuda:0. Pin the session
# public key independently, as with one-shot signing.
with client.register_model(model) as registered:
    receipt = registered.sign("my-model")
    verified = verify_evidence(receipt, trusted_pubkey=trusted_pubkey)
    # Every later sign hashes the current bytes again, including unchanged ones.
    receipt = registered.sign("my-model", streams={0: [writer_stream]})
```

`register_tensors(named_mapping, streams=...)` is the equivalent mapping API.
For a multi-GPU service, verification uses `trusted_pubkeys=trusted_keys`, the
independently pinned UUID-to-public-key mapping, just as with one-shot evidence.
`registered.names` gives the sorted tensor names and `attested_bytes` the
logical byte count (including aliases). Registered CUDA tensors must already
be contiguous with resolved negative/conjugate view flags. Materialize such
views **in the model** first, or use one-shot sharing; silently retaining an
export-time copy would become stale after later model updates.

Storage identity, shape, dtype, offsets, names and device placement must remain
fixed until close. They are checked before and after every observation;
replacement/resizing requires closing and re-registering. Values may change
**between completed observations**. PyTorch version counters also detect
in-place writes during a call. Neither check replaces the caller's obligation
to exclude concurrent writers: inference tensors and raw CUDA writes may not
increment a version counter.

Before import/sign, the client drains only the declared producer CUDA streams
on the host, not the whole device. By default this means the current stream
on each participating GPU. `streams={producer_ordinal: stream_or_list}` must
name exactly those GPUs. Join every other writer into these streams, or pass
all writers explicitly; keep bytes immutable through completion. Consumer
hashing, receipt finalization and result transfers stay on the notary's private
nonblocking streams, with independent GPUs dispatched concurrently.

An ambiguous import/sign/close raises `RegistrationUncertainError`; its
`.registration` handle stays pinned and cannot sign again. Retry its `close()`
to obtain the matching acknowledgement. `client.registrations()` retrieves
live/quarantined handles even if construction was interrupted or the last
user reference was lost. There is deliberately no GC-triggered unmap/free.

If the old service can no longer acknowledge, independently confirm that its
contexts/process have terminated before calling
`registered.close(server_completed=True)`. A timeout, connection refusal,
different service epoch, or PID number alone is not that proof. All models
remain pinned until acknowledged retirement or that explicit recovery assertion.

## CLI

```
cuattest info                       identity and registered code
cuattest serve [--host --port] [--max-request-tensors N]
                [--max-request-bytes N] [--max-request-tiles N]
                                      bounded HTTP front end
cuattest build-kernel [--arch ...]  compile CUBINs (no GPU needed)
cuattest measure FILE [--sign M]    measure tensors from a JSON file
cuattest expect PATH [--compare F --trusted-pubkey HEX]
                                      expected CID from an on-disk checkpoint
cuattest selftest                   exercise the whole path
```

## Environment

`CUATTEST_HASH_MODE=auto` (default) selects double-buffered asynchronous hashing
on qualified `sm_120` devices for requests with at least 1 GiB of logical spans
**per GPU**. Small requests and other architectures use the original kernel.
`standard` forces that kernel; `async` forces the new path on `sm_80+` for
qualification and is rejected on older devices. These modes use separate
cooperative occupancy limits, identical BLAKE3 tails/ROOT flags and the same
signed evidence format. The additional entry point changes the source/CUBIN
CIDs; update code allowlists after rebuilding.

| variable | meaning |
|---|---|
| `CUATTEST_KERNEL_DIR` | searched first for a prebuilt CUBIN |
| `CUATTEST_CACHE` | cache root (default `~/.cache/cuattest`) |
| `CUATTEST_KERNEL_SRC` | override the kernel source path |
| `CUATTEST_DISABLE_NATIVE_HOST` | use the Python host fallback when set to `1` (diagnostics/A-B benchmarking) |

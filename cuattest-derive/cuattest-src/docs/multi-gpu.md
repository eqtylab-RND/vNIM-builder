# Multi-GPU models and services

Run one HTTP service for all visible GPUs, or choose a subset:

```bash
cuattest serve --devices all
cuattest serve --devices 0,2,3 --port 8078
```

Each selected GPU owns an independent notary context and session key, and
loads a CUBIN for its own architecture. Existing `cuattest --device 1 serve`
and `Notary(device=1)` keep the single-GPU API and receipt format. The library
equivalent of the multi-GPU service is `MultiGpuNotary(devices=None)`; call
`close()` when finished, just as with `Notary`.

The existing producer calls work for an ordinary model or a model whose state
dictionary spans several GPUs:

```python
import json
from pathlib import Path
from cuattest.client import Client
from cuattest.ipc import share_model
from cuattest.expect import verify_evidence

client = Client("http://127.0.0.1:8077")
trusted_keys = json.loads(Path("trusted-gpu-keys.json").read_text())
names, refs, keepalive = share_model(model)
receipt = client.sign(refs, "my-model")
del keepalive  # the acknowledged Client call has already retired the leases
verified = verify_evidence(receipt, trusted_pubkeys=trusted_keys)
print(verified.vram_cid)
```

Quiesce writers on **all** source GPUs before sharing and keep their tensors
immutable throughout the complete request. Shards execute concurrently in
independent per-GPU workers, but this does not create a simultaneous snapshot
of an actively changing model.
Transport uncertainty quarantines the entire request's producer leases, as in
the single-GPU API. A later shard failure never permits acknowledgement while
another shard still has uncertain IPC ownership.

## Device identity and placement

`share_tensors()` and `share_model()` add `device_uuid` to every tensor reference.
Routing uses the CUDA driver's UUID, not the producer's local `device` ordinal.
The producer and service may have different `CUDA_VISIBLE_DEVICES` orderings or
subsets, provided every submitted GPU is visible and selected in the service.
An unknown UUID or a missing UUID is rejected before any handle is imported.
Older single-GPU wire references without UUIDs remain supported by `Notary`.

A client-supplied UUID is only a routing hint. Each notary still asks CUDA for
the imported allocation's real owner and bounds before hashing. Each span is
measured on its owning GPU, so this works without NVLink or peer-memory access
between GPUs and across different GPU architectures. No tensor payload is
copied through host memory by the notary; aggregation uses the 32-byte digests.
CUDA IPC still requires producer and notary to run on the same host and share
the appropriate IPC namespace. This is not remote-memory measurement over SSH.

One service can also handle independent producers on any selected GPU. A
multi-GPU service always emits the aggregate format below, even for a request
that happens to use only one GPU. Requests queue; within a request, the selected
GPUs run concurrently. Each session has one worker, so its CUDA context and
native workspaces are never used by overlapping calls. Request byte/tile/count
limits apply to the whole request before dispatch, not separately to each
shard. Aggregate receipts support up to 16,384 spans and 256 GPUs; operators
may lower these limits.

All dispatched shards must finish before any success or error response can
acknowledge producer storage. A fast failure on one GPU cannot release leases
while another GPU is still reading. Interrupted dispatch drains even work
whose submission did not return normally; if completion cannot be confirmed,
the service stops without acknowledgement and preserves quarantine state.
`close()` waits for worker-confirmed termination before destroying any CUDA
context. Library callers must still serialize requests and `close()`.

The native backend briefly serializes each shard's IPC import/validation batch
to reduce contention in driver bookkeeping. This gate is released before any
kernel launch or synchronization; GPU hashing and signing still overlap.
Mappings are not cached across requests, and every imported span retains its
owner and allocation-bounds checks.

## Keys and verification

For this service, `GET /v1/info` returns:

```json
{"multi_gpu": true, "devices": [
  {"device_uuid": "GPU-...", "device_ordinal": 0,
   "gpu_pubkey_uncompressed": "04...", "gpu_did": "did:key:...", "arch": "sm_120"}
]}
```

Each entry also contains the normal single-GPU identity/code fields. Through a
trusted setup channel, capture those UUID/key pairs for the **running session**
as `trusted-gpu-keys.json`, a JSON object mapping GPU UUID strings to full
uncompressed P-256 public-key hex strings. A separate `cuattest info` invocation
starts new sessions and cannot supply the running service's keys. Never derive
the trusted key map from an untrusted receipt. A restart creates new keys.

Use `verify_evidence(receipt, trusted_pubkeys=keys)` or
`compare(expected, receipt, trusted_pubkeys=keys)`. The single-key argument
`trusted_pubkey` remains for single-GPU receipts. Offline comparison supports:

```bash
cuattest expect ./checkpoint --compare receipt.json \
  --trusted-pubkeys trusted-gpu-keys.json
```

The global root preserves the original submitted span order (sorted state-dict
names for `share_model`), regardless of placement. Moving complete unchanged
tensors between GPUs therefore does not change the expected CID. Tensor
parallelism that splits individual tensors into new named slices still changes
the span count/bytes: cuAttest does not reconstruct framework-specific shards.

## Aggregate receipt contract

`receipt_type` is `cuattest.multi-gpu.v1`. `manifest` contains the original
`model`, a fresh 128-bit `nonce`, `tensor_count`, and `shards`. Each shard records
its `device_uuid`, the service's `device_ordinal`, and its strictly increasing
global `positions`. `receipts` contains one ordinary kernel-signed receipt per
shard, in the same order.

Each shard's signed `model` field is the 64-character hexadecimal BLAKE3 of:

```
UTF8("cuattest.multi-gpu.v1") || 0x00 || canonical_manifest_JSON || LE32(shard_index)
```

Canonical manifest JSON uses sorted object keys, no whitespace, and the
manifest's array order (`json.dumps(..., sort_keys=True, separators=(",", ":"))`).
This commitment fits the existing kernel's model field and binds every shard
to the same model, layout, and request. Verification rejects changed positions,
missing/duplicate spans, swapped GPUs, mixed requests, and untrusted keys. It
authenticates every measurement and credential proof before interleaving the
digest lists and recomputing `BLAKE3(LE32(N) || spanDigests)`.

The outer `model_root`, `vram_cid`, `tensor_count`, and `digests` must equal that
verified reconstruction; `seconds` is diagnostic host timing. This envelope is
an authenticated aggregation of per-GPU measurements, not an assertion that one
GPU assembled or signed the global root. `VerifiedMeasurement.document` is the
authenticated manifest for this format. All existing submitted-span trust
limitations continue to apply.

## Integration tests

```bash
CUATTEST_TEST_MULTIGPU=1 pytest -q tests/test_multigpu_integration.py
```

These opt-in tests use every visible GPU, reverse the service's visibility
order, and exercise real PyTorch exports, HTTP, independently verified global
folds, per-GPU requests, forged routing hints, fresh partial-leaf/tile bytes
across repeated parallel requests, and recovery after one GPU rejects an
import while the others receive valid spans. Both host backends are covered.
They require two or more GPUs, the native extension, PyTorch, and the `verify`
extra. CPU fault tests additionally force overlap, reverse completion order,
and interrupt worker startup, dispatch, and draining to guard against an early
HTTP acknowledgement or premature context destruction. The
[sanitizer runner](sanitizers.md#reproduce-cuda-checks) also has a `multigpu`
suite for instrumenting the successful cross-process paths.

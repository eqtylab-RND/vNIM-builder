# Operations

## Running against a container

The notary listens on loopback, and CUDA IPC handles only resolve inside a
shared IPC namespace. A containerised client needs **both**:

```bash
docker run --gpus all --network host --ipc host ...
```

`--network host` alone gets you a client that connects and then fails every
measurement, because the handles do not resolve. They are separate
requirements for separate reasons.

To avoid host networking, bind the notary somewhere the container can route to
(`cuattest serve --host 172.17.0.1`) and point the client at it. You still
need `--ipc host`.

## Placement

One notary session per GPU. `cuattest serve --devices all` manages these sessions
behind one HTTP endpoint and supports models distributed across the selected
GPUs; `--devices 0,2` selects a subset. See [multi-GPU operation](multi-gpu.md)
for UUID routing and aggregate verification. Each session holds a CUDA context
and a little VRAM for the module and constants. The native host backend grows
and reuses two global ping-pong
scratch buffers totalling approximately 48 bytes per 128 KiB scheduling tile
for large spans: 32 primary bytes per tile plus a half-sized peer. The
first seven BLAKE3 tree levels use about 8.2 KiB of block-local shared memory,
which is reused for every tile and is not a model-sized allocation. A 96 GiB
single tensor therefore needs 36 MiB of persistent CV scratch instead of
4.5 GiB. The buffers are released when the notary closes; retaining them avoids
repeated CUDA allocation/free synchronization on the request path.

Measurement competes with whatever else is using the GPU. On an idle device a
65 GB model hashes in well under a second; under active inference the same work
takes several times longer, because the hashing kernels contend for the same
compute units.

Every request is bounded before any CUDA IPC handle is imported. Defaults are
16,384 submitted spans, 128 GiB of aggregate logical input (duplicates count
again), and 1,048,576 scheduling tiles. The independent tile ceiling bounds
retained scratch even for many tiny spans. Operators can lower or deliberately
raise these limits with `cuattest serve --max-request-tensors`,
`--max-request-bytes`, and `--max-request-tiles`; clients cannot override them.

## Troubleshooting

**`cuInit: system not yet initialized (rc=802)`** — an H100/H200 in
confidential-compute mode that has not been unlocked. `nvidia-smi conf-compute
-grs` will say `not ready`; set it with `nvidia-smi conf-compute -srs 1`. The
state is volatile and resets on every driver load.

**`cannot load libcuda.so.1`** — no driver, or a container without the device
passed through. Add `--gpus all`.

**Comparing the native and Python host paths** — the compiled C++ backend is
the Linux default and appears as `"host_backend": "C++"` in `/v1/info`. Start
the server with `CUATTEST_DISABLE_NATIVE_HOST=1` only to diagnose or benchmark
the portable Python fallback.

**`device does not support cooperative kernel launches`** — the measurement
pipeline requires a grid-wide barrier. Use a device/driver configuration that
reports CUDA cooperative-launch support.

**`cannot load libnvrtc`** — only needed to *compile* the kernel. Either ship a
prebuilt CUBIN and set `CUATTEST_KERNEL_DIR`, or `pip install '.[build]'`.

### Client allocator compatibility

**`CUDA allocation does not support legacy CUDA IPC`** — the client found
storage that cannot be represented by cuAttest's legacy 64-byte CUDA memory
handles. This includes PyTorch `expandable_segments:True` (VMM-backed storage)
and `backend:cudaMallocAsync`. PyTorch's own VMM IPC support uses a different
format and does not make those allocations compatible with this protocol.
The exporter queries CUDA's [allocation capability attribute](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html),
so unrelated CUDA errors are not mislabeled as allocator incompatibility.

Restart the **model-owning client process**, with these settings in place
before importing torch or allocating the model:

```bash
PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:False \
PYTORCH_ALLOC_CONF=backend:native,expandable_segments:False \
python your_model_process.py
```

Both names are set consistently because newer PyTorch versions also accept
`PYTORCH_ALLOC_CONF`. Preserve any other compatible allocator options your
application needs. These are client-side settings; restarting only the notary
does not change the model's storage. Changing allocator settings, calling
`contiguous()` on an already-contiguous tensor, or emptying the cache does not
convert existing live allocations. Reallocate/reload the model after restart.
cuAttest does not change process-wide allocator settings or silently copy an
incompatible allocation into another pool.

### Other IPC and client errors

**`cuIpcOpenMemHandle: invalid argument`** — the handle is malformed or came
from a different machine, or the processes do not share an IPC namespace.

**`NotaryRequestUncertainError` after a timeout or disconnect** — the client
cannot know whether the notary opened the CUDA handles before the connection
failed. It therefore leaves the process-owned storage leases quarantined. Do
not unload the tensors or call ordinary `keepalive.release()` while the old
request may still run. After independently confirming completion/cancellation,
or after the notary process has exited, call
`keepalive.release(server_completed=True)`. This conservative failure mode may
retain VRAM, but prevents allocator reuse under an in-flight hash.

A refused connection or DNS failure is known to be unsent and does not enter
quarantine. Route errors such as `EHOSTUNREACH` and `ENETUNREACH` can arrive
after the POST was transmitted, so they remain quarantined. A response whose
headers arrived but whose body was truncated is safe to release: this protocol
emits headers only after IPC cleanup. If the notary cannot close an imported
mapping, it destroys the entire CUDA context before acknowledging; if context
destruction also fails, it withholds the response and terminates the server
session.

The client uses an explicit proxy-disabled HTTP opener. Do not insert an HTTP
proxy or gateway between producer and notary: only response headers received
directly from the same-host notary are a valid IPC-cleanup acknowledgement.

**CUDA IPC references are one-shot** — after a client request starts, obtain
fresh refs with `share_model()`/`share_tensors()` before measuring again. A
consumed or concurrently active list is rejected before any network I/O. This
state is stored with the process-owned allocation lease, so copying or JSON
round-tripping a refs list does not create a fresh lease. Do not call
`keepalive.release()` while `Client` is blocked: active claims are rejected and
remain pinned until that Client observes a definitive completion or quarantines
an ambiguous failure.

**producer tensor changed during CUDA IPC measurement** — tensors must remain
immutable from `share_model`/`share_tensors` until the response is acknowledged.
The client detected a PyTorch version-counter change and discarded the result.
Also serialize hot swaps, optimizer steps, and raw/other-stream CUDA writes;
those need not be visible to the version-counter tripwire.

Producer and notary CUDA ordinals may differ under `CUDA_VISIBLE_DEVICES`.
That alone is valid: the notary accepts the reference when the CUDA driver says
the imported allocation belongs to its selected device. A driver-reported
different owner is still rejected before hashing.

**`"tensors" must be a list` from `/v1/sign`** — signing is an atomic
measure-and-sign operation. Send the tensor references and model in the same
request; there is deliberately no signable process-global "last measurement".

**request body was not received within 30 seconds** — the client advertised a
body length but stalled. The single-threaded service closes such requests so
health and measurement traffic cannot be blocked indefinitely.

Each request line and complete header block has one 10-second monotonic
wall-clock deadline. The server recomputes the remaining budget after every
read, so sending a byte just before each socket inactivity timeout cannot keep
the single-threaded service occupied indefinitely.

## Repeated observations

Persistent registrations are opt-in via `Client.register_model` or
`register_tensors`. Close every registration when finished; dropping its Python
variable does not release storage. Following uncertain completion, recover the
handle through `error.registration` or `client.registrations()` and retry
`close()`. If the old server is gone, use `close(server_completed=True)` only
after independently confirming termination of its old CUDA contexts/process.
Ordinary sign/error headers never retire a persistent lease. See the
[complete registration contract](api.md#repeated-observations).

The automatic hashing mode retains the original path on small requests and
unqualified architectures. Set `CUATTEST_HASH_MODE=standard` on the service
for an A/B comparison; do not force `async` on older hardware or assume that a
small cached-buffer result predicts a near-capacity model's performance.

## Upgrading the kernel

Changing the `.cu` changes `kernel_cid`; recompiling changes `cubin_cid`. Both
appear in every signed document, so evidence produced by different builds is
distinguishable — which is the point. Rebuild CUBINs when you change the
kernel. Cache sidecars bind every architecture-named CUBIN to both the source
and CUBIN digests; stale or partially updated pairs are rejected automatically.

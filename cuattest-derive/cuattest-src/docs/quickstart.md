# Quickstart

## Install

```bash
pip install .                 # core + independent host-side BLAKE3
pip install '.[client]'       # + torch, if this process will share tensors
```

A Linux source install needs a C++17 compiler for the native host extension;
other platforms retain the Python fallback for offline tooling. `libcuda` and
`libnvrtc` are opened only at runtime, so no GPU or CUDA toolkit is needed
during installation.

## Build the kernel

The CUBIN is what runs, so it is what gets hashed and registered.

```bash
cuattest build-kernel               # for this GPU
cuattest build-kernel --arch sm_90 --arch sm_100 --out ./cubins
```

Compiling needs NVRTC but no GPU, so you can build on one machine and ship the
CUBINs to another. At startup the notary searches `$CUATTEST_KERNEL_DIR`,
then the package's `cubins/`, then `~/.cache/cuattest/cubins`, and compiles
on demand if it finds none.

## Check it works

```bash
cuattest selftest
```

Exercises keygen, in-GPU BLAKE3 against host reference digests (including
alignment, semantic-chunk, and 128 KiB scheduling-tile boundaries),
multi-tensor tree reduction, the model-root fold, the private one-shot handoff,
and receipt signing.

## Run it

```bash
cuattest serve                 # 127.0.0.1:8077
curl -s localhost:8077/v1/info
```

## Measure a model

From the process that owns the model — a different process from the notary:

```python
from cuattest.client import Client
from cuattest.ipc import share_model

names, refs, keepalive = share_model(model)   # zero copy
receipt = Client().sign(refs, "my-model")     # measure + sign atomically
del keepalive                                  # only after request returns
print(receipt["vram_cid"])
```

Stop all writers before sharing, then hold `keepalive` and do not mutate any
shared tensor until the request returns.
Freeing a tensor early lets the caching allocator recycle its segment; writing
one can produce a torn state while the notary hashes the request. The
client rejects detected PyTorch version-counter changes and then releases the
process-owned storage leases, so repeated model unloads can reclaim their VRAM.
On a timeout or disconnect it instead raises `NotaryRequestUncertainError` and
quarantines those leases, because the notary may still be hashing. Keep the
tensors alive and immutable; use
`keepalive.release(server_completed=True)` only after independently confirming
that the request finished, was cancelled, or the notary exited.
The returned refs are one-shot. For another measurement, call `share_model()`
again so the new request has a live storage lease and mutation guard. Copies
retain the same registry identity, and `keepalive.release()` cannot retire it
while a `Client` request is active.

Full example: [`examples/measure_torch_model.py`](../examples/measure_torch_model.py).

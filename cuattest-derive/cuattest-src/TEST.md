# Testing cuAttest

How to take a bare GPU box and convince yourself, step by step, that this stack
does what it claims: that a P-256 key was born inside the GPU, that another
process's live VRAM was hashed in place, and that the signature over the result
holds up when you check it with something other than this code.

Each section says what to run, what a good result looks like, and what it
proves. Work down the list — the later checks assume the earlier ones passed.
The whole run takes about ten minutes, most of it compiling the kernel once.

For the slower instrumented runs, see the [sanitizer audit and regression
guide](docs/sanitizers.md): all four CUDA tools, host ASan/UBSan/TSan/MSan,
leak checks, CFI, fuzzing, and documented tool/dependency limitations.

Reference numbers throughout come from an NVIDIA H100 NVL (95 GB, sm_90),
driver 610.43.02, CUDA 13.3, Python 3.12.

## 0. What the machine needs

An NVIDIA GPU and driver, Python 3.10+, and (on Linux) a C++17 compiler for the
native host extension. `libcuda` is opened at runtime; the only core Python
dependency is an independent host BLAKE3 implementation used to identify the
CUBIN before it is loaded.

To *compile* the kernel you also need NVRTC and the CUDA runtime/CCCL headers
used by cooperative groups. They come with the CUDA toolkit; if the box has no
toolkit, `pip install '.[build]'` fetches them as NVIDIA wheels. Compiling
needs no GPU, so a build machine works too.

A stock Ubuntu cloud image often ships Python without `pip` or `venv`. If
`python3 -m venv` complains, install them first:

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip
```

Confirm the GPU is visible before going further — most confusing failures later
are really this failing quietly:

```bash
nvidia-smi -L
```

## 1. Install

```bash
python3 -m venv .venv
.venv/bin/pip install '.[verify]' pytest
```

The `verify` extra pulls in `cryptography`, which section 5 uses to check a
signature independently. `pytest` is only for the unit tests.

## 2. Unit tests — no GPU required

For the required Release/AssertedRelease matrix, run
`python tests/build_matrix.py /tmp/new-cuattest-matrix`. See
[build configurations](docs/build-configurations.md#test-both-configurations)
for installation, debug assertions, isolated caches, and GPU matrix options.

```bash
.venv/bin/python -m pytest tests -q
```

**Good:** all CPU tests pass; explicitly opt-in GPU tests are skipped.

These cover the host-side encoders (CIDs, `did:key`, safetensors parsing,
tensor refs), signed-evidence verification, atomic server protocol, total
slow-trickle-resistant header/body deadlines, empty-tensor canonicalization,
stack-preserving multi-context activation, matching Python/C++ fused reduction
plans, native-host routing, fused launch count, CUBIN cache invalidation,
NVRTC/header wheel discovery, immutable/transport-state IPC
lease handling, request work ceilings, interrupt-safe fallback IPC cleanup,
strict bounded evidence parsing, canonical credential signatures, per-CUBIN
compiler provenance, initialized output copies, constant-work scalar structure,
fatal unmap cleanup, trust-anchor enforcement, and invalid-evidence cleanup.
Client-export regressions cover lazy conjugate/negative resolution, retention
and mutation guards for every copy, per-GPU synchronization before export, and
fake-driver allocation-capability checks without masking unrelated CUDA errors.
They run without a GPU, including in CI. The native-planner checks run on
Linux; other platforms exercise the portable Python/offline path instead.

## 3. Build the kernel

```bash
.venv/bin/cuattest build-kernel
```

**Good:** one line naming the architecture, a CUBIN of a few megabytes, and a
path under `~/.cache/cuattest/cubins`. On the H100 this took **87 seconds** and
produced a 3.85 MB `sm_90` CUBIN. It is slow because the kernel carries its own
BLAKE3, SHA-256, P-256 field arithmetic and JSON writer, and NVRTC inlines
aggressively.

Worth doing once, deliberately: build it twice into different caches and
compare.

```bash
CUATTEST_CACHE=/tmp/c1 .venv/bin/cuattest build-kernel
CUATTEST_CACHE=/tmp/c2 .venv/bin/cuattest build-kernel
sha256sum /tmp/c1/cubins/*.cubin /tmp/c2/cubins/*.cubin
```

**Good:** identical hashes.

**Proves:** the build is reproducible. This matters more than it looks. Every
signed document carries `cubin_cid`, and that number is only meaningful as
evidence if a third party can compile the same `.cu` and arrive at the same
bytes. If these two hashes ever diverge, the CUBIN CID stops being a claim
anyone can check.

## 4. Selftest

```bash
.venv/bin/cuattest selftest
```

**Good:** every check says `ok` and the run ends with `ALL CHECKS PASS` (the
credential-proof check says `skip` only when the optional `verify` extra is not
installed).

This walks the whole path in one process: generate a key on the device, hash
the empty input and compare against the published BLAKE3 digest
(`af1349b9…3262`), hash a multi-chunk input to exercise the tree reduction,
hash a device range and check it equals the same bytes uploaded from the host,
then compare multiple differently sized tensor roots and their model fold with
the independent host BLAKE3 before parsing the receipt signed by that same
cooperative kernel. With the `verify` extra installed, it also reconstructs
and verifies all three kernel-produced credential JWS signatures.

The empty-input vector is the one to watch. It is the cheapest possible proof
that the in-GPU BLAKE3 is really BLAKE3 and not something that merely hashes
consistently.

Then look at the session's identity:

```bash
.venv/bin/cuattest info
```

Run it twice. **Good:** `gpu_did` differs every time; `kernel_cid` and
`cubin_cid` do not.

**Proves:** the identity is per-process — the key is generated at startup and
dies with the process — while the code being run is stable and named.

## 4a. Independent CPU/GPU cryptographic comparisons

The selftest is a quick end-to-end check. The dedicated differential suite
adds thousands of reproducible vectors and compares exact device outputs with
independent CPU implementations, including the intermediate primitives:

```bash
.venv/bin/pip install '.[test-crypto]'
# Also install '.[build]' if NVRTC and CUDA/CCCL headers are not available.
CUATTEST_TEST_CRYPTO=1 .venv/bin/python -m pytest -q tests/test_crypto_differential.py

# Repeat on every visible GPU, including both native and Python host paths.
CUATTEST_TEST_CRYPTO=1 CUATTEST_CRYPTO_DEVICES=all \
  .venv/bin/python -m pytest -q tests/test_crypto_differential.py
```

`CUATTEST_CRYPTO_DEVICES` defaults to `0` and also accepts a comma-separated
list of visible ordinals, such as `0,3`. No CUDA access or optional oracle
imports occur when `CUATTEST_TEST_CRYPTO` is unset. Once enabled, missing
CUDA, NVRTC, reference packages, or the native extension is a failure rather
than a silently skipped correctness check. No PyTorch or model download is
needed. The first run compiles a test CUBIN once per GPU architecture within
that pytest process; allow a few minutes for compilation.

| Device code | Independent CPU reference | Comparisons |
| --- | --- | --- |
| BLAKE3 streaming and production tiled hashing | `blake3` package | Every tensor digest, concatenated-input digest, and final `BLAKE3(LE32(N) \|\| ordered digests)` |
| SHA-256 | standard-library `hashlib.sha256` | One-shot and fragmented/zero-length updates |
| HMAC-SHA-256 | standard-library `hmac` | Exact MAC bytes, including output aliasing the key or message used by RFC 6979 |
| P-256 field/order arithmetic | Python integers, `%`, and `pow(a, -1, modulus)` | Addition, subtraction, multiplication, nonzero inverses, sparse reduction of arbitrary 512-bit inputs, and Montgomery products with aliased outputs |
| RFC 6979 and P-256 ECDSA | `ecdsa` nonce generation/deterministic signing; `cryptography` public-key derivation and verification | Exact nonce and `r \|\| s` bytes, public-point coordinates, valid signatures, and rejection after a message-bit flip; both independent lanes and cooperative eight-lane groups |
| Session key generation | `hashlib.sha256`, Python integers, and `cryptography` | Public points for 16 known 32-byte entropy inputs |

The principal corpus contains **4,096 distinct 1-KiB blocks** (4 MiB): zero,
all-one, patterned, and seeded pseudorandom data. It is allocated in host RAM,
copied into GPU VRAM, and read back to assert identical bytes before any hash
comparison. Tests check the blocks separately and as one contiguous BLAKE3
tree; these are different operations, so agreement on one is insufficient.
The production measurement/signing kernels also process all 4,096 spans in
one request, checking every digest and the independent final model root.

Additional cases cover all 16 address alignments, SHA-256 padding boundaries,
1-KiB BLAKE3 chunk boundaries, 128-KiB scheduling-tile boundaries, odd tree
widths, and mixed-size tensors. A one-bit mutation must change exactly its
tensor digest and the final root. Arithmetic vectors force cross-limb
carries/borrows and values near both moduli. The 201 nonce/signature vectors
include hashes at or above the group order and keys across six-bit window
boundaries, which ordinary random inputs rarely cover. Neighboring cooperative
groups use different inputs, and the final warp is only partially occupied.
Sparse-reduction tests exhaust all 65,536 zero/max assignments of sixteen
32-bit words plus 2,048 random 512-bit inputs, including output/input aliasing.
Montgomery products are checked against `a*b*pow(2**256, -1, modulus)` with
disjoint output and aliases of either operand.

The ordinary CPU suite also symbolically executes the production inversion
schedules to prove the exact `p-2`/`n-2` exponents, derives the sparse reducer's
signed-intermediate/carry bounds for every 512-bit input, and checks all table
scan partitions. These proofs complement rather than replace live GPU tests.

Published [RFC 6979 Appendix A.2.5](https://www.rfc-editor.org/rfc/rfc6979.txt)
and [RFC 4231 section 4.2](https://www.rfc-editor.org/rfc/rfc4231.txt) vectors,
plus fixed SHA-256/BLAKE3 digests, anchor the library comparisons. The CPU
oracles do not call cuAttest's hashing or signing implementations. Every
production receipt also passes the ordinary trusted-key verifier.

The adapters in `tests/_crypto_oracle.cu` are appended to the **unchanged
production source** in memory, call its actual device functions, and are
compiled into a separate test-only module. They are not shipped in the
package, written into the notary's CUBIN cache, or exposed by its service.
Any scalars/nonces exported by those adapters belong to public test vectors,
not a notary session. The production-path checks still use the normal notary
CUBIN and its original entry points. Cleanup synchronizes before releasing
input allocations, including failed tests.

**Proves:** agreement with independent implementations for these concrete
vectors, boundary cases, and end-to-end folds on the tested hardware/compiler.
It is regression evidence, not an exhaustive mathematical proof, an entropy
quality assessment, or a GPU side-channel audit.

Validated on 2026-09-07: all 24 differential cases passed on an RTX 2080
(`sm_75`, NVRTC 12.9), and all 192 device/backend cases passed across the eight
RTX PRO 6000 Blackwell GPUs on `probqa.com` (`sm_120`, NVRTC 13.3). Both runs
covered the native C++ and Python fallback host paths; no cryptographic
mismatches were found.

## 4b. Client export correctness on real GPUs

```bash
.venv/bin/pip install '.[client,verify]'
CUATTEST_TEST_GPU=1 .venv/bin/python -m pytest -q tests/test_ipc_export_integration.py
```

The 22 cases use real PyTorch storage and a separate-process HTTP notary with
both host backends. Lazy conjugate, negative, and combined views are compared
against materialized tensors and independent Python `complex`/`struct`/`blake3`
references. Contiguous, transposed, and nonzero-offset views run on non-default
streams on **every visible GPU**; the notary reverses device visibility to
exercise UUID-based routing on multi-GPU hosts.

Four fresh-producer cases check 4-MiB tensors with expandable segments enabled
and disabled, using each PyTorch allocator environment-variable name. They
also change the environment after allocation: rejection/acceptance must depend
on actual storage, not the current setting. Rejected exports leave no leases
and do not alter tensors. These cases require CUDA support for expandable
segments; see [client allocator compatibility](docs/operations.md#client-allocator-compatibility)
for the production limitation and setup instructions.

Validated on 2026-09-07: all 22 cases passed on the local RTX 2080 and on
`probqa.com` with eight RTX PRO 6000 Blackwell GPUs. The remote run also passed
the existing multi-GPU integration suite with reversed device visibility.

## 4c. Stream isolation and transfer lifetime

```bash
.venv/bin/python -m pytest -q tests/test_streams.py tests/test_notary.py tests/test_native_ipc_cleanup.py
CUATTEST_TEST_GPU=1 .venv/bin/python -m pytest -q tests/test_stream_integration.py
```

Four live isolation cases exercise both backends with either the legacy default
stream or an unrelated stream held on a host-released memory wait. After
warming the workspaces, signed requests must finish and match independent
CPU hashes while the blocked stream still reports `CUDA_ERROR_NOT_READY`.
This checks real isolation, not an elapsed-time threshold. A watchdog releases
the gate if a regression introduces a context-wide wait, so failure cannot
strand the GPU. No spin kernel competes for cooperative-grid residency.

Four direct-pointer cases gate writes from the legacy default stream
(`cuMemsetD8`) or an explicitly supplied producer stream. Before releasing the
writer they assert that the notary stream already has a pending dependency;
then they compare the hash with Python BLAKE3 while an unrelated stream stays
blocked. This catches stale hashes and host/device-wide waits deterministically.
A live fallback cleanup test also injects a pinned free error and verifies that
real context destruction still succeeds and restores the caller's context.

CPU fault injection covers upload, measurement, signing, download, stream
drain, key-generation scrubbing, failed workspace growth, and stream/context
teardown, producer-event errors/interruptions, and pinned-free failures followed
by stream/module/context errors. HTTP tests check acknowledgement safety after
both confirmed and failed destruction. Pinned DMA and IPC storage must not be
freed after unconfirmed work; only successful completion or context destruction
permits retirement. The
native sanitizer driver double defers copies until completion to expose
premature buffer frees under ASan/MSan. See [stream performance results](docs/performance.md#private-cuda-streams-2026-09-08).
Run this isolation assertion without CUDA instrumentation: memcheck itself
introduced cross-stream waits in a standalone CUDA-only reproducer. The
[sanitizer follow-up](docs/sanitizers.md#private-stream-follow-up-2026-09-08)
records that limitation separately from the clean memory/race checks.

## 5. Check a signature without trusting this code

Everything so far is the notary vouching for itself. This step takes the
signature out and verifies it with an unrelated library.

Save as `verify_sig.py`:

```python
import json, time, struct
from cuattest._cuda import DeviceBuffer
from cuattest.notary import Notary
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives import hashes

n = Notary()
payload = b"pretend these are weights" * 100
timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
with n._activate():
    buf = DeviceBuffer.from_bytes(n.cu, payload)
    try:
        fused = n._launch_fused_active([(buf.ptr, len(payload))],
                                       timestamp, b"sig-test")
    finally:
        buf.close()
root = fused.roots
doc = json.loads(fused.receipt)

pub = bytes.fromhex(n.info.as_dict()["gpu_pubkey_uncompressed"])
key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), pub)
signed_bytes = bytes.fromhex(doc["measurementDocument"])
sig = bytes.fromhex(doc["measurementSignature"])
der = utils.encode_dss_signature(int.from_bytes(sig[:32], "big"),
                                 int.from_bytes(sig[32:], "big"))

key.verify(der, signed_bytes, ec.ECDSA(hashes.SHA256()))
print("signature verifies")

# the document must commit to the fold it claims
d = json.loads(signed_bytes)
assert d["modelHash"] == n.hash_bytes(struct.pack("<I", 1) + root).hex()
assert d["device"] == f"cuda:{n.info.device_ordinal}"
assert d["kernelCID"] == f"urn:cid:{n.info.kernel_cid}"
assert d["cubinCID"] == f"urn:cid:{n.info.cubin_cid}"
print("modelHash is BLAKE3(LE32(N) || digests), as the document says")

# and one flipped bit must break it
bad = bytearray(signed_bytes); bad[40] ^= 0x01
try:
    key.verify(der, bytes(bad), ec.ECDSA(hashes.SHA256()))
    print("PROBLEM: a tampered document verified")
except Exception:
    print("a tampered document is rejected")
n.close()
```

```bash
.venv/bin/python verify_sig.py
```

**Good:** all three lines, no traceback.

**Proves:** the bytes the kernel emitted are a real ECDSA P-256 signature over
SHA-256 of the exact document text, checkable by anyone holding the public key
— and the document's `modelHash` is the documented fold, not an unrelated
number pasted in beside it.

The test uses a directly allocated device span instead of CUDA IPC so it can
focus on independent signature verification. The same fused production kernel
hashes that span, folds its root, and signs it without accepting an
intermediate digest from the host.

## 6. The server, including the paths that should fail

```bash
.venv/bin/cuattest serve &
```

Happy path:

```bash
curl -s localhost:8077/healthz
curl -s localhost:8077/v1/info
```

Now the unhappy ones, which are the interesting half — a notary that crashes on
a malformed request is a notary an untrusted caller can take offline:

```bash
curl -s localhost:8077/v1/nope                                     # 404
curl -s -X POST -d '{"ts":"2099-01-01T00:00:00Z","model":"x"}' \
     localhost:8077/v1/sign                                        # 400, tensors required
curl -s -X POST -d '{"tensors":"nope"}'  localhost:8077/v1/measure # 400, must be a list
curl -s -X POST -d 'not json'            localhost:8077/v1/measure # 400, not valid JSON
curl -s -X POST -d '{"tensors":[{"handle":"00","nbytes":16,"seg_off":0,"t_off":0,"device":0}]}' \
     localhost:8077/v1/measure                                     # 400, handle is 1 byte
```

**Good:** each returns the status in its comment and a JSON `{"error": ...}`
saying what was wrong, the process stays up, and `/healthz` still answers
afterwards.

## 7. Two processes, for real

This is the claim the whole project rests on: one process holds the key and
never sees the model, the other holds the model and never sees the key.

With the notary still running, in a second shell:

```bash
.venv/bin/pip install torch transformers      # a few GB; only the client needs it
.venv/bin/python examples/measure_torch_model.py
```

The example downloads gpt2 and loads it into *its own* process, so the notary
never sees a weight.

**Good:** the notary's `did:key` and pid, then `sharing 149 tensors over CUDA
IPC (zero copy)`, a measurement time, the statement manifest, and the last line
`the signature and per-tensor digest fold verify`. See
[the performance results](docs/performance.md) for controlled timings.
That final verification also reconstructs the detached credential-JWS
inputs and verifies each ES256 proof against the pinned GPU public key. Removing
the entire statement graph is covered as a downgrade attempt and must fail.

The pids in the output are the thing to look at. They are different processes
with different CUDA contexts, and no secret crossed between them — only IPC
handles in one direction and a signed document back.

The client explicitly releases every process-owned CUDA allocation lease after
the response. An orderly model teardown should therefore leave no cuAttest
lease registered, and repeated load/share/unload cycles should reclaim their
VRAM.
The refs are deliberately one-shot; call `share_model()` again for each later
measurement rather than resending a handle whose lease has been retired. Tests
also hold a Client at the RPC boundary to prove concurrent keepalive release is
rejected, and reuse a pre-request JSON copy to prove state lives in the registry
rather than only in one dictionary object.

**If measurement fails here** and the client is in a container, see
[operations](docs/operations.md): CUDA IPC handles only resolve inside a shared
IPC namespace, so `--ipc host` is required in addition to `--network host`.
They are separate requirements and missing the second one looks exactly like
this.

## 8. Does the loaded model match the checkpoint?

`expect` hashes an on-disk checkpoint through the same kernels and tells you
what the notary *would* report, so you can compare the two.

```bash
.venv/bin/cuattest expect /path/to/checkpoint
```

To compare against a real measurement, save one and pass it in. Exit status is
part of the interface: **0** match, **2** mismatch, **1** error.

```bash
TRUSTED_GPU_PUBKEY=04...  # captured from /v1/info through a trusted channel
.venv/bin/cuattest expect /path/to/checkpoint --compare receipt.json \
  --trusted-pubkey "$TRUSTED_GPU_PUBKEY"
echo $?
```

### Prove it actually catches something

A test that only ever passes has told you nothing. Change a single weight
before sharing it, and confirm the mismatch is caught *and named*:

```python
import json
from cuattest.client import Client
from cuattest.ipc import share_tensors
from safetensors.torch import load_file

state = load_file("/path/to/model.safetensors")
state = {k: v.cuda() for k, v in state.items()}
name = sorted(k for k, v in state.items()
              if k.endswith(".weight") and v.is_floating_point())[0]
state[name].view(-1)[0] += 1          # one number, in one tensor

names, refs, keepalive = share_tensors(state)
json.dump(Client().sign(refs, "tamper-test"), open("tampered.json", "w"))
del keepalive
```

```bash
.venv/bin/cuattest expect /path/to/checkpoint --compare tampered.json \
  --trusted-pubkey "$TRUSTED_GPU_PUBKEY"
```

**Good:** `NO MATCH — 1 of N tensors differ`, followed by the name of the
tensor you touched, and exit status 2.

### When a mismatch is not a problem

A mismatch is a starting point, not a verdict. Loading a model legitimately
changes what is resident: tied weights appear under two names, dtype
conversion rewrites every byte, vLLM fuses q/k/v, a tensor-parallel rank holds
a slice. [Expected CID](docs/expected-cid.md) covers this properly.

Two more causes are worth knowing before they surprise you, because both look
like tampering and neither is:

- **Key names differ between the checkpoint and the runtime.** gpt2 is the
  example everyone hits first: the checkpoint stores `h.0.attn.c_attn.weight`,
  while the runtime `state_dict()` calls it
  `transformer.h.0.attn.c_attn.weight`. Both sides sort by name, so a prefix
  difference reorders the fold and changes the root even though every tensor is
  byte-identical.
- **Buffers the runtime never loads.** gpt2's checkpoint carries twelve
  `attn.bias` causal masks that transformers treats as non-persistent, so the
  disk has 160 tensors and VRAM has 149.

Run gpt2 through this and you get `NO MATCH — tensor count differs (160 on
disk, 149 measured)`, and the hint blames fusion or sharding. Both are wrong
for this model; the weights are untouched. Check per-tensor digests before
concluding anything.

## 9. Throughput at model scale

Measuring four tiny tensors proves correctness, not that this is usable on a
real model. Hash something the size of a 7B model in bf16:

```python
import ctypes, json, os, urllib.request
from cuattest._cuda import Cuda, DeviceBuffer, CUipcMemHandle

GB, NT = 14.0, 291
cu = Cuda(); cu.cuInit(0); cu.ctx_create(0)
gh = cu.lib.cuIpcGetMemHandle          # the producer half; the notary only opens handles
gh.argtypes = [ctypes.POINTER(CUipcMemHandle), ctypes.c_ulonglong]; gh.restype = ctypes.c_int

per = (int(GB * 1e9) // NT) & ~0xFFF
chunk = os.urandom(1 << 20)
keep, refs = [], []
for _ in range(NT):
    b = DeviceBuffer(cu, per)
    for off in range(0, per, len(chunk)):
        cu.htod(b.ptr + off, chunk[:min(len(chunk), per - off)])
    keep.append(b)
    h = CUipcMemHandle(); cu.check(gh(ctypes.byref(h), b.ptr), "cuIpcGetMemHandle")
    refs.append({"handle": h.to_bytes().hex(), "seg_off": 0, "t_off": 0,
                 "nbytes": per, "device": 0})
cu.cuCtxSynchronize()

req = urllib.request.Request("http://127.0.0.1:8077/v1/measure",
                             json.dumps({"tensors": refs}).encode(),
                             {"Content-Type": "application/json"})
m = json.loads(urllib.request.urlopen(req, timeout=1800).read())
print(f"{NT * per / 1e9:.2f} GB in {m['seconds']}s -> {NT * per / 1e9 / m['seconds']:.1f} GB/s")
del keep
```

**Good on an idle H100 NVL:** 14 GB across 291 tensors in **about 0.9 s —
roughly 15 GB/s**, with signing adding **84 ms**. Run it twice: the root must
be identical.

Expect less under load. Hashing competes for the same compute units as whatever
inference is running on the device.

## 10. Test the package, not just the checkout

Everything above can pass in a source checkout while the thing users actually
install is broken, because a checkout has files a wheel might not ship. The
kernel `.cu` is exactly such a file: it is hashed at startup and its CID goes
into every signed document, so it has to travel inside the package.

```bash
.venv/bin/pip install build
.venv/bin/python -m build --wheel -o /tmp/wheeltest

# the .cu must be in there
python3 -c "import zipfile,glob; print([n for n in \
  zipfile.ZipFile(glob.glob('/tmp/wheeltest/*.whl')[0]).namelist() if n.endswith('.cu')])"

# and it must work from a clean venv with a cold CUBIN cache
python3 -m venv /tmp/wheeltest/venv
/tmp/wheeltest/venv/bin/pip install /tmp/wheeltest/*.whl
cd /tmp && CUATTEST_CACHE=/tmp/wheeltest/cache /tmp/wheeltest/venv/bin/cuattest selftest
```

**Good:** the listing shows `cuattest/kernel/p256_cuda_notary_b3.cu`, and the
selftest passes from a directory that is not the repo — compiling the kernel
from scratch on the way, which is what a first run on a fresh machine does.

Run this after any change to `pyproject.toml`. A packaging mistake here does
not show up in the unit tests, in `selftest`, or anywhere in a checkout.

## Registered IPC and asynchronous hashing

The normal CPU suite covers ticket expiration/replay/budgets, lost or mismatched
acknowledgements, producer mutation/replacement, concurrent sign/close,
interrupted retirement, and context-destruction quarantine. Real GPU tests use
separate producer/consumer processes and reverse their visible GPU ordinals:

```bash
CUATTEST_TEST_GPU=1 pytest -q tests/test_registered_integration.py
```

On `sm_80+`, rerun with `CUATTEST_HASH_MODE=async` to force the new kernel through
small, unaligned and partial-block inputs; the default only selects it for
large `sm_120` requests. On an RTX 2080, leave the mode at `auto`/`standard`.
The test also keeps an unrelated producer stream blocked during an observation
and verifies that registration does not wait for that stream.

The sanitizer runner supports `--suite registered` and isolates each backend
in a fresh producer process. Its default does not suppress any checks. See
the [production registration validation report](docs/testing/registered-ipc-2026-09-09.md)
for the duplicate-import memcheck control, explicit diagnostic exclusion,
mixed-backend racecheck limitation, exact commands and results.

## Cleaning up

The notary holds a CUDA context for its lifetime, so stop it when you are done
and confirm the device is idle:

```bash
kill %1          # the notary from section 6, still a job in that shell
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

**Good:** the compute-apps table comes back empty.

From another shell, find it and kill it by pid. Resist `pkill -f` over ssh: the
remote shell's own command line contains the pattern you just passed, so it
matches itself and disconnects you.

```bash
pgrep -af 'bin/cuattest serve'
kill <pid>
```

## Where to look when something fails

[Operations](docs/operations.md) has the error-by-error list: confidential
compute not unlocked (`rc=802`), missing driver or NVRTC, IPC handles that do
not resolve, cross-device references, invalid spans, and stalled HTTP bodies.

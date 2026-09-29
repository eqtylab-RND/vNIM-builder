# The kernel

`kernel/p256_cuda_notary_b3.cu` is a deliberately expanded and heavily
commented CUDA translation unit, compiled by NVRTC straight to a CUBIN. It
includes only CUDA's cooperative-groups header, with the limited-dependencies
mode enabled, and does not link the CUDA runtime.

That self-containment is deliberate. The CUBIN's content id is part of the
notary's identity, so the fewer things that can change what gets compiled, the
more that id is worth. Everything it needs — BLAKE3, SHA-256, P-256, base58,
base32, a JSON writer — is in the file.

## Layout

| order | section | what lives there |
|---|---|---|
| 1 | BLAKE3 | scalar-scheduled compression, paired vector loads, and reduction helpers |
| 2 | SHA-256 + P-256 | ECDSA, RFC 6979, fixed-base table, and keygen |
| 3 | streaming BLAKE3 | small-buffer hashing and the final model fold |
| 4 | statement assembly | encoders, validators, and two-pass receipt construction |
| 5 | receipt finalization | parallel signing and one-shot root consumption |
| 6 | fused measurement | tiled chunk hashing, every reduction level, and model fold |

## Entry points

Three `__global__` functions. Everything else is `__device__` and unreachable
from the host.

| kernel | launch | does |
|---|---|---|
| `keygen_kernel` | 1×1 | derives the session key, exports the public point |
| `measure_model_fused_kernel` | cooperative grid×128 | hashes every tensor, completes every reduction level, folds the model root, and optionally arms its private handoff |
| `attest_measured_kernel` | 1×128 | consumes the pending model root and constructs the signed receipt |

Both request kernels are queued on the same stream, followed by one host
synchronization and one result copy. Splitting them is intentional: the
P-256/statement call graph needs 255 registers and a large stack, whereas the
tiled bulk kernel needs 96 registers and has 62.5% theoretical occupancy on the
profiled `sm_75` GPU. No intermediate digest is copied to or supplied by the
host.

## Device state

The sensitive globals are the reason the design works:

```c
__device__ static Byte g_private_scalar_big_endian[32];
__device__ static int  g_session_key_ready;
__device__ static Byte g_compressed_public_key[33];
__device__ static Byte g_pending_model_root[32];
__device__ static int  g_pending_tensor_count;
__device__ static int  g_pending_receipt_armed;
```

`g_private_scalar_big_endian` is written once by `keygen_kernel` and read only
by the signing path. There is no kernel that exports it, and no host call that
can read a device global that is not exposed through a module symbol lookup —
which this package never does. It dies when the context is destroyed.

`g_session_key_ready` is checked before receipt construction; signing before
keygen returns status `-2` rather than signing with zeros.

The pending root is a private, one-shot handoff. Measurement invalidates any
old value at entry, publishes the newly folded root only after all hashing has
completed, and arms it with an atomic operation after a device fence. The
attestation kernel atomically consumes that arm, checks the tensor count, and
wipes the root. Its ABI has no model-root argument, so the host cannot redirect
the signer to an arbitrary digest. Serialized service requests and same-stream
launch ordering complete this protocol.

The module also owns a fixed-base P-256 table built by `keygen_kernel`; it is
described below. It contains public curve points, not key material.

## BLAKE3

Two implementations, for two different jobs.

**Parallel**, for measuring tensors. The fused kernel preserves every tensor's
fixed 1 KiB BLAKE3 chunks but groups 128 of them into a 128 KiB scheduling tile.
Each 64-thread half-block owns one tile and each lane processes two independent
leaves, issuing both sets of loads before their dependent compression chains to
expose instruction-level parallelism. The common aligned path reads each
64-byte message block with four `uint4` loads per leaf and keeps all 16 message
words and compression-state words in named scalars. Arbitrarily aligned spans
and partial final blocks retain a byte-safe fallback.

Neighboring lanes own neighboring 2 KiB regions, so leaf-load addresses at one
warp instruction are 2 KiB-strided: they are aligned and vectorized per lane,
but are not contiguous warp-coalesced. This mapping keeps two full sequential
BLAKE3 chains in each lane's registers. Cross-tile reduction has a different
layout: adjacent threads load adjacent pairs of 32-byte chaining values, so
those global reads and writes are coalesced.

The 128 leaf CVs and their first seven pairwise tree levels live in an
approximately 8.2 KiB block-local shared array. It is transposed by CV word and
padded after every 32 logical slots, so simultaneous child reads cover the 32
shared-memory banks without conflicts. A barrier separates reading parents
into registers from overwriting their children. Only the tile's final 32-byte
CV is written to global scratch.

The host packs immutable tensor descriptors and per-level prefix sums before
launch. Prefix sums schedule cross-tile work from differently sized tensors;
they are not hash inputs. Two global workspaces ping-pong for those remaining
levels. Large spans approach 48 bytes per 128 KiB tile: 32 primary bytes plus
a half-sized peer. For one 96 GiB tensor that is 24 MiB in the primary
workspace plus 12 MiB in the secondary, or 36 MiB total instead of 4.5 GiB.
Odd trailing nodes are carried unchanged;
the final parent receives BLAKE3's `ROOT` flag, while a one-chunk tensor receives
it during chunk compression.

**Incremental**, for folding inside the fused kernel. `Blake3StreamingState`
implements an ordinary single-thread streaming hasher: a 1 KiB buffer and a
54-deep stack of chaining values, merged on the binary carry pattern of the
chunk counter:

```c
struct Blake3StreamingState {
    Uint32 subtree_stack[54][8];
    int subtree_count;
    Byte pending_chunk[1024];
    int pending_byte_count;
    Uint64 completed_chunk_count;
};
```

54 is enough for 2⁵⁴ chunks, which is more input than the device can hold.

Both produce the exact BLAKE3 digest for the same bytes. That matters because
the digest *is* the content id — CIDv1, raw codec `0x55`, multihash `0x1e` —
so it has to agree with every other BLAKE3 implementation in the world, not
just with itself.

## Cooperative launch

A grid-wide barrier is safe only when every block can reside concurrently.
At startup the host verifies cooperative-launch support and asks the CUDA
occupancy API how many 128-thread blocks of this exact function can be active
per multiprocessor. It multiplies that by the GPU's multiprocessor count and
never launches a larger grid. The kernel's grid-stride loops decouple resident
grid size from tile count, so a 96 GiB input's 786,432 scheduling tiles do not
require that many simultaneously resident blocks.

All measurement work has one cooperative launch. A signed request adds one
small, same-stream finalization launch; both finish before the only host
synchronization. The former path needed a chunk launch and one launch per
reduction depth for each tensor, followed by a model-fold launch and a signing
launch. Grid-wide synchronization between measurement stages remains
on-device. See NVIDIA's
[cooperative-groups documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
and [cooperative driver launch API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html).

## SHA-256 and P-256

SHA-256 is textbook, and used for three things: the ECDSA message hash, HMAC
inside RFC 6979, and deriving a UUID for each credential.

The curve arithmetic is straightforward but worth knowing the shape of:

- **Barrett reduction** rather than Montgomery. `barrett_reduce()` takes a 512-bit
  product and reduces mod `m` using a precomputed `mu = ⌊2^512 / m⌋`. Both `mu`
  values arrive in the `ctx` buffer, so the kernel never computes them.
- **Jacobian coordinates** for point arithmetic (`jacobian_double`,
  `jacobian_add`), converted back to affine once at the end of
  `multiply_generator_by_scalar`, which costs one modular
  inversion via Fermat (`modexp` with `p − 2`).
- **RFC 6979** deterministic nonces (`derive_rfc6979_nonce`). No randomness at signing
  time — a kernel has no good entropy source, and a repeated `k` would leak the
  private key. Determinism removes that failure mode entirely.

Secret scalar processing is constant-work. During keygen the kernel builds a
six-bit fixed-base table containing the 63 non-zero multiples needed by each
of 43 windows. `multiply_generator_by_scalar` scans all 63 entries of every
window using masks,
then always performs one point addition; it never indexes the table with the
secret digit. Exceptional Jacobian cases and modular corrections are likewise
mask-selected, and modular exponentiation always performs its multiply.

RFC 6979 evaluates exactly four candidates and mask-selects the first valid
one. A P-256 candidate is rejected with probability below 2⁻³², so exhausting
all four has probability below 2⁻¹²⁸; that exceptional case fails closed with
status `-7`. Thus neither nonce/private-key digits nor nonce rejection control
the amount of signing work.

`ecdsa_sign` returns `r ‖ s` as 64 big-endian bytes.

## The `ctx` buffer

The host uploads 26 little-endian `Uint64` limbs once at startup. The kernel
indexes it by fixed offsets:

| offset | limbs | contents |
|---|---|---|
| `CONTEXT_FIELD_PRIME_OFFSET` 0 | 4 | field prime `p` |
| `CONTEXT_GROUP_ORDER_OFFSET` 4 | 4 | group order `n` |
| `CONTEXT_FIELD_BARRETT_FACTOR_OFFSET` 8 | 5 | Barrett `mu` for `p` |
| `CONTEXT_ORDER_BARRETT_FACTOR_OFFSET` 13 | 5 | Barrett `mu` for `n` |
| `CONTEXT_GENERATOR_X_OFFSET` 18 | 4 | generator x |
| `CONTEXT_GENERATOR_Y_OFFSET` 22 | 4 | generator y |

These are public constants. Passing them in rather than compiling them in
keeps the kernel free of large initialised arrays, which NVRTC handles poorly.

## Keygen

```
d = SHA256( os_csprng_seed[32] ) mod n
```

The trusted notary host obtains the seed from the operating-system CSPRNG. It
is the sole credited entropy source. The kernel deliberately does not treat
CUDA timers, scheduling variation, data races, or atomic-arbitration order as
cryptographic entropy: CUDA documents no device-side entropy primitive, and
none of those effects has a portable lower bound on attacker-conditioned
min-entropy.

The host supplies and therefore knows the seed. It can reproduce `d` from the
seed and the public derivation algorithm. Retaining `d` in module-private
device storage isolates signing state from the untrusted model process; it is
not a defense against the trusted notary process or a privileged host. The
[trust model](trust-model.md) makes that boundary explicit.

Afterwards the kernel zeroes its scalar copy and SHA output. The host
separately scrubs the device-side seed buffer before freeing it.

## Statement assembly

The signing phase writes JSON directly into a caller-supplied buffer through a
tiny bounds-checked writer (`Wr`: pointer, length, capacity, sticky error flag).
No allocation, no formatting library. Overflow sets the error flag and the
kernel returns `-4` rather than truncating.

### Encoders

| function | produces |
|---|---|
| `raw_cid_str` | `b` + base32 of `0x01 0x55 0x1e 0x20 ‖ digest` — 59 chars, for content |
| `rdfc_cid_str` | `b` + base32 of `0x01 0x83 0xE8 0x02 0x1E 0x20 ‖ digest` — 62 chars, for canonicalized statements |
| `b58enc` | base58btc, for the `did:key` |
| `b64url_enc` | base64url, for the JWS signature |
| `uuid_text` | a UUIDv4-shaped string from a SHA-256, for credential ids |

The two CID prefixes are the distinction that matters: **raw** identifies
bytes, **RDFC-1** identifies a canonicalized RDF statement. A `StateAttestation`
points at raw content, is named by a UUID derived from its canonicalized
preimage, and its registration is itself named by an RDFC id.

### Validators

Inputs from the host-side atomic operation are checked before anything is signed:

- `is_valid_timestamp` accepts exactly `dddd-dd-ddTdd:dd:ddZ`, 20 bytes. The
  kernel reads a fixed width, so a short timestamp would read past the upload.
- `is_valid_model_name` accepts 1–64 bytes of `[A-Za-z0-9/._-]`. This keeps the
  model name from escaping its JSON string.
- the selected device ordinal must be non-negative; every IPC reference is
  separately checked against that same ordinal before this kernel can launch.

Either failing returns `-3`.

### What gets built

Receipt construction is two-pass. `prepare_signing_hashes` builds the two
independent signing inputs: the measurement document plus the Verifiable
Credential being registered. Threads 0–15 then compute their P-256
signatures concurrently in a single warp. Finally,
`assemble_attestation_from_root` emits the same deterministic structures with
those signatures supplied, avoiding serial scalar multiplications.

`write_measurement_prefix` composes the measurement document in a 1024-byte
buffer, SHA-256s it, and emits the document as hex alongside its detached
signature. The document is small and fixed-shape:

```json
{"claim":"modelHash and modelCID were computed in-GPU from client-submitted, bounds-checked VRAM spans; …",
 "cubinCID":"urn:cid:…","device":"cuda:N","gpuDID":"did:key:…",
 "hashScheme":"BLAKE3 per submitted span …","kernelCID":"urn:cid:…",
 "measuredAt":"…","model":"…","modelCID":"urn:cid:…","modelHash":"…",
 "operation":"gpu-hash-submitted-vram-spans","tensorCount":N}
```

`tensorCount` is retained as the legacy wire-field name; it counts submitted
spans and does not establish that those spans are runtime tensors.

Then two builders produce the manifest's single statement:

- `build_credential_id_preimage` — the credential's own triples with its name
  replaced by a blank node: `modelRoot` (what the bytes are), `instanceID`
  (which resident copy, folded by the host over the submitted IPC handles),
  `stateType`, and `previousStateCredential` linking to that copy's previous
  credential or null for the first in the session. Its RDFC digest becomes the
  credential's UUID.
- `build_credential_registration` — the `StateAttestation` itself, signed with
  the session key and wrapped in a `CredentialRegistration`

The preimage is a two-blank-node RDF dataset (the credential and its nested
`state`), so it needs real canonicalization rather than the single-subject
shortcut: `hash_credential_preimage_nquads` implements RDFC-1.0 sections
4.4/4.6 for that closed schema, handling 9 quads for a genesis credential and
10 once `previousStateCredential` is present. Which node takes `c14n0` is
value-dependent — adding `previousStateCredential` can flip it — so neither the
labels nor the final quad order may be templated. The registration wrapper is a
four-blank-node dataset (proof subject, proof graph, registration, and the
nested state) handled the same way by
`hash_credential_registration_nquads`, at 19 or 20 quads.

The JWS signing document has exactly one blank node, so its canonical label is
forced; the kernel emits it in sorted order and `nquads_are_sorted` verifies
that rather than trusting the template.

Per-instance chains live in module-private device globals
(`g_chains`, sized by `CUATTEST_CHAIN_SLOTS_MAX` and bounded at runtime by
`configure_chain_slots_kernel`). Keeping them on-device is what stops a host
from forking or rewriting a chain. A lookup miss with every active slot in use
is an error, not an eviction: reusing a slot would emit a genesis statement for
a copy that already has predecessors, which no verifier could distinguish from
a forked chain.

Before publishing them, the host independently reconstructs the complete
closed-schema graph and its canonical RDFC CIDs from the validated measurement
document. Extra fields, altered relationships and missing or duplicate
registration coverage are rejected rather than exposed as credentialed claims.
The external evidence verifier additionally rebuilds each detached-JWS signing
input and verifies all three raw ES256 signatures against the already verified
GPU public key; wrapper CIDs alone are not treated as authentication, and a
missing statement graph is rejected rather than treated as an optional layer.

The credential builder is the largest piece. It derives a UUID, writes the
credential's canonical N-Quads into a 2560-byte buffer, hashes those, hashes
the proof-options quads separately, concatenates the two digests as the JWS
signing input, and signs. The proof is deliberately *not* referenced in the
document quads — it does not exist yet at signing time.

## Two constraints worth knowing

**Only 32-byte digests cross function boundaries.** The three builders are
`__noinline__`, and they take subjects as bare digests rather than as
pre-formatted `urn:cid:…` strings. This is not style. The source records that
on this NVRTC/ptxas combination, caller-held buffers that had to stay valid
across large nested calls were observed being aliased and clobbered by the
optimizer, corrupting statements with unrelated bytes. Short CID strings are
now rebuilt locally where they are consumed. If you refactor this file, keep
that property.

**Pointer meaning remains a host trust boundary.** The production signer does
not accept signable digests: it consumes only the private, one-shot root
published by the preceding measurement kernel. It cannot establish that the
measured addresses semantically represent framework tensors, however. The
client chooses the handles and offsets; the host is responsible for importing
each CUDA IPC allocation, checking its driver-owned device and bounds, and
constructing the descriptors. Those checks do not bind the spans to an
executing inference graph. See
[trust-model.md](trust-model.md).

## Status codes

The request kernels share one status `int`:

| code | meaning |
|---|---|
| `0` | success; `out_len` holds the JSON length |
| `-2` | keygen has not run |
| `-3` | bad timestamp or model name |
| `-4` | output buffer too small |
| `-5` | invalid fused measurement plan |
| `-6` | no matching measured root is pending |
| `-7` | deterministic P-256 signing failed |
| `-8` | chain slot count out of range |
| `-9` | chain slots already configured |
| `-10` | no free chain slot for this instance |
| `-11` | chain slots not configured |

The host allocates 6144 bytes for the output, which fits the manifest and the
document with room to spare (~3.7 KiB in practice, the only variable part being
the 64-byte model name). It must match `kOutCapacity` in `_native.cpp`.

## Building it

```bash
cuattest build-kernel --arch sm_90
```

Compiled with `--std=c++11` and a **real** architecture (`sm_90`, not
`compute_90`) so NVRTC emits a CUBIN rather than PTX. Needs NVRTC and the CUDA
runtime/CCCL headers used by cooperative groups; `pip install '.[build]'`
supplies all three from NVIDIA wheels when no toolkit is installed. Compilation
needs no GPU.

Changing the source changes `kernel_cid`; recompiling changes `cubin_cid`. Both
are computed by an independent host BLAKE3 implementation before the module is
loaded, passed into the signing call, and included in every signed document.
Cache metadata binds a CUBIN to its exact source, binary digest, and its own
NVRTC compiler version. Per-CUBIN provenance remains correct when only one
architecture is rebuilt after a toolkit upgrade, and travels with artifacts
copied to another directory.

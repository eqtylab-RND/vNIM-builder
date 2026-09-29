# Trust model

## What a cuAttest signature proves

**The measurement kernel read these bytes from the submitted, bounds-checked
spans in this GPU during the request**, using code whose content id is in the
same document and a private scalar retained in device memory.

The claim is deliberately narrow, and it is checkable in two independent ways:
the caller re-folds the per-tensor digests and must arrive at the same root,
and anyone with the public key can verify the signature afterwards. The
comparison command and Python `compare()` API accept a match only from signed
evidence, check every duplicate count/digest/root field against that document,
and require the expected public key to be pinned from a separate trusted
channel. Invalid evidence is an error, never a content mismatch.

## What it does not prove

- **That the runtime computed correctly with those bytes.** The notary attests
  what its measurement kernel read from the submitted spans, not that an
  inference engine used them or behaved correctly.
- **That the supplied spans are semantically framework tensors or were used by
  an inference.** The client chooses every IPC handle and offset. The notary
  imports and bounds-checks those spans and the fused kernel hashes them itself,
  but neither step establishes their application-level meaning. A compromised
  workload can keep a pristine decoy checkpoint in unused VRAM and submit it
  while executing with different weights.
- **Anything about the host.** No TEE quote, no TPM, no platform attestation.
  If you need "this ran in a genuine confidential VM", that comes from
  elsewhere and is a separate signature.
- **That the model file on disk matches.** The measurement is of resident
  memory. A loader that transformed the weights produces a different root, by
  design.
- **That an uncooperative producer froze concurrent writes.** The sharing API
  requires tensors to be quiescent and immutable for the full IPC lease and
  checks PyTorch version counters where available. Raw CUDA writes can bypass
  that tripwire; a producer that violates the contract can request evidence
  over a torn state just as it can request evidence over arbitrary buffers.

## The key

The trusted notary host obtains a 32-byte seed from the operating-system CSPRNG.
A CUDA kernel hashes that seed, maps it to a P-256 private scalar, and retains
the scalar in device globals until the process exits. CUDA timers, scheduling
variation, memory races, and atomic-arbitration order are assigned zero
credited entropy.

The scalar is not copied back by cuAttest, but the host supplies the seed and
knows the derivation algorithm, so it can reproduce the scalar. Device-only
storage protects signing state from the separate, untrusted model process; it
does not create a security boundary against the trusted notary or a privileged
host.

That last point is a feature and a constraint: **a notary's identity spans one
run**. Restart it and the `did:key` changes. Bind evidence to a run, not to a
long-lived identity.

## Who has to be trusted

| party | can it forge a measurement? |
|---|---|
| the model process | it cannot forge signatures or use the private key directly, but it can request a genuine signature over arbitrary in-bounds buffers it submits |
| the notary process | yes — it supplies the seed and controls the signing service |
| a privileged host, driver, or hypervisor | outside this claim — cuAttest provides no host TEE or platform attestation |
| an unprivileged process on the box | only by breaking the OS/process boundary or the cryptography |

The notary is the trusted component. The process split keeps its signing state
away from the workload, but the signature alone authenticates only the bytes
and metadata submitted to the service. A claim about the weights actually used
for inference additionally requires a trusted runtime integration, tensor
enumerator, or manifest policy that binds the submitted spans to the executing
graph independently of the workload.

The compiled C++ host backend is part of that trusted notary process: it
imports and bounds-checks the client-selected spans passed to the GPU. The
`host_backend` value in `/v1/info` is diagnostic and is not part of the signed
measurement document. `kernel_cid` and `cubin_cid` identify the device code,
not the Python service or native host extension.

The source and CUBIN identities are hashed by an independent host BLAKE3
implementation before the CUDA module is loaded. Both CIDs, the selected GPU
ordinal, and the signer DID are in the signed measurement document. Consumers
that require a particular build still need to compare those values with their
allowlist and pin the notary public key obtained through their trusted channel.

## Confidential compute

On an H100/H200 with CC mode on, the GPU must be unlocked before CUDA runs at
all. That is a platform property, not something this tool provides — it will
simply fail to start, and say so.

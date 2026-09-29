# Expected CID from an on-disk model

[Multi-GPU receipts](multi-gpu.md) use the same global span order and CID fold.
Compare them with `--trusted-pubkeys trusted-gpu-keys.json` (a UUID-to-key map)
instead of the single `--trusted-pubkey` argument below.

```bash
cuattest expect ./checkpoint
TRUSTED_GPU_PUBKEY=04...  # pin captured through a trusted channel
cuattest expect ./checkpoint --compare measured.json \
  --trusted-pubkey "$TRUSTED_GPU_PUBKEY"
```

Computes the CID the notary *would* report, from a safetensors checkpoint,
without loading the model into a framework. Then you can compare it with what
the notary measured from the ordered VRAM spans submitted by a client.

A match means those submitted spans have the checkpoint's canonical byte
sequence and fold. It does not prove that the spans are runtime tensors or
that an inference graph used them; a client can submit a pristine decoy copy.

## How it computes it

The same fold the notary uses — per-tensor BLAKE3 in sorted name order, then
`BLAKE3(LE32(N) ‖ digests)` — run through **the notary's own kernels**, so the
hashing is identical by construction rather than by assumption. One tensor is
resident at a time, so peak extra VRAM is the largest tensor, not the model.
Zero-element tensors are excluded from both expectation and runtime
measurement because they have no non-empty CUDA byte span to authenticate.

Accepts a `.safetensors` file, a `*.safetensors.index.json` shard index, or a
directory. In a directory, conventional `model.safetensors.index.json` and
then `model.safetensors` take precedence; otherwise multiple safetensors
indexes are rejected as ambiguous and must be selected by their explicit path.

## Tied weights are handled

A checkpoint stores one buffer for tied weights while the runtime state dict
may present several names. `expect` restores only ties supported by model-side
architecture/configuration evidence, or by the versioned
`cuattest.aliases.v1` full-span schema. Generic safetensors metadata is not
enough: `save_model` uses the same free-form string map for dropped overlapping
views, but that map contains no offset, shape, or extent. Treating it as a
full-size alias would compute an expectation no runtime view could match.

The built-in configuration rules cover T5's shared, encoder, decoder, and
optionally output embeddings, plus a narrower causal-model
`tie_word_embeddings` fallback:

For checkpoint producers that need another proven full-span tie, the metadata
value is JSON text with this versioned shape:

```json
{"cuattest.aliases.v1":"{\"alias.weight\":{\"kind\":\"full-span\",\"source\":\"stored.weight\"}}"}
```

`source` must name a stored tensor and `alias.weight` must be absent from the
file. Partial or offset views intentionally have no representation in version
1 and are not reconstructed.

```
  tensors    291  (0.99 GB on disk)
  vram_cid   bafkr4igpm533s4lzx3cug5bx5lvavzrrd42xgmtgelgspqh33crd2ne4eq
  tied       lm_head.weight = model.embed_tokens.weight  (restored from explicit schema/model config)
```

Without it, that model reads 290 on disk against 291 measured and never
matches. `--no-tied` turns the aliasing off.

## When it will not match — legitimately

The notary measures **resident memory**, which equals the checkpoint only when
the loader changed nothing. All of these are normal, and none of them mean
anything is wrong:

| cause | what you see |
|---|---|
| dtype conversion (F32 checkpoint loaded as BF16) | same count, every tensor differs |
| fusion (vLLM merges q/k/v into `qkv_proj`) | fewer tensors measured than on disk |
| tensor parallelism | each rank holds a slice, not the whole tensor |
| quantisation | different bytes and usually a different count |
| a runtime that renames tensors | counts may match, names do not |

So: a match authenticates the submitted byte spans against this checkpoint
expectation. A mismatch is a starting point, and the tool tries to give you
the next step rather than just a verdict.

## Reading a mismatch

```
  expected  bafkr4iea3qsia5s…  (290 tensors)
  measured  bafkr4igpm533s4l…  (291 tensors)

  NO MATCH — tensor count differs (290 on disk, 291 measured): the submission
  has more spans than the checkpoint expectation — tied weights
  in the runtime are the usual honest-client cause
```

For an honest `share_model` client, both sides are sorted by name. When counts
agree, a positional diff is therefore useful and the tool labels differing
positions with the expected tensor names. Names are not part of the submitted
IPC references or signed fold, so these labels are not a semantic guarantee
against a malicious client:

```
  NO MATCH — 3 of 291 tensors differ

  3 tensor(s) differ:
    model.layers.3.self_attn.qkv_proj.weight
    model.layers.7.mlp.down_proj.weight
    model.norm.weight
```

That is the same mechanism the continuous re-attestation uses to say *which*
weight changed, rather than only that something did.

## Exit codes

| code | meaning |
|---|---|
| `0` | match, or no `--compare` given |
| `2` | compared and did not match |
| `1` | it could not run — unreadable checkpoint, no GPU, unavailable verifier, or invalid evidence |

`2` is separate from `1` on purpose: a mismatch is a finding, not a failure,
and a script should be able to tell them apart.

## What `--compare` accepts

A signed attestation receipt, either directly or under a `measurement` key in
`run.json`. An unsigned `/v1/measure` response is useful diagnostics but is not
accepted as proof, even if its caller-supplied `vram_cid` matches. The verifier
checks the detached P-256 signature and re-folds `digests` before reporting
`MATCH`.

`--trusted-pubkey` is mandatory for comparisons and must be the 65-byte
uncompressed P-256 key obtained through a trusted channel (for example, a
pinned local `/v1/info` response). Never copy this value from the receipt being
tested: a self-signed receipt can authenticate its contents, but cannot
establish that its signer is the notary you intended to trust. Signature
verification also requires `pip install 'cuattest[verify]'`; if that support is
missing, or if the document/signature is invalid, the command exits `1`, not
the authenticated content-mismatch status `2`. The Python `compare()` API has
the same mandatory `trusted_pubkey` and raises `EvidenceError` for invalid
evidence.

Receipt and `run.json` inputs are treated as untrusted: duplicate JSON keys at
any nesting depth are rejected, hexadecimal fields must use strict hex text,
and files larger than 8 MiB are refused before parsing. The limit is well above
the largest receipt produced under the server's default 16,384-span ceiling.

# SPDX-License-Identifier: Apache-2.0
"""What CID would result if these checkpoint bytes were submitted canonically?

This computes the same fold the notary computes — per-tensor BLAKE3 in sorted
name order, then ``BLAKE3(LE32(N) ‖ digests)`` — over the bytes in a
safetensors checkpoint, using the notary's own kernels so the hashing is
identical by construction rather than by assumption.

**It will not always match, and that is not a bug.** For an honest
``share_model`` client, submitted VRAM spans equal the checkpoint only when
the loader changed nothing. Common, legitimate reasons for a mismatch:

  * **tied weights** — a checkpoint stores one buffer, the runtime exposes it
    under two names, so the runtime measures more tensors than the disk has
  * **dtype conversion** — loading an F32 checkpoint as BF16 changes the bytes
  * **fusion** — vLLM fuses q/k/v into one ``qkv_proj``, so names and count
    both differ
  * **sharding** — a tensor-parallel rank holds a slice, not the whole tensor
  * **quantisation** — different bytes entirely

So treat a match as evidence that the *submitted VRAM spans* match this
checkpoint canonicalization. It does not bind those spans to the tensors an
inference runtime actually uses; a client can submit decoy buffers. A mismatch
is a starting point: the comparison names which submitted positions differ.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import ids
from ._cuda import DeviceBuffer
from ._hosthash import blake3_digest
from ._protocol import (
    MEASUREMENT_CLAIM,
    MEASUREMENT_HASH_SCHEME,
    MEASUREMENT_OPERATION,
    loads_no_duplicate_fields,
    parse_measurement_document,
)
from ._statements import validate_statement_graph
from .notary import GpuCleanupUncertainError, GpuSessionAbortedError
from .safetensors import DiskTensor, load_with_aliases

# Some older causal checkpoints store one embedding buffer while an honest
# runtime submission exposes both embedding and output-head state-dict names.
# This is the backward-compatible config fallback when no richer metadata or
# architecture group supplies the aliases.
TIED_ALIASES = {
    "tie_word_embeddings": ("model.embed_tokens.weight", "lm_head.weight"),
}
# These causal families share the fallback's input-embedding naming contract.
# An output head alone is not enough to identify it (GPT-2 uses transformer.wte,
# for example), so reverse reconstruction needs this model-side evidence.
CAUSAL_EMBEDDING_MODEL_TYPES = {
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "gemma",
    "gemma2",
    "phi3",
}

# T5 model_type alone does not identify the runtime: encoder-only, base
# encoder-decoder, and conditional-generation classes expose different names.
T5_MODEL_TYPES = {"t5", "mt5", "umt5", "longt5"}
T5_INPUT_EMBEDDINGS = (
    "shared.weight",
    "encoder.embed_tokens.weight",
    "decoder.embed_tokens.weight",
)
T5_ARCHITECTURE_GROUPS = {
    family + architecture: group
    for family in ("T5", "MT5", "UMT5", "LongT5")
    for architecture, group in (
        ("EncoderModel", T5_INPUT_EMBEDDINGS[:2]),
        ("Model", T5_INPUT_EMBEDDINGS),
        ("ForConditionalGeneration", T5_INPUT_EMBEDDINGS + ("lm_head.weight",)),
    )
}

# Kernel receipts are currently only a few KiB plus 64 hexadecimal characters
# per measured span. This cap comfortably covers the service's default tensor
# limit while preventing a local, untrusted --compare file from being read into
# memory without bound.
MAX_EVIDENCE_FILE_BYTES = 8 * 1024 * 1024
MAX_MEASUREMENT_DOCUMENT_BYTES = 16 * 1024


@dataclass(frozen=True)
class Expectation:
    model_root: str
    vram_cid: str
    tensor_count: int
    digests: str  # hex, 32 bytes per tensor, sorted by name
    names: list[str]
    total_bytes: int
    tied: tuple[str, ...] = ()  # aliases proven by explicit schema/model config

    def as_dict(self) -> dict:
        return {
            "model_root": self.model_root,
            "vram_cid": self.vram_cid,
            "tensor_count": self.tensor_count,
            "digests": self.digests,
            "total_bytes": self.total_bytes,
            "tied_aliases": list(self.tied),
        }

    def digest_of(self, name: str) -> str | None:
        try:
            i = self.names.index(name)
        except ValueError:
            return None
        return self.digests[i * 64 : (i + 1) * 64]


def resolve_ties(
    path: Path,
    tensors: list[DiskTensor],
    declared_full_span_aliases: dict[str, str] | None = None,
) -> tuple[list[DiskTensor], list[str]]:
    """Restore shared state-dict names omitted from a safetensors checkpoint.

    Explicit ``cuattest.aliases.v1`` declarations are full-span ties by schema.
    Free-form ``save_model`` metadata is deliberately not used: it cannot tell
    an identical tie from a shorter overlapping view. ``config.json`` supplies
    model-side proof for architecture-aware T5-family reconstruction plus the
    causal-language-model fallback used by earlier cuAttest releases.
    """
    present = {tensor.name: tensor for tensor in tensors}
    serialized_names = set(present)
    added: list[str] = []

    def add_alias(alias: str, source: str) -> None:
        if alias in present or source not in present:
            return
        base = present[source]
        # Copying the complete extent is valid only for callers below: either
        # the explicit full-span schema or a known architecture tie group.
        # Never feed raw safetensors save_model metadata into this helper;
        # dropped views do not carry their offset/shape/length there.
        present[alias] = DiskTensor(
            alias, base.path, base.offset, base.nbytes, base.dtype, base.shape
        )
        added.append(f"{alias} = {source}")

    def complete_group(names: tuple[str, ...]) -> None:
        # Prefer the first canonical name that was actually serialized. Each
        # generated alias then points directly at that disk span, rather than
        # forming chains whose result could depend on iteration order.
        source = next((name for name in names if name in serialized_names), None)
        if source is None:
            source = next((name for name in names if name in present), None)
        if source is None:
            return
        for alias in names:
            add_alias(alias, source)

    for alias, source in sorted((declared_full_span_aliases or {}).items()):
        add_alias(alias, source)

    cfg_path = (path if path.is_dir() else path.parent) / "config.json"
    if not cfg_path.is_file():
        return sorted(present.values(), key=lambda tensor: tensor.name), added
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        return sorted(present.values(), key=lambda tensor: tensor.name), added
    if not isinstance(cfg, dict):
        return sorted(present.values(), key=lambda tensor: tensor.name), added

    if cfg.get("model_type") in T5_MODEL_TYPES:
        architectures = cfg.get("architectures")
        if not architectures:
            # Preserve the legacy fallback for checkpoints with no saved
            # class. An explicit class takes precedence, including classes
            # whose naming contract we do not know (no inferred aliases).
            t5_group = T5_INPUT_EMBEDDINGS + ("lm_head.weight",)
        elif (
            isinstance(architectures, list)
            and len(architectures) == 1
            and isinstance(architectures[0], str)
        ):
            t5_group = T5_ARCHITECTURE_GROUPS.get(architectures[0], ())
        else:
            t5_group = ()
        if cfg.get("tie_word_embeddings") is not True:
            t5_group = tuple(name for name in t5_group if name != "lm_head.weight")
        complete_group(t5_group)
    else:
        for flag, group in TIED_ALIASES.items():
            if cfg.get(flag) is True and (
                group[0] in present
                or cfg.get("model_type") in CAUSAL_EMBEDDING_MODEL_TYPES
            ):
                # save_model may retain either member. The config proves the
                # full-span tie in both directions; free-form metadata cannot.
                # T5 uses its own group above, never this causal naming scheme.
                complete_group(group)
    return sorted(present.values(), key=lambda tensor: tensor.name), added


def compute(notary, path: str | Path, progress=None, tied: bool = True) -> Expectation:
    """Hash every tensor in `path` with the notary's kernels and fold them.

    One tensor is resident on the device at a time, so peak extra VRAM is the
    size of the largest tensor, not of the model. The resulting expectation
    describes ordered bytes, not their use by an inference runtime.
    """
    path = Path(path)
    # Runtime sharing intentionally omits zero-element tensors because CUDA
    # IPC has no non-empty byte span to authenticate for them. Apply the same
    # canonical rule to disk expectations before aliases, counts, or folding.
    checkpoint_tensors, declared_full_span_aliases = load_with_aliases(path)
    tensors: list[DiskTensor] = [
        tensor for tensor in checkpoint_tensors if tensor.nbytes
    ]
    if not tensors:
        raise ValueError(f"no non-empty tensors found in {path}")
    added: list[str] = []
    if tied:
        tensors, added = resolve_ties(path, tensors, declared_full_span_aliases)

    digests = bytearray()
    total = 0
    # A tied alias points at the same bytes; hash them once.
    seen: dict[tuple, bytes] = {}
    # DeviceBuffer uses the thread-current CUDA context. Keep this notary's
    # context active across allocation, hashing, and freeing so compute()
    # remains correct when several Notary instances coexist in one process.
    with notary._activate():
        for i, t in enumerate(tensors):
            key = (str(t.path), t.offset, t.nbytes)
            digest = seen.get(key)
            if digest is None:
                buf = DeviceBuffer.from_bytes(notary.cu, t.read())
                try:
                    digest = notary.hash_dptr(buf.ptr, t.nbytes)
                except (GpuSessionAbortedError, GpuCleanupUncertainError):
                    # hash_dptr destroys the owning context when queued work
                    # cannot be drained. The context owns this allocation from
                    # that point onward; freeing it here would either address a
                    # dead context or race work when destruction was uncertain.
                    buf.ptr = 0
                    raise
                finally:
                    buf.close()
                seen[key] = digest
                total += t.nbytes
            digests += digest
            if progress:
                progress(i + 1, len(tensors), t.name)

        root_input = len(tensors).to_bytes(4, "little") + bytes(digests)
        model_root = notary.hash_bytes(root_input)
    return Expectation(
        model_root=model_root.hex(),
        vram_cid=ids.raw_cid(model_root),
        tensor_count=len(tensors),
        digests=bytes(digests).hex(),
        names=[t.name for t in tensors],
        total_bytes=total,
        tied=tuple(added),
    )


@dataclass
class Comparison:
    matches: bool
    reason: str
    expected: Expectation
    measured_cid: str | None = None
    measured_count: int | None = None
    differing: list[str] | None = None  # only when counts line up

    def report(self) -> str:
        lines = [
            f"  expected  {self.expected.vram_cid}  ({self.expected.tensor_count} tensors)"
        ]
        if self.measured_cid:
            lines.append(
                f"  measured  {self.measured_cid}"
                + (f"  ({self.measured_count} tensors)" if self.measured_count else "")
            )
        lines.append("")
        lines.append(
            "  MATCH — the submitted VRAM spans match the checkpoint bytes"
            if self.matches
            else f"  NO MATCH — {self.reason}"
        )
        if self.differing is not None and not self.differing:
            lines.append("\n  every per-tensor digest agrees — only the fold differs")
        elif self.differing:
            lines.append(f"\n  {len(self.differing)} tensor(s) differ:")
            for n in self.differing[:12]:
                lines.append(f"    {n}")
            if len(self.differing) > 12:
                lines.append(f"    … and {len(self.differing) - 12} more")
        return "\n".join(lines)


class EvidenceError(ValueError):
    """A purported receipt is unsigned, malformed, or internally inconsistent."""


class VerificationUnavailableError(EvidenceError):
    """Evidence cannot be classified because signature support is not installed."""


@dataclass(frozen=True)
class VerifiedMeasurement:
    model_root: str
    vram_cid: str
    tensor_count: int
    digests: str | None
    document: dict
    # Statements present in the manifest that this verification did NOT
    # authenticate, because they were issued by someone other than the GPU --
    # the notary service's own IdentityAttestation, typically. Checking those
    # needs the issuer's key pinned separately, so they are reported rather
    # than silently counted as evidence.
    unverified_statements: tuple[str, ...] = ()


def _hex_bytes(
    value,
    field: str,
    length: int | None = None,
    *,
    min_length: int = 0,
    max_length: int | None = None,
) -> bytes:
    """Decode strict hex only after cheap text-size bounds have succeeded."""
    if not isinstance(value, str):
        raise EvidenceError(f"{field} must be hexadecimal text")

    # bytes.fromhex accepts embedded ASCII whitespace and allocates the output
    # before callers can inspect its length. Reject both properties up front:
    # fixed-size fields never scan or allocate an attacker-sized string, and
    # variable-size fields are capped before their characters are examined.
    text_length = len(value)
    if length is not None:
        if text_length != length * 2:
            raise EvidenceError(f"{field} must contain exactly {length} bytes")
    elif (
        text_length % 2
        or text_length < min_length * 2
        or (max_length is not None and text_length > max_length * 2)
    ):
        raise EvidenceError(f"{field} has an invalid size")
    if re.fullmatch(r"[0-9A-Fa-f]*", value) is None:
        raise EvidenceError(f"{field} is not valid hexadecimal")
    try:
        raw = bytes.fromhex(value)
    except ValueError as e:
        raise EvidenceError(f"{field} is not valid hexadecimal") from e
    return raw


def _require_equal(receipt: dict, field: str, expected) -> None:
    if field in receipt and receipt[field] != expected:
        raise EvidenceError(
            f"unsigned {field} contradicts the signed measurement document"
        )


def verify_evidence(
    receipt: dict,
    trusted_pubkey: str | None = None,
    *,
    trusted_pubkeys: dict[str, str] | None = None,
) -> VerifiedMeasurement:
    """Verify a receipt and return only measurement data authenticated by it.

    Top-level convenience fields are never treated as evidence. The root,
    count, device and code identities come from the detached, signed document;
    every credential JWS, duplicate top-level value and optional digest list is
    checked before a caller can report MATCH.
    """
    if not isinstance(receipt, dict):
        raise EvidenceError("evidence must be a JSON object")
    if "receipt_type" in receipt:
        from .multigpu import verify_multi_gpu_evidence

        if trusted_pubkey is not None:
            raise EvidenceError(
                "multi-GPU evidence requires trusted_pubkeys by GPU UUID"
            )
        return verify_multi_gpu_evidence(receipt, trusted_pubkeys)
    if trusted_pubkeys is not None:
        raise EvidenceError("single-GPU evidence requires trusted_pubkey")
    document_bytes = _hex_bytes(
        receipt.get("measurementDocument"),
        "measurementDocument",
        min_length=1,
        max_length=MAX_MEASUREMENT_DOCUMENT_BYTES,
    )
    signature = _hex_bytes(
        receipt.get("measurementSignature"), "measurementSignature", 64
    )
    public_hex = receipt.get("gpu_pubkey_uncompressed")
    public = _hex_bytes(public_hex, "gpu_pubkey_uncompressed", 65)
    if public[0] != 0x04:
        raise EvidenceError(
            "gpu_pubkey_uncompressed is not an uncompressed P-256 point"
        )
    if trusted_pubkey is not None:
        if not isinstance(trusted_pubkey, str):
            raise EvidenceError("trusted GPU public key must be hexadecimal text")
        if public_hex.lower() != trusted_pubkey.lower():
            raise EvidenceError("receipt was not signed by the trusted GPU public key")

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec, utils
    except ImportError as e:
        raise VerificationUnavailableError(
            "signature verification requires the 'verify' extra "
            "(pip install 'cuattest[verify]')"
        ) from e
    try:
        key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), public)
        der = utils.encode_dss_signature(
            int.from_bytes(signature[:32], "big"), int.from_bytes(signature[32:], "big")
        )
        key.verify(der, document_bytes, ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError) as e:
        raise EvidenceError("measurement signature does not verify") from e

    try:
        # Use the same duplicate-free, exact schema as the publishing host.
        # Otherwise a genuinely signed extra assertion (or a duplicate key
        # interpreted differently by another JSON parser) could overstate the
        # narrow submitted-span guarantee.
        document = parse_measurement_document(document_bytes)
    except (UnicodeDecodeError, ValueError) as e:
        raise EvidenceError(f"invalid signed measurementDocument: {e}") from e

    # A valid signature over a stronger, unverified semantic claim is still
    # misleading evidence. Pin the exact narrow statement made by the trusted
    # kernel, rather than accepting arbitrary caller-controlled prose.
    if document.get("claim") != MEASUREMENT_CLAIM:
        raise EvidenceError("signed claim does not describe submitted VRAM spans")
    if document.get("hashScheme") != MEASUREMENT_HASH_SCHEME:
        raise EvidenceError("signed hashScheme does not describe submitted spans")

    root = _hex_bytes(document.get("modelHash"), "signed modelHash", 32)
    root_hex = root.hex()
    cid = ids.raw_cid(root)
    if document.get("modelCID") != f"urn:cid:{cid}":
        raise EvidenceError("signed modelCID does not encode signed modelHash")
    count = document.get("tensorCount")
    if type(count) is not int or not 0 < count <= (1 << 31) - 1:
        raise EvidenceError("signed tensorCount must be a positive integer")

    derived_did = ids.did_key_p256(public[1:33], public[33:65])
    if document.get("gpuDID") != derived_did:
        raise EvidenceError("signed gpuDID does not identify the verification key")
    for field in ("device", "kernelCID", "cubinCID", "measuredAt"):
        if not isinstance(document.get(field), str) or not document[field]:
            raise EvidenceError(f"signed document has no {field}")
    device_match = re.fullmatch(r"cuda:(0|[1-9][0-9]*)", document["device"])
    if device_match is None or int(device_match.group(1)) > (1 << 31) - 1:
        raise EvidenceError("signed device is not a CUDA ordinal")
    for field in ("kernelCID", "cubinCID"):
        value = document[field]
        if (
            len(value) != len("urn:cid:") + 59
            or re.fullmatch(r"urn:cid:b[a-z2-7]+", value) is None
        ):
            raise EvidenceError(f"signed {field} is not a raw BLAKE3 CID")
    if document.get("operation") != MEASUREMENT_OPERATION:
        raise EvidenceError("signed operation is not a submitted-span measurement")
    if not isinstance(document.get("model"), str) or not document["model"]:
        raise EvidenceError("signed document has no model")
    if (
        re.fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
            document["measuredAt"],
        )
        is None
    ):
        raise EvidenceError("signed measuredAt is not an RFC3339 UTC timestamp")

    # Statement verification is not an optional enhancement. Every kernel
    # receipt contains the manifest; allowing an intermediary to delete it
    # would downgrade verification to the detached measurement signature and
    # silently discard the credential authentication check.
    try:
        manifest = receipt["manifest"]
    except KeyError:
        raise EvidenceError(
            "evidence is missing required credential statements"
        ) from None

    def verify_credential_signature(signing_input: bytes, raw_signature: bytes) -> None:
        """Verify one kernel-issued detached JWS with the pinned GPU key."""
        try:
            der_signature = utils.encode_dss_signature(
                int.from_bytes(raw_signature[:32], "big"),
                int.from_bytes(raw_signature[32:], "big"),
            )
            key.verify(
                der_signature,
                signing_input,
                ec.ECDSA(hashes.SHA256()),
            )
        except (InvalidSignature, ValueError) as error:
            raise ValueError("credential proof signature does not verify") from error

    try:
        # A receipt that has passed through the service also carries its
        # IdentityAttestation. It is reported, never trusted: verifying it
        # needs the service key pinned independently of the GPU's.
        unverified = validate_statement_graph(
            manifest,
            document,
            verify_credential_signature=verify_credential_signature,
            kernel_only=False,
        )
    except ValueError as error:
        raise EvidenceError(f"invalid statement graph: {error}") from error

    # Receipts intentionally duplicate these values for ergonomic tooling;
    # none is trusted unless it agrees byte-for-byte with signed content.
    _require_equal(receipt, "model_root", root_hex)
    _require_equal(receipt, "modelRoot", root_hex)
    _require_equal(receipt, "vram_cid", cid)
    _require_equal(receipt, "tensor_count", count)
    _require_equal(receipt, "gpu_did", derived_did)
    _require_equal(receipt, "device", document["device"])
    _require_equal(
        receipt, "kernel_cid", document["kernelCID"].removeprefix("urn:cid:")
    )
    _require_equal(receipt, "cubin_cid", document["cubinCID"].removeprefix("urn:cid:"))
    _require_equal(receipt, "measured_at", document["measuredAt"])

    digest_hex = receipt.get("digests")
    if digest_hex is not None:
        digests = _hex_bytes(digest_hex, "digests", count * 32)
        folded = blake3_digest(count.to_bytes(4, "little") + digests)
        if folded != root:
            raise EvidenceError("digests do not fold to the signed modelHash")
        digest_hex = digests.hex()

    return VerifiedMeasurement(
        root_hex, cid, count, digest_hex, document, tuple(unverified)
    )


def compare(
    expected: Expectation,
    measured: dict,
    trusted_pubkey: str | None = None,
    *,
    trusted_pubkeys: dict[str, str] | None = None,
) -> Comparison:
    """Compare an expectation against independently verified signed evidence."""
    if trusted_pubkey is None and not trusted_pubkeys:
        # A receipt can prove that one key signed its contents, but it cannot
        # prove that the self-supplied key is the notary the caller intended.
        # Refuse the comparison altogether instead of ever returning MATCH on
        # a caller-selected trust root.
        raise EvidenceError(
            "comparison requires a trusted GPU public key obtained through a trusted channel"
        )

    # Invalid signatures and malformed/contradictory documents are execution
    # failures, not ordinary content mismatches. Let EvidenceError (including
    # VerificationUnavailableError) reach the CLI's exit-1 path and library
    # callers' verification-error handling.
    verified = verify_evidence(
        measured, trusted_pubkey=trusted_pubkey, trusted_pubkeys=trusted_pubkeys
    )

    cid = verified.vram_cid
    count = verified.tensor_count

    if count is not None and count != expected.tensor_count:
        delta = count - expected.tensor_count
        hint = (
            "the submission has more spans than the checkpoint expectation — "
            "tied weights in the runtime are the usual honest-client cause"
            if delta > 0
            else "the submission has fewer spans — fusion or sharding is the usual "
            "honest-client cause"
        )
        return Comparison(
            False,
            f"tensor count differs ({expected.tensor_count} on disk, "
            f"{count} measured): {hint}",
            expected,
            cid,
            count,
        )

    if cid == expected.vram_cid and verified.model_root == expected.model_root.lower():
        return Comparison(True, "", expected, cid, count)

    # Same count: a positional diff is meaningful, since both sides sort by name.
    md = verified.digests
    if md and len(md) == len(expected.digests):
        differing = [
            expected.names[i]
            for i in range(expected.tensor_count)
            if md[i * 64 : (i + 1) * 64].lower()
            != expected.digests[i * 64 : (i + 1) * 64].lower()
        ]
        if not differing:
            # [] rather than None: the distinction between "compared, nothing
            # differs" and "could not compare" is worth keeping.
            return Comparison(
                False,
                "per-tensor digests all agree but the roots differ — "
                "the tensor ordering must differ",
                expected,
                cid,
                count,
                differing=[],
            )
        return Comparison(
            False,
            f"{len(differing)} of {expected.tensor_count} tensors differ",
            expected,
            cid,
            count,
            differing,
        )

    return Comparison(
        False,
        "the content ids differ; no per-tensor digests were supplied to narrow it down",
        expected,
        cid,
        count,
    )


def load_measurement(path: str | Path) -> dict:
    """Read signed evidence from a receipt or a run.json wrapper."""
    evidence_path = Path(path)
    # A bounded read remains safe if the file grows between stat/open/read.
    # Reading one extra byte distinguishes the exact limit from truncation.
    with evidence_path.open("rb") as evidence_file:
        encoded = evidence_file.read(MAX_EVIDENCE_FILE_BYTES + 1)
    if len(encoded) > MAX_EVIDENCE_FILE_BYTES:
        raise ValueError(
            f"{path} exceeds the {MAX_EVIDENCE_FILE_BYTES}-byte evidence limit"
        )
    d = loads_no_duplicate_fields(encoded)
    if not isinstance(d, dict):
        raise ValueError(  # noqa: TRY004 - malformed JSON value, not API misuse
            f"{path} must contain a JSON object"
        )
    if (
        "measurementDocument" not in d
        and "receipt_type" not in d
        and isinstance(d.get("measurement"), dict)
    ):
        d = d["measurement"]
    if "measurementDocument" not in d and "receipt_type" not in d:
        raise ValueError(f"{path} has no signed measurementDocument — is it a receipt?")
    return d

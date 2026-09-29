# SPDX-License-Identifier: Apache-2.0
"""Signed-evidence comparison logic. No GPU required."""

import base64
import json

import pytest

pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

from cuattest import _statements as statements_module
from cuattest import expect as expect_module
from cuattest import ids
from cuattest._hosthash import blake3_digest
from cuattest._protocol import (
    MEASUREMENT_CLAIM,
    MEASUREMENT_HASH_SCHEME,
    MEASUREMENT_OPERATION,
)
from cuattest.expect import (
    EvidenceError,
    Expectation,
    compare,
    verify_evidence,
)

D1, D2, D3 = "11" * 32, "22" * 32, "33" * 32


def make(names, digests):
    digest_hex = "".join(digests)
    root = blake3_digest(len(names).to_bytes(4, "little") + bytes.fromhex(digest_hex))
    return Expectation(
        model_root=root.hex(),
        vram_cid=ids.raw_cid(root),
        tensor_count=len(names),
        digests=digest_hex,
        names=list(names),
        total_bytes=0,
    )


# A fixed residency identity for fixtures. Real receipts fold this from the
# submitted IPC handles; tests only need it stable and distinct from a root.
INSTANCE_ROOT = blake3_digest(b"test-instance")


def signed_manifest(document, private, previous_uuid=None):
    """Build the kernel's one-statement manifest with a genuine ES256 proof."""
    issuer = document["gpuDID"]
    timestamp = document["measuredAt"]
    model_urn = document["modelCID"]
    instance_urn = f"urn:cid:{ids.raw_cid(INSTANCE_ROOT)}"
    credential_id = statements_module._state_credential_id(
        document["modelHash"], issuer, timestamp, instance_urn, model_urn,
        previous_uuid,
    )
    signing_input = statements_module._credential_signing_input(
        credential_id, issuer, timestamp, instance_urn, model_urn, previous_uuid
    )
    der = private.sign(signing_input, ec.ECDSA(hashes.SHA256()))
    signature_r, signature_s = utils.decode_dss_signature(der)
    raw_signature = signature_r.to_bytes(32, "big") + signature_s.to_bytes(32, "big")
    jws = (
        statements_module.JWS_PREFIX
        + base64.urlsafe_b64encode(raw_signature).rstrip(b"=").decode()
    )
    statement_id = statements_module._credential_registration_id(
        credential_id, issuer, timestamp, instance_urn, model_urn, previous_uuid, jws
    )
    statements = {
        statement_id: state_registration(
            statement_id, credential_id, issuer, timestamp, instance_urn,
            model_urn, previous_uuid, jws,
        )
    }
    return {"version": statements_module.MANIFEST_VERSION, "statements": statements}


def state_registration(
    statement_id, credential_id, issuer, timestamp, instance_urn, model_urn,
    previous_uuid, jws,
):
    """The registered StateAttestation exactly as the kernel serializes it."""
    return {
        "@context": statements_module.STATEMENT_CONTEXT,
        "@id": statement_id,
        "@type": "CredentialRegistration",
        "credential": {
            "@context": statements_module.VC_CONTEXT,
            "id": credential_id,
            "type": ["VerifiableCredential", statements_module.CREDENTIAL_TYPE],
            "credentialSubject": {
                "id": issuer,
                "state": {
                    "stateType": statements_module.STATE_TYPE,
                    "instanceID": instance_urn,
                    "modelRoot": model_urn,
                    "previousStateCredential": previous_uuid,
                },
            },
            "issuer": issuer,
            "proof": {
                "type": "EcdsaSecp256r1Signature2019",
                "proofPurpose": "assertionMethod",
                "verificationMethod": f"{issuer}#{issuer[8:]}",
                "created": timestamp,
                "jws": jws,
            },
            "validFrom": timestamp,
        },
        "registeredBy": issuer,
        "timestamp": timestamp,
    }


def recomputed_statement_id(wrapper):
    """Re-derive a wrapper's content id after its proof bytes were edited."""
    credential = wrapper["credential"]
    state = credential["credentialSubject"]["state"]
    return statements_module._credential_registration_id(
        credential["id"],
        credential["issuer"],
        wrapper["timestamp"],
        state["instanceID"],
        state["modelRoot"],
        state["previousStateCredential"],
        credential["proof"]["jws"],
    )


def signed_receipt(
    digests,
    *,
    include_digests=True,
    claim=MEASUREMENT_CLAIM,
    extra_fields=None,
    duplicate_claim=False,
    include_statements=True,
    private=None,
):
    count = len(digests)
    digest_hex = "".join(digests)
    root = blake3_digest(count.to_bytes(4, "little") + bytes.fromhex(digest_hex))
    cid = ids.raw_cid(root)
    private = private or ec.generate_private_key(ec.SECP256R1())
    public = private.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    did = ids.did_key_p256(public[1:33], public[33:])
    document = {
        "claim": claim,
        "cubinCID": f"urn:cid:{ids.raw_cid(bytes.fromhex(D2))}",
        "device": "cuda:3",
        "gpuDID": did,
        "hashScheme": MEASUREMENT_HASH_SCHEME,
        "kernelCID": f"urn:cid:{ids.raw_cid(bytes.fromhex(D3))}",
        "measuredAt": "2026-09-03T10:00:00Z",
        "model": "test/model",
        "modelCID": f"urn:cid:{cid}",
        "modelHash": root.hex(),
        "operation": MEASUREMENT_OPERATION,
        "tensorCount": count,
    }
    if extra_fields:
        document.update(extra_fields)
    serialized = json.dumps(document, separators=(",", ":"))
    if duplicate_claim:
        # Python's ordinary decoder chooses the final occurrence and would see
        # the valid claim below, while a first-occurrence parser would expose
        # this stronger statement.  Verifiers must reject the ambiguity.
        serialized = '{"claim":"inference used these weights",' + serialized[1:]
    document_bytes = serialized.encode()
    der = private.sign(document_bytes, ec.ECDSA(hashes.SHA256()))
    r, s = utils.decode_dss_signature(der)
    receipt = {
        "measurementDocument": document_bytes.hex(),
        "measurementSignature": (r.to_bytes(32, "big") + s.to_bytes(32, "big")).hex(),
        "gpu_pubkey_uncompressed": public.hex(),
        "gpu_did": did,
        "modelRoot": root.hex(),
        "model_root": root.hex(),
        "vram_cid": cid,
        "tensor_count": count,
        "kernel_cid": document["kernelCID"].removeprefix("urn:cid:"),
        "cubin_cid": document["cubinCID"].removeprefix("urn:cid:"),
        "device": document["device"],
        "measured_at": document["measuredAt"],
    }
    if include_digests:
        receipt["digests"] = digest_hex
    if include_statements:
        receipt["manifest"] = signed_manifest(document, private)
    return receipt


def compare_pinned(expected, receipt):
    return compare(expected, receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_comparison_rejects_a_receipts_self_selected_trust_root():
    exp = make(["a"], [D1])
    receipt = signed_receipt([D1])

    with pytest.raises(EvidenceError, match="trusted GPU public key"):
        compare(exp, receipt)


def test_identical_signed_evidence_matches():
    exp = make(["a", "b"], [D1, D2])
    receipt = signed_receipt([D1, D2])
    r = compare_pinned(exp, receipt)
    assert r.matches and "MATCH" in r.report()


def test_verifier_accepts_genuinely_signed_credential_proofs():
    receipt = signed_receipt([D1])

    verified = verify_evidence(
        receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"]
    )

    assert verified.tensor_count == 1


def test_comparison_rejects_credential_statement_downgrade():
    expectation = make(["a"], [D1])
    receipt = signed_receipt([D1])
    del receipt["manifest"]

    # The measurement signature remains genuine, so this specifically proves
    # that deleting the credential proof cannot downgrade verification into a
    # successful measurement-only comparison.
    with pytest.raises(EvidenceError, match="missing required credential statements"):
        compare_pinned(expectation, receipt)


def test_verifier_rejects_a_forged_proof_even_with_recomputed_wrapper_cid():
    receipt = signed_receipt([D1])
    graph = receipt["manifest"]["statements"]
    credentials = [
        (statement_id, statement)
        for statement_id, statement in graph.items()
        if statement["@type"] == "CredentialRegistration"
    ]
    old_statement_id, wrapper = credentials[0]
    proof = wrapper["credential"]["proof"]
    raw_signature = bytearray(
        base64.urlsafe_b64decode(
            proof["jws"][len(statements_module.JWS_PREFIX) :] + "=="
        )
    )
    raw_signature[0] ^= 1
    proof["jws"] = (
        statements_module.JWS_PREFIX
        + base64.urlsafe_b64encode(raw_signature).rstrip(b"=").decode()
    )

    # Model the actual attack: the wrapper CID is content-derived, so an
    # intermediary can recompute it after replacing the proof bytes.
    forged_statement_id = recomputed_statement_id(wrapper)
    wrapper["@id"] = forged_statement_id
    del graph[old_statement_id]
    graph[forged_statement_id] = wrapper

    with pytest.raises(EvidenceError, match="credential proof signature"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_verifier_rejects_noncanonical_credential_signature_pad_bits():
    receipt = signed_receipt([D1])
    graph = receipt["manifest"]["statements"]
    old_statement_id, wrapper = next(
        (statement_id, statement)
        for statement_id, statement in graph.items()
        if statement["@type"] == "CredentialRegistration"
    )
    proof = wrapper["credential"]["proof"]
    encoded = proof["jws"][len(statements_module.JWS_PREFIX) :]
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    last_index = alphabet.index(encoded[-1])
    assert last_index % 16 == 0  # canonical four zero pad bits
    noncanonical = encoded[:-1] + alphabet[last_index + 1]
    assert base64.urlsafe_b64decode(noncanonical + "==") == base64.urlsafe_b64decode(
        encoded + "=="
    )
    proof["jws"] = statements_module.JWS_PREFIX + noncanonical

    # Recompute the content-derived wrapper ID so canonical signature text is
    # the only failed invariant, not a stale map key.
    replacement_id = recomputed_statement_id(wrapper)
    wrapper["@id"] = replacement_id
    del graph[old_statement_id]
    graph[replacement_id] = wrapper

    with pytest.raises(EvidenceError, match="non-canonical JWS signature"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_verifier_rejects_old_tensor_semantics_even_when_genuinely_signed():
    receipt = signed_receipt(
        [D1],
        claim=(
            "modelHash and modelCID were computed in-GPU from VRAM-resident "
            "tensors; this measurement document was assembled and signed in-kernel"
        ),
    )

    with pytest.raises(EvidenceError, match="submitted VRAM spans"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_verifier_rejects_genuinely_signed_fields_outside_the_exact_schema():
    receipt = signed_receipt([D1], extra_fields={"inferenceUsedTheseWeights": True})

    with pytest.raises(EvidenceError, match="unexpected field"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_verifier_rejects_genuinely_signed_duplicate_fields():
    receipt = signed_receipt([D1], duplicate_claim=True)

    with pytest.raises(EvidenceError, match="duplicate field"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_verifier_rejects_oversized_hex_before_decoding():
    receipt = signed_receipt([D1])
    receipt["measurementDocument"] = "00" * (
        expect_module.MAX_MEASUREMENT_DOCUMENT_BYTES + 1
    )

    with pytest.raises(EvidenceError, match="measurementDocument has an invalid size"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])

    receipt = signed_receipt([D1])
    receipt["digests"] = "00" * 1024
    with pytest.raises(EvidenceError, match="digests must contain exactly 32 bytes"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_verifier_rejects_whitespace_in_fixed_length_hex():
    receipt = signed_receipt([D1])
    receipt["measurementSignature"] = "00 " * 42 + "00"
    receipt["measurementSignature"] = receipt["measurementSignature"][:128]

    with pytest.raises(EvidenceError, match="not valid hexadecimal"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


@pytest.mark.parametrize("invalid_count", [True, 1.0])
def test_verifier_rejects_non_integer_signed_tensor_counts(invalid_count):
    receipt = signed_receipt([D1], extra_fields={"tensorCount": invalid_count})

    with pytest.raises(EvidenceError, match="tensorCount must be an integer"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_verifier_rejects_an_unvalidated_statement_map():
    receipt = signed_receipt([D1])
    receipt["manifest"] = {
        "version": statements_module.MANIFEST_VERSION,
        "statements": {"claim": {"inferenceUsedTheseWeights": True}},
    }

    with pytest.raises(EvidenceError, match="not a credential"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_matching_caller_cid_without_a_signature_is_not_proof():
    """Regression: copying the expected CID into an unsigned dict cannot MATCH."""
    exp = make(["a"], [D1])
    with pytest.raises(EvidenceError, match="measurementDocument"):
        compare(
            exp,
            {"vram_cid": exp.vram_cid, "tensor_count": 1},
            trusted_pubkey="04" + "00" * 64,
        )


def test_unsigned_cid_cannot_override_a_different_signed_root():
    exp = make(["a"], [D1])
    receipt = signed_receipt([D2])
    receipt["vram_cid"] = exp.vram_cid
    with pytest.raises(EvidenceError, match="contradicts"):
        compare_pinned(exp, receipt)


def test_tampered_signature_is_rejected():
    exp = make(["a"], [D1])
    receipt = signed_receipt([D1])
    receipt["measurementSignature"] = "00" * 64
    with pytest.raises(EvidenceError, match="does not verify"):
        compare_pinned(exp, receipt)


def test_contradictory_count_is_rejected_before_match():
    exp = make(["a"], [D1])
    receipt = signed_receipt([D1])
    receipt["tensor_count"] = 99
    with pytest.raises(EvidenceError, match="tensor_count contradicts"):
        compare_pinned(exp, receipt)


def test_contradictory_digests_are_rejected_before_match():
    exp = make(["a"], [D1])
    receipt = signed_receipt([D1])
    receipt["digests"] = D2
    with pytest.raises(EvidenceError, match="do not fold"):
        compare_pinned(exp, receipt)


def test_more_measured_tensors_blames_tied_weights():
    exp = make(["a"], [D1])
    receipt = signed_receipt([D1, D2])
    r = compare_pinned(exp, receipt)
    assert not r.matches and "tied weights" in r.reason


def test_fewer_measured_tensors_blames_fusion():
    exp = make(["a", "b", "c"], [D1, D2, D3])
    receipt = signed_receipt([D1, D2])
    r = compare_pinned(exp, receipt)
    assert not r.matches and "fusion or sharding" in r.reason


def test_same_count_names_the_differing_tensors():
    exp = make(["a", "b", "c"], [D1, D2, D3])
    receipt = signed_receipt([D1, "99" * 32, D3])
    r = compare_pinned(exp, receipt)
    assert not r.matches and r.differing == ["b"]
    assert "b" in r.report()


def test_digest_comparison_is_case_insensitive():
    exp = make(["a"], [D1])
    receipt = signed_receipt([D2])
    receipt["digests"] = D2.upper()
    r = compare_pinned(exp, receipt)
    assert r.differing == ["a"]


def test_no_digests_supplied_says_so():
    exp = make(["a"], [D1])
    receipt = signed_receipt([D2], include_digests=False)
    r = compare_pinned(exp, receipt)
    assert not r.matches and "no per-tensor digests" in r.reason


def test_report_lists_at_most_twelve_then_counts():
    names = [f"t{i}" for i in range(20)]
    exp = make(names, [D1] * 20)
    receipt = signed_receipt([D2] * 20)
    r = compare_pinned(exp, receipt)
    assert len(r.differing) == 20 and "and 8 more" in r.report()


def test_verifier_can_pin_the_expected_public_key():
    receipt = signed_receipt([D1])
    verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])
    other = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
        )
        .hex()
    )
    with pytest.raises(ValueError, match="trusted GPU"):
        verify_evidence(receipt, trusted_pubkey=other)

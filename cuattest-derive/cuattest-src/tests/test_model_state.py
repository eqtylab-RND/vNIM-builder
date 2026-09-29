# SPDX-License-Identifier: Apache-2.0
"""The host's own gpuModelTensorsV1 report about a copy it loaded.

The GPU signs what it found in VRAM; this credential is the host saying what it
put there. The two roots are folded by the identical algorithm, so a faithful
load makes them equal -- which is only meaningful if the host really does hash
the file the same way the kernel hashes the spans. That equivalence, and the
fact that every reported field is inside the signature, is what these cover.
"""

import base64
import struct

import pytest

from cuattest import _hostkey, _modelstate, _statements, ids, safetensors
from cuattest._hosthash import blake3_digest

ec = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ec")
utils = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.utils")
hashes = pytest.importorskip("cryptography.hazmat.primitives.hashes")

INSTANCE = "urn:cid:" + ids.raw_cid(blake3_digest(b"instance"))
MODEL_CID = "urn:cid:" + ids.raw_cid(blake3_digest(b"on-disk"))
MODEL_ROOT = "urn:cid:" + ids.raw_cid(blake3_digest(b"in-vram"))
TS = "2026-09-16T19:00:00Z"


@pytest.fixture
def key(tmp_path):
    return _hostkey.load_or_create(tmp_path / "host_key")


def _build(key, **overrides):
    args = {
        "instance_urn": INSTANCE,
        "model_cid": MODEL_CID,
        "model_root_urn": MODEL_ROOT,
        "timestamp": TS,
    }
    args.update(overrides)
    return _modelstate.build(key, **args)


def test_the_report_carries_exactly_the_five_declared_facts(key):
    state = _build(key)["credentialSubject"]["state"]
    assert state == {
        "stateType": "gpuModelTensorsV1",
        "instanceID": INSTANCE,
        "modelRoot": MODEL_ROOT,
        "modelCID": MODEL_CID,
        "reportedBy": key.did,
    }


def test_its_state_type_is_distinct_from_the_gpus(key):
    # The GPU attests gpuModelTensorsStateV1 about the same copy. Same subject,
    # different reporter, so the two must not be confusable.
    assert _modelstate.STATE_TYPE == "gpuModelTensorsV1"
    assert _statements.STATE_TYPE == "gpuModelTensorsStateV1"
    assert _modelstate.STATE_TYPE != _statements.STATE_TYPE


def test_the_host_is_both_subject_and_issuer(key):
    credential = _build(key)
    assert credential["credentialSubject"]["id"] == key.did
    assert credential["issuer"] == key.did
    # "for now" -- reportedBy is a separate field precisely so it can diverge
    # from the issuer later without changing the credential's shape.
    assert credential["credentialSubject"]["state"]["reportedBy"] == key.did
    assert credential["type"] == ["VerifiableCredential", "StateAttestation"]


def test_the_proof_verifies_against_the_host_key(key):
    credential = _build(key)
    raw = base64.urlsafe_b64decode(credential["proof"]["jws"].split("..")[1] + "==")
    signing_input = _modelstate.signing_input(
        credential["id"], key.did, INSTANCE, MODEL_CID, MODEL_ROOT, TS
    )
    key._private_key.public_key().verify(
        utils.encode_dss_signature(
            int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
        ),
        signing_input,
        ec.ECDSA(hashes.SHA256()),
    )


@pytest.mark.parametrize(
    "field", ["instance_urn", "model_cid", "model_root_urn", "timestamp"]
)
def test_every_reported_field_is_covered_by_the_signature(key, field):
    """A field outside the signature is a field an intermediary can rewrite."""
    other = "2026-01-01T00:00:00Z" if field == "timestamp" else (
        "urn:cid:" + ids.raw_cid(blake3_digest(b"tampered"))
    )
    args = {
        "instance_urn": INSTANCE, "model_cid": MODEL_CID,
        "model_root_urn": MODEL_ROOT, "timestamp": TS,
    }
    genuine = _modelstate.signing_input(_build(key)["id"], key.did, **args)
    args[field] = other
    assert _modelstate.signing_input(_build(key)["id"], key.did, **args) != genuine


def test_the_id_is_derived_so_restating_a_fact_does_not_mint_a_new_one(key):
    # A manifest must not grow an identifier every receipt for one fixed fact.
    assert _build(key)["id"] == _build(key)["id"]


@pytest.mark.parametrize("field", ["instance_urn", "model_cid", "model_root_urn"])
def test_a_different_fact_gets_a_different_id(key, field):
    other = "urn:cid:" + ids.raw_cid(blake3_digest(b"other"))
    assert _build(key, **{field: other})["id"] != _build(key)["id"]


def test_the_registration_is_named_by_its_own_content_id(key):
    statement_id, statement = _modelstate.statement(
        key, INSTANCE, MODEL_CID, MODEL_ROOT, TS
    )
    assert statement_id.startswith("urn:cid:bagb6qaq")  # RDFC-1 codec
    assert statement["@id"] == statement_id
    assert statement["@type"] == "CredentialRegistration"
    assert statement["registeredBy"] == key.did


class _DiskTensor:
    """A DiskTensor without needing a real safetensors header."""

    def __init__(self, path, offset, nbytes, name="t"):
        self.path, self.offset, self.nbytes, self.name = path, offset, nbytes, name


def test_the_disk_fold_is_the_algorithm_the_gpu_applies_to_vram(tmp_path):
    """Equality between modelCID and modelRoot only means something if the two
    sides fold identically. This pins the host side to the documented scheme:
    BLAKE3(LE32(N) || per-span BLAKE3 digests)."""
    payloads = [b"", b"alpha" * 1000, bytes(range(256)) * 700]
    path = tmp_path / "weights.bin"
    path.write_bytes(b"".join(payloads))
    offsets, running = [], 0
    for payload in payloads:
        offsets.append(running)
        running += len(payload)
    tensors = [
        _DiskTensor(path, off, len(p)) for off, p in zip(offsets, payloads)
    ]
    expected = blake3_digest(
        struct.pack("<I", len(payloads))
        + b"".join(blake3_digest(p) for p in payloads)
    )
    assert safetensors.model_root(tensors) == expected
    assert safetensors.model_cid(tensors) == f"urn:cid:{ids.raw_cid(expected)}"


def test_the_disk_fold_streams_rather_than_loading_whole_tensors(tmp_path):
    # A multi-gigabyte shard must not have to be resident to be hashed, so a
    # small chunk size must not change the answer.
    payload = bytes((i * 31 + 7) & 0xFF for i in range(300_000))
    path = tmp_path / "big.bin"
    path.write_bytes(payload)
    tensors = [_DiskTensor(path, 0, len(payload))]
    assert safetensors.model_root(tensors, chunk=4096) == safetensors.model_root(
        tensors
    )


def test_a_short_file_is_refused_rather_than_silently_padded(tmp_path):
    path = tmp_path / "truncated.bin"
    path.write_bytes(b"only-ten-b")
    tensors = [_DiskTensor(path, 0, 4096)]
    with pytest.raises(safetensors.SafetensorsError, match="short"):
        safetensors.model_root(tensors)


class _FakeKey:
    """Just enough host key to build a report."""

    did = "did:key:zDnaeHOST"
    verification_method = "did:key:zDnaeHOST#zDnaeHOST"

    def sign(self, data):
        return blake3_digest(data) + blake3_digest(data[::-1])


def _notary_stub(model_cid):
    from cuattest.notary import Notary

    notary = object.__new__(Notary)
    notary._loaded_model_cid = model_cid
    notary._model_state_statement = None
    notary._host_key = _FakeKey()
    return notary


def _measurement(instance_cid, vram_cid, measured_at):
    from cuattest.notary import Measurement

    return Measurement(
        digests="11" * 32, model_root="22" * 32, vram_cid=vram_cid,
        tensor_count=1, measured_at=measured_at, instance_cid=instance_cid,
    )


def _sign_twice(notary, vram_cids=("model-cid", "model-cid")):
    receipts = []
    for index, vram_cid in enumerate(vram_cids):
        receipt = {"manifest": {"statements": {}}}
        notary._attach_model_state_statement(
            receipt, _measurement("instance-cid", vram_cid, f"2026-09-16T19:00:0{index}Z")
        )
        receipts.append(receipt["manifest"]["statements"])
    return receipts


def test_restating_one_fact_does_not_grow_the_manifest():
    """Signing the same copy repeatedly must reuse the identical statement.

    The credential id is derived from the copy, the file and the measured root,
    but the registration wrapper also covers the timestamp -- so a report
    rebuilt per receipt would appear under a new wrapper id every time.
    """
    first, second = _sign_twice(_notary_stub(MODEL_CID))
    assert list(first) == list(second)
    assert first == second


def test_a_different_measured_root_gets_its_own_report():
    # Same copy, different bytes found in it: that is a new fact, not a restatement.
    first, second = _sign_twice(_notary_stub(MODEL_CID), ("model-cid", "other-cid"))
    assert list(first) != list(second)


def test_declaring_a_new_model_invalidates_the_previous_report():
    notary = _notary_stub(MODEL_CID)
    (before,) = _sign_twice(notary)[:1]
    notary.declare_loaded_model("urn:cid:" + ids.raw_cid(blake3_digest(b"reloaded")))
    after = {}
    receipt = {"manifest": {"statements": after}}
    notary._attach_model_state_statement(
        receipt, _measurement("instance-cid", "model-cid", "2026-09-16T19:00:09Z")
    )
    assert list(before) != list(after)


def test_no_declared_model_means_no_report():
    receipt = {"manifest": {"statements": {}}}
    _notary_stub(None)._attach_model_state_statement(
        receipt, _measurement("instance-cid", "model-cid", TS)
    )
    assert receipt["manifest"]["statements"] == {}


def test_owned_buffers_have_no_residency_identity_so_no_report():
    # instance_cid is empty for buffers this process owns; there is no
    # cross-process copy to report on, and inventing one would be a lie.
    receipt = {"manifest": {"statements": {}}}
    _notary_stub(MODEL_CID)._attach_model_state_statement(
        receipt, _measurement("", "model-cid", TS)
    )
    assert receipt["manifest"]["statements"] == {}


def test_collection_model_cid_is_signed_without_becoming_gpu_root(key):
    cid = "urn:cid:bagaachrapozywkvg2va7l3crrq57m74r6gsbff5nkzcyjg22f7cjd3rvagqq"
    assert ids.validate_model_cid(cid) == cid
    credential = _build(key, model_cid=cid)
    state = credential["credentialSubject"]["state"]
    assert state["modelCID"] == cid
    assert state["modelRoot"] == MODEL_ROOT and cid != MODEL_ROOT
    raw = base64.urlsafe_b64decode(credential["proof"]["jws"].split("..")[1] + "==")
    signature = utils.encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
    key._private_key.public_key().verify(signature, _modelstate.signing_input(
        credential["id"], key.did, INSTANCE, cid, MODEL_ROOT, TS), ec.ECDSA(hashes.SHA256()))
    notary = _notary_stub(None)
    notary.declare_loaded_model(cid)
    assert notary._loaded_model_cid == cid


@pytest.mark.parametrize("cid", [True, "urn:cid:bad", "urn:cid:b" + "a" * 60,
    "URN:CID:BAGAACHRAPOZYWKVG2VA7L3CRRQ57M74R6GSBFF5NKZCYJG22F7CJD3RVAGQQ"])
def test_bad_declared_model_cids_rejected(cid):
    with pytest.raises(ValueError):
        ids.validate_model_cid(cid)

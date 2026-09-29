# SPDX-License-Identifier: Apache-2.0
"""The service's durable identity and the attestation it issues.

The GPU's key lives for one process and cannot vouch for anything outside it.
This key is the opposite: it persists, and it signs the claim the GPU is unable
to make about itself -- which CUBIN and kernel source the service loaded. These
tests cover the parts that would fail quietly: a key that does not survive a
restart, a key other local users can read, and a signature over the wrong bytes.
"""

import base64
import os
import stat

import pytest

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

from cuattest import _hostkey, _identity

GPU_DID = "did:key:zDnaerpANTZMJydLKeSnf1JD37x81vhji35WoWsC46aVJ2SKr"
CUBIN = "urn:cid:bafkr4iaaw3ecdv7x67jvoaafdmkrejzf3swpl4noxs5jfdmkqzg4byxnt4"
KERNEL = "urn:cid:bafkr4iguslpyylnf37oqpa2g5nltcm2x5qlipxbaxwa2p7s72rerscsvdu"
DEVICE_UUID = "GPU-56e263b9-7678-70a4-2cab-7f3b6f0ddf20"
# CUDA reports the "GPU-" prefixed form; a urn:uuid names the bare UUID.
DEVICE_URN = "urn:uuid:56e263b9-7678-70a4-2cab-7f3b6f0ddf20"
TS = "2026-09-11T14:00:30Z"


@pytest.fixture
def key(tmp_path):
    return _hostkey.load_or_create(tmp_path / "host_key")


def test_key_is_created_private_and_survives_a_restart(tmp_path):
    path = tmp_path / "host_key"
    first = _hostkey.load_or_create(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert len(path.read_bytes()) == 32
    # The service identity is worthless if it changes every start.
    assert _hostkey.load_or_create(path).did == first.did
    assert first.did.startswith("did:key:zDna")  # P-256, multicodec 0x1200


def test_a_group_readable_key_is_refused(tmp_path):
    path = tmp_path / "host_key"
    _hostkey.load_or_create(path)
    path.chmod(0o640)
    # Anyone who can read it can issue attestations in this service's name,
    # so continuing would make the identity meaningless.
    with pytest.raises(_hostkey.HostKeyError, match="group or others"):
        _hostkey.load_or_create(path)


def test_a_truncated_key_is_refused_rather_than_padded(tmp_path):
    path = tmp_path / "host_key"
    path.write_bytes(b"\x01" * 31)
    path.chmod(0o600)
    with pytest.raises(_hostkey.HostKeyError, match="32"):
        _hostkey.load_or_create(path)


def test_env_override_selects_the_key_path(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere" / "key"
    monkeypatch.setenv("CUATTEST_HOST_KEY", str(target))
    assert _hostkey.default_key_path() == target


def test_signature_is_raw_r_s_not_der(key):
    # The kernel emits raw r||s and expect.py decodes that form; a DER
    # signature here would be silently unverifiable by the same path.
    assert len(key.sign(b"message")) == 64


def test_attestation_binds_the_gpu_session_to_the_loaded_code(key):
    credential = _identity.build(key, GPU_DID, CUBIN, KERNEL, DEVICE_UUID, TS)
    subject = credential["credentialSubject"]
    assert subject["id"] == GPU_DID           # the GPU is the subject
    assert credential["issuer"] == key.did    # the service is the issuer
    assert subject["executedOn"] == key.did
    assert subject["identity"] == {
        "type": _identity.IDENTITY_TYPE,
        "cubin": CUBIN,
        "device": DEVICE_URN,
        "kernel": KERNEL,
    }
    # The attester kind describes the code/hardware binding, not the GPU DID,
    # so it belongs on the identity node rather than on the subject.
    assert "type" not in subject
    assert subject["identity"]["type"] == "CudaAttesterV1"
    # The NVIDIA prefix would make this a malformed urn:uuid.
    assert "GPU-" not in subject["identity"]["device"]
    assert credential["type"] == ["VerifiableCredential", "IdentityAttestation"]
    # VC 2.0 uses validFrom; issuanceDate was removed and must not reappear.
    assert "issuanceDate" not in credential
    assert credential["validFrom"] == TS
    assert credential["proof"]["type"] == "EcdsaSecp256r1Signature2019"


def test_attestation_proof_verifies_against_the_issuer_key(key):
    credential = _identity.build(key, GPU_DID, CUBIN, KERNEL, DEVICE_UUID, TS)
    raw = base64.urlsafe_b64decode(credential["proof"]["jws"].split("..")[1] + "==")
    signing_input = _identity.signing_input(
        credential["id"], GPU_DID, key.did, CUBIN, KERNEL,
        DEVICE_URN, TS,
    )
    key._private_key.public_key().verify(
        utils.encode_dss_signature(
            int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big")
        ),
        signing_input,
        ec.ECDSA(hashes.SHA256()),
    )


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("gpu_did", "did:key:zDnaeOTHER"),
        ("cubin", "urn:cid:bafkr4iaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        ("kernel", "urn:cid:bafkr4ibbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ("device", "urn:uuid:00000000-0000-0000-0000-000000000000"),
        ("timestamp", "2026-09-11T14:00:31Z"),
    ],
)
def test_every_attested_field_is_covered_by_the_signature(key, field, replacement):
    args = {
        "gpu_did": GPU_DID, "cubin": CUBIN, "kernel": KERNEL,
        "device": DEVICE_URN, "timestamp": TS,
    }
    credential = _identity.build(key, GPU_DID, CUBIN, KERNEL, DEVICE_UUID, TS)
    genuine = _identity.signing_input(
        credential["id"], args["gpu_did"], key.did, args["cubin"],
        args["kernel"], args["device"], args["timestamp"],
    )
    args[field] = replacement
    tampered = _identity.signing_input(
        credential["id"], args["gpu_did"], key.did, args["cubin"],
        args["kernel"], args["device"], args["timestamp"],
    )
    assert tampered != genuine


def test_credential_id_is_derived_not_random(key):
    # A random uuid would mint a new credential id on every receipt for facts
    # that never change, so the manifest would grow identifiers pointlessly.
    first = _identity.build(key, GPU_DID, CUBIN, KERNEL, DEVICE_UUID, TS)
    second = _identity.build(key, GPU_DID, CUBIN, KERNEL, DEVICE_UUID, TS)
    assert first["id"] == second["id"]
    other = _identity.build(key, GPU_DID, CUBIN, KERNEL, DEVICE_UUID, "2026-01-01T00:00:00Z")
    assert other["id"] == first["id"]  # timestamp is not part of the identity
    changed = _identity.build(
        key, "did:key:zDnaeOTHER", CUBIN, KERNEL, DEVICE_UUID, TS
    )
    assert changed["id"] != first["id"]


def test_registration_wraps_the_credential_under_its_own_content_id(key):
    credential = _identity.build(key, GPU_DID, CUBIN, KERNEL, DEVICE_UUID, TS)
    statement_id, statement = _identity.registration(credential, key.did, TS)
    assert statement_id.startswith("urn:cid:bagb6qaq")  # RDFC-1 codec
    assert statement["@id"] == statement_id
    assert statement["@type"] == "CredentialRegistration"
    assert statement["credential"] is credential
    assert statement["registeredBy"] == key.did


def test_host_identity_can_be_disabled_without_cryptography(monkeypatch):
    # The opt-out exists so a lean deployment can skip the service identity
    # deliberately, rather than losing it to a missing import.
    monkeypatch.setenv("CUATTEST_HOST_IDENTITY", "0")
    assert os.environ["CUATTEST_HOST_IDENTITY"].lower() in {"0", "false", "no"}

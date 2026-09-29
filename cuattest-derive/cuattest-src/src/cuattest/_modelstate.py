# SPDX-License-Identifier: Apache-2.0
"""The host's own report about a model it loaded into GPU memory.

The GPU signs what it found in VRAM. The host binds that measurement to the
model asset CID supplied by its caller. In an SDK integration, `modelCID` is
the same file or collection CID used as the Model input to the signed
computation. The GPU never sees the source asset, so this association is a
host claim issued under the service's durable key.

`modelRoot` identifies the GPU-measured tensor bytes. It is separate from
`modelCID`, which may identify an entire model package including configuration
and tokenizers. Equality is not required and a signed association does not
prove faithful loading. A caller can still use `safetensors.model_cid` for a
raw source tensor root; only in that case, with identical bytes and ordering,
can it compare that root directly with the GPU measurement.

Because `modelRoot` comes from the GPU, this credential cannot be complete at
load time; it is issued once the first measurement of that copy exists. What
the host contributes -- instanceID and modelCID -- is fixed at load.

The detached-JWS construction matches the kernel's and _identity's exactly
(ES256, b64=false, payload sha256(proof options) || sha256(document), raw r||s),
so one verification path covers all three.
"""

from __future__ import annotations

import base64
import hashlib
import uuid as uuidlib

from ._hosthash import blake3_digest
from ._identity import RDF_TYPE, SEC, TERMS, VC_TERMS, XSD, registration
from ._statements import CREDENTIAL_TYPE, JWS_PROTECTED_HEADER

#: The host's report about a resident copy. Deliberately distinct from the
#: kernel's ``gpuModelTensorsStateV1``: same subject, different reporter.
STATE_TYPE = "gpuModelTensorsV1"


def _credential_document(
    credential_id: str,
    host_did: str,
    instance_urn: str,
    model_cid: str,
    model_root_urn: str,
    timestamp: str,
) -> str:
    """The credential's own triples, without its proof.

    Mirrors the kernel's StateAttestation shape -- a `state` node hanging off
    the credential subject -- so a verifier walks host and GPU state reports
    the same way.
    """
    return "".join(
        sorted(
            (
                f"<{credential_id}> <{RDF_TYPE}> <{VC_TERMS}VerifiableCredential> .\n"
                f"<{credential_id}> <{RDF_TYPE}> <{TERMS}{CREDENTIAL_TYPE}> .\n"
                f"<{credential_id}> <{VC_TERMS}credentialSubject> <{host_did}> .\n"
                f"<{credential_id}> <{VC_TERMS}issuer> <{host_did}> .\n"
                f'<{credential_id}> <{VC_TERMS}validFrom> "{timestamp}"'
                f"^^<{XSD}dateTime> .\n"
                f"<{host_did}> <{TERMS}state> _:b0 .\n"
                f"_:b0 <{TERMS}instanceID> <{instance_urn}> .\n"
                f"_:b0 <{TERMS}modelCID> <{model_cid}> .\n"
                f"_:b0 <{TERMS}modelRoot> <{model_root_urn}> .\n"
                f'_:b0 <{TERMS}reportedBy> "{host_did}" .\n'
                f'_:b0 <{TERMS}stateType> "{STATE_TYPE}" .\n'
            ).splitlines(keepends=True)
        )
    )


def _proof_options(host_did: str, timestamp: str) -> str:
    return "".join(
        sorted(
            (
                f'_:c14n0 <http://purl.org/dc/terms/created> "{timestamp}"'
                f"^^<{XSD}dateTime> .\n"
                f"_:c14n0 <{RDF_TYPE}> <{SEC}EcdsaSecp256r1Signature2019> .\n"
                f"_:c14n0 <{SEC}proofPurpose> <{SEC}assertionMethod> .\n"
                f"_:c14n0 <{SEC}verificationMethod> "
                f"<{host_did}#{host_did[8:]}> .\n"
            ).splitlines(keepends=True)
        )
    )


def signing_input(
    credential_id: str,
    host_did: str,
    instance_urn: str,
    model_cid: str,
    model_root_urn: str,
    timestamp: str,
) -> bytes:
    """The exact bytes the detached JWS covers.

    RFC 7797 b64=false leaves the 64-byte payload unencoded; this suite
    defines it as proof-options hash || document hash, as the kernel does.
    """
    document = _credential_document(
        credential_id, host_did, instance_urn, model_cid, model_root_urn, timestamp
    )
    return (
        JWS_PROTECTED_HEADER.encode()
        + b"."
        + hashlib.sha256(_proof_options(host_did, timestamp).encode()).digest()
        + hashlib.sha256(document.encode()).digest()
    )


def _credential_uuid(
    host_did: str, instance_urn: str, model_cid: str, model_root_urn: str
) -> str:
    """A stable identifier for this report.

    Derived, not random, for the same reason the kernel's is: re-reporting the
    same facts about the same copy yields the same id, so a manifest does not
    grow a new identifier every receipt. The predecessor link is deliberately
    absent -- this credential has no chain; it restates a fixed fact about one
    resident copy.
    """
    seed = blake3_digest(
        f"{host_did}\n{instance_urn}\n{model_cid}\n{model_root_urn}".encode()
    )
    return str(uuidlib.UUID(bytes=seed[:16], version=4))


def build(
    host_key,
    instance_urn: str,
    model_cid: str,
    model_root_urn: str,
    timestamp: str,
) -> dict:
    """Issue the signed host report for one resident copy."""
    host_did = host_key.did
    credential_id = f"urn:uuid:{_credential_uuid(host_did, instance_urn, model_cid, model_root_urn)}"
    raw_signature = host_key.sign(
        signing_input(
            credential_id, host_did, instance_urn, model_cid, model_root_urn, timestamp
        )
    )
    jws = (
        JWS_PROTECTED_HEADER
        + ".."
        + base64.urlsafe_b64encode(raw_signature).rstrip(b"=").decode()
    )
    return {
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://w3id.org/security/v2",
        ],
        "id": credential_id,
        "type": ["VerifiableCredential", CREDENTIAL_TYPE],
        "credentialSubject": {
            "id": host_did,
            "state": {
                "stateType": STATE_TYPE,
                "instanceID": instance_urn,
                "modelRoot": model_root_urn,
                "modelCID": model_cid,
                "reportedBy": host_did,
            },
        },
        "issuer": host_did,
        "proof": {
            "type": "EcdsaSecp256r1Signature2019",
            "proofPurpose": "assertionMethod",
            "verificationMethod": host_key.verification_method,
            "created": timestamp,
            "jws": jws,
        },
        "validFrom": timestamp,
    }


def statement(
    host_key,
    instance_urn: str,
    model_cid: str,
    model_root_urn: str,
    timestamp: str,
) -> tuple[str, dict]:
    """Build the report and wrap it as a CredentialRegistration."""
    credential = build(host_key, instance_urn, model_cid, model_root_urn, timestamp)
    return registration(credential, host_key.did, timestamp)


__all__ = ["STATE_TYPE", "build", "signing_input", "statement"]

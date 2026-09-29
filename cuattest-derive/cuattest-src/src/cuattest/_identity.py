# SPDX-License-Identifier: Apache-2.0
"""The service's IdentityAttestation over a GPU session.

Once keygen establishes a session's GPU DID, the service issues one credential
binding that DID to the code and hardware it loaded: the CUBIN that runs, the
kernel source it was built from, and the device it runs on. The credential is
signed by the service's durable key, kept for the life of the session, and
added to every manifest the GPU produces.

This is the counterpart to a gap in the measurement document: cubinCID and
kernelCID are values the service computes and hands the kernel, which the GPU
signs but cannot check. Issuing them under the service's own durable identity
does not make them verifiable -- a compromised service can still name the wrong
CUBIN -- but it makes the claim attributable to a key that can be pinned and
revoked, instead of an anonymous assertion riding inside a GPU signature.

The detached-JWS construction matches the kernel's exactly (ES256, b64=false,
payload sha256(proof options) || sha256(document), raw r||s), so one
verification path covers both the service's credentials and the GPU's.
"""

from __future__ import annotations

import base64
import hashlib
import re
import uuid as uuidlib

from . import ids
from ._hosthash import blake3_digest
from ._statements import JWS_PROTECTED_HEADER, STATEMENT_CONTEXT

TERMS = "https://eqtylab.io/terms/"
VC_TERMS = "https://www.w3.org/2018/credentials#"
SEC = "https://w3id.org/security#"
XSD = "http://www.w3.org/2001/XMLSchema#"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
# The attester kind, carried on the nested `identity` node rather than on
# the subject: it describes the code/hardware binding, not the GPU DID.
IDENTITY_TYPE = "CudaAttesterV1"

_BLANK_NODE = re.compile(r"_:[A-Za-z0-9]+")


def _canonicalize(nquads: str, nodes: tuple[str, ...]) -> str:
    """RDFC-1.0 sections 4.4/4.6 for a closed schema with distinct nodes.

    The same specialization _statements uses for the credential wrapper, but
    over a caller-supplied node set rather than a fixed three. Every node in
    these schemas carries a different predicate set, so first-degree hashes
    cannot coincide; a collision must fail closed rather than fall back to
    role ordering. Arbitrary RDF would need N-degree canonicalization and must
    not be routed through here.
    """
    quads = nquads.splitlines(keepends=True)
    first_degree = {}
    for node in nodes:
        incident = [
            _BLANK_NODE.sub(
                lambda match, node=node: "_:a" if match[0] == node else "_:z",
                quad,
            )
            for quad in quads
            if node in _BLANK_NODE.findall(quad)
        ]
        if not incident:
            raise ValueError(f"{node} appears in no quad")
        first_degree[node] = hashlib.sha256(
            "".join(sorted(incident)).encode()
        ).digest()
    if len(set(first_degree.values())) != len(nodes):
        raise ValueError("blank-node first-degree hash collision")
    labels = {
        node: f"_:c14n{index}"
        for index, node in enumerate(
            sorted(first_degree, key=first_degree.__getitem__)
        )
    }
    return "".join(
        sorted(_BLANK_NODE.sub(lambda m: labels[m[0]], quad) for quad in quads)
    )


def _credential_document(
    credential_id: str,
    gpu_did: str,
    host_did: str,
    cubin_cid: str,
    kernel_cid: str,
    device_urn: str,
    timestamp: str,
) -> str:
    """The credential's own triples, without its proof."""
    return "".join(
        sorted(
            (
                f"<{credential_id}> <{RDF_TYPE}> <{VC_TERMS}VerifiableCredential> .\n"
                f"<{credential_id}> <{RDF_TYPE}> <{TERMS}IdentityAttestation> .\n"
                f"<{credential_id}> <{VC_TERMS}credentialSubject> <{gpu_did}> .\n"
                f"<{credential_id}> <{VC_TERMS}issuer> <{host_did}> .\n"
                f'<{credential_id}> <{VC_TERMS}validFrom> "{timestamp}"'
                f"^^<{XSD}dateTime> .\n"
                # Informational: the issuer already identifies the service.
                f"<{gpu_did}> <{TERMS}executedOn> <{host_did}> .\n"
                f"<{gpu_did}> <{TERMS}identity> _:b0 .\n"
                f"_:b0 <{RDF_TYPE}> <{TERMS}{IDENTITY_TYPE}> .\n"
                f"_:b0 <{TERMS}cubin> <{cubin_cid}> .\n"
                f"_:b0 <{TERMS}device> <{device_urn}> .\n"
                f"_:b0 <{TERMS}kernel> <{kernel_cid}> .\n"
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
    gpu_did: str,
    host_did: str,
    cubin_cid: str,
    kernel_cid: str,
    device_urn: str,
    timestamp: str,
) -> bytes:
    """The exact bytes the detached JWS covers.

    RFC 7797 b64=false leaves the 64-byte payload unencoded; this suite
    defines it as proof-options hash || document hash, as the kernel does.
    """
    document = _credential_document(
        credential_id, gpu_did, host_did, cubin_cid, kernel_cid,
        device_urn, timestamp,
    )
    return (
        JWS_PROTECTED_HEADER.encode()
        + b"."
        + hashlib.sha256(_proof_options(host_did, timestamp).encode()).digest()
        + hashlib.sha256(document.encode()).digest()
    )


def _credential_uuid(gpu_did: str, cubin_cid: str, kernel_cid: str) -> str:
    """A stable identifier for this session's attestation.

    Derived rather than random, so re-issuing the same claim about the same
    session yields the same credential id and the manifest does not grow a new
    identifier per receipt.
    """
    seed = blake3_digest(f"{gpu_did}\n{cubin_cid}\n{kernel_cid}".encode())
    return str(uuidlib.UUID(bytes=seed[:16], version=4))


def build(
    host_key,
    gpu_did: str,
    cubin_cid: str,
    kernel_cid: str,
    device_uuid: str,
    timestamp: str,
) -> dict:
    """Issue the signed IdentityAttestation for one GPU session."""
    # CUDA reports device UUIDs with NVIDIA's "GPU-" prefix, which the rest of
    # this project keeps as the routing key for a physical device. A urn:uuid
    # names a bare RFC 4122 UUID, though, so the prefix is dropped here -- at
    # the one place the URN is formatted -- rather than at the source.
    device_urn = f"urn:uuid:{device_uuid.removeprefix('GPU-')}"
    credential_id = f"urn:uuid:{_credential_uuid(gpu_did, cubin_cid, kernel_cid)}"
    raw_signature = host_key.sign(
        signing_input(
            credential_id, gpu_did, host_key.did, cubin_cid, kernel_cid,
            device_urn, timestamp,
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
        "type": ["VerifiableCredential", "IdentityAttestation"],
        "credentialSubject": {
            "id": gpu_did,
            "executedOn": host_key.did,
            "identity": {
                "type": IDENTITY_TYPE,
                "cubin": cubin_cid,
                "device": device_urn,
                "kernel": kernel_cid,
            },
        },
        "issuer": host_key.did,
        "proof": {
            "type": "EcdsaSecp256r1Signature2019",
            "proofPurpose": "assertionMethod",
            "verificationMethod": host_key.verification_method,
            "created": timestamp,
            "jws": jws,
        },
        "validFrom": timestamp,
    }


def registration(credential: dict, host_did: str, timestamp: str) -> tuple[str, dict]:
    """Wrap the credential as a CredentialRegistration statement.

    The manifest's statement map stays one shape: every entry is keyed by its
    own RDFC content id, so a verifier walks them uniformly regardless of who
    issued them.
    """
    credential_id = credential["id"]
    nquads = "".join(
        sorted(
            (
                f"_:b0 <{RDF_TYPE}> <{TERMS}CredentialRegistration> .\n"
                f"_:b0 <{TERMS}credential> <{credential_id}> .\n"
                f'_:b0 <{TERMS}registeredBy> "{host_did}" .\n'
                f'_:b0 <{TERMS}timestamp> "{timestamp}"^^<{XSD}dateTime> .\n'
            ).splitlines(keepends=True)
        )
    )
    digest = blake3_digest(_canonicalize(nquads, ("_:b0",)).encode())
    statement_id = f"urn:cid:{ids.rdfc_cid(digest)}"
    return statement_id, {
        "@context": STATEMENT_CONTEXT,
        "@id": statement_id,
        "@type": "CredentialRegistration",
        "credential": credential,
        "registeredBy": host_did,
        "timestamp": timestamp,
    }

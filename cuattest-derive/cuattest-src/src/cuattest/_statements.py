# SPDX-License-Identifier: Apache-2.0
"""Strict host validation for the deterministic EQTY statement manifest."""

from __future__ import annotations

import base64
import hashlib
import re
import uuid
from collections.abc import Callable

from . import ids
from ._hosthash import blake3_digest

STATEMENT_CONTEXT = (
    "urn:cid:bafkr4icploa577ziqnb57jlpoj7l2hi5kgt3knxpdtunlttjd3q33zeqpy"
)
VC_CONTEXT = [
    "https://www.w3.org/ns/credentials/v2",
    "https://w3id.org/security/v2",
]
JWS_PROTECTED_HEADER = "eyJhbGciOiJFUzI1NiIsImNyaXQiOlsiYjY0Il0sImI2NCI6ZmFsc2V9"
JWS_PREFIX = JWS_PROTECTED_HEADER + ".."

# The GPU's own attested state. The host issues a separate
# "gpuModelTensorsV1" report about the same copy; see _modelstate.
STATE_TYPE = "gpuModelTensorsStateV1"
CREDENTIAL_TYPE = "StateAttestation"
MANIFEST_VERSION = "3"

_CREDENTIAL_REGISTRATION_FIELDS = frozenset(
    {"@context", "@id", "@type", "credential", "registeredBy", "timestamp"}
)
_CREDENTIAL_FIELDS = frozenset(
    {
        "@context",
        "credentialSubject",
        "id",
        "issuer",
        "proof",
        "type",
        "validFrom",
    }
)
_SUBJECT_FIELDS = frozenset({"id", "state"})
_STATE_PAYLOAD_FIELDS = frozenset(
    {"stateType", "instanceID", "modelRoot", "previousStateCredential"}
)
_PROOF_FIELDS = frozenset(
    {"created", "jws", "proofPurpose", "type", "verificationMethod"}
)


def _require_exact_object(value, fields: frozenset[str], label: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    if frozenset(value) != fields:
        raise ValueError(f"{label} does not match its exact schema")
    return value


def _statement_urn(canonical_nquads: str) -> str:
    digest = blake3_digest(canonical_nquads.encode())
    return f"urn:cid:{ids.rdfc_cid(digest)}"


_CREDENTIAL_BLANK_NODE = re.compile(r"_:b[0-3]\b")


def _credential_quads(
    credential_ref: str,
    state_ref: str,
    issuer: str,
    timestamp: str,
    instance_urn: str,
    model_urn: str,
    previous_uuid: str | None,
) -> list[str]:
    """The StateAttestation's own triples, without its proof.

    `credential_ref` and `state_ref` are already-serialized N-Quads terms --
    either an angle-bracketed IRI or a blank-node label -- because these same
    triples are needed twice with different naming. The signing input names the
    credential by its final ``urn:uuid``; the id derivation cannot, since that
    is the value it is deriving, and uses a blank node instead.

    The subject is the GPU itself: this credential says what state that GPU
    observed, so credentialSubject and issuer are the same DID. RDF has no
    null, so a genesis credential simply carries no previousStateCredential
    triple -- which is also what JSON-LD expansion does with a null-valued key,
    making the emitted "previousStateCredential":null and the absent triple the
    same document.
    """
    quads = [
        f"<{issuer}> <https://eqtylab.io/terms/state> {state_ref} .\n",
        f"{credential_ref} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        f"<https://eqtylab.io/terms/{CREDENTIAL_TYPE}> .\n",
        f"{credential_ref} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<https://www.w3.org/2018/credentials#VerifiableCredential> .\n",
        f"{credential_ref} <https://www.w3.org/2018/credentials#credentialSubject> "
        f"<{issuer}> .\n",
        f"{credential_ref} <https://www.w3.org/2018/credentials#issuer> <{issuer}> .\n",
        f"{credential_ref} <https://www.w3.org/2018/credentials#validFrom> "
        f'"{timestamp}"^^<http://www.w3.org/2001/XMLSchema#dateTime> .\n',
        f"{state_ref} <https://eqtylab.io/terms/instanceID> <{instance_urn}> .\n",
        f"{state_ref} <https://eqtylab.io/terms/modelRoot> <{model_urn}> .\n",
        f'{state_ref} <https://eqtylab.io/terms/stateType> "{STATE_TYPE}" .\n',
    ]
    if previous_uuid is not None:
        quads.append(
            f"{state_ref} <https://eqtylab.io/terms/previousStateCredential> "
            f"<{previous_uuid}> .\n"
        )
    return quads


def _canonicalize_two_node_nquads(nquads: str) -> str:
    """RDFC-1.0 4.4/4.6 for the two-blank-node credential id preimage.

    The credential node and the nested `state` node carry disjoint predicates,
    so their first-degree hashes cannot coincide; a collision fails closed
    rather than falling back to role ordering. Which node wins c14n0 is
    value-dependent -- adding previousStateCredential can flip it -- so neither
    the labels nor the final quad order may be assumed.
    """
    quads = nquads.splitlines(keepends=True)
    first_degree = {}
    for node in ("_:b0", "_:b1"):
        incident = [
            _CREDENTIAL_BLANK_NODE.sub(
                lambda match, node=node: "_:a" if match[0] == node else "_:z", quad
            )
            for quad in quads
            if node in _CREDENTIAL_BLANK_NODE.findall(quad)
        ]
        if not incident:
            raise ValueError(f"{node} appears in no quad")
        first_degree[node] = hashlib.sha256("".join(sorted(incident)).encode()).digest()
    if len(set(first_degree.values())) != 2:
        raise ValueError("credential blank-node first-degree hash collision")
    labels = {
        node: f"_:c14n{index}"
        for index, node in enumerate(sorted(first_degree, key=first_degree.__getitem__))
    }
    return "".join(
        sorted(
            _CREDENTIAL_BLANK_NODE.sub(lambda m: labels[m[0]], quad) for quad in quads
        )
    )


def _credential_id_preimage(
    issuer: str,
    timestamp: str,
    instance_urn: str,
    model_urn: str,
    previous_uuid: str | None,
) -> str:
    """The content this credential's UUID is derived from.

    Everything the credential asserts, minus its own name. Deriving the id from
    this rather than drawing it at random is what lets a verifier recompute it:
    a receipt whose `state` was edited after signing no longer hashes to the
    id inside its own signed document. It is also what makes the kernel's two
    receipt passes agree, since the device has no entropy source to reuse.
    """
    return _canonicalize_two_node_nquads(
        "".join(
            _credential_quads(
                "_:b0", "_:b1", issuer, timestamp, instance_urn, model_urn,
                previous_uuid,
            )
        )
    )


def _credential_uuid(model_hash: str, subject_urn: str) -> str:
    uuid_bytes = bytearray(
        hashlib.sha256(bytes.fromhex(model_hash) + subject_urn.encode()).digest()
    )
    uuid_bytes[6] = 0x40 | (uuid_bytes[6] & 0x0F)
    uuid_bytes[8] = 0x80 | (uuid_bytes[8] & 0x3F)
    return str(uuid.UUID(bytes=bytes(uuid_bytes[:16])))


def _state_credential_id(
    model_hash: str,
    issuer: str,
    timestamp: str,
    instance_urn: str,
    model_urn: str,
    previous_uuid: str | None,
) -> str:
    """This measurement's credential id, as both the kernel and host derive it."""
    preimage_urn = _statement_urn(
        _credential_id_preimage(
            issuer, timestamp, instance_urn, model_urn, previous_uuid
        )
    )
    return f"urn:uuid:{_credential_uuid(model_hash, preimage_urn)}"


def _canonicalize_credential_nquads(nquads: str) -> str:
    """RDFC-1.0 for the closed credential-wrapper schema, not arbitrary RDF."""
    quads = nquads.splitlines(keepends=True)
    first_degree = {}
    for node in ("_:b0", "_:b1", "_:b2", "_:b3"):
        # RDFC 4.6: hash all incident quads with this node replaced by _:a
        # and every other blank node (including graph names) by _:z. Validated
        # field alphabets cannot contain these tokens inside IRIs or literals.
        incident = [
            _CREDENTIAL_BLANK_NODE.sub(
                lambda match, node=node: "_:a" if match[0] == node else "_:z", quad
            )
            for quad in quads
            if node in _CREDENTIAL_BLANK_NODE.findall(quad)
        ]
        if not incident:
            raise ValueError(f"{node} appears in no quad")
        first_degree[node] = hashlib.sha256("".join(sorted(incident)).encode()).digest()

    # The proof subject, proof graph, registration, and state node have
    # distinct incident quad shapes. Their first-degree inputs cannot coincide
    # in this schema; a SHA-256 collision must fail closed, never fall back to
    # role ordering. No N-degree search is needed for these four structurally
    # unique nodes.
    if len(set(first_degree.values())) != 4:
        raise ValueError("credential blank-node first-degree hash collision")
    labels = {
        node: f"_:c14n{index}"
        for index, node in enumerate(sorted(first_degree, key=first_degree.__getitem__))
    }
    return "".join(
        sorted(
            _CREDENTIAL_BLANK_NODE.sub(lambda match: labels[match[0]], quad)
            for quad in quads
        )
    )


def _credential_registration_nquads(
    credential_id: str,
    issuer: str,
    timestamp: str,
    instance_urn: str,
    model_urn: str,
    previous_uuid: str | None,
    jws: str,
) -> str:
    # b0/b1/b2/b3 are provisional roles, never preassigned canonical labels:
    # the signature, UUID, issuer, timestamp, and state can all change their
    # order.
    quads = _credential_quads(
        f"<{credential_id}>", "_:b3", issuer, timestamp, instance_urn, model_urn,
        previous_uuid,
    )
    quads += [
        f"<{credential_id}> <https://w3id.org/security#proof> _:b1 .\n",
        f'_:b0 <http://purl.org/dc/terms/created> "{timestamp}"'
        "^^<http://www.w3.org/2001/XMLSchema#dateTime> _:b1 .\n",
        "_:b0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<https://w3id.org/security#EcdsaSecp256r1Signature2019> _:b1 .\n",
        f'_:b0 <https://w3id.org/security#jws> "{jws}" _:b1 .\n',
        "_:b0 <https://w3id.org/security#proofPurpose> "
        "<https://w3id.org/security#assertionMethod> _:b1 .\n",
        "_:b0 <https://w3id.org/security#verificationMethod> "
        f"<{issuer}#{issuer[8:]}> _:b1 .\n",
        "_:b2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<https://eqtylab.io/terms/CredentialRegistration> .\n",
        f"_:b2 <https://eqtylab.io/terms/credential> <{credential_id}> .\n",
        f'_:b2 <https://eqtylab.io/terms/registeredBy> "{issuer}" .\n',
        f'_:b2 <https://eqtylab.io/terms/timestamp> "{timestamp}"'
        "^^<http://www.w3.org/2001/XMLSchema#dateTime> .\n",
    ]
    return "".join(quads)


def _credential_registration_id(
    credential_id: str,
    issuer: str,
    timestamp: str,
    instance_urn: str,
    model_urn: str,
    previous_uuid: str | None,
    jws: str,
) -> str:
    nquads = _credential_registration_nquads(
        credential_id, issuer, timestamp, instance_urn, model_urn, previous_uuid, jws
    )
    return _statement_urn(_canonicalize_credential_nquads(nquads))


def _credential_signing_input(
    credential_id: str,
    issuer: str,
    timestamp: str,
    instance_urn: str,
    model_urn: str,
    previous_uuid: str | None,
) -> bytes:
    """Reconstruct the exact detached-JWS input assembled by the CUDA kernel.

    The measured state is inside these bytes, not merely referenced by them, so
    the GPU signature covers instanceID, modelRoot, and the chain link
    directly rather than only through the derived credential id.
    """
    credential_document = "".join(
        sorted(
            _credential_quads(
                f"<{credential_id}>", "_:c14n0", issuer, timestamp, instance_urn,
                model_urn, previous_uuid,
            )
        )
    ).encode()
    proof_options = (
        f'_:c14n0 <http://purl.org/dc/terms/created> "{timestamp}"'
        "^^<http://www.w3.org/2001/XMLSchema#dateTime> .\n"
        "_:c14n0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
        "<https://w3id.org/security#EcdsaSecp256r1Signature2019> .\n"
        "_:c14n0 <https://w3id.org/security#proofPurpose> "
        "<https://w3id.org/security#assertionMethod> .\n"
        "_:c14n0 <https://w3id.org/security#verificationMethod> "
        f"<{issuer}#{issuer[8:]}> .\n"
    ).encode()
    # RFC 7797 b64=false leaves the 64-byte payload unencoded. The project's
    # fixed suite defines that payload as proof-options hash || document hash.
    return (
        JWS_PROTECTED_HEADER.encode()
        + b"."
        + hashlib.sha256(proof_options).digest()
        + hashlib.sha256(credential_document).digest()
    )


def _validate_jws(value) -> tuple[str, bytes]:
    if type(value) is not str or not value.startswith(JWS_PREFIX):
        raise ValueError("credential proof has an invalid JWS header")
    signature = value[len(JWS_PREFIX) :]
    if re.fullmatch(r"[A-Za-z0-9_-]{86}", signature) is None:
        raise ValueError("credential proof has an invalid JWS signature encoding")
    try:
        decoded = base64.urlsafe_b64decode(signature + "==")
    except ValueError as error:
        raise ValueError("credential proof has an invalid JWS signature") from error
    if len(decoded) != 64:
        raise ValueError("credential proof does not contain a P-256 signature")
    canonical = base64.urlsafe_b64encode(decoded).rstrip(b"=").decode("ascii")
    if signature != canonical:
        # With a 64-byte value, four low bits in the last base64url character
        # are padding. Some decoders ignore non-zero pad bits, which would give
        # one signature several text spellings and therefore several wrapper
        # CIDs. The graph is content-addressed, so require its unique spelling.
        raise ValueError("credential proof has a non-canonical JWS signature")
    return value, decoded


def _validate_state_attestation(
    map_key: str,
    statement,
    model_hash: str,
    issuer: str,
    timestamp: str,
    model_urn: str,
    expected_instance_urn: str | None,
    verify_credential_signature: Callable[[bytes, bytes], None] | None,
) -> None:
    """Re-derive the whole statement from the measurement and require equality.

    Both ids are rebuilt here from the same canonical N-Quads the CUDA kernel
    builds, so a valid CUBIN cannot attach a stronger assertion beside the
    narrow signed measurement document, nor rename a statement without the
    content id moving with it.
    """
    outer = _require_exact_object(
        statement, _CREDENTIAL_REGISTRATION_FIELDS, "CredentialRegistration"
    )
    credential = _require_exact_object(
        outer["credential"], _CREDENTIAL_FIELDS, "credential"
    )
    subject = _require_exact_object(
        credential["credentialSubject"], _SUBJECT_FIELDS, "credentialSubject"
    )
    proof = _require_exact_object(credential["proof"], _PROOF_FIELDS, "proof")
    payload = _require_exact_object(
        subject["state"], _STATE_PAYLOAD_FIELDS, "credentialSubject state"
    )

    if subject["id"] != issuer:
        raise ValueError("StateAttestation subject is not the issuing GPU")
    if payload["stateType"] != STATE_TYPE:
        raise ValueError(f"unsupported stateType {payload['stateType']!r}")
    if payload["modelRoot"] != model_urn:
        raise ValueError("StateAttestation modelRoot contradicts the measurement")

    instance_urn = payload["instanceID"]
    if not isinstance(instance_urn, str) or not instance_urn.startswith("urn:cid:"):
        raise ValueError("StateAttestation instanceID must be a urn:cid")
    # The kernel receives instanceID from this service; when the caller knows
    # which value it supplied, a substituted-but-self-consistent one is caught.
    if expected_instance_urn is not None and instance_urn != expected_instance_urn:
        raise ValueError("StateAttestation instanceID is not the value supplied")

    previous_uuid = payload["previousStateCredential"]
    if previous_uuid is not None and (
        not isinstance(previous_uuid, str) or not previous_uuid.startswith("urn:uuid:")
    ):
        raise ValueError("previousStateCredential must be null or a urn:uuid")

    expected_credential_id = _state_credential_id(
        model_hash, issuer, timestamp, instance_urn, model_urn, previous_uuid
    )
    jws, raw_signature = _validate_jws(proof["jws"])
    expected_statement_id = _credential_registration_id(
        expected_credential_id, issuer, timestamp, instance_urn, model_urn,
        previous_uuid, jws,
    )
    expected = {
        "@context": STATEMENT_CONTEXT,
        "@id": expected_statement_id,
        "@type": "CredentialRegistration",
        "credential": {
            "@context": VC_CONTEXT,
            "id": expected_credential_id,
            "type": ["VerifiableCredential", CREDENTIAL_TYPE],
            "credentialSubject": {
                "id": issuer,
                "state": {
                    "stateType": STATE_TYPE,
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
    if map_key != expected_statement_id or outer != expected:
        raise ValueError("StateAttestation contradicts the measurement")
    if verify_credential_signature is not None:
        signing_input = _credential_signing_input(
            expected_credential_id, issuer, timestamp, instance_urn, model_urn,
            previous_uuid,
        )
        verify_credential_signature(signing_input, raw_signature)


def _validate_statement_graph(
    manifest,
    document: dict,
    verify_credential_signature: Callable[[bytes, bytes], None] | None,
    expected_instance_urn: str | None = None,
    kernel_only: bool = True,
) -> list[str]:
    """Require exactly the one GPU-issued StateAttestation.

    The service's own IdentityAttestation is deliberately not expected here:
    this validates what the *kernel* produced, and the service adds its
    statement to the manifest afterwards.
    """
    if type(manifest) is not dict:
        raise ValueError("kernel receipt must contain a manifest object")
    if set(manifest) != {"version", "statements"}:
        raise ValueError("manifest must carry exactly a version and statements")
    if manifest["version"] != MANIFEST_VERSION:
        raise ValueError(
            f"unsupported manifest version {manifest['version']!r}; "
            f"expected {MANIFEST_VERSION!r}"
        )
    statements = manifest["statements"]
    if type(statements) is not dict:
        raise ValueError("manifest statements must be an object")
    # The kernel emits exactly the one state credential. A manifest that has
    # since passed through the service also carries the service's own
    # attestation, which is signed by a different key and cannot be checked
    # here; those are returned to the caller rather than silently accepted.
    if kernel_only and len(statements) != 1:
        raise ValueError("kernel manifest must contain exactly one statement node")
    if len(statements) < 1:
        raise ValueError("manifest must contain a state credential")

    issuer = document["gpuDID"]
    timestamp = document["measuredAt"]
    model_urn = document["modelCID"]

    # Exactly one credential must be issued by the GPU. Anything else must
    # still be a CredentialRegistration, but is foreign: issued by the service,
    # over facts the GPU never signed.
    gpu_credentials = []
    foreign = []
    for statement_id, statement in statements.items():
        if type(statement) is not dict or statement.get("@type") != "CredentialRegistration":
            raise ValueError("manifest carries a statement that is not a credential")
        if statement.get("registeredBy") == issuer:
            gpu_credentials.append((statement_id, statement))
        else:
            foreign.append(statement_id)
    if len(gpu_credentials) != 1:
        raise ValueError("manifest must contain exactly one GPU-issued credential")
    credential_id, credential = gpu_credentials[0]
    _validate_state_attestation(
        credential_id,
        credential,
        document["modelHash"],
        issuer,
        timestamp,
        model_urn,
        expected_instance_urn,
        verify_credential_signature,
    )
    return foreign


def validate_statement_graph_structure(
    manifest, document: dict, *,
    expected_instance_urn: str | None = None,
) -> None:
    """Validate the closed manifest emitted by the already pinned CUDA program.

    This dependency-free host check prevents the trusted service from
    publishing altered semantics. Evidence consumers must use
    :func:`validate_statement_graph`, which additionally authenticates every
    credential proof against the receipt's verified GPU public key.
    """
    _validate_statement_graph(
        manifest, document, None,
        expected_instance_urn=expected_instance_urn,
    )


def validate_statement_graph(
    manifest,
    document: dict,
    *,
    verify_credential_signature: Callable[[bytes, bytes], None],
    expected_instance_urn: str | None = None,
    kernel_only: bool = True,
) -> list[str]:
    """Validate the manifest and cryptographically verify the GPU's proofs.

    Returns the ids of any statements issued by someone other than the GPU --
    the service's IdentityAttestation, typically. Those are NOT verified here:
    authenticating them needs the issuer's key, pinned separately from the
    GPU's, so the caller must decide what that identity is worth.
    """
    if not callable(verify_credential_signature):
        raise TypeError("credential signature verifier must be callable")
    return _validate_statement_graph(
        manifest, document, verify_credential_signature,
        expected_instance_urn=expected_instance_urn, kernel_only=kernel_only,
    )

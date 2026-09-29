# SPDX-License-Identifier: Apache-2.0
"""Credential CIDs checked across the RDFC implementation boundary."""

import base64
import re
from itertools import permutations

import pytest

from cuattest import _statements as statements
from cuattest import ids
from cuattest._hosthash import blake3_digest

EQTY = "https://eqtylab.io/terms/"
VC = "https://www.w3.org/2018/credentials#"
SEC = "https://w3id.org/security#"
CREATED = "http://purl.org/dc/terms/created"
DATETIME = "http://www.w3.org/2001/XMLSchema#dateTime"


def credential_fields(wrapper):
    credential = wrapper["credential"]
    state = credential["credentialSubject"]["state"]
    return (
        credential["id"],
        credential["issuer"],
        wrapper["timestamp"],
        state["instanceID"],
        state["modelRoot"],
        state["previousStateCredential"],
        credential["proof"]["jws"],
    )


def credential_vector(case):
    """One synthetic registered StateAttestation.

    Odd cases carry a chain link and even ones do not, so the vectors cover
    both the nineteen- and twenty-quad wrapper shapes.
    """
    issuer = ids.did_key_p256(
        bytes.fromhex(
            "6b17d1f2e12c4247f8bce6e563a440f277037d812deb33a0f4a13945d898c296"
        ),
        bytes(31) + b"\x01",
    )
    timestamp = f"2026-09-03T10:00:{case:02d}Z"
    previous = (
        None
        if case % 2 == 0
        else f"urn:uuid:00000000-0000-4000-8000-{case:012d}"
    )
    return {
        "credential": {
            "id": f"urn:uuid:11111111-0000-4000-8000-{case:012d}",
            "type": ["VerifiableCredential", statements.CREDENTIAL_TYPE],
            "credentialSubject": {
                "id": issuer,
                "state": {
                    "stateType": statements.STATE_TYPE,
                    "instanceID": "urn:cid:" + ids.raw_cid(bytes([case]) * 32),
                    "modelRoot": "urn:cid:" + ids.raw_cid(bytes([255 - case]) * 32),
                    "previousStateCredential": previous,
                },
            },
            "issuer": issuer,
            "validFrom": timestamp,
            "proof": {
                "created": timestamp,
                "jws": statements.JWS_PREFIX
                + base64.urlsafe_b64encode(bytes([case]) * 64).rstrip(b"=").decode(),
                "proofPurpose": "assertionMethod",
                "type": "EcdsaSecp256r1Signature2019",
                "verificationMethod": f"{issuer}#{issuer[8:]}",
            },
        },
        "registeredBy": issuer,
        "timestamp": timestamp,
    }


def independently_normalize_credential(wrapper):
    """Expand receipt fields into RDF without using production N-Quads templates.

    No context downloads are needed. URDNA2015 and RDFC-1.0 have identical
    first-degree rules for this dataset (no blank predicates or hash ties).
    """
    jsonld = pytest.importorskip("pyld.jsonld")
    credential = wrapper["credential"]
    proof = credential["proof"]
    state = credential["credentialSubject"]["state"]
    state_node = {
        "@id": "_:state",
        EQTY + "stateType": [{"@value": state["stateType"]}],
        EQTY + "instanceID": [{"@id": state["instanceID"]}],
        EQTY + "modelRoot": [{"@id": state["modelRoot"]}],
    }
    if state["previousStateCredential"] is not None:
        state_node[EQTY + "previousStateCredential"] = [
            {"@id": state["previousStateCredential"]}
        ]
    expanded = {
        # The content-addressed wrapper @id is omitted from its own preimage.
        "@id": "_:registration",
        "@type": [EQTY + "CredentialRegistration"],
        EQTY + "registeredBy": [{"@value": wrapper["registeredBy"]}],
        EQTY + "timestamp": [{"@value": wrapper["timestamp"], "@type": DATETIME}],
        EQTY + "credential": [
            {
                "@id": credential["id"],
                "@type": [VC + "VerifiableCredential", EQTY + statements.CREDENTIAL_TYPE],
                VC + "issuer": [{"@id": credential["issuer"]}],
                VC + "credentialSubject": [
                    {
                        "@id": credential["credentialSubject"]["id"],
                        EQTY + "state": [state_node],
                    }
                ],
                VC + "validFrom": [
                    {"@value": credential["validFrom"], "@type": DATETIME}
                ],
                SEC + "proof": [
                    {
                        "@id": "_:proofGraph",
                        "@graph": [
                            {
                                "@id": "_:proofSubject",
                                "@type": [SEC + proof["type"]],
                                CREATED: [
                                    {"@value": proof["created"], "@type": DATETIME}
                                ],
                                SEC + "jws": [{"@value": proof["jws"]}],
                                SEC + "proofPurpose": [
                                    {"@id": SEC + proof["proofPurpose"]}
                                ],
                                SEC + "verificationMethod": [
                                    {"@id": proof["verificationMethod"]}
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    return jsonld.normalize(
        expanded,
        {
            "algorithm": "URDNA2015",
            "format": "application/n-quads",
        },
    )


def credential_role_labels(canonical):
    return (
        re.search(r"(_:c14n\d) <http://purl.org/dc/terms/created>", canonical)[1],
        re.search(r"<https://w3id.org/security#proof> (_:c14n\d)", canonical)[1],
        re.search(r"(_:c14n\d) <https://eqtylab.io/terms/credential>", canonical)[1],
        re.search(r"(_:c14n\d) <https://eqtylab.io/terms/stateType>", canonical)[1],
    )


@pytest.mark.parametrize(
    "case,expected_cid",
    [
        (0, "bagb6qaq6ec4kg6vy4pfhyxvgc6dgfakhmmcmrofnk5rjmvgjao3lo4svl2h4y"),
        (1, "bagb6qaq6echtrup6uk5rfw3us3yxcbiroren4uk6vcaefjsdpwsj6op2zewww"),
        (2, "bagb6qaq6ebj3qcjspq5dsakbrol4tuphqbyb7ptopwenvibctnygtp7lfnl6k"),
        (5, "bagb6qaq6ebvs3iwxrkewhc5d5vsopzpveqfzuzc63fkmzacrqnuas4dvn3v4a"),
        (6, "bagb6qaq6ecrib563vlgcpb34hfdsl7gebedv3y4eskeu5tjg3wtg266cuhvfu"),
        (7, "bagb6qaq6ecm7xmszlsu63cjimafng2hwptfnzl55franjzwwcrrevrx47umta"),
        (8, "bagb6qaq6ea7es2cb6ndu4l632csviv2ucwsqgfwyl7uihc3qwkhsqooftegxm"),
        (11, "bagb6qaq6ecud36hnqrdttkgcis7xq3ij5wdixd2wwzf6korm7kk6c53jqyhqo"),
        (12, "bagb6qaq6eazq2tro7c33j3tyxmvjea2oojlgfg5r3yvcuqoaor5ggizyelddk"),
        (14, "bagb6qaq6ebimtqqc36rvg2htazg3f75gaclt43yh4zhg7vr4ochllrtnvca7w"),
        (16, "bagb6qaq6eaqdroxxaui75ibxnwes4b4w4vsymsliaso4oly3iy6bkm76oyq5y"),
        (18, "bagb6qaq6ebe7hjmtijit3gb4u3u7wifd2ksj2hprx3xepo3kpivito2x5bx5i"),
        (20, "bagb6qaq6ecfx66cb7xjawsqo5vkkjwpk7jrbrx3tuekirw3uhy7lnjwklc7bq"),
        (25, "bagb6qaq6easune7jhm3p5g5arclylt3u3m2qnj6snnnc4gdncfai24talakay"),
        (27, "bagb6qaq6ecztuha6cocygx53phodukzayq7g5amishbnbsrctqsk5jfhiwjtk"),
        (28, "bagb6qaq6ecsybauhsk7a43ban7unc2atu7xlshktdk6fomi3d4rapwgaztzja"),
        (29, "bagb6qaq6ebe7c7qhzy3mafinchpsuobzoltgz46ypellgbw5uwd72txtgxgee"),
        (31, "bagb6qaq6ebkic5q7f7ekdtyceusih26xs4phfym4erosxdsxbezcm7fdoxyam"),
        (32, "bagb6qaq6ecprlucidn4vrahf3vne6xymtsmv4vwjlnwilou7qbezgquymihaq"),
        (35, "bagb6qaq6ebdzi6ihqcatpbuwxv43ukwmozhcadzrbachcvmngrpdi5jxjvn3e"),
        (36, "bagb6qaq6ed746gd7ngcqs5hzznzqardczinfg3ci6z2x6vjkdaaso5zmbo7wy"),
        (39, "bagb6qaq6ebcgewtmapoglt37cwyc7vuofq6uwu5s4euxapge6tdolwtfpjubw"),
        (48, "bagb6qaq6ecsejszoi7evhz2uj6kq3skaz3tcbubmdqmjenq7ihlulero3ezcy"),
        (52, "bagb6qaq6ecvbwnpuzz4xmysjuuxzbsbcsfr4tdcj3k2qj2oo7j34sqiksodp2"),
    ],
)
def test_credential_cids_against_independently_normalized_vectors(case, expected_cid):
    # Frozen PyLD-normalized vectors cover all twenty-four role orders of the
    # four blank nodes, even on a base installation without PyLD. Relabel and
    # reorder the input to catch template-order dependencies as well as
    # value-dependent canonical-label regressions.
    fields = credential_fields(credential_vector(case))
    assert statements._credential_registration_id(*fields) == "urn:cid:" + expected_cid
    provisional = statements._credential_registration_nquads(*fields)
    for labels in permutations(("_:b0", "_:b1", "_:b2", "_:b3")):
        renamed = re.sub(
            r"_:b([0-3])",
            lambda match, labels=labels: labels[int(match[1])],
            provisional,
        )
        shuffled = "".join(reversed(renamed.splitlines(keepends=True)))
        canonical = statements._canonicalize_credential_nquads(shuffled)
        assert ids.rdfc_cid(blake3_digest(canonical.encode())) == expected_cid


def test_credential_canonicalization_matches_independent_rdf_normalization():
    observed_orders = set()
    for case in range(128):
        wrapper = credential_vector(case)
        fields = credential_fields(wrapper)
        expected = independently_normalize_credential(wrapper)
        actual = statements._canonicalize_credential_nquads(
            statements._credential_registration_nquads(*fields)
        )
        assert actual == expected
        assert statements._credential_registration_id(*fields) == (
            "urn:cid:" + ids.rdfc_cid(blake3_digest(expected.encode()))
        )
        observed_orders.add(credential_role_labels(expected))
    # Every assignment of the four blank nodes: proof subject, proof graph,
    # registration, and the credential subject's nested state.
    assert len(observed_orders) == 24


def test_verifier_accepts_independently_recomputed_wrapper_cids():
    from test_expect import signed_receipt

    from cuattest.expect import verify_evidence

    receipt = signed_receipt(["11" * 32])
    graph = receipt["manifest"]["statements"]
    for old_id, wrapper in list(graph.items()):
        if wrapper["@type"] != "CredentialRegistration":
            continue
        canonical = independently_normalize_credential(wrapper)
        new_id = "urn:cid:" + ids.rdfc_cid(blake3_digest(canonical.encode()))
        del graph[old_id]
        wrapper["@id"] = new_id
        graph[new_id] = wrapper
    verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])


def test_verifier_rejects_legacy_role_labels_even_with_valid_proofs():
    from test_expect import signed_receipt

    from cuattest.expect import EvidenceError, verify_evidence

    receipt = signed_receipt(["11" * 32])
    graph = receipt["manifest"]["statements"]
    for old_id, wrapper in list(graph.items()):
        if wrapper["@type"] != "CredentialRegistration":
            continue
        provisional = statements._credential_registration_nquads(
            *credential_fields(wrapper)
        )
        canonical = statements._canonicalize_credential_nquads(provisional)
        # Force a noncanonical assignment while retaining the genuine JWS.
        labels = ["_:c14n0", "_:c14n1", "_:c14n2", "_:c14n3"]
        if credential_role_labels(canonical) == tuple(labels):
            labels[0], labels[1] = labels[1], labels[0]
        legacy = re.sub(
            r"_:b([0-3])",
            lambda match, labels=labels: labels[int(match[1])],
            provisional,
        )
        bad_id = "urn:cid:" + ids.rdfc_cid(blake3_digest(legacy.encode()))
        assert bad_id != old_id
        del graph[old_id]
        wrapper["@id"] = bad_id
        graph[bad_id] = wrapper
        break
    with pytest.raises(EvidenceError, match="StateAttestation contradicts"):
        verify_evidence(receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])

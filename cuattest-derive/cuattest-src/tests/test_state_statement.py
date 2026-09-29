# SPDX-License-Identifier: Apache-2.0
"""The StateAttestation's canonical form, its chain, and residency identity.

The identifiers below were produced by the CUDA kernel on an sm_75 device and
independently recomputed on the host. Pinning them here is the point: the
credential id is derived from the canonicalized preimage, so any drift in the
N-Quads template, the predicate set, or the RDFC-1 labelling silently changes
every credential identity the project has ever issued. A test that recomputed
them from the implementation would agree with any such drift.
"""

import struct

import pytest

from cuattest import _statements as statements
from cuattest._hosthash import blake3_digest
from cuattest.notary import _instance_root, _owned_instance_root

# One session on a GTX 1660 Ti; two signatures over the same resident copy.
GPU_DID = "did:key:zDnaeg1f7LPo4EK3VSTj3x4WwPtQ67cGyocfhSL9t8uS1rib3"
MODEL_HASH = "629555e24c1bc1d8aac870efc4e11b6bbee695bec8f3a790c88b909d79c77f20"
MODEL_URN = "urn:cid:bafkr4idcsvk6eta3yhmkvsdq57cocg3lx3tjlpwi6otzbselscoxtr37ea"
INSTANCE_URN = "urn:cid:bafkr4iaha4dqobyha4dqobyha4dqobyha4dqobyha4dqobyha4dqobyha4"
GENESIS_TS = "2026-09-16T18:38:48Z"
SECOND_TS = "2026-09-16T18:38:49Z"
GENESIS_ID = "urn:uuid:6a6c84cd-a571-468a-aff7-77352091d798"
SECOND_ID = "urn:uuid:51f29547-4221-4475-a8ac-bfb396581d62"


def _credential_id(timestamp, previous):
    return statements._state_credential_id(
        MODEL_HASH, GPU_DID, timestamp, INSTANCE_URN, MODEL_URN, previous
    )


def _preimage(timestamp, previous):
    return statements._credential_id_preimage(
        GPU_DID, timestamp, INSTANCE_URN, MODEL_URN, previous
    )


def test_genesis_credential_id_matches_the_kernel():
    assert _credential_id(GENESIS_TS, None) == GENESIS_ID


def test_chained_credential_id_matches_the_kernel():
    assert _credential_id(SECOND_TS, GENESIS_ID) == SECOND_ID


def test_genesis_omits_the_previous_triple_entirely():
    # RDF has no null. A genesis preimage carries nine quads, not ten with an
    # empty one, which is also what JSON-LD expansion does with a null key.
    genesis = _preimage(GENESIS_TS, None)
    chained = _preimage(GENESIS_TS, GENESIS_ID)
    assert len(genesis.splitlines()) == 9
    assert len(chained.splitlines()) == 10
    assert "previousStateCredential" not in genesis


def test_blank_node_labels_are_value_dependent_not_positional():
    """Which node wins c14n0 depends on the values, not on its role.

    The labels come from comparing first-degree hashes, so across a range of
    otherwise identical credentials the nested state node sometimes sorts first
    and sometimes second. The kernel must compute this per receipt; a fixed
    assignment would be right only by luck. This sweeps timestamps until both
    outcomes are observed, and fails if one never occurs.
    """

    def subject_of_model_root(timestamp, previous):
        canonical = _preimage(timestamp, previous)
        line = next(l for l in canonical.splitlines() if "modelRoot" in l)
        return line.split()[0]

    observed = set()
    for second in range(40):
        stamp = f"2026-09-16T18:38:{second:02d}Z"
        observed.add(subject_of_model_root(stamp, None))
        observed.add(subject_of_model_root(stamp, GENESIS_ID))
    assert observed == {"_:c14n0", "_:c14n1"}, observed


def test_canonical_form_is_sorted_and_relabelled():
    canonical = _preimage(SECOND_TS, GENESIS_ID)
    lines = canonical.splitlines(keepends=True)
    assert lines == sorted(lines)
    assert "_:b0" not in canonical and "_:b1" not in canonical


def test_the_preimage_omits_the_id_it_derives():
    # The credential cannot name itself in the bytes its own name comes from.
    canonical = _preimage(SECOND_TS, GENESIS_ID)
    assert SECOND_ID not in canonical
    assert GENESIS_ID in canonical  # but the predecessor's name is covered


def test_a_different_predecessor_changes_the_credential_id():
    other = "urn:uuid:00000000-0000-4000-8000-000000000001"
    assert _credential_id(SECOND_TS, other) != SECOND_ID


def test_a_different_instance_changes_the_credential_id():
    other_instance = "urn:cid:" + statements.ids.raw_cid(blake3_digest(b"other"))
    assert (
        statements._state_credential_id(
            MODEL_HASH, GPU_DID, GENESIS_TS, other_instance, MODEL_URN, None
        )
        != GENESIS_ID
    )


def test_a_different_timestamp_changes_the_credential_id():
    # Two genesis measurements of one copy must not collide on an id.
    assert _credential_id(SECOND_TS, None) != _credential_id(GENESIS_TS, None)


class _Ref:
    """The fields _instance_root folds, without importing torch."""

    def __init__(self, handle, nbytes, seg_off=0, t_off=0):
        self._handle = handle
        self.nbytes = nbytes
        self.seg_off = seg_off
        self.t_off = t_off

    def raw_handle(self):
        return self._handle


def _copy(byte):
    return [
        _Ref(bytes([byte]) * 64, 1300),
        _Ref(bytes([byte + 1]) * 64, 132099),
        _Ref(bytes([byte + 2]) * 64, 4096, seg_off=4096),
    ]


def test_instance_root_is_stable_for_one_resident_copy():
    assert _instance_root(_copy(0x11)) == _instance_root(_copy(0x11))


def test_two_identical_models_get_different_instance_roots():
    # Same sizes and offsets, different allocations: modelRoot could not tell
    # these apart, which is the whole reason instanceID exists.
    assert _instance_root(_copy(0x11)) != _instance_root(_copy(0xAA))


def test_reallocating_one_tensor_changes_the_instance_root():
    moved = _copy(0x11)
    moved[2] = _Ref(b"\x99" * 64, 4096, seg_off=4096)
    assert _instance_root(moved) != _instance_root(_copy(0x11))


def test_instance_root_covers_offsets_not_only_handles():
    shifted = _copy(0x11)
    shifted[2] = _Ref(shifted[2].raw_handle(), 4096, seg_off=8192)
    assert _instance_root(shifted) != _instance_root(_copy(0x11))


def test_instance_root_is_order_sensitive():
    reordered = list(reversed(_copy(0x11)))
    assert _instance_root(reordered) != _instance_root(_copy(0x11))


def test_instance_root_folds_the_documented_layout():
    refs = _copy(0x11)
    expected = blake3_digest(
        struct.pack("<I", len(refs))
        + b"".join(
            blake3_digest(r.raw_handle())
            + struct.pack("<QQQ", r.seg_off, r.t_off, r.nbytes)
            for r in refs
        )
    )
    assert _instance_root(refs) == expected


def test_owned_buffers_are_not_comparable_with_real_handles():
    # Pointer identity means nothing across processes; it must never collide
    # with a handle-derived value for the same spans.
    spans = [(0x7F0012340000, 1300)]
    assert _owned_instance_root(spans) != _instance_root(
        [_Ref(b"\x11" * 64, 1300)]
    )


def test_first_degree_tie_fails_closed():
    """Two nodes with identical incident quads must be rejected, not guessed.

    RDFC-1 resolves such ties with N-degree canonicalization, which this
    specialized helper does not implement.
    """
    tied = (
        "_:b0 <https://eqtylab.io/terms/x> _:b1 .\n"
        "_:b1 <https://eqtylab.io/terms/x> _:b0 .\n"
    )
    with pytest.raises(ValueError, match="collision"):
        statements._canonicalize_two_node_nquads(tied)

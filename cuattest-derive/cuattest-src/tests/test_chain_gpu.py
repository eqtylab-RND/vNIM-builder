# SPDX-License-Identifier: Apache-2.0
"""Per-instance statement chains, on real hardware.

The chain lives in module-private device globals, so none of this is reachable
from host-side tests: whether a second signature actually links to the first,
whether two resident copies stay on separate chains, and whether a full table
refuses rather than silently emitting a second genesis for a copy that already
has predecessors.
"""

import json
import os
import time

import pytest

from cuattest import Notary, NotaryError
from cuattest._cuda import DeviceBuffer
from cuattest.cli import _launch_owned_fused_buffers

pytestmark = pytest.mark.skipif(
    os.environ.get("CUATTEST_TEST_GPU") != "1",
    reason="set CUATTEST_TEST_GPU=1 to run real CUDA chain tests",
)

PAYLOADS = [b"chain-a" * 64, b"chain-b" * 128]


def _sign(notary, buffers, instance_root):
    """One signed receipt over buffers this process owns."""
    result = notary._launch_fused_active(
        [(b.ptr, n) for b, n in buffers],
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        b"chain/test-model",
        instance_root,
    )
    return json.loads(result.receipt)


def _credential_of(receipt):
    """The manifest's one GPU-issued StateAttestation."""
    return next(
        statement["credential"]
        for statement in receipt["manifest"]["statements"].values()
        if statement["@type"] == "CredentialRegistration"
    )


def _state_of(receipt):
    return _credential_of(receipt)["credentialSubject"]["state"]


@pytest.fixture
def notary_with_buffers():
    notary = Notary()
    try:
        with notary._activate():
            buffers = [
                (DeviceBuffer.from_bytes(notary.cu, payload), len(payload))
                for payload in PAYLOADS
            ]
            try:
                yield notary, buffers
            finally:
                for buffer, _ in buffers:
                    buffer.close()
    finally:
        notary.close()


def test_first_signature_for_a_copy_is_a_genesis(notary_with_buffers):
    notary, buffers = notary_with_buffers
    state = _state_of(_sign(notary, buffers, b"\x01" * 32))
    assert state["previousStateCredential"] is None


def test_second_signature_links_to_the_first(notary_with_buffers):
    notary, buffers = notary_with_buffers
    first = _credential_of(_sign(notary, buffers, b"\x02" * 32))
    second = _credential_of(_sign(notary, buffers, b"\x02" * 32))
    first_state = first["credentialSubject"]["state"]
    second_state = second["credentialSubject"]["state"]
    assert second_state["previousStateCredential"] == first["id"]
    assert second["id"] != first["id"]
    # Same copy, same content: only the chain and timestamp move.
    assert second_state["instanceID"] == first_state["instanceID"]
    assert second_state["modelRoot"] == first_state["modelRoot"]


def test_a_third_signature_continues_the_same_chain(notary_with_buffers):
    notary, buffers = notary_with_buffers
    ids = [
        _credential_of(_sign(notary, buffers, b"\x03" * 32))["id"] for _ in range(3)
    ]
    previous = [
        _state_of(_sign(notary, buffers, b"\x03" * 32))["previousStateCredential"]
    ]
    assert ids[1] != ids[0] and ids[2] != ids[1]
    assert previous[0] == ids[2]


def test_two_resident_copies_keep_separate_chains(notary_with_buffers):
    notary, buffers = notary_with_buffers
    a_first = _credential_of(_sign(notary, buffers, b"\x04" * 32))
    b_first = _credential_of(_sign(notary, buffers, b"\x05" * 32))
    a_second = _credential_of(_sign(notary, buffers, b"\x04" * 32))
    a_first_state = a_first["credentialSubject"]["state"]
    b_first_state = b_first["credentialSubject"]["state"]

    # A different instanceID starts its own chain rather than continuing A's.
    assert b_first_state["previousStateCredential"] is None
    assert a_second["credentialSubject"]["state"]["previousStateCredential"] == (
        a_first["id"]
    )
    assert a_first_state["instanceID"] != b_first_state["instanceID"]
    # Identical bytes measured twice: modelRoot cannot separate the copies.
    assert a_first_state["modelRoot"] == b_first_state["modelRoot"]


def test_a_full_chain_table_refuses_rather_than_evicting(monkeypatch):
    """Eviction would emit a genesis for a copy that already has a history.

    A verifier could not distinguish that from a forked chain, so the kernel
    returns -10 instead.
    """
    monkeypatch.setenv("CUATTEST_CHAIN_SLOTS", "1")
    notary = Notary()
    try:
        with notary._activate():
            buffers = [
                (DeviceBuffer.from_bytes(notary.cu, payload), len(payload))
                for payload in PAYLOADS
            ]
            try:
                _sign(notary, buffers, b"\x06" * 32)  # claims the only slot
                with pytest.raises(NotaryError, match="no free chain slot"):
                    _sign(notary, buffers, b"\x07" * 32)
            finally:
                for buffer, _ in buffers:
                    buffer.close()
    finally:
        notary.close()


def test_chain_slots_cannot_be_reconfigured(notary_with_buffers):
    # Shrinking or re-pointing the table mid-session would strand live chains
    # in slots no lookup scans again, turning their next statement into a
    # silent genesis.
    notary, _ = notary_with_buffers
    with pytest.raises(NotaryError, match="already configured"):
        notary._configure_chain_slots()

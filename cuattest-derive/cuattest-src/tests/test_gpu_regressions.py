# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-driver regressions: CUATTEST_TEST_GPU=1 pytest tests/test_gpu_regressions.py."""

import json
import os
from contextlib import closing
from types import SimpleNamespace

import pytest
from test_statements import independently_normalize_credential

from cuattest import ids
from cuattest._cuda import DeviceBuffer
from cuattest._hosthash import blake3_digest
from cuattest.cli import _launch_owned_fused_buffers
from cuattest.expect import verify_evidence
from cuattest.notary import Notary
from cuattest.server import make_handler

pytestmark = pytest.mark.skipif(
    os.environ.get("CUATTEST_TEST_GPU") != "1",
    reason="set CUATTEST_TEST_GPU=1 to run real CUDA regressions",
)


@pytest.fixture(scope="module", params=["native", "fallback"])
def gpu_notary(request):
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(
            "CUATTEST_DISABLE_NATIVE_HOST", "1" if request.param == "fallback" else "0"
        )
        with closing(Notary()) as notary:
            assert (notary._native_runner is None) == (request.param == "fallback")
            yield notary


def test_rejected_ipc_request_keeps_http_session_open(gpu_notary):
    notary = gpu_notary
    context = notary.ctx
    handler = object.__new__(make_handler(notary))
    handler.path = "/v1/measure"
    handler._read_json = lambda: {
        "tensors": [
            {
                "handle": bytes(64).hex(),
                "nbytes": 8,
                "device": 0,
                "seg_off": 0,
                "t_off": 0,
            }
        ]
    }
    handler.server = SimpleNamespace(cuattest_fatal_error=None)
    handler.close_connection = False
    sent = []
    handler._send = lambda code, payload, raw=False: sent.append((code, payload))

    handler.do_POST()

    assert len(sent) == 1 and sent[0][0] in (400, 500)
    assert "cuIpcOpenMemHandle" in sent[0][1]["error"]
    assert handler.server.cuattest_fatal_error is None
    assert not handler.close_connection
    assert not notary.ipc_cleanup_required
    assert notary.ctx == context and not notary._closed
    assert notary.hash_bytes(b"still serving") == blake3_digest(b"still serving")


def test_gpu_credential_cids_match_independent_normalization(gpu_notary):
    pytest.importorskip("pyld")
    pytest.importorskip("cryptography")
    notary = gpu_notary
    # Exercise the GPU's credential registration over different measurements
    # and timestamps. The verifier and generator agreeing alone is insufficient.
    for case in range(4):
        data = bytes([case]) * 32
        with notary._activate():
            buffer = DeviceBuffer.from_bytes(notary.cu, data)
            try:
                fused = _launch_owned_fused_buffers(
                    notary,
                    [buffer],
                    [len(data)],
                    f"2026-09-03T10:00:{case:02d}Z",
                    b"canonicalization-regression",
                )
            finally:
                buffer.close()
        receipt = json.loads(fused.receipt)
        receipt["gpu_pubkey_uncompressed"] = notary.info.gpu_pubkey_uncompressed
        verify_evidence(receipt, trusted_pubkey=notary.info.gpu_pubkey_uncompressed)
        wrappers = [
            wrapper
            for wrapper in receipt["manifest"]["statements"].values()
            if wrapper["@type"] == "CredentialRegistration"
        ]
        # A raw kernel receipt carries exactly the GPU's own registration; the
        # service adds its IdentityAttestation only on the higher-level path.
        assert len(wrappers) == 1
        for wrapper in wrappers:
            canonical = independently_normalize_credential(wrapper)
            assert wrapper["@id"] == (
                "urn:cid:" + ids.rdfc_cid(blake3_digest(canonical.encode()))
            )

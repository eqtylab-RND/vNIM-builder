# SPDX-License-Identifier: Apache-2.0
"""Opt-in near-full-VRAM signer regression; never run on a busy GPU."""

import ctypes
import json
import os
from contextlib import closing

import pytest

from cuattest._cuda import DeviceBuffer
from cuattest._hosthash import blake3_digest
from cuattest.cli import _launch_owned_fused_buffers
from cuattest.expect import verify_evidence
from cuattest.notary import Notary

pytestmark = pytest.mark.skipif(
    os.environ.get("CUATTEST_TEST_LARGE_CAPACITY") != "1",
    reason="opt-in only: temporarily reserves nearly all free VRAM on cuda:0",
)


@pytest.mark.parametrize("backend", ["native", "fallback"])
def test_cold_signer_with_only_one_gib_free(monkeypatch, backend):
    monkeypatch.setenv(
        "CUATTEST_DISABLE_NATIVE_HOST", "1" if backend == "fallback" else "0"
    )
    with closing(Notary()) as notary:
        assert (notary._native_runner is None) == (backend == "fallback")
        with notary._activate():
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            query = notary.cu.lib.cuMemGetInfo_v2
            query.argtypes = [ctypes.POINTER(ctypes.c_size_t)] * 2
            query.restype = ctypes.c_int
            notary.cu.check(
                query(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo"
            )
            reserve = 1024**3
            if free.value < reserve + 1024**2:
                pytest.skip("needs at least 1 GiB free VRAM")
            # Do not prewarm the signer: the defect is CUDA allocating its
            # context-wide local-memory backing at the FIRST receipt launch.
            # This filler is not submitted or reported as benchmark weights.
            with closing(DeviceBuffer(notary.cu, free.value - reserve)):
                data = b"cold-signer-memory-regression"
                with closing(DeviceBuffer.from_bytes(notary.cu, data)) as tensor:
                    result = _launch_owned_fused_buffers(
                        notary,
                        [tensor],
                        [len(data)],
                        "2026-09-07T00:00:00Z",
                        b"cold-signer-memory-regression",
                    )
            receipt = json.loads(result.receipt)
            receipt["gpu_pubkey_uncompressed"] = notary.info.gpu_pubkey_uncompressed
            verified = verify_evidence(
                receipt, trusted_pubkey=notary.info.gpu_pubkey_uncompressed
            )
            assert (
                verified.digests is None
                or verified.digests == blake3_digest(data).hex()
            )
            assert (
                verified.model_root
                == blake3_digest((1).to_bytes(4, "little") + blake3_digest(data)).hex()
            )

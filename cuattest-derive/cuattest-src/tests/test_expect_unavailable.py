# SPDX-License-Identifier: Apache-2.0
"""Verification dependency failures are execution errors, not mismatches."""

import builtins

import pytest

from cuattest.expect import (
    Expectation,
    VerificationUnavailableError,
    compare,
)


def test_compare_propagates_missing_signature_support(monkeypatch):
    receipt = {
        "measurementDocument": b"{}".hex(),
        "measurementSignature": "00" * 64,
        "gpu_pubkey_uncompressed": "04" + "00" * 64,
    }
    expected = Expectation(
        model_root="00" * 32,
        vram_cid="not-used",
        tensor_count=1,
        digests="00" * 32,
        names=["weight"],
        total_bytes=32,
    )
    real_import = builtins.__import__

    def without_cryptography(name, *args, **kwargs):
        if name.startswith("cryptography"):
            raise ImportError("cryptography intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cryptography)
    with pytest.raises(VerificationUnavailableError, match="verify.*extra"):
        compare(expected, receipt, trusted_pubkey=receipt["gpu_pubkey_uncompressed"])

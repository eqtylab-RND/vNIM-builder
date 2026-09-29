# SPDX-License-Identifier: Apache-2.0
"""Encoders, against fixed vectors. No GPU required."""

import pytest

from cuattest.ids import b58encode, did_key_p256, raw_cid, uncompressed_hex

# BLAKE3 of the empty input, and the CID an EQTY manifest carries for it.
EMPTY_B3 = bytes.fromhex("af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262")
EMPTY_CID = "bafkr4ifpcne3t5pzugtkaqcn5i3nzskjtpfslsnnyejlpte2spfoihzsmi"


def test_raw_cid_matches_reference_vector():
    assert raw_cid(EMPTY_B3) == EMPTY_CID


def test_raw_cid_rejects_wrong_length():
    with pytest.raises(ValueError):
        raw_cid(b"\x00" * 31)


def test_raw_cid_is_injective_for_a_one_bit_change():
    other = bytearray(EMPTY_B3)
    other[0] ^= 0x01
    assert raw_cid(bytes(other)) != EMPTY_CID


def test_b58_leading_zero_bytes_become_ones():
    assert b58encode(b"\x00\x00\x01").startswith("11")


@pytest.mark.parametrize("parity_byte,expected_prefix", [(0x00, "did:key:z"), (0x01, "did:key:z")])
def test_did_key_shape(parity_byte, expected_prefix):
    y = bytes(31) + bytes([parity_byte])
    assert did_key_p256(bytes(32), y).startswith(expected_prefix)


def test_did_key_encodes_point_parity():
    """The compressed prefix depends on y's parity, so the DIDs must differ."""
    even = did_key_p256(bytes(32), bytes(31) + b"\x00")
    odd = did_key_p256(bytes(32), bytes(31) + b"\x01")
    assert even != odd


def test_did_key_rejects_short_coordinates():
    with pytest.raises(ValueError):
        did_key_p256(bytes(31), bytes(32))


def test_uncompressed_hex_is_65_bytes():
    h = uncompressed_hex(bytes(32), bytes(32))
    assert h.startswith("04") and len(h) == 130

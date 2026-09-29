# SPDX-License-Identifier: Apache-2.0
"""Content identifiers and key identifiers, exactly as the kernel emits them.

Both encodings are reimplemented here rather than pulled from a multiformats
library, so the host and kernel encoders can be checked against each other.
"""

from __future__ import annotations

_B32 = "abcdefghijklmnopqrstuvwxyz234567"
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _base32_multibase(data: bytes) -> str:
    out, acc, bits = ["b"], 0, 0
    for byte in data:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= 5:
            out.append(_B32[(acc >> (bits - 5)) & 31])
            bits -= 5
    if bits:
        out.append(_B32[(acc << (5 - bits)) & 31])
    return "".join(out)


def raw_cid(digest: bytes) -> str:
    """CID for a 32-byte BLAKE3 digest: multibase base32 of raw/blake3 prefix.

    Prefix 0x01 0x55 0x1e 0x20 is CIDv1, raw codec (0x55), blake3-256 (0x1e),
    32 bytes. Lower-case base32 with no padding, 'b' multibase prefix.
    """
    if len(digest) != 32:
        raise ValueError(f"expected a 32-byte digest, got {len(digest)}")
    return _base32_multibase(bytes((0x01, 0x55, 0x1E, 0x20)) + digest)


def rdfc_cid(digest: bytes) -> str:
    """CID for a canonicalized statement using the kernel's RDFC-1 codec."""
    if len(digest) != 32:
        raise ValueError(f"expected a 32-byte digest, got {len(digest)}")
    # 0x83 0xe8 0x02 is the unsigned-varint encoding of the RDFC-1 codec.
    return _base32_multibase(bytes((0x01, 0x83, 0xE8, 0x02, 0x1E, 0x20)) + digest)


def b58encode(data: bytes) -> str:
    n = int.from_bytes(data, "big")
    out = ""
    while n:
        n, rem = divmod(n, 58)
        out = _B58[rem] + out
    return "1" * (len(data) - len(data.lstrip(b"\x00"))) + out


def did_key_p256(x: bytes, y: bytes) -> str:
    """did:key for a P-256 public key: multicodec 0x1200 + compressed point."""
    if len(x) != 32 or len(y) != 32:
        raise ValueError("P-256 coordinates must be 32 bytes each")
    compressed = bytes((0x02 | (y[31] & 1),)) + x
    return "did:key:z" + b58encode(bytes((0x80, 0x24)) + compressed)


def uncompressed_hex(x: bytes, y: bytes) -> str:
    return "04" + x.hex() + y.hex()


def validate_model_cid(value: str) -> str:
    """Validate a host-declared SDK Model asset CID, preserving it unchanged.

    Supports canonical CIDv1 raw blobs and Iroh hashseq collections using
    BLAKE3-256. This identifies the model asset; it is not the GPU tensor root.
    """
    import base64
    if not isinstance(value, str) or not value.startswith("urn:cid:b") or len(value) not in (67, 69):
        raise ValueError("model CID must be a canonical raw or collection BLAKE3-256 urn:cid")
    try:
        encoded = value[len("urn:cid:b"):]
        decoded = base64.b32decode(encoded.upper() + "=" * (-len(encoded) % 8))
        canonical = base64.b32encode(decoded).decode().lower().rstrip("=")
        if encoded != canonical or decoded[:-32] not in (bytes.fromhex("01551e20"), bytes.fromhex("0180011e20")):
            raise ValueError()
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError("invalid model CID") from error
    return value

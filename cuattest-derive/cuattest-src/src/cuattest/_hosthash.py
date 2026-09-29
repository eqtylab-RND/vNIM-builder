# SPDX-License-Identifier: Apache-2.0
"""Host-side hashing used to identify and verify GPU-produced evidence.

This module deliberately uses the independently built ``blake3`` package.
Code provenance must not be computed by the CUBIN whose identity is being
checked: a substituted executable could otherwise simply claim an allowlisted
digest for itself.
"""

from __future__ import annotations

from blake3 import blake3


def blake3_digest(data: bytes) -> bytes:
    """Return an independently computed BLAKE3-256 digest."""
    return blake3(data).digest()

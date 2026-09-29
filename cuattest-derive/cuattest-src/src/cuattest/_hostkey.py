# SPDX-License-Identifier: Apache-2.0
"""The notary service's own durable identity.

The GPU's key is generated per process and dies with it, which is what makes a
session identity meaningful but also means it cannot vouch for anything that
outlives the run. This key is the opposite: it belongs to the service, persists
across restarts, and is what relying parties pin.

It signs the IdentityAttestation binding a session's GPU DID to the CUBIN,
kernel source and device the service loaded. That claim is not verifiable by
the GPU -- a compromised service can still assert the wrong CUBIN -- but it
becomes attributable to a durable identity rather than anonymous, which is the
whole point of issuing it.

P-256, matching the kernel, so one signature suite and one verification path
cover every credential in a manifest.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from . import ids


class HostKeyError(RuntimeError):
    """The service identity could not be loaded or created."""


def default_key_path() -> Path:
    """Where the service keeps its identity unless told otherwise."""
    override = os.environ.get("CUATTEST_HOST_KEY")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "cuattest" / "host_key"


def _require_cryptography():
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
    except ImportError as error:  # pragma: no cover - dependency presence
        raise HostKeyError(
            "host-side signing needs the 'cryptography' package"
        ) from error
    return ec


class HostKey:
    """A loaded service identity: the private scalar and its did:key."""

    def __init__(self, private_key, did: str, path: Path):
        self._private_key = private_key
        self.did = did
        self.path = path

    @property
    def verification_method(self) -> str:
        return f"{self.did}#{self.did[8:]}"

    def sign(self, message: bytes) -> bytes:
        """Return a raw r||s P-256 signature over `message`.

        Raw rather than DER, matching what the kernel emits and what
        expect.py already verifies, so both credentials decode identically.
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec, utils

        der = self._private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        r, s = utils.decode_dss_signature(der)
        return r.to_bytes(32, "big") + s.to_bytes(32, "big")


def _did_for(private_key) -> str:
    numbers = private_key.public_key().public_numbers()
    return ids.did_key_p256(
        numbers.x.to_bytes(32, "big"), numbers.y.to_bytes(32, "big")
    )


def load_or_create(path: Path | None = None) -> HostKey:
    """Load the service identity, creating it on first run.

    The file holds the raw 32-byte private scalar. It is written 0600 and its
    mode is re-checked on every load: a service identity readable by other
    local users is not an identity, and silently continuing would let anyone
    who can read it issue attestations in this service's name.
    """
    ec = _require_cryptography()
    key_path = Path(path) if path is not None else default_key_path()

    if key_path.exists():
        mode = stat.S_IMODE(key_path.stat().st_mode)
        if mode & 0o077:
            raise HostKeyError(
                f"{key_path} is mode {mode:04o}; it must not be readable by "
                "group or others. Fix with: chmod 600 "
                f"{key_path}"
            )
        scalar = key_path.read_bytes()
        if len(scalar) != 32:
            raise HostKeyError(
                f"{key_path} holds {len(scalar)} bytes; a P-256 private "
                "scalar is 32"
            )
        private_key = ec.derive_private_key(
            int.from_bytes(scalar, "big"), ec.SECP256R1()
        )
        return HostKey(private_key, _did_for(private_key), key_path)

    private_key = ec.generate_private_key(ec.SECP256R1())
    scalar = private_key.private_numbers().private_value.to_bytes(32, "big")
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Create the file empty at 0600 before any secret reaches it, so the
    # scalar is never briefly present under a wider mode.
    descriptor = os.open(
        key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        os.write(descriptor, scalar)
    finally:
        os.close(descriptor)
    return HostKey(private_key, _did_for(private_key), key_path)

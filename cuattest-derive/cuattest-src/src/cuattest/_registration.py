# SPDX-License-Identifier: Apache-2.0
"""Server-owned, bounded IPC registrations; never a cache of measured hashes."""

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import secrets
from threading import RLock
import time

from . import ids
from ._build_config import ASSERTIONS_ENABLED
from ._cuda import IpcImportRejectedError
from .notary import (
    IpcCleanupUncertainError,
    IpcSessionAbortedError,
    Measurement,
    NotaryError,
    TensorRef,
    _instance_root,
    _utc_timestamp,
    _validated_model,
)

MAX_REGISTRATIONS = 16
PENDING_SECONDS = 60


class _ImportedSpans:
    """One context's persistent mappings, with Notary-owned error quarantine.

    Normal ownership belongs to this object, retained by the manager BEFORE
    import. Any interrupted driver/bookkeeping boundary destroys the context.
    The sentinel lives on Notary, not this object's fallible exception path.
    """

    def __init__(self, owner, tensors):
        self.owner = owner
        self.tensors = tensors
        self.mappings = [None] * len(tensors)
        self.spans = []
        self.closed = False

    @contextmanager
    def _operation(self):
        owner = self.owner
        # A context-query failure entering _activate must not hide uncertainty
        # inherited from a previous failed import/launch/close.
        try:
            with owner._activate():
                owner._registration_cleanup_required = True
                yield
                owner._registration_cleanup_required = False
        except BaseException as error:
            owner._registration_cleanup_required = True
            cleanup_error = error
            try:
                owner.close()
            except BaseException as failure:
                cleanup_error = failure
            if not owner._context_destroyed:
                raise IpcCleanupUncertainError(
                    "registered IPC cleanup requires confirmed context destruction"
                ) from cleanup_error
            if not isinstance(error, Exception):
                raise
            raise IpcSessionAbortedError(
                "registered IPC operation failed; the context was destroyed"
            ) from error

    def open(self):
        if ASSERTIONS_ENABLED:
            assert not self.closed and not self.spans
            assert len(self.mappings) == len(self.tensors)
            assert all(mapped is None for mapped in self.mappings)
        by_handle = {}
        rejected = None
        with self._operation():
            try:
                for tensor in self.tensors:
                    raw = tensor.raw_handle()
                    if raw not in by_handle:
                        mapped = self.owner.cu.ipc_open(raw)
                        # Fixed ledger slots avoid allocation after driver success.
                        # The sentinel also covers interruption BEFORE assignment.
                        self.mappings[len(by_handle)] = mapped
                        if (
                            self.owner.cu.pointer_device(mapped)
                            != self.owner.device_ordinal
                        ):
                            raise NotaryError(
                                "registered allocation belongs to another GPU"
                            )
                        base, size = self.owner.cu.address_range(mapped)
                        by_handle[raw] = mapped, base, size
                    self.spans.append(
                        (tensor.pointer_in(*by_handle[raw]), tensor.nbytes)
                    )
            except (NotaryError, IpcImportRejectedError) as error:
                # Driver-confirmed rejection/pure bounds checks happen before
                # ANY launch. Retire the known prefix without killing a healthy
                # service. An interrupted import/free instead takes _operation's
                # context-destruction path; never infer rejection from an error
                # that may have arrived after CUDA accepted the mapping.
                rejected = error
                for index, mapped in enumerate(self.mappings):
                    if mapped is not None:
                        self.owner.cu.ipc_close(mapped)
                        self.mappings[index] = None
                self.closed = True
                self.spans.clear()
        if rejected is not None:
            raise NotaryError(str(rejected)) from rejected
        if ASSERTIONS_ENABLED:
            assert len(self.spans) == len(self.tensors)
            assert sum(mapped is not None for mapped in self.mappings) == len(by_handle)

    def sign(self, model):
        model_bytes = _validated_model(model)
        if self.closed:
            raise NotaryError("registration is closed")
        with self._operation():
            if ASSERTIONS_ENABLED:
                assert len(self.spans) == len(self.tensors)
            timestamp = _utc_timestamp()
            started = time.perf_counter()
            # Producer-stream readiness precedes the HTTP request. Run the same
            # fresh hashing, private handoff, signing and stream drain as v1/sign.
            # Registration pins one resident copy for its lifetime, so its
            # residency identity is fixed too: every observation chains to the
            # previous one for this copy rather than starting a new chain.
            instance_root = _instance_root(self.tensors)
            result = self.owner._launch_fused_active(
                self.spans, timestamp, model_bytes, instance_root
            )
            measurement = Measurement(
                digests=result.roots.hex(),
                model_root=result.model_root.hex(),
                vram_cid=ids.raw_cid(result.model_root),
                instance_cid=ids.raw_cid(instance_root),
                tensor_count=len(self.tensors),
                measured_at=timestamp,
                seconds=time.perf_counter() - started,
            )
            return self.owner._signed_receipt(measurement, result.receipt, model)

    def close(self):
        if self.closed:
            return
        if not self.owner._context_destroyed:
            with self._operation():
                for index, mapped in enumerate(self.mappings):
                    if mapped is not None:
                        self.owner.cu.ipc_close(mapped)
                        self.mappings[index] = None
        # Never infer destruction from owner._closed: close() can have failed.
        self.closed = True
        self.spans.clear()
        self.mappings.clear()


@dataclass
class _Entry:
    created: float
    state: str = "pending"
    tensors: list = field(default_factory=list)
    shards: dict = field(default_factory=dict)
    sequence: int = 0


class RegistrationManager:
    """One HTTP service's serial registration namespace and resource budget.

    Reserve a server-generated ticket BEFORE submitting IPC handles. Closing
    removes it; import only accepts existing pending tickets. Consequently an
    idempotent close of an unknown ticket is safe even if an old import request
    arrives later. No unbounded tombstone cache or reused client-chosen IDs.
    """

    def __init__(self, notary):
        self.notary = notary
        self.session = secrets.token_hex(32)
        self._entries = {}
        self._next = 0
        self._lock = RLock()

    def _check(self, body):
        if not isinstance(body, dict) or body.get("session") != self.session:
            raise NotaryError("registration belongs to a different service session")
        token = body.get("token")
        if not isinstance(token, str) or not 1 <= len(token) <= 96:
            raise NotaryError("invalid registration token")
        return token

    def reserve(self):
        with self._lock:
            now = time.monotonic()
            self._entries = {
                key: value
                for key, value in self._entries.items()
                if value.state != "pending" or now - value.created < PENDING_SECONDS
            }
            if len(self._entries) >= MAX_REGISTRATIONS:
                raise NotaryError(
                    "registration limit reached; close an existing registration"
                )
            self._next += 1
            token = f"{self._next:x}:{secrets.token_hex(32)}"
            self._entries[token] = _Entry(now)
            return {"session": self.session, "token": token, "pid": os.getpid()}

    def import_tensors(self, body):
        with self._lock:
            token = self._check(body)
            entry = self._entries.get(token)
            if (
                entry is None
                or entry.state != "pending"
                or time.monotonic() - entry.created >= PENDING_SECONDS
            ):
                raise NotaryError(
                    "registration ticket is closed, expired or already imported"
                )
            raw = body.get("tensors")
            if (
                not isinstance(raw, list)
                or not 1 <= len(raw) <= self.notary.max_request_tensors
            ):
                raise NotaryError("registered tensor count exceeds request limit")
            tensors = [TensorRef.from_dict(ref) for ref in raw]
            if hasattr(self.notary, "notaries"):
                groups = [
                    (self.notary.notaries[uid], [tensors[i] for i in positions])
                    for uid, positions in self.notary._partition(tensors)
                ]
            else:
                groups = [(self.notary, tensors)]
            # Bound all retained logical spans, not just the latest request.
            retained = [
                ref for value in self._entries.values() for ref in value.tensors
            ]
            if (
                len(retained) + len(tensors) > self.notary.max_request_tensors
                or sum(t.nbytes for t in retained + tensors)
                > self.notary.max_request_bytes
                or sum((t.nbytes + 131071) // 131072 for t in retained + tensors)
                > self.notary.max_request_tiles
            ):
                raise NotaryError("aggregate registration budget exceeded")
            for owner, selected in groups:
                owner._validate_tensors(selected)
            entry.state = "importing"
            entry.tensors = tensors
            # Publish every ownership object before opening the first mapping.
            # A partial failure leaves the entry recoverable by close(), or by
            # service teardown if a context has become unusable.
            for owner, selected in groups:
                entry.shards[owner.info.device_uuid] = _ImportedSpans(owner, selected)
            for shard in entry.shards.values():
                shard.open()
            entry.state = "active"
            if ASSERTIONS_ENABLED:
                assert sum(len(shard.tensors) for shard in entry.shards.values()) == len(tensors)
            return {
                "session": self.session,
                "token": token,
                "registered": True,
                "tensor_count": len(tensors),
            }

    def sign(self, body):
        with self._lock:
            token = self._check(body)
            entry = self._entries.get(token)
            if entry is None or entry.state != "active":
                raise NotaryError("registration is not active")
            sequence = body.get("sequence")
            if type(sequence) is not int or sequence != entry.sequence + 1:
                raise NotaryError("registration sequence is stale or out of order")
            _validated_model(body.get("model"))
            entry.sequence = sequence  # Never replay after an ambiguous result.
            if ASSERTIONS_ENABLED:
                assert entry.sequence > 0 and entry.shards
            if hasattr(self.notary, "notaries"):
                receipt = self.notary.sign(
                    entry.tensors, body["model"], _registered=entry.shards
                )
            else:
                receipt = next(iter(entry.shards.values())).sign(body["model"])
            return {
                "session": self.session,
                "token": token,
                "sequence": sequence,
                "receipt": receipt,
            }

    def close(self, body):
        with self._lock:
            token = self._check(body)
            entry = self._entries.get(token)
            if entry is not None:
                for shard in entry.shards.values():
                    shard.close()
                del self._entries[token]
            # An absent ticket can never accept a delayed import: reserve()
            # generates a NEW, non-reused ID. Exact-session close is idempotent.
            return {"session": self.session, "token": token, "released": True}

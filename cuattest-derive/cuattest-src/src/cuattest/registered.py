# SPDX-License-Identifier: Apache-2.0
"""Reusable producer-owned IPC leases with explicit, idempotent retirement."""

from threading import RLock

from ._build_config import ASSERTIONS_ENABLED
from .client import NotaryRequestUncertainError
from .ipc import (
    IpcTensorMutatedError,
    _complete_ipc_refs,
    _require_byte_addressable,
    _tensor_version,
    assert_ipc_refs_immutable,
    claim_ipc_refs,
    quarantine_ipc_refs,
    share_tensors,
    wire_refs,
)
from .notary import _validated_model

_LIVE = {}
_LIVE_LOCK = RLock()


def live_registrations(url):
    with _LIVE_LOCK:
        return tuple(value for value in _LIVE.values() if value.client.url == url)


class RegistrationUncertainError(NotaryRequestUncertainError):
    """Storage remains pinned; registration.close() can retry its retirement."""

    def __init__(self, message, registration):
        super().__init__(message)
        self.registration = registration


def _state(provider):
    import torch

    state = dict(provider())
    state = {
        name: tensor
        for name, tensor in state.items()
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda and tensor.numel() > 0
    }
    if not state:
        raise ValueError("no GPU-resident tensors to register")
    signatures = {}
    for name, tensor in sorted(state.items()):
        _require_byte_addressable(tensor, name)
        if not tensor.is_contiguous() or tensor.is_conj() or tensor.is_neg():
            raise ValueError(
                f"registered tensor {name!r} must be contiguous with resolved view flags; "
                "materialize it in the model first, or use one-shot share_model"
            )
        storage = torch.Tensor.untyped_storage(tensor)
        signatures[name] = (
            storage._cdata,
            storage.data_ptr(),
            storage.nbytes(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.dtype,
            tensor.device,
        )
    return state, signatures


def _streams(state, selected):
    import torch

    devices = {int(t.device.index) for t in state.values()}
    if selected is None:
        selected = {device: torch.cuda.current_stream(device) for device in devices}
    if set(selected) != devices or any(type(d) is not int for d in selected):
        raise ValueError(
            "streams must name exactly the registered producer CUDA ordinals"
        )
    result = {}
    for device, value in selected.items():
        streams = list(value) if isinstance(value, (tuple, list)) else [value]
        if not streams or any(
            not isinstance(s, torch.cuda.Stream) or s.device.index != device
            for s in streams
        ):
            raise ValueError(
                "each producer stream must belong to its declared CUDA device"
            )
        result[device] = streams
    return result


def _check_versions(versions):
    if any(
        version is not None and _tensor_version(tensor) != version
        for tensor, version in versions
    ):
        raise IpcTensorMutatedError("registered tensors changed during measurement")


class RegisteredModel:
    """Pin storage through repeated fresh observations, then explicitly close.

    sign() drains ONLY declared producer streams (default: current per GPU)
    before sending HTTP. Callers must join other writer streams, or provide all
    of them as {ordinal: [stream, ...]}, and keep bytes immutable until return.
    Version counters are a tripwire, not protection against raw CUDA writes.
    """

    @classmethod
    def create(cls, client, provider, *, streams=None):
        self = cls()
        self.client, self._provider = client, provider
        self._lock = RLock()
        self._state = "preparing"
        self._sequence = 0
        state, self._signatures = _state(provider)
        self.names, self._refs, self._keepalive = share_tensors(
            state, _producer_streams=_streams(state, streams)
        )
        # Recovery can discover this handle as soon as _LIVE publishes it.
        # Hold the per-handle lock BEFORE publication through activation or
        # quarantine: close() must not clear refs/provider during import, then
        # have creation resurrect the retired handle as active. Keep _LIVE_LOCK
        # short-lived so other handles remain discoverable during these RPCs.
        with self._lock:
            # Reserve carries NO memory handles. A failed/lost reserve response
            # owns no producer mapping; its empty ticket expires on the service.
            try:
                ticket = client._rpc("/v1/registrations/open", {})
                self.session, self.token = ticket["session"], ticket["token"]
                if (
                    not isinstance(self.session, str)
                    or len(self.session) != 64
                    or not isinstance(self.token, str)
                    or not 1 <= len(self.token) <= 96
                ):
                    raise ValueError("invalid registration ticket")
                self._key = (client.url, self.session, self.token)
                with _LIVE_LOCK:
                    if self._key in _LIVE:
                        raise ValueError("service reused a live registration ticket")
                    _LIVE[self._key] = self
            except BaseException:
                _complete_ipc_refs(self._refs)
                raise
            try:
                self._refs = claim_ipc_refs(self._refs)
                self._state = "uncertain"  # Publish before exposing a single handle.
                result = client._rpc(
                    "/v1/registrations/import",
                    {
                        **self._identity(),
                        "tensors": wire_refs(self._refs),
                    },
                )
                self._check_reply(result)
                if (
                    result.get("registered") is not True
                    or type(result.get("tensor_count")) is not int
                    or result["tensor_count"] != len(self.names)
                ):
                    raise ValueError("incomplete registration acknowledgement")
                assert_ipc_refs_immutable(self._refs)
                self._state = "active"
                return self
            except BaseException as error:
                self._uncertain(error)

    def _identity(self):
        return {"session": self.session, "token": self.token}

    @property
    def attested_bytes(self):
        """Logical bytes rehashed per observation (aliases count separately)."""
        return sum(ref["nbytes"] for ref in self._refs)

    def _check_reply(self, result):
        if not isinstance(result, dict) or any(
            result.get(k) != v for k, v in self._identity().items()
        ):
            raise ValueError(
                "registration acknowledgement belongs to another session/ticket"
            )

    def _uncertain(self, error):
        self._state = "uncertain"
        quarantine_ipc_refs(self._refs)
        if ASSERTIONS_ENABLED:
            # Check only after quarantine is published. A diagnostic must not
            # interrupt the lifetime transition that protects producer storage.
            with _LIVE_LOCK:
                assert _LIVE.get(self._key) is self
        if not isinstance(error, Exception):
            # _LIVE and the allocation registry survive even if the handle's
            # construction or this exception is interrupted. Never GC-unmap.
            raise error
        raise RegistrationUncertainError(
            "registered IPC completion is uncertain; storage remains pinned; "
            "retry registration.close() or independently confirm server termination",
            self,
        ) from error

    def sign(self, model: str, *, streams=None):
        with self._lock:
            if self._state != "active":
                raise RuntimeError(
                    "registration is closed or uncertain; close it before reuse"
                )
            _validated_model(model)
            state, signatures = _state(self._provider)
            if signatures != self._signatures:
                raise IpcTensorMutatedError(
                    "registered tensor names, layout or storage changed; re-register"
                )
            versions = [(tensor, _tensor_version(tensor)) for tensor in state.values()]
            for values in _streams(state, streams).values():
                for stream in values:
                    stream.synchronize()
            _check_versions(versions)
            self._sequence += 1
            self._state = "uncertain"
            try:
                if ASSERTIONS_ENABLED:
                    assert self._sequence > 0
                result = self.client._rpc(
                    "/v1/registrations/sign",
                    {
                        **self._identity(),
                        "sequence": self._sequence,
                        "model": model,
                    },
                )
                self._check_reply(result)
                if (
                    type(result.get("sequence")) is not int
                    or result["sequence"] != self._sequence
                    or not isinstance(result.get("receipt"), dict)
                ):
                    raise ValueError("invalid registered observation acknowledgement")
            except BaseException as error:
                self._uncertain(error)
            self._state = "active"
            # Sign completion allows later writes, NOT allocation release. A
            # mutation-invalid receipt is rejected without retiring this lease.
            _check_versions(versions)
            # state_dict snapshots can outlive a Parameter's .data replacement
            # without changing their version counters. Re-read the provider so
            # a concurrent structural swap cannot silently attest the old model.
            if _state(self._provider)[1] != self._signatures:
                raise IpcTensorMutatedError(
                    "registered tensor names, layout or storage changed during measurement"
                )
            return result["receipt"]

    def close(self, *, server_completed: bool = False):
        """Confirm unregistration, then release producer storage.

        Retry after a lost close response: tickets cannot be imported again.
        server_completed=True is an explicit recovery assertion, identical in
        meaning to IpcKeepalive.release: the caller has INDEPENDENTLY confirmed
        that the old consumer's contexts/process are gone. A timeout, HTTP 500,
        connection refusal or a different server session is NOT such proof.
        """
        if type(server_completed) is not bool:
            raise TypeError("server_completed must be a boolean")
        with self._lock:
            if self._state == "closed":
                self._retire_local()
                return
            self._state = "uncertain"
            try:
                if not server_completed:
                    result = self.client._rpc(
                        "/v1/registrations/close", self._identity()
                    )
                    self._check_reply(result)
                    if result.get("released") is not True:
                        raise ValueError(
                            "unregister did not acknowledge storage release"
                        )
                _complete_ipc_refs(self._refs)
            except BaseException as error:
                self._uncertain(error)
            self._state = "closed"
            self._retire_local()

    def _retire_local(self):
        if ASSERTIONS_ENABLED:
            assert self._state == "closed"
        # Retry local bookkeeping too: an interrupt after marking closed must
        # not leave an immortal model in the process-owned recovery registry.
        with _LIVE_LOCK:
            _LIVE.pop(self._key, None)
        self._keepalive.clear()
        self._refs.clear()
        self._provider = None

    def __enter__(self):
        if self._state != "active":
            raise RuntimeError("registration is not active")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

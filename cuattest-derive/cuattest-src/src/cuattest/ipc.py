# SPDX-License-Identifier: Apache-2.0
"""Client side: hand a model's resident weights to the notary over CUDA IPC.

Import this in the process that owns the model. It needs torch; the notary
itself does not. Ordinary contiguous tensors are shared without copying;
non-contiguous or lazy conjugate/negative views are materialized first.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import count
from threading import Lock
from typing import Any

from ._build_config import ASSERTIONS_ENABLED
from ._cuda import IPC_HANDLE_BYTES

try:
    from ._native import _export_cuda_allocation as _native_export_cuda_allocation
except ImportError:  # non-Linux install or a source tree not built in place
    _native_export_cuda_allocation = None

# New clients pin the exact StorageImpl in this process instead of using
# PyTorch's private, non-transactional producer counter.  Keep the old fields
# readable so applications can safely retire refs created before this change.
_REF_ALLOCATION_LEASE = "_cuattest_allocation_lease"
_REF_COUNTER_HANDLE = "_cuattest_ref_counter_handle"
_REF_COUNTER_OFFSET = "_cuattest_ref_counter_offset"
_REF_CONSUMED = "_cuattest_reference_consumed"
_REF_IN_FLIGHT = "_cuattest_request_in_flight"
_REF_LEASE_UNCERTAIN = "_cuattest_lease_completion_uncertain"
_REF_SOURCE_TENSOR = "_cuattest_source_tensor"
_REF_SOURCE_VERSION = "_cuattest_source_version"
_REF_SOURCE_NAME = "_cuattest_source_name"
_WIRE_FIELDS = ("handle", "nbytes", "seg_off", "t_off", "device", "device_uuid")

_LEASE_FRESH = "fresh"
_LEASE_IN_FLIGHT = "in-flight"
_LEASE_UNCERTAIN = "uncertain"

# CUDA IPC references are byte ranges. PyTorch reports element_size() == 1
# for these sub-byte dtypes even though several logical elements share a byte,
# so numel() * element_size() would hash adjacent storage. Reject them until
# the wire protocol can describe bit offsets and lengths.
_PACKED_DTYPE_NAMES = frozenset(
    {
        "torch.quint2x4",
        "torch.quint4x2",
    }
)


class IpcLeaseUncertainError(RuntimeError):
    """An IPC lease cannot be retired while its consumer may still be running."""


class IpcTensorMutatedError(RuntimeError):
    """A producer tensor changed while its bytes were being measured."""


class IpcReferenceReuseError(RuntimeError):
    """One-shot CUDA IPC references were reused after or during a request."""


class _AllocationLease:
    """Strong storage ownership plus the authoritative one-shot state."""

    __slots__ = ("guards", "state", "storage")

    def __init__(self, storage, guards) -> None:
        self.storage = storage
        self.guards = guards
        self.state = _LEASE_FRESH


# Leases are registered before share_tensors returns.  This process-owned
# table keeps the allocator storage alive even if an application drops its
# model, refs, and IpcKeepalive after an ambiguous transport failure.
_ACTIVE_ALLOCATION_LEASES: dict[int, _AllocationLease] = {}
# This lock protects both registry membership and each lease's request state.
# Keeping ownership and state under one lock makes a copied ref observe the
# same fresh -> in-flight -> uncertain/retired transition as the original.
_ALLOCATION_LEASE_LOCK = Lock()
_ALLOCATION_LEASE_IDS = count(1)


def _reserve_allocation_lease_id() -> int:
    with _ALLOCATION_LEASE_LOCK:
        return next(_ALLOCATION_LEASE_IDS)


def _publish_allocation_lease(lease_id: int, storage, guards) -> None:
    lease = _AllocationLease(storage, guards)
    with _ALLOCATION_LEASE_LOCK:
        if lease_id in _ACTIVE_ALLOCATION_LEASES:
            raise RuntimeError("CUDA IPC allocation lease ID was reused")
        _ACTIVE_ALLOCATION_LEASES[lease_id] = lease
        if ASSERTIONS_ENABLED:
            assert lease.state == _LEASE_FRESH and lease.storage is storage


def _discard_allocation_lease(lease_id: int) -> bool:
    with _ALLOCATION_LEASE_LOCK:
        return _ACTIVE_ALLOCATION_LEASES.pop(lease_id, None) is not None


def _active_allocation_lease_count() -> int:
    """Return process-owned leases; exposed only for deterministic tests."""
    with _ALLOCATION_LEASE_LOCK:
        return len(_ACTIVE_ALLOCATION_LEASES)


class _TensorLeaseTag(str):
    """JSON-compatible private metadata that also keeps its tensor alive."""

    def __new__(cls, name: str, guards):
        tag = super().__new__(cls, name)
        tag.guards = guards
        return tag


def _tensor_version(tensor) -> int | None:
    """Return PyTorch's mutation counter when this tensor type exposes one."""
    try:
        return int(tensor._version)
    except (AttributeError, RuntimeError):
        # Inference tensors may intentionally omit a version counter. They are
        # still governed by the immutable-lease contract documented below.
        return None


def _require_byte_addressable(tensor, name: str) -> None:
    """Reject logical elements whose occupied extent is not byte-addressable."""
    if str(getattr(tensor, "dtype", "")) in _PACKED_DTYPE_NAMES:
        raise ValueError(
            f"tensor {name!r} uses packed dtype {tensor.dtype}; "
            "CUDA IPC attestation currently requires byte-addressable elements"
        )


def _export_storage_allocation(
    torch, storage, expected_device: int
) -> tuple[bytes, int]:
    """Export a storage's containing driver allocation without mutating it."""
    if _native_export_cuda_allocation is None:
        raise RuntimeError(
            "safe CUDA IPC sharing requires the cuattest native extension"
        )

    # Invoke the base descriptors directly.  A Tensor subclass is allowed as
    # input, but it must not interpose Python callbacks while the allocation
    # address and extent are being captured.
    storage_pointer = int(torch.UntypedStorage.data_ptr(storage))
    storage_nbytes = int(torch.UntypedStorage.nbytes(storage))
    # cuIpcGetMemHandle operates in the current CUDA context.  PyTorch's device
    # guard selects the storage's primary context and restores the caller's
    # previous device afterwards, including for multi-GPU state dictionaries.
    with torch.cuda.device(expected_device):
        handle, allocation_base, allocation_nbytes, allocation_device = (
            _native_export_cuda_allocation(storage_pointer)
        )

    if type(handle) is not bytes or len(handle) != IPC_HANDLE_BYTES:
        raise RuntimeError("CUDA driver returned an invalid IPC handle")
    if type(allocation_device) is not int or allocation_device != expected_device:
        raise RuntimeError(
            "CUDA storage device changed during IPC export "
            f"(expected cuda:{expected_device}, got cuda:{allocation_device})"
        )
    if (
        type(allocation_base) is not int
        or type(allocation_nbytes) is not int
        or allocation_base > storage_pointer
        or storage_nbytes < 0
        or allocation_nbytes < 0
        or storage_pointer - allocation_base + storage_nbytes > allocation_nbytes
    ):
        raise RuntimeError("CUDA storage lies outside its driver allocation")
    return handle, storage_pointer - allocation_base


def assert_ipc_refs_immutable(refs: Iterable[dict]) -> None:
    """Reject a result if PyTorch observed an in-place write during its lease.

    This is a best-effort tripwire, not a lock: raw CUDA writes and hostile
    code can bypass PyTorch's version counter. Callers must ensure there are no
    writes of any kind from ``share_tensors`` until acknowledged completion (or
    independently confirmed termination after an uncertain request).
    """
    snapshots = []
    with _ALLOCATION_LEASE_LOCK:
        for ref in refs:
            if _REF_ALLOCATION_LEASE in ref:
                lease = _ACTIVE_ALLOCATION_LEASES.get(ref[_REF_ALLOCATION_LEASE])
                if lease is None:
                    # Lease IDs are monotonic and never reused. A missing entry
                    # therefore means that this stale dictionary was copied
                    # before another copy completed and retired the request.
                    # Mutation after that completion is legal, so do not fall
                    # back to this dictionary's old _TensorLeaseTag snapshot.
                    continue
                # The registry is authoritative and survives a JSON round
                # trip of refs, unlike attributes on _TensorLeaseTag.
                snapshots.append(
                    (ref.get(_REF_SOURCE_NAME, "<unnamed>"), tuple(lease.guards))
                )
                continue

            lease_tag = ref.get(_REF_SOURCE_TENSOR)
            if lease_tag is not None:
                snapshots.append(
                    (
                        ref.get(_REF_SOURCE_NAME, str(lease_tag)),
                        getattr(lease_tag, "guards", ()),
                    )
                )

    changed = []
    detached = []
    for name, guards in snapshots:
        if not guards:
            # A detached tag without a live registry entry has no trustworthy
            # mutation guard. This can occur only for legacy/stale refs.
            detached.append(name)
            continue
        if any(
            expected is not None and _tensor_version(tensor) != expected
            for tensor, expected in guards
        ):
            changed.append(name)
    if changed:
        raise IpcTensorMutatedError(
            "producer tensor(s) changed during CUDA IPC measurement: "
            + ", ".join(changed[:5])
        )
    if detached:
        raise IpcTensorMutatedError(
            "producer mutation guard was detached for tensor(s): "
            + ", ".join(detached[:5])
        )


def claim_ipc_refs(refs: Iterable[dict]) -> list[dict]:
    """Atomically reserve fresh, one-shot references for one client request.

    Process-owned refs carry a monotonic lease ID. Its registry state, rather
    than flags on a copyable dictionary, is authoritative for the transition.
    """
    refs = list(refs)
    with _ALLOCATION_LEASE_LOCK:
        for ref in refs:
            lease = None
            has_process_lease = _REF_ALLOCATION_LEASE in ref
            if has_process_lease:
                lease = _ACTIVE_ALLOCATION_LEASES.get(ref[_REF_ALLOCATION_LEASE])
                if lease is None:
                    raise IpcReferenceReuseError(
                        "CUDA IPC references name a retired allocation lease; "
                        "call share_tensors() again"
                    )
                if lease.state == _LEASE_UNCERTAIN:
                    raise IpcLeaseUncertainError(
                        "CUDA IPC request completion is unknown; these references "
                        "cannot be sent again"
                    )
                if lease.state == _LEASE_IN_FLIGHT:
                    raise IpcReferenceReuseError(
                        "CUDA IPC references are already in use by another request"
                    )
            if ref.get(_REF_IN_FLIGHT):
                raise IpcReferenceReuseError(
                    "CUDA IPC references are already in use by another request"
                )
            if ref.get(_REF_LEASE_UNCERTAIN):
                raise IpcLeaseUncertainError(
                    "CUDA IPC request completion is unknown; these references "
                    "cannot be sent again"
                )
            if ref.get(_REF_CONSUMED):
                raise IpcReferenceReuseError(
                    "CUDA IPC references are one-shot and have already been "
                    "consumed; call share_tensors() again"
                )

        for ref in refs:
            # Publish in the process registry before touching the copyable ref.
            # If a dict update or process-control exception then interrupts the
            # claim, the authoritative lease remains pinned and non-reusable.
            if _REF_ALLOCATION_LEASE in ref:
                _ACTIVE_ALLOCATION_LEASES[
                    ref[_REF_ALLOCATION_LEASE]
                ].state = _LEASE_IN_FLIGHT
            # Do not roll registry state back if either assignment is
            # interrupted: a pinned, non-reusable lease is safer than letting
            # an old handle escape after its ownership becomes unclear.
            ref[_REF_CONSUMED] = True
            ref[_REF_IN_FLIGHT] = True
    return refs


def wire_refs(refs: Iterable[dict]) -> list[dict]:
    """Strip private metadata and reject references that are no longer usable."""
    result = []
    with _ALLOCATION_LEASE_LOCK:
        for ref in refs:
            if _REF_ALLOCATION_LEASE in ref:
                lease = _ACTIVE_ALLOCATION_LEASES.get(ref[_REF_ALLOCATION_LEASE])
                if lease is None:
                    raise IpcReferenceReuseError(
                        "CUDA IPC references name a retired allocation lease"
                    )
                if lease.state == _LEASE_UNCERTAIN:
                    raise IpcLeaseUncertainError(
                        "CUDA IPC references remain quarantined"
                    )
                if lease.state == _LEASE_IN_FLIGHT and not ref.get(_REF_IN_FLIGHT):
                    raise IpcReferenceReuseError(
                        "CUDA IPC references are already in use by another request"
                    )
            if ref.get(_REF_CONSUMED) and not ref.get(_REF_IN_FLIGHT):
                raise IpcReferenceReuseError(
                    "CUDA IPC references are one-shot and have already been consumed"
                )
            if ref.get(_REF_LEASE_UNCERTAIN):
                raise IpcLeaseUncertainError(
                    "CUDA IPC request completion is unknown; references remain quarantined"
                )
            result.append({key: ref[key] for key in _WIRE_FIELDS if key in ref})
    return result


def quarantine_ipc_refs(refs: Iterable[dict]) -> None:
    """Mark producer leases whose RPC may still be using their allocations."""
    with _ALLOCATION_LEASE_LOCK:
        for ref in refs:
            # Even manually assembled refs become unsafe to retry after an
            # ambiguous send. Generated refs additionally retain their storage
            # lease below until server completion is independently known.
            if _REF_ALLOCATION_LEASE in ref:
                lease = _ACTIVE_ALLOCATION_LEASES.get(ref[_REF_ALLOCATION_LEASE])
                if lease is not None:
                    lease.state = _LEASE_UNCERTAIN
            ref[_REF_LEASE_UNCERTAIN] = True
            ref.pop(_REF_IN_FLIGHT, None)


def _release_ipc_refs(
    refs: Iterable[dict], *, server_completed: bool, client_finished: bool
) -> None:
    """Retire leases only after their active consumer is known to be done."""
    refs = list(refs)
    pending_legacy = [ref for ref in refs if _REF_COUNTER_HANDLE in ref]
    retired_leases = []
    with _ALLOCATION_LEASE_LOCK:
        for ref in refs:
            lease = None
            if _REF_ALLOCATION_LEASE in ref:
                lease = _ACTIVE_ALLOCATION_LEASES.get(ref[_REF_ALLOCATION_LEASE])
            state = lease.state if lease is not None else None
            if (
                state == _LEASE_IN_FLIGHT or ref.get(_REF_IN_FLIGHT)
            ) and not client_finished:
                # In particular, IpcKeepalive.__del__ can run on another thread
                # while Client is blocked. Only that Client's completion path
                # may clear an active claim; server_completed is for a request
                # already transitioned to uncertain after transport failure.
                raise IpcLeaseUncertainError(
                    "CUDA IPC request is still in flight; its allocation lease "
                    "cannot be released"
                )
            if (
                state == _LEASE_UNCERTAIN or ref.get(_REF_LEASE_UNCERTAIN)
            ) and not server_completed:
                raise IpcLeaseUncertainError(
                    "CUDA IPC request completion is unknown; keep the lease "
                    "quarantined until server completion or cancellation is "
                    "independently confirmed, then pass server_completed=True"
                )

        # Mark every dictionary consumed before removing even one registry
        # entry. A MemoryError therefore leaves all actual storage pinned and
        # no partially retired handle can look fresh.
        for ref in refs:
            ref[_REF_CONSUMED] = True

        # Retain local references until after the lock is released. Removing a
        # registry entry is the ownership handoff that makes any pre-existing
        # JSON/dict copy stale: its monotonic lease ID can never resolve again.
        for ref in refs:
            if _REF_ALLOCATION_LEASE in ref:
                lease = _ACTIVE_ALLOCATION_LEASES.get(ref[_REF_ALLOCATION_LEASE])
                if lease is not None:
                    retired_leases.append(lease)
        for ref in refs:
            if _REF_ALLOCATION_LEASE in ref:
                _ACTIVE_ALLOCATION_LEASES.pop(ref[_REF_ALLOCATION_LEASE], None)
                ref.pop(_REF_ALLOCATION_LEASE, None)
            ref.pop(_REF_IN_FLIGHT, None)
            ref.pop(_REF_LEASE_UNCERTAIN, None)
            # Drop dictionary-level guards only after the authoritative lease
            # has been retired or found already retired.
            ref.pop(_REF_SOURCE_TENSOR, None)
            ref.pop(_REF_SOURCE_VERSION, None)
            ref.pop(_REF_SOURCE_NAME, None)

    used_legacy_counter = False
    for ref in pending_legacy:
        if _REF_COUNTER_HANDLE in ref:
            # Compatibility for refs exported by older cuAttest clients.
            import torch

            handle = bytes.fromhex(ref[_REF_COUNTER_HANDLE])
            offset = ref[_REF_COUNTER_OFFSET]
            torch.UntypedStorage._release_ipc_counter_cuda(handle, offset)
            ref.pop(_REF_COUNTER_HANDLE, None)
            ref.pop(_REF_COUNTER_OFFSET, None)
            used_legacy_counter = True
    if used_legacy_counter:
        # Old refs may already have moved their allocation to PyTorch's IPC
        # limbo; collect immediately once their counters reach zero.
        torch.cuda.ipc_collect()
    # Keep process-owned storage alive through all fallible legacy cleanup.
    retired_leases.clear()


def _complete_ipc_refs(refs: Iterable[dict]) -> None:
    """Client-only completion after acknowledgement or a definitely-unsent RPC."""
    _release_ipc_refs(refs, server_completed=True, client_finished=True)


def release_ipc_refs(refs: Iterable[dict], *, server_completed: bool = False) -> None:
    """Release process-owned/legacy leases, but never an active Client claim."""
    _release_ipc_refs(refs, server_completed=server_completed, client_finished=False)


class IpcKeepalive(list):
    """Tensor list that retires acknowledged CUDA allocation leases."""

    def __init__(self, tensors: Iterable[Any], refs: list[dict]) -> None:
        super().__init__(tensors)
        self._refs = refs

    def release(self, *, server_completed: bool = False) -> None:
        """Validate immutability and retire an acknowledged IPC lease.

        Mutation does not make an already acknowledged mapping unsafe to
        release, so report it only after the allocation lease is retired.
        An uncertain request still takes precedence and remains quarantined.
        An active Client claim is never releasable here; ``server_completed``
        applies only after that Client has transitioned it to uncertain.
        """
        mutation = None
        try:
            self.assert_unchanged()
        except IpcTensorMutatedError as e:
            mutation = e
        release_ipc_refs(self._refs, server_completed=server_completed)
        if mutation is not None:
            raise mutation

    def assert_unchanged(self) -> None:
        """Check PyTorch's mutation counters before accepting a direct-HTTP result."""
        assert_ipc_refs_immutable(self._refs)

    def __del__(self) -> None:
        # Interpreter shutdown can tear torch down first; explicit Client calls
        # remain the reliable path, while this is a safety net for raw HTTP use.
        # release() deliberately refuses quarantined refs, so destruction never
        # guesses that a disconnected consumer has stopped using the storage.
        try:
            self.release()
        except Exception:  # noqa: BLE001,S110 - best effort during interpreter teardown
            pass


def _device_uuids(devices: list[int]) -> dict[int, str]:
    from ._cuda import Cuda

    cuda = Cuda()
    cuda.init()
    return {ordinal: cuda.device_uuid(cuda.device(ordinal)) for ordinal in devices}


def share_tensors(tensors: dict[str, Any] | Iterable[tuple[str, Any]], *, _producer_streams=None):
    """Return (names, refs, keepalive) for every GPU-resident tensor.

    `refs` are JSON-ready dicts to POST to the notary (plus underscore-prefixed
    producer cleanup metadata stripped by :class:`cuattest.client.Client`).
    All writers must be quiesced before this call; `keepalive` must stay
    referenced and every shared tensor must remain immutable until the
    measurement returns. Freeing or writing a tensor can
    respectively recycle its segment or give the notary a torn, cross-tensor
    state. Client calls reject results when PyTorch's version counter detects a
    write, release process-owned storage leases after an acknowledged response,
    and quarantine them after an ambiguous transport failure. Direct HTTP users
    should call ``keepalive.assert_unchanged()`` before accepting the response,
    followed by ``keepalive.release()``.
    A refs list is one-shot; call this function again for every later request.

    Lazy conjugate/negative flags and non-contiguous layouts are materialized
    before export; the resulting copies are synchronized and retained too.
    Storage must support legacy CUDA IPC. PyTorch expandable-segment/VMM and
    cudaMallocAsync allocations are unsupported: configure the native allocator
    with ``expandable_segments:False`` before allocating the model (see the
    operations guide). Changing settings cannot convert existing allocations.

    Names are sorted so the digest order — and therefore the folded model
    root — is reproducible for anyone re-measuring the same model.
    """
    import torch

    if _native_export_cuda_allocation is None:
        raise RuntimeError(
            "safe CUDA IPC sharing requires the cuattest native extension"
        )

    items = tensors.items() if isinstance(tensors, dict) else tensors
    state = {k: v for k, v in items}
    names = sorted(
        k
        for k, t in state.items()
        if isinstance(t, torch.Tensor) and t.is_cuda and t.numel() > 0
    )
    if not names:
        raise ValueError("no GPU-resident tensors to share")

    prepared = []
    for name in names:
        source = state[name].detach()
        _require_byte_addressable(source, name)
        guards = [(source, _tensor_version(source))]
        t = source
        # Lazy conjugate/negative views can be contiguous yet expose the
        # untransformed backing bytes. Resolve values, not just strides, before
        # exporting storage. Retain and version-guard every asynchronous copy
        # as well as the original view, including intermediate resolutions.
        if t.is_conj():
            t = t.resolve_conj()
            guards.append((t, _tensor_version(t)))
        if t.is_neg():
            t = t.resolve_neg()
            guards.append((t, _tensor_version(t)))
        if not t.is_contiguous():
            # A non-contiguous view's bytes are not the tensor's bytes; the
            # copy is what we then share and what gets measured.
            t = t.contiguous()
            guards.append((t, _tensor_version(t)))
        # Capture before synchronization so a concurrent PyTorch write queued
        # during preparation cannot become the unobserved baseline version.
        prepared.append((name, t, guards))

    # This synchronization must follow all resolve_conj()/resolve_neg() and
    # contiguous() copies: they are asynchronous CUDA work too. Sync each device,
    # rather than only whichever device happens to be current.
    # Unlike the notary's private execution stream, source tensors may have been
    # written on ANY producer stream. Waiting only on current_stream() here
    # would silently export unfinished writes from other streams. Keep this
    # readiness barrier unless the protocol gains explicit producer events.
    devices = sorted(
        {
            int(t.device.index if t.device.index is not None else t.get_device())
            for _, t, _ in prepared
        }
    )
    for device in devices:
        if _producer_streams is None:
            torch.cuda.synchronize(device)
        else:
            # Private registered-model path: it rejects views requiring copies
            # and declares every writer stream explicitly before export. Do
            # not wait for unrelated work on the whole producer device.
            for stream in _producer_streams[device]:
                stream.synchronize()
    # Producer ordinals are process-local. Route by physical identity when a
    # service sees a different CUDA_VISIBLE_DEVICES order or only a subset.
    device_uuids = _device_uuids(devices)

    changed_while_preparing = [
        name
        for name, _, guards in prepared
        if any(
            version is not None and _tensor_version(tensor) != version
            for tensor, version in guards
        )
    ]
    if changed_while_preparing:
        raise IpcTensorMutatedError(
            "producer tensor(s) changed while preparing CUDA IPC measurement: "
            + ", ".join(changed_while_preparing[:5])
        )

    tensors_kept, refs = [], []
    registered_lease_ids = []
    try:
        for name, t, guards in prepared:
            tensors_kept.append(t)
            # Call Tensor's descriptor directly so a tensor subclass cannot
            # substitute a Python callback for storage discovery.  Unlike
            # torch's private _share_cuda_, the driver export that follows is
            # observational and has no partially acquired counter to roll back.
            storage = torch.Tensor.untyped_storage(t)
            device = int(
                t.device.index if t.device.index is not None else t.get_device()
            )
            handle, seg_off = _export_storage_allocation(torch, storage, device)
            # Finish tensor/subclass calls and wire-value conversions before
            # publishing the process lease. Once published, only operations on
            # exact built-in values remain, and the surrounding BaseException
            # handler owns the lease ID on every exceptional exit.
            handle_hex = handle.hex()
            tensor_offset = int(t.storage_offset()) * int(t.element_size())
            tensor_nbytes = int(t.numel()) * int(t.element_size())
            lease_tag = _TensorLeaseTag(name, guards)

            # Publish strong ownership before returning an exportable handle.
            # Remember the ID locally first, so every exceptional exit can
            # remove a registration that may already have completed.
            lease_id = _reserve_allocation_lease_id()
            registered_lease_ids.append(lease_id)
            _publish_allocation_lease(lease_id, storage, guards)
            refs.append(
                {
                    "handle": handle_hex,
                    "seg_off": int(seg_off),
                    "t_off": tensor_offset,
                    "nbytes": tensor_nbytes,
                    "device": device,
                    "device_uuid": device_uuids[device],
                    # This process-private ID retains the exact StorageImpl;
                    # wire_refs strips it before the request leaves the client.
                    _REF_ALLOCATION_LEASE: lease_id,
                    # Keep both a strong reference and the version observed at
                    # export. Client checks it again only after the server has
                    # closed the mapping, covering the full measurement window.
                    # A str subclass keeps refs JSON-serializable for callers
                    # while privately holding strong source/copy references and
                    # their versions for the post-response check.
                    _REF_SOURCE_TENSOR: lease_tag,
                    _REF_SOURCE_VERSION: guards[-1][1],
                    _REF_SOURCE_NAME: name,
                }
            )
        keepalive = IpcKeepalive(tensors_kept, refs)
    except BaseException:
        for lease_id in registered_lease_ids:
            _discard_allocation_lease(lease_id)
        raise
    return names, refs, keepalive


def share_model(model) -> tuple[list[str], list[dict], list]:
    """`share_tensors` over a torch module's ``state_dict()``."""
    return share_tensors(model.state_dict())

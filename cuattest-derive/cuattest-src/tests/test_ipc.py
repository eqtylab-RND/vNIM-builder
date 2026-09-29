# SPDX-License-Identifier: Apache-2.0
"""PyTorch CUDA-IPC handoff invariants, with a dependency-free fake torch."""

import errno
import gc
import http.client
import json
import socket
import sys
import urllib.error
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from cuattest import client as client_module
from cuattest import ipc as ipc_module
from cuattest.client import Client, NotaryClientError, NotaryRequestUncertainError
from cuattest.ipc import (
    _REF_ALLOCATION_LEASE,
    _REF_CONSUMED,
    _REF_COUNTER_HANDLE,
    _REF_COUNTER_OFFSET,
    _REF_LEASE_UNCERTAIN,
    IpcKeepalive,
    IpcLeaseUncertainError,
    IpcReferenceReuseError,
    IpcTensorMutatedError,
    _active_allocation_lease_count,
    release_ipc_refs,
    share_tensors,
)


class FakeStorage:
    def __init__(self, tensor, events):
        self.tensor = tensor
        self.events = events

    def ipc_handle(self):
        return bytes([self.tensor.device.index + 1]) * 64


class FakeTensor:
    is_cuda = True

    def __init__(
        self, name, events, device=0, contiguous=True, dtype="torch.float16",
        *, conjugate=False, negative=False,
    ):
        self.name = name
        self.events = events
        self.device = SimpleNamespace(index=device)
        self._contiguous = contiguous
        self._conjugate = conjugate
        self._negative = negative
        self._version = 0
        self.dtype = dtype

    def numel(self):
        return 4

    def element_size(self):
        return 2

    def storage_offset(self):
        return 3

    def detach(self):
        return self

    def is_contiguous(self):
        return self._contiguous

    def is_conj(self):
        return self._conjugate

    def is_neg(self):
        return self._negative

    def resolve_conj(self):
        if not self.is_conj():
            return self
        self.events.append(("resolve_conj", self.name))
        return FakeTensor(
            self.name, self.events, self.device.index, self._contiguous,
            self.dtype, negative=self._negative,
        )

    def resolve_neg(self):
        if not self.is_neg():
            return self
        self.events.append(("resolve_neg", self.name))
        return FakeTensor(
            self.name, self.events, self.device.index, self._contiguous,
            self.dtype, conjugate=self._conjugate,
        )

    def contiguous(self):
        # Real contiguous lazy views return themselves: contiguity alone must
        # never be used as a proxy for resolved logical tensor values.
        if self.is_contiguous():
            return self
        self.events.append(("copy", self.name))
        return FakeTensor(
            self.name,
            self.events,
            self.device.index,
            contiguous=True,
            dtype=self.dtype,
        )

    def untyped_storage(self):
        storage_type = getattr(self, "storage_type", FakeStorage)
        return storage_type(self, self.events)


def fake_torch(events):
    class FakeCuda:
        @staticmethod
        def synchronize(device):
            events.append(("sync", device))

        @staticmethod
        def ipc_collect():
            events.append(("collect",))

    class FakeUntypedStorage:
        @staticmethod
        def _release_ipc_counter_cuda(handle, offset):
            events.append(("release", handle, offset))

    return SimpleNamespace(
        Tensor=FakeTensor, cuda=FakeCuda, UntypedStorage=FakeUntypedStorage
    )


def test_registered_export_drains_only_each_declared_producer_stream(monkeypatch):
    events = []
    torch = fake_torch(events)
    monkeypatch.setitem(sys.modules, "torch", torch)
    torch.cuda.synchronize = lambda device: pytest.fail("device-wide registered export wait")
    streams = {
        device: [SimpleNamespace(synchronize=lambda d=device, i=i: events.append(("stream", d, i)))
                 for i in range(2)]
        for device in (0, 2)
    }
    names, refs, keepalive = share_tensors(
        {f"d{d}": FakeTensor(str(d), events, device=d) for d in (0, 2)},
        _producer_streams=streams,
    )
    try:
        assert names == ["d0", "d2"]
        assert [event for event in events if event[0] == "stream"] == [
            ("stream", 0, 0), ("stream", 0, 1), ("stream", 2, 0), ("stream", 2, 1)
        ]
        assert len(refs) == 2
    finally:
        keepalive.release()


@pytest.fixture(autouse=True)
def fake_cuda_allocation_export(monkeypatch):
    assert _active_allocation_lease_count() == 0
    monkeypatch.setattr(
        ipc_module,
        "_device_uuids",
        lambda devices: {
            device: f"GPU-00000000-0000-0000-0000-{device:012d}" for device in devices
        },
    )

    def export(torch, storage, expected_device):
        assert storage.tensor.device.index == expected_device
        storage.events.append(("share", storage.tensor.name))
        return storage.ipc_handle(), 7

    monkeypatch.setattr(ipc_module, "_export_storage_allocation", export)
    yield
    assert _active_allocation_lease_count() == 0


def test_noncontiguous_copies_finish_before_any_handle_is_shared(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    tensors = {
        "z": FakeTensor("z", events, device=1, contiguous=True),
        "a": FakeTensor("a", events, device=0, contiguous=False),
    }

    names, refs, keepalive = share_tensors(tensors)

    assert names == ["a", "z"]
    assert [r["device"] for r in refs] == [0, 1]
    assert [r["device_uuid"] for r in ipc_module.wire_refs(refs)] == [
        "GPU-00000000-0000-0000-0000-000000000000",
        "GPU-00000000-0000-0000-0000-000000000001",
    ]
    first_share = min(i for i, event in enumerate(events) if event[0] == "share")
    assert ("copy", "a") in events[:first_share]
    assert ("sync", 0) in events[:first_share]
    assert ("sync", 1) in events[:first_share]
    keepalive.release()


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("conjugate,negative", [(True, False), (False, True), (True, True)])
def test_lazy_view_copies_are_resolved_synchronized_and_guarded(
    monkeypatch, contiguous, conjugate, negative
):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    source = FakeTensor(
        "weight", events, device=1, contiguous=contiguous,
        conjugate=conjugate, negative=negative,
    )
    _, refs, keepalive = share_tensors({"weight": source})
    try:
        exported = keepalive[0]
        assert exported is not source
        assert exported.is_contiguous() and not exported.is_conj() and not exported.is_neg()
        copies = []
        if conjugate:
            copies.append(("resolve_conj", "weight"))
        if negative:
            copies.append(("resolve_neg", "weight"))
        if not contiguous:
            copies.append(("copy", "weight"))
        assert events == copies + [("sync", 1), ("share", "weight")]
        lease = ipc_module._ACTIVE_ALLOCATION_LEASES[refs[0][_REF_ALLOCATION_LEASE]]
        assert lease.storage.tensor is exported
        assert len(lease.guards) == 1 + len(copies)
        assert lease.guards[0][0] is source and lease.guards[-1][0] is exported
        # Keep the original and every asynchronous intermediate copy alive;
        # writes to any of them must trip the existing immutable-lease guard.
        for tensor, version in lease.guards:
            tensor._version += 1
            try:
                with pytest.raises(IpcTensorMutatedError):
                    keepalive.assert_unchanged()
            finally:
                tensor._version = version
    finally:
        keepalive.release()


def test_lazy_view_copies_on_all_devices_finish_before_export(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    _, _, keepalive = share_tensors({
        "a": FakeTensor("a", events, device=0, conjugate=True),
        "z": FakeTensor("z", events, device=1, negative=True),
    })
    try:
        assert events == [
            ("resolve_conj", "a"), ("resolve_neg", "z"),
            ("sync", 0), ("sync", 1), ("share", "a"), ("share", "z"),
        ]
    finally:
        keepalive.release()


@pytest.mark.parametrize("dtype", ["torch.quint4x2", "torch.quint2x4"])
def test_packed_dtypes_are_rejected_instead_of_hashing_adjacent_bytes(
    monkeypatch, dtype
):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    tensor = FakeTensor("packed", events, dtype=dtype)

    with pytest.raises(ValueError, match="packed dtype.*byte-addressable"):
        share_tensors({"packed": tensor})

    assert events == []  # no copy, synchronization, or IPC lease was acquired


def test_wire_conversion_failure_does_not_publish_an_allocation_lease(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    tensor = FakeTensor("weight", events)

    class FailingHex(bytes):
        def hex(self):
            raise MemoryError("wire handle allocation failed")

    class FailAfterExportStorage(FakeStorage):
        def ipc_handle(self):
            # Handle export itself changes no storage ownership. Fail during
            # subsequent wire conversion, before the process lease is made
            # externally observable.
            return FailingHex(super().ipc_handle())

    tensor.storage_type = FailAfterExportStorage

    with pytest.raises(MemoryError, match="wire handle allocation failed"):
        share_tensors({"weight": tensor})

    assert events == [("sync", 0), ("share", "weight")]
    assert _active_allocation_lease_count() == 0


def test_interrupted_lease_publication_is_rolled_back(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    publish = ipc_module._publish_allocation_lease

    def publish_then_fail(lease_id, storage, guards):
        publish(lease_id, storage, guards)
        raise MemoryError("interrupted after registry insertion")

    monkeypatch.setattr(ipc_module, "_publish_allocation_lease", publish_then_fail)
    with pytest.raises(MemoryError, match="after registry insertion"):
        share_tensors({"weight": FakeTensor("weight", events)})

    assert _active_allocation_lease_count() == 0


def test_keepalive_construction_failure_retires_all_published_leases(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))

    def fail_keepalive(*args):
        raise MemoryError("keepalive allocation failed")

    monkeypatch.setattr(ipc_module, "IpcKeepalive", fail_keepalive)
    with pytest.raises(MemoryError, match="keepalive allocation failed"):
        share_tensors({"weight": FakeTensor("weight", events)})

    assert _active_allocation_lease_count() == 0


def test_tensor_storage_python_share_callback_is_never_invoked(monkeypatch):
    events = []
    callback_state = ["must remain intact"]
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    tensor = FakeTensor("weight", events)

    class HostileStorage(FakeStorage):
        def _share_cuda_(self):
            callback_state.clear()
            pytest.fail("private PyTorch sharing callback was invoked")

    tensor.storage_type = HostileStorage
    _, _, keepalive = share_tensors({"weight": tensor})

    assert callback_state == ["must remain intact"]
    keepalive.release()


def test_driver_export_failure_has_no_partial_ownership_to_roll_back(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))

    def failed_export(*args):
        raise MemoryError("driver export result")

    monkeypatch.setattr(ipc_module, "_export_storage_allocation", failed_export)

    with pytest.raises(MemoryError, match="driver export result"):
        share_tensors({"weight": FakeTensor("weight", events)})

    assert _active_allocation_lease_count() == 0


def test_incompatible_later_allocation_retires_earlier_export_leases(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    export = ipc_module._export_storage_allocation

    def reject_second(torch, storage, device):
        if storage.tensor.name == "z":
            assert _active_allocation_lease_count() == 1
            raise RuntimeError("CUDA allocation does not support legacy CUDA IPC")
        return export(torch, storage, device)

    monkeypatch.setattr(ipc_module, "_export_storage_allocation", reject_second)
    with pytest.raises(RuntimeError, match="does not support legacy CUDA IPC"):
        share_tensors({
            "a": FakeTensor("a", events),
            "z": FakeTensor("z", events),
        })
    assert _active_allocation_lease_count() == 0


def test_private_immutability_metadata_keeps_refs_json_serializable(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))

    _, refs, keepalive = share_tensors({"weight": FakeTensor("weight", events)})

    # Some direct-HTTP integrations serialize the returned refs before using
    # wire_refs(). Private lifetime metadata must not break that old contract.
    assert json.loads(json.dumps(refs))[0]["_cuattest_source_tensor"] == "weight"
    keepalive.release()


@pytest.mark.parametrize("contiguous,conjugate,negative", [
    (True, False, False), (False, False, False),
    (True, True, False), (True, False, True), (False, True, True),
])
def test_write_during_export_preparation_is_rejected_before_sharing(
    monkeypatch, contiguous, conjugate, negative
):
    events = []
    tensor = FakeTensor(
        "weight", events, contiguous=contiguous,
        conjugate=conjugate, negative=negative,
    )
    torch = fake_torch(events)

    def mutate_during_sync(device):
        events.append(("sync", device))
        tensor._version += 1

    torch.cuda.synchronize = mutate_during_sync
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(IpcTensorMutatedError, match="while preparing"):
        share_tensors({"weight": tensor})

    assert not any(event[0] == "share" for event in events)


def test_release_uses_pytorch_counter_metadata_once(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [
        {
            "handle": "ab" * 64,
            _REF_COUNTER_HANDLE: b"/torch-counter".hex(),
            _REF_COUNTER_OFFSET: 9,
        }
    ]

    release_ipc_refs(refs)
    release_ipc_refs(refs)

    assert events == [("release", b"/torch-counter", 9), ("collect",)]
    assert _REF_COUNTER_HANDLE not in refs[0]


def test_client_strips_cleanup_metadata_and_releases_after_response(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [
        {
            "handle": "ab" * 64,
            "nbytes": 8,
            "seg_off": 0,
            "t_off": 0,
            "device": 0,
            _REF_COUNTER_HANDLE: b"/torch-counter".hex(),
            _REF_COUNTER_OFFSET: 4,
        }
    ]
    client = Client()

    def rpc(path, body):
        events.append(("rpc", path, body))
        assert _REF_COUNTER_HANDLE not in body["tensors"][0]
        return {"ok": True}

    monkeypatch.setattr(client, "_rpc", rpc)
    assert client.measure(refs) == {"ok": True}
    assert [event[0] for event in events] == ["rpc", "release", "collect"]


def test_atomic_sign_sends_tensors_and_model_in_one_request(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [{"handle": "ab" * 64, "nbytes": 8, "device": 0}]
    client = Client()
    monkeypatch.setattr(client, "_rpc", lambda path, body: {"path": path, "body": body})

    result = client.sign(refs, "model/a")

    assert result["path"] == "/v1/sign"
    assert result["body"] == {
        "tensors": [{"handle": "ab" * 64, "nbytes": 8, "device": 0}],
        "model": "model/a",
    }
    assert refs[0][_REF_CONSUMED] is True


def test_acknowledged_refs_cannot_be_sent_a_second_time(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    tensor = FakeTensor("weight", events, contiguous=False)
    _, refs, keepalive = share_tensors({"weight": tensor})
    client = Client()
    requests = []
    monkeypatch.setattr(
        client,
        "_rpc",
        lambda path, body: requests.append((path, body)) or {"ok": True},
    )

    assert client.measure(refs) == {"ok": True}
    assert refs[0][_REF_CONSUMED] is True
    assert _REF_ALLOCATION_LEASE not in refs[0]

    with pytest.raises(IpcReferenceReuseError, match="one-shot.*consumed"):
        client.sign(refs, "must-not-be-sent")

    assert len(requests) == 1
    del keepalive


def test_keepalive_cannot_release_a_lease_while_client_is_in_flight(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    _, refs, keepalive = share_tensors(
        {"weight": FakeTensor("weight", events, contiguous=False)}
    )
    lease_id = refs[0][_REF_ALLOCATION_LEASE]
    entered_rpc = Event()
    finish_rpc = Event()
    results = []
    failures = []
    client = Client()

    def blocked_rpc(path, body):
        entered_rpc.set()
        assert finish_rpc.wait(2), "test did not release the blocked RPC"
        return {"ok": True}

    def request():
        try:
            results.append(client.measure(refs))
        except Exception as error:  # noqa: BLE001 - capture worker failures
            failures.append(error)

    monkeypatch.setattr(client, "_rpc", blocked_rpc)
    worker = Thread(target=request)
    worker.start()
    assert entered_rpc.wait(2), "client did not enter its RPC"
    try:
        for server_completed in (False, True):
            with pytest.raises(IpcLeaseUncertainError, match="still in flight"):
                keepalive.release(server_completed=server_completed)
        # The failed release must preserve both authoritative storage ownership
        # and the dictionary metadata until the owning Client acknowledges.
        assert lease_id in ipc_module._ACTIVE_ALLOCATION_LEASES
        assert refs[0][_REF_ALLOCATION_LEASE] == lease_id
    finally:
        finish_rpc.set()
        worker.join(2)

    assert not worker.is_alive()
    assert failures == []
    assert results == [{"ok": True}]
    assert _active_allocation_lease_count() == 0
    del keepalive


def test_json_copied_refs_share_the_authoritative_one_shot_lease(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    _, refs, keepalive = share_tensors({"weight": FakeTensor("weight", events)})
    copied_refs = json.loads(json.dumps(refs))
    requests = []
    client = Client()
    monkeypatch.setattr(
        client,
        "_rpc",
        lambda path, body: requests.append((path, body)) or {"ok": True},
    )

    assert client.measure(refs) == {"ok": True}
    assert _active_allocation_lease_count() == 0

    # The copy retained the monotonic lease ID. Its missing registry entry now
    # proves retirement before a stale CUDA handle can reach the network.
    with pytest.raises(IpcReferenceReuseError, match="retired allocation lease"):
        client.measure(copied_refs)

    assert len(requests) == 1
    del keepalive


def test_original_keepalive_ignores_guards_retired_through_copied_refs(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    tensor = FakeTensor("weight", events)
    _, original_refs, keepalive = share_tensors({"weight": tensor})
    copied_refs = json.loads(json.dumps(original_refs))
    lease_id = original_refs[0][_REF_ALLOCATION_LEASE]
    client = Client()
    monkeypatch.setattr(client, "_rpc", lambda path, body: {"ok": True})

    assert client.measure(copied_refs) == {"ok": True}
    assert lease_id not in ipc_module._ACTIVE_ALLOCATION_LEASES
    # Completion cleaned only the submitted dictionaries. The keepalive's
    # originals intentionally retain a stale monotonic ID and mutation tag.
    assert original_refs[0][_REF_ALLOCATION_LEASE] == lease_id

    # The producer may legally reuse or mutate storage after Client returns.
    # Idempotent keepalive cleanup must recognize that the copied request
    # already retired its global lease instead of checking a stale snapshot.
    tensor._version += 1
    keepalive.release()
    keepalive.release()

    assert _REF_ALLOCATION_LEASE not in original_refs[0]
    assert _active_allocation_lease_count() == 0


def test_client_bypasses_environment_proxies(monkeypatch):
    created_handlers = []
    opened = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b'{"direct": true}'

    class Opener:
        def open(self, request, timeout):
            opened.append((request.full_url, timeout))
            return Response()

    def build_opener(*handlers):
        created_handlers.extend(handlers)
        return Opener()

    monkeypatch.setenv("http_proxy", "http://untrusted-proxy.invalid:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setattr(client_module.urllib.request, "build_opener", build_opener)
    monkeypatch.setattr(
        client_module.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("global proxy-aware urlopen was used"),
    )

    client = Client("http://notary.internal:8077", timeout=3)

    assert len(created_handlers) == 1
    assert isinstance(created_handlers[0], client_module.urllib.request.ProxyHandler)
    assert created_handlers[0].proxies == {}
    assert client.info() == {"direct": True}
    assert opened == [("http://notary.internal:8077/v1/info", 3)]


def test_ambiguous_rpc_failure_quarantines_ipc_lease(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [
        {
            "handle": "ab" * 64,
            "nbytes": 8,
            "device": 0,
            _REF_COUNTER_HANDLE: b"/torch-counter".hex(),
            _REF_COUNTER_OFFSET: 4,
        }
    ]
    keepalive = IpcKeepalive([object()], refs)
    client = Client()

    def timed_out(request, timeout):
        raise TimeoutError("connection timed out")

    # Exercise the transport classifier too: urllib may surface a socket
    # timeout as a bare OSError subclass rather than URLError.
    monkeypatch.setattr(client._opener, "open", timed_out)
    with pytest.raises(NotaryRequestUncertainError):
        client.measure(refs)

    assert refs[0][_REF_LEASE_UNCERTAIN] is True
    assert events == []
    # Model teardown invokes the keepalive destructor, which must not
    # decrement the counter while the raw consumer may still be reading.
    del keepalive
    gc.collect()
    assert events == []

    # A later ordinary cleanup attempt is guarded for the same reason.
    with pytest.raises(IpcLeaseUncertainError, match="completion is unknown"):
        release_ipc_refs(refs)

    # An operator can retire the quarantined lease after independently
    # confirming that the server completed, cancelled, or restarted.
    release_ipc_refs(refs, server_completed=True)
    assert events == [("release", b"/torch-counter", 4), ("collect",)]


def test_ambiguous_rpc_keeps_the_exported_storage_in_the_process_registry(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    tensor = FakeTensor("weight", events)
    _, refs, keepalive = share_tensors({"weight": tensor})
    client = Client()

    def timed_out(path, body):
        raise NotaryRequestUncertainError("request completion is unknown")

    monkeypatch.setattr(client, "_rpc", timed_out)
    with pytest.raises(NotaryRequestUncertainError):
        client.measure(refs)

    assert refs[0][_REF_LEASE_UNCERTAIN] is True
    assert _active_allocation_lease_count() == 1
    del keepalive, tensor
    gc.collect()
    assert _active_allocation_lease_count() == 1

    release_ipc_refs(refs, server_completed=True)
    assert _active_allocation_lease_count() == 0


def test_definitive_server_error_releases_ipc_lease(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [
        {
            "handle": "ab" * 64,
            "nbytes": 8,
            "device": 0,
            _REF_COUNTER_HANDLE: b"/torch-counter".hex(),
            _REF_COUNTER_OFFSET: 4,
        }
    ]
    client = Client()

    def rejected(path, body):
        raise NotaryClientError("HTTP 400", ipc_completion_known=True)

    monkeypatch.setattr(client, "_rpc", rejected)
    with pytest.raises(NotaryClientError):
        client.measure(refs)

    assert events == [("release", b"/torch-counter", 4), ("collect",)]
    assert _REF_LEASE_UNCERTAIN not in refs[0]


class TruncatedResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        raise http.client.IncompleteRead(b'{"ok":', 1)


def test_truncated_acknowledged_response_releases_ipc_lease(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [
        {
            "handle": "ab" * 64,
            "nbytes": 8,
            "device": 0,
            _REF_COUNTER_HANDLE: b"/torch-counter".hex(),
            _REF_COUNTER_OFFSET: 4,
        }
    ]
    client = Client()
    monkeypatch.setattr(
        client._opener, "open", lambda request, timeout: TruncatedResponse()
    )

    with pytest.raises(NotaryClientError, match="incomplete JSON") as caught:
        client.measure(refs)

    assert not isinstance(caught.value, NotaryRequestUncertainError)
    assert caught.value.ipc_completion_known is True
    assert events == [("release", b"/torch-counter", 4), ("collect",)]


@pytest.mark.parametrize(
    "reason",
    [
        ConnectionRefusedError("connection refused"),
        socket.gaierror(socket.EAI_NONAME, "name not known"),
    ],
)
def test_definitely_preconnect_failure_releases_ipc_lease(monkeypatch, reason):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [
        {
            "handle": "ab" * 64,
            "nbytes": 8,
            "device": 0,
            _REF_COUNTER_HANDLE: b"/torch-counter".hex(),
            _REF_COUNTER_OFFSET: 4,
        }
    ]

    def not_connected(request, timeout):
        raise urllib.error.URLError(reason)

    client = Client()
    monkeypatch.setattr(client._opener, "open", not_connected)
    with pytest.raises(NotaryClientError, match="request was not sent") as caught:
        client.measure(refs)

    assert not isinstance(caught.value, NotaryRequestUncertainError)
    assert caught.value.ipc_completion_known is True
    assert events == [("release", b"/torch-counter", 4), ("collect",)]


@pytest.mark.parametrize("route_errno", [errno.EHOSTUNREACH, errno.ENETUNREACH])
def test_route_failure_with_unknown_phase_stays_quarantined(monkeypatch, route_errno):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [
        {
            "handle": "ab" * 64,
            "nbytes": 8,
            "device": 0,
            _REF_COUNTER_HANDLE: b"/torch-counter".hex(),
            _REF_COUNTER_OFFSET: 4,
        }
    ]

    def route_failed(request, timeout):
        raise urllib.error.URLError(OSError(route_errno, "route failed"))

    client = Client()
    monkeypatch.setattr(client._opener, "open", route_failed)
    with pytest.raises(NotaryRequestUncertainError):
        client.measure(refs)

    assert refs[0][_REF_LEASE_UNCERTAIN] is True
    assert events == []
    release_ipc_refs(refs, server_completed=True)
    assert events == [("release", b"/torch-counter", 4), ("collect",)]


def test_client_rejects_a_result_if_a_shared_tensor_changed_during_rpc(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    tensor = FakeTensor("weight", events)
    _, refs, keepalive = share_tensors({"weight": tensor})
    lease_id = refs[0][_REF_ALLOCATION_LEASE]
    client = Client()

    def mutate_while_measuring(path, body):
        tensor._version += 1
        return {"vram_cid": "must-not-escape"}

    monkeypatch.setattr(client, "_rpc", mutate_while_measuring)

    with pytest.raises(IpcTensorMutatedError, match="weight"):
        client.measure(refs)

    # The acknowledged mapping is gone, so rejecting its result must still
    # retire the process-owned allocation lease rather than leak it.
    assert _REF_ALLOCATION_LEASE not in refs[0]
    assert lease_id not in ipc_module._ACTIVE_ALLOCATION_LEASES
    del keepalive


def test_direct_http_keepalive_reports_mutation_after_safe_release(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    tensor = FakeTensor("weight", events)
    _, refs, keepalive = share_tensors({"weight": tensor})
    tensor._version += 1

    with pytest.raises(IpcTensorMutatedError, match="weight"):
        keepalive.release()

    assert _REF_ALLOCATION_LEASE not in refs[0]


def test_malformed_status_is_wrapped_and_quarantined(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [
        {
            "handle": "ab" * 64,
            "nbytes": 8,
            "device": 0,
            _REF_COUNTER_HANDLE: b"/torch-counter".hex(),
            _REF_COUNTER_OFFSET: 4,
        }
    ]

    def malformed_status(request, timeout):
        raise http.client.BadStatusLine("not HTTP")

    client = Client()
    monkeypatch.setattr(client._opener, "open", malformed_status)
    with pytest.raises(
        NotaryRequestUncertainError, match="completion is unknown"
    ) as caught:
        client.measure(refs)

    assert isinstance(caught.value.__cause__, http.client.BadStatusLine)
    assert refs[0][_REF_LEASE_UNCERTAIN] is True
    assert events == []
    release_ipc_refs(refs, server_completed=True)
    assert events == [("release", b"/torch-counter", 4), ("collect",)]


def test_unexpected_rpc_exception_uses_documented_uncertainty_error(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "torch", fake_torch(events))
    refs = [
        {
            "handle": "ab" * 64,
            "nbytes": 8,
            "device": 0,
            _REF_COUNTER_HANDLE: b"/torch-counter".hex(),
            _REF_COUNTER_OFFSET: 4,
        }
    ]
    client = Client()

    def unexpected(path, body):
        raise RuntimeError("alternate transport failed")

    monkeypatch.setattr(client, "_rpc", unexpected)
    with pytest.raises(NotaryRequestUncertainError) as caught:
        client.measure(refs)

    assert isinstance(caught.value.__cause__, RuntimeError)
    assert refs[0][_REF_LEASE_UNCERTAIN] is True
    assert events == []
    release_ipc_refs(refs, server_completed=True)
    assert events == [("release", b"/torch-counter", 4), ("collect",)]

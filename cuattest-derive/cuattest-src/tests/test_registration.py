# SPDX-License-Identifier: Apache-2.0
"""Persistent IPC protocol and fail-closed ownership, without a GPU."""

import io
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from cuattest import _registration as reg
from cuattest import registered as producer
from cuattest._cuda import CudaError, IpcImportRejectedError
from cuattest.client import Client, NotaryClientError
from cuattest.notary import (
    IpcCleanupUncertainError,
    IpcSessionAbortedError,
    NotaryError,
    TensorRef,
    _async_hash_threshold,
    _ASYNC_HASH_MIN_BYTES,
)
from test_notary import ImportedCuda, bare_notary


@pytest.mark.parametrize("arch", ["sm_75", "sm_80", "sm_90", "sm_100", "sm_120"])
def test_async_dispatch_only_qualifies_large_blackwell_automatically(arch):
    assert _async_hash_threshold(arch, "standard") is None
    assert _async_hash_threshold(arch, "auto") == (
        _ASYNC_HASH_MIN_BYTES if arch == "sm_120" else None
    )
    if arch == "sm_75":
        with pytest.raises(ValueError, match="sm_80"):
            _async_hash_threshold(arch, "async")
    else:
        assert _async_hash_threshold(arch, "async") == 0
    with pytest.raises(ValueError, match="CUATTEST_HASH_MODE"):
        _async_hash_threshold(arch, "typo")


def owner(**kwargs):
    result = bare_notary(ImportedCuda(allocation_nbytes=4096, **kwargs))
    result.info = SimpleNamespace(device_uuid="GPU-" + "1" * 32)
    result.max_request_tensors = 4
    result.max_request_bytes = 4096
    result.max_request_tiles = 4
    return result


def refs():
    return [TensorRef(bytes(64).hex(), 1024, 0, offset, 0) for offset in (0, 1024)]


def request(manager, **extra):
    ticket = manager.reserve()
    return {"session": ticket["session"], "token": ticket["token"], **extra}


def wire():
    return [
        dict(
            handle=t.handle,
            nbytes=t.nbytes,
            seg_off=t.seg_off,
            t_off=t.t_off,
            device=t.device,
        )
        for t in refs()
    ]


def test_ticket_close_is_idempotent_and_delayed_import_cannot_resurrect_it():
    n = owner()
    manager = reg.RegistrationManager(n)
    body = request(manager, tensors=wire())
    assert manager.close(body) == manager.close(body)
    with pytest.raises(NotaryError, match="closed"):
        manager.import_tensors(body)
    assert not n.cu.opened
    with pytest.raises(NotaryError, match="different service"):
        reg.RegistrationManager(n).close(body)
    assert manager.reserve()["token"] != body["token"]


def test_registration_imports_once_and_enforces_sequence_and_aggregate_budget(
    monkeypatch,
):
    n = owner()
    manager = reg.RegistrationManager(n)
    body = request(manager, tensors=wire())
    assert manager.import_tensors(body)["tensor_count"] == 2
    assert len(n.cu.opened) == 1 and not n.cu.closed
    shard = manager._entries[body["token"]].shards[n.info.device_uuid]
    assert shard.spans == [(0x1000, 1024), (0x1400, 1024)]
    observed = []
    monkeypatch.setattr(
        shard, "sign", lambda model: observed.append(model) or {"fresh": len(observed)}
    )
    for sequence in (1, 2):
        assert manager.sign({**body, "sequence": sequence, "model": "test"})[
            "receipt"
        ] == {"fresh": sequence}
    for sequence in (1, 2, 4, True, "3"):
        with pytest.raises(NotaryError, match="sequence"):
            manager.sign({**body, "sequence": sequence, "model": "test"})
    with pytest.raises(NotaryError, match="already imported"):
        manager.import_tensors(body)
    n.max_request_tensors = 3
    with pytest.raises(NotaryError, match="aggregate"):
        manager.import_tensors(request(manager, tensors=wire()))
    assert len(n.cu.opened) == 1
    manager.close(body)
    assert n.cu.closed == [0x1000]
    manager.close(body)
    assert n.cu.closed == [0x1000]


def test_pending_tickets_expire_without_import_and_slots_are_bounded(monkeypatch):
    manager = reg.RegistrationManager(owner())
    monkeypatch.setattr(reg.time, "monotonic", lambda: 100)
    tickets = [request(manager) for _ in range(reg.MAX_REGISTRATIONS)]
    with pytest.raises(NotaryError, match="limit"):
        manager.reserve()
    monkeypatch.setattr(reg.time, "monotonic", lambda: 160)
    with pytest.raises(NotaryError, match="expired"):
        manager.import_tensors({**tickets[0], "tensors": wire()})
    manager.reserve()
    assert len(manager._entries) == 1


@pytest.mark.parametrize("destroy_fails", [False, True])
@pytest.mark.parametrize("stage", ["import", "bounds", "close", "enter", "unwind"])
def test_driver_interruptions_preserve_quarantine_until_confirmed_destruction(
    monkeypatch, destroy_fails, stage
):
    n = owner(ctx_destroy_fails=destroy_fails)
    shard = reg._ImportedSpans(n, refs())

    def fail(*args):
        raise CudaError("injected uncertain driver error")

    if stage == "import":
        monkeypatch.setattr(n.cu, "ipc_open", fail)
    elif stage == "bounds":
        monkeypatch.setattr(n.cu, "address_range", fail)
    elif stage == "close":
        shard.open()
        monkeypatch.setattr(n.cu, "ipc_close", fail)
    elif stage == "enter":
        # close() also queries the current context; failure must leave the
        # independent sentinel set even when no specialized error survives.
        monkeypatch.setattr(n.cu, "ctx_get_current", fail)
        destroy_fails = True
    elif stage == "unwind":
        from contextlib import contextmanager

        @contextmanager
        def activation():
            yield
            fail()

        monkeypatch.setattr(n, "_activate", activation)
    with pytest.raises(
        IpcCleanupUncertainError if destroy_fails else IpcSessionAbortedError
    ):
        shard.close() if stage == "close" else shard.open()
    assert n.ipc_cleanup_required == destroy_fails
    assert n._context_destroyed != destroy_fails
    if destroy_fails:
        with pytest.raises(IpcCleanupUncertainError):
            shard.close()
        assert not shard.closed
    else:
        shard.close()
        assert shard.closed


def test_known_rejected_import_retires_prefix_without_destroying_service(monkeypatch):
    n = owner()
    tensors = refs()
    tensors[1] = TensorRef((bytes([1]) * 64).hex(), 1024)
    real_open = n.cu.ipc_open

    def reject_second(raw):
        if raw[0]:
            raise IpcImportRejectedError("driver rejected handle")
        return real_open(raw)

    monkeypatch.setattr(n.cu, "ipc_open", reject_second)
    shard = reg._ImportedSpans(n, tensors)
    with pytest.raises(NotaryError, match="rejected"):
        shard.open()
    assert n.cu.closed == [0x1000]
    assert shard.closed and not n.ipc_cleanup_required and not n._closed


@pytest.mark.parametrize("kind", ["error-headers", "truncated-body"])
def test_registered_http_headers_never_acknowledge_storage_release(kind):
    client = Client()

    def open_request(*args, **kwargs):
        if kind == "error-headers":
            raise HTTPError(client.url, 500, "error", {}, io.BytesIO(b"{}"))
        return io.BytesIO(b"{")

    client._opener = SimpleNamespace(open=open_request)
    for operation in ("import", "sign", "close"):
        with pytest.raises(NotaryClientError) as caught:
            client._rpc("/v1/registrations/" + operation, {})
        assert not caught.value.ipc_completion_known
    with pytest.raises(NotaryClientError) as caught:
        client._rpc("/v1/sign", {})
    assert caught.value.ipc_completion_known


@pytest.fixture
def fake_producer(monkeypatch):
    events = []
    tensor = SimpleNamespace(version=0)
    signature = {"weight": (1, 2, 3)}
    client = Client("http://registration-unit-test")

    def rpc(path, body):
        events.append(path)
        return dict(
            session="a" * 64,
            token="ticket",
            registered=True,
            tensor_count=1,
            sequence=body.get("sequence"),
            receipt={"fresh": body.get("sequence")},
            released=True,
        )

    monkeypatch.setattr(client, "_rpc", rpc)
    monkeypatch.setattr(
        producer, "_state", lambda provider: ({"weight": tensor}, dict(signature))
    )
    monkeypatch.setattr(
        producer,
        "_streams",
        lambda state, selected: {
            0: [SimpleNamespace(synchronize=lambda: events.append("stream-sync"))]
        },
    )
    monkeypatch.setattr(producer, "_tensor_version", lambda t: t.version)
    monkeypatch.setattr(
        producer,
        "share_tensors",
        lambda *a, **k: (["weight"], [{"handle": "x"}], [tensor]),
    )
    monkeypatch.setattr(
        producer, "claim_ipc_refs", lambda refs: events.append("claim") or refs
    )
    monkeypatch.setattr(producer, "wire_refs", lambda refs: refs)
    monkeypatch.setattr(producer, "assert_ipc_refs_immutable", lambda refs: None)
    monkeypatch.setattr(
        producer, "quarantine_ipc_refs", lambda refs: events.append("quarantine")
    )
    monkeypatch.setattr(
        producer, "_complete_ipc_refs", lambda refs: events.append("release")
    )
    yield client, events, tensor, signature
    for handle in client.registrations():
        handle.close(server_completed=True)


def test_producer_rehashes_updated_values_but_rejects_replaced_storage(fake_producer):
    client, events, tensor, signature = fake_producer
    with client.register_tensors({}) as handle:
        assert handle.sign("model") == {"fresh": 1}
        tensor.version += 1  # writes BETWEEN completed observations are legal
        assert handle.sign("model") == {"fresh": 2}
        assert "release" not in events
        signature["weight"] = (2, 3, 4)
        with pytest.raises(producer.IpcTensorMutatedError, match="storage changed"):
            handle.sign("model")
    assert events.count("release") == 1 and not client.registrations()
    handle.close()
    assert events.count("release") == 1


@pytest.mark.parametrize("stage", ["import", "sign", "close"])
@pytest.mark.parametrize("failure", ["lost-response", "wrong-session", "interrupt"])
def test_lost_or_mismatched_ack_keeps_storage_recoverable(
    monkeypatch, fake_producer, stage, failure
):
    client, events, _, _ = fake_producer
    original = client._rpc

    def rpc(path, body):
        response = original(path, body)
        if path.endswith("/" + stage):
            if failure == "wrong-session":
                response["session"] = "b" * 64
            else:
                raise (KeyboardInterrupt if failure == "interrupt" else OSError)(
                    "lost ACK"
                )
        return response

    monkeypatch.setattr(client, "_rpc", rpc)
    with pytest.raises(
        KeyboardInterrupt
        if failure == "interrupt"
        else producer.RegistrationUncertainError
    ):
        handle = client.register_tensors({})
        if stage == "sign":
            handle.sign("model")
        elif stage == "close":
            handle.close()
    assert "release" not in events
    (recovered,) = client.registrations()
    with pytest.raises(RuntimeError, match="uncertain"):
        recovered.sign("model")
    monkeypatch.setattr(client, "_rpc", original)
    recovered.close()
    assert events.count("release") == 1 and not client.registrations()


def test_mutation_during_sign_discards_receipt_but_preserves_registration(
    monkeypatch, fake_producer
):
    client, events, tensor, _ = fake_producer
    original = client._rpc
    with client.register_tensors({}) as handle:

        def mutate(path, body):
            tensor.version += 1
            return original(path, body)

        monkeypatch.setattr(client, "_rpc", mutate)
        with pytest.raises(producer.IpcTensorMutatedError):
            handle.sign("model")
        assert "release" not in events and handle._state == "active"


def test_storage_swap_during_sign_is_detected_without_a_version_increment(
    monkeypatch, fake_producer
):
    client, events, _, signature = fake_producer
    original = client._rpc
    with client.register_tensors({}) as handle:

        def swap(path, body):
            signature["weight"] = (99, 99, 99)
            return original(path, body)

        monkeypatch.setattr(client, "_rpc", swap)
        with pytest.raises(producer.IpcTensorMutatedError, match="during measurement"):
            handle.sign("model")
        assert "release" not in events and handle._state == "active"


def test_stream_selection_validates_each_device_without_a_global_wait(monkeypatch):
    class Stream:
        def __init__(self, device):
            self.device = SimpleNamespace(index=device)

    fake = SimpleNamespace(cuda=SimpleNamespace(Stream=Stream, current_stream=Stream))
    monkeypatch.setitem(sys.modules, "torch", fake)
    state = {str(d): SimpleNamespace(device=SimpleNamespace(index=d)) for d in (0, 2)}
    assert set(producer._streams(state, None)) == {0, 2}
    for bad in (
        {0: Stream(0)},
        {0: Stream(0), 2: Stream(0)},
        {0: [], 2: Stream(2)},
        {False: Stream(0), 2: Stream(2)},
    ):
        with pytest.raises(ValueError, match="stream"):
            producer._streams(state, bad)


@pytest.mark.parametrize("pause_at", ["publication", "import-reply"])
@pytest.mark.parametrize("outcome", ["success", "lost-response", "interrupt"])
def test_recovery_close_waits_for_registration_creation(
    monkeypatch, fake_producer, pause_at, outcome
):
    client, events, _, _ = fake_producer
    original_rpc, original_claim = client._rpc, producer.claim_ipc_refs
    started, finish, closing = Event(), Event(), Event()

    def pause():
        started.set()
        assert finish.wait(5)

    def claim(refs):
        # This is the first step AFTER publication in Client.registrations(),
        # before the import RPC. Locking only around the RPC is too late.
        if pause_at == "publication":
            pause()
        return original_claim(refs)

    def rpc(path, body):
        response = original_rpc(path, body)
        if path.endswith("/import"):
            if pause_at == "import-reply":
                pause()
            events.append("import-outcome")
            if outcome != "success":
                raise (KeyboardInterrupt if outcome == "interrupt" else OSError)(
                    "injected import reply failure"
                )
        return response

    monkeypatch.setattr(producer, "claim_ipc_refs", claim)
    monkeypatch.setattr(client, "_rpc", rpc)

    def close(handle):
        closing.set()
        handle.close()

    with ThreadPoolExecutor(max_workers=2) as workers:
        creating = workers.submit(client.register_tensors, {})
        try:
            assert started.wait(5)
            (recovered,) = client.registrations()
            # Probe ownership instead of depending on a sleep to schedule the
            # race. Release a mistakenly unlocked lock even when the test fails.
            acquired = recovered._lock.acquire(blocking=False)
            if acquired:
                recovered._lock.release()
            assert not acquired, "creation must lock before publishing its handle"
            retirement = workers.submit(close, recovered)
            assert closing.wait(5)
            assert not retirement.done()
            assert recovered._refs and recovered._keepalive
            assert callable(recovered._provider)
            assert "/v1/registrations/close" not in events
            assert "release" not in events
        finally:
            finish.set()
        if outcome == "success":
            assert creating.result(timeout=5) is recovered
        else:
            with pytest.raises(
                KeyboardInterrupt
                if outcome == "interrupt"
                else producer.RegistrationUncertainError
            ) as caught:
                creating.result(timeout=5)
            if outcome == "lost-response":
                assert caught.value.registration is recovered
        retirement.result(timeout=5)

    assert events.index("import-outcome") < events.index("/v1/registrations/close")
    if outcome != "success":
        assert events.index("quarantine") < events.index("/v1/registrations/close")
    # Retirement may win immediately AFTER creation unlocks. It must never be
    # overwritten back to active by creation, nor invoke a cleared provider.
    assert recovered._state == "closed"
    assert not recovered._refs and not recovered._keepalive
    assert recovered._provider is None and not client.registrations()
    with pytest.raises(RuntimeError, match="closed or uncertain"):
        recovered.sign("model")
    recovered.close()
    assert events.count("release") == events.count("/v1/registrations/close") == 1


def test_close_cannot_retire_storage_while_an_observation_is_inflight(
    monkeypatch, fake_producer
):
    client, events, _, _ = fake_producer
    original = client._rpc
    started, finish, closing = Event(), Event(), Event()
    handle = client.register_tensors({})

    def rpc(path, body):
        if path.endswith("/sign"):
            started.set()
            assert finish.wait(5)
        return original(path, body)

    monkeypatch.setattr(client, "_rpc", rpc)

    def close():
        closing.set()
        handle.close()

    with ThreadPoolExecutor(max_workers=2) as workers:
        signing = workers.submit(handle.sign, "model")
        try:
            assert started.wait(5)
            # Check actual lock ownership, not a flaky elapsed-time assertion.
            assert not handle._lock.acquire(blocking=False)
            retirement = workers.submit(close)
            assert closing.wait(5)
            assert "release" not in events
        finally:
            finish.set()
        assert signing.result() == {"fresh": 1}
        retirement.result()
    assert events[-1] == "release"


def test_exception_allocation_failure_cannot_hide_registered_quarantine(monkeypatch):
    from cuattest.server import make_handler

    n = owner(ipc_close_fails=True, ctx_destroy_fails=True)
    manager = reg.RegistrationManager(n)
    body = request(manager, tensors=wire())
    manager.import_tensors(body)
    monkeypatch.setattr(reg, "RegistrationManager", lambda notary: manager)

    def allocation_failure(*args):
        raise MemoryError("allocating specialized IPC exception")

    monkeypatch.setattr(reg, "IpcCleanupUncertainError", allocation_failure)
    handler = object.__new__(make_handler(n))
    handler.path = "/v1/registrations/close"
    handler.server = SimpleNamespace(cuattest_fatal_error=None)
    handler._read_json = lambda: body
    sent = []
    handler._send = lambda *args: sent.append(args)
    handler.do_POST()
    assert not sent and handler.close_connection and n.ipc_cleanup_required
    assert isinstance(handler.server.cuattest_fatal_error, MemoryError)


def test_interrupted_local_retirement_can_be_retried_without_another_remote_close(monkeypatch, fake_producer):
    client, events, _, _ = fake_producer
    handle = client.register_tensors({})
    retire = handle._retire_local
    def interrupted():
        raise KeyboardInterrupt("after acknowledged close, before registry removal")
    monkeypatch.setattr(handle, "_retire_local", interrupted)
    with pytest.raises(KeyboardInterrupt):
        handle.close()
    assert handle._state == "closed" and handle in client.registrations()
    monkeypatch.setattr(handle, "_retire_local", retire)
    handle.close()
    assert not client.registrations()
    assert events.count("release") == events.count("/v1/registrations/close") == 1

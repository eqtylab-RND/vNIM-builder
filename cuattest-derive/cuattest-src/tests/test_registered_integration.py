# SPDX-License-Identifier: Apache-2.0
"""Real, separate-process registered IPC; CUATTEST_TEST_GPU=1 opts in.

Run again with CUATTEST_HASH_MODE=async on sm_80+ to force double buffering
through partial tiles, unaligned spans and both host backends. The same suite
uses every visible GPU, reversing service ordinals through ipc_test_service.
"""

import gc
import ctypes
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import pytest
from blake3 import blake3
from _ipc_test_service import ipc_test_service

from cuattest.expect import verify_evidence
from cuattest._cuda import Cuda, PinnedBuffer
from cuattest.ipc import (
    IpcTensorMutatedError,
    _active_allocation_lease_count,
    share_tensors,
)
from cuattest.registered import RegistrationUncertainError

pytestmark = pytest.mark.skipif(
    os.environ.get("CUATTEST_TEST_GPU") != "1", reason="set CUATTEST_TEST_GPU=1"
)


BACKENDS = ["native", "fallback"]
if os.environ.get("CUATTEST_TEST_REGISTERED_BACKEND"):
    selected = os.environ["CUATTEST_TEST_REGISTERED_BACKEND"]
    if selected not in BACKENDS:
        raise pytest.UsageError("CUATTEST_TEST_REGISTERED_BACKEND must be native or fallback")
    BACKENDS = [selected]


@pytest.fixture(scope="module", params=BACKENDS)
def service(request):
    with ipc_test_service(request.param) as setup:
        yield setup
        client, _, torch = setup
        assert not client.registrations()
    gc.collect()
    torch.cuda.empty_cache()


def expected(state):
    roots = b"".join(
        blake3(t.cpu().numpy().tobytes()).digest() for _, t in sorted(state.items())
    )
    return roots.hex(), blake3(struct.pack("<I", len(state)) + roots).hexdigest()


def assert_receipt(receipt, state, keys):
    verified = verify_evidence(receipt, trusted_pubkeys=keys)
    digests, root = expected(state)
    assert verified.digests == digests and verified.model_root == root


def test_registered_tail_spans_and_fresh_updates(service):
    client, keys, torch = service
    state, writers = {}, {}
    for device in range(torch.cuda.device_count()):
        writer = writers[device] = torch.cuda.Stream(device=device)
        with torch.cuda.stream(writer):
            source = torch.arange(4 * 1024 * 1024, dtype=torch.int64, device=device).to(
                torch.uint8
            )
            for offset, size in (
                (0, 1),
                (1, 1023),
                (0, 1024),
                (3, 1025),
                (0, 131072),
                (16, 131073),
                (0, 3 * 1024 * 1024 + 19),
            ):
                state[f"d{device}_{offset}_{size}"] = source[offset : offset + size]
    before = _active_allocation_lease_count()
    with client.register_tensors(state, streams=writers) as registered:
        assert _active_allocation_lease_count() > before
        assert_receipt(
            registered.sign("registered/tails", streams=writers), state, keys
        )
        for tensor in state.values():
            with torch.cuda.stream(writers[tensor.device.index]):
                tensor.add_(1)
        assert_receipt(
            registered.sign("registered/updated", streams=writers), state, keys
        )
    assert _active_allocation_lease_count() == before


def test_duplicate_imports_retain_the_original_registration(service):
    # CUDA refcounts repeated opens. Keep this case separate because Compute
    # Sanitizer 2026.2 memcheck forgets a live mapping at its FIRST close; the
    # raw CUDA-only ipc_refcount_probe.py reproduces that instrumentation bug.
    # Do not weaken production ownership or skip this ordinary GPU regression.
    client, keys, torch = service
    state = {
        f"d{d}": torch.ones(1025, dtype=torch.uint8, device=d)
        for d in range(torch.cuda.device_count())
    }
    with client.register_tensors(state) as registered:
        assert_receipt(registered.sign("registered/original"), state, keys)
        names, refs, keep = share_tensors(state)
        try:
            assert names == registered.names
            assert_receipt(client.sign(refs, "one-shot/overlap"), state, keys)
        finally:
            keep.release()
        assert_receipt(registered.sign("registered/after-oneshot"), state, keys)
        with client.register_tensors(state) as second:
            assert_receipt(second.sign("registered/second"), state, keys)
        assert_receipt(registered.sign("registered/after-second"), state, keys)


def test_registered_model_rejects_storage_replacement_and_lazy_views(service):
    client, keys, torch = service
    model = torch.nn.Module()
    model.register_buffer(
        "weight", torch.ones(1025, dtype=torch.uint8, device="cuda:0")
    )
    with client.register_model(model) as registered:
        assert_receipt(registered.sign("registered/model"), model.state_dict(), keys)
        original = model.weight
        model.weight = model.weight.clone()
        with pytest.raises(IpcTensorMutatedError, match="storage changed"):
            registered.sign("registered/replaced")
        model.weight = original
        model.weight.add_(3)
        assert_receipt(registered.sign("registered/restored"), model.state_dict(), keys)
    complex_tensor = torch.ones((4, 5), dtype=torch.complex64, device="cuda:0")
    before = _active_allocation_lease_count()
    for tensor in (
        complex_tensor.conj(),
        torch._neg_view(complex_tensor),
        complex_tensor.t(),
    ):
        with pytest.raises(ValueError, match="contiguous with resolved"):
            client.register_tensors({"lazy": tensor})
    assert _active_allocation_lease_count() == before


@pytest.mark.parametrize("operation", ["import", "sign", "close"])
def test_lost_real_response_can_retire_the_exact_ticket(
    service, monkeypatch, operation
):
    client, _, torch = service
    state = {"weight": torch.ones(1025, dtype=torch.uint8, device="cuda:0")}
    before = _active_allocation_lease_count()
    rpc = client._rpc

    def lose_response(path, body):
        reply = rpc(path, body)  # actually complete the consumer operation
        if path.endswith("/" + operation):
            raise OSError("injected lost response after remote completion")
        return reply

    monkeypatch.setattr(client, "_rpc", lose_response)
    with pytest.raises(RegistrationUncertainError) as caught:
        handle = client.register_tensors(state)
        if operation == "sign":
            handle.sign("registered/lost")
        elif operation == "close":
            handle.close()
    assert _active_allocation_lease_count() > before
    handle = caught.value.registration
    assert handle in client.registrations()
    monkeypatch.setattr(client, "_rpc", rpc)
    handle.close()  # repeated close after a lost ACK is safe and idempotent
    assert _active_allocation_lease_count() == before


@pytest.mark.parametrize("outcome", ["success", "lost-response", "interrupt"])
def test_recovery_close_waits_for_real_import_reply(service, monkeypatch, outcome):
    client, _, torch = service
    state = {
        f"d{d}": torch.ones(1025, dtype=torch.uint8, device=d)
        for d in range(torch.cuda.device_count())
    }
    streams = {
        d: torch.cuda.current_stream(d) for d in range(torch.cuda.device_count())
    }
    before = _active_allocation_lease_count()
    rpc = client._rpc
    imported, resume, closing = threading.Event(), threading.Event(), threading.Event()
    remote_close = threading.Event()

    def delayed_reply(path, body):
        if path.endswith("/close"):
            remote_close.set()
        reply = rpc(path, body)
        if path.endswith("/import"):
            # The separate-process GPU service really owns these mappings now,
            # but create() has not accepted its reply or activated the handle.
            imported.set()
            assert resume.wait(15)
            if outcome != "success":
                raise (KeyboardInterrupt if outcome == "interrupt" else OSError)(
                    "injected import reply failure after remote completion"
                )
        return reply

    monkeypatch.setattr(client, "_rpc", delayed_reply)

    def close(handle):
        closing.set()
        handle.close()

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            creating = workers.submit(client.register_tensors, state, streams=streams)
            try:
                assert imported.wait(15)
                (recovered,) = client.registrations()
                acquired = recovered._lock.acquire(blocking=False)
                if acquired:
                    recovered._lock.release()
                assert not acquired, "pending import must serialize recovery close"
                retirement = workers.submit(close, recovered)
                assert closing.wait(15)
                assert not retirement.done() and not remote_close.is_set()
                assert _active_allocation_lease_count() > before
                assert recovered._refs and recovered._keepalive
                assert callable(recovered._provider)
            finally:
                resume.set()
            if outcome == "success":
                assert creating.result(timeout=15) is recovered
            else:
                with pytest.raises(
                    KeyboardInterrupt
                    if outcome == "interrupt"
                    else RegistrationUncertainError
                ) as caught:
                    creating.result(timeout=15)
                if outcome == "lost-response":
                    assert caught.value.registration is recovered
            retirement.result(timeout=15)
        assert remote_close.is_set()
        assert recovered._state == "closed" and recovered._provider is None
        assert not recovered._refs and not recovered._keepalive
        assert not client.registrations()
        assert _active_allocation_lease_count() == before
        # A concurrent close may retire the just-created handle, but creation
        # must not resurrect it and cause sign() to call a cleared provider.
        with pytest.raises(RuntimeError, match="closed or uncertain"):
            recovered.sign("registered/retired-during-creation")
        recovered.close()
    finally:
        monkeypatch.setattr(client, "_rpc", rpc)
        for handle in client.registrations():
            handle.close()


def test_registered_sign_does_not_wait_for_unrelated_producer_stream(service):
    client, keys, torch = service
    tensor = torch.ones(1025, dtype=torch.uint8, device="cuda:0")
    state = {"weight": tensor}
    cu = Cuda()
    cu.init()
    with (
        torch.cuda.device(0),
        client.register_tensors(state) as handle,
        closing(PinnedBuffer(cu, 4)) as gate,
    ):
        assert_receipt(handle.sign("registered/warmup"), state, keys)
        gate.write(bytes(4))
        other = cu.stream_create()
        wait = cu.lib.cuStreamWaitValue32_v2
        wait.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        wait.restype = ctypes.c_int
        query = cu.lib.cuStreamQuery
        query.argtypes, query.restype = [ctypes.c_void_p], ctypes.c_int
        pointer = cu.lib.cuMemHostGetDevicePointer_v2
        pointer.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        pointer.restype = ctypes.c_int
        device_gate = ctypes.c_ulonglong()
        cu.check(pointer(ctypes.byref(device_gate), gate.ptr, 0), "mapped stream gate")
        watchdog_fired = threading.Event()

        def release():
            watchdog_fired.set()
            gate.write(b"\x01\0\0\0")

        watchdog = threading.Timer(15, release)
        try:
            watchdog.start()
            cu.check(wait(other, device_gate.value, 1, 1), "producer stream wait")
            assert query(other) == 600
            assert_receipt(handle.sign("registered/isolated"), state, keys)
            assert not watchdog_fired.is_set(), (
                "registration waited on unrelated CUDA work"
            )
            assert query(other) == 600
        finally:
            gate.write(b"\x01\0\0\0")
            watchdog.cancel()
            watchdog.join()
            cu.stream_sync(other)
            cu.stream_destroy(other)

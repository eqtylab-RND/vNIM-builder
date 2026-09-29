# SPDX-License-Identifier: Apache-2.0
"""Real stream isolation; enable with CUATTEST_TEST_GPU=1."""

import ctypes
import json
import os
import struct
import threading
from contextlib import closing

import pytest

from cuattest._cuda import CudaError, DeviceBuffer, PinnedBuffer
from cuattest._hosthash import blake3_digest
from cuattest.expect import verify_evidence
from cuattest.notary import Notary


pytestmark = pytest.mark.skipif(
    os.environ.get("CUATTEST_TEST_GPU") != "1", reason="set CUATTEST_TEST_GPU=1"
)


@pytest.mark.parametrize("backend", ["native", "fallback"])
@pytest.mark.parametrize("blocked_stream", ["legacy-default", "unrelated"])
def test_signed_requests_complete_while_another_stream_is_blocked(
    monkeypatch, backend, blocked_stream
):
    monkeypatch.setenv("CUATTEST_DISABLE_NATIVE_HOST", str(int(backend == "fallback")))
    with closing(Notary()) as notary, notary._activate():
        assert (notary._native_runner is None) == (backend == "fallback")
        cu = notary.cu
        data = bytes(range(256)) * 17
        with closing(DeviceBuffer.from_bytes(cu, data)) as source, closing(PinnedBuffer(cu, 4)) as gate:
            gate.write(bytes(4))
            other = cu.stream_create() if blocked_stream == "unrelated" else None
            wait = cu.lib.cuStreamWaitValue32_v2
            wait.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_uint]
            wait.restype = ctypes.c_int
            query = cu.lib.cuStreamQuery
            query.argtypes, query.restype = [ctypes.c_void_p], ctypes.c_int
            get_pointer = cu.lib.cuMemHostGetDevicePointer_v2
            get_pointer.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p, ctypes.c_uint]
            get_pointer.restype = ctypes.c_int
            device_gate = ctypes.c_ulonglong()
            cu.check(get_pointer(ctypes.byref(device_gate), gate.ptr, 0), "mapped gate")
            released_by_watchdog = threading.Event()

            def release_watchdog():
                released_by_watchdog.set()
                gate.write(b"\x01\0\0\0")

            watchdog = threading.Timer(10, release_watchdog)
            watchdog.daemon = True
            try:
                # Cold allocation/growth is allowed to synchronize. Warm the
                # largest shape first; steady-state/shrinking calls must not.
                notary._launch_fused_active(
                    [(source.ptr, len(data))] * 4, "2026-09-08T00:00:00Z", b"stream-test"
                )
                watchdog.start()
                # A stream memory wait uses no SMs, unlike a spin kernel that
                # could prevent cooperative-grid residency and obscure the bug.
                # The HOST releases it: no hidden cross-stream GPU dependency.
                cu.check(wait(other, device_gate.value, 1, 1), "stream gate wait")
                assert query(other) == 600  # CUDA_ERROR_NOT_READY
                for count, nbytes in ((1, 1023), (4, 1025)):
                    result = notary._launch_fused_active(
                        [(source.ptr, nbytes)] * count,
                        "2026-09-08T00:00:00Z", b"stream-test"
                    )
                    expected = blake3_digest(data[:nbytes]) * count
                    root = blake3_digest(struct.pack("<I", count) + expected)
                    assert result.roots == expected and result.model_root == root
                    receipt = json.loads(result.receipt)
                    receipt["gpu_pubkey_uncompressed"] = notary.info.gpu_pubkey_uncompressed
                    verified = verify_evidence(receipt, trusted_pubkey=notary.info.gpu_pubkey_uncompressed)
                    assert verified.model_root == root.hex()
                    assert not released_by_watchdog.is_set(), "request waited for unrelated work"
                    assert query(other) == 600, "request synchronized the other stream"
            finally:
                gate.write(b"\x01\0\0\0")
                watchdog.cancel()
                if watchdog.ident is not None:
                    watchdog.join()
                cu.stream_sync(other)
                if other is not None:
                    cu.stream_destroy(other)


@pytest.mark.parametrize("backend", ["native", "fallback"])
@pytest.mark.parametrize("producer", ["legacy-memset", "explicit-stream"])
def test_direct_hash_waits_for_producer_without_waiting_for_unrelated_work(
    monkeypatch, backend, producer
):
    monkeypatch.setenv("CUATTEST_DISABLE_NATIVE_HOST", str(int(backend == "fallback")))
    with closing(Notary()) as notary, notary._activate():
        assert (notary._native_runner is None) == (backend == "fallback")
        cu = notary.cu
        data = b"\xA5" * 1025  # include a partial BLAKE3 chunk
        with closing(DeviceBuffer.from_bytes(cu, data)) as source, closing(PinnedBuffer(cu, 8)) as gate:
            gate.write(bytes(8))
            producer_stream = cu.stream_create() if producer == "explicit-stream" else None
            other = cu.stream_create()
            wait = cu.lib.cuStreamWaitValue32_v2
            wait.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint, ctypes.c_uint]
            wait.restype = ctypes.c_int
            query = cu.lib.cuStreamQuery
            query.argtypes, query.restype = [ctypes.c_void_p], ctypes.c_int
            get_pointer = cu.lib.cuMemHostGetDevicePointer_v2
            get_pointer.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p, ctypes.c_uint]
            get_pointer.restype = ctypes.c_int
            device_gate = ctypes.c_ulonglong()
            cu.check(get_pointer(ctypes.byref(device_gate), gate.ptr, 0), "mapped gates")
            released_by_watchdog = threading.Event()

            def release_watchdog():
                released_by_watchdog.set()
                gate.write(struct.pack("<II", 1, 1))

            watchdog = threading.Timer(10, release_watchdog)
            watchdog.daemon = True
            try:
                assert notary.hash_dptr(source.ptr, len(data)) == blake3_digest(data)
                launch = notary._launch_fused_active

                def launch_after_handoff(*args, **kwargs):
                    # Deterministic ordering check BEFORE submitting the kernel:
                    # the producer is gated and the private stream must already
                    # be waiting on its event. No elapsed-time race is needed.
                    # A host/global wait deadlocks here until the watchdog fires.
                    assert not released_by_watchdog.is_set(), "handoff blocked the host"
                    assert query(producer_stream) == 600
                    assert query(notary._stream) == 600, "missing producer dependency"
                    gate.write(struct.pack("<II", 1, 0))  # release only the writer
                    return launch(*args, **kwargs)

                monkeypatch.setattr(notary, "_launch_fused_active", launch_after_handoff)
                watchdog.start()
                cu.check(wait(other, device_gate.value + 4, 1, 1), "unrelated wait")
                cu.check(wait(producer_stream, device_gate.value, 1, 1), "producer wait")
                if producer_stream is None:
                    # The apparently synchronous legacy API still queues device
                    # work: this is the original stale-digest reproduction.
                    cu.memset0(source.ptr, len(data))
                    kwargs = {}
                else:
                    cu.memset0_async(source.ptr, len(data), producer_stream)
                    kwargs = {"producer_stream": producer_stream}
                assert query(producer_stream) == 600
                assert notary.hash_dptr(source.ptr, len(data), **kwargs) == blake3_digest(bytes(len(data)))
                assert not released_by_watchdog.is_set()
                assert query(other) == 600, "hash synchronized unrelated work"
            finally:
                gate.write(struct.pack("<II", 1, 1))
                watchdog.cancel()
                if watchdog.ident is not None:
                    watchdog.join()
                for stream in (producer_stream, other):
                    cu.stream_sync(stream)
                    if stream is not None:
                        cu.stream_destroy(stream)


def test_real_fallback_context_is_destroyed_after_pinned_free_error(monkeypatch):
    monkeypatch.setenv("CUATTEST_DISABLE_NATIVE_HOST", "1")
    with closing(Notary()) as notary:
        assert notary._native_runner is None
        data = b"pinned cleanup regression" * 100
        assert notary.hash_bytes(data) == blake3_digest(data)
        buffers = list(notary._fallback_host_buffers.values())
        assert len(buffers) == 2
        previous = notary.cu.ctx_get_current().value
        destroyed = []
        destroy = notary.cu.ctx_destroy

        def destroy_context(ctx):
            assert all(not buffer.ptr for buffer in buffers)
            destroy(ctx)  # real CUDA destruction, not just a fake call counter
            destroyed.append(ctx)

        def fail_free(ptr):
            # Deliberately omit the driver free. Memcheck's leak checker will
            # report these buffers at context destruction; this error-path test
            # verifies reclamation by the context, not explicit free success.
            raise CudaError("injected cuMemFreeHost failure")

        monkeypatch.setattr(notary.cu, "ctx_destroy", destroy_context)
        monkeypatch.setattr(notary.cu, "host_free", fail_free)
        with pytest.raises(CudaError, match="injected cuMemFreeHost failure"):
            notary.close()
        assert len(destroyed) == 1 and notary._context_destroyed
        assert notary.cu.ctx_get_current().value == previous
        for buffer in buffers:
            buffer.close()  # detached wrappers cannot issue another host free

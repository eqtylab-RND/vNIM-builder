# SPDX-License-Identifier: Apache-2.0
"""Stream ABI and pinned staging regressions; no CUDA driver required."""

import ctypes

import pytest

from cuattest._cuda import Cuda, CudaError, DeviceBuffer, PinnedBuffer


def test_notary_stream_is_nonblocking_and_launches_preserve_it():
    cuda = object.__new__(Cuda)
    events = []

    def create(pointer, flags):
        assert flags == 1  # CU_STREAM_NON_BLOCKING, not ordinary stream flags=0
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 41
        return 0

    def launch(*args):
        assert args[8].value == 41
        assert ctypes.cast(args[9][0], ctypes.POINTER(ctypes.c_int))[0] == 17
        events.append("launch")
        return 0

    def synchronize(stream):
        assert stream.value == 41
        events.append("sync")
        return 0

    cuda.cuStreamCreate = create
    cuda.cuLaunchKernel = cuda.cuLaunchCooperativeKernel = launch
    cuda.cuStreamSynchronize = synchronize
    cuda.cuCtxSynchronize = lambda: pytest.fail("context-wide wait")
    stream = cuda.stream_create()
    cuda.launch(1, 1, 128, [ctypes.c_int(17)], stream=stream)
    cuda.launch_cooperative(1, 1, 128, [ctypes.c_int(17)], stream=stream)
    cuda.stream_sync(stream)
    assert events == ["launch", "launch", "sync"]


@pytest.mark.parametrize("producer", [None, 42])
@pytest.mark.parametrize("failure", [None, "create", "record", "wait", "destroy"])
@pytest.mark.parametrize("interrupted", [False, True])
def test_producer_handoff_is_device_ordered_and_retires_event(producer, failure, interrupted):
    cuda = object.__new__(Cuda)
    events = []

    def result(stage):
        events.append(stage)
        if stage == failure:
            if interrupted:
                raise KeyboardInterrupt("after accepted " + stage)
            return 1
        return 0

    def create(pointer, flags):
        assert flags == 2  # CU_EVENT_DISABLE_TIMING; no timestamp overhead
        if failure != "create" or interrupted:
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = 99
        return result("create")

    def record(event, stream):
        assert event.value == 99 and stream == producer
        return result("record")

    def wait(stream, event, flags):
        assert stream == 41 and event.value == 99 and flags == 0
        return result("wait")

    def destroy(event):
        assert event.value == 99
        return result("destroy")

    def check(code, label):
        if code:
            raise CudaError(label)

    cuda.check = check
    cuda.cuEventCreate, cuda.cuEventRecord = create, record
    cuda.cuStreamWaitEvent, cuda.cuEventDestroy_v2 = wait, destroy
    cuda.cuStreamSynchronize = cuda.cuCtxSynchronize = lambda *a: pytest.fail("host wait")
    if failure is None:
        cuda.stream_wait_stream(41, producer)
    else:
        with pytest.raises(KeyboardInterrupt if interrupted else CudaError):
            cuda.stream_wait_stream(41, producer)
    stages = ["create", "record", "wait", "destroy"]
    expected = stages if failure is None else stages[:stages.index(failure) + 1]
    if failure in {"record", "wait"} or (failure == "create" and interrupted):
        expected += ["destroy"]
    assert events == expected


def test_pinned_staging_bounds_and_explicit_retirement():
    class Host:
        def host_alloc(self, size):
            self.data = ctypes.create_string_buffer(size)
            return ctypes.cast(self.data, ctypes.c_void_p)

        def host_free(self, pointer):
            self.freed += 1

        freed = 0

    host = Host()
    buffer = PinnedBuffer(host, 8)
    buffer.write(b"abcdefgh")
    assert buffer.read() == b"abcdefgh"
    assert buffer.read(3) == b"abc"
    for size in (-1, 9):
        with pytest.raises(ValueError, match="capacity"):
            buffer.read(size)
    with pytest.raises(ValueError, match="capacity"):
        buffer.write(b"abcdefghi")
    buffer.close()
    buffer.close()
    assert host.freed == 1
    # CUDA, not Python's garbage collector, owns a quarantined DMA allocation.
    assert "__del__" not in PinnedBuffer.__dict__


@pytest.mark.parametrize("buffer_class", [DeviceBuffer, PinnedBuffer])
def test_interruption_after_free_never_retries_the_same_pointer(buffer_class):
    class Driver:
        def alloc(self, size):
            return 42

        host_alloc = alloc
        frees = 0

        def free(self, ptr):
            assert ptr == 42
            self.frees += 1
            raise KeyboardInterrupt("after successful driver free")

        host_free = free

    driver = Driver()
    buffer = buffer_class(driver, 8)
    with pytest.raises(KeyboardInterrupt):
        buffer.close()
    buffer.close()
    assert not buffer.ptr and driver.frees == 1

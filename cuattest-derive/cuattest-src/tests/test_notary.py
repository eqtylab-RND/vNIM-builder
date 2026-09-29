# SPDX-License-Identifier: Apache-2.0
"""Notary validation and lifecycle regressions that need no CUDA device."""

import base64
import inspect
import json
import sys
from types import SimpleNamespace

import pytest

from cuattest import _statements as statements_module
from cuattest import ids
from cuattest import notary as notary_module
from cuattest._cuda import CudaError, IpcImportRejectedError, PinnedBuffer
from cuattest._hosthash import blake3_digest
from cuattest._protocol import (
    MEASUREMENT_CLAIM,
    MEASUREMENT_HASH_SCHEME,
    MEASUREMENT_OPERATION,
)
from cuattest.notary import (
    GpuCleanupUncertainError,
    GpuSessionAbortedError,
    IpcCleanupUncertainError,
    IpcSessionAbortedError,
    Measurement,
    Notary,
    NotaryError,
    TensorRef,
    _fused_plan,
    _FusedResult,
    _validated_model,
)
from cuattest.server import make_handler

RAW = bytes(range(64))
STREAM = 0xA11


def install_staging(monkeypatch, cuda, events, output=b""):
    """Deterministic DMA double: output is unreadable until its stream drains."""
    instances = []

    class Pinned:
        def __init__(self, cuda, nbytes):
            self.ptr = self
            self.nbytes = nbytes
            self.closed = False
            instances.append(self)

        def write(self, data):
            assert len(data) == self.nbytes

        def read(self, nbytes):
            assert events.index(("download",)) < events.index(("sync",))
            assert nbytes == len(output) and nbytes <= self.nbytes
            events.append(("read", 3, nbytes))
            return output

        def close(self):
            self.closed = True
            events.append(("host_close",))

    def upload(dst, host, nbytes, stream):
        assert stream == STREAM and host.nbytes == nbytes
        events.append(("upload",))

    def download(host, src, nbytes, stream):
        assert stream == STREAM and host.nbytes == nbytes
        events.append(("download",))

    monkeypatch.setattr(notary_module, "PinnedBuffer", Pinned)
    cuda.htod_async = upload
    cuda.dtoh_async = download
    return instances


class ImportedCuda:
    def __init__(
        self,
        allocation_nbytes=64,
        allocation_device=0,
        ipc_close_fails=False,
        ctx_destroy_fails=False,
        module_unload_fails=False,
    ):
        self.allocation_nbytes = allocation_nbytes
        self.allocation_device = allocation_device
        self.ipc_close_fails = ipc_close_fails
        self.ctx_destroy_fails = ctx_destroy_fails
        self.module_unload_fails = module_unload_fails
        self.opened = []
        self.closed = []
        self.close_attempts = []
        self.destroyed = []
        self.unload_attempts = []
        self.current = None
        self.context_switches = []

    def ipc_open(self, raw):
        self.opened.append(raw)
        return 0x1000

    def address_range(self, ptr):
        return 0x1000, self.allocation_nbytes

    def pointer_device(self, ptr):
        return self.allocation_device

    def ipc_close(self, ptr):
        self.close_attempts.append(ptr)
        if self.ipc_close_fails:
            raise CudaError("cuIpcCloseMemHandle failed")
        self.closed.append(ptr)

    def ctx_destroy(self, ctx):
        self.destroyed.append(ctx)
        if self.ctx_destroy_fails:
            raise CudaError("cuCtxDestroy failed")
        if self.current == ctx:
            # CUDA pops a successfully destroyed current context.
            self.current = None

    def module_unload(self, module):
        self.unload_attempts.append(module)
        if self.module_unload_fails:
            raise CudaError("cuModuleUnload failed")

    def stream_destroy(self, stream):
        assert stream == STREAM

    def ctx_get_current(self):
        return self.current

    def ctx_set_current(self, ctx):
        self.context_switches.append(ctx)
        self.current = ctx


def bare_notary(cuda):
    notary = object.__new__(Notary)
    notary._closed = False
    notary._context_destroyed = False
    notary.ctx = object()
    notary._stream = STREAM
    notary._fallback_device_buffers = {}
    notary._fallback_host_buffers = {}
    notary.device_ordinal = 0
    notary._fused_grid_limit = 1
    notary.cu = cuda
    cuda.current = notary.ctx
    return notary


def test_model_validation_reports_non_ascii_as_a_request_error():
    with pytest.raises(NotaryError, match="ASCII"):
        _validated_model("model/\ud800")


def test_host_entropy_is_exactly_one_256_bit_os_csprng_sample(monkeypatch):
    expected_seed = bytes(range(32))
    requested_sizes = []

    def fake_urandom(size):
        requested_sizes.append(size)
        assert size == len(expected_seed)
        return expected_seed

    monkeypatch.setattr(notary_module.os, "urandom", fake_urandom)
    monkeypatch.setattr(
        notary_module.time,
        "perf_counter_ns",
        lambda: pytest.fail("timer jitter must not be treated as key entropy"),
    )

    assert notary_module.host_entropy() == expected_seed
    assert requested_sizes == [32]


@pytest.mark.parametrize("stage", [None, "launch", "scrub", "sync"])
@pytest.mark.parametrize("drain_fails", [False, True])
def test_keygen_scrubs_and_drains_its_own_stream_before_free(
    monkeypatch, stage, drain_fails
):
    events, buffers = [], []

    class Buffer:
        def __init__(self, cuda, nbytes):
            self.ptr = len(buffers) + 10
            self.closed = False
            buffers.append(self)

        @classmethod
        def from_bytes(cls, cuda, data):
            return cls(cuda, len(data))

        def read(self, size):
            assert events[-1] == "sync"
            return b"\x01" * size

        def close(self):
            self.closed = True
            self.ptr = 0

    class Driver:
        def launch(self, *args, stream):
            assert stream == STREAM
            events.append("launch")
            if stage == "launch":
                raise CudaError("injected launch")

        def memset0_async(self, ptr, size, stream):
            assert stream == STREAM and ptr == buffers[0].ptr and size == 32
            events.append("scrub")
            if stage == "scrub":
                raise CudaError("injected scrub")

        def stream_sync(self, stream):
            assert stream == STREAM
            events.append("sync")
            if drain_fails or (stage == "sync" and events.count("sync") == 1):
                raise CudaError("injected sync")

    n = bare_notary(Driver())
    n._fn = {"keygen_kernel": 1}
    n._ctx_dev = SimpleNamespace(ptr=1)
    monkeypatch.setattr(notary_module, "DeviceBuffer", Buffer)
    if stage is not None or drain_fails:
        with pytest.raises(CudaError):
            n._keygen()
    else:
        assert n._keygen() == (b"\x01" * 32,) * 2
        assert events == ["launch", "scrub", "sync"]
    assert all(b.closed == (not drain_fails) for b in buffers)
    assert all(b.ptr == 0 for b in buffers)
    assert bool(getattr(n, "_abandon_context_allocations", False)) == drain_fails


@pytest.mark.parametrize("destroy_fails", [False, True])
@pytest.mark.parametrize("unload_fails", [False, True])
def test_failed_stream_teardown_is_not_failed_context_destruction(destroy_fails, unload_fails):
    class Driver(ImportedCuda):
        def stream_destroy(self, stream):
            assert stream == STREAM
            raise CudaError("injected stream teardown failure")

    cu = Driver(ctx_destroy_fails=destroy_fails, module_unload_fails=unload_fails)
    n = bare_notary(cu)
    n.module = "module"
    n._fallback_ipc_cleanup_required = True
    # Model failed IPC close with all queued work already drained. Stream and
    # module teardown errors must not conceal a successful final destruction.
    expected = IpcCleanupUncertainError if destroy_fails else IpcSessionAbortedError
    with pytest.raises(expected):
        n._abort_uncertain_fallback_ipc(CudaError("injected IPC close failure"))
    assert len(cu.destroyed) == 1 and cu.unload_attempts == ["module"]
    assert n._context_destroyed == (not destroy_fails)
    assert n.ipc_cleanup_required == destroy_fails


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_request_tensors", 0),
        ("max_request_bytes", True),
        ("max_request_tiles", notary_module._MAX_U64 + 1),
    ],
)
def test_invalid_request_limits_are_rejected_before_cuda_load(
    monkeypatch, keyword, value
):
    monkeypatch.setattr(
        notary_module,
        "Cuda",
        lambda: pytest.fail("invalid service limits must not initialize CUDA"),
    )

    with pytest.raises(NotaryError, match=keyword):
        Notary(**{keyword: value})


def test_fused_plan_packs_disjoint_tile_workspaces_and_reduction_levels():
    plan = _fused_plan(
        [
            (0x1000, 1),
            (0x2000, 129 * 1024),
            (0x3000, 300 * 1024),
        ]
    )

    descriptors = [
        notary_module._TENSOR_SPAN.unpack_from(
            plan.descriptors, i * notary_module._TENSOR_SPAN.size
        )
        for i in range(3)
    ]
    offsets = [
        int.from_bytes(plan.reduction_offsets[i : i + 8], "little")
        for i in range(0, len(plan.reduction_offsets), 8)
    ]

    assert descriptors == [
        (0x1000, 1, 0, 0, 1),
        (0x2000, 129 * 1024, 1, 0, 129),
        (0x3000, 300 * 1024, 3, 1, 300),
    ]
    assert plan.total_tiles == 6
    assert plan.secondary_tiles == 3
    assert plan.levels == 2
    assert offsets == [
        0,
        0,
        1,
        3,
        0,
        0,
        0,
        1,
    ]


def test_fused_plan_limits_96_gib_scratch_to_36_mib():
    plan = _fused_plan([(0x1000, 96 * 1024**3)])

    assert plan.total_tiles == 786_432
    assert plan.secondary_tiles == 393_216
    assert (plan.total_tiles + plan.secondary_tiles) * 32 == 36 * 1024**2


def test_fused_plan_rejects_a_workspace_size_that_cannot_be_addressed():
    with pytest.raises(NotaryError, match="workspace exceeds"):
        _fused_plan([(1, notary_module._MAX_U64)] * 4096)


@pytest.mark.parametrize("async_threshold", [None, 1000 * 1024, 1000 * 1024 + 1])
def test_fused_hash_is_one_occupancy_bounded_launch_and_one_read(monkeypatch, async_threshold):
    events = []
    expected_roots = b"\x11" * 32
    expected_model_root = b"\x22" * 32
    # Unsigned output contains only roots, model root, and initialized status;
    # there is no unwritten receipt-length slot in the DtoH copy.
    output_bytes = expected_roots + expected_model_root + bytes(4)
    instances = []

    class Buffer:
        next_ptr = 0x10000

        def __init__(self, cuda, nbytes):
            self.ptr = Buffer.next_ptr
            Buffer.next_ptr += max(nbytes, 1) + 0x100
            self.nbytes = nbytes
            self.index = len(instances)
            self.closed = False
            instances.append(self)

        @classmethod
        def from_bytes(cls, cuda, data):
            return cls(cuda, len(data))

        def read(self, nbytes):
            events.append(("read", self.index, nbytes))
            assert self.index == 3 and nbytes == len(output_bytes)
            return output_bytes

        def close(self):
            self.closed = True

    class LaunchCuda:
        def launch_cooperative(self, function, grid, block, args, *, stream):
            assert stream == STREAM
            events.append(
                (
                    "cooperative",
                    function,
                    grid,
                    block,
                    tuple(argument.value for argument in args),
                )
            )

        def launch(self, function, grid, block, args, *, stream):
            assert stream == STREAM
            events.append(
                (
                    "launch",
                    function,
                    grid,
                    block,
                    tuple(argument.value for argument in args),
                )
            )

        def stream_sync(self, stream):
            assert stream == STREAM
            events.append(("sync",))

    cuda = LaunchCuda()
    install_staging(monkeypatch, cuda, events, output_bytes)
    notary = bare_notary(cuda)
    notary._fused_grid_limit = 2
    notary._async_min_bytes = async_threshold
    notary._async_grid_limit = 1
    notary._fn = {
        "measure_model_fused_kernel": "measure",
        "measure_model_fused_async_kernel": "async-measure",
        "attest_measured_kernel": "attest",
    }
    notary._ctx_dev = SimpleNamespace(ptr=1)
    notary._cubin_digest_dev = SimpleNamespace(ptr=2)
    notary._kernel_digest_dev = SimpleNamespace(ptr=3)
    monkeypatch.setattr(notary_module, "DeviceBuffer", Buffer)

    result = notary._launch_fused_active([(0x9000, 1000 * 1024)])

    launches = [event for event in events if event[0] == "cooperative"]
    assert len(launches) == 1
    _, function, grid, block, args = launches[0]
    # The high-shared-memory variant has a DIFFERENT cooperative limit. Its
    # threshold is inclusive and must not reuse the standard kernel's grid.
    expected_launch = ("async-measure", 1, 128) if async_threshold == 1000 * 1024 else ("measure", 2, 128)
    assert (function, grid, block) == expected_launch
    assert args[1] == 1  # tensor count
    assert args[3] == 3  # ceil(log2(ceil(1000 / 128))) global levels
    assert args[4] == 8  # 128 KiB scheduling tiles
    assert events.count(("sync",)) == 1
    assert [e[0] for e in events[:5]] == ["upload", "cooperative", "download", "sync", "read"]
    assert [event for event in events if event[0] == "read"] == [
        ("read", 3, len(output_bytes))
    ]
    assert all(not buffer.closed for buffer in instances)
    assert result == _FusedResult(expected_roots, expected_model_root, None)


def test_signing_uses_private_handoff_and_one_host_synchronization(monkeypatch):
    events = []
    expected_roots = b"\x11" * 32
    expected_model_root = b"\x22" * 32
    receipt = b"{}"
    raw_output = bytearray(32 + 32 + notary_module._OUT_CAP + 8)
    raw_output[:32] = expected_roots
    raw_output[32:64] = expected_model_root
    raw_output[64 : 64 + len(receipt)] = receipt
    out_len_offset = 64 + notary_module._OUT_CAP
    raw_output[out_len_offset : out_len_offset + 4] = len(receipt).to_bytes(4, "little")
    instances = []

    class Buffer:
        next_ptr = 0x20000

        def __init__(self, cuda, nbytes):
            self.ptr = Buffer.next_ptr
            Buffer.next_ptr += max(nbytes, 1) + 0x100
            self.index = len(instances)
            self.nbytes = nbytes
            self.closed = False
            instances.append(self)

        @classmethod
        def from_bytes(cls, cuda, data):
            return cls(cuda, len(data))

        def read(self, nbytes):
            events.append(("read", self.index, nbytes))
            assert self.index == 3 and nbytes == len(raw_output)
            return bytes(raw_output)

        def close(self):
            self.closed = True

    class LaunchCuda:
        def launch_cooperative(self, function, grid, block, args, *, stream):
            assert stream == STREAM
            events.append(
                (
                    "cooperative",
                    function,
                    grid,
                    block,
                    tuple(argument.value for argument in args),
                )
            )

        def launch(self, function, grid, block, args, *, stream):
            assert stream == STREAM
            events.append(
                (
                    "launch",
                    function,
                    grid,
                    block,
                    tuple(argument.value for argument in args),
                )
            )

        def stream_sync(self, stream):
            assert stream == STREAM
            events.append(("sync",))

    notary = bare_notary(LaunchCuda())
    install_staging(monkeypatch, notary.cu, events, bytes(raw_output))
    notary._fused_grid_limit = 2
    notary._fn = {
        "measure_model_fused_kernel": "measure",
        "attest_measured_kernel": "attest",
    }
    notary._ctx_dev = SimpleNamespace(ptr=1)
    notary._cubin_digest_dev = SimpleNamespace(ptr=2)
    notary._kernel_digest_dev = SimpleNamespace(ptr=3)
    monkeypatch.setattr(notary_module, "DeviceBuffer", Buffer)

    result = notary._launch_fused_active(
        [(0x9000, 1000 * 1024)],
        "2026-09-04T12:00:00Z",
        b"test-model",
    )

    launches = [event for event in events if event[0] in {"cooperative", "launch"}]
    assert [(event[0], event[1], event[2], event[3]) for event in launches] == [
        ("cooperative", "measure", 2, 128),
        ("launch", "attest", 1, 128),
    ]
    measure_args = launches[0][4]
    attest_args = launches[1][4]
    assert measure_args[9] == 1  # arm the module-private one-shot root handoff
    assert measure_args[8] not in attest_args  # no host-provided model-root pointer
    assert events.count(("sync",)) == 1
    assert [e[0] for e in events[:6]] == [
        "upload", "cooperative", "launch", "download", "sync", "read"
    ]
    assert [event for event in events if event[0] == "read"] == [
        ("read", 3, len(raw_output))
    ]
    assert all(not buffer.closed for buffer in instances)
    assert result == _FusedResult(expected_roots, expected_model_root, "{}")


@pytest.mark.parametrize("failure_stage", ["measurement", "attestation"])
def test_python_fallback_conservatively_drains_when_launch_reports_failure(
    monkeypatch, failure_stage
):
    events = []
    instances = []

    class Buffer:
        next_ptr = 0x30000

        def __init__(self, cuda, nbytes):
            self.ptr = Buffer.next_ptr
            Buffer.next_ptr += max(nbytes, 1) + 0x100
            self.name = f"buffer-{len(instances)}"
            self.nbytes = nbytes
            instances.append(self)

        @classmethod
        def from_bytes(cls, cuda, data):
            return cls(cuda, len(data))

        def close(self):
            events.append(("close", self.name))
            self.ptr = 0

    class FailingLaunchCuda:
        current = None

        def launch_cooperative(self, function, grid, block, args, *, stream):
            assert stream == STREAM
            events.append(("launch", "measurement"))
            if failure_stage == "measurement":
                # Model a driver wrapper that enqueued work before surfacing
                # an exception at the Python call-return boundary.
                raise CudaError("injected measurement launch failure")

        def launch(self, function, grid, block, args, *, stream):
            assert stream == STREAM
            events.append(("launch", "attestation"))
            if failure_stage == "attestation":
                raise CudaError("injected attestation launch failure")

        def stream_sync(self, stream):
            assert stream == STREAM
            events.append(("sync",))

    cuda = FailingLaunchCuda()
    install_staging(monkeypatch, cuda, events)
    notary = bare_notary(cuda)
    notary._fn = {
        "measure_model_fused_kernel": "measure",
        "attest_measured_kernel": "attest",
    }
    notary._ctx_dev = SimpleNamespace(ptr=1)
    notary._cubin_digest_dev = SimpleNamespace(ptr=2)
    notary._kernel_digest_dev = SimpleNamespace(ptr=3)
    monkeypatch.setattr(notary_module, "DeviceBuffer", Buffer)

    with pytest.raises(CudaError, match=f"{failure_stage} launch"):
        notary._launch_fused_active([(0x9000, 8)], "2026-09-05T00:00:00Z", b"model")

    sync_index = events.index(("sync",))
    expected_launches = [("launch", "measurement")]
    if failure_stage == "attestation":
        expected_launches.append(("launch", "attestation"))
    assert events[:sync_index] == [("upload",), *expected_launches]
    assert not any(event[0] == "close" for event in events)
    assert all(buffer.ptr for buffer in instances)  # drained workspaces are reusable


def test_native_runner_owns_complete_ipc_and_fused_hot_path():
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    calls = []

    class NativeRunner:
        def run_ipc(self, inputs, timestamp, model, instance_root=None):
            calls.append((inputs, timestamp, model))
            return b"\x11" * 32, b"\x22" * 32, "native receipt"

    notary._native_runner = NativeRunner()
    ref = TensorRef(RAW.hex(), 8, seg_off=3, t_off=4, device=7)

    measurement, roots, receipt = notary._measure_request(
        [ref], "2026-09-03T10:00:00Z", b"test-model"
    )

    assert calls == [([(RAW, 8, 3, 4)], b"2026-09-03T10:00:00Z", b"test-model")]
    assert cuda.opened == [] and cuda.closed == []
    assert roots == b"\x11" * 32 and receipt == "native receipt"
    assert measurement.model_root == "22" * 32


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("failure", ["allocation", "free-interruption"])
def test_fallback_failed_workspace_growth_never_reuses_a_freed_buffer(
    monkeypatch, host, failure
):
    events = []

    class Buffer:
        def __init__(self, cu, nbytes):
            events.append(("alloc", nbytes))
            if nbytes == 32:
                raise CudaError("out of memory")
            self.ptr, self.nbytes = nbytes, nbytes

        def close(self):
            events.append(("free", self.ptr))
            self.ptr = 0
            if failure == "free-interruption":
                raise KeyboardInterrupt("after free")

    monkeypatch.setattr(notary_module, "PinnedBuffer" if host else "DeviceBuffer", Buffer)
    n = bare_notary(ImportedCuda())
    first = n._fallback_buffer("test", 8, host=host)
    assert n._fallback_buffer("test", 4, host=host) is first
    with pytest.raises(CudaError if failure == "allocation" else KeyboardInterrupt):
        n._fallback_buffer("test", 32, host=host)
    smaller = n._fallback_buffer("test", 4, host=host)
    assert smaller is not first and smaller.ptr != 0
    assert events == [("alloc", 8), ("free", 8),
                      *([("alloc", 32)] if failure == "allocation" else []), ("alloc", 4)]


@pytest.mark.parametrize("stage", ["upload", "measurement", "attestation", "download", "sync"])
@pytest.mark.parametrize("drain_fails,destroy_fails", [(False, False), (True, False), (True, True)])
def test_fallback_dma_lifetime_until_stream_completion_or_context_destruction(
    monkeypatch, stage, drain_fails, destroy_fails
):
    events, device_buffers = [], []

    class Buffer:
        def __init__(self, cu, nbytes):
            self.ptr = 0x10000 + len(device_buffers) * 0x10000
            self.closed = False
            self.nbytes = nbytes
            device_buffers.append(self)

        def close(self):
            self.closed = True
            self.ptr = 0

    class Driver(ImportedCuda):
        def launch_cooperative(self, fn, grid, block, args, *, stream):
            assert stream == STREAM
            if stage == "measurement":
                raise CudaError("injected measurement")

        def launch(self, fn, grid, block, args, *, stream):
            assert stream == STREAM
            if stage == "attestation":
                raise CudaError("injected attestation")

        def stream_sync(self, stream):
            assert stream == STREAM
            events.append(("sync",))
            if drain_fails or (stage == "sync" and events.count(("sync",)) == 1):
                raise CudaError("injected sync")

        def ctx_destroy(self, ctx):
            # DMA buffers are still pinned while destruction is IN PROGRESS.
            assert all(b.closed == (not drain_fails) for b in pinned)
            if drain_fails:
                assert all(b.ptr == 0 for b in pinned)
            assert all(b.closed == (not drain_fails) and b.ptr == 0 for b in device_buffers)
            super().ctx_destroy(ctx)

    cuda = Driver(ctx_destroy_fails=destroy_fails)
    notary = bare_notary(cuda)
    notary._fn = {"measure_model_fused_kernel": 1, "attest_measured_kernel": 2}
    notary._ctx_dev = SimpleNamespace(ptr=3, close=lambda: None)
    notary._cubin_digest_dev = SimpleNamespace(ptr=4, close=lambda: None)
    notary._kernel_digest_dev = SimpleNamespace(ptr=5, close=lambda: None)
    pinned = install_staging(monkeypatch, cuda, events)
    monkeypatch.setattr(notary_module, "DeviceBuffer", Buffer)
    if stage in {"upload", "download"}:
        method = "htod_async" if stage == "upload" else "dtoh_async"

        def fail_copy(*args):
            # Model a wrapper interrupted after DMA may have been accepted.
            raise CudaError(f"injected {stage}")

        setattr(cuda, method, fail_copy)

    error = (IpcCleanupUncertainError if destroy_fails else IpcSessionAbortedError) if drain_fails else CudaError
    with pytest.raises(error):
        notary._measure_request([TensorRef(RAW.hex(), 8)], "2026-09-08T00:00:00Z", b"model")

    if not drain_fails:
        assert all(not b.closed for b in [*pinned, *device_buffers])
        notary.close()

    assert len(pinned) == 2
    assert all(b.closed == (not drain_fails) for b in pinned)
    assert all(b.closed == (not drain_fails) for b in device_buffers)
    assert cuda.close_attempts == ([] if drain_fails else [0x1000])
    assert notary.ipc_cleanup_required == destroy_fails
    if drain_fails:
        assert notary._context_destroyed == (not destroy_fails)


def test_native_ipc_cleanup_flag_destroys_context_even_if_error_translation_fails():
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    context = notary.ctx

    class NativeRunner:
        def __init__(self):
            self.closed = False
            self.ipc_cleanup_failed = True

        def run_ipc(self, inputs, timestamp, model, instance_root=None):
            # Model an allocation failure while the extension translates its
            # native close exception. The persistent flag remains authoritative.
            raise MemoryError("could not construct native close exception")

        def close(self):
            self.closed = True

    runner = NativeRunner()
    notary._native_runner = runner

    with pytest.raises(IpcSessionAbortedError, match="context was destroyed"):
        notary._measure([TensorRef(RAW.hex(), 8, device=0)], "2026-09-03T10:00:00Z")

    assert runner.closed
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None
    assert not notary.ipc_cleanup_required


class QuarantinedNativeRunner:
    """Expose either native sentinel spelling or the typed cleanup error."""

    def __init__(self, signal):
        self.signal = signal
        self.closed = False

    def run_ipc(self, inputs, timestamp, model, instance_root=None):
        if self.signal == "typed_error":
            raise notary_module._NativeIpcCloseError("native IPC close failed")
        setattr(self, self.signal, True)
        raise MemoryError("could not construct native close exception")

    def close(self):
        self.closed = True


@pytest.mark.parametrize("signal", ["context_cleanup_required", "ipc_cleanup_failed"])
@pytest.mark.parametrize("destroy_fails", [False, True])
def test_close_preserves_native_quarantine_until_destroy_returns(signal, destroy_fails):
    cuda = ImportedCuda(ctx_destroy_fails=destroy_fails)
    notary = bare_notary(cuda)
    context = notary.ctx
    runner = QuarantinedNativeRunner(signal)
    setattr(runner, signal, True)
    notary._native_runner = runner
    notary.module = object()
    buffers = [SimpleNamespace(ptr=i + 1) for i in range(3)]
    notary._ctx_dev, notary._cubin_digest_dev, notary._kernel_digest_dev = buffers
    observed = []
    destroy = cuda.ctx_destroy

    def destroy_with_detached_runner(ctx):
        # This is the ownership handoff: the runner is already detached, but
        # the driver has not yet released any outstanding mappings or work.
        observed.append((notary._native_runner, notary.ipc_cleanup_required))
        destroy(ctx)

    cuda.ctx_destroy = destroy_with_detached_runner
    if destroy_fails:
        with pytest.raises(CudaError, match="cuCtxDestroy failed"):
            notary.close()
    else:
        notary.close()

    assert observed == [(None, True)]
    assert runner.closed and notary._native_runner is None
    assert [buffer.ptr for buffer in buffers] == [0, 0, 0]
    assert cuda.unload_attempts == []  # potentially active modules stay quarantined
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None
    assert notary._context_destroyed == (not destroy_fails)
    assert notary.ipc_cleanup_required == destroy_fails
    # Idempotent close must retain the failed-destruction quarantine; being
    # closed is not equivalent to having destroyed the CUDA context.
    notary.close()
    assert cuda.destroyed == [context]
    assert notary.ipc_cleanup_required == destroy_fails


@pytest.mark.parametrize("interrupt_type", [MemoryError, KeyboardInterrupt])
def test_native_quarantine_survives_interrupt_after_runner_detachment(interrupt_type):
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    runner = QuarantinedNativeRunner("context_cleanup_required")
    runner.context_cleanup_required = True
    notary._native_runner = runner
    lines, first_line = inspect.getsourcelines(Notary.close)
    target_line = first_line + next(
        i for i, line in enumerate(lines) if "self._native_runner = None" in line
    ) + 1

    def interrupt_after_detach(frame, event, arg):
        if (
            frame.f_code is Notary.close.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            sys.settrace(None)
            raise interrupt_type("interrupted after runner detachment")
        return interrupt_after_detach

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_after_detach)
    try:
        with pytest.raises(interrupt_type, match="after runner detachment"):
            notary.close()
    finally:
        sys.settrace(previous_trace)

    # Failure before cuCtxDestroy is attempted needs the same quarantine as
    # a driver-reported destruction failure. The detached runner is no guard.
    assert runner.closed and notary._native_runner is None
    assert cuda.destroyed == []
    assert notary._closed and notary.ctx is None
    assert not notary._context_destroyed
    assert notary.ipc_cleanup_required
    notary.close()
    assert notary.ipc_cleanup_required


@pytest.mark.parametrize(
    "signal", ["context_cleanup_required", "ipc_cleanup_failed", "typed_error"]
)
@pytest.mark.parametrize("replacement", ["context_query", "exception_allocation"])
@pytest.mark.parametrize("route", ["/v1/measure", "/v1/sign"])
def test_native_quarantine_withholds_http_ack_when_cleanup_error_is_replaced(
    monkeypatch, signal, replacement, route
):
    replacement_error = (
        CudaError("context query failed during unwinding")
        if replacement == "context_query"
        else MemoryError("could not allocate IpcCleanupUncertainError")
    )

    class UnwindingCuda(ImportedCuda):
        def ctx_get_current(self):
            if self.destroyed and replacement == "context_query":
                raise replacement_error
            return self.current

        def ctx_push_current(self, ctx):
            self.previous = self.current
            self.current = ctx

        def ctx_pop_current(self):
            ctx = self.current
            self.current = self.previous
            return ctx

    cuda = UnwindingCuda(ctx_destroy_fails=True)
    notary = bare_notary(cuda)
    context = notary.ctx
    # Force measure() to push a context so its finally queries CUDA after
    # close() fails, reproducing replacement during real Python unwinding.
    cuda.current = None
    runner = QuarantinedNativeRunner(signal)
    notary._native_runner = runner
    assert not notary.ipc_cleanup_required
    if replacement == "exception_allocation":
        def fail_error_construction(*args):
            raise replacement_error

        monkeypatch.setattr(
            notary_module, "IpcCleanupUncertainError", fail_error_construction
        )

    handler = object.__new__(make_handler(notary))
    handler.path = route
    handler._read_json = lambda: {
        "tensors": [{"handle": RAW.hex(), "nbytes": 8, "device": 0}],
        "model": "quarantine-regression",
    }
    handler.server = SimpleNamespace(cuattest_fatal_error=None)
    handler.close_connection = False
    sent = []
    handler._send = lambda code, payload, raw=False: sent.append((code, payload))

    handler.do_POST()

    # Even 500 headers acknowledge completion and release the producer lease.
    # The generic HTTP exception guard must see the Notary-owned quarantine,
    # not depend on either the lost runner or the specialized exception type.
    assert sent == []
    assert handler.close_connection
    assert handler.server.cuattest_fatal_error is replacement_error
    assert runner.closed and notary._native_runner is None
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None
    assert not notary._context_destroyed
    assert notary.ipc_cleanup_required


@pytest.mark.parametrize(
    ("destroy_fails", "expected_error"),
    [
        (False, GpuSessionAbortedError),
        (True, GpuCleanupUncertainError),
    ],
)
def test_native_direct_span_uncertainty_quarantines_source_and_workspaces(
    monkeypatch, destroy_fails, expected_error
):
    events = []
    cuda = ImportedCuda(ctx_destroy_fails=destroy_fails)
    notary = bare_notary(cuda)
    context = notary.ctx

    class NativeRunner:
        context_cleanup_required = True
        ipc_cleanup_failed = True

        def run_spans(self, spans, timestamp, model, instance_root=None):
            raise RuntimeError("both context synchronizations failed")

        def close(self):
            events.append("runner-abandoned")

    class SourceBuffer:
        ptr = 0xD000

        @classmethod
        def from_bytes(cls, cuda, data):
            return cls()

        def close(self):
            # hash_bytes clears the wrapper without issuing a potentially
            # racing cuMemFree; context destruction owns the allocation.
            events.append(("source-close", self.ptr))
            self.ptr = 0

    notary._native_runner = NativeRunner()
    monkeypatch.setattr(notary_module, "DeviceBuffer", SourceBuffer)

    with pytest.raises(expected_error, match="queued CUDA work"):
        notary.hash_bytes(b"source bytes")

    assert events == ["runner-abandoned", ("source-close", 0)]
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None


def test_all_spans_are_validated_before_first_hash_kernel(monkeypatch):
    cuda = ImportedCuda(allocation_nbytes=64)
    notary = bare_notary(cuda)
    launched = []
    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: launched.append(args),
    )
    refs = [
        TensorRef(RAW.hex(), 8, device=0),
        TensorRef(RAW.hex(), 8, seg_off=60, device=0),
    ]

    with pytest.raises(NotaryError, match="outside the imported allocation"):
        notary._measure(refs, "2026-09-03T10:00:00Z")

    assert launched == []
    assert cuda.closed == [0x1000]


@pytest.mark.parametrize(
    ("limit_name", "limit", "refs", "message"),
    [
        (
            "max_request_tensors",
            1,
            [TensorRef(RAW.hex(), 1), TensorRef(RAW.hex(), 1)],
            "2 tensors; limit is 1",
        ),
        (
            "max_request_bytes",
            15,
            [TensorRef(RAW.hex(), 8), TensorRef(RAW.hex(), 8)],
            "16 aggregate bytes; limit is 15",
        ),
        (
            "max_request_tiles",
            1,
            [TensorRef(RAW.hex(), 1), TensorRef(RAW.hex(), 1)],
            "2 scheduling tiles; limit is 1",
        ),
    ],
)
def test_request_work_limits_reject_repeated_spans_before_ipc_import(
    monkeypatch, limit_name, limit, refs, message
):
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    notary.max_request_tensors = 100
    notary.max_request_bytes = 1024**3
    notary.max_request_tiles = 100
    setattr(notary, limit_name, limit)
    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: pytest.fail("over-limit request reached the GPU"),
    )

    with pytest.raises(NotaryError, match=message):
        notary._measure(refs, "2026-09-03T10:00:00Z")

    # Both refs deliberately repeat one valid handle. Mapping deduplication
    # cannot let duplicated logical hashing evade aggregate work accounting.
    assert cuda.opened == []


def test_fallback_closes_an_import_if_handle_cache_insertion_runs_out_of_memory(
    monkeypatch,
):
    class FailingHash:
        def __init__(self):
            self.calls = 0

        def __hash__(self):
            self.calls += 1
            if self.calls == 2:
                # The empty-dict lookup happens first; fail the insertion that
                # follows cuIpcOpenMemHandle to exercise its narrow leak window.
                raise MemoryError("simulated dict growth failure")
            return 123

        def __eq__(self, other):
            return self is other

    raw = FailingHash()
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    monkeypatch.setattr(TensorRef, "raw_handle", lambda self: raw)
    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: pytest.fail("hash launched after failed ownership record"),
    )

    with pytest.raises(MemoryError, match="dict growth"):
        notary._measure([TensorRef(RAW.hex(), 8, device=0)], "2026-09-03T10:00:00Z")

    assert raw.calls == 2
    assert cuda.opened == [raw]
    assert cuda.closed == [0x1000]


def _fallback_line_containing(fragment):
    lines, first_line = inspect.getsourcelines(Notary._run_fallback_ipc_guarded)
    return first_line + next(
        index for index, line in enumerate(lines) if fragment in line
    )


@pytest.mark.parametrize("with_earlier_import", [False, True])
def test_fallback_rejected_import_closes_batch_and_keeps_serving(
    monkeypatch,
    with_earlier_import,
):
    class RejectingCuda(ImportedCuda):
        def ipc_open(self, raw):
            if raw == bytes(64):
                raise IpcImportRejectedError("cuIpcOpenMemHandle: invalid argument")
            return super().ipc_open(raw)

    cuda = RejectingCuda()
    notary = bare_notary(cuda)
    launched = []

    def launch(*args):
        launched.append(args)
        return _FusedResult(b"\x11" * 32, b"\x22" * 32, None)

    monkeypatch.setattr(notary, "_launch_fused_active", launch)
    valid = TensorRef(RAW.hex(), 8)
    invalid = TensorRef(bytes(64).hex(), 8)
    batch = [valid, invalid] if with_earlier_import else [invalid]
    with pytest.raises(IpcImportRejectedError, match="invalid argument"):
        notary._measure(batch, "2026-09-03T10:00:00Z")

    assert cuda.closed == ([0x1000] if with_earlier_import else [])
    assert not launched
    assert not notary.ipc_cleanup_required
    assert not notary._closed
    assert cuda.destroyed == []
    # A subsequent request must reach measurement in the same live session.
    notary._measure([valid], "2026-09-03T10:00:00Z")
    assert len(launched) == 1
    assert len(cuda.closed) == (2 if with_earlier_import else 1)


def test_fallback_interrupt_after_ipc_open_destroys_context_before_acknowledgement(
    monkeypatch,
):
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    context = notary.ctx
    target_line = _fallback_line_containing("owned_mappings[owned_count] = mapped_base")

    def interrupt_after_open(frame, event, arg):
        if (
            frame.f_code is Notary._run_fallback_ipc_guarded.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            sys.settrace(None)
            raise MemoryError("interrupted after cuIpcOpenMemHandle returned")
        return interrupt_after_open

    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: pytest.fail("interrupted import reached the GPU"),
    )
    previous_trace = sys.gettrace()
    sys.settrace(interrupt_after_open)
    try:
        with pytest.raises(IpcSessionAbortedError, match="context was destroyed"):
            notary._measure([TensorRef(RAW.hex(), 8)], "2026-09-03T10:00:00Z")
    finally:
        sys.settrace(previous_trace)

    assert cuda.opened == [RAW]
    assert cuda.close_attempts == []  # returned pointer never reached the ledger
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None


def test_fallback_interrupt_after_ipc_close_still_destroys_context(monkeypatch):
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    context = notary.ctx
    target_line = _fallback_line_containing("mapping_index += 1")
    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: _FusedResult(b"\x11" * 32, b"\x22" * 32, None),
    )

    def interrupt_after_close(frame, event, arg):
        if (
            frame.f_code is Notary._run_fallback_ipc_guarded.__code__
            and event == "line"
            and frame.f_lineno == target_line
        ):
            sys.settrace(None)
            raise MemoryError("interrupted after cuIpcCloseMemHandle returned")
        return interrupt_after_close

    previous_trace = sys.gettrace()
    sys.settrace(interrupt_after_close)
    try:
        with pytest.raises(IpcSessionAbortedError, match="context was destroyed"):
            notary._measure([TensorRef(RAW.hex(), 8)], "2026-09-03T10:00:00Z")
    finally:
        sys.settrace(previous_trace)

    assert cuda.closed == [0x1000]
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None


def test_fallback_failed_drain_destroys_context_without_explicit_ipc_close(monkeypatch):
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    context = notary.ctx

    def unconfirmed_work(*args):
        raise notary_module._QueuedGpuWorkUnconfirmedError(
            "queued CUDA work could not be synchronized"
        )

    monkeypatch.setattr(notary, "_launch_fused_active", unconfirmed_work)

    with pytest.raises(IpcSessionAbortedError, match="context was destroyed"):
        notary._measure([TensorRef(RAW.hex(), 8, device=0)], "2026-09-03T10:00:00Z")

    # Explicit unmap is forbidden while work may still be queued. Successful
    # context destruction releases the import before the server acknowledges.
    assert cuda.close_attempts == []
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None


def test_remapped_producer_ordinal_uses_driver_owner(monkeypatch):
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: _FusedResult(b"\x11" * 32, b"\x22" * 32, None),
    )

    measurement, roots = notary._measure(
        [TensorRef(RAW.hex(), 8, device=7)], "2026-09-03T10:00:00Z"
    )

    assert measurement.tensor_count == 1 and roots == b"\x11" * 32
    assert cuda.opened == [RAW] and cuda.closed == [0x1000]


def test_driver_reported_peer_allocation_is_rejected_before_hash(monkeypatch):
    cuda = ImportedCuda(allocation_device=1)
    notary = bare_notary(cuda)
    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: pytest.fail("hash launched"),
    )

    # The client claims cuda:0, but IPC lazy peer access could otherwise make
    # a cuda:1 allocation readable and produce a falsely labelled receipt.
    with pytest.raises(NotaryError, match="allocation belongs to cuda:1"):
        notary._measure([TensorRef(RAW.hex(), 8, device=0)], "2026-09-03T10:00:00Z")

    assert cuda.closed == [0x1000]


@pytest.mark.parametrize("unload_fails", [False, True])
def test_ipc_close_failure_destroys_context_before_safe_error(monkeypatch, unload_fails):
    cuda = ImportedCuda(ipc_close_fails=True, module_unload_fails=unload_fails)
    notary = bare_notary(cuda)
    notary.module = module = object()
    context = notary.ctx
    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: _FusedResult(b"\x11" * 32, b"\x22" * 32, None),
    )

    with pytest.raises(IpcSessionAbortedError, match="context was destroyed") as caught:
        notary._measure([TensorRef(RAW.hex(), 8, device=0)], "2026-09-03T10:00:00Z")

    assert isinstance(caught.value.__cause__, CudaError)
    assert cuda.close_attempts == [0x1000]
    assert cuda.unload_attempts == [module]
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None
    assert notary._context_destroyed
    assert not notary.ipc_cleanup_required


def test_ipc_close_and_context_destroy_failure_remains_unacknowledgeable(monkeypatch):
    cuda = ImportedCuda(ipc_close_fails=True, ctx_destroy_fails=True)
    notary = bare_notary(cuda)
    context = notary.ctx
    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: _FusedResult(b"\x11" * 32, b"\x22" * 32, None),
    )

    with pytest.raises(
        IpcCleanupUncertainError, match="destruction was not confirmed"
    ) as caught:
        notary._measure([TensorRef(RAW.hex(), 8, device=0)], "2026-09-03T10:00:00Z")

    assert isinstance(caught.value.__cause__, CudaError)
    assert cuda.close_attempts == [0x1000]
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None
    assert not notary._context_destroyed
    assert notary.ipc_cleanup_required

    # close() is now a no-op, not evidence that the failed destruction worked.
    # A second abort must not turn this closed-but-uncertain session into an ACK.
    with pytest.raises(IpcCleanupUncertainError, match="destruction was not confirmed"):
        notary._abort_uncertain_fallback_ipc(caught.value)
    assert cuda.destroyed == [context]
    assert notary.ipc_cleanup_required


@pytest.mark.parametrize("destroy_fails", [False, True])
@pytest.mark.parametrize("cleanup_failure", ["module", "pinned", "both"])
def test_http_ack_after_ipc_and_resource_cleanup_failure(monkeypatch, destroy_fails, cleanup_failure):
    cuda = ImportedCuda(
        ipc_close_fails=True, module_unload_fails=cleanup_failure != "pinned",
        ctx_destroy_fails=destroy_fails,
    )
    notary = bare_notary(cuda)
    if cleanup_failure != "module":
        cuda.host_alloc = lambda size: 123

        def fail_host_free(ptr):
            raise CudaError("cuMemFreeHost failed")

        cuda.host_free = fail_host_free
        notary._fallback_host_buffers["metadata"] = PinnedBuffer(cuda, 8)
    notary.module = module = object()
    context = notary.ctx
    monkeypatch.setattr(
        notary,
        "_launch_fused_active",
        lambda *args: _FusedResult(b"\x11" * 32, b"\x22" * 32, None),
    )
    handler = object.__new__(make_handler(notary))
    handler.path = "/v1/measure"
    handler._read_json = lambda: {
        "tensors": [{"handle": RAW.hex(), "nbytes": 8, "device": 0}]
    }
    handler.server = SimpleNamespace(cuattest_fatal_error=None)
    handler.close_connection = False
    sent = []

    def send(code, payload, raw=False):
        # Response headers release the producer's IPC lease. Check at the
        # actual acknowledgement boundary, not just after the handler returns.
        assert cuda.destroyed == [context]
        assert notary._context_destroyed
        assert not notary.ipc_cleanup_required
        sent.append((code, payload))

    handler._send = send
    handler.do_POST()

    assert cuda.opened == [RAW] and cuda.close_attempts == [0x1000]
    assert cuda.unload_attempts == [module] and cuda.destroyed == [context]
    assert handler.close_connection
    assert notary.ipc_cleanup_required == destroy_fails
    if destroy_fails:
        assert sent == []
        assert isinstance(handler.server.cuattest_fatal_error, IpcCleanupUncertainError)
    else:
        assert len(sent) == 1 and sent[0][0] == 500
        assert "context was destroyed" in sent[0][1]["error"]
        assert isinstance(handler.server.cuattest_fatal_error, IpcSessionAbortedError)


@pytest.mark.parametrize("destroy_fails", [False, True])
def test_abort_after_buffer_free_error_uses_actual_context_destruction(destroy_fails):
    cuda = ImportedCuda(ctx_destroy_fails=destroy_fails)
    notary = bare_notary(cuda)
    context = notary.ctx
    notary._fallback_ipc_cleanup_required = True

    def fail_free():
        raise CudaError("cuMemFree failed before context destruction")

    notary._ctx_dev = SimpleNamespace(close=fail_free)
    expected = IpcCleanupUncertainError if destroy_fails else IpcSessionAbortedError
    with pytest.raises(expected):
        notary._abort_uncertain_fallback_ipc(CudaError("cuIpcCloseMemHandle failed"))
    assert cuda.destroyed == [context]
    assert notary._closed and notary.ctx is None
    assert notary._context_destroyed == (not destroy_fails)
    assert notary.ipc_cleanup_required == destroy_fails


@pytest.mark.parametrize("interrupt_type", [CudaError, KeyboardInterrupt])
@pytest.mark.parametrize("later_failure", [None, "stream", "module", "context"])
@pytest.mark.parametrize("already_current", [False, True])
def test_pinned_free_failure_still_tears_down_cuda_and_detaches_buffers(
    interrupt_type, later_failure, already_current
):
    events = []
    original_error = interrupt_type("cuMemFreeHost failed")
    notary = object.__new__(Notary)
    notary._closed = False
    notary._context_destroyed = False
    notary.ctx, notary.module, notary._stream = "context", "module", STREAM

    class Driver:
        def __init__(self):
            self.stack = ["embedding"] + (["context"] if already_current else [])
            self.allocated = 0

        def ctx_get_current(self):
            return self.stack[-1]

        def ctx_push_current(self, ctx):
            events.append("push")
            self.stack.append(ctx)

        def ctx_pop_current(self):
            events.append("pop")
            return self.stack.pop()

        def host_alloc(self, nbytes):
            self.allocated += 1
            return self.allocated

        def host_free(self, ptr):
            assert self.stack[-1] == "context"
            events.append(("free", ptr))
            raise original_error

        def stream_destroy(self, stream):
            assert self.stack[-1] == "context" and stream == STREAM
            events.append("stream")
            if later_failure == "stream":
                raise CudaError("stream failed")

        def module_unload(self, module):
            assert self.stack[-1] == "context" and module == "module"
            events.append("module")
            if later_failure == "module":
                raise CudaError("module failed")

        def ctx_destroy(self, ctx):
            assert ctx == "context"
            # The unvisited second buffer must never free a context-reclaimed
            # allocation on a later close; exercise REAL PinnedBuffer wrappers.
            assert all(not buffer.ptr for buffer in buffers)
            assert not notary._fallback_host_buffers
            events.append("context")
            if later_failure == "context":
                raise CudaError("context failed")
            if self.stack[-1] == ctx:
                self.stack.pop()

    notary.cu = cuda = Driver()
    buffers = [PinnedBuffer(cuda, 8), PinnedBuffer(cuda, 8)]
    notary._fallback_host_buffers = dict(zip(("metadata", "output"), buffers))
    expected_error = CudaError if later_failure else interrupt_type
    with pytest.raises(expected_error) as caught:
        notary.close()
    if later_failure is None:
        assert caught.value is original_error  # successful destruction does not hide errors
    assert events == [
        *([] if already_current else ["push"]), ("free", 1), "stream", "module",
        *([] if already_current else ["pop"]), "context",
    ]
    assert cuda.stack == ["embedding"] + (
        ["context"] if already_current and later_failure == "context" else []
    )
    assert notary._context_destroyed == (later_failure != "context")
    assert notary._closed and notary.ctx is None and notary._stream is None
    before = list(events)
    notary.close()
    for buffer in buffers:
        buffer.close()
    assert events == before  # no retry/double free after the context was destroyed


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("interrupt_during_unload", [False, True])
def test_abort_preserves_interrupt_after_confirmed_destruction(
    monkeypatch, interrupt_type, interrupt_during_unload
):
    cuda = ImportedCuda(module_unload_fails=True)
    notary = bare_notary(cuda)
    notary.module = object()
    notary._fallback_ipc_cleanup_required = True
    interrupt = interrupt_type("interrupted cleanup")
    if interrupt_during_unload:
        def unload(module):
            raise interrupt

        monkeypatch.setattr(cuda, "module_unload", unload)
        original_error = CudaError("cuIpcCloseMemHandle failed")
    else:
        original_error = interrupt

    with pytest.raises(interrupt_type) as caught:
        notary._abort_uncertain_fallback_ipc(original_error)
    assert caught.value is interrupt
    assert notary._context_destroyed
    assert not notary.ipc_cleanup_required


class ClosedBuffer:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self.calls = 0

    def close(self):
        self.calls += 1
        self.events.append(("close", self.name))


@pytest.mark.parametrize("unload_fails,abandon", [(False, False), (True, False), (False, True)])
def test_close_releases_buffers_then_destroys_context_once(unload_fails, abandon):
    events = []
    notary = object.__new__(Notary)
    notary._closed = False
    notary.ctx = "context"
    notary.module = "module"
    notary._fn = {"kernel": object()}
    notary._abandon_context_allocations = abandon
    buffers = [ClosedBuffer(name, events) for name in ("ctx", "cubin", "kernel")]
    notary._ctx_dev, notary._cubin_digest_dev, notary._kernel_digest_dev = buffers

    class ContextCuda:
        def __init__(self):
            self.stack = ["other-context"]

        def ctx_get_current(self):
            return self.stack[-1] if self.stack else None

        def ctx_push_current(self, ctx):
            events.append(("push", ctx))
            self.stack.append(ctx)

        def ctx_pop_current(self):
            ctx = self.stack.pop()
            events.append(("pop", ctx))
            return ctx

        def ctx_destroy(self, ctx):
            assert ctx not in self.stack
            events.append(("destroy", ctx))

        def module_unload(self, module):
            assert self.stack[-1] == "context"
            events.append(("unload", module))
            if unload_fails:
                raise CudaError("injected unload failure")

    notary.cu = ContextCuda()

    if unload_fails:
        with pytest.raises(CudaError, match="injected unload failure"):
            notary.close()
    else:
        notary.close()
    notary.close()

    assert [buffer.calls for buffer in buffers] == [0 if abandon else 1] * 3
    assert events == [
        ("push", "context"),
        *([] if abandon else [
            ("close", "ctx"), ("close", "cubin"), ("close", "kernel"),
            ("unload", "module"),
        ]),
        ("pop", "context"),
        ("destroy", "context"),
    ]
    assert notary.cu.stack == ["other-context"]
    assert notary._closed and notary.ctx is None and notary.module is None
    assert notary._context_destroyed
    with pytest.raises(NotaryError, match="closed"):
        notary._ensure_open()


def test_source_and_cubin_are_host_hashed_before_module_load(monkeypatch, tmp_path):
    # This fixture deliberately models sm_75, which has no cp.async. A real
    # Blackwell matrix may force async globally; do not let that unrelated
    # device override replace this test's source/CUBIN trust-boundary checks.
    monkeypatch.setenv("CUATTEST_HASH_MODE", "standard")
    events = []
    source = "// exact source"
    cubin = b"\x7fELFexact-cubin"
    source_digest = b"\x11" * 32
    cubin_digest = b"\x22" * 32

    class InitCuda:
        def launch(self, function, grid, block, args, stream=None):
            # Startup launches configure_chain_slots_kernel before keygen; the
            # status buffer double already reports success.
            events.append(("launch", function))

        def stream_sync(self, stream):
            events.append(("stream_sync", stream))

        def stream_create(self):
            return STREAM

        def stream_destroy(self, stream):
            assert stream == STREAM

        def __init__(self):
            self.stack = ["embedding-context"]

        @property
        def current(self):
            return self.stack[-1] if self.stack else None

        def init(self):
            pass

        def device(self, ordinal):
            return ordinal

        def compute_capability(self, dev):
            return 7, 5

        def device_name(self, dev):
            return "fake GPU"

        def device_uuid(self, dev):
            return "GPU-00000000-0000-0000-0000-000000000000"

        def ctx_create(self, dev):
            self.stack.append("ctx")
            return "ctx"

        def ctx_get_current(self):
            return self.current

        def ctx_push_current(self, ctx):
            events.append(("ctx_push", ctx))
            self.stack.append(ctx)

        def ctx_pop_current(self):
            ctx = self.stack.pop()
            events.append(("ctx_pop", ctx))
            return ctx

        def module_load(self, image):
            events.append(("module_load", image))
            return "module"

        def module_unload(self, module):
            events.append(("module_unload", module))

        def function(self, module, name):
            return name

        def cooperative_grid(self, dev, fn, block):
            assert (dev, fn, block) == (0, "measure_model_fused_kernel", 128)
            return 4

        def ctx_destroy(self, ctx):
            assert ctx not in self.stack
            events.append(("ctx_destroy", ctx))

    class InitBuffer:
        next_ptr = 1

        def __init__(self, cuda=None, nbytes=0):
            # Startup also allocates a small status buffer directly, for the
            # chain-slot configuration launch, not only via from_bytes.
            self.ptr = InitBuffer.next_ptr
            InitBuffer.next_ptr += 1
            self.nbytes = nbytes

        @classmethod
        def from_bytes(cls, cuda, data):
            return cls(cuda, len(data))

        def read(self, nbytes):
            # Status zero: the configuration launch succeeded.
            return bytes(nbytes)

        def close(self):
            self.ptr = 0

    def host_hash(data):
        events.append(("host_hash", data))
        return source_digest if data == source.encode() else cubin_digest

    monkeypatch.setattr(notary_module, "Cuda", InitCuda)
    monkeypatch.setattr(notary_module, "DeviceBuffer", InitBuffer)
    monkeypatch.setattr(notary_module, "kernel_source", lambda: source)
    monkeypatch.setattr(
        notary_module,
        "load_cubin",
        lambda arch, frozen_source: (cubin, "test", "kernel.cubin"),
    )
    monkeypatch.setattr(notary_module, "blake3_digest", host_hash)
    monkeypatch.setattr(Notary, "_keygen", lambda self: (b"\x01" * 32, b"\x02" * 32))

    notary = Notary(artifact_dir=tmp_path)
    try:
        load_index = next(
            i for i, event in enumerate(events) if event[0] == "module_load"
        )
        hash_indexes = [i for i, event in enumerate(events) if event[0] == "host_hash"]
        assert len(hash_indexes) == 2 and max(hash_indexes) < load_index
        assert notary.info.kernel_cid == ids.raw_cid(source_digest)
        assert notary.info.cubin_cid == ids.raw_cid(cubin_digest)
        assert notary.cu.current == "embedding-context"
        assert notary.cu.stack == ["embedding-context"]
        # cuCtxCreate pushed exactly one entry and initialization popped that
        # same entry instead of replacing it with a duplicate embedding ctx.
        assert events.count(("ctx_pop", "ctx")) == 1
        artifact_metadata = json.loads(
            (tmp_path / notary_module.cubin_metadata_filename("sm_75")).read_text()
        )
        assert artifact_metadata["compiler"] == "test"
    finally:
        notary.close()
    assert notary.cu.stack == ["embedding-context"]


def test_construction_interrupt_after_context_creation_cleans_up_and_reraises(
    monkeypatch,
):
    events = []

    class InterruptedInitCuda:
        def stream_create(self):
            return STREAM

        def stream_destroy(self, stream):
            assert stream == STREAM

        def __init__(self):
            self.stack = ["embedding-context"]

        def init(self):
            pass

        def device(self, ordinal):
            return ordinal

        def compute_capability(self, dev):
            return 9, 0

        def device_name(self, dev):
            return "fake GPU"

        def ctx_create(self, dev):
            self.stack.append("notary-context")
            return "notary-context"

        def module_load(self, image):
            raise KeyboardInterrupt("stop construction")

        def ctx_get_current(self):
            return self.stack[-1] if self.stack else None

        def ctx_push_current(self, ctx):
            events.append(("push", ctx))
            self.stack.append(ctx)

        def ctx_pop_current(self):
            ctx = self.stack.pop()
            events.append(("pop", ctx))
            return ctx

        def ctx_destroy(self, ctx):
            assert ctx not in self.stack
            events.append(("destroy", ctx))

    cuda = InterruptedInitCuda()
    monkeypatch.setattr(notary_module, "Cuda", lambda: cuda)
    monkeypatch.setattr(notary_module, "kernel_source", lambda: "// source")
    monkeypatch.setattr(
        notary_module,
        "load_cubin",
        lambda arch, source: (b"\x7fELFcubin", "test", "kernel.cubin"),
    )

    with pytest.raises(KeyboardInterrupt, match="stop construction"):
        Notary()

    assert events == [
        ("pop", "notary-context"),
        ("push", "notary-context"),
        ("pop", "notary-context"),
        ("destroy", "notary-context"),
    ]
    assert cuda.stack == ["embedding-context"]


def test_each_operation_activates_its_own_context_then_restores_the_previous(
    monkeypatch,
):
    events = []

    class SharedCuda:
        def __init__(self):
            self.stack = ["ctx-second-notary"]

        @property
        def current(self):
            return self.stack[-1] if self.stack else None

        def ctx_get_current(self):
            return self.current

        def ctx_push_current(self, ctx):
            events.append(("push", ctx))
            self.stack.append(ctx)

        def ctx_pop_current(self):
            ctx = self.stack.pop()
            events.append(("pop", ctx))
            return ctx

        def stream_wait_stream(self, stream, producer):
            assert self.current == "ctx-first-notary" and stream == STREAM
            events.append(("producer", producer))

    first = object.__new__(Notary)
    first._closed = False
    first.ctx = "ctx-first-notary"
    first._stream = STREAM
    first.cu = SharedCuda()

    def hash_active(ptr, nbytes):
        events.append(("hash", first.cu.current, ptr, nbytes))
        return b"\x42" * 32

    monkeypatch.setattr(first, "_hash_dptr_active", hash_active)

    assert first.hash_dptr(0x1000, 8) == b"\x42" * 32
    assert events == [
        ("push", "ctx-first-notary"),
        ("producer", None),
        ("hash", "ctx-first-notary", 0x1000, 8),
        ("pop", "ctx-first-notary"),
    ]
    assert first.cu.stack == ["ctx-second-notary"]

    events.clear()

    def fail_active(ptr, nbytes):
        events.append(("hash-failed", first.cu.current))
        raise CudaError("launch failed")

    monkeypatch.setattr(first, "_hash_dptr_active", fail_active)
    with pytest.raises(CudaError, match="launch failed"):
        first.hash_dptr(0x1000, 8)
    assert events == [
        ("push", "ctx-first-notary"),
        ("producer", None),
        ("hash-failed", "ctx-first-notary"),
        ("pop", "ctx-first-notary"),
    ]
    assert first.cu.stack == ["ctx-second-notary"]


@pytest.mark.parametrize("producer", [None, 0xB22])
@pytest.mark.parametrize("failure", [None, CudaError, KeyboardInterrupt])
def test_direct_hash_hands_off_producer_before_either_backend(monkeypatch, producer, failure):
    cuda = ImportedCuda()
    notary = bare_notary(cuda)
    events = []

    def handoff(stream, producer_stream):
        assert cuda.current is notary.ctx and stream == STREAM
        events.append(("handoff", producer_stream))
        if failure:
            raise failure("producer handoff failed")

    def launch(spans):
        assert events == [("handoff", producer)]
        events.append(("hash", spans))
        return _FusedResult(b"\x42" * 32, b"\x43" * 32, None)

    cuda.stream_wait_stream = handoff
    monkeypatch.setattr(notary, "_launch_fused_active", launch)
    kwargs = {} if producer is None else {"producer_stream": producer}
    if failure:
        with pytest.raises(failure, match="producer handoff failed"):
            notary.hash_dptr(0x1000, 1025, **kwargs)
        assert events == [("handoff", producer)]  # no kernel accepted the source
    else:
        assert notary.hash_dptr(0x1000, 1025, **kwargs) == b"\x42" * 32
        assert events[-1] == ("hash", [(0x1000, 1025)])


def test_close_inside_an_activation_does_not_pop_the_embedding_context():
    events = []

    class StackCuda:
        def __init__(self):
            self.stack = ["embedding-context"]

        def ctx_get_current(self):
            return self.stack[-1] if self.stack else None

        def ctx_push_current(self, ctx):
            events.append(("push", ctx))
            self.stack.append(ctx)

        def ctx_pop_current(self):
            ctx = self.stack.pop()
            events.append(("pop", ctx))
            return ctx

        def ctx_destroy(self, ctx):
            assert self.stack[-1] == ctx
            events.append(("destroy", ctx))
            # A successful cuCtxDestroy pops a current context itself.
            self.stack.pop()

        def module_unload(self, module):
            assert self.stack[-1] == "notary-context"
            events.append(("unload", module))

    notary = object.__new__(Notary)
    notary._closed = False
    notary.ctx = "notary-context"
    notary.module = "module"
    notary._fn = {}
    notary._ctx_dev = None
    notary._cubin_digest_dev = None
    notary._kernel_digest_dev = None
    notary.cu = StackCuda()

    with notary._activate():
        assert notary.cu.stack == ["embedding-context", "notary-context"]
        notary.close()

    assert events == [
        ("push", "notary-context"), ("unload", "module"), ("destroy", "notary-context")
    ]
    assert notary.cu.stack == ["embedding-context"]


def _valid_host_validation_case():
    notary = object.__new__(Notary)
    notary.device_ordinal = 0
    notary.info = SimpleNamespace(
        gpu_did="did:key:trusted", kernel_cid="kernel", cubin_cid="cubin"
    )
    measurement = Measurement(
        digests="11" * 32,
        model_root="22" * 32,
        vram_cid="model-cid",
        tensor_count=1,
        measured_at="2026-09-03T10:00:00Z",
    )
    document = {
        "claim": MEASUREMENT_CLAIM,
        "hashScheme": MEASUREMENT_HASH_SCHEME,
        "operation": MEASUREMENT_OPERATION,
        "modelHash": measurement.model_root,
        "modelCID": f"urn:cid:{measurement.vram_cid}",
        "tensorCount": measurement.tensor_count,
        "measuredAt": measurement.measured_at,
        "model": "requested-model",
        "device": "cuda:0",
        "gpuDID": notary.info.gpu_did,
        "kernelCID": f"urn:cid:{notary.info.kernel_cid}",
        "cubinCID": f"urn:cid:{notary.info.cubin_cid}",
    }
    return notary, measurement, document


def _valid_manifest(document, previous_uuid=None):
    """The kernel's one-statement manifest, structurally valid but unsigned.

    _validate_kernel_receipt checks structure only -- it never verifies a
    proof -- so a well-formed placeholder JWS is enough here. Signature
    verification is covered in test_expect.
    """
    issuer = document["gpuDID"]
    timestamp = document["measuredAt"]
    model_urn = document["modelCID"]
    instance_urn = f"urn:cid:{ids.raw_cid(blake3_digest(b'host-validation'))}"
    credential_id = statements_module._state_credential_id(
        document["modelHash"], issuer, timestamp, instance_urn, model_urn,
        previous_uuid,
    )
    jws = statements_module.JWS_PREFIX + base64.urlsafe_b64encode(
        bytes(64)
    ).rstrip(b"=").decode()
    statement_id = statements_module._credential_registration_id(
        credential_id, issuer, timestamp, instance_urn, model_urn, previous_uuid, jws
    )
    statements = {
        statement_id: {
            "@context": statements_module.STATEMENT_CONTEXT,
            "@id": statement_id,
            "@type": "CredentialRegistration",
            "credential": {
                "@context": statements_module.VC_CONTEXT,
                "id": credential_id,
                "type": ["VerifiableCredential", statements_module.CREDENTIAL_TYPE],
                "credentialSubject": {
                    "id": issuer,
                    "state": {
                        "stateType": statements_module.STATE_TYPE,
                        "instanceID": instance_urn,
                        "modelRoot": model_urn,
                        "previousStateCredential": previous_uuid,
                    },
                },
                "issuer": issuer,
                "proof": {
                    "type": "EcdsaSecp256r1Signature2019",
                    "proofPurpose": "assertionMethod",
                    "verificationMethod": f"{issuer}#{issuer[8:]}",
                    "created": timestamp,
                    "jws": jws,
                },
                "validFrom": timestamp,
            },
            "registeredBy": issuer,
            "timestamp": timestamp,
        }
    }
    return {"version": statements_module.MANIFEST_VERSION, "statements": statements}


def test_host_refuses_a_cubin_that_adds_a_stronger_signed_assertion():
    notary, measurement, document = _valid_host_validation_case()
    document["inferenceUsedTheseWeights"] = True
    receipt = {
        "measurementDocument": json.dumps(document).encode().hex(),
        "measurementSignature": "00" * 64,
        "modelRoot": measurement.model_root,
        "manifest": {"version": statements_module.MANIFEST_VERSION, "statements": {}},
    }

    with pytest.raises(NotaryError, match="unexpected field"):
        notary._validate_kernel_receipt(receipt, measurement, "requested-model")


def test_host_refuses_a_cubin_that_emits_duplicate_signed_fields():
    notary, measurement, document = _valid_host_validation_case()
    serialized = json.dumps(document, separators=(",", ":"))
    # json.loads would choose the later, valid claim; a first-key-wins verifier
    # could instead expose this unverified semantic assertion.
    serialized = '{"claim":"inference used these weights",' + serialized[1:]
    receipt = {
        "measurementDocument": serialized.encode().hex(),
        "modelRoot": measurement.model_root,
    }

    with pytest.raises(NotaryError, match="duplicate field"):
        notary._validate_kernel_receipt(receipt, measurement, "requested-model")


@pytest.mark.parametrize("invalid_count", [True, 1.0])
def test_host_refuses_non_integer_signed_tensor_counts(invalid_count):
    notary, measurement, document = _valid_host_validation_case()
    document["tensorCount"] = invalid_count
    receipt = {
        "measurementDocument": json.dumps(document).encode().hex(),
        "modelRoot": measurement.model_root,
    }

    with pytest.raises(NotaryError, match="tensorCount must be an integer"):
        notary._validate_kernel_receipt(receipt, measurement, "requested-model")


def test_host_refuses_a_stronger_assertion_in_the_statement_graph():
    notary, measurement, document = _valid_host_validation_case()
    document_bytes = json.dumps(document, separators=(",", ":")).encode()
    manifest = _valid_manifest(document)
    state = next(
        statement["credential"]["credentialSubject"]["state"]
        for statement in manifest["statements"].values()
        if statement["@type"] == "CredentialRegistration"
    )
    state["inferenceUsedTheseWeights"] = True
    receipt = {
        "measurementDocument": document_bytes.hex(),
        "measurementSignature": "00" * 64,
        "modelRoot": measurement.model_root,
        "manifest": manifest,
    }

    with pytest.raises(NotaryError, match="invalid statements.*exact schema"):
        notary._validate_kernel_receipt(receipt, measurement, "requested-model")


def test_host_accepts_the_exact_manifest():
    notary, measurement, document = _valid_host_validation_case()
    document_bytes = json.dumps(document, separators=(",", ":")).encode()
    receipt = {
        "measurementDocument": document_bytes.hex(),
        "measurementSignature": "00" * 64,
        "modelRoot": measurement.model_root,
        "manifest": _valid_manifest(document),
    }

    notary._validate_kernel_receipt(receipt, measurement, "requested-model")


def test_host_refuses_a_cubin_that_signs_a_false_code_identity():
    notary = object.__new__(Notary)
    notary.device_ordinal = 0
    notary.info = SimpleNamespace(
        gpu_did="did:key:trusted", kernel_cid="kernel-trusted", cubin_cid="cubin-actual"
    )
    measurement = Measurement(
        digests="11" * 32,
        model_root="22" * 32,
        vram_cid="model-cid",
        tensor_count=1,
        measured_at="2026-09-03T10:00:00Z",
    )
    document = {
        "claim": MEASUREMENT_CLAIM,
        "hashScheme": MEASUREMENT_HASH_SCHEME,
        "operation": MEASUREMENT_OPERATION,
        "modelHash": measurement.model_root,
        "modelCID": f"urn:cid:{measurement.vram_cid}",
        "tensorCount": 1,
        "measuredAt": measurement.measured_at,
        "model": "requested-model",
        "device": "cuda:0",
        "gpuDID": notary.info.gpu_did,
        "kernelCID": f"urn:cid:{notary.info.kernel_cid}",
        # A malicious executable claims the allowlisted digest, rather than
        # the digest the trusted host computed for its loaded bytes.
        "cubinCID": "urn:cid:cubin-allowlisted-but-false",
    }
    receipt = {
        "measurementDocument": json.dumps(document).encode().hex(),
        "modelRoot": measurement.model_root,
    }

    with pytest.raises(NotaryError, match="signed cubinCID contradicts"):
        notary._validate_kernel_receipt(receipt, measurement, "requested-model")


def test_host_refuses_a_cubin_that_claims_submitted_spans_are_tensors():
    notary = object.__new__(Notary)
    notary.device_ordinal = 0
    notary.info = SimpleNamespace(
        gpu_did="did:key:trusted", kernel_cid="kernel", cubin_cid="cubin"
    )
    measurement = Measurement(
        digests="11" * 32,
        model_root="22" * 32,
        vram_cid="model-cid",
        tensor_count=1,
        measured_at="2026-09-03T10:00:00Z",
    )
    document = {
        "claim": MEASUREMENT_CLAIM.replace(
            "client-submitted, bounds-checked VRAM spans", "VRAM-resident tensors"
        ),
        "hashScheme": MEASUREMENT_HASH_SCHEME,
        "operation": MEASUREMENT_OPERATION,
        "modelHash": measurement.model_root,
        "modelCID": f"urn:cid:{measurement.vram_cid}",
        "tensorCount": 1,
        "measuredAt": measurement.measured_at,
        "model": "requested-model",
        "device": "cuda:0",
        "gpuDID": notary.info.gpu_did,
        "kernelCID": f"urn:cid:{notary.info.kernel_cid}",
        "cubinCID": f"urn:cid:{notary.info.cubin_cid}",
    }
    receipt = {
        "measurementDocument": json.dumps(document).encode().hex(),
        "modelRoot": measurement.model_root,
    }

    with pytest.raises(NotaryError, match="signed claim contradicts"):
        notary._validate_kernel_receipt(receipt, measurement, "requested-model")


def test_sign_refuses_a_signed_document_for_another_model(monkeypatch):
    notary = object.__new__(Notary)
    notary._closed = False
    notary.ctx = object()
    notary.cu = SimpleNamespace(
        ctx_get_current=lambda: notary.ctx,
        ctx_set_current=lambda ctx: None,
    )
    notary.device_ordinal = 0
    notary.info = SimpleNamespace(
        gpu_did="did:key:trusted",
        kernel_cid="kernel-trusted",
        cubin_cid="cubin-trusted",
    )
    measurement = Measurement(
        digests="11" * 32,
        model_root="22" * 32,
        vram_cid="model-cid",
        tensor_count=1,
        measured_at="2026-09-03T10:00:00Z",
    )
    document = {
        "claim": MEASUREMENT_CLAIM,
        "hashScheme": MEASUREMENT_HASH_SCHEME,
        "operation": MEASUREMENT_OPERATION,
        "modelHash": measurement.model_root,
        "modelCID": f"urn:cid:{measurement.vram_cid}",
        "tensorCount": 1,
        "measuredAt": measurement.measured_at,
        "model": "different-model",
        "device": "cuda:0",
        "gpuDID": notary.info.gpu_did,
        "kernelCID": f"urn:cid:{notary.info.kernel_cid}",
        "cubinCID": f"urn:cid:{notary.info.cubin_cid}",
    }
    receipt = {
        "measurementDocument": json.dumps(document).encode().hex(),
        "measurementSignature": "00" * 64,
        "modelRoot": measurement.model_root,
        "manifest": {
            "version": statements_module.MANIFEST_VERSION,
            "statements": {},
        },
    }
    monkeypatch.setattr(
        notary_module, "_utc_timestamp", lambda: measurement.measured_at
    )

    def wrong_model_receipt(tensors, measured_at, model_bytes):
        assert model_bytes == b"requested-model"
        return measurement, bytes.fromhex(measurement.digests), json.dumps(receipt)

    monkeypatch.setattr(notary, "_measure_request", wrong_model_receipt)

    with pytest.raises(NotaryError, match="signed model contradicts"):
        notary.sign([object()], "requested-model")

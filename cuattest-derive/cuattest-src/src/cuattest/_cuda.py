# SPDX-License-Identifier: Apache-2.0
"""Minimal CUDA driver API binding, via ctypes.

Only the entry points the notary needs, loaded from ``libcuda.so.1`` at
runtime. Binding by dlopen rather than linking means this package installs
and imports on a machine with no CUDA at all - you only need a driver when
you actually open a device.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import uuid
from typing import Iterable

CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95
CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9
CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0x1
CU_STREAM_NON_BLOCKING = 0x1
CU_EVENT_DISABLE_TIMING = 0x2

IPC_HANDLE_BYTES = 64


class CudaError(RuntimeError):
    """A CUDA driver call failed."""


class IpcImportRejectedError(CudaError):
    """The driver returned failure without creating an IPC mapping."""


class CUipcMemHandle(ctypes.Structure):
    # c_ubyte, NOT c_char: ctypes gives a c_char array string semantics and
    # silently truncates an assignment at the first NUL. IPC handles are
    # binary and routinely contain NULs, so a c_char field passes a mangled
    # handle and the driver rejects it with "invalid argument".
    _fields_ = [("reserved", ctypes.c_ubyte * IPC_HANDLE_BYTES)]

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CUipcMemHandle":
        if len(raw) != IPC_HANDLE_BYTES:
            raise CudaError(f"IPC handle must be {IPC_HANDLE_BYTES} bytes, got {len(raw)}")
        h = cls()
        h.reserved = (ctypes.c_ubyte * IPC_HANDLE_BYTES)(*raw)
        return h

    def to_bytes(self) -> bytes:
        return bytes(self.reserved)


class Cuda:
    """The driver API, bound lazily."""

    def __init__(self, soname: str = "libcuda.so.1") -> None:
        try:
            self.lib = ctypes.CDLL(soname)
        except OSError as e:
            raise CudaError(
                f"cannot load {soname}: {e}. An NVIDIA driver must be installed; "
                "inside a container the device also has to be passed through "
                "(docker --gpus all)."
            ) from e
        self._bind()

    def _bind(self) -> None:
        c, v, p = ctypes.c_int, ctypes.c_void_p, ctypes.POINTER
        u64, u32 = ctypes.c_ulonglong, ctypes.c_uint
        sig = {
            "cuInit": ([u32], c),
            "cuDeviceGet": ([p(c), c], c),
            "cuDeviceGetCount": ([p(c)], c),
            "cuDeviceGetAttribute": ([p(c), c, c], c),
            "cuDeviceGetName": ([ctypes.c_char_p, c, c], c),
            "cuCtxCreate_v2": ([p(v), u32, c], c),
            "cuCtxDestroy_v2": ([v], c),
            "cuCtxGetCurrent": ([p(v)], c),
            "cuCtxSetCurrent": ([v], c),
            "cuCtxPushCurrent_v2": ([v], c),
            "cuCtxPopCurrent_v2": ([p(v)], c),
            "cuCtxSynchronize": ([], c),
            "cuStreamCreate": ([p(v), u32], c),
            "cuStreamDestroy_v2": ([v], c),
            "cuStreamSynchronize": ([v], c),
            "cuStreamWaitEvent": ([v, v, u32], c),
            "cuEventCreate": ([p(v), u32], c),
            "cuEventRecord": ([v, v], c),
            "cuEventDestroy_v2": ([v], c),
            "cuModuleLoadData": ([p(v), v], c),
            "cuModuleUnload": ([v], c),
            "cuModuleGetFunction": ([p(v), v, ctypes.c_char_p], c),
            "cuMemAlloc_v2": ([p(u64), ctypes.c_size_t], c),
            "cuMemFree_v2": ([u64], c),
            "cuMemcpyHtoD_v2": ([u64, v, ctypes.c_size_t], c),
            "cuMemcpyDtoH_v2": ([v, u64, ctypes.c_size_t], c),
            "cuMemAllocHost_v2": ([p(v), ctypes.c_size_t], c),
            "cuMemFreeHost": ([v], c),
            "cuMemcpyHtoDAsync_v2": ([u64, v, ctypes.c_size_t, v], c),
            "cuMemcpyDtoHAsync_v2": ([v, u64, ctypes.c_size_t, v], c),
            "cuMemsetD8_v2": ([u64, ctypes.c_ubyte, ctypes.c_size_t], c),
            "cuMemsetD8Async": ([u64, ctypes.c_ubyte, ctypes.c_size_t, v], c),
            "cuPointerGetAttribute": ([v, c, u64], c),
            "cuLaunchKernel": ([v, u32, u32, u32, u32, u32, u32, u32, v, p(v), p(v)], c),
            "cuLaunchCooperativeKernel": (
                [v, u32, u32, u32, u32, u32, u32, u32, v, p(v)],
                c,
            ),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor": (
                [p(c), v, c, ctypes.c_size_t],
                c,
            ),
            "cuIpcCloseMemHandle": ([u64], c),
            "cuGetErrorString": ([c, p(ctypes.c_char_p)], c),
        }
        for name, (argtypes, restype) in sig.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = argtypes, restype
            setattr(self, name, fn)
        # v2 distinguishes MIG instances; older supported drivers expose the
        # original UUID query. UUIDs survive CUDA_VISIBLE_DEVICES reordering.
        for name in ("cuDeviceGetUuid_v2", "cuDeviceGetUuid"):
            try:
                fn = getattr(self.lib, name)
            except AttributeError:
                continue
            fn.argtypes, fn.restype = [v, c], c
            self.cuDeviceGetUuid = fn
            break
        else:
            raise CudaError("libcuda exports no cuDeviceGetUuid")
        # cuIpcOpenMemHandle takes the handle struct by value; the _v2 form is
        # what current drivers export, with the original as a fallback.
        for cand in ("cuIpcOpenMemHandle_v2", "cuIpcOpenMemHandle"):
            fn = getattr(self.lib, cand, None)
            if fn is not None:
                fn.argtypes = [p(u64), CUipcMemHandle, u32]
                fn.restype = c
                self.cuIpcOpenMemHandle = fn
                break
        else:
            raise CudaError("libcuda exports no cuIpcOpenMemHandle")
        # Querying the imported allocation is the only trustworthy way to
        # bounds-check offsets supplied by an IPC client. Request metadata is
        # not an authority for how large the exported allocation really is.
        for cand in ("cuMemGetAddressRange_v2", "cuMemGetAddressRange"):
            fn = getattr(self.lib, cand, None)
            if fn is not None:
                fn.argtypes = [p(u64), p(ctypes.c_size_t), u64]
                fn.restype = c
                self.cuMemGetAddressRange = fn
                break
        else:
            raise CudaError("libcuda exports no cuMemGetAddressRange")

    # ── error handling ───────────────────────────────────────────────────────

    def check(self, rc: int, what: str) -> None:
        if rc == 0:
            return
        msg = ctypes.c_char_p()
        self.cuGetErrorString(rc, ctypes.byref(msg))
        text = msg.value.decode() if msg.value else f"error {rc}"
        hint = ""
        if rc == 802:
            hint = (
                " — the GPU is in confidential-compute mode and has not been "
                "unlocked; set its ready state (nvidia-smi conf-compute -srs 1)"
            )
        raise CudaError(f"{what}: {text} (rc={rc}){hint}")

    # ── the calls ────────────────────────────────────────────────────────────

    def init(self) -> None:
        self.check(self.cuInit(0), "cuInit")

    def device_count(self) -> int:
        count = ctypes.c_int()
        self.check(self.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
        return count.value

    def device_uuid(self, dev: int) -> str:
        raw = (ctypes.c_ubyte * 16)()
        self.check(self.cuDeviceGetUuid(ctypes.byref(raw), dev), "cuDeviceGetUuid")
        return "GPU-" + str(uuid.UUID(bytes=bytes(raw)))

    def device(self, ordinal: int = 0) -> int:
        dev = ctypes.c_int()
        self.check(self.cuDeviceGet(ctypes.byref(dev), ordinal), "cuDeviceGet")
        return dev.value

    def compute_capability(self, dev: int) -> tuple[int, int]:
        out = []
        for attr, label in (
            (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "major"),
            (CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "minor"),
        ):
            v = ctypes.c_int()
            self.check(self.cuDeviceGetAttribute(ctypes.byref(v), attr, dev),
                       f"cuDeviceGetAttribute(cc {label})")
            out.append(v.value)
        return out[0], out[1]

    def device_name(self, dev: int) -> str:
        buf = ctypes.create_string_buffer(128)
        self.check(self.cuDeviceGetName(buf, 128, dev), "cuDeviceGetName")
        return buf.value.decode(errors="replace")

    def device_attribute(self, dev: int, attribute: int, label: str) -> int:
        value = ctypes.c_int()
        self.check(
            self.cuDeviceGetAttribute(ctypes.byref(value), attribute, dev),
            f"cuDeviceGetAttribute({label})",
        )
        return value.value

    def ctx_create(self, dev: int) -> ctypes.c_void_p:
        ctx = ctypes.c_void_p()
        self.check(self.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
        return ctx

    def ctx_destroy(self, ctx) -> None:
        self.check(self.cuCtxDestroy_v2(ctx), "cuCtxDestroy")

    def stream_create(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        # A plain stream still implicitly joins legacy stream 0. NON_BLOCKING
        # is necessary to isolate this session from unrelated default-stream work.
        self.check(self.cuStreamCreate(ctypes.byref(stream), CU_STREAM_NON_BLOCKING),
                   "cuStreamCreate")
        return stream

    def stream_destroy(self, stream) -> None:
        # Destruction is asynchronous, NOT proof that queued work has finished.
        self.check(self.cuStreamDestroy_v2(stream), "cuStreamDestroy")

    def stream_sync(self, stream) -> None:
        self.check(self.cuStreamSynchronize(stream), "cuStreamSynchronize")

    def stream_wait_stream(self, stream, producer_stream=None) -> None:
        """Queue a dependency on previously submitted producer work, without a host wait."""
        event = ctypes.c_void_p()
        try:
            self.check(self.cuEventCreate(ctypes.byref(event), CU_EVENT_DISABLE_TIMING),
                       "cuEventCreate(producer readiness)")
            self.check(self.cuEventRecord(event, producer_stream),
                       "cuEventRecord(producer readiness)")
            self.check(self.cuStreamWaitEvent(stream, event, 0),
                       "cuStreamWaitEvent(producer readiness)")
        finally:
            if event.value:
                # CUDA retains an incomplete event and its queued dependencies
                # until completion. Destroy is asynchronous, so retiring this
                # per-call handle neither waits on the CPU nor loses ordering.
                # Retire it on errors/interrupts too, even after a queued wait.
                self.check(self.cuEventDestroy_v2(event), "cuEventDestroy")

    def ctx_get_current(self) -> ctypes.c_void_p:
        ctx = ctypes.c_void_p()
        self.check(self.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        return ctx

    def ctx_set_current(self, ctx) -> None:
        self.check(self.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    def ctx_push_current(self, ctx) -> None:
        self.check(self.cuCtxPushCurrent_v2(ctx), "cuCtxPushCurrent")

    def ctx_pop_current(self) -> ctypes.c_void_p:
        ctx = ctypes.c_void_p()
        self.check(self.cuCtxPopCurrent_v2(ctypes.byref(ctx)), "cuCtxPopCurrent")
        return ctx

    def module_load(self, image: bytes) -> ctypes.c_void_p:
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(image, len(image))
        self.check(self.cuModuleLoadData(ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p)),
                   "cuModuleLoadData")
        return mod

    def function(self, mod, name: str) -> ctypes.c_void_p:
        fn = ctypes.c_void_p()
        self.check(self.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()),
                   f"cuModuleGetFunction({name})")
        return fn

    def module_unload(self, mod) -> None:
        self.check(self.cuModuleUnload(mod), "cuModuleUnload")

    def alloc(self, nbytes: int) -> int:
        ptr = ctypes.c_ulonglong()
        # CUDA rejects zero-byte allocations. Empty BLAKE3 inputs therefore
        # have a one-byte sentinel that is deliberately never read or written;
        # initcheck's optional unused-memory scan can report that sentinel.
        self.check(self.cuMemAlloc_v2(ctypes.byref(ptr), max(nbytes, 1)), "cuMemAlloc")
        return ptr.value

    def free(self, ptr: int) -> None:
        self.check(self.cuMemFree_v2(ptr), "cuMemFree")

    def htod(self, ptr: int, data: bytes) -> None:
        buf = ctypes.create_string_buffer(data, len(data))
        self.check(self.cuMemcpyHtoD_v2(ptr, ctypes.cast(buf, ctypes.c_void_p), len(data)),
                   "cuMemcpyHtoD")
        # Pageable uploads may return after staging, before DMA completes.
        # Finish this blocking setup/direct-input helper before a nonblocking
        # stream can read it. Requests use pinned, same-stream DMA instead.
        self.stream_sync(None)

    def dtoh(self, ptr: int, nbytes: int) -> bytes:
        buf = ctypes.create_string_buffer(nbytes)
        self.check(self.cuMemcpyDtoH_v2(ctypes.cast(buf, ctypes.c_void_p), ptr, nbytes),
                   "cuMemcpyDtoH")
        return buf.raw

    def memset0(self, ptr: int, nbytes: int) -> None:
        self.check(self.cuMemsetD8_v2(ptr, 0, nbytes), "cuMemsetD8")

    def memset0_async(self, ptr: int, nbytes: int, stream) -> None:
        self.check(self.cuMemsetD8Async(ptr, 0, nbytes, stream), "cuMemsetD8Async")

    def host_alloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.cuMemAllocHost_v2(ctypes.byref(ptr), max(1, nbytes)),
                   "cuMemAllocHost")
        return ptr

    def host_free(self, ptr) -> None:
        self.check(self.cuMemFreeHost(ptr), "cuMemFreeHost")

    def htod_async(self, ptr: int, host, nbytes: int, stream) -> None:
        """Enqueue from pinned host storage retained until stream completion."""
        self.check(self.cuMemcpyHtoDAsync_v2(ptr, host, nbytes, stream),
                   "cuMemcpyHtoDAsync")

    def dtoh_async(self, host, ptr: int, nbytes: int, stream) -> None:
        """Enqueue into pinned host storage; read it only after completion."""
        self.check(self.cuMemcpyDtoHAsync_v2(host, ptr, nbytes, stream),
                   "cuMemcpyDtoHAsync")

    def launch(self, fn, grid: int, block: int, args: Iterable, *, stream=None) -> None:
        """Launch a 1-D kernel. `args` are ctypes values, passed by reference."""
        arr = (ctypes.c_void_p * len(args := list(args)))()
        for i, a in enumerate(args):
            arr[i] = ctypes.cast(ctypes.byref(a), ctypes.c_void_p)
        self.check(
            self.cuLaunchKernel(fn, grid, 1, 1, block, 1, 1, 0, stream, arr, None),
            "cuLaunchKernel",
        )

    def cooperative_grid(self, dev: int, fn, block: int) -> int:
        """Return the largest occupancy-safe grid for a cooperative kernel."""
        supported = self.device_attribute(
            dev, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, "cooperative launch"
        )
        if not supported:
            raise CudaError("device does not support cooperative kernel launches")
        multiprocessors = self.device_attribute(
            dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "multiprocessor count"
        )
        blocks_per_multiprocessor = ctypes.c_int()
        self.check(
            self.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(blocks_per_multiprocessor), fn, block, 0
            ),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )
        if multiprocessors <= 0 or blocks_per_multiprocessor.value <= 0:
            raise CudaError("cooperative kernel has zero occupancy")
        return multiprocessors * blocks_per_multiprocessor.value

    def launch_cooperative(
        self, fn, grid: int, block: int, args: Iterable, *, stream=None
    ) -> None:
        """Launch a 1-D grid whose blocks may synchronize with each other."""
        arr = (ctypes.c_void_p * len(args := list(args)))()
        for i, argument in enumerate(args):
            arr[i] = ctypes.cast(ctypes.byref(argument), ctypes.c_void_p)
        self.check(
            self.cuLaunchCooperativeKernel(
                fn, grid, 1, 1, block, 1, 1, 0, stream, arr
            ),
            "cuLaunchCooperativeKernel",
        )

    def sync(self) -> None:
        self.check(self.cuCtxSynchronize(), "cuCtxSynchronize")

    def ipc_open(self, raw_handle: bytes) -> int:
        h = CUipcMemHandle.from_bytes(raw_handle)
        ptr = ctypes.c_ulonglong()
        rc = self.cuIpcOpenMemHandle(
            ctypes.byref(ptr), h, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS
        )
        if rc != 0:
            # Only a returned failure code proves that no mapping was opened.
            # An exception crossing a successful call's return boundary leaves
            # ownership uncertain and must not receive this classification.
            try:
                self.check(rc, "cuIpcOpenMemHandle")
            except CudaError as error:
                raise IpcImportRejectedError(str(error)) from error
        return ptr.value

    def ipc_close(self, ptr: int) -> None:
        self.check(self.cuIpcCloseMemHandle(ptr), "cuIpcCloseMemHandle")

    def address_range(self, ptr: int) -> tuple[int, int]:
        """Return the actual allocation base and size containing ``ptr``."""
        base = ctypes.c_ulonglong()
        size = ctypes.c_size_t()
        self.check(self.cuMemGetAddressRange(ctypes.byref(base), ctypes.byref(size), ptr),
                   "cuMemGetAddressRange")
        return base.value, size.value

    def pointer_device(self, ptr: int) -> int:
        """Return the driver-reported ordinal that owns an allocation."""
        ordinal = ctypes.c_int()
        self.check(
            self.cuPointerGetAttribute(
                ctypes.byref(ordinal), CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ptr
            ),
            "cuPointerGetAttribute(DEVICE_ORDINAL)",
        )
        return ordinal.value


class PinnedBuffer:
    """Explicitly retired DMA staging memory, not a Python-managed byte array.

    No destructor: an unconfirmed stream drain must never free a DMA source or
    destination just because Python unwinds. The notary frees these only after
    stream completion, otherwise abandons them to their allocating context's
    destruction. On failed destruction, they stay quarantined until process exit.
    """

    def __init__(self, cuda: Cuda, nbytes: int) -> None:
        self._cuda, self.nbytes = cuda, nbytes
        self.ptr = cuda.host_alloc(nbytes)

    def write(self, data: bytes) -> None:
        if len(data) > self.nbytes:
            raise ValueError("data exceeds pinned buffer capacity")
        ctypes.memmove(self.ptr, data, len(data))

    def read(self, nbytes: int | None = None) -> bytes:
        nbytes = self.nbytes if nbytes is None else nbytes
        if not 0 <= nbytes <= self.nbytes:
            raise ValueError("read exceeds pinned buffer capacity")
        return ctypes.string_at(self.ptr, nbytes)

    def close(self) -> None:
        if self.ptr:
            ptr = self.ptr
            # Detach BEFORE entering the driver: an interruption after success
            # must not turn a retry/GC into a second free of the same address.
            self.ptr = None
            self._cuda.host_free(ptr)


class DeviceBuffer:
    """Device allocation with a Python lifetime."""

    def __init__(self, cuda: Cuda, nbytes: int) -> None:
        self._cuda, self.nbytes = cuda, nbytes
        self.ptr = 0
        self.ptr = cuda.alloc(nbytes)

    @classmethod
    def from_bytes(cls, cuda: Cuda, data: bytes) -> "DeviceBuffer":
        b = cls(cuda, len(data))
        cuda.htod(b.ptr, data)
        return b

    def read(self, nbytes: int | None = None) -> bytes:
        return self._cuda.dtoh(self.ptr, self.nbytes if nbytes is None else nbytes)

    def close(self) -> None:
        if self.ptr:
            ptr = self.ptr
            self.ptr = 0
            try:
                self._cuda.free(ptr)
            except CudaError:
                pass

    def __del__(self) -> None:
        self.close()

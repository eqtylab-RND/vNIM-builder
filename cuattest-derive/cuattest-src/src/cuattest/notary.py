# SPDX-License-Identifier: Apache-2.0
"""The GPU notary: signing state retained on the device.

One process owns a CUDA context, generates a P-256 key inside the GPU, loads
the kernel from bytes it has hashed, and thereafter answers three questions:

    who are you          the session's did:key and the code that is running
    measure these        BLAKE3 over another process's live VRAM, via CUDA IPC
    sign what you saw    a measurement document plus EQTY statements, all
                         assembled and signed inside the kernel

The private scalar is derived in the kernel and is not copied out of device
globals by cuAttest. The trusted notary supplies its OS-CSPRNG seed and can
therefore reproduce it. The scalar dies with the process, so a notary's
identity spans exactly one run.

This process never loads a model; the process that owns the model never sees
the signing state. The notary is the trusted component; separating the model
process prevents a compromised workload from directly using that state. The
caller can check the measurement's internal fold and anyone with the pinned
public key can verify its signature.
"""

from __future__ import annotations

from ._build_config import ASSERTIONS_ENABLED

import ctypes
import json
import os
import re
import struct
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import ids
from ._cuda import IPC_HANDLE_BYTES, Cuda, DeviceBuffer, IpcImportRejectedError, PinnedBuffer
from ._hosthash import blake3_digest
from ._protocol import (
    MEASUREMENT_CLAIM,
    MEASUREMENT_DOCUMENT_FIELDS,
    MEASUREMENT_HASH_SCHEME,
    MEASUREMENT_OPERATION,
    parse_kernel_receipt,
    parse_measurement_document,
)
from ._statements import validate_statement_graph_structure
from .kernel import cubin_filename, cubin_metadata_filename, kernel_source, load_cubin

_REAL_CUDA_TYPE = Cuda
try:
    from ._native import Error as _NativeError
    from ._native import FusedRunner as _NativeFusedRunner
    from ._native import IpcCloseError as _NativeIpcCloseError
except ModuleNotFoundError as error:  # direct source-tree use before a package build
    if error.name != "cuattest._native":
        raise

    class _NativeError(RuntimeError):
        pass

    class _NativeIpcCloseError(_NativeError):
        pass

    _NativeFusedRunner = None

# P-256 constants in the layout the kernel's OFF_* offsets expect:
# P[4] ‖ N[4] ‖ mu_p[5] ‖ mu_n[5] ‖ Gx[4] ‖ Gy[4], little-endian u64 limbs,
# where mu = floor(2^512 / m) for Barrett reduction.
P256_CTX = (
    0xFFFFFFFFFFFFFFFF,
    0x00000000FFFFFFFF,
    0x0000000000000000,
    0xFFFFFFFF00000001,
    0xF3B9CAC2FC632551,
    0xBCE6FAADA7179E84,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFF00000000,
    0x0000000000000003,
    0xFFFFFFFEFFFFFFFF,
    0xFFFFFFFEFFFFFFFE,
    0x00000000FFFFFFFF,
    0x0000000000000001,
    0x012FFD85EEDF9BFE,
    0x43190552DF1A6C21,
    0xFFFFFFFEFFFFFFFF,
    0x00000000FFFFFFFF,
    0x0000000000000001,
    0xF4A13945D898C296,
    0x77037D812DEB33A0,
    0xF8BCE6E563A440F2,
    0x6B17D1F2E12C4247,
    0xCBB6406837BF51F5,
    0x2BCE33576B315ECE,
    0x8EE7EB4A7C0F9E16,
    0x4FE342E2FE1A7F9B,
)

# One slot per resident model copy that may chain statements concurrently.
# Bounded by CUATTEST_CHAIN_SLOTS_MAX in the kernel, which is covered by cubinCID.
_CHAIN_SLOTS_DEFAULT = 16

_THREADS = 128  # two 64-thread groups, one 128 KiB tile per group
_CHUNK = 1024  # BLAKE3's fixed semantic chunk size
_ASYNC_HASH_MIN_BYTES = 1 << 30


def _async_hash_threshold(arch: str, mode: str) -> int | None:
    """Select only qualified hardware automatically; never tax cached inputs.

    The extra shared memory helps large Blackwell working sets but hurts small
    cached inputs. Keep a separately compiled standard kernel and occupancy
    limit. 'async' is an explicit diagnostic override, not an older-GPU fallback.
    """
    if mode not in {"auto", "standard", "async"}:
        raise ValueError("CUATTEST_HASH_MODE must be auto, standard or async")
    capable = int(arch.removeprefix("sm_")) >= 80
    if mode == "async":
        if not capable:
            raise ValueError("asynchronous hashing requires sm_80 or newer")
        return 0
    if mode == "auto" and arch == "sm_120":
        return _ASYNC_HASH_MIN_BYTES
    return None


_TILE_CHUNKS = 128  # block-local scheduling tile: 128 BLAKE3 chunks = 128 KiB
# State statement, its credential and the measurement document: ~3.7 KiB in
# practice, the only variable part being the 64-byte model name. Must match
# kOutCapacity in _native.cpp; an overflow is a hard -4, so keep the headroom.
_OUT_CAP = 6144
_TS_LEN = 20  # the kernel reads exactly 20 bytes: YYYY-MM-DDTHH:MM:SSZ
_MAX_U64 = (1 << 64) - 1
_MAX_I32 = (1 << 31) - 1
_TENSOR_SPAN = struct.Struct("<QQQQQ")

# Defaults admit a full 96 GiB accelerator while bounding attacker-selected
# repeated spans. Count limits tiny-span metadata/output growth; byte limits
# bound repeated hashing; tile limits independently cap retained CV workspace.
DEFAULT_MAX_REQUEST_TENSORS = 16_384
DEFAULT_MAX_REQUEST_BYTES = 128 * 1024**3
DEFAULT_MAX_REQUEST_TILES = 1_048_576  # 128 GiB worth of full 128 KiB tiles


class NotaryError(RuntimeError):
    """The notary could not do what was asked."""


class GpuSessionAbortedError(NotaryError):
    """Unconfirmed queued work was stopped by destroying its CUDA context."""


class GpuCleanupUncertainError(NotaryError):
    """Queued GPU work and context destruction could not be confirmed."""


class IpcSessionAbortedError(GpuSessionAbortedError):
    """An IPC unmap failed, but context destruction completed cleanup."""


class IpcCleanupUncertainError(GpuCleanupUncertainError):
    """Neither IPC unmapping nor CUDA context destruction was confirmed."""


class _QueuedGpuWorkUnconfirmedError(RuntimeError):
    """A launch succeeded and the fallback backend could not drain its context."""


def _context_id(ctx):
    """Return a comparable value for real and test-double context handles."""
    return getattr(ctx, "value", ctx)


def _native_runner_requires_context_cleanup(runner) -> bool:
    """Read the native persistent sentinel, including its compatibility name."""
    return bool(
        getattr(runner, "context_cleanup_required", False)
        or getattr(runner, "ipc_cleanup_failed", False)
    )


def _require_popped_context(popped, expected, operation: str) -> None:
    if _context_id(popped) != _context_id(expected):
        raise NotaryError(
            f"CUDA context stack changed during {operation}: popped an unexpected context"
        )


@dataclass(frozen=True)
class GpuInfo:
    gpu_did: str
    gpu_pubkey_uncompressed: str
    kernel_cid: str
    cubin_cid: str
    compiler: str
    arch: str
    device: str
    device_ordinal: int
    pid: int
    host_backend: str = "unknown"
    device_uuid: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TensorRef:
    """One tensor to measure, in another process's address space.

    `handle` is a CUDA IPC memory handle for the allocator *segment*, hex
    encoded. Older torch ``_share_cuda_()`` clients prefix a 2-byte header,
    which is stripped if present. The tensor's bytes start at
    ``segment_base + seg_off + t_off``.

    References are client-selected. Driver ownership and bounds checks make
    them safe to read; they cannot establish that a span is a framework tensor
    or that an inference runtime used it.
    """

    handle: str
    nbytes: int
    seg_off: int = 0
    t_off: int = 0
    device: int = 0
    device_uuid: str | None = None

    def raw_handle(self) -> bytes:
        # Reject oversized text before bytes.fromhex allocates a decoded copy.
        # Wire handles have exactly 64 bytes, optionally prefixed by torch's
        # two-byte header, so no other input length can become valid.
        if not isinstance(self.handle, str):
            raise NotaryError("IPC handle must be a hexadecimal string")
        if len(self.handle) not in {IPC_HANDLE_BYTES * 2, (IPC_HANDLE_BYTES + 2) * 2}:
            raise NotaryError(
                f"IPC handle has {len(self.handle)} hexadecimal characters; "
                f"expected {IPC_HANDLE_BYTES} bytes ({IPC_HANDLE_BYTES * 2} characters) "
                f"or torch's {IPC_HANDLE_BYTES + 2}-byte form"
            )
        try:
            h = bytes.fromhex(self.handle)
        except (TypeError, ValueError) as e:
            raise NotaryError(f"IPC handle is not valid hexadecimal: {e}") from e
        if len(h) == IPC_HANDLE_BYTES + 2:
            h = h[2:]
        if len(h) != IPC_HANDLE_BYTES:
            raise NotaryError(
                f"IPC handle is {len(h)} bytes after header strip, expected {IPC_HANDLE_BYTES}"
            )
        return h

    @classmethod
    def from_dict(cls, d: dict) -> TensorRef:
        try:
            if not isinstance(d, dict):
                raise TypeError("reference must be an object")
            handle = d["handle"]
            if not isinstance(handle, str):
                raise TypeError("handle must be a hexadecimal string")
            values = {name: d.get(name, 0) for name in ("nbytes", "seg_off", "t_off")}
            values["device"] = d["device"]
            for name, value in values.items():
                # bool is an int subclass, but accepting true as an offset or
                # device ordinal is never meaningful protocol behaviour.
                if type(value) is not int:
                    raise TypeError(f"{name} must be an integer")
            ref = cls(handle=handle, device_uuid=d.get("device_uuid"), **values)
            ref.validate_metadata()
            return ref
        except KeyError as e:
            # Field names below are fixed by this parser. Never interpolate
            # the whole untrusted object: one request may be 64 MiB, and
            # reflecting it into an error would create another huge response.
            raise NotaryError(
                f"bad tensor reference: missing required field {e.args[0]!r}"
            ) from e
        except (TypeError, ValueError) as e:
            raise NotaryError(f"bad tensor reference: {e}") from e

    def validate_metadata(self) -> None:
        if self.device_uuid is not None and (
            type(self.device_uuid) is not str
            or re.fullmatch(
                r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", self.device_uuid
            )
            is None
        ):
            raise NotaryError("device_uuid must be a canonical GPU UUID")
        for name, value in (
            ("nbytes", self.nbytes),
            ("seg_off", self.seg_off),
            ("t_off", self.t_off),
            ("device", self.device),
        ):
            # TensorRef is also a public library type, so repeat the wire
            # parser's strict check here. Otherwise bool/float instances built
            # directly in Python can reach arithmetic later in this method.
            if type(value) is not int:
                raise NotaryError(f"tensor {name} must be an integer")
        if self.nbytes <= 0:
            raise NotaryError("tensor nbytes must be positive")
        if self.seg_off < 0 or self.t_off < 0:
            raise NotaryError("tensor offsets must be non-negative")
        for name, value in (
            ("nbytes", self.nbytes),
            ("seg_off", self.seg_off),
            ("t_off", self.t_off),
        ):
            if value > _MAX_U64:
                raise NotaryError(f"tensor {name} exceeds the CUDA address width")
        if not 0 <= self.device <= _MAX_I32:
            raise NotaryError("tensor device must be a non-negative CUDA ordinal")
        # Check additions independently of an allocation so no Python integer
        # can later be truncated through ctypes into a wrapped CUdeviceptr.
        if self.seg_off + self.t_off > _MAX_U64:
            raise NotaryError("tensor offsets overflow the CUDA address width")

    def pointer_in(
        self, mapped_base: int, allocation_base: int, allocation_nbytes: int
    ) -> int:
        """Validate this complete span against the imported allocation."""
        self.validate_metadata()
        relative = self.seg_off + self.t_off
        start = mapped_base + relative
        end = start + self.nbytes
        allocation_end = allocation_base + allocation_nbytes
        if (
            mapped_base > _MAX_U64
            or start > _MAX_U64
            or end > _MAX_U64 + 1
            or allocation_end > _MAX_U64 + 1
        ):
            raise NotaryError("tensor span overflows the CUDA address width")
        if start < allocation_base or end > allocation_end:
            raise NotaryError(
                "tensor span lies outside the imported allocation "
                f"(offset={relative}, nbytes={self.nbytes}, allocation={allocation_nbytes})"
            )
        return start


@dataclass(frozen=True)
class Measurement:
    digests: str
    model_root: str
    vram_cid: str
    tensor_count: int
    measured_at: str
    # Residency identity: which copy in memory, as opposed to what its bytes
    # are. Empty when the spans are buffers this process owns, which have no
    # IPC handle and therefore no cross-process identity.
    instance_cid: str = ""
    seconds: float = field(default=0.0)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _FusedPlan:
    descriptors: bytes
    reduction_offsets: bytes
    total_tiles: int
    secondary_tiles: int
    levels: int


@dataclass(frozen=True)
class _FusedResult:
    roots: bytes
    model_root: bytes
    receipt: str | None


def _fused_plan(spans: list[tuple[int, int]]) -> _FusedPlan:
    """Pack tensor descriptors and cross-tile reduction prefixes.

    Each descriptor retains the semantic 1 KiB BLAKE3 chunk count, but its
    workspace bases count 128-chunk scheduling tiles. The first seven tree
    levels stay in block-local shared memory and therefore need no global CV
    allocation.
    """
    descriptors = bytearray()
    tile_counts = []
    total_tiles = 0
    secondary_tiles = 0
    for ptr, nbytes in spans:
        nchunks = max(1, (nbytes + _CHUNK - 1) // _CHUNK)
        ntiles = (nchunks + _TILE_CHUNKS - 1) // _TILE_CHUNKS
        if ntiles > _MAX_U64 // 32 - total_tiles:
            raise NotaryError("fused BLAKE3 workspace exceeds the CUDA address width")
        descriptors += _TENSOR_SPAN.pack(
            ptr, nbytes, total_tiles, secondary_tiles, nchunks
        )
        tile_counts.append(ntiles)
        total_tiles += ntiles
        if ntiles > 1:
            secondary_tiles += (ntiles + 1) // 2

    levels = max(((count - 1).bit_length() for count in tile_counts), default=0)
    offsets = []
    current = tile_counts
    for _ in range(levels):
        prefix = 0
        offsets.append(prefix)
        next_counts = []
        for count in current:
            work = (count + 1) // 2 if count > 1 else 0
            prefix += work
            offsets.append(prefix)
            next_counts.append((count + 1) // 2 if count > 1 else 1)
        current = next_counts

    # The kernel does not dereference this pointer when levels == 0, but a
    # real allocation keeps its argument valid for CUDA's parameter checking.
    reduction_offsets = (
        struct.pack(f"<{len(offsets)}Q", *offsets) if offsets else bytes(8)
    )
    if ASSERTIONS_ENABLED:
        # Mirror the native planner's ABI/reduction postconditions. Never put
        # packing, state changes or driver operations inside an assertion.
        assert len(descriptors) == len(spans) * _TENSOR_SPAN.size
        assert len(reduction_offsets) == max(8, levels * (len(spans) + 1) * 8)
        assert 0 <= secondary_tiles <= total_tiles
        assert all(count == 1 for count in current)
        assert total_tiles >= len(spans) and 0 <= levels < 64
    return _FusedPlan(
        descriptors=bytes(descriptors),
        reduction_offsets=reduction_offsets,
        total_tiles=total_tiles,
        secondary_tiles=secondary_tiles,
        levels=levels,
    )


def host_entropy() -> bytes:
    """Return the key-generation seed from the operating-system CSPRNG.

    This is the sole entropy source credited by the key-generation design.
    The trusted notary host necessarily knows the seed; the CUDA kernel hashes
    it before mapping the result into a P-256 scalar and retains only that
    scalar in module-private device storage.
    """
    return os.urandom(32)


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _instance_root(tensors: list["TensorRef"]) -> bytes:
    """Fold which resident copy these spans came from, ignoring their content.

    modelRoot answers what the bytes are, and is identical for two identical
    models by construction. This answers which of them was measured: it folds
    the allocations themselves, so it stays constant across repeated
    observations of one resident copy and differs between two copies holding
    the same weights.

    Handles are hashed rather than carried. A raw CUipcMemHandle is live read
    access to the producer's memory, and receipts are made to be shared.

    Span order is significant, matching the submitted order the model fold
    uses. Reordering the same allocations yields a different instance and
    therefore starts a separate chain, so callers must keep the order stable
    across observations of one model.
    """
    parts = [struct.pack("<I", len(tensors))]
    for tensor in tensors:
        parts.append(blake3_digest(tensor.raw_handle()))
        parts.append(
            struct.pack("<QQQ", tensor.seg_off, tensor.t_off, tensor.nbytes)
        )
    return blake3_digest(b"".join(parts))


def _owned_instance_root(spans: list[tuple[int, int]]) -> bytes:
    """instanceID for device buffers this process owns, which have no IPC
    handle to identify them. Selftest and diagnostic paths only: a device
    pointer identifies an allocation just as well *within* one process, but it
    carries no cross-process meaning, so it must not be compared with an
    instanceID folded from real handles.
    """
    parts = [struct.pack("<I", len(spans))]
    for pointer, nbytes in spans:
        parts.append(blake3_digest(struct.pack("<Q", pointer)))
        parts.append(struct.pack("<QQQ", 0, 0, nbytes))
    return blake3_digest(b"".join(parts))


def _validated_model(model: str) -> bytes:
    if not isinstance(model, str):
        raise NotaryError("model must be a string")
    try:
        encoded = model.encode("ascii")
    except UnicodeEncodeError as e:
        raise NotaryError(
            "model must be 1-64 ASCII letters, digits, '/', '.', '_' or '-'"
        ) from e
    allowed = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-"
    if not 1 <= len(encoded) <= 64 or any(byte not in allowed for byte in encoded):
        raise NotaryError(
            "model must be 1-64 ASCII letters, digits, '/', '.', '_' or '-'"
        )
    return encoded


def _validated_request_limit(value: int, name: str, maximum: int) -> int:
    """Validate an operator-supplied denial-of-service boundary."""
    if type(value) is not int or not 0 < value <= maximum:
        raise NotaryError(f"{name} must be an integer from 1 through {maximum}")
    return value


class Notary:
    """A live notary session; callers must serialize requests and close().

    Each operation activates this session's context on its calling thread.
    MultiGpuNotary assigns distinct sessions to distinct workers; a single
    session's native workspaces must never be used concurrently.
    """

    def __init__(
        self,
        device: int = 0,
        artifact_dir: str | Path | None = None,
        *,
        max_request_tensors: int = DEFAULT_MAX_REQUEST_TENSORS,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_request_tiles: int = DEFAULT_MAX_REQUEST_TILES,
    ) -> None:
        if type(device) is not int or not 0 <= device <= _MAX_I32:
            raise NotaryError("device must be a non-negative CUDA ordinal")
        self.device_ordinal = device
        # Validate service limits before loading CUDA. Operators can raise
        # these deliberately, but no network client can override them.
        self.max_request_tensors = _validated_request_limit(
            max_request_tensors, "max_request_tensors", _MAX_I32
        )
        self.max_request_bytes = _validated_request_limit(
            max_request_bytes, "max_request_bytes", _MAX_U64
        )
        self.max_request_tiles = _validated_request_limit(
            max_request_tiles, "max_request_tiles", _MAX_U64
        )
        self._closed = False
        self._context_destroyed = False
        self.ctx = None
        self.module = None
        self._stream = None
        self._fallback_device_buffers: dict[str, DeviceBuffer] = {}
        self._fallback_host_buffers: dict[str, PinnedBuffer] = {}
        self._ctx_dev: DeviceBuffer | None = None
        self._chain_slots = int(
            os.environ.get("CUATTEST_CHAIN_SLOTS", _CHAIN_SLOTS_DEFAULT)
        )
        if self._chain_slots <= 0:
            raise NotaryError("CUATTEST_CHAIN_SLOTS must be positive")
        self._cubin_digest_dev: DeviceBuffer | None = None
        self._kernel_digest_dev: DeviceBuffer | None = None
        self._native_runner = None
        # Native teardown can detach the runner before CUDA destruction
        # succeeds. Its quarantine must outlive both that Python reference
        # and any specialized exception replaced during stack unwinding.
        self._native_context_cleanup_required = False
        # The Python IPC fallback keeps this sentinel true across every
        # interruptible import/close boundary. The HTTP layer consults it as a
        # final fail-closed guard before emitting acknowledgement headers.
        self._fallback_ipc_cleanup_required = False
        self._registration_cleanup_required = False
        # Set by declare_loaded_model(); no declared model, no host report.
        self._loaded_model_cid = None
        # The built report, frozen on first use. See _attach_model_state_statement.
        self._model_state_statement = None

        try:
            self.cu = Cuda()
            self.cu.init()
            self.dev = self.cu.device(device)
            major, minor = self.cu.compute_capability(self.dev)
            self.arch = f"sm_{major}{minor}"
            self._async_min_bytes = _async_hash_threshold(
                self.arch, os.environ.get("CUATTEST_HASH_MODE", "auto")
            )
            self.device_name = self.cu.device_name(self.dev)

            # Freeze the source once so cache validation and the advertised
            # identity cannot observe different CUATTEST_KERNEL_SRC contents.
            source = kernel_source()
            cubin, compiler, self.cubin_path = load_cubin(self.arch, source)
            if not cubin.startswith(b"\x7fELF"):
                raise NotaryError(f"{self.cubin_path} is not an ELF CUBIN")

            # These identities are computed by an independent host library
            # *before* module_load. The executable being identified must never
            # be trusted to report its own allowlist digest.
            kernel_digest = blake3_digest(source.encode())
            cubin_digest = blake3_digest(cubin)

            try:
                # cuCtxCreate *pushes* the new context. It must be popped after
                # initialization: cuCtxSetCurrent(previous) would replace that
                # entry and leave a duplicate of the embedding's context below
                # it, corrupting callers that later use cuCtxPopCurrent.
                self.ctx = self.cu.ctx_create(self.dev)
                self._stream = self.cu.stream_create()
                self.module = self.cu.module_load(cubin)
                self._fn = {
                    n: self.cu.function(self.module, n)
                    for n in (
                        "keygen_kernel",
                        "measure_model_fused_kernel",
                        "attest_measured_kernel",
                        "configure_chain_slots_kernel",
                    )
                }
                # A cooperative grid may contain only blocks that can all be
                # resident concurrently. The kernel uses grid-stride loops,
                # so this occupancy limit is independent of model size.
                self._fused_grid_limit = self.cu.cooperative_grid(
                    self.dev, self._fn["measure_model_fused_kernel"], _THREADS
                )
                self._async_grid_limit = 0
                if self._async_min_bytes is not None:
                    name = "measure_model_fused_async_kernel"
                    self._fn[name] = self.cu.function(self.module, name)
                    self._async_grid_limit = self.cu.cooperative_grid(
                        self.dev, self._fn[name], _THREADS
                    )

                ctx_bytes = b"".join(v.to_bytes(8, "little") for v in P256_CTX)
                self._ctx_dev = DeviceBuffer.from_bytes(self.cu, ctx_bytes)
                self._cubin_digest_dev = DeviceBuffer.from_bytes(self.cu, cubin_digest)
                self._kernel_digest_dev = DeviceBuffer.from_bytes(
                    self.cu, kernel_digest
                )

                self._configure_chain_slots()
                pub_x, pub_y = self._keygen()
                native_disabled = os.environ.get(
                    "CUATTEST_DISABLE_NATIVE_HOST", ""
                ).lower() in {"1", "true", "yes"}
                if (
                    _NativeFusedRunner is not None
                    and type(self.cu) is _REAL_CUDA_TYPE
                    and not native_disabled
                ):
                    try:
                        async_options = {}
                        if self._async_min_bytes is not None:
                            async_options = {
                                "async_measure_function": _context_id(
                                    self._fn["measure_model_fused_async_kernel"]
                                ),
                                "async_grid_limit": self._async_grid_limit,
                                "async_min_bytes": self._async_min_bytes,
                            }
                        self._native_runner = _NativeFusedRunner(
                            _context_id(self._fn["measure_model_fused_kernel"]),
                            _context_id(self._fn["attest_measured_kernel"]),
                            self._fused_grid_limit,
                            self._ctx_dev.ptr,
                            self._cubin_digest_dev.ptr,
                            self._kernel_digest_dev.ptr,
                            self.device_ordinal,
                            _context_id(self._stream),
                            **async_options,
                        )
                    except _NativeError as error:
                        raise NotaryError(
                            f"native CUDA host initialization failed: {error}"
                        ) from error
                self.info = GpuInfo(
                    gpu_did=ids.did_key_p256(pub_x, pub_y),
                    gpu_pubkey_uncompressed=ids.uncompressed_hex(pub_x, pub_y),
                    kernel_cid=ids.raw_cid(kernel_digest),
                    cubin_cid=ids.raw_cid(cubin_digest),
                    compiler=compiler,
                    host_backend=(
                        "C++" if self._native_runner is not None else "Python"
                    ),
                    arch=self.arch,
                    device=self.device_name,
                    device_ordinal=device,
                    device_uuid=self.cu.device_uuid(self.dev),
                    pid=os.getpid(),
                )

                # Issue the service's attestation over this session now that
                # the GPU DID exists. It is signed once and reused for every
                # receipt, because the facts it binds -- session key, CUBIN,
                # kernel source, device -- are fixed for the life of the
                # process. A failure to load the service identity is fatal:
                # publishing receipts whose code identity nobody vouches for
                # is exactly the situation this credential exists to remove.
                self._identity_statement = self._issue_identity_attestation()


                if artifact_dir:
                    d = Path(artifact_dir)
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "p256_cuda_notary_b3.cu").write_text(source)
                    (d / cubin_filename(self.arch)).write_bytes(cubin)
                    (d / cubin_metadata_filename(self.arch)).write_text(
                        json.dumps(
                            {
                                "source_blake3": kernel_digest.hex(),
                                "cubin_blake3": cubin_digest.hex(),
                                # load_cubin decorates cached versions for
                                # display; the copied artifact records the raw
                                # compiler version in its own bound sidecar.
                                "compiler": compiler.removesuffix(" (prebuilt)"),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            finally:
                if self.ctx is not None:
                    popped = self.cu.ctx_pop_current()
                    _require_popped_context(popped, self.ctx, "notary initialization")
        except BaseException:
            # Construction has transferred context ownership to this object as
            # soon as cuCtxCreate succeeds. Process-control exceptions need the
            # same cleanup as ordinary failures; never mask the original if
            # best-effort teardown itself reports another exception.
            try:
                self.close()
            except BaseException:  # noqa: BLE001,S110 - preserve constructor failure
                pass
            raise

    def _ensure_open(self) -> None:
        if self._closed or self.ctx is None:
            raise NotaryError("notary session is closed")

    @contextmanager
    def _activate(self):
        """Make this instance's context current, then restore the caller's.

        CUDA contexts are thread-current rather than object-current. Without
        this guard, constructing a second Notary makes the first one's module
        and allocations invalid for subsequent driver calls on the same
        thread.
        """
        self._ensure_open()
        activated_ctx = self.ctx
        pushed = _context_id(self.cu.ctx_get_current()) != _context_id(activated_ctx)
        if pushed:
            # A library must not replace an embedding's stack entry with
            # cuCtxSetCurrent. Push our context and pop exactly that entry so
            # nested CUDA users observe an unchanged stack on return.
            self.cu.ctx_push_current(activated_ctx)
        try:
            yield
        finally:
            if pushed:
                current = self.cu.ctx_get_current()
                if _context_id(current) == _context_id(activated_ctx):
                    popped = self.cu.ctx_pop_current()
                    _require_popped_context(popped, activated_ctx, "notary operation")
                elif not self._closed:
                    # close() legitimately destroys and therefore pops an
                    # active context. Any other stack change is a caller bug
                    # that must not be mistaken for successful restoration.
                    raise NotaryError(
                        "CUDA context stack changed during notary operation"
                    )

    # ── key ──────────────────────────────────────────────────────────────────

    def _issue_identity_attestation(self) -> tuple[str, dict] | None:
        """Sign this session's IdentityAttestation with the service key.

        Returns None when the service identity is disabled, in which case
        receipts carry only GPU-signed statements. That is a real reduction --
        nobody then vouches for which CUBIN the kernel was handed -- so it has
        to be asked for explicitly rather than happening on a missing import.
        """
        from . import _identity
        from ._hostkey import load_or_create

        if os.environ.get("CUATTEST_HOST_IDENTITY", "1").lower() in {
            "0",
            "false",
            "no",
        }:
            self._host_key = None
            return None
        self._host_key = load_or_create()
        credential = _identity.build(
            self._host_key,
            self.info.gpu_did,
            f"urn:cid:{self.info.cubin_cid}",
            f"urn:cid:{self.info.kernel_cid}",
            self.info.device_uuid,
            _utc_timestamp(),
        )
        return _identity.registration(
            credential, self._host_key.did, credential["validFrom"]
        )

    def _attach_identity_statement(self, receipt: dict) -> None:
        """Add the service's attestation to a manifest the GPU produced.

        The kernel cannot carry this statement: it is signed by a key the GPU
        never sees, over facts the GPU cannot check. Adding it here is what
        makes the manifest a joint document -- GPU-signed statements about what
        was measured, plus a service-signed statement about what was running --
        so the measurement document and its signature remain untouched kernel
        bytes either way.
        """
        if self._identity_statement is None:
            return
        statement_id, statement = self._identity_statement
        manifest = receipt.get("manifest")
        if not isinstance(manifest, dict):  # pragma: no cover - kernel invariant
            raise NotaryError("kernel receipt has no manifest to extend")
        statements = manifest.get("statements")
        if not isinstance(statements, dict):  # pragma: no cover
            raise NotaryError("kernel manifest has no statement map")
        if statement_id in statements:  # pragma: no cover - id is content-derived
            raise NotaryError("identity statement collides with a kernel statement")
        statements[statement_id] = statement

    def declare_loaded_model(self, model_cid: str | None) -> None:
        """Bind the loaded copy to the caller's signed Model asset CID.

        Use the same raw-file or collection CID used by the SDK's signed
        computation. The host reports it beside the independently measured
        GPU modelRoot; equality is not required or proof of faithful loading.
        A caller may also supply a raw source tensor CID for direct comparison.
        Passing None clears the declaration.
        """
        if model_cid is not None:
            from .ids import validate_model_cid
            try:
                model_cid = validate_model_cid(model_cid)
            except ValueError as error:
                raise NotaryError(str(error)) from error
        self._loaded_model_cid = model_cid
        # A newly declared model invalidates any report about the old one.
        self._model_state_statement = None

    def _attach_model_state_statement(
        self, receipt: dict, measurement: "Measurement"
    ) -> None:
        """Add the host's report about the copy this measurement covers.

        Deliberately issued here rather than at load: the report names both the
        content id the host folded from disk and the root the GPU folded from
        VRAM, and the second does not exist until a measurement does. What the
        host contributes was fixed at load time; this is only where the two
        halves meet.

        Silent when the host has no durable key or no declared model -- the
        same posture as the identity attestation, and for the same reason: an
        unsigned or invented claim would be worse than none.
        """
        model_cid = self._loaded_model_cid
        host_key = getattr(self, "_host_key", None)
        if model_cid is None or host_key is None or not measurement.instance_cid:
            return
        from . import _modelstate

        # Built once per (copy, file, measured root) and reused verbatim. The
        # report restates one fixed fact, so signing the same copy repeatedly
        # must not grow the manifest a statement per receipt. The credential id
        # is already derived from those three values, but the registration
        # wrapper also covers the timestamp -- so without this, one fact would
        # appear under a new wrapper id every time.
        key = (measurement.instance_cid, model_cid, measurement.vram_cid)
        if self._model_state_statement is None or self._model_state_statement[0] != key:
            self._model_state_statement = (
                key,
                *_modelstate.statement(
                    host_key,
                    f"urn:cid:{measurement.instance_cid}",
                    model_cid,
                    f"urn:cid:{measurement.vram_cid}",
                    measurement.measured_at,
                ),
            )
        _, statement_id, statement = self._model_state_statement
        statements = receipt["manifest"]["statements"]
        if statement_id in statements:  # pragma: no cover - id is content-derived
            raise NotaryError("model state statement collides with another")
        statements[statement_id] = statement

    def _configure_chain_slots(self) -> None:
        """Fix how many resident copies may chain statements in this session.

        One slot tracks one instanceID, so this bounds how many distinct
        model copies the notary can sign for concurrently -- not how many
        signatures each may accumulate. Exceeding it fails the request rather
        than recycling a slot, because a recycled slot would emit a genesis
        statement for a copy that already has predecessors.
        """
        status = DeviceBuffer(self.cu, 4)
        try:
            self.cu.launch(
                self._fn["configure_chain_slots_kernel"],
                1,
                1,
                [
                    ctypes.c_int(self._chain_slots),
                    ctypes.c_ulonglong(status.ptr),
                ],
                stream=self._stream,
            )
            self.cu.stream_sync(self._stream)
            rc = int.from_bytes(status.read(4), "little", signed=True)
            if rc != 0:
                raise NotaryError(
                    f"chain slot configuration failed: status={rc} "
                    f"({_STATUS.get(rc, 'unknown')})"
                )
        finally:
            status.close()

    def _keygen(self) -> tuple[bytes, bytes]:
        self._ensure_open()
        seed = host_entropy()
        ent = DeviceBuffer.from_bytes(self.cu, seed)
        px, py = DeviceBuffer(self.cu, 32), DeviceBuffer(self.cu, 32)
        work_unconfirmed = False
        try:
            work_unconfirmed = True
            self.cu.launch(
                self._fn["keygen_kernel"],
                1,
                1,
                [
                    ctypes.c_ulonglong(self._ctx_dev.ptr),
                    ctypes.c_ulonglong(ent.ptr),
                    ctypes.c_ulonglong(px.ptr),
                    ctypes.c_ulonglong(py.ptr),
                ],
                stream=self._stream,
            )
            # Scrub after keygen in the SAME stream, before its one completion
            # wait. A default-stream memset could race this nonblocking stream.
            self.cu.memset0_async(ent.ptr, 32, self._stream)
            self.cu.stream_sync(self._stream)
            work_unconfirmed = False
            x, y = px.read(32), py.read(32)
        finally:
            try:
                if work_unconfirmed:
                    self.cu.stream_sync(self._stream)
                    work_unconfirmed = False
            finally:
                for buffer in (ent, px, py):
                    if work_unconfirmed:
                        self._abandon_context_allocations = True
                        buffer.ptr = 0
                    else:
                        buffer.close()
        if x == bytes(32) or y == bytes(32):
            raise NotaryError("keygen produced a zero public key")
        return x, y

    # ── hashing ──────────────────────────────────────────────────────────────

    def hash_dptr(self, ptr: int, nbytes: int, *, producer_stream=None) -> bytes:
        """Hash a device range after prior work on its producer CUDA stream.

        By default the producer is this notary context's legacy default stream.
        Pass a same-context CUDA stream handle for non-default producers. All
        writers must precede that stream's handoff and remain quiescent until
        this call returns; the allocation must remain alive throughout.
        """
        with self._activate():
            try:
                # Direct pointers have no IPC exporter's readiness barrier.
                # Our NON_BLOCKING stream cannot see even preceding legacy
                # cuMemsetD8 writes without an explicit dependency. Queue an
                # event handoff, not a context-wide or CPU-side producer wait.
                # Keep this at the public boundary: hash_bytes uploads are
                # already complete, and IPC spans were synchronized on export.
                self.cu.stream_wait_stream(self._stream, producer_stream)
                return self._hash_dptr_active(ptr, nbytes)
            except _QueuedGpuWorkUnconfirmedError as error:
                self._abort_unconfirmed_gpu_work(error)

    def _hash_dptr_active(self, ptr: int, nbytes: int) -> bytes:
        """Hash already-ordered input with this notary's context current."""
        self._ensure_open()
        if type(ptr) is not int or not 0 <= ptr <= _MAX_U64:
            raise NotaryError("device pointer is outside the CUDA address width")
        if type(nbytes) is not int or not 0 <= nbytes <= _MAX_U64:
            raise NotaryError("hash size is outside the CUDA address width")
        if ptr + nbytes > _MAX_U64 + 1:
            raise NotaryError("hash span overflows the CUDA address width")
        return self._launch_fused_active([(ptr, nbytes)]).roots

    def hash_bytes(self, data: bytes) -> bytes:
        with self._activate():
            buf = DeviceBuffer.from_bytes(self.cu, data)
            try:
                return self._hash_dptr_active(buf.ptr, len(data))
            except _QueuedGpuWorkUnconfirmedError as error:
                # The context now owns this allocation until destruction. Do
                # not let DeviceBuffer.__del__ issue cuMemFree if destruction
                # itself fails and queued work may still reference the bytes.
                buf.ptr = 0
                self._abort_unconfirmed_gpu_work(error)
            finally:
                buf.close()

    def _abort_unconfirmed_gpu_work(self, error: BaseException) -> None:
        """Destroy a poisoned session before direct-span storage can be reused."""
        self._abandon_context_allocations = True
        try:
            self.close()
        except BaseException as destroy_error:
            raise GpuCleanupUncertainError(
                "queued CUDA work could not be synchronized and context "
                "destruction was not confirmed; caller-owned spans must not be freed"
            ) from destroy_error
        raise GpuSessionAbortedError(
            "queued CUDA work could not be synchronized; the notary context was destroyed"
        ) from error

    def _launch_fused_active(
        self,
        spans: list[tuple[int, int]],
        measured_at: str | None = None,
        model_bytes: bytes | None = None,
        instance_root: bytes | None = None,
    ) -> _FusedResult:
        """Hash every span, fold the model root, and optionally sign once.

        The notary context must already be current. All chunk hashing and
        tree levels execute inside one occupancy-bounded cooperative grid;
        an optional same-stream finalization kernel consumes its private root.
        Metadata upload, both kernels, and the final output download share one
        private nonblocking stream. Wait once for that stream, not for unrelated
        work in the context. Pinned DMA buffers survive even a failed drain.
        """
        self._ensure_open()
        if not spans or len(spans) > _MAX_I32:
            raise NotaryError("fused hashing requires 1 to 2147483647 spans")
        signing = model_bytes is not None
        if signing:
            if measured_at is None or len(measured_at) != _TS_LEN:
                raise NotaryError(
                    f"ts must be exactly {_TS_LEN} bytes "
                    f"(YYYY-MM-DDTHH:MM:SSZ), got {len(measured_at or '')!r}"
                )
            try:
                timestamp = measured_at.encode("ascii")
            except UnicodeEncodeError as e:
                raise NotaryError("timestamp must be ASCII") from e
            model_blob = model_bytes or b"\0"
            # Callers holding IPC handles supply the real residency fold; owned
            # device buffers have no handle, so they fall back to pointer
            # identity. Both backends need it, so resolve it before branching.
            instance_blob = (
                instance_root
                if instance_root is not None
                else _owned_instance_root(spans)
            )
            if len(instance_blob) != 32:
                raise NotaryError("instance_root must be a 32-byte digest")
        else:
            timestamp = bytes(_TS_LEN)
            model_blob = b"\0"
            instance_blob = bytes(32)

        native_runner = getattr(self, "_native_runner", None)
        if native_runner is not None:
            try:
                roots, model_root, receipt = native_runner.run_spans(
                    spans,
                    timestamp if signing else None,
                    model_bytes if signing else None,
                    instance_blob,
                )
            except BaseException as error:
                # launch() persists this sentinel when both synchronization
                # attempts fail after CUDA accepted work. Preserve that state
                # as a distinct exception so the owning call destroys the
                # context before any source or workspace can be released.
                if _native_runner_requires_context_cleanup(native_runner):
                    raise _QueuedGpuWorkUnconfirmedError(
                        "queued CUDA work could not be synchronized"
                    ) from error
                if isinstance(error, _NativeError):
                    raise NotaryError(str(error)) from error
                if isinstance(error, UnicodeDecodeError):
                    raise NotaryError(
                        f"kernel produced invalid UTF-8: {error}"
                    ) from error
                raise
            return _FusedResult(roots, model_root, receipt)

        plan = _fused_plan(spans)
        reduction_offset = len(plan.descriptors)
        timestamp_offset = reduction_offset + len(plan.reduction_offsets)
        model_offset = timestamp_offset + len(timestamp)
        instance_offset = model_offset + len(model_blob)
        metadata = (
            plan.descriptors
            + plan.reduction_offsets
            + timestamp
            + model_blob
            + instance_blob
        )

        roots_nbytes = len(spans) * 32
        model_root_offset = roots_nbytes
        json_offset = model_root_offset + 32
        json_capacity = _OUT_CAP if signing else 0
        out_len_offset = json_offset + json_capacity
        # Unsigned calls have no receipt length. Overlay the status on that
        # otherwise unwritten slot so the DtoH copy never reads uninitialized
        # device memory. Signed layout remains length followed by status.
        status_offset = out_len_offset + (4 if signing else 0)

        buffers = []
        work_unconfirmed = False
        try:
            metadata_d = self._fallback_buffer("metadata", len(metadata))
            buffers.append(metadata_d)
            work_a = self._fallback_buffer("work_a", plan.total_tiles * 32)
            buffers.append(work_a)
            work_b = self._fallback_buffer("work_b", plan.secondary_tiles * 32)
            buffers.append(work_b)
            output = self._fallback_buffer("output", status_offset + 4)
            buffers.append(output)
            metadata_h = self._fallback_buffer("metadata", len(metadata), host=True)
            output_h = self._fallback_buffer("output", status_offset + 4, host=True)
            metadata_h.write(metadata)
            requested_blocks = (plan.total_tiles + 1) // 2
            threshold = getattr(self, "_async_min_bytes", None)
            use_async = threshold is not None and sum(n for _, n in spans) >= threshold
            function_name = "measure_model_fused_async_kernel" if use_async else "measure_model_fused_kernel"
            grid_limit = self._async_grid_limit if use_async else self._fused_grid_limit
            grid = min(grid_limit, max(1, requested_blocks))
            if ASSERTIONS_ENABLED:
                assert 1 <= grid <= grid_limit
                assert output_h.nbytes >= status_offset + 4
                assert metadata_h.nbytes >= len(metadata)
            work_may_be_queued = False
            try:
                # Publish the conservative state before entering the driver.
                # A process-control exception can be delivered at the Python
                # call-return boundary after CUDA accepted the launch but
                # before a following assignment would execute.
                work_may_be_queued = True
                work_unconfirmed = True
                self.cu.htod_async(
                    metadata_d.ptr, metadata_h.ptr, len(metadata), self._stream
                )
                self.cu.launch_cooperative(
                    self._fn[function_name],
                    grid,
                    _THREADS,
                    [
                        ctypes.c_ulonglong(metadata_d.ptr),
                        ctypes.c_int(len(spans)),
                        ctypes.c_ulonglong(metadata_d.ptr + reduction_offset),
                        ctypes.c_int(plan.levels),
                        ctypes.c_ulonglong(plan.total_tiles),
                        ctypes.c_ulonglong(work_a.ptr),
                        ctypes.c_ulonglong(work_b.ptr),
                        ctypes.c_ulonglong(output.ptr),
                        ctypes.c_ulonglong(output.ptr + model_root_offset),
                        ctypes.c_int(1 if signing else 0),
                        ctypes.c_ulonglong(output.ptr + status_offset),
                    ],
                    stream=self._stream,
                )
                if signing:
                    self.cu.launch(
                        self._fn["attest_measured_kernel"],
                        1,
                        _THREADS,
                        [
                            ctypes.c_int(len(spans)),
                            ctypes.c_ulonglong(self._ctx_dev.ptr),
                            ctypes.c_ulonglong(metadata_d.ptr + timestamp_offset),
                            ctypes.c_ulonglong(metadata_d.ptr + model_offset),
                            ctypes.c_int(len(model_bytes) if signing else 0),
                            ctypes.c_ulonglong(self._cubin_digest_dev.ptr),
                            ctypes.c_ulonglong(self._kernel_digest_dev.ptr),
                            ctypes.c_ulonglong(metadata_d.ptr + instance_offset),
                            ctypes.c_int(self.device_ordinal),
                            ctypes.c_ulonglong(output.ptr + json_offset),
                            ctypes.c_int(json_capacity),
                            ctypes.c_ulonglong(output.ptr + out_len_offset),
                            ctypes.c_ulonglong(output.ptr + status_offset),
                        ],
                        stream=self._stream,
                    )
                self.cu.dtoh_async(
                    output_h.ptr, output.ptr, status_offset + 4, self._stream
                )
                self.cu.stream_sync(self._stream)
                work_unconfirmed = False
                work_may_be_queued = False
            except BaseException:
                if work_may_be_queued:
                    # The measurement can still be using imported IPC memory
                    # when either launch reports failure. Drain conservatively
                    # before a caller is allowed to unmap those allocations.
                    try:
                        self.cu.stream_sync(self._stream)
                        work_unconfirmed = False
                    except BaseException as drain_error:
                        # These allocations must survive until context
                        # destruction; abandoning their Python wrappers avoids
                        # a later close/GC issuing cuMemFree before completion.
                        for buffer in buffers:
                            buffer.ptr = 0
                        self._abandon_context_allocations = True
                        raise _QueuedGpuWorkUnconfirmedError(
                            "queued CUDA work could not be synchronized"
                        ) from drain_error
                raise
            raw = output_h.read(status_offset + 4)

            rc = int.from_bytes(
                raw[status_offset : status_offset + 4], "little", signed=True
            )
            if rc != 0:
                raise NotaryError(
                    "fused measurement/finalization "
                    f"status={rc} ({_STATUS.get(rc, 'unknown')})"
                )

            receipt = None
            if signing:
                n = int.from_bytes(
                    raw[out_len_offset : out_len_offset + 4], "little", signed=True
                )
                if not 0 < n <= json_capacity:
                    raise NotaryError(f"kernel reported an output length of {n}")
                try:
                    receipt = raw[json_offset : json_offset + n].decode()
                except UnicodeDecodeError as e:
                    raise NotaryError(f"kernel produced invalid UTF-8: {e}") from e
            return _FusedResult(
                roots=raw[:roots_nbytes],
                model_root=raw[model_root_offset : model_root_offset + 32],
                receipt=receipt,
            )
        finally:
            for buffer in reversed(buffers):
                if work_unconfirmed:
                    self._abandon_context_allocations = True
                    buffer.ptr = 0

    def _fallback_buffer(self, name: str, nbytes: int, *, host: bool = False):
        """Grow drained workspaces before submission; retain them between calls.

        Even with nonblocking streams, cuMemAlloc/cuMemAllocHost/cuMemFree can
        synchronize. Like the native runner, the fallback keeps those calls off
        its steady-state request path. Growth frees the old buffer first to avoid
        transiently doubling VRAM use for near-capacity models.
        """
        cache = self._fallback_host_buffers if host else self._fallback_device_buffers
        buffer = cache.get(name)
        if buffer is None or not buffer.ptr or buffer.nbytes < nbytes:
            if buffer is not None:
                # Detach before the interruptible free/allocation pair. Failed
                # growth must not leave a closed old buffer reusable by a
                # subsequent smaller request.
                del cache[name]
                buffer.close()
            buffer = (PinnedBuffer if host else DeviceBuffer)(self.cu, nbytes)
            cache[name] = buffer
        return buffer

    # ── measure ──────────────────────────────────────────────────────────────

    @property
    def ipc_cleanup_required(self) -> bool:
        """Whether sending response headers could release a live IPC producer."""
        return bool(
            getattr(self, "_fallback_ipc_cleanup_required", False)
            or getattr(self, "_registration_cleanup_required", False)
            or getattr(self, "_native_context_cleanup_required", False)
            or _native_runner_requires_context_cleanup(
                getattr(self, "_native_runner", None)
            )
        )

    def _abort_uncertain_fallback_ipc(self, error: BaseException) -> None:
        """Destroy a fallback context whose mapping ownership is uncertain."""
        cleanup_error = error
        try:
            self.close()
        except BaseException as close_error:  # noqa: BLE001 - classify after checking destruction
            cleanup_error = close_error
        # close() can propagate a module-unload error even after cuCtxDestroy
        # succeeded. Conversely, a repeated close() is a no-op after *failed*
        # destruction. Neither its return nor _closed/ctx=None proves mappings
        # were released: only the driver's confirmed destruction permits ACKs.
        if not getattr(self, "_context_destroyed", False):
            # Keep the persistent sentinel set. The server checks it even if
            # allocating this specialized exception fails under memory pressure.
            raise IpcCleanupUncertainError(
                "Python fallback IPC cleanup failed and context destruction "
                "was not confirmed"
            ) from cleanup_error
        self._fallback_ipc_cleanup_required = False
        if not isinstance(error, Exception):
            # KeyboardInterrupt/SystemExit still terminate their caller, but
            # only after context destruction has released any hidden mapping.
            raise error
        if not isinstance(cleanup_error, Exception):
            # An interrupt during close deserves the same treatment.
            raise cleanup_error
        raise IpcSessionAbortedError(
            "Python fallback IPC ownership became uncertain; "
            "the notary context was destroyed"
        ) from cleanup_error

    def _run_fallback_ipc(
        self,
        tensors: list[TensorRef],
        raw_handles: list[bytes],
        measured_at: str,
        model_bytes: bytes | None,
        instance_root: bytes,
    ) -> _FusedResult:
        """Import, hash, and close mappings with a fail-closed ownership guard.

        Python cannot atomically pair a CUDA driver call with a following list
        update: an asynchronous exception can arrive at either call-return
        boundary. The persistent sentinel is therefore set before any import
        and cleared only after every close is confirmed. Any exception in a
        gap destroys the entire context instead of acknowledging the request.
        """
        self._fallback_ipc_cleanup_required = True
        try:
            return self._run_fallback_ipc_guarded(
                tensors, raw_handles, measured_at, model_bytes, instance_root
            )
        except BaseException as error:
            if self._fallback_ipc_cleanup_required:
                self._abort_uncertain_fallback_ipc(error)
            raise

    def _run_fallback_ipc_guarded(
        self,
        tensors: list[TensorRef],
        raw_handles: list[bytes],
        measured_at: str,
        model_bytes: bytes | None,
        instance_root: bytes,
    ) -> _FusedResult:
        """Inner fallback operation; caller owns the persistent safety guard."""
        by_handle: dict[bytes, tuple[int, int, int]] = {}
        owned_mappings: list[int | None] = [None] * len(tensors)
        owned_count = 0
        untracked_import_possible = False
        validated: list[tuple[TensorRef, int]] = []
        fused_result: _FusedResult | None = None
        operation_error: BaseException | None = None

        try:
            for tensor, raw in zip(tensors, raw_handles):
                allocation = by_handle.get(raw)
                if allocation is None:
                    # Set this before entering the driver. If ipc_open returns
                    # a pointer and an exception lands before fixed-ledger
                    # publication, context destruction still owns that import.
                    untracked_import_possible = True
                    try:
                        mapped_base = self.cu.ipc_open(raw)
                    except IpcImportRejectedError:
                        # A driver-confirmed rejection owns no mapping. Keep
                        # serving after closing earlier imports in this batch;
                        # other exceptions retain the uncertain-import guard.
                        untracked_import_possible = False
                        raise
                    owned_mappings[owned_count] = mapped_base
                    owned_count += 1
                    untracked_import_possible = False
                    by_handle[raw] = (mapped_base, 0, 0)
                    allocation_device = self.cu.pointer_device(mapped_base)
                    if allocation_device != self.device_ordinal:
                        raise NotaryError(
                            f"IPC allocation belongs to cuda:{allocation_device}, "
                            f"but this notary uses cuda:{self.device_ordinal}"
                        )
                    allocation_base, allocation_nbytes = self.cu.address_range(
                        mapped_base
                    )
                    allocation = (mapped_base, allocation_base, allocation_nbytes)
                    by_handle[raw] = allocation
                mapped_base, allocation_base, allocation_nbytes = allocation
                validated.append(
                    (
                        tensor,
                        tensor.pointer_in(
                            mapped_base, allocation_base, allocation_nbytes
                        ),
                    )
                )

            fused_result = self._launch_fused_active(
                [(ptr, tensor.nbytes) for tensor, ptr in validated],
                measured_at,
                model_bytes,
                instance_root if model_bytes is not None else None,
            )
        except BaseException as error:  # noqa: BLE001 - includes process-control errors
            operation_error = error

        if operation_error is not None and (
            isinstance(operation_error, _QueuedGpuWorkUnconfirmedError)
            or getattr(self, "_abandon_context_allocations", False)
        ):
            # Explicit unmapping could race the kernel. Leave all imports to
            # context destruction in the outer persistent-guard handler, even
            # if unwinding replaced the specialized exception after failed DMA.
            raise operation_error

        first_close_error: BaseException | None = None
        mapping_index = 0
        while mapping_index < owned_count:
            ptr = owned_mappings[mapping_index]
            try:
                if ptr is None:  # pragma: no cover - fixed-ledger invariant
                    raise NotaryError("CUDA IPC ownership ledger is incomplete")
                self.cu.ipc_close(ptr)
            except BaseException as error:  # noqa: BLE001 - cleanup must fail closed
                if first_close_error is None:
                    first_close_error = error
            mapping_index += 1

        if first_close_error is not None:
            # Keep the sentinel set so the outer handler destroys the context.
            raise first_close_error
        if untracked_import_possible:
            # The driver may have installed a mapping whose returned pointer
            # never reached the fixed ledger. Only context destruction can
            # prove cleanup; preserve the original error when one exists.
            if operation_error is not None:
                raise operation_error
            raise NotaryError("CUDA IPC import ownership was not recorded")

        # Every successfully returned import is now confirmed closed. Clear
        # the sentinel before re-raising an ordinary operation failure so the
        # server may safely send its normal error response.
        self._fallback_ipc_cleanup_required = False
        if operation_error is not None:
            raise operation_error
        if fused_result is None:  # pragma: no cover - defensive invariant
            raise NotaryError("fused measurement returned no result")
        return fused_result

    def _validate_tensors(self, tensors: list[TensorRef]) -> list[bytes]:
        """Preflight every bound and routing hint before any IPC import."""
        self._ensure_open()
        if not tensors:
            raise NotaryError("no tensors to measure")
        # Reject all request-only invariants before importing anything. More
        # importantly, all allocation-backed spans below are validated before
        # the first hash kernel launches, so one bad trailing tensor cannot
        # poison the CUDA context after earlier work has begun.
        # This is deliberately only a memory-safety boundary: clients choose
        # the spans, and validation cannot bind them to an executing model.
        max_tensors = getattr(self, "max_request_tensors", DEFAULT_MAX_REQUEST_TENSORS)
        max_bytes = getattr(self, "max_request_bytes", DEFAULT_MAX_REQUEST_BYTES)
        max_tiles = getattr(self, "max_request_tiles", DEFAULT_MAX_REQUEST_TILES)
        if len(tensors) > max_tensors:
            raise NotaryError(
                f"request has {len(tensors)} tensors; limit is {max_tensors}"
            )

        aggregate_bytes = 0
        aggregate_tiles = 0
        for tensor in tensors:
            if not isinstance(tensor, TensorRef):
                raise NotaryError("tensors must contain TensorRef objects")
            # Producer ordinals are process-local and may be remapped by
            # CUDA_VISIBLE_DEVICES. Validate their shape here, but use the
            # driver-reported owner after import as the authoritative check.
            tensor.validate_metadata()
            aggregate_bytes += tensor.nbytes
            if aggregate_bytes > max_bytes:
                raise NotaryError(
                    f"request spans {aggregate_bytes} aggregate bytes; limit is {max_bytes}"
                )
            aggregate_tiles += 1 + (tensor.nbytes - 1) // (_CHUNK * _TILE_CHUNKS)
            if aggregate_tiles > max_tiles:
                raise NotaryError(
                    f"request needs {aggregate_tiles} scheduling tiles; "
                    f"limit is {max_tiles}"
                )

        # Decode handles only after all aggregate work bounds pass. Repeating
        # one valid allocation still counts once per logical span because the
        # fused kernel hashes every occurrence even though mapping is deduped.
        raw_handles = [tensor.raw_handle() for tensor in tensors]
        for tensor in tensors:
            if (
                tensor.device_uuid is not None
                and tensor.device_uuid != self.info.device_uuid
            ):
                # The UUID is only a routing hint, not proof of ownership.
                # The driver must still confirm the imported allocation's
                # owner below, even when the client names the expected UUID.
                raise NotaryError("tensor device_uuid does not match this notary's GPU")

        return raw_handles

    def _measure_request(
        self,
        tensors: list[TensorRef],
        measured_at: str,
        model_bytes: bytes | None = None,
    ) -> tuple[Measurement, bytes, str | None]:
        """Map another process's allocations over IPC and hash them in place.

        Each unique segment is mapped once. Nothing is copied: one cooperative
        kernel computes every tensor digest and the model root. When requested,
        a small same-stream kernel consumes that root privately and signs the
        receipt over the caller's live VRAM measurement.
        """
        raw_handles = self._validate_tensors(tensors)
        # Which resident copy this is, folded from the handles the client
        # submitted. Content-independent, so two identical models in separate
        # allocations get separate identities -- and separate chains.
        instance_root = _instance_root(tensors) if model_bytes is not None else b"\0" * 32

        started = time.perf_counter()
        native_runner = getattr(self, "_native_runner", None)
        if native_runner is not None:
            if model_bytes is None:
                native_timestamp = None
            else:
                if len(measured_at) != _TS_LEN:
                    raise NotaryError(
                        f"ts must be exactly {_TS_LEN} bytes "
                        f"(YYYY-MM-DDTHH:MM:SSZ), got {len(measured_at)!r}"
                    )
                try:
                    native_timestamp = measured_at.encode("ascii")
                except UnicodeEncodeError as error:
                    raise NotaryError("timestamp must be ASCII") from error
            try:
                roots, model_root, receipt = native_runner.run_ipc(
                    [
                        (raw, tensor.nbytes, tensor.seg_off, tensor.t_off)
                        for tensor, raw in zip(tensors, raw_handles)
                    ],
                    native_timestamp,
                    model_bytes,
                    instance_root,
                )
                fused_result = _FusedResult(roots, model_root, receipt)
            except BaseException as error:
                # A response is safe only after context destruction has
                # released a mapping whose explicit native close failed, or
                # stopped queued work that could not be synchronized before
                # unmapping. The persistent flag also covers an extreme
                # low-memory failure while CPython constructs the exception.
                if isinstance(
                    error, _NativeIpcCloseError
                ) or _native_runner_requires_context_cleanup(native_runner):
                    # Latch even a typed close error from a runner without a
                    # readable sentinel. close() must preserve this uncertainty
                    # if teardown or exception construction subsequently fails.
                    self._native_context_cleanup_required = True
                    try:
                        self.close()
                    except Exception as destroy_error:
                        raise IpcCleanupUncertainError(
                            "CUDA IPC unmapping failed and context destruction was not confirmed"
                        ) from destroy_error
                    raise IpcSessionAbortedError(
                        "CUDA IPC unmapping failed; the notary context was destroyed"
                    ) from error
                if isinstance(error, _NativeError):
                    raise NotaryError(str(error)) from error
                if isinstance(error, UnicodeDecodeError):
                    raise NotaryError(
                        f"kernel produced invalid UTF-8: {error}"
                    ) from error
                raise

            measurement = Measurement(
                digests=roots.hex(),
                model_root=model_root.hex(),
                vram_cid=ids.raw_cid(model_root),
                instance_cid=ids.raw_cid(instance_root),
                tensor_count=len(tensors),
                measured_at=measured_at,
                seconds=round(time.perf_counter() - started, 4),
            )
            return measurement, roots, fused_result.receipt

        fused_result = self._run_fallback_ipc(
            tensors, raw_handles, measured_at, model_bytes, instance_root
        )
        roots = fused_result.roots
        model_root = fused_result.model_root
        measurement = Measurement(
            digests=roots.hex(),
            model_root=model_root.hex(),
            vram_cid=ids.raw_cid(model_root),
            instance_cid=ids.raw_cid(instance_root),
            tensor_count=len(tensors),
            measured_at=measured_at,
            seconds=round(time.perf_counter() - started, 4),
        )
        return measurement, roots, fused_result.receipt

    def _measure(
        self, tensors: list[TensorRef], measured_at: str
    ) -> tuple[Measurement, bytes]:
        """Compatibility wrapper for an unsigned atomic measurement."""
        measurement, roots, _ = self._measure_request(tensors, measured_at)
        return measurement, roots

    def measure(self, tensors: list[TensorRef]) -> Measurement:
        """Return an unsigned diagnostic measurement.

        Unsigned results are intentionally not retained for a later signing
        request. Use :meth:`sign` when evidence, rather than diagnostics, is
        required.
        """
        with self._activate():
            measurement, _ = self._measure(tensors, _utc_timestamp())
            return measurement

    # ── sign ─────────────────────────────────────────────────────────────────

    def sign(self, tensors: list[TensorRef], model: str) -> dict:
        """Atomically measure, assemble and sign one immutable observation.

        The timestamp is captured here rather than accepted from a separate
        request. No process-global "last roots" exist, so another client can
        neither replace this observation nor revive it after a failed measure.
        """
        with self._activate():
            model_bytes = _validated_model(model)
            measured_at = _utc_timestamp()
            measurement, _, text = self._measure_request(
                tensors, measured_at, model_bytes
            )
            return self._signed_receipt(measurement, text, model)

    def _signed_receipt(self, measurement: Measurement, text: str | None, model: str) -> dict:
        """Apply identical signed-fact checks to one-shot and registered work."""
        if text is None:  # pragma: no cover - defensive invariant
            raise NotaryError("fused signing returned no receipt")
        try:
            # Parse the entire CUBIN response without duplicate keys before
            # validating either its signed document or statement graph.
            receipt = parse_kernel_receipt(text)
        except ValueError as e:
            raise NotaryError(f"kernel produced invalid JSON: {e}") from e
        self._validate_kernel_receipt(receipt, measurement, model)
        self._attach_identity_statement(receipt)
        self._attach_model_state_statement(receipt, measurement)
        receipt.update(measurement.as_dict())
        if self._host_key is not None:
            receipt["host_did"] = self._host_key.did
        receipt.update(
            {
                "gpu_did": self.info.gpu_did,
                "gpu_pubkey_uncompressed": self.info.gpu_pubkey_uncompressed,
                "kernel_cid": self.info.kernel_cid,
                "cubin_cid": self.info.cubin_cid,
                "device": f"cuda:{self.device_ordinal}",
            }
        )
        return receipt

    def attest(self, tensors: list[TensorRef], model: str) -> dict:
        """Descriptive alias for the atomic :meth:`sign` operation."""
        return self.sign(tensors, model)

    def _validate_kernel_receipt(
        self, receipt: dict, measurement: Measurement, expected_model: str
    ) -> None:
        """Refuse to publish signed claims that differ from trusted host facts.

        In particular, a substituted CUBIN cannot ignore its uploaded digest
        and claim an allowlisted one: the host computed the real CUBIN digest
        before loading it and checks the exact signed document here.
        """
        try:
            document_bytes = bytes.fromhex(receipt["measurementDocument"])
            document = parse_measurement_document(document_bytes)
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as e:
            raise NotaryError(
                f"kernel returned a malformed measurement document: {e}"
            ) from e
        expected = {
            # These protocol fields prevent a valid CUBIN from emitting a
            # stronger semantic assertion than the host actually established.
            "claim": MEASUREMENT_CLAIM,
            "hashScheme": MEASUREMENT_HASH_SCHEME,
            "operation": MEASUREMENT_OPERATION,
            "modelHash": measurement.model_root,
            "modelCID": f"urn:cid:{measurement.vram_cid}",
            "tensorCount": measurement.tensor_count,
            "measuredAt": measurement.measured_at,
            # The kernel cannot substitute even another valid model name: the
            # caller asked to attest this exact host-validated string.
            "model": expected_model,
            "device": f"cuda:{self.device_ordinal}",
            "gpuDID": self.info.gpu_did,
            "kernelCID": f"urn:cid:{self.info.kernel_cid}",
            "cubinCID": f"urn:cid:{self.info.cubin_cid}",
        }
        if frozenset(expected) != MEASUREMENT_DOCUMENT_FIELDS:
            raise RuntimeError("host validation does not cover the signed schema")
        for field_name, value in expected.items():
            if document.get(field_name) != value:
                raise NotaryError(
                    f"kernel's signed {field_name} contradicts the host measurement"
                )
        if receipt.get("modelRoot") != measurement.model_root:
            raise NotaryError(
                "kernel response modelRoot contradicts the host measurement"
            )
        try:
            measurement_signature = bytes.fromhex(receipt["measurementSignature"])
        except (KeyError, TypeError, ValueError) as error:
            raise NotaryError(
                "kernel returned a malformed measurement signature"
            ) from error
        if len(measurement_signature) != 64:
            raise NotaryError("kernel returned a malformed measurement signature")
        try:
            validate_statement_graph_structure(
                receipt.get("manifest"),
                document,
                expected_instance_urn=(
                    f"urn:cid:{measurement.instance_cid}"
                    if measurement.instance_cid
                    else None
                ),
            )
        except ValueError as error:
            raise NotaryError(f"kernel returned invalid statements: {error}") from error

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        native_runner = getattr(self, "_native_runner", None)
        if _native_runner_requires_context_cleanup(native_runner):
            # Snapshot before any interruptible cleanup, especially before
            # runner.close() / _native_runner=None removes the native owner.
            # Do not reset an existing latch when the runner is already gone.
            self._native_context_cleanup_required = True
        ctx = getattr(self, "ctx", None)
        cu = getattr(self, "cu", None)
        pushed = False
        if (
            ctx is not None
            and cu is not None
            and _context_id(cu.ctx_get_current()) != _context_id(ctx)
        ):
            # DeviceBuffer.free and cuCtxDestroy must both execute with
            # this instance's context current, even if another Notary was
            # the last object used on this thread. Pair this push with a
            # pop before destroying the now-detached context.
            cu.ctx_push_current(ctx)
            pushed = True
        try:
            abandon_allocations = bool(
                getattr(self, "_abandon_context_allocations", False)
                or getattr(self, "_native_context_cleanup_required", False)
            )
            if native_runner is not None:
                native_runner.close()
            self._native_runner = None
            buffers = (
                getattr(self, "_ctx_dev", None),
                getattr(self, "_cubin_digest_dev", None),
                getattr(self, "_kernel_digest_dev", None),
                *getattr(self, "_fallback_device_buffers", {}).values(),
                *getattr(self, "_fallback_host_buffers", {}).values(),
            )
            try:
                for b in buffers:
                    if b is not None and not abandon_allocations:
                        b.close()
            finally:
                # A pinned-buffer free (or an interrupt during any free) must
                # not skip stream/module/context teardown. Unvisited buffers
                # now belong solely to CUDA: detach their wrappers BEFORE
                # destruction so GC cannot double-free context-owned memory.
                # cuCtxDestroy reclaims cuMemAllocHost as well as device memory;
                # if it fails these allocations remain quarantined, not freed.
                for b in buffers:
                    if b is not None and hasattr(b, "ptr"):
                        b.ptr = 0
                self._ctx_dev = None
                self._cubin_digest_dev = None
                self._kernel_digest_dev = None
                self._fallback_device_buffers = {}
                self._fallback_host_buffers = {}
                self._abandon_context_allocations = False
                if ctx is not None and cu is not None:
                    try:
                        try:
                            stream = getattr(self, "_stream", None)
                            if stream is not None and not abandon_allocations:
                                # Destroy is asynchronous, never a replacement
                                # for the request's drain or proof for IPC ACKs.
                                cu.stream_destroy(stream)
                        finally:
                            module = getattr(self, "module", None)
                            if module is not None and not abandon_allocations:
                                cu.module_unload(module)
                    finally:
                        # Even a failed free/unload must not skip destruction
                        # or leave the caller's stack with our temporary push.
                        if pushed:
                            popped = cu.ctx_pop_current()
                            pushed = False
                            _require_popped_context(popped, ctx, "notary close")
                        # This also erases the device-global private key. If
                        # ctx was already current, destruction itself pops it.
                        cu.ctx_destroy(ctx)
                        # A free/unload error may propagate AFTER destruction.
                        # Track actual success independently from exceptions or
                        # _closed, which is set even on failed destruction.
                        self._context_destroyed = True
                        # Only confirmed destruction ends native quarantine;
                        # a masked exception must not permit unsafe HTTP ACKs.
                        self._native_context_cleanup_required = False
                        self._registration_cleanup_required = False
        finally:
            self.ctx = None
            self.module = None
            self._stream = None
            self._fn = {}
            self._closed = True
            if pushed:
                # Preserve the caller's stack even if buffer cleanup failed
                # before normal detachment. Do not use cuCtxSetCurrent here:
                # replacing the top entry is not equivalent to popping it.
                popped = cu.ctx_pop_current()
                _require_popped_context(popped, ctx, "failed notary close")


_STATUS = {
    -2: "key not ready — keygen did not run",
    -3: "malformed timestamp or model name",
    -4: "output buffer too small",
    -5: "invalid fused measurement plan",
    -6: "no matching measured root is pending",
    -7: "deterministic P-256 signing failed",
    -8: "chain slot count out of range",
    -9: "chain slots already configured",
    -10: "no free chain slot for this instance",
    -11: "chain slots not configured",
}

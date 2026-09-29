# SPDX-License-Identifier: Apache-2.0
"""Per-GPU notaries and authenticated aggregation without peer-access assumptions.

Every span is hashed on its owning GPU. The host only interleaves authenticated
digest lists; it does not claim that a single GPU measured the complete model.
"""

from __future__ import annotations

from ._build_config import ASSERTIONS_ENABLED

import json
import re
import secrets
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from threading import Event, Thread
from typing import Any

from . import ids
from ._cuda import Cuda
from ._hosthash import blake3_digest
from .notary import (
    _CHUNK,
    _TILE_CHUNKS,
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_MAX_REQUEST_TENSORS,
    DEFAULT_MAX_REQUEST_TILES,
    IpcCleanupUncertainError,
    IpcSessionAbortedError,
    Measurement,
    Notary,
    NotaryError,
    TensorRef,
    _validated_model,
    _validated_request_limit,
)

RECEIPT_TYPE = "cuattest.multi-gpu.v1"
MAX_AGGREGATE_TENSORS = 16384
MAX_AGGREGATE_DEVICES = 256
_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")


def _shard_model(manifest: dict, index: int) -> str:
    # The existing kernel authenticates a 1..64-character model field. A
    # domain-separated 256-bit commitment fits that contract without changing
    # its signed schema or accepting host-provided measurement digests.
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return blake3_digest(
        RECEIPT_TYPE.encode() + b"\0" + encoded + index.to_bytes(4, "little")
    ).hex()


def _combine(manifest: dict, digests: list[bytes]) -> tuple[bytes, bytes]:
    count = manifest["tensor_count"]
    combined = bytearray(count * 32)
    for shard, data in zip(manifest["shards"], digests):
        if len(data) != len(shard["positions"]) * 32:
            raise ValueError("shard digest count does not match its positions")
        for local, position in enumerate(shard["positions"]):
            combined[position * 32 : (position + 1) * 32] = data[
                local * 32 : (local + 1) * 32
            ]
    return bytes(combined), blake3_digest(count.to_bytes(4, "little") + combined)


@dataclass(frozen=True)
class MultiGpuInfo:
    devices: list[dict]

    def as_dict(self) -> dict:
        return {"multi_gpu": True, "devices": self.devices}


@dataclass
class _ShardCall:
    """Keep outcomes independently of interruptible queue-submission bookkeeping."""

    uid: str
    function: Callable
    args: tuple
    result: Any = None
    error: BaseException | None = None
    done: Event = field(default_factory=Event)

    def run(self) -> None:
        try:
            self.result = self.function(*self.args)
        except BaseException as error:  # noqa: BLE001 - rethrow only after every shard drains
            self.error = error
        finally:
            self.done.set()


class _GpuWorker:
    """One FIFO worker per GPU, with an explicit proof that its CUDA work ended."""

    def __init__(self) -> None:
        self._queue: SimpleQueue = SimpleQueue()
        self.stopped = Event()
        # The thread must not keep this owner alive. If construction returns
        # just before the pool's dictionary insertion fails, finalization still
        # wakes the otherwise-unowned idle worker; no IPC was submitted yet.
        weakref.finalize(self, self._queue.put, None)
        self._thread = Thread(
            target=self._run, args=(self._queue, self.stopped),
            name="cuattest-gpu", daemon=True,
        )
        try:
            # No GPU task is accepted until startup returns. If start() is
            # interrupted after creating the thread, the stop token below lets
            # that otherwise-unowned thread exit without ever touching CUDA.
            self._thread.start()
        except BaseException:
            self._queue.put(None)
            raise

    @staticmethod
    def _run(queue: SimpleQueue, stopped: Event) -> None:
        try:
            while (call := queue.get()) is not None:
                call.run()
                del call
        finally:
            # Do not infer this from Thread.join()/is_alive(): interruption of
            # join can change Python's thread bookkeeping before the worker has
            # actually exited. Only the worker can confirm this ownership end.
            stopped.set()

    def submit(self, call: _ShardCall) -> None:
        self._queue.put(call)

    def close(self) -> None:
        # FIFO order drains ALL accepted calls, including one whose queue.put
        # returned just before the submitting thread was interrupted.
        self._queue.put(None)
        self.stopped.wait()
        self._thread.join()


class MultiGpuNotary:
    """Serial requests, with GPU shards executing concurrently in independent sessions.

    ``devices=None`` selects all GPUs visible to this process. Producer UUIDs
    route requests even when its visible ordinals differ from ours. As with
    Notary, callers must serialize operations and quiesce all producer writes.
    """

    def __init__(
        self,
        devices: list[int] | None = None,
        artifact_dir: str | Path | None = None,
        *,
        max_request_tensors: int = DEFAULT_MAX_REQUEST_TENSORS,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_request_tiles: int = DEFAULT_MAX_REQUEST_TILES,
    ) -> None:
        self.notaries: dict[str, Notary] = {}
        self._workers: dict[str, _GpuWorker] = {}
        # Covers the entire dispatch/join interval, including an accepted job
        # whose queue submission never reached Python bookkeeping. Never ACK in that gap.
        self._shards_inflight = False
        self._closed = False
        self.max_request_tensors = _validated_request_limit(
            max_request_tensors, "max_request_tensors", MAX_AGGREGATE_TENSORS
        )
        self.max_request_bytes = _validated_request_limit(
            max_request_bytes, "max_request_bytes", (1 << 64) - 1
        )
        self.max_request_tiles = _validated_request_limit(
            max_request_tiles, "max_request_tiles", (1 << 64) - 1
        )
        if devices is None:
            cuda = Cuda()
            cuda.init()
            devices = list(range(cuda.device_count()))
        if (
            not devices
            or len(devices) > MAX_AGGREGATE_DEVICES
            or any(type(device) is not int or device < 0 for device in devices)
            or len(set(devices)) != len(devices)
        ):
            raise NotaryError("devices must name distinct visible CUDA ordinals")
        pending = None
        try:
            for device in devices:
                pending = Notary(
                    device=device,
                    artifact_dir=Path(artifact_dir) / f"cuda-{device}"
                    if artifact_dir
                    else None,
                    max_request_tensors=self.max_request_tensors,
                    max_request_bytes=self.max_request_bytes,
                    max_request_tiles=self.max_request_tiles,
                )
                if pending.info.device_uuid in self.notaries:
                    raise NotaryError("selected CUDA devices have duplicate UUIDs")
                self.notaries[pending.info.device_uuid] = pending
                pending = None
            self.info = MultiGpuInfo([n.info.as_dict() for n in self.notaries.values()])
        except BaseException:
            try:
                if pending is not None:
                    pending.close()
            finally:
                self.close()
            raise

    @property
    def ipc_cleanup_required(self) -> bool:
        # Do not query native flags while their owner threads might still be
        # changing them with the GIL released. The batch sentinel is sufficient
        # until all workers have crossed the completion barrier.
        return self._shards_inflight or any(
            n.ipc_cleanup_required for n in self.notaries.values()
        )

    def _ensure_workers(self, calls: list[_ShardCall]) -> None:
        for call in calls:
            if call.uid in self._workers:
                continue
            self._workers[call.uid] = _GpuWorker()

    def _drain_workers(self) -> None:
        first_error = None
        for worker in self._workers.values():
            try:
                worker.close()
            except BaseException as error:  # noqa: BLE001 - try every worker before failing closed
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        self._workers.clear()
        self._shards_inflight = False

    def _run_shards(self, calls: list[_ShardCall]) -> list[Any]:
        if ASSERTIONS_ENABLED:
            assert len({call.uid for call in calls}) == len(calls)
        self._ensure_workers(calls)
        self._shards_inflight = True
        dispatch_error = None
        try:
            for call in calls:
                self._workers[call.uid].submit(call)
            for call in calls:
                while not call.done.wait(0.1):
                    if self._workers[call.uid].stopped.is_set():
                        raise RuntimeError("GPU worker exited without completing its shard")
        except BaseException as error:  # noqa: BLE001 - interrupts must also drain foreign mappings
            dispatch_error = error
            try:
                self._drain_workers()
            except BaseException as drain_error:
                self._closed = True
                # Keep the batch sentinel and worker references: close() can
                # retry the join, but must not destroy contexts under live work.
                raise IpcCleanupUncertainError(
                    "GPU shard workers could not be joined; IPC completion is unconfirmed"
                ) from drain_error
        self._shards_inflight = False
        errors = [call.error for call in calls if call.error is not None]
        if dispatch_error is not None:
            errors.insert(0, dispatch_error)
        if self.ipc_cleanup_required:
            self._closed = True
            # A fast request-validation error must not hide another GPU's
            # uncertain cleanup and turn the whole batch into a safe HTTP 400.
            raise IpcCleanupUncertainError(
                "GPU shard cleanup is uncertain; producer storage must remain quarantined"
            ) from (errors[0] if errors else None)
        for error in errors:
            if isinstance(error, IpcCleanupUncertainError):
                self._closed = True
                raise error
        for error in errors:
            if isinstance(error, IpcSessionAbortedError):
                self._closed = True
                raise error
        for error in errors:
            if not isinstance(error, Exception):
                raise error
        if errors:
            raise errors[0]
        # Completion order must never affect the authenticated shard layout.
        if ASSERTIONS_ENABLED:
            assert all(call.done.is_set() and call.error is None for call in calls)
        return [call.result for call in calls]

    def _partition(self, tensors: list[TensorRef]) -> list[tuple[str, list[int]]]:
        if self._closed:
            raise NotaryError("multi-GPU notary is closed")
        if not tensors or len(tensors) > self.max_request_tensors:
            raise NotaryError(
                f"request must contain 1..{self.max_request_tensors} tensors"
            )
        groups: dict[str, list[int]] = {}
        total_bytes = total_tiles = 0
        # Preflight the ENTIRE request before any GPU opens a handle. Per-shard
        # limits alone would allow a many-GPU request to bypass operator caps.
        for position, tensor in enumerate(tensors):
            if not isinstance(tensor, TensorRef):
                raise NotaryError("tensors must contain TensorRef objects")
            tensor.validate_metadata()
            if tensor.device_uuid is None:
                raise NotaryError(
                    "multi-GPU requests require device_uuid; export with share_tensors/share_model"
                )
            if tensor.device_uuid not in self.notaries:
                raise NotaryError(
                    "tensor GPU is not among this service's selected devices"
                )
            total_bytes += tensor.nbytes
            total_tiles += 1 + (tensor.nbytes - 1) // (_CHUNK * _TILE_CHUNKS)
            if (
                total_bytes > self.max_request_bytes
                or total_tiles > self.max_request_tiles
            ):
                raise NotaryError(
                    "request exceeds aggregate byte or scheduling-tile limit"
                )
            groups.setdefault(tensor.device_uuid, []).append(position)
        for tensor in tensors:
            tensor.raw_handle()
        if ASSERTIONS_ENABLED:
            # Name-order positions must form a disjoint, complete partition;
            # a device-major concatenation would attest a different model.
            assert sorted(i for positions in groups.values() for i in positions) == list(range(len(tensors)))
            assert all(positions == sorted(positions) for positions in groups.values())
        return sorted(groups.items())

    def measure(self, tensors: list[TensorRef]) -> Measurement:
        started = time.perf_counter()
        groups = self._partition(tensors)
        measurements = self._run_shards([
            _ShardCall(uid, self.notaries[uid].measure, ([tensors[i] for i in positions],))
            for uid, positions in groups
        ])
        layout = {
            "tensor_count": len(tensors),
            "shards": [{"positions": p} for _, p in groups],
        }
        digests, root = _combine(
            layout, [bytes.fromhex(m.digests) for m in measurements]
        )
        return Measurement(
            digests.hex(),
            root.hex(),
            ids.raw_cid(root),
            len(tensors),
            measurements[0].measured_at,
            time.perf_counter() - started,
        )

    def sign(self, tensors: list[TensorRef], model: str, *, _registered=None) -> dict:
        _validated_model(model)
        groups = self._partition(tensors)
        manifest = {
            "model": model,
            "nonce": secrets.token_hex(16),
            "tensor_count": len(tensors),
            "shards": [
                {
                    "device_uuid": uid,
                    "device_ordinal": self.notaries[uid].device_ordinal,
                    "positions": positions,
                }
                for uid, positions in groups
            ],
        }
        started = time.perf_counter()
        receipts = self._run_shards([
            _ShardCall(
                uid, self.notaries[uid].sign if _registered is None else _registered[uid].sign,
                ([tensors[i] for i in positions], _shard_model(manifest, index))
                if _registered is None else (_shard_model(manifest, index),),
            )
            for index, (uid, positions) in enumerate(groups)
        ])
        digests, root = _combine(
            manifest, [bytes.fromhex(r["digests"]) for r in receipts]
        )
        return {
            "receipt_type": RECEIPT_TYPE,
            "manifest": manifest,
            "receipts": receipts,
            "model_root": root.hex(),
            "vram_cid": ids.raw_cid(root),
            "tensor_count": len(tensors),
            "digests": digests.hex(),
            "seconds": time.perf_counter() - started,
        }

    attest = sign

    def close(self) -> None:
        self._closed = True
        # A failed join must leave contexts and their sentinels intact. Only
        # after every worker exits may close() migrate a context to this thread
        # and free its resources without racing a still-running GPU request.
        self._drain_workers()
        first_error = None
        for notary in self.notaries.values():
            try:
                notary.close()
            except BaseException as error:  # noqa: BLE001 - close every GPU before reraising
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def verify_multi_gpu_evidence(receipt: dict, trusted_pubkeys: dict[str, str]):
    """Authenticate every shard, the exact layout, and the recomputed global fold.

    Keys are pinned by GPU UUID through a trusted channel. Neither keys copied
    out of a receipt nor its advertised aggregate CID establish trust.
    """
    from .expect import EvidenceError, VerifiedMeasurement, verify_evidence

    def require(condition, message):
        if not condition:
            raise EvidenceError(message)

    require(
        type(receipt) is dict and receipt.get("receipt_type") == RECEIPT_TYPE,
        "invalid multi-GPU receipt type",
    )
    require(
        set(receipt)
        == {
            "receipt_type",
            "manifest",
            "receipts",
            "model_root",
            "vram_cid",
            "tensor_count",
            "digests",
            "seconds",
        },
        "multi-GPU receipt does not match its exact schema",
    )
    require(
        type(trusted_pubkeys) is dict and bool(trusted_pubkeys),
        "multi-GPU verification requires trusted GPU public keys by UUID",
    )
    manifest = receipt["manifest"]
    require(
        type(manifest) is dict
        and set(manifest) == {"model", "nonce", "tensor_count", "shards"},
        "invalid multi-GPU manifest schema",
    )
    try:
        _validated_model(manifest["model"])
    except NotaryError as error:
        raise EvidenceError(str(error)) from error
    require(
        type(manifest["nonce"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", manifest["nonce"]) is not None,
        "invalid multi-GPU request nonce",
    )
    count = manifest["tensor_count"]
    require(
        type(count) is int and 0 < count <= MAX_AGGREGATE_TENSORS,
        "invalid aggregate tensor count",
    )
    shards, receipts = manifest["shards"], receipt["receipts"]
    require(
        type(shards) is list and 0 < len(shards) <= MAX_AGGREGATE_DEVICES,
        "invalid GPU shard list",
    )
    require(
        type(receipts) is list and len(receipts) == len(shards),
        "missing or extra shard receipts",
    )
    covered, uuids, ordinals = set(), set(), set()
    for shard in shards:
        require(
            type(shard) is dict
            and set(shard) == {"device_uuid", "device_ordinal", "positions"},
            "invalid shard layout schema",
        )
        uid, ordinal, positions = (
            shard["device_uuid"],
            shard["device_ordinal"],
            shard["positions"],
        )
        require(
            type(uid) is str and _UUID.fullmatch(uid) is not None and uid not in uuids,
            "invalid or duplicate shard GPU UUID",
        )
        require(
            type(ordinal) is int
            and 0 <= ordinal < (1 << 31)
            and ordinal not in ordinals,
            "invalid or duplicate shard device ordinal",
        )
        require(
            type(positions) is list and 0 < len(positions) <= count,
            "invalid shard positions",
        )
        require(
            all(type(p) is int and 0 <= p < count for p in positions),
            "shard position outside aggregate",
        )
        require(
            positions == sorted(set(positions)) and not covered.intersection(positions),
            "duplicate or unordered shard positions",
        )
        require(
            uid in trusted_pubkeys and type(trusted_pubkeys[uid]) is str,
            "missing trusted key for shard GPU",
        )
        covered.update(positions)
        uuids.add(uid)
        ordinals.add(ordinal)
    require(len(covered) == count, "shard layout does not cover every tensor")
    # Only after all bounded structural checks do we allocate digest buffers
    # and verify signatures. Each signed model field commits to this SAME
    # manifest plus its shard index, preventing reordering and request mixing.
    digest_lists = []
    for index, (shard, evidence) in enumerate(zip(shards, receipts)):
        require(
            type(evidence) is dict and "receipt_type" not in evidence,
            "nested multi-GPU receipts are not allowed",
        )
        verified = verify_evidence(
            evidence, trusted_pubkey=trusted_pubkeys[shard["device_uuid"]]
        )
        require(
            verified.document["model"] == _shard_model(manifest, index),
            "shard signature does not bind this multi-GPU manifest",
        )
        require(
            verified.document["device"] == f"cuda:{shard['device_ordinal']}",
            "shard signed device contradicts the manifest",
        )
        require(
            verified.tensor_count == len(shard["positions"]),
            "shard count contradicts its layout",
        )
        require(
            verified.digests is not None,
            "multi-GPU evidence requires every shard's digests",
        )
        digest_lists.append(bytes.fromhex(verified.digests))
    digests, root = _combine(manifest, digest_lists)
    cid = ids.raw_cid(root)
    require(
        type(receipt["tensor_count"]) is int and receipt["tensor_count"] == count,
        "aggregate count contradicts its manifest",
    )
    require(
        receipt["model_root"] == root.hex()
        and receipt["vram_cid"] == cid
        and receipt["digests"] == digests.hex(),
        "aggregate values contradict authenticated shard digests",
    )
    # document is explicitly the authenticated layout, not a fabricated
    # single-GPU measurement document or a new host signature.
    return VerifiedMeasurement(root.hex(), cid, count, digests.hex(), manifest)

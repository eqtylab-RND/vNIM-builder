# SPDX-License-Identifier: Apache-2.0
"""GPU routing, authenticated layout, and whole-request cleanup regressions."""

import copy
from contextlib import contextmanager
from threading import Barrier, Event, Thread, get_ident
from types import SimpleNamespace

import pytest

from cuattest import multigpu
from cuattest._hosthash import blake3_digest
from cuattest.notary import GpuInfo, Measurement, NotaryError, TensorRef


def uid(ordinal):
    return f"GPU-00000000-0000-0000-0000-{ordinal:012d}"


def refs():
    # Producer sees local devices 9/8, while the service sees 0/1. Global
    # positions interleave devices and must not become device-major order.
    return [
        TensorRef(
            bytes([i + 1]).hex() * 64, 8, device=9 - i % 2, device_uuid=uid(i % 2)
        )
        for i in range(4)
    ]


def tensor_digest(tensor):
    return blake3_digest(bytes.fromhex(tensor.handle))


@pytest.fixture
def pool(monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    from cuattest import ids

    class FakeNotary:
        def __init__(self, device, **kwargs):
            self.device_ordinal = device
            self.private = ec.generate_private_key(ec.SECP256R1())
            public = self.private.public_key().public_bytes(
                serialization.Encoding.X962,
                serialization.PublicFormat.UncompressedPoint,
            )
            self.info = GpuInfo(
                ids.did_key_p256(public[1:33], public[33:]),
                public.hex(),
                "kernel",
                "cubin",
                "test",
                "sm_90",
                "fake",
                device,
                0,
                device_uuid=uid(device),
            )
            self.calls = []
            self.ipc_cleanup_required = False
            self.closed = False

        def sign(self, tensors, model):
            from test_expect import signed_receipt

            self.calls.append(tensors)
            result = signed_receipt(
                [tensor_digest(t).hex() for t in tensors],
                private=self.private,
                extra_fields={"model": model, "device": f"cuda:{self.device_ordinal}"},
            )
            result["seconds"] = 0
            return result

        def measure(self, tensors):
            receipt = self.sign(tensors, "diagnostic")
            return Measurement(
                receipt["digests"],
                receipt["model_root"],
                receipt["vram_cid"],
                receipt["tensor_count"],
                receipt["measured_at"],
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(multigpu, "Notary", FakeNotary)
    result = multigpu.MultiGpuNotary([0, 1])
    yield result
    result.close()


def pins(pool):
    return {
        entry["device_uuid"]: entry["gpu_pubkey_uncompressed"]
        for entry in pool.info.devices
    }


def test_uuid_routing_preserves_original_tensor_order(pool):
    from cuattest.expect import Expectation, compare, verify_evidence

    tensors = refs()
    receipt = pool.sign(tensors, "test/model")
    verified = verify_evidence(receipt, trusted_pubkeys=pins(pool))
    assert verified.document["model"] == "test/model"
    expected_digests = b"".join(tensor_digest(t) for t in tensors)
    expected_root = blake3_digest((4).to_bytes(4, "little") + expected_digests)
    assert verified.digests == expected_digests.hex()
    assert verified.model_root == expected_root.hex()
    assert pool.notaries[uid(0)].calls == [[tensors[0], tensors[2]]]
    assert pool.notaries[uid(1)].calls == [[tensors[1], tensors[3]]]
    expected = Expectation(
        verified.model_root,
        verified.vram_cid,
        4,
        verified.digests,
        ["a", "b", "c", "d"],
        32,
    )
    assert compare(expected, receipt, trusted_pubkeys=pins(pool)).matches
    assert pool.measure(tensors).model_root == expected_root.hex()


@pytest.mark.parametrize(
    "failure",
    ["uuid", "missing_uuid", "handle", "byte_limit", "tile_limit", "count_limit"],
)
def test_whole_request_preflight_happens_before_any_gpu_import(pool, failure):
    tensors = refs()
    if failure == "uuid":
        tensors[-1] = TensorRef("11" * 64, 8, device_uuid=uid(9))
    elif failure == "missing_uuid":
        tensors[-1] = TensorRef("11" * 64, 8)
    elif failure == "handle":
        tensors[-1] = TensorRef("zz" * 64, 8, device_uuid=uid(1))
    else:
        setattr(
            pool,
            {
                "byte_limit": "max_request_bytes",
                "tile_limit": "max_request_tiles",
                "count_limit": "max_request_tensors",
            }[failure],
            1,
        )
    with pytest.raises(NotaryError):
        pool.sign(tensors, "test/model")
    assert all(not n.calls for n in pool.notaries.values())


@pytest.mark.parametrize(
    "mutation",
    [
        "model",
        "nonce",
        "positions",
        "ordinal",
        "uuid",
        "missing",
        "duplicate",
        "bool_position",
        "huge_count",
        "root",
        "digests",
        "untrusted",
        "proof",
        "nested",
        "extra_field",
    ],
)
def test_aggregate_verification_rejects_tampering(pool, mutation):
    from cuattest.expect import EvidenceError, verify_evidence

    receipt = pool.sign(refs(), "test/model")
    manifest = receipt["manifest"]
    shard = manifest["shards"][0]
    keys = pins(pool)
    if mutation == "model":
        manifest["model"] = "other/model"
    elif mutation == "nonce":
        manifest["nonce"] = "00" * 16
    elif mutation == "positions":
        manifest["shards"][0]["positions"], manifest["shards"][1]["positions"] = (
            [1, 3],
            [0, 2],
        )
    elif mutation == "ordinal":
        shard["device_ordinal"] = 3
    elif mutation == "uuid":
        shard["device_uuid"] = uid(9)
        keys[uid(9)] = keys[uid(0)]
    elif mutation == "missing":
        shard["positions"].pop()
    elif mutation == "duplicate":
        shard["positions"] = [0, 0]
    elif mutation == "bool_position":
        shard["positions"][0] = False
    elif mutation == "huge_count":
        manifest["tensor_count"] = 2**31
    elif mutation == "root":
        receipt["model_root"] = "00" * 32
    elif mutation == "digests":
        receipt["digests"] = "00" * 128
    elif mutation == "untrusted":
        keys[uid(0)] = keys[uid(1)]
    elif mutation == "proof":
        receipt["receipts"][0]["manifest"]["statements"] = {}
    elif mutation == "nested":
        receipt["receipts"][0] = copy.deepcopy(receipt)
    elif mutation == "extra_field":
        manifest["claim"] = "inference used these weights"
    with pytest.raises(EvidenceError):
        verify_evidence(receipt, trusted_pubkeys=keys)


def test_shards_from_different_requests_cannot_be_mixed(pool):
    from cuattest.expect import EvidenceError, verify_evidence

    first, second = pool.sign(refs(), "test/model"), pool.sign(refs(), "test/model")
    first["receipts"][1] = second["receipts"][1]
    with pytest.raises(EvidenceError, match="manifest"):
        verify_evidence(first, trusted_pubkeys=pins(pool))


def test_aggregate_requires_pinned_keys(pool):
    from cuattest.expect import EvidenceError, verify_evidence

    with pytest.raises(EvidenceError, match="trusted GPU public keys"):
        verify_evidence(pool.sign(refs(), "test/model"))


def test_uncertain_shard_cleanup_withholds_http_acknowledgement(pool, monkeypatch):
    from cuattest.notary import IpcCleanupUncertainError
    from cuattest.server import make_handler

    failing = pool.notaries[uid(1)]

    def fail(*args):
        failing.ipc_cleanup_required = True
        raise IpcCleanupUncertainError("GPU cleanup is uncertain")

    monkeypatch.setattr(failing, "sign", fail)
    handler = object.__new__(make_handler(pool))
    handler.path = "/v1/sign"
    handler._read_json = lambda: {
        "model": "test/model",
        "tensors": [vars(t) for t in refs()],
    }
    handler._send = lambda *args, **kwargs: pytest.fail(
        "acknowledged unsafe IPC cleanup"
    )
    handler.server = SimpleNamespace(cuattest_fatal_error=None)
    handler.do_POST()
    assert handler.close_connection
    assert pool.ipc_cleanup_required
    assert isinstance(handler.server.cuattest_fatal_error, IpcCleanupUncertainError)


def test_close_attempts_every_gpu_after_one_close_fails(pool, monkeypatch):
    def fail():
        raise RuntimeError("destroy failed")

    with monkeypatch.context() as patch:
        patch.setattr(pool.notaries[uid(0)], "close", fail)
        with pytest.raises(RuntimeError, match="destroy failed"):
            pool.close()
    assert pool.notaries[uid(1)].closed


@contextmanager
def background_call(function, release):
    state = SimpleNamespace(done=Event(), error=None, result=None)

    def run():
        try:
            state.result = function()
        except BaseException as error:  # noqa: BLE001 - surface worker-thread test failures
            state.error = error
        finally:
            state.done.set()

    thread = Thread(target=run)
    thread.start()
    try:
        yield state
    finally:
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive(), "test request did not drain"


def parallel_handler(pool):
    from cuattest.server import make_handler

    handler = object.__new__(make_handler(pool))
    handler.path = "/v1/sign"
    handler._read_json = lambda: {
        "model": "test/model", "tensors": [vars(t) for t in refs()]
    }
    handler.server = SimpleNamespace(cuattest_fatal_error=None)
    handler.close_connection = False
    sent = []
    handler._send = lambda code, payload, raw=False: sent.append((code, payload))
    return handler, sent


@pytest.mark.parametrize("operation", ["measure", "sign"])
def test_shards_overlap_on_stable_workers_without_reordering_results(pool, monkeypatch, operation):
    barrier = Barrier(2, timeout=5)
    second_finished = Event()
    threads = {uid(0): [], uid(1): []}
    finished = []
    for key, notary in pool.notaries.items():
        original = getattr(notary, operation)

        def run(*args, key=key, original=original):
            threads[key].append(get_ident())
            # A serial implementation cannot cross this barrier. Finish in
            # reverse UUID order to check that scheduling cannot reorder CIDs.
            barrier.wait()
            if key == uid(0):
                assert second_finished.wait(5)
            result = original(*args)
            finished.append(key)
            if key == uid(1):
                second_finished.set()
            return result

        monkeypatch.setattr(notary, operation, run)

    roots = []
    for _ in range(2):
        second_finished.clear()
        if operation == "sign":
            from cuattest.expect import verify_evidence

            receipt = pool.sign(refs(), "test/model")
            roots.append(verify_evidence(receipt, trusted_pubkeys=pins(pool)).model_root)
        else:
            roots.append(pool.measure(refs()).model_root)
    assert roots[0] == roots[1]
    assert finished == [uid(1), uid(0)] * 2
    assert len(set(threads[uid(0)])) == len(set(threads[uid(1)])) == 1
    assert threads[uid(0)][0] != threads[uid(1)][0]
    assert not pool.ipc_cleanup_required


def test_fast_validation_error_waits_for_other_gpu_before_http_headers(pool, monkeypatch):
    started, failed, release, retired = Event(), Event(), Event(), Event()
    original = pool.notaries[uid(1)].sign

    def slow(*args):
        started.set()
        assert release.wait(5)
        result = original(*args)
        retired.set()
        return result

    def fail(*args):
        assert started.wait(5)
        failed.set()
        raise NotaryError("rejected on first GPU")

    monkeypatch.setattr(pool.notaries[uid(0)], "sign", fail)
    monkeypatch.setattr(pool.notaries[uid(1)], "sign", slow)
    handler, sent = parallel_handler(pool)
    with background_call(handler.do_POST, release) as state:
        assert failed.wait(5)
        assert not state.done.wait(0.05)
        assert sent == [] and pool.ipc_cleanup_required
    assert state.error is None
    assert retired.is_set()
    assert sent == [(400, {"error": "rejected on first GPU"})]
    assert not pool.ipc_cleanup_required


@pytest.mark.parametrize("failure", ["cleanup_flag", "cleanup_error", "aborted"])
def test_fatal_shard_error_takes_precedence_over_earlier_validation_error(
    pool, monkeypatch, failure
):
    from cuattest.notary import IpcCleanupUncertainError, IpcSessionAbortedError

    def invalid(*args):
        raise NotaryError("ordinary request rejection")

    def fatal(*args):
        if failure == "cleanup_flag":
            pool.notaries[uid(1)].ipc_cleanup_required = True
            raise MemoryError("could not allocate specialized cleanup exception")
        if failure == "cleanup_error":
            raise IpcCleanupUncertainError("mapping still uncertain")
        raise IpcSessionAbortedError("context destroyed")

    monkeypatch.setattr(pool.notaries[uid(0)], "sign", invalid)
    monkeypatch.setattr(pool.notaries[uid(1)], "sign", fatal)
    handler, sent = parallel_handler(pool)
    handler.do_POST()
    assert handler.close_connection and pool._closed
    if failure == "aborted":
        assert sent == [(500, {"error": "context destroyed"})]
    else:
        assert sent == []
        assert isinstance(handler.server.cuattest_fatal_error, IpcCleanupUncertainError)


def test_interrupt_after_submission_drains_accepted_job_and_allows_safe_retry(pool, monkeypatch):
    # Warm workers without fault injection, then simulate queue.put succeeding
    # just before the parent loses its normal return from submit().
    expected = pool.sign(refs(), "test/model")["model_root"]
    worker = pool._workers[uid(0)]
    submit = worker.submit
    original = pool.notaries[uid(0)].sign
    started, release = Event(), Event()

    def slow(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    def interrupted(call):
        submit(call)
        raise MemoryError("interrupted after accepted submission")

    with monkeypatch.context() as patch:
        patch.setattr(worker, "submit", interrupted)
        patch.setattr(pool.notaries[uid(0)], "sign", slow)
        handler, sent = parallel_handler(pool)
        with background_call(handler.do_POST, release) as state:
            assert started.wait(5)
            assert not state.done.wait(0.05)
            assert sent == [] and pool.ipc_cleanup_required
        assert state.error is None
    assert len(sent) == 1 and sent[0][0] == 500
    assert worker.stopped.is_set() and not pool._workers
    assert not pool.ipc_cleanup_required
    assert pool.sign(refs(), "test/model")["model_root"] == expected


def test_failed_worker_drain_keeps_quarantine_and_never_closes_live_contexts(pool, monkeypatch):
    from cuattest.notary import IpcCleanupUncertainError

    pool.sign(refs(), "test/model")
    worker = pool._workers[uid(0)]
    submit = worker.submit
    original = pool.notaries[uid(0)].sign
    started, release = Event(), Event()

    def slow(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    def interrupted(call):
        submit(call)
        raise MemoryError("submission interrupted")

    def fail_join():
        raise MemoryError("worker join interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(worker, "submit", interrupted)
        patch.setattr(worker, "close", fail_join)
        patch.setattr(pool.notaries[uid(0)], "sign", slow)
        handler, sent = parallel_handler(pool)
        with background_call(handler.do_POST, release) as state:
            assert started.wait(5) and state.done.wait(5)
            assert sent == [] and pool.ipc_cleanup_required
            assert isinstance(handler.server.cuattest_fatal_error, IpcCleanupUncertainError)
            with pytest.raises(MemoryError, match="join interrupted"):
                pool.close()
            assert all(not n.closed for n in pool.notaries.values())
            assert not worker.stopped.is_set()
        assert state.error is None
    pool.close()
    assert worker.stopped.is_set()
    assert all(n.closed for n in pool.notaries.values())
    assert not pool.ipc_cleanup_required


def test_worker_exit_signal_not_thread_join_bookkeeping_confirms_completion(monkeypatch):
    started, release = Event(), Event()
    worker = multigpu._GpuWorker()

    def slow():
        started.set()
        assert release.wait(5)

    call = multigpu._ShardCall("gpu", slow, ())
    worker.submit(call)
    assert started.wait(5)
    # On affected Python versions an interrupted join can mark a thread as
    # stopped too early. Model that misleading return without corrupting real
    # interpreter internals: the explicit worker-exit event must still block.
    monkeypatch.setattr(worker._thread, "join", lambda: None)
    with background_call(worker.close, release) as state:
        assert not state.done.wait(0.05)
        assert not worker.stopped.is_set()
    assert state.error is None
    assert call.done.is_set() and worker.stopped.is_set()


def test_interrupted_thread_start_cannot_accept_ipc_work(pool, monkeypatch):
    original = Thread.start
    started = []

    def interrupted(thread):
        original(thread)
        started.append(thread)
        raise MemoryError("interrupted after thread start")

    monkeypatch.setattr(multigpu.Thread, "start", interrupted)
    with pytest.raises(MemoryError, match="after thread start"):
        pool.sign(refs(), "test/model")
    for thread in started:
        thread.join(5)
        assert not thread.is_alive()
    assert all(not n.calls for n in pool.notaries.values())
    assert not pool.ipc_cleanup_required


def test_interrupted_worker_registration_does_not_leak_idle_thread(pool):
    class FailingRegistry(dict):
        def __setitem__(self, key, worker):
            self.thread = worker._thread
            raise MemoryError("could not register worker")

    pool._workers = registry = FailingRegistry()
    with pytest.raises(MemoryError, match="could not register"):
        pool.sign(refs(), "test/model")
    # The worker's target cannot hold its owner alive after the failed insert.
    # Dropping the exception traceback permits collection on non-refcounted
    # Python implementations too; finalization must wake the idle queue.
    import gc

    gc.collect()
    registry.thread.join(5)
    assert not registry.thread.is_alive()
    assert all(not n.calls for n in pool.notaries.values())
    assert not pool.ipc_cleanup_required


@pytest.mark.parametrize("failure", ["initialization", "duplicate_uuid"])
def test_partial_startup_closes_every_acquired_session(pool, monkeypatch, failure):
    from dataclasses import replace

    original = multigpu.Notary
    acquired = []

    def construct(device, **kwargs):
        if failure == "initialization" and device == 1:
            raise NotaryError("injected initialization failure")
        notary = original(device, **kwargs)
        acquired.append(notary)
        if failure == "duplicate_uuid":
            notary.info = replace(notary.info, device_uuid=uid(0))
        return notary

    monkeypatch.setattr(multigpu, "Notary", construct)
    with pytest.raises(NotaryError, match="initialization failure|duplicate UUIDs"):
        multigpu.MultiGpuNotary([0, 1])
    assert acquired and all(notary.closed for notary in acquired)


@pytest.mark.parametrize("selection", [[], [0, 0], [-1], [True]])
def test_invalid_device_selection_is_rejected_before_initialization(
    monkeypatch, selection
):
    monkeypatch.setattr(
        multigpu, "Notary", lambda **kwargs: pytest.fail("opened a GPU")
    )
    with pytest.raises(NotaryError, match="distinct visible"):
        multigpu.MultiGpuNotary(selection)

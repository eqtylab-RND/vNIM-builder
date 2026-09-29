# SPDX-License-Identifier: Apache-2.0
"""Real sharded PyTorch models over HTTP; CUATTEST_TEST_MULTIGPU=1 enables this."""

import gc
import os

import pytest
from _ipc_test_service import ipc_test_service

from cuattest import ids
from cuattest._hosthash import blake3_digest
from cuattest.client import NotaryClientError
from cuattest.expect import Expectation, compare, verify_evidence
from cuattest.ipc import _active_allocation_lease_count, share_model, share_tensors

pytestmark = pytest.mark.skipif(
    os.environ.get("CUATTEST_TEST_MULTIGPU") != "1",
    reason="requires opt-in and at least two CUDA GPUs",
)


@pytest.fixture(scope="module", params=["native", "fallback"])
def service(request):
    with ipc_test_service(request.param, minimum_devices=2) as setup:
        yield setup


@pytest.fixture(scope="module", autouse=True)
def release_unused_framework_cache():
    yield
    import torch

    # Retired producer tensors leave reusable allocator blocks on every GPU.
    # Return only unused cache to CUDA so memcheck's exit-time leak report
    # measures ownership, not PyTorch's next-allocation cache. Live quarantined
    # leases remain strongly referenced and must never be force-released here.
    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


def test_sharded_model_matches_global_canonical_fold(service):
    client, keys, torch = service
    model = torch.nn.Module()
    for device in range(torch.cuda.device_count()):
        # One noncontiguous view and one alias per device, interleaved by name.
        tensor = (
            (torch.arange(64, dtype=torch.uint8, device=f"cuda:{device}") + device)
            .reshape(8, 8)
            .t()
        )
        model.register_buffer(f"a_{device}", tensor)
        model.register_buffer(f"z_{device}", tensor)
    previous_device = torch.cuda.current_device()
    names, refs, keepalive = share_model(model)
    assert torch.cuda.current_device() == previous_device
    assert {r["device_uuid"] for r in refs} == set(keys)
    receipt = client.sign(refs, "test/sharded-model")
    assert _active_allocation_lease_count() == 0
    keepalive.release()
    del keepalive  # the list itself retains ordinary tensor references
    verified = verify_evidence(receipt, trusted_pubkeys=keys)
    state = model.state_dict()
    digests = b"".join(
        blake3_digest(
            bytes(state[name].cpu().contiguous().view(torch.uint8).reshape(-1).tolist())
        )
        for name in names
    )
    root = blake3_digest(len(names).to_bytes(4, "little") + digests)
    expected = Expectation(
        root.hex(), ids.raw_cid(root), len(names), digests.hex(), names, len(names) * 64
    )
    assert verified.model_root == root.hex()
    assert compare(expected, receipt, trusted_pubkeys=keys).matches
    assert len(receipt["receipts"]) == torch.cuda.device_count()
    assert client.info()["multi_gpu"] is True


def test_independent_requests_on_every_gpu_and_rejected_wrong_owner(service):
    client, keys, torch = service
    for device in range(torch.cuda.device_count()):
        tensor = torch.full((32,), device, dtype=torch.uint8, device=f"cuda:{device}")
        _, refs, keepalive = share_tensors({"weight": tensor})
        receipt = client.sign(refs, "test/independent")
        keepalive.release()
        assert _active_allocation_lease_count() == 0
        del keepalive
        assert len(receipt["receipts"]) == 1
        verified = verify_evidence(receipt, trusted_pubkeys=keys)
        assert verified.digests == blake3_digest(bytes([device]) * 32).hex()

    tensor = torch.zeros(32, dtype=torch.uint8, device="cuda:0")
    _, refs, keepalive = share_tensors({"weight": tensor})
    # A client-provided UUID routes but never authorizes access to a foreign
    # allocation. Exercise the actual driver owner check with a forged hint.
    refs[0]["device_uuid"] = next(uid for uid in keys if uid != refs[0]["device_uuid"])
    with pytest.raises(NotaryClientError) as rejected:
        client.sign(refs, "test/wrong-owner")
    assert rejected.value.ipc_completion_known
    assert _active_allocation_lease_count() == 0
    keepalive.release()
    del keepalive

    # Rejection must leave the service and all independent GPU sessions usable.
    _, refs, keepalive = share_tensors({"weight": tensor})
    verify_evidence(client.sign(refs, "test/recovered"), trusted_pubkeys=keys)
    keepalive.release()
    assert _active_allocation_lease_count() == 0
    del keepalive


def test_parallel_repeated_requests_hash_fresh_partial_tiles_on_every_gpu(service):
    client, keys, torch = service
    previous_device = torch.cuda.current_device()
    host = {
        f"layer.{layer}.gpu.{device}": torch.arange(size, dtype=torch.int64).to(torch.uint8) + device
        for device in range(torch.cuda.device_count())
        for layer, size in enumerate((1023, 1025, 128 * 1024 + 65))
    }
    device_tensors = {
        name: value.to(f"cuda:{name.rsplit('.', 1)[1]}")
        for name, value in host.items()
    }
    previous_root = None
    for _ in range(3):
        names, refs, keepalive = share_tensors(device_tensors)
        receipt = client.sign(refs, "test/parallel-partial-tiles")
        assert _active_allocation_lease_count() == 0
        keepalive.release()
        del keepalive
        digests = b"".join(blake3_digest(host[name].numpy().tobytes()) for name in names)
        root = blake3_digest(len(names).to_bytes(4, "little") + digests).hex()
        verified = verify_evidence(receipt, trusted_pubkeys=keys)
        assert verified.digests == digests.hex() and verified.model_root == root
        assert root != previous_root
        assert len(receipt["receipts"]) == torch.cuda.device_count()
        assert torch.cuda.current_device() == previous_device
        previous_root = root
        # Reuse the same allocations only AFTER the complete batch ACK. Stable
        # per-GPU workers must hash fresh bytes, not retain a previous request's
        # imported pointers, partial-leaf lengths, or mutable native workspace.
        for name in names:
            host[name].add_(1)
            device_tensors[name].add_(1)


def test_rejected_shard_drains_other_gpus_and_preserves_sessions(service):
    client, keys, torch = service
    tensors = {
        f"gpu.{device}": torch.full((128 * 1024 + 1,), device, dtype=torch.uint8,
                                     device=f"cuda:{device}")
        for device in range(torch.cuda.device_count())
    }
    _, refs, keepalive = share_tensors(tensors)
    # Unlike a preflight metadata rejection, this valid-size handle reaches
    # the real CUDA driver while all other GPU workers receive valid IPC spans.
    refs[0]["handle"] = "00" * 64
    with pytest.raises(NotaryClientError) as rejected:
        client.sign(refs, "test/parallel-rejected-import")
    assert rejected.value.ipc_completion_known
    assert _active_allocation_lease_count() == 0
    keepalive.release()
    del keepalive
    names, refs, keepalive = share_tensors(tensors)
    verified = verify_evidence(
        client.sign(refs, "test/parallel-after-rejection"), trusted_pubkeys=keys
    )
    assert verified.digests == b"".join(
        blake3_digest(tensors[name].cpu().numpy().tobytes()) for name in names
    ).hex()
    keepalive.release()
    del keepalive
    assert _active_allocation_lease_count() == 0

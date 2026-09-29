# SPDX-License-Identifier: Apache-2.0
"""Real PyTorch export regressions; enable with CUATTEST_TEST_GPU=1."""

import gc
import os
import struct
import subprocess
import sys
import textwrap
from contextlib import ExitStack

import pytest
from _ipc_test_service import ipc_test_service
from blake3 import blake3

from cuattest.expect import verify_evidence
from cuattest.ipc import _active_allocation_lease_count, share_model

pytestmark = pytest.mark.skipif(
    os.environ.get("CUATTEST_TEST_GPU") != "1",
    reason="set CUATTEST_TEST_GPU=1 to run real CUDA export regressions",
)


@pytest.fixture(scope="module", params=["native", "fallback"])
def export_service(request):
    with ipc_test_service(request.param) as setup:
        yield setup


@pytest.fixture(scope="module", autouse=True)
def release_unused_framework_cache():
    yield
    # These tests intentionally exercise PyTorch's caching allocator. Release
    # its now-unused blocks before process exit so CUDA leak checking measures
    # allocation ownership, not memory retained for a future tensor operation.
    import torch

    for device in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "offset"])
@pytest.mark.parametrize("conjugate,negative", [(True, False), (False, True), (True, True)])
def test_lazy_views_match_materialized_values_over_ipc(
    export_service, layout, conjugate, negative
):
    client, keys, torch = export_service
    model = torch.nn.Module()
    expected = {}
    # Compute reference values using Python complex arithmetic and struct,
    # not the GPU view's storage or the exporter's resolution methods.
    values = [complex(i + 1, 101 + i) for i in range(24)]
    cpu = torch.tensor(values, dtype=torch.complex64).reshape(4, 6)
    if layout == "transposed":
        values = [values[row * 6 + col] for col in range(6) for row in range(4)]
    elif layout == "offset":
        values = values[6:]
    transformed = [value.conjugate() if conjugate else value for value in values]
    if negative:
        transformed = [-value for value in transformed]

    def digest(items):
        raw = b"".join(struct.pack("<ff", value.real, value.imag) for value in items)
        return blake3(raw).digest()

    with ExitStack() as streams:
        for device in range(torch.cuda.device_count()):
            # Every device has outstanding work on a non-default stream when
            # share_model resolves the flags. Synchronizing only cuda:0 or
            # only before those copies is insufficient for the consumer.
            stream = torch.cuda.Stream(device=device)
            streams.enter_context(torch.cuda.stream(stream))
            base = cpu.to(f"cuda:{device}")
            if layout == "transposed":
                base = base.t()
            elif layout == "offset":
                base = base[1:]
            lazy = base.conj() if conjugate else base
            if negative:
                # PyTorch exposes no public lazy-negation constructor; this
                # test-only call creates the flag that resolve_neg handles.
                lazy = torch._neg_view(lazy)
            assert lazy.is_conj() == conjugate and lazy.is_neg() == negative
            assert lazy.is_contiguous() == (layout != "transposed")
            materialized = torch.tensor(
                transformed, dtype=torch.complex64, device=f"cuda:{device}"
            ).reshape(base.shape)
            for suffix, tensor, logical in (
                ("base", base, values), ("lazy", lazy, transformed),
                ("materialized", materialized, transformed),
            ):
                name = f"d{device}_{suffix}"
                model.register_buffer(name, tensor)
                expected[name] = digest(logical)
        previous_device = torch.cuda.current_device()
        names, refs, keepalive = share_model(model)
        assert torch.cuda.current_device() == previous_device

    try:
        assert all(
            tensor.is_contiguous() and not tensor.is_conj() and not tensor.is_neg()
            for tensor in keepalive
        )
        receipt = client.sign(refs, "test/lazy-views")
        verified = verify_evidence(receipt, trusted_pubkeys=keys)
        digests = b"".join(expected[name] for name in names)
        assert verified.digests == digests.hex()
        assert verified.model_root == blake3(struct.pack("<I", len(names)) + digests).hexdigest()
        for device in range(torch.cuda.device_count()):
            assert expected[f"d{device}_lazy"] == expected[f"d{device}_materialized"]
            assert expected[f"d{device}_lazy"] != expected[f"d{device}_base"]
        assert _active_allocation_lease_count() == 0
    finally:
        keepalive.release()


@pytest.mark.parametrize("variable", ["PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"])
@pytest.mark.parametrize("expandable", [True, False])
def test_expandable_allocation_diagnosis_in_fresh_producer(variable, expandable):
    # Allocator configuration is read during CUDA initialization. Each case
    # needs a fresh process, or a test may accidentally exercise cached legacy
    # storage and pass even though expandable-segment export is still broken.
    script = textwrap.dedent(
        f"""
        import gc
        import os
        import torch
        from cuattest.ipc import _active_allocation_lease_count, share_model

        assert torch.cuda.is_available()
        for device in range(torch.cuda.device_count()):
            tensor = torch.ones(1024 * 1024, dtype=torch.float32, device=f"cuda:{{device}}")
            assert tensor.is_contiguous() and tensor.nbytes == 4 * 1024 * 1024
            model = torch.nn.Module()
            model.register_buffer("weight", tensor)
            # Deliberately contradict the current environment after allocation.
            # Compatibility belongs to the storage, not an environment string.
            os.environ[{variable!r}] = "backend:native,expandable_segments:{not expandable}"
            before = _active_allocation_lease_count()
            if {expandable!r}:
                for attempt in range(2):
                    try:
                        share_model(model)
                    except RuntimeError as error:
                        message = str(error)
                        assert "does not support legacy CUDA IPC" in message, message
                        assert "expandable_segments" in message, message
                        assert "backend:native,expandable_segments:False" in message, message
                        assert "before" in message and "allocating" in message, message
                        assert "existing" in message, message
                    else:
                        raise AssertionError("VMM allocation was exported as a legacy handle")
                    assert _active_allocation_lease_count() == before
            else:
                _, refs, keepalive = share_model(model)
                assert len(refs) == 1 and len(bytes.fromhex(refs[0]["handle"])) == 64
                assert refs[0]["device"] == device
                keepalive.release()
                assert _active_allocation_lease_count() == before
            # Compare every downloaded value on CPU. GPU torch.equal would
            # additionally create PyTorch's persistent pinned scalar-return
            # cache, obscuring leak checks of this export-only regression.
            assert torch.equal(tensor.cpu(), torch.ones_like(tensor, device="cpu"))
            # Releasing an IPC lease does not destroy the caller's Tensor or
            # clear PyTorch's allocator cache. Drop those independent owners
            # explicitly, including in these short-lived VMM probe processes.
            del model, tensor
            if not {expandable!r}:
                del keepalive
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
        """
    )
    environment = os.environ.copy()
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    environment.pop("PYTORCH_ALLOC_CONF", None)
    environment.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
    environment[variable] = f"backend:native,expandable_segments:{expandable}"
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment,
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

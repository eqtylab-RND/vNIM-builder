# SPDX-License-Identifier: Apache-2.0
"""Small checkpoint through the exact large-benchmark path on real GPUs."""

import json
import os

import pytest
import test_multigpu_integration as integration
from performance import benchmark_large_model as large

from cuattest.ipc import _active_allocation_lease_count

service = integration.service

pytestmark = pytest.mark.skipif(
    os.environ.get("CUATTEST_TEST_MULTIGPU") != "1",
    reason="requires opt-in and at least two CUDA GPUs",
)


def test_checkpoint_benchmark_loads_every_gpu_and_verifies_json(service, tmp_path):
    client, _, torch = service
    st = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "model.safetensors"
    # More than one stored tensor per device; distinct bytes catch accidental
    # substitution, repetition, or device-major rather than name-major folds.
    stored = {
        f"layer.{i}.weight": torch.arange(1024, dtype=torch.bfloat16) + i
        for i in range(torch.cuda.device_count() * 2)
    }
    st.save_file(stored, str(checkpoint))
    output = tmp_path / "result.json"
    assert (
        large.main(
            [
                "--checkpoint",
                str(checkpoint),
                "--url",
                client.url,
                "--min-utilization",
                "0",
                "--runs",
                "2",
                "--warmup-runs",
                "1",
                "--json-output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert len(report["devices"]) == torch.cuda.device_count()
    assert all(d["resident_weight_bytes"] > 0 for d in report["devices"])
    assert len(report["result"]["samples"]) == 2
    assert report["result"]["revision"] == "local-checkpoint"
    assert all(
        s["model_root"] == report["independent_model_root"]
        for s in report["result"]["samples"]
    )
    assert _active_allocation_lease_count() == 0

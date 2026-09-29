# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the compiled attestation host planner."""

import importlib
import sys

import pytest

from cuattest.notary import _MAX_U64, _fused_plan

try:
    _native = importlib.import_module("cuattest._native")
except ModuleNotFoundError:
    if sys.platform.startswith("linux"):
        raise
    pytest.skip(
        "the optimized native host backend is Linux-only", allow_module_level=True
    )


@pytest.mark.parametrize(
    "spans",
    [
        [(0x1000, 0)],
        [(0x1000, 1)],
        [(0x1000, 1024), (0x2000, 1025)],
        [(0x1000, 1), (0x2000, 5 * 1024), (0x3000, 10 * 1024)],
        [(0x1000, 129 * 1024), (0x2000, 300 * 1024 + 17)],
        [(0x1000, 96 * 1024**3)],
        [(0x1000 + index * 0x1000, (index + 1) * 713) for index in range(149)],
    ],
)
def test_native_plan_exactly_matches_python_reference(spans):
    expected = _fused_plan(spans)

    assert _native.plan(spans) == (
        expected.descriptors,
        expected.reduction_offsets,
        expected.total_tiles,
        expected.secondary_tiles,
        expected.levels,
    )


def test_native_plan_rejects_cuda_workspace_overflow():
    with pytest.raises(_native.Error, match="workspace exceeds"):
        _native.plan([(1, _MAX_U64)] * 4096)


def test_native_plan_requires_at_least_one_span():
    with pytest.raises(_native.Error, match="requires 1"):
        _native.plan([])


def test_native_cuda_export_rejects_null_before_loading_the_driver():
    with pytest.raises(ValueError, match="null CUDA storage pointer"):
        _native._export_cuda_allocation(0)


def test_native_cuda_export_requires_an_integer_pointer():
    with pytest.raises(TypeError):
        _native._export_cuda_allocation(object())

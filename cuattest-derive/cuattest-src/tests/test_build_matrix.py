# SPDX-License-Identifier: Apache-2.0
"""Matrix isolation/order and failure reporting, without a compiler or GPU."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_matrix


def test_matrix_prebuilds_kernels_and_copies_only_caches(monkeypatch, tmp_path):
    previous = tmp_path / "previous"
    output = tmp_path / "next"
    for mode in ("Release", "AssertedRelease"):
        for name in ("lib", "cache", "oracle-cache"):
            directory = previous / mode / name
            directory.mkdir(parents=True)
            (directory / "marker").write_text(name)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(build_matrix.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "/stale/source")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k silently_omit_tests")
    assert build_matrix.main([str(output), "--gpu", "--crypto", "--reuse-artifacts", str(previous)]) == 0
    records = json.loads((output / "results.json").read_text())
    assert [r["phase"] for r in records] == [
        "build", "identity", "kernels", "pytest", "selftest-native", "selftest-fallback",
    ] * 2
    for _, env in calls:
        mode = env["CUATTEST_BUILD_TYPE"]
        assert env["PYTHONPATH"].split(os.pathsep)[0] == str(output / mode / "lib")
        assert env["CUATTEST_CACHE"] == str(output / mode / "cache")
        assert env["CUATTEST_SANITIZER_CACHE"] == str(output / mode / "oracle-cache")
        assert "PYTEST_ADDOPTS" not in env
        assert (output / mode / "cache/marker").read_text() == "cache"
        assert (output / mode / "oracle-cache/marker").read_text() == "oracle-cache"
        assert not (output / mode / "lib/marker").exists()


@pytest.mark.parametrize("phase", ["build", "kernels"])
def test_matrix_failed_prerequisite_skips_tests_but_checks_other_configuration(monkeypatch, tmp_path, phase):
    def run(command, **kwargs):
        mode = kwargs["env"]["CUATTEST_BUILD_TYPE"]
        label = Path(kwargs["stdout"].name).stem
        return SimpleNamespace(returncode=17 if mode == "Release" and label == phase else 0)
    monkeypatch.setattr(build_matrix.subprocess, "run", run)
    output = tmp_path / "matrix"
    assert build_matrix.main([str(output), "--gpu"]) == 1
    records = json.loads((output / "results.json").read_text())
    assert not any(r["configuration"] == "Release" and r["phase"] == "pytest" for r in records)
    assert any(r["configuration"] == "AssertedRelease" and r["phase"] == "pytest" for r in records)
    assert [r["returncode"] for r in records if r["returncode"]] == [17]

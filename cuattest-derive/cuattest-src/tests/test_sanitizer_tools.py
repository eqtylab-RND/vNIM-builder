# SPDX-License-Identifier: Apache-2.0
"""CPU-only guards for sanitizer coverage and failure propagation."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuattest import _nvrtc, kernel

SPEC = importlib.util.spec_from_file_location(
    "run_cuda", Path(__file__).with_name("sanitizers") / "run_cuda.py"
)
run_cuda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_cuda)


def test_initcheck_includes_shared_memory():
    options = run_cuda.tool_options("initcheck", driver_only=True)
    assert options[options.index("--initcheck-address-space") + 1] == "all"


@pytest.mark.parametrize("space", ["all", "shared"])
def test_ipc_initcheck_limit_is_explicit_and_raw_check_is_available(space):
    options = run_cuda.tool_options("initcheck", driver_only=False, ipc_initcheck_space=space)
    assert options[options.index("--initcheck-address-space") + 1] == space


@pytest.mark.parametrize("driver_only", [True, False])
def test_module_unload_check_is_driver_only(driver_only):
    options = run_cuda.tool_options("memcheck", driver_only=driver_only)
    assert ("--detect-missing-module-unload" in options) == driver_only
    assert options[options.index("--target-processes") + 1] == "all"
    assert options[options.index("--error-exitcode") + 1] == "86"
    assert options[options.index("--report-api-errors") + 1] == (
        "all" if driver_only else "explicit"
    )


def test_raw_runtime_api_diagnostics_remain_available():
    options = run_cuda.tool_options("memcheck", driver_only=False, api_errors="all")
    assert options[options.index("--report-api-errors") + 1] == "all"


def test_cuda_runner_keeps_testing_but_returns_failure(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=86 if len(calls) == 1 else 0)

    monkeypatch.setattr(run_cuda.subprocess, "run", run)
    assert run_cuda.main([str(tmp_path), "--suite", "selftest", "--tool", "initcheck"]) == 1
    assert [env["CUATTEST_DISABLE_NATIVE_HOST"] for _, env in calls] == ["0", "1"]
    assert "import cuattest._native" in calls[0][0][-1]
    assert "import cuattest._native" not in calls[1][0][-1]
    results = json.loads((tmp_path / "results.json").read_text())
    assert [result["returncode"] for result in results] == [86, 0]


def test_registered_sanitizers_isolate_backends_without_hiding_failures(monkeypatch, tmp_path):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=86 if len(calls) == 1 else 0)
    monkeypatch.setattr(run_cuda.subprocess, "run", run)
    assert run_cuda.main([str(tmp_path), "--suite", "registered", "--tool", "racecheck"]) == 1
    assert [env["CUATTEST_TEST_REGISTERED_BACKEND"] for _, env in calls] == ["native", "fallback"]
    assert all(command[-1] == "tests/test_registered_integration.py" for command, _ in calls)


@pytest.mark.parametrize("tool", run_cuda.TOOLS)
def test_multigpu_sanitizers_use_current_source_and_runtime_safe_options(
    monkeypatch, tmp_path, tool
):
    calls = []
    monkeypatch.setenv("PYTHONPATH", "/stale/installed/checkout")

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_cuda.subprocess, "run", run)
    assert run_cuda.main([str(tmp_path), "--suite", "multigpu", "--tool", tool]) == 0
    assert len(calls) == 1
    command, env = calls[0]
    assert env["CUATTEST_TEST_MULTIGPU"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(
        Path(run_cuda.__file__).resolve().parents[2] / "src"
    )
    assert "requires at least two GPUs" in command[-1]
    assert "tests/test_multigpu_integration.py" in command[-1]
    assert "parallel_repeated_requests" in command[-1]
    assert "--detect-missing-module-unload" not in command
    if tool == "initcheck":
        assert command[command.index("--initcheck-address-space") + 1] == "shared"
    if tool == "memcheck":
        assert command[command.index("--report-api-errors") + 1] == "explicit"
    results = json.loads((tmp_path / "results.json").read_text())
    assert results[0]["suite"] == "multigpu" and results[0]["backend"] == "both"


@pytest.mark.parametrize("inherited_cache", [False, True])
@pytest.mark.parametrize("kernel_dir_override", [False, True])
def test_cuda_runner_isolates_cold_cache_writes(
    monkeypatch, tmp_path, inherited_cache, kernel_dir_override
):
    output = tmp_path / "audit"
    simulated_home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: simulated_home))
    if inherited_cache:
        normal_cache = tmp_path / "normal-cache"
        monkeypatch.setenv("CUATTEST_CACHE", str(normal_cache))
    else:
        normal_cache = simulated_home / ".cache" / "cuattest"
        monkeypatch.delenv("CUATTEST_CACHE", raising=False)
    if kernel_dir_override:
        lookup = tmp_path / "prebuilt"
        lookup.mkdir()
        monkeypatch.setenv("CUATTEST_KERNEL_DIR", str(lookup))
    else:
        lookup = output / "cubins"
        monkeypatch.delenv("CUATTEST_KERNEL_DIR", raising=False)

    # A source mismatch must neither replace the normal cache's existing
    # artifacts nor depend on its writability. Only the NVRTC compiler is
    # doubled: exercise real load_cubin lookup, publication, and cache reuse.
    normal_cubins = normal_cache / "cubins"
    normal_cubins.mkdir(parents=True)
    untouched = {
        kernel.cubin_filename("sm_75"): b"\x7fELFnormal cached kernel",
        kernel.cubin_metadata_filename("sm_75"): b"{}",
    }
    for name, data in untouched.items():
        (normal_cubins / name).write_bytes(data)
    compilations = []
    cubin = b"\x7fELFisolated sanitizer kernel"
    source = "// sanitizer cold-cache regression"

    class FakeNvrtc:
        def compile_cubin(self, text, arch, *, build_type):
            compilations.append((text, arch))
            return cubin

        def version(self):
            return "NVRTC test compiler"

    monkeypatch.setattr(_nvrtc, "Nvrtc", FakeNvrtc)
    calls = []
    parent_cache = os.environ.get("CUATTEST_CACHE")
    parent_lookup = os.environ.get("CUATTEST_KERNEL_DIR")

    def run(command, **kwargs):
        env = kwargs["env"]
        calls.append(env)
        # Model the subprocess's cache environment without mutating the
        # runner's parent environment or requiring CUDA on the CPU test host.
        with monkeypatch.context() as child:
            for name in ("CUATTEST_CACHE", "CUATTEST_KERNEL_DIR"):
                if name in env:
                    child.setenv(name, env[name])
                else:
                    child.delenv(name, raising=False)
            blob, _, path = kernel.load_cubin("sm_75", source)
            assert path.parent == output / "cubins"
            assert blob == cubin
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_cuda.subprocess, "run", run)
    normal_cubins.chmod(0o555)
    try:
        assert run_cuda.main(
            [str(output), "--suite", "selftest", "--tool", "memcheck"]
        ) == 0
    finally:
        normal_cubins.chmod(0o755)

    assert len(calls) == 2
    assert compilations == [(source, "sm_75")]
    assert all(env["CUATTEST_CACHE"] == str(output) for env in calls)
    assert all(env["CUATTEST_KERNEL_DIR"] == str(lookup) for env in calls)
    assert os.environ.get("CUATTEST_CACHE") == parent_cache
    assert os.environ.get("CUATTEST_KERNEL_DIR") == parent_lookup
    assert {path.name: path.read_bytes() for path in normal_cubins.iterdir()} == untouched
    if kernel_dir_override:
        assert list(lookup.iterdir()) == []
    metadata_path = output / "cubins" / kernel.cubin_metadata_filename("sm_75")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source_blake3"] == kernel.source_digest(source).hex()

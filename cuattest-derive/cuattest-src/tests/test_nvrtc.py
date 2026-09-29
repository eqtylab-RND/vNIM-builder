# SPDX-License-Identifier: Apache-2.0
"""NVRTC discovery for NVIDIA's pip wheel layout."""

from types import SimpleNamespace

from cuattest import _nvrtc


def test_wheel_library_is_an_absolute_candidate_before_bare_sonames(monkeypatch, tmp_path):
    package = tmp_path / "site-packages" / "nvidia" / "cuda_nvrtc"
    lib = package / "lib"
    lib.mkdir(parents=True)
    wheel_nvrtc = lib / "libnvrtc.so.12"
    wheel_nvrtc.write_bytes(b"stub")
    spec = SimpleNamespace(submodule_search_locations=[str(package)])
    monkeypatch.setattr(_nvrtc.importlib.util, "find_spec", lambda name: spec)

    candidates = _nvrtc.nvrtc_candidates()

    assert candidates[0] == str(wheel_nvrtc.resolve())
    assert candidates.index(str(wheel_nvrtc.resolve())) < candidates.index("libnvrtc.so.12")


def test_missing_wheel_falls_back_to_loader_sonames(monkeypatch):
    monkeypatch.setattr(_nvrtc.importlib.util, "find_spec", lambda name: None)
    assert _nvrtc.nvrtc_candidates() == _nvrtc._CANDIDATES


def test_cuda_include_dirs_find_runtime_and_cccl_wheels_only(monkeypatch, tmp_path):
    nvidia = tmp_path / "site-packages" / "nvidia"
    runtime = nvidia / "cuda_runtime" / "include"
    cccl = nvidia / "cuda_cccl" / "include"
    versioned = nvidia / "cu13" / "include"
    unrelated = nvidia / "cudnn" / "include"
    for directory in (runtime, cccl, versioned, unrelated):
        directory.mkdir(parents=True)
    spec = SimpleNamespace(submodule_search_locations=[str(nvidia)])
    monkeypatch.setattr(_nvrtc.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)

    include_dirs = _nvrtc.cuda_include_dirs()

    assert include_dirs[:3] == (
        runtime.resolve(), cccl.resolve(), versioned.resolve()
    )
    assert unrelated.resolve() not in include_dirs

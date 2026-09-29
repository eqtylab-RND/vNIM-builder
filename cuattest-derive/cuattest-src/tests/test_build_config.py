# SPDX-License-Identifier: Apache-2.0
"""The build mode controls actual assertions, not just a configuration label."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import sysconfig
from types import SimpleNamespace

import pytest

from cuattest import _build_config as config, kernel, notary
from cuattest._build_options import BUILD_TYPES, cuda_options, native_options


@pytest.mark.parametrize("mode", BUILD_TYPES)
def test_compiler_policy_overrides_cpython_ndebug_and_keeps_release_optimization(mode):
    flags = native_options(mode)
    assert ("-DNDEBUG" in flags) == (mode == "Release")
    assert ("-UNDEBUG" in flags) == (mode != "Release")
    assert ("-O3" in flags) == (mode != "Debug")
    assert ("-O0" in flags) == (mode == "Debug")
    assert ("--device-debug" in cuda_options(mode)) == (mode == "Debug")
    assert ("--dopt=on" in cuda_options(mode)) == (mode != "Debug")


@pytest.mark.parametrize("value", ["release", "RelWithDebInfo", "", "Typo"])
def test_invalid_mode_is_never_silently_release(value):
    for select in (native_options, cuda_options):
        with pytest.raises(ValueError, match="CUATTEST_BUILD_TYPE"):
            select(value)


def test_installed_native_and_python_agree():
    native = pytest.importorskip("cuattest._native")
    assert native.BUILD_TYPE == config.BUILD_TYPE
    assert bool(native.ASSERTIONS_ENABLED) == config.ASSERTIONS_ENABLED


def test_runtime_cannot_change_the_compiled_native_assertion_policy():
    assert config._select_build_type(None, None, None) == "Release"
    for mode in BUILD_TYPES:
        assert config._select_build_type(None, mode, None) == mode
        assert config._select_build_type(mode, None, mode) == mode
    with pytest.raises(RuntimeError, match="rebuild/reinstall"):
        config._select_build_type("AssertedRelease", "Release", "Release")


def test_python_asserted_build_refuses_optimized_interpreter():
    result = subprocess.run(
        [sys.executable, "-O", "-c", "import cuattest._build_config"],
        capture_output=True, text=True,
    )
    if config.ASSERTIONS_ENABLED:
        assert result.returncode != 0
        assert "without -O" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


def test_python_plan_postcondition_uses_the_active_build_policy(monkeypatch):
    # Corrupt an INTERNAL packer's ABI, not a client's input. Release skips the
    # diagnostic; asserted modes catch it before it could reach the driver.
    monkeypatch.setattr(notary, "_TENSOR_SPAN", SimpleNamespace(size=40, pack=lambda *args: b"bad"))
    if config.ASSERTIONS_ENABLED:
        with pytest.raises(AssertionError):
            notary._fused_plan([(1, 1)])
    else:
        assert notary._fused_plan([(1, 1)]).descriptors == b"bad"


@pytest.mark.parametrize("mode", BUILD_TYPES)
def test_native_internal_assertion_really_aborts_only_asserted_builds(tmp_path, mode):
    resource = pytest.importorskip("resource")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a C++ compiler")
    source = Path(__file__).resolve().parents[1] / "src/cuattest/_native.cpp"
    core = source.read_text().split("\nbool py_u64(", 1)[0] + "\n}\n"
    probe = tmp_path / "assert_probe.cpp"
    executable = tmp_path / "probe"
    # append_u64_le requires an aligned internal serialization cursor. There
    # is no UB in the Release control (one prefix byte + eight appended bytes).
    probe.write_text(core + '''
int main() {
  std::vector<unsigned char> bytes(1);
  append_u64_le(bytes, 123);
  return bytes.size() == 9 ? 0 : 2;
}
''')
    subprocess.run([
        compiler, "-DNDEBUG", *native_options(mode),
        f"-I{sysconfig.get_paths()['include']}", str(probe), "-ldl", "-o", str(executable),
    ], check=True, capture_output=True, text=True)
    def no_core():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = subprocess.run([str(executable)], capture_output=True, text=True, preexec_fn=no_core)
    if mode == "Release":
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode == -signal.SIGABRT
        assert "output.size() % sizeof(value) == 0" in result.stderr


def test_rebuilding_same_object_directory_changes_native_mode(tmp_path):
    if not sys.platform.startswith("linux") or shutil.which("c++") is None:
        pytest.skip("requires the Linux extension compiler")
    root = Path(__file__).resolve().parents[1]
    library = tmp_path / "lib"
    for mode in ("Release", "AssertedRelease", "Debug", "Release"):
        env = dict(os.environ, CUATTEST_BUILD_TYPE=mode)
        result = subprocess.run([
            sys.executable, "setup.py", "build_ext", "--build-temp", str(tmp_path / "objects"),
            "--build-lib", str(library),
        ], cwd=root, env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        native = next((library / "cuattest").glob("_native*.so"))
        # A fresh interpreter reads the new DSO, never a previously dlopened
        # image. Reuse precisely the SAME object/output paths without --force.
        result = subprocess.run([sys.executable, "-c", '''
import importlib.util, sys
spec = importlib.util.spec_from_file_location("_native", sys.argv[1])
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
assert module.BUILD_TYPE == sys.argv[2]
assert bool(module.ASSERTIONS_ENABLED) == (sys.argv[2] != "Release")
''', str(native), mode], capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", BUILD_TYPES)
def test_cubin_cache_binds_mode_and_options(monkeypatch, tmp_path, mode):
    import json
    from cuattest import _nvrtc
    class Compiler:
        def compile_cubin(self, source, arch, *, build_type):
            return b"\x7fELF" + build_type.encode()
        def version(self):
            return "test"
    monkeypatch.setattr(_nvrtc, "Nvrtc", Compiler)
    monkeypatch.setattr(kernel, "search_dirs", lambda: [tmp_path])
    _, _, path = kernel.build_cubin("sm_75", tmp_path, "source", build_type=mode)
    assert kernel.find_cubin("sm_75", "source", build_type=mode) is not None
    for other in set(BUILD_TYPES) - {mode}:
        assert kernel.find_cubin("sm_75", "source", build_type=other) is None
    metadata_path = Path(str(path) + ".source-blake3")
    metadata = json.loads(metadata_path.read_text())
    metadata["build_options"] = ["--wrong-assertion-policy"]
    metadata_path.write_text(json.dumps(metadata))
    assert kernel.find_cubin("sm_75", "source", build_type=mode) is None
    metadata.pop("build_options")
    metadata.pop("build_type")
    metadata_path.write_text(json.dumps(metadata))
    assert kernel.find_cubin("sm_75", "source", build_type=mode) is None


@pytest.mark.skipif(os.environ.get("CUATTEST_TEST_GPU") != "1", reason="set CUATTEST_TEST_GPU=1")
def test_actual_device_assertion_is_enabled_by_build_mode(tmp_path):
    from cuattest._cuda import Cuda
    from cuattest._nvrtc import Nvrtc
    cu = Cuda()
    cu.init()
    major, minor = cu.compute_capability(cu.device(0))
    # Compile the actual BLAKE3 implementation, without the unused signing
    # graph. An oversized internal chunk is safe for this 1,056-byte control
    # allocation, but violates BLAKE3's semantic 1,024-byte chunk invariant.
    source = kernel.kernel_source().split("// ===== SHA-256", 1)[0]
    source += '''
extern "C" __global__ void invariant_probe(const Byte *input, Uint32 *output) {
    blake3_hash_chunk(input, 1025, 0, 1, output);
}
'''
    cubin = Nvrtc().compile_cubin(source, f"sm_{major}{minor}")
    assert (b"__assertfail" in cubin) == config.ASSERTIONS_ENABLED
    path = tmp_path / "probe.cubin"
    path.write_bytes(cubin)
    # A CUDA assertion poisons its context. Keep fault injection in a child,
    # never the pytest/Torch context used by subsequent real-GPU tests.
    result = subprocess.run([sys.executable, "-c", '''
import ctypes, sys
from pathlib import Path
from cuattest._cuda import Cuda, CudaError, DeviceBuffer
cu = Cuda(); cu.init(); ctx = cu.ctx_create(cu.device(0))
try:
    mod = cu.module_load(Path(sys.argv[1]).read_bytes())
    data = DeviceBuffer.from_bytes(cu, bytes(1056))
    out = DeviceBuffer(cu, 32)
    cu.launch(cu.function(mod, "invariant_probe"), 1, 1,
              [ctypes.c_uint64(data.ptr), ctypes.c_uint64(out.ptr)])
    try:
        cu.sync()
    except CudaError as error:
        print(str(error), flush=True)
        sys.exit(23)
finally:
    cu.ctx_destroy(ctx)
''', str(path)], capture_output=True, text=True, timeout=60)
    if config.ASSERTIONS_ENABLED:
        assert result.returncode == 23, result.stdout + result.stderr
        assert "chunk_byte_count <= 1024" in result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr

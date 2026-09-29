# SPDX-License-Identifier: Apache-2.0
"""NVRTC binding: compile the kernel source to a CUBIN.

NVRTC needs no GPU and no driver, so a CUBIN can be built on a machine that
has none - which is how `cuattest build-kernel` produces artefacts to ship
alongside the package.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path

from ._build_config import BUILD_TYPE
from ._build_options import cuda_options


class NvrtcError(RuntimeError):
    """NVRTC is unavailable, or the kernel did not compile."""


_CANDIDATES = ("libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11.2")


def _wheel_candidates() -> list[str]:
    """Absolute NVRTC paths installed by NVIDIA's Python wheel.

    ``nvidia-cuda-nvrtc-cu12`` installs into a namespace package rather than a
    normal dynamic-loader directory. A bare ``libnvrtc.so.12`` therefore does
    not find an otherwise correctly installed build extra.
    """
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
    except (ImportError, ModuleNotFoundError, ValueError):
        return []
    locations = () if spec is None else (spec.submodule_search_locations or ())
    found: list[str] = []
    for location in locations:
        lib_dir = Path(location) / "lib"
        # Prefer the documented sonames, then accept newer minor-versioned
        # wheels without needing a package release for each CUDA revision.
        paths = [lib_dir / name for name in _CANDIDATES]
        paths.extend(sorted(lib_dir.glob("libnvrtc.so.*"), reverse=True))
        for path in paths:
            if path.is_file():
                absolute = str(path.resolve())
                if absolute not in found:
                    found.append(absolute)
    return found


def nvrtc_candidates() -> tuple[str, ...]:
    """Load wheel-local libraries before falling back to loader sonames."""
    return tuple(_wheel_candidates()) + _CANDIDATES


def cuda_include_dirs() -> tuple[Path, ...]:
    """Find CUDA runtime and CCCL headers supplied by wheels or a toolkit."""
    candidates: list[Path] = []

    try:
        nvidia = importlib.util.find_spec("nvidia")
    except (ImportError, ModuleNotFoundError, ValueError):
        nvidia = None
    for location in () if nvidia is None else (nvidia.submodule_search_locations or ()):
        root = Path(location)
        candidates.extend(
            [
                root / "cuda_runtime" / "include",
                root / "cuda_cccl" / "include",
            ]
        )
        # Namespace-package roots can also contain unrelated libraries such
        # as cuDNN or cuSPARSELt. Only the versioned CUDA umbrella directory
        # (for example ``nvidia/cu13/include``) is a compiler include root.
        candidates.extend(sorted(root.glob("cu[0-9]*/include"), reverse=True))

    for variable in ("CUDA_HOME", "CUDA_PATH"):
        if root := os.environ.get(variable):
            candidates.append(Path(root) / "include")
    candidates.append(Path("/usr/local/cuda/include"))

    found: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            resolved = candidate.resolve()
            if resolved not in found:
                found.append(resolved)
            cccl = (candidate / "cccl")
            if cccl.is_dir() and (resolved_cccl := cccl.resolve()) not in found:
                found.append(resolved_cccl)
    return tuple(found)


class Nvrtc:
    def __init__(self, soname: str | None = None) -> None:
        names = (soname,) if soname else nvrtc_candidates()
        errors = []
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError as e:
                errors.append(f"{n}: {e}")
        else:
            raise NvrtcError(
                "cannot load libnvrtc (tried " + ", ".join(names) + "). It ships with "
                "the CUDA toolkit; without a toolkit, `pip install 'cuattest[build]'` "
                "provides it and the required headers, or build the CUBIN elsewhere and point "
                "CUATTEST_KERNEL_DIR at it.\n  " + "\n  ".join(errors)
            )
        c, v, p, sz = ctypes.c_int, ctypes.c_void_p, ctypes.POINTER, ctypes.c_size_t
        for name, argtypes in {
            "nvrtcVersion": [p(c), p(c)],
            "nvrtcCreateProgram": [p(v), ctypes.c_char_p, ctypes.c_char_p, c, v, v],
            "nvrtcCompileProgram": [v, c, p(ctypes.c_char_p)],
            "nvrtcGetProgramLogSize": [v, p(sz)],
            "nvrtcGetProgramLog": [v, ctypes.c_char_p],
            "nvrtcGetCUBINSize": [v, p(sz)],
            "nvrtcGetCUBIN": [v, ctypes.c_char_p],
            "nvrtcDestroyProgram": [p(v)],
            "nvrtcGetErrorString": [c],
        }.items():
            fn = getattr(self.lib, name)
            fn.argtypes = argtypes
            fn.restype = ctypes.c_char_p if name == "nvrtcGetErrorString" else c
            setattr(self, name, fn)

    def _err(self, rc: int) -> str:
        s = self.nvrtcGetErrorString(rc)
        return s.decode() if s else f"nvrtc error {rc}"

    def version(self) -> str:
        major, minor = ctypes.c_int(), ctypes.c_int()
        self.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
        return f"{major.value}.{minor.value}"

    def compile_cubin(self, source: str, arch: str, *, build_type: str | None = None) -> bytes:
        """Compile for a *real* architecture (sm_90, not compute_90) → CUBIN."""
        policy = cuda_options(BUILD_TYPE if build_type is None else build_type)
        prog = ctypes.c_void_p()
        rc = self.nvrtcCreateProgram(ctypes.byref(prog), source.encode(),
                                     b"p256_cuda_notary_b3.cu", 0, None, None)
        if rc != 0:
            raise NvrtcError(f"nvrtcCreateProgram: {self._err(rc)}")
        try:
            option_values = [b"--std=c++11", f"--gpu-architecture={arch}".encode()]
            option_values.extend(option.encode() for option in policy)
            option_values.extend(
                f"--include-path={path}".encode() for path in cuda_include_dirs()
            )
            opts = (ctypes.c_char_p * len(option_values))(*option_values)
            rc = self.nvrtcCompileProgram(prog, len(option_values), opts)
            if rc != 0:
                raise NvrtcError(f"nvrtcCompileProgram ({arch}): {self._err(rc)}\n{self._log(prog)}")
            size = ctypes.c_size_t()
            rc = self.nvrtcGetCUBINSize(prog, ctypes.byref(size))
            if rc != 0:
                raise NvrtcError(f"nvrtcGetCUBINSize: {self._err(rc)}")
            buf = ctypes.create_string_buffer(size.value)
            rc = self.nvrtcGetCUBIN(prog, buf)
            if rc != 0:
                raise NvrtcError(f"nvrtcGetCUBIN: {self._err(rc)}")
            return buf.raw[: size.value]
        finally:
            self.nvrtcDestroyProgram(ctypes.byref(prog))

    def _log(self, prog) -> str:
        size = ctypes.c_size_t()
        if self.nvrtcGetProgramLogSize(prog, ctypes.byref(size)) != 0 or size.value <= 1:
            return ""
        buf = ctypes.create_string_buffer(size.value)
        if self.nvrtcGetProgramLog(prog, buf) != 0:
            return ""
        return buf.value.decode(errors="replace")

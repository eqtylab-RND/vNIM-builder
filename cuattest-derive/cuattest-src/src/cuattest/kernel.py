# SPDX-License-Identifier: Apache-2.0
"""Installing the kernel: locating the source, compiling it, caching the CUBIN.

"Installing" means producing machine code for this GPU's architecture and
putting it where the notary will find it. The CUBIN is what actually runs, so
it is what gets hashed and registered - the source is kept beside it for
reference, never substituted for it.

Search order for a CUBIN:
  1. $CUATTEST_KERNEL_DIR
  2. the package's own ``cubins/`` directory (populated by `build-kernel`)
  3. the user cache, ~/.cache/cuattest/cubins
and if none has one for this architecture, compile with NVRTC and cache it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from ._build_config import BUILD_TYPE
from ._build_options import cuda_options, validate_build_type
from ._hosthash import blake3_digest

KERNEL_FILENAME = "p256_cuda_notary_b3.cu"


def kernel_source_path() -> Path:
    """The .cu shipped with this package."""
    here = Path(__file__).resolve().parent
    for cand in (
        here / "kernel" / KERNEL_FILENAME,
        here.parent.parent / "kernel" / KERNEL_FILENAME,
    ):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"{KERNEL_FILENAME} not found; expected it beside the package or at "
        "<repo>/kernel/. Set CUATTEST_KERNEL_SRC to override."
    )


def kernel_source() -> str:
    override = os.environ.get("CUATTEST_KERNEL_SRC")
    path = Path(override) if override else kernel_source_path()
    return path.read_text()


def cubin_filename(arch: str, build_type: str | None = None) -> str:
    mode = validate_build_type(BUILD_TYPE if build_type is None else build_type)
    suffix = "" if mode == "Release" else f".{mode}"
    return f"p256_cuda_notary_b3.{arch}{suffix}.cubin"


def cubin_metadata_filename(arch: str, build_type: str | None = None) -> str:
    """Sidecar binding a cached CUBIN to its exact source and bytes."""
    return f"{cubin_filename(arch, build_type)}.source-blake3"


def source_digest(source: str | None = None) -> bytes:
    return blake3_digest((kernel_source() if source is None else source).encode())


def cache_dir() -> Path:
    return (
        Path(os.environ.get("CUATTEST_CACHE", Path.home() / ".cache" / "cuattest"))
        / "cubins"
    )


def search_dirs() -> list[Path]:
    dirs = []
    if env := os.environ.get("CUATTEST_KERNEL_DIR"):
        dirs.append(Path(env))
    dirs.append(Path(__file__).resolve().parent / "cubins")
    dirs.append(cache_dir())
    return dirs


def find_cubin(arch: str, source: str | None = None, *, build_type: str | None = None) -> tuple[bytes, str, Path] | None:
    """Find a CUBIN built from exactly ``source``.

    Architecture-only cache keys are unsafe: both package upgrades and
    ``CUATTEST_KERNEL_SRC`` can change the source without changing ``sm_XX``.
    The sidecar also commits to the CUBIN bytes so an interrupted/concurrent
    cache update cannot pair new code with old metadata.
    """
    mode = validate_build_type(BUILD_TYPE if build_type is None else build_type)
    name = cubin_filename(arch, mode)
    wanted_source = source_digest(source).hex()
    for d in search_dirs():
        path = d / name
        metadata_path = d / cubin_metadata_filename(arch, mode)
        if not path.is_file() or not metadata_path.is_file():
            continue
        try:
            cubin = path.read_bytes()
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError, TypeError):
            continue
        if not isinstance(metadata, dict):
            continue
        compiler = metadata.get("compiler")
        if (
            metadata.get("source_blake3") != wanted_source
            or metadata.get("cubin_blake3") != blake3_digest(cubin).hex()
            or not isinstance(compiler, str)
            or not compiler.strip()
            or metadata.get("build_type") != mode
            or metadata.get("build_options") != cuda_options(mode)
        ):
            continue
        # Compiler provenance belongs to this exact digest-bound artifact.
        # A directory-global version file is wrong after a partial rebuild in
        # which another architecture was compiled by a newer toolkit.
        return cubin, f"{compiler} (prebuilt)", path
    return None


def build_cubin(
    arch: str, out_dir: Path | None = None, source: str | None = None, *, build_type: str | None = None
) -> tuple[bytes, str, Path]:
    """Compile the kernel for `arch` and cache the result. Needs NVRTC, not a GPU."""
    from ._nvrtc import Nvrtc

    mode = validate_build_type(BUILD_TYPE if build_type is None else build_type)
    source = kernel_source() if source is None else source
    nvrtc = Nvrtc()
    cubin = nvrtc.compile_cubin(source, arch, build_type=mode)
    compiler = nvrtc.version()
    if not cubin.startswith(b"\x7fELF"):
        raise RuntimeError(
            f"NVRTC returned {len(cubin)} bytes that are not an ELF CUBIN — "
            f"'{arch}' must be a real architecture (sm_90), not a virtual one"
        )
    out_dir = out_dir or cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / cubin_filename(arch, mode)
    metadata_path = out_dir / cubin_metadata_filename(arch, mode)
    metadata = {
        "source_blake3": source_digest(source).hex(),
        "cubin_blake3": blake3_digest(cubin).hex(),
        "compiler": compiler,
        "build_type": mode,
        "build_options": cuda_options(mode),
    }
    # Publish each file by rename. Readers verify both digests, so even the
    # brief interval between renames is a cache miss, never stale execution.
    # A PID is not unique to a call: independent notaries can compile in two
    # threads. Give each invocation its own directory on the same filesystem
    # and clean up its unpublished files even when a write/rename fails.
    with TemporaryDirectory(prefix=f".{path.name}.", dir=out_dir) as staging:
        tmp_path = Path(staging) / path.name
        tmp_metadata = Path(staging) / metadata_path.name
        tmp_path.write_bytes(cubin)
        tmp_metadata.write_text(json.dumps(metadata, sort_keys=True) + "\n")
        tmp_path.replace(path)
        tmp_metadata.replace(metadata_path)
    return cubin, compiler, path


def load_cubin(arch: str, source: str | None = None) -> tuple[bytes, str, Path]:
    """Prebuilt if there is one for this arch, otherwise compile and cache."""
    source = kernel_source() if source is None else source
    found = find_cubin(arch, source)
    if found is not None:
        return found
    return build_cubin(arch, source=source)

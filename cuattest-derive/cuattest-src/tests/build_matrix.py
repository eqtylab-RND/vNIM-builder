# SPDX-License-Identifier: Apache-2.0
"""Build/test isolated Release and AssertedRelease packages without reinstalling."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new artifact directory")
    parser.add_argument("--configuration", action="append", choices=["Release", "Debug", "AssertedRelease"])
    parser.add_argument("--gpu", action="store_true", help="real GPU/IPC tests and both backend selftests")
    parser.add_argument("--crypto", action="store_true", help="independent CUDA cryptographic oracles")
    parser.add_argument("--crypto-devices", default="0", help="0, comma-separated ordinals, or all")
    parser.add_argument("--multigpu", action="store_true", help="requires at least two visible idle GPUs")
    parser.add_argument("--large-capacity", action="store_true", help="reserves almost all free VRAM; idle GPU only")
    parser.add_argument("--hash-mode", choices=["auto", "standard", "async"], default="auto")
    parser.add_argument("--kernel-dir", type=Path, help="optional matching prebuilt CUBINs; cache remains isolated")
    parser.add_argument("--reuse-artifacts", type=Path, help="copy verified kernel/oracle caches from a previous matrix; always rebuild the host")
    args = parser.parse_args(argv)
    if args.reuse_artifacts is not None and not args.reuse_artifacts.is_dir():
        parser.error("--reuse-artifacts must name a previous matrix directory")
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    # Never replace a previous audit's evidence or a user's build directory.
    output.mkdir(parents=True, exist_ok=False)
    records = []

    def run(mode, label, command, env):
        log = output / mode / (label + ".log")
        print(f"{mode}: {label} -> {log}", flush=True)
        started = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        records.append(dict(configuration=mode, phase=label, command=command,
                            returncode=result.returncode, seconds=time.monotonic() - started,
                            log=str(log)))
        (output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"{mode}: {label} exit {result.returncode}", flush=True)
        return result.returncode == 0

    for mode in args.configuration or ["Release", "AssertedRelease"]:
        directory = output / mode
        library = directory / "lib"
        directory.mkdir()
        if args.reuse_artifacts is not None:
            # Only copy caches, never the previous Python package/extension.
            # Normal source/policy/digest checks still validate every hit.
            for name in ("cache", "oracle-cache"):
                previous = args.reuse_artifacts / mode / name
                if previous.is_dir():
                    shutil.copytree(previous, directory / name)
        env = dict(os.environ, CUATTEST_BUILD_TYPE=mode, CUATTEST_HASH_MODE=args.hash_mode,
                   CUATTEST_CACHE=str(directory / "cache"),
                   CUATTEST_SANITIZER_CACHE=str(directory / "oracle-cache"),
                   CUATTEST_CRYPTO_DEVICES=args.crypto_devices)
        # The build's lib comes first for BOTH pytest and all producer/server
        # children. Never overwrite/import the editable in-place extension.
        env["PYTHONPATH"] = os.pathsep.join([str(library), str(root / "tests/sanitizers")])
        for key in ("PYTHONOPTIMIZE", "CUATTEST_KERNEL_SRC", "CUATTEST_DISABLE_NATIVE_HOST",
                    "CUATTEST_TEST_REGISTERED_BACKEND", "PYTEST_ADDOPTS"):
            env.pop(key, None)
        env["CUATTEST_KERNEL_DIR"] = str(args.kernel_dir.resolve()) if args.kernel_dir else str(directory / "cache/cubins")
        for flag, enabled in (("GPU", args.gpu), ("CRYPTO", args.crypto),
                              ("MULTIGPU", args.multigpu), ("LARGE_CAPACITY", args.large_capacity)):
            env[f"CUATTEST_TEST_{flag}"] = "1" if enabled else "0"
        if not run(mode, "build", [sys.executable, "setup.py", "build", "--build-base",
                                     str(directory / "build"), "--build-lib", str(library)], env):
            continue
        verify = '''
import pathlib, sys
import cuattest
from cuattest import _native
from cuattest._build_config import BUILD_TYPE, ASSERTIONS_ENABLED
assert pathlib.Path(cuattest.__file__).resolve().parent.parent == pathlib.Path(sys.argv[1])
assert BUILD_TYPE == _native.BUILD_TYPE == sys.argv[2]
assert ASSERTIONS_ENABLED == bool(_native.ASSERTIONS_ENABLED) == (BUILD_TYPE != "Release")
print(cuattest.__file__, _native.__file__, BUILD_TYPE, ASSERTIONS_ENABLED)
'''
        if args.multigpu:
            verify += '\nfrom cuattest._cuda import Cuda\nc = Cuda(); c.init(); assert c.device_count() >= 2\n'
        if not run(mode, "identity", [sys.executable, "-c", verify, str(library), mode], env):
            continue
        if args.gpu or args.crypto or args.multigpu or args.large_capacity:
            # Asserted CUDA translation units can take several minutes to
            # compile. Build once outside HTTP fixture startup deadlines; those
            # deadlines should detect a stuck service, not an optimizing NVRTC.
            kernels = '''
from cuattest._cuda import Cuda
from cuattest.kernel import load_cubin
cu = Cuda(); cu.init()
arches = {"sm_%d%d" % cu.compute_capability(cu.device(i)) for i in range(cu.device_count())}
if not arches: raise RuntimeError("GPU matrix requires a visible CUDA device")
for arch in sorted(arches):
    data, compiler, path = load_cubin(arch)
    print(arch, len(data), compiler, path, flush=True)
'''
            if not run(mode, "kernels", [sys.executable, "-c", kernels], env):
                continue
        command = [sys.executable, "-m", "pytest", "-q", "tests"]
        if args.crypto:
            command += ["-p", "cache_oracle"]
        run(mode, "pytest", command, env)
        if args.gpu:
            for backend in ("native", "fallback"):
                selected = dict(env, CUATTEST_DISABLE_NATIVE_HOST="1" if backend == "fallback" else "0")
                run(mode, "selftest-" + backend,
                    [sys.executable, "-m", "cuattest.cli", "selftest"], selected)
    return int(any(record["returncode"] != 0 for record in records))


if __name__ == "__main__":
    raise SystemExit(main())

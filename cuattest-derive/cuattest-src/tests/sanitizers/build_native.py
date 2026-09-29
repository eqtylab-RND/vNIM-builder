# SPDX-License-Identifier: Apache-2.0
"""Build a host-only sanitizer harness without modifying the installed extension."""

import argparse
import platform
import shlex
import subprocess
import sysconfig
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sanitizer", choices=[
        "address", "undefined", "leak", "thread", "memory", "integer",
        "cfi", "safe-stack", "hwaddress", "none",
    ])
    parser.add_argument("output", type=Path)
    parser.add_argument("--msan-libcxx", type=Path)
    parser.add_argument("--pointer-pairs", action="store_true")
    parser.add_argument("--cxx", default="clang++")
    parser.add_argument("--linker", default="lld")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = here.parents[1] / "src/cuattest/_native.cpp"
    text = source.read_text()
    # Do not maintain a fork of the planner/runner. Extract its exact prefix;
    # fail loudly if the binding boundary moves. #line preserves source sites.
    assert text.count("\nbool py_u64(") == 1
    core = text.split("\nbool py_u64(", 1)[0] + "\n} // namespace\n"
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "native_core.hpp").write_text(f'#line 1 "{source}"\n' + core)
    flags = ["-O1", "-g", "-fno-omit-frame-pointer", "-fno-optimize-sibling-calls"]
    if args.sanitizer != "none":
        flags += [f"-fsanitize={args.sanitizer}", "-fno-sanitize-recover=all"]
    if args.sanitizer == "address":
        flags += ["-fsanitize=undefined,bounds",
                  "-fsanitize-address-use-after-scope",
                  "-fsanitize-address-use-after-return=always"]
        if args.pointer_pairs:
            flags += ["-fsanitize=pointer-compare,pointer-subtract", "-O0"]
    if args.sanitizer == "undefined":
        flags += ["-fsanitize=bounds,local-bounds,implicit-conversion"]
    if args.sanitizer == "memory":
        flags += ["-fsanitize-memory-track-origins=2"]
    if args.sanitizer == "integer":
        flags += [f"-fsanitize-ignorelist={here / 'integer.ignore'}"]
    if args.sanitizer == "hwaddress" and platform.machine() == "x86_64":
        # This host-only harness never forks. x86 page aliasing checks the
        # heap only; do not mistake it for AArch64's full tagged-memory mode.
        flags += ["-fsanitize-hwaddress-experimental-aliasing"]
    if args.sanitizer == "cfi":
        flags += ["-flto", "-fvisibility=default", "-fsanitize-cfi-cross-dso",
                  "-fno-sanitize-trap=all", f"-fuse-ld={args.linker}"]
    cxx = []
    if args.msan_libcxx:
        prefix = args.msan_libcxx.resolve()
        cxx = ["-stdlib=libc++", "-nostdinc++", f"-I{prefix}/include/c++/v1",
               f"-L{prefix}/lib", f"-Wl,-rpath,{prefix}/lib"]
    common = [args.cxx, *flags, *cxx, "-std=c++17",
              f"-I{args.output.resolve()}", f"-I{sysconfig.get_paths()['include']}"]
    commands = [
        [*common, "-fPIC", "-shared", str(here / "fake_cuda.cpp"),
         "-Wl,-soname,libcuda.so.1", "-o", str(args.output / "libcuda.so.1")],
        [*common,
         str(here / "native_harness.cpp"), f"-L{args.output.resolve()}",
         f"-Wl,-rpath,{args.output.resolve()}", "-l:libcuda.so.1", "-pthread", "-ldl",
         "-o", str(args.output / "harness")],
    ]
    for command in commands:
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""Dependency-free compiler policy shared by setuptools and runtime NVRTC."""

BUILD_TYPES = ("Release", "Debug", "AssertedRelease")


def validate_build_type(value):
    if value not in BUILD_TYPES:
        raise ValueError(f"CUATTEST_BUILD_TYPE must be one of {', '.join(BUILD_TYPES)}")
    return value


def native_options(build_type):
    validate_build_type(build_type)
    # These go LAST, after Python's sysconfig/environment flags. In particular
    # CPython normally supplies -DNDEBUG even when building another extension
    # in debug mode; merely omitting our own definition would leave asserts off.
    flags = ["-std=c++17", "-fvisibility=hidden"]
    flags += ["-O0", "-g3", "-UNDEBUG"] if build_type == "Debug" else (
        ["-O3", "-g", "-UNDEBUG"] if build_type == "AssertedRelease" else
        ["-O3", "-g0", "-DNDEBUG"]
    )
    return flags + [f'-DCUATTEST_BUILD_TYPE="{build_type}"']


def cuda_options(build_type):
    validate_build_type(build_type)
    if build_type == "Release":
        return ["--define-macro=NDEBUG", "--dopt=on"]
    if build_type == "Debug":
        # NVRTC -G without -dopt turns optimization off. --dopt=off is NOT a
        # supported spelling. AssertedRelease intentionally does not use -G.
        return ["--undefine-macro=NDEBUG", "--device-debug"]
    return ["--undefine-macro=NDEBUG", "--dopt=on", "--generate-line-info"]

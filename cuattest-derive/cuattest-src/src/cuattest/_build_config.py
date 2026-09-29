# SPDX-License-Identifier: Apache-2.0
"""One assertion policy for native, CUDA and Python internal invariants."""

import os

from ._build_info import BUILD_TYPE as _packaged_type
from ._build_options import validate_build_type


def _select_build_type(requested, packaged, native):
    types = [validate_build_type(v) for v in (requested, packaged, native) if v is not None]
    if len(set(types)) > 1:
        raise RuntimeError(
            "cuAttest build configuration mismatch; rebuild/reinstall with "
            "CUATTEST_BUILD_TYPE set consistently (it cannot toggle native asserts at runtime)"
        )
    return types[0] if types else "Release"


try:
    from . import _native
except ImportError:
    _native = None

BUILD_TYPE = _select_build_type(
    os.environ.get("CUATTEST_BUILD_TYPE"), _packaged_type,
    getattr(_native, "BUILD_TYPE", None),
)
ASSERTIONS_ENABLED = BUILD_TYPE != "Release"
if ASSERTIONS_ENABLED and not __debug__:
    raise RuntimeError("Debug/AssertedRelease require Python without -O or PYTHONOPTIMIZE")
if ASSERTIONS_ENABLED and _native is not None and not getattr(_native, "ASSERTIONS_ENABLED", False):
    raise RuntimeError("rebuild the native extension to enable its requested assertions")

# SPDX-License-Identifier: Apache-2.0
"""Optional pytest plugin caching only the test crypto CUBIN between tool runs."""

import os
from pathlib import Path

from blake3 import blake3
from cuattest._build_config import BUILD_TYPE
from cuattest._build_options import cuda_options


def pytest_collection_modifyitems(items):
    for module in {item.module for item in items}:
        if module.__name__ != "test_crypto_differential":
            continue
        directory = Path(os.environ["CUATTEST_SANITIZER_CACHE"])
        directory.mkdir(parents=True, exist_ok=True)
        original = module.compile_oracle

        def cached(arch, source, *, module=module, directory=directory, original=original):
            adapters = Path(module.__file__).with_name("_crypto_oracle.cu").read_text()
            version = module.Nvrtc().version()
            # Oracle artifacts also contain the production assertions. Never
            # reuse a Release oracle when qualifying AssertedRelease/Debug.
            policy = BUILD_TYPE + "\0" + "\0".join(cuda_options(BUILD_TYPE))
            key = blake3((arch + "\0" + version + "\0" + policy + "\0" + source + "\0" + adapters).encode()).hexdigest()
            artifact = directory / (key + ".cubin")
            checksum = directory / (key + ".blake3")
            if artifact.exists() and checksum.exists():
                data = artifact.read_bytes()
                if blake3(data).hexdigest() == checksum.read_text():
                    return data
            data = original(arch, source)
            # This directory is explicitly test-only, never a production cache.
            # A missing/mismatched checksum triggers recompilation after an
            # interrupted write. Concurrent sanitizer runs should use separate
            # directories (the usual workflow runs GPU tools sequentially).
            artifact.write_bytes(data)
            checksum.write_text(blake3(data).hexdigest())
            return data

        module.compile_oracle = cached

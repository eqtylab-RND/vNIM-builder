#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify the async dispatch cutoff with independent hashes and full timings."""

import argparse
from contextlib import closing
import ctypes
import json
import os
from pathlib import Path
import statistics
import time

from blake3 import blake3
from cuattest._cuda import DeviceBuffer
from cuattest.notary import Notary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument(
        "--sizes-mib", type=int, nargs="+", default=[16, 64, 128, 256, 512, 1024, 4096]
    )
    args = parser.parse_args()
    pattern = bytes(range(256)) * 65536
    results = []
    for mode in ("standard", "async"):
        os.environ["CUATTEST_HASH_MODE"] = mode
        with closing(Notary()) as notary, notary._activate():
            function = notary._fn[
                "measure_model_fused_async_kernel"
                if mode == "async"
                else "measure_model_fused_kernel"
            ]
            attributes = {}
            for name, attribute in (
                ("registers", 4),
                ("shared_bytes", 1),
                ("local_bytes", 3),
            ):
                value = ctypes.c_int()
                notary.cu.check(
                    notary.cu.lib.cuFuncGetAttribute(
                        ctypes.byref(value), attribute, function
                    ),
                    name,
                )
                attributes[name] = value.value
            for size in args.sizes_mib:
                nbytes = size << 20
                oracle = blake3()
                with closing(DeviceBuffer(notary.cu, nbytes)) as buffer:
                    for offset in range(0, nbytes, len(pattern)):
                        data = pattern[: min(nbytes - offset, len(pattern))]
                        oracle.update(data)
                        notary.cu.htod(buffer.ptr + offset, data)
                    expected = oracle.digest()
                    seconds = []
                    for iteration in range(args.runs + 5):
                        started = time.perf_counter()
                        digest = notary.hash_dptr(buffer.ptr, nbytes)
                        elapsed = time.perf_counter() - started
                        assert digest == expected
                        if iteration >= 5:
                            seconds.append(elapsed)
                    results.append(
                        dict(
                            mode=mode,
                            nbytes=nbytes,
                            seconds=seconds,
                            attributes=attributes,
                            grid_limit=notary._async_grid_limit
                            if mode == "async"
                            else notary._fused_grid_limit,
                            notary=notary.info.as_dict(),
                        )
                    )
                    print(
                        f"{mode} {size} MiB: {statistics.median(seconds) * 1000:.3f} ms",
                        flush=True,
                    )
    with args.output.open("x") as output:
        json.dump(results, output, indent=2)


if __name__ == "__main__":
    main()

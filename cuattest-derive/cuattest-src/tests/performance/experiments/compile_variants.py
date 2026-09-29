# SPDX-License-Identifier: Apache-2.0
"""Compile isolated performance variants without creating CUDA contexts."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from cuattest.kernel import build_cubin, kernel_source
from variants import (
    NAMES,
    HASH_NAMES,
    TABLE_NAMES,
    transform,
    hash_transform,
    table_transform,
    require_baseline,
)


def compile_one(output, arch, name):
    started = time.monotonic()
    directory = Path(output) / name
    directory.mkdir(parents=True, exist_ok=False)
    source = require_baseline(kernel_source())
    for part in name.split("+"):
        if part == "baseline":
            continue
        source = (
            hash_transform(source, part)
            if part in HASH_NAMES
            else table_transform(source, part)
            if part in TABLE_NAMES
            else transform(source, part)
        )
    path = directory / "kernel.cu"
    path.write_text(source)
    binary, compiler, cubin = build_cubin(arch, directory / "cubins", source)
    result = {
        "variant": name,
        "arch": arch,
        "compiler": compiler,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "cubin_sha256": hashlib.sha256(binary).hexdigest(),
        "source": str(path.resolve()),
        "cubin_dir": str(cubin.parent.resolve()),
        "compile_seconds": time.monotonic() - started,
    }
    (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--variant", action="append")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    require_baseline(kernel_source())
    variants = args.variant or [*NAMES[1:], *HASH_NAMES]
    for variant in variants:
        assert all(
            part in (*NAMES, *HASH_NAMES, *TABLE_NAMES) for part in variant.split("+")
        ), variant
    failures = 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(compile_one, args.output, args.arch, name): name
            for name in variants
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                failures += 1
                print(json.dumps({"variant": name, "error": repr(error)}), flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())

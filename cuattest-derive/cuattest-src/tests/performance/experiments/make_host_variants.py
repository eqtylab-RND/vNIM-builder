# SPDX-License-Identifier: Apache-2.0
"""Mechanically derive isolated host scheduling trials; never change ownership."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = (args.root / "src/cuattest/_native.cpp").read_text()
    if (
        hashlib.sha256(original.encode()).hexdigest()
        != "987edac87ee283d629b16992a9554f3e8ac77718272db29a5397d2862b5b3ac7"
    ):
        raise ValueError("host trials require an isolated source snapshot of 4708698")
    for name, batch in (
        ("ipc_ungated", 0),
        ("ipc_each", 1),
        ("ipc_batch4", 4),
        ("ipc_batch16", 16),
    ):
        source = original
        lock = "std::unique_lock<std::mutex> import_lock(import_mutex);"
        if batch == 0:
            source = source.replace(
                "static std::mutex import_mutex;",
                "// Trial: allow independent driver calls to contend directly.",
            )
            source = source.replace(lock, "").replace("      import_lock.unlock();", "")
        else:
            source = source.replace(
                lock,
                """std::unique_lock<std::mutex> import_lock(import_mutex, std::defer_lock);
      unsigned imports_in_batch = 0;""",
            )
            source = source.replace(
                "if (found == by_handle.end()) {",
                """if (found == by_handle.end()) {
          if (!import_lock.owns_lock()) import_lock.lock();""",
            )
            source = source.replace(
                "          opened[index] = allocation;",
                f"""          opened[index] = allocation;
          // Scheduling only: every mapping was owned before validation, and
          // RAII still releases this gate on every exceptional path.
          if (++imports_in_batch == {batch}) {{
            import_lock.unlock(); imports_in_batch = 0;
          }}""",
            )
            source = source.replace(
                "      import_lock.unlock();\n      result = launch",
                "      if (import_lock.owns_lock()) import_lock.unlock();\n      result = launch",
            )
        assert source != original
        destination = args.output / name
        shutil.copytree(
            args.root,
            destination,
            ignore=shutil.ignore_patterns(
                "build", "__pycache__", ".pytest_cache", "*.so"
            ),
        )
        (destination / "src/cuattest/_native.cpp").write_text(source)
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=destination,
            check=True,
        )
        metadata = {
            "variant": name,
            "imports_per_batch": batch,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        }
        (destination / "trial.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()

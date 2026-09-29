"""Run every Compute Sanitizer tool over the actual experimental hash paths."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.root / "tests" / "sanitizers"))
    from run_cuda import TOOLS, tool_options

    env = dict(
        os.environ,
        CUATTEST_CACHE=str(args.output / "cache"),
        PYTHONPATH=str(args.root / "src"),
    )
    reports = []
    for tool in TOOLS:
        for backend in ("native", "fallback"):
            run_env = dict(
                env, CUATTEST_DISABLE_NATIVE_HOST=str(int(backend == "fallback"))
            )
            command = [
                "compute-sanitizer",
                *tool_options(tool, driver_only=True),
                sys.executable,
                str(Path(__file__).with_name("probe.py")),
                "--runs",
                "2",
                "--require-backend",
                backend,
            ]
            log = args.output / f"{tool}-{backend}.log"
            with log.open("x") as stream:
                result = subprocess.run(
                    command, env=run_env, stdout=stream, stderr=subprocess.STDOUT
                )
            text = log.read_text()
            clean = result.returncode == 0 and (
                "ERROR SUMMARY: 0 errors" in text
                or "RACECHECK SUMMARY: 0 hazards" in text
            )
            reports.append(
                dict(
                    tool=tool,
                    backend=backend,
                    returncode=result.returncode,
                    clean=clean,
                    command=command,
                    log=str(log),
                )
            )
            print(json.dumps(reports[-1]), flush=True)
            (args.output / "results.json").write_text(json.dumps(reports, indent=2))
            if not clean:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

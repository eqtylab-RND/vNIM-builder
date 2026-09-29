"""Queue one explicitly labeled trial for resident.py's owned server."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="resident.py output directory")
    parser.add_argument("label")
    parser.add_argument(
        "--variant", type=Path, help="compiled variant directory (omit for baseline)"
    )
    parser.add_argument("--registered", action="store_true")
    parser.add_argument("--runs", type=int, default=60)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.label):
        parser.error("label must contain only letters, digits, underscores and hyphens")
    if args.runs < 1 or args.warmup_runs < 0:
        parser.error("runs must be positive and warmups nonnegative")
    label = args.label + ("-trace" if args.trace else "")
    job = dict(
        label=label,
        registered=args.registered,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        profile=args.trace,
    )
    if args.variant:
        variant = args.variant.resolve()
        manifest = json.loads((variant / "manifest.json").read_text())
        source = variant / "kernel.cu"
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
            parser.error("variant source contradicts its compile manifest")
        job["env"] = {
            "CUATTEST_KERNEL_SRC": str(source),
            "CUATTEST_KERNEL_DIR": str(variant / "cubins"),
        }
    with (args.output / "jobs" / f"{label}.json").open("x") as output:
        json.dump(job, output, indent=2)


if __name__ == "__main__":
    main()

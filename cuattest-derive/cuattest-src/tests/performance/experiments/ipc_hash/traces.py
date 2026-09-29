"""Summarize actual request kernels, excluding the registration-only hash."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    results = []
    for trace in sorted(args.directory.glob("*.nsys-rep")):
        path = trace.with_suffix(".sqlite")
        if not path.exists():
            subprocess.run(
                [
                    "nsys",
                    "export",
                    "--type=sqlite",
                    "--output=" + str(path),
                    str(trace),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
            names = dict(db.execute("SELECT id,value FROM StringIds"))
            kernels = defaultdict(lambda: defaultdict(list))
            for row in db.execute(
                "SELECT shortName,deviceId,start,end,registersPerThread,localMemoryPerThread,gridX FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
            ):
                name, device, start, end, regs, local, grid = row
                kernels[names[name]][device].append(
                    dict(start=start, end=end, registers=regs, local=local, grid=grid)
                )
            paired = {}
            for device, receipts in kernels["attest_measured_kernel"].items():
                hashes = kernels["measure_model_fused_kernel"][device]
                paired[device] = [
                    (
                        max(
                            (h for h in hashes if h["end"] <= r["start"]),
                            key=lambda h: h["start"],
                        ),
                        r,
                    )
                    for r in receipts
                ]
            assert len({len(p) for p in paired.values()}) == 1
            count = len(next(iter(paired.values())))
            warm = [p for pairs in paired.values() for p in pairs[1:]]
            record = dict(
                trace=trace.stem, kernels={}, windows_ms=[], start_spread_ms=[], api={}
            )
            for name, index in (("hash", 0), ("receipt", 1)):
                rows = [p[index] for p in warm]
                times = [(r["end"] - r["start"]) / 1e6 for r in rows]
                record["kernels"][name] = dict(
                    count=len(times),
                    median_ms=statistics.median(times),
                    mean_ms=statistics.fmean(times),
                    registers=sorted({r["registers"] for r in rows}),
                    local=sorted({r["local"] for r in rows}),
                    grid=sorted({r["grid"] for r in rows}),
                )
            for index in range(1, count):
                starts = [pairs[index][0]["start"] for pairs in paired.values()]
                ends = [pairs[index][1]["end"] for pairs in paired.values()]
                record["windows_ms"].append((max(ends) - min(starts)) / 1e6)
                record["start_spread_ms"].append((max(starts) - min(starts)) / 1e6)
            low = max(pairs[0][1]["end"] for pairs in paired.values())
            high = max(pairs[-1][1]["end"] for pairs in paired.values())
            calls = defaultdict(list)
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            api_rows = [
                row
                for table in (
                    "CUPTI_ACTIVITY_KIND_DRIVER",
                    "CUPTI_ACTIVITY_KIND_RUNTIME",
                )
                if table in tables
                for row in db.execute(f"SELECT nameId,start,end FROM {table}")
            ]
            for name, start, end in api_rows:
                name = names[name]
                if any(
                    word in name
                    for word in (
                        "IpcOpen",
                        "IpcClose",
                        "StreamSynchronize",
                        "LaunchCooperative",
                    )
                ):
                    calls[name].append((start, end))
            for name, rows in calls.items():
                # high excludes the last request's trailing IPC closes, so
                # only open counts are an exact per-request diagnostic here.
                record["api"][name] = dict(
                    total_count=len(rows),
                    warm_kernel_window_count=sum(low <= s < high for s, _ in rows),
                    median_us=statistics.median((e - s) / 1000 for s, e in rows),
                )
            results.append(record)
            print(json.dumps(record), flush=True)
    (args.directory / "trace-summary.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

"""Compact raw benchmark JSON without dropping timing samples or outliers."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def collect(directory):
    trials = []
    groups = defaultdict(list)
    layouts = {}
    profiles = []
    for layout in ("ordinary", "packed"):
        parent = directory / layout
        layouts[layout] = json.loads((parent / "resident-layout.json").read_text())
        profiles.extend(json.loads((parent / "trace-summary.json").read_text()))
        for path in sorted(parent.glob("*results.json")):
            result = json.loads(path.read_text())
            samples = result.pop("samples")
            assert all(
                s["model_root"] == result["independent_model_root"]
                and s["tensor_count"] == result["tensor_count"]
                and s["attested_bytes"] == result["attested_bytes"]
                for s in samples
            )
            assert result.pop("placement") == layouts[layout]["placement"]
            result["layout"] = layout
            result["samples_seconds"] = [s["seconds"] for s in samples]
            trials.append(result)
            if not result.get("profile"):
                label = (
                    result["label"]
                    .removesuffix("-a")
                    .removesuffix("-b")
                    .removesuffix("-c")
                    .removesuffix("-d")
                )
                # Job labels (including repetition suffixes) are local to each
                # layout directory. Pooling by label alone silently averages
                # ordinary and packed A/B results when both use 01-baseline.
                groups[layout, label].extend(result["samples_seconds"])
    assert layouts["ordinary"]["placement"] == layouts["packed"]["placement"]
    # Store identical tensor placement once; layout-specific offsets are still
    # retained so packed alignment holes and logical spans can be audited.
    placement = layouts["ordinary"].pop("placement")
    layouts["packed"].pop("placement")
    summary = {
        f"{layout}/{label}": dict(
            samples=len(values),
            mean_ms=statistics.fmean(values) * 1000,
            median_ms=statistics.median(values) * 1000,
            stdev_ms=statistics.stdev(values) * 1000,
            min_ms=min(values) * 1000,
            max_ms=max(values) * 1000,
        )
        for (layout, label), values in sorted(groups.items())
    }
    manifests = [
        json.loads(path.read_text())
        for parent in ("remote-variants", "variants")
        for path in sorted((directory / parent).glob("*/manifest.json"))
    ]
    probes = [
        json.loads(path.read_text())
        for path in sorted((directory / "remote-probes").glob("probe-*.log"))
    ]
    fresh = json.loads((directory / "remote-probes" / "fresh.log").read_text())
    sanitizers = json.loads(
        (directory / "async-sanitizers" / "results.json").read_text()
    )
    assert len(sanitizers) == 8 and all(
        s["clean"] and s["returncode"] == 0 for s in sanitizers
    )
    return dict(
        captured_on="2026-09-09",
        hostname="Sarge-SRV2025 (probqa.com)",
        baseline_commit="1ce0fbe",
        checkpoint="Qwen/Qwen3.5-397B-A17B",
        revision="8472618112abcbd45acbcdc58436aff4233c23f7",
        excluded_prefixes=["mtp."],
        timing="full Client.sign for one-shot; token HTTP sign for registration; verification outside both",
        placement=placement,
        layouts=layouts,
        summary=summary,
        trials=trials,
        profiles=profiles,
        compile_manifests=manifests,
        blackwell_hash_probes=probes,
        registration_regression=fresh,
        async_sanitizers=sanitizers,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = collect(args.directory)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

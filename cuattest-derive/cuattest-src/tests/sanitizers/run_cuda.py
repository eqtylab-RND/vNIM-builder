# SPDX-License-Identifier: Apache-2.0
"""Run CUDA sanitizers serially, retaining separate logs and nonzero failures."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

TOOLS = ("memcheck", "initcheck", "racecheck", "synccheck")


def tool_options(tool, *, driver_only, api_errors=None, ipc_initcheck_space="shared"):
    options = ["--tool", tool, "--target-processes", "all", "--error-exitcode", "86",
               "--check-bulk-copy", "yes", "--check-tensor-ops", "yes"]
    if tool == "memcheck":
        # PyTorch/CUDA Runtime probes cuCtxGetDevice before creating a context
        # and handles INVALID_CONTEXT itself (also reproduced without cuAttest).
        # Keep every explicit API failure visible; driver-only runs additionally
        # report implicit calls. --api-errors all preserves the raw diagnostic.
        options += ["--leak-check", "full", "--padding", "32",
                    "--report-api-errors", api_errors or ("all" if driver_only else "explicit"),
                    "--check-cache-control",
                    "--track-stream-ordered-races", "all"]
        # NVIDIA explicitly disallows this optional check with CUDA Runtime
        # clients (including PyTorch). Our driver-only selftest/oracle own
        # their modules and can check explicit module lifetime as well.
        if driver_only:
            options += ["--detect-missing-module-unload"]
    elif tool == "initcheck":
        # The default checks GLOBAL only and missed descriptor padding reads
        # in shared memory. Keep BOTH spaces for driver-only workloads. NVIDIA
        # documents that initcheck cannot track initialization across IPC; a
        # raw driver-only 4096-byte IPC copy reproduces that false positive.
        # Check shared memory for IPC and leave the raw global check opt-in.
        options += ["--initcheck-address-space",
                    "all" if driver_only else ipc_initcheck_space]
    elif tool == "racecheck":
        options += ["--racecheck-report", "all", "--racecheck-detect-level", "info",
                    "--print-level", "info", "--racecheck-deadlock-timeout", "30000",
                    "--racecheck-continue-on-deadlock", "no"]
    return options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tool", action="append", choices=TOOLS)
    parser.add_argument("--suite", action="append", choices=("selftest", "crypto", "ipc", "registered", "multigpu"))
    parser.add_argument("--compute-sanitizer", default="compute-sanitizer")
    parser.add_argument("--api-errors", choices=("all", "explicit"))
    parser.add_argument("--ipc-initcheck-address-space", choices=("all", "shared"),
                        default="shared")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # KERNEL_DIR is only a lookup override; load_cubin builds cache misses in
    # CACHE/cubins. Override even an inherited writable cache so an audit never
    # publishes into the normal cache or needs a writable home directory.
    env["CUATTEST_CACHE"] = str(output)
    # An explicit prebuilt directory remains a read-only search location.
    env.setdefault("CUATTEST_KERNEL_DIR", str(output / "cubins"))
    env["CUATTEST_TEST_CRYPTO"] = env["CUATTEST_TEST_GPU"] = "1"
    env["CUATTEST_SANITIZER_CACHE"] = str(output / "oracle-cache")
    env["PYTHONPATH"] = os.pathsep.join(
        # A shared test venv may be editable-installed against another source
        # copy. Audit THIS checkout, including subprocess consumers, not that
        # stale installation while reporting the current source's results.
        [str(root / "src"), str(Path(__file__).parent.resolve()),
         env.get("PYTHONPATH", "")]
    )
    results = []
    for tool in args.tool or TOOLS:
        for suite in args.suite or ("selftest", "crypto"):
            backends = ("native", "fallback") if suite in {"selftest", "registered"} else ("both",)
            for backend in backends:
                run_env = env.copy()
                if suite == "selftest":
                    run_env["CUATTEST_DISABLE_NATIVE_HOST"] = str(int(backend == "fallback"))
                    # A missing extension must not silently run the fallback
                    # twice while the results claim to cover both backends.
                    native_import = "import cuattest._native; " if backend == "native" else ""
                    target = [sys.executable, "-c", native_import +
                              "from cuattest.cli import main; raise SystemExit(main(['selftest']))"]
                elif suite == "crypto":
                    target = [sys.executable, "-m", "pytest", "-q", "-p", "cache_oracle",
                              "tests/test_crypto_differential.py"]
                elif suite == "ipc":
                    target = [sys.executable, "-m", "pytest", "-q",
                              "tests/test_ipc_export_integration.py"]
                elif suite == "registered":
                    # Keep each instrumented backend in a fresh producer
                    # process. A mixed-backend racecheck run stalled in the
                    # second producer-stream drain after the first service's
                    # teardown. Ordinary GPU tests still cover that transition.
                    run_env["CUATTEST_TEST_REGISTERED_BACKEND"] = backend
                    target = [sys.executable, "-m", "pytest", "-q",
                              "tests/test_registered_integration.py"]
                else:
                    run_env["CUATTEST_TEST_MULTIGPU"] = "1"
                    # Deliberate invalid-driver-call tests run uninstrumented;
                    # memcheck correctly reports those injected API errors.
                    # These success paths cover both backends, fresh bytes,
                    # partial tiles, reversed ordinals and every visible GPU.
                    tests = ["-q", "tests/test_multigpu_integration.py", "-k",
                             "sharded_model_matches or parallel_repeated_requests"]
                    target = [sys.executable, "-c", (
                              "import torch, pytest\n"
                              "if torch.cuda.device_count() < 2:\n"
                              "    raise RuntimeError('multigpu sanitizer suite requires at least two GPUs')\n"
                              f"raise SystemExit(pytest.main({tests!r}))")]
                command = [args.compute_sanitizer,
                           *tool_options(tool, driver_only=suite in {"selftest", "crypto"},
                                         api_errors=args.api_errors,
                                         ipc_initcheck_space=args.ipc_initcheck_address_space),
                           *target]
                logfile = output / f"{suite}-{backend}-{tool}.log"
                print(f"{shlex.join(command)}\n  -> {logfile}", flush=True)
                with logfile.open("w") as stream:
                    result = subprocess.run(command, cwd=root, env=run_env,
                                            stdout=stream, stderr=subprocess.STDOUT,
                                            check=False)
                results.append({"suite": suite, "backend": backend, "tool": tool,
                                "returncode": result.returncode, "log": str(logfile)})
                print(f"  exit={result.returncode}", flush=True)
                # Persist progress even if a later test is interrupted. Keep
                # running other tools after a failure, but NEVER hide its exit.
                (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(result["returncode"] != 0 for result in results))


if __name__ == "__main__":
    raise SystemExit(main())

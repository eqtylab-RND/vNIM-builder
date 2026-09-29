# SPDX-License-Identifier: Apache-2.0
"""Local benchmark server with Nsight capture starting AFTER CUDA initialization.

Launch under nsys --capture-range=cudaProfilerApi --capture-range-end=stop.
This is a profiling helper, not a production service or a latency benchmark.
"""

import argparse
import ctypes
import json

from cuattest.multigpu import MultiGpuNotary
from cuattest.server import _NotaryHTTPServer, make_handler


def profiler_call(notary, name):
    # Nsight's process-wide capture needs one current context for the driver
    # trigger. Preserve the embedding thread's stack just like normal requests.
    first = next(iter(notary.notaries.values()))
    with first._activate():
        function = getattr(first.cu.lib, name)
        function.argtypes, function.restype = [], ctypes.c_int
        first.cu.check(function(), name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--max-request-bytes", type=int, default=1 << 40)
    parser.add_argument("--max-request-tiles", type=int, default=8388608)
    args = parser.parse_args(argv)
    notary = MultiGpuNotary(
        max_request_bytes=args.max_request_bytes,
        max_request_tiles=args.max_request_tiles,
    )
    server = None
    capturing = False
    try:
        server = _NotaryHTTPServer(("127.0.0.1", args.port), make_handler(notary))
        # On the audited driver/Nsight combination, tracing initialization of
        # multiple modules aborted inside the instrumented process. Deferred
        # capture covers complete requests (including IPC), with no startup
        # kernels. Never mix profiler samples into ordinary latency averages.
        profiler_call(notary, "cuProfilerStart")
        capturing = True
        print(json.dumps({"port": server.server_port, "info": notary.info.as_dict()}),
              flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if server is not None:
                server.server_close()
        finally:
            try:
                if capturing:
                    profiler_call(notary, "cuProfilerStop")
            finally:
                notary.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Loopback-only, owned-process registration prototype. NOT a supported API.

Only /experiment/unregister (or confirmed death of this exact process) permits
producer storage release. In particular, sign response headers do NOT. Never
use Client.sign or its one-shot completion classification with these routes.
Any failure terminates this single-client process and its CUDA contexts.
"""

import argparse
import json
import os
from pathlib import Path
import secrets
import sys


class RegisteredRunner:
    def __init__(self, notary, tensors):
        self.delegate = notary._native_runner
        if self.delegate is None:
            raise RuntimeError("registration experiment requires the native runner")
        self.cu = notary.cu
        self.inputs = [(t.raw_handle(), t.nbytes, t.seg_off, t.t_off) for t in tensors]
        self.mappings = []
        self.spans = []
        by_handle = {}
        # The owning server tears down the context on *every* failure, even an
        # interrupt between driver return and publication in this Python ledger.
        for tensor, (raw, _, _, _) in zip(tensors, self.inputs):
            if raw not in by_handle:
                mapped = self.cu.ipc_open(raw)
                self.mappings.append(mapped)
                if self.cu.pointer_device(mapped) != notary.device_ordinal:
                    raise RuntimeError("registered allocation belongs to another GPU")
                base, size = self.cu.address_range(mapped)
                by_handle[raw] = (mapped, base, size)
            self.spans.append((tensor.pointer_in(*by_handle[raw]), tensor.nbytes))

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def run_ipc(self, inputs, timestamp, model):
        if inputs != self.inputs:
            raise RuntimeError("registration cannot substitute handles or spans")
        # Fresh kernels and same private one-shot signing handoff, every call.
        # The prototype producer synchronizes writes before submitting HTTP.
        return self.delegate.run_spans(self.spans, timestamp, model)

    def unregister(self):
        # Requests serialize and run_spans confirms stream completion. If any
        # close fails, no release ACK is sent; context/process teardown owns it.
        for mapped in self.mappings:
            self.cu.ipc_close(mapped)
        self.mappings.clear()


def registered_handler(notary):
    from cuattest.server import make_handler

    token = None
    tensors = None
    parent = make_handler(notary)

    class Handler(parent):
        def do_POST(self):
            nonlocal token, tensors
            try:
                body = self._read_json()
                if self.path == "/experiment/register":
                    if token is not None:
                        raise ValueError("one active registration per owned server")
                    tensors = self._tensor_refs(body)
                    # Exercise all ordinary aggregate/span/owner checks before
                    # retaining any mapping. Registration cost is reported on
                    # its own, including this deliberately conservative hash.
                    notary.measure(tensors)
                    groups = notary._partition(tensors)
                    count = 0
                    for uid, positions in groups:
                        owner = notary.notaries[uid]
                        with owner._activate():
                            runner = RegisteredRunner(
                                owner, [tensors[i] for i in positions]
                            )
                            owner._native_runner = runner
                            count += len(runner.mappings)
                    token = secrets.token_hex(32)
                    return self._send(200, {"token": token, "unique_handles": count})
                if token is None or not secrets.compare_digest(
                    body.get("token", ""), token
                ):
                    raise ValueError("unknown registration generation")
                if self.path == "/experiment/sign":
                    return self._send(200, notary.sign(tensors, body["model"]))
                if self.path == "/experiment/unregister":
                    for owner in notary.notaries.values():
                        with owner._activate():
                            runner = owner._native_runner
                            if isinstance(runner, RegisteredRunner):
                                runner.unregister()
                                owner._native_runner = runner.delegate
                    token, tensors = None, None
                    return self._send(200, {"released": True})
                # No v1 measure/sign escape hatch while persistent maps exist.
                raise ValueError("unsupported experiment route")
            except BaseException as error:
                self.close_connection = True
                self.server.cuattest_fatal_error = RuntimeError(str(error))
                # Silence is intentional. No ambiguous error response can be
                # mistaken for confirmation that imported storage is released.

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("ready", type=Path)
    parser.add_argument("--registered", action="store_true")
    args = parser.parse_args()
    tracing = args.ready.stem.endswith("-trace-ready")
    if tracing and os.environ.get("CUATTEST_TRACE_CHILD") != "1":
        os.environ["CUATTEST_TRACE_CHILD"] = "1"
        report = args.ready.with_name(args.ready.stem.removesuffix("-ready"))
        os.execvp(
            "nsys",
            [
                "nsys",
                "profile",
                "--trace=cuda,nvtx",
                "--sample=none",
                "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--output=" + str(report),
                sys.executable,
                __file__,
                *sys.argv[1:],
            ],
        )
    sys.path.insert(0, str(args.root / "src"))
    from cuattest.multigpu import MultiGpuNotary
    from cuattest.server import _NotaryHTTPServer, make_handler

    notary = MultiGpuNotary(max_request_bytes=1 << 40, max_request_tiles=8388608)
    server = None
    try:
        if tracing:
            first = next(iter(notary.notaries.values()))
            with first._activate():
                first.cu.check(first.cu.lib.cuProfilerStart(), "cuProfilerStart")
        handler = (
            registered_handler(notary) if args.registered else make_handler(notary)
        )
        server = _NotaryHTTPServer(("127.0.0.1", 0), handler)
        with args.ready.open("x") as output:
            json.dump({"pid": os.getpid(), "port": server.server_port}, output)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        if tracing:
            with first._activate():
                first.cu.check(first.cu.lib.cuProfilerStop(), "cuProfilerStop")
        notary.close()


if __name__ == "__main__":
    main()

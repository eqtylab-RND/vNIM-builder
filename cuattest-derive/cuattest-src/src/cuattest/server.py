# SPDX-License-Identifier: Apache-2.0
"""HTTP front end for one notary session.

HTTP requests deliberately queue: a notary session's mutable workspaces and
IPC ownership cannot serve overlapping requests. MultiGpuNotary may execute
one request's independent GPU shards concurrently, and joins them before ACK.

    GET  /v1/info      the session identity and the code that is running
    POST /v1/measure   {"tensors": [{handle, nbytes, seg_off, t_off, device}, ...]}
    POST /v1/sign      {"tensors": [...], "model": "..."} (measure + sign atomically)
    GET  /healthz
"""

from __future__ import annotations

import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from .notary import (
    IpcCleanupUncertainError,
    IpcSessionAbortedError,
    Notary,
    NotaryError,
    TensorRef,
)

logger = logging.getLogger("cuattest")
MAX_BODY = 64 * 1024 * 1024  # a very large model still fits its handle list
HEADER_READ_TIMEOUT = 10.0
BODY_READ_TIMEOUT = 30.0


class RequestBodyTimeout(NotaryError):
    """The peer did not finish its declared body within the deadline."""


class _HeaderDeadlineReader:
    """Buffered request reader with one wall-clock deadline per header block.

    A socket timeout alone is an inactivity timeout: every trickled byte starts
    it again. ``readline`` instead performs at most one underlying read at a
    time and recomputes the remaining monotonic budget after every chunk.
    Bytes read past a newline are retained for the next header or request body.
    """

    _CHUNK = 8192

    def __init__(self, reader, connection) -> None:
        self._reader = reader
        self._connection = connection
        self._pending = bytearray()
        self._deadline: float | None = None

    def begin_header(self) -> None:
        self._deadline = time.monotonic() + HEADER_READ_TIMEOUT

    def finish_header(self) -> None:
        self._deadline = None

    def _apply_remaining_timeout(self) -> None:
        if self._deadline is None:
            return
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("request-line/header deadline expired")
        self._connection.settimeout(remaining)

    def _complete_line_end(self, limit: int) -> int | None:
        available = len(self._pending)
        considered = available if limit < 0 else min(available, limit)
        newline = self._pending.find(b"\n", 0, considered)
        if newline >= 0:
            return newline + 1
        if limit >= 0 and available >= limit:
            return limit
        return None

    def readline(self, limit: int = -1) -> bytes:
        if limit == 0:
            return b""
        while True:
            # Check the wall clock even when a previous read filled multiple
            # header lines. The deadline covers parsing the complete block,
            # not merely time spent waiting inside recv().
            self._apply_remaining_timeout()
            end = self._complete_line_end(limit)
            if end is not None:
                line = bytes(self._pending[:end])
                del self._pending[:end]
                return line

            read_size = self._CHUNK
            if limit >= 0:
                read_size = min(read_size, limit - len(self._pending))
            chunk = self._reader.read1(read_size)
            if chunk:
                self._pending.extend(chunk)
                continue

            end = len(self._pending) if limit < 0 else min(len(self._pending), limit)
            line = bytes(self._pending[:end])
            del self._pending[:end]
            return line

    def read1(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if self._pending:
            end = len(self._pending) if size < 0 else min(len(self._pending), size)
            chunk = bytes(self._pending[:end])
            del self._pending[:end]
            return chunk
        return self._reader.read1(size)

    def read(self, size: int = -1) -> bytes:
        if not self._pending:
            return self._reader.read(size)
        if size < 0:
            prefix = bytes(self._pending)
            self._pending.clear()
            return prefix + self._reader.read()
        prefix = self.read1(size)
        if len(prefix) == size:
            return prefix
        return prefix + self._reader.read(size - len(prefix))

    def __getattr__(self, name):
        return getattr(self._reader, name)


class _NotaryHTTPServer(HTTPServer):
    """Stop serving as soon as a fatal CUDA session error is recorded."""

    def __init__(self, *args, **kwargs) -> None:
        self.cuattest_fatal_error: Exception | None = None
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, client_address = super().get_request()
        # BaseHTTPRequestHandler parses the request line and every header
        # before do_POST can apply the body deadline. This initial timeout also
        # covers setup; Handler's deadline reader then enforces one decreasing
        # wall-clock budget across the complete request-line/header block.
        request.settimeout(HEADER_READ_TIMEOUT)
        return request, client_address

    def service_actions(self) -> None:
        # serve_forever calls this only after the request handler has returned
        # and flushed its response (if an acknowledgement was safe to send).
        if self.cuattest_fatal_error is not None:
            raise self.cuattest_fatal_error


def make_handler(notary: Notary):
    from ._registration import RegistrationManager
    registrations = RegistrationManager(notary)

    class Handler(BaseHTTPRequestHandler):
        server_version = "cuattest"

        def setup(self) -> None:
            super().setup()
            self.rfile = _HeaderDeadlineReader(self.rfile, self.connection)

        def handle_one_request(self) -> None:
            # Start before BaseHTTPRequestHandler reads the request line. The
            # finally also covers an overlong line or timeout before headers.
            self.rfile.begin_header()
            try:
                super().handle_one_request()
            finally:
                self.rfile.finish_header()

        def parse_request(self) -> bool:
            try:
                return super().parse_request()
            finally:
                # Body reads have their own independent total deadline.
                self.rfile.finish_header()

        # ── plumbing ─────────────────────────────────────────────────────────
        def _send(self, code: int, payload, raw: bool = False) -> None:
            # Client response headers are the CUDA-IPC completion
            # acknowledgement. POST handlers call this only after every map
            # closed or successful context destruction released all of them.
            body = payload.encode() if raw else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except (TypeError, ValueError) as e:
                raise NotaryError("Content-Length must be an integer") from e
            if length <= 0:
                raise NotaryError("empty request body")
            if length > MAX_BODY:
                raise NotaryError(f"request body of {length} bytes exceeds the limit")
            # HTTPServer is deliberately single-threaded to keep the CUDA
            # context on its owner thread. A body deadline is therefore a
            # service-wide availability boundary, not merely client hygiene.
            deadline = time.monotonic() + BODY_READ_TIMEOUT
            chunks = []
            remaining = length
            try:
                while remaining:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        raise TimeoutError("request body deadline expired")
                    # BufferedReader.read1 consumes bytes already buffered by
                    # the header parser, then performs at most one socket read.
                    # Recomputing the timeout makes this a total wall-clock
                    # deadline; a peer cannot stay alive by trickling a byte
                    # shortly before each inactivity timeout.
                    self.connection.settimeout(timeout)
                    chunk = self.rfile.read1(min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
            except TimeoutError as e:
                raise RequestBodyTimeout(
                    f"request body was not received within {BODY_READ_TIMEOUT:g} seconds"
                ) from e
            raw = b"".join(chunks)
            if len(raw) != length:
                raise NotaryError(
                    f"request body ended after {len(raw)} of {length} declared bytes"
                )
            try:
                return json.loads(raw)
            except ValueError as e:
                raise NotaryError(f"body is not valid JSON: {e}") from e

        def _tensor_refs(self, body) -> list[TensorRef]:
            if not isinstance(body, dict):
                raise NotaryError("request body must be a JSON object")
            raw = body.get("tensors")
            if not isinstance(raw, list):
                raise NotaryError('"tensors" must be a list')
            max_tensors = getattr(notary, "max_request_tensors", None)
            if max_tensors is not None and len(raw) > max_tensors:
                # Reject oversized lists before constructing one TensorRef per
                # entry. Notary._measure_request repeats the authoritative
                # check before any IPC handle can be imported.
                raise NotaryError(
                    f"request has {len(raw)} tensors; limit is {max_tensors}"
                )
            return [TensorRef.from_dict(t) for t in raw]

        def log_message(self, fmt, *args):
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _stop_server(self, error: Exception) -> None:
            # Do not accept a pipelined request on this connection after the
            # CUDA session has been destroyed or become untrustworthy.
            self.close_connection = True
            server = getattr(self, "server", None)
            if server is not None:
                server.cuattest_fatal_error = error

        # ── routes ───────────────────────────────────────────────────────────
        def do_GET(self):
            if self.path == "/healthz":
                return self._send(200, {"status": "ok"})
            if self.path == "/v1/info":
                return self._send(200, notary.info.as_dict())
            self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            try:
                if self.path.startswith("/v1/registrations/"):
                    # These headers are NOT the one-shot allocation-release
                    # contract. Only a structured, matching close ACK releases
                    # a persistent producer; sign retains every mapping.
                    body = self._read_json()
                    if self.path == "/v1/registrations/open":
                        return self._send(200, registrations.reserve())
                    if self.path == "/v1/registrations/import":
                        return self._send(200, registrations.import_tensors(body))
                    if self.path == "/v1/registrations/sign":
                        return self._send(200, registrations.sign(body))
                    if self.path == "/v1/registrations/close":
                        return self._send(200, registrations.close(body))
                    return self._send(404, {"error": "unknown registration operation"})
                if self.path == "/v1/measure":
                    body = self._read_json()
                    refs = self._tensor_refs(body)
                    result = notary.measure(refs)
                    logger.info(
                        "measured %d tensors in %.3fs -> %s",
                        result.tensor_count,
                        result.seconds,
                        result.vram_cid,
                    )
                    return self._send(200, result.as_dict())

                if self.path == "/v1/sign":
                    body = self._read_json()
                    refs = self._tensor_refs(body)
                    model = body.get("model")
                    if not isinstance(model, str):
                        raise NotaryError('"model" must be a string')
                    # One call maps, measures and signs before this handler can
                    # service another client. The timestamp is captured by the
                    # notary; no mutable "last measurement" is addressable.
                    receipt = notary.sign(refs, model)
                    logger.info(
                        "measured and signed %d tensors in %.3fs -> %s",
                        receipt["tensor_count"],
                        receipt["seconds"],
                        receipt["vram_cid"],
                    )
                    return self._send(200, receipt)

                self._send(404, {"error": f"no route {self.path}"})
            except IpcCleanupUncertainError as e:
                # If both unmap and context destruction failed, any response
                # would falsely authorize producer memory reuse. Close this
                # connection without headers and stop the poisoned session;
                # Client turns the resulting disconnect into a quarantined
                # NotaryRequestUncertainError.
                logger.critical("unsafe CUDA IPC cleanup; withholding response: %s", e)
                self._stop_server(e)
            except IpcSessionAbortedError as e:
                # Successful context destruction guarantees the mapping is
                # gone. A 500 is therefore safe to use as an acknowledgement,
                # but the destroyed notary session cannot serve again.
                self._stop_server(e)
                self._send(500, {"error": str(e)})
            except RequestBodyTimeout as e:
                self._send(408, {"error": str(e)})
            except NotaryError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:
                if getattr(notary, "ipc_cleanup_required", False):
                    # Either backend can lose its specialized cleanup error
                    # to an allocation failure or another unwinding exception.
                    # Notary-owned quarantine survives native runner teardown.
                    # Withhold headers: the client must quarantine its lease.
                    logger.critical(
                        "unclassified error with unsafe CUDA IPC cleanup; "
                        "withholding response: %s",
                        e,
                    )
                    self._stop_server(e)
                    return
                logger.exception("unhandled error on %s", self.path)
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def serve(notary: Notary, host: str = "127.0.0.1", port: int = 8077) -> None:
    httpd = _NotaryHTTPServer((host, port), make_handler(notary))
    logger.info("cuattest listening on %s:%d", host, port)
    if hasattr(notary.info, "devices"):
        for device in notary.info.devices:
            logger.info(
                "  cuda:%s %s %s %s (%s)",
                device["device_ordinal"],
                device["device_uuid"],
                device["gpu_did"],
                device["arch"],
                device["host_backend"],
            )
    else:
        logger.info("  did    %s", notary.info.gpu_did)
        logger.info("  device %s (%s)", notary.info.device, notary.info.arch)
        logger.info("  host   %s", notary.info.host_backend)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down; the session key dies with this process")
    finally:
        httpd.server_close()

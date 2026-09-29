# SPDX-License-Identifier: Apache-2.0
"""HTTP protocol safety without opening a real listening socket."""

import json
from types import SimpleNamespace

import pytest

from cuattest import server as server_module
from cuattest.notary import IpcCleanupUncertainError, IpcSessionAbortedError
from cuattest.server import (
    BODY_READ_TIMEOUT,
    HEADER_READ_TIMEOUT,
    RequestBodyTimeout,
    _HeaderDeadlineReader,
    _NotaryHTTPServer,
    make_handler,
)

RAW = bytes(range(64)).hex()


def test_accepted_socket_gets_a_deadline_before_header_parsing(monkeypatch):
    accepted = Connection()
    monkeypatch.setattr(
        server_module.HTTPServer,
        "get_request",
        lambda self: (accepted, ("127.0.0.1", 12345)),
    )
    server = object.__new__(_NotaryHTTPServer)

    request, address = server.get_request()

    assert request is accepted
    assert address == ("127.0.0.1", 12345)
    assert accepted.timeouts == [HEADER_READ_TIMEOUT]


class Connection:
    def __init__(self):
        self.timeouts = []

    def settimeout(self, seconds):
        self.timeouts.append(seconds)


class ChunkReader:
    def __init__(self, *chunks):
        self.chunks = iter(chunks)
        self.calls = 0

    def read1(self, length):
        self.calls += 1
        return next(self.chunks, b"")


def test_header_timeout_is_total_not_reset_by_trickled_bytes(monkeypatch):
    # One byte arrives before each socket inactivity timeout, but the absolute
    # deadline still expires after the second byte.
    moments = iter((100.0, 100.0, 106.0, 111.0))
    monkeypatch.setattr(server_module.time, "monotonic", lambda: next(moments))
    connection = Connection()
    raw = ChunkReader(b"G", b"E", b"T")
    reader = _HeaderDeadlineReader(raw, connection)

    reader.begin_header()
    with pytest.raises(TimeoutError, match="header deadline expired"):
        reader.readline()

    assert raw.calls == 2
    assert connection.timeouts == [pytest.approx(10.0), pytest.approx(4.0)]


def test_header_reader_preserves_bytes_read_ahead_into_the_body(monkeypatch):
    monkeypatch.setattr(server_module.time, "monotonic", lambda: 100.0)
    raw = ChunkReader(b"POST / HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}")
    reader = _HeaderDeadlineReader(raw, Connection())

    reader.begin_header()
    assert reader.readline() == b"POST / HTTP/1.1\r\n"
    assert reader.readline() == b"Content-Length: 2\r\n"
    assert reader.readline() == b"\r\n"
    reader.finish_header()

    assert reader.read1(2) == b"{}"
    assert raw.calls == 1


def test_handler_installs_and_starts_the_total_header_deadline(monkeypatch):
    connection = Connection()
    raw = ChunkReader()

    def base_setup(handler):
        handler.connection = connection
        handler.rfile = raw

    observed = []
    monkeypatch.setattr(server_module.BaseHTTPRequestHandler, "setup", base_setup)
    monkeypatch.setattr(
        server_module.BaseHTTPRequestHandler,
        "handle_one_request",
        lambda handler: observed.append(handler.rfile._deadline),
    )
    monkeypatch.setattr(server_module.time, "monotonic", lambda: 100.0)
    Handler = make_handler(SimpleNamespace())
    handler = object.__new__(Handler)

    handler.setup()
    assert isinstance(handler.rfile, _HeaderDeadlineReader)
    handler.handle_one_request()

    assert observed == [100.0 + HEADER_READ_TIMEOUT]
    assert handler.rfile._deadline is None


class TimeoutReader:
    def read1(self, length):
        raise TimeoutError("peer stalled")


def test_incomplete_body_has_a_service_wide_deadline():
    Handler = make_handler(SimpleNamespace())
    handler = object.__new__(Handler)
    handler.headers = {"Content-Length": "100"}
    handler.connection = Connection()
    handler.rfile = TimeoutReader()

    with pytest.raises(RequestBodyTimeout, match="not received"):
        handler._read_json()

    assert handler.connection.timeouts == [pytest.approx(BODY_READ_TIMEOUT, abs=0.01)]


class TrickleReader:
    def __init__(self):
        self.calls = 0

    def read1(self, length):
        self.calls += 1
        return b"x"


def test_body_timeout_is_total_not_reset_by_trickled_bytes(monkeypatch):
    moments = iter((100.0, 100.0, 131.0))
    monkeypatch.setattr(server_module.time, "monotonic", lambda: next(moments))
    Handler = make_handler(SimpleNamespace())
    handler = object.__new__(Handler)
    handler.headers = {"Content-Length": "2"}
    handler.connection = Connection()
    handler.rfile = TrickleReader()

    with pytest.raises(RequestBodyTimeout, match="not received"):
        handler._read_json()

    assert handler.rfile.calls == 1
    assert handler.connection.timeouts == [pytest.approx(BODY_READ_TIMEOUT, abs=0.01)]


def run_post(notary, path, body):
    Handler = make_handler(notary)
    handler = object.__new__(Handler)
    handler.path = path
    handler._read_json = lambda: body
    sent = []
    handler._send = lambda code, payload, raw=False: sent.append((code, payload))
    handler.do_POST()
    assert len(sent) == 1
    return sent[0]


def test_sign_is_one_atomic_measure_and_sign_request():
    calls = []
    receipt = {"ok": True, "tensor_count": 1, "seconds": 0.1, "vram_cid": "cid"}
    notary = SimpleNamespace(
        sign=lambda refs, model: calls.append((refs, model)) or receipt
    )
    body = {
        "model": "model/a",
        "tensors": [
            {"handle": RAW, "nbytes": 8, "seg_off": 0, "t_off": 0, "device": 0}
        ],
    }

    code, payload = run_post(notary, "/v1/sign", body)

    assert code == 200 and payload == receipt
    assert calls[0][1] == "model/a"
    assert calls[0][0][0].nbytes == 8


def test_legacy_sign_cannot_reuse_whichever_measurement_ran_last():
    calls = []
    notary = SimpleNamespace(sign=lambda refs, model: calls.append((refs, model)))

    code, payload = run_post(
        notary, "/v1/sign", {"ts": "2099-01-01T00:00:00Z", "model": "model/a"}
    )

    assert code == 400
    assert '"tensors" must be a list' in payload["error"]
    assert calls == []


def test_malformed_tensor_error_does_not_echo_a_large_request_field():
    huge_handle = "feedface" * (1024 * 128)
    notary = SimpleNamespace(
        measure=lambda refs: pytest.fail("invalid references must not reach CUDA")
    )

    code, payload = run_post(
        notary,
        "/v1/measure",
        {
            "tensors": [
                {
                    "handle": huge_handle,
                    "nbytes": "not-an-integer",
                    "device": 0,
                }
            ]
        },
    )

    assert code == 400
    assert payload == {"error": "bad tensor reference: nbytes must be an integer"}
    assert len(json.dumps(payload)) < 128


def test_tensor_count_limit_is_enforced_before_reference_construction(monkeypatch):
    notary = SimpleNamespace(
        max_request_tensors=1,
        measure=lambda refs: pytest.fail("over-limit refs reached the notary"),
    )
    monkeypatch.setattr(
        server_module.TensorRef,
        "from_dict",
        lambda value: pytest.fail("over-limit refs were constructed"),
    )

    code, payload = run_post(
        notary,
        "/v1/measure",
        {"tensors": [{"anything": 1}, {"anything": 2}]},
    )

    assert code == 400
    assert payload == {"error": "request has 2 tensors; limit is 1"}


def fatal_post(error):
    def fail(refs):
        raise error

    notary = SimpleNamespace(measure=fail)
    Handler = make_handler(notary)
    handler = object.__new__(Handler)
    handler.path = "/v1/measure"
    handler._read_json = lambda: {
        "tensors": [{"handle": RAW, "nbytes": 8, "seg_off": 0, "t_off": 0, "device": 0}]
    }
    sent = []
    handler._send = lambda code, payload, raw=False: sent.append((code, payload))
    handler.server = SimpleNamespace(cuattest_fatal_error=None)
    handler.close_connection = False
    handler.do_POST()
    return handler, sent


def test_destroyed_context_is_acknowledged_then_server_stops():
    error = IpcSessionAbortedError("context destroyed")

    handler, sent = fatal_post(error)

    assert sent == [(500, {"error": "context destroyed"})]
    assert handler.close_connection is True
    assert handler.server.cuattest_fatal_error is error


def test_uncertain_ipc_cleanup_stops_without_response_headers():
    error = IpcCleanupUncertainError("mapping may remain open")

    handler, sent = fatal_post(error)

    assert sent == []
    assert handler.close_connection is True
    assert handler.server.cuattest_fatal_error is error


def test_unclassified_error_with_cleanup_sentinel_withholds_headers():
    error = MemoryError("specialized exception allocation failed")

    def fail(refs):
        raise error

    notary = SimpleNamespace(measure=fail, ipc_cleanup_required=True)
    Handler = make_handler(notary)
    handler = object.__new__(Handler)
    handler.path = "/v1/measure"
    handler._read_json = lambda: {
        "tensors": [{"handle": RAW, "nbytes": 8, "seg_off": 0, "t_off": 0, "device": 0}]
    }
    sent = []
    handler._send = lambda code, payload, raw=False: sent.append((code, payload))
    handler.server = SimpleNamespace(cuattest_fatal_error=None)
    handler.close_connection = False

    handler.do_POST()

    assert sent == []
    assert handler.close_connection is True
    assert handler.server.cuattest_fatal_error is error


def test_fatal_session_error_breaks_the_server_loop():
    error = IpcCleanupUncertainError("fatal")
    server = object.__new__(_NotaryHTTPServer)
    server.cuattest_fatal_error = error

    with pytest.raises(IpcCleanupUncertainError) as stopped:
        server.service_actions()

    assert stopped.value is error

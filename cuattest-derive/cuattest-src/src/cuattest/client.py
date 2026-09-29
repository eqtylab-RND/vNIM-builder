# SPDX-License-Identifier: Apache-2.0
"""Talk to a running `cuattest serve`, using only the standard library."""

from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request

from .ipc import (
    _complete_ipc_refs,
    assert_ipc_refs_immutable,
    claim_ipc_refs,
    quarantine_ipc_refs,
    wire_refs,
)


class NotaryClientError(RuntimeError):
    """The notary refused, failed, or could not be reached."""

    def __init__(self, message: str, *, ipc_completion_known: bool = False) -> None:
        super().__init__(message)
        # True means producer cleanup is safe: either no request bytes could
        # have reached the notary, or response headers acknowledge that every
        # imported mapping is gone. Other transport failures are ambiguous.
        self.ipc_completion_known = ipc_completion_known


class NotaryRequestUncertainError(NotaryClientError):
    """The request may still be executing after its connection was lost."""


def _request_was_definitely_unsent(error: Exception) -> bool:
    """Whether a transport failure proves that no HTTP bytes were sent."""
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    # Do not classify generic errno values such as EHOSTUNREACH/ENETUNREACH
    # here: urllib can surface them while sending or awaiting headers, after
    # the notary has already opened the CUDA handles. These three concrete
    # failures happen while resolving or establishing the connection itself.
    return isinstance(
        reason, (ConnectionRefusedError, socket.gaierror, http.client.InvalidURL)
    )


class Client:
    def __init__(
        self, url: str = "http://127.0.0.1:8077", timeout: float = 600.0
    ) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        # CUDA IPC works only between processes on the same host. Environment
        # proxy settings would insert an untrusted acknowledgement boundary: a
        # proxy-generated 502/504 does not prove that the backend closed its
        # mapping. An explicit empty ProxyHandler forces every request directly
        # to the configured notary address.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _rpc(self, path: str, body=None):
        header_ack = not path.startswith("/v1/registrations/")
        try:
            req = urllib.request.Request(self.url + path)
        except ValueError as e:
            raise NotaryClientError(
                f"{path}: invalid notary URL: {e}", ipc_completion_known=True
            ) from e
        if body is not None:
            try:
                req.data = json.dumps(body).encode()
            except (TypeError, ValueError) as e:
                raise NotaryClientError(
                    f"{path}: request is not JSON encodable: {e}",
                    ipc_completion_known=True,
                ) from e
            req.add_header("Content-Type", "application/json")
        try:
            response = self._opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            # Only ONE-SHOT headers acknowledge mapping retirement. Registered
            # storage survives sign/errors; only its matching close body can
            # authorize release, even when this HTTP error was definitely sent.
            try:
                raw = e.read().decode(errors="replace")
            except Exception:  # noqa: BLE001 - headers already establish completion
                raw = str(e)
            try:
                raw = json.loads(raw).get("error", raw)
            except (AttributeError, ValueError):
                pass
            raise NotaryClientError(f"{path}: {raw}", ipc_completion_known=header_ack) from e
        except Exception as e:
            reason = e.reason if isinstance(e, urllib.error.URLError) else e
            if _request_was_definitely_unsent(e):
                # Connection refusal, name-resolution failure, and invalid
                # destinations happen before urllib can transmit the body.
                raise NotaryClientError(
                    f"cannot reach the notary at {self.url} ({reason}); request was not sent",
                    ipc_completion_known=True,
                ) from e
            # Timeouts, resets, malformed status lines, and other protocol
            # errors can happen after the server imported the CUDA handles.
            # Normalize all of them to the documented recovery exception.
            raise NotaryRequestUncertainError(
                f"{path}: transport failed ({reason}); request completion is unknown, "
                "so any CUDA IPC leases remain quarantined. The service must run on this "
                "host, and a container needs --network host plus --ipc host."
            ) from e

        try:
            with response as r:
                parsed = json.load(r)
        except Exception as e:
            # One-shot headers survive a truncated body as a release ACK.
            # Persistent registration headers deliberately do not: its close
            # body must identify the exact service session and ticket.
            raise NotaryClientError(
                f"{path}: invalid or incomplete JSON response: {e}",
                ipc_completion_known=header_ack,
            ) from e
        if isinstance(parsed, dict) and "error" in parsed:
            raise NotaryClientError(
                f"{path}: {parsed['error']}", ipc_completion_known=header_ack
            )
        return parsed

    def info(self) -> dict:
        return self._rpc("/v1/info")

    def register_model(self, model, *, streams=None):
        """Pin and import a model once; the returned context manager rehashes it.

        Registered tensors must be contiguous, resolved CUDA views. Structure
        and storage cannot change until close(); bytes may change BETWEEN
        completed signs. All writers must precede the declared streams' handoff.
        """
        from .registered import RegisteredModel
        return RegisteredModel.create(self, model.state_dict, streams=streams)

    def register_tensors(self, tensors, *, streams=None):
        """Register a named tensor mapping, retaining and checking its storage."""
        from .registered import RegisteredModel
        return RegisteredModel.create(self, lambda: dict(tensors), streams=streams)

    def registrations(self):
        """Recover live/quarantined handles, including interrupted registrations."""
        from .registered import live_registrations
        return live_registrations(self.url)

    def measure(self, refs: list[dict]) -> dict:
        return self._refs_rpc("/v1/measure", refs)

    def sign(self, refs: list[dict], model: str) -> dict:
        """Atomically measure ``refs`` and return signed evidence."""
        return self._refs_rpc("/v1/sign", refs, {"model": model})

    def attest(self, refs: list[dict], model: str) -> dict:
        """Descriptive alias for :meth:`sign`."""
        return self.sign(refs, model)

    def _refs_rpc(self, path: str, refs: list[dict], extra: dict | None = None) -> dict:
        # CUDA IPC refs are one-shot. Claim them before any potentially
        # blocking work so concurrent or sequential reuse cannot outlive the
        # storage lease and mutation guard associated with the first request.
        refs = claim_ipc_refs(refs)
        # Build the wire payload before starting I/O. A local preparation error
        # is known not to have exposed the handles and can release normally.
        try:
            body = {"tensors": wire_refs(refs)}
            if extra:
                body.update(extra)
        except BaseException:
            try:
                # The claim is active, but no bytes were sent. Only this
                # owning Client path may retire an in-flight lease.
                _complete_ipc_refs(refs)
            except Exception:  # noqa: BLE001,S110 - retain and re-raise the request error
                pass
            raise

        try:
            result = self._rpc(path, body)
        except NotaryClientError as e:
            if e.ipc_completion_known:
                # A concrete server response is the consumer acknowledgement.
                try:
                    _complete_ipc_refs(refs)
                except Exception:  # noqa: BLE001,S110 - retain the request error
                    pass
            else:
                quarantine_ipc_refs(refs)
            raise
        except Exception as e:
            # _rpc normalizes transport errors, but keep this boundary safe for
            # alternate transports/subclasses too: ordinary ambiguous failures
            # must remain catchable through the documented exception type.
            quarantine_ipc_refs(refs)
            raise NotaryRequestUncertainError(
                f"{path}: request completion is unknown; CUDA IPC leases remain quarantined"
            ) from e
        except BaseException:
            # Preserve process-control exceptions while still preventing an
            # unsafe decrement during unwinding.
            quarantine_ipc_refs(refs)
            raise
        # The response is usable only if the producer kept every tensor stable
        # for the entire lease. Always retire an acknowledged lease, even when
        # the version tripwire rejects a result after detecting mutation.
        try:
            assert_ipc_refs_immutable(refs)
        finally:
            _complete_ipc_refs(refs)
        return result

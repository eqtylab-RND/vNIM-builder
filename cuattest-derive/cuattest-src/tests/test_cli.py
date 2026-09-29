# SPDX-License-Identifier: Apache-2.0
"""CLI trust-anchor, verification, and self-test contracts."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from cuattest import cli
from cuattest._statements import MANIFEST_VERSION
from cuattest.expect import EvidenceError, VerificationUnavailableError
from cuattest.notary import (
    GpuCleanupUncertainError,
    GpuSessionAbortedError,
    NotaryError,
    _QueuedGpuWorkUnconfirmedError,
)


@pytest.mark.parametrize("explicit_arch", [None, "sm_89"])
def test_build_kernel_uses_selected_device_for_automatic_arch(
    monkeypatch,
    tmp_path,
    explicit_arch,
):
    from cuattest import _cuda

    selected = []
    built = []

    class HeterogeneousCuda:
        def init(self):
            pass

        def device(self, ordinal):
            selected.append(ordinal)
            return ordinal + 100

        def compute_capability(self, device):
            return {100: (7, 5), 101: (9, 0)}[device]

    def build(arch, out):
        built.append(arch)
        return b"\x7fELFstub", "test", out / "kernel.cubin"

    monkeypatch.setattr(_cuda, "Cuda", HeterogeneousCuda)
    monkeypatch.setattr(cli.kmod, "build_cubin", build)
    argv = ["--device", "1", "build-kernel", "--out", str(tmp_path)]
    if explicit_arch:
        argv += ["--arch", explicit_arch]
    assert cli.main(argv) == 0
    assert built == [explicit_arch or "sm_90"]
    assert selected == ([] if explicit_arch else [1])


def expect_args(**overrides):
    values = {
        "model": "checkpoint",
        "compare": "receipt.json",
        "trusted_pubkey": None,
        "quiet": True,
        "no_tied": False,
        "json": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("mode", ["Release", "Debug", "AssertedRelease"])
def test_build_kernel_forwards_explicit_configuration(monkeypatch, tmp_path, mode):
    from cuattest._build_config import BUILD_TYPE
    seen = []
    def build(arch, out, *, build_type):
        seen.append((arch, out, build_type))
        return b"\x7fELFstub", "test", out / cli.kmod.cubin_filename(arch, build_type)
    monkeypatch.setattr(cli.kmod, "build_cubin", build)
    assert cli.main(["build-kernel", "--arch", "sm_120", "--out", str(tmp_path),
                     "--build-type", mode]) == 0
    assert seen == [("sm_120", tmp_path, mode)]
    # A cross-build choice affects that artifact, not the imported host policy.
    assert cli.kmod.BUILD_TYPE == BUILD_TYPE


def test_compare_requires_a_trust_anchor_before_opening_cuda(monkeypatch):
    monkeypatch.setattr(
        cli, "with_notary", lambda args, fn: pytest.fail("CUDA should not be opened")
    )

    # Even an accidentally empty path still means --compare was supplied; it
    # must not fall through to the successful compute-only command.
    for compare_path in ("receipt.json", ""):
        with pytest.raises(NotaryError, match="requires --trusted-pubkey"):
            cli.cmd_expect(expect_args(compare=compare_path))


def test_compare_passes_the_pinned_public_key(monkeypatch):
    pin = "04" + "11" * 64
    seen = []
    result = SimpleNamespace(
        matches=True, reason="", differing=None, report=lambda: "MATCH"
    )
    monkeypatch.setattr(cli.expect_mod, "compute", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli.expect_mod, "load_measurement", lambda path: {"receipt": path}
    )

    def compare(expected, receipt, trusted_pubkey=None):
        seen.append((receipt, trusted_pubkey))
        return result

    monkeypatch.setattr(cli.expect_mod, "compare", compare)
    monkeypatch.setattr(cli, "with_notary", lambda args, fn: fn(object()))

    with pytest.raises(SystemExit) as stopped:
        cli.cmd_expect(expect_args(trusted_pubkey=pin.upper()))

    assert stopped.value.code == 0
    assert seen == [({"receipt": "receipt.json"}, pin)]


def test_serve_passes_operator_request_limits_to_notary(monkeypatch):
    from cuattest import server as server_module

    observed = []

    class FakeNotary:
        def __init__(self, **kwargs):
            observed.append(("notary", kwargs))

        def close(self):
            observed.append(("close",))

    monkeypatch.setattr(cli, "Notary", FakeNotary)
    monkeypatch.setattr(
        server_module,
        "serve",
        lambda notary, host, port: observed.append(("serve", host, port)),
    )
    args = SimpleNamespace(
        device=2,
        artifacts="artifacts",
        host="127.0.0.2",
        port=9000,
        max_request_tensors=7,
        max_request_bytes=8,
        max_request_tiles=9,
    )

    assert cli.cmd_serve(args) == 0
    assert observed == [
        (
            "notary",
            {
                "device": 2,
                "artifact_dir": "artifacts",
                "max_request_tensors": 7,
                "max_request_bytes": 8,
                "max_request_tiles": 9,
            },
        ),
        ("serve", "127.0.0.2", 9000),
        ("close",),
    ]


@pytest.mark.parametrize("selection,devices", [("all", None), ("3,1", [3, 1])])
def test_serve_multiple_devices_selects_multi_gpu_notary(
    monkeypatch, selection, devices
):
    from cuattest import multigpu, server

    observed = []

    class FakePool:
        def __init__(self, **kwargs):
            observed.append(kwargs)

        def close(self):
            observed.append("closed")

    monkeypatch.setattr(multigpu, "MultiGpuNotary", FakePool)
    monkeypatch.setattr(server, "serve", lambda *args: None)
    assert (
        cli.main(["serve", "--devices", selection, "--max-request-tensors", "7"]) == 0
    )
    assert observed[0]["devices"] == devices
    assert observed[0]["max_request_tensors"] == 7
    assert observed[-1] == "closed"


def test_compare_accepts_operator_pinned_multi_gpu_key_file(monkeypatch, tmp_path):
    keys = {"GPU-00000000-0000-0000-0000-000000000001": "04" + "11" * 64}
    key_file = tmp_path / "keys.json"
    key_file.write_text(json.dumps(keys))
    monkeypatch.setattr(cli.expect_mod, "compute", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli.expect_mod, "load_measurement", lambda *args: {})
    seen = []

    def compare(expected, measured, *, trusted_pubkeys):
        seen.append(trusted_pubkeys)
        return SimpleNamespace(matches=True, report=lambda: "MATCH")

    monkeypatch.setattr(cli.expect_mod, "compare", compare)
    monkeypatch.setattr(cli, "with_notary", lambda args, fn: fn(object()))
    with pytest.raises(SystemExit) as stopped:
        cli.main(
            [
                "expect",
                "checkpoint",
                "--quiet",
                "--compare",
                "receipt.json",
                "--trusted-pubkeys",
                str(key_file),
            ]
        )
    assert stopped.value.code == 0 and seen == [keys]


def test_invalid_multi_gpu_key_file_is_rejected_before_cuda(monkeypatch, tmp_path):
    key_file = tmp_path / "keys.json"
    key_file.write_text('{"GPU-bad":"04","GPU-bad":"05"}')
    monkeypatch.setattr(cli, "with_notary", lambda *args: pytest.fail("opened CUDA"))
    assert (
        cli.main(
            [
                "expect",
                "checkpoint",
                "--compare",
                "receipt.json",
                "--trusted-pubkeys",
                str(key_file),
            ]
        )
        == 1
    )


def test_missing_verifier_reaches_cli_execution_error_path(monkeypatch, capsys):
    pin = "04" + "11" * 64
    monkeypatch.setattr(cli.expect_mod, "compute", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli.expect_mod, "load_measurement", lambda path: {})

    def verifier_unavailable(*args, **kwargs):
        raise VerificationUnavailableError("install the verify extra")

    monkeypatch.setattr(cli.expect_mod, "compare", verifier_unavailable)
    monkeypatch.setattr(cli, "with_notary", lambda args, fn: fn(object()))

    rc = cli.main(
        [
            "expect",
            "checkpoint",
            "--compare",
            "receipt.json",
            "--trusted-pubkey",
            pin,
        ]
    )

    assert rc == 1
    assert "install the verify extra" in capsys.readouterr().err


def test_invalid_evidence_reaches_cli_execution_error_path(monkeypatch, capsys):
    pin = "04" + "11" * 64
    monkeypatch.setattr(cli.expect_mod, "compute", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli.expect_mod, "load_measurement", lambda path: {})
    monkeypatch.setattr(
        cli.expect_mod,
        "compare",
        lambda *args, **kwargs: (_ for _ in ()).throw(EvidenceError("bad signature")),
    )
    monkeypatch.setattr(cli, "with_notary", lambda args, fn: fn(object()))

    rc = cli.main(
        [
            "expect",
            "checkpoint",
            "--compare",
            "receipt.json",
            "--trusted-pubkey",
            pin,
        ]
    )

    assert rc == 1
    assert "bad signature" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error_type", [GpuSessionAbortedError, GpuCleanupUncertainError]
)
def test_cli_owned_hash_buffer_is_abandoned_after_context_cleanup(error_type):
    buffer = SimpleNamespace(ptr=0x1234)

    class FailedNotary:
        def hash_dptr(self, ptr, nbytes):
            raise error_type("queued work")

    with pytest.raises(error_type):
        cli._hash_owned_device_buffer(FailedNotary(), buffer, 8, offset=4)

    assert buffer.ptr == 0


def test_cli_fused_buffers_are_abandoned_before_context_abort():
    buffers = [SimpleNamespace(ptr=0x1000), SimpleNamespace(ptr=0x2000)]
    observed = []

    class FailedNotary:
        def _launch_fused_active(self, spans, measured_at, model):
            raise _QueuedGpuWorkUnconfirmedError("queued work")

        def _abort_unconfirmed_gpu_work(self, error):
            observed.append(([buffer.ptr for buffer in buffers], error))
            raise GpuSessionAbortedError("context destroyed")

    with pytest.raises(GpuSessionAbortedError):
        cli._launch_owned_fused_buffers(
            FailedNotary(), buffers, [8, 16], "2026-09-05T00:00:00Z", b"selftest"
        )

    assert observed[0][0] == [0, 0]
    assert isinstance(observed[0][1], _QueuedGpuWorkUnconfirmedError)


def test_selftest_rejects_a_deterministic_but_non_blake3_tree_hash(monkeypatch, capsys):
    from cuattest import _cuda

    empty_digest = bytes.fromhex(
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    )
    wrong_digest = b"\x99" * 32
    cid = "bafkr4ifpcne3t5pzugtkaqcn5i3nzskjtpfslsnnyejlpte2spfoihzsmi"

    class Buffer:
        ptr = 1

        @classmethod
        def from_bytes(cls, cuda, data):
            return cls()

        def close(self):
            pass

    class FakeNotary:
        device_ordinal = 0
        cu = object()
        info = SimpleNamespace(gpu_did="did:key:z-test", kernel_cid=cid, cubin_cid=cid)

        def hash_bytes(self, data):
            return empty_digest if not data else wrong_digest

        def hash_dptr(self, ptr, nbytes):
            return wrong_digest

        def _activate(self):
            return nullcontext()

        def _launch_fused_active(self, spans, timestamp, model):
            signed = {
                "device": "cuda:0",
                "kernelCID": f"urn:cid:{cid}",
                "cubinCID": f"urn:cid:{cid}",
            }
            # Selftest fails at hashing, well before the statement graph, so
            # this only has to be a structurally real receipt.
            receipt = json.dumps(
                {
                    "measurementDocument": json.dumps(signed).encode().hex(),
                    "measurementSignature": "00" * 64,
                    "modelRoot": wrong_digest.hex(),
                    "manifest": {"version": MANIFEST_VERSION, "statements": {}},
                }
            )
            roots = wrong_digest * len(spans)
            return SimpleNamespace(
                roots=roots,
                model_root=wrong_digest,
                receipt=receipt,
            )

    monkeypatch.setattr(_cuda, "DeviceBuffer", Buffer)
    monkeypatch.setattr(cli, "with_notary", lambda args, fn: fn(FakeNotary()))

    assert cli.cmd_selftest(SimpleNamespace()) == 1
    output = capsys.readouterr().out
    assert "FAIL multi-chunk hashing matches host BLAKE3" in output

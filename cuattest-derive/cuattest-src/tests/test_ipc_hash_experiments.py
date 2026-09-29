"""Deterministic invariants of the opt-in, non-production latency prototypes."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

DIRECTORY = Path(__file__).parent / "performance" / "experiments" / "ipc_hash"


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"ipc_hash_{name}", DIRECTORY / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packed_plan_preserves_names_devices_and_disjoint_byte_spans():
    resident = load("resident")
    specs = [
        SimpleNamespace(name=name, nbytes=size)
        for name, size in (("c", 1025), ("a", 1024), ("b", 3), ("d", 0x400000001))
    ]
    placement = {"a": 0, "b": 1, "c": 0, "d": 1}
    offsets, sizes = resident.pack_plan(specs, placement)
    assert offsets == {"a": 0, "b": 0, "c": 1024, "d": 256}
    assert sizes == {0: 2049, 1: 256 + 0x400000001}
    assert resident.pack_plan(list(reversed(specs)), placement) == (offsets, sizes)
    for device in sizes:
        spans = sorted(
            (offsets[t.name], offsets[t.name] + t.nbytes)
            for t in specs
            if placement[t.name] == device
        )
        assert all(start % 256 == 0 for start, _ in spans)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_registered_runner_rehashes_every_call_and_refuses_substitution():
    server = load("server")
    calls = []

    class Driver:
        def ipc_open(self, raw):
            calls.append(("open", raw))
            return 4096

        def ipc_close(self, ptr):
            calls.append(("close", ptr))

        def pointer_device(self, ptr):
            return 2

        def address_range(self, ptr):
            return ptr, 8192

    class Runner:
        def run_spans(self, spans, timestamp, model, instance_root=None):
            calls.append(("fresh", spans, timestamp, model))
            return len(calls)

    from cuattest.notary import TensorRef

    handle = bytes(64).hex()
    refs = [TensorRef(handle, 1024, 0, offset, 2) for offset in (0, 1024)]
    notary = SimpleNamespace(_native_runner=Runner(), cu=Driver(), device_ordinal=2)
    runner = server.RegisteredRunner(notary, refs)
    assert calls == [("open", bytes(64))]
    assert runner.run_ipc(runner.inputs, b"time1", b"model") != runner.run_ipc(
        runner.inputs, b"time2", b"model"
    )
    assert len([call for call in calls if call[0] == "fresh"]) == 2
    assert not any(call[0] == "close" for call in calls)
    with pytest.raises(RuntimeError, match="substitute"):
        runner.run_ipc(runner.inputs[:-1], b"time3", b"model")
    runner.unregister()
    assert calls[-1] == ("close", 4096)


def test_generator_refuses_a_changed_baseline():
    variants = load("variants")
    with pytest.raises(ValueError, match="exact 1ce0fbe"):
        variants.transform("not the reviewed source", "async")


def test_async_generator_has_a_bounded_waited_copy_and_old_gpu_fallback(monkeypatch):
    variants = load("variants")
    from cuattest.kernel import kernel_source
    import hashlib

    source = kernel_source()
    # Exercise transformation structure without freezing the production file
    # forever. The separate mismatch test covers the real compiler's gate.
    monkeypatch.setattr(
        variants, "BASELINE", hashlib.sha256(source.encode()).hexdigest()
    )
    generated = variants.transform(source, "async")
    assert "__shared__ uint4 stage[2][8][128]" in generated
    assert "#if __CUDA_ARCH__ >= 800" in generated
    assert "cp.async.wait_group 0;" in generated
    assert "if (block_index + 1 < 16)" in generated
    assert "int next = slot ^ 1;" in generated
    assert (
        "__syncthreads"
        not in generated.split("__shared__ uint4 stage")[1].split("#else")[0]
    )


def test_registration_error_sends_no_storage_release_ack():
    server = load("server")
    notary = SimpleNamespace()
    handler = object.__new__(server.registered_handler(notary))
    handler.server = SimpleNamespace(cuattest_fatal_error=None)
    handler.path = "/v1/sign"
    handler._read_json = lambda: {"token": "stale", "model": "test"}
    sent = []
    handler._send = lambda *args: sent.append(args)
    handler.do_POST()
    assert sent == []
    assert handler.close_connection
    assert isinstance(handler.server.cuattest_fatal_error, RuntimeError)


def test_failed_unregister_never_forgets_owned_maps():
    server = load("server")
    runner = object.__new__(server.RegisteredRunner)
    runner.mappings = [4096, 8192]
    calls = []

    def close(ptr):
        calls.append(ptr)
        raise RuntimeError("injected driver failure")

    runner.cu = SimpleNamespace(ipc_close=close)
    with pytest.raises(RuntimeError, match="injected"):
        runner.unregister()
    # The server must terminate and destroy its context. No empty ledger may
    # allow a later handler to manufacture an unregister ACK after this error.
    assert runner.mappings == [4096, 8192]
    assert calls == [4096]


@pytest.fixture
def report_campaign(tmp_path):
    placement = {"weight": 0}
    for layout in ("ordinary", "packed"):
        directory = tmp_path / layout
        directory.mkdir()
        (directory / "resident-layout.json").write_text(
            json.dumps({"placement": placement})
        )
        (directory / "trace-summary.json").write_text("[]")
    (tmp_path / "remote-probes").mkdir()
    (tmp_path / "remote-probes" / "fresh.log").write_text("{}")
    (tmp_path / "async-sanitizers").mkdir()
    (tmp_path / "async-sanitizers" / "results.json").write_text(
        json.dumps([dict(clean=True, returncode=0) for _ in range(8)])
    )

    def write_trial(layout, label, values, *, profile=False):
        data = dict(
            label=label,
            samples=[
                dict(seconds=s, tensor_count=1, attested_bytes=1024, model_root="root")
                for s in values
            ],
            placement=placement,
            independent_model_root="root",
            tensor_count=1,
            attested_bytes=1024,
            profile=profile,
        )
        (tmp_path / layout / f"{label}-results.json").write_text(json.dumps(data))

    return tmp_path, write_trial


def test_report_keeps_outliers_and_excludes_profiles(report_campaign):
    directory, write_trial = report_campaign
    for layout in ("ordinary", "packed"):
        for suffix, values, profile in (
            ("a", [1, 100], False),
            ("b", [3], False),
            ("trace", [9999], True),
        ):
            write_trial(layout, f"{layout}-baseline-{suffix}", values, profile=profile)
    data = load("report").collect(directory)
    assert set(data["summary"]) == {
        f"{layout}/{layout}-baseline" for layout in ("ordinary", "packed")
    }
    for summary in data["summary"].values():
        assert summary["samples"] == 3
        assert summary["max_ms"] == 100000
        assert summary["mean_ms"] == pytest.approx(104000 / 3)
    assert len(data["trials"]) == 6  # Profiler samples remain in raw evidence.
    assert [t["samples_seconds"] for t in data["trials"] if t["profile"]] == [
        [9999], [9999]
    ]


@pytest.mark.parametrize("suffixes", [("",), ("-a", "-b", "-c", "-d")])
def test_report_separates_reused_labels_by_layout(
    report_campaign, monkeypatch, capsys, suffixes
):
    directory, write_trial = report_campaign
    expected_summary, expected_trials = {}, {}
    for layout, baseline in (("ordinary", 1), ("packed", 10)):
        # Both layouts may use the README's 01-baseline job label. Their 1 s
        # and 10 s measurements must not become a fictitious 5.5 s baseline,
        # even when -a/-b/-c/-d repetitions are folded into the same job.
        for label, seconds in (("01-baseline", baseline), ("02-async", baseline / 2)):
            for suffix in suffixes:
                values = [seconds, seconds]
                write_trial(layout, label + suffix, values)
                expected_trials[layout, label + suffix] = values
            expected_summary[f"{layout}/{label}"] = dict(
                samples=2 * len(suffixes),
                mean_ms=seconds * 1000,
                median_ms=seconds * 1000,
                stdev_ms=0,
                min_ms=seconds * 1000,
                max_ms=seconds * 1000,
            )

    # Exercise the real JSON/stdout format too: allocation identity must be
    # retained in the published summary, not just in an internal grouping key.
    output = directory / "report.json"
    monkeypatch.setattr(
        "sys.argv", [str(DIRECTORY / "report.py"), str(directory), str(output)]
    )
    load("report").main()
    data = json.loads(output.read_text())
    assert data["summary"] == expected_summary
    assert json.loads(capsys.readouterr().out) == expected_summary
    assert {
        (trial["layout"], trial["label"]): trial["samples_seconds"]
        for trial in data["trials"]
    } == expected_trials

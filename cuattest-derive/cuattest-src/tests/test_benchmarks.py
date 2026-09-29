# SPDX-License-Identifier: Apache-2.0
"""Benchmark timing, safe capacity planning, and pretrained-byte regressions."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from performance import benchmark_large_model as large
from performance import benchmark_torch_models as small

from cuattest._hosthash import blake3_digest


def spec(name, size, shard="model.safetensors"):
    return large.TensorSpec(name, (size,), "U8", size, shard)


def device(ordinal, total=100, free=96):
    return large.DeviceMemory(
        ordinal, f"GPU-{ordinal}", total * large.GIB, free * large.GIB
    )


def test_default_small_workloads_are_unchanged_and_large_is_explicit():
    assert small.parse_args([]).models == list(small.DEFAULT_MODELS)
    assert [m.model_id for m in small.DEFAULT_MODELS] == [
        "openai-community/gpt2",
        "openai-community/gpt2-large",
    ]
    args = large.parse_args([])
    assert args.model == large.DEFAULT_MODEL and len(args.model.revision) == 40
    assert args.exclude_prefix == ["mtp."]
    assert args.min_utilization == 0.9 and args.reserve_gib == 1.0
    assert args.runs == 5 and args.warmup_runs == 1
    # A preset's architectural exclusion must not silently apply to an
    # unrelated checkpoint or to a user's custom exclusion list.
    assert large.parse_args(["--model", "other/model"]).exclude_prefix == []
    assert large.parse_args(["--checkpoint", "local"]).exclude_prefix == []
    assert large.parse_args(["--exclude-prefix", "aux."]).exclude_prefix == ["aux."]


@pytest.mark.parametrize(
    "argument,value",
    [
        ("--min-utilization", "nan"),
        ("--min-utilization", "inf"),
        ("--min-utilization", "1"),
        ("--min-utilization", "-0.1"),
        ("--reserve-gib", "nan"),
        ("--reserve-gib", "inf"),
        ("--reserve-gib", "0"),
        ("--runs", "0"),
        ("--warmup-runs", "-1"),
        ("--exclude-prefix", ""),
    ],
)
def test_invalid_benchmark_limits_are_rejected(argument, value):
    with pytest.raises(SystemExit):
        large.parse_args([argument, value])


@pytest.mark.parametrize("selection", ["", "0,0", "-1", "2", "cpu", "0,"])
def test_invalid_device_selection(selection):
    with pytest.raises(ValueError):
        large.select_devices(selection, 2)


def test_device_selection_preserves_explicit_order():
    assert large.select_devices("all", 3) == [0, 1, 2]
    assert large.select_devices("2,0", 3) == [2, 0]
    with pytest.raises(ValueError):
        large.select_devices("all", 0)


def test_selection_never_pads_repeats_or_silently_drops_main_weights():
    tensors = [spec("model.weight", 10), spec("mtp.weight", 20), spec("empty", 0)]
    assert large.select_tensors(tensors, ["mtp."]) == [tensors[0]]
    assert sum(t.nbytes for t in large.select_tensors(tensors, [])) == 30
    with pytest.raises(ValueError, match="duplicate"):
        large.select_tensors([tensors[0], tensors[0]], [])
    with pytest.raises(ValueError, match="nonempty"):
        large.select_tensors(tensors, ["model", "mtp"])


def test_largest_first_planning_fits_without_slicing_or_changing_span_order():
    tensors = [
        spec("a.small", 3 * large.GIB),
        spec("b.big", 8 * large.GIB),
        spec("c.small", 3 * large.GIB),
        spec("d.big", 8 * large.GIB),
    ]
    devices = [device(0, total=12, free=12), device(1, total=12, free=12)]
    placement = large.plan_placement(tensors, devices, large.GIB, 0.9)
    assert placement == {"b.big": 0, "d.big": 1, "a.small": 0, "c.small": 1}
    assert [t.name for t in tensors] == ["a.small", "b.big", "c.small", "d.big"]


def test_planning_uses_free_vram_not_total_and_reserves_overhead():
    tensors = [spec("too_large", 95 * large.GIB)]
    with pytest.raises(ValueError, match="does not fit"):
        large.plan_placement(tensors, [device(0, free=95)], large.GIB, 0)
    with pytest.raises(ValueError, match="headroom"):
        large.plan_placement(tensors, [device(0, free=1)], large.GIB, 0)
    # Account for CUDA allocator rounding, not merely logical byte counts.
    tensors = [spec("rounds_up", 11 * 1024**2 + 1)]
    tight = large.DeviceMemory(0, "uuid", 20 * 1024**2, 12 * 1024**2)
    with pytest.raises(ValueError, match="does not fit"):
        large.plan_placement(tensors, [tight], 1, 0)


def test_minimum_utilization_is_per_device_not_just_aggregate():
    with pytest.raises(ValueError, match="cuda:1.*min-utilization"):
        large.plan_placement(
            [spec("weight", 94 * large.GIB)], [device(0), device(1)], large.GIB, 0.9
        )


def test_heterogeneous_capacity_planning():
    tensors = [spec(f"weight.{i}", 10 * large.GIB) for i in range(14)]
    devices = [device(0, total=50, free=51), device(1)]
    placement = large.plan_placement(tensors, devices, large.GIB, 0.9)
    assert sorted(placement.values()).count(0) == 5
    assert sorted(placement.values()).count(1) == 9


def test_inspection_routes_by_uuid_even_with_remapped_ordinals(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        large,
        "Cuda",
        lambda: SimpleNamespace(
            init=lambda: None, device=lambda d: d, device_uuid=lambda d: f"uuid-{1 - d}"
        ),
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device", lambda d: nullcontext())
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (90, 100))
    info = {
        "multi_gpu": True,
        "devices": [
            {"device_ordinal": 0, "device_uuid": "uuid-0"},
            {"device_ordinal": 1, "device_uuid": "uuid-1"},
        ],
    }
    assert [d.uuid for d in large.inspect_devices("all", info)] == ["uuid-1", "uuid-0"]
    info["devices"].pop()
    with pytest.raises(ValueError, match="not served"):
        large.inspect_devices("all", info)


@pytest.mark.parametrize(
    "keys,keyword",
    [("single-key", "trusted_pubkey"), ({"uuid": "key"}, "trusted_pubkeys")],
)
def test_timing_excludes_export_and_verified_keys_are_passed(
    monkeypatch, keys, keyword
):
    events = []
    refs = [{"nbytes": 123}]

    def share(model):
        events.append("export")
        return ["weight"], refs, [object()]

    def clock():
        events.append("clock")
        return len(events) * 0.1

    def verify(receipt, **kwargs):
        events.append("verify")
        assert kwargs == {keyword: keys}
        return SimpleNamespace(tensor_count=1, model_root="root")

    def sign(received, model_id):
        events.append("sign")
        assert received is refs
        return {"receipt": True}

    monkeypatch.setattr(small, "share_model", share)
    monkeypatch.setattr(small.time, "perf_counter", clock)
    monkeypatch.setattr(small, "verify_evidence", verify)
    result = small.attest_once(SimpleNamespace(sign=sign), object(), "model", keys)
    assert events == ["export", "clock", "sign", "clock", "verify"]
    assert result.seconds == pytest.approx(0.2)
    assert result.attested_bytes == 123


def test_warmups_excluded_and_independent_root_enforced(monkeypatch):
    calls = []

    def attest(*args):
        calls.append(args)
        return small.AttestationSample(len(calls), 1, 123, "root")

    monkeypatch.setattr(small, "attest_once", attest)
    samples = small.sample_model(None, None, "model", 2, 1, "key", expected_root="root")
    assert [s.seconds for s in samples] == [2, 3]
    with pytest.raises(RuntimeError, match="root differs"):
        small.sample_model(None, None, "model", 1, 0, "key", expected_root="wrong")


def test_registered_benchmark_times_the_full_public_call_but_not_setup_or_verification(monkeypatch):
    events = []
    class Handle:
        names = ["weight"]
        attested_bytes = 123
        def sign(self, model):
            events.append("sign-with-readiness-and-metadata")
            return {}
        def __enter__(self):
            events.append("register")
            return self
        def __exit__(self, *args):
            events.append("close")
    def clock():
        events.append("clock")
        return events.count("clock") * 0.2
    def verify(receipt, **keys):
        events.append("verify")
        assert keys == {"trusted_pubkeys": {"gpu": "key"}}
        return SimpleNamespace(tensor_count=1, model_root="root")
    monkeypatch.setattr(small.time, "perf_counter", clock)
    monkeypatch.setattr(small, "verify_evidence", verify)
    client = SimpleNamespace(register_model=lambda model: Handle())
    samples = small.sample_model(client, object(), "model", 1, 1, {"gpu": "key"}, registered=True, expected_root="root")
    assert len(samples) == 1 and samples[0].seconds == pytest.approx(0.2)
    assert samples[0].attested_bytes == 123
    assert events == ["register", *(["clock", "sign-with-readiness-and-metadata", "clock", "verify"] * 2), "close"]
    assert small.parse_args(["--registered"]).registered
    assert large.parse_args(["--registered"]).registered


def test_loader_uses_actual_pretrained_bytes_and_independent_sorted_fold(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    stored = {
        "z.weight": torch.arange(32, dtype=torch.bfloat16),
        "a.weight": torch.tensor([17, 99], dtype=torch.uint8),
    }
    checkpoint = tmp_path / "model.safetensors"
    st.save_file(stored, str(checkpoint))
    specs, files = large.local_specs(checkpoint)
    copies = []
    original_to = torch.Tensor.to

    def copy_to(tensor, destination, *args, **kwargs):
        if isinstance(destination, str) and destination.startswith("cuda:"):
            copies.append((destination, tensor.dtype, tuple(tensor.shape)))
            return tensor.clone()
        return original_to(tensor, destination, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", copy_to)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d: None)
    placement = {"a.weight": 1, "z.weight": 0}
    model, root = large.load_resident(specs, files, placement, 1)
    for name, tensor in stored.items():
        assert torch.equal(model.tensors[name], tensor)
        assert model.tensors[name].dtype == tensor.dtype
    assert {d for d, _, _ in copies} == {"cuda:0", "cuda:1"}
    digests = b"".join(
        blake3_digest(stored[name].reshape(-1).view(torch.uint8).numpy())
        for name in sorted(stored)
    )
    assert root == blake3_digest((2).to_bytes(4, "little") + digests).hex()


@pytest.mark.parametrize(
    "shape,offsets", [((2,), (0, 4)), ((2,), (0, 3)), ((-2,), (0, 4))]
)
def test_hub_headers_use_resolved_revision_and_validate_stored_dtype(
    monkeypatch, shape, offsets
):
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    class Api:
        def model_info(self, model, revision):
            assert (model, revision) == ("org/model", "main")
            return SimpleNamespace(sha="a" * 40)

        def get_safetensors_metadata(self, model, revision, timeout):
            calls.append((model, revision))
            return SimpleNamespace(
                weight_map={"weight": "part.safetensors"},
                files_metadata={
                    "part.safetensors": SimpleNamespace(
                        tensors={
                            "weight": SimpleNamespace(
                                shape=shape, dtype="BF16", data_offsets=offsets
                            )
                        }
                    )
                },
            )

    monkeypatch.setattr(hub, "HfApi", Api)
    if shape == (2,) and offsets == (0, 4):
        tensors, revision = large.hub_specs(small.ModelSpec("org/model", "main"), None)
        assert revision == "a" * 40
        assert tensors == [
            large.TensorSpec("weight", (2,), "BF16", 4, "part.safetensors")
        ]
    else:
        with pytest.raises(ValueError, match="invalid checkpoint"):
            large.hub_specs(small.ModelSpec("org/model", "main"), None)
    assert calls == [("org/model", "a" * 40)]


def test_download_pins_selected_shards_and_rechecks_actual_headers(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    hub = pytest.importorskip("huggingface_hub")
    path = tmp_path / "part.safetensors"
    st.save_file({"weight": torch.zeros(2, dtype=torch.bfloat16)}, str(path))
    calls = []

    def snapshot(model, **kwargs):
        calls.append((model, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(hub, "snapshot_download", snapshot)
    tensor = large.TensorSpec("weight", (2,), "BF16", 4, path.name)
    files = large.download_checkpoint(
        small.ModelSpec("org/model", "main"), "fixed", [tensor], tmp_path
    )
    assert files == {path.name: path}
    assert calls == [
        (
            "org/model",
            {
                "revision": "fixed",
                "allow_patterns": [path.name],
                "cache_dir": tmp_path,
                "max_workers": 4,
            },
        )
    ]
    # A header fetched before a long download is not sufficient validation:
    # the downloaded shape/dtype/size must still match, even at equal byte size.
    st.save_file({"weight": torch.zeros(1, dtype=torch.float32)}, str(path))
    with pytest.raises(ValueError, match="contradicts pinned metadata"):
        large.download_checkpoint(small.ModelSpec("org/model"), "fixed", [tensor], None)


def test_existing_result_file_is_not_overwritten(tmp_path):
    result = tmp_path / "results.json"
    result.write_text("old benchmark")
    with pytest.raises(SystemExit):
        large.parse_args(["--json-output", str(result)])
    assert result.read_text() == "old benchmark"


@pytest.mark.parametrize("defect", ["cpu", "wrong_gpu", "alias", "padding", "missing"])
def test_resident_accounting_rejects_fake_occupancy(defect):
    def tensor(ptr):
        return SimpleNamespace(
            is_cuda=True,
            device=SimpleNamespace(index=0),
            is_contiguous=lambda: True,
            numel=lambda: 1024,
            element_size=lambda: 1,
            untyped_storage=lambda: SimpleNamespace(
                data_ptr=lambda: ptr, nbytes=lambda: 1024
            ),
        )

    state = {"a": tensor(100), "b": tensor(200)}
    specs = [spec("a", 1024), spec("b", 1024)]
    placement = {"a": 0, "b": 0}
    assert large.resident_bytes(large.ResidentCheckpoint(state), specs, placement) == {
        0: 2048
    }
    if defect == "cpu":
        state["b"].is_cuda = False
    elif defect == "wrong_gpu":
        state["b"].device.index = 1
    elif defect == "alias":
        state["b"] = state["a"]
    elif defect == "padding":
        state["b"].untyped_storage = lambda: SimpleNamespace(
            data_ptr=lambda: 200, nbytes=lambda: 4096
        )
    else:
        del state["b"]
    with pytest.raises(RuntimeError):
        large.resident_bytes(large.ResidentCheckpoint(state), specs, placement)


def test_missing_native_exporter_fails_before_network_or_allocation(monkeypatch):
    pytest.importorskip("torch")
    from cuattest import ipc

    monkeypatch.setattr(ipc, "_native_export_cuda_allocation", None)
    monkeypatch.setattr(large, "Client", lambda *args: pytest.fail("contacted server"))
    with pytest.raises(RuntimeError, match="native extension"):
        large.main([])


def test_dry_run_budgets_after_signer_initialization_and_never_loads(monkeypatch):
    pytest.importorskip("torch")
    from cuattest import ipc

    events = []
    monkeypatch.setattr(ipc, "_native_export_cuda_allocation", object())
    monkeypatch.setattr(large, "Client", lambda *args: SimpleNamespace(info=dict))
    monkeypatch.setattr(large, "describe_notary", lambda info: "pin")

    def inspect(*args):
        events.append("inspect")
        return [device(0)]

    def headers(path):
        events.append("headers")
        return [spec("weight", 94 * large.GIB)], {}

    monkeypatch.setattr(large, "inspect_devices", inspect)
    monkeypatch.setattr(large, "warm_notaries", lambda *args: events.append("warm"))
    monkeypatch.setattr(large, "local_specs", headers)
    monkeypatch.setattr(
        large, "load_resident", lambda *args: pytest.fail("loaded model")
    )
    assert large.main(["--checkpoint", "checkpoint", "--dry-run"]) == 0
    assert events == ["inspect", "warm", "inspect", "headers"]


@pytest.mark.parametrize("failure", [None, "start", "stop", "bind"])
def test_deferred_profiler_covers_requests_and_always_closes_contexts(monkeypatch, failure):
    from performance import profile_multigpu_server as profile

    events = []

    def construct(**kwargs):
        events.append("initialize all GPUs")
        return SimpleNamespace(
            info=SimpleNamespace(as_dict=dict),
            close=lambda: events.append("close contexts"),
        )

    def capture(notary, function):
        stage = "start" if function == "cuProfilerStart" else "stop"
        events.append(stage)
        if failure == stage:
            raise RuntimeError(stage)

    def serve():
        events.append("serve requests")
        raise KeyboardInterrupt

    def server(*args):
        events.append("bind")
        if failure == "bind":
            raise RuntimeError("bind")
        return SimpleNamespace(
            server_port=1234, serve_forever=serve,
            server_close=lambda: events.append("close socket"),
        )

    monkeypatch.setattr(profile, "MultiGpuNotary", construct)
    monkeypatch.setattr(profile, "profiler_call", capture)
    monkeypatch.setattr(profile, "_NotaryHTTPServer", server)
    if failure is None:
        assert profile.main([]) == 0
    else:
        with pytest.raises(RuntimeError, match=failure):
            profile.main([])
    expected = ["initialize all GPUs", "bind"]
    if failure != "bind":
        expected += ["start"]
        if failure != "start":
            expected += ["serve requests"]
        expected += ["close socket"]
        if failure != "start":
            expected += ["stop"]
    assert events == expected + ["close contexts"]

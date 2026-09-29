# SPDX-License-Identifier: Apache-2.0
"""Checkpoint expectation canonicalization without a CUDA dependency."""

import json
import struct
from contextlib import contextmanager
from typing import ClassVar

import pytest

from cuattest import expect as expect_module
from cuattest._hosthash import blake3_digest
from cuattest.notary import GpuCleanupUncertainError


def write_safetensors(path, tensors, metadata=None):
    header = {}
    payload = bytearray()
    for name, shape, data in tensors:
        start = len(payload)
        payload += data
        header[name] = {
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


class FakeBuffer:
    next_ptr = 1
    contents: ClassVar[dict[int, bytes]] = {}

    @classmethod
    def from_bytes(cls, cuda, data):
        assert cuda.active, "allocation used the wrong thread-current CUDA context"
        buffer = cls()
        buffer.ptr = cls.next_ptr
        cls.next_ptr += 1
        cls.contents[buffer.ptr] = data
        return buffer

    def close(self):
        self.contents.pop(self.ptr, None)
        self.ptr = 0


class FakeNotary:
    def __init__(self):
        self.cu = type("FakeCuda", (), {"active": False})()

    @contextmanager
    def _activate(self):
        assert not self.cu.active
        self.cu.active = True
        try:
            yield
        finally:
            self.cu.active = False

    def hash_dptr(self, ptr, nbytes):
        assert self.cu.active
        return blake3_digest(FakeBuffer.contents[ptr][:nbytes])

    def hash_bytes(self, data):
        assert self.cu.active
        return blake3_digest(data)


def test_expectation_omits_empty_tensors_like_runtime_sharing(monkeypatch, tmp_path):
    checkpoint = write_safetensors(
        tmp_path / "model.safetensors",
        [
            ("empty", [0], b""),
            ("weight", [3], b"abc"),
        ],
    )
    monkeypatch.setattr(expect_module, "DeviceBuffer", FakeBuffer)

    expectation = expect_module.compute(FakeNotary(), checkpoint, tied=False)

    digest = blake3_digest(b"abc")
    root = blake3_digest((1).to_bytes(4, "little") + digest)
    assert expectation.names == ["weight"]
    assert expectation.tensor_count == 1
    assert expectation.total_bytes == 3
    assert expectation.digests == digest.hex()
    assert expectation.model_root == root.hex()


def test_checkpoint_with_only_empty_tensors_has_no_measurable_state(tmp_path):
    checkpoint = write_safetensors(
        tmp_path / "empty.safetensors", [("empty", [0], b"")]
    )

    with pytest.raises(ValueError, match="no non-empty tensors"):
        expect_module.compute(FakeNotary(), checkpoint, tied=False)


def test_expectation_does_not_free_source_after_uncertain_context_cleanup(
    monkeypatch, tmp_path
):
    checkpoint = write_safetensors(
        tmp_path / "model.safetensors", [("weight", [3], b"abc")]
    )
    closed_pointers = []

    class QuarantinedBuffer:
        ptr = 0x1234

        @classmethod
        def from_bytes(cls, cuda, data):
            return cls()

        def close(self):
            closed_pointers.append(self.ptr)

    class UncertainNotary(FakeNotary):
        def hash_dptr(self, ptr, nbytes):
            raise GpuCleanupUncertainError("context destruction was not confirmed")

    monkeypatch.setattr(expect_module, "DeviceBuffer", QuarantinedBuffer)

    with pytest.raises(GpuCleanupUncertainError):
        expect_module.compute(UncertainNotary(), checkpoint, tied=False)

    # A zero pointer means DeviceBuffer.close cannot issue cuMemFree against
    # either a destroyed context or storage still referenced by queued work.
    assert closed_pointers == [0]


def test_expectation_restores_every_alias_declared_by_full_span_schema(
    monkeypatch, tmp_path
):
    aliases = {
        "shared.weight": "decoder.embed_tokens.weight",
        "encoder.embed_tokens.weight": "decoder.embed_tokens.weight",
        "lm_head.weight": "decoder.embed_tokens.weight",
    }
    checkpoint = write_safetensors(
        tmp_path / "model.safetensors",
        [("decoder.embed_tokens.weight", [4], b"tied")],
        metadata={
            "format": "pt",
            "cuattest.aliases.v1": json.dumps(
                {
                    alias: {"kind": "full-span", "source": source}
                    for alias, source in aliases.items()
                }
            ),
        },
    )
    monkeypatch.setattr(expect_module, "DeviceBuffer", FakeBuffer)

    expectation = expect_module.compute(FakeNotary(), checkpoint)

    digest = blake3_digest(b"tied")
    assert expectation.names == [
        "decoder.embed_tokens.weight",
        "encoder.embed_tokens.weight",
        "lm_head.weight",
        "shared.weight",
    ]
    assert expectation.tensor_count == 4
    assert expectation.total_bytes == 4  # aliases reuse one disk/device hash
    assert expectation.digests == (digest * 4).hex()
    assert set(expectation.tied) == {
        f"{alias} = decoder.embed_tokens.weight" for alias in aliases
    }


def test_save_model_view_metadata_is_not_expanded_to_the_base_extent(
    monkeypatch, tmp_path
):
    # safetensors.torch.save_model records only "view": "base" when it drops
    # an overlapping view. There is no offset/length here, so manufacturing a
    # full-size alias would make valid runtime evidence impossible to match.
    checkpoint = write_safetensors(
        tmp_path / "model.safetensors",
        [("base", [8], b"12345678")],
        metadata={"format": "pt", "view": "base"},
    )
    monkeypatch.setattr(expect_module, "DeviceBuffer", FakeBuffer)

    expectation = expect_module.compute(FakeNotary(), checkpoint)

    assert expectation.names == ["base"]
    assert expectation.tensor_count == 1
    assert expectation.total_bytes == 8
    assert expectation.tied == ()


def test_t5_config_restores_all_runtime_names_when_sharded_metadata_has_no_aliases(
    monkeypatch, tmp_path
):
    checkpoint = write_safetensors(
        tmp_path / "model.safetensors",
        [("decoder.embed_tokens.weight", [4], b"tied")],
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "t5",
                "tie_word_embeddings": True,
            }
        )
    )
    monkeypatch.setattr(expect_module, "DeviceBuffer", FakeBuffer)

    expectation = expect_module.compute(FakeNotary(), checkpoint)

    assert expectation.names == [
        "decoder.embed_tokens.weight",
        "encoder.embed_tokens.weight",
        "lm_head.weight",
        "shared.weight",
    ]
    assert set(expectation.tied) == {
        "shared.weight = decoder.embed_tokens.weight",
        "encoder.embed_tokens.weight = decoder.embed_tokens.weight",
        "lm_head.weight = decoder.embed_tokens.weight",
    }


@pytest.mark.parametrize("family", ["T5", "MT5", "UMT5", "LongT5"])
@pytest.mark.parametrize(
    "architecture", ["EncoderModel", "Model", "ForConditionalGeneration"]
)
@pytest.mark.parametrize("tie_output", [False, True])
def test_t5_aliases_follow_saved_architecture(
    monkeypatch,
    tmp_path,
    family,
    architecture,
    tie_output,
):
    names = ["shared.weight", "encoder.embed_tokens.weight"]
    if architecture != "EncoderModel":
        names.append("decoder.embed_tokens.weight")
    if architecture == "ForConditionalGeneration" and tie_output:
        names.append("lm_head.weight")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": family.lower(),
                "architectures": [family + architecture],
                "tie_word_embeddings": tie_output,
                # T5EncoderModel can retain this family default in the saved config.
                "is_encoder_decoder": True,
            }
        )
    )
    monkeypatch.setattr(expect_module, "DeviceBuffer", FakeBuffer)
    # Any member, including an output head, can be the serialized survivor.
    for source in names:
        checkpoint = write_safetensors(
            tmp_path / "model.safetensors",
            [(source, [4], b"tied")],
        )
        expectation = expect_module.compute(FakeNotary(), checkpoint)
        assert expectation.names == sorted(names)
        assert expectation.digests == (blake3_digest(b"tied") * len(names)).hex()
        assert expectation.total_bytes == 4


@pytest.mark.parametrize("model_type", ["gpt2", "bart", "custom"])
def test_output_head_alone_does_not_identify_causal_embedding_names(
    monkeypatch,
    tmp_path,
    model_type,
):
    checkpoint = write_safetensors(
        tmp_path / "model.safetensors",
        [("lm_head.weight", [4], b"tied")],
        metadata={"model.embed_tokens.weight": "lm_head.weight"},
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "tie_word_embeddings": True,
            }
        )
    )
    monkeypatch.setattr(expect_module, "DeviceBuffer", FakeBuffer)
    expectation = expect_module.compute(FakeNotary(), checkpoint)
    assert expectation.names == ["lm_head.weight"]
    assert not expectation.tied


def test_unknown_t5_architecture_does_not_invent_a_generation_head(
    monkeypatch, tmp_path
):
    checkpoint = write_safetensors(
        tmp_path / "model.safetensors",
        [("shared.weight", [4], b"tied")],
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "t5",
                "architectures": ["CustomT5Model"],
                "tie_word_embeddings": True,
            }
        )
    )
    monkeypatch.setattr(expect_module, "DeviceBuffer", FakeBuffer)
    expectation = expect_module.compute(FakeNotary(), checkpoint)
    assert expectation.names == ["shared.weight"]
    assert not expectation.tied


@pytest.mark.parametrize("source", ["lm_head.weight", "model.embed_tokens.weight"])
@pytest.mark.parametrize("tie_output", [True, False, "true", None])
def test_causal_config_restores_ties_in_either_serialization_direction(
    monkeypatch,
    tmp_path,
    source,
    tie_output,
):
    checkpoint = write_safetensors(
        tmp_path / "model.safetensors",
        [(source, [4], b"tied")],
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "tie_word_embeddings": tie_output,
            }
        )
    )
    monkeypatch.setattr(expect_module, "DeviceBuffer", FakeBuffer)

    expectation = expect_module.compute(FakeNotary(), checkpoint)
    names = (
        ["lm_head.weight", "model.embed_tokens.weight"]
        if tie_output is True
        else [source]
    )
    assert expectation.names == names
    assert expectation.digests == (blake3_digest(b"tied") * len(names)).hex()


@pytest.mark.parametrize(
    "model_class",
    ["T5EncoderModel", "T5Model", "T5ForConditionalGeneration", "LlamaForCausalLM"],
)
@pytest.mark.parametrize("serialization", ["save_pretrained", "save_model"])
def test_real_saved_checkpoint_matches_runtime_state_dict(
    monkeypatch,
    tmp_path,
    model_class,
    serialization,
):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    safetensors = pytest.importorskip("safetensors.torch")
    if model_class.startswith("T5"):
        config = transformers.T5Config(
            vocab_size=16,
            d_model=8,
            d_kv=4,
            d_ff=16,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            tie_word_embeddings=True,
        )
    else:
        config = transformers.LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
    model = getattr(transformers, model_class)(config)
    if serialization == "save_pretrained":
        model.save_pretrained(tmp_path)
    else:
        config.architectures = [model_class]
        config.save_pretrained(tmp_path)
        safetensors.save_model(model, tmp_path / "model.safetensors")
    monkeypatch.setattr(expect_module, "DeviceBuffer", FakeBuffer)

    expectation = expect_module.compute(FakeNotary(), tmp_path)
    runtime = sorted(
        (name, tensor) for name, tensor in model.state_dict().items() if tensor.numel()
    )
    digests = b"".join(
        blake3_digest(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes())
        for _, tensor in runtime
    )
    assert expectation.names == [name for name, _ in runtime]
    assert expectation.digests == digests.hex()
    assert (
        expectation.model_root
        == blake3_digest(len(runtime).to_bytes(4, "little") + digests).hex()
    )


def test_receipt_file_loader_rejects_duplicate_keys_recursively(tmp_path):
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        '{"measurement":{"measurementDocument":"first","measurementDocument":"second"}}'
    )

    with pytest.raises(ValueError, match="duplicate field"):
        expect_module.load_measurement(receipt_path)


def test_receipt_file_loader_caps_input_before_json_parsing(monkeypatch, tmp_path):
    monkeypatch.setattr(expect_module, "MAX_EVIDENCE_FILE_BYTES", 32)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(b"{" + b" " * 31 + b"}")

    with pytest.raises(ValueError, match="32-byte evidence limit"):
        expect_module.load_measurement(receipt_path)

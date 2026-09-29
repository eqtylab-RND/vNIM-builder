# SPDX-License-Identifier: Apache-2.0
"""The safetensors reader, against files built in the test. No GPU required."""

import json
import struct

import pytest

from cuattest.safetensors import (
    SafetensorsError,
    load,
    load_with_aliases,
    tensors_in_file,
)


def write_st(path, tensors: dict[str, tuple[str, list, bytes]]):
    """tensors: name -> (dtype, shape, payload)"""
    header, blob, off = {}, b"", 0
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [off, off + len(payload)],
        }
        blob += payload
        off += len(payload)
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + blob)
    return path


def write_raw_st(path, header, payload=b""):
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def test_reads_tensors_and_skips_metadata(tmp_path):
    p = write_st(
        tmp_path / "m.safetensors",
        {"b.w": ("F32", [2], b"\x00" * 8), "a.w": ("BF16", [4], b"\x11" * 8)},
    )
    ts = tensors_in_file(p)
    assert {t.name for t in ts} == {"a.w", "b.w"}  # __metadata__ excluded
    assert all(t.nbytes == 8 for t in ts)


def test_load_sorts_by_name(tmp_path):
    p = write_st(
        tmp_path / "m.safetensors",
        {"z": ("U8", [1], b"\x01"), "a": ("U8", [1], b"\x02")},
    )
    assert [t.name for t in load(p)] == ["a", "z"]


def test_tensor_bytes_read_back_exactly(tmp_path):
    payload = bytes(range(16))
    p = write_st(tmp_path / "m.safetensors", {"x": ("U8", [16], payload)})
    assert load(p)[0].read() == payload


def test_shape_dtype_byte_range_mismatch_is_caught(tmp_path):
    """A header claiming a shape its byte range cannot hold is corrupt."""
    p = tmp_path / "bad.safetensors"
    header = {"x": {"dtype": "F32", "shape": [100], "data_offsets": [0, 8]}}
    hb = json.dumps(header).encode()
    p.write_bytes(struct.pack("<Q", len(hb)) + hb + b"\x00" * 8)
    with pytest.raises(SafetensorsError, match="but its range is"):
        tensors_in_file(p)


@pytest.mark.parametrize("shape", [[-1], [True], [1.5]])
def test_shape_dimensions_must_be_nonnegative_integers(tmp_path, shape):
    p = write_raw_st(
        tmp_path / "bad-shape.safetensors",
        {"x": {"dtype": "U8", "shape": shape, "data_offsets": [0, 1]}},
        b"x",
    )

    with pytest.raises(
        SafetensorsError, match="shape dimensions.*non-negative integers"
    ):
        tensors_in_file(p)


@pytest.mark.parametrize("offsets", [[-1, 0], [False, 1], [0.0, 1]])
def test_data_offsets_must_be_nonnegative_integers(tmp_path, offsets):
    p = write_raw_st(
        tmp_path / "bad-offset.safetensors",
        {"x": {"dtype": "U8", "shape": [1], "data_offsets": offsets}},
        b"x",
    )

    with pytest.raises(SafetensorsError, match="data offsets.*non-negative integers"):
        tensors_in_file(p)


def test_negative_offset_cannot_point_a_tensor_into_its_json_header(tmp_path):
    p = write_raw_st(
        tmp_path / "header-pointer.safetensors",
        {"x": {"dtype": "U8", "shape": [4], "data_offsets": [-4, 0]}},
        b"data",
    )

    with pytest.raises(SafetensorsError, match="data offsets.*non-negative"):
        tensors_in_file(p)


def test_duplicate_json_tensor_names_are_rejected_instead_of_overwritten(tmp_path):
    tensor = '{"dtype":"U8","shape":[1],"data_offsets":[0,1]}'
    encoded = f'{{"x":{tensor},"x":{tensor}}}'.encode()
    p = tmp_path / "duplicate-key.safetensors"
    p.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"x")

    with pytest.raises(SafetensorsError, match="duplicate key"):
        tensors_in_file(p)


@pytest.mark.parametrize(
    "header,payload,error",
    [
        (
            {
                "a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
                "b": {"dtype": "U8", "shape": [1], "data_offsets": [2, 3]},
            },
            b"abc",
            "leaves a gap",
        ),
        (
            {
                "a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                "b": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]},
            },
            b"ab",
            "overlaps an earlier tensor",
        ),
        (
            {"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}},
            b"ab",
            "ranges cover 1 data bytes.*contains 2",
        ),
        (
            {"a": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}},
            b"a",
            "ranges cover 2 data bytes.*contains 1",
        ),
    ],
)
def test_tensor_ranges_must_cover_the_data_section_exactly_once(
    tmp_path, header, payload, error
):
    p = write_raw_st(tmp_path / "bad-layout.safetensors", header, payload)

    with pytest.raises(SafetensorsError, match=error):
        tensors_in_file(p)


def test_truncated_file_is_rejected(tmp_path):
    p = tmp_path / "short.safetensors"
    p.write_bytes(b"\x01\x02")
    with pytest.raises(SafetensorsError, match="too short"):
        tensors_in_file(p)


def test_implausible_header_length_is_rejected(tmp_path):
    p = tmp_path / "huge.safetensors"
    p.write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(SafetensorsError, match="implausible header"):
        tensors_in_file(p)


def test_shard_index_is_followed(tmp_path):
    write_st(tmp_path / "s1.safetensors", {"a": ("U8", [1], b"\x01")})
    write_st(tmp_path / "s2.safetensors", {"b": ("U8", [1], b"\x02")})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
    )
    assert [t.name for t in load(tmp_path)] == ["a", "b"]


def test_index_naming_a_missing_shard_fails_loudly(tmp_path):
    write_st(tmp_path / "s1.safetensors", {"a": ("U8", [1], b"\x01")})
    idx = tmp_path / "model.safetensors.index.json"
    idx.write_text(
        json.dumps({"weight_map": {"a": "s1.safetensors", "b": "gone.safetensors"}})
    )
    with pytest.raises(SafetensorsError, match="missing shard"):
        load(idx)


def test_index_ignores_stale_tensor_not_in_weight_map(tmp_path):
    write_st(
        tmp_path / "s1.safetensors",
        {
            "wanted": ("U8", [1], b"\x01"),
            "stale": ("U8", [1], b"\x02"),
        },
    )
    idx = tmp_path / "model.safetensors.index.json"
    idx.write_text(json.dumps({"weight_map": {"wanted": "s1.safetensors"}}))

    assert [tensor.name for tensor in load(idx)] == ["wanted"]


def test_index_rejects_tensor_found_in_the_wrong_mapped_shard(tmp_path):
    write_st(tmp_path / "s1.safetensors", {"b": ("U8", [1], b"\x02")})
    write_st(
        tmp_path / "s2.safetensors",
        {
            "a": ("U8", [1], b"\x01"),
            "b": ("U8", [1], b"\x02"),
        },
    )
    idx = tmp_path / "model.safetensors.index.json"
    idx.write_text(
        json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
    )

    with pytest.raises(SafetensorsError, match="stored in .*weight_map assigns it"):
        load(idx)


@pytest.mark.parametrize(
    "filename",
    [
        "../outside.safetensors",
        "/tmp/outside.safetensors",
        "C:\\outside.safetensors",
        "\\\\server\\share\\outside.safetensors",
    ],
)
def test_index_rejects_absolute_and_parent_traversing_shard_paths(tmp_path, filename):
    idx = tmp_path / "model.safetensors.index.json"
    idx.write_text(json.dumps({"weight_map": {"a": filename}}))

    with pytest.raises(SafetensorsError, match="relative paths without"):
        load(idx)


def test_index_allows_an_ordinary_nested_relative_shard_path(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    write_st(nested / "s1.safetensors", {"a": ("U8", [1], b"x")})
    idx = tmp_path / "model.safetensors.index.json"
    idx.write_text(json.dumps({"weight_map": {"a": "nested/s1.safetensors"}}))

    tensors = load(idx)

    assert [tensor.name for tensor in tensors] == ["a"]
    assert tensors[0].read() == b"x"


def test_index_rejects_an_unrecognized_symlink_that_escapes_the_checkpoint(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.safetensors"
    write_st(outside, {"a": ("U8", [1], b"x")})
    (tmp_path / "escape.safetensors").symlink_to(outside)
    idx = tmp_path / "model.safetensors.index.json"
    idx.write_text(json.dumps({"weight_map": {"a": "escape.safetensors"}}))

    with pytest.raises(SafetensorsError, match="escapes the checkpoint"):
        load(idx)


def test_standard_hub_snapshot_symlink_into_its_blob_store_is_allowed(tmp_path):
    model_cache = tmp_path / "models--example--model"
    snapshot = model_cache / "snapshots" / "0123456789abcdef"
    blobs = model_cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    write_st(blobs / "deadbeef", {"a": ("U8", [1], b"x")})
    (snapshot / "s1.safetensors").symlink_to("../../blobs/deadbeef")
    idx = snapshot / "model.safetensors.index.json"
    idx.write_text(json.dumps({"weight_map": {"a": "s1.safetensors"}}))

    tensors = load(snapshot)

    assert [tensor.name for tensor in tensors] == ["a"]
    assert tensors[0].read() == b"x"


def test_free_form_metadata_is_not_treated_as_alias_provenance(tmp_path):
    path = write_st(tmp_path / "model.safetensors", {"weight": ("U8", [1], b"x")})
    header = {
        "weight": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]},
        "__metadata__": {"description": "weight", "view": "weight"},
    }
    write_raw_st(path, header, b"x")

    tensors, declared = load_with_aliases(path)

    assert [tensor.name for tensor in tensors] == ["weight"]
    assert declared == {}


def test_explicit_schema_exposes_only_full_span_aliases(tmp_path):
    write_st(
        tmp_path / "s1.safetensors",
        {"decoder.embed_tokens.weight": ("U8", [1], b"x")},
    )
    aliases = {
        "shared.weight": "decoder.embed_tokens.weight",
        "encoder.embed_tokens.weight": "decoder.embed_tokens.weight",
        "lm_head.weight": "decoder.embed_tokens.weight",
    }
    idx = tmp_path / "model.safetensors.index.json"
    idx.write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": 1,
                    "cuattest.aliases.v1": json.dumps(
                        {
                            alias: {"kind": "full-span", "source": source}
                            for alias, source in aliases.items()
                        }
                    ),
                },
                "weight_map": {"decoder.embed_tokens.weight": "s1.safetensors"},
            }
        )
    )

    tensors, declared = load_with_aliases(idx)

    assert [tensor.name for tensor in tensors] == ["decoder.embed_tokens.weight"]
    assert declared == aliases


def test_explicit_alias_schema_rejects_a_view_without_full_span_proof(tmp_path):
    path = write_st(tmp_path / "model.safetensors", {"base": ("U8", [8], b"12345678")})
    header = {
        "base": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]},
        "__metadata__": {
            "cuattest.aliases.v1": json.dumps(
                {"view": {"kind": "view", "source": "base"}}
            )
        },
    }
    write_raw_st(path, header, b"12345678")

    with pytest.raises(SafetensorsError, match="invalid full-span alias"):
        load_with_aliases(path)


def test_directory_selects_conventional_model_index_over_adapter_index(tmp_path):
    write_st(tmp_path / "model-shard.safetensors", {"model": ("U8", [1], b"m")})
    write_st(tmp_path / "adapter-shard.safetensors", {"adapter": ("U8", [1], b"a")})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model": "model-shard.safetensors"}})
    )
    (tmp_path / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"adapter": "adapter-shard.safetensors"}})
    )

    assert [tensor.name for tensor in load(tmp_path)] == ["model"]


def test_directory_rejects_multiple_nonconventional_safetensors_indexes(tmp_path):
    for name in ("first.safetensors.index.json", "second.safetensors.index.json"):
        (tmp_path / name).write_text(json.dumps({"weight_map": {}}))

    with pytest.raises(SafetensorsError, match="multiple safetensors indexes"):
        load(tmp_path)


def test_directory_ignores_unrelated_bin_index(tmp_path):
    write_st(tmp_path / "model.safetensors", {"weight": ("U8", [1], b"x")})
    (tmp_path / "pytorch_model.bin.index.json").write_text("not even JSON")

    assert [tensor.name for tensor in load(tmp_path)] == ["weight"]


def test_directory_prefers_conventional_model_file_over_adapter_index(tmp_path):
    write_st(tmp_path / "model.safetensors", {"model": ("U8", [1], b"m")})
    write_st(tmp_path / "adapter-shard.safetensors", {"adapter": ("U8", [1], b"a")})
    (tmp_path / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"adapter": "adapter-shard.safetensors"}})
    )

    assert [tensor.name for tensor in load(tmp_path)] == ["model"]


def test_direct_bin_index_is_not_accepted_as_a_safetensors_index(tmp_path):
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(json.dumps({"weight_map": {}}))

    with pytest.raises(SafetensorsError, match="not a safetensors file"):
        load(index)


def test_duplicate_tensor_names_across_shards_are_rejected(tmp_path):
    write_st(tmp_path / "s1.safetensors", {"duplicate": ("U8", [1], b"a")})
    write_st(tmp_path / "s2.safetensors", {"duplicate": ("U8", [1], b"b")})

    with pytest.raises(SafetensorsError, match="duplicate tensor names"):
        load(tmp_path)


def test_directory_without_safetensors_fails(tmp_path):
    with pytest.raises(SafetensorsError, match="no .safetensors"):
        load(tmp_path)

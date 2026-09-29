# SPDX-License-Identifier: Apache-2.0
"""Parsing of the tensor references clients send. No GPU required."""

import pytest

from cuattest.notary import NotaryError, TensorRef

RAW = bytes(range(64))


def test_plain_64_byte_handle():
    t = TensorRef(handle=RAW.hex(), nbytes=1024)
    assert t.raw_handle() == RAW


def test_torch_two_byte_header_is_stripped():
    """torch's _share_cuda_() prefixes two bytes before the real handle."""
    t = TensorRef(handle=(b"\xff\xee" + RAW).hex(), nbytes=1024)
    assert t.raw_handle() == RAW


def test_wrong_length_handle_is_rejected():
    with pytest.raises(NotaryError, match="expected 64"):
        TensorRef(handle=(b"\x01" * 40).hex(), nbytes=8).raw_handle()


def test_from_dict_defaults_offsets_to_zero():
    t = TensorRef.from_dict({"handle": RAW.hex(), "nbytes": 16, "device": 2})
    assert (t.seg_off, t.t_off, t.device) == (0, 0, 2)


def test_from_dict_reports_a_missing_field():
    with pytest.raises(NotaryError, match="bad tensor reference"):
        TensorRef.from_dict({"nbytes": 16})


def test_from_dict_reports_a_non_numeric_size():
    with pytest.raises(NotaryError, match="bad tensor reference"):
        TensorRef.from_dict({"handle": RAW.hex(), "nbytes": "big", "device": 0})


def test_from_dict_does_not_reflect_a_huge_handle_in_its_error():
    huge_handle = "feedface" * (1024 * 128)

    with pytest.raises(NotaryError) as caught:
        TensorRef.from_dict({"handle": huge_handle, "nbytes": "big", "device": 0})

    message = str(caught.value)
    assert message == "bad tensor reference: nbytes must be an integer"
    assert huge_handle not in message


def test_raw_handle_rejects_huge_text_before_hex_decoding():
    huge_handle = "ab" * (1024 * 1024)

    with pytest.raises(NotaryError) as caught:
        TensorRef(huge_handle, nbytes=1).raw_handle()

    message = str(caught.value)
    assert "expected 64 bytes" in message
    assert len(message) < 160
    assert huge_handle not in message


@pytest.mark.parametrize("field,value", [
    ("nbytes", -1), ("nbytes", 0), ("seg_off", -1), ("t_off", -1),
])
def test_negative_or_empty_spans_are_rejected(field, value):
    ref = {"handle": RAW.hex(), "nbytes": 16, "seg_off": 0, "t_off": 0, "device": 0}
    ref[field] = value
    with pytest.raises(NotaryError, match="positive|non-negative"):
        TensorRef.from_dict(ref)


def test_boolean_offsets_are_not_silently_coerced_to_integers():
    with pytest.raises(NotaryError, match="t_off must be an integer"):
        TensorRef.from_dict({"handle": RAW.hex(), "nbytes": 16,
                             "t_off": True, "device": 0})


def test_direct_tensor_ref_construction_still_requires_strict_integers():
    with pytest.raises(NotaryError, match="nbytes must be an integer"):
        TensorRef(handle=RAW.hex(), nbytes=True, device=0).validate_metadata()


def test_source_device_is_required_on_wire():
    with pytest.raises(NotaryError, match="bad tensor reference"):
        TensorRef.from_dict({"handle": RAW.hex(), "nbytes": 16})


@pytest.mark.parametrize("value", [True, 1, "GPU-not-a-uuid", "0", "GPU-" + "f" * 4096])
def test_invalid_routing_uuid_is_rejected(value):
    with pytest.raises(NotaryError, match="device_uuid"):
        TensorRef.from_dict({"handle": RAW.hex(), "nbytes": 16, "device": 0, "device_uuid": value})


def test_span_must_fit_the_imported_allocation():
    t = TensorRef(handle=RAW.hex(), nbytes=17, seg_off=32, t_off=16, device=0)
    with pytest.raises(NotaryError, match="outside the imported allocation"):
        t.pointer_in(mapped_base=0x1000, allocation_base=0x1000, allocation_nbytes=64)


def test_checked_pointer_addition_rejects_u64_wrap():
    t = TensorRef(handle=RAW.hex(), nbytes=2, device=0)
    with pytest.raises(NotaryError, match="overflows"):
        t.pointer_in(mapped_base=(1 << 64) - 1,
                     allocation_base=(1 << 64) - 1, allocation_nbytes=1)

    offset_wrap = TensorRef(
        handle=RAW.hex(), nbytes=1, seg_off=(1 << 64) - 1, t_off=1, device=0
    )
    with pytest.raises(NotaryError, match="offsets overflow"):
        offset_wrap.pointer_in(mapped_base=0, allocation_base=0, allocation_nbytes=1)


def test_producer_device_ordinal_is_only_validated_not_compared():
    t = TensorRef(handle=RAW.hex(), nbytes=16, device=1)
    # CUDA_VISIBLE_DEVICES can map producer cuda:1 to notary cuda:0. The
    # imported pointer's driver-reported owner is authoritative instead.
    t.validate_metadata()

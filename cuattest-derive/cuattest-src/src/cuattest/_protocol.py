# SPDX-License-Identifier: Apache-2.0
"""Constants and validation shared by signed-measurement producers/consumers."""

from __future__ import annotations

import json

# The client chooses the IPC handles and byte ranges. Bounds and device checks
# establish memory safety, but they cannot prove that an inference runtime used
# those bytes or that a submitted range has tensor semantics. Keep this claim
# byte-for-byte aligned with write_measurement_prefix() in the CUDA kernel.
MEASUREMENT_CLAIM = (
    "modelHash and modelCID were computed in-GPU from client-submitted, "
    "bounds-checked VRAM spans; this measurement document was assembled and "
    "signed in-kernel"
)

MEASUREMENT_HASH_SCHEME = (
    "BLAKE3 per submitted span (= EQTY raw CID hash); "
    "modelRoot=BLAKE3(LE32(N)||spanDigests)"
)
MEASUREMENT_OPERATION = "gpu-hash-submitted-vram-spans"

# This is an exact schema, not merely a list of required fields.  Permitting a
# substituted kernel to add assertions would let it turn a narrow, verified
# statement into a stronger signed claim.  Duplicate JSON keys are similarly
# unsafe because different verifier implementations can select different
# occurrences of the same field.
MEASUREMENT_DOCUMENT_FIELDS = frozenset(
    {
        "claim",
        "cubinCID",
        "device",
        "gpuDID",
        "hashScheme",
        "kernelCID",
        "measuredAt",
        "model",
        "modelCID",
        "modelHash",
        "operation",
        "tensorCount",
    }
)
# Statements live inside the versioned manifest envelope, not beside the
# measurement document. This set is the outer receipt only; the manifest's own
# contents are checked by _statements.validate_statement_graph.
KERNEL_RECEIPT_FIELDS = frozenset(
    {"measurementDocument", "measurementSignature", "modelRoot", "manifest"}
)


def _object_without_duplicate_fields(pairs: list[tuple[str, object]]) -> dict:
    """Build one JSON object while refusing parser-dependent duplicate keys."""
    result = {}
    for field_name, value in pairs:
        if field_name in result:
            # Do not reflect an attacker-controlled field name into an HTTP
            # error response; it may occupy most of the document size limit.
            raise ValueError("JSON object contains a duplicate field")
        result[field_name] = value
    return result


def loads_no_duplicate_fields(encoded: str | bytes) -> object:
    """Decode untrusted JSON while rejecting duplicate keys at every depth."""
    return json.loads(encoded, object_pairs_hook=_object_without_duplicate_fields)


def parse_measurement_document(encoded: bytes) -> dict:
    """Decode a duplicate-free document containing exactly the signed schema."""
    document = loads_no_duplicate_fields(encoded)
    if not isinstance(document, dict):
        # The bytes have the right Python input type but the decoded JSON has
        # the wrong protocol shape, so this is a value/schema error.
        raise ValueError(  # noqa: TRY004
            "signed measurement document must be a JSON object"
        )

    actual_fields = frozenset(document)
    if actual_fields != MEASUREMENT_DOCUMENT_FIELDS:
        missing_count = len(MEASUREMENT_DOCUMENT_FIELDS - actual_fields)
        unexpected_count = len(actual_fields - MEASUREMENT_DOCUMENT_FIELDS)
        raise ValueError(
            "signed measurement document does not match the exact schema "
            f"({missing_count} missing field(s), "
            f"{unexpected_count} unexpected field(s))"
        )

    # JSON booleans and integral floats compare equal to Python integers.  An
    # exact shared type check keeps host publication and independent verifier
    # acceptance identical instead of relying on == for protocol values.
    for field_name in MEASUREMENT_DOCUMENT_FIELDS - {"tensorCount"}:
        if type(document[field_name]) is not str:
            raise ValueError(f"signed measurement field {field_name} must be a string")
    if type(document["tensorCount"]) is not int:
        raise ValueError("signed measurement field tensorCount must be an integer")
    return document


def parse_kernel_receipt(encoded: str) -> dict:
    """Decode the duplicate-free, exact outer object emitted by the CUBIN."""
    receipt = loads_no_duplicate_fields(encoded)
    if not isinstance(receipt, dict):
        raise ValueError("kernel receipt must be a JSON object")  # noqa: TRY004
    if frozenset(receipt) != KERNEL_RECEIPT_FIELDS:
        raise ValueError("kernel receipt does not match the exact outer schema")
    return receipt

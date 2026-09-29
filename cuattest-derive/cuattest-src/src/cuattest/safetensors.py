# SPDX-License-Identifier: Apache-2.0
"""A minimal safetensors reader — header parsing only, no dependencies.

The format is simple enough to read directly, and doing so keeps this package
dependency-free: an 8-byte little-endian header length, that many bytes of
JSON describing every tensor, then the raw data.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from blake3 import blake3

# Bits per element, for validating that a tensor's declared shape and dtype
# agree with the byte range the header gives it. Keeping the sub-byte dtypes
# as bits mirrors safetensors' own validation instead of rounding them down.
DTYPE_BITS = {
    "F64": 64,
    "I64": 64,
    "U64": 64,
    "C64": 64,
    "F32": 32,
    "I32": 32,
    "U32": 32,
    "F16": 16,
    "BF16": 16,
    "I16": 16,
    "U16": 16,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "F8_E8M0": 8,
    "F8_E4M3FNUZ": 8,
    "F8_E5M2FNUZ": 8,
    "I8": 8,
    "U8": 8,
    "BOOL": 8,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "F4": 4,
}

# Safetensors __metadata__ is otherwise an untyped, free-form string map.
# Only this versioned key has alias semantics in cuAttest. Its JSON value maps
# a missing name to {"kind": "full-span", "source": "stored.name"}; the
# explicit kind prevents metadata emitted for an overlapping view from being
# misread as proof that the view covers its source's entire byte range.
_FULL_SPAN_ALIASES_KEY = "cuattest.aliases.v1"


class SafetensorsError(RuntimeError):
    """The file is not a readable safetensors checkpoint."""


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    """Build a JSON object while enforcing safetensors' unique-key rule."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key in JSON object")
        result[key] = value
    return result


@dataclass(frozen=True)
class DiskTensor:
    name: str
    path: Path
    offset: int  # absolute byte offset in `path`
    nbytes: int
    dtype: str
    shape: tuple[int, ...]

    def read(self) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read(self.nbytes)
        if len(data) != self.nbytes:
            raise SafetensorsError(
                f"{self.name}: wanted {self.nbytes} bytes at {self.offset}, got {len(data)}"
            )
        return data


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header, data_start). `data_start` is where tensor bytes begin."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise SafetensorsError(f"{path}: too short to be safetensors")
        (n,) = struct.unpack("<Q", raw_len)
        if not 0 < n < 512 * 1024 * 1024:
            raise SafetensorsError(f"{path}: implausible header length {n}")
        blob = f.read(n)
        if len(blob) != n:
            raise SafetensorsError(f"{path}: header truncated")
    try:
        header = json.loads(blob, object_pairs_hook=_unique_json_object)
    except ValueError as e:
        raise SafetensorsError(f"{path}: header is not JSON: {e}") from e
    if not isinstance(header, dict):
        raise SafetensorsError(f"{path}: safetensors header must be a JSON object")
    return header, 8 + n


def _tensors_and_metadata(path: Path) -> tuple[list[DiskTensor], dict[str, str]]:
    header, data_start = read_header(path)
    try:
        data_nbytes = path.stat().st_size - data_start
    except OSError as error:
        raise SafetensorsError(f"{path}: cannot inspect file size: {error}") from error
    if data_nbytes < 0:  # A concurrent truncation after read_header().
        raise SafetensorsError(f"{path}: file ends before its declared data section")

    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise SafetensorsError(
            f"{path}: __metadata__ must contain only string keys and values"
        )

    out: list[DiskTensor] = []
    ranges: list[tuple[int, int, str]] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            raise SafetensorsError(
                f"{path}: malformed entry for {name!r}: expected an object"
            )
        try:
            dtype = meta["dtype"]
            raw_shape = meta["shape"]
            raw_offsets = meta["data_offsets"]
        except KeyError as e:
            raise SafetensorsError(f"{path}: malformed entry for {name!r}: {e}") from e

        if not isinstance(dtype, str) or dtype not in DTYPE_BITS:
            raise SafetensorsError(f"{path}: {name} has unsupported dtype {dtype!r}")
        if not isinstance(raw_shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in raw_shape
        ):
            # bool is an int subclass in Python. Exact type checks keep true
            # and false from becoming one- and zero-element dimensions.
            raise SafetensorsError(
                f"{path}: {name} shape dimensions must be non-negative integers"
            )
        if (
            not isinstance(raw_offsets, list)
            or len(raw_offsets) != 2
            or any(type(offset) is not int or offset < 0 for offset in raw_offsets)
        ):
            raise SafetensorsError(
                f"{path}: {name} data offsets must be two non-negative integers"
            )

        shape = tuple(raw_shape)
        start, end = raw_offsets
        if end < start:
            raise SafetensorsError(f"{path}: {name} has a negative byte range")
        nbytes = end - start

        # Cross-check the declared shape against the byte range; a mismatch
        # means the header is lying and every digest downstream would be wrong.
        expected_bits = DTYPE_BITS[dtype]
        for dimension in shape:
            expected_bits *= dimension
        if expected_bits % 8:
            raise SafetensorsError(
                f"{path}: {name} declares a sub-byte tensor not aligned to a whole byte"
            )
        expected_nbytes = expected_bits // 8
        if expected_nbytes != nbytes:
            raise SafetensorsError(
                f"{path}: {name} declares {dtype}{list(shape)} ({expected_nbytes} bytes) "
                f"but its range is {nbytes} bytes"
            )

        out.append(DiskTensor(name, path, data_start + start, nbytes, dtype, shape))
        ranges.append((start, end, name))

    # Safetensors requires the byte buffer to be covered exactly once. Sorting
    # by offsets makes this a linear scan after O(n log n) ordering and rejects
    # leading/trailing gaps, internal holes, overlaps, and out-of-file ranges.
    next_offset = 0
    for start, end, name in sorted(ranges):
        if start != next_offset:
            defect = (
                "overlaps an earlier tensor" if start < next_offset else "leaves a gap"
            )
            raise SafetensorsError(
                f"{path}: {name} starts at {start}, which {defect} at byte {next_offset}"
            )
        next_offset = end
    if next_offset != data_nbytes:
        raise SafetensorsError(
            f"{path}: tensor ranges cover {next_offset} data bytes, "
            f"but the file contains {data_nbytes}"
        )

    return out, metadata


def tensors_in_file(path: Path) -> list[DiskTensor]:
    """Return tensors only after validating the complete safetensors layout."""
    tensors, _ = _tensors_and_metadata(path)
    return tensors


def load(path: str | Path) -> list[DiskTensor]:
    """Every tensor in a checkpoint: one file, a shard index, or a directory.

    Returned in sorted name order — the same order the notary measures in, so
    the folds are comparable.
    """
    tensors, _ = load_with_aliases(path)
    return tensors


def load_with_aliases(path: str | Path) -> tuple[list[DiskTensor], dict[str, str]]:
    """Return tensors and explicitly declared full-span aliases.

    Generic safetensors metadata such as ``{"description": "weight"}`` has no
    alias provenance, and ``save_model`` metadata does not carry a dropped
    view's offset or extent. Consequently only ``cuattest.aliases.v1`` is read
    here; architecture-aware aliases are established separately from model
    configuration in :mod:`cuattest.expect`.
    """
    p = Path(path)
    metadata_blocks: list[dict] = []
    if p.is_file() and p.suffix == ".safetensors":
        found, metadata = _tensors_and_metadata(p)
        metadata_blocks.append(metadata)
    elif p.is_file() and p.name.endswith(".safetensors.index.json"):
        found, metadata_blocks = _from_index(p)
    elif p.is_dir():
        index = _select_directory_index(p)
        if index is not None:
            found, metadata_blocks = _from_index(index)
        elif (p / "model.safetensors").is_file():
            # Prefer the conventional unsharded model over an unrelated index
            # or auxiliary safetensors files in the same repository checkout.
            found, metadata = _tensors_and_metadata(p / "model.safetensors")
            metadata_blocks.append(metadata)
        else:
            shards = sorted(p.glob("*.safetensors"))
            if not shards:
                raise SafetensorsError(f"no .safetensors files in {p}")
            found = []
            for shard in shards:
                tensors, metadata = _tensors_and_metadata(shard)
                found.extend(tensors)
                metadata_blocks.append(metadata)
    else:
        raise SafetensorsError(f"{p} is not a safetensors file, index or directory")

    # Count once. Calling names.count() for every entry made a malformed large
    # checkpoint's duplicate-name error path quadratic.
    counts = Counter(tensor.name for tensor in found)
    dupes: list[str] = []
    for name, count in counts.items():
        if count > 1:
            dupes.append(name)
            if len(dupes) == 5:
                break
    if dupes:
        raise SafetensorsError(f"duplicate tensor names across shards: {dupes}")

    tensor_names = set(counts)
    aliases: dict[str, str] = {}
    for metadata in metadata_blocks:
        raw_aliases = metadata.get(_FULL_SPAN_ALIASES_KEY)
        if raw_aliases is None:
            continue
        if not isinstance(raw_aliases, str):
            raise SafetensorsError(
                f"{_FULL_SPAN_ALIASES_KEY} metadata must be JSON text"
            )
        try:
            declared = json.loads(raw_aliases, object_pairs_hook=_unique_json_object)
        except ValueError as error:
            raise SafetensorsError(
                f"invalid {_FULL_SPAN_ALIASES_KEY} metadata: {error}"
            ) from error
        if not isinstance(declared, dict):
            raise SafetensorsError(
                f"{_FULL_SPAN_ALIASES_KEY} metadata must be a JSON object"
            )
        for alias, declaration in declared.items():
            if (
                not isinstance(alias, str)
                or not alias
                or alias in tensor_names
                or not isinstance(declaration, dict)
                or set(declaration) != {"kind", "source"}
                or declaration.get("kind") != "full-span"
                or not isinstance(declaration.get("source"), str)
                or declaration["source"] not in tensor_names
            ):
                raise SafetensorsError(
                    f"invalid full-span alias declaration for {alias!r}"
                )
            stored_name = declaration["source"]
            previous = aliases.get(alias)
            if previous is not None and previous != stored_name:
                raise SafetensorsError(
                    f"conflicting full-span alias metadata for {alias!r}: "
                    f"{previous!r} and {stored_name!r}"
                )
            aliases[alias] = stored_name

    return sorted(found, key=lambda tensor: tensor.name), aliases


def _select_directory_index(directory: Path) -> Path | None:
    """Choose a safetensors index without silently selecting another artifact."""
    conventional = directory / "model.safetensors.index.json"
    if conventional.is_file():
        # A directory may also contain an adapter index. The conventional model
        # name is an explicit selection and must not depend on lexical order.
        return conventional
    if (directory / "model.safetensors").is_file():
        return None
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        names = [index.name for index in indexes[:5]]
        raise SafetensorsError(
            f"multiple safetensors indexes in {directory}: {names}; "
            "pass the intended index path explicitly"
        )
    return indexes[0] if indexes else None


def _confined_shard_path(index_path: Path, filename: str) -> Path:
    """Resolve a safe lexical entry, including standard Hub blob symlinks."""
    # PureWindowsPath is intentional even on POSIX: it recognizes drive,
    # rooted, UNC, and backslash-separated traversal forms before Path joins
    # the untrusted value to the checkpoint directory.
    windows_path = PureWindowsPath(filename)
    native_path = Path(filename)
    if (
        not filename
        or windows_path.drive
        or windows_path.root
        or native_path.is_absolute()
        or ".." in windows_path.parts
        or ".." in native_path.parts
    ):
        raise SafetensorsError(
            f"{index_path}: shard filenames must be relative paths without '..' components"
        )

    checkpoint_root = index_path.parent.resolve()
    try:
        candidate = (index_path.parent / native_path).resolve()
    except (OSError, RuntimeError) as error:
        raise SafetensorsError(
            f"{index_path}: cannot resolve shard path {filename!r}: {error}"
        ) from error
    try:
        candidate.relative_to(checkpoint_root)
    except ValueError:
        if not _is_same_hub_cache_blob(index_path, candidate):
            raise SafetensorsError(
                f"{index_path}: shard path escapes the checkpoint directory"
            ) from None
    return candidate


def _is_same_hub_cache_blob(index_path: Path, candidate: Path) -> bool:
    """Allow a snapshot symlink only into its own Hugging Face blob store."""
    snapshot = index_path.parent
    snapshots = snapshot.parent
    model_cache = snapshots.parent
    if (
        snapshots.name != "snapshots"
        or not snapshot.name
        or not model_cache.name.startswith("models--")
    ):
        return False

    # The normal Hub layout has a real models--ORG--NAME/blobs directory. Do
    # not let an attacker turn this narrow compatibility exception into an
    # arbitrary escape by replacing blobs itself with another symlink.
    blob_root = model_cache / "blobs"
    if blob_root.is_symlink() or not blob_root.is_dir():
        return False
    try:
        resolved_model_cache = model_cache.resolve()
        resolved_blob_root = blob_root.resolve()
        resolved_blob_root.relative_to(resolved_model_cache)
        candidate.relative_to(resolved_blob_root)
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _from_index(index_path: Path) -> tuple[list[DiskTensor], list[dict]]:
    try:
        index = json.loads(
            index_path.read_text(), object_pairs_hook=_unique_json_object
        )
        if not isinstance(index, dict):
            raise TypeError("index must be an object")
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict):
            raise TypeError("weight_map must be an object")
        for name, filename in weight_map.items():
            if not isinstance(name, str) or not isinstance(filename, str):
                raise TypeError("weight_map names and shard filenames must be strings")
        index_metadata = index.get("metadata", {})
        if not isinstance(index_metadata, dict):
            raise TypeError("metadata must be an object")
    except (ValueError, KeyError, TypeError) as e:
        raise SafetensorsError(f"{index_path}: not a shard index: {e}") from e

    wanted = set(weight_map)
    shard_paths = {
        filename: _confined_shard_path(index_path, filename)
        for filename in set(weight_map.values())
    }
    by_shard: dict[Path, list[DiskTensor]] = {}
    metadata_blocks = [index_metadata]
    for shard in sorted(set(shard_paths.values())):
        if not shard.is_file():
            raise SafetensorsError(
                f"{index_path} references a missing shard: {shard.name}"
            )
        tensors, metadata = _tensors_and_metadata(shard)
        by_shard[shard] = tensors
        metadata_blocks.append(metadata)

    found = []
    for shard, tensors in by_shard.items():
        for tensor in tensors:
            assigned = weight_map.get(tensor.name)
            if assigned is None:
                # Shards are sometimes reused while regenerating an index. An
                # actual loader follows weight_map, so stale extras must not be
                # folded into the expected runtime state.
                continue
            assigned_path = shard_paths[assigned]
            if assigned_path != shard:
                raise SafetensorsError(
                    f"{index_path}: tensor {tensor.name!r} is stored in {shard.name} "
                    f"but weight_map assigns it to {assigned_path.name}"
                )
            found.append(tensor)
    missing = wanted - {t.name for t in found}
    if missing:
        raise SafetensorsError(
            f"{index_path} lists {len(missing)} tensor(s) absent from the shards, "
            f"e.g. {sorted(missing)[:3]}"
        )
    return found, metadata_blocks


def model_root(tensors: list[DiskTensor], *, chunk: int = 8 << 20) -> bytes:
    """Fold the same model root the GPU computes, but over the file on disk.

    Identical algorithm to the kernel's: BLAKE3(LE32(N) || per-span BLAKE3
    digests), in the order the spans are submitted. So a faithful load makes
    this equal to the GPU's measured root, and any difference is real rather
    than an artefact of two hashing schemes.

    Tensors are streamed in bounded chunks; a multi-gigabyte shard must not
    have to be resident to be hashed.
    """
    from ._hosthash import blake3_digest

    digests = []
    for tensor in tensors:
        hasher = blake3()
        remaining = tensor.nbytes
        with open(tensor.path, "rb") as handle:
            handle.seek(tensor.offset)
            while remaining:
                block = handle.read(min(chunk, remaining))
                if not block:
                    raise SafetensorsError(
                        f"{tensor.name}: wanted {tensor.nbytes} bytes at "
                        f"{tensor.offset}, file ended {remaining} short"
                    )
                hasher.update(block)
                remaining -= len(block)
        digests.append(hasher.digest())
    return blake3_digest(len(digests).to_bytes(4, "little") + b"".join(digests))


def model_cid(tensors: list[DiskTensor]) -> str:
    """`urn:cid:` naming the on-disk model root, for the host's state report."""
    from . import ids

    return f"urn:cid:{ids.raw_cid(model_root(tensors))}"

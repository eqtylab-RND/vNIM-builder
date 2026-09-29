# SPDX-License-Identifier: Apache-2.0
"""Independent CPU/GPU crypto oracles; opt in with CUATTEST_TEST_CRYPTO=1.

No CUDA access or optional oracle imports on ordinary CPU-only test runs.
Once opted in, missing CUDA/NVRTC/oracles are errors, not misleading skips.
See TEST.md for installation, all-device execution, and coverage limitations.
"""

import ctypes
import hashlib
import hmac
import json
import os
import random
import struct
from contextlib import closing, contextmanager
from functools import cache
from pathlib import Path

import pytest

from cuattest import kernel
from cuattest._cuda import Cuda, DeviceBuffer
from cuattest._nvrtc import Nvrtc
from cuattest.expect import verify_evidence
from cuattest.notary import Notary

ENABLED = os.environ.get("CUATTEST_TEST_CRYPTO") == "1"
pytestmark = pytest.mark.skipif(not ENABLED, reason="set CUATTEST_TEST_CRYPTO=1")
BLOCK_COUNT = 4096
BLOCK_BYTES = 1024
SEED = 0xC0A77E57


def pytest_generate_tests(metafunc):
    if "crypto_device" not in metafunc.fixturenames:
        return
    devices = [0]
    if ENABLED:
        cu = Cuda()
        cu.init()
        count = cu.device_count()
        selection = os.environ.get("CUATTEST_CRYPTO_DEVICES", "0")
        try:
            devices = (
                list(range(count))
                if selection == "all"
                else [int(v) for v in selection.split(",")]
            )
        except ValueError as error:
            raise pytest.UsageError(
                "CUATTEST_CRYPTO_DEVICES must be all or comma-separated ordinals"
            ) from error
        if (
            not devices
            or len(set(devices)) != len(devices)
            or any(not 0 <= d < count for d in devices)
        ):
            raise pytest.UsageError(
                "CUATTEST_CRYPTO_DEVICES must select distinct visible GPUs"
            )
    metafunc.parametrize(
        "crypto_device", devices, ids=lambda d: f"cuda{d}", scope="module"
    )


@pytest.fixture(scope="module", params=["native", "fallback"])
def gpu_notary(request, crypto_device):
    # Explicitly exercise both real host paths, rather than accidentally
    # counting an unavailable native extension as a second fallback run.
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(
            "CUATTEST_DISABLE_NATIVE_HOST", "1" if request.param == "fallback" else "0"
        )
        with closing(Notary(device=crypto_device)) as notary:
            assert (notary._native_runner is None) == (request.param == "fallback")
            yield notary


@contextmanager
def device_arena(notary):
    """Keep every uploaded byte alive through synchronization, including errors."""
    with notary._activate():
        buffers = []

        def upload(data):
            buffer = DeviceBuffer.from_bytes(notary.cu, data)
            buffers.append(buffer)
            # This is a real host/device equality check, not two independently
            # generated random streams assumed to have produced the same data.
            assert buffer.read() == data
            return buffer

        try:
            yield upload
        finally:
            try:
                notary.cu.sync()
            except BaseException as error:  # noqa: BLE001 - interrupts also require GPU quarantine
                for buffer in buffers:
                    buffer.ptr = 0
                notary._abort_unconfirmed_gpu_work(error)
            finally:
                for buffer in reversed(buffers):
                    buffer.close()


@cache
def compile_oracle(arch, source):
    # An in-memory, test-only CUBIN: never overwrite the production cache or
    # change its source identity to expose test scalars/nonces to a service.
    adapters = Path(__file__).with_name("_crypto_oracle.cu").read_text()
    return Nvrtc().compile_cubin(source + "\n" + adapters, arch)


@pytest.fixture(scope="module")
def oracle(gpu_notary):
    n = gpu_notary
    cubin = compile_oracle(n.arch, kernel.kernel_source())
    with n._activate():
        module = n.cu.module_load(cubin)
    try:
        # This module's known seed is test data, unrelated to the notary key.
        with device_arena(n) as upload:
            seed = upload(bytes(32))
            public = upload(bytes(64))
            n.cu.launch(
                n.cu.function(module, "keygen_kernel"), 1, 1,
                [
                    ctypes.c_uint64(n._ctx_dev.ptr), ctypes.c_uint64(seed.ptr),
                    ctypes.c_uint64(public.ptr), ctypes.c_uint64(public.ptr + 32),
                ],
            )
        yield module
    finally:
        if not n._closed:
            with n._activate():
                try:
                    n.cu.sync()
                except BaseException as error:  # noqa: BLE001 - preserve quarantine on interrupts
                    n._abort_unconfirmed_gpu_work(error)
                # The strict sanitizer module-leak check also applies to this
                # test-only module; Notary.close owns only its production one.
                n.cu.module_unload(module)


def run_oracle(n, module, name, blobs, count, stride, *, context=False, extra=()):
    with device_arena(n) as upload:
        inputs = [upload(blob) for blob in blobs]
        output = upload(b"\xa5" * (count * stride))
        pointers = (
            ([n._ctx_dev.ptr] if context else [])
            + [b.ptr for b in inputs]
            + [output.ptr]
        )
        n.cu.launch(
            n.cu.function(module, name),
            min(64, (count + 31) // 32),
            32,
            [
                *(ctypes.c_uint64(p) for p in pointers),
                ctypes.c_int(count),
                *(ctypes.c_int(v) for v in extra),
            ],
        )
        n.cu.sync()
        raw = output.read()
    return [raw[i * stride : (i + 1) * stride] for i in range(count)]


@cache
def blocks():
    rng = random.Random(SEED)  # Reproducible public vectors, NEVER key entropy.
    values = [
        bytes(BLOCK_BYTES),
        b"\xff" * BLOCK_BYTES,
        bytes(range(256)) * (BLOCK_BYTES // 256),
        b"\x55\xaa" * (BLOCK_BYTES // 2),
    ]
    values.extend(
        i.to_bytes(4, "little") + rng.randbytes(BLOCK_BYTES - 4)
        for i in range(4, BLOCK_COUNT)
    )
    assert len(values) == len(set(values)) == BLOCK_COUNT
    return tuple(values)


@cache
def boundary_messages():
    rng = random.Random(SEED + 1)
    sizes = (
        0,
        1,
        3,
        31,
        32,
        33,
        55,
        56,
        57,
        63,
        64,
        65,
        119,
        120,
        127,
        128,
        129,
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
        127 * 1024,
        128 * 1024,
        129 * 1024,
        255 * 1024 - 1,
        256 * 1024,
        257 * 1024 + 1,
        511 * 1024,
        512 * 1024 + 1,
    )
    return tuple(rng.randbytes(size) for size in sizes)


def pack_messages(messages, unaligned=False):
    packed = bytearray()
    offsets = []
    for i, message in enumerate(messages):
        # Cover ALL sixteen address residues, including the uint4 fast path,
        # without changing any byte within the logical input span.
        residue = i % 16 if unaligned else 0
        packed.extend(b"\xa5" * ((residue - len(packed)) % 16))
        offsets.append(len(packed))
        packed.extend(message)
    return bytes(packed), offsets


def check_hash_oracles(n, module, messages):
    from blake3 import blake3

    packed, offsets = pack_messages(messages, unaligned=True)
    rng = random.Random(SEED + 2)
    keys = [rng.randbytes(32) for _ in messages]
    records = run_oracle(
        n,
        module,
        "crypto_test_hashes",
        [
            packed,
            struct.pack(f"<{len(messages)}Q", *offsets),
            struct.pack(f"<{len(messages)}Q", *(len(m) for m in messages)),
            b"".join(keys),
        ],
        len(messages),
        192,
    )
    for i, (message, key, record) in enumerate(zip(messages, keys, records)):
        sha = hashlib.sha256(message).digest()
        mac = hmac.digest(key, message, "sha256")
        expected = (
            sha,
            sha,
            blake3(message).digest(),
            mac,
            mac,
            hmac.digest(key, sha, "sha256"),
        )
        labels = (
            "SHA256",
            "SHA256 streaming",
            "BLAKE3 streaming",
            "HMAC",
            "HMAC aliased key",
            "HMAC aliased message",
        )
        for j, (reference, label) in enumerate(zip(expected, labels)):
            assert record[j * 32 : (j + 1) * 32] == reference, (
                f"{label}: vector {i}, {len(message)} bytes"
            )


def production_measure(n, messages, *, unaligned=False):
    from blake3 import blake3

    packed, offsets = pack_messages(messages, unaligned)
    with device_arena(n) as upload:
        data = upload(packed)
        result = n._launch_fused_active(
            [
                (data.ptr + offset, len(message))
                for offset, message in zip(offsets, messages)
            ],
            "2026-09-07T00:00:00Z",
            b"crypto-differential",
        )
    expected = b"".join(blake3(m).digest() for m in messages)
    # Compare EACH leaf/tensor digest and the final fold. Root determinism or
    # CPU hashing of GPU-returned leaves alone cannot establish correctness.
    for i in range(len(messages)):
        assert result.roots[i * 32 : (i + 1) * 32] == expected[i * 32 : (i + 1) * 32], (
            f"tensor {i}"
        )
    assert len(result.roots) == len(expected)
    root = blake3(struct.pack("<I", len(messages)) + expected).digest()
    assert result.model_root == root
    receipt = json.loads(result.receipt)
    receipt["gpu_pubkey_uncompressed"] = n.info.gpu_pubkey_uncompressed
    verified = verify_evidence(receipt, trusted_pubkey=n.info.gpu_pubkey_uncompressed)
    assert verified.model_root == root.hex() and verified.tensor_count == len(messages)
    return result


def test_thousands_of_identical_host_device_1kib_blocks(gpu_notary, oracle):
    check_hash_oracles(gpu_notary, oracle, blocks())


def test_padding_chunk_tile_and_odd_tree_boundaries(gpu_notary, oracle):
    messages = boundary_messages()
    check_hash_oracles(gpu_notary, oracle, messages)
    production_measure(gpu_notary, messages, unaligned=True)


def test_production_aligned_partial_chunks_and_odd_trees(gpu_notary):
    # Full 1-KiB inputs do not exercise the aligned paired-load path followed
    # by a partial final chunk/odd tile. Repeat the mixed-depth corpus with
    # every tensor aligned, independently of the unaligned fallback above.
    production_measure(gpu_notary, boundary_messages())


@pytest.mark.parametrize("unaligned", [False, True], ids=["aligned", "all_alignments"])
def test_production_4096_tensor_digests_and_final_model_hash(gpu_notary, unaligned):
    # 4096 digests + the LE32 count also straddle the 128-KiB boundary in
    # the final streaming fold, independently of individual 1-KiB inputs.
    production_measure(gpu_notary, blocks(), unaligned=unaligned)


def test_same_blocks_as_one_contiguous_blake3_tree(gpu_notary):
    # Hashing independent blocks and hashing their concatenation are NOT the
    # same operation: nonzero chunk counters and cross-tile parents matter.
    production_measure(gpu_notary, [b"".join(blocks())])


def test_single_bit_mutation_changes_exactly_one_tensor_digest(gpu_notary):
    original = production_measure(gpu_notary, blocks())
    changed = list(blocks())
    index = 2051
    data = bytearray(changed[index])
    data[511] ^= 0x80
    changed[index] = bytes(data)
    modified = production_measure(gpu_notary, changed)
    assert original.roots[: index * 32] == modified.roots[: index * 32]
    assert original.roots[(index + 1) * 32 :] == modified.roots[(index + 1) * 32 :]
    assert (
        original.roots[index * 32 : (index + 1) * 32]
        != modified.roots[index * 32 : (index + 1) * 32]
    )
    assert original.model_root != modified.model_root


def arithmetic_pairs(modulus):
    # Exercise carries/borrows across EVERY 64-bit limb and full-width
    # products near m^2. CPU constants come from an independent curve library.
    edge = [0, 1, 2, modulus - 2, modulus - 1]
    edge += [
        ((1 << bit) + delta) % modulus
        for bit in (32, 64, 128, 192, 255)
        for delta in (-1, 0, 1)
    ]
    pairs = [(a, b) for a in edge for b in edge]
    rng = random.Random(SEED + 3)
    pairs += [(rng.randrange(modulus), rng.randrange(modulus)) for _ in range(256)]
    return pairs


@pytest.mark.parametrize("use_order", [False, True], ids=["field_prime", "group_order"])
def test_p256_arithmetic_matches_python_integers(gpu_notary, oracle, use_order):
    from ecdsa import NIST256p

    modulus = NIST256p.order if use_order else NIST256p.curve.p()
    pairs = arithmetic_pairs(modulus)
    raw = b"".join(a.to_bytes(32, "big") + b.to_bytes(32, "big") for a, b in pairs)
    records = run_oracle(
        gpu_notary,
        oracle,
        "crypto_test_arithmetic",
        [raw],
        len(pairs),
        128,
        context=True,
        extra=(int(use_order),),
    )
    for i, ((a, b), record) in enumerate(zip(pairs, records)):
        expected = (
            (a + b) % modulus,
            (a - b) % modulus,
            a * b % modulus,
            pow(a, -1, modulus) if a else 0,
        )
        assert record == b"".join(v.to_bytes(32, "big") for v in expected), (
            f"arithmetic vector {i}"
        )


@cache
def wide_product_vectors():
    # Exhaust all 0/max assignments of sixteen 32-bit words. Random reduced
    # operands almost never hit the extreme signed-carry combinations.
    values = [sum(((mask >> word) & 1) * (2**32 - 1) << (32 * word)
                  for word in range(16)) for mask in range(2**16)]
    rng = random.Random(SEED + 17)
    values += [rng.getrandbits(512) for _ in range(2048)]
    return tuple(values)


def test_sparse_prime_reduction_matches_python_for_wide_products(gpu_notary, oracle):
    from ecdsa import NIST256p

    values = wide_product_vectors()
    records = run_oracle(
        gpu_notary, oracle, "crypto_test_prime_reduction",
        [b"".join(value.to_bytes(64, "little") for value in values)],
        len(values), 64, context=True,
    )
    for index, (value, record) in enumerate(zip(values, records)):
        assert record == (value % NIST256p.curve.p()).to_bytes(32, "big") * 2, index


@pytest.mark.parametrize("use_order", [False, True], ids=["field_prime", "group_order"])
def test_montgomery_products_and_aliases_match_python(gpu_notary, oracle, use_order):
    from ecdsa import NIST256p

    modulus = NIST256p.order if use_order else NIST256p.curve.p()
    pairs = arithmetic_pairs(modulus)
    records = run_oracle(
        gpu_notary, oracle, "crypto_test_montgomery_products",
        [b"".join(a.to_bytes(32, "big") + b.to_bytes(32, "big") for a, b in pairs)],
        len(pairs), 96, context=True, extra=(int(use_order),),
    )
    inverse_radix = pow(2**256, -1, modulus)
    for index, ((a, b), record) in enumerate(zip(pairs, records)):
        expected = (a * b * inverse_radix % modulus).to_bytes(32, "big")
        assert record == expected * 3, index


@cache
def signature_vectors():
    from ecdsa import NIST256p, SigningKey
    from ecdsa.rfc6979 import generate_k

    order = NIST256p.order
    known_private = int(
        "C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721", 16
    )
    vectors = [(known_private, hashlib.sha256(b"sample").digest())]
    # Force hashes >= n: a random SHA256 digest almost never reaches this
    # range, so fuzzing alone misses RFC 6979 bits2octets/order reduction.
    for key in (1, 2, order - 2, order - 1):
        vectors.extend(
            (key, value.to_bytes(32, "big"))
            for value in (0, 1, order - 1, order, order + 1, 2**256 - 1)
        )
    keys = [1 << bit for bit in range(0, 256, 6)]
    keys += [(1 << bit) - 1 for bit in (64, 128, 192, 255)]
    rng = random.Random(SEED + 4)
    keys += [rng.randrange(1, order) for _ in range(128)]
    vectors.extend(
        (key, hashlib.sha256(blocks()[i]).digest()) for i, key in enumerate(keys)
    )
    # 201 groups leave a partially occupied final warp in the cooperative
    # oracle. A full-mask shuffle must not wait for absent neighboring groups.
    vectors.append((order - 3, hashlib.sha256(b"partial signing warp").digest()))
    expected = []
    for key, digest in vectors:
        nonce = generate_k(order, key, hashlib.sha256, digest)
        signer = SigningKey.from_secret_exponent(
            key, curve=NIST256p, hashfunc=hashlib.sha256
        )
        # Byte-for-byte deterministic r||s, NOT merely "a valid signature".
        signature = signer.sign_digest_deterministic(digest, hashfunc=hashlib.sha256)
        expected.append(nonce.to_bytes(32, "big") + signature)
    # Published RFC 6979 A.2.5, SHA256("sample"), anchors the library oracle.
    # https://www.rfc-editor.org/rfc/rfc6979.txt
    assert expected[0].hex() == (
        "a6e3c57dd01abe90086538398355dd4c3b17aa873382b0f24d6129493d8aad60"
        "efd48b2aacb6a8fd1140dd9cd45e81d69d2c877b56aaf991c34d0ea84eaf3716"
        "f7cb1c942d657c41d436c7a1b6e29f65f3e900dbb9aff4064dc4ab2f843acda8"
    )
    return vectors, expected


@pytest.mark.parametrize("entry_point", ["crypto_test_signatures", "crypto_test_grouped_signatures"])
def test_exact_rfc6979_nonces_and_p256_signatures(gpu_notary, oracle, entry_point):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, utils

    vectors, expected = signature_vectors()
    raw = b"".join(key.to_bytes(32, "big") + digest for key, digest in vectors)
    records = run_oracle(
        gpu_notary,
        oracle,
        entry_point,
        [raw],
        len(vectors),
        168,
        context=True,
    )
    for i, ((scalar, digest), reference, record) in enumerate(
        zip(vectors, expected, records)
    ):
        assert record[160:] == bytes(8), f"nonce/signing status for vector {i}"
        assert record[:96] == reference, f"deterministic nonce/r/s vector {i}"
        public = ec.derive_private_key(scalar, ec.SECP256R1()).public_key()
        numbers = public.public_numbers()
        assert record[96:160] == numbers.x.to_bytes(32, "big") + numbers.y.to_bytes(
            32, "big"
        ), f"public point {i}"
        r, s = (
            int.from_bytes(record[32:64], "big"),
            int.from_bytes(record[64:96], "big"),
        )
        signature = utils.encode_dss_signature(r, s)
        public.verify(signature, digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
        changed = bytes([digest[0] ^ 1]) + digest[1:]
        with pytest.raises(InvalidSignature):
            public.verify(
                signature, changed, ec.ECDSA(utils.Prehashed(hashes.SHA256()))
            )


def test_keygen_public_points_match_sha256_and_cryptography(gpu_notary, oracle):
    from cryptography.hazmat.primitives.asymmetric import ec
    from ecdsa import NIST256p

    rng = random.Random(SEED + 5)
    seeds = [bytes(32), b"\xff" * 32, bytes(range(32))] + [
        rng.randbytes(32) for _ in range(13)
    ]
    n = gpu_notary
    for seed in seeds:
        with device_arena(n) as upload:
            entropy, output = upload(seed), upload(b"\xa5" * 64)
            n.cu.launch(
                n.cu.function(oracle, "keygen_kernel"),
                1,
                1,
                [
                    ctypes.c_uint64(n._ctx_dev.ptr),
                    ctypes.c_uint64(entropy.ptr),
                    ctypes.c_uint64(output.ptr),
                    ctypes.c_uint64(output.ptr + 32),
                ],
            )
            n.cu.sync()
            actual = output.read()
        scalar = (
            int.from_bytes(hashlib.sha256(seed).digest(), "big") % NIST256p.order or 1
        )
        point = (
            ec.derive_private_key(scalar, ec.SECP256R1()).public_key().public_numbers()
        )
        assert actual == point.x.to_bytes(32, "big") + point.y.to_bytes(32, "big")


def test_published_sha256_and_hmac_vectors(gpu_notary, oracle):
    messages = (b"", b"abc", b"Hi There")
    packed, offsets = pack_messages(messages)
    # RFC 4231 case 1 has a 20-byte key. Padding it to 32 bytes preserves its
    # HMAC block while respecting the device helper's fixed-32-byte contract.
    key = b"\x0b" * 20 + bytes(12)
    records = run_oracle(
        gpu_notary,
        oracle,
        "crypto_test_hashes",
        [
            packed,
            struct.pack("<3Q", *offsets),
            struct.pack("<3Q", 0, 3, 8),
            key * 3,
        ],
        3,
        192,
    )
    assert (
        records[0][:32].hex()
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert (
        records[1][:32].hex()
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert (
        records[0][64:96].hex()
        == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    )
    # https://www.rfc-editor.org/rfc/rfc4231.txt section 4.2
    assert (
        records[2][96:128].hex()
        == "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
    )

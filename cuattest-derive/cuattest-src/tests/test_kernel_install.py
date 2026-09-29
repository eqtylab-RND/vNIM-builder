# SPDX-License-Identifier: Apache-2.0
"""Kernel discovery and caching. No GPU required."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from cuattest import kernel as kmod
from cuattest._hosthash import blake3_digest
from cuattest._build_config import BUILD_TYPE
from cuattest._build_options import cuda_options


def test_kernel_source_ships_with_the_package():
    src = kmod.kernel_source()
    # The identity of the notary includes this source's CID, so its presence
    # is not optional.
    assert "measure_model_fused_kernel" in src
    assert "attest_measured_kernel" in src
    assert "keygen_kernel" in src
    # Measurement, every reduction level, and folding remain one cooperative
    # launch. Receipt construction is isolated so its large P-256 frame cannot
    # reduce bulk-hash occupancy, and consumes only a private measured root.
    # Both measurement variants share the same private-root handoff. The
    # async entry point has a different occupancy limit, not a public signer.
    assert src.count('extern "C" __global__ void') == 5
    assert "measure_model_fused_async_kernel" in src
    attest = src.split("void attest_measured_kernel", 1)[1]
    assert "g_pending_model_root" in attest
    assert "model_root" not in attest.split("{", 1)[0]
    assert '\\"kernelCID\\"' in src and '\\"cubinCID\\"' in src
    assert '\\"device\\":\\"cuda:' in src and '\\"gpuDID\\"' in src


def test_secret_scalar_path_uses_fixed_work_and_masked_selection():
    src = kmod.kernel_source()
    scalar = src.split("void multiply_generator_by_scalar", 1)[1].split(
        "void big_endian_bytes_to_uint256", 1
    )[0]
    nonce = src.split("derive_rfc6979_nonce", 1)[1].split("sign_ecdsa_p256", 1)[0]

    # A set nonce bit must not conditionally execute the expensive point add.
    # Keep both the fixed-window point schedule and RFC 6979 candidate schedule
    # fixed, with secret values affecting only mask selection. Four candidates
    # make the all-rejected probability less than 2^-128 and fail closed.
    assert "g_p256_generator_table" in scalar
    assert "candidate <= P256_WINDOW_POINTS" in scalar
    assert "constant_time_equality_mask" in scalar and "jacobian_add(" in scalar
    assert "[secret_digit]" not in scalar
    assert "candidate_index < 4" in nonce
    assert "constant_time_select(output_nonce" in nonce
    assert "volatile Uint64 schedule_guard" in nonce
    assert "return candidate_was_selected ? 0 : -1" in nonce


def test_keygen_credits_only_the_fixed_size_host_csprng_seed():
    src = kmod.kernel_source()
    keygen = src.split("void keygen_kernel", 1)[1].split(
        "// ===== SINGLE-THREAD STREAMING BLAKE3 =====", 1
    )[0]
    compact_keygen = "".join(keygen.split())

    # Device timers and scheduling races have no portable, attacker-conditioned
    # min-entropy guarantee. Keep the production derivation deterministic from
    # the trusted host's 32-byte OS-CSPRNG sample instead of silently crediting
    # unspecified GPU behavior as cryptographic entropy.
    assert "sha256_digest(host_entropy,32,private_scalar_digest)" in compact_keygen
    assert "clock64()" not in keygen
    assert "%%globaltimer" not in keygen
    assert "atomicExch(" not in keygen
    assert "atomicCAS(" not in keygen
    assert "entropy_pool" not in keygen


def test_aligned_blake3_path_keeps_vector_loads_paired_ilp_and_local_tiles():
    src = kmod.kernel_source()
    pair = src.split("void blake3_hash_two_aligned_full_chunks", 1)[1].split(
        "blake3_compress_aligned_parent", 1
    )[0]
    measure = src.split("measure_model_fused_kernel", 1)[1]
    compact_pair = "".join(pair.split())
    compact_measure = "".join(measure.split())

    # The profiled fast path exposes two independent compression chains per
    # lane and loads both aligned message blocks as 16-byte vectors. Reverting
    # to runtime-indexed word arrays recreates the original local-memory stall.
    assert "uint4first_0_to_3=first_vectors[0]" in compact_pair
    assert "uint4second_0_to_3=second_vectors[0]" in compact_pair
    assert "blake3_compress_words(first_output_0" in compact_pair
    assert "blake3_compress_words(second_output_0" in compact_pair
    assert "#define BLAKE3_CHUNKS_PER_TILE 128" in src
    assert "shared_tile_chaining_values[2][8][PADDED_TILE_STRIDE]" in compact_measure
    assert "shared_memory_tile_slot(left_child_index)" in compact_measure
    assert "first_local_chunk_index=(Uint64)tile_lane_index*2" in compact_measure
    assert "tile_reduction_level<7" in compact_measure
    assert "total_tile_count" in compact_measure


def test_kernel_source_explains_its_parallelism_and_optimized_storage():
    src = kmod.kernel_source()

    # Keep the performance-sensitive CUDA ownership model understandable in the
    # trusted source itself, rather than relying only on an external document.
    assert "CUDA PARALLELISM MODEL" in src
    assert "four warps split into two 64-thread tile groups" in src
    assert "2 KiB apart" in src
    assert "intentionally strided" in src
    assert "coalesced" in src
    assert "word-transposed shared array" in src
    assert "Hold the complete parent in registers" in src
    assert "grid-stride" in src

    # Descriptive names are part of the readability contract for the dense
    # cryptographic and scheduling code.
    assert "struct TensorMeasurementSpan" in src
    assert "semantic_chunk_count" in src
    assert "struct Blake3StreamingState" in src
    assert "struct FieldParameters" in src
    assert "struct BufferWriter" in src


def test_cubin_filename_includes_the_arch():
    assert kmod.cubin_filename("sm_90", "Release") == "p256_cuda_notary_b3.sm_90.cubin"
    for mode in ("Debug", "AssertedRelease"):
        assert kmod.cubin_filename("sm_90", mode) == f"p256_cuda_notary_b3.sm_90.{mode}.cubin"


def test_search_dirs_prefers_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CUATTEST_KERNEL_DIR", str(tmp_path))
    assert kmod.search_dirs()[0] == tmp_path


def test_find_cubin_reads_a_prebuilt_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("CUATTEST_KERNEL_DIR", str(tmp_path))
    cubin = b"\x7fELFstub"
    (tmp_path / kmod.cubin_filename("sm_90")).write_bytes(cubin)
    (tmp_path / kmod.cubin_metadata_filename("sm_90")).write_text(
        json.dumps(
            {
                "source_blake3": kmod.source_digest().hex(),
                "cubin_blake3": blake3_digest(cubin).hex(),
                "compiler": "12.9",
                "build_type": BUILD_TYPE,
                "build_options": cuda_options(BUILD_TYPE),
            }
        )
    )
    found = kmod.find_cubin("sm_90")
    assert found is not None
    cubin, tag, path = found
    assert cubin.startswith(b"\x7fELF") and "12.9" in tag and Path(path).exists()


def test_find_cubin_returns_none_for_an_unbuilt_arch(monkeypatch, tmp_path):
    # Keep cache-miss tests independent of any prebuilt artifacts a release
    # wheel may eventually ship in its package directory.
    monkeypatch.setattr(kmod, "search_dirs", lambda: [tmp_path])
    assert kmod.find_cubin("sm_61") is None


def test_find_cubin_rejects_missing_source_sidecar(monkeypatch, tmp_path):
    monkeypatch.setattr(kmod, "search_dirs", lambda: [tmp_path])
    (tmp_path / kmod.cubin_filename("sm_90")).write_bytes(b"\x7fELFold")
    assert kmod.find_cubin("sm_90") is None


def test_find_cubin_treats_malformed_sidecar_as_a_cache_miss(monkeypatch, tmp_path):
    monkeypatch.setattr(kmod, "search_dirs", lambda: [tmp_path])
    (tmp_path / kmod.cubin_filename("sm_90")).write_bytes(b"\x7fELFold")
    (tmp_path / kmod.cubin_metadata_filename("sm_90")).write_text("[]")
    assert kmod.find_cubin("sm_90") is None


def test_find_cubin_rejects_legacy_directory_global_compiler_metadata(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(kmod, "search_dirs", lambda: [tmp_path])
    cubin = b"\x7fELFlegacy"
    (tmp_path / kmod.cubin_filename("sm_90")).write_bytes(cubin)
    (tmp_path / kmod.cubin_metadata_filename("sm_90")).write_text(
        json.dumps(
            {
                "source_blake3": kmod.source_digest().hex(),
                "cubin_blake3": blake3_digest(cubin).hex(),
            }
        )
    )
    (tmp_path / "NVRTC_VERSION").write_text("wrong-after-partial-rebuild\n")

    assert kmod.find_cubin("sm_90") is None


def test_source_change_invalidates_architecture_named_cubin(monkeypatch, tmp_path):
    cubin_dir = tmp_path / "cubins"
    cubin_dir.mkdir()
    monkeypatch.setattr(kmod, "search_dirs", lambda: [cubin_dir])
    cubin = b"\x7fELFold"
    (cubin_dir / kmod.cubin_filename("sm_90")).write_bytes(cubin)
    (cubin_dir / kmod.cubin_metadata_filename("sm_90")).write_text(
        json.dumps(
            {
                "source_blake3": kmod.source_digest("// old source").hex(),
                "cubin_blake3": blake3_digest(cubin).hex(),
                "compiler": "12.8",
            }
        )
    )
    assert kmod.find_cubin("sm_90", "// upgraded source") is None


def test_cubin_byte_change_invalidates_sidecar(monkeypatch, tmp_path):
    monkeypatch.setattr(kmod, "search_dirs", lambda: [tmp_path])
    (tmp_path / kmod.cubin_filename("sm_90")).write_bytes(b"\x7fELFtampered")
    (tmp_path / kmod.cubin_metadata_filename("sm_90")).write_text(
        json.dumps(
            {
                "source_blake3": kmod.source_digest().hex(),
                "cubin_blake3": blake3_digest(b"\x7fELForiginal").hex(),
                "compiler": "12.8",
            }
        )
    )
    assert kmod.find_cubin("sm_90") is None


def test_build_records_source_and_cubin_digests(monkeypatch, tmp_path):
    cubin = b"\x7fELFcompiled"

    class FakeNvrtc:
        def compile_cubin(self, source, arch, *, build_type):
            assert (source, arch) == ("// source", "sm_90")
            assert build_type == BUILD_TYPE
            return cubin

        def version(self):
            return "test"

    from cuattest import _nvrtc

    monkeypatch.setattr(_nvrtc, "Nvrtc", FakeNvrtc)

    kmod.build_cubin("sm_90", tmp_path, source="// source")
    metadata = json.loads(
        (tmp_path / kmod.cubin_metadata_filename("sm_90")).read_text()
    )
    assert metadata == {
        "source_blake3": kmod.source_digest("// source").hex(),
        "cubin_blake3": blake3_digest(cubin).hex(),
        "compiler": "test",
        "build_type": BUILD_TYPE,
        "build_options": cuda_options(BUILD_TYPE),
    }
    assert not (tmp_path / "NVRTC_VERSION").exists()


@pytest.mark.parametrize("fail_publication", [False, True])
def test_concurrent_builds_own_their_staging_files(
    monkeypatch,
    tmp_path,
    fail_publication,
):
    from cuattest import _nvrtc

    cubin = b"\x7fELFconcurrent"

    class FakeNvrtc:
        def compile_cubin(self, source, arch, *, build_type):
            return cubin

        def version(self):
            return "test"

    monkeypatch.setattr(_nvrtc, "Nvrtc", FakeNvrtc)
    monkeypatch.setattr(kmod, "search_dirs", lambda: [tmp_path])
    both_staged = Barrier(2)
    original_replace = Path.replace
    staged_paths = []

    def publish(path, target):
        if target == tmp_path / kmod.cubin_filename("sm_90"):
            staged_paths.append(path)
            # Both callers have written both files before either can rename
            # one. PID-only names deterministically lose one caller's source.
            both_staged.wait(timeout=10)
        elif fail_publication:
            raise OSError("injected metadata publication failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(kmod.build_cubin, "sm_90", tmp_path, "// source")
            for _ in range(2)
        ]
        for future in futures:
            if fail_publication:
                with pytest.raises(OSError, match="metadata publication"):
                    future.result(timeout=15)
            else:
                assert future.result(timeout=15) == (
                    cubin,
                    "test",
                    tmp_path / kmod.cubin_filename("sm_90"),
                )

    assert len(set(staged_paths)) == 2
    assert not list(tmp_path.glob(".*"))  # no abandoned staging files/directories
    found = kmod.find_cubin("sm_90", "// source")
    if fail_publication:
        assert found is None
    else:
        assert found == (
            cubin,
            "test (prebuilt)",
            tmp_path / kmod.cubin_filename("sm_90"),
        )


def test_each_cubin_reads_its_own_compiler_version(monkeypatch, tmp_path):
    monkeypatch.setattr(kmod, "search_dirs", lambda: [tmp_path])

    for arch, compiler in (("sm_90", "12.8"), ("sm_100", "13.0")):
        cubin = b"\x7fELF" + arch.encode()
        (tmp_path / kmod.cubin_filename(arch)).write_bytes(cubin)
        (tmp_path / kmod.cubin_metadata_filename(arch)).write_text(
            json.dumps(
                {
                    "source_blake3": kmod.source_digest().hex(),
                    "cubin_blake3": blake3_digest(cubin).hex(),
                    "compiler": compiler,
                    "build_type": BUILD_TYPE,
                    "build_options": cuda_options(BUILD_TYPE),
                }
            )
        )

    # A legacy directory-global value must neither overwrite per-artifact
    # provenance nor be required when an artifact is copied elsewhere.
    (tmp_path / "NVRTC_VERSION").write_text("99.0\n")
    assert kmod.find_cubin("sm_90")[1] == "12.8 (prebuilt)"
    assert kmod.find_cubin("sm_100")[1] == "13.0 (prebuilt)"


def test_kernel_source_env_override(monkeypatch, tmp_path):
    p = tmp_path / "custom.cu"
    p.write_text("// custom")
    monkeypatch.setenv("CUATTEST_KERNEL_SRC", str(p))
    assert kmod.kernel_source() == "// custom"

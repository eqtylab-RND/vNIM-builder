"""Frozen, test-only hashing experiments against 1ce0fbe; never a runtime switch."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

from cuattest.kernel import build_cubin, kernel_source

BASELINE = "638ee4bf4209a262c304eb6f9798c3be054c1613f034295a85c794b344648a5b"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one source anchor: {old[:80]}")
    return source.replace(old, new, 1)


def transform(source, name):
    if hashlib.sha256(source.encode()).hexdigest() != BASELINE:
        raise ValueError("experiment requires the exact 1ce0fbe CUDA source")
    if name == "baseline":
        return source
    if name == "fulltile":
        # Keep uncommon byte-safe/tail paths out of the hot kernel's inlined
        # register allocation. Full aligned tiles need neither ROOT leaves nor
        # per-lane partial-block calculations. No register cap is imposed.
        source = replace_once(
            source,
            "__device__ static void blake3_hash_chunk(",
            "__device__ __noinline__ void blake3_hash_chunk(",
        )
        source = replace_once(
            source,
            "__forceinline__ void blake3_hash_aligned_full_chunk(",
            "__noinline__ void blake3_hash_aligned_full_chunk(",
        )
        anchor = "            // Derive the first chunk's tensor-wide index"
        source = replace_once(
            source,
            anchor,
            "            const bool full_tile = chunks_in_tile == 128 &&\n"
            "                tensor_span->byte_count - first_semantic_chunk * 1024 >= 131072 &&\n"
            "                (((Uint64)tensor_span->device_bytes & 15ull) == 0);\n"
            + anchor,
        )
        source = replace_once(
            source,
            "(Uint32)(first_chunk_remaining_bytes > 1024 ? 1024",
            "(Uint32)(full_tile || first_chunk_remaining_bytes > 1024 ? 1024",
        )
        source = replace_once(
            source,
            "(Uint32)(second_chunk_remaining_bytes > 1024 ? 1024",
            "(Uint32)(full_tile || second_chunk_remaining_bytes > 1024 ? 1024",
        )
        return source
    if name not in ("async", "async_single"):
        raise ValueError(name)
    start = source.index(
        "__device__ __forceinline__ void blake3_hash_two_aligned_full_chunks("
    )
    end = source.index("// Compress two aligned global-memory chaining values", start)
    function = source[start:end]
    body = function.index("    // Initialize two independent")
    original_body = function[body : function.rfind("}")]
    # Each lane owns its shared locations: no cross-lane reads, divergent
    # barrier, or block-wide wait inside a partial tile. cp.async.wait_group 0
    # confirms this lane's writes before it reads/reuses its ping-pong buffer.
    staging = """
#if __CUDA_ARCH__ >= 800
    __shared__ uint4 stage[BUFFERS][8][128];
    const int lane = (int)threadIdx.x;
    for (int v = 0; v < 8; ++v) {
        const Byte *src = v < 4 ? first_chunk_bytes + v * 16
                               : second_chunk_bytes + (v - 4) * 16;
        unsigned dst = (unsigned)__cvta_generic_to_shared(&stage[0][v][lane]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
""".replace("BUFFERS", "2" if name == "async" else "1")
    loop = original_body.index("        // Issue four 16-byte loads")
    loads_end = original_body.index("        // Compute each chunk's independent", loop)
    loads = """        asm volatile("cp.async.wait_group 0;" ::: "memory");
        int slot = SLOT;
""".replace("SLOT", "(int)block_index & 1" if name == "async" else "0")
    for leaf, offset in (("first", 0), ("second", 4)):
        for vector, words in enumerate(("0_to_3", "4_to_7", "8_to_11", "12_to_15")):
            loads += f"        uint4 {leaf}_{words} = stage[slot][{offset + vector}][lane];\n"
    loads += """
        // The current vectors are now registers. Stage the next compression
        // block while computing this one; never read past the 1 KiB chunk.
        if (block_index + 1 < 16) {
            int next = NEXT;
            for (int v = 0; v < 8; ++v) {
                const Byte *src = (v < 4 ? first_chunk_bytes + v * 16
                                        : second_chunk_bytes + (v - 4) * 16)
                                  + (Uint64)(block_index + 1) * 64;
                unsigned dst = (unsigned)__cvta_generic_to_shared(&stage[next][v][lane]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
""".replace("NEXT", "slot ^ 1" if name == "async" else "0")
    altered_body = original_body[:loop] + loads + original_body[loads_end:]
    function = (
        function[:body]
        + staging
        + altered_body
        + "#else\n"
        + original_body
        + "#endif\n}\n\n"
    )
    return source[:start] + function + source[end:]


def compile_one(output, arch, name):
    started = time.monotonic()
    source = transform(kernel_source(), name)
    directory = Path(output) / name
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "kernel.cu"
    path.write_text(source)
    binary, compiler, cubin = build_cubin(arch, directory / "cubins", source)
    result = dict(
        variant=name,
        arch=arch,
        compiler=compiler,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        cubin_sha256=hashlib.sha256(binary).hexdigest(),
        source=str(path.resolve()),
        cubin_dir=str(cubin.parent.resolve()),
        compile_seconds=time.monotonic() - started,
    )
    (directory / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--variant", action="append")
    args = parser.parse_args()
    with ProcessPoolExecutor(3) as pool:
        futures = [
            pool.submit(compile_one, args.output, args.arch, name)
            for name in (args.variant or ["fulltile", "async", "async_single"])
        ]
        for future in futures:
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()

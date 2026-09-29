// CUDA implementation of the cuAttest notary.
//
// This translation unit deliberately contains every device-side primitive used by the notary. It
// is compiled directly to a CUBIN by NVRTC and does not link the CUDA runtime. Keeping the trusted
// kernel self-contained makes its content identifier an understandable description of the code
// that measured and signed a model.
//
// ALGORITHMS AND DATA FLOW
// ------------------------
// 1. BLAKE3 measures each tensor directly in VRAM. The public BLAKE3 format still uses semantic
//    1 KiB chunks; a larger 128 KiB "scheduling tile" only controls how CUDA assigns work.
// 2. SHA-256 hashes signed documents and implements HMAC-SHA-256 for deterministic RFC 6979 ECDSA
//    nonces.
// 3. P-256 arithmetic signs the measurement document and one credential. Secret-dependent
//    choices use fixed work and mask selection rather than branches or secret table indices.
// 4. Small encoding and writer helpers construct the complete EQTY receipt on the GPU. The host
//    supplies metadata, tensor spans, and output storage; it never supplies a model root or a
//    signable digest.
//
// CUDA PARALLELISM MODEL
// ----------------------
// `measure_model_fused_kernel` is launched cooperatively with 128 threads per block. Every block
// is divided into two independent 64-thread groups. A group owns one 128 KiB scheduling tile:
//
//   * 64 lanes x 2 semantic chunks per lane x 1 KiB per chunk = 128 KiB per group.
//   * Each lane hashes two independent chunks. Loading both inputs before advancing either
//     dependency chain exposes instruction-level parallelism to the warp scheduler.
//   * A full 1 KiB chunk is sixteen 64-byte BLAKE3 blocks. Each block is fetched as four aligned
//     `uint4` values (four 16-byte vector loads). Each lane advances linearly through two adjacent
//     chunks, while neighboring lanes start 2 KiB apart. Leaf reads are therefore aligned and
//     vectorized but intentionally strided, not contiguous/coalesced across a warp. This ownership
//     keeps two complete dependency chains in registers for measured instruction-level parallelism.
//     Cross-tile reduction is different: adjacent threads consume adjacent 32-byte CV pairs, so
//     those global loads and stores are coalesced.
//   * The 128 leaf chaining values and seven tile-local tree levels stay in shared memory. The
//     array is transposed by chaining-value word and padded after each 32 logical entries to avoid
//     shared-memory bank conflicts. A read/barrier/write/barrier sequence makes in-place reduction
//     safe.
//   * Only one 32-byte chaining value per 128 KiB tile reaches global scratch. Later tree levels
//     ping-pong between two global workspaces. Cooperative-grid barriers separate those levels, so
//     the CPU launches the measurement kernel only once.
//   * All grid work uses grid-stride loops. The grid therefore contains only as many blocks as can
//     be resident concurrently (required by cooperative barriers), regardless of model size.
//
// `attest_measured_kernel` is intentionally separate. Its SHA/P-256/JSON call graph needs a large
// stack and register set that would reduce occupancy in every measurement thread. Four eight-lane
// groups sign independent hashes in SIMT lockstep; lane zero assembles the final response.
//
// `keygen_kernel` uses one thread. It hashes a 32-byte operating-system CSPRNG seed supplied by
// the trusted notary host, stores the resulting private scalar only in module-private device
// memory, and exports the public point. CUDA timer and scheduling behavior are deliberately not
// treated as cryptographic entropy.
//
// HOST-FACING ENTRY POINTS
// ------------------------
//   keygen_kernel               Derive the session key and export only its public point.
//   measure_model_fused_kernel  Hash tensors, finish their BLAKE3 trees, fold the model root, and
//                               optionally arm the private one-shot signing handoff.
//   attest_measured_kernel      Consume that handoff, sign four hashes, and construct six EQTY
//                               statements.
//
// Standards: BLAKE3 specification; FIPS 180-4 (SHA-256); SEC 1 and FIPS 186-4 (P-256 ECDSA);
// RFC 6979 (deterministic nonces).
#define _CG_LIMIT_INCLUDED_DEPENDENCIES
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// NVRTC has CUDA's __assertfail builtin but no host libc assert.h in its
// header-free compilation environment. Match standard assert semantics,
// including UNEVALUATED expressions under NDEBUG. Offline NVCC uses assert.h.
// These diagnose internal bugs, not untrusted input: status/bounds checks stay
// live in Release. A failed device assert poisons the context; the host must
// retain its existing fail-closed cleanup. Never print key/nonce/operand values.
#if defined(__CUDACC_RTC__)
#ifdef assert
#undef assert
#endif
#ifdef NDEBUG
#define assert(expression) ((void)0)
#else
#define assert(expression) ((expression) ? (void)0 : \
    __assertfail(#expression, __FILE__, __LINE__, __func__, sizeof(char)))
#endif
#else
#include <assert.h>
#endif

// NVRTC provides these CUDA-compatible fundamental types without requiring <cstdint>.
using Byte = unsigned char;
using Uint32 = unsigned int;
using Uint64 = unsigned long long;
static_assert(sizeof(Byte) == 1 && sizeof(Uint32) == 4 && sizeof(Uint64) == 8,
              "host/device wire widths must agree");

// Offsets, in 64-bit limbs, into the immutable P-256 context uploaded by the host.
#define CONTEXT_FIELD_PRIME_OFFSET 0
#define CONTEXT_GROUP_ORDER_OFFSET 4
#define CONTEXT_FIELD_BARRETT_FACTOR_OFFSET 8
#define CONTEXT_ORDER_BARRETT_FACTOR_OFFSET 13
#define CONTEXT_GENERATOR_X_OFFSET 18
#define CONTEXT_GENERATOR_Y_OFFSET 22

// ===== BLAKE3 (model measurement) =====
// BLAKE3 on GPU, from the spec (reference: BLAKE3 team). Unkeyed hash.
// Parallel chunk compression + tree merge. Produces the exact 32-byte BLAKE3
// digest -> validated against the `blake3` python lib. This is the hash EQTY
// CIDs use (raw codec 0x55, multihash 0x1e), so a tensor's digest here IS its
// content id.

// BLAKE3 uses the SHA-256 initial vector as its eight-word initialization value.
__device__ __constant__ Uint32 BLAKE3_INITIAL_VECTOR[8] = {0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u,
                                                           0xA54FF53Au, 0x510E527Fu, 0x9B05688Cu,
                                                           0x1F83D9ABu, 0x5BE0CD19u};

// Domain-separation flags defined by the BLAKE3 specification.
#define BLAKE3_CHUNK_START_FLAG 1
#define BLAKE3_CHUNK_END_FLAG 2
#define BLAKE3_PARENT_FLAG 4
#define BLAKE3_ROOT_FLAG 8

// CUDA scheduling constants. These do not change BLAKE3's semantic 1 KiB chunk size.
#define BLAKE3_CHUNKS_PER_TILE 128
#define THREADS_PER_TILE 64

// Three padding words provide one extra shared-memory slot after each 32-entry bank cycle.
#define PADDED_TILE_STRIDE (BLAKE3_CHUNKS_PER_TILE + 3)

// Rotate a 32-bit word right, as required by each BLAKE3 mixing operation.
__device__ __forceinline__ Uint32 rotate_right_32(Uint32 value, int distance) {
    assert(distance > 0 && distance < 32);
    return (value >> distance) | (value << (32 - distance));
}
// Perform one BLAKE3 G mixing function. Every argument is a scalar lvalue so NVRTC can retain the
// entire compression state in registers.
#define BLAKE3_MIX(state_a, state_b, state_c, state_d, message_x, message_y)                       \
    do {                                                                                           \
        (state_a) = (state_a) + (state_b) + (message_x);                                           \
        (state_d) = rotate_right_32((state_d) ^ (state_a), 16);                                    \
        (state_c) = (state_c) + (state_d);                                                         \
        (state_b) = rotate_right_32((state_b) ^ (state_c), 12);                                    \
        (state_a) = (state_a) + (state_b) + (message_y);                                           \
        (state_d) = rotate_right_32((state_d) ^ (state_a), 8);                                     \
        (state_c) = (state_c) + (state_d);                                                         \
        (state_b) = rotate_right_32((state_b) ^ (state_c), 7);                                     \
    } while (0)

// Apply one complete BLAKE3 round: four column mixes followed by four diagonal mixes. The caller
// expresses the message permutation by passing the sixteen message words in the required order.
#define BLAKE3_ROUND(message_0, message_1, message_2, message_3, message_4, message_5, message_6,  \
                     message_7, message_8, message_9, message_10, message_11, message_12,          \
                     message_13, message_14, message_15)                                           \
    do {                                                                                           \
        BLAKE3_MIX(state_0, state_4, state_8, state_12, message_0, message_1);                     \
        BLAKE3_MIX(state_1, state_5, state_9, state_13, message_2, message_3);                     \
        BLAKE3_MIX(state_2, state_6, state_10, state_14, message_4, message_5);                    \
        BLAKE3_MIX(state_3, state_7, state_11, state_15, message_6, message_7);                    \
        BLAKE3_MIX(state_0, state_5, state_10, state_15, message_8, message_9);                    \
        BLAKE3_MIX(state_1, state_6, state_11, state_12, message_10, message_11);                  \
        BLAKE3_MIX(state_2, state_7, state_8, state_13, message_12, message_13);                   \
        BLAKE3_MIX(state_3, state_4, state_9, state_14, message_14, message_15);                   \
    } while (0)

// Compress one 64-byte BLAKE3 block represented as scalar words.
//
// The long parameter list is deliberate. An earlier array-indexed implementation made NVRTC spill
// state and message arrays into thread-local memory, which overloaded the load/store unit. Named
// scalar words give ptxas an unambiguous register allocation and preserve the profiled fast path.
__device__ __forceinline__ void
blake3_compress_words(Uint32 chaining_value_0, Uint32 chaining_value_1, Uint32 chaining_value_2,
                      Uint32 chaining_value_3, Uint32 chaining_value_4, Uint32 chaining_value_5,
                      Uint32 chaining_value_6, Uint32 chaining_value_7, Uint32 message_0,
                      Uint32 message_1, Uint32 message_2, Uint32 message_3, Uint32 message_4,
                      Uint32 message_5, Uint32 message_6, Uint32 message_7, Uint32 message_8,
                      Uint32 message_9, Uint32 message_10, Uint32 message_11, Uint32 message_12,
                      Uint32 message_13, Uint32 message_14, Uint32 message_15, Uint64 chunk_counter,
                      Uint32 block_byte_count, Uint32 domain_flags, Uint32 &output_0,
                      Uint32 &output_1, Uint32 &output_2, Uint32 &output_3, Uint32 &output_4,
                      Uint32 &output_5, Uint32 &output_6, Uint32 &output_7) {
    assert(block_byte_count <= 64);
    assert((domain_flags & ~15u) == 0); // This implementation is unkeyed BLAKE3.
    assert(!(domain_flags & BLAKE3_PARENT_FLAG) ||
           (chunk_counter == 0 && block_byte_count == 64 &&
            !(domain_flags & (BLAKE3_CHUNK_START_FLAG | BLAKE3_CHUNK_END_FLAG))));
    // Initialize the first half of the compression state from the caller's chaining value.
    Uint32 state_0 = chaining_value_0;
    Uint32 state_1 = chaining_value_1;
    Uint32 state_2 = chaining_value_2;
    Uint32 state_3 = chaining_value_3;
    Uint32 state_4 = chaining_value_4;
    Uint32 state_5 = chaining_value_5;
    Uint32 state_6 = chaining_value_6;
    Uint32 state_7 = chaining_value_7;

    // Initialize the second half from BLAKE3's fixed vector and block metadata.
    Uint32 state_8 = BLAKE3_INITIAL_VECTOR[0];
    Uint32 state_9 = BLAKE3_INITIAL_VECTOR[1];
    Uint32 state_10 = BLAKE3_INITIAL_VECTOR[2];
    Uint32 state_11 = BLAKE3_INITIAL_VECTOR[3];
    Uint32 state_12 = (Uint32)chunk_counter;
    Uint32 state_13 = (Uint32)(chunk_counter >> 32);
    Uint32 state_14 = block_byte_count;
    Uint32 state_15 = domain_flags;

    // Execute BLAKE3's seven rounds with the specification's message permutations.
    BLAKE3_ROUND(message_0, message_1, message_2, message_3, message_4, message_5, message_6,
                 message_7, message_8, message_9, message_10, message_11, message_12, message_13,
                 message_14, message_15);
    BLAKE3_ROUND(message_2, message_6, message_3, message_10, message_7, message_0, message_4,
                 message_13, message_1, message_11, message_12, message_5, message_9, message_14,
                 message_15, message_8);
    BLAKE3_ROUND(message_3, message_4, message_10, message_12, message_13, message_2, message_7,
                 message_14, message_6, message_5, message_9, message_0, message_11, message_15,
                 message_8, message_1);
    BLAKE3_ROUND(message_10, message_7, message_12, message_9, message_14, message_3, message_13,
                 message_15, message_4, message_0, message_11, message_2, message_5, message_8,
                 message_1, message_6);
    BLAKE3_ROUND(message_12, message_13, message_9, message_11, message_15, message_10, message_14,
                 message_8, message_7, message_2, message_5, message_3, message_0, message_1,
                 message_6, message_4);
    BLAKE3_ROUND(message_9, message_14, message_11, message_5, message_8, message_12, message_15,
                 message_1, message_13, message_3, message_0, message_10, message_2, message_6,
                 message_4, message_7);
    BLAKE3_ROUND(message_11, message_15, message_5, message_0, message_1, message_9, message_8,
                 message_6, message_14, message_10, message_2, message_12, message_3, message_4,
                 message_7, message_13);

    // Fold the two state halves together to produce the eight-word chaining value.
    output_0 = state_0 ^ state_8;
    output_1 = state_1 ^ state_9;
    output_2 = state_2 ^ state_10;
    output_3 = state_3 ^ state_11;
    output_4 = state_4 ^ state_12;
    output_5 = state_5 ^ state_13;
    output_6 = state_6 ^ state_14;
    output_7 = state_7 ^ state_15;
}

// Array adapter used by the byte-safe and single-threaded BLAKE3 paths.
__device__ __forceinline__ void blake3_compress_block(const Uint32 chaining_value[8],
                                                      const Uint32 message_words[16],
                                                      Uint64 chunk_counter, Uint32 block_byte_count,
                                                      Uint32 domain_flags, Uint32 output[8]) {
    // Forward every array element as a scalar to the register-only compression implementation.
    blake3_compress_words(chaining_value[0], chaining_value[1], chaining_value[2],
                          chaining_value[3], chaining_value[4], chaining_value[5],
                          chaining_value[6], chaining_value[7], message_words[0], message_words[1],
                          message_words[2], message_words[3], message_words[4], message_words[5],
                          message_words[6], message_words[7], message_words[8], message_words[9],
                          message_words[10], message_words[11], message_words[12],
                          message_words[13], message_words[14], message_words[15], chunk_counter,
                          block_byte_count, domain_flags, output[0], output[1], output[2],
                          output[3], output[4], output[5], output[6], output[7]);
}

// Hash one full, 16-byte-aligned, semantic 1 KiB chunk.
__device__ __forceinline__ void blake3_hash_aligned_full_chunk(const Byte *chunk_bytes,
                                                               Uint64 chunk_counter, int is_root,
                                                               Uint32 &output_0, Uint32 &output_1,
                                                               Uint32 &output_2, Uint32 &output_3,
                                                               Uint32 &output_4, Uint32 &output_5,
                                                               Uint32 &output_6, Uint32 &output_7) {
    assert(chunk_bytes != 0 && ((Uint64)chunk_bytes & 15) == 0);
    assert(is_root == 0 || is_root == 1);
    // Start the chunk's chaining value at BLAKE3's initial vector.
    output_0 = BLAKE3_INITIAL_VECTOR[0];
    output_1 = BLAKE3_INITIAL_VECTOR[1];
    output_2 = BLAKE3_INITIAL_VECTOR[2];
    output_3 = BLAKE3_INITIAL_VECTOR[3];
    output_4 = BLAKE3_INITIAL_VECTOR[4];
    output_5 = BLAKE3_INITIAL_VECTOR[5];
    output_6 = BLAKE3_INITIAL_VECTOR[6];
    output_7 = BLAKE3_INITIAL_VECTOR[7];

    // Keep the sixteen-block dependency chain as a loop. Full unrolling increases register
    // pressure and instruction-cache cost without exposing additional parallelism.
#pragma unroll 1
    for (Uint32 block_index = 0; block_index < 16; block_index++) {
        // Fetch the 64-byte message block with four aligned 16-byte vector loads.
        const uint4 *vector_words = (const uint4 *)(chunk_bytes + (Uint64)block_index * 64);
        uint4 message_0_to_3 = vector_words[0];
        uint4 message_4_to_7 = vector_words[1];
        uint4 message_8_to_11 = vector_words[2];
        uint4 message_12_to_15 = vector_words[3];

        // Mark the first and last compression blocks; a one-chunk hash also marks its last block
        // as the BLAKE3 root output.
        Uint32 domain_flags = (block_index == 0 ? BLAKE3_CHUNK_START_FLAG : 0) |
                              (block_index == 15 ? (Uint32)BLAKE3_CHUNK_END_FLAG |
                                                           (is_root ? (Uint32)BLAKE3_ROOT_FLAG : 0u)
                                                 : 0u);

        // Feed the previous chaining value back into the next sequential block compression.
        blake3_compress_words(output_0, output_1, output_2, output_3, output_4, output_5, output_6,
                              output_7, message_0_to_3.x, message_0_to_3.y, message_0_to_3.z,
                              message_0_to_3.w, message_4_to_7.x, message_4_to_7.y,
                              message_4_to_7.z, message_4_to_7.w, message_8_to_11.x,
                              message_8_to_11.y, message_8_to_11.z, message_8_to_11.w,
                              message_12_to_15.x, message_12_to_15.y, message_12_to_15.z,
                              message_12_to_15.w, chunk_counter, 64, domain_flags, output_0,
                              output_1, output_2, output_3, output_4, output_5, output_6, output_7);
    }
}

// Process two independent leaves per lane. Issuing both sets of vector loads
// before either dependent compression chain gives the scheduler useful ILP;
// three leaves costs 128 registers and is slower on the profiled sm_75 GPU.
__device__ __forceinline__ void blake3_hash_two_aligned_full_chunks(
        const Byte *first_chunk_bytes, Uint64 first_chunk_counter, int first_chunk_is_root,
        const Byte *second_chunk_bytes, Uint64 second_chunk_counter, int second_chunk_is_root,
        Uint32 &first_output_0, Uint32 &first_output_1, Uint32 &first_output_2,
        Uint32 &first_output_3, Uint32 &first_output_4, Uint32 &first_output_5,
        Uint32 &first_output_6, Uint32 &first_output_7, Uint32 &second_output_0,
        Uint32 &second_output_1, Uint32 &second_output_2, Uint32 &second_output_3,
        Uint32 &second_output_4, Uint32 &second_output_5, Uint32 &second_output_6,
        Uint32 &second_output_7) {
    // Initialize two independent register-resident chaining values.
    first_output_0 = BLAKE3_INITIAL_VECTOR[0];
    first_output_1 = BLAKE3_INITIAL_VECTOR[1];
    first_output_2 = BLAKE3_INITIAL_VECTOR[2];
    first_output_3 = BLAKE3_INITIAL_VECTOR[3];
    first_output_4 = BLAKE3_INITIAL_VECTOR[4];
    first_output_5 = BLAKE3_INITIAL_VECTOR[5];
    first_output_6 = BLAKE3_INITIAL_VECTOR[6];
    first_output_7 = BLAKE3_INITIAL_VECTOR[7];
    second_output_0 = BLAKE3_INITIAL_VECTOR[0];
    second_output_1 = BLAKE3_INITIAL_VECTOR[1];
    second_output_2 = BLAKE3_INITIAL_VECTOR[2];
    second_output_3 = BLAKE3_INITIAL_VECTOR[3];
    second_output_4 = BLAKE3_INITIAL_VECTOR[4];
    second_output_5 = BLAKE3_INITIAL_VECTOR[5];
    second_output_6 = BLAKE3_INITIAL_VECTOR[6];
    second_output_7 = BLAKE3_INITIAL_VECTOR[7];

    // Advance the two independent compression chains together. The loads for both chunks issue
    // before either chain's arithmetic, giving each warp useful work while the other load waits.
#pragma unroll 1
    for (Uint32 block_index = 0; block_index < 16; block_index++) {
        // Issue four 16-byte loads for the first chunk.
        const uint4 *first_vectors = (const uint4 *)(first_chunk_bytes + (Uint64)block_index * 64);
        uint4 first_0_to_3 = first_vectors[0];
        uint4 first_4_to_7 = first_vectors[1];
        uint4 first_8_to_11 = first_vectors[2];
        uint4 first_12_to_15 = first_vectors[3];

        // Issue the corresponding loads for the second chunk before starting compression.
        const uint4 *second_vectors =
                (const uint4 *)(second_chunk_bytes + (Uint64)block_index * 64);
        uint4 second_0_to_3 = second_vectors[0];
        uint4 second_4_to_7 = second_vectors[1];
        uint4 second_8_to_11 = second_vectors[2];
        uint4 second_12_to_15 = second_vectors[3];

        // Compute each chunk's independent BLAKE3 domain flags.
        Uint32 first_flags =
                (block_index == 0 ? BLAKE3_CHUNK_START_FLAG : 0) |
                (block_index == 15 ? (Uint32)BLAKE3_CHUNK_END_FLAG |
                                             (first_chunk_is_root ? (Uint32)BLAKE3_ROOT_FLAG : 0u)
                                   : 0u);
        Uint32 second_flags =
                (block_index == 0 ? BLAKE3_CHUNK_START_FLAG : 0) |
                (block_index == 15 ? (Uint32)BLAKE3_CHUNK_END_FLAG |
                                             (second_chunk_is_root ? (Uint32)BLAKE3_ROOT_FLAG : 0u)
                                   : 0u);

        // Advance the first chunk's register-only compression chain.
        blake3_compress_words(first_output_0, first_output_1, first_output_2, first_output_3,
                              first_output_4, first_output_5, first_output_6, first_output_7,
                              first_0_to_3.x, first_0_to_3.y, first_0_to_3.z, first_0_to_3.w,
                              first_4_to_7.x, first_4_to_7.y, first_4_to_7.z, first_4_to_7.w,
                              first_8_to_11.x, first_8_to_11.y, first_8_to_11.z, first_8_to_11.w,
                              first_12_to_15.x, first_12_to_15.y, first_12_to_15.z,
                              first_12_to_15.w, first_chunk_counter, 64, first_flags,
                              first_output_0, first_output_1, first_output_2, first_output_3,
                              first_output_4, first_output_5, first_output_6, first_output_7);

        // Advance the second chain only after the compiler has seen both independent load sets.
        blake3_compress_words(
                second_output_0, second_output_1, second_output_2, second_output_3, second_output_4,
                second_output_5, second_output_6, second_output_7, second_0_to_3.x, second_0_to_3.y,
                second_0_to_3.z, second_0_to_3.w, second_4_to_7.x, second_4_to_7.y, second_4_to_7.z,
                second_4_to_7.w, second_8_to_11.x, second_8_to_11.y, second_8_to_11.z,
                second_8_to_11.w, second_12_to_15.x, second_12_to_15.y, second_12_to_15.z,
                second_12_to_15.w, second_chunk_counter, 64, second_flags, second_output_0,
                second_output_1, second_output_2, second_output_3, second_output_4, second_output_5,
                second_output_6, second_output_7);
    }
}

// Double-buffered, lane-owned copies. Only the paired aligned/full-chunk
// path calls this; incomplete BLAKE3 blocks are never padded or over-read.
__device__ __forceinline__ void blake3_hash_two_async_full_chunks(
        const Byte *first_chunk_bytes, Uint64 first_chunk_counter, int first_chunk_is_root,
        const Byte *second_chunk_bytes, Uint64 second_chunk_counter, int second_chunk_is_root,
        Uint32 &first_output_0, Uint32 &first_output_1, Uint32 &first_output_2,
        Uint32 &first_output_3, Uint32 &first_output_4, Uint32 &first_output_5,
        Uint32 &first_output_6, Uint32 &first_output_7, Uint32 &second_output_0,
        Uint32 &second_output_1, Uint32 &second_output_2, Uint32 &second_output_3,
        Uint32 &second_output_4, Uint32 &second_output_5, Uint32 &second_output_6,
        Uint32 &second_output_7) {

#if __CUDA_ARCH__ >= 800
    assert(blockDim.x == 128 && threadIdx.x < 128);
    assert(first_chunk_bytes != 0 && second_chunk_bytes != 0);
    assert(((Uint64)first_chunk_bytes & 15) == 0 && ((Uint64)second_chunk_bytes & 15) == 0);
    // Each lane exclusively owns its column. Tails take divergent paths, so
    // a block-wide barrier here would deadlock. Wait on this lane's copies
    // before reading/reusing its slots; no copy remains pending on return.
    __shared__ uint4 stage[2][8][128];
    const int lane = (int)threadIdx.x;
    for (int v = 0; v < 8; ++v) {
        const Byte *src = v < 4 ? first_chunk_bytes + v * 16
                               : second_chunk_bytes + (v - 4) * 16;
        unsigned dst = (unsigned)__cvta_generic_to_shared(&stage[0][v][lane]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    // Initialize two independent register-resident chaining values.
    first_output_0 = BLAKE3_INITIAL_VECTOR[0];
    first_output_1 = BLAKE3_INITIAL_VECTOR[1];
    first_output_2 = BLAKE3_INITIAL_VECTOR[2];
    first_output_3 = BLAKE3_INITIAL_VECTOR[3];
    first_output_4 = BLAKE3_INITIAL_VECTOR[4];
    first_output_5 = BLAKE3_INITIAL_VECTOR[5];
    first_output_6 = BLAKE3_INITIAL_VECTOR[6];
    first_output_7 = BLAKE3_INITIAL_VECTOR[7];
    second_output_0 = BLAKE3_INITIAL_VECTOR[0];
    second_output_1 = BLAKE3_INITIAL_VECTOR[1];
    second_output_2 = BLAKE3_INITIAL_VECTOR[2];
    second_output_3 = BLAKE3_INITIAL_VECTOR[3];
    second_output_4 = BLAKE3_INITIAL_VECTOR[4];
    second_output_5 = BLAKE3_INITIAL_VECTOR[5];
    second_output_6 = BLAKE3_INITIAL_VECTOR[6];
    second_output_7 = BLAKE3_INITIAL_VECTOR[7];

    // Advance the two independent compression chains together. The loads for both chunks issue
    // before either chain's arithmetic, giving each warp useful work while the other load waits.
#pragma unroll 1
    for (Uint32 block_index = 0; block_index < 16; block_index++) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        int slot = (int)block_index & 1;
        uint4 first_0_to_3 = stage[slot][0][lane];
        uint4 first_4_to_7 = stage[slot][1][lane];
        uint4 first_8_to_11 = stage[slot][2][lane];
        uint4 first_12_to_15 = stage[slot][3][lane];
        uint4 second_0_to_3 = stage[slot][4][lane];
        uint4 second_4_to_7 = stage[slot][5][lane];
        uint4 second_8_to_11 = stage[slot][6][lane];
        uint4 second_12_to_15 = stage[slot][7][lane];

        // The current vectors are now registers. Stage the next compression
        // block while computing this one; never read past the 1 KiB chunk.
        if (block_index + 1 < 16) {
            int next = slot ^ 1;
            for (int v = 0; v < 8; ++v) {
                const Byte *src = (v < 4 ? first_chunk_bytes + v * 16
                                        : second_chunk_bytes + (v - 4) * 16)
                                  + (Uint64)(block_index + 1) * 64;
                unsigned dst = (unsigned)__cvta_generic_to_shared(&stage[next][v][lane]);
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        // Compute each chunk's independent BLAKE3 domain flags.
        Uint32 first_flags =
                (block_index == 0 ? BLAKE3_CHUNK_START_FLAG : 0) |
                (block_index == 15 ? (Uint32)BLAKE3_CHUNK_END_FLAG |
                                             (first_chunk_is_root ? (Uint32)BLAKE3_ROOT_FLAG : 0u)
                                   : 0u);
        Uint32 second_flags =
                (block_index == 0 ? BLAKE3_CHUNK_START_FLAG : 0) |
                (block_index == 15 ? (Uint32)BLAKE3_CHUNK_END_FLAG |
                                             (second_chunk_is_root ? (Uint32)BLAKE3_ROOT_FLAG : 0u)
                                   : 0u);

        // Advance the first chunk's register-only compression chain.
        blake3_compress_words(first_output_0, first_output_1, first_output_2, first_output_3,
                              first_output_4, first_output_5, first_output_6, first_output_7,
                              first_0_to_3.x, first_0_to_3.y, first_0_to_3.z, first_0_to_3.w,
                              first_4_to_7.x, first_4_to_7.y, first_4_to_7.z, first_4_to_7.w,
                              first_8_to_11.x, first_8_to_11.y, first_8_to_11.z, first_8_to_11.w,
                              first_12_to_15.x, first_12_to_15.y, first_12_to_15.z,
                              first_12_to_15.w, first_chunk_counter, 64, first_flags,
                              first_output_0, first_output_1, first_output_2, first_output_3,
                              first_output_4, first_output_5, first_output_6, first_output_7);

        // Advance the second chain only after the compiler has seen both independent load sets.
        blake3_compress_words(
                second_output_0, second_output_1, second_output_2, second_output_3, second_output_4,
                second_output_5, second_output_6, second_output_7, second_0_to_3.x, second_0_to_3.y,
                second_0_to_3.z, second_0_to_3.w, second_4_to_7.x, second_4_to_7.y, second_4_to_7.z,
                second_4_to_7.w, second_8_to_11.x, second_8_to_11.y, second_8_to_11.z,
                second_8_to_11.w, second_12_to_15.x, second_12_to_15.y, second_12_to_15.z,
                second_12_to_15.w, second_chunk_counter, 64, second_flags, second_output_0,
                second_output_1, second_output_2, second_output_3, second_output_4, second_output_5,
                second_output_6, second_output_7);
    }
#else
    // Older targets retain the original path, not a second copy to maintain.
    blake3_hash_two_aligned_full_chunks(
        first_chunk_bytes, first_chunk_counter, first_chunk_is_root,
        second_chunk_bytes, second_chunk_counter, second_chunk_is_root,
        first_output_0, first_output_1, first_output_2, first_output_3,
        first_output_4, first_output_5, first_output_6, first_output_7,
        second_output_0, second_output_1, second_output_2, second_output_3,
        second_output_4, second_output_5, second_output_6, second_output_7);
#endif
}


// Specializations keep the 32-KiB staging allocation OUT of the standard
// kernel, rather than taxing small/older-GPU requests for an untaken branch.
template <bool UseAsync> struct FullChunkPair;
template <> struct FullChunkPair<false> {
    static __device__ __forceinline__ void hash(
        const Byte *first_chunk_bytes, Uint64 first_chunk_counter, int first_chunk_is_root,
        const Byte *second_chunk_bytes, Uint64 second_chunk_counter, int second_chunk_is_root,
        Uint32 &first_output_0, Uint32 &first_output_1, Uint32 &first_output_2,
        Uint32 &first_output_3, Uint32 &first_output_4, Uint32 &first_output_5,
        Uint32 &first_output_6, Uint32 &first_output_7, Uint32 &second_output_0,
        Uint32 &second_output_1, Uint32 &second_output_2, Uint32 &second_output_3,
        Uint32 &second_output_4, Uint32 &second_output_5, Uint32 &second_output_6,
        Uint32 &second_output_7) {
        blake3_hash_two_aligned_full_chunks(
            first_chunk_bytes, first_chunk_counter, first_chunk_is_root,
            second_chunk_bytes, second_chunk_counter, second_chunk_is_root,
            first_output_0, first_output_1, first_output_2, first_output_3,
            first_output_4, first_output_5, first_output_6, first_output_7,
            second_output_0, second_output_1, second_output_2, second_output_3,
            second_output_4, second_output_5, second_output_6, second_output_7);
    }
};
template <> struct FullChunkPair<true> {
    static __device__ __forceinline__ void hash(
        const Byte *first_chunk_bytes, Uint64 first_chunk_counter, int first_chunk_is_root,
        const Byte *second_chunk_bytes, Uint64 second_chunk_counter, int second_chunk_is_root,
        Uint32 &first_output_0, Uint32 &first_output_1, Uint32 &first_output_2,
        Uint32 &first_output_3, Uint32 &first_output_4, Uint32 &first_output_5,
        Uint32 &first_output_6, Uint32 &first_output_7, Uint32 &second_output_0,
        Uint32 &second_output_1, Uint32 &second_output_2, Uint32 &second_output_3,
        Uint32 &second_output_4, Uint32 &second_output_5, Uint32 &second_output_6,
        Uint32 &second_output_7) {
        blake3_hash_two_async_full_chunks(first_chunk_bytes, first_chunk_counter, first_chunk_is_root, second_chunk_bytes, second_chunk_counter, second_chunk_is_root, first_output_0, first_output_1, first_output_2, first_output_3, first_output_4, first_output_5, first_output_6, first_output_7, second_output_0, second_output_1, second_output_2, second_output_3, second_output_4, second_output_5, second_output_6, second_output_7);
    }
};
// Compress two aligned global-memory chaining values into one parent chaining value.
__device__ __forceinline__ void
blake3_compress_aligned_parent(const Uint32 *left_child, const Uint32 *right_child, int is_root,
                               Uint32 &output_0, Uint32 &output_1, Uint32 &output_2,
                               Uint32 &output_3, Uint32 &output_4, Uint32 &output_5,
                               Uint32 &output_6, Uint32 &output_7) {
    // Read each 32-byte child with two aligned 16-byte vector loads.
    const uint4 *left_vectors = (const uint4 *)left_child;
    const uint4 *right_vectors = (const uint4 *)right_child;
    uint4 left_0_to_3 = left_vectors[0];
    uint4 left_4_to_7 = left_vectors[1];
    uint4 right_0_to_3 = right_vectors[0];
    uint4 right_4_to_7 = right_vectors[1];

    // A parent node always starts from the fixed IV, has a zero counter, and consumes 64 bytes.
    blake3_compress_words(
            BLAKE3_INITIAL_VECTOR[0], BLAKE3_INITIAL_VECTOR[1], BLAKE3_INITIAL_VECTOR[2],
            BLAKE3_INITIAL_VECTOR[3], BLAKE3_INITIAL_VECTOR[4], BLAKE3_INITIAL_VECTOR[5],
            BLAKE3_INITIAL_VECTOR[6], BLAKE3_INITIAL_VECTOR[7], left_0_to_3.x, left_0_to_3.y,
            left_0_to_3.z, left_0_to_3.w, left_4_to_7.x, left_4_to_7.y, left_4_to_7.z,
            left_4_to_7.w, right_0_to_3.x, right_0_to_3.y, right_0_to_3.z, right_0_to_3.w,
            right_4_to_7.x, right_4_to_7.y, right_4_to_7.z, right_4_to_7.w, 0, 64,
            (Uint32)BLAKE3_PARENT_FLAG | (is_root ? (Uint32)BLAKE3_ROOT_FLAG : 0u), output_0,
            output_1, output_2, output_3, output_4, output_5, output_6, output_7);
}

// Shared-memory CVs are transposed by word. One padding word after each group
// of 32 logical CV slots makes the even/odd child indices of a warp cover all
// 32 banks instead of repeatedly hitting the same eight banks.
__device__ __forceinline__ int shared_memory_tile_slot(Uint64 logical_index) {
    assert(logical_index < BLAKE3_CHUNKS_PER_TILE);
    // Insert one physical slot for every complete 32-entry shared-memory bank cycle.
    return (int)(logical_index + (logical_index >> 5));
}
__device__ __forceinline__ void blake3_compress_parent_from_shared_memory(
        const Uint32 *transposed_tile, int left_slot, int right_slot, int is_root, Uint32 &output_0,
        Uint32 &output_1, Uint32 &output_2, Uint32 &output_3, Uint32 &output_4, Uint32 &output_5,
        Uint32 &output_6, Uint32 &output_7) {
    // The first array dimension is flattened: all word-0 values precede all word-1 values, etc.
    blake3_compress_words(
            BLAKE3_INITIAL_VECTOR[0], BLAKE3_INITIAL_VECTOR[1], BLAKE3_INITIAL_VECTOR[2],
            BLAKE3_INITIAL_VECTOR[3], BLAKE3_INITIAL_VECTOR[4], BLAKE3_INITIAL_VECTOR[5],
            BLAKE3_INITIAL_VECTOR[6], BLAKE3_INITIAL_VECTOR[7],
            transposed_tile[0 * PADDED_TILE_STRIDE + left_slot],
            transposed_tile[1 * PADDED_TILE_STRIDE + left_slot],
            transposed_tile[2 * PADDED_TILE_STRIDE + left_slot],
            transposed_tile[3 * PADDED_TILE_STRIDE + left_slot],
            transposed_tile[4 * PADDED_TILE_STRIDE + left_slot],
            transposed_tile[5 * PADDED_TILE_STRIDE + left_slot],
            transposed_tile[6 * PADDED_TILE_STRIDE + left_slot],
            transposed_tile[7 * PADDED_TILE_STRIDE + left_slot],
            transposed_tile[0 * PADDED_TILE_STRIDE + right_slot],
            transposed_tile[1 * PADDED_TILE_STRIDE + right_slot],
            transposed_tile[2 * PADDED_TILE_STRIDE + right_slot],
            transposed_tile[3 * PADDED_TILE_STRIDE + right_slot],
            transposed_tile[4 * PADDED_TILE_STRIDE + right_slot],
            transposed_tile[5 * PADDED_TILE_STRIDE + right_slot],
            transposed_tile[6 * PADDED_TILE_STRIDE + right_slot],
            transposed_tile[7 * PADDED_TILE_STRIDE + right_slot], 0, 64,
            (Uint32)BLAKE3_PARENT_FLAG | (is_root ? (Uint32)BLAKE3_ROOT_FLAG : 0u), output_0,
            output_1, output_2, output_3, output_4, output_5, output_6, output_7);
}
#undef BLAKE3_ROUND
#undef BLAKE3_MIX

// Decode one little-endian 32-bit word from an arbitrarily aligned byte address.
__device__ __forceinline__ Uint32 load_little_endian_uint32(const Byte *bytes) {
    return (Uint32)bytes[0] | ((Uint32)bytes[1] << 8) | ((Uint32)bytes[2] << 16) |
           ((Uint32)bytes[3] << 24);
}

// Hash one semantic BLAKE3 chunk containing between zero and 1,024 bytes. This general path handles
// the last partial chunk and arbitrary caller alignment; full aligned chunks use the scalar fast
// paths above.
__device__ static void blake3_hash_chunk(const Byte *chunk_bytes, Uint32 chunk_byte_count,
                                         Uint64 chunk_counter, int is_root,
                                         Uint32 output_chaining_value[8]) {
    assert(chunk_byte_count <= 1024);
    assert(chunk_byte_count == 0 || chunk_bytes != 0);
    assert(output_chaining_value != 0 && (is_root == 0 || is_root == 1));
    // Initialize this chunk's chaining value.
    Uint32 chaining_value[8];
    for (int word_index = 0; word_index < 8; word_index++) {
        chaining_value[word_index] = BLAKE3_INITIAL_VECTOR[word_index];
    }

    // Even an empty BLAKE3 input has one zero-length compression block.
    Uint32 block_count = (chunk_byte_count + 63) / 64;
    if (block_count == 0) {
        block_count = 1;
    }

    // Compress the chunk's 64-byte blocks sequentially because each depends on the previous CV.
    for (Uint32 block_index = 0; block_index < block_count; block_index++) {
        // Determine how many input bytes belong to this block.
        Uint32 block_offset = block_index * 64;
        Uint32 block_byte_count = chunk_byte_count - block_offset;
        if (block_byte_count > 64) {
            block_byte_count = 64;
        }

        // BLAKE3 defines absent tail bytes as zero.
        Uint32 message_words[16];
        for (int word_index = 0; word_index < 16; word_index++) {
            message_words[word_index] = 0;
        }

        // All ordinary tensor blocks are full and naturally aligned. Load
        // their words directly instead of copying 64 individual bytes through
        // a thread-local temporary array. Keep a byte-safe tail path because
        // the public span ABI permits arbitrary byte lengths and offsets.
        const Byte *block_bytes = chunk_bytes + block_offset;
        if (block_byte_count == 64 && (((Uint64)block_bytes & 15ull) == 0)) {
            // Use four vector reads when the address satisfies `uint4`'s 16-byte alignment.
            const uint4 *vector_words = (const uint4 *)block_bytes;
            uint4 words_0_to_3 = vector_words[0];
            uint4 words_4_to_7 = vector_words[1];
            uint4 words_8_to_11 = vector_words[2];
            uint4 words_12_to_15 = vector_words[3];
            message_words[0] = words_0_to_3.x;
            message_words[1] = words_0_to_3.y;
            message_words[2] = words_0_to_3.z;
            message_words[3] = words_0_to_3.w;
            message_words[4] = words_4_to_7.x;
            message_words[5] = words_4_to_7.y;
            message_words[6] = words_4_to_7.z;
            message_words[7] = words_4_to_7.w;
            message_words[8] = words_8_to_11.x;
            message_words[9] = words_8_to_11.y;
            message_words[10] = words_8_to_11.z;
            message_words[11] = words_8_to_11.w;
            message_words[12] = words_12_to_15.x;
            message_words[13] = words_12_to_15.y;
            message_words[14] = words_12_to_15.z;
            message_words[15] = words_12_to_15.w;
        } else if (block_byte_count == 64 && (((Uint64)block_bytes & 3ull) == 0)) {
            // Fall back to naturally aligned 32-bit reads when vector alignment is unavailable.
            const Uint32 *aligned_words = (const Uint32 *)block_bytes;
#pragma unroll
            for (int word_index = 0; word_index < 16; word_index++) {
                message_words[word_index] = aligned_words[word_index];
            }
        } else {
            // Pack an unaligned or partial block one byte at a time in little-endian order.
            for (Uint32 byte_index = 0; byte_index < block_byte_count; byte_index++) {
                message_words[byte_index >> 2] |= (Uint32)block_bytes[byte_index]
                                                  << ((byte_index & 3) * 8);
            }
        }

        // Domain-separate the first block, final block, and final root output.
        Uint32 domain_flags = 0;
        if (block_index == 0) {
            domain_flags |= BLAKE3_CHUNK_START_FLAG;
        }
        if (block_index == block_count - 1) {
            domain_flags |= BLAKE3_CHUNK_END_FLAG;
            if (is_root) {
                domain_flags |= BLAKE3_ROOT_FLAG;
            }
        }

        // Compress this block and carry its output into the next block.
        Uint32 compressed_value[8];
        blake3_compress_block(chaining_value, message_words, chunk_counter, block_byte_count,
                              domain_flags, compressed_value);
        for (int word_index = 0; word_index < 8; word_index++) {
            chaining_value[word_index] = compressed_value[word_index];
        }
    }

    // Return the final eight-word chunk chaining value to the caller.
    for (int word_index = 0; word_index < 8; word_index++) {
        output_chaining_value[word_index] = chaining_value[word_index];
    }
}

// Combine two ordinary array-resident chaining values into one BLAKE3 parent node.
__device__ static void blake3_compress_parent(const Uint32 left_child[8],
                                              const Uint32 right_child[8], int is_root,
                                              Uint32 output_chaining_value[8]) {
    // A parent message is exactly the left child followed by the right child.
    Uint32 parent_message[16];
    for (int word_index = 0; word_index < 8; word_index++) {
        parent_message[word_index] = left_child[word_index];
        parent_message[8 + word_index] = right_child[word_index];
    }

    // Domain-separate parent nodes and mark only the final parent as the root output.
    Uint32 domain_flags = BLAKE3_PARENT_FLAG | (is_root ? BLAKE3_ROOT_FLAG : 0);
    blake3_compress_block(BLAKE3_INITIAL_VECTOR, parent_message, 0, 64, domain_flags,
                          output_chaining_value);
}

// One descriptor per tensor. `primary_tile_offset` and `secondary_tile_offset` are 128 KiB
// scheduling-tile CV indices into the global ping-pong workspaces; `semantic_chunk_count` remains
// the exact count of BLAKE3's semantic 1 KiB chunks. Host packing is five little-endian Uint64
// values.
struct TensorMeasurementSpan {
    const Byte *device_bytes;
    Uint64 byte_count;
    Uint64 primary_tile_offset;
    Uint64 secondary_tile_offset;
    Uint64 semantic_chunk_count;
};
static_assert(sizeof(TensorMeasurementSpan) == 40, "descriptor must match host <5Q packing");

// Locate the tensor that owns one flattened scheduling-tile index. Descriptor offsets are sorted,
// so a binary search avoids a linear scan for models with many tensors.
__device__ __forceinline__ int find_tensor_for_tile(const TensorMeasurementSpan *tensor_spans,
                                                    int tensor_count, Uint64 global_tile_index) {
    int lower_bound = 0;
    int upper_bound = tensor_count;
    while (lower_bound + 1 < upper_bound) {
        int midpoint = lower_bound + (upper_bound - lower_bound) / 2;
        if (tensor_spans[midpoint].primary_tile_offset <= global_tile_index) {
            lower_bound = midpoint;
        } else {
            upper_bound = midpoint;
        }
    }
    return lower_bound;
}

// `level_prefix_offsets` has tensor_count + 1 entries. Repeated values represent tensors that have
// already reduced to one CV at this level.
__device__ __forceinline__ int find_tensor_for_reduction_output(const Uint64 *level_prefix_offsets,
                                                                int tensor_count,
                                                                Uint64 global_output_index) {
    int lower_bound = 0;
    int upper_bound = tensor_count;
    while (lower_bound < upper_bound) {
        int midpoint = lower_bound + (upper_bound - lower_bound) / 2;
        if (level_prefix_offsets[midpoint + 1] <= global_output_index) {
            lower_bound = midpoint + 1;
        } else {
            upper_bound = midpoint;
        }
    }
    return lower_bound;
}

// Count the pairwise tree levels required to reduce `node_count` nodes to one root.
__device__ __forceinline__ int count_reduction_rounds(Uint64 node_count) {
    assert(node_count > 0 && node_count < (1ull << 63));
    int round_count = 0;
    while (node_count > 1) {
        node_count = (node_count + 1) / 2;
        round_count++;
    }
    return round_count;
}

// ===== SHA-256 + HMAC-SHA-256 =====
//
// SHA-256 processes 64-byte blocks through a 64-round compression function. The incremental state
// below buffers partial blocks, applies the FIPS 180-4 padding rule, and writes the final digest in
// big-endian byte order. HMAC wraps that primitive with the standard inner and outer keyed hashes.
// The signing path uses SHA-256 for documents and HMAC-SHA-256 for RFC 6979 nonce derivation.
__device__ static const Uint32 SHA256_ROUND_CONSTANTS[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u};
// Rotate one SHA-256 working word right.
__device__ __forceinline__ Uint32 sha256_rotate_right(Uint32 value, int distance) {
    assert(distance > 0 && distance < 32);
    return (value >> distance) | (value << (32 - distance));
}
// Compress one complete SHA-256 block into an eight-word hash state.
__device__ static void sha256_compress_block(Uint32 hash_state[8], const Byte block_bytes[64]) {
    // Parse the first sixteen schedule words as big-endian integers.
    Uint32 message_schedule[64];
    for (int word_index = 0; word_index < 16; word_index++) {
        message_schedule[word_index] = ((Uint32)block_bytes[4 * word_index] << 24) |
                                       ((Uint32)block_bytes[4 * word_index + 1] << 16) |
                                       ((Uint32)block_bytes[4 * word_index + 2] << 8) |
                                       block_bytes[4 * word_index + 3];
    }

    // Expand the remaining 48 schedule words with SHA-256's two small-sigma functions.
    for (int word_index = 16; word_index < 64; word_index++) {
        Uint32 small_sigma_0 = sha256_rotate_right(message_schedule[word_index - 15], 7) ^
                               sha256_rotate_right(message_schedule[word_index - 15], 18) ^
                               (message_schedule[word_index - 15] >> 3);
        Uint32 small_sigma_1 = sha256_rotate_right(message_schedule[word_index - 2], 17) ^
                               sha256_rotate_right(message_schedule[word_index - 2], 19) ^
                               (message_schedule[word_index - 2] >> 10);
        message_schedule[word_index] = message_schedule[word_index - 16] + small_sigma_0 +
                                       message_schedule[word_index - 7] + small_sigma_1;
    }

    // Copy the current hash into the eight conventional SHA-256 working words.
    Uint32 working_a = hash_state[0];
    Uint32 working_b = hash_state[1];
    Uint32 working_c = hash_state[2];
    Uint32 working_d = hash_state[3];
    Uint32 working_e = hash_state[4];
    Uint32 working_f = hash_state[5];
    Uint32 working_g = hash_state[6];
    Uint32 working_h = hash_state[7];

    // Run the 64 compression rounds defined by FIPS 180-4.
    for (int round_index = 0; round_index < 64; round_index++) {
        Uint32 big_sigma_1 = sha256_rotate_right(working_e, 6) ^
                             sha256_rotate_right(working_e, 11) ^
                             sha256_rotate_right(working_e, 25);
        Uint32 choice = (working_e & working_f) ^ (~working_e & working_g);
        Uint32 temporary_1 = working_h + big_sigma_1 + choice +
                             SHA256_ROUND_CONSTANTS[round_index] + message_schedule[round_index];
        Uint32 big_sigma_0 = sha256_rotate_right(working_a, 2) ^
                             sha256_rotate_right(working_a, 13) ^
                             sha256_rotate_right(working_a, 22);
        Uint32 majority =
                (working_a & working_b) ^ (working_a & working_c) ^ (working_b & working_c);
        Uint32 temporary_2 = big_sigma_0 + majority;

        // Shift the working words and inject this round's two temporary values.
        working_h = working_g;
        working_g = working_f;
        working_f = working_e;
        working_e = working_d + temporary_1;
        working_d = working_c;
        working_c = working_b;
        working_b = working_a;
        working_a = temporary_1 + temporary_2;
    }

    // Feed the block result forward into the caller's running hash state.
    hash_state[0] += working_a;
    hash_state[1] += working_b;
    hash_state[2] += working_c;
    hash_state[3] += working_d;
    hash_state[4] += working_e;
    hash_state[5] += working_f;
    hash_state[6] += working_g;
    hash_state[7] += working_h;
}

// Incremental SHA-256 state for arbitrarily long input assembled from multiple memory regions.
struct Sha256State {
    Uint32 hash_words[8];
    Byte pending_block[64];
    int pending_byte_count;
    Uint64 total_byte_count;
};

// Reset an incremental SHA-256 state to the standard initial vector.
__device__ static void sha256_initialize(Sha256State *state) {
    assert(state != 0);
    const Uint32 initial_hash[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    for (int word_index = 0; word_index < 8; word_index++) {
        state->hash_words[word_index] = initial_hash[word_index];
    }
    state->pending_byte_count = 0;
    state->total_byte_count = 0;
}

// Append bytes to an incremental SHA-256 state and compress every complete block.
__device__ static void sha256_update(Sha256State *state, const Byte *input_bytes,
                                     Uint64 input_byte_count) {
    assert(state != 0 && (input_byte_count == 0 || input_bytes != 0));
    assert(state->pending_byte_count >= 0 && state->pending_byte_count < 64);
    assert((state->total_byte_count & 63) == (Uint64)state->pending_byte_count);
    assert(input_byte_count <= (~0ull >> 3) - state->total_byte_count);
    state->total_byte_count += input_byte_count;
    while (input_byte_count != 0) {
        // Fill only the unused portion of the pending 64-byte block.
        int bytes_to_copy = 64 - state->pending_byte_count;
        if ((Uint64)bytes_to_copy > input_byte_count) {
            bytes_to_copy = (int)input_byte_count;
        }
        for (int byte_index = 0; byte_index < bytes_to_copy; byte_index++) {
            state->pending_block[state->pending_byte_count + byte_index] = input_bytes[byte_index];
        }

        // Advance the input and buffered-byte counts.
        state->pending_byte_count += bytes_to_copy;
        input_bytes += bytes_to_copy;
        input_byte_count -= bytes_to_copy;

        // Compress and release the pending buffer whenever it becomes full.
        if (state->pending_byte_count == 64) {
            sha256_compress_block(state->hash_words, state->pending_block);
            state->pending_byte_count = 0;
        }
    }
}

// Apply SHA-256 padding and serialize the final digest in big-endian order.
__device__ static void sha256_finalize(Sha256State *state, Byte output_digest[32]) {
    assert(state != 0 && output_digest != 0);
    assert(state->pending_byte_count >= 0 && state->pending_byte_count < 64);
    assert((state->total_byte_count & 63) == (Uint64)state->pending_byte_count);
    assert(state->total_byte_count <= (~0ull >> 3));
    // Preserve the pre-padding message length as a 64-bit bit count.
    Uint64 total_bit_count = state->total_byte_count * 8;
    int padded_byte_count = state->pending_byte_count;

    // Append the mandatory one bit, represented as 0x80 followed by zero bits.
    state->pending_block[padded_byte_count++] = 0x80;
    if (padded_byte_count > 56) {
        while (padded_byte_count < 64) {
            state->pending_block[padded_byte_count++] = 0;
        }
        sha256_compress_block(state->hash_words, state->pending_block);
        padded_byte_count = 0;
    }

    // Zero-pad through byte 55, then append the big-endian message length.
    while (padded_byte_count < 56) {
        state->pending_block[padded_byte_count++] = 0;
    }
    for (int byte_index = 0; byte_index < 8; byte_index++) {
        state->pending_block[56 + byte_index] = (Byte)(total_bit_count >> (56 - 8 * byte_index));
    }
    sha256_compress_block(state->hash_words, state->pending_block);

    // Convert the eight final words into the standard 32-byte representation.
    for (int word_index = 0; word_index < 8; word_index++) {
        output_digest[4 * word_index] = (Byte)(state->hash_words[word_index] >> 24);
        output_digest[4 * word_index + 1] = (Byte)(state->hash_words[word_index] >> 16);
        output_digest[4 * word_index + 2] = (Byte)(state->hash_words[word_index] >> 8);
        output_digest[4 * word_index + 3] = (Byte)state->hash_words[word_index];
    }
}

// Compute a one-shot SHA-256 digest using the incremental implementation.
__device__ static void sha256_digest(const Byte *message, Uint64 message_byte_count,
                                     Byte output_digest[32]) {
    Sha256State state;
    sha256_initialize(&state);
    sha256_update(&state, message, message_byte_count);
    sha256_finalize(&state, output_digest);
}

// Compute HMAC-SHA-256 for the fixed 32-byte keys used by RFC 6979.
__device__ static void hmac_sha256(const Byte key[32], const Byte *message, int message_byte_count,
                                   Byte output_digest[32]) {
    assert(key != 0 && output_digest != 0 && message_byte_count >= 0);
    assert(message_byte_count == 0 || message != 0);
    Byte padded_key[64];
    Sha256State state;

    // XOR the key with HMAC's inner pad and extend the unused half with the pad itself.
    int byte_index = 0;
    for (; byte_index < 32; byte_index++) {
        padded_key[byte_index] = key[byte_index] ^ 0x36;
    }
    for (; byte_index < 64; byte_index++) {
        padded_key[byte_index] = 0x36;
    }

    // Hash the inner pad followed by the caller's message.
    sha256_initialize(&state);
    sha256_update(&state, padded_key, 64);
    sha256_update(&state, message, message_byte_count);
    Byte inner_digest[32];
    sha256_finalize(&state, inner_digest);

    // Rebuild the padded key with HMAC's outer pad.
    for (byte_index = 0; byte_index < 32; byte_index++) {
        padded_key[byte_index] = key[byte_index] ^ 0x5c;
    }
    for (; byte_index < 64; byte_index++) {
        padded_key[byte_index] = 0x5c;
    }

    // Hash the outer pad followed by the inner digest.
    sha256_initialize(&state);
    sha256_update(&state, padded_key, 64);
    sha256_update(&state, inner_digest, 32);
    sha256_finalize(&state, output_digest);
}

// ===== MULTIPRECISION ARITHMETIC AND P-256 ECDSA =====
//
// A 256-bit integer is four little-endian 64-bit limbs. Products use schoolbook multiplication,
// and generic modular products use Barrett reduction with a host-supplied reciprocal. The P-256
// base field uses its sparse prime for reduction; order inversion stays in the Montgomery domain.
// P-256 points use
// Jacobian (X:Y:Z) coordinates so additions and doublings avoid inversions; one Fermat inversion
// converts the final point back to affine coordinates. Six-bit fixed-window generator
// multiplication scans every table entry and mask-selects the requested point. Signing replicas
// cooperatively partition that public scan, keeping addresses and work independent of scalar digits.
//
// These primitives are validated by the repository's GPU self-test against Python
// `cryptography`. Constant-work here reduces timing leakage within this implementation, but it is
// not a formal side-channel proof for the complete GPU platform.

// Multiply two arbitrary-length little-endian limb arrays with schoolbook multiplication.
__device__ static void multiply_limbs(const Uint64 *left, int left_limb_count, const Uint64 *right,
                                      int right_limb_count, Uint64 *product) {
    assert(left != 0 && right != 0 && product != 0);
    assert(left_limb_count > 0 && right_limb_count > 0);
    assert(left_limb_count <= 5 && right_limb_count <= 5);
    // Unlike modular_multiply, the low-level schoolbook output cannot alias
    // either operand: it is zeroed before the first multiply.
    assert(product != left && product != right);
    // Zero every output limb before accumulating partial products.
    for (int output_index = 0; output_index < left_limb_count + right_limb_count; output_index++) {
        product[output_index] = 0;
    }

    // Accumulate one row of the multiplication matrix at a time.
    for (int left_index = 0; left_index < left_limb_count; left_index++) {
        Uint64 carry = 0;
        for (int right_index = 0; right_index < right_limb_count; right_index++) {
            // CUDA exposes the upper half separately from ordinary 64-bit multiplication.
            Uint64 product_low = left[left_index] * right[right_index];
            Uint64 product_high = __umul64hi(left[left_index], right[right_index]);

            // Add the low product and incoming carry while recording both overflows.
            int output_index = left_index + right_index;
            Uint64 previous_output = product[output_index];
            Uint64 sum = previous_output + product_low;
            Uint64 overflow_count = (sum < previous_output);
            sum += carry;
            overflow_count += (sum < carry);

            // Store the completed low limb and carry the high half plus overflow forward.
            product[output_index] = sum;
            assert(product_high <= ~0ull - overflow_count);
            carry = product_high + overflow_count;
        }
        product[left_index + right_limb_count] = carry;
    }
}

// Subtract equal-length little-endian limb arrays and return the final borrow bit.
__device__ static Uint64 subtract_limbs(const Uint64 *minuend, const Uint64 *subtrahend,
                                        int limb_count, Uint64 *difference) {
    assert(minuend != 0 && subtrahend != 0 && difference != 0 && limb_count > 0);
    Uint64 borrow = 0;
    for (int limb_index = 0; limb_index < limb_count; limb_index++) {
        // First subtract the matching limb and detect unsigned underflow.
        Uint64 minuend_limb = minuend[limb_index];
        Uint64 subtrahend_limb = subtrahend[limb_index];
        Uint64 partial_difference = minuend_limb - subtrahend_limb;
        Uint64 limb_borrow = (minuend_limb < subtrahend_limb);

        // Then subtract the borrow from the preceding less-significant limb.
        Uint64 final_difference = partial_difference - borrow;
        Uint64 carry_borrow = (partial_difference < borrow);
        difference[limb_index] = final_difference;
        borrow = limb_borrow + carry_borrow;
        assert(borrow <= 1);
    }
    return borrow;
}

// Expand a bit to either all-zero or all-one bits without branching.
__device__ __forceinline__ Uint64 constant_time_mask(Uint64 bit) {
    return 0ull - (bit & 1ull);
}

// Select `when_mask_set` for an all-one mask or `when_mask_clear` for an all-zero mask.
__device__ static void constant_time_select(Uint64 *output, const Uint64 *when_mask_set,
                                            const Uint64 *when_mask_clear, int limb_count,
                                            Uint64 mask) {
    assert(output != 0 && when_mask_set != 0 && when_mask_clear != 0 && limb_count > 0);
    assert(mask == 0 || mask == ~0ull);
    for (int limb_index = 0; limb_index < limb_count; limb_index++) {
        output[limb_index] =
                (when_mask_set[limb_index] & mask) | (when_mask_clear[limb_index] & ~mask);
    }
}

// Return one when any limb is nonzero, otherwise return zero.
__device__ static Uint64 constant_time_uint256_is_nonzero(const Uint64 value[4]) {
    Uint64 combined = value[0] | value[1] | value[2] | value[3];
    return (combined | (0ull - combined)) >> 63;
}

// Return all-one bits when a 256-bit value is zero, otherwise return all-zero bits.
__device__ static Uint64 constant_time_uint256_zero_mask(const Uint64 value[4]) {
    return constant_time_mask(constant_time_uint256_is_nonzero(value) ^ 1ull);
}

// Reduce a 256-bit value known to be below twice the modulus with one masked subtraction.
#ifndef NDEBUG
// Assertion-only comparisons: do not add secret-dependent diagnostic branches
// to Release arithmetic. Zero remains a valid field element (including the
// internal inverse-of-zero convention for points at infinity).
__device__ static bool uint256_is_reduced(const Uint64 value[4], const Uint64 modulus[4]) {
    for (int i = 3; i >= 0; --i) {
        if (value[i] != modulus[i]) return value[i] < modulus[i];
    }
    return false;
}
__device__ static bool is_p256_prime(const Uint64 modulus[4]) {
    return modulus[0] == 0xffffffffffffffffull && modulus[1] == 0x00000000ffffffffull &&
           modulus[2] == 0 && modulus[3] == 0xffffffff00000001ull;
}
#endif
__device__ static void reduce_uint256_once(const Uint64 value[4], const Uint64 modulus[4],
                                           Uint64 output[4]) {
    Uint64 reduced_candidate[4];
    Uint64 borrow = subtract_limbs(value, modulus, 4, reduced_candidate);
    Uint64 value_was_at_least_modulus = constant_time_mask(borrow ^ 1ull);
    constant_time_select(output, reduced_candidate, value, 4, value_was_at_least_modulus);
    assert(uint256_is_reduced(output, modulus));
}

__device__ static void p256_reduce_product(const Uint64 product[8], const Uint64 modulus[4],
                                           Uint64 output[4]) {
    assert(is_p256_prime(modulus));
    // B=2^32: B^8 == B^7-B^6-B^3+1 (mod P-256). Signed coefficients remain
    // bounded by 9*(B-1), far below signed 64-bit overflow. Let R=B^8 and h=R-P.
    // The folded integer's quotient by R is in [-4,4]. One normalization maps
    // z to z%R + floor(z/R)*h, in [-4*h,R+4*h). The next is in [0,R), since
    // 5*h<R. The third normalizes its words with carry zero. Do not drop that
    // third pass: folding a carry does not itself normalize the affected words.
    // Finally R<2*P, so ONE masked subtraction suffices for any 512-bit input.
    // tests/test_p256_optimizations.py proves these bounds independently.
    long long w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        w[i] = (Uint32)(product[i / 2] >> (32 * (i % 2)));
#pragma unroll
    for (int i = 15; i >= 8; --i) {
        long long v = w[i];
        w[i] = 0;
        w[i - 1] += v;
        w[i - 2] -= v;
        w[i - 5] -= v;
        w[i - 8] += v;
    }
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        long long carry = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            long long t = w[i] + carry;
            w[i] = (Uint32)t;
            carry = t >> 32;
        }
        w[0] += carry;
        w[3] -= carry;
        w[6] -= carry;
        w[7] += carry;
        assert(pass != 2 || carry == 0); // The third pass must finish normalization.
    }
    Uint64 r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        r[i] = (Uint64)(Uint32)w[2 * i] | ((Uint64)(Uint32)w[2 * i + 1] << 32);
    reduce_uint256_once(r, modulus, output);
}

// Reduce a 512-bit product modulo a 256-bit modulus with Barrett reduction.
// `barrett_factor` is floor(2^512 / modulus), precomputed by the host as five limbs.
__device__ static void barrett_reduce(const Uint64 product[8], const Uint64 modulus[4],
                                      const Uint64 barrett_factor[5], Uint64 output[4]) {
    // This public dispatch checks ALL limbs. The group order needs its own
    // reduction; sharing the low limb or calling both moduli "P-256" is not enough.
    if (modulus[0] == 0xffffffffffffffffull && modulus[1] == 0x00000000ffffffffull &&
        modulus[2] == 0x0000000000000000ull && modulus[3] == 0xffffffff00000001ull) {
        p256_reduce_product(product, modulus, output);
        return;
    }

    // Approximate the quotient: q1 = floor(product / 2^192).
    const Uint64 *approximate_quotient_high = product + 3;

    // Multiply q1 by mu, then take q3 = floor(q1 * mu / 2^320).
    Uint64 quotient_product[10];
    multiply_limbs(approximate_quotient_high, 5, barrett_factor, 5, quotient_product);
    const Uint64 *approximate_quotient = quotient_product + 5;

    // Compute the low five limbs of q3 * modulus.
    Uint64 quotient_times_modulus[9];
    multiply_limbs(approximate_quotient, 5, modulus, 4, quotient_times_modulus);

    // Subtract those low limbs from the low five limbs of the original product.
    Uint64 product_low[5];
    Uint64 quotient_product_low[5];
    for (int limb_index = 0; limb_index < 5; limb_index++) {
        product_low[limb_index] = product[limb_index];
        quotient_product_low[limb_index] = quotient_times_modulus[limb_index];
    }
    Uint64 remainder[5];
    subtract_limbs(product_low, quotient_product_low, 5, remainder);

    // Extend the modulus to five limbs for constant-width correction passes.
    Uint64 extended_modulus[5];
    for (int limb_index = 0; limb_index < 4; limb_index++) {
        extended_modulus[limb_index] = modulus[limb_index];
    }
    extended_modulus[4] = 0;

    // The approximation leaves fewer than four moduli; execute all four corrections and mask
    // away any subtraction that borrowed. Fixed iteration count avoids value-dependent timing.
    for (int correction_index = 0; correction_index < 4; correction_index++) {
        Uint64 corrected_candidate[5];
        Uint64 borrow = subtract_limbs(remainder, extended_modulus, 5, corrected_candidate);
        constant_time_select(remainder, corrected_candidate, remainder, 5,
                             constant_time_mask(borrow ^ 1ull));
    }

    // Return the low four limbs; the correction passes guarantee the fifth limb is zero.
    assert(remainder[4] == 0);
    assert(uint256_is_reduced(remainder, modulus));
    for (int limb_index = 0; limb_index < 4; limb_index++) {
        output[limb_index] = remainder[limb_index];
    }
}

// Multiply two 256-bit values and reduce the 512-bit result modulo `modulus`.
__device__ static void modular_multiply(const Uint64 left[4], const Uint64 right[4],
                                        const Uint64 modulus[4], const Uint64 barrett_factor[5],
                                        Uint64 output[4]) {
    Uint64 full_product[8];
    multiply_limbs(left, 4, right, 4, full_product);
    barrett_reduce(full_product, modulus, barrett_factor, output);
}

__device__ static void modular_square(const Uint64 value[4],const Uint64 modulus[4],const Uint64 barrett_factor[5],Uint64 output[4]) {
    modular_multiply(value, value, modulus, barrett_factor, output);
}

// These field-only helpers require the exact P-256 prime. FieldParameters is
// constructed only from the fixed P-256 context; generic callers must use the
// full-modulus dispatch above. Retain the factor argument for the common
// inversion-schedule call shape, but no Barrett reciprocal is read here.
__device__ static void p256_modular_multiply(const Uint64 left[4], const Uint64 right[4],
                                             const Uint64 modulus[4], const Uint64 *,
                                             Uint64 output[4]) {
    Uint64 product[8];
    multiply_limbs(left, 4, right, 4, product);
    p256_reduce_product(product, modulus, output);
}
__device__ static void p256_modular_square(const Uint64 value[4], const Uint64 modulus[4],
                                           const Uint64 *factor, Uint64 output[4]) {
    p256_modular_multiply(value, value, modulus, factor, output);
}

// Add two reduced 256-bit values and reduce their sum modulo `modulus`.
__device__ static void modular_add(const Uint64 left[4], const Uint64 right[4],
                                   const Uint64 modulus[4], Uint64 output[4]) {
    assert(uint256_is_reduced(left, modulus) && uint256_is_reduced(right, modulus));
    Uint64 extended_sum[5];
    Uint64 carry = 0;
    for (int limb_index = 0; limb_index < 4; limb_index++) {
        Uint64 partial_sum = left[limb_index] + right[limb_index];
        Uint64 first_carry = (partial_sum < left[limb_index]);
        partial_sum += carry;
        Uint64 second_carry = (partial_sum < carry);
        extended_sum[limb_index] = partial_sum;
        carry = first_carry + second_carry;
    }
    extended_sum[4] = carry;

    // Extend the modulus so an overflowed 256-bit sum can be compared correctly.
    Uint64 extended_modulus[5];
    for (int limb_index = 0; limb_index < 4; limb_index++) {
        extended_modulus[limb_index] = modulus[limb_index];
    }
    extended_modulus[4] = 0;

    // Values are already reduced, so at most one subtraction is necessary.
    Uint64 reduced_candidate[5];
    Uint64 borrow = subtract_limbs(extended_sum, extended_modulus, 5, reduced_candidate);
    constant_time_select(output, reduced_candidate, extended_sum, 4,
                         constant_time_mask(borrow ^ 1ull));
    assert(uint256_is_reduced(output, modulus));
}

// Subtract two reduced values, adding the modulus back when the subtraction borrows.
__device__ static void modular_subtract(const Uint64 minuend[4], const Uint64 subtrahend[4],
                                        const Uint64 modulus[4], Uint64 output[4]) {
    assert(uint256_is_reduced(minuend, modulus) && uint256_is_reduced(subtrahend, modulus));
    Uint64 direct_difference[4];
    Uint64 wrapped_difference[4];
    Uint64 borrow = subtract_limbs(minuend, subtrahend, 4, direct_difference);

    // Compute the wrapped candidate unconditionally to keep operand values out of control flow.
    Uint64 carry = 0;
    for (int limb_index = 0; limb_index < 4; limb_index++) {
        Uint64 partial_sum = direct_difference[limb_index] + modulus[limb_index];
        Uint64 first_carry = (partial_sum < direct_difference[limb_index]);
        partial_sum += carry;
        Uint64 second_carry = (partial_sum < carry);
        wrapped_difference[limb_index] = partial_sum;
        carry = first_carry + second_carry;
    }

    // Select the wrapped result only when the original subtraction underflowed.
    constant_time_select(output, wrapped_difference, direct_difference, 4,
                         constant_time_mask(borrow));
    assert(uint256_is_reduced(output, modulus));
}

// Set a 256-bit value from one ordinary 64-bit integer.
__device__ static void set_uint256(Uint64 output[4], Uint64 value) {
    output[0] = value;
    output[1] = 0;
    output[2] = 0;
    output[3] = 0;
}

// Copy one four-limb value.
__device__ static void copy_uint256(Uint64 destination[4], const Uint64 source[4]) {
    for (int limb_index = 0; limb_index < 4; limb_index++) {
        destination[limb_index] = source[limb_index];
    }
}

// Raise `base` to a 256-bit exponent with fixed-work square-and-multiply.
__device__ static void modular_exponentiate(const Uint64 base[4], const Uint64 exponent[4],
                                            const Uint64 modulus[4], const Uint64 barrett_factor[5],
                                            Uint64 output[4]) {
    Uint64 accumulated_result[4];
    set_uint256(accumulated_result, 1);
    Uint64 current_power[4];
    copy_uint256(current_power, base);

    // Process all 256 exponent bits, including leading zeroes, so runtime is value-independent.
    for (int bit_index = 0; bit_index < 256; bit_index++) {
        // Compute the multiply candidate on every iteration.
        Uint64 multiplied_candidate[4];
        modular_multiply(accumulated_result, current_power, modulus, barrett_factor,
                         multiplied_candidate);

        // Select the candidate only when this exponent bit is set.
        Uint64 exponent_bit = (exponent[bit_index >> 6] >> (bit_index & 63)) & 1ull;
        constant_time_select(accumulated_result, multiplied_candidate, accumulated_result, 4,
                             constant_time_mask(exponent_bit));

        // Square the base for the next, more-significant exponent bit.
        Uint64 squared_power[4];
        modular_square(current_power, modulus, barrett_factor, squared_power);
        copy_uint256(current_power, squared_power);
    }

    copy_uint256(output, accumulated_result);
}

// 32-bit word-by-word Montgomery reduction, with R=2^256. For reduced inputs,
// return left*right*R^-1 modulo the odd modulus. The complete 17-word temporary
// retains the 257th bit; dropping it breaks carries near the group order.
// Each inner sum is at most (2^32-1)^2 + 2*(2^32-1) = 2^64-1. REDC leaves
// less than twice the modulus, so one extended, masked subtraction suffices.
// Loading both operands before the final stores permits output/input aliasing.
// Conversion is paid once around a complete inversion, not at each product.
__device__ static void montgomery_multiply(const Uint64 left[4], const Uint64 right[4],
                                           const Uint64 modulus[4], const Uint64 *,
                                           Uint64 output[4]) {
    assert((modulus[0] & 1) == 1);
    assert(uint256_is_reduced(left, modulus) && uint256_is_reduced(right, modulus));
    Uint32 a[8], b[8], m[8], t[17] = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        a[i] = (Uint32)(left[i / 2] >> (32 * (i % 2)));
        b[i] = (Uint32)(right[i / 2] >> (32 * (i % 2)));
        m[i] = (Uint32)(modulus[i / 2] >> (32 * (i % 2)));
    }
    Uint32 inv = 1;
#pragma unroll
    for (int i = 0; i < 5; ++i)
        inv *= 2u - m[0] * inv;
    inv = 0u - inv;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        Uint64 carry = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            Uint64 v = (Uint64)a[i] * b[j] + t[i + j] + carry;
            t[i + j] = (Uint32)v;
            carry = v >> 32;
        }
        t[i + 8] = (Uint32)carry;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        Uint32 q = t[i] * inv;
        Uint64 carry = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            Uint64 v = (Uint64)q * m[j] + t[i + j] + carry;
            t[i + j] = (Uint32)v;
            carry = v >> 32;
        }
#pragma unroll
        for (int k = i + 8; k < 17; ++k) {
            Uint64 v = (Uint64)t[k] + carry;
            t[k] = (Uint32)v;
            carry = v >> 32;
        }
        assert(t[i] == 0 && carry == 0);
    }
    Uint64 r[5], extended[5], reduced[5];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r[i] = (Uint64)t[8 + 2 * i] | ((Uint64)t[9 + 2 * i] << 32);
        extended[i] = modulus[i];
    }
    r[4] = t[16];
    extended[4] = 0;
    Uint64 borrow = subtract_limbs(r, extended, 5, reduced);
    constant_time_select(output, reduced, r, 4, constant_time_mask(borrow ^ 1ull));
    assert(r[4] <= 1 && uint256_is_reduced(output, modulus));
}

// Invert with public, fixed addition chains for P-256's p-2 and n-2. The full
// modulus comparison matters: neither chain is valid for a lookalike modulus.
// All loop counts and named powers come from PUBLIC exponents, never the base.
// The field chain uses 255 squares and 12 multiplies; notably its last multiply
// is essential for p-2 (an inverse-square chain for p-3 is NOT interchangeable).
// The order chain is a four-bit sliding-window schedule in Montgomery form;
// its r2 constant is 2^512 modulo n, not the field prime's Montgomery constant.
// CPU tests symbolically execute these exact schedules; GPU tests compare
// boundary/random inverses and byte-exact RFC 6979 signatures to CPU packages.
// Keep the generic fixed-work Fermat fallback for other public moduli.
__device__ static void modular_inverse(const Uint64 value[4], const Uint64 modulus[4],
                                       const Uint64 barrett_factor[5], Uint64 output[4]) {
    if (modulus[0] == 0xffffffffffffffffull && modulus[1] == 0x00000000ffffffffull &&
        modulus[2] == 0x0000000000000000ull && modulus[3] == 0xffffffff00000001ull) {
        const Uint64 *base = value;
        Uint64 x2[4];
        copy_uint256(x2, base);
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            p256_modular_square(x2, modulus, barrett_factor, x2);
        }
        p256_modular_multiply(x2, base, modulus, barrett_factor, x2);
        Uint64 x4[4];
        copy_uint256(x4, x2);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            p256_modular_square(x4, modulus, barrett_factor, x4);
        }
        p256_modular_multiply(x4, x2, modulus, barrett_factor, x4);
        Uint64 x6[4];
        copy_uint256(x6, x4);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            p256_modular_square(x6, modulus, barrett_factor, x6);
        }
        p256_modular_multiply(x6, x2, modulus, barrett_factor, x6);
        Uint64 x12[4];
        copy_uint256(x12, x6);
#pragma unroll 1
        for (int repeat = 0; repeat < 6; ++repeat) {
            p256_modular_square(x12, modulus, barrett_factor, x12);
        }
        p256_modular_multiply(x12, x6, modulus, barrett_factor, x12);
        Uint64 x24[4];
        copy_uint256(x24, x12);
#pragma unroll 1
        for (int repeat = 0; repeat < 12; ++repeat) {
            p256_modular_square(x24, modulus, barrett_factor, x24);
        }
        p256_modular_multiply(x24, x12, modulus, barrett_factor, x24);
        Uint64 x30[4];
        copy_uint256(x30, x24);
#pragma unroll 1
        for (int repeat = 0; repeat < 6; ++repeat) {
            p256_modular_square(x30, modulus, barrett_factor, x30);
        }
        p256_modular_multiply(x30, x6, modulus, barrett_factor, x30);
        Uint64 x32[4];
        copy_uint256(x32, x30);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            p256_modular_square(x32, modulus, barrett_factor, x32);
        }
        p256_modular_multiply(x32, x2, modulus, barrett_factor, x32);
        Uint64 r[4];
        copy_uint256(r, x32);
#pragma unroll 1
        for (int repeat = 0; repeat < 32; ++repeat) {
            p256_modular_square(r, modulus, barrett_factor, r);
        }
        p256_modular_multiply(r, base, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 128; ++repeat) {
            p256_modular_square(r, modulus, barrett_factor, r);
        }
        p256_modular_multiply(r, x32, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 32; ++repeat) {
            p256_modular_square(r, modulus, barrett_factor, r);
        }
        p256_modular_multiply(r, x32, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 30; ++repeat) {
            p256_modular_square(r, modulus, barrett_factor, r);
        }
        p256_modular_multiply(r, x30, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            p256_modular_square(r, modulus, barrett_factor, r);
        }
        p256_modular_multiply(r, base, modulus, barrett_factor, r);
        copy_uint256(output, r);
        return;
    }
    if (modulus[0] == 0xf3b9cac2fc632551ull && modulus[1] == 0xbce6faada7179e84ull &&
        modulus[2] == 0xffffffffffffffffull && modulus[3] == 0xffffffff00000000ull) {
        Uint64 r2[4] = {0x83244c95be79eea2ull, 0x4699799c49bd6fa6ull, 0x2845b2392b6bec59ull,
                        0x66e12d94f3d95620ull};
        Uint64 one[4] = {1, 0, 0, 0}, base_mont[4];
        montgomery_multiply(value, r2, modulus, barrett_factor, base_mont);
        const Uint64 *base = base_mont;
        Uint64 a2[4];
        copy_uint256(a2, base);
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            montgomery_multiply(a2, a2, modulus, barrett_factor, a2);
        }
        Uint64 a3[4];
        copy_uint256(a3, base);
        montgomery_multiply(a3, a2, modulus, barrett_factor, a3);
        Uint64 a5[4];
        copy_uint256(a5, a3);
        montgomery_multiply(a5, a2, modulus, barrett_factor, a5);
        Uint64 a7[4];
        copy_uint256(a7, a5);
        montgomery_multiply(a7, a2, modulus, barrett_factor, a7);
        Uint64 a9[4];
        copy_uint256(a9, a7);
        montgomery_multiply(a9, a2, modulus, barrett_factor, a9);
        Uint64 a11[4];
        copy_uint256(a11, a9);
        montgomery_multiply(a11, a2, modulus, barrett_factor, a11);
        Uint64 a13[4];
        copy_uint256(a13, a11);
        montgomery_multiply(a13, a2, modulus, barrett_factor, a13);
        Uint64 a15[4];
        copy_uint256(a15, a13);
        montgomery_multiply(a15, a2, modulus, barrett_factor, a15);
        Uint64 r[4];
        copy_uint256(r, a15);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 32; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a11, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a3, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a7, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a13, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a5, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a11, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a13, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a7, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a11, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a3, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, base, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a9, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a7, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a7, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a7, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a7, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a5, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a3, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a11, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a3, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a3, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a9, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 1; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 3; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a5, modulus, barrett_factor, r);
#pragma unroll 1
        for (int repeat = 0; repeat < 2; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
#pragma unroll 1
        for (int repeat = 0; repeat < 4; ++repeat) {
            montgomery_multiply(r, r, modulus, barrett_factor, r);
        }
        montgomery_multiply(r, a15, modulus, barrett_factor, r);
        montgomery_multiply(r, one, modulus, barrett_factor, output);
        return;
    }
    Uint64 exponent[4];
    Uint64 two[4];
    set_uint256(two, 2);
    subtract_limbs(modulus, two, 4, exponent);
    modular_exponentiate(value, exponent, modulus, barrett_factor, output);
}

// References to the P-256 base field modulus and its Barrett reciprocal.
struct FieldParameters {
    const Uint64 *prime;
    const Uint64 *barrett_factor;
};

// Point formulas use the P-256 field only. Avoid a repeated public modulus
// comparison at each multiply/square; the order is never a FieldParameters.
__device__ static void field_multiply(const FieldParameters &field, const Uint64 left[4],
                                      const Uint64 right[4], Uint64 output[4]) {
    p256_modular_multiply(left, right, field.prime, field.barrett_factor, output);
}
__device__ static void field_add(const FieldParameters &field, const Uint64 left[4],
                                 const Uint64 right[4], Uint64 output[4]) {
    modular_add(left, right, field.prime, output);
}
__device__ static void field_subtract(const FieldParameters &field, const Uint64 minuend[4],
                                      const Uint64 subtrahend[4], Uint64 output[4]) {
    modular_subtract(minuend, subtrahend, field.prime, output);
}
__device__ static void field_square(const FieldParameters &field, const Uint64 value[4],
                                    Uint64 output[4]) {
    p256_modular_square(value, field.prime, field.barrett_factor, output);
}
// Double one P-256 point in Jacobian coordinates. The formula takes advantage of P-256's a = -3.
__device__ static void jacobian_double(const FieldParameters &field, const Uint64 input_x[4],
                                       const Uint64 input_y[4], const Uint64 input_z[4],
                                       Uint64 output_x[4], Uint64 output_y[4], Uint64 output_z[4]) {
    Uint64 delta[4];
    Uint64 gamma[4];
    Uint64 beta[4];
    Uint64 alpha[4];
    Uint64 temporary_1[4];
    Uint64 temporary_2[4];
    Uint64 temporary_3[4];

    // delta = Z^2, gamma = Y^2, beta = X * gamma.
    field_square(field, input_z, delta);
    field_square(field, input_y, gamma);
    field_multiply(field, input_x, gamma, beta);

    // alpha = 3 * (X - delta) * (X + delta).
    field_subtract(field, input_x, delta, temporary_1);
    field_add(field, input_x, delta, temporary_2);
    field_multiply(field, temporary_1, temporary_2, temporary_3);
    field_add(field, temporary_3, temporary_3, temporary_1);
    field_add(field, temporary_1, temporary_3, alpha);

    // X3 = alpha^2 - 8 * beta.
    field_square(field, alpha, temporary_1);
    field_add(field, beta, beta, temporary_2);
    field_add(field, temporary_2, temporary_2, temporary_2);
    field_add(field, temporary_2, temporary_2, temporary_2);
    field_subtract(field, temporary_1, temporary_2, output_x);

    // Z3 = (Y + Z)^2 - gamma - delta.
    field_add(field, input_y, input_z, temporary_1);
    field_square(field, temporary_1, temporary_2);
    field_subtract(field, temporary_2, gamma, temporary_3);
    field_subtract(field, temporary_3, delta, output_z);

    // Y3 = alpha * (4 * beta - X3) - 8 * gamma^2.
    field_add(field, beta, beta, temporary_1);
    field_add(field, temporary_1, temporary_1, temporary_1);
    field_subtract(field, temporary_1, output_x, temporary_2);
    field_multiply(field, alpha, temporary_2, temporary_3);
    field_square(field, gamma, temporary_1);
    field_add(field, temporary_1, temporary_1, temporary_2);
    field_add(field, temporary_2, temporary_2, temporary_2);
    field_add(field, temporary_2, temporary_2, temporary_2);
    field_subtract(field, temporary_3, temporary_2, output_y);
}
// Add two P-256 points in Jacobian coordinates. Infinity is represented by Z = 0.
__device__ static void jacobian_add(const FieldParameters &field, const Uint64 first_x[4],
                                    const Uint64 first_y[4], const Uint64 first_z[4],
                                    const Uint64 second_x[4], const Uint64 second_y[4],
                                    const Uint64 second_z[4], Uint64 output_x[4],
                                    Uint64 output_y[4], Uint64 output_z[4]) {
    Uint64 first_z_squared[4];
    Uint64 second_z_squared[4];
    Uint64 first_scaled_x[4];
    Uint64 second_scaled_x[4];
    Uint64 first_scaled_y[4];
    Uint64 second_scaled_y[4];
    Uint64 x_difference[4];
    Uint64 doubled_x_difference_squared[4];
    Uint64 x_difference_cubed_term[4];
    Uint64 doubled_y_difference[4];
    Uint64 first_x_times_squared_difference[4];
    Uint64 temporary_1[4];
    Uint64 temporary_2[4];
    Uint64 temporary_3[4];

    // Scale both X coordinates into the same projective frame.
    field_square(field, first_z, first_z_squared);
    field_square(field, second_z, second_z_squared);
    field_multiply(field, first_x, second_z_squared, first_scaled_x);
    field_multiply(field, second_x, first_z_squared, second_scaled_x);

    // Scale both Y coordinates into that same frame.
    field_multiply(field, second_z, second_z_squared, temporary_1);
    field_multiply(field, first_y, temporary_1, first_scaled_y);
    field_multiply(field, first_z, first_z_squared, temporary_2);
    field_multiply(field, second_y, temporary_2, second_scaled_y);

    // H = U2 - U1; I = (2H)^2; J = H*I; r = 2(S2-S1); V = U1*I.
    field_subtract(field, second_scaled_x, first_scaled_x, x_difference);
    field_add(field, x_difference, x_difference, temporary_1);
    field_square(field, temporary_1, doubled_x_difference_squared);
    field_multiply(field, x_difference, doubled_x_difference_squared, x_difference_cubed_term);
    field_subtract(field, second_scaled_y, first_scaled_y, temporary_2);
    field_add(field, temporary_2, temporary_2, doubled_y_difference);
    field_multiply(field, first_scaled_x, doubled_x_difference_squared,
                   first_x_times_squared_difference);

    // Equal scaled points require doubling rather than the generic addition formula.
    Uint64 equal_points_mask = constant_time_uint256_zero_mask(x_difference) &
                               constant_time_uint256_zero_mask(temporary_2);

    // Compute the generic candidate X3 = r^2 - J - 2V.
    Uint64 generic_x[4];
    Uint64 generic_y[4];
    Uint64 generic_z[4];
    field_square(field, doubled_y_difference, temporary_1);
    field_subtract(field, temporary_1, x_difference_cubed_term, temporary_3);
    field_add(field, first_x_times_squared_difference, first_x_times_squared_difference,
              temporary_1);
    field_subtract(field, temporary_3, temporary_1, generic_x);

    // Compute the generic candidate Y3 = r*(V-X3) - 2*S1*J.
    field_subtract(field, first_x_times_squared_difference, generic_x, temporary_1);
    field_multiply(field, doubled_y_difference, temporary_1, temporary_2);
    field_multiply(field, first_scaled_y, x_difference_cubed_term, temporary_3);
    field_add(field, temporary_3, temporary_3, temporary_3);
    field_subtract(field, temporary_2, temporary_3, generic_y);

    // Compute the generic candidate Z3 = ((Z1+Z2)^2-Z1^2-Z2^2)*H.
    field_add(field, first_z, second_z, temporary_1);
    field_square(field, temporary_1, temporary_2);
    field_subtract(field, temporary_2, first_z_squared, temporary_3);
    field_subtract(field, temporary_3, second_z_squared, temporary_1);
    field_multiply(field, temporary_1, x_difference, generic_z);

    // The ordinary Jacobian formula is incomplete for equal points and either
    // infinity. Compute every candidate and mask-select instead of branching
    // on secret-dependent intermediate values.
    Uint64 doubled_x[4];
    Uint64 doubled_y[4];
    Uint64 doubled_z[4];
    jacobian_double(field, first_x, first_y, first_z, doubled_x, doubled_y, doubled_z);
    constant_time_select(output_x, doubled_x, generic_x, 4, equal_points_mask);
    constant_time_select(output_y, doubled_y, generic_y, 4, equal_points_mask);
    constant_time_select(output_z, doubled_z, generic_z, 4, equal_points_mask);

    // If the second input is infinity, select the first input unchanged.
    Uint64 second_is_infinity_mask = constant_time_uint256_zero_mask(second_z);
    constant_time_select(output_x, first_x, output_x, 4, second_is_infinity_mask);
    constant_time_select(output_y, first_y, output_y, 4, second_is_infinity_mask);
    constant_time_select(output_z, first_z, output_z, 4, second_is_infinity_mask);

    // If the first input is infinity, select the second input unchanged.
    Uint64 first_is_infinity_mask = constant_time_uint256_zero_mask(first_z);
    constant_time_select(output_x, second_x, output_x, 4, first_is_infinity_mask);
    constant_time_select(output_y, second_y, output_y, 4, first_is_infinity_mask);
    constant_time_select(output_z, second_z, output_z, 4, first_is_infinity_mask);
}
#define P256_WINDOW_BITS 6
#define P256_WINDOWS ((256 + P256_WINDOW_BITS - 1) / P256_WINDOW_BITS)
#define P256_WINDOW_POINTS ((1 << P256_WINDOW_BITS) - 1)
#define P256_POINT_LIMBS 12
// The key-generation thread constructs this table once from the compiled scalar implementation.
// Keeping it module-private avoids accepting host-precomputed curve points. Each window contains
// the 63 nonzero multiples needed by a six-bit digit, stored as X, Y, and Z limb arrays.
__device__ static Uint64
        g_p256_generator_table[P256_WINDOWS * P256_WINDOW_POINTS * P256_POINT_LIMBS];
__device__ static int g_p256_generator_table_ready = 0;

// Build the fixed-base table used by every later public-key derivation and ECDSA signature.
__device__ static void build_p256_generator_table(const Uint64 *p256_context) {
    // Reference the host-uploaded field constants.
    FieldParameters field;
    field.prime = p256_context + CONTEXT_FIELD_PRIME_OFFSET;
    field.barrett_factor = p256_context + CONTEXT_FIELD_BARRETT_FACTOR_OFFSET;

    // Start window zero at the standard P-256 generator in Jacobian coordinates.
    Uint64 window_base_x[4];
    Uint64 window_base_y[4];
    Uint64 window_base_z[4];
    copy_uint256(window_base_x, p256_context + CONTEXT_GENERATOR_X_OFFSET);
    copy_uint256(window_base_y, p256_context + CONTEXT_GENERATOR_Y_OFFSET);
    set_uint256(window_base_z, 1);

    // Each successive window base is the preceding base multiplied by 2^6.
    for (int window_index = 0; window_index < P256_WINDOWS; window_index++) {
        Uint64 multiple_x[4];
        Uint64 multiple_y[4];
        Uint64 multiple_z[4];
        copy_uint256(multiple_x, window_base_x);
        copy_uint256(multiple_y, window_base_y);
        copy_uint256(multiple_z, window_base_z);

        // Store the 63 nonzero multiples of this window's base point.
        for (int digit = 1; digit <= P256_WINDOW_POINTS; digit++) {
            Uint64 *stored_point =
                    &g_p256_generator_table[(window_index * P256_WINDOW_POINTS + digit - 1) *
                                            P256_POINT_LIMBS];
            for (int limb_index = 0; limb_index < 4; limb_index++) {
                stored_point[limb_index] = multiple_x[limb_index];
                stored_point[4 + limb_index] = multiple_y[limb_index];
                stored_point[8 + limb_index] = multiple_z[limb_index];
            }

            // Add the window base once to obtain the next multiple.
            if (digit < P256_WINDOW_POINTS) {
                Uint64 next_x[4];
                Uint64 next_y[4];
                Uint64 next_z[4];
                jacobian_add(field, multiple_x, multiple_y, multiple_z, window_base_x,
                             window_base_y, window_base_z, next_x, next_y, next_z);
                copy_uint256(multiple_x, next_x);
                copy_uint256(multiple_y, next_y);
                copy_uint256(multiple_z, next_z);
            }
        }

        // Double six times to advance the base point to the next scalar window.
        for (int bit_index = 0; bit_index < P256_WINDOW_BITS; bit_index++) {
            Uint64 doubled_x[4];
            Uint64 doubled_y[4];
            Uint64 doubled_z[4];
            jacobian_double(field, window_base_x, window_base_y, window_base_z, doubled_x,
                            doubled_y, doubled_z);
            copy_uint256(window_base_x, doubled_x);
            copy_uint256(window_base_y, doubled_y);
            copy_uint256(window_base_z, doubled_z);
        }
    }

    // Publish all table writes before advertising the table as ready.
    __threadfence();
    g_p256_generator_table_ready = 1;
}

// Return all-one bits when two 64-bit values are equal, otherwise all-zero bits.
__device__ __forceinline__ Uint64 constant_time_equality_mask(Uint64 left, Uint64 right) {
    Uint64 difference = left ^ right;
    Uint64 is_nonzero = (difference | (0ull - difference)) >> 63;
    return constant_time_mask(is_nonzero ^ 1ull);
}

// Multiply the standard P-256 generator by a secret scalar with fixed-window, constant work.
// A cooperative group MUST have identical scalar/context inputs and all its
// lanes must call together. Ordinary key generation and independent oracle
// lanes use width one. A template makes unsupported widths a compile error.
template <int TableWidth = 1>
__device__ static void multiply_generator_by_scalar(const Uint64 *p256_context,
                                                    const Uint64 scalar[4], Uint64 affine_x[4],
                                                    Uint64 affine_y[4]) {
    static_assert(TableWidth >= 1 && TableWidth <= 16 && (TableWidth & (TableWidth - 1)) == 0,
                  "invalid P-256 table group width");
    assert(p256_context != 0 && scalar != 0 && affine_x != 0 && affine_y != 0);
    assert(g_p256_generator_table_ready == 1);
    // Reference the base-field constants used by the point formulas.
    FieldParameters field;
    field.prime = p256_context + CONTEXT_FIELD_PRIME_OFFSET;
    field.barrett_factor = p256_context + CONTEXT_FIELD_BARRETT_FACTOR_OFFSET;

    // Initialize the accumulator to the point at infinity (Z = 0).
    Uint64 accumulated_x[4];
    Uint64 accumulated_y[4];
    Uint64 accumulated_z[4];
    set_uint256(accumulated_x, 0);
    set_uint256(accumulated_y, 0);
    set_uint256(accumulated_z, 0);

    // Process every six-bit scalar window, including leading zero windows.
#pragma unroll 1
    for (int window_index = 0; window_index < P256_WINDOWS; window_index++) {
        // Extract this window's six bits, joining adjacent limbs when necessary.
        int first_bit_index = window_index * P256_WINDOW_BITS;
        int first_limb_index = first_bit_index >> 6;
        int shift_within_limb = first_bit_index & 63;
        assert(first_limb_index >= 0 && first_limb_index < 4);
        Uint64 secret_digit = scalar[first_limb_index] >> shift_within_limb;
        if (shift_within_limb > 64 - P256_WINDOW_BITS && first_limb_index < 3) {
            secret_digit |= scalar[first_limb_index + 1] << (64 - shift_within_limb);
        }
        secret_digit &= (1ull << P256_WINDOW_BITS) - 1ull;

        // Initialize the selected point to infinity, which represents a zero digit.
        Uint64 selected_x[4];
        Uint64 selected_y[4];
        Uint64 selected_z[4];
        set_uint256(selected_x, 0);
        set_uint256(selected_y, 0);
        set_uint256(selected_z, 0);

        // Scan all 63 points instead of indexing global memory with the secret digit.
#pragma unroll 1
        // Each group has identical signing inputs. Partition the PUBLIC
        // candidate scan, never the secret digit or a secret-indexed address.
        for (int candidate = 1 + ((int)threadIdx.x & (TableWidth - 1));
             candidate <= P256_WINDOW_POINTS; candidate += TableWidth) {
            const Uint64 *candidate_point =
                &g_p256_generator_table[(window_index * P256_WINDOW_POINTS + candidate - 1) *
                                        P256_POINT_LIMBS];
            Uint64 selection_mask = constant_time_equality_mask(secret_digit, (Uint64)candidate);
            for (int limb_index = 0; limb_index < 4; limb_index++) {
                selected_x[limb_index] |= candidate_point[limb_index] & selection_mask;
                selected_y[limb_index] |= candidate_point[4 + limb_index] & selection_mask;
                selected_z[limb_index] |= candidate_point[8 + limb_index] & selection_mask;
            }
        }

        if (TableWidth > 1) {
            unsigned group_base = ((unsigned)threadIdx.x & 31u) & ~(unsigned)(TableWidth - 1);
            unsigned group_mask = ((1u << TableWidth) - 1u) << group_base;
            // Reconverge the complete group after its public, uneven 63-point
            // scan. XOR-shuffle OR reduction gives every replica the same point.
#pragma unroll
            for (int offset = TableWidth / 2; offset > 0; offset >>= 1) {
#pragma unroll
                for (int limb = 0; limb < 4; ++limb) {
                    selected_x[limb] |=
                        __shfl_xor_sync(group_mask, selected_x[limb], offset, TableWidth);
                    selected_y[limb] |=
                        __shfl_xor_sync(group_mask, selected_y[limb], offset, TableWidth);
                    selected_z[limb] |=
                        __shfl_xor_sync(group_mask, selected_z[limb], offset, TableWidth);
                }
            }
        }

        // Always add one selected point; adding infinity handles a zero digit without a branch.
        Uint64 next_x[4];
        Uint64 next_y[4];
        Uint64 next_z[4];
        jacobian_add(field, accumulated_x, accumulated_y, accumulated_z, selected_x, selected_y,
                     selected_z, next_x, next_y, next_z);
        copy_uint256(accumulated_x, next_x);
        copy_uint256(accumulated_y, next_y);
        copy_uint256(accumulated_z, next_z);
    }

    // Convert (X:Y:Z) to affine (X/Z^2, Y/Z^3) with one field inversion.
    Uint64 inverse_z[4];
    Uint64 inverse_z_squared[4];
    Uint64 inverse_z_cubed[4];
    modular_inverse(accumulated_z, field.prime, field.barrett_factor, inverse_z);
    field_square(field, inverse_z, inverse_z_squared);
    field_multiply(field, inverse_z_squared, inverse_z, inverse_z_cubed);
    field_multiply(field, accumulated_x, inverse_z_squared, affine_x);
    field_multiply(field, accumulated_y, inverse_z_cubed, affine_y);
}
// Decode a 32-byte big-endian integer into four little-endian 64-bit limbs.
__device__ static void big_endian_bytes_to_uint256(const Byte input_bytes[32],
                                                   Uint64 output_limbs[4]) {
    for (int input_limb_index = 0; input_limb_index < 4; input_limb_index++) {
        Uint64 decoded_limb = 0;
        for (int byte_index = 0; byte_index < 8; byte_index++) {
            decoded_limb = (decoded_limb << 8) | input_bytes[input_limb_index * 8 + byte_index];
        }
        output_limbs[3 - input_limb_index] = decoded_limb;
    }
}

// Encode four little-endian 64-bit limbs as a 32-byte big-endian integer.
__device__ static void uint256_to_big_endian_bytes(const Uint64 input_limbs[4],
                                                   Byte output_bytes[32]) {
    for (int output_limb_index = 0; output_limb_index < 4; output_limb_index++) {
        Uint64 encoded_limb = input_limbs[3 - output_limb_index];
        for (int byte_index = 0; byte_index < 8; byte_index++) {
            output_bytes[output_limb_index * 8 + byte_index] =
                    (Byte)(encoded_limb >> (56 - 8 * byte_index));
        }
    }
}

// Decode a 32-byte integer and reduce it once modulo the P-256 group order.
__device__ static void reduce_big_endian_bytes_mod_order(const Byte input_bytes[32],
                                                         const Uint64 *group_order,
                                                         Uint64 output[4]) {
    Uint64 decoded_value[4];
    big_endian_bytes_to_uint256(input_bytes, decoded_value);
    reduce_uint256_once(decoded_value, group_order, output);
}

// Derive a deterministic ECDSA nonce with RFC 6979 section 3.2's HMAC-DRBG procedure.
__device__ static int derive_rfc6979_nonce(const Uint64 *group_order,
                                           const Uint64 private_scalar[4],
                                           const Byte message_hash[32], Uint64 output_nonce[4]) {
    // RFC 6979 calls these fixed-width representations int2octets(x) and bits2octets(h1).
    Byte private_scalar_octets[32];
    uint256_to_big_endian_bytes(private_scalar, private_scalar_octets);
    Byte reduced_hash_octets[32];
    {
        Uint64 reduced_hash[4];
        reduce_big_endian_bytes_mod_order(message_hash, group_order, reduced_hash);
        uint256_to_big_endian_bytes(reduced_hash, reduced_hash_octets);
    }

    // Initialize the RFC 6979 HMAC-DRBG state: V = 0x01^32 and K = 0x00^32.
    Byte value_state[32];
    Byte key_state[32];
    for (int byte_index = 0; byte_index < 32; byte_index++) {
        value_state[byte_index] = 0x01;
        key_state[byte_index] = 0x00;
    }

    // Update K and V with V || 0x00 || private_scalar || reduced_hash.
    Byte seed_material[97];
    for (int byte_index = 0; byte_index < 32; byte_index++) {
        seed_material[byte_index] = value_state[byte_index];
    }
    seed_material[32] = 0x00;
    for (int byte_index = 0; byte_index < 32; byte_index++) {
        seed_material[33 + byte_index] = private_scalar_octets[byte_index];
        seed_material[65 + byte_index] = reduced_hash_octets[byte_index];
    }
    hmac_sha256(key_state, seed_material, 97, key_state);
    hmac_sha256(key_state, value_state, 32, value_state);

    // Repeat with V || 0x01 || private_scalar || reduced_hash.
    for (int byte_index = 0; byte_index < 32; byte_index++) {
        seed_material[byte_index] = value_state[byte_index];
    }
    seed_material[32] = 0x01;
    for (int byte_index = 0; byte_index < 32; byte_index++) {
        seed_material[33 + byte_index] = private_scalar_octets[byte_index];
        seed_material[65 + byte_index] = reduced_hash_octets[byte_index];
    }
    hmac_sha256(key_state, seed_material, 97, key_state);
    hmac_sha256(key_state, value_state, 32, value_state);

    // Four fixed candidates keep the schedule independent of secret data.
    // A P-256 candidate is rejected with probability below 2^-32, so failure
    // after all four is below 2^-128; fail closed instead of using a fallback.
    set_uint256(output_nonce, 1);
    Uint64 candidate_was_selected = 0;
    volatile Uint64 schedule_guard = 0;
    for (int candidate_index = 0; candidate_index < 4; candidate_index++) {
        // Generate a candidate from the next V value.
        hmac_sha256(key_state, value_state, 32, value_state);
        Uint64 candidate[4];
        Uint64 unused_difference[4];
        big_endian_bytes_to_uint256(value_state, candidate);

        // Prevent the compiler from deleting fixed-work rounds after the first accepted candidate.
        schedule_guard ^= candidate[0];

        // A valid ECDSA nonce is in the closed interval [1, group_order - 1].
        Uint64 candidate_is_below_order =
                subtract_limbs(candidate, group_order, 4, unused_difference);
        Uint64 candidate_is_valid =
                constant_time_uint256_is_nonzero(candidate) & candidate_is_below_order;
        Uint64 select_this_candidate = candidate_is_valid & (candidate_was_selected ^ 1ull);

        // Keep the first valid candidate without branching on the candidate value.
        constant_time_select(output_nonce, candidate, output_nonce, 4,
                             constant_time_mask(select_this_candidate));
        candidate_was_selected |= candidate_is_valid;

        // Perform RFC 6979's rejection update after every candidate to keep a fixed schedule.
        Byte rejection_material[33];
        for (int byte_index = 0; byte_index < 32; byte_index++) {
            rejection_material[byte_index] = value_state[byte_index];
        }
        rejection_material[32] = 0x00;
        hmac_sha256(key_state, rejection_material, 33, key_state);
        hmac_sha256(key_state, value_state, 32, value_state);
    }

    // Fail closed in the negligible event that all four candidates were invalid.
    return candidate_was_selected ? 0 : -1;
}

// Sign a 32-byte message hash with deterministic P-256 ECDSA.
// TableWidth replicas must have identical signing inputs; only their leader
// may publish a signature. See multiply_generator_by_scalar's group contract.
template<int TableWidth = 1>
__device__ static int sign_ecdsa_p256(const Uint64 *p256_context, const Uint64 private_scalar[4],
                                      const Byte message_hash[32], Uint64 output_r[4],
                                      Uint64 output_s[4]) {
    // Reference the group order and its Barrett reciprocal.
    const Uint64 *group_order = p256_context + CONTEXT_GROUP_ORDER_OFFSET;
    const Uint64 *order_barrett_factor = p256_context + CONTEXT_ORDER_BARRETT_FACTOR_OFFSET;

    // Derive deterministic k; propagate the all-candidates-rejected failure.
    Uint64 nonce[4];
    if (derive_rfc6979_nonce(group_order, private_scalar, message_hash, nonce) != 0) {
        return -1;
    }

    // Compute R = nonce * G and r = R.x mod group_order.
    Uint64 nonce_point_x[4];
    Uint64 nonce_point_y[4];
    multiply_generator_by_scalar<TableWidth>(p256_context, nonce, nonce_point_x, nonce_point_y);
    Uint64 signature_r[4];
    reduce_uint256_once(nonce_point_x, group_order, signature_r);

    // Decode z from the message hash and compute nonce^-1.
    Uint64 reduced_message_hash[4];
    reduce_big_endian_bytes_mod_order(message_hash, group_order, reduced_message_hash);
    Uint64 inverse_nonce[4];
    modular_inverse(nonce, group_order, order_barrett_factor, inverse_nonce);

    // Compute s = nonce^-1 * (z + r * private_scalar) mod group_order.
    Uint64 r_times_private_scalar[4];
    modular_multiply(signature_r, private_scalar, group_order, order_barrett_factor,
                     r_times_private_scalar);
    Uint64 hash_plus_product[4];
    modular_add(reduced_message_hash, r_times_private_scalar, group_order, hash_plus_product);
    Uint64 signature_s[4];
    modular_multiply(inverse_nonce, hash_plus_product, group_order, order_barrett_factor,
                     signature_s);

    // Return both signature scalars to the caller.
    copy_uint256(output_r, signature_r);
    copy_uint256(output_s, signature_s);

    // ECDSA forbids zero r or s. Report failure without branching on either scalar.
    Uint64 invalid_signature_mask = constant_time_uint256_zero_mask(signature_r) |
                                    constant_time_uint256_zero_mask(signature_s);
    return -(int)(invalid_signature_mask & 1ull);
}

// ===== MODULE-PRIVATE NOTARY STATE =====

// The private scalar is created and consumed only by device code; no entry point exports it.
__device__ static Byte g_private_scalar_big_endian[32];

// Signing helpers reject requests until key generation has completed.
__device__ static int g_session_key_ready = 0;

// The public key is retained in SEC 1 compressed form for did:key construction.
__device__ static Byte g_compressed_public_key[33];

// This private one-shot handoff is the only path from measurement to receipt construction. The
// signing entry point deliberately has no host-provided model-root parameter.
__device__ static Byte g_pending_model_root[32];
__device__ static int g_pending_tensor_count = 0;
__device__ static int g_pending_receipt_armed = 0;

// ===== PER-INSTANCE STATEMENT CHAINS =====
//
// Every resident model copy -- identified by the instanceID folded over its
// hashed IPC handles and extents -- owns a separate chain, so a statement for
// one copy can never link to another's. Two byte-identical models share a
// modelRoot but not an instanceID, which is the case this table exists for.
//
// The table is module-private for the same reason the private scalar is: a
// host able to rewrite these links could fork a chain or replay a predecessor,
// and keeping them in device storage is what prevents it. The host still
// chooses which chain to extend, because it supplies the instanceID, so the
// chain is exactly as trustworthy as that value.
//
// CUATTEST_CHAIN_SLOTS_MAX is a compile-time ceiling and therefore covered by
// cubinCID; the host selects the active count at initialization and can never
// exceed it. A lookup miss with every active slot in use is an error rather
// than an eviction: silently reusing a slot would emit a genesis statement for
// a copy that already has predecessors, which no verifier could distinguish
// from a forked chain.
#ifndef CUATTEST_CHAIN_SLOTS_MAX
#define CUATTEST_CHAIN_SLOTS_MAX 64
#endif
static_assert(CUATTEST_CHAIN_SLOTS_MAX > 0 && CUATTEST_CHAIN_SLOTS_MAX <= 4096,
              "chain slot ceiling must be positive and bounded");

struct ChainEntry {
    Byte instance_root[32];        // key: which resident copy
    Byte last_credential_uuid[16];  // UUID of that copy's most recent StateAttestation
    int in_use;
};

// Zero-initialized by module load, so every slot starts free.
__device__ static ChainEntry g_chains[CUATTEST_CHAIN_SLOTS_MAX];

// Active slot count, set once at initialization. Zero until then, which makes
// every signing attempt fail closed rather than silently chain nothing.
__device__ static int g_chain_slots = 0;

// Return the active slot already chaining `instance_root`, or -1 if this copy
// has never been signed in this session.
__device__ static int chain_find(const Byte instance_root[32]) {
    assert(instance_root != 0);
    for (int slot = 0; slot < g_chain_slots; slot++) {
        if (!g_chains[slot].in_use) {
            continue;
        }
        int matches = 1;
        for (int index = 0; index < 32; index++) {
            if (g_chains[slot].instance_root[index] != instance_root[index]) {
                matches = 0;
                break;
            }
        }
        if (matches) {
            return slot;
        }
    }
    return -1;
}

// Return the lowest free slot, or -1 when the table is full. Claiming is
// deliberately separate from finding: both receipt passes must observe the
// same predecessor, so nothing is mutated until the second pass commits.
__device__ static int chain_free_slot() {
    for (int slot = 0; slot < g_chain_slots; slot++) {
        if (!g_chains[slot].in_use) {
            return slot;
        }
    }
    return -1;
}

// Publish this copy's new chain head. Called exactly once per signature, only
// after the receipt has been assembled successfully -- advancing earlier would
// leave the next statement pointing at a predecessor that was never emitted.
__device__ static void chain_commit(int slot, const Byte instance_root[32],
                                    const Byte credential_uuid[16]) {
    assert(slot >= 0 && slot < g_chain_slots);
    assert(instance_root != 0 && credential_uuid != 0);
    for (int index = 0; index < 32; index++) {
        g_chains[slot].instance_root[index] = instance_root[index];
    }
    for (int index = 0; index < 16; index++) {
        g_chains[slot].last_credential_uuid[index] = credential_uuid[index];
    }
    g_chains[slot].in_use = 1;
}

// Derive one session key as SHA256(host_entropy) mod group_order.
//
// This kernel is launched as exactly one block containing one thread. The 32-byte input comes from
// the operating-system CSPRNG and is the sole credited entropy source. The notary process supplies
// and therefore knows that seed; keeping the derived scalar in device storage isolates it from the
// untrusted workload, not from the trusted notary host. CUDA exposes no documented device-side
// entropy primitive, and timer values, scheduling variation, data races, and unspecified atomic
// arbitration are not substitutes for one. Only the public point is copied out of device storage.
extern "C" __global__ void keygen_kernel(const Uint64 *p256_context, const Byte *host_entropy,
                                         Byte *output_public_x, Byte *output_public_y) {
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.x == 1 && blockDim.y == 1 && blockDim.z == 1);
    assert(p256_context != 0 && host_entropy != 0 && output_public_x != 0 && output_public_y != 0);
    // Reference the P-256 group order used to map the digest into a private scalar.
    const Uint64 *group_order = p256_context + CONTEXT_GROUP_ORDER_OFFSET;

    // Condition the fixed-size OS seed and reduce the digest modulo the P-256 group order.
    Byte private_scalar_digest[32];
    sha256_digest(host_entropy, 32, private_scalar_digest);
    Uint64 private_scalar[4];
    reduce_big_endian_bytes_mod_order(private_scalar_digest, group_order, private_scalar);

    // ECDSA requires a nonzero scalar; replace the single zero value with one using a mask.
    Uint64 scalar_one[4];
    set_uint256(scalar_one, 1);
    constant_time_select(private_scalar, scalar_one, private_scalar, 4,
                         constant_time_uint256_zero_mask(private_scalar));

    // Construct the fixed-base table before the first scalar multiplication.
    if (!g_p256_generator_table_ready) {
        build_p256_generator_table(p256_context);
    }

    // Retain the private scalar only in module-private device storage.
    uint256_to_big_endian_bytes(private_scalar, g_private_scalar_big_endian);

    // Derive and export the public affine point private_scalar * G.
    Uint64 public_x[4];
    Uint64 public_y[4];
    multiply_generator_by_scalar(p256_context, private_scalar, public_x, public_y);
    uint256_to_big_endian_bytes(public_x, output_public_x);
    uint256_to_big_endian_bytes(public_y, output_public_y);

    // Cache the compressed public key: 0x02/0x03 according to Y parity, followed by X.
    g_compressed_public_key[0] = (Byte)(0x02 | (output_public_y[31] & 1));
    for (int byte_index = 0; byte_index < 32; byte_index++) {
        g_compressed_public_key[1 + byte_index] = output_public_x[byte_index];
    }

    // Publish readiness only after private and public key material is complete.
    g_session_key_ready = 1;

    // Best-effort clearing removes sensitive temporary copies from this thread's local storage.
    for (int limb_index = 0; limb_index < 4; limb_index++) {
        private_scalar[limb_index] = 0;
    }
    for (int byte_index = 0; byte_index < 32; byte_index++) {
        private_scalar_digest[byte_index] = 0;
    }
}

// Set how many per-instance chain slots this session may use.
//
// Launched once, as one thread, during notary startup. The value is bounded by
// the compile-time ceiling rather than trusted outright, and the table is only
// ever widened from zero at initialization -- shrinking it later would strand
// live chains in slots no lookup would scan again, turning their next
// statement into a silent genesis.
extern "C" __global__ void configure_chain_slots_kernel(int slots, int *status) {
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.x == 1 && blockDim.y == 1 && blockDim.z == 1);
    assert(status != 0);
    if (slots <= 0 || slots > CUATTEST_CHAIN_SLOTS_MAX) {
        *status = -8;
        return;
    }
    if (g_chain_slots != 0) {
        *status = -9;
        return;
    }
    g_chain_slots = slots;
    *status = 0;
}

// ===== SINGLE-THREAD STREAMING BLAKE3 =====
//
// Receipt-sized buffers and the final model-root fold are too small to benefit from CUDA-wide
// scheduling. This conventional streaming implementation keeps one pending semantic chunk plus a
// stack of completed subtree chaining values. When the completed-chunk count carries through a
// binary bit, equal-sized subtrees merge exactly as specified by BLAKE3. Fifty-four stack levels
// cover more bytes than any supported GPU can hold.
struct Blake3StreamingState {
    Uint32 subtree_stack[54][8];
    int subtree_count;
    Byte pending_chunk[1024];
    int pending_byte_count;
    Uint64 completed_chunk_count;
};

// Initialize an empty streaming BLAKE3 state.
__device__ static void blake3_stream_initialize(Blake3StreamingState *state) {
    assert(state != 0);
    state->subtree_count = 0;
    state->pending_byte_count = 0;
    state->completed_chunk_count = 0;
}
// Merge a completed chunk CV through every binary carry, then push the resulting subtree.
__device__ static void blake3_stream_push_chunk_cv(Blake3StreamingState *state,
                                                   Uint32 chaining_value[8],
                                                   Uint64 total_chunk_count) {
    assert(total_chunk_count != 0 && total_chunk_count < (1ull << 54));
    assert(state->subtree_count >= 0 && state->subtree_count < 54);
    // Every trailing zero in the new chunk count joins two equal-sized adjacent subtrees.
    while ((total_chunk_count & 1ull) == 0) {
        assert(state->subtree_count > 0);
        state->subtree_count--;
        Uint32 parent_value[8];
        blake3_compress_parent(state->subtree_stack[state->subtree_count], chaining_value, 0,
                               parent_value);
        for (int word_index = 0; word_index < 8; word_index++) {
            chaining_value[word_index] = parent_value[word_index];
        }
        total_chunk_count >>= 1;
    }

    // Retain the unmatched subtree until a future chunk or finalization can merge it.
    assert(state->subtree_count < 54);
    for (int word_index = 0; word_index < 8; word_index++) {
        state->subtree_stack[state->subtree_count][word_index] = chaining_value[word_index];
    }
    state->subtree_count++;
}

// Append bytes while deliberately retaining the final chunk for ROOT-aware finalization.
__device__ static void blake3_stream_update(Blake3StreamingState *state, const Byte *input_bytes,
                                            Uint64 input_byte_count) {
    assert(state != 0 && (input_byte_count == 0 || input_bytes != 0));
    assert(state->pending_byte_count >= 0 && state->pending_byte_count <= 1024);
    assert(state->subtree_count == __popcll(state->completed_chunk_count));
    while (input_byte_count != 0) {
        // A full pending chunk becomes non-final only when another input byte arrives.
        if (state->pending_byte_count == 1024) {
            Uint32 chunk_chaining_value[8];
            blake3_hash_chunk(state->pending_chunk, 1024, state->completed_chunk_count, 0,
                              chunk_chaining_value);
            blake3_stream_push_chunk_cv(state, chunk_chaining_value,
                                        state->completed_chunk_count + 1);
            state->completed_chunk_count++;
            state->pending_byte_count = 0;
        }

        // Copy as much input as fits in the pending semantic chunk.
        int bytes_to_copy = 1024 - state->pending_byte_count;
        if ((Uint64)bytes_to_copy > input_byte_count) {
            bytes_to_copy = (int)input_byte_count;
        }
        for (int byte_index = 0; byte_index < bytes_to_copy; byte_index++) {
            state->pending_chunk[state->pending_byte_count + byte_index] = input_bytes[byte_index];
        }

        // Advance both source and destination cursors.
        state->pending_byte_count += bytes_to_copy;
        input_bytes += bytes_to_copy;
        input_byte_count -= bytes_to_copy;
    }
}

// Finish the BLAKE3 tree and serialize its 32-byte root digest in little-endian word order.
__device__ static void blake3_stream_finalize(Blake3StreamingState *state, Byte output_digest[32]) {
    assert(state != 0 && output_digest != 0);
    assert(state->pending_byte_count >= 0 && state->pending_byte_count <= 1024);
    assert(state->subtree_count == __popcll(state->completed_chunk_count));
    assert(state->completed_chunk_count == 0 || state->pending_byte_count > 0);
    Uint32 root_chaining_value[8];
    if (state->completed_chunk_count == 0) {
        // A one-chunk input receives ROOT during the chunk's final block compression.
        blake3_hash_chunk(state->pending_chunk, (Uint32)state->pending_byte_count, 0, 1,
                          root_chaining_value);
    } else {
        // Hash the retained final chunk without ROOT because at least one left subtree exists.
        Uint32 right_subtree[8];
        blake3_hash_chunk(state->pending_chunk, (Uint32)state->pending_byte_count,
                          state->completed_chunk_count, 0, right_subtree);

        // Fold stack entries from newest to oldest, preserving BLAKE3's left/right ordering.
        for (int stack_index = state->subtree_count - 1; stack_index >= 1; stack_index--) {
            Uint32 parent_value[8];
            blake3_compress_parent(state->subtree_stack[stack_index], right_subtree, 0,
                                   parent_value);
            for (int word_index = 0; word_index < 8; word_index++) {
                right_subtree[word_index] = parent_value[word_index];
            }
        }

        // Apply ROOT only to the final parent compression.
        blake3_compress_parent(state->subtree_stack[0], right_subtree, 1, root_chaining_value);
    }

    // BLAKE3 digest bytes are the little-endian serialization of the root output words.
    for (int word_index = 0; word_index < 8; word_index++) {
        output_digest[4 * word_index] = (Byte)root_chaining_value[word_index];
        output_digest[4 * word_index + 1] = (Byte)(root_chaining_value[word_index] >> 8);
        output_digest[4 * word_index + 2] = (Byte)(root_chaining_value[word_index] >> 16);
        output_digest[4 * word_index + 3] = (Byte)(root_chaining_value[word_index] >> 24);
    }
}

// ===== RECEIPT ENCODING AND IN-KERNEL STATEMENT ASSEMBLY =====
//
// The receipt is intentionally constructed from compiled-in templates on the GPU. This prevents
// the host from choosing document bytes for the device key to sign. The result contains a signed
// measurement document plus the notary-issued CredentialRegistration wrapping
// its StateAttestation. Each statement identifier is:
//
//   canonical N-Quads --BLAKE3--> 32-byte digest --CIDv1/RDFC codec--> statement @id
//
// The fixed templates reproduce the EQTY fields; multi-blank-node wrappers are canonicalized
// from their actual quads before hashing, since role order is not RDFC label order. A
// deterministic credential UUID comes from SHA256(model_root || subject). Host-supplied timestamp
// and model name fields are validated before any signing hash is accepted.

// A bounds-checking writer shared by JSON, N-Quads, hexadecimal, and decimal encoders.
struct BufferWriter {
    Byte *destination;
    int length;
    int capacity;
    int overflowed;
};

// Append a byte range, counting the required length even after the fixed output buffer fills.
__device__ static void writer_append_bytes(BufferWriter *writer, const Byte *source,
                                           int byte_count) {
    assert(writer != 0 && byte_count >= 0);
    assert(writer->length >= 0 && writer->capacity >= 0);
    // An overflowed counting writer intentionally has length >= capacity.
    // Preserve its graceful error path rather than asserting that it fits.
    assert(byte_count <= 0x7fffffff - writer->length);
    assert(byte_count == 0 || source != 0);
    assert(writer->capacity == 0 || writer->destination != 0);
    for (int byte_index = 0; byte_index < byte_count; byte_index++) {
        if (writer->length < writer->capacity) {
            writer->destination[writer->length] = source[byte_index];
        }
        writer->length++;
    }
    if (writer->length >= writer->capacity) {
        writer->overflowed = 1;
    }
}

// Append a null-terminated compile-time string without writing its terminator.
__device__ static void writer_append_c_string(BufferWriter *writer, const char *text) {
    int text_length = 0;
    while (text[text_length] != '\0') {
        text_length++;
    }
    writer_append_bytes(writer, (const Byte *)text, text_length);
}

// Append bytes as lower-case hexadecimal.
__device__ static void writer_append_hex(BufferWriter *writer, const Byte *bytes, int byte_count) {
    const char *hex_alphabet = "0123456789abcdef";
    for (int byte_index = 0; byte_index < byte_count; byte_index++) {
        Byte encoded_pair[2] = {(Byte)hex_alphabet[bytes[byte_index] >> 4],
                                (Byte)hex_alphabet[bytes[byte_index] & 15]};
        writer_append_bytes(writer, encoded_pair, 2);
    }
}

// Append an unsigned 32-bit value as base-10 ASCII.
__device__ static void writer_append_uint32(BufferWriter *writer, Uint32 value) {
    // Generate digits least-significant first into a bounded temporary buffer.
    char reversed_digits[10];
    int digit_count = 0;
    if (value == 0) {
        reversed_digits[digit_count++] = '0';
    }
    while (value != 0) {
        reversed_digits[digit_count++] = (char)('0' + value % 10);
        value /= 10;
    }

    // Reverse the temporary digits as they are appended to the output.
    while (digit_count > 0) {
        Byte digit = (Byte)reversed_digits[--digit_count];
        writer_append_bytes(writer, &digit, 1);
    }
}

// Encode a big-endian byte string with the Bitcoin base58 alphabet used by did:key.
__device__ static int base58_encode(const Byte *input, int input_byte_count, char *output,
                                    int output_capacity) {
    assert(input_byte_count >= 0 && input_byte_count <= 64 && output_capacity >= 0);
    const char *base58_alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Copy the input because repeated long division updates it in place.
    Byte current_number[64];
    for (int byte_index = 0; byte_index < input_byte_count; byte_index++) {
        current_number[byte_index] = input[byte_index];
    }
    int current_byte_count = input_byte_count;

    // Long-divide the base-256 number by 58, collecting remainders in reverse order.
    char reversed_output[96];
    int reversed_length = 0;
    while (current_byte_count > 0) {
        Uint32 remainder = 0;
        Byte quotient[64];
        int quotient_length = 0;
        for (int byte_index = 0; byte_index < current_byte_count; byte_index++) {
            Uint32 accumulator = remainder * 256u + current_number[byte_index];
            Byte quotient_digit = (Byte)(accumulator / 58u);
            remainder = accumulator % 58u;
            if (quotient_length != 0 || quotient_digit != 0) {
                quotient[quotient_length++] = quotient_digit;
            }
        }
        assert(reversed_length < 96 && remainder < 58);
        reversed_output[reversed_length++] = base58_alphabet[remainder];
        for (int byte_index = 0; byte_index < quotient_length; byte_index++) {
            current_number[byte_index] = quotient[byte_index];
        }
        current_byte_count = quotient_length;
    }

    // Base58 preserves each leading zero byte as the alphabet's zero character, '1'.
    for (int byte_index = 0; byte_index < input_byte_count && input[byte_index] == 0;
         byte_index++) {
        assert(reversed_length < 96);
        reversed_output[reversed_length++] = '1';
    }

    // Reverse the collected digits into the caller's output buffer.
    int output_length = 0;
    while (reversed_length > 0 && output_length < output_capacity) {
        output[output_length++] = reversed_output[--reversed_length];
    }
    return output_length;
}

// Encode base64url without '=' padding, matching urlsafe_b64encode(...).rstrip("=").
__device__ static int base64url_encode(const Byte *input, int input_byte_count, Byte *output) {
    assert(input_byte_count >= 0 && input_byte_count <= 0x3fffffff);
    assert(input_byte_count == 0 || (input != 0 && output != 0));
    const char *base64url_alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    int output_length = 0;
    int input_offset = 0;

    // Every complete three-byte group expands into four six-bit characters.
    for (; input_offset + 3 <= input_byte_count; input_offset += 3) {
        Uint32 group = ((Uint32)input[input_offset] << 16) |
                       ((Uint32)input[input_offset + 1] << 8) | input[input_offset + 2];
        output[output_length++] = (Byte)base64url_alphabet[(group >> 18) & 63];
        output[output_length++] = (Byte)base64url_alphabet[(group >> 12) & 63];
        output[output_length++] = (Byte)base64url_alphabet[(group >> 6) & 63];
        output[output_length++] = (Byte)base64url_alphabet[group & 63];
    }

    // Encode one or two trailing bytes without appending padding characters.
    int remaining_bytes = input_byte_count - input_offset;
    if (remaining_bytes == 1) {
        Uint32 group = (Uint32)input[input_offset] << 16;
        output[output_length++] = (Byte)base64url_alphabet[(group >> 18) & 63];
        output[output_length++] = (Byte)base64url_alphabet[(group >> 12) & 63];
    } else if (remaining_bytes == 2) {
        Uint32 group = ((Uint32)input[input_offset] << 16) | ((Uint32)input[input_offset + 1] << 8);
        output[output_length++] = (Byte)base64url_alphabet[(group >> 18) & 63];
        output[output_length++] = (Byte)base64url_alphabet[(group >> 12) & 63];
        output[output_length++] = (Byte)base64url_alphabet[(group >> 6) & 63];
    }

    return output_length;
}

// Receipt-only, global-thread-zero helper. In particular, do not reuse this
// shared state from the parallel span-hashing path or the four signing lanes.
// A local 54-level streaming state would otherwise inflate every thread's
// stack backing even though there is only one receipt writer per block.
__device__ static void receipt_blake3_digest(const Byte *input, int input_byte_count,
                                             Byte output_digest[32]) {
    assert(threadIdx.x == 0 && blockIdx.x == 0); // Shared scratch has exactly one owner.
    assert(input_byte_count >= 0);
    __shared__ Blake3StreamingState state;
    blake3_stream_initialize(&state);
    blake3_stream_update(&state, input, (Uint64)input_byte_count);
    blake3_stream_finalize(&state, output_digest);
}

// Format a multibase base32-lower CID with no padding and a leading 'b'. The result goes into a
// plain buffer, rather than a BufferWriter, because callers embed the same CID multiple times.
__device__ static int format_cid(const Byte *binary_prefix, int prefix_byte_count,
                                 const Byte digest[32], Byte *output) {
    assert(prefix_byte_count >= 0 && prefix_byte_count <= 6);
    assert(binary_prefix != 0 && digest != 0 && output != 0);
    const char *base32_alphabet = "abcdefghijklmnopqrstuvwxyz234567";

    // Concatenate the CID version/codec/multihash prefix and the 32-byte digest.
    Byte binary_cid[38];
    for (int byte_index = 0; byte_index < prefix_byte_count; byte_index++) {
        binary_cid[byte_index] = binary_prefix[byte_index];
    }
    for (int byte_index = 0; byte_index < 32; byte_index++) {
        binary_cid[prefix_byte_count + byte_index] = digest[byte_index];
    }

    // Prefix the lower-case base32 multibase discriminator.
    int binary_byte_count = prefix_byte_count + 32;
    int output_length = 0;
    output[output_length++] = 'b';

    // Stream bits from the binary CID into five-bit base32 digits.
    Uint32 bit_accumulator = 0;
    int available_bits = 0;
    for (int byte_index = 0; byte_index < binary_byte_count; byte_index++) {
        bit_accumulator = (bit_accumulator << 8) | binary_cid[byte_index];
        available_bits += 8;
        while (available_bits >= 5) {
            output[output_length++] =
                    (Byte)base32_alphabet[(bit_accumulator >> (available_bits - 5)) & 31];
            available_bits -= 5;
        }
    }

    // Pad a final partial digit with zero bits; no '=' characters are emitted.
    if (available_bits > 0) {
        output[output_length++] =
                (Byte)base32_alphabet[(bit_accumulator << (5 - available_bits)) & 31];
    }
    assert(output_length == 1 + (binary_byte_count * 8 + 4) / 5);
    return output_length;
}

// EQTY raw CID (multicodec 0x55) for raw bytes -- data/document content.
__device__ static int format_raw_cid(const Byte digest[32], Byte *output) {
    const Byte raw_cid_prefix[4] = {0x01, 0x55, 0x1e, 0x20};
    return format_cid(raw_cid_prefix, 4, digest, output);
} // 59 chars

// EQTY RDFC-1 CID (multicodec 0xb403) for a canonicalized statement -- the
// @id of a CredentialRegistration node, and the credential-id preimage,
// matching what the EQTY SDK computes for the same canonical N-Quads.
__device__ static int format_rdfc_cid(const Byte digest[32], Byte *output) {
    const Byte rdfc_cid_prefix[6] = {0x01, 0x83, 0xE8, 0x02, 0x1E, 0x20};
    return format_cid(rdfc_cid_prefix, 6, digest, output);
} // 62 chars

// Format 16 UUID bytes with the conventional 8-4-4-4-12 lower-case hexadecimal grouping.
__device__ static int format_uuid(const Byte uuid_bytes[16], Byte *output) {
    const char *hex_alphabet = "0123456789abcdef";
    const int segment_byte_counts[5] = {4, 2, 2, 2, 6};
    int output_length = 0;
    int input_offset = 0;
    for (int segment_index = 0; segment_index < 5; segment_index++) {
        if (segment_index > 0) {
            output[output_length++] = '-';
        }
        for (int byte_index = 0; byte_index < segment_byte_counts[segment_index]; byte_index++) {
            Byte value = uuid_bytes[input_offset + byte_index];
            output[output_length++] = (Byte)hex_alphabet[value >> 4];
            output[output_length++] = (Byte)hex_alphabet[value & 15];
        }
        input_offset += segment_byte_counts[segment_index];
    }
    return output_length;
} // 36 chars

// Validate the fixed 20-byte lexical shape YYYY-MM-DDTHH:MM:SSZ.
__device__ static int is_valid_timestamp(const Byte *timestamp) {
    const char *shape = "dddd-dd-ddTdd:dd:ddZ";
    for (int byte_index = 0; byte_index < 20; byte_index++) {
        char expected = shape[byte_index];
        Byte actual = timestamp[byte_index];
        if (expected == 'd') {
            if (actual < '0' || actual > '9') {
                return 0;
            }
        } else if (actual != (Byte)expected) {
            return 0;
        }
    }
    return 1;
}

// Permit only a compact path-like ASCII model name, preventing JSON escaping and injection.
__device__ static int is_valid_model_name(const Byte *model_name, int model_name_length) {
    if (model_name_length < 1 || model_name_length > 64) {
        return 0;
    }
    for (int byte_index = 0; byte_index < model_name_length; byte_index++) {
        Byte character = model_name[byte_index];
        int is_allowed = (character >= 'a' && character <= 'z') ||
                         (character >= 'A' && character <= 'Z') ||
                         (character >= '0' && character <= '9') || character == '/' ||
                         character == '.' || character == '_' || character == '-';
        if (!is_allowed) {
            return 0;
        }
    }
    return 1;
}

// Format the session public key as a did:key identifier. The two multicodec bytes identify a
// compressed P-256 public key; the complete binary value is then base58btc encoded after "z".
__device__ static int build_issuer_did(char output_did[64]) {
    // Copy the textual did:key multibase prefix.
    const char *did_prefix = "did:key:z";
    int output_length = 0;
    while (*did_prefix != '\0') {
        output_did[output_length++] = *did_prefix++;
    }

    // Prefix the 33-byte SEC 1 compressed key with the varint P-256 public-key multicodec.
    Byte multicodec_public_key[35];
    multicodec_public_key[0] = 0x80;
    multicodec_public_key[1] = 0x24;
    for (int byte_index = 0; byte_index < 33; byte_index++) {
        multicodec_public_key[2 + byte_index] = g_compressed_public_key[byte_index];
    }

    // Append the base58btc payload without a null terminator.
    output_length += base58_encode(multicodec_public_key, 35, output_did + output_length,
                                   64 - output_length);
    return output_length;
}

// context CID for the statement vocabulary used below (StateAttestation /
// CredentialRegistration terms) -- display-only: @context never enters the
// RDFC canonicalization (JSON-LD framing is resolved before canonicalizing),
// so this constant does not affect any @id computed here.
#define MANIFEST_VERSION "3"
#define STATE_TYPE "gpuModelTensorsStateV1"
#define CREDENTIAL_TYPE "StateAttestation"
#define CTX_CID "bafkr4icploa577ziqnb57jlpoj7l2hi5kgt3knxpdtunlttjd3q33zeqpy"

struct CredentialNQuad {
    const Byte *bytes;
    // A 32-bit length left four uninitialized padding bytes after it. NVRTC
    // copies these shared-memory descriptors with 64-bit loads during sorting,
    // so initcheck correctly reported those padding reads. A full-width length
    // makes every copied byte initialized without clearing large scratch areas.
    Uint64 length;
};
static_assert(sizeof(CredentialNQuad) == sizeof(CredentialNQuad::bytes) +
                                       sizeof(CredentialNQuad::length),
              "credential descriptors must not contain uninitialized padding");

__device__ static int compare_credential_bytes(const Byte *left, int left_length,
                                              const Byte *right, int right_length) {
    int common_length = left_length < right_length ? left_length : right_length;
    for (int index = 0; index < common_length; index++) {
        if (left[index] != right[index]) {
            return (int)left[index] - (int)right[index];
        }
    }
    return left_length - right_length;
}

// Confirm a newline-terminated N-Quads buffer is already in canonical order.
//
// Used where the emission template is known to be sorted for every value this
// kernel can produce -- the JWS signing document -- so that a regression in
// the template fails closed instead of silently changing every signature.
__device__ static int nquads_are_sorted(const Byte *nquads, int length) {
    assert(nquads != 0 && length >= 0);
    int previous_start = -1;
    int previous_length = 0;
    int start = 0;
    for (int index = 0; index < length; index++) {
        if (nquads[index] != '\n') continue;
        int quad_length = index + 1 - start;
        if (previous_start >= 0 &&
            compare_credential_bytes(nquads + previous_start, previous_length, nquads + start,
                                     quad_length) > 0) {
            return 0;
        }
        previous_start = start;
        previous_length = quad_length;
        start = index + 1;
    }
    return start == length;
}

__device__ static void sort_credential_quads(CredentialNQuad *quads, int count) {
    assert(quads != 0 && count >= 0 && count <= 20);
    // At most twenty ASCII quads: byte order is Unicode code point order.
    for (int index = 1; index < count; index++) {
        CredentialNQuad next = quads[index];
        int slot = index;
        while (slot > 0 && compare_credential_bytes(
                quads[slot - 1].bytes, quads[slot - 1].length,
                next.bytes, next.length) > 0) {
            quads[slot] = quads[slot - 1];
            slot--;
        }
        quads[slot] = next;
    }
}

__device__ static int credential_blank_node(const Byte *bytes, int remaining) {
    // These provisional tokens occur only in the trusted template. Validated
    // timestamps, did:key, UUID/CID URIs, and base64url JWS cannot contain them.
    if (remaining >= 8 && bytes[0] == '_' && bytes[1] == ':' &&
        bytes[2] == 't' && bytes[3] == 'e' && bytes[4] == 'm' && bytes[5] == 'p' &&
        bytes[6] >= '0' && bytes[6] <= '3' && bytes[7] == ' ') {
        return bytes[6] - '0';
    }
    return -1;
}

__device__ __noinline__ static int
hash_credential_registration_nquads(Byte *nquads, int length, Byte digest[32]) {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    assert(nquads != 0 && digest != 0 && length >= 0);
    // RDFC-1.0 sections 4.4 and 4.6, specialized to this closed schema:
    // https://www.w3.org/TR/rdf-canon/#hash-1d-quads
    // Nineteen quads for a genesis credential, twenty once the nested state
    // carries a previousStateCredential link.
    // The proof subject, proof graph, wrapper, and state node have distinct
    // first-degree inputs. Hash ties can only be SHA-256 collisions here;
    // reject them rather than assigning a role-based label. Arbitrary RDF
    // would need N-degree canonicalization too and must not be routed through
    // this helper.
    __shared__ CredentialNQuad quads[20];
    int count = 0;
    int start = 0;
    for (int index = 0; index < length; index++) {
        if (nquads[index] == '\n') {
            if (count == 20) return 0;
            quads[count++] = {nquads + start, index + 1 - start};
            start = index + 1;
        }
    }
    if ((count != 19 && count != 20) || start != length) return 0;

    __shared__ Byte first_degree_hashes[4][32];
    // Only global thread zero calls the receipt builders, before/after the
    // four independent signing lanes. Keep their large PUBLIC serialization
    // scratch in block-shared memory: thread-local arrays make CUDA reserve
    // gigabytes of context-wide local-memory backing on many-SM GPUs, even
    // for this one-block kernel. These helpers must not be called in parallel
    // by the signing lanes; their private ECDSA state remains lane-local.
    __shared__ Byte first_degree_bytes[2560];
    for (int node = 0; node < 4; node++) {
        BufferWriter writer = {first_degree_bytes, 0, 2560, 0};
        __shared__ CredentialNQuad incident[20];
        int incident_count = 0;
        for (int quad_index = 0; quad_index < count; quad_index++) {
            CredentialNQuad quad = quads[quad_index];
            int includes_node = 0;
            for (int index = 0; index < quad.length; index++) {
                if (credential_blank_node(quad.bytes + index, quad.length - index) == node) {
                    includes_node = 1;
                    break;
                }
            }
            if (!includes_node) continue;
            int quad_start = writer.length;
            for (int index = 0; index < quad.length;) {
                int related = credential_blank_node(quad.bytes + index, quad.length - index);
                if (related >= 0) {
                    // Include graph names as well as subjects/objects in the
                    // replacement. The JWS value changes these incident hashes.
                    writer_append_c_string(&writer, related == node ? "_:a" : "_:z");
                    index += 7;
                } else {
                    writer_append_bytes(&writer, quad.bytes + index, 1);
                    index++;
                }
            }
            if (writer.overflowed) return 0;
            incident[incident_count++] = {
                first_degree_bytes + quad_start, writer.length - quad_start};
        }
        if (incident_count == 0) return 0;
        sort_credential_quads(incident, incident_count);
        Sha256State hasher;
        sha256_initialize(&hasher);
        for (int index = 0; index < incident_count; index++) {
            sha256_update(&hasher, incident[index].bytes, incident[index].length);
        }
        sha256_finalize(&hasher, first_degree_hashes[node]);
    }

    int labels[4] = {0, 0, 0, 0};
    for (int node = 0; node < 4; node++) {
        for (int other = 0; other < 4; other++) {
            if (node == other) continue;
            int order = compare_credential_bytes(first_degree_hashes[node], 32,
                                                 first_degree_hashes[other], 32);
            if (order == 0) return 0;
            if (order > 0) labels[node]++;
        }
    }
    // Equal-length replacements leave the original quad slices valid. Sorting
    // must follow relabeling: proof/wrapper subject order is value-dependent.
    for (int index = 0; index < length; index++) {
        int node = credential_blank_node(nquads + index, length - index);
        if (node >= 0) {
            nquads[index + 2] = 'c';
            nquads[index + 3] = '1';
            nquads[index + 4] = '4';
            nquads[index + 5] = 'n';
            nquads[index + 6] = (Byte)('0' + labels[node]);
            index += 6;
        }
    }
    sort_credential_quads(quads, count);
    __shared__ Blake3StreamingState hasher;
    blake3_stream_initialize(&hasher);
    for (int index = 0; index < count; index++) {
        blake3_stream_update(&hasher, quads[index].bytes, quads[index].length);
    }
    blake3_stream_finalize(&hasher, digest);
    return 1;
}

// Canonicalize and hash the credential-id preimage's N-Quads.
//
// RDFC-1.0 sections 4.4 and 4.6 again, specialized to a second closed schema:
// the credential node (_:temp0, standing in for the urn:uuid it is deriving)
// and the nested `state` node (_:temp1). A genesis preimage has 9 quads; one
// carrying previousStateCredential has 10.
//
// This is deliberately a sibling of hash_credential_registration_nquads rather
// than a generalization of it. That function's output is pinned byte-for-byte
// to a real EQTY-SDK-issued credential, and widening its bounds to serve two
// schemas would put that fixture at risk for no gain -- the algorithm is
// short, and the two callers have different quad counts and node counts.
//
// The two nodes carry disjoint predicates, so their first-degree hashes can
// collide only if SHA-256 does. Reject that case rather than fall back to a
// role-based label. Arbitrary RDF would need N-degree canonicalization and
// must not be routed through here.
__device__ __noinline__ static int
hash_credential_preimage_nquads(Byte *nquads, int length, Byte digest[32]) {
    assert(threadIdx.x == 0 && blockIdx.x == 0);
    assert(nquads != 0 && digest != 0 && length >= 0);
    __shared__ CredentialNQuad quads[10];
    int count = 0;
    int start = 0;
    for (int index = 0; index < length; index++) {
        if (nquads[index] == '\n') {
            if (count == 10) return 0;
            quads[count++] = {nquads + start, index + 1 - start};
            start = index + 1;
        }
    }
    if ((count != 9 && count != 10) || start != length) return 0;

    __shared__ Byte first_degree_hashes[2][32];
    __shared__ Byte first_degree_bytes[2048];
    for (int node = 0; node < 2; node++) {
        BufferWriter writer = {first_degree_bytes, 0, 2048, 0};
        __shared__ CredentialNQuad incident[10];
        int incident_count = 0;
        for (int quad_index = 0; quad_index < count; quad_index++) {
            CredentialNQuad quad = quads[quad_index];
            int includes_node = 0;
            for (int index = 0; index < quad.length; index++) {
                if (credential_blank_node(quad.bytes + index, quad.length - index) == node) {
                    includes_node = 1;
                    break;
                }
            }
            if (!includes_node) continue;
            int quad_start = writer.length;
            for (int index = 0; index < quad.length;) {
                int related = credential_blank_node(quad.bytes + index, quad.length - index);
                if (related >= 0) {
                    writer_append_c_string(&writer, related == node ? "_:a" : "_:z");
                    index += 7;
                } else {
                    writer_append_bytes(&writer, quad.bytes + index, 1);
                    index++;
                }
            }
            if (writer.overflowed) return 0;
            incident[incident_count++] = {
                first_degree_bytes + quad_start, writer.length - quad_start};
        }
        if (incident_count == 0) return 0;
        sort_credential_quads(incident, incident_count);
        Sha256State hasher;
        sha256_initialize(&hasher);
        for (int index = 0; index < incident_count; index++) {
            sha256_update(&hasher, incident[index].bytes, incident[index].length);
        }
        sha256_finalize(&hasher, first_degree_hashes[node]);
    }

    // Ascending first-degree hash order assigns c14n0 then c14n1. Which node
    // wins is value-dependent -- adding previousStateCredential can flip it --
    // so neither the labels nor the final quad order may be templated.
    int labels[2] = {0, 0};
    int order = compare_credential_bytes(first_degree_hashes[0], 32, first_degree_hashes[1], 32);
    if (order == 0) return 0;
    if (order > 0) {
        labels[0] = 1;
    } else {
        labels[1] = 1;
    }

    // Equal-length replacement keeps the existing quad slices valid. Sorting
    // must follow relabeling, because subject order depends on the new labels.
    for (int index = 0; index < length; index++) {
        int node = credential_blank_node(nquads + index, length - index);
        if (node >= 0) {
            nquads[index + 2] = 'c';
            nquads[index + 3] = '1';
            nquads[index + 4] = '4';
            nquads[index + 5] = 'n';
            nquads[index + 6] = (Byte)('0' + labels[node]);
            index += 6;
        }
    }
    sort_credential_quads(quads, count);
    __shared__ Blake3StreamingState hasher;
    blake3_stream_initialize(&hasher);
    for (int index = 0; index < count; index++) {
        blake3_stream_update(&hasher, quads[index].bytes, quads[index].length);
    }
    blake3_stream_finalize(&hasher, digest);
    return 1;
}

// Builds the N-Quads the credential's own id is derived from, and returns
// their RDFC digest through `output_preimage_digest`.
//
// This is everything the StateAttestation asserts, minus its own name: the
// credential node stands in as `_:temp0` because the value being derived is
// exactly the urn:uuid that would otherwise sit there. `_:temp1` is the nested
// `state` -- what schema this payload is, which resident copy it came from,
// and what its bytes hash to. modelRoot is GPU-measured; instanceID is
// folded by the host over the IPC handles it resolved, since handles never
// reach the device.
//
// `previous_credential_uuid` is the 16-byte UUID of this instance's last
// credential, or null for the first signature of that copy in this session.
// RDF has no null, so a genesis preimage simply omits the triple -- which is
// also what JSON-LD expansion does with a null-valued key, making the emitted
// "previousStateCredential":null and the absent triple the same document.
//
// Deriving the id from this preimage rather than drawing it at random is what
// lets a verifier recompute it, and what makes the kernel's two receipt passes
// agree: the device has no entropy source to reuse across them.
//
// Takes the subject values as bare digests rather than precomputed
// "urn:cid:..." strings. An earlier version received that string from a
// caller-held buffer that had to stay valid across other large nested calls
// (credential building, ECDSA signing), and on this GPU/toolchain combination
// that cross-call survival is NOT reliable -- the optimizer was observed to
// alias/clobber such buffers, corrupting the statement with unrelated
// leftover bytes. Recomputing the short-lived "urn:cid:..." form fresh, in a
// purely local buffer right where it is consumed, avoids relying on that
// guarantee; only small fixed-size digests (proven stable in the same
// testing) cross function boundaries. __noinline__ keeps this function's own
// locals out of the caller's frame, which independently helps: the corruption
// was traced to this project's specific nvrtc/ptxas combination
// over-aggressively reusing stack slots when everything is inlined into one
// giant kernel body.
__device__ __noinline__ static void
build_credential_id_preimage(const char *issuer_did, int issuer_did_length,
                             const Byte timestamp[20], const Byte model_root[32],
                             const Byte instance_root[32],
                             const Byte *previous_credential_uuid,
                             Byte output_preimage_digest[32], BufferWriter *output_writer) {
    assert(issuer_did != 0 && timestamp != 0 && model_root != 0 && instance_root != 0);
    assert(output_preimage_digest != 0 && output_writer != 0);

    // Content and residency are raw CIDs; a chain link names another credential.
    Byte model_cid[59];
    int model_cid_length = format_raw_cid(model_root, model_cid);
    Byte instance_cid[59];
    int instance_cid_length = format_raw_cid(instance_root, instance_cid);
    Byte previous_uuid_text[36];
    int previous_uuid_text_length =
            previous_credential_uuid ? format_uuid(previous_credential_uuid, previous_uuid_text)
                                     : 0;

    // Receipt builders run only in thread zero; share this public serialization
    // scratch rather than reserving it in every thread's stack.
    __shared__ Byte canonical_nquads[1536];
    BufferWriter canonical_writer;
    canonical_writer.destination = canonical_nquads;
    canonical_writer.length = 0;
    canonical_writer.capacity = 1536;
    canonical_writer.overflowed = 0;
    writer_append_c_string(&canonical_writer, "<");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer,
                           "> <https://eqtylab.io/terms/state> _:temp1 .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                           "<https://eqtylab.io/terms/" CREDENTIAL_TYPE "> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                           "<https://www.w3.org/2018/credentials#VerifiableCredential> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp0 <https://www.w3.org/2018/credentials#credentialSubject> <");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp0 <https://www.w3.org/2018/credentials#issuer> <");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp0 <https://www.w3.org/2018/credentials#validFrom> \"");
    writer_append_bytes(&canonical_writer, timestamp, 20);
    writer_append_c_string(&canonical_writer,
                           "\"^^<http://www.w3.org/2001/XMLSchema#dateTime> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp1 <https://eqtylab.io/terms/instanceID> <urn:cid:");
    writer_append_bytes(&canonical_writer, instance_cid, instance_cid_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp1 <https://eqtylab.io/terms/modelRoot> <urn:cid:");
    writer_append_bytes(&canonical_writer, model_cid, model_cid_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    if (previous_credential_uuid) {
        writer_append_c_string(
                &canonical_writer,
                "_:temp1 <https://eqtylab.io/terms/previousStateCredential> <urn:uuid:");
        writer_append_bytes(&canonical_writer, previous_uuid_text, previous_uuid_text_length);
        writer_append_c_string(&canonical_writer, "> .\n");
    }
    writer_append_c_string(&canonical_writer,
                           "_:temp1 <https://eqtylab.io/terms/stateType> \"" STATE_TYPE "\" .\n");

    // Propagate bounded-buffer failures before hashing incomplete bytes.
    if (canonical_writer.overflowed) {
        output_writer->overflowed = 1;
        return;
    }

    // Canonicalize the two-node dataset; the caller turns this into the UUID.
    if (!hash_credential_preimage_nquads(canonical_nquads, canonical_writer.length,
                                         output_preimage_digest)) {
        output_writer->overflowed = 1;
    }
}

// Builds the notary-signed StateAttestation over this measurement, wraps it in
// `{"@type":"CredentialRegistration","credential":<vc>,...}`, writes it to
// `out`, and returns the wrapper statement's own RDFC digest through
// `output_statement_digest` and the credential's UUID through
// `output_credential_uuid`.
//
// The credential's subject is the GPU itself -- credentialSubject and issuer
// are the same DID -- and the measured state is nested inside that subject
// rather than referenced as a separate statement. So the state travels inside
// the signed document, not merely beside it: the JWS covers instanceID,
// modelRoot, and the chain link directly.
//
// The VC's proof is a spec-correct EcdsaSecp256r1Signature2019: JWS over
// b64(hdr) '.' sha256_digest(URDNA proof-options) || sha256_digest(URDNA
// document). Those signing inputs have at most one blank node, whose canonical
// label is therefore forced. The full wrapper has four and requires
// content-dependent canonical labeling and quad sorting before its CID is
// hashed.
__device__ __noinline__ static void
build_credential_registration(const Byte model_root[32], const char *issuer_did,
                              int issuer_did_length, const Byte timestamp[20],
                              const Byte instance_root[32],
                              const Byte *previous_credential_uuid,
                              const Byte preimage_digest[32], const Byte raw_signature[64],
                              Byte output_signing_hash[32], Byte output_credential_uuid[16],
                              Byte output_statement_digest[32], BufferWriter *output_writer) {
    assert(issuer_did != 0 && timestamp != 0 && model_root != 0 && instance_root != 0);
    assert(preimage_digest != 0 && output_writer != 0);

    // The state's own terms, recomputed here rather than carried in from the
    // preimage builder: only small fixed-size digests cross call boundaries.
    Byte model_cid[59];
    int model_cid_length = format_raw_cid(model_root, model_cid);
    Byte instance_cid[59];
    int instance_cid_length = format_raw_cid(instance_root, instance_cid);
    Byte previous_uuid_text[36];
    int previous_uuid_text_length =
            previous_credential_uuid ? format_uuid(previous_credential_uuid, previous_uuid_text)
                                     : 0;

    // The credential id commits to everything the credential asserts, via the
    // canonicalized preimage the caller just hashed.
    Byte preimage_cid[62];
    int preimage_cid_length = format_rdfc_cid(preimage_digest, preimage_cid);
    Byte uuid_seed[32 + 8 + 62];
    int uuid_seed_length = 0;
    for (int byte_index = 0; byte_index < 32; byte_index++) {
        uuid_seed[uuid_seed_length++] = model_root[byte_index];
    }
    const char *preimage_urn_prefix = "urn:cid:";
    for (int byte_index = 0; byte_index < 8; byte_index++) {
        uuid_seed[uuid_seed_length++] = (Byte)preimage_urn_prefix[byte_index];
    }
    for (int byte_index = 0; byte_index < preimage_cid_length; byte_index++) {
        uuid_seed[uuid_seed_length++] = preimage_cid[byte_index];
    }
    Byte uuid_hash[32];
    sha256_digest(uuid_seed, (Uint64)uuid_seed_length, uuid_hash);

    // Set RFC 4122 version-4 and variant bits while retaining deterministic remaining bits.
    uuid_hash[6] = (Byte)(0x40 | (uuid_hash[6] & 0x0f));
    uuid_hash[8] = (Byte)(0x80 | (uuid_hash[8] & 0x3f));
    Byte uuid_text[36];
    int uuid_text_length = format_uuid(uuid_hash, uuid_text);
    if (output_credential_uuid) {
        for (int byte_index = 0; byte_index < 16; byte_index++) {
            output_credential_uuid[byte_index] = uuid_hash[byte_index];
        }
    }

    // Thread zero reuses this block-shared public buffer for each canonical
    // representation. Do not turn it back into per-thread stack storage;
    // context-wide local-memory reservation would defeat near-full-VRAM use.
    __shared__ Byte canonical_nquads[4096];
    BufferWriter canonical_writer;
    canonical_writer.destination = canonical_nquads;
    canonical_writer.length = 0;
    canonical_writer.capacity = 4096;
    canonical_writer.overflowed = 0;
    // SIGNING STEP 1: canonicalize the credential document without a proof. The proof does not
    // exist yet at signing time, so it is NOT referenced here (no "security#proof" line, no named
    // graph) -- this is the fixed ad-hoc proof-options canonicalization this
    // project's crypto suite uses as JWS signing input, distinct from the
    // full JSON-LD/RDFC expansion used below for the wrapper's own @id.
    //
    // These are the preimage's quads with _:temp0 resolved to the derived
    // urn:uuid and the state node taking the only blank label left, _:c14n0.
    // They are emitted in sorted order: a did:key subject precedes a urn:uuid
    // one, which precedes a blank node, and each subject's predicates are
    // already ascending. The order is asserted below rather than assumed.
    writer_append_c_string(&canonical_writer, "<");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "> <https://eqtylab.io/terms/state> _:c14n0 .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer,
                           "> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                           "<https://eqtylab.io/terms/" CREDENTIAL_TYPE "> .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer,
                           "> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                           "<https://www.w3.org/2018/credentials#VerifiableCredential> .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer,
                           "> <https://www.w3.org/2018/credentials#credentialSubject> <");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer, "> <https://www.w3.org/2018/credentials#issuer> <");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer,
                           "> <https://www.w3.org/2018/credentials#validFrom> \"");
    writer_append_bytes(&canonical_writer, timestamp, 20);
    writer_append_c_string(&canonical_writer,
                           "\"^^<http://www.w3.org/2001/XMLSchema#dateTime> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:c14n0 <https://eqtylab.io/terms/instanceID> <urn:cid:");
    writer_append_bytes(&canonical_writer, instance_cid, instance_cid_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:c14n0 <https://eqtylab.io/terms/modelRoot> <urn:cid:");
    writer_append_bytes(&canonical_writer, model_cid, model_cid_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    if (previous_credential_uuid) {
        writer_append_c_string(
                &canonical_writer,
                "_:c14n0 <https://eqtylab.io/terms/previousStateCredential> <urn:uuid:");
        writer_append_bytes(&canonical_writer, previous_uuid_text, previous_uuid_text_length);
        writer_append_c_string(&canonical_writer, "> .\n");
    }
    writer_append_c_string(&canonical_writer,
                           "_:c14n0 <https://eqtylab.io/terms/stateType> \"" STATE_TYPE "\" .\n");
    if (canonical_writer.overflowed) {
        output_writer->overflowed = 1;
        return;
    }
    // Canonical N-Quads are sorted. The template above is already in order for
    // every value this kernel can produce, but a silent regression here would
    // change every signature, so verify rather than trust the template.
    if (!nquads_are_sorted(canonical_nquads, canonical_writer.length)) {
        output_writer->overflowed = 1;
        return;
    }
    Byte document_canonical_hash[32];
    sha256_digest(canonical_nquads, (Uint64)canonical_writer.length, document_canonical_hash);

    // SIGNING STEP 2: canonicalize proof options as plain triples. The JWS value being computed is
    // necessarily absent from these options.
    canonical_writer.length = 0;
    writer_append_c_string(&canonical_writer, "_:c14n0 <http://purl.org/dc/terms/created> \"");
    writer_append_bytes(&canonical_writer, timestamp, 20);
    writer_append_c_string(&canonical_writer,
                           "\"^^<http://www.w3.org/2001/XMLSchema#dateTime> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:c14n0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                           "<https://w3id.org/security#EcdsaSecp256r1Signature2019> .\n");
    writer_append_c_string(&canonical_writer, "_:c14n0 <https://w3id.org/security#proofPurpose> "
                                              "<https://w3id.org/security#assertionMethod> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:c14n0 <https://w3id.org/security#verificationMethod> <");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "#");
    writer_append_bytes(&canonical_writer, (const Byte *)(issuer_did + 8), issuer_did_length - 8);
    writer_append_c_string(&canonical_writer, "> .\n");
    if (canonical_writer.overflowed) {
        output_writer->overflowed = 1;
        return;
    }
    Byte proof_options_hash[32];
    sha256_digest(canonical_nquads, (Uint64)canonical_writer.length, proof_options_hash);

    // SIGNING STEP 3: hash protected-header + '.' + proof-options hash + document hash.
    Byte jws_signing_input[136];
    BufferWriter signing_input_writer;
    signing_input_writer.destination = jws_signing_input;
    signing_input_writer.length = 0;
    signing_input_writer.capacity = 136;
    signing_input_writer.overflowed = 0;
    writer_append_c_string(&signing_input_writer,
                           "eyJhbGciOiJFUzI1NiIsImNyaXQiOlsiYjY0Il0sImI2NCI6ZmFsc2V9.");
    writer_append_bytes(&signing_input_writer, proof_options_hash, 32);
    writer_append_bytes(&signing_input_writer, document_canonical_hash, 32);
    if (signing_input_writer.overflowed) {
        output_writer->overflowed = 1;
        return;
    }
    Byte signing_hash[32];
    sha256_digest(jws_signing_input, (Uint64)signing_input_writer.length, signing_hash);

    // The first pass requests this hash and stops before receipt construction.
    if (output_signing_hash) {
        for (int byte_index = 0; byte_index < 32; byte_index++) {
            output_signing_hash[byte_index] = signing_hash[byte_index];
        }
    }
    if (!raw_signature) {
        return;
    }

    // The second pass receives the corresponding raw r || s signature and embeds it as base64url.
    Byte base64url_signature[90];
    int base64url_signature_length =
            base64url_encode(raw_signature, 64, base64url_signature); // 86 chars

    // Thread zero constructs the public credential, sequentially reusing the
    // shared buffer across the registrations and both receipt passes.
    __shared__ Byte credential_json[1536];
    BufferWriter credential_writer;
    credential_writer.destination = credential_json;
    credential_writer.length = 0;
    credential_writer.capacity = 1536;
    credential_writer.overflowed = 0;
    writer_append_c_string(
            &credential_writer,
            "{\"@context\":[\"https://www.w3.org/ns/credentials/v2\",\"https://w3id.org/security/"
            "v2\"],\"id\":\"urn:uuid:");
    writer_append_bytes(&credential_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&credential_writer,
                           "\",\"type\":[\"VerifiableCredential\",\"" CREDENTIAL_TYPE "\"],"
                           "\"credentialSubject\":{\"id\":\"");
    writer_append_bytes(&credential_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&credential_writer,
                           "\",\"state\":{\"stateType\":\"" STATE_TYPE
                           "\",\"instanceID\":\"urn:cid:");
    writer_append_bytes(&credential_writer, instance_cid, instance_cid_length);
    writer_append_c_string(&credential_writer, "\",\"modelRoot\":\"urn:cid:");
    writer_append_bytes(&credential_writer, model_cid, model_cid_length);
    writer_append_c_string(&credential_writer, "\",\"previousStateCredential\":");
    if (previous_credential_uuid) {
        writer_append_c_string(&credential_writer, "\"urn:uuid:");
        writer_append_bytes(&credential_writer, previous_uuid_text, previous_uuid_text_length);
        writer_append_c_string(&credential_writer, "\"");
    } else {
        writer_append_c_string(&credential_writer, "null");
    }
    writer_append_c_string(&credential_writer, "}},\"issuer\":\"");
    writer_append_bytes(&credential_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&credential_writer,
                           "\",\"proof\":{\"type\":\"EcdsaSecp256r1Signature2019\","
                           "\"proofPurpose\":\"assertionMethod\",\"verificationMethod\":\"");
    writer_append_bytes(&credential_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&credential_writer, "#");
    writer_append_bytes(&credential_writer, (const Byte *)(issuer_did + 8), issuer_did_length - 8);
    writer_append_c_string(&credential_writer, "\",\"created\":\"");
    writer_append_bytes(&credential_writer, timestamp, 20);
    writer_append_c_string(
            &credential_writer,
            "\",\"jws\":\"eyJhbGciOiJFUzI1NiIsImNyaXQiOlsiYjY0Il0sImI2NCI6ZmFsc2V9..");
    writer_append_bytes(&credential_writer, base64url_signature, base64url_signature_length);
    writer_append_c_string(&credential_writer, "\"},\"validFrom\":\"");
    writer_append_bytes(&credential_writer, timestamp, 20);
    writer_append_c_string(&credential_writer, "\"}");
    if (credential_writer.overflowed) {
        output_writer->overflowed = 1;
        return;
    }

    // Assemble the wrapper with provisional role labels. The proof subject,
    // proof graph, registration, and state node receive canonical labels only
    // after their actual quads (including the completed signature) have been
    // hashed below.
    canonical_writer.length = 0;
    writer_append_c_string(&canonical_writer, "<");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "> <https://eqtylab.io/terms/state> _:temp3 .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer,
                           "> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                           "<https://eqtylab.io/terms/" CREDENTIAL_TYPE "> .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer,
                           "> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                           "<https://www.w3.org/2018/credentials#VerifiableCredential> .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer, "> <https://w3id.org/security#proof> _:temp1 .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer,
                           "> <https://www.w3.org/2018/credentials#credentialSubject> <");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer, "> <https://www.w3.org/2018/credentials#issuer> <");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer, "<urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer,
                           "> <https://www.w3.org/2018/credentials#validFrom> \"");
    writer_append_bytes(&canonical_writer, timestamp, 20);
    writer_append_c_string(&canonical_writer,
                           "\"^^<http://www.w3.org/2001/XMLSchema#dateTime> .\n");
    writer_append_c_string(&canonical_writer, "_:temp0 <http://purl.org/dc/terms/created> \"");
    writer_append_bytes(&canonical_writer, timestamp, 20);
    writer_append_c_string(&canonical_writer,
                           "\"^^<http://www.w3.org/2001/XMLSchema#dateTime> _:temp1 .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                           "<https://w3id.org/security#EcdsaSecp256r1Signature2019> _:temp1 .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp0 <https://w3id.org/security#jws> "
                           "\"eyJhbGciOiJFUzI1NiIsImNyaXQiOlsiYjY0Il0sImI2NCI6ZmFsc2V9..");
    writer_append_bytes(&canonical_writer, base64url_signature, base64url_signature_length);
    writer_append_c_string(&canonical_writer, "\" _:temp1 .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp0 <https://w3id.org/security#proofPurpose> "
                           "<https://w3id.org/security#assertionMethod> _:temp1 .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp0 <https://w3id.org/security#verificationMethod> <");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "#");
    writer_append_bytes(&canonical_writer, (const Byte *)(issuer_did + 8), issuer_did_length - 8);
    writer_append_c_string(&canonical_writer, "> _:temp1 .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp2 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
                           "<https://eqtylab.io/terms/CredentialRegistration> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp2 <https://eqtylab.io/terms/credential> <urn:uuid:");
    writer_append_bytes(&canonical_writer, uuid_text, uuid_text_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer, "_:temp2 <https://eqtylab.io/terms/registeredBy> \"");
    writer_append_bytes(&canonical_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&canonical_writer, "\" .\n");
    writer_append_c_string(&canonical_writer, "_:temp2 <https://eqtylab.io/terms/timestamp> \"");
    writer_append_bytes(&canonical_writer, timestamp, 20);
    writer_append_c_string(&canonical_writer,
                           "\"^^<http://www.w3.org/2001/XMLSchema#dateTime> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp3 <https://eqtylab.io/terms/instanceID> <urn:cid:");
    writer_append_bytes(&canonical_writer, instance_cid, instance_cid_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    writer_append_c_string(&canonical_writer,
                           "_:temp3 <https://eqtylab.io/terms/modelRoot> <urn:cid:");
    writer_append_bytes(&canonical_writer, model_cid, model_cid_length);
    writer_append_c_string(&canonical_writer, "> .\n");
    if (previous_credential_uuid) {
        writer_append_c_string(
                &canonical_writer,
                "_:temp3 <https://eqtylab.io/terms/previousStateCredential> <urn:uuid:");
        writer_append_bytes(&canonical_writer, previous_uuid_text, previous_uuid_text_length);
        writer_append_c_string(&canonical_writer, "> .\n");
    }
    writer_append_c_string(&canonical_writer,
                           "_:temp3 <https://eqtylab.io/terms/stateType> \"" STATE_TYPE "\" .\n");
    if (canonical_writer.overflowed) {
        output_writer->overflowed = 1;
        return;
    }

    // Hash the complete wrapper canonicalization to obtain the wrapper statement's own RDFC CID.
    if (!hash_credential_registration_nquads(
            canonical_nquads, canonical_writer.length, output_statement_digest)) {
        output_writer->overflowed = 1;
        return;
    }

    // Append the completed wrapper JSON under that CID in the receipt's statement map.
    // It is the receipt's only statement, so no leading comma.
    Byte statement_cid[62];
    int statement_cid_length = format_rdfc_cid(output_statement_digest, statement_cid);
    writer_append_c_string(output_writer, "\"urn:cid:");
    writer_append_bytes(output_writer, statement_cid, statement_cid_length);
    writer_append_c_string(output_writer, "\":");
    writer_append_c_string(output_writer,
                           "{\"@context\":\"urn:cid:" CTX_CID "\",\"@id\":\"urn:cid:");
    writer_append_bytes(output_writer, statement_cid, statement_cid_length);
    writer_append_c_string(output_writer,
                           "\",\"@type\":\"CredentialRegistration\",\"credential\":");
    writer_append_bytes(output_writer, credential_json, credential_writer.length);
    writer_append_c_string(output_writer, ",\"registeredBy\":\"");
    writer_append_bytes(output_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(output_writer, "\",\"timestamp\":\"");
    writer_append_bytes(output_writer, timestamp, 20);
    writer_append_c_string(output_writer, "\"}");
}

// Assemble the measurement document and append "measurementDocument", "measurementSignature",
// "modelRoot", and the opening of the statement map to `output_writer`. The first pass uses the
// document's SHA-256 hash for signing; the second pass embeds the supplied signature.
// __noinline__ for the same cross-call-survival reason
// as the statement builders above -- keeps `document_bytes` and friends out of the
// caller's frame instead of all living in one giant kernel body.
__device__ __noinline__ static int
write_measurement_prefix(const Byte model_root[32], const Byte timestamp[20],
                         const Byte *model_name, int model_name_length, Uint32 tensor_count,
                         const Byte cubin_digest[32], const Byte kernel_digest[32],
                         Uint32 device_ordinal, const char *issuer_did, int issuer_did_length,
                         const Byte raw_signature[64], Byte output_document_signing_hash[32],
                         BufferWriter *output_writer) {
    // Format raw CIDs for the measured model, loaded CUBIN, and trusted kernel source.
    Byte model_cid[59];
    int model_cid_length = format_raw_cid(model_root, model_cid);
    Byte cubin_cid[59];
    int cubin_cid_length = format_raw_cid(cubin_digest, cubin_cid);
    Byte kernel_cid[59];
    int kernel_cid_length = format_raw_cid(kernel_digest, kernel_cid);

    // Serialize the exact JSON bytes covered by the measurement signature.
    // This receipt-only helper also runs exclusively in global thread zero.
    __shared__ Byte document_bytes[1024];
    BufferWriter document_writer;
    document_writer.destination = document_bytes;
    document_writer.length = 0;
    document_writer.capacity = 1024;
    document_writer.overflowed = 0;
    writer_append_c_string(
            &document_writer,
            // IPC validation establishes only that the client-selected ranges
            // are in-bounds VRAM. It cannot establish tensor semantics or that
            // the workload used those bytes, so the signed prose must say spans.
            "{\"claim\":\"modelHash and modelCID were computed in-GPU from client-submitted, "
            "bounds-checked VRAM spans; "
            "this measurement document was assembled and signed "
            "in-kernel\",\"cubinCID\":\"urn:cid:");
    writer_append_bytes(&document_writer, cubin_cid, cubin_cid_length);
    writer_append_c_string(&document_writer, "\",\"device\":\"cuda:");
    writer_append_uint32(&document_writer, device_ordinal);
    writer_append_c_string(&document_writer, "\",\"gpuDID\":\"");
    writer_append_bytes(&document_writer, (const Byte *)issuer_did, issuer_did_length);
    writer_append_c_string(&document_writer,
                           "\",\"hashScheme\":\"BLAKE3 per submitted span "
                           "(= EQTY raw CID hash); "
                           "modelRoot=BLAKE3(LE32(N)||spanDigests)\",\"kernelCID\":\"urn:cid:");
    writer_append_bytes(&document_writer, kernel_cid, kernel_cid_length);
    writer_append_c_string(&document_writer, "\",\"measuredAt\":\"");
    writer_append_bytes(&document_writer, timestamp, 20);
    writer_append_c_string(&document_writer, "\",\"model\":\"");
    writer_append_bytes(&document_writer, model_name, model_name_length);
    writer_append_c_string(&document_writer, "\",\"modelCID\":\"urn:cid:");
    writer_append_bytes(&document_writer, model_cid, model_cid_length);
    writer_append_c_string(&document_writer, "\",\"modelHash\":\"");
    writer_append_hex(&document_writer, model_root, 32);
    writer_append_c_string(&document_writer,
                           "\",\"operation\":\"gpu-hash-submitted-vram-spans\","
                           "\"tensorCount\":");
    writer_append_uint32(&document_writer, tensor_count);
    writer_append_c_string(&document_writer, "}");
    if (document_writer.overflowed) {
        return -4;
    }

    // SHA-256 is the ECDSA message hash for the exact document bytes.
    sha256_digest(document_bytes, (Uint64)document_writer.length, output_document_signing_hash);

    // Begin the outer receipt and leave its statement map open for the six builders.
    writer_append_c_string(output_writer, "{\"measurementDocument\":\"");
    writer_append_hex(output_writer, document_bytes, document_writer.length);
    writer_append_c_string(output_writer, "\",\"measurementSignature\":\"");
    writer_append_hex(output_writer, raw_signature, 64);
    writer_append_c_string(output_writer, "\",\"modelRoot\":\"");
    writer_append_hex(output_writer, model_root, 32);
    // Statements live inside a versioned manifest envelope. The measurement
    // document and its signature remain siblings for now, because the document
    // is still what measurementSignature covers.
    writer_append_c_string(output_writer, "\",\"manifest\":{\"version\":\"" MANIFEST_VERSION
                                          "\",\"statements\":{");
    return output_writer->overflowed ? -4 : 0;
}

// Construct the two independent ECDSA message hashes without signing them. This is pass one of
// receipt construction: the measurement document contributes hash 0, and the
// StateAttestation contributes hash 1. A scratch writer reuses the final output allocation; pass two starts again
// at byte zero after the two signatures are ready.
__device__ __noinline__ static int
prepare_signing_hashes(const Byte model_root[32], const Byte instance_root[32],
                       const Byte *previous_credential_uuid, int tensor_count, const Byte *timestamp,
                       const Byte *model_name, int model_name_length, const Byte *cubin_digest,
                       const Byte *kernel_digest, int device_ordinal, Byte *scratch_bytes,
                       int scratch_capacity, Byte output_signing_hashes[64]) {
    // Reject signing before key generation.
    if (!g_session_key_ready) {
        return -2;
    }

    // Constrain host strings to forms that can be safely inserted into fixed JSON templates.
    if (!is_valid_timestamp(timestamp) || !is_valid_model_name(model_name, model_name_length) ||
        device_ordinal < 0) {
        return -3;
    }

    // Derive the issuer identifier from the module-private compressed public key.
    char issuer_did[64];
    int issuer_did_length = build_issuer_did(issuer_did);

    // Initialize a writer over the reusable receipt allocation.
    BufferWriter scratch_writer;
    scratch_writer.destination = scratch_bytes;
    scratch_writer.length = 0;
    scratch_writer.capacity = scratch_capacity;
    scratch_writer.overflowed = 0;

    // A zero placeholder preserves the final receipt's exact measurement-document shape.
    Byte zero_signature[64];
    for (int byte_index = 0; byte_index < 64; byte_index++) {
        zero_signature[byte_index] = 0;
    }

    // Hash 0 signs the measurement document itself.
    int preparation_status = write_measurement_prefix(
            model_root, timestamp, model_name, model_name_length, (Uint32)tensor_count,
            cubin_digest, kernel_digest, (Uint32)device_ordinal, issuer_did, issuer_did_length,
            zero_signature, output_signing_hashes, &scratch_writer);
    if (preparation_status != 0) {
        return preparation_status;
    }

    // Hash 1 signs this measurement's StateAttestation.
    //
    // The credential is serialized here and again in pass two, and the two must
    // agree byte for byte or the signature will not verify against what the
    // receipt actually carries. Both passes are therefore handed the SAME
    // predecessor by the caller; neither reads the chain table itself, and the
    // table is not advanced until pass two has succeeded.
    Byte preimage_digest[32];
    scratch_writer.length = 0;
    scratch_writer.overflowed = 0;
    build_credential_id_preimage(issuer_did, issuer_did_length, timestamp, model_root,
                                 instance_root, previous_credential_uuid, preimage_digest,
                                 &scratch_writer);
    if (scratch_writer.overflowed) {
        return -4;
    }
    scratch_writer.length = 0;
    scratch_writer.overflowed = 0;
    build_credential_registration(model_root, issuer_did, issuer_did_length, timestamp,
                                  instance_root, previous_credential_uuid, preimage_digest,
                                  /*raw_signature=*/(const Byte *)0, output_signing_hashes + 32,
                                  /*output_credential_uuid=*/(Byte *)0,
                                  /*output_statement_digest=*/(Byte *)0, &scratch_writer);
    return scratch_writer.overflowed ? -4 : 0;
}

// Build the complete response from a measured root and the two signatures already computed
// on-device. This is pass two of receipt construction. It stays out of the cooperative
// measurement kernel's call graph so large statement-builder locals do not consume every
// hashing thread's stack.
//
// `previous_credential_uuid` must be the same value pass one was given, and
// `output_credential_uuid` returns this measurement's credential id so the
// caller can advance that instance's chain -- but only after this function succeeds.
__device__ __noinline__ static int
assemble_attestation_from_root(const Byte model_root[32], const Byte instance_root[32],
                               const Byte *previous_credential_uuid, int tensor_count,
                               const Byte *timestamp, const Byte *model_name,
                               int model_name_length, const Byte *cubin_digest,
                               const Byte *kernel_digest, int device_ordinal,
                               const Byte signatures[128], Byte *output_json, int output_capacity,
                               int *output_length, Byte output_credential_uuid[16]) {
    // Defense in depth: repeat readiness and host-metadata validation before final assembly.
    if (!g_session_key_ready) {
        return -2;
    }
    if (!is_valid_timestamp(timestamp) || !is_valid_model_name(model_name, model_name_length) ||
        device_ordinal < 0) {
        return -3;
    }

    // Derive the issuer identifier from the module-private compressed public key.
    char issuer_did[64];
    int issuer_did_length = build_issuer_did(issuer_did);

    // The statement builder appends its own "urn:cid:<id>":{...} entry directly to the writer
    // and hands back only fixed-size digests -- never a precomputed "urn:cid:..." string --
    // so nothing that has to survive across the next (large, nested) call is wider than 32
    // bytes. See build_credential_id_preimage's compiler-workaround comment.
    BufferWriter response_writer;
    response_writer.destination = output_json;
    response_writer.length = 0;
    response_writer.capacity = output_capacity;
    response_writer.overflowed = 0;

    // Begin the receipt with the signed measurement document. Signature slot zero belongs to it,
    // and this also opens the manifest envelope and its statement map.
    Byte unused_document_signing_hash[32];
    int assembly_status = write_measurement_prefix(
            model_root, timestamp, model_name, model_name_length, (Uint32)tensor_count,
            cubin_digest, kernel_digest, (Uint32)device_ordinal, issuer_did, issuer_did_length,
            signatures, unused_document_signing_hash, &response_writer);
    if (assembly_status != 0) {
        return assembly_status;
    }

    // Derive this credential's id from everything it is about to assert: what
    // these bytes are, which resident copy they came from, and which
    // credential preceded it for that copy.
    Byte preimage_digest[32];
    build_credential_id_preimage(issuer_did, issuer_did_length, timestamp, model_root,
                                 instance_root, previous_credential_uuid, preimage_digest,
                                 &response_writer);

    // The manifest's one statement: the notary's StateAttestation, registered.
    // Signature slot one is its JWS.
    Byte state_credential_digest[32];
    build_credential_registration(model_root, issuer_did, issuer_did_length, timestamp,
                                  instance_root, previous_credential_uuid, preimage_digest,
                                  signatures + 64, /*output_signing_hash=*/(Byte *)0,
                                  output_credential_uuid, state_credential_digest,
                                  &response_writer);

    // Close the statement map, the manifest, and the outer receipt, then publish the byte length.
    writer_append_c_string(&response_writer, "}}}");
    if (response_writer.overflowed) {
        return -4;
    }
    *output_length = response_writer.length;
    return 0;
}

// ===== RECEIPT-FINALIZATION KERNEL =====
//
// The host launches one 128-thread block after a measurement kernel on the same CUDA stream. All
// threads clear the reusable output allocation with a grid-stride loop. Thread zero atomically
// consumes the private measured-root handoff and prepares four independent signing hashes. Threads
// 0-31 then form four eight-lane signing groups in one warp. Replicas perform
// identical arithmetic but partition each constant-time table scan, reducing
// serial loads without secret-indexed memory. Only each group's leader writes
// the shared signature/status; identical concurrent stores would still race.
// Finally thread zero assembles the receipt and wipes the consumed root.
//
// Keeping this call graph in a separate entry prevents its large P-256 stack and register footprint
// from reducing occupancy for every bulk hashing thread.
extern "C" __global__ void attest_measured_kernel(int tensor_count, const Uint64 *p256_context,
                                                  const Byte *timestamp, const Byte *model_name,
                                                  int model_name_length, const Byte *cubin_digest,
                                                  const Byte *kernel_digest,
                                                  const Byte *instance_root, int device_ordinal,
                                                  Byte *output_json, int output_capacity,
                                                  int *output_length, int *status) {
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.x == 128 && blockDim.y == 1 && blockDim.z == 1);
    assert(output_capacity >= 0 && output_json != 0 && output_length != 0 && status != 0);
    assert(instance_root != 0);
    // Share the two 32-byte signing hashes and two 64-byte signatures across the block.
    __shared__ Byte shared_signing_hashes[2 * 32];
    __shared__ Byte shared_signatures[2 * 64];

    // Share stage results so barriers can hand work from thread zero to the signing lanes and back.
    __shared__ int preparation_status;
    __shared__ int signature_statuses[2];

    // This instance's chain slot and predecessor, resolved once so both receipt
    // passes serialize the identical statement.
    __shared__ int chain_slot;
    __shared__ int chain_has_previous;
    __shared__ Byte chain_previous[16];

    // Compute a conventional global thread index and grid-stride width.
    Uint64 global_thread_index = (Uint64)blockIdx.x * blockDim.x + threadIdx.x;
    Uint64 grid_thread_count = (Uint64)gridDim.x * blockDim.x;

    // Mark the output as in progress before any parallel work starts.
    if (global_thread_index == 0) {
        *output_length = 0;
        *status = 1;
    }

    // The host reads the entire reusable output allocation. Clearing its tail
    // prevents data from an earlier request from crossing that boundary.
    for (Uint64 output_offset = global_thread_index; output_offset < (Uint64)output_capacity;
         output_offset += grid_thread_count) {
        output_json[output_offset] = 0;
    }

    // Wait until every output byte is clear before thread zero reuses that allocation as scratch.
    __syncthreads();

    // Thread zero consumes the one-shot root, resolves this instance's chain,
    // and creates both signing hashes.
    if (global_thread_index == 0) {
        chain_slot = -1;
        chain_has_previous = 0;
        int pending_receipt_was_armed = atomicExch(&g_pending_receipt_armed, 0);
        if (pending_receipt_was_armed != 1 || g_pending_tensor_count != tensor_count) {
            preparation_status = -6;
        } else if (g_chain_slots <= 0) {
            // Signing before configure_chain_slots_kernel would chain nothing.
            preparation_status = -11;
        } else {
            int existing_slot = chain_find(instance_root);
            if (existing_slot >= 0) {
                chain_slot = existing_slot;
                chain_has_previous = 1;
                for (int byte_index = 0; byte_index < 16; byte_index++) {
                    chain_previous[byte_index] =
                            g_chains[existing_slot].last_credential_uuid[byte_index];
                }
            } else {
                // First signature for this copy: a genesis statement, and a slot
                // is reserved but not claimed until pass two succeeds.
                chain_slot = chain_free_slot();
            }
            // Refusing is deliberate. Evicting a live slot would emit a genesis
            // statement for a copy that already has predecessors, which no
            // verifier could tell apart from a forked chain.
            preparation_status =
                    chain_slot < 0
                            ? -10
                            : prepare_signing_hashes(
                                      g_pending_model_root, instance_root,
                                      chain_has_previous ? chain_previous : (const Byte *)0,
                                      tensor_count, timestamp, model_name, model_name_length,
                                      cubin_digest, kernel_digest, device_ordinal, output_json,
                                      output_capacity, shared_signing_hashes);
        }
    }

    // Publish the prepared hashes and status to the two signing groups.
    __syncthreads();

    // Two complete eight-lane groups; all replicas receive the SAME inputs.
    if (global_thread_index < 16 && preparation_status == 0) {
        int signature_index = (int)global_thread_index / 8;
        Uint64 private_scalar[4];
        Uint64 signature_r[4];
        Uint64 signature_s[4];
        big_endian_bytes_to_uint256(g_private_scalar_big_endian, private_scalar);
        int signing_status = sign_ecdsa_p256<8>(p256_context, private_scalar,
                                             shared_signing_hashes + (Uint64)signature_index * 32,
                                             signature_r, signature_s);
        if (((int)global_thread_index & 7) == 0)
            signature_statuses[signature_index] = signing_status;

        // Serialize a successful ECDSA signature as raw big-endian r || s.
        if (signing_status == 0 && ((int)global_thread_index & 7) == 0) {
            uint256_to_big_endian_bytes(signature_r,
                                        shared_signatures + (Uint64)signature_index * 64);
            uint256_to_big_endian_bytes(signature_s,
                                        shared_signatures + (Uint64)signature_index * 64 + 32);
        }

        // Best-effort clearing removes this lane's local private-scalar copy.
        for (int limb_index = 0; limb_index < 4; limb_index++) {
            private_scalar[limb_index] = 0;
        }
    }

    // Wait until both signature leaders have published their results.
    __syncthreads();

    // Thread zero validates both signatures and performs pass-two receipt assembly.
    if (global_thread_index == 0) {
        if (preparation_status == 0) {
            for (int signature_index = 0; signature_index < 2; signature_index++) {
                if (signature_statuses[signature_index] != 0) {
                    preparation_status = -7;
                }
            }
        }
        Byte credential_uuid[16];
        *status = preparation_status == 0
                          ? assemble_attestation_from_root(
                                    g_pending_model_root, instance_root,
                                    chain_has_previous ? chain_previous : (const Byte *)0,
                                    tensor_count, timestamp, model_name, model_name_length,
                                    cubin_digest, kernel_digest, device_ordinal, shared_signatures,
                                    output_json, output_capacity, output_length,
                                    credential_uuid)
                          : preparation_status;

        // Advance this copy's chain only once the receipt is complete. Advancing
        // earlier would leave the next statement pointing at a predecessor that
        // was never emitted, which still verifies alone but breaks the sequence.
        if (*status == 0) {
            chain_commit(chain_slot, instance_root, credential_uuid);
        }

        // Wipe and invalidate the consumed one-shot handoff regardless of assembly success.
        for (int byte_index = 0; byte_index < 32; byte_index++) {
            g_pending_model_root[byte_index] = 0;
        }
        g_pending_tensor_count = 0;
    }
}

// ===== COOPERATIVE MODEL-MEASUREMENT KERNEL =====
//
// This entry is separate from P-256 and statement assembly so their large local frames do not lower
// hashing occupancy. The host chooses a cooperative grid small enough for every block to reside at
// once, which makes `cooperative_grid.sync()` legal. The algorithm proceeds through five phases:
//
// 1. Tile hashing: each 128-thread block is four warps split into two 64-thread tile groups. Every
//    group hashes one 128 KiB scheduling tile; each of its lanes owns two adjacent 1 KiB chunks.
// 2. Shared reduction: seven levels reduce 128 leaf CVs to one tile CV in an 8,384-byte padded,
//    word-transposed shared array. Both groups use block-wide barriers in exactly the same order.
// 3. Global reduction: all threads cooperatively reduce tile CVs for every tensor. Adjacent global
//    ranks read adjacent CV pairs, so this phase's vector loads and stores are coalesced.
// 4. Root extraction: grid-stride threads serialize one root digest per tensor.
// 5. Model fold: thread zero computes BLAKE3(LE32(tensor_count) || tensor_digests), optionally
//    publishing it through the private one-shot receipt handoff.
template <bool UseAsync>
__device__ __forceinline__ void measure_model_fused_impl(const TensorMeasurementSpan *tensor_spans, int tensor_count,
                           const Uint64 *reduction_level_offsets, int global_reduction_level_count,
                           Uint64 total_tile_count, Uint32 *primary_workspace,
                           Uint32 *secondary_workspace, Byte *tensor_digests,
                           Byte *output_model_root, int arm_receipt, int *status) {
    // Allocate independent shared arrays for the block's two 64-thread tile groups. Transposing the
    // eight CV words makes one warp's same-word accesses span banks; three padding slots break bank
    // aliasing at each 32-entry boundary.
    __shared__ Uint32 shared_tile_chaining_values[2][8][PADDED_TILE_STRIDE];

    // Obtain cooperative-grid identity for global barriers and grid-stride loops.
    cg::grid_group cooperative_grid = cg::this_grid();
    Uint64 global_thread_index = cooperative_grid.thread_rank();
    Uint64 grid_thread_count = cooperative_grid.size();

    // Thread zero invalidates any stale signing handoff and validates launch invariants.
    if (global_thread_index == 0) {
        atomicExch(&g_pending_receipt_armed, 0);
        g_pending_tensor_count = 0;
        int launch_is_valid = tensor_count > 0 && total_tile_count > 0 &&
                              global_reduction_level_count >= 0 && blockDim.x == 128;
        *status = launch_is_valid ? 1 : -5;
    }

    // Make the launch decision visible to every block before any thread can return.
    cooperative_grid.sync();
    if (*status != 1) {
        return;
    }

    // The inexpensive release status checks above remain authoritative for
    // malformed launch input. These deeper internal layout/schedule checks
    // disappear entirely with NDEBUG, including the descriptor/prefix scan.
    assert(cooperative_grid.is_valid());
    assert(blockDim.x == 128 && blockDim.y == 1 && blockDim.z == 1);
    assert(gridDim.y == 1 && gridDim.z == 1 && grid_thread_count > 0);
    assert(tensor_spans != 0 && reduction_level_offsets != 0 && tensor_digests != 0);
    assert(output_model_root != 0 && primary_workspace != 0 && secondary_workspace != 0);
    assert(primary_workspace != secondary_workspace);
    assert(((Uint64)primary_workspace & 15) == 0 && ((Uint64)secondary_workspace & 15) == 0);
    assert(global_reduction_level_count < 64 && (arm_receipt == 0 || arm_receipt == 1));
#ifndef NDEBUG
    if (global_thread_index == 0) {
        Uint64 primary = 0, secondary = 0;
        int maximum_rounds = 0;
        for (int i = 0; i < tensor_count; ++i) {
            const TensorMeasurementSpan &span = tensor_spans[i];
            Uint64 chunks = span.byte_count == 0 ? 1 : 1 + (span.byte_count - 1) / 1024;
            Uint64 tiles = 1 + (chunks - 1) / BLAKE3_CHUNKS_PER_TILE;
            assert(span.semantic_chunk_count == chunks);
            assert(span.byte_count == 0 || span.device_bytes != 0);
            assert(span.primary_tile_offset == primary && span.secondary_tile_offset == secondary);
            assert(tiles <= (~0ull / 32) - primary);
            primary += tiles;
            secondary += tiles > 1 ? (tiles + 1) / 2 : 0;
            int rounds = count_reduction_rounds(tiles);
            if (rounds > maximum_rounds) maximum_rounds = rounds;
        }
        assert(primary == total_tile_count && secondary <= primary);
        assert(maximum_rounds == global_reduction_level_count);
        for (int level = 0; level < global_reduction_level_count; ++level) {
            const Uint64 *row = reduction_level_offsets + (Uint64)level * (tensor_count + 1);
            Uint64 prefix = 0;
            assert(row[0] == 0);
            for (int i = 0; i < tensor_count; ++i) {
                Uint64 tiles = 1 + (tensor_spans[i].semantic_chunk_count - 1) / BLAKE3_CHUNKS_PER_TILE;
                Uint64 width = 1ull << level;
                Uint64 nodes = 1 + (tiles - 1) / width;
                prefix += nodes > 1 ? (nodes + 1) / 2 : 0;
                assert(row[i + 1] == prefix);
            }
        }
    }
    // All lanes rendezvous before using a plan that thread zero just checked.
    cooperative_grid.sync();
#endif

    // Split four physical warps into two groups. Group 0 is warps 0-1; group 1 is warps 2-3.
    int tile_group_index = (int)threadIdx.x / THREADS_PER_TILE;
    int tile_lane_index = (int)threadIdx.x & (THREADS_PER_TILE - 1);
    assert(tile_group_index < 2 && tile_lane_index < THREADS_PER_TILE);

    // Each block walks a grid-stride sequence of tile pairs. Adding the group index selects one of
    // the two tiles in the pair without requiring independent control flow per half-block.
    for (Uint64 tile_pair_base = (Uint64)blockIdx.x * 2; tile_pair_base < total_tile_count;
         tile_pair_base += (Uint64)gridDim.x * 2) {
        Uint64 flattened_tile_index = tile_pair_base + (Uint64)tile_group_index;
        int tile_exists = flattened_tile_index < total_tile_count;

        // Resolve the flattened tile to its tensor and tensor-local semantic chunk range.
        int tensor_index = 0;
        Uint64 tensor_tile_index = 0;
        Uint64 first_semantic_chunk = 0;
        Uint64 chunks_in_tile = 0;
        Uint64 tiles_in_tensor = 0;
        const TensorMeasurementSpan *tensor_span = 0;
        if (tile_exists) {
            tensor_index = find_tensor_for_tile(tensor_spans, tensor_count, flattened_tile_index);
            tensor_span = &tensor_spans[tensor_index];
            assert(tensor_index >= 0 && tensor_index < tensor_count);
            assert(flattened_tile_index >= tensor_span->primary_tile_offset);
            tensor_tile_index = flattened_tile_index - tensor_span->primary_tile_offset;
            first_semantic_chunk = tensor_tile_index * BLAKE3_CHUNKS_PER_TILE;
            Uint64 remaining_chunks = tensor_span->semantic_chunk_count - first_semantic_chunk;
            assert(first_semantic_chunk < tensor_span->semantic_chunk_count);
            chunks_in_tile = remaining_chunks > BLAKE3_CHUNKS_PER_TILE ? BLAKE3_CHUNKS_PER_TILE
                                                                       : remaining_chunks;
            tiles_in_tensor = (tensor_span->semantic_chunk_count + BLAKE3_CHUNKS_PER_TILE - 1) /
                              BLAKE3_CHUNKS_PER_TILE;
        }

        // A 64-thread group assigns semantic chunks [2*lane, 2*lane+1] to each lane. Within a warp,
        // base addresses are therefore 2 KiB apart. Each individual load is aligned/vectorized,
        // while two per-lane dependency chains provide the measured instruction-level parallelism.
        Uint64 first_local_chunk_index = (Uint64)tile_lane_index * 2;
        if (tile_exists && first_local_chunk_index < chunks_in_tile) {
            // Derive the first chunk's tensor-wide index, byte offset, and bounded length.
            Uint64 first_chunk_index = first_semantic_chunk + first_local_chunk_index;
            Uint64 first_chunk_byte_offset = first_chunk_index * 1024;
            assert(first_chunk_byte_offset <= tensor_span->byte_count);
            Uint64 first_chunk_remaining_bytes = tensor_span->byte_count - first_chunk_byte_offset;
            Uint32 first_chunk_byte_count =
                    (Uint32)(first_chunk_remaining_bytes > 1024 ? 1024
                                                                : first_chunk_remaining_bytes);

            // The second adjacent chunk exists for every lane except the final odd leaf.
            int has_second_chunk = first_local_chunk_index + 1 < chunks_in_tile;
            Uint64 second_chunk_index = first_chunk_index + 1;
            Uint64 second_chunk_byte_offset = second_chunk_index * 1024;
            Uint64 second_chunk_remaining_bytes =
                    has_second_chunk ? tensor_span->byte_count - second_chunk_byte_offset : 0;
            Uint32 second_chunk_byte_count =
                    (Uint32)(second_chunk_remaining_bytes > 1024 ? 1024
                                                                 : second_chunk_remaining_bytes);

            // Keep both eight-word chaining values in named scalar registers. Arrays here caused
            // NVRTC to spill into thread-local memory in the profiled implementation.
            Uint32 first_cv_0;
            Uint32 first_cv_1;
            Uint32 first_cv_2;
            Uint32 first_cv_3;
            Uint32 first_cv_4;
            Uint32 first_cv_5;
            Uint32 first_cv_6;
            Uint32 first_cv_7;
            Uint32 second_cv_0;
            Uint32 second_cv_1;
            Uint32 second_cv_2;
            Uint32 second_cv_3;
            Uint32 second_cv_4;
            Uint32 second_cv_5;
            Uint32 second_cv_6;
            Uint32 second_cv_7;

            // Use the paired fast path only when both complete chunks meet uint4 alignment.
            int first_chunk_is_aligned =
                    (((Uint64)(tensor_span->device_bytes + first_chunk_byte_offset) & 15ull) == 0);
            int second_chunk_is_aligned =
                    (((Uint64)(tensor_span->device_bytes + second_chunk_byte_offset) & 15ull) == 0);
            int tensor_has_one_semantic_chunk = tensor_span->semantic_chunk_count == 1;
            if (has_second_chunk && first_chunk_byte_count == 1024 &&
                second_chunk_byte_count == 1024 && first_chunk_is_aligned &&
                second_chunk_is_aligned) {
                FullChunkPair<UseAsync>::hash(
                        tensor_span->device_bytes + first_chunk_byte_offset, first_chunk_index,
                        tensor_has_one_semantic_chunk,
                        tensor_span->device_bytes + second_chunk_byte_offset, second_chunk_index,
                        tensor_has_one_semantic_chunk, first_cv_0, first_cv_1, first_cv_2,
                        first_cv_3, first_cv_4, first_cv_5, first_cv_6, first_cv_7, second_cv_0,
                        second_cv_1, second_cv_2, second_cv_3, second_cv_4, second_cv_5,
                        second_cv_6, second_cv_7);
            } else {
                // Hash the first leaf with the aligned scalar path or general byte-safe path.
                if (first_chunk_byte_count == 1024 && first_chunk_is_aligned) {
                    blake3_hash_aligned_full_chunk(
                            tensor_span->device_bytes + first_chunk_byte_offset, first_chunk_index,
                            tensor_has_one_semantic_chunk, first_cv_0, first_cv_1, first_cv_2,
                            first_cv_3, first_cv_4, first_cv_5, first_cv_6, first_cv_7);
                } else {
                    Uint32 first_chaining_value[8];
                    blake3_hash_chunk(tensor_span->device_bytes + first_chunk_byte_offset,
                                      first_chunk_byte_count, first_chunk_index,
                                      tensor_has_one_semantic_chunk, first_chaining_value);
                    first_cv_0 = first_chaining_value[0];
                    first_cv_1 = first_chaining_value[1];
                    first_cv_2 = first_chaining_value[2];
                    first_cv_3 = first_chaining_value[3];
                    first_cv_4 = first_chaining_value[4];
                    first_cv_5 = first_chaining_value[5];
                    first_cv_6 = first_chaining_value[6];
                    first_cv_7 = first_chaining_value[7];
                }

                // Hash the second leaf only when this lane owns one.
                if (has_second_chunk) {
                    if (second_chunk_byte_count == 1024 && second_chunk_is_aligned) {
                        blake3_hash_aligned_full_chunk(
                                tensor_span->device_bytes + second_chunk_byte_offset,
                                second_chunk_index, tensor_has_one_semantic_chunk, second_cv_0,
                                second_cv_1, second_cv_2, second_cv_3, second_cv_4, second_cv_5,
                                second_cv_6, second_cv_7);
                    } else {
                        Uint32 second_chaining_value[8];
                        blake3_hash_chunk(tensor_span->device_bytes + second_chunk_byte_offset,
                                          second_chunk_byte_count, second_chunk_index,
                                          tensor_has_one_semantic_chunk, second_chaining_value);
                        second_cv_0 = second_chaining_value[0];
                        second_cv_1 = second_chaining_value[1];
                        second_cv_2 = second_chaining_value[2];
                        second_cv_3 = second_chaining_value[3];
                        second_cv_4 = second_chaining_value[4];
                        second_cv_5 = second_chaining_value[5];
                        second_cv_6 = second_chaining_value[6];
                        second_cv_7 = second_chaining_value[7];
                    }
                }
            }

            // Transpose the first CV into shared memory: [group][word][padded leaf slot].
            int first_shared_slot = shared_memory_tile_slot(first_local_chunk_index);
            shared_tile_chaining_values[tile_group_index][0][first_shared_slot] = first_cv_0;
            shared_tile_chaining_values[tile_group_index][1][first_shared_slot] = first_cv_1;
            shared_tile_chaining_values[tile_group_index][2][first_shared_slot] = first_cv_2;
            shared_tile_chaining_values[tile_group_index][3][first_shared_slot] = first_cv_3;
            shared_tile_chaining_values[tile_group_index][4][first_shared_slot] = first_cv_4;
            shared_tile_chaining_values[tile_group_index][5][first_shared_slot] = first_cv_5;
            shared_tile_chaining_values[tile_group_index][6][first_shared_slot] = first_cv_6;
            shared_tile_chaining_values[tile_group_index][7][first_shared_slot] = first_cv_7;

            // Store the adjacent second CV in the following logical slot when present.
            if (has_second_chunk) {
                int second_shared_slot = shared_memory_tile_slot(first_local_chunk_index + 1);
                shared_tile_chaining_values[tile_group_index][0][second_shared_slot] = second_cv_0;
                shared_tile_chaining_values[tile_group_index][1][second_shared_slot] = second_cv_1;
                shared_tile_chaining_values[tile_group_index][2][second_shared_slot] = second_cv_2;
                shared_tile_chaining_values[tile_group_index][3][second_shared_slot] = second_cv_3;
                shared_tile_chaining_values[tile_group_index][4][second_shared_slot] = second_cv_4;
                shared_tile_chaining_values[tile_group_index][5][second_shared_slot] = second_cv_5;
                shared_tile_chaining_values[tile_group_index][6][second_shared_slot] = second_cv_6;
                shared_tile_chaining_values[tile_group_index][7][second_shared_slot] = second_cv_7;
            }
        }

        // A block-wide barrier publishes both groups' leaf CVs before either starts reduction.
        __syncthreads();

        // Seven fixed rounds reduce at most 128 leaves. Both 64-thread groups must execute every
        // block-wide barrier even when one owns a shorter final tile or no tile at all.
        Uint64 nodes_in_level = chunks_in_tile;
#pragma unroll 1
        for (int tile_reduction_level = 0; tile_reduction_level < 7; tile_reduction_level++) {
            Uint64 next_level_node_count = (nodes_in_level + 1) / 2;
            int lane_produces_output = tile_exists && nodes_in_level > 1 &&
                                       (Uint64)tile_lane_index < next_level_node_count;

            // Hold the complete parent in registers until all lanes have finished reading children.
            Uint32 parent_cv_0 = 0;
            Uint32 parent_cv_1 = 0;
            Uint32 parent_cv_2 = 0;
            Uint32 parent_cv_3 = 0;
            Uint32 parent_cv_4 = 0;
            Uint32 parent_cv_5 = 0;
            Uint32 parent_cv_6 = 0;
            Uint32 parent_cv_7 = 0;
            if (lane_produces_output) {
                Uint64 left_child_index = (Uint64)tile_lane_index * 2;
                if (left_child_index + 1 < nodes_in_level) {
                    // Compress a complete pair. The final tile-local parent is ROOT only when this
                    // tensor has no additional tiles waiting for global reduction.
                    int parent_is_tensor_root = tiles_in_tensor == 1 && nodes_in_level == 2;
                    blake3_compress_parent_from_shared_memory(
                            &shared_tile_chaining_values[tile_group_index][0][0],
                            shared_memory_tile_slot(left_child_index),
                            shared_memory_tile_slot(left_child_index + 1), parent_is_tensor_root,
                            parent_cv_0, parent_cv_1, parent_cv_2, parent_cv_3, parent_cv_4,
                            parent_cv_5, parent_cv_6, parent_cv_7);
                } else {
                    // BLAKE3 carries an unmatched odd right edge upward without recompression.
                    int child_slot = shared_memory_tile_slot(left_child_index);
                    parent_cv_0 = shared_tile_chaining_values[tile_group_index][0][child_slot];
                    parent_cv_1 = shared_tile_chaining_values[tile_group_index][1][child_slot];
                    parent_cv_2 = shared_tile_chaining_values[tile_group_index][2][child_slot];
                    parent_cv_3 = shared_tile_chaining_values[tile_group_index][3][child_slot];
                    parent_cv_4 = shared_tile_chaining_values[tile_group_index][4][child_slot];
                    parent_cv_5 = shared_tile_chaining_values[tile_group_index][5][child_slot];
                    parent_cv_6 = shared_tile_chaining_values[tile_group_index][6][child_slot];
                    parent_cv_7 = shared_tile_chaining_values[tile_group_index][7][child_slot];
                }
            }

            // No lane may overwrite a child while another lane still reads it.
            __syncthreads();

            // Compact this level's register-resident outputs over the old shared-memory children.
            if (lane_produces_output) {
                int output_slot = shared_memory_tile_slot((Uint64)tile_lane_index);
                shared_tile_chaining_values[tile_group_index][0][output_slot] = parent_cv_0;
                shared_tile_chaining_values[tile_group_index][1][output_slot] = parent_cv_1;
                shared_tile_chaining_values[tile_group_index][2][output_slot] = parent_cv_2;
                shared_tile_chaining_values[tile_group_index][3][output_slot] = parent_cv_3;
                shared_tile_chaining_values[tile_group_index][4][output_slot] = parent_cv_4;
                shared_tile_chaining_values[tile_group_index][5][output_slot] = parent_cv_5;
                shared_tile_chaining_values[tile_group_index][6][output_slot] = parent_cv_6;
                shared_tile_chaining_values[tile_group_index][7][output_slot] = parent_cv_7;
            }

            // Publish the compacted level before any lane reads it in the next round.
            __syncthreads();
            if (nodes_in_level > 1) {
                nodes_in_level = next_level_node_count;
            }
        }

        // One group leader writes the final 32-byte tile CV to primary global scratch as two
        // aligned uint4 vectors. This is the only global scratch write for 128 KiB of input.
        if (tile_exists && tile_lane_index == 0) {
            uint4 *tile_output = (uint4 *)&primary_workspace[(tensor_span->primary_tile_offset +
                                                              tensor_tile_index) *
                                                             8];
            tile_output[0] = make_uint4(shared_tile_chaining_values[tile_group_index][0][0],
                                        shared_tile_chaining_values[tile_group_index][1][0],
                                        shared_tile_chaining_values[tile_group_index][2][0],
                                        shared_tile_chaining_values[tile_group_index][3][0]);
            tile_output[1] = make_uint4(shared_tile_chaining_values[tile_group_index][4][0],
                                        shared_tile_chaining_values[tile_group_index][5][0],
                                        shared_tile_chaining_values[tile_group_index][6][0],
                                        shared_tile_chaining_values[tile_group_index][7][0]);
        }

        // Protect the final CV until both group leaders copy it, then allow shared-memory reuse.
        __syncthreads();
    }

    // Ensure every tile CV is globally visible before cross-tile reduction begins.
    cooperative_grid.sync();

    // Reduce tile CVs level by level. Host-provided prefix rows flatten the active outputs of
    // differently sized tensors while preserving a separate tree for each tensor.
    for (int reduction_level = 0; reduction_level < global_reduction_level_count;
         reduction_level++) {
        const Uint64 *level_prefix_offsets =
                reduction_level_offsets + (Uint64)reduction_level * (tensor_count + 1);
        Uint64 total_outputs_in_level = level_prefix_offsets[tensor_count];
        Uint64 subtree_width_in_tiles = 1ull << reduction_level;

        // Only ranks below the first loop width need enter this grid-stride reduction loop.
        if (global_thread_index < total_outputs_in_level) {
            int tensor_index = find_tensor_for_reduction_output(level_prefix_offsets, tensor_count,
                                                                global_thread_index);
            for (Uint64 flattened_output_index = global_thread_index;
                 flattened_output_index < total_outputs_in_level;
                 flattened_output_index += grid_thread_count) {
                // Advance monotonically through tensor boundaries in the flattened prefix row.
                while (tensor_index + 1 < tensor_count &&
                       level_prefix_offsets[tensor_index + 1] <= flattened_output_index) {
                    tensor_index++;
                }

                // Resolve this output to its tensor-local pair of child nodes.
                const TensorMeasurementSpan &tensor_span = tensor_spans[tensor_index];
                Uint64 tensor_output_index =
                        flattened_output_index - level_prefix_offsets[tensor_index];
                Uint64 tensor_tile_count =
                        (tensor_span.semantic_chunk_count + BLAKE3_CHUNKS_PER_TILE - 1) /
                        BLAKE3_CHUNKS_PER_TILE;
                Uint64 nodes_in_level =
                        (tensor_tile_count + subtree_width_in_tiles - 1) / subtree_width_in_tiles;
                Uint64 left_child_index = 2 * tensor_output_index;
                assert(nodes_in_level > 1 && left_child_index < nodes_in_level);
                assert(flattened_output_index >= level_prefix_offsets[tensor_index] &&
                       flattened_output_index < level_prefix_offsets[tensor_index + 1]);

                // Ping-pong source and destination workspaces at each global level.
                const Uint32 *source_workspace =
                        (reduction_level & 1) ? secondary_workspace : primary_workspace;
                Uint32 *destination_workspace =
                        (reduction_level & 1) ? primary_workspace : secondary_workspace;
                Uint64 source_tensor_offset = (reduction_level & 1)
                                                      ? tensor_span.secondary_tile_offset
                                                      : tensor_span.primary_tile_offset;
                Uint64 destination_tensor_offset = (reduction_level & 1)
                                                           ? tensor_span.primary_tile_offset
                                                           : tensor_span.secondary_tile_offset;
                uint4 *output_vectors = (uint4 *)&destination_workspace[(destination_tensor_offset +
                                                                         tensor_output_index) *
                                                                        8];

                if (left_child_index + 1 < nodes_in_level) {
                    // Adjacent ranks consume adjacent 64-byte child pairs. Their aligned uint4
                    // transactions are contiguous across the warp and therefore coalesced.
                    Uint32 parent_cv_0;
                    Uint32 parent_cv_1;
                    Uint32 parent_cv_2;
                    Uint32 parent_cv_3;
                    Uint32 parent_cv_4;
                    Uint32 parent_cv_5;
                    Uint32 parent_cv_6;
                    Uint32 parent_cv_7;
                    blake3_compress_aligned_parent(
                            &source_workspace[(source_tensor_offset + left_child_index) * 8],
                            &source_workspace[(source_tensor_offset + left_child_index + 1) * 8],
                            /*is_root=*/nodes_in_level == 2, parent_cv_0, parent_cv_1, parent_cv_2,
                            parent_cv_3, parent_cv_4, parent_cv_5, parent_cv_6, parent_cv_7);
                    output_vectors[0] =
                            make_uint4(parent_cv_0, parent_cv_1, parent_cv_2, parent_cv_3);
                    output_vectors[1] =
                            make_uint4(parent_cv_4, parent_cv_5, parent_cv_6, parent_cv_7);
                } else {
                    // Carry an odd final child unchanged into the destination workspace.
                    const uint4 *input_vectors =
                            (const uint4
                                     *)&source_workspace[(source_tensor_offset + left_child_index) *
                                                         8];
                    output_vectors[0] = input_vectors[0];
                    output_vectors[1] = input_vectors[1];
                }
            }
        }

        // All outputs at this level must be complete before source/destination roles swap.
        cooperative_grid.sync();
    }

    // Assign tensors to threads with another grid-stride loop and serialize each final CV.
    for (Uint64 tensor_index = global_thread_index; tensor_index < (Uint64)tensor_count;
         tensor_index += grid_thread_count) {
        const TensorMeasurementSpan &tensor_span = tensor_spans[tensor_index];
        Uint64 tensor_tile_count = (tensor_span.semantic_chunk_count + BLAKE3_CHUNKS_PER_TILE - 1) /
                                   BLAKE3_CHUNKS_PER_TILE;
        int completed_round_count = count_reduction_rounds(tensor_tile_count);
        assert(completed_round_count <= global_reduction_level_count);

        // Reduction parity determines which workspace contains the final tensor root.
        const Uint32 *root_workspace =
                (completed_round_count & 1) ? secondary_workspace : primary_workspace;
        Uint64 root_workspace_offset = (completed_round_count & 1)
                                               ? tensor_span.secondary_tile_offset
                                               : tensor_span.primary_tile_offset;

        // BLAKE3 digests serialize each root word in little-endian byte order.
        for (int word_index = 0; word_index < 8; word_index++) {
            Uint32 root_word = root_workspace[root_workspace_offset * 8 + word_index];
            tensor_digests[tensor_index * 32 + 4 * word_index] = (Byte)root_word;
            tensor_digests[tensor_index * 32 + 4 * word_index + 1] = (Byte)(root_word >> 8);
            tensor_digests[tensor_index * 32 + 4 * word_index + 2] = (Byte)(root_word >> 16);
            tensor_digests[tensor_index * 32 + 4 * word_index + 3] = (Byte)(root_word >> 24);
        }
    }

    // Make every serialized tensor digest visible before thread zero folds the model root.
    cooperative_grid.sync();

    // One thread folds a short tensor-count prefix and the ordered tensor digests. Parallelizing
    // this receipt-sized input would cost more synchronization than it saves.
    if (global_thread_index == 0) {
        // This final fold is also exclusively global-thread-zero work, AFTER
        // all parallel hashing has finished. A local 54-level streaming state
        // would charge its stack backing to every possible resident thread,
        // wasting VRAM when the model already occupies nearly all of it.
        // The independent per-tile/per-thread hash states above stay private.
        __shared__ Blake3StreamingState model_hasher;
        blake3_stream_initialize(&model_hasher);

        // Bind tensor count and order into the model root with a four-byte little-endian prefix.
        Byte tensor_count_little_endian[4];
        tensor_count_little_endian[0] = (Byte)tensor_count;
        tensor_count_little_endian[1] = (Byte)(tensor_count >> 8);
        tensor_count_little_endian[2] = (Byte)(tensor_count >> 16);
        tensor_count_little_endian[3] = (Byte)(tensor_count >> 24);
        blake3_stream_update(&model_hasher, tensor_count_little_endian, 4);
        blake3_stream_update(&model_hasher, tensor_digests, (Uint64)tensor_count * 32);
        blake3_stream_finalize(&model_hasher, output_model_root);

        // Optionally publish the root for exactly one subsequent same-stream receipt kernel.
        if (arm_receipt) {
            for (int byte_index = 0; byte_index < 32; byte_index++) {
                g_pending_model_root[byte_index] = output_model_root[byte_index];
            }
            g_pending_tensor_count = tensor_count;

            // Order root/count writes before the atomic flag that makes them consumable.
            __threadfence();
            atomicExch(&g_pending_receipt_armed, 1);
        }

        // Publish successful completion to the host.
        *status = 0;
    }
}

// Both entry points share exact tails, reduction, ROOT flags and the private
// one-shot signing handoff. Host dispatch is based only on architecture/size.
extern "C" __global__ void
measure_model_fused_kernel(const TensorMeasurementSpan *tensor_spans, int tensor_count,
                           const Uint64 *reduction_level_offsets, int global_reduction_level_count,
                           Uint64 total_tile_count, Uint32 *primary_workspace,
                           Uint32 *secondary_workspace, Byte *tensor_digests,
                           Byte *output_model_root, int arm_receipt, int *status) {
    measure_model_fused_impl<false>(
            tensor_spans, tensor_count, reduction_level_offsets, global_reduction_level_count,
            total_tile_count, primary_workspace, secondary_workspace, tensor_digests,
            output_model_root, arm_receipt, status);
}

extern "C" __global__ void
measure_model_fused_async_kernel(const TensorMeasurementSpan *tensor_spans, int tensor_count,
                           const Uint64 *reduction_level_offsets, int global_reduction_level_count,
                           Uint64 total_tile_count, Uint32 *primary_workspace,
                           Uint32 *secondary_workspace, Byte *tensor_digests,
                           Byte *output_model_root, int arm_receipt, int *status) {
    measure_model_fused_impl<true>(
            tensor_spans, tensor_count, reduction_level_offsets, global_reduction_level_count,
            total_tile_count, primary_workspace, secondary_workspace, tensor_digests,
            output_model_root, arm_receipt, status);
}

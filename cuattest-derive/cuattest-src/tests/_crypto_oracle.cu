// SPDX-License-Identifier: Apache-2.0
// TEST ONLY: appended to the unchanged production translation unit by pytest.
// These are adapters, not alternate implementations of the algorithms. Never
// ship this source/CUBIN or add these entry points to the notary: they accept
// known test keys and export intermediate values for independent comparison.

// Every eight-lane group has identical inputs; adjacent groups deliberately
// differ. Only group leaders publish results, so no identical stores race.
extern "C" __global__ void crypto_test_grouped_signatures(
        const Uint64 *context, const Byte *inputs, Byte *output, int count) {
    int rank = blockIdx.x * blockDim.x + threadIdx.x;
    int groups = blockDim.x * gridDim.x / 8;
    for (int i = rank / 8; i < count; i += groups) {
        Uint64 scalar[4], nonce[4], r[4], s[4], x[4], y[4];
        big_endian_bytes_to_uint256(inputs + i * 64, scalar);
        const Byte *digest = inputs + i * 64 + 32;
        int nonce_status = derive_rfc6979_nonce(
                context + CONTEXT_GROUP_ORDER_OFFSET, scalar, digest, nonce);
        int sign_status = sign_ecdsa_p256<8>(context, scalar, digest, r, s);
        multiply_generator_by_scalar<8>(context, scalar, x, y);
        if ((threadIdx.x & 7) != 0) continue;
        Byte *record = output + i * 168;
        for (int j = 0; j < 168; ++j) record[j] = 0;
        ((int *)(record + 160))[0] = nonce_status;
        ((int *)(record + 160))[1] = sign_status;
        if (nonce_status != 0 || sign_status != 0) continue;
        uint256_to_big_endian_bytes(nonce, record);
        uint256_to_big_endian_bytes(r, record + 32);
        uint256_to_big_endian_bytes(s, record + 64);
        uint256_to_big_endian_bytes(x, record + 96);
        uint256_to_big_endian_bytes(y, record + 128);
    }
}

// Arbitrary 512-bit values expose the worst signed coefficients/carries, not
// only the narrower products of already reduced field elements. Check aliasing.
extern "C" __global__ void crypto_test_prime_reduction(
        const Uint64 *context, const Byte *inputs, Byte *output, int count) {
    const Uint64 *prime = context + CONTEXT_FIELD_PRIME_OFFSET;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        Uint64 product[8], alias[8], reduced[4];
        const Uint64 *words = (const Uint64 *)(inputs + i * 64);
        for (int j = 0; j < 8; ++j) product[j] = alias[j] = words[j];
        p256_reduce_product(product, prime, reduced);
        p256_reduce_product(alias, prime, alias);
        uint256_to_big_endian_bytes(reduced, output + i * 64);
        uint256_to_big_endian_bytes(alias, output + i * 64 + 32);
    }
}

extern "C" __global__ void crypto_test_montgomery_products(
        const Uint64 *context, const Byte *inputs, Byte *output, int count, int use_order) {
    const Uint64 *modulus = context + (use_order ? CONTEXT_GROUP_ORDER_OFFSET
                                               : CONTEXT_FIELD_PRIME_OFFSET);
    const Uint64 *factor = context + (use_order ? CONTEXT_ORDER_BARRETT_FACTOR_OFFSET
                                              : CONTEXT_FIELD_BARRETT_FACTOR_OFFSET);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        Uint64 a[4], b[4], alias_a[4], alias_b[4], result[4];
        big_endian_bytes_to_uint256(inputs + i * 64, a);
        big_endian_bytes_to_uint256(inputs + i * 64 + 32, b);
        copy_uint256(alias_a, a);
        copy_uint256(alias_b, b);
        montgomery_multiply(a, b, modulus, factor, result);
        montgomery_multiply(alias_a, b, modulus, factor, alias_a);
        montgomery_multiply(a, alias_b, modulus, factor, alias_b);
        uint256_to_big_endian_bytes(result, output + i * 96);
        uint256_to_big_endian_bytes(alias_a, output + i * 96 + 32);
        uint256_to_big_endian_bytes(alias_b, output + i * 96 + 64);
    }
}

extern "C" __global__ void crypto_test_hashes(
        const Byte *data, const Uint64 *offsets, const Uint64 *lengths,
        const Byte *keys, Byte *output, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        const Byte *message = data + offsets[i];
        Uint64 length = lengths[i];
        Byte *record = output + i * 192;
        sha256_digest(message, length, record);

        // Fragmented and zero-length updates must agree with one-shot hashing.
        // Per-vector fragment sizes exercise partial SHA blocks/BLAKE3 chunks.
        Sha256State sha;
        Blake3StreamingState b3;
        sha256_initialize(&sha);
        blake3_stream_initialize(&b3);
        sha256_update(&sha, message, 0);
        blake3_stream_update(&b3, message, 0);
        Uint64 step = 1 + (i * 67 % 2053);
        for (Uint64 cursor = 0; cursor < length; cursor += step) {
            Uint64 amount = length - cursor;
            if (amount > step) amount = step;
            sha256_update(&sha, message + cursor, amount);
            blake3_stream_update(&b3, message + cursor, amount);
        }
        sha256_finalize(&sha, record + 32);
        blake3_stream_finalize(&b3, record + 64);
        hmac_sha256(keys + i * 32, message, (int)length, record + 96);

        // RFC 6979 aliases the HMAC output with K or V. Testing only disjoint
        // buffers would miss a destructive-write regression in that path.
        Byte alias_key[32], alias_message[32];
        for (int j = 0; j < 32; j++) {
            alias_key[j] = keys[i * 32 + j];
            alias_message[j] = record[j];
        }
        hmac_sha256(alias_key, message, (int)length, alias_key);
        hmac_sha256(keys + i * 32, alias_message, 32, alias_message);
        for (int j = 0; j < 32; j++) {
            record[128 + j] = alias_key[j];
            record[160 + j] = alias_message[j];
        }
    }
}

extern "C" __global__ void crypto_test_arithmetic(
        const Uint64 *context, const Byte *inputs, Byte *output, int count,
        int use_order) {
    const Uint64 *modulus = context + (use_order ? CONTEXT_GROUP_ORDER_OFFSET
                                               : CONTEXT_FIELD_PRIME_OFFSET);
    const Uint64 *factor = context + (use_order ? CONTEXT_ORDER_BARRETT_FACTOR_OFFSET
                                              : CONTEXT_FIELD_BARRETT_FACTOR_OFFSET);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        Uint64 a[4], b[4], value[4];
        big_endian_bytes_to_uint256(inputs + i * 64, a);
        big_endian_bytes_to_uint256(inputs + i * 64 + 32, b);
        modular_add(a, b, modulus, value);
        uint256_to_big_endian_bytes(value, output + i * 128);
        modular_subtract(a, b, modulus, value);
        uint256_to_big_endian_bytes(value, output + i * 128 + 32);
        modular_multiply(a, b, modulus, factor, value);
        uint256_to_big_endian_bytes(value, output + i * 128 + 64);
        // Zero has no inverse; it is an explicit sentinel, never an oracle
        // claim that inversion of zero is mathematically defined.
        set_uint256(value, 0);
        if (constant_time_uint256_is_nonzero(a)) {
            modular_inverse(a, modulus, factor, value);
        }
        uint256_to_big_endian_bytes(value, output + i * 128 + 96);
    }
}

extern "C" __global__ void crypto_test_signatures(
        const Uint64 *context, const Byte *inputs, Byte *output, int count) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        // Only caller-supplied PUBLIC TEST scalars, never g_private_scalar_*.
        Uint64 scalar[4], nonce[4], r[4], s[4], x[4], y[4];
        big_endian_bytes_to_uint256(inputs + i * 64, scalar);
        const Byte *digest = inputs + i * 64 + 32;
        Byte *record = output + i * 168;
        int nonce_status = derive_rfc6979_nonce(
                context + CONTEXT_GROUP_ORDER_OFFSET, scalar, digest, nonce);
        int sign_status = sign_ecdsa_p256(context, scalar, digest, r, s);
        // Initialize outputs even on error so failure cannot pass by reading
        // leftover bytes from a preceding vector or allocation.
        for (int j = 0; j < 168; j++) record[j] = 0;
        ((int *)(record + 160))[0] = nonce_status;
        ((int *)(record + 160))[1] = sign_status;
        if (nonce_status != 0 || sign_status != 0) continue;
        multiply_generator_by_scalar(context, scalar, x, y);
        uint256_to_big_endian_bytes(nonce, record);
        uint256_to_big_endian_bytes(r, record + 32);
        uint256_to_big_endian_bytes(s, record + 64);
        uint256_to_big_endian_bytes(x, record + 96);
        uint256_to_big_endian_bytes(y, record + 128);
    }
}

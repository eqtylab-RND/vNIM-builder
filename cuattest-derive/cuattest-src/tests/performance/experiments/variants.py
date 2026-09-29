# SPDX-License-Identifier: Apache-2.0
"""Frozen 2026-09-09 experiments against 4708698, never loaded implicitly.

These include REJECTED candidates, not alternative production algorithms.
In particular writer_cached failed real GPT-2 receipt validation on sm_75.
"""

import hashlib
import re

BASELINE_SHA256 = "2f6240892c46648a5d13600b5ebf0ed5d09361ed6afb590bbb2625fafa7ee7ab"


def require_baseline(source):
    # Source-aware CUBIN lookup correctly ignores binaries for another source.
    # Without this guard, running from an edited tree silently benchmarks a new
    # build under an old baseline label. Export the pinned source explicitly.
    if hashlib.sha256(source.encode()).hexdigest() != BASELINE_SHA256:
        raise ValueError(
            "experiments require the pristine 4708698 CUDA source; set CUATTEST_KERNEL_SRC"
        )
    return source


P = 2**256 - 2**224 + 2**192 + 2**96 - 1
N = int("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551", 16)
NAMES = (
    "baseline",
    "unrolled",
    "karatsuba",
    "square",
    "prime_reduce",
    "chains",
    "mont_inverse",
    "prime_square_chains",
    "prime_square_mont",
    "prime_chains",
    "prime_mont",
    "limbs32",
    "prime_fast_chains",
    "prime_fast_mont",
    "prime_fast_limbs32_mont",
)
HASH_NAMES = (
    "hash_warp_lookup",
    "hash_ldg",
    "hash_single_chain",
    "hash_blocks8",
    "hash_blocks12",
)
TABLE_NAMES = ("table2", "table4", "table8", "table16", "stages", "writer_cached")


def writer_transform(source):
    return replace_body(
        source,
        "writer_append_bytes",
        r"""
    // BufferWriter metadata is disjoint from its source and destination byte
    // ranges. Cache it so alias conservatism does not spill/reload length for
    // every output byte. Preserve forward-copy and exact-full overflow rules.
    const int initial_length = writer->length;
    const int capacity = writer->capacity;
    Byte *destination = writer->destination;
    int writable = 0;
    if (byte_count > 0 && initial_length < capacity) {
        const int available = capacity - initial_length;
        writable = byte_count < available ? byte_count : available;
    }
    for (int byte_index = 0; byte_index < writable; ++byte_index) {
        destination[initial_length + byte_index] = source[byte_index];
    }
    if (byte_count > 0) writer->length = initial_length + byte_count;
    if (writer->length >= capacity) writer->overflowed = 1;
""",
    )


def stage_transform(source):
    source = before(
        source,
        "// ===== MULTIPRECISION ARITHMETIC AND P-256 ECDSA =====",
        'extern "C" { __device__ unsigned long long trial_stage_ticks[16]; }',
    )
    begin, end = body_range(source, "sign_ecdsa_p256")
    body = source[begin:end]
    body = "\nif(threadIdx.x==0) trial_stage_ticks[4]=clock64();\n" + body
    for marker, slot in (
        ("    // Compute R =", 5),
        ("    // Decode z from", 6),
        ("    // Compute s =", 7),
        ("    // Return both signature scalars", 8),
    ):
        assert body.count(marker) == 1
        body = body.replace(
            marker,
            f"    if(threadIdx.x==0) trial_stage_ticks[{slot}]=clock64();\n" + marker,
        )
    source = source[:begin] + body + source[end:]
    marker = "    // Convert (X:Y:Z) to affine"
    assert source.count(marker) == 1
    source = source.replace(
        marker, "    if(threadIdx.x==0) trial_stage_ticks[9]=clock64();\n" + marker
    )
    begin, end = body_range(source, "attest_measured_kernel")
    body = source[begin:end]
    body = "\nif(threadIdx.x==0) trial_stage_ticks[0]=clock64();\n" + body
    for marker, slot in (
        ("    // Publish the prepared hashes", 1),
        ("    // Thread zero validates all signatures", 2),
    ):
        assert body.count(marker) == 1
        body = body.replace(
            marker,
            f"    if(threadIdx.x==0) trial_stage_ticks[{slot}]=clock64();\n" + marker,
        )
    body += "\nif(threadIdx.x==0) trial_stage_ticks[3]=clock64();\n"
    return source[:begin] + body + source[end:]


def table_transform(source, name):
    assert name in TABLE_NAMES
    if name == "stages":
        return stage_transform(source)
    if name == "writer_cached":
        return writer_transform(source)
    width = int(name.removeprefix("table"))

    def once(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    once("Uint64 affine_y[4]) {", "Uint64 affine_y[4], int table_width = 1) {")
    once("Uint64 output_s[4]) {", "Uint64 output_s[4], int table_width = 1) {")
    once(
        "multiply_generator_by_scalar(p256_context, nonce, nonce_point_x, nonce_point_y);",
        "multiply_generator_by_scalar(p256_context, nonce, nonce_point_x, nonce_point_y, table_width);",
    )
    once(
        "for (int candidate = 1; candidate <= P256_WINDOW_POINTS; candidate++) {",
        """// Each group has identical signing inputs. Partition the PUBLIC
        // candidate scan, never the secret digit or a secret-indexed address.
        for (int candidate = 1 + ((int)threadIdx.x & (table_width-1));
             candidate <= P256_WINDOW_POINTS; candidate += table_width) {""",
    )
    before_add = "        // Always add one selected point; adding infinity handles a zero digit without a branch."
    reduction = """        if (table_width > 1) {
            unsigned group_base = ((unsigned)threadIdx.x & 31u) & ~(unsigned)(table_width-1);
            unsigned group_mask = ((1u << table_width)-1u) << group_base;
            // Reconverge the complete group after its public, uneven 63-point
            // scan. XOR-shuffle OR reduction gives every replica the same point.
#pragma unroll
            for (int offset=table_width/2; offset>0; offset>>=1) {
#pragma unroll
                for (int limb=0; limb<4; ++limb) {
                    selected_x[limb] |= __shfl_xor_sync(group_mask, selected_x[limb], offset, table_width);
                    selected_y[limb] |= __shfl_xor_sync(group_mask, selected_y[limb], offset, table_width);
                    selected_z[limb] |= __shfl_xor_sync(group_mask, selected_z[limb], offset, table_width);
                }
            }
        }

"""
    once(before_add, reduction + before_add)
    once(
        "if (global_thread_index < 4 && preparation_status == 0) {",
        f"if (global_thread_index < {4 * width} && preparation_status == 0) {{",
    )
    once(
        "int signature_index = (int)global_thread_index;",
        f"int signature_index = (int)global_thread_index / {width};",
    )
    once(
        "                                             signature_r, signature_s);",
        f"                                             signature_r, signature_s, {width});",
    )
    once(
        "        signature_statuses[signature_index] = signing_status;",
        f"        if (((int)global_thread_index & {width - 1}) == 0)\n            signature_statuses[signature_index] = signing_status;",
    )
    once(
        "        if (signing_status == 0) {",
        f"        if (signing_status == 0 && ((int)global_thread_index & {width - 1}) == 0) {{",
    )
    return source


def hash_transform(source, name):
    assert name in HASH_NAMES
    if name == "hash_warp_lookup":
        old = "tensor_index = find_tensor_for_tile(tensor_spans, tensor_count, flattened_tile_index);"
        new = """// The tile index and existence predicate are uniform within each full warp.
            // Only the lookup is shared; every lane keeps its own semantic chunk counter.
            if ((threadIdx.x & 31) == 0)
                tensor_index = find_tensor_for_tile(tensor_spans, tensor_count, flattened_tile_index);
            tensor_index = __shfl_sync(0xffffffffu, tensor_index, 0);"""
        assert source.count(old) == 1
        return source.replace(old, new)
    if name == "hash_ldg":
        # Only aligned uint4 reads use the read-only cache; partial/unaligned
        # byte paths and all output stores remain unchanged.
        return re.sub(r"(uint4 \w+ = )(\w+)\[([0-3])\];", r"\1__ldg(\2 + \3);", source)
    if name == "hash_single_chain":
        calls = []
        for which in ("first", "second"):
            outputs = ",".join(f"{which}_output_{i}" for i in range(8))
            calls.append(
                f"blake3_hash_aligned_full_chunk({which}_chunk_bytes, {which}_chunk_counter, {which}_chunk_is_root, {outputs});"
            )
        return replace_body(
            source, "blake3_hash_two_aligned_full_chunks", "\n".join(calls)
        )
    blocks = 8 if name == "hash_blocks8" else 12
    old = 'extern "C" __global__ void\nmeasure_model_fused_kernel('
    assert source.count(old) == 1
    return source.replace(
        old,
        f'extern "C" __global__ __launch_bounds__(128, {blocks}) void\nmeasure_model_fused_kernel(',
    )


def body_range(source, name):
    begin = source.index("{", source.index(name + "("))
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return begin + 1, end - 1


def replace_body(source, name, body):
    begin, end = body_range(source, name)
    return source[:begin] + "\n" + body + "\n" + source[end:]


def before(source, marker, extra):
    assert source.count(marker) == 1, marker
    return source.replace(marker, extra + "\n" + marker)


def limbs(value):
    return ", ".join(f"0x{(value >> (64 * i)) & (2**64 - 1):016x}ull" for i in range(4))


def matches(value):
    return " && ".join(
        f"modulus[{i}] == 0x{(value >> (64 * i)) & (2**64 - 1):016x}ull"
        for i in range(4)
    )


# Each tuple computes dst = src^(2^squares) * factor. Indices and iteration
# counts describe PUBLIC exponents, never the secret base or scalar.
def chain_plan(modulus):
    if modulus == P:
        return [
            ("x2", "base", 1, "base"),
            ("x4", "x2", 2, "x2"),
            ("x6", "x4", 2, "x2"),
            ("x12", "x6", 6, "x6"),
            ("x24", "x12", 12, "x12"),
            ("x30", "x24", 6, "x6"),
            ("x32", "x30", 2, "x2"),
            ("r", "x32", 32, "base"),
            ("r", "r", 128, "x32"),
            ("r", "r", 32, "x32"),
            ("r", "r", 30, "x30"),
            ("r", "r", 2, "base"),
        ]
    # Public four-bit sliding-window chain for the group order. Keep all
    # references statically named so there is no secret-indexed power table.
    plan = [("a2", "base", 1, None)]
    for odd in range(3, 16, 2):
        plan.append((f"a{odd}", "base" if odd == 3 else f"a{odd - 2}", 0, "a2"))
    bits = bin(modulus - 2)[2:]
    cursor = 0
    first = True
    while cursor < len(bits):
        if bits[cursor] == "0":
            end = cursor
            while end < len(bits) and bits[end] == "0":
                end += 1
            plan.append(("r", "r", end - cursor, None))
        else:
            end = min(cursor + 4, len(bits))
            while bits[end - 1] == "0":
                end -= 1
            digit = int(bits[cursor:end], 2)
            factor = "base" if digit == 1 else f"a{digit}"
            plan.append(
                (
                    "r",
                    factor if first else "r",
                    0 if first else end - cursor,
                    None if first else factor,
                )
            )
            first = False
        cursor = end
    return plan


def validate_chain(modulus):
    powers = {"base": 1}
    squares = multiplies = 0
    for dst, src, count, factor in chain_plan(modulus):
        powers[dst] = (powers[src] << count) + (powers[factor] if factor else 0)
        squares += count
        multiplies += factor is not None
    assert powers["r"] == modulus - 2
    return {"squares": squares, "multiplies": multiplies}


def chain_code(modulus, mont=False, fast=False):
    validate_chain(modulus)
    mul = (
        "trial_montgomery"
        if mont
        else "trial_prime_multiply"
        if fast
        else "modular_multiply"
    )
    code = []
    declared = {"base"}
    if mont:
        code += [
            f"Uint64 r2[4] = {{{limbs(pow(2, 512, modulus))}}};",
            "Uint64 one[4] = {1,0,0,0}, base_mont[4];",
            "trial_montgomery(value, r2, modulus, barrett_factor, base_mont);",
            "const Uint64 *base = base_mont;",
        ]
    else:
        code += ["const Uint64 *base = value;"]
    for dst, src, count, factor in chain_plan(modulus):
        if dst not in declared:
            code.append(f"Uint64 {dst}[4];")
            declared.add(dst)
        if dst != src:
            code.append(f"copy_uint256({dst}, {src});")
        if count:
            sq = (
                f"trial_montgomery({dst}, {dst}, modulus, barrett_factor, {dst});"
                if mont
                else f"{'trial_prime_square' if fast else 'trial_square'}({dst}, modulus, barrett_factor, {dst});"
            )
            code += [
                "#pragma unroll 1",
                f"for (int repeat=0; repeat<{count}; ++repeat) {{ {sq} }}",
            ]
        if factor:
            code.append(f"{mul}({dst}, {factor}, modulus, barrett_factor, {dst});")
    code.append(
        "trial_montgomery(r, one, modulus, barrett_factor, output);"
        if mont
        else "copy_uint256(output, r);"
    )
    return "\n".join(code)


KARATSUBA = r"""
// One Karatsuba level for four 64-bit limbs. Both 128-bit differences are
// made absolute without secret branches. The middle coefficient is 257 bits.
__device__ static void trial_karatsuba4(const Uint64 *a, const Uint64 *b, Uint64 *out) {
    Uint64 ad[2], bd[2], sa=0, sb=0;
#pragma unroll
    for (int i=0;i<2;++i) {
        Uint64 t=a[i+2]-a[i], r=t-sa;
        sa=(a[i+2]<a[i])+(t<sa); ad[i]=r;
        t=b[i+2]-b[i]; r=t-sb;
        sb=(b[i+2]<b[i])+(t<sb); bd[i]=r;
    }
    Uint64 ca=sa, cb=sb, ma=0-sa, mb=0-sb;
#pragma unroll
    for (int i=0;i<2;++i) {
        Uint64 t=ad[i]^ma, r=t+ca; ca=(r<t); ad[i]=r;
        t=bd[i]^mb; r=t+cb; cb=(r<t); bd[i]=r;
    }
    Uint64 lo[4], hi[4], d[4], mid[5];
    trial_schoolbook<2,2>(a,b,lo);
    trial_schoolbook<2,2>(a+2,b+2,hi);
    trial_schoolbook<2,2>(ad,bd,d);
    Uint64 carry=0;
#pragma unroll
    for (int i=0;i<4;++i) {
        Uint64 t=lo[i]+hi[i], c=(t<lo[i]);
        Uint64 r=t+carry; carry=c+(r<t); mid[i]=r;
        out[i]=lo[i]; out[i+4]=hi[i];
    }
    mid[4]=carry;
    Uint64 plus_carry=0, borrow=0, mask=0-(sa^sb);
#pragma unroll
    for (int i=0;i<5;++i) {
        Uint64 v=(i<4)?d[i]:0, t=mid[i]+v, c=(t<mid[i]);
        Uint64 added=t+plus_carry; plus_carry=c+(added<t);
        t=mid[i]-v; c=(mid[i]<v);
        Uint64 subtracted=t-borrow; borrow=c+(t<borrow);
        mid[i]=(added&mask)|(subtracted&~mask);
    }
    carry=0;
#pragma unroll
    for(int i=0;i<5;++i) {
        Uint64 t=out[i+2]+mid[i], c=(t<out[i+2]);
        Uint64 r=t+carry; carry=c+(r<t); out[i+2]=r;
    }
    out[7]+=carry;
}
"""

SQUARE = r"""
    // Comba columns with a three-limb accumulator. Off-diagonal products are
    // computed once, then added twice; no 129-bit doubled product is truncated.
    Uint64 product[8], c0=0, c1=0, c2=0;
#pragma unroll
    for (int column=0;column<7;++column) {
#pragma unroll
        for (int i=0;i<4;++i) {
            int j=column-i;
            if(j>=i && j<4) {
                Uint64 lo=value[i]*value[j], hi=__umul64hi(value[i],value[j]);
#pragma unroll
                for(int twice=0;twice<(i==j?1:2);++twice) {
                    Uint64 t=c0+lo, carry=(t<c0); c0=t;
                    t=c1+hi; Uint64 extra=(t<c1); c1=t+carry;
                    c2+=extra+(c1<t);
                }
            }
        }
        product[column]=c0; c0=c1; c1=c2; c2=0;
    }
    product[7]=c0;
    barrett_reduce(product, modulus, barrett_factor, output);
"""

PRIME_REDUCE = r"""
__device__ static void trial_prime_reduce(const Uint64 product[8],
                                         const Uint64 modulus[4], Uint64 output[4]) {
    // B=2^32: B^8 == B^7-B^6-B^3+1 (mod P-256). Signed coefficients remain
    // well within 64 bits. Four fixed normalization passes cover worst carries;
    // no carry-dependent loop or modulus lookup depends on secret input.
    long long w[16];
#pragma unroll
    for(int i=0;i<16;++i) w[i]=(Uint32)(product[i/2] >> (32*(i%2)));
#pragma unroll
    for(int i=15;i>=8;--i) {
        long long v=w[i]; w[i]=0;
        w[i-1]+=v; w[i-2]-=v; w[i-5]-=v; w[i-8]+=v;
    }
#pragma unroll
    for(int pass=0;pass<4;++pass) {
        long long carry=0;
#pragma unroll
        for(int i=0;i<8;++i) {
            long long t=w[i]+carry;
            w[i]=(Uint32)t; carry=t>>32;
        }
        w[0]+=carry; w[3]-=carry; w[6]-=carry; w[7]+=carry;
    }
    Uint64 r[4];
#pragma unroll
    for(int i=0;i<4;++i) r[i]=(Uint64)(Uint32)w[2*i] | ((Uint64)(Uint32)w[2*i+1]<<32);
    reduce_uint256_once(r, modulus, output);
}
"""

MONTGOMERY = r"""
// 32-bit word-by-word Montgomery reduction; inputs/outputs are in the
// Montgomery domain. Conversion is paid once around a complete inversion.
__device__ static void trial_montgomery(const Uint64 left[4], const Uint64 right[4],
        const Uint64 modulus[4], const Uint64 *, Uint64 output[4]) {
    Uint32 a[8], b[8], m[8], t[17]={};
#pragma unroll
    for(int i=0;i<8;++i) {
        a[i]=(Uint32)(left[i/2]>>(32*(i%2)));
        b[i]=(Uint32)(right[i/2]>>(32*(i%2)));
        m[i]=(Uint32)(modulus[i/2]>>(32*(i%2)));
    }
    Uint32 inv=1;
#pragma unroll
    for(int i=0;i<5;++i) inv*=2u-m[0]*inv;
    inv=0u-inv;
#pragma unroll
    for(int i=0;i<8;++i) {
        Uint64 carry=0;
#pragma unroll
        for(int j=0;j<8;++j) {
            Uint64 v=(Uint64)a[i]*b[j]+t[i+j]+carry;
            t[i+j]=(Uint32)v; carry=v>>32;
        }
        t[i+8]=(Uint32)carry;
    }
#pragma unroll
    for(int i=0;i<8;++i) {
        Uint32 q=t[i]*inv;
        Uint64 carry=0;
#pragma unroll
        for(int j=0;j<8;++j) {
            Uint64 v=(Uint64)q*m[j]+t[i+j]+carry;
            t[i+j]=(Uint32)v; carry=v>>32;
        }
#pragma unroll
        for(int k=i+8;k<17;++k) {
            Uint64 v=(Uint64)t[k]+carry; t[k]=(Uint32)v; carry=v>>32;
        }
    }
    Uint64 r[5], extended[5], reduced[5];
#pragma unroll
    for(int i=0;i<4;++i) {r[i]=(Uint64)t[8+2*i]|((Uint64)t[9+2*i]<<32);extended[i]=modulus[i];}
    r[4]=t[16]; extended[4]=0;
    Uint64 borrow=subtract_limbs(r,extended,5,reduced);
    constant_time_select(output,reduced,r,4,constant_time_mask(borrow^1ull));
}
"""

LIMBS32 = r"""
template<int L, int R> __device__ __forceinline__ void trial_schoolbook32(
        const Uint64 *left, const Uint64 *right, Uint64 *product) {
    Uint32 a[2*L], b[2*R], t[2*(L+R)]={};
#pragma unroll
    for(int i=0;i<2*L;++i) a[i]=(Uint32)(left[i/2]>>(32*(i%2)));
#pragma unroll
    for(int i=0;i<2*R;++i) b[i]=(Uint32)(right[i/2]>>(32*(i%2)));
#pragma unroll
    for(int i=0;i<2*L;++i) {
        Uint64 carry=0;
#pragma unroll
        for(int j=0;j<2*R;++j) {
            // (B-1)^2 + (B-1) + (B-1) == B^2-1, so no 64-bit carry is lost.
            Uint64 value=(Uint64)a[i]*b[j]+t[i+j]+carry;
            t[i+j]=(Uint32)value; carry=value>>32;
        }
        t[i+2*R]=(Uint32)carry;
    }
#pragma unroll
    for(int i=0;i<L+R;++i) product[i]=(Uint64)t[2*i]|((Uint64)t[2*i+1]<<32);
}
"""

PRIME_DIRECT = r"""
__device__ static void trial_prime_multiply(const Uint64 left[4], const Uint64 right[4],
        const Uint64 modulus[4], const Uint64 *, Uint64 output[4]) {
    Uint64 product[8];
    multiply_limbs(left,4,right,4,product);
    trial_prime_reduce(product,modulus,output);
}
__device__ static void trial_prime_square(const Uint64 value[4], const Uint64 modulus[4],
        const Uint64 *factor, Uint64 output[4]) {
    trial_prime_multiply(value,value,modulus,factor,output);
}
"""


def transform(source, name):
    assert name in NAMES, name
    if "limbs32" in name:
        source = before(source, "// Multiply two arbitrary-length", LIMBS32)
        begin, end = body_range(source, "multiply_limbs")
        dispatch = "\n".join(
            f"if(left_limb_count=={l} && right_limb_count=={r}) {{trial_schoolbook32<{l},{r}>(left,right,product); return;}}"
            for l, r in ((4, 4), (5, 5), (5, 4))
        )
        source = source[:begin] + "\n" + dispatch + source[begin:]
    if name in {"unrolled", "karatsuba"}:
        begin, end = body_range(source, "multiply_limbs")
        body = (
            source[begin:end]
            .replace("left_limb_count", "L")
            .replace("right_limb_count", "R")
        )
        body = re.sub(r"(?m)^(\s*)for \(", r"\1#pragma unroll\n\1for (", body)
        helper = (
            "template<int L,int R> __device__ __forceinline__ void trial_schoolbook("
            "const Uint64 *left,const Uint64 *right,Uint64 *product) {" + body + "}\n"
        )
        source = before(
            source,
            "// Multiply two arbitrary-length",
            helper + (KARATSUBA if name == "karatsuba" else ""),
        )
        begin, end = body_range(source, "multiply_limbs")
        dispatch = (
            "if(left_limb_count==4 && right_limb_count==4) {trial_karatsuba4(left,right,product); return;}\n"
            if name == "karatsuba"
            else "\n".join(
                f"if(left_limb_count=={l} && right_limb_count=={r}) {{trial_schoolbook<{l},{r}>(left,right,product); return;}}"
                for l, r in ((4, 4), (5, 5), (5, 4))
            )
        )
        source = source[:begin] + "\n" + dispatch + source[begin:]
    if "prime" in name:
        reducer = (
            PRIME_REDUCE.replace("pass<4", "pass<3") if "fast" in name else PRIME_REDUCE
        )
        source = before(source, "// Reduce a 512-bit product", reducer)
        begin, end = body_range(source, "barrett_reduce")
        source = (
            source[:begin]
            + f"\nif({matches(P)}) {{ trial_prime_reduce(product,modulus,output); return; }}\n"
            + source[begin:]
        )
    sqbody = (
        SQUARE
        if "square" in name
        else "modular_multiply(value,value,modulus,barrett_factor,output);"
    )
    square = (
        "__device__ static void trial_square(const Uint64 value[4],const Uint64 modulus[4],const Uint64 barrett_factor[5],Uint64 output[4]) {\n"
        + sqbody
        + "\n}\n"
    )
    source = before(source, "// Add two reduced 256-bit values", square)
    if "fast" in name:
        source = before(source, "// Add two reduced 256-bit values", PRIME_DIRECT)
        source = replace_body(
            source,
            "field_multiply",
            "trial_prime_multiply(left,right,field.prime,field.barrett_factor,output);",
        )
        source = replace_body(
            source,
            "field_square",
            "trial_prime_square(value,field.prime,field.barrett_factor,output);",
        )
    source = source.replace(
        "modular_multiply(current_power, current_power, modulus, barrett_factor, squared_power);",
        "trial_square(current_power, modulus, barrett_factor, squared_power);",
    )
    source = source.replace(
        "modular_multiply(value, value, field.prime, field.barrett_factor, output);",
        "trial_square(value, field.prime, field.barrett_factor, output);",
    )
    if "chains" in name or "mont" in name:
        mont = "mont" in name
        if mont:
            source = before(source, "// Invert a nonzero field element", MONTGOMERY)
        begin, end = body_range(source, "modular_inverse")
        original = source[begin:end]
        code = "\n".join(
            f"if({matches(m)}) {{\n{chain_code(m, mont and not ('fast' in name and m == P), 'fast' in name and m == P)}\nreturn;\n}}"
            for m in (P, N)
        )
        source = replace_body(source, "modular_inverse", code + original)
    return source

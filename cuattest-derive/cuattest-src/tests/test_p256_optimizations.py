# SPDX-License-Identifier: Apache-2.0
"""CPU proofs for the public schedules/bounds used by the CUDA fast paths.

These do not replace device differential tests. Reading the actual schedules
avoids validating an unrelated, duplicated Python addition-chain template.
"""

import re

import pytest

from cuattest.kernel import kernel_source_path

P = 2**256 - 2**224 + 2**192 + 2**96 - 1
N = int("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551", 16)


def brace_body(source, opening):
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[opening + 1:cursor - 1]


def function_body(name):
    source = kernel_source_path().read_text()
    return brace_body(source, source.index("{", source.index(name + "(")))


def inversion_branches():
    source = function_body("modular_inverse")
    branches = {}
    for match in re.finditer(r"\bif\s*\((modulus\[.*?\))\s*\{", source, re.S):
        limbs = re.findall(r"modulus\[(\d)\]\s*==\s*(0x[0-9a-f]+)ull", match[1])
        assert [int(i) for i, _ in limbs] == list(range(4))
        modulus = sum(int(value, 16) << (64 * int(i)) for i, value in limbs)
        branches[modulus] = brace_body(source, match.end() - 1)
    assert set(branches) == {P, N}
    return branches


@pytest.mark.parametrize("modulus", [P, N], ids=["field", "order"])
def test_cuda_inversion_chain_has_the_exact_public_exponent(modulus):
    source = inversion_branches()[modulus]
    operations = re.compile(
        r"(?P<square>#pragma\s+unroll\s+1\s+for\s*\(\s*int repeat\s*=\s*0;"
        r"\s*repeat\s*<\s*(?P<count>\d+);\s*\+\+repeat\s*\)\s*\{\s*"
        r"(?P<square_function>p256_modular_square|montgomery_multiply)\((?P<square_args>.*?)\);\s*\})"
        r"|(?P<copy>copy_uint256\((?P<copy_args>.*?)\);)"
        r"|(?P<multiply>(?:p256_modular_multiply|montgomery_multiply)\((?P<multiply_args>.*?)\);)",
        re.S,
    )
    powers = {"base": 1, "one": 0}
    squares = multiplies = conversions = 0
    for match in operations.finditer(source):
        if match["square"]:
            args = [v.strip() for v in match["square_args"].split(",")]
            assert args[0] == args[-1]
            if match["square_function"] == "montgomery_multiply":
                assert args[0] == args[1]
            count = int(match["count"])
            powers[args[-1]] = powers[args[0]] << count
            squares += count
        elif match["copy"]:
            destination, value = [v.strip() for v in match["copy_args"].split(",")]
            powers[destination] = powers[value]
        else:
            left, right, _, _, destination = [v.strip() for v in match["multiply_args"].split(",")]
            if (left, right, destination) == ("value", "r2", "base_mont"):
                assert modulus == N
                powers[destination] = 1
                conversions += 1
            else:
                powers[destination] = powers[left] + powers[right]
                if right == "one":
                    conversions += 1
                else:
                    multiplies += 1
    # p-3 (inverse-square), a missing final multiply, or an off-by-one run of
    # squarings cannot silently pass by testing only a=1 or a=p-1.
    assert powers["output"] == modulus - 2
    if modulus == P:
        assert (squares, multiplies, conversions) == (255, 12, 0)
    else:
        assert conversions == 2
        assert squares + multiplies < 320
        constants = re.search(r"Uint64 r2\[4\]\s*=\s*\{(.*?)\};", source, re.S)
        r2 = sum(int(v, 16) << (64 * i) for i, v in enumerate(
            re.findall(r"(0x[0-9a-f]+)ull", constants[1])
        ))
        assert r2 == pow(2, 512, N)


def test_sparse_prime_reduction_bounds_cover_every_512_bit_input():
    body = function_body("p256_reduce_product")
    folds = [(int(offset), 1 if sign == "+" else -1) for offset, sign in re.findall(
        r"w\[i\s*-\s*(\d+)\]\s*([+-])=\s*v;", body
    )]
    carry_folds = [(int(index), 1 if sign == "+" else -1) for index, sign in re.findall(
        r"w\[(\d+)\]\s*([+-])=\s*carry;", body
    )]
    assert len(folds) == len(carry_folds) == 4
    B = 2**32
    R = B**8
    h = R - P
    assert sum(sign * B**(8-offset) for offset, sign in folds) == h
    assert sum(sign * B**index for index, sign in carry_folds) == h
    assert re.search(r"pass\s*<\s*3", body)
    # Propagate the actual CUDA folding coefficients symbolically. Each
    # original word ranges independently over [0,B-1], so these are bounds
    # over ALL inputs, not a sample-based assertion about a carry loop.
    matrix = [[int(i == j) for j in range(16)] for i in range(16)]
    bound = B - 1
    for i in range(15, 7, -1):
        row = matrix[i]
        matrix[i] = [0] * 16
        for offset, sign in folds:
            matrix[i-offset] = [a + sign*b for a, b in zip(matrix[i-offset], row)]
            bound = max(bound, sum(abs(v) for v in matrix[i-offset]) * (B-1))
    coefficients = [sum(matrix[i][j] * B**i for i in range(8)) for j in range(16)]
    low = sum(min(0, v) * (B-1) for v in coefficients)
    high = sum(max(0, v) * (B-1) for v in coefficients)
    qmin, qmax = low // R, high // R
    assert (qmin, qmax) == (-4, 4)
    assert bound <= 9 * (B-1)
    assert bound + max(abs(qmin), abs(qmax)) + 32 < 2**63
    # f(z)=z%R+floor(z/R)*h. Its second result is in [0,R), so the third
    # normalization has carry zero, and one final subtraction is sufficient.
    assert qmin*h > -R and qmax*h < R
    assert R + (qmin-1)*h > 0 and (qmax+1)*h < R
    assert R < 2*P


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16])
def test_cooperative_table_scan_visits_every_public_candidate_once(width):
    partitions = [list(range(1 + lane, 64, width)) for lane in range(width)]
    assert sorted(v for part in partitions for v in part) == list(range(1, 64))
    for digit in range(64):
        values = [sum(1 << candidate for candidate in part if candidate == digit)
                  for part in partitions]
        offset = width // 2
        while offset:
            values = [v | values[lane ^ offset] for lane, v in enumerate(values)]
            offset //= 2
        assert values == [0 if digit == 0 else 1 << digit] * width

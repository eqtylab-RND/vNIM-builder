# SPDX-License-Identifier: Apache-2.0
"""Bound P-256 folding and exercise the exact CUDA reducer under CPU sanitizers."""

import argparse
from pathlib import Path
import subprocess

from cuattest.kernel import kernel_source
from variants import P, PRIME_REDUCE, limbs, require_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = require_baseline(kernel_source())
    args.output.mkdir(parents=True, exist_ok=True)
    B = 1 << 32
    R = B**8
    h = R - P
    matrix = [[int(i == j) for j in range(16)] for i in range(16)]
    maximum = B - 1
    for i in range(15, 7, -1):
        row = matrix[i]
        matrix[i] = [0] * 16
        for position, sign in ((i - 1, 1), (i - 2, -1), (i - 5, -1), (i - 8, 1)):
            matrix[position] = [a + sign * b for a, b in zip(matrix[position], row)]
            maximum = max(maximum, sum(abs(v) for v in matrix[position]) * (B - 1))
    coefficients = [sum(matrix[i][j] * B**i for i in range(8)) for j in range(16)]
    low = sum(min(0, v) * (B - 1) for v in coefficients)
    high = sum(max(0, v) * (B - 1) for v in coefficients)
    qmin, qmax = low // R, high // R
    # f(z) = z % R + (z // R) * h. After one fold its quotient is -1, 0 or 1.
    # In either nonzero case, the second fold is strictly within [0, R).
    assert qmin * h > -R and qmax * h < R
    assert R + (qmin - 1) * h > 0
    assert (qmax + 1) * h < R
    # One more pass normalizes all eight words with carry zero. Four passes
    # therefore suffice too. The final one-subtraction correction is valid.
    assert R < 2 * P
    assert maximum + max(abs(qmin), abs(qmax)) + 32 < 2**63
    print(
        {
            "initial_quotient_bounds": [qmin, qmax],
            "absolute_signed_intermediate_bound": maximum,
            "minimum_normalization_passes_proved": 3,
        },
        flush=True,
    )
    helpers = source[
        source.index("// Subtract equal-length") : source.index(
            "// Reduce a 512-bit product"
        )
    ]
    code = r"""
#include <boost/multiprecision/cpp_int.hpp>
#include <iostream>
#include <random>
#include <cassert>
using Uint64 = unsigned long long;
using Uint32 = unsigned int;
#define __device__
#define __forceinline__ inline
"""
    code += (
        helpers
        + PRIME_REDUCE
        + PRIME_REDUCE.replace("trial_prime_reduce", "trial_prime_reduce3").replace(
            "pass<4", "pass<3"
        )
    )
    code += f"\nstatic const Uint64 prime[4] = {{{limbs(P)}}};\n"
    code += r"""
using boost::multiprecision::cpp_int;
cpp_int decode(const Uint64 *words, int count) {
    cpp_int value=0;
    for(int i=count-1;i>=0;--i) { value <<= 64; value += words[i]; }
    return value;
}
int main() {
    const cpp_int p=decode(prime,4);
    std::mt19937_64 rng(0xc0a77e57);
    for(unsigned vector=0;vector<1065536;++vector) {
        Uint64 input[8], output[4], output3[4];
        for(int i=0;i<8;++i) {
            if(vector<65536) {
                Uint64 low=(vector & (1u<<(2*i))) ? 0xffffffffull : 0;
                Uint64 high=(vector & (1u<<(2*i+1))) ? 0xffffffffull : 0;
                input[i]=low | (high<<32);
            } else input[i]=rng();
        }
        trial_prime_reduce(input,prime,output);
        trial_prime_reduce3(input,prime,output3);
        const cpp_int expected=decode(input,8)%p;
        assert(decode(output,4)==expected);
        assert(decode(output3,4)==expected);
    }
    std::cout << "PASS: 65536 limb corners + 1000000 random 512-bit values, 3/4 passes\n";
}
"""
    target = args.output / "reducer.cpp"
    target.write_text(code)
    binary = args.output / "reducer"
    subprocess.run(
        [
            "clang++-21",
            "-std=c++17",
            "-O1",
            "-g",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
            str(target),
            "-o",
            str(binary),
        ],
        check=True,
    )
    subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()

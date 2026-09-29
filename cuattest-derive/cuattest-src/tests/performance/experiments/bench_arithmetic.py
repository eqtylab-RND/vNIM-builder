# SPDX-License-Identifier: Apache-2.0
"""Isolated, differential-checked GPU arithmetic experiments (public test data)."""

import argparse
from contextlib import ExitStack, closing
import ctypes as c
import hashlib
import json
from pathlib import Path
import random
import statistics

from variants import NAMES, P, N, transform, validate_chain, require_baseline
from cuattest._cuda import Cuda, DeviceBuffer
from cuattest._nvrtc import Nvrtc
from cuattest.kernel import kernel_source

ADAPTER = r"""
extern "C" __global__ void validate(const Uint64 *inputs,const Uint64 *modulus,
        const Uint64 *mu,Uint64 *out,int count) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) {
        const Uint64 *a=inputs+i*8,*b=a+4; Uint64 *r=out+i*20;
        multiply_limbs(a,4,b,4,r);
        modular_multiply(a,b,modulus,mu,r+8);
        trial_square(a,modulus,mu,r+12);
        modular_inverse(a,modulus,mu,r+16);
    }
}
extern "C" __global__ void chain(const Uint64 *inputs,const Uint64 *modulus,
        const Uint64 *mu,Uint64 *out,int mode,int iterations) {
    int i=threadIdx.x;
    if(i>=4) return;
    Uint64 a[4],b[4]; copy_uint256(a,inputs+i*8);copy_uint256(b,inputs+i*8+4);
#pragma unroll 1
    for(int repeat=0;repeat<iterations;++repeat) {
        if(mode==0) modular_multiply(a,b,modulus,mu,a);
        else if(mode==1) trial_square(a,modulus,mu,a);
        else modular_inverse(a,modulus,mu,a);
    }
    copy_uint256(out+i*4,a);
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=NAMES, action="append")
    parser.add_argument("--runs", type=int, default=11)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    source = require_baseline(kernel_source())
    args.output.mkdir(parents=True, exist_ok=True)
    cu = Cuda()
    cu.init()
    dev = cu.device(args.device)
    arch = "sm_" + "".join(map(str, cu.compute_capability(dev)))
    ctx = cu.ctx_create(dev)
    nvrtc = Nvrtc()
    results = {
        "arch": arch,
        "gpu": cu.device_name(dev),
        "compiler": nvrtc.version(),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "chains": {"field": validate_chain(P), "order": validate_chain(N)},
        "variants": {},
    }
    try:
        # Driver events measure only the owning stream's kernels, not uploads,
        # Python dispatch, compilation or independent result verification.
        stream = cu.stream_create()
        event_create = cu.lib.cuEventCreate
        event_create.argtypes = [c.POINTER(c.c_void_p), c.c_uint]
        event_record = cu.lib.cuEventRecord
        event_record.argtypes = [c.c_void_p, c.c_void_p]
        elapsed = cu.lib.cuEventElapsedTime
        elapsed.argtypes = [c.POINTER(c.c_float), c.c_void_p, c.c_void_p]
        destroy = cu.lib.cuEventDestroy_v2
        destroy.argtypes = [c.c_void_p]
        start, end = c.c_void_p(), c.c_void_p()
        cu.check(event_create(c.byref(start), 0), "event_create")
        cu.check(event_create(c.byref(end), 0), "event_create")
        for name in args.variant or NAMES:
            trial = transform(source, name)
            prefix = trial.split("// References to the P-256 base field", 1)[0]
            code = prefix + ADAPTER
            key = hashlib.sha256((code + arch + nvrtc.version()).encode()).hexdigest()
            target = args.output / (name + "-" + key + ".cubin")
            print(f"compiling {name}", flush=True)
            if target.exists():
                binary = target.read_bytes()
            else:
                binary = nvrtc.compile_cubin(code, arch)
                target.write_bytes(binary)
            (args.output / (name + ".cu")).write_text(trial)
            if args.compile_only:
                continue
            module = cu.module_load(binary)
            record = {}
            try:
                for label, modulus in (("field", P), ("order", N)):
                    rng = random.Random(99173)
                    edge = [0, 1, 2, modulus - 2, modulus - 1]
                    edge += [
                        ((1 << bit) + delta) % modulus
                        for bit in (32, 64, 128, 192, 255)
                        for delta in (-1, 0, 1)
                    ]
                    pairs = [(a, b) for a in edge for b in edge] + [
                        (rng.randrange(1, modulus), rng.randrange(1, modulus))
                        for _ in range(1024)
                    ]
                    # Nonzero first four operands make timed inversion chains meaningful.
                    pairs = pairs[-4:] + pairs
                    with ExitStack() as stack:

                        def upload(raw):
                            return stack.enter_context(
                                closing(DeviceBuffer.from_bytes(cu, raw))
                            )

                        data = upload(
                            b"".join(
                                a.to_bytes(32, "little") + b.to_bytes(32, "little")
                                for a, b in pairs
                            )
                        )
                        mod = upload(modulus.to_bytes(32, "little"))
                        mu = upload(((1 << 512) // modulus).to_bytes(40, "little"))
                        output = upload(bytes(len(pairs) * 160))
                        base = [c.c_uint64(b.ptr) for b in (data, mod, mu, output)]
                        cu.launch(
                            cu.function(module, "validate"),
                            32,
                            32,
                            [*base, c.c_int(len(pairs))],
                            stream=stream,
                        )
                        cu.stream_sync(stream)
                        raw = output.read()
                        for i, (a, b) in enumerate(pairs):
                            want = (
                                (a * b).to_bytes(64, "little")
                                + (a * b % modulus).to_bytes(32, "little")
                                + (a * a % modulus).to_bytes(32, "little")
                                + (pow(a, -1, modulus) if a else 0).to_bytes(
                                    32, "little"
                                )
                            )
                            assert raw[i * 160 : (i + 1) * 160] == want, (
                                name,
                                label,
                                i,
                                a,
                                b,
                            )
                        timings = {}
                        for mode, operation, repeats in (
                            (0, "mul", 256),
                            (1, "square", 256),
                            (2, "inverse", 2),
                        ):
                            samples = []
                            for sample in range(args.runs + 2):
                                cu.check(event_record(start, stream), "event_record")
                                cu.launch(
                                    cu.function(module, "chain"),
                                    1,
                                    32,
                                    [*base, c.c_int(mode), c.c_int(repeats)],
                                    stream=stream,
                                )
                                cu.check(event_record(end, stream), "event_record")
                                cu.stream_sync(stream)
                                ms = c.c_float()
                                cu.check(elapsed(c.byref(ms), start, end), "elapsed")
                                got = output.read(128)
                                for lane, (a, b) in enumerate(pairs[:4]):
                                    want = (
                                        a * pow(b, repeats, modulus) % modulus
                                        if mode == 0
                                        else pow(a, 1 << repeats, modulus)
                                        if mode == 1
                                        else a
                                    )
                                    assert (
                                        int.from_bytes(
                                            got[lane * 32 : (lane + 1) * 32], "little"
                                        )
                                        == want
                                    ), (name, operation, lane)
                                if sample >= 2:
                                    samples.append(ms.value / repeats)
                            timings[operation] = {
                                "samples_ms_per_operation": samples,
                                "median_ms": statistics.median(samples),
                            }
                        record[label] = {"vectors": len(pairs), "timings": timings}
                results["variants"][name] = record
                print(
                    name,
                    json.dumps(
                        {
                            label: {
                                op: round(t["median_ms"] * 1000, 3)
                                for op, t in rec["timings"].items()
                            }
                            for label, rec in record.items()
                        }
                    ),
                    flush=True,
                )
                (args.output / "results.json").write_text(
                    json.dumps(results, indent=2) + "\n"
                )
            finally:
                cu.module_unload(module)
        cu.check(destroy(start), "event_destroy")
        cu.check(destroy(end), "event_destroy")
        cu.stream_destroy(stream)
    finally:
        cu.ctx_destroy(ctx)


if __name__ == "__main__":
    main()

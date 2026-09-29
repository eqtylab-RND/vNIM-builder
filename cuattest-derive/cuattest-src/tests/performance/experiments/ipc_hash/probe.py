"""Independent BLAKE3 checks of complete, partial and unaligned GPU spans."""

import argparse
from contextlib import closing
import ctypes
import json
import random
import statistics
import time

from blake3 import blake3
from cuattest._cuda import DeviceBuffer
from cuattest.notary import Notary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--require-backend", choices=("native", "fallback"))
    args = parser.parse_args()
    rng = random.Random(20260909)
    lengths = sorted(
        {
            1,
            2,
            3,
            15,
            16,
            17,
            63,
            64,
            65,
            1023,
            1024,
            1025,
            2047,
            2048,
            2049,
            8191,
            8192,
            8193,
            131071,
            131072,
            131073,
            262143,
            262144,
            262145,
            524289,
            *[rng.randrange(1, 524288) for _ in range(20)],
        }
    )
    raw = rng.randbytes(max(lengths) + 32)
    with closing(Notary(device=args.device)) as notary:
        if args.require_backend is not None:
            actual_backend = "fallback" if notary._native_runner is None else "native"
            if actual_backend != args.require_backend:
                raise RuntimeError(
                    f"requested {args.require_backend}, got {actual_backend}"
                )
        with notary._activate():
            buffer = DeviceBuffer.from_bytes(notary.cu, raw)
            try:
                for offset in (0, 1, 7, 16):
                    for size in lengths:
                        actual = notary.hash_dptr(buffer.ptr + offset, size)
                        assert actual == blake3(raw[offset : offset + size]).digest(), (
                            offset,
                            size,
                        )
                spans = [
                    (buffer.ptr + (i % 17), size) for i, size in enumerate(lengths)
                ]
                roots = b"".join(
                    blake3(raw[i % 17 : i % 17 + size]).digest()
                    for i, size in enumerate(lengths)
                )
                result = notary._launch_fused_active(spans)
                assert result.roots == roots
                assert (
                    result.model_root
                    == blake3(len(spans).to_bytes(4, "little") + roots).digest()
                )
            finally:
                buffer.close()
            large = rng.randbytes(64 << 20)
            buffer = DeviceBuffer.from_bytes(notary.cu, large)
            try:
                expected = blake3(large).digest()
                seconds = []
                for i in range(args.runs + 3):
                    start = time.perf_counter()
                    actual = notary.hash_dptr(buffer.ptr, len(large))
                    elapsed = time.perf_counter() - start
                    assert actual == expected
                    if i >= 3:
                        seconds.append(elapsed)
                attrs = {}
                fn = notary._fn["measure_model_fused_kernel"]
                for name, attribute in (
                    ("registers", 4),
                    ("shared_bytes", 1),
                    ("local_bytes", 3),
                ):
                    value = ctypes.c_int()
                    notary.cu.check(
                        notary.cu.lib.cuFuncGetAttribute(
                            ctypes.byref(value), attribute, fn
                        ),
                        name,
                    )
                    attrs[name] = value.value
                print(
                    json.dumps(
                        dict(
                            info=notary.info.as_dict(),
                            attributes=attrs,
                            differential_cases=4 * len(lengths) + 1,
                            bytes=len(large),
                            samples_seconds=seconds,
                            median_seconds=statistics.median(seconds),
                        )
                    ),
                    flush=True,
                )
            finally:
                buffer.close()


if __name__ == "__main__":
    main()

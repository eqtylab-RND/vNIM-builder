# SPDX-License-Identifier: Apache-2.0
"""Raw CUDA-only control for duplicate IPC imports; no torch/cuAttest imports.

CUDA specifies one reference per successful open, released by one close. This
control distinguishes driver/sanitizer tracking from the application's lease
machinery. Compare --opens 1 and --opens 2 under memcheck, then uninstrumented.
"""

import argparse
import ctypes as c
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handle")
    parser.add_argument("--opens", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    library = c.CDLL("libcuda.so.1")

    class Handle(c.Structure):
        _fields_ = [("bytes", c.c_ubyte * 64)]

    def call(name, types, *values):
        function = getattr(library, name)
        function.argtypes, function.restype = types, c.c_int
        status = function(*values)
        if status:
            raise RuntimeError(f"{name}: CUDA status {status}")

    ptr, context = c.c_uint64(), c.c_void_p()
    call("cuInit", [c.c_uint], 0)
    call(
        "cuCtxCreate_v2",
        [c.POINTER(c.c_void_p), c.c_uint, c.c_int],
        c.byref(context),
        0,
        0,
    )
    try:
        if args.handle:
            handle = Handle.from_buffer_copy(bytes.fromhex(args.handle))
            for _ in range(args.opens):
                call(
                    "cuIpcOpenMemHandle_v2",
                    [c.POINTER(c.c_uint64), Handle, c.c_uint],
                    c.byref(ptr),
                    handle,
                    1,
                )
            for _ in range(args.opens - 1):
                call("cuIpcCloseMemHandle", [c.c_uint64], ptr)
            # A host copy alone does not exercise memcheck's kernel shadow
            # allocation table. Load a one-byte PTX copy, no crypto or NVRTC.
            ptx = b""".version 7.8
.target sm_75
.address_size 64
.visible .entry copy_byte(.param .u64 dst, .param .u64 src) {
.reg .b64 d, s;
.reg .b32 v;
ld.param.u64 d, [dst];
ld.param.u64 s, [src];
ld.global.u8 v, [s];
st.global.u8 [d], v;
ret;
}\n"""
            module, function, destination = c.c_void_p(), c.c_void_p(), c.c_uint64()
            call(
                "cuModuleLoadData",
                [c.POINTER(c.c_void_p), c.c_char_p],
                c.byref(module),
                ptx,
            )
            call(
                "cuModuleGetFunction",
                [c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p],
                c.byref(function),
                module,
                b"copy_byte",
            )
            call(
                "cuMemAlloc_v2",
                [c.POINTER(c.c_uint64), c.c_size_t],
                c.byref(destination),
                1,
            )
            parameters = (c.c_void_p * 2)(c.addressof(destination), c.addressof(ptr))
            call(
                "cuLaunchKernel",
                [c.c_void_p] + [c.c_uint] * 7 + [c.c_void_p, c.c_void_p, c.c_void_p],
                function,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                None,
                parameters,
                None,
            )
            call("cuCtxSynchronize", [])
            byte = c.c_ubyte()
            call(
                "cuMemcpyDtoH_v2",
                [c.c_void_p, c.c_uint64, c.c_size_t],
                c.byref(byte),
                destination,
                1,
            )
            assert byte.value == 0x5A
            call("cuMemFree_v2", [c.c_uint64], destination)
            call("cuModuleUnload", [c.c_void_p], module)
            output = c.create_string_buffer(4096)
            call(
                "cuMemcpyDtoH_v2",
                [c.c_void_p, c.c_uint64, c.c_size_t],
                output,
                ptr,
                4096,
            )
            assert output.raw == b"\x5a" * 4096
            call("cuIpcCloseMemHandle", [c.c_uint64], ptr)
            print("raw CUDA reference-counted IPC copy matches", flush=True)
            return 0
        call("cuMemAlloc_v2", [c.POINTER(c.c_uint64), c.c_size_t], c.byref(ptr), 4096)
        data = c.create_string_buffer(b"\x5a" * 4096)
        call("cuMemcpyHtoD_v2", [c.c_uint64, c.c_void_p, c.c_size_t], ptr, data, 4096)
        handle = Handle()
        call("cuIpcGetMemHandle", [c.POINTER(Handle), c.c_uint64], c.byref(handle), ptr)
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--handle",
                bytes(handle.bytes).hex(),
                "--opens",
                str(args.opens),
            ],
            check=False,
        )
        # subprocess completion proves the consumer process/context is gone.
        call("cuMemFree_v2", [c.c_uint64], ptr)
        return result.returncode
    finally:
        call("cuCtxDestroy_v2", [c.c_void_p], context)


if __name__ == "__main__":
    raise SystemExit(main())

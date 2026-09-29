# SPDX-License-Identifier: Apache-2.0
"""The IPC handle struct must carry bytes verbatim. No GPU required.

This is a regression test for a real failure: declaring the field as
``c_char * 64`` gives it string semantics, so ctypes truncates the assignment
at the first NUL. Real handles contain NULs, and the driver then rejects them
with a bare "invalid argument".
"""

import ctypes

import pytest

from cuattest._cuda import (
    IPC_HANDLE_BYTES,
    Cuda,
    CudaError,
    CUipcMemHandle,
    IpcImportRejectedError,
)

# The opening bytes of a handle observed from torch's _share_cuda_(); note the
# NUL in third position, which is what broke the c_char version.
REALISTIC = bytes([0x03, 0x63, 0x00, 0x0C, 0x90, 0x10, 0x00, 0x00]) + bytes(range(56))


def test_handle_round_trips_exactly():
    h = CUipcMemHandle.from_bytes(REALISTIC)
    assert h.to_bytes() == REALISTIC


def test_handle_memory_is_the_raw_bytes():
    """What the driver reads is the struct's memory, so check that directly."""
    h = CUipcMemHandle.from_bytes(REALISTIC)
    assert ctypes.string_at(ctypes.byref(h), IPC_HANDLE_BYTES) == REALISTIC


def test_embedded_nuls_survive():
    raw = bytes(64)                       # every byte a NUL
    assert CUipcMemHandle.from_bytes(raw).to_bytes() == raw
    raw2 = b"\x00" * 32 + b"\xff" * 32    # trailing data after a run of NULs
    assert CUipcMemHandle.from_bytes(raw2).to_bytes() == raw2


def test_wrong_length_is_rejected():
    with pytest.raises(CudaError, match="64 bytes"):
        CUipcMemHandle.from_bytes(b"\x01" * 63)


def test_struct_is_exactly_the_handle_size():
    assert ctypes.sizeof(CUipcMemHandle) == IPC_HANDLE_BYTES


def test_driver_rejection_is_distinct_from_an_interrupted_import():
    cuda = object.__new__(Cuda)
    cuda.cuIpcOpenMemHandle = lambda *args: 1
    cuda.cuGetErrorString = lambda *args: 0

    with pytest.raises(IpcImportRejectedError, match="cuIpcOpenMemHandle.*rc=1"):
        cuda.ipc_open(bytes(64))

    def interrupted(*args):
        raise CudaError("exception without a driver return code")

    cuda.cuIpcOpenMemHandle = interrupted
    with pytest.raises(CudaError) as caught:
        cuda.ipc_open(REALISTIC)
    assert not isinstance(caught.value, IpcImportRejectedError)

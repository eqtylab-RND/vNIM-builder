# SPDX-License-Identifier: Apache-2.0
"""Native CUDA-IPC launch cleanup tested against a deterministic fake driver."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

_FAKE_CUDA = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

typedef struct { unsigned char reserved[64]; } CUipcMemHandle;

static int launch_calls = 0;
static int synchronize_calls = 0;
static int close_calls = 0;
static int free_calls = 0;
static int export_calls = 0;
static int range_calls = 0;
static int host_free_calls = 0;
static int synchronized_after_launch = 0;
static uint64_t next_allocation = 0x100000;

int cuGetErrorString(int result, const char **text) {
    (void)result;
    *text = "injected fake-driver failure";
    return 0;
}
int cuMemAlloc_v2(uint64_t *pointer, size_t nbytes) {
    *pointer = next_allocation;
    next_allocation += (uint64_t)nbytes + 0x1000;
    return 0;
}
int cuMemFree_v2(uint64_t pointer) {
    (void)pointer; free_calls += 1; return 0;
}
int cuMemAllocHost_v2(void **ptr, size_t nbytes) {
    *ptr = malloc(nbytes); return *ptr ? 0 : 2;
}
int cuMemFreeHost(void *ptr) { free(ptr); host_free_calls += 1; return 0; }
int cuMemcpyHtoDAsync_v2(uint64_t dst, const void *src, size_t nbytes, void *stream) {
    assert(stream == (void *)6);
    (void)dst; (void)src; (void)nbytes; return 0;
}
int cuMemcpyDtoHAsync_v2(void *dst, uint64_t src, size_t nbytes, void *stream) {
    assert(stream == (void *)6);
    (void)src; memset(dst, 0, nbytes); return 0;
}
int cuLaunchCooperativeKernel(void *function,
                              unsigned gx, unsigned gy, unsigned gz,
                              unsigned bx, unsigned by, unsigned bz,
                              unsigned shared, void *stream, void **arguments) {
    (void)function; (void)gx; (void)gy; (void)gz;
    (void)bx; (void)by; (void)bz; (void)shared;
    assert(stream == (void *)6); (void)arguments;
    launch_calls += 1;
    synchronized_after_launch = 0;
    return launch_calls == 2 ? 701 : 0;
}
int cuCtxSynchronize(void) { abort(); } /* Never wait on unrelated streams. */
int cuStreamSynchronize(void *stream) {
    assert(stream == (void *)6);
    synchronize_calls += 1;
    if (getenv("CUATTEST_FAKE_SYNC_FAIL") != NULL) return 702;
    synchronized_after_launch = 1;
    return 0;
}
int cuIpcGetMemHandle(CUipcMemHandle *handle, uint64_t pointer) {
    (void)pointer;
    export_calls += 1;
    if (getenv("CUATTEST_FAKE_EXPORT_FAIL") != NULL) return 1;
    memset(handle, 0xab, sizeof(*handle)); return 0;
}
int cuIpcOpenMemHandle_v2(uint64_t *pointer, CUipcMemHandle handle,
                          unsigned flags) {
    (void)handle; (void)flags; *pointer = 0x4000; return 0;
}
int cuIpcCloseMemHandle(uint64_t pointer) {
    (void)pointer;
    close_calls += 1;
    return synchronized_after_launch ? 0 : 703;
}
int cuPointerGetAttribute(void *data, int attribute, uint64_t pointer) {
    (void)pointer;
    if (attribute == 10) { /* IS_LEGACY_CUDA_IPC_CAPABLE */
        if (getenv("CUATTEST_FAKE_ATTRIBUTE_FAIL") != NULL) return 700;
        *(int *)data = getenv("CUATTEST_FAKE_NONLEGACY") == NULL;
        return 0;
    }
    *(int *)data = 0; return 0;
}
int cuMemGetAddressRange_v2(uint64_t *base, size_t *nbytes, uint64_t pointer) {
    range_calls += 1;
    (void)pointer; *base = 0x4000; *nbytes = 0x1000; return 0;
}
int cuattestFakeLaunchCalls(void) { return launch_calls; }
int cuattestFakeSynchronizeCalls(void) { return synchronize_calls; }
int cuattestFakeCloseCalls(void) { return close_calls; }
int cuattestFakeFreeCalls(void) { return free_calls; }
int cuattestFakeExportCalls(void) { return export_calls; }
int cuattestFakeRangeCalls(void) { return range_calls; }
int cuattestFakeHostFreeCalls(void) { return host_free_calls; }
"""


@pytest.fixture(scope="module")
def fake_cuda_driver(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the native fake-driver test")
    directory = tmp_path_factory.mktemp("fake-cuda")
    source = directory / "fake_cuda.c"
    library = directory / "libcuda.so.1"
    source.write_text(_FAKE_CUDA)
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-Wl,-soname,libcuda.so.1",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return directory


def _run_fake_driver_case(fake_cuda_driver, *, sync_fails: bool) -> None:
    script = textwrap.dedent(
        f"""
        import ctypes
        from cuattest import _native

        driver = ctypes.CDLL("libcuda.so.1")
        handle, base, nbytes, device = _native._export_cuda_allocation(0x4080)
        assert handle == bytes([0xab]) * 64
        assert (base, nbytes, device) == (0x4000, 0x1000, 0)
        runner = _native.FusedRunner(1, 2, 1, 3, 4, 5, 0, 6)
        try:
            runner.run_ipc(
                [(bytes(range(64)), 8, 0, 0)],
                b"2026-09-05T00:00:00Z",
                b"model",
                bytes(32),
            )
        except _native.Error as error:
            assert type(error) is _native.Error, type(error)
            assert "attestation" in str(error), str(error)
        else:
            raise AssertionError("injected attestation launch failure was ignored")

        assert driver.cuattestFakeLaunchCalls() == 2
        assert driver.cuattestFakeSynchronizeCalls() == 1
        assert driver.cuattestFakeCloseCalls() == {0 if sync_fails else 1}
        assert runner.ipc_cleanup_failed is {sync_fails}
        runner.close()
        # A poisoned runner leaves all four retained workspaces to context
        # destruction; an ordinarily drained failure can free them directly.
        assert driver.cuattestFakeFreeCalls() == {0 if sync_fails else 4}
        assert driver.cuattestFakeHostFreeCalls() == {0 if sync_fails else 2}
        """
    )
    environment = os.environ.copy()
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(fake_cuda_driver)
        if not existing
        else f"{fake_cuda_driver}{os.pathsep}{existing}"
    )
    if sync_fails:
        environment["CUATTEST_FAKE_SYNC_FAIL"] = "1"
    else:
        environment.pop("CUATTEST_FAKE_SYNC_FAIL", None)
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_failed_followup_launch_is_drained_before_ipc_unmap(fake_cuda_driver):
    _run_fake_driver_case(fake_cuda_driver, sync_fails=False)


@pytest.mark.parametrize(
    "setting,operation,export_calls,range_calls",
    [
        ("NONLEGACY", "does not support legacy CUDA IPC", 0, 0),
        ("ATTRIBUTE_FAIL", "cuPointerGetAttribute(client export IPC capability)", 0, 0),
        ("EXPORT_FAIL", "cuIpcGetMemHandle(client export)", 1, 1),
    ],
)
def test_export_capability_is_checked_without_masking_driver_errors(
    fake_cuda_driver, setting, operation, export_calls, range_calls
):
    script = textwrap.dedent(
        f"""
        import ctypes
        from cuattest import _native

        driver = ctypes.CDLL("libcuda.so.1")
        try:
            _native._export_cuda_allocation(0x4080)
        except _native.Error as error:
            message = str(error)
            assert {operation!r} in message, message
            if {setting!r} == "NONLEGACY":
                assert "expandable_segments" in message
                assert "PYTORCH_CUDA_ALLOC_CONF" in message
                assert "backend:native,expandable_segments:False" in message
                assert "before" in message and "allocating" in message
                assert "existing" in message
            else:
                # An invalid pointer/context or unrelated export failure is
                # not evidence that expandable segments caused the failure.
                assert "expandable_segments" not in message, message
                assert "rc=" in message, message
        else:
            raise AssertionError("injected export failure was ignored")
        assert driver.cuattestFakeExportCalls() == {export_calls}
        assert driver.cuattestFakeRangeCalls() == {range_calls}
        """
    )
    environment = os.environ.copy()
    existing = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        str(fake_cuda_driver)
        if not existing
        else f"{fake_cuda_driver}{os.pathsep}{existing}"
    )
    environment[f"CUATTEST_FAKE_{setting}"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment,
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unsigned_native_output_omits_unwritten_length_slot(fake_cuda_driver):
    script = textwrap.dedent(
        """
        from cuattest import _native

        runner = _native.FusedRunner(1, 2, 1, 3, 4, 5, 0, 6)
        roots, model_root, receipt = runner.run_spans([(0x4080, 8)], None, None, bytes(32))
        assert (len(roots), len(model_root), receipt) == (32, 32, None)
        # 32-byte tensor root + 32-byte model root + initialized status. The
        # old 72-byte layout copied an unwritten four-byte receipt length.
        assert runner.capacities[3] == 68, runner.capacities
        runner.close()
        """
    )
    environment = os.environ.copy()
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(fake_cuda_driver)
        if not existing
        else f"{fake_cuda_driver}{os.pathsep}{existing}"
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_failed_drain_leaves_ipc_mappings_for_context_destruction(fake_cuda_driver):
    _run_fake_driver_case(fake_cuda_driver, sync_fails=True)


def test_failed_direct_span_drain_requires_context_destruction(fake_cuda_driver):
    script = textwrap.dedent(
        """
        import ctypes
        from cuattest import _native

        driver = ctypes.CDLL("libcuda.so.1")
        runner = _native.FusedRunner(1, 2, 1, 3, 4, 5, 0, 6)
        try:
            runner.run_spans([(0x4080, 8)], None, None, bytes(32))
        except _native.Error as error:
            assert "cuStreamSynchronize" in str(error), str(error)
        else:
            raise AssertionError("injected synchronization failure was ignored")

        # launch() attempted both its normal synchronization and its catch-path
        # drain. The persistent flag prevents workspace reuse or cuMemFree
        # until the owning Notary destroys the whole CUDA context.
        assert driver.cuattestFakeLaunchCalls() == 1
        assert driver.cuattestFakeSynchronizeCalls() == 2
        assert runner.context_cleanup_required is True
        assert runner.ipc_cleanup_failed is True  # compatibility alias
        runner.close()
        assert driver.cuattestFakeFreeCalls() == 0
        """
    )
    environment = os.environ.copy()
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(fake_cuda_driver)
        if not existing
        else f"{fake_cuda_driver}{os.pathsep}{existing}"
    )
    environment["CUATTEST_FAKE_SYNC_FAIL"] = "1"
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

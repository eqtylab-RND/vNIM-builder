// SPDX-License-Identifier: Apache-2.0
// Instrumented driver double: actual host allocations/copies, synthetic
// kernel output, and deterministic failures. Never loads or touches a GPU.
#include "native_core.hpp"

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <condition_variable>
#include <mutex>

// Use the SAME driver ABI declarations as the caller, including the by-value
// handle's C++ type identity. Independently redeclaring a layout-compatible C
// struct makes function-type sanitizers report the test double, not the host.
static thread_local int mode;
static thread_local int mappings;
static thread_local void *allocations[32];
static thread_local unsigned char ipc_storage[4096];
struct PendingCopy { void *dst; const void *src; size_t size; };
static thread_local PendingCopy pending[16];
static thread_local unsigned pending_count;
static thread_local uintptr_t last_measure_function;
static thread_local unsigned last_measure_grid;
static std::mutex sync_mutex;
static std::condition_variable sync_ready;
static unsigned sync_arrived;

extern "C" {

void audit_driver_finish_work(void) {
    for (unsigned i = 0; i < pending_count; ++i) {
        memcpy(pending[i].dst, pending[i].src, pending[i].size);
    }
    pending_count = 0;
}

void audit_driver_reset(int new_mode) {
    assert(pending_count == 0);
    for (unsigned i = 0; i < 32; ++i) {
        free(allocations[i]);
        allocations[i] = NULL;
    }
    mode = new_mode;
    mappings = 0;
}
int audit_driver_live_allocations(void) {
    int count = 0;
    for (unsigned i = 0; i < 32; ++i) count += allocations[i] != NULL;
    return count;
}
int audit_driver_open_mappings(void) { return mappings; }
uintptr_t audit_driver_measure_function(void) { return last_measure_function; }
unsigned audit_driver_measure_grid(void) { return last_measure_grid; }
int cuGetErrorString(int result, const char **text) {
    (void)result; *text = "sanitizer-harness injected failure"; return 0;
}
int cuMemAlloc_v2(CUdeviceptr *pointer, size_t nbytes) {
    if (mode == 64) return 2;
    for (unsigned i = 0; i < 32; ++i) {
        if (!allocations[i]) {
            allocations[i] = malloc(nbytes);
            assert(allocations[i]);
            *pointer = (uintptr_t)allocations[i];
            return 0;
        }
    }
    abort();
}
int cuMemFree_v2(CUdeviceptr pointer) {
    for (unsigned i = 0; i < 32; ++i) {
        if ((uintptr_t)allocations[i] == pointer) {
            free(allocations[i]); allocations[i] = NULL; return 0;
        }
    }
    abort();
}
int cuMemAllocHost_v2(void **pointer, size_t nbytes) {
    CUdeviceptr raw = 0;
    int rc = cuMemAlloc_v2(&raw, nbytes);
    *pointer = (void *)(uintptr_t)raw;
    return rc;
}
int cuMemFreeHost(void *pointer) { return cuMemFree_v2((uintptr_t)pointer); }
int cuMemcpyHtoDAsync_v2(CUdeviceptr dst, const void *src, size_t nbytes, void *stream) {
    assert(stream == (void *)6);
    assert(pending_count < 16);
    pending[pending_count++] = {(void *)(uintptr_t)dst, src, nbytes};
    return mode == 256 ? 700 : 0;
}
int cuMemcpyDtoHAsync_v2(void *dst, CUdeviceptr src, size_t nbytes, void *stream) {
    assert(stream == (void *)6);
    if (mode == 8) return 700;
    assert(pending_count < 16);
    pending[pending_count++] = {dst, (void *)(uintptr_t)src, nbytes};
    return 0;
}
static void *pointer_arg(void **args, unsigned index) {
    return (void *)(uintptr_t)*(const CUdeviceptr *)args[index];
}
int cuLaunchCooperativeKernel(void *function, unsigned gx, unsigned gy, unsigned gz,
    unsigned bx, unsigned by, unsigned bz, unsigned shared, void *stream, void **args) {
    (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
    (void)shared; assert(stream == (void *)6);
    if ((uintptr_t)function == 1 || (uintptr_t)function == 7) {
        last_measure_function = (uintptr_t)function;
        last_measure_grid = gx;
        int count = *(int *)args[1];
        memset(pointer_arg(args, 7), 0x5a, (size_t)count * 32);
        memset(pointer_arg(args, 8), 0xa5, 32);
        *(int32_t *)pointer_arg(args, 10) = 0;
    } else {
        assert((uintptr_t)function == 2);
        if (mode == 1) return 701;
        memcpy(pointer_arg(args, 8), "{}", 2);
        *(int32_t *)pointer_arg(args, 10) = mode == 16 ? 8193 : 2;
        *(int32_t *)pointer_arg(args, 11) = mode == 32 ? -7 : 0;
        // The real driver copies the full output capacity. Initialize that
        // capacity here so MSan can inspect host reads rather than mock gaps.
        memset((char *)pointer_arg(args, 8) + 2, 0, (size_t)*(int *)args[9] - 2);
    }
    return 0;
}
int cuCtxSynchronize(void) { abort(); }
int cuStreamSynchronize(void *stream) {
    assert(stream == (void *)6);
    if (mode == 2) return 702;
    // Actual copies happen at completion, not submission. Sanitizers now catch
    // freeing either DMA endpoint before the successful drain or mock context
    // destruction, including when an upload reports an error after submission.
    audit_driver_finish_work();
    if (mode == 128) {
        // A lock held across GPU completion would serialize these eight
        // independent sessions and fail the barrier. Driver double only:
        // no real kernel or context should ever wait on a host test barrier.
        std::unique_lock<std::mutex> lock(sync_mutex);
        ++sync_arrived;
        sync_ready.notify_all();
        assert(sync_ready.wait_for(lock, std::chrono::seconds(15),
                                  [] { return sync_arrived == 8; }));
    }
    return 0;
}
int cuIpcGetMemHandle(CUipcMemHandle *handle, CUdeviceptr pointer) {
    (void)pointer; memset(handle, 0xab, sizeof(*handle)); return 0;
}
int cuIpcOpenMemHandle_v2(CUdeviceptr *pointer, CUipcMemHandle handle, unsigned flags) {
    (void)handle; (void)flags; *pointer = (uintptr_t)ipc_storage; ++mappings; return 0;
}
int cuIpcCloseMemHandle(CUdeviceptr pointer) {
    (void)pointer; if (mode == 4) return 703; --mappings; return 0;
}
int cuPointerGetAttribute(void *data, int attribute, CUdeviceptr pointer) {
    (void)pointer; *(int *)data = attribute == 10 ? 1 : 0; return 0;
}
int cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *nbytes, CUdeviceptr pointer) {
    (void)pointer; *base = (uintptr_t)ipc_storage; *nbytes = sizeof(ipc_storage); return 0;
}
} // extern "C"

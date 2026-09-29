// SPDX-License-Identifier: Apache-2.0
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>

// Standalone CUDA only: no cuAttest, Python, PyTorch, IPC, or cooperative grid.
__global__ void noop() {}
#define CUDA(call) do { auto rc = (call); if (rc != cudaSuccess) { \
    std::fprintf(stderr, "%s: %d\n", #call, int(rc)); std::abort(); } } while (0)
int main() {
    cudaStream_t own, other;
    CUDA(cudaStreamCreateWithFlags(&own, cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
    unsigned *gate;
    CUDA(cudaMallocHost(&gate, sizeof(*gate)));
    *gate = 0;
    noop<<<1,1,0,own>>>();
    CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(own));
    std::mutex mutex;
    std::condition_variable ready;
    bool done = false;
    std::thread watchdog([&] {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait_for(lock, std::chrono::seconds(3), [&] { return done; });
        *static_cast<volatile unsigned *>(gate) = 1;
    });
    CUresult rc = cuStreamWaitValue32(other, reinterpret_cast<CUdeviceptr>(gate), 1,
                                     CU_STREAM_WAIT_VALUE_EQ);
    if (rc != CUDA_SUCCESS) std::abort();
    auto start = std::chrono::steady_clock::now();
    noop<<<1,1,0,own>>>();
    CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(own));
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    bool still_blocked = cudaStreamQuery(other) == cudaErrorNotReady;
    {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
        ready.notify_all();
    }
    watchdog.join();
    CUDA(cudaStreamSynchronize(other));
    CUDA(cudaStreamDestroy(other));
    CUDA(cudaStreamDestroy(own));
    CUDA(cudaFreeHost(gate));
    CUDA(cudaDeviceReset());
    std::printf("independent stream: %.6f seconds; other still blocked: %s\n",
                elapsed, still_blocked ? "yes" : "no");
    return still_blocked ? 0 : 3;
}

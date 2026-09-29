// SPDX-License-Identifier: Apache-2.0
//
// Latency-sensitive CUDA host orchestration for cuAttest.
//
// The extension deliberately uses the CUDA driver ABI through dlopen rather
// than CUDA headers or libcuda at link time.  Installing and importing the
// package therefore still works without a CUDA toolkit or NVIDIA driver; the
// driver is required only when a real FusedRunner is constructed or a client
// exports a CUDA allocation.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

// Keep the driver ABI's by-value record at external linkage. A record in an
// anonymous namespace has a different type identity in every translation
// unit, so cross-DSO CFI rejects an otherwise layout-compatible instrumented
// driver. The public CUDA tag also makes that ABI contract explicit.
struct CUipcMemHandle_st {
  char reserved[64];
};

namespace {

using CUdeviceptr = unsigned long long;
using CUfunction = void *;
using CUresult = int;
using CUipcMemHandle = CUipcMemHandle_st;

constexpr std::size_t kIpcHandleBytes = 64;
constexpr std::uint64_t kChunkBytes = 1024;
constexpr std::uint64_t kTileChunks = 128;
constexpr std::uint64_t kCvBytes = 32;
constexpr std::size_t kDescriptorBytes = 5 * sizeof(std::uint64_t);
constexpr unsigned kThreads = 128;
// Must match _OUT_CAP in notary.py.
constexpr int kOutCapacity = 6144;
constexpr std::size_t kTimestampBytes = 20;
// instanceID: which resident copy these spans came from, folded by the
// caller over the IPC handles it resolved. The device never sees a handle.
constexpr std::size_t kInstanceRootBytes = 32;
constexpr int kPointerAttributeDeviceOrdinal = 9;
constexpr int kPointerAttributeIsLegacyCudaIpcCapable = 10;
constexpr unsigned kIpcLazyEnablePeerAccess = 1;

// Assertions diagnose OUR contracts, never replace fallible input/driver
// validation. They must be side-effect-free: Release defines NDEBUG. Asserted
// builds may abort the process and must not be used with production secrets.
#ifndef CUATTEST_BUILD_TYPE
#ifdef NDEBUG
#define CUATTEST_BUILD_TYPE "Release"
#else
#define CUATTEST_BUILD_TYPE "AssertedRelease"
#endif
#endif

static_assert(sizeof(CUdeviceptr) == 8);
static_assert(sizeof(int) == 4);
static_assert(sizeof(CUipcMemHandle) == kIpcHandleBytes);
static_assert(kDescriptorBytes == 40);
static_assert(kThreads == 2 * kTileChunks / 2);
static_assert(kChunkBytes == 1024 && kCvBytes == 32);

class NativeFailure : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class IpcCloseFailure : public std::exception {
public:
  explicit IpcCloseFailure(CUresult result) noexcept : result_(result) {}

  const char *what() const noexcept override {
    return "cuIpcCloseMemHandle failed";
  }

  CUresult result() const noexcept { return result_; }

private:
  CUresult result_;
};

struct CudaApi {
  using GetErrorString = CUresult (*)(CUresult, const char **);
  using MemAlloc = CUresult (*)(CUdeviceptr *, std::size_t);
  using MemFree = CUresult (*)(CUdeviceptr);
  using MemcpyHtoD = CUresult (*)(CUdeviceptr, const void *, std::size_t, void *);
  using MemcpyDtoH = CUresult (*)(void *, CUdeviceptr, std::size_t, void *);
  using MemAllocHost = CUresult (*)(void **, std::size_t);
  using MemFreeHost = CUresult (*)(void *);
  using LaunchCooperative = CUresult (*)(CUfunction, unsigned, unsigned,
                                         unsigned, unsigned, unsigned, unsigned,
                                         unsigned, void *, void **);
  using StreamSynchronize = CUresult (*)(void *);
  using IpcGet = CUresult (*)(CUipcMemHandle *, CUdeviceptr);
  using IpcOpen = CUresult (*)(CUdeviceptr *, CUipcMemHandle, unsigned);
  using IpcClose = CUresult (*)(CUdeviceptr);
  using PointerGetAttribute = CUresult (*)(void *, int, CUdeviceptr);
  using MemGetAddressRange = CUresult (*)(CUdeviceptr *, std::size_t *,
                                          CUdeviceptr);

  void *library = nullptr;
  GetErrorString get_error_string = nullptr;
  MemAlloc mem_alloc = nullptr;
  MemFree mem_free = nullptr;
  MemcpyHtoD memcpy_htod = nullptr;
  MemcpyDtoH memcpy_dtoh = nullptr;
  MemAllocHost mem_alloc_host = nullptr;
  MemFreeHost mem_free_host = nullptr;
  LaunchCooperative launch_cooperative = nullptr;
  StreamSynchronize stream_synchronize = nullptr;
  IpcGet ipc_get = nullptr;
  IpcOpen ipc_open = nullptr;
  IpcClose ipc_close = nullptr;
  PointerGetAttribute pointer_get_attribute = nullptr;
  MemGetAddressRange mem_get_address_range = nullptr;

  CudaApi() {
    library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      const char *error = dlerror();
      throw NativeFailure(std::string("cannot load libcuda.so.1: ") +
                          (error == nullptr ? "unknown dlopen error" : error));
    }
    try {
      get_error_string = symbol<GetErrorString>("cuGetErrorString");
      mem_alloc = symbol<MemAlloc>("cuMemAlloc_v2");
      mem_free = symbol<MemFree>("cuMemFree_v2");
      memcpy_htod = symbol<MemcpyHtoD>("cuMemcpyHtoDAsync_v2");
      memcpy_dtoh = symbol<MemcpyDtoH>("cuMemcpyDtoHAsync_v2");
      mem_alloc_host = symbol<MemAllocHost>("cuMemAllocHost_v2");
      mem_free_host = symbol<MemFreeHost>("cuMemFreeHost");
      launch_cooperative =
          symbol<LaunchCooperative>("cuLaunchCooperativeKernel");
      stream_synchronize = symbol<StreamSynchronize>("cuStreamSynchronize");
      ipc_get = symbol<IpcGet>("cuIpcGetMemHandle");
      ipc_open =
          symbol_either<IpcOpen>("cuIpcOpenMemHandle_v2", "cuIpcOpenMemHandle");
      ipc_close = symbol<IpcClose>("cuIpcCloseMemHandle");
      pointer_get_attribute =
          symbol<PointerGetAttribute>("cuPointerGetAttribute");
      mem_get_address_range = symbol_either<MemGetAddressRange>(
          "cuMemGetAddressRange_v2", "cuMemGetAddressRange");
    } catch (...) {
      dlclose(library);
      library = nullptr;
      throw;
    }
  }

  ~CudaApi() {
    if (library != nullptr) {
      dlclose(library);
    }
  }

  CudaApi(const CudaApi &) = delete;
  CudaApi &operator=(const CudaApi &) = delete;

  template <typename T> T symbol(const char *name) {
    dlerror();
    void *value = dlsym(library, name);
    const char *error = dlerror();
    if (error != nullptr || value == nullptr) {
      throw NativeFailure(std::string("libcuda exports no ") + name);
    }
    return reinterpret_cast<T>(value);
  }

  template <typename T>
  T symbol_either(const char *preferred, const char *fallback) {
    dlerror();
    void *value = dlsym(library, preferred);
    if (dlerror() == nullptr && value != nullptr) {
      return reinterpret_cast<T>(value);
    }
    return symbol<T>(fallback);
  }

  std::string describe(CUresult result) const {
    const char *text = nullptr;
    if (get_error_string != nullptr && get_error_string(result, &text) == 0 &&
        text != nullptr) {
      return text;
    }
    return std::string("error ") + std::to_string(result);
  }

  void check(CUresult result, const char *operation) const {
    if (result != 0) {
      throw NativeFailure(std::string(operation) + ": " + describe(result) +
                          " (rc=" + std::to_string(result) + ")");
    }
  }
};

struct InputSpan {
  std::uint64_t pointer;
  std::uint64_t nbytes;
};

struct IpcInput {
  std::array<unsigned char, kIpcHandleBytes> handle{};
  std::uint64_t nbytes = 0;
  std::uint64_t segment_offset = 0;
  std::uint64_t tensor_offset = 0;
};

struct FusedPlan {
  std::vector<unsigned char> descriptors;
  std::vector<unsigned char> reduction_offsets;
  std::uint64_t total_tiles = 0;
  std::uint64_t secondary_tiles = 0;
  int levels = 0;
};

struct FusedResult {
  std::vector<unsigned char> roots;
  std::array<unsigned char, 32> model_root{};
  std::string receipt;
  bool signing = false;
};

void append_u64_le(std::vector<unsigned char> &output, std::uint64_t value) {
  assert(output.size() % sizeof(value) == 0);
  for (unsigned shift = 0; shift != 64; shift += 8) {
    output.push_back(static_cast<unsigned char>(value >> shift));
  }
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const char *message) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw NativeFailure(message);
  }
  return left + right;
}

std::size_t checked_size(std::uint64_t count, std::uint64_t width,
                         const char *message) {
  assert(width != 0); // All internal callers supply a nonzero ABI element size.
  if (count > std::numeric_limits<std::uint64_t>::max() / width) {
    throw NativeFailure(message);
  }
  const std::uint64_t bytes = count * width;
  if (bytes > std::numeric_limits<std::size_t>::max()) {
    throw NativeFailure(message);
  }
  return static_cast<std::size_t>(bytes);
}

FusedPlan make_plan(const std::vector<InputSpan> &spans) {
  if (spans.empty() || spans.size() > static_cast<std::size_t>(
                                          std::numeric_limits<int>::max())) {
    throw NativeFailure("fused hashing requires 1 to 2147483647 spans");
  }

  FusedPlan plan;
  if (spans.size() >
      std::numeric_limits<std::size_t>::max() / kDescriptorBytes) {
    throw NativeFailure("fused descriptor size exceeds host address space");
  }
  plan.descriptors.reserve(spans.size() * kDescriptorBytes);
  std::vector<std::uint64_t> tile_counts;
  tile_counts.reserve(spans.size());

  for (const InputSpan &span : spans) {
    // Direct device-range hashing intentionally supports BLAKE3's empty
    // input. IPC tensor references are validated as non-empty in Python.
    const std::uint64_t chunks =
        span.nbytes == 0 ? 1 : 1 + (span.nbytes - 1) / kChunkBytes;
    const std::uint64_t tiles = 1 + (chunks - 1) / kTileChunks;
    assert(chunks >= 1 && tiles >= 1);
    if (tiles > std::numeric_limits<std::uint64_t>::max() / kCvBytes -
                    plan.total_tiles) {
      throw NativeFailure(
          "fused BLAKE3 workspace exceeds the CUDA address width");
    }
    append_u64_le(plan.descriptors, span.pointer);
    append_u64_le(plan.descriptors, span.nbytes);
    append_u64_le(plan.descriptors, plan.total_tiles);
    append_u64_le(plan.descriptors, plan.secondary_tiles);
    append_u64_le(plan.descriptors, chunks);
    tile_counts.push_back(tiles);
    plan.total_tiles += tiles;
    if (tiles > 1) {
      plan.secondary_tiles += (tiles + 1) / 2;
    }

    std::uint64_t remaining = tiles - 1;
    int levels = 0;
    while (remaining != 0) {
      ++levels;
      remaining >>= 1;
    }
    plan.levels = std::max(plan.levels, levels);
    assert(plan.secondary_tiles <= plan.total_tiles);
    assert(plan.levels >= 0 && plan.levels < 64);
  }

  const std::size_t row_values = spans.size() + 1;
  if (plan.levels > 0 &&
      row_values > std::numeric_limits<std::size_t>::max() /
                       static_cast<std::size_t>(plan.levels)) {
    throw NativeFailure("fused reduction plan exceeds host address space");
  }
  plan.reduction_offsets.reserve(std::max<std::size_t>(
      8, row_values * static_cast<std::size_t>(plan.levels) * 8));
  std::vector<std::uint64_t> current = tile_counts;
  for (int level = 0; level < plan.levels; ++level) {
    std::uint64_t prefix = 0;
    append_u64_le(plan.reduction_offsets, 0);
    for (std::uint64_t &count : current) {
      assert(count >= 1);
      const std::uint64_t work = count > 1 ? (count + 1) / 2 : 0;
      prefix = checked_add(
          prefix, work, "fused reduction plan exceeds the CUDA address width");
      append_u64_le(plan.reduction_offsets, prefix);
      count = count > 1 ? (count + 1) / 2 : 1;
    }
  }
  if (plan.reduction_offsets.empty()) {
    plan.reduction_offsets.resize(8, 0);
  }
  assert(plan.descriptors.size() == spans.size() * kDescriptorBytes);
  assert(plan.reduction_offsets.size() == std::max<std::size_t>(
      8, row_values * static_cast<std::size_t>(plan.levels) * 8));
  assert(plan.total_tiles >= spans.size());
  assert(std::all_of(current.begin(), current.end(),
                     [](std::uint64_t count) { return count == 1; }));
  return plan;
}

struct DeviceAllocation {
  CUdeviceptr pointer = 0;
  std::size_t capacity = 0;
};

class FusedRunner {
public:
  FusedRunner(std::uint64_t measure_function,
              std::uint64_t attest_function, int grid_limit,
              std::uint64_t p256_context, std::uint64_t cubin_digest,
              std::uint64_t kernel_digest, int device_ordinal,
              std::uint64_t stream, std::uint64_t async_measure_function = 0,
              int async_grid_limit = 0, std::uint64_t async_min_bytes = 0)
      : api_(std::make_unique<CudaApi>()),
        measure_function_(reinterpret_cast<CUfunction>(
            static_cast<std::uintptr_t>(measure_function))),
        attest_function_(reinterpret_cast<CUfunction>(
            static_cast<std::uintptr_t>(attest_function))),
        grid_limit_(grid_limit), p256_context_(p256_context),
        cubin_digest_(cubin_digest), kernel_digest_(kernel_digest),
        device_ordinal_(device_ordinal),
        stream_(reinterpret_cast<void *>(static_cast<std::uintptr_t>(stream))),
        async_measure_function_(reinterpret_cast<CUfunction>(
            static_cast<std::uintptr_t>(async_measure_function))),
        async_grid_limit_(async_grid_limit), async_min_bytes_(async_min_bytes) {
    if (measure_function == 0 || attest_function == 0 || grid_limit <= 0 ||
        p256_context == 0 ||
        cubin_digest == 0 || kernel_digest == 0 || device_ordinal < 0 || stream == 0) {
      throw NativeFailure("invalid native fused-runner configuration");
    }
    if ((async_measure_function == 0) != (async_grid_limit == 0) || async_grid_limit < 0) {
      throw NativeFailure("invalid asynchronous measurement configuration");
    }
  }

  ~FusedRunner() = default;
  FusedRunner(const FusedRunner &) = delete;
  FusedRunner &operator=(const FusedRunner &) = delete;

  FusedResult
  run_spans(const std::vector<InputSpan> &spans,
            const std::array<unsigned char, kTimestampBytes> &timestamp,
            const std::array<unsigned char, kInstanceRootBytes> &instance_root,
            const std::string &model, bool signing) {
    // Internal, already-ordered spans: public Notary.hash_dptr queues a
    // producer event on stream_ first; hash_bytes finishes its upload first.
    // NON_BLOCKING does not implicitly wait for legacy-stream source writes.
    // Do not add a global wait here: IPC/export-ready inputs share launch().
    require_open();
    bool queued_work_unconfirmed = false;
    try {
      return launch(spans, timestamp, instance_root, model, signing,
                    &queued_work_unconfirmed);
    } catch (...) {
      if (queued_work_unconfirmed) {
        // Direct spans can point at a caller-owned DeviceBuffer just as IPC
        // spans point at an imported allocation. Persist the unsafe state so
        // Python destroys the context before either source or our retained
        // workspaces can be freed or reused.
        context_cleanup_required_ = true;
      }
      throw;
    }
  }

  FusedResult
  run_ipc(const std::vector<IpcInput> &inputs,
          const std::array<unsigned char, kTimestampBytes> &timestamp,
          const std::array<unsigned char, kInstanceRootBytes> &instance_root,
          const std::string &model, bool signing) {
    require_open();
    if (inputs.empty() ||
        inputs.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw NativeFailure("fused hashing requires 1 to 2147483647 spans");
    }

    struct Allocation {
      CUdeviceptr mapped = 0;
      CUdeviceptr base = 0;
      std::size_t nbytes = 0;
    };
    std::vector<Allocation> opened;
    opened.reserve(inputs.size());
    // Views point into `inputs`, whose storage is fixed for this whole call.
    // Avoid allocating a temporary std::string for every tensor just to find
    // the usually much smaller set of unique allocator segments.
    std::unordered_map<std::string_view, std::size_t> by_handle;
    by_handle.reserve(inputs.size());
    std::vector<InputSpan> spans;
    spans.reserve(inputs.size());
    std::exception_ptr operation_error;
    bool queued_work_unconfirmed = false;
    FusedResult result;

    try {
      // Batch one session's imports instead of making many GPU threads
      // contend inside the driver's IPC bookkeeping on every handle. This
      // process-wide gate protects scheduling, not mapping ownership: each
      // runner still validates and retires its own imports on every request.
      // Release it BEFORE launch/synchronization so GPU work stays concurrent;
      // unique_lock also releases it before the error-path cleanup below.
      static std::mutex import_mutex;
      std::unique_lock<std::mutex> import_lock(import_mutex);
      for (const IpcInput &input : inputs) {
        const std::string_view key(
            reinterpret_cast<const char *>(input.handle.data()),
            input.handle.size());
        auto found = by_handle.find(key);
        std::size_t index;
        if (found == by_handle.end()) {
          CUipcMemHandle handle{};
          std::memcpy(handle.reserved, input.handle.data(),
                      input.handle.size());
          Allocation allocation;
          api_->check(api_->ipc_open(&allocation.mapped, handle,
                                     kIpcLazyEnablePeerAccess),
                      "cuIpcOpenMemHandle");
          // Ownership is recorded before subsequent queries so every
          // successfully imported mapping is closed on any failure.
          opened.push_back(allocation);
          assert(allocation.mapped != 0);
          index = opened.size() - 1;
          by_handle.emplace(key, index);

          int allocation_device = -1;
          api_->check(api_->pointer_get_attribute(
                          &allocation_device, kPointerAttributeDeviceOrdinal,
                          allocation.mapped),
                      "cuPointerGetAttribute(DEVICE_ORDINAL)");
          if (allocation_device != device_ordinal_) {
            throw NativeFailure("IPC allocation belongs to cuda:" +
                                std::to_string(allocation_device) +
                                ", but this notary uses cuda:" +
                                std::to_string(device_ordinal_));
          }
          api_->check(api_->mem_get_address_range(&allocation.base,
                                                  &allocation.nbytes,
                                                  allocation.mapped),
                      "cuMemGetAddressRange");
          opened[index] = allocation;
        } else {
          index = found->second;
        }

        const Allocation &allocation = opened[index];
        const std::uint64_t relative =
            checked_add(input.segment_offset, input.tensor_offset,
                        "tensor offsets overflow the CUDA address width");
        const std::uint64_t start =
            checked_add(allocation.mapped, relative,
                        "tensor span overflows the CUDA address width");
        const std::uint64_t allocation_size =
            static_cast<std::uint64_t>(allocation.nbytes);
        const bool before_base = start < allocation.base;
        const std::uint64_t inside = before_base ? 0 : start - allocation.base;
        if (before_base || inside > allocation_size ||
            input.nbytes > allocation_size - inside) {
          throw NativeFailure(
              "tensor span lies outside the imported allocation "
              "(offset=" +
              std::to_string(relative) +
              ", nbytes=" + std::to_string(input.nbytes) +
              ", allocation=" + std::to_string(allocation.nbytes) + ")");
        }
        spans.push_back({start, input.nbytes});
        assert(index < opened.size() && by_handle.size() == opened.size());
      }
      import_lock.unlock();
      assert(spans.size() == inputs.size());
      result = launch(spans, timestamp, instance_root, model, signing,
                      &queued_work_unconfirmed);
    } catch (...) {
      operation_error = std::current_exception();
    }

    if (queued_work_unconfirmed) {
      // At least one launch succeeded, but neither the normal synchronize nor
      // the exception-path drain confirmed completion. Unmapping an imported
      // allocation here could race a still-running measurement kernel. Leave
      // every mapping owned by the CUDA context and set the persistent
      // sentinel: Python destroys that context before acknowledging the
      // request (and withholds acknowledgement if destruction is uncertain).
      context_cleanup_required_ = true;
      assert(operation_error != nullptr);
      std::rethrow_exception(operation_error);
    }

    CUresult first_close_error = 0;
    for (const Allocation &allocation : opened) {
      const CUresult rc = api_->ipc_close(allocation.mapped);
      if (rc != 0 && first_close_error == 0) {
        first_close_error = rc;
      }
    }
    if (first_close_error != 0) {
      // Set the sentinel before throwing. Python checks it for every exception
      // from run_ipc, so even failure to allocate an exception message cannot
      // turn an unconfirmed unmap into a normal, acknowledgeable error.
      context_cleanup_required_ = true;
      throw IpcCloseFailure(first_close_error);
    }
    if (operation_error != nullptr) {
      std::rethrow_exception(operation_error);
    }
    return result;
  }

  void close() noexcept {
    if (closed_) {
      return;
    }
    if (context_cleanup_required_) {
      // cuMemFree or later reuse is unsafe while a kernel may still reference
      // these buffers. Abandon the wrappers and let cuCtxDestroy reclaim the
      // allocations together with any still-open IPC mappings.
      abandon(metadata_);
      abandon(work_a_);
      abandon(work_b_);
      abandon(output_);
      // cuMemAllocHost allocations, like cuMemAlloc, belong to this context.
      // Let destruction retire DMA endpoints too; cuMemFreeHost AFTER destroy
      // would access an already-retired resource (unlike async/pool allocations).
      metadata_host_ = {};
      output_host_ = {};
    } else {
      release(metadata_);
      release(work_a_);
      release(work_b_);
      release(output_);
      release_host_buffers();
    }
    closed_ = true;
    assert(metadata_.pointer == 0 && work_a_.pointer == 0 &&
           work_b_.pointer == 0 && output_.pointer == 0);
    assert(metadata_host_.pointer == nullptr && output_host_.pointer == nullptr);
  }

  void release_host_buffers() noexcept {
    assert(!context_cleanup_required_);
    // Called only for a drained session. Pinned memory is not vector-owned
    // and must survive exceptional unwinding while DMA may reference it.
    for (HostAllocation *allocation : {&metadata_host_, &output_host_}) {
      if (allocation->pointer != nullptr) {
        (void)api_->mem_free_host(allocation->pointer);
        allocation->pointer = nullptr;
        allocation->capacity = 0;
      }
    }
  }

  std::array<std::size_t, 4> capacities() const {
    return {metadata_.capacity, work_a_.capacity, work_b_.capacity,
            output_.capacity};
  }

  bool context_cleanup_required() const noexcept {
    return context_cleanup_required_;
  }

  bool ipc_cleanup_failed() const noexcept {
    return context_cleanup_required_;
  }

private:
  struct HostAllocation {
    void *pointer = nullptr;
    std::size_t capacity = 0;
  };

  void reserve_host(HostAllocation &allocation, std::size_t requested) {
    assert(!closed_ && !context_cleanup_required_ && requested > 0);
    assert((allocation.pointer == nullptr) == (allocation.capacity == 0));
    if (allocation.capacity >= requested) return;
    if (allocation.pointer != nullptr) {
      api_->check(api_->mem_free_host(allocation.pointer), "cuMemFreeHost(growing staging)");
      allocation.pointer = nullptr;
      allocation.capacity = 0;
    }
    api_->check(api_->mem_alloc_host(&allocation.pointer, requested),
                "cuMemAllocHost(native staging)");
    allocation.capacity = requested;
    assert(allocation.pointer != nullptr && allocation.capacity >= requested);
  }

  void require_open() const {
    if (closed_) {
      throw NativeFailure("native fused runner is closed");
    }
    assert(api_ != nullptr && stream_ != nullptr);
    assert(!context_cleanup_required_); // An uncertain runner cannot be reused.
  }

  void release(DeviceAllocation &allocation) noexcept {
    if (allocation.pointer != 0 && api_ != nullptr) {
      (void)api_->mem_free(allocation.pointer);
    }
    allocation.pointer = 0;
    allocation.capacity = 0;
  }

  static void abandon(DeviceAllocation &allocation) noexcept {
    allocation.pointer = 0;
    allocation.capacity = 0;
  }

  void reserve(DeviceAllocation &allocation, std::size_t requested,
               const char *label) {
    assert(!closed_ && !context_cleanup_required_);
    assert((allocation.pointer == 0) == (allocation.capacity == 0));
    // CUDA rejects zero-byte allocations. Plans without a secondary reduction
    // retain a one-byte sentinel, never read or written by the kernel. That is
    // intentional unused capacity, not an uninitialized-memory access.
    requested = std::max<std::size_t>(requested, 1);
    if (allocation.capacity >= requested) {
      return;
    }
    // Do not overlap the old and larger allocations: for a near-capacity
    // model that transient peak could reject a request whose actual workspace
    // fits. Growth is off the steady-state path. For large spans, the retained
    // buffers are about 0.037% of attested bytes (32 bytes per 128 KiB tile
    // plus the half-sized peer); the first seven BLAKE3 levels live in shared
    // memory.
    if (allocation.pointer != 0) {
      api_->check(api_->mem_free(allocation.pointer),
                  "cuMemFree(growing native workspace)");
      allocation.pointer = 0;
      allocation.capacity = 0;
    }
    CUdeviceptr replacement = 0;
    api_->check(api_->mem_alloc(&replacement, requested), label);
    allocation.pointer = replacement;
    allocation.capacity = requested;
    assert(allocation.pointer != 0 && allocation.capacity >= requested);
  }

  FusedResult
  launch(const std::vector<InputSpan> &spans,
         const std::array<unsigned char, kTimestampBytes> &timestamp,
         const std::array<unsigned char, kInstanceRootBytes> &instance_root,
         const std::string &model, bool signing,
         bool *queued_work_unconfirmed) {
    assert(queued_work_unconfirmed != nullptr && !*queued_work_unconfirmed);
    const FusedPlan plan = make_plan(spans);
    // Validate the aggregate before allocating workspaces. Dispatch must not
    // wrap the byte count (or attempt enormous allocations before rejecting it).
    std::uint64_t input_bytes = 0;
    for (const InputSpan &span : spans) {
      input_bytes = checked_add(input_bytes, span.nbytes, "aggregate input bytes overflow");
    }
    if (signing && (model.empty() || model.size() > 64)) {
      throw NativeFailure("model must contain 1 to 64 bytes");
    }

    std::vector<unsigned char> metadata;
    const std::size_t model_bytes = signing ? model.size() : 1;
    if (plan.descriptors.size() > std::numeric_limits<std::size_t>::max() -
                                      plan.reduction_offsets.size() -
                                      kTimestampBytes - model_bytes -
                                      kInstanceRootBytes) {
      throw NativeFailure("fused metadata exceeds host address space");
    }
    metadata.reserve(plan.descriptors.size() + plan.reduction_offsets.size() +
                     kTimestampBytes + model_bytes + kInstanceRootBytes);
    metadata.insert(metadata.end(), plan.descriptors.begin(),
                    plan.descriptors.end());
    const std::size_t reduction_offset = metadata.size();
    metadata.insert(metadata.end(), plan.reduction_offsets.begin(),
                    plan.reduction_offsets.end());
    const std::size_t timestamp_offset = metadata.size();
    metadata.insert(metadata.end(), timestamp.begin(), timestamp.end());
    const std::size_t model_offset = metadata.size();
    if (signing) {
      metadata.insert(metadata.end(), model.begin(), model.end());
    } else {
      metadata.push_back(0);
    }
    const std::size_t instance_offset = metadata.size();
    metadata.insert(metadata.end(), instance_root.begin(), instance_root.end());

    const std::size_t roots_bytes =
        checked_size(static_cast<std::uint64_t>(spans.size()), kCvBytes,
                     "fused output exceeds host address space");
    const std::size_t model_root_offset = roots_bytes;
    const std::size_t json_offset = model_root_offset + 32;
    int json_capacity = signing ? kOutCapacity : 0;
    const std::size_t out_length_offset = json_offset + json_capacity;
    // Unsigned measurement has no receipt length to write. Overlay status on
    // that slot instead of copying four uninitialized device bytes to host.
    // Signed output retains the adjacent [length, status] pair.
    const std::size_t status_offset =
        out_length_offset + (signing ? sizeof(std::int32_t) : 0);
    const std::size_t output_bytes = status_offset + 4;
    const std::size_t work_a_bytes =
        checked_size(plan.total_tiles, kCvBytes,
                     "fused BLAKE3 workspace exceeds host address space");
    const std::size_t work_b_bytes =
        checked_size(plan.secondary_tiles, kCvBytes,
                     "fused BLAKE3 workspace exceeds host address space");

    reserve(metadata_, metadata.size(), "cuMemAlloc(native metadata)");
    reserve(work_a_, work_a_bytes, "cuMemAlloc(native work_a)");
    reserve(work_b_, work_b_bytes, "cuMemAlloc(native work_b)");
    reserve(output_, output_bytes, "cuMemAlloc(native output)");
    reserve_host(metadata_host_, metadata.size());
    reserve_host(output_host_, output_bytes);
    assert(metadata_.capacity >= metadata.size() && metadata_host_.capacity >= metadata.size());
    assert(output_.capacity >= output_bytes && output_host_.capacity >= output_bytes);
    assert(work_a_.capacity >= work_a_bytes && work_b_.capacity >= work_b_bytes);
    assert(reduction_offset % 8 == 0 && timestamp_offset % 8 == 0);
    assert(instance_offset + kInstanceRootBytes == metadata.size());
    assert(model_offset + model_bytes == instance_offset);
    assert(status_offset + sizeof(std::int32_t) == output_bytes);
    std::memcpy(metadata_host_.pointer, metadata.data(), metadata.size());

    const std::uint64_t requested_blocks =
        (plan.total_tiles + 1) / 2;
    // Each kernel has its OWN occupancy ceiling: async staging consumes more
    // shared memory. Reusing the standard grid could deadlock a cooperative
    // launch. Both paths retain the same drained workspaces and private stream.
    const bool use_async = async_measure_function_ != nullptr && input_bytes >= async_min_bytes_;
    const int grid_limit = use_async ? async_grid_limit_ : grid_limit_;
    CUfunction measure_function = use_async ? async_measure_function_ : measure_function_;
    const unsigned grid = static_cast<unsigned>(
        std::min<std::uint64_t>(static_cast<std::uint64_t>(grid_limit),
                                std::max<std::uint64_t>(1, requested_blocks)));
    int span_count = static_cast<int>(spans.size());
    assert(grid > 0 && grid <= static_cast<unsigned>(grid_limit));
    assert(measure_function != nullptr && spans.size() == static_cast<std::size_t>(span_count));
    int levels = plan.levels;
    std::uint64_t total_tiles = plan.total_tiles;
    int sign_receipt = signing ? 1 : 0;
    int model_length = signing ? static_cast<int>(model.size()) : 0;
    CUdeviceptr reduction_pointer = metadata_.pointer + reduction_offset;
    CUdeviceptr timestamp_pointer = metadata_.pointer + timestamp_offset;
    CUdeviceptr model_pointer = metadata_.pointer + model_offset;
    CUdeviceptr instance_pointer = metadata_.pointer + instance_offset;
    CUdeviceptr model_root_pointer = output_.pointer + model_root_offset;
    CUdeviceptr json_pointer = output_.pointer + json_offset;
    CUdeviceptr out_length_pointer = output_.pointer + out_length_offset;
    CUdeviceptr status_pointer = output_.pointer + status_offset;
    void *measure_arguments[] = {
        &metadata_.pointer,  &span_count,       &reduction_pointer,
        &levels,             &total_tiles,      &work_a_.pointer,
        &work_b_.pointer,    &output_.pointer,  &model_root_pointer,
        &sign_receipt,       &status_pointer,
    };
    bool work_queued = false;
    try {
      // Upload, measurement, private-root signing, and download are one FIFO
      // on the session's nonblocking stream. Synchronous/default-stream copies
      // would either serialize unrelated work or race this nonblocking stream.
      // Set the drain obligation BEFORE even the first DMA submission.
      work_queued = true;
      api_->check(api_->memcpy_htod(metadata_.pointer, metadata_host_.pointer,
                                    metadata.size(), stream_),
                  "cuMemcpyHtoDAsync(native metadata)");
      const CUresult measurement_launch = api_->launch_cooperative(
          measure_function, grid, 1, 1, kThreads, 1, 1, 0, stream_,
          measure_arguments);
      // A successful launch is asynchronous. Record that fact before the next
      // fallible host operation so every exceptional exit knows it must drain
      // the stream before an IPC mapping can be closed.
      if (measurement_launch == 0) {
        work_queued = true;
      }
      api_->check(measurement_launch,
                  "cuLaunchCooperativeKernel(measurement)");
      if (signing) {
        void *attest_arguments[] = {
            &span_count,        &p256_context_,     &timestamp_pointer,
            &model_pointer,     &model_length,      &cubin_digest_,
            &kernel_digest_,    &instance_pointer,  &device_ordinal_,
            &json_pointer,      &json_capacity,     &out_length_pointer,
            &status_pointer,
        };
        const CUresult attestation_launch = api_->launch_cooperative(
            attest_function_, 1, 1, 1, kThreads, 1, 1, 0, stream_,
            attest_arguments);
        if (attestation_launch == 0) {
          work_queued = true;
        }
        api_->check(attestation_launch,
                    "cuLaunchCooperativeKernel(attestation)");
      }
      api_->check(api_->memcpy_dtoh(output_host_.pointer, output_.pointer,
                                    output_bytes, stream_),
                  "cuMemcpyDtoHAsync(native output)");
      const CUresult synchronized = api_->stream_synchronize(stream_);
      if (synchronized == 0) {
        work_queued = false;
      }
      api_->check(synchronized, "cuStreamSynchronize");
    } catch (...) {
      if (work_queued) {
        // In particular, the attestation launch can fail synchronously after
        // the measurement launch has started. A best-effort drain is required
        // before run_ipc may unmap the producer's allocation.
        if (api_->stream_synchronize(stream_) == 0) {
          work_queued = false;
        }
      }
      if (work_queued && queued_work_unconfirmed != nullptr) {
        *queued_work_unconfirmed = true;
      }
      throw;
    }

    const auto *raw = static_cast<const unsigned char *>(output_host_.pointer);
    assert(!work_queued && !*queued_work_unconfirmed);
    const auto read_i32 = [raw, output_bytes](std::size_t offset) {
      assert(offset <= output_bytes && sizeof(std::int32_t) <= output_bytes - offset);
      const std::uint32_t value =
          static_cast<std::uint32_t>(raw[offset]) |
          (static_cast<std::uint32_t>(raw[offset + 1]) << 8) |
          (static_cast<std::uint32_t>(raw[offset + 2]) << 16) |
          (static_cast<std::uint32_t>(raw[offset + 3]) << 24);
      return static_cast<std::int32_t>(value);
    };
    const std::int32_t status = read_i32(status_offset);
    if (status != 0) {
      const char *description = "unknown";
      switch (status) {
      case -2:
        description = "key not ready - keygen did not run";
        break;
      case -3:
        description = "malformed timestamp or model name";
        break;
      case -4:
        description = "output buffer too small";
        break;
      case -5:
        description = "invalid fused measurement plan";
        break;
      case -6:
        description = "no matching measured root is pending";
        break;
      case -7:
        description = "deterministic P-256 signing failed";
        break;
      case -8:
        description = "chain slot count out of range";
        break;
      case -9:
        description = "chain slots already configured";
        break;
      case -10:
        description = "no free chain slot for this instance";
        break;
      case -11:
        description = "chain slots not configured";
        break;
      }
      throw NativeFailure("fused measurement/finalization status=" +
                          std::to_string(status) + " (" + description + ")");
    }

    FusedResult result;
    result.signing = signing;
    result.roots.assign(raw, raw + roots_bytes);
    std::copy_n(raw + model_root_offset, result.model_root.size(),
                result.model_root.begin());
    if (signing) {
      const std::int32_t length = read_i32(out_length_offset);
      if (length <= 0 || length > json_capacity) {
        throw NativeFailure("kernel reported an output length of " +
                            std::to_string(length));
      }
      result.receipt.assign(
          reinterpret_cast<const char *>(raw + json_offset),
          static_cast<std::size_t>(length));
    }
    assert(result.roots.size() == spans.size() * kCvBytes);
    assert(result.signing == !result.receipt.empty());
    return result;
  }

  std::unique_ptr<CudaApi> api_;
  CUfunction measure_function_ = nullptr;
  CUfunction attest_function_ = nullptr;
  int grid_limit_ = 0;
  CUdeviceptr p256_context_ = 0;
  CUdeviceptr cubin_digest_ = 0;
  CUdeviceptr kernel_digest_ = 0;
  int device_ordinal_ = 0;
  void *stream_ = nullptr; // Borrowed; Notary owns it and the containing context.
  CUfunction async_measure_function_ = nullptr;
  int async_grid_limit_ = 0;
  std::uint64_t async_min_bytes_ = 0;
  HostAllocation metadata_host_;
  HostAllocation output_host_;
  DeviceAllocation metadata_;
  DeviceAllocation work_a_;
  DeviceAllocation work_b_;
  DeviceAllocation output_;
  bool closed_ = false;
  bool context_cleanup_required_ = false;
};

bool py_u64(PyObject *object, std::uint64_t &output) {
  const unsigned long long value = PyLong_AsUnsignedLongLong(object);
  if (PyErr_Occurred()) {
    return false;
  }
  output = static_cast<std::uint64_t>(value);
  return true;
}

bool parse_spans(PyObject *object, std::vector<InputSpan> &output) {
  PyObject *sequence = PySequence_Fast(object, "spans must be a sequence");
  if (sequence == nullptr) {
    return false;
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
  try {
    output.reserve(static_cast<std::size_t>(std::max<Py_ssize_t>(count, 0)));
    for (Py_ssize_t index = 0; index < count; ++index) {
      PyObject *item = PySequence_Fast_GET_ITEM(sequence, index);
      PyObject *pair =
          PySequence_Fast(item, "each span must be a (pointer, nbytes) pair");
      if (pair == nullptr) {
        Py_DECREF(sequence);
        return false;
      }
      if (PySequence_Fast_GET_SIZE(pair) != 2) {
        Py_DECREF(pair);
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError,
                        "each span must contain exactly pointer and nbytes");
        return false;
      }
      InputSpan span{};
      const bool valid =
          py_u64(PySequence_Fast_GET_ITEM(pair, 0), span.pointer) &&
          py_u64(PySequence_Fast_GET_ITEM(pair, 1), span.nbytes);
      Py_DECREF(pair);
      if (!valid) {
        Py_DECREF(sequence);
        return false;
      }
      output.push_back(span);
    }
  } catch (...) {
    Py_DECREF(sequence);
    throw;
  }
  Py_DECREF(sequence);
  return true;
}

bool parse_ipc_inputs(PyObject *object, std::vector<IpcInput> &output) {
  PyObject *sequence = PySequence_Fast(object, "IPC inputs must be a sequence");
  if (sequence == nullptr) {
    return false;
  }
  const Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
  try {
    output.reserve(static_cast<std::size_t>(std::max<Py_ssize_t>(count, 0)));
    for (Py_ssize_t index = 0; index < count; ++index) {
      PyObject *item = PySequence_Fast_GET_ITEM(sequence, index);
      PyObject *fields = PySequence_Fast(
          item,
          "each IPC input must contain handle, nbytes, seg_off and t_off");
      if (fields == nullptr) {
        Py_DECREF(sequence);
        return false;
      }
      if (PySequence_Fast_GET_SIZE(fields) != 4) {
        Py_DECREF(fields);
        Py_DECREF(sequence);
        PyErr_SetString(
            PyExc_ValueError,
            "each IPC input must contain handle, nbytes, seg_off and t_off");
        return false;
      }
      IpcInput input{};
      PyObject *handle = PySequence_Fast_GET_ITEM(fields, 0);
      char *handle_data = nullptr;
      Py_ssize_t handle_size = 0;
      if (PyBytes_AsStringAndSize(handle, &handle_data, &handle_size) < 0) {
        Py_DECREF(fields);
        Py_DECREF(sequence);
        return false;
      }
      if (handle_size != static_cast<Py_ssize_t>(kIpcHandleBytes)) {
        Py_DECREF(fields);
        Py_DECREF(sequence);
        PyErr_Format(PyExc_ValueError, "IPC handle must be %zu bytes, got %zd",
                     kIpcHandleBytes, handle_size);
        return false;
      }
      std::memcpy(input.handle.data(), handle_data, kIpcHandleBytes);
      const bool valid =
          py_u64(PySequence_Fast_GET_ITEM(fields, 1), input.nbytes) &&
          py_u64(PySequence_Fast_GET_ITEM(fields, 2), input.segment_offset) &&
          py_u64(PySequence_Fast_GET_ITEM(fields, 3), input.tensor_offset);
      Py_DECREF(fields);
      if (!valid) {
        Py_DECREF(sequence);
        return false;
      }
      output.push_back(input);
    }
  } catch (...) {
    Py_DECREF(sequence);
    throw;
  }
  Py_DECREF(sequence);
  return true;
}

bool parse_signing(PyObject *timestamp_object, PyObject *model_object,
                   std::array<unsigned char, kTimestampBytes> &timestamp,
                   std::string &model, bool &signing) {
  timestamp.fill(0);
  signing = model_object != Py_None;
  if (!signing) {
    if (timestamp_object != Py_None) {
      PyErr_SetString(PyExc_ValueError,
                      "timestamp must be None when signing is disabled");
      return false;
    }
    return true;
  }
  if (!PyBytes_Check(timestamp_object) || !PyBytes_Check(model_object)) {
    PyErr_SetString(PyExc_TypeError, "timestamp and model must be bytes");
    return false;
  }
  char *timestamp_data = nullptr;
  Py_ssize_t timestamp_size = 0;
  if (PyBytes_AsStringAndSize(timestamp_object, &timestamp_data,
                              &timestamp_size) < 0) {
    return false;
  }
  if (timestamp_size != static_cast<Py_ssize_t>(kTimestampBytes)) {
    PyErr_Format(PyExc_ValueError, "timestamp must be %zu bytes, got %zd",
                 kTimestampBytes, timestamp_size);
    return false;
  }
  std::memcpy(timestamp.data(), timestamp_data, kTimestampBytes);
  char *model_data = nullptr;
  Py_ssize_t model_size = 0;
  if (PyBytes_AsStringAndSize(model_object, &model_data, &model_size) < 0) {
    return false;
  }
  model.assign(model_data, static_cast<std::size_t>(model_size));
  return true;
}

PyObject *result_to_python(FusedResult &&result) {
  PyObject *roots = PyBytes_FromStringAndSize(
      reinterpret_cast<const char *>(result.roots.data()),
      static_cast<Py_ssize_t>(result.roots.size()));
  if (roots == nullptr) {
    return nullptr;
  }
  PyObject *model_root = PyBytes_FromStringAndSize(
      reinterpret_cast<const char *>(result.model_root.data()),
      static_cast<Py_ssize_t>(result.model_root.size()));
  if (model_root == nullptr) {
    Py_DECREF(roots);
    return nullptr;
  }
  PyObject *receipt;
  if (result.signing) {
    receipt = PyUnicode_DecodeUTF8(
        result.receipt.data(), static_cast<Py_ssize_t>(result.receipt.size()),
        "strict");
  } else {
    receipt = Py_NewRef(Py_None);
  }
  if (receipt == nullptr) {
    Py_DECREF(roots);
    Py_DECREF(model_root);
    return nullptr;
  }
  PyObject *tuple = PyTuple_New(3);
  if (tuple == nullptr) {
    Py_DECREF(roots);
    Py_DECREF(model_root);
    Py_DECREF(receipt);
    return nullptr;
  }
  PyTuple_SET_ITEM(tuple, 0, roots);
  PyTuple_SET_ITEM(tuple, 1, model_root);
  PyTuple_SET_ITEM(tuple, 2, receipt);
  return tuple;
}

PyObject *native_error = nullptr;
PyObject *ipc_close_error = nullptr;

typedef struct {
  PyObject_HEAD FusedRunner *runner;
} PyFusedRunner;

PyObject *runner_new(PyTypeObject *type, PyObject *, PyObject *) {
  PyFusedRunner *self =
      reinterpret_cast<PyFusedRunner *>(type->tp_alloc(type, 0));
  if (self != nullptr) {
    self->runner = nullptr;
  }
  return reinterpret_cast<PyObject *>(self);
}

int runner_init(PyFusedRunner *self, PyObject *args, PyObject *kwargs) {
  static const char *names[] = {
      "measure_function", "attest_function", "grid_limit",    "p256_context",
      "cubin_digest",     "kernel_digest",   "device_ordinal", "stream",
      "async_measure_function", "async_grid_limit", "async_min_bytes", nullptr};
  unsigned long long measure_function = 0;
  unsigned long long attest_function = 0;
  unsigned long long p256_context = 0;
  unsigned long long cubin_digest = 0;
  unsigned long long kernel_digest = 0;
  unsigned long long stream = 0;
  unsigned long long async_measure_function = 0, async_min_bytes = 0;
  int async_grid_limit = 0;
  int grid_limit = 0;
  int device_ordinal = 0;
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "KKiKKKiK|KiK:FusedRunner", const_cast<char **>(names),
          &measure_function, &attest_function, &grid_limit, &p256_context,
          &cubin_digest, &kernel_digest, &device_ordinal, &stream,
          &async_measure_function, &async_grid_limit, &async_min_bytes)) {
    return -1;
  }
  try {
    if (self->runner != nullptr) {
      PyErr_SetString(native_error,
                      "native fused runner cannot be initialized twice");
      return -1;
    }
    self->runner = new FusedRunner(measure_function, attest_function, grid_limit,
                                   p256_context, cubin_digest, kernel_digest,
                                   device_ordinal, stream, async_measure_function,
                                   async_grid_limit, async_min_bytes);
    return 0;
  } catch (const std::exception &error) {
    PyErr_SetString(native_error, error.what());
    return -1;
  }
}

void runner_dealloc(PyFusedRunner *self) {
  // Notary.close() releases cached allocations while its CUDA context is
  // current. Do not issue CUDA calls from tp_dealloc: Python finalization may
  // run after that context has already been destroyed.
  delete self->runner;
  self->runner = nullptr;
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

template <typename Callable> PyObject *run_without_gil(Callable &&callable) {
  std::unique_ptr<FusedResult> result;
  std::exception_ptr failure;
  Py_BEGIN_ALLOW_THREADS try {
    result = std::make_unique<FusedResult>(callable());
  } catch (...) {
    failure = std::current_exception();
  }
  Py_END_ALLOW_THREADS if (failure != nullptr) {
    try {
      std::rethrow_exception(failure);
    } catch (const IpcCloseFailure &error) {
      PyErr_Format(ipc_close_error, "%s (rc=%d)", error.what(), error.result());
    } catch (const std::exception &error) {
      PyErr_SetString(native_error, error.what());
    } catch (...) {
      PyErr_SetString(native_error, "unknown native host failure");
    }
    return nullptr;
  }
  return result_to_python(std::move(*result));
}

PyObject *runner_run_spans(PyFusedRunner *self, PyObject *args) {
  PyObject *spans_object = nullptr;
  PyObject *timestamp_object = nullptr;
  PyObject *model_object = nullptr;
  PyObject *instance_object = nullptr;
  if (!PyArg_ParseTuple(args, "OOOO:run_spans", &spans_object,
                        &timestamp_object, &model_object, &instance_object)) {
    return nullptr;
  }
  if (self->runner == nullptr) {
    PyErr_SetString(native_error, "native fused runner is uninitialized");
    return nullptr;
  }
  std::vector<InputSpan> spans;
  std::array<unsigned char, kTimestampBytes> timestamp{};
  std::array<unsigned char, kInstanceRootBytes> instance_root{};
  std::string model;
  bool signing = false;
  try {
    if (!parse_spans(spans_object, spans) ||
        !parse_signing(timestamp_object, model_object, timestamp, model,
                       signing)) {
      return nullptr;
    }
    // A signed request must carry the residency fold; an unsigned measurement
    // never reaches the statement builders, so zeros are never serialized.
    if (signing) {
      char *instance_bytes = nullptr;
      Py_ssize_t instance_length = 0;
      if (PyBytes_AsStringAndSize(instance_object, &instance_bytes,
                                  &instance_length) != 0) {
        return nullptr;
      }
      if (instance_length != static_cast<Py_ssize_t>(kInstanceRootBytes)) {
        PyErr_SetString(native_error, "instance_root must be 32 bytes");
        return nullptr;
      }
      std::memcpy(instance_root.data(), instance_bytes, kInstanceRootBytes);
    }

  } catch (const std::exception &error) {
    PyErr_SetString(native_error, error.what());
    return nullptr;
  }
  return run_without_gil([&] {
    return self->runner->run_spans(spans, timestamp, instance_root, model,
                                   signing);
  });
}

PyObject *runner_run_ipc(PyFusedRunner *self, PyObject *args) {
  PyObject *inputs_object = nullptr;
  PyObject *timestamp_object = nullptr;
  PyObject *model_object = nullptr;
  PyObject *instance_object = nullptr;
  if (!PyArg_ParseTuple(args, "OOOO:run_ipc", &inputs_object, &timestamp_object,
                        &model_object, &instance_object)) {
    return nullptr;
  }
  if (self->runner == nullptr) {
    PyErr_SetString(native_error, "native fused runner is uninitialized");
    return nullptr;
  }
  std::vector<IpcInput> inputs;
  std::array<unsigned char, kTimestampBytes> timestamp{};
  std::array<unsigned char, kInstanceRootBytes> instance_root{};
  std::string model;
  bool signing = false;
  try {
    if (!parse_ipc_inputs(inputs_object, inputs) ||
        !parse_signing(timestamp_object, model_object, timestamp, model,
                       signing)) {
      return nullptr;
    }
    // A signed request must carry the residency fold; an unsigned measurement
    // never reaches the statement builders, so zeros are never serialized.
    if (signing) {
      char *instance_bytes = nullptr;
      Py_ssize_t instance_length = 0;
      if (PyBytes_AsStringAndSize(instance_object, &instance_bytes,
                                  &instance_length) != 0) {
        return nullptr;
      }
      if (instance_length != static_cast<Py_ssize_t>(kInstanceRootBytes)) {
        PyErr_SetString(native_error, "instance_root must be 32 bytes");
        return nullptr;
      }
      std::memcpy(instance_root.data(), instance_bytes, kInstanceRootBytes);
    }
  } catch (const std::exception &error) {
    PyErr_SetString(native_error, error.what());
    return nullptr;
  }
  return run_without_gil(
      [&] {
        return self->runner->run_ipc(inputs, timestamp, instance_root, model,
                                     signing);
      });
}

PyObject *runner_close(PyFusedRunner *self, PyObject *) {
  if (self->runner != nullptr) {
    Py_BEGIN_ALLOW_THREADS self->runner->close();
    Py_END_ALLOW_THREADS
  }
  Py_RETURN_NONE;
}

PyObject *runner_capacities(PyFusedRunner *self, void *) {
  if (self->runner == nullptr) {
    PyErr_SetString(native_error, "native fused runner is uninitialized");
    return nullptr;
  }
  const auto values = self->runner->capacities();
  return Py_BuildValue("(KKKK)", static_cast<unsigned long long>(values[0]),
                       static_cast<unsigned long long>(values[1]),
                       static_cast<unsigned long long>(values[2]),
                       static_cast<unsigned long long>(values[3]));
}

PyObject *runner_ipc_cleanup_failed(PyFusedRunner *self, void *) {
  if (self->runner == nullptr) {
    PyErr_SetString(native_error, "native fused runner is uninitialized");
    return nullptr;
  }
  return PyBool_FromLong(self->runner->ipc_cleanup_failed());
}

PyObject *runner_context_cleanup_required(PyFusedRunner *self, void *) {
  if (self->runner == nullptr) {
    PyErr_SetString(native_error, "native fused runner is uninitialized");
    return nullptr;
  }
  return PyBool_FromLong(self->runner->context_cleanup_required());
}

PyMethodDef runner_methods[] = {
    {"run_spans", reinterpret_cast<PyCFunction>(runner_run_spans), METH_VARARGS,
     PyDoc_STR(
         "run_spans(spans, timestamp, model, instance_root) -> "
         "(roots, model_root, receipt)")},
    {"run_ipc", reinterpret_cast<PyCFunction>(runner_run_ipc), METH_VARARGS,
     PyDoc_STR(
         "run_ipc(inputs, timestamp, model, instance_root) -> "
         "(roots, model_root, receipt)")},
    {"close", reinterpret_cast<PyCFunction>(runner_close), METH_NOARGS,
     PyDoc_STR("Release reusable CUDA workspaces while the owning context is "
               "current.")},
    {nullptr, nullptr, 0, nullptr},
};

PyGetSetDef runner_getset[] = {
    {const_cast<char *>("capacities"),
     reinterpret_cast<getter>(runner_capacities), nullptr,
     const_cast<char *>("Allocated metadata/work/output capacities in bytes."),
     nullptr},
    {const_cast<char *>("ipc_cleanup_failed"),
     reinterpret_cast<getter>(runner_ipc_cleanup_failed), nullptr,
     const_cast<char *>("Compatibility alias for context_cleanup_required."),
     nullptr},
    {const_cast<char *>("context_cleanup_required"),
     reinterpret_cast<getter>(runner_context_cleanup_required), nullptr,
     const_cast<char *>("Whether context destruction is required before CUDA "
                        "allocations may be released."),
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

PyType_Slot runner_slots[] = {
    {Py_tp_doc, const_cast<char *>("Native CUDA fused-request runner.")},
    {Py_tp_new, reinterpret_cast<void *>(runner_new)},
    {Py_tp_init, reinterpret_cast<void *>(runner_init)},
    {Py_tp_dealloc, reinterpret_cast<void *>(runner_dealloc)},
    {Py_tp_methods, runner_methods},
    {Py_tp_getset, runner_getset},
    {0, nullptr},
};

PyType_Spec runner_spec = {
    "cuattest._native.FusedRunner",           sizeof(PyFusedRunner), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, runner_slots,
};

PyObject *module_plan(PyObject *, PyObject *object) {
  std::vector<InputSpan> spans;
  try {
    if (!parse_spans(object, spans)) {
      return nullptr;
    }
    FusedPlan plan = make_plan(spans);
    PyObject *descriptors = PyBytes_FromStringAndSize(
        reinterpret_cast<const char *>(plan.descriptors.data()),
        static_cast<Py_ssize_t>(plan.descriptors.size()));
    PyObject *offsets = PyBytes_FromStringAndSize(
        reinterpret_cast<const char *>(plan.reduction_offsets.data()),
        static_cast<Py_ssize_t>(plan.reduction_offsets.size()));
    if (descriptors == nullptr || offsets == nullptr) {
      Py_XDECREF(descriptors);
      Py_XDECREF(offsets);
      return nullptr;
    }
    PyObject *tuple = Py_BuildValue(
        "NNKKi", descriptors, offsets,
        static_cast<unsigned long long>(plan.total_tiles),
        static_cast<unsigned long long>(plan.secondary_tiles), plan.levels);
    return tuple;
  } catch (const std::exception &error) {
    PyErr_SetString(native_error, error.what());
    return nullptr;
  }
}

PyObject *module_export_cuda_allocation(PyObject *, PyObject *pointer_object) {
  const unsigned long long storage_pointer =
      PyLong_AsUnsignedLongLong(pointer_object);
  if (PyErr_Occurred()) {
    return nullptr;
  }
  if (storage_pointer == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "cannot export a null CUDA storage pointer");
    return nullptr;
  }

  try {
    // Export the driver allocation itself instead of calling PyTorch's private
    // _share_cuda_ method.  That method mutates StorageImpl ownership before
    // several fallible Python-object and CUDA-event operations, so a failure
    // can expose neither a result nor enough state to roll the mutation back.
    // cuIpcGetMemHandle is observational: if a later Python allocation fails,
    // no producer-side counter or storage ownership transition needs cleanup.
    static CudaApi api;
    // VMM-backed expandable segments and stream-ordered pool allocations
    // need a different IPC transport, not a legacy 64-byte memory handle.
    // Query the allocation itself: environment settings may have changed
    // since allocation, and mixed pools can coexist in the same process.
    // Do this before legacy range/export calls so unsupported storage gets a
    // useful diagnosis, while failed attribute queries retain the CUDA error.
    unsigned int legacy_ipc_capable = 0;
    api.check(api.pointer_get_attribute(&legacy_ipc_capable,
                                        kPointerAttributeIsLegacyCudaIpcCapable,
                                        storage_pointer),
              "cuPointerGetAttribute(client export IPC capability)");
    if (!legacy_ipc_capable) {
      throw NativeFailure(
          "CUDA allocation does not support legacy CUDA IPC; cuAttest cannot "
          "export expandable_segments/VMM or cudaMallocAsync storage. Restart "
          "the producer with PYTORCH_CUDA_ALLOC_CONF="
          "backend:native,expandable_segments:False (and the same value for "
          "PYTORCH_ALLOC_CONF if set), before importing torch and allocating "
          "the model. Changing settings does not convert existing allocations.");
    }
    CUdeviceptr allocation_base = 0;
    std::size_t allocation_bytes = 0;
    api.check(api.mem_get_address_range(&allocation_base, &allocation_bytes,
                                        storage_pointer),
              "cuMemGetAddressRange(client export)");
    if (storage_pointer < allocation_base ||
        storage_pointer - allocation_base > allocation_bytes) {
      throw NativeFailure("CUDA driver returned an invalid allocation range");
    }

    int allocation_device = -1;
    api.check(api.pointer_get_attribute(&allocation_device,
                                        kPointerAttributeDeviceOrdinal,
                                        allocation_base),
              "cuPointerGetAttribute(client export device)");
    CUipcMemHandle handle{};
    api.check(api.ipc_get(&handle, allocation_base),
              "cuIpcGetMemHandle(client export)");

    return Py_BuildValue(
        "y#KKi", reinterpret_cast<const char *>(handle.reserved),
        static_cast<Py_ssize_t>(sizeof(handle.reserved)),
        static_cast<unsigned long long>(allocation_base),
        static_cast<unsigned long long>(allocation_bytes), allocation_device);
  } catch (const std::exception &error) {
    PyErr_SetString(native_error, error.what());
    return nullptr;
  }
}

PyMethodDef module_methods[] = {
    {"plan", reinterpret_cast<PyCFunction>(module_plan), METH_O,
     PyDoc_STR("Build the fused-kernel descriptor and reduction schedule.")},
    {"_export_cuda_allocation",
     reinterpret_cast<PyCFunction>(module_export_cuda_allocation), METH_O,
     PyDoc_STR("Export the driver allocation containing a CUDA storage pointer.")},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native host orchestration for cuAttest.",
    -1,
    module_methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

} // namespace

PyMODINIT_FUNC PyInit__native() {
  PyObject *module = PyModule_Create(&module_definition);
  if (module == nullptr) {
    return nullptr;
  }

  native_error =
      PyErr_NewException("cuattest._native.Error", PyExc_RuntimeError, nullptr);
  if (native_error == nullptr ||
      PyModule_AddObjectRef(module, "Error", native_error) < 0) {
    Py_XDECREF(native_error);
    Py_DECREF(module);
    return nullptr;
  }
  ipc_close_error = PyErr_NewException("cuattest._native.IpcCloseError",
                                       native_error, nullptr);
  if (ipc_close_error == nullptr ||
      PyModule_AddObjectRef(module, "IpcCloseError", ipc_close_error) < 0) {
    Py_XDECREF(ipc_close_error);
    Py_DECREF(module);
    return nullptr;
  }

  PyObject *runner_type = PyType_FromSpec(&runner_spec);
  if (runner_type == nullptr ||
      PyModule_AddObject(module, "FusedRunner", runner_type) < 0) {
    Py_XDECREF(runner_type);
    Py_DECREF(module);
    return nullptr;
  }
  if (PyModule_AddStringConstant(module, "BACKEND", "C++") < 0 ||
      PyModule_AddStringConstant(module, "BUILD_TYPE", CUATTEST_BUILD_TYPE) < 0 ||
      PyModule_AddIntConstant(module, "ASSERTIONS_ENABLED",
#ifdef NDEBUG
                              0
#else
                              1
#endif
                              ) < 0) {
    Py_DECREF(module);
    return nullptr;
  }
  return module;
}

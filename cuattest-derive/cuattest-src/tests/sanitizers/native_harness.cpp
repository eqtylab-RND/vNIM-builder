// SPDX-License-Identifier: Apache-2.0
// Compile the verbatim native core extracted by build_native.py. The CPython
// bindings are tested separately in the instrumented extension, not simulated.
// This does NOT execute GPU instructions or emulate cryptographic results.
#include "native_core.hpp"

#include <cassert>
#include <cstdio>
#include <thread>

extern "C" void audit_driver_reset(int mode);
extern "C" int audit_driver_live_allocations();
extern "C" int audit_driver_open_mappings();
extern "C" void audit_driver_finish_work();
extern "C" uintptr_t audit_driver_measure_function();
extern "C" unsigned audit_driver_measure_grid();

template <class Function> void expect_failure(Function function) {
  bool rejected = false;
  try {
    function();
  } catch (const NativeFailure &) {
    rejected = true;
  } catch (const IpcCloseFailure &) {
    rejected = true;
  }
  assert(rejected);
}

void exercise(unsigned worker) {
  std::array<unsigned char, kTimestampBytes> timestamp{};
  std::memcpy(timestamp.data(), "2026-09-07T00:00:00Z", timestamp.size());
  for (unsigned iteration = 0; iteration < 100; ++iteration) {
    std::vector<InputSpan> spans;
    const std::uint64_t boundaries[] = {
        0, 1, 63, 64, 65, 1023, 1024, 1025, 131071, 131072, 131073,
        257 * 1024 + 1, 96ull * 1024 * 1024 * 1024};
    for (std::size_t i = 0; i < 64 + iteration; ++i) {
      spans.push_back({0x1000 + i * 4096,
                       boundaries[(i + worker) % std::size(boundaries)]});
    }
    auto plan = make_plan(spans);
    assert(plan.descriptors.size() == spans.size() * kDescriptorBytes);
    assert(plan.total_tiles > 0 && plan.levels > 0);
    assert(plan.reduction_offsets.size() == (spans.size() + 1) * plan.levels * 8);
  }
  expect_failure([] { make_plan({}); });
  expect_failure([] {
    make_plan(std::vector<InputSpan>(4096, {1, UINT64_MAX}));
  });
  expect_failure([] { checked_add(UINT64_MAX, 1, "overflow"); });
  expect_failure([] { checked_size(UINT64_MAX, 32, "overflow"); });

  audit_driver_reset(0);
  {
    // Inclusive aggregate-byte dispatch, not per-tensor size; each entry has
    // its own cooperative residency ceiling. Both use the same workspaces.
    FusedRunner runner(1, 2, 16, 3, 4, 5, 0, 6, 7, 3, 262144);
    expect_failure([&] {
      runner.run_spans({{1, UINT64_MAX / 2}, {1, UINT64_MAX / 2 + 2}}, timestamp, "audit", true);
    });
    assert(audit_driver_live_allocations() == 0);
    for (std::uint64_t bytes : {262143ull, 262144ull, 16ull * 1024 * 1024}) {
      auto result = runner.run_spans({{0x1000, bytes / 2}, {0x2000, bytes - bytes / 2}},
                                    timestamp, "audit", true);
      assert(result.roots.size() == 64);
      assert(audit_driver_measure_function() == (bytes < 262144 ? 1 : 7));
      assert(audit_driver_measure_grid() <= (bytes < 262144 ? 16 : 3));
      if (bytes == 16ull * 1024 * 1024) assert(audit_driver_measure_grid() == 3);
    }
    runner.close();
    assert(audit_driver_live_allocations() == 0);
  }

  for (int mode : {0, 1, 2, 4, 8, 16, 32, 64, 256}) {
    audit_driver_reset(mode);
    FusedRunner runner(1, 2, 16, 3, 4, 5, 0, 6);
    std::vector<IpcInput> inputs(12);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      // Three distinct handles, each repeated, exercise mapping deduplication
      // and string_view keys whose lifetime belongs to this inputs vector.
      inputs[i].handle.fill(static_cast<unsigned char>(i % 3));
      inputs[i].nbytes = i * 17 + 1;
      inputs[i].segment_offset = i;
      inputs[i].tensor_offset = i + 1;
    }
    if (mode == 0) {
      for (unsigned size : {1, 12, 128, 2}) {
        auto result = runner.run_spans(
            std::vector<InputSpan>(size, {0x1000, 257 * 1024 + 1}),
            timestamp, "audit", true);
        assert(result.roots.size() == size * 32 && result.receipt == "{}");
      }
      auto result = runner.run_ipc(inputs, timestamp, "", false);
      assert(result.roots.size() == inputs.size() * 32 && result.receipt.empty());
      inputs[0].nbytes = UINT64_MAX;
      expect_failure([&] { runner.run_ipc(inputs, timestamp, "audit", true); });
      inputs[0].nbytes = 1;
      inputs[0].segment_offset = UINT64_MAX;
      expect_failure([&] { runner.run_ipc(inputs, timestamp, "audit", true); });
      expect_failure([&] { runner.run_spans({{1, 1}}, timestamp, "", true); });
      expect_failure([&] {
        runner.run_spans({{1, 1}}, timestamp, std::string(65, 'a'), true);
      });
    } else {
      expect_failure([&] { runner.run_ipc(inputs, timestamp, "audit", true); });
    }
    // A simulated context teardown owns abandoned allocations/mappings only
    // for unconfirmed synchronization or failed IPC close, like the real host.
    const bool poisoned = mode == 2 || mode == 4;
    assert(runner.context_cleanup_required() == poisoned);
    runner.close();
    runner.close();
    expect_failure([&] { runner.run_spans({{1, 1}}, timestamp, "", false); });
    if (!poisoned) {
      assert(audit_driver_live_allocations() == 0);
      assert(audit_driver_open_mappings() == 0);
    }
    // Mock context destruction completes DMA before reclaiming BOTH device and
    // cuMemAllocHost allocations. No explicit free is valid after destruction.
    audit_driver_finish_work();
    audit_driver_reset(0);
  }
}

void exercise_parallel_ipc() {
  audit_driver_reset(128);
  std::array<unsigned char, kTimestampBytes> timestamp{};
  FusedRunner runner(1, 2, 16, 3, 4, 5, 0, 6);
  std::vector<IpcInput> inputs(3);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    inputs[i].handle.fill(static_cast<unsigned char>(i));
    inputs[i].nbytes = 1025;
  }
  // The driver's synchronization barrier requires all eight runners to get
  // past their import batches. Accidentally extending the IPC lock through
  // launch or synchronize must fail deterministically, not just run slowly.
  auto result = runner.run_ipc(inputs, timestamp, "", false);
  assert(result.roots.size() == inputs.size() * 32);
  assert(audit_driver_open_mappings() == 0);
  runner.close();
  assert(audit_driver_live_allocations() == 0);
}

int main() {
  // Production serializes a given runner; independent sessions may run at
  // once. Do not invent unsupported concurrent access to the same runner.
  std::vector<std::thread> workers;
  for (unsigned i = 0; i < 8; ++i) workers.emplace_back(exercise, i);
  for (auto &worker : workers) worker.join();
  workers.clear();
  for (unsigned i = 0; i < 8; ++i) workers.emplace_back(exercise_parallel_ipc);
  for (auto &worker : workers) worker.join();
  std::puts("native sanitizer harness: 8 concurrent sessions passed");
}

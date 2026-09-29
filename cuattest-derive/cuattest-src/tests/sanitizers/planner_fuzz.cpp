// SPDX-License-Identifier: Apache-2.0
// Fuzz the actual checked planner, not a duplicate implementation.
#include "native_core.hpp"
#include <cassert>

extern "C" int LLVMFuzzerTestOneInput(const unsigned char *data, std::size_t size) {
  std::vector<InputSpan> spans;
  while (size >= 16 && spans.size() < 128) {
    InputSpan span{};
    std::memcpy(&span.pointer, data, 8);
    std::memcpy(&span.nbytes, data + 8, 8);
    spans.push_back(span);
    data += 16;
    size -= 16;
  }
  try {
    auto plan = make_plan(spans);
    assert(!spans.empty());
    assert(plan.descriptors.size() == spans.size() * kDescriptorBytes);
    assert(plan.total_tiles >= spans.size());
    assert(plan.levels >= 0 && plan.levels < 64);
    assert(plan.reduction_offsets.size() ==
           (plan.levels ? (spans.size() + 1) * plan.levels * 8 : 8));
  } catch (const NativeFailure &) {
    // Empty requests and arithmetic overflows are deliberate validation paths.
  }
  return 0;
}

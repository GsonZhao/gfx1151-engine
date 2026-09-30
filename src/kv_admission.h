#pragma once

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

struct KvAdmissionState {
  bool busy = false;
  int mapped = 0;
  int reserved = 0;
};

inline bool kv_admission_decode_cap(const char* text, int& cap) {
  if (!text) {
    cap = 4096;
    return true;
  }
  char* end = nullptr;
  errno = 0;
  const long long value = std::strtoll(text, &end, 10);
  if (errno || end == text || *end || value < 0 || value > INT_MAX) return false;
  cap = (int)value;
  return true;
}

inline int kv_admission_target(size_t prompt_tokens, int max_tokens,
                               int decode_cap, int maxctx, int page_tokens) {
  const int64_t prompt = (int64_t)std::min(prompt_tokens, (size_t)maxctx);
  const int64_t decode = std::max(0, std::min(max_tokens, decode_cap));
  const int64_t end = std::min<int64_t>(maxctx, prompt + decode);
  return (int)((end + page_tokens - 1) / page_tokens);
}

inline int kv_admission_need(int target, int mapped, bool continuation) {
  return continuation ? std::max(target, mapped) : target;
}

inline int64_t kv_admission_available(int pool_pages,
                                      const std::vector<KvAdmissionState>& slots,
                                      size_t chosen) {
  int64_t available = pool_pages;
  for (size_t index = 0; index < slots.size(); index++)
    if (index != chosen && slots[index].busy)
      available -= std::max(slots[index].reserved, slots[index].mapped);
  return available;
}

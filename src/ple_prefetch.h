#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ple_prefetch {

inline int chunk_size(int remaining, int batch, int capacity) {
  return remaining > batch && remaining <= capacity
             ? remaining : std::min(batch, remaining);
}

struct Hash {
  std::array<int64_t, 3> mult{};
  std::array<int64_t, 16> sizes{}, offsets{};
  int64_t pad = 0;
  uint64_t table = 0;
  unsigned row_bytes = 160;
};

class Queue {
 public:
  using Clock = std::chrono::steady_clock;
  using Reader = std::function<void(const uint64_t*, uint8_t*, size_t)>;
  using Reuse = std::function<void(size_t)>;
  struct Timing {
    Clock::time_point start, done;
  };

  Queue(const std::vector<int64_t>& prefix, const std::vector<int>& tokens,
        Hash hash, std::array<uint8_t*, 2> buffers, Reader reader,
        Reuse reuse = {})
      : history_(prefix), base_(prefix.size()), hash_(hash),
        reader_(std::move(reader)), reuse_(std::move(reuse)) {
    history_.insert(history_.end(), tokens.begin(), tokens.end());
    for (size_t slot = 0; slot < slots_.size(); slot++)
      slots_[slot].data = buffers[slot];
  }

  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;
  ~Queue() {
    for (auto& slot : slots_)
      if (slot.task.valid()) slot.task.wait();
  }

  bool pending(size_t slot) const { return slots_.at(slot).task.valid(); }

  void start(size_t slot_index, int offset, int count) {
    Slot& slot = slots_.at(slot_index);
    for (const auto& pending_slot : slots_)
      if (pending_slot.task.valid())
        throw std::logic_error("PLE read is still pending");
    if (offset < 0 || count <= 0 ||
        (size_t)offset + (size_t)count > history_.size() - base_)
      throw std::out_of_range("PLE prefetch range");
    if (reuse_) reuse_(slot_index);
    slot.offset = offset;
    slot.count = count;
    slot.timing.start = Clock::now();
    slot.task = std::async(std::launch::async, [this, slot_index] {
      Slot& current = slots_[slot_index];
      std::vector<uint64_t> addrs((size_t)current.count * 16);
      for (int token = 0; token < current.count; token++) {
        const size_t position = base_ + current.offset + token;
        auto cat = [&](size_t back) -> int64_t {
          return position < back ? hash_.pad : history_[position - back];
        };
        for (size_t head = 0; head < 16; head++) {
          int64_t mix = cat(0) * hash_.mult[0] ^ (cat(1) * hash_.mult[1]);
          if (head >= 8) mix ^= cat(2) * hash_.mult[2];
          uint64_t row = (uint64_t)(hash_.offsets[head] +
                                   mix % hash_.sizes[head]);
          addrs[(size_t)token * 16 + head] =
              hash_.table + row * hash_.row_bytes;
        }
      }
      reader_(addrs.data(), current.data, addrs.size());
      current.timing.done = Clock::now();
    });
  }

  Timing wait(size_t slot_index, int offset, int count) {
    Slot& slot = slots_.at(slot_index);
    if (!slot.task.valid() || slot.offset != offset || slot.count != count)
      throw std::logic_error("PLE prefetch chunk mismatch");
    slot.task.get();
    return slot.timing;
  }

 private:
  struct Slot {
    uint8_t* data = nullptr;
    int offset = -1, count = 0;
    Timing timing{};
    std::future<void> task;
  };
  std::vector<int64_t> history_;
  size_t base_;
  Hash hash_;
  Reader reader_;
  Reuse reuse_;
  std::array<Slot, 2> slots_;
};

}

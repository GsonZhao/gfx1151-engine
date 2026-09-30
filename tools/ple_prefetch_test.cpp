#include "../src/ple_prefetch.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

static void check(bool condition, const std::string& name) {
  if (!condition) {
    std::fprintf(stderr, "FAIL %s\n", name.c_str());
    std::exit(1);
  }
}

template <typename Exception, typename Function>
static void rejects(Function function, const std::string& name) {
  bool rejected = false;
  try {
    function();
  } catch (const Exception&) {
    rejected = true;
  }
  check(rejected, name);
}

static ple_prefetch::Hash make_hash(unsigned row_bytes) {
  ple_prefetch::Hash hash;
  hash.mult = {7919, -104729, 15485863};
  hash.pad = 248044;
  hash.table = 0x100000000ull;
  hash.row_bytes = row_bytes;
  for (size_t head = 0; head < 16; head++) {
    hash.sizes[head] = 127 + (int64_t)head * 2;
    hash.offsets[head] = 10000 + (int64_t)head * 1000;
  }
  return hash;
}

static std::vector<uint64_t> reference(const std::vector<int64_t>& prefix,
                                     const std::vector<int>& tokens, int offset,
                                     int count, const ple_prefetch::Hash& hash) {
  std::vector<int64_t> history(prefix);
  history.insert(history.end(), tokens.begin(), tokens.end());
  std::vector<uint64_t> addrs((size_t)count * 16);
  for (int token = 0; token < count; token++) {
    const int position = (int)prefix.size() + offset + token;
    auto cat = [&](int back) -> int64_t {
      int previous = position - back;
      return previous < 0 ? hash.pad : history[(size_t)previous];
    };
    for (int head = 0; head < 16; head++) {
      int64_t mix = cat(0) * hash.mult[0] ^ (cat(1) * hash.mult[1]);
      if (head >= 8) mix ^= cat(2) * hash.mult[2];
      uint64_t row = (uint64_t)(hash.offsets[head] + mix % hash.sizes[head]);
      addrs[(size_t)token * 16 + head] = hash.table + row * hash.row_bytes;
    }
  }
  return addrs;
}

static void read_rows(const ple_prefetch::Hash& hash, const uint64_t* addrs,
                      uint8_t* staging, size_t rows) {
  for (size_t row = 0; row < rows; row++)
    for (unsigned byte = 0; byte < hash.row_bytes; byte++)
      staging[row * hash.row_bytes + byte] =
          (uint8_t)((addrs[row] - hash.table) / hash.row_bytes + byte);
}

static void test_equivalence(unsigned row_bytes, size_t prefix_count) {
  const auto hash = make_hash(row_bytes);
  std::vector<int64_t> prefix(prefix_count, 411);
  std::vector<int> tokens(21);
  for (size_t token = 0; token < tokens.size(); token++)
    tokens[token] = (int)(token * 173 + 7);
  const auto saved_prefix = prefix;
  const auto saved_tokens = tokens;
  std::array<std::vector<uint8_t>, 2> buffers;
  for (auto& buffer : buffers) buffer.resize(5 * 16 * row_bytes, 0xa5);
  std::atomic<size_t> calls{0};
  ple_prefetch::Queue queue(
      prefix, tokens, hash, {buffers[0].data(), buffers[1].data()},
      [&](const uint64_t* addrs, uint8_t* staging, size_t rows) {
        calls.fetch_add(1);
        read_rows(hash, addrs, staging, rows);
      });
  check(prefix == saved_prefix && tokens == saved_tokens,
        "prefetch does not advance authoritative history");
  prefix.assign(100, 999);
  tokens.assign(100, 555);
  int offset = 0;
  size_t slot = 0, chunks = 0;
  while (offset < (int)saved_tokens.size()) {
    int count = ple_prefetch::chunk_size((int)saved_tokens.size() - offset, 4, 5);
    if (!queue.pending(slot)) queue.start(slot, offset, count);
    auto timing = queue.wait(slot, offset, count);
    check(timing.done >= timing.start, "per-chunk completion timing");
    auto addresses = reference(saved_prefix, saved_tokens, offset, count, hash);
    std::vector<uint8_t> expected((size_t)count * 16 * row_bytes);
    read_rows(hash, addresses.data(), expected.data(), addresses.size());
    check(std::memcmp(expected.data(), buffers[slot].data(), expected.size()) == 0,
          "raw rows match chunk-local gather including prefix and pad");
    if (expected.size() < buffers[slot].size())
      check(buffers[slot][expected.size()] == 0xa5, "no write beyond chunk rows");
    offset += count;
    int next = ple_prefetch::chunk_size((int)saved_tokens.size() - offset, 4, 5);
    if (next > 0) queue.start(1 - slot, offset, next);
    slot = 1 - slot;
    chunks++;
  }
  check(calls == chunks, "each chunk is read exactly once");
}

static void test_overlap() {
  auto hash = make_hash(160);
  std::array<std::vector<uint8_t>, 2> buffers;
  for (auto& buffer : buffers) buffer.resize(16 * hash.row_bytes);
  std::promise<void> entered, release;
  auto entered_future = entered.get_future();
  auto release_future = release.get_future().share();
  std::atomic<int> calls{0};
  ple_prefetch::Queue queue(
      {}, {1, 2}, hash, {buffers[0].data(), buffers[1].data()},
      [&](const uint64_t* addrs, uint8_t* staging, size_t rows) {
        if (calls.fetch_add(1) == 1) {
          entered.set_value();
          release_future.wait();
        }
        read_rows(hash, addrs, staging, rows);
      });
  queue.start(0, 0, 1);
  queue.wait(0, 0, 1);
  auto current = buffers[0];
  queue.start(1, 1, 1);
  check(entered_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
        "next chunk starts before current computation finishes");
  check(buffers[0] == current, "prefetch leaves current staging unchanged");
  rejects<std::logic_error>([&] { queue.start(0, 0, 1); },
                            "one IOCP reader never pumps two batches at once");
  release.set_value();
  queue.wait(1, 1, 1);
}

static void test_reuse_fence() {
  auto hash = make_hash(90);
  std::array<std::vector<uint8_t>, 2> buffers;
  for (auto& buffer : buffers) buffer.resize(16 * hash.row_bytes);
  std::promise<void> reuse_entered, gpu_done;
  auto reuse_future = reuse_entered.get_future();
  auto gpu_future = gpu_done.get_future().share();
  std::atomic<int> calls{0};
  ple_prefetch::Queue queue(
      {}, {1, 2, 3}, hash, {buffers[0].data(), buffers[1].data()},
      [&](const uint64_t* addrs, uint8_t* staging, size_t rows) {
        calls.fetch_add(1);
        read_rows(hash, addrs, staging, rows);
      },
      [&](size_t slot) {
        if (slot == 0 && calls == 2) {
          reuse_entered.set_value();
          gpu_future.wait();
        }
      });
  queue.start(0, 0, 1);
  queue.wait(0, 0, 1);
  auto original = buffers[0];
  queue.start(1, 1, 1);
  queue.wait(1, 1, 1);
  auto reuse_task = std::async(std::launch::async, [&] { queue.start(0, 2, 1); });
  check(reuse_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
        "slot reuse reaches GPU fence");
  check(calls == 2 && buffers[0] == original, "no overwrite before GPU consumption");
  gpu_done.set_value();
  reuse_task.get();
  queue.wait(0, 2, 1);
  check(calls == 3, "slot can be refilled after GPU consumption");
}

static void test_independent_requests() {
  auto hash = make_hash(160);
  std::array<std::array<std::vector<uint8_t>, 2>, 2> buffers;
  std::array<std::promise<void>, 2> entered;
  std::array<std::future<void>, 2> entered_futures;
  std::array<std::unique_ptr<ple_prefetch::Queue>, 2> queues;
  std::promise<void> release;
  auto release_future = release.get_future().share();
  for (size_t request = 0; request < queues.size(); request++) {
    entered_futures[request] = entered[request].get_future();
    for (auto& buffer : buffers[request]) buffer.resize(16 * hash.row_bytes);
    queues[request] = std::make_unique<ple_prefetch::Queue>(
        std::vector<int64_t>{(int64_t)request + 11},
        std::vector<int>{(int)request + 101}, hash,
        std::array<uint8_t*, 2>{buffers[request][0].data(), buffers[request][1].data()},
        [&, request](const uint64_t* addrs, uint8_t* staging, size_t rows) {
          entered[request].set_value();
          release_future.wait();
          read_rows(hash, addrs, staging, rows);
        });
    queues[request]->start(0, 0, 1);
  }
  for (auto& future : entered_futures)
    check(future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "independent requests can have reads in flight together");
  release.set_value();
  for (size_t request = 0; request < queues.size(); request++) {
    queues[request]->wait(0, 0, 1);
    auto addresses = reference({(int64_t)request + 11}, {(int)request + 101}, 0, 1, hash);
    std::vector<uint8_t> expected(16 * hash.row_bytes);
    read_rows(hash, addresses.data(), expected.data(), addresses.size());
    check(buffers[request][0] == expected, "requests keep their own history and staging");
  }
}

static void test_cleanup_and_errors() {
  auto hash = make_hash(160);
  std::vector<uint8_t> buffer(16 * hash.row_bytes);
  std::promise<void> reading, release, cleanup;
  auto reading_future = reading.get_future();
  auto release_future = release.get_future().share();
  auto cleanup_future = cleanup.get_future();
  auto queue = std::make_unique<ple_prefetch::Queue>(
      std::vector<int64_t>{}, std::vector<int>{1}, hash,
      std::array<uint8_t*, 2>{buffer.data(), buffer.data()},
      [&](const uint64_t*, uint8_t*, size_t) {
        reading.set_value();
        release_future.wait();
      });
  queue->start(0, 0, 1);
  check(reading_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
        "pending read has started");
  auto destructor = std::async(std::launch::async, [&, owned = std::move(queue)]() mutable {
    cleanup.set_value();
    owned.reset();
  });
  cleanup_future.wait();
  check(destructor.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
        "cancellation drains IO before releasing staging");
  release.set_value();
  destructor.get();
  ple_prefetch::Queue failing(
      {}, {1}, hash, {buffer.data(), buffer.data()},
      [](const uint64_t*, uint8_t*, size_t) { throw std::runtime_error("read failure"); });
  rejects<std::out_of_range>([&] { failing.start(0, -1, 1); }, "negative offset");
  rejects<std::out_of_range>([&] { failing.start(0, 0, 2); }, "past prompt end");
  rejects<std::out_of_range>([&] { failing.start(0, 0, 0); }, "empty read");
  failing.start(0, 0, 1);
  rejects<std::logic_error>([&] { failing.wait(0, 1, 1); }, "chunk mismatch");
  rejects<std::runtime_error>([&] { failing.wait(0, 0, 1); }, "worker error propagated");
  check(!failing.pending(0), "failed read leaves no joinable task");
}

int main() {
  check(ple_prefetch::chunk_size(0, 8192, 8448) == 0, "empty prompt");
  check(ple_prefetch::chunk_size(8193, 8192, 8448) == 8193, "merged tail");
  check(ple_prefetch::chunk_size(8449, 8192, 8448) == 8192, "unmerged tail");
  check(ple_prefetch::chunk_size(32, 8192, 8448) == 32, "small chunk");
  for (unsigned row_bytes : {160u, 90u})
    for (size_t prefix_count : {0u, 1u, 9u}) test_equivalence(row_bytes, prefix_count);
  test_overlap();
  test_reuse_fence();
  test_independent_requests();
  test_cleanup_and_errors();
  std::puts("PASS PLE prefetch: FP8/IQ4 rows, history, tails, overlap, reuse, cleanup");
  return 0;
}

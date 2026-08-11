#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace franka_control::cpp {

// Bounded single-producer/single-consumer queue for policy actions.
// Complete blocks become visible with one release store, so the realtime
// consumer cannot observe a partially enqueued block. clear() and pop()
// coordinate through read_head_, allowing clear while control is running.
template <std::size_t ActionDimension, std::size_t Capacity>
class SpscActionQueue {
  static_assert(Capacity > 0, "SpscActionQueue capacity must be positive");
  static_assert(std::atomic<double>::is_always_lock_free,
                "realtime action queue requires lock-free double atomics");

 public:
  using Action = std::array<double, ActionDimension>;

  static constexpr std::size_t capacity() { return Capacity; }

  bool tryPush(const Action& value) { return tryPushBlock(&value, 1); }

  bool tryPushBlock(const Action* values, std::size_t count) {
    if (count == 0) return true;
    if (count > Capacity) return false;

    const uint64_t write = write_head_.load(std::memory_order_relaxed);
    const uint64_t read = read_head_.load(std::memory_order_acquire);
    if (write - read + count > Capacity) return false;

    for (std::size_t i = 0; i < count; ++i) {
      ring_[(write + i) % Capacity].store(values[i]);
    }
    write_head_.store(write + count, std::memory_order_release);
    return true;
  }

  bool pop(Action& value) {
    uint64_t read = read_head_.load(std::memory_order_relaxed);
    while (true) {
      const uint64_t write = write_head_.load(std::memory_order_acquire);
      if (read >= write) return false;

      const Action candidate = ring_[read % Capacity].load();
      if (read_head_.compare_exchange_weak(
              read, read + 1, std::memory_order_release, std::memory_order_relaxed)) {
        value = candidate;
        return true;
      }
    }
  }

  void clear() {
    uint64_t read = read_head_.load(std::memory_order_relaxed);
    while (true) {
      const uint64_t write = write_head_.load(std::memory_order_acquire);
      if (read >= write) return;
      if (read_head_.compare_exchange_weak(
              read, write, std::memory_order_release, std::memory_order_relaxed)) {
        return;
      }
    }
  }

  std::size_t size() const {
    const uint64_t write = write_head_.load(std::memory_order_acquire);
    const uint64_t read = read_head_.load(std::memory_order_acquire);
    return static_cast<std::size_t>(write - read);
  }

 private:
  class AtomicAction {
   public:
    void store(const Action& value) {
      for (std::size_t i = 0; i < ActionDimension; ++i) {
        values_[i].store(value[i], std::memory_order_relaxed);
      }
    }

    Action load() const {
      Action value{};
      for (std::size_t i = 0; i < ActionDimension; ++i) {
        value[i] = values_[i].load(std::memory_order_relaxed);
      }
      return value;
    }

   private:
    std::array<std::atomic<double>, ActionDimension> values_{};
  };

  std::array<AtomicAction, Capacity> ring_{};
  alignas(64) std::atomic<uint64_t> write_head_{0};
  alignas(64) std::atomic<uint64_t> read_head_{0};
};

}  // namespace franka_control::cpp

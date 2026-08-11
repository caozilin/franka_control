#pragma once

#include <array>
#include <atomic>
#include <cstdint>

namespace franka_control::cpp {

struct LatestRobotState {
  std::array<double, 7> q{};
  std::array<double, 8> pose{};
  bool valid{false};
};

// One-writer/many-reader snapshot for the realtime state publication path.
// Individual values are atomic to avoid C++ data races; the sequence counter
// makes the group consistent as one robot-state sample.
class AtomicRobotStateSnapshot {
 public:
  AtomicRobotStateSnapshot() {
    static_assert(std::atomic<double>::is_always_lock_free,
                  "realtime state snapshot requires lock-free double atomics");
  }

  void store(const LatestRobotState& state) {
    sequence_.fetch_add(1, std::memory_order_acq_rel);  // odd: write in progress
    for (std::size_t i = 0; i < state.q.size(); ++i) {
      q_[i].store(state.q[i], std::memory_order_relaxed);
    }
    for (std::size_t i = 0; i < state.pose.size(); ++i) {
      pose_[i].store(state.pose[i], std::memory_order_relaxed);
    }
    valid_.store(state.valid, std::memory_order_relaxed);
    sequence_.fetch_add(1, std::memory_order_release);  // even: publish sample
  }

  LatestRobotState load() const {
    LatestRobotState state;
    while (true) {
      const uint64_t before = sequence_.load(std::memory_order_acquire);
      if ((before & 1U) != 0U) continue;

      for (std::size_t i = 0; i < state.q.size(); ++i) {
        state.q[i] = q_[i].load(std::memory_order_relaxed);
      }
      for (std::size_t i = 0; i < state.pose.size(); ++i) {
        state.pose[i] = pose_[i].load(std::memory_order_relaxed);
      }
      state.valid = valid_.load(std::memory_order_relaxed);

      std::atomic_thread_fence(std::memory_order_acquire);
      const uint64_t after = sequence_.load(std::memory_order_relaxed);
      if (before == after) return state;
    }
  }

 private:
  alignas(64) std::atomic<uint64_t> sequence_{0};
  std::array<std::atomic<double>, 7> q_{};
  std::array<std::atomic<double>, 8> pose_{};
  std::atomic<bool> valid_{false};
};

}  // namespace franka_control::cpp

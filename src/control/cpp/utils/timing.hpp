#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

namespace franka_control::cpp {

inline constexpr std::array<const char*, 23> kTimingFieldNames{
    "loop_total",
    "read_once",
    "policy_total",
    "action_get",
    "transform_action",
    "update_pose_goal",
    "state_lock",
    "controller_update_goal",
    "gripper",
    "print_events",
    "controller_step",
    "controller_reference",
    "controller_slerp",
    "controller_model_coriolis",
    "controller_model_jacobian",
    "controller_velocity_math",
    "controller_pose_error",
    "controller_wrench_torque",
    "controller_torque_limit",
    "raw_trace_write",
    "trace_callback",
    "torques_build",
    "write_once",
};

enum TimingField : int {
  kLoopTotal = 0,
  kReadOnce,
  kPolicyTotal,
  kActionGet,
  kTransformAction,
  kUpdatePoseGoal,
  kStateLock,
  kControllerUpdateGoal,
  kGripper,
  kPrintEvents,
  kControllerStep,
  kControllerReference,
  kControllerSlerp,
  kControllerModelCoriolis,
  kControllerModelJacobian,
  kControllerVelocityMath,
  kControllerPoseError,
  kControllerWrenchTorque,
  kControllerTorqueLimit,
  kRawTraceWrite,
  kTraceCallback,
  kTorquesBuild,
  kWriteOnce,
  kTimingFieldCount,
};

using Clock = std::chrono::steady_clock;

inline double secondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct alignas(64) TimingFrame {
  double elapsed{0.0};
  double robot_dt{0.0};
  std::array<double, kTimingFieldCount> fields{};
};

class TimingRing {
 public:
  explicit TimingRing(size_t capacity) : capacity_(capacity > 0 ? capacity : 1), ring_(capacity_) {}

  void write(const TimingFrame& frame) {
    const uint64_t idx = write_head_.load(std::memory_order_relaxed);
    ring_[idx % capacity_] = frame;
    write_head_.store(idx + 1, std::memory_order_release);
  }

  uint64_t head() const { return write_head_.load(std::memory_order_acquire); }
  size_t capacity() const { return capacity_; }
  void clear() { write_head_.store(0, std::memory_order_release); }
  const TimingFrame* data() const { return ring_.data(); }

 private:
  size_t capacity_;
  std::vector<TimingFrame> ring_;
  std::atomic<uint64_t> write_head_{0};
};

}  // namespace franka_control::cpp

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/gripper.h>
#include <franka/gripper_state.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <exception>
#include <cstring>
#include <memory>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "nullspace/nullspace_torque.hpp"
#include "reference/cartesian_reference.hpp"
#include "reference/joint_reference.hpp"
#include "reference/reference_factory.hpp"
#include "safety/torque_rate_limiter.hpp"
#include "tracker/cartesian_impedance_tracker.hpp"
#include "tracker/joint_impedance_tracker.hpp"
#include "utils/control.hpp"
#include "utils/atomic_robot_state.hpp"
#include "utils/joint_motion_generator.hpp"
#include "utils/pose.hpp"
#include "utils/spsc_action_queue.hpp"
#include "utils/timing.hpp"

namespace py = pybind11;
using namespace franka_control::cpp;

namespace {

// time(1) + goal_xyz(3) + goal_rotvec(3) + ref_xyz(3) + ref_rotvec(3)
// + actual_xyz(3) + actual_rotvec(3) + tau_cmd(7) + tau_desired(7) + tau_j_d(7) + tau_j(7) = 47
static constexpr int kTraceDim = 47;
static constexpr std::size_t kActionQueueCapacity = 4096;
static constexpr int kDefaultTraceSeconds = 180;  // 1kHz × 180s
static constexpr std::array<double, 7> kJointLowerLimits{
    {-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
static constexpr std::array<double, 7> kJointUpperLimits{
    {2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};

struct alignas(64) TraceFrame {
  double time;
  double goal_xyz[3];
  double goal_rotvec[3];
  double ref_xyz[3];
  double ref_rotvec[3];
  double actual_xyz[3];
  double actual_rotvec[3];
  double tau_cmd[7];
  double tau_desired[7];
  double tau_j_d[7];
  double tau_j[7];
};

// Ring buffer: single writer (1kHz callback), single reader (Python GIL thread).
// write_head_ is monotonic, never wraps. Reader computes index % capacity.
class TraceRing {
 public:
  explicit TraceRing(size_t capacity = kDefaultTraceSeconds * 1000)
      : capacity_(capacity > 0 ? capacity : 1), ring_(capacity_) {}

  void write(const TraceFrame& frame) {
    if (capacity_ == 0) return;
    const uint64_t idx = write_head_.load(std::memory_order_relaxed);
    ring_[idx % capacity_] = frame;
    write_head_.store(idx + 1, std::memory_order_release);
  }
  uint64_t head() const { return write_head_.load(std::memory_order_acquire); }
  size_t capacity() const { return capacity_; }
  void clear() { write_head_.store(0, std::memory_order_release); }
  const TraceFrame* data() const { return ring_.data(); }

 private:
  size_t capacity_;
  std::vector<TraceFrame> ring_;
  std::atomic<uint64_t> write_head_{0};
};

std::vector<std::array<double, 8>> actionsFromArray(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array) {
  if (array.ndim() == 1 && (array.shape(0) == 7 || array.shape(0) == 8)) {
    std::array<double, 8> out{};
    const auto* data = array.data();
    for (size_t i = 0; i < static_cast<size_t>(array.shape(0)); ++i) out[i] = data[i];
    return {out};
  }
  if (array.ndim() == 2 && array.shape(0) > 0 && (array.shape(1) == 7 || array.shape(1) == 8)) {
    const auto rows = static_cast<size_t>(array.shape(0));
    const auto columns = static_cast<size_t>(array.shape(1));
    const auto* data = array.data();
    std::vector<std::array<double, 8>> out(rows);
    for (size_t row = 0; row < rows; ++row) {
      for (size_t column = 0; column < columns; ++column) {
        out[row][column] = data[row * columns + column];
      }
    }
    return out;
  }
  throw std::invalid_argument("action must have shape (7,), (8,), (N, 7), or (N, 8)");
}

std::array<double, 7> array7FromArray(const py::array_t<double, py::array::c_style | py::array::forcecast>& array,
                                      const char* name) {
  if (array.ndim() != 1 || array.shape(0) != 7) {
    throw std::invalid_argument(std::string(name) + " must have shape (7,)");
  }
  std::array<double, 7> out{};
  const auto* data = array.data();
  for (size_t i = 0; i < 7; ++i) out[i] = data[i];
  return out;
}

std::array<double, 6> array6FromArray(const py::array_t<double, py::array::c_style | py::array::forcecast>& array,
                                      const char* name) {
  if (array.ndim() != 1 || array.shape(0) != 6) {
    throw std::invalid_argument(std::string(name) + " must have shape (6,)");
  }
  std::array<double, 6> out{};
  const auto* data = array.data();
  for (size_t i = 0; i < 6; ++i) out[i] = data[i];
  return out;
}

Matrix6d matrix6FromArray(const py::array_t<double, py::array::c_style | py::array::forcecast>& array,
                          const char* name) {
  if (array.ndim() != 2 || array.shape(0) != 6 || array.shape(1) != 6) {
    throw std::invalid_argument(std::string(name) + " must have shape (6, 6)");
  }
  Matrix6d out;
  const auto* data = array.data();
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) out(r, c) = data[r * 6 + c];
  }
  return out;
}

template <size_t N>
std::array<double, N> fixedArrayFromHandle(py::handle value, const char* name) {
  auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
  if (!array || array.ndim() != 1 || array.shape(0) != static_cast<py::ssize_t>(N)) {
    throw std::invalid_argument(std::string(name) + " has the wrong shape");
  }
  std::array<double, N> out{};
  std::copy(array.data(), array.data() + N, out.begin());
  return out;
}

struct BackendConfig {
  double policy_period_s;
  double joint_min_jerk_duration_s;
  std::array<double, 7> joint_stiffness;
  std::array<double, 7> joint_damping;
  double reference_position_epsilon;
  double reference_linear_velocity_epsilon;
  double reference_rotation_epsilon;
  double reference_angular_velocity_epsilon;
  std::array<double, 7> collision_lower_torque;
  std::array<double, 7> collision_upper_torque;
  std::array<double, 6> collision_lower_force;
  std::array<double, 6> collision_upper_force;
  double gripper_width_max;
};

BackendConfig backendConfigFromDict(const py::dict& config) {
  return {
      config["policy_period_s"].cast<double>(),
      config["joint_min_jerk_duration_s"].cast<double>(),
      fixedArrayFromHandle<7>(config["joint_stiffness"], "joint_stiffness"),
      fixedArrayFromHandle<7>(config["joint_damping"], "joint_damping"),
      config["reference_position_epsilon"].cast<double>(),
      config["reference_linear_velocity_epsilon"].cast<double>(),
      config["reference_rotation_epsilon"].cast<double>(),
      config["reference_angular_velocity_epsilon"].cast<double>(),
      fixedArrayFromHandle<7>(config["collision_lower_torque"], "collision_lower_torque"),
      fixedArrayFromHandle<7>(config["collision_upper_torque"], "collision_upper_torque"),
      fixedArrayFromHandle<6>(config["collision_lower_force"], "collision_lower_force"),
      fixedArrayFromHandle<6>(config["collision_upper_force"], "collision_upper_force"),
      config["gripper_width_max"].cast<double>(),
  };
}

}  // namespace

class RealtimeFrankaBackend {
 public:
  RealtimeFrankaBackend(std::string robot_ip,
                        double max_translation_step,
                        double max_rotation_step,
                        double motion_limited_max_translation_velocity,
                        double motion_limited_max_rotation_velocity,
                        double motion_limited_max_translation_acceleration,
                        double motion_limited_max_rotation_acceleration,
                        double max_torque_rate,
                        std::string reference_name,
                        std::string control_mode,
                        double trace_capacity_sec,
                        py::array_t<double, py::array::c_style | py::array::forcecast> home_q,
                        py::array_t<double, py::array::c_style | py::array::forcecast> stiffness,
                        py::array_t<double, py::array::c_style | py::array::forcecast> damping,
                        bool nullspace_enabled,
                        py::array_t<double, py::array::c_style | py::array::forcecast> nullspace_q_target,
                        double nullspace_stiffness,
                        double nullspace_damping,
                        std::string nullspace_pinv,
                        std::string nullspace_projector,
                        double nullspace_lambda,
                        py::array_t<double, py::array::c_style | py::array::forcecast> task_constraint_mask,
                        py::dict backend_config)
      : robot_ip_(std::move(robot_ip)),
        robot_(robot_ip_),
        max_translation_step_(max_translation_step),
        max_rotation_step_(max_rotation_step),
        motion_limited_max_translation_velocity_(motion_limited_max_translation_velocity),
        motion_limited_max_rotation_velocity_(motion_limited_max_rotation_velocity),
        motion_limited_max_translation_acceleration_(motion_limited_max_translation_acceleration),
        motion_limited_max_rotation_acceleration_(motion_limited_max_rotation_acceleration),
        max_torque_rate_(max_torque_rate),
        reference_generator_(makeReferenceGenerator(reference_name)),
        control_mode_(std::move(control_mode)),
        home_q_(array7FromArray(home_q, "home_q")),
        stiffness_(matrix6FromArray(stiffness, "stiffness")),
        damping_(matrix6FromArray(damping, "damping")),
        nullspace_config_{nullspace_enabled,
                          parsePseudoInverseMode(nullspace_pinv),
                          parseNullspaceProjectorMode(nullspace_projector),
                          nullspace_lambda,
                          nullspace_stiffness,
                          nullspace_damping,
                          array7FromArray(nullspace_q_target, "nullspace_q_target"),
                          array6FromArray(task_constraint_mask, "task_constraint_mask")},
        config_(backendConfigFromDict(backend_config)),
        trace_ring_(static_cast<size_t>(trace_capacity_sec > 0.0 ? trace_capacity_sec * 1000.0 : 1000)),
        timing_ring_(static_cast<size_t>(trace_capacity_sec > 0.0 ? trace_capacity_sec * 1000.0 : 1000)) {
    robot_.setCollisionBehavior(config_.collision_lower_torque,
                                config_.collision_upper_torque,
                                config_.collision_lower_force,
                                config_.collision_upper_force);
    updateLatestRobotState(robot_.readOnce());
  }

  ~RealtimeFrankaBackend() { stop(); }

  void enqueue_action(py::array_t<double, py::array::c_style | py::array::forcecast> action) {
    const auto parsed = actionsFromArray(action);
    if (!action_queue_.tryPushBlock(parsed.data(), parsed.size())) {
      throw std::overflow_error("realtime action queue capacity exceeded");
    }
  }

  void clear_actions() { action_queue_.clear(); }

  size_t get_pending_action_count() const { return action_queue_.size(); }

  std::vector<double> get_joint_positions() {
    const auto cached = latestRobotState();
    return {cached.q.begin(), cached.q.end()};
  }

  void start_control_loop(double max_duration) {
    if (running_.load()) throw std::runtime_error("control thread is already running");
    joinControlThread(true);
    throwIfError();
    stop_requested_.store(false);
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_.clear();
    }
    trace_ring_.clear();
    timing_ring_.clear();
    resetTraceContinuity();
    updateLatestRobotState(robot_.readOnce());
    running_.store(true);
    control_thread_ = std::thread([this, max_duration]() {
      try {
        runControlLoop(max_duration);
      } catch (const std::exception& exc) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_ = exc.what();
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_ = "unknown C++ realtime control exception";
      }
      running_.store(false);
    });
  }

  void joinControlThread(bool may_release_gil) {
    if (!control_thread_.joinable()) return;
    if (may_release_gil && PyGILState_Check()) {
      py::gil_scoped_release release;
      control_thread_.join();
    } else {
      control_thread_.join();
    }
  }

  void wait() {
    joinControlThread(true);
    throwIfError();
  }

  void stop() {
    stop_requested_.store(true);
    joinControlThread(true);
    try {
      robot_.stop();
    } catch (...) {
    }
  }

  bool is_running() const { return running_.load(); }

  uint64_t get_trace_head() const { return trace_ring_.head(); }

  uint64_t get_timing_head() const { return timing_ring_.head(); }

  py::array_t<double> get_trace_since(uint64_t after) {
    uint64_t head = trace_ring_.head();
    size_t cap = trace_ring_.capacity();
    if (head <= after) {
      return py::array_t<double>(std::vector<py::ssize_t>{0, static_cast<py::ssize_t>(kTraceDim)});
    }
    uint64_t count = head - after;
    if (count > cap) count = cap;
    const uint64_t start = head - count;
    std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(kTraceDim)};
    auto result = py::array_t<double>(shape);
    auto* buf = result.mutable_data();
    const TraceFrame* src = trace_ring_.data();
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t idx = (start + i) % cap;
      const auto& f = src[idx];
      double* out = buf + i * kTraceDim;
      out[0] = f.time;
      std::memcpy(out + 1, f.goal_xyz, sizeof(f.goal_xyz));
      std::memcpy(out + 4, f.goal_rotvec, sizeof(f.goal_rotvec));
      std::memcpy(out + 7, f.ref_xyz, sizeof(f.ref_xyz));
      std::memcpy(out + 10, f.ref_rotvec, sizeof(f.ref_rotvec));
      std::memcpy(out + 13, f.actual_xyz, sizeof(f.actual_xyz));
      std::memcpy(out + 16, f.actual_rotvec, sizeof(f.actual_rotvec));
      std::memcpy(out + 19, f.tau_cmd, sizeof(f.tau_cmd));
      std::memcpy(out + 26, f.tau_desired, sizeof(f.tau_desired));
      std::memcpy(out + 33, f.tau_j_d, sizeof(f.tau_j_d));
      std::memcpy(out + 40, f.tau_j, sizeof(f.tau_j));
    }
    return result;
  }

  void clear_trace() {
    trace_ring_.clear();
    timing_ring_.clear();
    resetTraceContinuity();
  }

  py::list get_timing_field_names() const {
    py::list names;
    for (const char* name : kTimingFieldNames) names.append(name);
    return names;
  }

  py::array_t<double> get_timing_since(uint64_t after) {
    uint64_t head = timing_ring_.head();
    size_t cap = timing_ring_.capacity();
    static constexpr int kTimingDim = 3 + kTimingFieldCount;  // frame, elapsed, robot_dt, fields...
    if (head <= after) {
      return py::array_t<double>(std::vector<py::ssize_t>{0, static_cast<py::ssize_t>(kTimingDim)});
    }
    uint64_t count = head - after;
    if (count > cap) count = cap;
    const uint64_t start = head - count;
    std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(kTimingDim)};
    auto result = py::array_t<double>(shape);
    auto* buf = result.mutable_data();
    const TimingFrame* src = timing_ring_.data();
    for (uint64_t i = 0; i < count; ++i) {
      const uint64_t frame_index = start + i;
      const auto& f = src[frame_index % cap];
      double* out = buf + i * kTimingDim;
      out[0] = static_cast<double>(frame_index);
      out[1] = f.elapsed;
      out[2] = f.robot_dt;
      for (int field = 0; field < kTimingFieldCount; ++field) out[3 + field] = f.fields[field];
    }
    return result;
  }

  void set_reference(std::string reference_name) {
    if (running_.load()) throw std::runtime_error("cannot change reference while control thread is running");
    reference_generator_ = makeReferenceGenerator(reference_name);
  }

  void set_control_mode(std::string control_mode) {
    if (running_.load()) throw std::runtime_error("cannot change control_mode while control thread is running");
    if (control_mode != "cartesian" && control_mode != "joint") {
      throw std::invalid_argument("control_mode must be 'cartesian' or 'joint'");
    }
    control_mode_ = std::move(control_mode);
  }

  void reset(double speed_factor, double reset_duration) {
    if (running_.load()) throw std::runtime_error("cannot reset while control thread is running");
    if (reset_duration > 0.0) {
      DurationJointMotionGenerator motion_generator(reset_duration, home_q_);
      robot_.control(motion_generator);
    } else {
      JointMotionGenerator motion_generator(speed_factor, home_q_);
      robot_.control(motion_generator);
    }
    updateLatestRobotState(robot_.readOnce());
  }

  void probe_model() {
    auto model = robot_.loadModel();
    const auto state = robot_.readOnce();
    (void)model.coriolis(state);
    (void)model.zeroJacobian(franka::Frame::kEndEffector, state);
  }

  std::vector<double> get_robot_state_vector() {
    const auto cached = latestRobotState();
    return {cached.pose.begin(), cached.pose.end()};
  }

 private:
  void throwIfError() {
    std::string error;
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error = error_;
      error_.clear();
    }
    if (!error.empty()) throw std::runtime_error(error);
  }

  std::array<double, 7> popAction(TimingFrame* timing = nullptr) {
    const auto start = Clock::now();
    std::array<double, 8> queued{};
    const bool has_action = action_queue_.pop(queued);
    if (timing != nullptr) timing->fields[kActionGet] += secondsSince(start);
    if (!has_action) return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0};
    std::array<double, 7> action{};
    for (size_t i = 0; i < 7; ++i) action[i] = queued[i];
    return action;
  }

  std::array<double, 7> popAccumulatedJointAction() {
    std::array<double, 7> action{};
    std::array<double, 8> next{};
    while (action_queue_.pop(next)) {
      for (size_t i = 0; i < 7; ++i) action[i] += next[i];
    }
    return action;
  }

  void runJointMinJerkImpedanceLoop(double max_duration, double segment_duration) {
    auto model = robot_.loadModel();
    const franka::RobotState initial_state = robot_.readOnce();
    JointMinJerkReferenceGenerator reference(
        segment_duration, vector7FromArray(kJointLowerLimits), vector7FromArray(kJointUpperLimits));
    reference.reset(vector7FromArray(initial_state.q));
    JointImpedanceTracker tracker(
        vector7FromArray(config_.joint_stiffness), vector7FromArray(config_.joint_damping));
    TorqueRateLimiter safety(max_torque_rate_);
    double elapsed = 0.0;
    double next_policy_time = 0.0;

    robot_.control([&](const franka::RobotState& state, franka::Duration step) -> franka::Torques {
      const double dt = std::max(step.toSec(), 0.001);
      elapsed += dt;
      updateLatestRobotState(state);

      while (elapsed + 1e-12 >= next_policy_time) {
        reference.acceptDelta(popAccumulatedJointAction(), elapsed);
        next_policy_time += config_.policy_period_s;
      }

      const JointReferenceSample reference_sample = reference.sample(elapsed);
      const Vector7d q = vector7FromArray(state.q);
      const Vector7d dq = vector7FromArray(state.dq);
      const Vector7d coriolis = vector7FromArray(model.coriolis(state));
      const Vector7d tau_desired = tracker.compute(q, dq, coriolis, reference_sample);
      const Vector7d tau_limited = safety.apply(tau_desired, state.tau_J_d, dt);
      franka::Torques command(arrayFromVector7(tau_limited));
      if (stop_requested_.load() || (max_duration > 0.0 && elapsed >= max_duration)) {
        return franka::MotionFinished(command);
      }
      return command;
    });
  }

  void runControlLoop(double max_duration) {
    if (control_mode_ == "joint") {
      runJointMinJerkImpedanceLoop(max_duration, config_.joint_min_jerk_duration_s);
      return;
    }

    auto model = robot_.loadModel();
    const franka::RobotState initial_state = robot_.readOnce();
    CartesianReferenceGenerator reference(
        reference_generator_, max_translation_step_, max_rotation_step_,
        motion_limited_max_translation_velocity_, motion_limited_max_rotation_velocity_,
        motion_limited_max_translation_acceleration_, motion_limited_max_rotation_acceleration_,
        config_.policy_period_s, config_.reference_position_epsilon,
        config_.reference_linear_velocity_epsilon, config_.reference_rotation_epsilon,
        config_.reference_angular_velocity_epsilon);
    reference.reset(poseFromArray(initial_state.O_T_EE));
    CartesianImpedanceTracker tracker(stiffness_, damping_, nullspace_config_);
    TorqueRateLimiter safety(max_torque_rate_);
    double elapsed = 0.0;
    double next_policy_time = 0.0;

    robot_.control([&](const franka::RobotState& state, franka::Duration step) -> franka::Torques {
      TimingFrame timing{};
      const auto loop_start = Clock::now();
      const double dt = std::max(step.toSec(), 0.001);
      elapsed += dt;
      timing.elapsed = elapsed;
      timing.robot_dt = dt;
      updateLatestRobotState(state);

      auto start = Clock::now();
      while (elapsed + 1e-12 >= next_policy_time) {
        reference.acceptAction(popAction(&timing), elapsed);
        next_policy_time += config_.policy_period_s;
      }
      timing.fields[kPolicyTotal] += secondsSince(start);

      start = Clock::now();
      const CartesianReferenceSample reference_sample = reference.sample(elapsed, dt);
      timing.fields[kControllerReference] += secondsSince(start);

      start = Clock::now();
      const auto coriolis_array = model.coriolis(state);
      const Vector7d coriolis = vector7FromArray(coriolis_array);
      timing.fields[kControllerModelCoriolis] += secondsSince(start);

      start = Clock::now();
      const auto jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, state);
      const Eigen::Map<const Matrix67d> jacobian(jacobian_array.data());
      timing.fields[kControllerModelJacobian] += secondsSince(start);

      start = Clock::now();
      Matrix7d mass = Matrix7d::Zero();
      const Matrix7d* mass_ptr = nullptr;
      if (nullspace_config_.enabled &&
          nullspace_config_.projector_mode == NullspaceProjectorMode::kDynamic) {
        const auto mass_array = model.mass(state);
        mass = Eigen::Map<const Matrix7d>(mass_array.data());
        mass_ptr = &mass;
      }
      timing.fields[kControllerModelJacobian] += secondsSince(start);

      start = Clock::now();
      const Vector7d q = vector7FromArray(state.q);
      const Vector7d dq = vector7FromArray(state.dq);
      const Pose actual_pose = poseFromArray(state.O_T_EE);
      timing.fields[kControllerVelocityMath] += secondsSince(start);

      CartesianTrackerTiming tracker_timing{};
      const CartesianTrackingOutput tracking =
          tracker.compute(actual_pose, q, dq, coriolis, jacobian, mass_ptr, reference_sample, &tracker_timing);
      timing.fields[kControllerPoseError] += tracker_timing.pose_error;
      timing.fields[kControllerWrenchTorque] +=
          tracker_timing.wrench_torque + tracker_timing.nullspace_torque;

      start = Clock::now();
      const Vector7d tau_limited = safety.apply(tracking.desired_torque, state.tau_J_d, dt);
      timing.fields[kControllerTorqueLimit] += secondsSince(start);

      start = Clock::now();
      franka::Torques command(arrayFromVector7(tau_limited));
      timing.fields[kTorquesBuild] += secondsSince(start);

      // 1kHz trace: write every frame with pre-computed xyz + continuous rotvec
      {
        start = Clock::now();
        auto fill_xyz_rotvec = [](double* dst_xyz, double* dst_rotvec,
                                   const Pose& pose, Eigen::Vector3d& prev) {
          dst_xyz[0] = pose(0, 3);
          dst_xyz[1] = pose(1, 3);
          dst_xyz[2] = pose(2, 3);
          Eigen::Vector3d rv = matrixToRotvecContinuous(pose.block<3, 3>(0, 0), prev);
          dst_rotvec[0] = rv.x();
          dst_rotvec[1] = rv.y();
          dst_rotvec[2] = rv.z();
          prev = rv;
        };

        TraceFrame frame{};
        frame.time = elapsed;
        fill_xyz_rotvec(frame.goal_xyz, frame.goal_rotvec, reference_sample.target_pose, prev_goal_rotvec_);
        fill_xyz_rotvec(frame.ref_xyz, frame.ref_rotvec, reference_sample.pose, prev_ref_rotvec_);
        fill_xyz_rotvec(frame.actual_xyz, frame.actual_rotvec, actual_pose, prev_actual_rotvec_);
        for (Eigen::Index i = 0; i < 7; ++i) {
          frame.tau_cmd[i] = tau_limited[i];
          frame.tau_desired[i] = tracking.desired_torque[i];
          frame.tau_j_d[i] = state.tau_J_d[static_cast<size_t>(i)];
          frame.tau_j[i] = state.tau_J[static_cast<size_t>(i)];
        }
        trace_ring_.write(frame);
        timing.fields[kRawTraceWrite] += secondsSince(start);
      }

      timing.fields[kControllerStep] =
          timing.fields[kControllerReference] + timing.fields[kControllerModelCoriolis] +
          timing.fields[kControllerModelJacobian] + timing.fields[kControllerVelocityMath] +
          timing.fields[kControllerPoseError] + timing.fields[kControllerWrenchTorque] +
          timing.fields[kControllerTorqueLimit] + timing.fields[kTorquesBuild];
      timing.fields[kLoopTotal] = secondsSince(loop_start);
      timing_ring_.write(timing);

      if (stop_requested_.load() || (max_duration > 0.0 && elapsed >= max_duration)) {
        return franka::MotionFinished(command);
      }
      return command;
    });
  }

  std::string robot_ip_;
  franka::Robot robot_;
  double max_translation_step_;
  double max_rotation_step_;
  double motion_limited_max_translation_velocity_;
  double motion_limited_max_rotation_velocity_;
  double motion_limited_max_translation_acceleration_;
  double motion_limited_max_rotation_acceleration_;
  double max_torque_rate_;
  std::shared_ptr<ReferenceGenerator> reference_generator_;
  std::string control_mode_;
  std::array<double, 7> home_q_;
  Matrix6d stiffness_;
  Matrix6d damping_;
  NullspaceConfig nullspace_config_;
  BackendConfig config_;
  SpscActionQueue<8, kActionQueueCapacity> action_queue_;
  std::thread control_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  mutable std::mutex error_mutex_;
  std::string error_;
  TraceRing trace_ring_;
  TimingRing timing_ring_;
  Eigen::Vector3d prev_goal_rotvec_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d prev_ref_rotvec_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d prev_actual_rotvec_{Eigen::Vector3d::Zero()};
  AtomicRobotStateSnapshot latest_robot_state_;

  void updateLatestRobotState(const franka::RobotState& state) {
    const Pose pose = poseFromArray(state.O_T_EE);
    const Eigen::Vector3d rotvec = matrixToRotvec(pose.block<3, 3>(0, 0));
    LatestRobotState latest;
    std::copy(state.q.begin(), state.q.end(), latest.q.begin());
    latest.pose = {pose(0, 3), pose(1, 3), pose(2, 3), rotvec.x(), rotvec.y(), rotvec.z(),
                   config_.gripper_width_max, config_.gripper_width_max};
    latest.valid = true;
    latest_robot_state_.store(latest);
  }

  LatestRobotState latestRobotState() {
    const LatestRobotState cached = latest_robot_state_.load();
    if (cached.valid) return cached;
    const auto state = robot_.readOnce();
    updateLatestRobotState(state);
    return latest_robot_state_.load();
  }

  void resetTraceContinuity() {
    const Eigen::Vector3d home_rotvec{kPi, 0.0, 0.0};
    prev_goal_rotvec_ = home_rotvec;
    prev_ref_rotvec_ = home_rotvec;
    prev_actual_rotvec_ = home_rotvec;
  }
};

class RealtimeGripperBackend {
 public:
  RealtimeGripperBackend(std::string robot_ip,
                         double width_tolerance,
                         double close_threshold,
                         double grasp_epsilon_inner,
                         double grasp_epsilon_outer)
      : gripper_(std::move(robot_ip)),
        width_tolerance_(width_tolerance),
        close_threshold_(close_threshold),
        grasp_epsilon_inner_(grasp_epsilon_inner),
        grasp_epsilon_outer_(grasp_epsilon_outer) {}

  py::dict read_once() const {
    const franka::GripperState state = [this]() {
      py::gil_scoped_release release;
      return gripper_.readOnce();
    }();
    py::dict out;
    out["width"] = state.width;
    out["max_width"] = state.max_width;
    out["is_grasped"] = state.is_grasped;
    out["temperature"] = state.temperature;
    return out;
  }

  bool command(double target, double speed, double force) const {
    target = std::clamp(target, 0.0, kGripperWidthMax);
    try {
      const franka::GripperState state = gripper_.readOnce();
      if (target <= close_threshold_) {
        if (state.is_grasped || state.width <= width_tolerance_) return true;
      } else if (std::abs(state.width - target) <= width_tolerance_) {
        return true;
      }
    } catch (...) {
    }
    // Close commands should use grasp so the gripper applies holding force once contact is made.
    if (target <= close_threshold_) {
      return gripper_.grasp(0.0, speed, force, grasp_epsilon_inner_, grasp_epsilon_outer_);
    }
    return gripper_.move(target, speed);
  }

  bool stop() const { return gripper_.stop(); }

 private:
  franka::Gripper gripper_;
  double width_tolerance_;
  double close_threshold_;
  double grasp_epsilon_inner_;
  double grasp_epsilon_outer_;
};

PYBIND11_MODULE(_franka_backend, m) {
  py::class_<RealtimeFrankaBackend>(m, "RealtimeFrankaBackend")
      .def(py::init<std::string, double, double, double, double, double, double, double, std::string, std::string,
                    double,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    bool,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    double, double, std::string, std::string, double,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::dict>(),
           py::arg("robot_ip"), py::arg("max_translation_step"),
           py::arg("max_rotation_step"),
           py::arg("motion_limited_max_translation_velocity"),
           py::arg("motion_limited_max_rotation_velocity"),
           py::arg("motion_limited_max_translation_acceleration"),
           py::arg("motion_limited_max_rotation_acceleration"),
           py::arg("max_torque_rate"),
           py::arg("reference_name"),
           py::arg("control_mode"), py::arg("trace_capacity_sec"), py::arg("home_q"),
           py::arg("stiffness"), py::arg("damping"),
           py::arg("nullspace_enabled"),
           py::arg("nullspace_q_target"),
           py::arg("nullspace_stiffness"),
           py::arg("nullspace_damping"),
           py::arg("nullspace_pinv"),
           py::arg("nullspace_projector"),
           py::arg("nullspace_lambda"),
           py::arg("task_constraint_mask"),
           py::arg("backend_config"))
      .def("enqueue_action", &RealtimeFrankaBackend::enqueue_action)
      .def("clear_actions", &RealtimeFrankaBackend::clear_actions)
      .def("get_pending_action_count", &RealtimeFrankaBackend::get_pending_action_count)
      .def("get_joint_positions", &RealtimeFrankaBackend::get_joint_positions)
      .def("start_control_loop", &RealtimeFrankaBackend::start_control_loop, py::arg("max_duration") = -1.0)
      .def("wait", &RealtimeFrankaBackend::wait)
      .def("stop", &RealtimeFrankaBackend::stop)
      .def("is_running", &RealtimeFrankaBackend::is_running)
      .def("set_reference", &RealtimeFrankaBackend::set_reference)
      .def("set_control_mode", &RealtimeFrankaBackend::set_control_mode)
      .def("reset", &RealtimeFrankaBackend::reset, py::arg("speed_factor") = 0.5, py::arg("reset_duration") = -1.0)
      .def("probe_model", &RealtimeFrankaBackend::probe_model)
      .def("get_robot_state_vector", &RealtimeFrankaBackend::get_robot_state_vector)
      .def("get_trace_head", &RealtimeFrankaBackend::get_trace_head)
      .def("get_trace_since", &RealtimeFrankaBackend::get_trace_since, py::arg("after") = 0)
      .def("get_timing_head", &RealtimeFrankaBackend::get_timing_head)
      .def("get_timing_since", &RealtimeFrankaBackend::get_timing_since, py::arg("after") = 0)
      .def("get_timing_field_names", &RealtimeFrankaBackend::get_timing_field_names)
      .def("clear_trace", &RealtimeFrankaBackend::clear_trace);

  py::class_<RealtimeGripperBackend>(m, "RealtimeGripperBackend")
      .def(py::init<std::string, double, double, double, double>(),
           py::arg("robot_ip"), py::arg("width_tolerance"), py::arg("close_threshold"),
           py::arg("grasp_epsilon_inner"), py::arg("grasp_epsilon_outer"),
           py::call_guard<py::gil_scoped_release>())
      .def("read_once", &RealtimeGripperBackend::read_once)
      .def("command", &RealtimeGripperBackend::command, py::arg("target"), py::arg("speed"), py::arg("force"),
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &RealtimeGripperBackend::stop, py::call_guard<py::gil_scoped_release>());
}

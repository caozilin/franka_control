#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

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
#include <deque>
#include <exception>
#include <cstring>
#include <memory>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "reference/reference_factory.hpp"
#include "nullspace/nullspace_torque.hpp"
#include "utils/control.hpp"
#include "utils/joint_motion_generator.hpp"
#include "utils/pose.hpp"
#include "utils/timing.hpp"

namespace py = pybind11;
using namespace franka_control::cpp;

namespace {

// time(1) + goal_xyz(3) + goal_rotvec(3) + ref_xyz(3) + ref_rotvec(3)
// + actual_xyz(3) + actual_rotvec(3) + tau_cmd(7) = 26
static constexpr int kTraceDim = 26;
static constexpr int kDefaultTraceSeconds = 180;  // 1kHz × 180s
static constexpr double kMaxTranslationGoalError = 0.03;
static constexpr double kMaxRotationGoalError = kPi / 6.0;
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
};

// Ring buffer: single writer (1kHz callback), single reader (Python GIL thread).
// write_head_ is monotonic, never wraps. Reader computes index % capacity.
class TraceRing {
 public:
  explicit TraceRing(size_t capacity = kDefaultTraceSeconds * 1000)
      : capacity_(capacity > 0 ? capacity : 1), ring_(capacity_) {}

  void write(const TraceFrame& frame) {
    if (capacity_ == 0) return;
    uint64_t idx = write_head_.fetch_add(1, std::memory_order_release);
    ring_[idx % capacity_] = frame;
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

std::array<double, 8> actionFromArray(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) {
  if (array.ndim() == 1 && (array.shape(0) == 7 || array.shape(0) == 8)) {
    std::array<double, 8> out{};
    const auto* data = array.data();
    for (size_t i = 0; i < static_cast<size_t>(array.shape(0)); ++i) out[i] = data[i];
    return out;
  }
  if (array.ndim() == 2 && array.shape(0) > 0 && (array.shape(1) == 7 || array.shape(1) == 8)) {
    std::array<double, 8> out{};
    const auto* data = array.data();
    for (size_t i = 0; i < static_cast<size_t>(array.shape(1)); ++i) out[i] = data[i];
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

Eigen::Vector3d clampNorm(const Eigen::Vector3d& value, double limit) {
  const double norm = value.norm();
  if (norm <= limit || norm < 1e-12) return value;
  return value * (limit / norm);
}
}  // namespace

class RealtimeFrankaBackend {
 public:
  RealtimeFrankaBackend(std::string robot_ip,
                        double max_translation_step,
                        double max_rotation_step,
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
                        double nullspace_lambda)
      : robot_ip_(std::move(robot_ip)),
        robot_(robot_ip_),
        max_translation_step_(max_translation_step),
        max_rotation_step_(max_rotation_step),
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
                          array7FromArray(nullspace_q_target, "nullspace_q_target")},
        trace_ring_(static_cast<size_t>(trace_capacity_sec > 0.0 ? trace_capacity_sec * 1000.0 : 1000)),
        timing_ring_(static_cast<size_t>(trace_capacity_sec > 0.0 ? trace_capacity_sec * 1000.0 : 1000)) {
    robot_.setCollisionBehavior({{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
                                {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
                                {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
                                {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});
  }

  ~RealtimeFrankaBackend() { stop(); }

  void enqueue_action(py::array_t<double, py::array::c_style | py::array::forcecast> action) {
    const auto parsed = actionFromArray(action);
    std::lock_guard<std::mutex> lock(action_mutex_);
    action_queue_.push_back(parsed);
  }

  void clear_actions() {
    std::lock_guard<std::mutex> lock(action_mutex_);
    action_queue_.clear();
  }

  size_t get_pending_action_count() const {
    std::lock_guard<std::mutex> lock(action_mutex_);
    return action_queue_.size();
  }

  py::array_t<double> get_joint_positions() {
    const auto state = robot_.readOnce();
    auto result = py::array_t<double>(static_cast<py::ssize_t>(7));
    std::memcpy(result.mutable_data(), state.q.data(), 7 * sizeof(double));
    return result;
  }

  void start_control_loop(double max_duration) {
    if (running_.load()) throw std::runtime_error("control thread is already running");
    stop_requested_.store(false);
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_.clear();
    }
    trace_ring_.clear();
    timing_ring_.clear();
    resetTraceContinuity();
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
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (!error_.empty()) throw std::runtime_error(error_);
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
  }

  void probe_model() {
    auto model = robot_.loadModel();
    const auto state = robot_.readOnce();
    (void)model.coriolis(state);
    (void)model.zeroJacobian(franka::Frame::kEndEffector, state);
  }

  py::array_t<double> get_robot_state_vector() {
    const auto state = robot_.readOnce();
    const Pose pose = poseFromArray(state.O_T_EE);
    const Eigen::Vector3d rotvec = matrixToRotvec(pose.block<3, 3>(0, 0));
    std::array<double, 8> out{pose(0, 3), pose(1, 3), pose(2, 3), rotvec.x(), rotvec.y(), rotvec.z(),
                             kGripperWidthMax, kGripperWidthMax};
    auto result = py::array_t<double>(static_cast<py::ssize_t>(out.size()));
    std::memcpy(result.mutable_data(), out.data(), out.size() * sizeof(double));
    return result;
  }

 private:
  std::array<double, 7> popAction(TimingFrame* timing = nullptr) {
    const auto start = Clock::now();
    std::lock_guard<std::mutex> lock(action_mutex_);
    if (timing != nullptr) timing->fields[kActionGet] += secondsSince(start);
    if (action_queue_.empty()) return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0};
    const auto queued = action_queue_.front();
    action_queue_.pop_front();
    std::array<double, 7> action{};
    for (size_t i = 0; i < 7; ++i) action[i] = queued[i];
    return action;
  }

  void startNewSegment(double elapsed,
                       const Pose& current_command_pose,
                       const Pose& actual_pose,
                       Pose* segment_start_pose,
                       Pose* segment_target_pose,
                       Eigen::Vector3d* segment_delta_translation,
                       Eigen::Vector3d* segment_delta_rotvec,
                       Eigen::Vector3d* last_segment_rotvec,
                       TimingFrame* timing) {
    *segment_start_pose = current_command_pose;
    const auto action = popAction(timing);
    Pose candidate = *segment_target_pose;

    auto start = Clock::now();
    candidate.block<3, 1>(0, 3) += transformTranslation(action, max_translation_step_);
    candidate.block<3, 3>(0, 0) =
        rotvecToMatrix(transformRotation(action, max_rotation_step_)) * candidate.block<3, 3>(0, 0);
    if (timing != nullptr) timing->fields[kTransformAction] += secondsSince(start);

    start = Clock::now();
    Pose limited = candidate;
    const Eigen::Vector3d position_error =
        candidate.block<3, 1>(0, 3) - actual_pose.block<3, 1>(0, 3);
    limited.block<3, 1>(0, 3) =
        actual_pose.block<3, 1>(0, 3) + clampNorm(position_error, kMaxTranslationGoalError);

    const Eigen::Vector3d rotation_error = matrixToRotvecContinuous(
        candidate.block<3, 3>(0, 0) * actual_pose.block<3, 3>(0, 0).transpose(), last_goal_rotation_error_);
    last_goal_rotation_error_ = rotation_error;
    limited.block<3, 3>(0, 0) =
        rotvecToMatrix(clampNorm(rotation_error, kMaxRotationGoalError)) * actual_pose.block<3, 3>(0, 0);
    *segment_target_pose = limited;
    if (timing != nullptr) timing->fields[kUpdatePoseGoal] += secondsSince(start);

    start = Clock::now();
    *segment_delta_translation =
        segment_target_pose->block<3, 1>(0, 3) - segment_start_pose->block<3, 1>(0, 3);
    *segment_delta_rotvec = matrixToRotvecContinuous(
        segment_target_pose->block<3, 3>(0, 0) * segment_start_pose->block<3, 3>(0, 0).transpose(), *last_segment_rotvec);
    *last_segment_rotvec = *segment_delta_rotvec;
    segment_start_time_ = elapsed;
    if (timing != nullptr) timing->fields[kControllerUpdateGoal] += secondsSince(start);
  }

  void updateMotionLimitedReference(double dt,
                                    const Pose& target_pose,
                                    Pose* command_pose,
                                    Vector6d* desired_velocity,
                                    Eigen::Vector3d* v_ref,
                                    Eigen::Vector3d* a_ref,
                                    Eigen::Vector3d* omega_ref,
                                    Eigen::Vector3d* alpha_ref,
                                    Eigen::Vector3d* last_ref_rotation_error) {
    desired_velocity->setZero();

    const Eigen::Vector3d p_goal = target_pose.block<3, 1>(0, 3);
    const Eigen::Vector3d p_ref = command_pose->block<3, 1>(0, 3);
    const Eigen::Vector3d position_error = p_goal - p_ref;
    const double position_distance = position_error.norm();

    if (position_distance < kRefPositionEps && v_ref->norm() < kRefLinearVelocityEps) {
      command_pose->block<3, 1>(0, 3) = p_goal;
      v_ref->setZero();
      a_ref->setZero();
    } else {
      Eigen::Vector3d v_des = Eigen::Vector3d::Zero();
      if (position_distance >= 1e-12) {
        const Eigen::Vector3d direction = position_error / position_distance;
        const double v_allow = std::sqrt(std::max(0.0, 2.0 * kMaxRefLinearAcceleration * position_distance));
        const double v_step_allow = position_distance / dt;
        v_des = direction * std::min(std::min(kMaxRefLinearVelocity, v_allow), v_step_allow);
      }

      *a_ref = clampNorm((v_des - *v_ref) / dt, kMaxRefLinearAcceleration);
      *v_ref = clampNorm(*v_ref + *a_ref * dt, kMaxRefLinearVelocity);

      const Eigen::Vector3d p_next = p_ref + *v_ref * dt;
      if (position_distance >= 1e-12 && position_error.dot(p_goal - p_next) <= 0.0) {
        command_pose->block<3, 1>(0, 3) = p_goal;
        v_ref->setZero();
        a_ref->setZero();
      } else {
        command_pose->block<3, 1>(0, 3) = p_next;
      }
    }

    const Eigen::Vector3d phi = matrixToRotvecContinuous(
        target_pose.block<3, 3>(0, 0) * command_pose->block<3, 3>(0, 0).transpose(), *last_ref_rotation_error);
    *last_ref_rotation_error = phi;
    const double rotation_distance = phi.norm();

    if (rotation_distance < kRefRotationEps && omega_ref->norm() < kRefAngularVelocityEps) {
      command_pose->block<3, 3>(0, 0) = target_pose.block<3, 3>(0, 0);
      omega_ref->setZero();
      alpha_ref->setZero();
      last_ref_rotation_error->setZero();
    } else {
      Eigen::Vector3d omega_des = Eigen::Vector3d::Zero();
      if (rotation_distance >= 1e-12) {
        const Eigen::Vector3d direction = phi / rotation_distance;
        const double omega_allow = std::sqrt(std::max(0.0, 2.0 * kMaxRefAngularAcceleration * rotation_distance));
        const double omega_step_allow = rotation_distance / dt;
        omega_des = direction * std::min(std::min(kMaxRefAngularVelocity, omega_allow), omega_step_allow);
      }

      *alpha_ref = clampNorm((omega_des - *omega_ref) / dt, kMaxRefAngularAcceleration);
      *omega_ref = clampNorm(*omega_ref + *alpha_ref * dt, kMaxRefAngularVelocity);

      const Eigen::Vector3d rotation_step = *omega_ref * dt;
      if (rotation_distance >= 1e-12 && phi.dot(rotation_step) >= rotation_distance * rotation_distance) {
        command_pose->block<3, 3>(0, 0) = target_pose.block<3, 3>(0, 0);
        omega_ref->setZero();
        alpha_ref->setZero();
        last_ref_rotation_error->setZero();
      } else {
        command_pose->block<3, 3>(0, 0) = rotvecToMatrix(rotation_step) * command_pose->block<3, 3>(0, 0);
      }
    }

    desired_velocity->head<3>() = *v_ref;
    desired_velocity->tail<3>() = *omega_ref;
  }

  std::array<double, 7> popAccumulatedJointAction() {
    std::array<double, 7> action{};
    std::lock_guard<std::mutex> lock(action_mutex_);
    while (!action_queue_.empty()) {
      const auto next = action_queue_.front();
      action_queue_.pop_front();
      for (size_t i = 0; i < 7; ++i) action[i] += next[i];
    }
    return action;
  }

  void runJointMinJerkImpedanceLoop(double max_duration, double segment_duration) {
    const double duration = std::max(segment_duration, kPolicyPeriod);
    const Vector7d joint_stiffness = (Vector7d() << 80.0, 80.0, 80.0, 60.0, 25.0, 15.0, 10.0).finished();
    Vector7d joint_damping = Vector7d::Zero();
    for (Eigen::Index i = 0; i < 7; ++i) joint_damping[i] = 2.0 * std::sqrt(joint_stiffness[i]);

    auto model = robot_.loadModel();
    const franka::RobotState initial_state = robot_.readOnce();
    Vector7d command_q = vector7FromArray(initial_state.q);
    Vector7d command_dq = Vector7d::Zero();
    Vector7d segment_start_q = command_q;
    Vector7d segment_target_q = command_q;
    double elapsed = 0.0;
    double next_policy_time = 0.0;
    double segment_start_time = 0.0;

    robot_.control([&](const franka::RobotState& state, franka::Duration step) -> franka::Torques {
      const double dt = std::max(step.toSec(), 0.001);
      elapsed += dt;

      while (elapsed + 1e-12 >= next_policy_time) {
        const auto action = popAccumulatedJointAction();
        bool has_action = false;
        for (double value : action) has_action = has_action || std::abs(value) > 1e-12;
        if (has_action) {
          segment_start_q = command_q;
          for (Eigen::Index i = 0; i < 7; ++i) {
            segment_target_q[i] = std::clamp(segment_target_q[i] + action[static_cast<size_t>(i)],
                                             kJointLowerLimits[static_cast<size_t>(i)],
                                             kJointUpperLimits[static_cast<size_t>(i)]);
          }
          segment_start_time = elapsed;
        }
        next_policy_time += kPolicyPeriod;
      }

      const double alpha = std::clamp((elapsed - segment_start_time) / duration, 0.0, 1.0);
      const double weight = 10.0 * std::pow(alpha, 3.0) - 15.0 * std::pow(alpha, 4.0) + 6.0 * std::pow(alpha, 5.0);
      const double dweight = (alpha >= 1.0) ? 0.0 : (30.0 * std::pow(alpha, 2.0) - 60.0 * std::pow(alpha, 3.0) + 30.0 * std::pow(alpha, 4.0)) / duration;
      command_q = segment_start_q + weight * (segment_target_q - segment_start_q);
      command_dq = dweight * (segment_target_q - segment_start_q);

      const Vector7d q = vector7FromArray(state.q);
      const Vector7d dq = vector7FromArray(state.dq);
      const Vector7d coriolis = vector7FromArray(model.coriolis(state));
      const Vector7d tau_desired = joint_stiffness.cwiseProduct(command_q - q) +
                                   joint_damping.cwiseProduct(command_dq - dq) + coriolis;
      const Vector7d tau_limited = limitTorqueRate(tau_desired, state.tau_J_d, dt);
      franka::Torques command(arrayFromVector7(tau_limited));
      if (stop_requested_.load() || (max_duration > 0.0 && elapsed >= max_duration)) {
        return franka::MotionFinished(command);
      }
      return command;
    });
  }

  void runControlLoop(double max_duration) {
    if (control_mode_ == "joint") {
      runJointMinJerkImpedanceLoop(max_duration, joint_min_jerk_duration_);
      return;
    }

    auto model = robot_.loadModel();
    const franka::RobotState initial_state = robot_.readOnce();
    Pose command_pose = poseFromArray(initial_state.O_T_EE);
    Pose segment_start_pose = command_pose;
    Pose segment_target_pose = command_pose;
    Eigen::Vector3d segment_delta_translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d segment_delta_rotvec = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_segment_rotvec = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_error_rotvec = Eigen::Vector3d::Zero();
    Eigen::Vector3d v_ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d a_ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d omega_ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d alpha_ref = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_ref_rotation_error = Eigen::Vector3d::Zero();
    double elapsed = 0.0;
    double next_policy_time = 0.0;
    const std::shared_ptr<const ReferenceGenerator> reference = reference_generator_;
    const bool motion_limited = std::string(reference->name()) == "motion_limited";
    segment_start_time_ = 0.0;

    uint64_t step_count = 0;
    robot_.control([&](const franka::RobotState& state, franka::Duration step) -> franka::Torques {
      TimingFrame timing{};
      const auto loop_start = Clock::now();
      const double dt = std::max(step.toSec(), 0.001);
      elapsed += dt;
      ++step_count;
      timing.elapsed = elapsed;
      timing.robot_dt = dt;

      auto start = Clock::now();
      while (elapsed + 1e-12 >= next_policy_time) {
        startNewSegment(elapsed, command_pose, poseFromArray(state.O_T_EE), &segment_start_pose, &segment_target_pose,
                        &segment_delta_translation, &segment_delta_rotvec, &last_segment_rotvec, &timing);
        next_policy_time += kPolicyPeriod;
      }
      timing.fields[kPolicyTotal] += secondsSince(start);

      start = Clock::now();
      Vector6d desired_velocity = Vector6d::Zero();
      if (motion_limited) {
        updateMotionLimitedReference(dt, segment_target_pose, &command_pose, &desired_velocity, &v_ref, &a_ref,
                                     &omega_ref, &alpha_ref, &last_ref_rotation_error);
      } else {
        const auto weights = reference->weights((elapsed - segment_start_time_) / kPolicyPeriod);
        command_pose = segment_start_pose;
        command_pose.block<3, 1>(0, 3) += weights.position * segment_delta_translation;
        command_pose.block<3, 3>(0, 0) =
            rotvecToMatrix(weights.position * segment_delta_rotvec) * segment_start_pose.block<3, 3>(0, 0);

        desired_velocity.head<3>() = weights.velocity * segment_delta_translation;
        desired_velocity.tail<3>() = weights.velocity * segment_delta_rotvec;
      }
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
      timing.fields[kControllerVelocityMath] += secondsSince(start);

      start = Clock::now();
      const Vector6d error = poseError(state.O_T_EE, command_pose, last_error_rotvec);
      last_error_rotvec = error.tail<3>();
      timing.fields[kControllerPoseError] += secondsSince(start);

      start = Clock::now();
      const Vector7d tau_task = jacobian.transpose() * (-stiffness_ * error + damping_ * (desired_velocity - jacobian * dq));
      timing.fields[kControllerWrenchTorque] += secondsSince(start);

      start = Clock::now();
      const Vector7d tau_null = computeNullspaceTorque(jacobian, mass_ptr, q, dq, nullspace_config_);
      const Vector7d tau_limited = limitTorqueRate(tau_task + tau_null + coriolis, state.tau_J_d, dt);
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

        Pose actual = poseFromArray(state.O_T_EE);
        TraceFrame frame{};
        frame.time = elapsed;
        fill_xyz_rotvec(frame.goal_xyz, frame.goal_rotvec, segment_target_pose, prev_goal_rotvec_);
        fill_xyz_rotvec(frame.ref_xyz, frame.ref_rotvec, command_pose, prev_ref_rotvec_);
        fill_xyz_rotvec(frame.actual_xyz, frame.actual_rotvec, actual, prev_actual_rotvec_);
        for (Eigen::Index i = 0; i < 7; ++i) frame.tau_cmd[i] = tau_limited[i];
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
  std::shared_ptr<ReferenceGenerator> reference_generator_;
  std::string control_mode_;
  double joint_min_jerk_duration_{0.25};
  std::array<double, 7> home_q_;
  Matrix6d stiffness_;
  Matrix6d damping_;
  NullspaceConfig nullspace_config_;
  double segment_start_time_{0.0};
  mutable std::mutex action_mutex_;
  std::deque<std::array<double, 8>> action_queue_;
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
  Eigen::Vector3d last_goal_rotation_error_{Eigen::Vector3d::Zero()};
  void resetTraceContinuity() {
    const Eigen::Vector3d home_rotvec{kPi, 0.0, 0.0};
    prev_goal_rotvec_ = home_rotvec;
    prev_ref_rotvec_ = home_rotvec;
    prev_actual_rotvec_ = home_rotvec;
  }
};

class RealtimeGripperBackend {
 public:
  explicit RealtimeGripperBackend(std::string robot_ip) : gripper_(std::move(robot_ip)) {}

  py::dict read_once() const {
    const franka::GripperState state = gripper_.readOnce();
    py::dict out;
    out["width"] = state.width;
    out["max_width"] = state.max_width;
    out["is_grasped"] = state.is_grasped;
    out["temperature"] = state.temperature;
    return out;
  }

  bool command(double target, double speed, double force) const {
    static constexpr double kWidthTolerance = 0.003;
    target = std::clamp(target, 0.0, kGripperWidthMax);
    try {
      const franka::GripperState state = gripper_.readOnce();
      if (target <= 1e-6) {
        if (state.is_grasped || state.width <= kWidthTolerance) return true;
      } else if (std::abs(state.width - target) <= kWidthTolerance) {
        return true;
      }
    } catch (...) {
    }
    if (target <= 1e-6) return gripper_.grasp(0.0, speed, force, 0.08, 0.08);
    return gripper_.move(target, speed);
  }

  bool stop() const { return gripper_.stop(); }

 private:
  franka::Gripper gripper_;
};

PYBIND11_MODULE(_franka_backend, m) {
  py::class_<RealtimeFrankaBackend>(m, "RealtimeFrankaBackend")
      .def(py::init<std::string, double, double, std::string, std::string, double,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    bool,
                    py::array_t<double, py::array::c_style | py::array::forcecast>,
                    double, double, std::string, std::string, double>(),
           py::arg("robot_ip"), py::arg("max_translation_step"),
           py::arg("max_rotation_step"), py::arg("reference_name"),
           py::arg("control_mode"), py::arg("trace_capacity_sec"), py::arg("home_q"),
           py::arg("stiffness"), py::arg("damping"),
           py::arg("nullspace_enabled"),
           py::arg("nullspace_q_target"),
           py::arg("nullspace_stiffness"),
           py::arg("nullspace_damping"),
           py::arg("nullspace_pinv"),
           py::arg("nullspace_projector"),
           py::arg("nullspace_lambda"))
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
      .def(py::init<std::string>(), py::arg("robot_ip"), py::call_guard<py::gil_scoped_release>())
      .def("read_once", &RealtimeGripperBackend::read_once, py::call_guard<py::gil_scoped_release>())
      .def("command", &RealtimeGripperBackend::command, py::arg("target"), py::arg("speed"), py::arg("force"),
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &RealtimeGripperBackend::stop, py::call_guard<py::gil_scoped_release>());
}

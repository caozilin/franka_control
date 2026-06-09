#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <franka/control_types.h>
#include <franka/duration.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
#include <exception>
#include <memory>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include "controllers/controller_factory.hpp"
#include "utils/control.hpp"
#include "utils/joint_motion_generator.hpp"
#include "utils/pose.hpp"

namespace py = pybind11;
using namespace franka_control::cpp;

namespace {

// time(1) + goal_xyz(3) + goal_rotvec(3) + ref_xyz(3) + ref_rotvec(3)
// + actual_xyz(3) + actual_rotvec(3) + tau_cmd(7) = 26
static constexpr int kTraceDim = 26;
static constexpr int kDefaultTraceSeconds = 180;  // 1kHz × 180s

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

std::array<double, 7> actionFromArray(const py::array_t<double, py::array::c_style | py::array::forcecast>& array) {
  if (array.ndim() == 1 && array.shape(0) == 7) {
    std::array<double, 7> out{};
    const auto* data = array.data();
    for (size_t i = 0; i < 7; ++i) out[i] = data[i];
    return out;
  }
  if (array.ndim() == 2 && array.shape(0) > 0 && array.shape(1) == 7) {
    std::array<double, 7> out{};
    const auto* data = array.data();
    for (size_t i = 0; i < 7; ++i) out[i] = data[i];
    return out;
  }
  throw std::invalid_argument("action must have shape (7,) or (N, 7)");
}
}  // namespace

class RealtimeFrankaBackend {
 public:
  RealtimeFrankaBackend(std::string robot_ip,
                        double max_translation_step,
                        double max_rotation_step,
                        std::string controller_name,
                        double trace_capacity_sec = kDefaultTraceSeconds)
      : robot_ip_(std::move(robot_ip)),
        robot_(robot_ip_),
        max_translation_step_(max_translation_step),
        max_rotation_step_(max_rotation_step),
        controller_(makeReferenceController(controller_name)),
        stiffness_(defaultStiffness()),
        damping_(defaultDamping()),
        trace_ring_(static_cast<size_t>(trace_capacity_sec > 0.0 ? trace_capacity_sec * 1000.0 : 1000)) {
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

  void start_control_loop(double max_duration) {
    if (running_.load()) throw std::runtime_error("control thread is already running");
    stop_requested_.store(false);
    {
      std::lock_guard<std::mutex> lock(error_mutex_);
      error_.clear();
    }
    trace_ring_.clear();
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

  py::array_t<double> get_trace_since(uint64_t after) {
    uint64_t head = trace_ring_.head();
    size_t cap = trace_ring_.capacity();
    if (head <= after) {
      return py::array_t<double>(std::vector<py::ssize_t>{0, static_cast<py::ssize_t>(kTraceDim)});
    }
    uint64_t count = head - after;
    if (count > cap) count = cap;
    std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(kTraceDim)};
    auto result = py::array_t<double>(shape);
    auto* buf = result.mutable_data();
    const TraceFrame* src = trace_ring_.data();
    for (uint64_t i = 0; i < count; ++i) {
      uint64_t idx = (after + i) % cap;
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
    prev_goal_rotvec_ = Eigen::Vector3d::Zero();
    prev_ref_rotvec_ = Eigen::Vector3d::Zero();
    prev_actual_rotvec_ = Eigen::Vector3d::Zero();
  }

  void set_controller(std::string controller_name) {
    if (running_.load()) throw std::runtime_error("cannot change controller while control thread is running");
    controller_ = makeReferenceController(controller_name);
  }

  void reset(double speed_factor) {
    if (running_.load()) throw std::runtime_error("cannot reset while control thread is running");
    static constexpr std::array<double, 7> kHomeJoints{
        0.0, -0.7853981633974483, 0.0, -2.356194490192345, 0.0, 1.5707963267948966, 0.7853981633974483};
    JointMotionGenerator motion_generator(speed_factor, kHomeJoints);
    robot_.control(motion_generator);
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
    return py::array_t<double>(out.size(), out.data());
  }

 private:
  std::array<double, 7> popAction() {
    std::lock_guard<std::mutex> lock(action_mutex_);
    if (action_queue_.empty()) return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0};
    auto action = action_queue_.front();
    action_queue_.pop_front();
    return action;
  }

  void startNewSegment(double elapsed,
                       const Pose& current_command_pose,
                       Pose* segment_start_pose,
                       Pose* segment_target_pose,
                       Eigen::Vector3d* segment_delta_translation,
                       Eigen::Vector3d* segment_delta_rotvec,
                       Eigen::Vector3d* last_segment_rotvec) {
    *segment_start_pose = current_command_pose;
    const auto action = popAction();
    segment_target_pose->block<3, 1>(0, 3) += transformTranslation(action, max_translation_step_);
    segment_target_pose->block<3, 3>(0, 0) =
        rotvecToMatrix(transformRotation(action, max_rotation_step_)) * segment_target_pose->block<3, 3>(0, 0);
    *segment_delta_translation =
        segment_target_pose->block<3, 1>(0, 3) - segment_start_pose->block<3, 1>(0, 3);
    *segment_delta_rotvec = matrixToRotvecContinuous(
        segment_target_pose->block<3, 3>(0, 0) * segment_start_pose->block<3, 3>(0, 0).transpose(), *last_segment_rotvec);
    *last_segment_rotvec = *segment_delta_rotvec;
    segment_start_time_ = elapsed;
  }

  void runControlLoop(double max_duration) {
    auto model = robot_.loadModel();
    const franka::RobotState initial_state = robot_.readOnce();
    Pose command_pose = poseFromArray(initial_state.O_T_EE);
    Pose segment_start_pose = command_pose;
    Pose segment_target_pose = command_pose;
    Eigen::Vector3d segment_delta_translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d segment_delta_rotvec = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_segment_rotvec = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_error_rotvec = Eigen::Vector3d::Zero();
    double elapsed = 0.0;
    double next_policy_time = 0.0;
    const std::shared_ptr<const ReferenceController> controller = controller_;
    segment_start_time_ = 0.0;

    uint64_t step_count = 0;
    robot_.control([&](const franka::RobotState& state, franka::Duration step) -> franka::Torques {
      const double dt = std::max(step.toSec(), 0.001);
      elapsed += dt;
      ++step_count;

      while (elapsed + 1e-12 >= next_policy_time) {
        startNewSegment(elapsed, command_pose, &segment_start_pose, &segment_target_pose, &segment_delta_translation,
                        &segment_delta_rotvec, &last_segment_rotvec);
        next_policy_time += kPolicyPeriod;
      }

      const auto weights = controller->weights((elapsed - segment_start_time_) / kPolicyPeriod);

      command_pose = segment_start_pose;
      command_pose.block<3, 1>(0, 3) += weights.position * segment_delta_translation;
      command_pose.block<3, 3>(0, 0) =
          rotvecToMatrix(weights.position * segment_delta_rotvec) * segment_start_pose.block<3, 3>(0, 0);

      Vector6d desired_velocity = Vector6d::Zero();
      desired_velocity.head<3>() = weights.velocity * segment_delta_translation;
      desired_velocity.tail<3>() = weights.velocity * segment_delta_rotvec;

      const auto coriolis_array = model.coriolis(state);
      const Vector7d coriolis = vector7FromArray(coriolis_array);
      const auto jacobian_array = model.zeroJacobian(franka::Frame::kEndEffector, state);
      const Eigen::Map<const Matrix67d> jacobian(jacobian_array.data());
      const Vector7d dq = vector7FromArray(state.dq);
      const Vector6d error = poseError(state.O_T_EE, command_pose, last_error_rotvec);
      last_error_rotvec = error.tail<3>();

      const Vector7d tau_task = jacobian.transpose() * (-stiffness_ * error + damping_ * (desired_velocity - jacobian * dq));
      const Vector7d tau_limited = limitTorqueRate(tau_task + coriolis, state.tau_J_d, dt);
      franka::Torques command(arrayFromVector7(tau_limited));

      // 1kHz trace: write every frame with pre-computed xyz + continuous rotvec
      {
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
      }

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
  std::shared_ptr<ReferenceController> controller_;
  Matrix6d stiffness_;
  Matrix6d damping_;
  double segment_start_time_{0.0};
  std::mutex action_mutex_;
  std::deque<std::array<double, 7>> action_queue_;
  std::thread control_thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_requested_{false};
  mutable std::mutex error_mutex_;
  std::string error_;
  TraceRing trace_ring_;
  Eigen::Vector3d prev_goal_rotvec_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d prev_ref_rotvec_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d prev_actual_rotvec_{Eigen::Vector3d::Zero()};
};

PYBIND11_MODULE(_franka_backend, m) {
  py::class_<RealtimeFrankaBackend>(m, "RealtimeFrankaBackend")
      .def(py::init<std::string, double, double, std::string, double>(),
           py::arg("robot_ip"), py::arg("max_translation_step"),
           py::arg("max_rotation_step"), py::arg("controller_name") = "min_jerk",
           py::arg("trace_capacity_sec") = kDefaultTraceSeconds)
      .def("enqueue_action", &RealtimeFrankaBackend::enqueue_action)
      .def("start_control_loop", &RealtimeFrankaBackend::start_control_loop, py::arg("max_duration") = -1.0)
      .def("wait", &RealtimeFrankaBackend::wait)
      .def("stop", &RealtimeFrankaBackend::stop)
      .def("is_running", &RealtimeFrankaBackend::is_running)
      .def("set_controller", &RealtimeFrankaBackend::set_controller)
      .def("reset", &RealtimeFrankaBackend::reset, py::arg("speed_factor") = 0.5)
      .def("probe_model", &RealtimeFrankaBackend::probe_model)
      .def("get_robot_state_vector", &RealtimeFrankaBackend::get_robot_state_vector)
      .def("get_trace_head", &RealtimeFrankaBackend::get_trace_head)
      .def("get_trace_since", &RealtimeFrankaBackend::get_trace_since, py::arg("after") = 0)
      .def("clear_trace", &RealtimeFrankaBackend::clear_trace);
}

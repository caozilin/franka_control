from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
CPP = ROOT / "src" / "control" / "cpp"


def test_realtime_reference_tracker_and_safety_are_separate_components() -> None:
    expected = (
        CPP / "reference" / "cartesian_reference.hpp",
        CPP / "reference" / "joint_reference.hpp",
        CPP / "tracker" / "cartesian_impedance_tracker.hpp",
        CPP / "tracker" / "joint_impedance_tracker.hpp",
        CPP / "tracker" / "joint_pid_tracker.hpp",
        CPP / "safety" / "torque_rate_limiter.hpp",
    )
    assert all(path.is_file() for path in expected)

    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    assert "reference.sample(" in backend
    assert "tracker.compute(" in backend
    assert "safety.apply(" in backend
    assert "updateMotionLimitedReference" not in backend
    assert "startNewSegment" not in backend

    joint_reference = (CPP / "reference" / "joint_reference.hpp").read_text(encoding="utf-8")
    assert "std::shared_ptr<const ReferenceGenerator> profile_" in joint_reference
    assert "profile_->weights(" in joint_reference
    assert "motion_limited is not implemented for joint references" in joint_reference

    joint_pid = (CPP / "tracker" / "joint_pid_tracker.hpp").read_text(encoding="utf-8")
    assert "proportional_gain{0.18}" in joint_pid
    assert "integral_gain_s{0.30}" in joint_pid
    assert "velocity_gain_s{0.04}" in joint_pid
    assert "pushing" not in joint_pid or "same_direction" in joint_pid

    safety = (CPP / "safety" / "torque_rate_limiter.hpp").read_text(encoding="utf-8")
    control = (CPP / "utils" / "control.hpp").read_text(encoding="utf-8")
    assert "kTorqueControlPeriodS = 1e-3" in control
    assert "franka::kMaxTorqueRate" in control
    assert "kTorqueControlPeriodS" in control
    assert "double dt" not in safety
    assert "state.tau_J_d, dt" not in backend


def test_realtime_ring_publishes_head_after_frame_write() -> None:
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    timing = (CPP / "utils" / "timing.hpp").read_text(encoding="utf-8")
    for source in (backend, timing):
        write_index = source.index("ring_[idx % capacity_] = frame;")
        publish_index = source.index("write_head_.store(idx + 1, std::memory_order_release);", write_index)
        assert write_index < publish_index


def test_every_realtime_control_route_records_trace_and_timing() -> None:
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    joint_start = backend.index("void runJointTrackerLoop")
    joint_end = backend.index("void runControlLoop", joint_start)
    joint_loop = backend[joint_start:joint_end]
    assert "trace_ring_.write(frame)" in joint_loop
    assert "timing_ring_.write(timing)" in joint_loop


def test_only_cpp_control_thread_keeps_realtime_priority() -> None:
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    constructor_restore = backend.index("setCurrentThreadToNormalPriority();")
    control_thread = backend.index("control_thread_ = std::thread")
    realtime_raise = backend.index("setCurrentThreadToRealtimePriority();", control_thread)
    run_loop = backend.index("runControlLoop(max_duration);")

    assert constructor_restore < control_thread < realtime_raise < run_loop
    assert 'pthread_setname_np(pthread_self(), "franka_ctrl_rt")' in backend
    reset_start = backend.index("void reset(double speed_factor")
    reset_end = backend.index("void probe_model()", reset_start)
    assert "ScopedRealtimePriority realtime_scope;" in backend[reset_start:reset_end]


def test_realtime_timing_records_host_callback_period() -> None:
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    timing = (CPP / "utils" / "timing.hpp").read_text(encoding="utf-8")
    profiler = (ROOT / "src" / "recording" / "realtime_timing.py").read_text(encoding="utf-8")

    assert '"callback_period"' in timing
    assert "loop_start - previous_callback_start" in backend
    assert '"callback_period"' in profiler


def test_joint_pose_diagnostics_are_decimated_but_torque_trace_is_not() -> None:
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    joint_start = backend.index("void runJointTrackerLoop")
    joint_end = backend.index("void runControlLoop", joint_start)
    joint_loop = backend[joint_start:joint_end]

    assert "trace_frame_index % kJointPoseTraceDecimation" in joint_loop
    assert joint_loop.index("trace_frame_index % kJointPoseTraceDecimation") < joint_loop.index(
        "frame.tau_cmd[i]"
    )
    assert joint_loop.index("frame.tau_cmd[i]") < joint_loop.index("trace_ring_.write(frame)")


def test_action_mailbox_is_bounded_lock_free_and_accepts_complete_blocks() -> None:
    queue = CPP / "utils" / "spsc_action_queue.hpp"
    assert queue.is_file()

    queue_source = queue.read_text(encoding="utf-8")
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    assert "std::atomic<uint64_t> write_head_" in queue_source
    assert "tryPushBlock" in queue_source
    assert "actionsFromArray" in backend
    assert "parsed.data(), parsed.size()" in backend
    assert "action_mutex_" not in backend
    assert "std::deque" not in backend


def test_robot_state_uses_consistent_lock_free_snapshot() -> None:
    snapshot = CPP / "utils" / "atomic_robot_state.hpp"
    assert snapshot.is_file()

    snapshot_source = snapshot.read_text(encoding="utf-8")
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    assert "AtomicRobotStateSnapshot" in snapshot_source
    assert "sequence_.fetch_add" in snapshot_source
    assert "latest_robot_state_.store(latest)" in backend
    assert "latest_robot_state_.load()" in backend
    assert "latest_robot_state_mutex_" not in backend


def test_realtime_experiment_parameters_come_from_python() -> None:
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    reference = (CPP / "reference" / "cartesian_reference.hpp").read_text(encoding="utf-8")
    tracker = (CPP / "tracker" / "joint_impedance_tracker.hpp").read_text(encoding="utf-8")
    types = (CPP / "utils" / "types.hpp").read_text(encoding="utf-8")

    assert 'config["policy_period_s"]' in backend
    assert "reference_generator_, config_.policy_period_s" in backend
    assert "joint_reference_duration_s" not in backend
    assert "config_.collision_lower_torque" in backend
    assert "position_epsilon_" in reference
    assert "JointImpedanceTracker(Vector7d stiffness, Vector7d damping)" in tracker
    assert "kPolicyPeriod" not in types


def test_control_stop_request_is_nonblocking_and_join_is_serialized() -> None:
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    env = (ROOT / "src" / "control" / "franka_env.py").read_text(encoding="utf-8")
    teleop = (ROOT / "src" / "data_collection" / "key_control.py").read_text(encoding="utf-8")

    assert "void request_stop() noexcept" in backend
    assert '.def("request_stop", &RealtimeFrankaBackend::request_stop)' in backend
    assert "control_thread_join_mutex_" in backend
    assert 'hasattr(self._backend, "request_stop")' in env

    escape_start = teleop.index('if char == "escape":')
    escape_end = teleop.index("with self._keys_lock:", escape_start)
    escape_handler = teleop[escape_start:escape_end]
    assert "self.env.request_stop()" in escape_handler
    assert "self.stop()" not in escape_handler

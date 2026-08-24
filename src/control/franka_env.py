from __future__ import annotations

import math
import pathlib
import queue
import threading
import time

import numpy as np

from client import image_tools
from devices import AsyncGripperDriver
from planning.action_planner import CartesianActionPlanner, PlannedRobotCommand
from planning.control_route import (
    JOINT_REFERENCE_CHOICES,
    ROUTE_TRACKER_MODE_CHOICES,
    TRACKER_MODE_CHOICES,
    ControlRoute,
)
from utils.control import ActionConfig, GRIPPER_WIDTH_MAX, POLICY_HZ, transform_action
from utils.pose import matrix_to_rotvec, matrix_to_rotvec_continuous, rotvec_to_matrix

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    rs = None
    HAS_REALSENSE = False

ROBOT_IP = "172.16.0.2"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CAM1_SERIAL: str | None = "346222072769"
DEFAULT_CAM2_SERIAL: str | None = "938422075745"
REFERENCE_CHOICES = ("min_jerk", "linear", "cubic", "motion_limited")
NULLSPACE_PINV_CHOICES = ("plain", "damped")
NULLSPACE_PROJECTOR_CHOICES = ("kinematic", "dynamic")
CONTROL_MODE_CHOICES = ("cartesian", "joint")
DEFAULT_MAX_TRANSLATION_VELOCITY = 0.2
DEFAULT_MAX_ROTATION_VELOCITY = math.pi / 4.0
DEFAULT_MAX_TRANSLATION_GOAL_ERROR = 0.3
DEFAULT_MAX_ROTATION_GOAL_ERROR = math.pi / 6.0
DEFAULT_MOTION_LIMITED_VELOCITY_SCALE = 1.2
DEFAULT_MOTION_LIMITED_ACCELERATION_SCALE = 5.0
DEFAULT_MAX_TORQUE_RATE = 1000.0
DEFAULT_JOINT_STIFFNESS = np.array([80.0, 80.0, 80.0, 60.0, 25.0, 15.0, 10.0], dtype=np.float64)
DEFAULT_JOINT_DAMPING = 2.0 * np.sqrt(DEFAULT_JOINT_STIFFNESS)
DEFAULT_PID_PROPORTIONAL_GAIN = 0.18
DEFAULT_PID_INTEGRAL_GAIN_S = 0.30
DEFAULT_PID_VELOCITY_GAIN_S = 0.04
DEFAULT_PID_MAXIMUM_CORRECTION_RAD = math.radians(3.0)
DEFAULT_PID_INTEGRATION_ERROR_LIMIT_RAD = math.radians(4.0)
DEFAULT_PID_INTEGRAL_TIME_CONSTANT_S = 1.0
DEFAULT_PID_STATIONARY_INTEGRAL_TIME_CONSTANT_S = 0.25
DEFAULT_PID_STATIONARY_VELOCITY_THRESHOLD_RAD_S = 0.02
DEFAULT_REFERENCE_POSITION_EPSILON = 0.0005
DEFAULT_REFERENCE_LINEAR_VELOCITY_EPSILON = 0.001
DEFAULT_REFERENCE_ROTATION_EPSILON = 0.001
DEFAULT_REFERENCE_ANGULAR_VELOCITY_EPSILON = 0.001
DEFAULT_COLLISION_TORQUE = np.array([20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0], dtype=np.float64)
DEFAULT_COLLISION_FORCE = np.array([20.0, 20.0, 20.0, 25.0, 25.0, 25.0], dtype=np.float64)

DEFAULT_HOME_Q = np.array(
    [0.0, -math.pi / 4.0, 0.0, -3.0 * math.pi / 4.0, 0.0, math.pi / 2.0, math.pi / 4.0],
    dtype=np.float64,
)
DEFAULT_STIFFNESS = np.diag([600.0, 600.0, 600.0, 25.0, 25.0, 25.0]).astype(np.float64)
DEFAULT_DAMPING = np.diag([2.0 * math.sqrt(DEFAULT_STIFFNESS[i, i]) for i in range(6)]).astype(np.float64)
GRIPPER_SPEED = 0.08
GRIPPER_FORCE = 30.0
GRIPPER_CONNECT_TIMEOUT = 8.0
GRIPPER_STATE_POLL_PERIOD = 1.0 / POLICY_HZ
GRIPPER_WIDTH_TOLERANCE = 0.003
GRIPPER_CLOSE_THRESHOLD = 1e-6
GRIPPER_GRASP_EPSILON_INNER = 0.08
GRIPPER_GRASP_EPSILON_OUTER = 0.08
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15
D435_START_RETRIES = 5
D435_START_RETRY_DELAY = 0.5
OBS_IMAGE_SIZE = 224


class D435Camera:
    def __init__(
        self,
        serial_number: str | None = None,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ) -> None:
        self._serial = serial_number
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._pipeline = None
        self._latest_frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._frame_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None

    def _release_pipeline(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

    def start(self) -> None:
        if not HAS_REALSENSE or rs is None:
            raise RuntimeError("pyrealsense2 未安装，无法启动相机")
        if self._pipeline is not None:
            self.stop()

        pipeline = rs.pipeline()
        try:
            config = rs.config()
            if self._serial:
                config.enable_device(self._serial)
            config.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
            pipeline.start(config)
            self._pipeline = pipeline
            self._stop_event.clear()
            self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader_thread.start()
            return
        except Exception as exc:
            self._release_pipeline()
            raise RuntimeError(f"相机 {self._serial or '<auto>'} 启动失败: {exc}") from exc

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set() and self._pipeline is not None:
            try:
                frames = self._pipeline.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data())
                with self._frame_lock:
                    self._latest_frame = frame.copy()
            except Exception as exc:
                print(f"  [相机] 读帧失败，将保持上一帧缓存: {exc}", flush=True)
                self._stop_event.set()
                self._release_pipeline()
                break

    def get_frame(self) -> np.ndarray:
        with self._frame_lock:
            return self._latest_frame.copy()

    def stop(self) -> None:
        self._stop_event.set()
        if (
            self._reader_thread is not None
            and self._reader_thread.is_alive()
            and threading.current_thread() is not self._reader_thread
        ):
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None
        self._release_pipeline()


class DualD435:
    def __init__(self, cam1_serial: str | None = None, cam2_serial: str | None = None) -> None:
        self._cam1 = D435Camera(cam1_serial)
        self._cam2 = D435Camera(cam2_serial)
        self._black = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    def start(self) -> None:
        for cam in (self._cam1, self._cam2):
            try:
                cam.start()
            except Exception as exc:
                print(f"  [相机] 启动失败，将使用全黑帧: {exc}", flush=True)
            time.sleep(0.2)

    def stop(self) -> None:
        for cam in (self._cam1, self._cam2):
            try:
                cam.stop()
            except Exception:
                pass

    def get_frames(self) -> tuple[np.ndarray, np.ndarray]:
        frames = []
        for cam in (self._cam1, self._cam2):
            if cam._pipeline is None:
                frames.append(self._black.copy())
                continue
            try:
                frames.append(cam.get_frame())
            except Exception as exc:
                print(f"  [相机] 读帧失败，返回全黑帧: {exc}", flush=True)
                cam._release_pipeline()
                frames.append(self._black.copy())
        return frames[0], frames[1]


def _resize_observation_image(image: np.ndarray) -> np.ndarray:
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(image, OBS_IMAGE_SIZE, OBS_IMAGE_SIZE))


def _validate_control_mode(control_mode: str) -> str:
    control_mode = str(control_mode)
    if control_mode not in CONTROL_MODE_CHOICES:
        raise ValueError(f"control_mode must be one of {CONTROL_MODE_CHOICES}; got {control_mode!r}")
    return control_mode


def _validate_reference_name(reference_name: str) -> str:
    reference_name = str(reference_name)
    if reference_name not in REFERENCE_CHOICES:
        raise ValueError(f"reference_name must be one of {REFERENCE_CHOICES}; got {reference_name!r}")
    return reference_name


def _validate_nullspace_pinv(mode: str) -> str:
    mode = str(mode)
    if mode not in NULLSPACE_PINV_CHOICES:
        raise ValueError(f"nullspace_pinv must be one of {NULLSPACE_PINV_CHOICES}; got {mode!r}")
    return mode


def _validate_nullspace_projector(mode: str) -> str:
    mode = str(mode)
    if mode not in NULLSPACE_PROJECTOR_CHOICES:
        raise ValueError(f"nullspace_projector must be one of {NULLSPACE_PROJECTOR_CHOICES}; got {mode!r}")
    return mode


def _validate_task_constraint_mask(mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return np.ones(6, dtype=np.float64)
    array = np.asarray(mask, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"task_constraint_mask must have shape (6,); got {array.shape}")
    validated = np.where(array > 0.5, 1.0, 0.0).astype(np.float64, copy=False)
    if not np.any(validated):
        raise ValueError("task_constraint_mask must keep at least one constrained task dimension")
    return validated


class _NoRobotBackend:
    def __init__(
        self,
        max_translation_velocity: float,
        max_rotation_velocity: float,
        max_translation_goal_error: float = DEFAULT_MAX_TRANSLATION_GOAL_ERROR,
        max_rotation_goal_error: float = DEFAULT_MAX_ROTATION_GOAL_ERROR,
        motion_limited_max_translation_velocity: float | None = None,
        motion_limited_max_rotation_velocity: float | None = None,
        motion_limited_max_translation_acceleration: float | None = None,
        motion_limited_max_rotation_acceleration: float | None = None,
        max_torque_rate: float = DEFAULT_MAX_TORQUE_RATE,
        reference_name: str = "min_jerk",
        control_mode: str = "cartesian",
        home_q: np.ndarray | None = None,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
        nullspace_enabled: bool = False,
        nullspace_q_target: np.ndarray | None = None,
        nullspace_stiffness: float = 10.0,
        nullspace_damping: float = 2.0,
        nullspace_pinv: str = "plain",
        nullspace_projector: str = "kinematic",
        nullspace_lambda: float = 0.05,
        task_constraint_mask: np.ndarray | None = None,
        control_hz: float = POLICY_HZ,
    ):
        del stiffness, damping, nullspace_q_target
        self.max_translation_velocity = float(max_translation_velocity)
        self.max_rotation_velocity = float(max_rotation_velocity)
        self.max_translation_goal_error = float(max_translation_goal_error)
        self.max_rotation_goal_error = float(max_rotation_goal_error)
        self.motion_limited_max_translation_velocity = float(
            motion_limited_max_translation_velocity
            if motion_limited_max_translation_velocity is not None
            else DEFAULT_MOTION_LIMITED_VELOCITY_SCALE * self.max_translation_velocity
        )
        self.motion_limited_max_rotation_velocity = float(
            motion_limited_max_rotation_velocity
            if motion_limited_max_rotation_velocity is not None
            else DEFAULT_MOTION_LIMITED_VELOCITY_SCALE * self.max_rotation_velocity
        )
        self.motion_limited_max_translation_acceleration = float(
            motion_limited_max_translation_acceleration
            if motion_limited_max_translation_acceleration is not None
            else DEFAULT_MOTION_LIMITED_ACCELERATION_SCALE * self.motion_limited_max_translation_velocity
        )
        self.motion_limited_max_rotation_acceleration = float(
            motion_limited_max_rotation_acceleration
            if motion_limited_max_rotation_acceleration is not None
            else DEFAULT_MOTION_LIMITED_ACCELERATION_SCALE * self.motion_limited_max_rotation_velocity
        )
        self.max_torque_rate = float(max_torque_rate)
        self.control_hz = float(control_hz)
        self.max_translation_step = self.max_translation_velocity / self.control_hz
        self.max_rotation_step = self.max_rotation_velocity / self.control_hz
        self.reference_name = _validate_reference_name(reference_name)
        self.control_mode = _validate_control_mode(control_mode)
        self.nullspace_enabled = bool(nullspace_enabled)
        self.nullspace_stiffness = float(nullspace_stiffness)
        self.nullspace_damping = float(nullspace_damping)
        self.nullspace_pinv = _validate_nullspace_pinv(nullspace_pinv)
        self.nullspace_projector = _validate_nullspace_projector(nullspace_projector)
        self.nullspace_lambda = float(nullspace_lambda)
        self.task_constraint_mask = _validate_task_constraint_mask(task_constraint_mask)
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._joint_target_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._home_q = np.asarray(home_q if home_q is not None else DEFAULT_HOME_Q, dtype=np.float64).copy()
        self._joint_positions = self._home_q.copy()
        self._state = np.array([0.0, 0.0, 0.0, math.pi, 0.0, 0.0, GRIPPER_WIDTH_MAX, GRIPPER_WIDTH_MAX], dtype=np.float64)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def enqueue_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        self._queue.put((action[0] if action.ndim == 2 else action).copy())

    def enqueue_joint_target(self, target: np.ndarray) -> None:
        self._joint_target_queue.put(np.asarray(target, dtype=np.float64).copy())

    def clear_actions(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._joint_target_queue.get_nowait()
            except queue.Empty:
                return

    def get_pending_action_count(self) -> int:
        return int(self._queue.qsize() + self._joint_target_queue.qsize())

    def get_joint_positions(self) -> np.ndarray:
        return self._joint_positions.copy()

    def start_control_loop(self, max_duration: float = -1.0) -> None:
        if self.is_running():
            raise RuntimeError("control thread is already running")
        self._stop.clear()

        def run() -> None:
            start = time.monotonic()
            while not self._stop.is_set():
                latest_target = None
                while True:
                    try:
                        latest_target = self._joint_target_queue.get_nowait()
                    except queue.Empty:
                        break
                if latest_target is not None:
                    self._joint_positions = latest_target.copy()
                try:
                    action = self._queue.get_nowait()
                except queue.Empty:
                    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
                if self.control_mode == "joint" and action.shape == (8,):
                    self._joint_positions += action[:7]
                    self._state[6:] = 0.0 if action[7] > 0.0 else GRIPPER_WIDTH_MAX
                else:
                    self._state[:3] += np.clip(action[:3], -self.max_translation_step, self.max_translation_step)
                    self._state[3:6] += np.clip(action[3:6], -self.max_rotation_step, self.max_rotation_step)
                    self._state[6:] = 0.0 if action[6] > 0.0 else GRIPPER_WIDTH_MAX
                if max_duration > 0.0 and time.monotonic() - start >= max_duration:
                    break
                time.sleep(0.1)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def stop(self) -> None:
        self._stop.set()
        self.wait()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_reference(self, reference_name: str) -> None:
        if self.is_running():
            raise RuntimeError("cannot change reference while control thread is running")
        self.reference_name = _validate_reference_name(reference_name)

    def reset(self, speed_factor: float = 0.5, reset_duration: float = -1.0) -> None:
        del speed_factor, reset_duration
        self.clear_actions()
        self._joint_positions = self._home_q.copy()
        self._state = np.array([0.0, 0.0, 0.0, math.pi, 0.0, 0.0, GRIPPER_WIDTH_MAX, GRIPPER_WIDTH_MAX], dtype=np.float64)

    def get_robot_state_vector(self) -> np.ndarray:
        return self._state.copy()

    def get_trace_head(self) -> int:
        return 0

    def get_trace_since(self, after: int = 0) -> np.ndarray:
        del after
        return np.zeros((0, 47), dtype=np.float64)

    def clear_trace(self) -> None:
        pass

    def get_timing_head(self) -> int:
        return 0

    def get_timing_since(self, after: int = 0) -> np.ndarray:
        del after
        from recording.realtime_timing import TIMING_FIELDS

        return np.zeros((0, 3 + len(TIMING_FIELDS)), dtype=np.float64)

    def get_timing_field_names(self) -> list[str]:
        from recording.realtime_timing import TIMING_FIELDS

        return list(TIMING_FIELDS)


class FrankaEnv:
    """Python high-level interface; realtime torque control runs in C++."""

    def __init__(
        self,
        robot_ip: str = ROBOT_IP,
        *,
        home_q: np.ndarray | None = None,
        reset_duration: float = 5.0,
        max_translation_velocity: float = DEFAULT_MAX_TRANSLATION_VELOCITY,
        max_rotation_velocity: float = DEFAULT_MAX_ROTATION_VELOCITY,
        control_hz: float = POLICY_HZ,
        max_translation_goal_error: float = DEFAULT_MAX_TRANSLATION_GOAL_ERROR,
        max_rotation_goal_error: float = DEFAULT_MAX_ROTATION_GOAL_ERROR,
        motion_limited_max_translation_velocity: float | None = None,
        motion_limited_max_rotation_velocity: float | None = None,
        motion_limited_max_translation_acceleration: float | None = None,
        motion_limited_max_rotation_acceleration: float | None = None,
        max_torque_rate: float = DEFAULT_MAX_TORQUE_RATE,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
        reference_name: str = "min_jerk",
        control_mode: str = "cartesian",
        action_planner: CartesianActionPlanner | None = None,
        tracker_mode: str = "auto",
        joint_stiffness: np.ndarray | None = None,
        joint_damping: np.ndarray | None = None,
        pid_proportional_gain: float = DEFAULT_PID_PROPORTIONAL_GAIN,
        pid_integral_gain_s: float = DEFAULT_PID_INTEGRAL_GAIN_S,
        pid_velocity_gain_s: float = DEFAULT_PID_VELOCITY_GAIN_S,
        pid_maximum_correction_rad: float = DEFAULT_PID_MAXIMUM_CORRECTION_RAD,
        pid_integration_error_limit_rad: float = DEFAULT_PID_INTEGRATION_ERROR_LIMIT_RAD,
        pid_integral_time_constant_s: float = DEFAULT_PID_INTEGRAL_TIME_CONSTANT_S,
        pid_stationary_integral_time_constant_s: float = DEFAULT_PID_STATIONARY_INTEGRAL_TIME_CONSTANT_S,
        pid_stationary_velocity_threshold_rad_s: float = DEFAULT_PID_STATIONARY_VELOCITY_THRESHOLD_RAD_S,
        reference_position_epsilon: float = DEFAULT_REFERENCE_POSITION_EPSILON,
        reference_linear_velocity_epsilon: float = DEFAULT_REFERENCE_LINEAR_VELOCITY_EPSILON,
        reference_rotation_epsilon: float = DEFAULT_REFERENCE_ROTATION_EPSILON,
        reference_angular_velocity_epsilon: float = DEFAULT_REFERENCE_ANGULAR_VELOCITY_EPSILON,
        collision_lower_torque: np.ndarray | None = None,
        collision_upper_torque: np.ndarray | None = None,
        collision_lower_force: np.ndarray | None = None,
        collision_upper_force: np.ndarray | None = None,
        nullspace_enabled: bool = False,
        nullspace_q_target: np.ndarray | None = None,
        nullspace_stiffness: float = 10.0,
        nullspace_damping: float = 2.0,
        nullspace_pinv: str = "plain",
        nullspace_projector: str = "kinematic",
        nullspace_lambda: float = 0.05,
        task_constraint_mask: np.ndarray | None = None,
        reset_speed_factor: float = 0.5,
        trace_capacity_sec: float = 180.0,
        use_gripper: bool = True,
        gripper_speed: float = GRIPPER_SPEED,
        gripper_force: float = GRIPPER_FORCE,
        gripper_width_tolerance: float = GRIPPER_WIDTH_TOLERANCE,
        gripper_close_threshold: float = GRIPPER_CLOSE_THRESHOLD,
        gripper_grasp_epsilon_inner: float = GRIPPER_GRASP_EPSILON_INNER,
        gripper_grasp_epsilon_outer: float = GRIPPER_GRASP_EPSILON_OUTER,
        no_robot: bool = False,
        no_cameras: bool = True,
        cam1_serial: str | None = DEFAULT_CAM1_SERIAL,
        cam2_serial: str | None = DEFAULT_CAM2_SERIAL,
        print_events: bool = True,
        auto_record: bool | None = None,
        save_recording: bool = False,
        log_root: pathlib.Path | str | None = None,
        log_subdir: str = "runs",
        print_timing_summary: bool = True,
    ):
        self.robot_ip = robot_ip
        self.max_translation_velocity = float(max_translation_velocity)
        self.max_rotation_velocity = float(max_rotation_velocity)
        self.max_translation_goal_error = float(max_translation_goal_error)
        self.max_rotation_goal_error = float(max_rotation_goal_error)
        self.motion_limited_max_translation_velocity = float(
            motion_limited_max_translation_velocity
            if motion_limited_max_translation_velocity is not None
            else DEFAULT_MOTION_LIMITED_VELOCITY_SCALE * self.max_translation_velocity
        )
        self.motion_limited_max_rotation_velocity = float(
            motion_limited_max_rotation_velocity
            if motion_limited_max_rotation_velocity is not None
            else DEFAULT_MOTION_LIMITED_VELOCITY_SCALE * self.max_rotation_velocity
        )
        self.motion_limited_max_translation_acceleration = float(
            motion_limited_max_translation_acceleration
            if motion_limited_max_translation_acceleration is not None
            else DEFAULT_MOTION_LIMITED_ACCELERATION_SCALE * self.motion_limited_max_translation_velocity
        )
        self.motion_limited_max_rotation_acceleration = float(
            motion_limited_max_rotation_acceleration
            if motion_limited_max_rotation_acceleration is not None
            else DEFAULT_MOTION_LIMITED_ACCELERATION_SCALE * self.motion_limited_max_rotation_velocity
        )
        self.max_torque_rate = float(max_torque_rate)
        self.control_hz = float(control_hz)
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        self.max_translation_step = self.max_translation_velocity / self.control_hz
        self.max_rotation_step = self.max_rotation_velocity / self.control_hz
        self.reset_duration = float(reset_duration)
        self.reset_speed_factor = float(reset_speed_factor)
        reference_name = _validate_reference_name(reference_name)
        self.action_planner = action_planner
        if self.action_planner is not None:
            self.control_route = ControlRoute(self.action_planner, reference_name, tracker_mode)
            control_mode = self.control_route.control_mode
        elif control_mode == "cartesian":
            self.action_planner = CartesianActionPlanner()
            self.control_route = ControlRoute(self.action_planner, reference_name, tracker_mode)
            control_mode = self.control_route.control_mode
        else:
            tracker_mode = str(tracker_mode).lower().strip()
            if tracker_mode not in ROUTE_TRACKER_MODE_CHOICES:
                raise ValueError(
                    f"tracker_mode must be one of {ROUTE_TRACKER_MODE_CHOICES}; got {tracker_mode!r}"
                )
            if tracker_mode not in {"auto", "pid", "joint_impedance", "joint_pid"}:
                raise ValueError(
                    "raw joint control requires tracker_mode='auto', 'pid', "
                    "'joint_impedance', or 'joint_pid'"
                )
            if reference_name not in JOINT_REFERENCE_CHOICES:
                raise ValueError(
                    f"joint reference must be one of {JOINT_REFERENCE_CHOICES}; got {reference_name!r}"
                )
            self.control_route = None
        self.control_mode = _validate_control_mode(control_mode)
        self.tracker_mode = (
            self.control_route.tracker_mode
            if self.control_route is not None
            else ("joint_pid" if tracker_mode in TRACKER_MODE_CHOICES else tracker_mode)
        )
        self.reference_name = reference_name
        self.latest_planner_telemetry: dict[str, object] | None = None
        self.joint_stiffness = np.asarray(
            DEFAULT_JOINT_STIFFNESS if joint_stiffness is None else joint_stiffness, dtype=np.float64
        )
        self.joint_damping = np.asarray(
            DEFAULT_JOINT_DAMPING if joint_damping is None else joint_damping, dtype=np.float64
        )
        self.pid_proportional_gain = float(pid_proportional_gain)
        self.pid_integral_gain_s = float(pid_integral_gain_s)
        self.pid_velocity_gain_s = float(pid_velocity_gain_s)
        self.pid_maximum_correction_rad = float(pid_maximum_correction_rad)
        self.pid_integration_error_limit_rad = float(pid_integration_error_limit_rad)
        self.pid_integral_time_constant_s = float(pid_integral_time_constant_s)
        self.pid_stationary_integral_time_constant_s = float(pid_stationary_integral_time_constant_s)
        self.pid_stationary_velocity_threshold_rad_s = float(pid_stationary_velocity_threshold_rad_s)
        pid_values = np.asarray(
            [
                self.pid_proportional_gain,
                self.pid_integral_gain_s,
                self.pid_velocity_gain_s,
                self.pid_maximum_correction_rad,
                self.pid_integration_error_limit_rad,
                self.pid_integral_time_constant_s,
                self.pid_stationary_integral_time_constant_s,
                self.pid_stationary_velocity_threshold_rad_s,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(pid_values)) or np.any(pid_values < 0.0):
            raise ValueError("PID tracker settings must be finite and non-negative")
        if (
            self.pid_maximum_correction_rad <= 0.0
            or self.pid_integral_time_constant_s <= 0.0
            or self.pid_stationary_integral_time_constant_s <= 0.0
        ):
            raise ValueError("PID correction limit and integral time constants must be positive")
        self.reference_position_epsilon = float(reference_position_epsilon)
        self.reference_linear_velocity_epsilon = float(reference_linear_velocity_epsilon)
        self.reference_rotation_epsilon = float(reference_rotation_epsilon)
        self.reference_angular_velocity_epsilon = float(reference_angular_velocity_epsilon)
        self.collision_lower_torque = np.asarray(
            DEFAULT_COLLISION_TORQUE if collision_lower_torque is None else collision_lower_torque, dtype=np.float64
        )
        self.collision_upper_torque = np.asarray(
            DEFAULT_COLLISION_TORQUE if collision_upper_torque is None else collision_upper_torque, dtype=np.float64
        )
        self.collision_lower_force = np.asarray(
            DEFAULT_COLLISION_FORCE if collision_lower_force is None else collision_lower_force, dtype=np.float64
        )
        self.collision_upper_force = np.asarray(
            DEFAULT_COLLISION_FORCE if collision_upper_force is None else collision_upper_force, dtype=np.float64
        )
        self.trace_capacity_sec = float(trace_capacity_sec)
        self.home_q = np.asarray(home_q if home_q is not None else DEFAULT_HOME_Q, dtype=np.float64)
        self.stiffness = np.asarray(stiffness if stiffness is not None else DEFAULT_STIFFNESS, dtype=np.float64)
        self.damping = np.asarray(damping if damping is not None else DEFAULT_DAMPING, dtype=np.float64)
        self.nullspace_enabled = bool(nullspace_enabled)
        self.nullspace_q_target = np.asarray(
            nullspace_q_target if nullspace_q_target is not None else self.home_q,
            dtype=np.float64,
        )
        if self.nullspace_q_target.shape != (7,):
            raise ValueError(f"nullspace_q_target must have shape (7,); got {self.nullspace_q_target.shape}")
        self.nullspace_stiffness = float(nullspace_stiffness)
        self.nullspace_damping = float(nullspace_damping)
        self.nullspace_pinv = _validate_nullspace_pinv(nullspace_pinv)
        self.nullspace_projector = _validate_nullspace_projector(nullspace_projector)
        self.nullspace_lambda = float(nullspace_lambda)
        self.task_constraint_mask = _validate_task_constraint_mask(task_constraint_mask)
        self.action_config = ActionConfig(self.max_translation_velocity, self.max_rotation_velocity, self.control_hz)
        self.no_robot = bool(no_robot)
        self.no_cameras = bool(no_cameras)
        self.robot = None if self.no_robot else object()
        self.cam1_serial = cam1_serial
        self.cam2_serial = cam2_serial
        self._cameras: DualD435 | None = None if self.no_cameras else DualD435(cam1_serial, cam2_serial)
        if self._cameras is not None:
            self._cameras.start()
        self.print_events = bool(print_events)
        self.auto_record = (not self.no_robot) if auto_record is None else bool(auto_record)
        self.save_recording = bool(save_recording)
        self.log_root = pathlib.Path(log_root) if log_root is not None else REPO_ROOT / "logs" / log_subdir
        self.print_timing_summary = bool(print_timing_summary)
        self._active_run_paths = None
        self._record_finalized = True
        self._policy_tick = 0
        self._last_torque_print_trace_head = 0
        self._last_gripper_target: float = GRIPPER_WIDTH_MAX
        self._last_gripper_width: float = GRIPPER_WIDTH_MAX
        self._gripper_enabled = False
        self._gripper_driver: AsyncGripperDriver | None = None
        self._gripper_speed = float(gripper_speed)
        self._gripper_force = float(gripper_force)
        self.gripper_width_tolerance = float(gripper_width_tolerance)
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.gripper_grasp_epsilon_inner = float(gripper_grasp_epsilon_inner)
        self.gripper_grasp_epsilon_outer = float(gripper_grasp_epsilon_outer)
        self.commanded_pose_array = np.zeros(6, dtype=np.float64)
        self.ee_force_torque = np.zeros(6, dtype=np.float64)
        self._last_goal_rotation_error = np.zeros(3, dtype=np.float64)

        if self.no_robot:
            self._backend = _NoRobotBackend(
                self.max_translation_velocity,
                self.max_rotation_velocity,
                self.max_translation_goal_error,
                self.max_rotation_goal_error,
                self.motion_limited_max_translation_velocity,
                self.motion_limited_max_rotation_velocity,
                self.motion_limited_max_translation_acceleration,
                self.motion_limited_max_rotation_acceleration,
                self.max_torque_rate,
                self.reference_name,
                self.control_mode,
                self.home_q,
                self.stiffness,
                self.damping,
                self.nullspace_enabled,
                self.nullspace_q_target,
                self.nullspace_stiffness,
                self.nullspace_damping,
                self.nullspace_pinv,
                self.nullspace_projector,
                self.nullspace_lambda,
                self.task_constraint_mask,
                control_hz=self.control_hz,
            )
        else:
            from control._franka_backend import RealtimeFrankaBackend

            self._backend = RealtimeFrankaBackend(
                self.robot_ip,
                self.max_translation_step,
                self.max_rotation_step,
                self.motion_limited_max_translation_velocity,
                self.motion_limited_max_rotation_velocity,
                self.motion_limited_max_translation_acceleration,
                self.motion_limited_max_rotation_acceleration,
                self.max_torque_rate,
                self.reference_name,
                self.control_mode,
                self.trace_capacity_sec,
                self.home_q,
                self.stiffness,
                self.damping,
                self.nullspace_enabled,
                self.nullspace_q_target,
                self.nullspace_stiffness,
                self.nullspace_damping,
                self.nullspace_pinv,
                self.nullspace_projector,
                self.nullspace_lambda,
                self.task_constraint_mask,
                {
                    "tracker_mode": self.tracker_mode,
                    "policy_period_s": 1.0 / self.control_hz,
                    "joint_stiffness": self.joint_stiffness,
                    "joint_damping": self.joint_damping,
                    "pid_proportional_gain": self.pid_proportional_gain,
                    "pid_integral_gain_s": self.pid_integral_gain_s,
                    "pid_velocity_gain_s": self.pid_velocity_gain_s,
                    "pid_maximum_correction_rad": self.pid_maximum_correction_rad,
                    "pid_integration_error_limit_rad": self.pid_integration_error_limit_rad,
                    "pid_integral_time_constant_s": self.pid_integral_time_constant_s,
                    "pid_stationary_integral_time_constant_s": self.pid_stationary_integral_time_constant_s,
                    "pid_stationary_velocity_threshold_rad_s": self.pid_stationary_velocity_threshold_rad_s,
                    "reference_position_epsilon": self.reference_position_epsilon,
                    "reference_linear_velocity_epsilon": self.reference_linear_velocity_epsilon,
                    "reference_rotation_epsilon": self.reference_rotation_epsilon,
                    "reference_angular_velocity_epsilon": self.reference_angular_velocity_epsilon,
                    "collision_lower_torque": self.collision_lower_torque,
                    "collision_upper_torque": self.collision_upper_torque,
                    "collision_lower_force": self.collision_lower_force,
                    "collision_upper_force": self.collision_upper_force,
                    "gripper_width_max": GRIPPER_WIDTH_MAX,
                },
            )
            if use_gripper:
                self._start_gripper_driver(float(gripper_speed), float(gripper_force))
        self._refresh_status_from_state()

    def _start_gripper_driver(self, speed: float, force: float) -> None:
        from control._franka_backend import RealtimeGripperBackend

        driver = AsyncGripperDriver(
            self.robot_ip,
            speed=speed,
            force=force,
            width_max=GRIPPER_WIDTH_MAX,
            poll_period=GRIPPER_STATE_POLL_PERIOD,
            connect_timeout=GRIPPER_CONNECT_TIMEOUT,
            backend_factory=lambda robot_ip: RealtimeGripperBackend(
                robot_ip,
                self.gripper_width_tolerance,
                self.gripper_close_threshold,
                self.gripper_grasp_epsilon_inner,
                self.gripper_grasp_epsilon_outer,
            ),
        )
        self._gripper_driver = driver
        try:
            driver.start()
            self._last_gripper_width = driver.width
            self._gripper_enabled = driver.enabled
            print("  [夹爪] worker 已启动")
        except Exception as exc:
            print(f"  [夹爪] worker 启动失败: {exc}")
            self._stop_gripper_driver()

    def _stop_gripper_driver(self) -> None:
        driver = self._gripper_driver
        self._gripper_driver = None
        self._gripper_enabled = False
        if driver is not None:
            if not driver.stop(timeout=2.0):
                print("  [夹爪] worker 线程未在 2s 内退出", flush=True)

    def _set_gripper_target(self, target: float) -> None:
        target = float(np.clip(target, 0.0, GRIPPER_WIDTH_MAX))
        self._refresh_gripper_state()
        self._last_gripper_target = target
        driver = self._gripper_driver
        if not self._gripper_enabled or driver is None:
            return
        driver.set_target(target)

    def _format_torque_window(self) -> str:
        try:
            trace = self.get_trace_since(self._last_torque_print_trace_head)
            self._last_torque_print_trace_head = self.read_trace_head()
        except Exception:
            return ""
        if trace.shape[0] == 0 or trace.shape[1] < 47:
            return ""

        tau_cmd = trace[:, 19:26]
        tau_desired = trace[:, 26:33]
        tau_j_d = trace[:, 33:40]
        tau_j = trace[:, 40:47]
        tau_max = np.max(np.abs(tau_cmd), axis=0)
        tau_text = ", ".join(f"{value:+.3f}" for value in tau_max)
        frame_index = trace.shape[0] - 1
        jd_rate = float("nan")
        jd_rate_joint = 0
        if frame_index > 0:
            diff = np.abs(tau_j_d[frame_index] - tau_j_d[frame_index - 1])
            flat_idx = int(np.argmax(diff))
            jd_rate = diff[flat_idx]
            jd_rate_joint = flat_idx + 1

        return f" tau_abs_max=[{tau_text}] jd_rate={jd_rate:.3f}@J{jd_rate_joint}"

    def _print_action_event(
        self,
        action: np.ndarray,
        *,
        planned_command: PlannedRobotCommand | None = None,
    ) -> None:
        if not self.print_events:
            return
        if action.shape == (8,):
            dq_deg = np.degrees(action[:7])
            dq_text = ",".join(f"{value:+.1f}" for value in dq_deg)
            gripper = "close" if action[7] > 0.0 else "open"
            print(f"10Hz joint tick {self._policy_tick + 1:04d} dq_deg=[{dq_text}] gripper={gripper}", flush=True)
            return
        torque_window = self._format_torque_window()
        dx, dy, dz = action[0], action[1], action[2]
        drx, dry, drz = action[3], action[4], action[5]
        sqp_text = ""
        pose_error_text = ""
        if planned_command is not None and self.action_planner is not None:
            telemetry = planned_command.telemetry
            if telemetry is not None:
                sqp_text = (
                    f" sqp_status={telemetry.get('status', 'unknown')}"
                    f" feasible={telemetry.get('feasible', False)}"
                    f" iter={telemetry.get('iterations', 0)}"
                    f" solve_ms={float(telemetry.get('elapsed_ms', 0.0)):.2f}"
                )
            actual_pose = planned_command.actual_pose
            planned_pose = planned_command.planned_pose
            if planned_command.reference_space == "cartesian":
                actual_pose = self._pose_array_to_transform(self.get_robot_state_vector()[:6])
                planned_pose = self._pose_array_to_transform(self.commanded_pose_array)
            if actual_pose is not None and planned_pose is not None:
                actual_plan_position, actual_plan_rotation = self._pose_delta(actual_pose, planned_pose)
                pose_error_text += self._format_pose_delta(
                    "actual_plan", actual_plan_position, actual_plan_rotation
                )
        print(
            f"10Hz tick {self._policy_tick + 1:04d} "
            f"dxyz=[{dx:+.3f},{dy:+.3f},{dz:+.3f}]  "
            f"drot=[{drx:+.3f},{dry:+.3f},{drz:+.3f}]"
            f"{pose_error_text}"
            f"{torque_window}"
            f"{sqp_text}",
            flush=True,
        )

    def _refresh_gripper_state(self) -> None:
        driver = self._gripper_driver
        if driver is not None:
            self._last_gripper_width = driver.width
            self._gripper_enabled = driver.enabled

    def _refresh_status_from_state(self) -> np.ndarray:
        state = self.get_robot_state_vector()
        self.commanded_pose_array = state[:6].copy()
        self._last_goal_rotation_error = np.zeros(3, dtype=np.float64)
        self.ee_force_torque = np.zeros(6, dtype=np.float64)
        return state

    @staticmethod
    def _clamp_norm(value: np.ndarray, limit: float) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm <= limit or norm < 1e-12:
            return vector.copy()
        return vector * (limit / norm)

    @staticmethod
    def _pose_array_to_transform(pose: np.ndarray) -> np.ndarray:
        pose_array = np.asarray(pose, dtype=np.float64)
        if pose_array.shape != (6,):
            raise ValueError(f"pose must have shape (6,); got {pose_array.shape}")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotvec_to_matrix(pose_array[3:6])
        transform[:3, 3] = pose_array[:3]
        return transform

    @staticmethod
    def _pose_delta(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        source_pose = np.asarray(source, dtype=np.float64)
        target_pose = np.asarray(target, dtype=np.float64)
        position_delta = target_pose[:3, 3] - source_pose[:3, 3]
        rotation_delta = matrix_to_rotvec(target_pose[:3, :3] @ source_pose[:3, :3].T)
        return position_delta, rotation_delta

    @staticmethod
    def _format_pose_delta(name: str, position: np.ndarray, rotation: np.ndarray) -> str:
        return (
            f" {name}_dxyz_m=[{position[0]:+.4f},{position[1]:+.4f},{position[2]:+.4f}]"
            f" {name}_drot_rad=[{rotation[0]:+.4f},{rotation[1]:+.4f},{rotation[2]:+.4f}]"
        )

    def _prepare_cartesian_action(self, action: np.ndarray) -> np.ndarray:
        transformed = transform_action(action, self.action_config)
        current_goal = np.asarray(self.commanded_pose_array, dtype=np.float64).copy()
        if current_goal.shape != (6,):
            current_goal = self.get_robot_state_vector()[:6].copy()
        actual_state = self.get_robot_state_vector()[:6].copy()

        candidate_position = current_goal[:3] + transformed[:3]
        limited_position = actual_state[:3] + self._clamp_norm(
            candidate_position - actual_state[:3], self.max_translation_goal_error
        )

        current_rotation = rotvec_to_matrix(current_goal[3:6])
        actual_rotation = rotvec_to_matrix(actual_state[3:6])
        candidate_rotation = rotvec_to_matrix(transformed[3:6]) @ current_rotation
        rotation_error = matrix_to_rotvec_continuous(
            candidate_rotation @ actual_rotation.T, self._last_goal_rotation_error
        )
        self._last_goal_rotation_error = rotation_error.copy()
        limited_rotation = rotvec_to_matrix(
            self._clamp_norm(rotation_error, self.max_rotation_goal_error)
        ) @ actual_rotation

        limited_goal = np.zeros(6, dtype=np.float64)
        limited_goal[:3] = limited_position
        limited_goal[3:6] = matrix_to_rotvec_continuous(limited_rotation, current_goal[3:6])

        limited_action = transformed.copy()
        limited_action[:3] = limited_goal[:3] - current_goal[:3]
        limited_action[3:6] = matrix_to_rotvec_continuous(
            limited_rotation @ current_rotation.T, transformed[3:6]
        )
        self.commanded_pose_array = limited_goal
        return limited_action

    def enqueue_action_block(
        self,
        action_block: np.ndarray,
        *,
        print_event: bool = True,
        planned_command: PlannedRobotCommand | None = None,
    ) -> None:
        block = np.asarray(action_block, dtype=np.float64)
        first = block[0] if block.ndim == 2 else block
        expected_dim = 8 if self.control_mode == "joint" else 7
        if first.shape != (expected_dim,):
            raise ValueError(
                f"{self.control_mode} action must have shape ({expected_dim},) or (N, {expected_dim}); got {block.shape}"
            )
        if self.control_mode == "cartesian":
            if block.ndim == 1:
                transformed = self._prepare_cartesian_action(first)
                self._set_gripper_target(transformed[6])
                block = transformed
            else:
                transformed_rows = np.stack([self._prepare_cartesian_action(row) for row in block], axis=0)
                self._set_gripper_target(transformed_rows[-1, 6])
                block = transformed_rows
        else:
            self._set_gripper_target(GRIPPER_WIDTH_MAX if first[7] <= 0.0 else 0.0)
        if print_event:
            self._print_action_event(first, planned_command=planned_command)
        self._backend.enqueue_action(block)
        self._policy_tick += 1

    def enqueue_action(self, action: np.ndarray) -> None:
        self.enqueue_action_block(action)

    def enqueue_cartesian_action(
        self,
        action: np.ndarray,
        *,
        semantic_key=None,
    ) -> PlannedRobotCommand:
        """Plan and enqueue one physical 7D Cartesian action through the configured planner."""
        if self.action_planner is None:
            raise RuntimeError("this FrankaEnv was created for raw joint control and has no Cartesian planner")
        command = self.action_planner.plan(
            self.get_joint_positions(),
            np.asarray(action, dtype=np.float64),
            self.action_config,
            semantic_key=semantic_key,
        )
        if command.reference_space == "cartesian":
            assert command.cartesian_action is not None
            self.enqueue_action_block(command.cartesian_action, planned_command=command)
        else:
            assert command.joint_target is not None
            self._print_action_event(np.asarray(action, dtype=np.float64), planned_command=command)
            self.enqueue_joint_target(command.joint_target, gripper_target=command.gripper_target)
        self.latest_planner_telemetry = command.telemetry
        return command

    def reset_action_planner(self) -> None:
        if self.action_planner is not None:
            self.action_planner.reset(self.get_joint_positions())
        self.latest_planner_telemetry = None

    def enqueue_joint_target(self, target: np.ndarray, *, gripper_target: float | None = None) -> None:
        if self.control_mode != "joint":
            raise RuntimeError("joint targets require control_mode='joint'")
        target = np.asarray(target, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError(f"joint target must have shape (7,); got {target.shape}")
        if gripper_target is not None:
            self._set_gripper_target(float(gripper_target))
        self._backend.enqueue_joint_target(target)
        self._policy_tick += 1

    def clear_actions(self) -> None:
        if hasattr(self._backend, "clear_actions"):
            self._backend.clear_actions()

    def get_pending_action_count(self) -> int:
        if hasattr(self._backend, "get_pending_action_count"):
            return int(self._backend.get_pending_action_count())
        return 0

    def get_joint_positions(self) -> np.ndarray:
        if hasattr(self._backend, "get_joint_positions"):
            return np.asarray(self._backend.get_joint_positions(), dtype=np.float64).copy()
        return np.zeros(7, dtype=np.float64)

    def set_control_mode(self, control_mode: str) -> None:
        if self.is_control_running():
            raise RuntimeError("cannot change control_mode while control thread is running")
        control_mode = _validate_control_mode(control_mode)
        if self.action_planner is not None and control_mode != self.action_planner.control_mode:
            raise ValueError(
                f"planner {self.action_planner.mode!r} requires control_mode={self.action_planner.control_mode!r}"
            )
        self.control_mode = control_mode
        if hasattr(self._backend, "set_control_mode"):
            self._backend.set_control_mode(self.control_mode)

    def set_reference(self, reference_name: str) -> None:
        reference_name = _validate_reference_name(reference_name)
        if self.action_planner is not None:
            self.control_route = ControlRoute(self.action_planner, reference_name, self.tracker_mode)
        elif self.control_mode == "joint" and reference_name not in JOINT_REFERENCE_CHOICES:
            raise ValueError(f"joint reference must be one of {JOINT_REFERENCE_CHOICES}; got {reference_name!r}")
        self.reference_name = reference_name
        self._backend.set_reference(self.reference_name)

    def start_control_loop(
        self,
        *,
        max_duration: float | None = None,
        reference_name: str | None = None,
        print_events: bool | None = None,
        **_unused,
    ) -> None:
        if reference_name is not None:
            self.set_reference(reference_name)
        if print_events is not None:
            self.print_events = bool(print_events)
        self._begin_auto_recording()
        self._last_torque_print_trace_head = self.read_trace_head()
        self._backend.start_control_loop(-1.0 if max_duration is None else float(max_duration))

    def start_control(self, *, home_first: bool = False, reference_name: str | None = None, **kwargs) -> None:
        if home_first:
            self.reset()
        self.start_control_loop(reference_name=reference_name, **kwargs)

    def hold_pose(self) -> None:
        if not self.is_control_running():
            self.clear_actions()
            self.enqueue_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64))
            self.start_control_loop()

    def wait_control_loop(self) -> None:
        try:
            self._backend.wait()
        finally:
            self._finalize_auto_recording()

    def run_action_loop(self, *, max_duration: float | None = None, action_source=None, trace_callback=None, **kwargs) -> None:
        if action_source is not None or trace_callback is not None:
            raise ValueError("Python callbacks are not allowed in the C++ realtime control loop; enqueue actions at 10Hz")
        self.start_control_loop(max_duration=max_duration, **kwargs)
        self.wait_control_loop()

    def request_stop(self) -> None:
        if hasattr(self._backend, "request_stop"):
            self._backend.request_stop()
        else:
            self.stop_control()

    def stop_control(self) -> None:
        try:
            self._backend.stop()
        finally:
            self._finalize_auto_recording()

    def check_control_error(self) -> None:
        self._backend.wait()

    def stop(self) -> None:
        self.stop_control()
        if self._cameras is not None:
            self._cameras.stop()
        self._stop_gripper_driver()

    def read_trace_head(self) -> int:
        return int(self._backend.get_trace_head())

    def get_trace_since(self, after: int = 0) -> np.ndarray:
        return np.asarray(self._backend.get_trace_since(int(after)), dtype=np.float64)

    def get_trace_all(self) -> np.ndarray:
        return self.get_trace_since(0)

    def clear_trace(self) -> None:
        self._backend.clear_trace()

    def read_timing_head(self) -> int:
        if hasattr(self._backend, "get_timing_head"):
            return int(self._backend.get_timing_head())
        return 0

    def get_timing_since(self, after: int = 0) -> np.ndarray:
        if hasattr(self._backend, "get_timing_since"):
            return np.asarray(self._backend.get_timing_since(int(after)), dtype=np.float64)
        return np.zeros((0, 0), dtype=np.float64)

    def get_timing_all(self) -> np.ndarray:
        return self.get_timing_since(0)

    def get_timing_field_names(self) -> tuple[str, ...]:
        if hasattr(self._backend, "get_timing_field_names"):
            return tuple(str(name) for name in self._backend.get_timing_field_names())
        from recording.realtime_timing import TIMING_FIELDS

        return TIMING_FIELDS

    def get_timing_profiler(self):
        from recording import RealtimeTimingProfiler

        return RealtimeTimingProfiler.from_cpp_array(self.get_timing_all(), self.get_timing_field_names())

    def save_timing_csv(self, path: pathlib.Path | str) -> None:
        self.get_timing_profiler().write_csv(pathlib.Path(path))

    def print_realtime_timing_summary(self) -> None:
        self.get_timing_profiler().print_summary()

    def save_trace_to_recorder(self, recorder, reference_name: str = "") -> None:
        import time

        t0 = time.time()
        trace = self.get_trace_all()
        t1 = time.time()
        n = trace.shape[0]
        if n == 0:
            return
        print(
            f"  [trace] got {n} frames from C++ in {t1 - t0:.3f}s "
            f"(head={self.read_trace_head()}, capacity={self.trace_capacity_sec}s)"
        )
        reference_name_value = reference_name or self.reference_name
        rows = [
            {"time": float(t), "reference": reference_name_value,
             "goal_x": gx, "goal_y": gy, "goal_z": gz,
             "goal_rx": grx, "goal_ry": gry, "goal_rz": grz,
             "ref_x": rx, "ref_y": ry, "ref_z": rz,
             "ref_rx": rrx, "ref_ry": rry, "ref_rz": rrz,
             "actual_x": ax, "actual_y": ay, "actual_z": az,
             "actual_rx": arx, "actual_ry": ary, "actual_rz": arz}
            for t, gx, gy, gz, grx, gry, grz,
                rx, ry, rz, rrx, rry, rrz,
                ax, ay, az, arx, ary, arz
            in zip(trace[:, 0],
                   trace[:, 1], trace[:, 2], trace[:, 3],
                   trace[:, 4], trace[:, 5], trace[:, 6],
                   trace[:, 7], trace[:, 8], trace[:, 9],
                   trace[:, 10], trace[:, 11], trace[:, 12],
                   trace[:, 13], trace[:, 14], trace[:, 15],
                   trace[:, 16], trace[:, 17], trace[:, 18])
        ]
        t2 = time.time()
        recorder.rows.extend(rows)
        t3 = time.time()
        print(f"  [trace] {n} rows in {t2 - t1:.3f}s -> {n / max(t2 - t1, 1e-6):.0f} fps, extend {t3 - t2:.3f}s")

    def _begin_auto_recording(self) -> None:
        if not self.auto_record:
            self._active_run_paths = None
            self._record_finalized = True
            return
        if self.save_recording:
            from recording import create_run_paths

            self._active_run_paths = create_run_paths(self.log_root, self.reference_name)
        else:
            self._active_run_paths = None
        self._record_finalized = False

    def _finalize_auto_recording(self) -> None:
        if self._record_finalized:
            return
        self._record_finalized = True
        run_paths = self._active_run_paths
        self._active_run_paths = None
        if self.print_timing_summary:
            self.print_realtime_timing_summary()
        if run_paths is None:
            return

        import time

        from analysis import analyze_trace_csv, write_plot_svg, write_summary_json
        from recording import RealtimeTimingProfiler, TraceRecorder

        def timed_step(label, func):
            start = time.time()
            result = func()
            print(f"  [record] {label} in {time.time() - start:.3f}s")
            return result

        recorder = TraceRecorder(reference_name=self.reference_name)
        self.save_trace_to_recorder(recorder, reference_name=self.reference_name)
        timed_step("trace csv", lambda: recorder.write_trace_csv(run_paths.trace_csv))
        timing_rows = timed_step("timing csv", lambda: self.get_timing_all())
        timed_step("write timing csv", lambda: RealtimeTimingProfiler.from_cpp_array(timing_rows, self.get_timing_field_names()).write_csv(run_paths.timing_csv))
        timed_step(
            "metadata json",
            lambda: recorder.write_metadata_json(
                run_paths.metadata_json,
                {
                    "reference": self.reference_name,
                    "tracker_mode": self.tracker_mode,
                    "pid_proportional_gain": self.pid_proportional_gain,
                    "pid_integral_gain_s": self.pid_integral_gain_s,
                    "pid_velocity_gain_s": self.pid_velocity_gain_s,
                    "pid_maximum_correction_rad": self.pid_maximum_correction_rad,
                    "pid_integration_error_limit_rad": self.pid_integration_error_limit_rad,
                    "pid_integral_time_constant_s": self.pid_integral_time_constant_s,
                    "pid_stationary_integral_time_constant_s": self.pid_stationary_integral_time_constant_s,
                    "pid_stationary_velocity_threshold_rad_s": self.pid_stationary_velocity_threshold_rad_s,
                    "max_translation_velocity": self.max_translation_velocity,
                    "max_rotation_velocity": self.max_rotation_velocity,
                    "max_translation_step": self.max_translation_step,
                    "max_rotation_step": self.max_rotation_step,
                    "max_translation_goal_error": self.max_translation_goal_error,
                    "max_rotation_goal_error": self.max_rotation_goal_error,
                    "max_translation_goal_error": self.max_translation_goal_error,
                    "max_rotation_goal_error": self.max_rotation_goal_error,
                    "max_torque_rate": self.max_torque_rate,
                    "task_constraint_mask": self.task_constraint_mask.tolist(),
                    "reset_duration": self.reset_duration,
                    "reset_speed_factor": self.reset_speed_factor,
                    "sample_count": len(recorder.rows),
                    "timing_sample_count": int(timing_rows.shape[0]),
                    "trajectory_files": {
                        "pose_tracks_csv": run_paths.trace_csv.name,
                        "timing_csv": run_paths.timing_csv.name,
                        "summary_json": run_paths.summary_json.name,
                        "plot_svg": run_paths.plot_svg.name,
                    },
                },
            ),
        )
        summary = timed_step("analyze trace", lambda: analyze_trace_csv(run_paths.trace_csv))
        timed_step("summary json", lambda: write_summary_json(run_paths.summary_json, summary))
        timed_step("plot svg", lambda: write_plot_svg(run_paths.trace_csv, run_paths.plot_svg))
        print(f"Saved control recording: {run_paths.run_dir}")

    def is_control_running(self) -> bool:
        return bool(self._backend.is_running())

    def reset(self) -> None:
        self.clear_actions()
        try:
            self._backend.reset(self.reset_speed_factor, self.reset_duration)
        except TypeError:
            self._backend.reset(self.reset_speed_factor)
        self._set_gripper_target(GRIPPER_WIDTH_MAX)
        self._refresh_status_from_state()
        self.reset_action_planner()

    def get_robot_state_vector(self) -> np.ndarray:
        self._refresh_gripper_state()
        state = np.asarray(self._backend.get_robot_state_vector(), dtype=np.float64).copy()
        if state.shape[0] >= 8:
            half_width = float(np.clip(self._last_gripper_width, 0.0, GRIPPER_WIDTH_MAX)) / 2.0
            state[6] = half_width
            state[7] = -half_width
        return state

    def get_control_status(self) -> dict[str, object]:
        return {
            "control_running": self.is_control_running(),
            "reference_name": self.reference_name,
            "planner_mode": None if self.action_planner is None else self.action_planner.mode,
            "tracker_mode": self.tracker_mode,
            "gripper_enabled": self._gripper_enabled,
            "gripper_target": self._last_gripper_target,
            "pending_action_count": self.get_pending_action_count(),
        }

    def get_camera_frames(self) -> tuple[np.ndarray, np.ndarray]:
        if self._cameras is None:
            black = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            return black.copy(), black.copy()
        return self._cameras.get_frames()

    def get_observation(self, prompt: str = ""):
        state = self._refresh_status_from_state()
        img1, img2 = self.get_camera_frames()
        obs = {
            "prompt": prompt,
            "observation/image": _resize_observation_image(img1),
            "observation/wrist_image": _resize_observation_image(img2),
            "observation/state": state.astype(np.float64),
            "observation/joints": self.get_joint_positions().astype(np.float64),
        }
        return obs, img1, img2

from __future__ import annotations

import math
import multiprocessing as mp
import pathlib
import queue
import threading
import time

import numpy as np

from utils.control import ActionConfig, GRIPPER_WIDTH_MAX, transform_action

ROBOT_IP = "172.16.0.2"
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CAM1_SERIAL: str | None = None
DEFAULT_CAM2_SERIAL: str | None = None
REFERENCE_CHOICES = ("min_jerk", "linear", "cubic", "motion_limited")
NULLSPACE_PINV_CHOICES = ("plain", "damped")
NULLSPACE_PROJECTOR_CHOICES = ("kinematic", "dynamic")
CONTROL_MODE_CHOICES = ("cartesian", "joint")

DEFAULT_HOME_Q = np.array(
    [0.0, -math.pi / 4.0, 0.0, -3.0 * math.pi / 4.0, 0.0, math.pi / 2.0, math.pi / 4.0],
    dtype=np.float64,
)
DEFAULT_STIFFNESS = np.diag([600.0, 600.0, 600.0, 50.0, 50.0, 50.0]).astype(np.float64)
DEFAULT_DAMPING = np.diag([2.0 * math.sqrt(DEFAULT_STIFFNESS[i, i]) for i in range(6)]).astype(np.float64)
GRIPPER_SPEED = 0.08
GRIPPER_FORCE = 60.0
GRIPPER_CONNECT_TIMEOUT = 3.0


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


def _gripper_worker(robot_ip: str, command_queue, status_queue, speed: float, force: float) -> None:
    from control._franka_backend import RealtimeGripperBackend

    gripper = None

    def drain_latest(first_target):
        target = first_target
        while True:
            try:
                latest = command_queue.get_nowait()
            except queue.Empty:
                return target
            if latest is None:
                return None
            target = latest

    try:
        gripper = RealtimeGripperBackend(robot_ip)
        status_queue.put((True, ""))
        while True:
            target = drain_latest(command_queue.get())
            if target is None:
                break
            target = float(np.clip(float(target), 0.0, GRIPPER_WIDTH_MAX))
            try:
                ok = gripper.command(target, float(speed), float(force))
                if ok is False:
                    print(f"  [夹爪] command({target:.3f}) 返回失败", flush=True)
            except Exception as exc:
                print(f"  [夹爪] command({target:.3f}) 异常: {exc}", flush=True)
    except Exception as exc:
        status_queue.put((False, str(exc)))
    finally:
        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass


class _NoRobotBackend:
    def __init__(
        self,
        max_translation_step: float,
        max_rotation_step: float,
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
    ):
        del home_q, stiffness, damping, nullspace_q_target
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step = float(max_rotation_step)
        self.reference_name = _validate_reference_name(reference_name)
        self.control_mode = _validate_control_mode(control_mode)
        self.nullspace_enabled = bool(nullspace_enabled)
        self.nullspace_stiffness = float(nullspace_stiffness)
        self.nullspace_damping = float(nullspace_damping)
        self.nullspace_pinv = _validate_nullspace_pinv(nullspace_pinv)
        self.nullspace_projector = _validate_nullspace_projector(nullspace_projector)
        self.nullspace_lambda = float(nullspace_lambda)
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._state = np.array([0.0, 0.0, 0.0, math.pi, 0.0, 0.0, GRIPPER_WIDTH_MAX, GRIPPER_WIDTH_MAX], dtype=np.float64)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def enqueue_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        self._queue.put((action[0] if action.ndim == 2 else action).copy())

    def clear_actions(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def get_pending_action_count(self) -> int:
        return int(self._queue.qsize())

    def get_joint_positions(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float64)

    def start_control_loop(self, max_duration: float = -1.0) -> None:
        if self.is_running():
            raise RuntimeError("control thread is already running")
        self._stop.clear()

        def run() -> None:
            start = time.monotonic()
            while not self._stop.is_set():
                try:
                    action = self._queue.get_nowait()
                except queue.Empty:
                    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
                self._state[:3] += np.clip(action[:3], -1.0, 1.0) * self.max_translation_step
                self._state[3:6] += np.clip(action[3:6], -6.0, 6.0) * self.max_rotation_step / 6.0
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
        self._state = np.array([0.0, 0.0, 0.0, math.pi, 0.0, 0.0, GRIPPER_WIDTH_MAX, GRIPPER_WIDTH_MAX], dtype=np.float64)

    def get_robot_state_vector(self) -> np.ndarray:
        return self._state.copy()

    def get_trace_head(self) -> int:
        return 0

    def get_trace_since(self, after: int = 0) -> np.ndarray:
        del after
        return np.zeros((0, 26), dtype=np.float64)

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
    """Python 10Hz high-level interface; realtime torque control runs inside the C++ extension."""

    def __init__(
        self,
        robot_ip: str = ROBOT_IP,
        *,
        home_q: np.ndarray | None = None,
        reset_duration: float = 5.0,
        max_translation_step: float = 0.1,
        max_rotation_step: float = math.pi / 4.0,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
        reference_name: str = "min_jerk",
        control_mode: str = "cartesian",
        joint_min_jerk_duration: float = 0.25,
        nullspace_enabled: bool = False,
        nullspace_q_target: np.ndarray | None = None,
        nullspace_stiffness: float = 10.0,
        nullspace_damping: float = 2.0,
        nullspace_pinv: str = "plain",
        nullspace_projector: str = "kinematic",
        nullspace_lambda: float = 0.05,
        reset_speed_factor: float = 0.5,
        trace_capacity_sec: float = 180.0,
        use_gripper: bool = True,
        gripper_speed: float = GRIPPER_SPEED,
        gripper_force: float = GRIPPER_FORCE,
        no_robot: bool = False,
        no_cameras: bool = False,
        cam1_serial: str | None = DEFAULT_CAM1_SERIAL,
        cam2_serial: str | None = DEFAULT_CAM2_SERIAL,
        print_events: bool = True,
        auto_record: bool | None = None,
        save_recording: bool = False,
        log_root: pathlib.Path | str | None = None,
        log_subdir: str = "runs",
        print_timing_summary: bool = True,
        **_unused,
    ):
        del no_cameras
        self.robot_ip = robot_ip
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step = float(max_rotation_step)
        self.reset_duration = float(reset_duration)
        self.reset_speed_factor = float(reset_speed_factor)
        self.reference_name = _validate_reference_name(reference_name)
        self.control_mode = _validate_control_mode(control_mode)
        self.joint_min_jerk_duration = float(joint_min_jerk_duration)
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
        self.action_config = ActionConfig(self.max_translation_step, self.max_rotation_step)
        self.no_robot = bool(no_robot)
        self.robot = None if self.no_robot else object()
        self.cam1_serial = cam1_serial
        self.cam2_serial = cam2_serial
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
        self._gripper_enabled = False
        self._gripper_ctx = mp.get_context("spawn")
        self._gripper_command_queue = None
        self._gripper_status_queue = None
        self._gripper_process = None
        self.commanded_pose_array = np.zeros(6, dtype=np.float64)
        self.ee_force_torque = np.zeros(6, dtype=np.float64)

        if self.no_robot:
            self._backend = _NoRobotBackend(
                self.max_translation_step,
                self.max_rotation_step,
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
            )
        else:
            from control._franka_backend import RealtimeFrankaBackend

            self._backend = RealtimeFrankaBackend(
                self.robot_ip,
                self.max_translation_step,
                self.max_rotation_step,
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
            )
            if use_gripper:
                self._start_gripper_worker(float(gripper_speed), float(gripper_force))
        self._refresh_status_from_state()

    def _start_gripper_worker(self, speed: float, force: float) -> None:
        self._gripper_command_queue = self._gripper_ctx.Queue(maxsize=1)
        self._gripper_status_queue = self._gripper_ctx.Queue(maxsize=1)
        self._gripper_process = self._gripper_ctx.Process(
            target=_gripper_worker,
            args=(self.robot_ip, self._gripper_command_queue, self._gripper_status_queue, speed, force),
            daemon=True,
        )
        self._gripper_process.start()
        try:
            ok, message = self._gripper_status_queue.get(timeout=GRIPPER_CONNECT_TIMEOUT)
        except queue.Empty:
            ok, message = False, "startup timeout"
        self._gripper_enabled = bool(ok)
        if self._gripper_enabled:
            print("  [夹爪] worker 已启动")
        else:
            print(f"  [夹爪] worker 启动失败: {message}")
            self._stop_gripper_worker()

    def _stop_gripper_worker(self) -> None:
        if self._gripper_process is None:
            return
        if self._gripper_command_queue is not None:
            try:
                self._gripper_command_queue.put_nowait(None)
            except Exception:
                pass
        self._gripper_process.join(timeout=2.0)
        if self._gripper_process.is_alive():
            self._gripper_process.terminate()
            self._gripper_process.join(timeout=1.0)
        self._gripper_process = None
        self._gripper_enabled = False

    def _set_gripper_target(self, target: float) -> None:
        target = float(np.clip(target, 0.0, GRIPPER_WIDTH_MAX))
        self._last_gripper_target = target
        if not self._gripper_enabled or self._gripper_command_queue is None:
            return
        try:
            while True:
                self._gripper_command_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._gripper_command_queue.put_nowait(target)
        except queue.Full:
            pass

    def _format_torque_window(self) -> str:
        try:
            trace = self.get_trace_since(self._last_torque_print_trace_head)
            self._last_torque_print_trace_head = self.read_trace_head()
        except Exception:
            return ""
        if trace.shape[0] == 0 or trace.shape[1] < 26:
            return ""

        tau = trace[:, 19:26]
        tau_max = np.max(tau, axis=0)
        tau_text = ", ".join(f"{value:+.3f}" for value in tau_max)
        if tau.shape[0] < 2:
            return f" tau_max=[{tau_text}] max_d_tau=n/a"

        delta = np.abs(np.diff(tau, axis=0))
        flat_index = int(np.argmax(delta))
        frame_index, joint_index = np.unravel_index(flat_index, delta.shape)
        del frame_index
        return f" tau_max=[{tau_text}] max_d_tau={delta.max():.3f}@J{joint_index + 1}"

    def _print_action_event(self, action: np.ndarray) -> None:
        if not self.print_events:
            return
        if self.control_mode == "joint":
            dq_deg = np.degrees(action[:7])
            dq_text = ",".join(f"{value:+.1f}" for value in dq_deg)
            gripper = "close" if action[7] > 0.0 else "open"
            print(f"10Hz joint tick {self._policy_tick + 1:04d} dq_deg=[{dq_text}] gripper={gripper}", flush=True)
            return
        torque_window = self._format_torque_window()
        dx, dy, dz = action[0], action[1], action[2]
        drx, dry, drz = action[3], action[4], action[5]
        print(
            f"10Hz tick {self._policy_tick + 1:04d} "
            f"dxyz=[{dx:+.3f},{dy:+.3f},{dz:+.3f}]  "
            f"drot=[{drx:+.3f},{dry:+.3f},{drz:+.3f}]"
            f"{torque_window}",
            flush=True,
        )

    def _refresh_status_from_state(self) -> np.ndarray:
        state = self.get_robot_state_vector()
        self.commanded_pose_array = state[:6].copy()
        self.ee_force_torque = np.zeros(6, dtype=np.float64)
        return state

    def enqueue_action_block(self, action_block: np.ndarray) -> None:
        block = np.asarray(action_block, dtype=np.float64)
        first = block[0] if block.ndim == 2 else block
        expected_dim = 8 if self.control_mode == "joint" else 7
        if first.shape != (expected_dim,):
            raise ValueError(
                f"{self.control_mode} action must have shape ({expected_dim},) or (N, {expected_dim}); got {block.shape}"
            )
        self._print_action_event(first)
        if self.control_mode == "cartesian":
            transformed = transform_action(first, self.action_config)
            self._set_gripper_target(transformed[6])
        else:
            self._set_gripper_target(GRIPPER_WIDTH_MAX if first[7] <= 0.0 else 0.0)
        self._backend.enqueue_action(block)
        self._policy_tick += 1

    def enqueue_action(self, action: np.ndarray) -> None:
        self.enqueue_action_block(action)

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
        self.control_mode = _validate_control_mode(control_mode)
        if hasattr(self._backend, "set_control_mode"):
            self._backend.set_control_mode(self.control_mode)

    def set_reference(self, reference_name: str) -> None:
        self.reference_name = _validate_reference_name(reference_name)
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
        self.stop_control()

    def stop_control(self) -> None:
        try:
            self._backend.stop()
        finally:
            self._finalize_auto_recording()

    def stop(self) -> None:
        self.stop_control()
        self._stop_gripper_worker()

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
                    "max_translation_step": self.max_translation_step,
                    "max_rotation_step": self.max_rotation_step,
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

    def get_robot_state_vector(self) -> np.ndarray:
        state = np.asarray(self._backend.get_robot_state_vector(), dtype=np.float64).copy()
        if state.shape[0] >= 8:
            state[6] = self._last_gripper_target
            state[7] = self._last_gripper_target
        return state

    def get_control_status(self) -> dict[str, object]:
        return {
            "control_running": self.is_control_running(),
            "reference_name": self.reference_name,
            "gripper_enabled": self._gripper_enabled,
            "gripper_target": self._last_gripper_target,
            "pending_action_count": self.get_pending_action_count(),
        }

    def get_observation(self, prompt: str = ""):
        state = self._refresh_status_from_state()
        obs = {
            "prompt": prompt,
            "observation/state": state,
            "observation/joints": np.zeros(7, dtype=np.float64),
        }
        img1 = np.zeros((480, 640, 3), dtype=np.uint8)
        img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        return obs, img1, img2

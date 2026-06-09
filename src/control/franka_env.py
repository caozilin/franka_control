from __future__ import annotations

import math
import queue
import threading
import time

import numpy as np

ROBOT_IP = "172.16.0.2"
GRIPPER_WIDTH_MAX = 0.08
CONTROLLER_CHOICES = ("min_jerk", "linear", "cubic")


def _validate_controller_name(controller_name: str) -> str:
    controller_name = str(controller_name)
    if controller_name not in CONTROLLER_CHOICES:
        raise ValueError(f"controller_name must be one of {CONTROLLER_CHOICES}; got {controller_name!r}")
    return controller_name


class _NoRobotBackend:
    def __init__(self, max_translation_step: float, max_rotation_step: float, controller_name: str = "min_jerk"):
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step = float(max_rotation_step)
        self.controller_name = _validate_controller_name(controller_name)
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._state = np.array([0.0, 0.0, 0.0, math.pi, 0.0, 0.0, GRIPPER_WIDTH_MAX, GRIPPER_WIDTH_MAX], dtype=np.float64)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def enqueue_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        self._queue.put((action[0] if action.ndim == 2 else action).copy())

    def start_control_loop(self, max_duration: float = -1.0) -> None:
        self._stop.clear()

        def run() -> None:
            start = time.monotonic()
            while not self._stop.is_set():
                try:
                    action = self._queue.get_nowait()
                except queue.Empty:
                    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64)
                self._state[:3] += np.clip(action[:3], -1.0, 1.0) * self.max_translation_step
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

    def set_controller(self, controller_name: str) -> None:
        if self.is_running():
            raise RuntimeError("cannot change controller while control thread is running")
        self.controller_name = _validate_controller_name(controller_name)

    def reset(self, speed_factor: float = 0.5) -> None:
        del speed_factor
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
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


class FrankaEnv:
    """Python 10Hz high-level interface; realtime torque control runs inside the C++ extension."""

    def __init__(
        self,
        robot_ip: str = ROBOT_IP,
        *,
        max_translation_step: float = 0.01,
        max_rotation_step: float = math.pi / 4.0,
        controller_name: str = "min_jerk",
        reset_speed_factor: float = 0.5,
        trace_capacity_sec: float = 180.0,
        no_robot: bool = False,
        no_cameras: bool = False,
        **_unused,
    ):
        del no_cameras
        self.robot_ip = robot_ip
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step = float(max_rotation_step)
        self.reset_speed_factor = float(reset_speed_factor)
        self.controller_name = _validate_controller_name(controller_name)
        self.trace_capacity_sec = float(trace_capacity_sec)
        if no_robot:
            self._backend = _NoRobotBackend(self.max_translation_step, self.max_rotation_step, self.controller_name)
        else:
            from control._franka_backend import RealtimeFrankaBackend

            self._backend = RealtimeFrankaBackend(
                self.robot_ip,
                self.max_translation_step,
                self.max_rotation_step,
                self.controller_name,
                self.trace_capacity_sec,
            )

    def enqueue_action_block(self, action_block: np.ndarray) -> None:
        self._backend.enqueue_action(np.asarray(action_block, dtype=np.float64))

    def set_controller(self, controller_name: str) -> None:
        self.controller_name = _validate_controller_name(controller_name)
        self._backend.set_controller(self.controller_name)

    def start_control_loop(self, *, max_duration: float | None = None, controller_name: str | None = None, **_unused) -> None:
        if controller_name is not None:
            self.set_controller(controller_name)
        self._backend.start_control_loop(-1.0 if max_duration is None else float(max_duration))

    def wait_control_loop(self) -> None:
        self._backend.wait()

    def run_action_loop(self, *, max_duration: float | None = None, action_source=None, trace_callback=None, **kwargs) -> None:
        if action_source is not None or trace_callback is not None:
            raise ValueError("Python callbacks are not allowed in the C++ realtime control loop; enqueue actions at 10Hz")
        self.start_control_loop(max_duration=max_duration, **kwargs)
        self.wait_control_loop()

    def request_stop(self) -> None:
        self._backend.stop()

    def stop_control(self) -> None:
        self._backend.stop()

    def stop(self) -> None:
        self._backend.stop()

    # ---- trace / timing ---------------------------------------------------

    def read_trace_head(self) -> int:
        """Return current write index (monotonic, no copy)."""
        return int(self._backend.get_trace_head())

    def get_trace_since(self, after: int = 0) -> np.ndarray:
        """Return trace frames written after *after* (numpy ndarray, shape (N, 56))."""
        return np.asarray(self._backend.get_trace_since(int(after)), dtype=np.float64)

    def get_trace_all(self) -> np.ndarray:
        """Return all trace frames in ring buffer."""
        return self.get_trace_since(0)

    def clear_trace(self) -> None:
        """Reset trace ring buffer."""
        self._backend.clear_trace()

    def save_trace_to_recorder(self, recorder, controller_name: str = "") -> None:
        """Write C++ trace ring to TraceRecorder rows (xyz + rotvec already computed in C++).

        C++ trace layout (26 floats per frame):
          [0]    time
          [1:4]  goal_xyz
          [4:7]  goal_rotvec
          [7:10] ref_xyz
          [10:13] ref_rotvec
          [13:16] actual_xyz
          [16:19] actual_rotvec
          [19:26] tau_cmd
        """
        import time

        t0 = time.time()
        trace = self.get_trace_all()
        t1 = time.time()
        n = trace.shape[0]
        if n == 0:
            return
        print(f"  [trace] got {n} frames from C++ in {t1 - t0:.3f}s "
              f"(head={self.read_trace_head()}, capacity={self.trace_capacity_sec}s)")

        ctrl_name = controller_name or self.controller_name
        # Batch-convert to dict list via numpy column extraction (much faster than per-row loop)
        rows = [
            {"time": float(t), "controller": ctrl_name,
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
        print(f"  [trace] {n} rows in {t2 - t1:.3f}s → {n / max(t2 - t1, 1e-6):.0f} fps, "
              f"extend {t3 - t2:.3f}s")

    # -----------------------------------------------------------------------

    def is_control_running(self) -> bool:
        return bool(self._backend.is_running())

    def reset(self) -> None:
        self._backend.reset(self.reset_speed_factor)

    def get_robot_state_vector(self) -> np.ndarray:
        return np.asarray(self._backend.get_robot_state_vector(), dtype=np.float64).copy()

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any


BackendFactory = Callable[[str], Any]


def _default_backend_factory(robot_ip: str):
    from control._franka_backend import RealtimeGripperBackend

    return RealtimeGripperBackend(robot_ip)


class AsyncGripperDriver:
    """Single-connection gripper driver with asynchronous command execution."""

    def __init__(
        self,
        robot_ip: str,
        *,
        speed: float,
        force: float,
        width_max: float,
        poll_period: float = 0.1,
        connect_timeout: float = 8.0,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if poll_period <= 0.0:
            raise ValueError("poll_period must be positive")
        self._robot_ip = str(robot_ip)
        self._speed = float(speed)
        self._force = float(force)
        self._width_max = float(width_max)
        self._poll_period = float(poll_period)
        self._connect_timeout = float(connect_timeout)
        self._backend_factory = backend_factory or _default_backend_factory

        self._state_lock = threading.Lock()
        self._desired_target = self._width_max
        self._desired_revision = 0
        self._width = self._width_max
        self._enabled = False
        self._last_error = ""

        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._startup_queue: queue.Queue[tuple[bool, str]] = queue.Queue(maxsize=1)
        self._worker: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        with self._state_lock:
            return self._enabled

    @property
    def width(self) -> float:
        with self._state_lock:
            return self._width

    @property
    def last_error(self) -> str:
        with self._state_lock:
            return self._last_error

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._wake_event.clear()
        self._startup_queue = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._run, name="franka-gripper", daemon=True)
        self._worker.start()
        try:
            ok, message = self._startup_queue.get(timeout=self._connect_timeout)
        except queue.Empty as exc:
            self.stop()
            raise RuntimeError("gripper startup timed out") from exc
        if not ok:
            self.stop()
            raise RuntimeError(message)

    def set_target(self, target: float) -> None:
        clipped = min(max(float(target), 0.0), self._width_max)
        with self._state_lock:
            if clipped == self._desired_target:
                return
            self._desired_target = clipped
            self._desired_revision += 1
        self._wake_event.set()

    def stop(self, timeout: float = 2.0) -> bool:
        worker = self._worker
        self._worker = None
        self._stop_event.set()
        self._wake_event.set()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=max(float(timeout), 0.0))
        with self._state_lock:
            self._enabled = False
        return worker is None or not worker.is_alive()

    def _publish_width(self, width: float) -> None:
        with self._state_lock:
            self._width = min(max(float(width), 0.0), self._width_max)

    def _publish_error(self, message: str) -> None:
        with self._state_lock:
            self._last_error = str(message)

    def _desired(self) -> tuple[float, int]:
        with self._state_lock:
            return self._desired_target, self._desired_revision

    def _read_state(self, backend) -> None:
        try:
            state = backend.read_once()
            self._publish_width(float(state.get("width", self._width_max)))
        except Exception as exc:
            self._publish_error(f"gripper state read failed: {exc}")

    def _run(self) -> None:
        backend = None
        command_thread: threading.Thread | None = None
        command_results: queue.Queue[tuple[float, bool, str]] = queue.Queue()
        dispatched_revision = 0
        last_dispatched_target = self._width_max

        def run_command(target: float) -> None:
            try:
                ok = bool(backend.command(target, self._speed, self._force))
                message = "" if ok else f"gripper command({target:.3f}) returned false"
                command_results.put((target, ok, message))
            except Exception as exc:
                command_results.put((target, False, f"gripper command({target:.3f}) failed: {exc}"))
            finally:
                self._wake_event.set()

        try:
            backend = self._backend_factory(self._robot_ip)
            initial_state = backend.read_once()
            self._publish_width(float(initial_state.get("width", self._width_max)))
            with self._state_lock:
                self._enabled = True
                self._last_error = ""
            self._startup_queue.put_nowait((True, ""))

            next_poll = time.monotonic()
            while not self._stop_event.is_set():
                if command_thread is not None and not command_thread.is_alive():
                    command_thread.join()
                    command_thread = None
                    while True:
                        try:
                            _, ok, message = command_results.get_nowait()
                        except queue.Empty:
                            break
                        if not ok:
                            self._publish_error(message)

                desired_target, desired_revision = self._desired()
                if command_thread is None and desired_revision != dispatched_revision:
                    dispatched_revision = desired_revision
                    if desired_target != last_dispatched_target:
                        last_dispatched_target = desired_target
                        command_thread = threading.Thread(
                            target=run_command,
                            args=(desired_target,),
                            name="franka-gripper-command",
                            daemon=True,
                        )
                        command_thread.start()

                now = time.monotonic()
                if now >= next_poll:
                    self._read_state(backend)
                    next_poll = now + self._poll_period

                wait_time = max(0.0, min(next_poll - time.monotonic(), self._poll_period))
                self._wake_event.wait(wait_time)
                self._wake_event.clear()
        except Exception as exc:
            self._publish_error(str(exc))
            try:
                self._startup_queue.put_nowait((False, str(exc)))
            except queue.Full:
                pass
        finally:
            if backend is not None:
                if command_thread is not None and command_thread.is_alive():
                    try:
                        backend.stop()
                    except Exception:
                        pass
                    command_thread.join(timeout=1.0)
                try:
                    backend.stop()
                except Exception:
                    pass
            with self._state_lock:
                self._enabled = False

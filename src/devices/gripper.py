from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any


def _default_backend_factory(robot_ip: str):
    from control._franka_backend import RealtimeGripperBackend

    return RealtimeGripperBackend(robot_ip, 0.003, 1e-6, 0.08, 0.08)


class AsyncGripperDriver:
    """One gripper connection with 10 Hz state polling and async commands."""

    def __init__(
        self,
        robot_ip: str,
        *,
        speed: float,
        force: float,
        width_max: float,
        poll_period: float = 0.1,
        connect_timeout: float = 8.0,
        backend_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.robot_ip = robot_ip
        self.speed = float(speed)
        self.force = float(force)
        self.width_max = float(width_max)
        self.poll_period = float(poll_period)
        self.connect_timeout = float(connect_timeout)
        self.backend_factory = backend_factory or _default_backend_factory

        self._lock = threading.Lock()
        self._target = self.width_max
        self._width = self.width_max
        self._enabled = False
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._startup: queue.Queue[tuple[bool, str]] = queue.Queue(maxsize=1)
        self._worker: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def width(self) -> float:
        with self._lock:
            return self._width

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._wake.clear()
        self._startup = queue.Queue(maxsize=1)
        self._worker = threading.Thread(target=self._run, name="franka-gripper", daemon=True)
        self._worker.start()
        try:
            ok, message = self._startup.get(timeout=self.connect_timeout)
        except queue.Empty as exc:
            self.stop()
            raise RuntimeError("gripper startup timed out") from exc
        if not ok:
            self.stop()
            raise RuntimeError(message)

    def set_target(self, target: float) -> None:
        target = min(max(float(target), 0.0), self.width_max)
        with self._lock:
            self._target = target
        self._wake.set()

    def stop(self, timeout: float = 2.0) -> bool:
        worker = self._worker
        self._worker = None
        self._stop.set()
        self._wake.set()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout)
        with self._lock:
            self._enabled = False
        return worker is None or not worker.is_alive()

    def _set_width(self, width: float) -> None:
        with self._lock:
            self._width = min(max(float(width), 0.0), self.width_max)

    def _get_target(self) -> float:
        with self._lock:
            return self._target

    def _run(self) -> None:
        backend = None
        command_thread: threading.Thread | None = None
        last_target = self.width_max

        def command(target: float) -> None:
            try:
                if backend.command(target, self.speed, self.force) is False:
                    print(f"  [夹爪] command({target:.3f}) 返回失败", flush=True)
            except Exception as exc:
                print(f"  [夹爪] command({target:.3f}) 异常: {exc}", flush=True)
            finally:
                self._wake.set()

        try:
            backend = self.backend_factory(self.robot_ip)
            state = backend.read_once()
            self._set_width(state.get("width", self.width_max))
            with self._lock:
                self._enabled = True
            self._startup.put((True, ""))

            while not self._stop.is_set():
                if command_thread is not None and not command_thread.is_alive():
                    command_thread.join()
                    command_thread = None

                target = self._get_target()
                if command_thread is None and target != last_target:
                    last_target = target
                    command_thread = threading.Thread(
                        target=command,
                        args=(target,),
                        name="franka-gripper-command",
                        daemon=True,
                    )
                    command_thread.start()

                try:
                    state = backend.read_once()
                    self._set_width(state.get("width", self.width_max))
                except Exception:
                    pass
                self._wake.wait(self.poll_period)
                self._wake.clear()
        except Exception as exc:
            self._startup.put((False, str(exc)))
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
            with self._lock:
                self._enabled = False

from __future__ import annotations

import threading
import time

import pytest

from devices import AsyncGripperDriver


def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


class _BlockingGripperBackend:
    def __init__(self, _robot_ip: str) -> None:
        self.command_started = threading.Event()
        self.release_command = threading.Event()
        self._lock = threading.Lock()
        self.read_count = 0
        self.commands: list[float] = []

    def read_once(self) -> dict[str, float]:
        with self._lock:
            self.read_count += 1
        return {"width": 0.08}

    def command(self, target: float, _speed: float, _force: float) -> bool:
        with self._lock:
            self.commands.append(target)
        self.command_started.set()
        assert self.release_command.wait(timeout=1.0)
        return True

    def stop(self) -> bool:
        self.release_command.set()
        return True


def test_gripper_keeps_polling_and_coalesces_targets_during_command() -> None:
    backend = _BlockingGripperBackend("unused")
    driver = AsyncGripperDriver(
        "robot",
        speed=0.08,
        force=60.0,
        width_max=0.08,
        poll_period=0.01,
        connect_timeout=0.5,
        backend_factory=lambda _robot_ip: backend,
    )
    driver.start()
    try:
        reads_before = backend.read_count
        driver.set_target(0.0)
        assert backend.command_started.wait(timeout=0.5)

        _wait_until(lambda: backend.read_count >= reads_before + 3)
        driver.set_target(0.02)
        driver.set_target(0.04)
        driver.set_target(0.08)
        backend.release_command.set()

        _wait_until(lambda: len(backend.commands) >= 2)
        assert backend.commands == pytest.approx([0.0, 0.08])
        assert driver.width == pytest.approx(0.08)
    finally:
        assert driver.stop()

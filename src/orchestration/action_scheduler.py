"""Thread-safe scheduling for policy action chunks and RTC model tails."""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass

import numpy as np

from control.contracts import POLICY_ACTION_DIM


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = False
    execution_horizon: int = 5
    inference_delay: int = 3

    def __post_init__(self) -> None:
        if int(self.execution_horizon) < 1:
            raise ValueError("RTC execution_horizon must be at least one")
        if int(self.inference_delay) < 0:
            raise ValueError("RTC inference_delay cannot be negative")


@dataclass(frozen=True)
class ActionScheduleSnapshot:
    action_count: int
    rtc_model_count: int
    rtc_model_shape: tuple[int, int]
    rtc_activation_count: int


def _action_rows(values, *, columns: int | None, name: str) -> list[np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (H, A); got {array.shape}")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (H, {columns}); got {array.shape}")
    if array.shape[1] < 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite action rows")
    return [row.copy() for row in array]


class ActionPlanScheduler:
    """Own policy execution queues without knowing the policy or robot.

    One lock protects both the executable plan and its RTC model tail so UI
    snapshots and inference requests cannot observe mismatched queue states.
    """

    def __init__(self, replan_steps: int, rtc: RTCConfig = RTCConfig()) -> None:
        self.replan_steps = max(1, int(replan_steps))
        self.rtc = rtc
        self._actions: collections.deque[np.ndarray] = collections.deque()
        self._rtc_model_actions: collections.deque[np.ndarray] = collections.deque()
        self._rtc_activation_count = 0
        self._lock = threading.Lock()

    @property
    def prefetch_threshold(self) -> int:
        if self.rtc.enabled:
            return min(self.replan_steps, max(1, int(self.rtc.inference_delay)))
        return max(1, self.replan_steps // 2)

    def clear(self) -> None:
        with self._lock:
            self._actions.clear()
            self._rtc_model_actions.clear()
            self._rtc_activation_count = 0

    def append_inference_result(self, actions, rtc_model_actions=None) -> int:
        action_rows = _action_rows(actions, columns=POLICY_ACTION_DIM, name="actions")
        action_rows = action_rows[: self.replan_steps]
        rtc_rows = None
        if self.rtc.enabled and rtc_model_actions is not None:
            rtc_rows = _action_rows(rtc_model_actions, columns=None, name="rtc.model_actions")

        with self._lock:
            queued_before_append = len(self._actions)
            self._actions.extend(action_rows)
            if rtc_rows is not None:
                self._rtc_model_actions = collections.deque(rtc_rows)
                self._rtc_activation_count = queued_before_append
            return len(action_rows)

    def pop_next(self) -> np.ndarray | None:
        with self._lock:
            if not self._actions:
                return None
            return self._actions.popleft().copy()

    def consume_executed_step(self) -> None:
        if not self.rtc.enabled:
            return
        with self._lock:
            if self._rtc_activation_count > 0:
                self._rtc_activation_count -= 1
            elif self._rtc_model_actions:
                self._rtc_model_actions.popleft()

    def should_prefetch(self) -> bool:
        with self._lock:
            return len(self._actions) <= self.prefetch_threshold

    def rtc_request(self) -> dict | None:
        if not self.rtc.enabled:
            return None
        with self._lock:
            if not self._rtc_model_actions:
                return None
            tail = np.stack([row.copy() for row in self._rtc_model_actions], axis=0)
        return {
            "enabled": True,
            "prev_chunk_left_over": tail.astype(np.float32, copy=False),
            "inference_delay": int(self.rtc.inference_delay),
            "execution_horizon": int(self.rtc.execution_horizon),
        }

    def snapshot(self) -> ActionScheduleSnapshot:
        with self._lock:
            count = len(self._rtc_model_actions)
            width = len(self._rtc_model_actions[0]) if self._rtc_model_actions else 0
            return ActionScheduleSnapshot(
                action_count=len(self._actions),
                rtc_model_count=count,
                rtc_model_shape=(count, width),
                rtc_activation_count=self._rtc_activation_count,
            )

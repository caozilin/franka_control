from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import ActionPlanScheduler, RTCConfig  # noqa: E402


def _actions(start: float, count: int) -> np.ndarray:
    rows = np.zeros((count, 7), dtype=np.float64)
    rows[:, 0] = np.arange(start, start + count)
    return rows


def test_scheduler_limits_chunks_and_preserves_fifo_order() -> None:
    scheduler = ActionPlanScheduler(replan_steps=3)
    assert scheduler.prefetch_threshold == 1
    assert scheduler.should_prefetch()

    assert scheduler.append_inference_result(_actions(1.0, 5)) == 3
    assert scheduler.snapshot().action_count == 3
    assert not scheduler.should_prefetch()
    assert scheduler.pop_next()[0] == 1.0
    assert scheduler.pop_next()[0] == 2.0
    assert scheduler.should_prefetch()
    assert scheduler.pop_next()[0] == 3.0
    assert scheduler.pop_next() is None


def test_rtc_tail_activation_tracks_preexisting_actions() -> None:
    scheduler = ActionPlanScheduler(
        replan_steps=4,
        rtc=RTCConfig(enabled=True, execution_horizon=5, inference_delay=2),
    )
    scheduler.append_inference_result(_actions(10.0, 2))
    scheduler.append_inference_result(
        _actions(20.0, 2),
        np.asarray([[100.0, 101.0], [200.0, 201.0]]),
    )

    snapshot = scheduler.snapshot()
    assert snapshot.action_count == 4
    assert snapshot.rtc_model_shape == (2, 2)
    assert snapshot.rtc_activation_count == 2

    scheduler.pop_next()
    scheduler.consume_executed_step()
    scheduler.pop_next()
    scheduler.consume_executed_step()
    assert scheduler.snapshot().rtc_activation_count == 0
    assert scheduler.snapshot().rtc_model_count == 2

    scheduler.pop_next()
    scheduler.consume_executed_step()
    assert scheduler.snapshot().rtc_model_count == 1
    request = scheduler.rtc_request()
    assert request is not None
    assert request["prev_chunk_left_over"].dtype == np.float32
    np.testing.assert_allclose(request["prev_chunk_left_over"], [[200.0, 201.0]])


def test_scheduler_clear_resets_action_and_rtc_state() -> None:
    scheduler = ActionPlanScheduler(2, RTCConfig(enabled=True))
    scheduler.append_inference_result(_actions(1.0, 1), [[2.0]])
    scheduler.clear()
    assert scheduler.snapshot().action_count == 0
    assert scheduler.snapshot().rtc_model_count == 0
    assert scheduler.rtc_request() is None


@pytest.mark.parametrize(
    "actions",
    [np.zeros((2, 6)), np.full((2, 7), np.nan), np.zeros((0, 7))],
)
def test_scheduler_rejects_invalid_action_chunks(actions) -> None:
    with pytest.raises(ValueError):
        ActionPlanScheduler(2).append_inference_result(actions)

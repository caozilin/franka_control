from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planning import (  # noqa: E402
    BaselineSQPPlanner,
    AxisTask,
    TaskKind,
    TargetPose,
    ShadowOrientationReference,
    ShadowSQPPlanner,
    SQPSettings,
    project_historical_tolerance_bias,
)
from control.franka_env import DEFAULT_HOME_Q  # noqa: E402
from utils.pose import matrix_to_rotvec, rotvec_to_matrix  # noqa: E402
from planning.sqp import SQPPlan  # noqa: E402
from planning.sqp.solver import ConstraintValues, SQPSolverResult  # noqa: E402


def _bias_coordinates(rotation: np.ndarray, shadow: np.ndarray, frame: np.ndarray) -> np.ndarray:
    return matrix_to_rotvec(frame.T @ (rotation @ shadow.T) @ frame)


def _released(reference: np.ndarray, frame: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    return frame @ rotvec_to_matrix(coordinates) @ frame.T @ reference


def test_shadow_projection_supports_all_rotation_masks() -> None:
    frame = rotvec_to_matrix(np.array([0.31, -0.24, 0.17]))
    shadow = rotvec_to_matrix(np.array([-0.22, 0.09, 0.14]))
    raw_coordinates = np.array([0.43, -0.36, 0.28])
    raw = _released(shadow, frame, raw_coordinates)

    for values in itertools.product((False, True), repeat=3):
        mask = np.asarray(values, dtype=bool)
        corrected, diagnostics = project_historical_tolerance_bias(raw, shadow, frame, mask)
        expected = mask.astype(float) * raw_coordinates
        np.testing.assert_allclose(_bias_coordinates(corrected, shadow, frame), expected, atol=1e-10)
        np.testing.assert_allclose(diagnostics.kept_bias_tolerance_frame, expected, atol=1e-10)
        assert diagnostics.non_tolerance_residual_rad <= 1e-10


def test_shadow_never_leaks_history_to_strict_axes() -> None:
    frame = rotvec_to_matrix(np.array([0.27, -0.18, 0.33]))
    mask = np.array([True, False, True])
    executed = rotvec_to_matrix(np.array([0.13, -0.21, 0.08]))
    shadow = ShadowOrientationReference(executed)

    for action in (
        np.array([0.19, -0.11, 0.07]),
        np.array([-0.08, 0.16, 0.13]),
        np.array([0.12, 0.05, -0.17]),
    ):
        corrected, diagnostics = shadow.advance(
            executed,
            action,
            frame,
            mask,
            semantic_key=("adjust_bottle", "pregrasp"),
        )
        coordinates = _bias_coordinates(corrected, shadow.shadow_rotation, frame)
        np.testing.assert_allclose(coordinates[~mask], 0.0, atol=1e-10)
        assert diagnostics.non_tolerance_residual_rad <= 1e-10
        executed = _released(corrected, frame, np.array([0.41, 0.0, -0.47]))


def test_shadow_reanchors_when_stage_mask_changes() -> None:
    first_frame = rotvec_to_matrix(np.array([0.21, -0.12, 0.06]))
    second_frame = rotvec_to_matrix(np.array([-0.09, 0.18, 0.15]))
    executed = rotvec_to_matrix(np.array([0.16, -0.23, 0.11]))
    shadow = ShadowOrientationReference(executed)
    _, first = shadow.advance(
        executed,
        np.array([0.08, -0.03, 0.04]),
        first_frame,
        np.array([True, False, False]),
        semantic_key=(0, 0),
    )
    assert first.anchor_reset

    corrected, changed = shadow.advance(
        executed,
        np.zeros(3),
        second_frame,
        np.array([False, False, True]),
        semantic_key=(2, 0),
    )
    assert changed.anchor_reset
    np.testing.assert_allclose(corrected, executed, atol=1e-10)


def test_shadow_planner_uses_the_same_baseline_sqp() -> None:
    baseline = BaselineSQPPlanner(
        solver_settings=SQPSettings(max_iterations=30, max_time_s=2.0),
    )
    planner = ShadowSQPPlanner(baseline)
    plan = planner.step(
        DEFAULT_HOME_Q,
        np.array([0.001, 0.0, 0.0, 0.02, 0.0, 0.0]),
        tolerance_frame=np.eye(3),
        ranged_axes=np.array([True, False, False]),
        semantic_key=("task", "pregrasp"),
    )

    assert planner.baseline is baseline
    assert plan.baseline.solver.feasible, plan.baseline.solver.status
    assert plan.shadow.anchor_reset


def test_shadow_uses_projected_reference_then_builds_tasks_from_corrected_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = BaselineSQPPlanner()
    baseline.reset(DEFAULT_HOME_Q)
    assert baseline.target is not None
    projected = rotvec_to_matrix(np.array([0.22, -0.08, 0.13]))
    corrected = rotvec_to_matrix(np.array([-0.11, 0.19, 0.07]))
    baseline._target = TargetPose(baseline.target.position.copy(), projected)
    planner = ShadowSQPPlanner(baseline)
    seen: dict[str, np.ndarray | tuple[AxisTask, ...]] = {}

    def fake_advance(
        previous: np.ndarray,
        action: np.ndarray,
        frame: np.ndarray,
        mask: np.ndarray,
        *,
        semantic_key: object,
    ):
        del action, frame, mask, semantic_key
        seen["previous"] = previous.copy()
        from planning import ShadowOrientationDiagnostics

        zeros = np.zeros(3)
        return corrected.copy(), ShadowOrientationDiagnostics(
            zeros, zeros, zeros, 0.0, False,
        )

    def task_factory(rotation: np.ndarray) -> tuple[AxisTask, ...]:
        seen["task_rotation"] = rotation.copy()
        return tuple(AxisTask(TaskKind.SPECIFIC) for _ in range(6))

    def fake_solve_target(
        measured_q: np.ndarray,
        target: TargetPose,
        **kwargs: object,
    ) -> SQPPlan:
        seen["solve_rotation"] = target.rotation.copy()
        seen["tasks"] = kwargs["axis_tasks"]  # type: ignore[assignment]
        constraints = ConstraintValues(np.zeros(6), np.empty(0), 0.0, 0.0, 0.0)
        result = SQPSolverResult(
            measured_q.copy(), 0.0, 0, 0, 0.0, 0.0, True, True,
            "converged", constraints, {},
        )
        return SQPPlan(measured_q.copy(), target, target, result)

    monkeypatch.setattr(planner.shadow, "advance", fake_advance)
    monkeypatch.setattr(baseline, "solve_target", fake_solve_target)
    planner.step(
        DEFAULT_HOME_Q,
        np.zeros(6),
        tolerance_frame=np.eye(3),
        ranged_axes=np.array([True, False, False]),
        semantic_key=("task", "pregrasp"),
        axis_task_factory=task_factory,
    )

    np.testing.assert_allclose(seen["previous"], projected)
    np.testing.assert_allclose(seen["task_rotation"], corrected)
    np.testing.assert_allclose(seen["solve_rotation"], corrected)

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
from utils.pose import (  # noqa: E402
    rotation_from_tolerance_coordinates,
    rotation_tolerance_coordinates,
    rotvec_to_matrix,
)
from planning.sqp import SQPPlan  # noqa: E402
from planning.sqp.solver import ConstraintValues, SQPSolverResult  # noqa: E402


def _bias_coordinates(
    rotation: np.ndarray,
    shadow: np.ndarray,
    stage_target: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    return (
        rotation_tolerance_coordinates(rotation, stage_target, frame)
        - rotation_tolerance_coordinates(shadow, stage_target, frame)
    )


def _released(
    rotation: np.ndarray,
    stage_target: np.ndarray,
    frame: np.ndarray,
    coordinates: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    result = rotation_tolerance_coordinates(rotation, stage_target, frame)
    result[np.asarray(mask, dtype=bool)] = np.asarray(coordinates)[mask]
    return rotation_from_tolerance_coordinates(stage_target, frame, result)


def test_shadow_projection_supports_all_rotation_masks() -> None:
    frame = rotvec_to_matrix(np.array([0.31, -0.24, 0.17]))
    stage_target = rotvec_to_matrix(np.array([0.18, -0.13, 0.22]))
    shadow_coordinates = np.array([-0.22, 0.09, 0.14])
    raw_coordinates = np.array([0.43, -0.36, 0.28])
    shadow = rotation_from_tolerance_coordinates(stage_target, frame, shadow_coordinates)
    raw = rotation_from_tolerance_coordinates(stage_target, frame, raw_coordinates)

    for values in itertools.product((False, True), repeat=3):
        mask = np.asarray(values, dtype=bool)
        corrected, diagnostics = project_historical_tolerance_bias(
            raw, shadow, stage_target, frame, mask
        )
        expected = np.where(mask, raw_coordinates, shadow_coordinates)
        np.testing.assert_allclose(
            rotation_tolerance_coordinates(corrected, stage_target, frame),
            expected,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            diagnostics.kept_bias_tolerance_frame,
            expected - shadow_coordinates,
            atol=1e-10,
        )
        assert diagnostics.non_tolerance_residual_rad <= 1e-10


def test_shadow_never_leaks_history_to_strict_axes() -> None:
    frame = rotvec_to_matrix(np.array([0.27, -0.18, 0.33]))
    stage_target = rotvec_to_matrix(np.array([-0.14, 0.11, -0.08]))
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
            stage_target,
            frame,
            mask,
            semantic_key=("adjust_bottle", "pregrasp"),
        )
        coordinates = _bias_coordinates(
            corrected, shadow.shadow_rotation, stage_target, frame
        )
        np.testing.assert_allclose(coordinates[~mask], 0.0, atol=1e-10)
        assert diagnostics.non_tolerance_residual_rad <= 1e-10
        executed = _released(
            corrected,
            stage_target,
            frame,
            np.array([0.41, 0.0, -0.47]),
            mask,
        )


def test_shadow_reanchors_when_stage_mask_changes() -> None:
    first_frame = rotvec_to_matrix(np.array([0.21, -0.12, 0.06]))
    second_frame = rotvec_to_matrix(np.array([-0.09, 0.18, 0.15]))
    first_target = rotvec_to_matrix(np.array([-0.13, 0.07, 0.16]))
    second_target = rotvec_to_matrix(np.array([0.08, -0.11, 0.19]))
    executed = rotvec_to_matrix(np.array([0.16, -0.23, 0.11]))
    shadow = ShadowOrientationReference(executed)
    _, first = shadow.advance(
        executed,
        np.array([0.08, -0.03, 0.04]),
        first_target,
        first_frame,
        np.array([True, False, False]),
        semantic_key=(0, 0),
    )
    assert first.anchor_reset

    corrected, changed = shadow.advance(
        executed,
        np.zeros(3),
        second_target,
        second_frame,
        np.array([False, False, True]),
        semantic_key=(2, 0),
    )
    assert changed.anchor_reset
    np.testing.assert_allclose(corrected, executed, atol=1e-10)


def test_mask_011_shadow_removes_accumulated_stage_roll_leakage() -> None:
    frame = rotvec_to_matrix(np.array([0.0, 0.0, 0.47]))
    stage_target = frame @ np.diag((-1.0, 1.0, -1.0))
    mask = np.array([False, True, True])
    shadow = ShadowOrientationReference(stage_target)
    semantic_key = ("upright_cylinder", "pregrasp")
    shadow.reset(
        stage_target,
        semantic_key=semantic_key,
        ranged_axes=mask,
        stage_target_rotation=stage_target,
        tolerance_frame=frame,
    )
    leaked = rotation_from_tolerance_coordinates(
        stage_target,
        frame,
        np.radians((31.32, 30.0, 42.0)),
    )
    corrected, diagnostics = shadow.advance(
        leaked,
        np.zeros(3),
        stage_target,
        frame,
        mask,
        semantic_key=semantic_key,
    )
    coordinates = rotation_tolerance_coordinates(
        corrected, stage_target, frame
    )
    np.testing.assert_allclose(
        np.degrees(coordinates), (0.0, 30.0, 42.0), atol=1e-9
    )
    assert abs(float(corrected[:, 1] @ frame[:, 2])) <= 1e-12
    assert diagnostics.non_tolerance_residual_rad <= 1e-12


def test_same_stage_target_and_frame_update_reexpresses_without_reset() -> None:
    first_frame = rotvec_to_matrix(np.array([0.21, -0.12, 0.06]))
    second_frame = rotvec_to_matrix(np.array([-0.09, 0.18, 0.15]))
    first_target = rotvec_to_matrix(np.array([-0.13, 0.07, 0.16]))
    second_target = rotvec_to_matrix(np.array([0.08, -0.11, 0.19]))
    mask = np.array([False, True, True])
    semantic_key = ("upright_cylinder", "pregrasp")
    optimized = rotvec_to_matrix(np.array([0.16, -0.23, 0.11]))
    shadow = ShadowOrientationReference(optimized)

    corrected, initialized = shadow.advance(
        optimized,
        np.array([0.08, -0.03, 0.04]),
        first_target,
        first_frame,
        mask,
        semantic_key=semantic_key,
    )
    assert initialized.anchor_reset
    old_absolute_shadow = shadow.shadow_rotation.copy()
    optimized = _released(
        corrected,
        first_target,
        first_frame,
        np.array([0.0, 0.55, -0.37]),
        mask,
    )
    corrected, chart_updated = shadow.advance(
        optimized,
        np.zeros(3),
        second_target,
        second_frame,
        mask,
        semantic_key=semantic_key,
    )
    assert not chart_updated.anchor_reset
    np.testing.assert_allclose(
        shadow.shadow_rotation, old_absolute_shadow, atol=1e-12
    )
    corrected_coordinates = rotation_tolerance_coordinates(
        corrected, second_target, second_frame
    )
    shadow_coordinates = rotation_tolerance_coordinates(
        old_absolute_shadow, second_target, second_frame
    )
    optimized_coordinates = rotation_tolerance_coordinates(
        optimized, second_target, second_frame
    )
    np.testing.assert_allclose(
        corrected_coordinates,
        np.where(mask, optimized_coordinates, shadow_coordinates),
        atol=1e-10,
    )


def test_shadow_planner_uses_the_same_baseline_sqp() -> None:
    baseline = BaselineSQPPlanner(
        solver_settings=SQPSettings(max_iterations=30, max_time_s=2.0),
    )
    planner = ShadowSQPPlanner(baseline)
    plan = planner.step(
        DEFAULT_HOME_Q,
        np.array([0.001, 0.0, 0.0, 0.02, 0.0, 0.0]),
        stage_target_rotation=np.eye(3),
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
        stage_target: np.ndarray,
        frame: np.ndarray,
        mask: np.ndarray,
        *,
        semantic_key: object,
    ):
        del action, stage_target, frame, mask, semantic_key
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
        stage_target_rotation=np.eye(3),
        tolerance_frame=np.eye(3),
        ranged_axes=np.array([True, False, False]),
        semantic_key=("task", "pregrasp"),
        axis_task_factory=task_factory,
    )

    np.testing.assert_allclose(seen["previous"], projected)
    np.testing.assert_allclose(seen["task_rotation"], corrected)
    np.testing.assert_allclose(seen["solve_rotation"], corrected)

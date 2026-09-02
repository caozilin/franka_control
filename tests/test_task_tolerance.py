from __future__ import annotations

import pathlib
import sys
import itertools

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from planning import (  # noqa: E402
    PANDA_TASK_TOLERANCE_IDS,
    PANDA_TOLERANCE_PROFILES,
    CartesianActionPlanner,
    GripperPhaseClassifier,
    ManipulationPhase,
    PlannerConfig,
    TaskKind,
    RotationalToleranceState,
    box_tolerance_frame,
)
from control.franka_env import DEFAULT_HOME_Q  # noqa: E402
from utils.control import ActionConfig  # noqa: E402
from utils.pose import (  # noqa: E402
    rotation_from_tolerance_coordinates,
    rotation_tolerance_coordinates,
    rotvec_to_matrix,
)
from planning.shadow_reference import project_release_target  # noqa: E402


def test_csv_task_rows_are_grouped_into_unique_tolerance_ids() -> None:
    assert len(PANDA_TOLERANCE_PROFILES) == 13
    assert len(PANDA_TASK_TOLERANCE_IDS) == 25
    assert set(PANDA_TASK_TOLERANCE_IDS.values()) == set(PANDA_TOLERANCE_PROFILES)
    assert PANDA_TASK_TOLERANCE_IDS["geometry_plate_cube"] == "T09"
    assert PANDA_TASK_TOLERANCE_IDS["geometry_plate_box_lying"] == "T09"


def test_phase_classifier_matches_mujoco_stable_width_rule() -> None:
    classifier = GripperPhaseClassifier()
    assert classifier.update(0.08, False).phase is ManipulationPhase.PREGRASP
    assert classifier.update(0.08, True).phase is ManipulationPhase.GRASP
    assert classifier.update(0.04, True).phase is ManipulationPhase.GRASP
    assert classifier.update(0.04, True).phase is ManipulationPhase.GRASP
    assert classifier.update(0.04, True).phase is ManipulationPhase.POSTGRASP
    assert classifier.update(0.04, False).phase is ManipulationPhase.RELEASE
    assert classifier.update(0.08, False).phase is ManipulationPhase.RELEASE
    assert classifier.update(0.08, False).phase is ManipulationPhase.RELEASE
    assert classifier.update(0.08, False).phase is ManipulationPhase.PREGRASP


def test_stage_configuration_preserves_asymmetric_bounds() -> None:
    planner = CartesianActionPlanner(PlannerConfig(mode="baseline_sqp"))
    planner.configure_rotation_tolerance(
        np.eye(3),
        np.eye(3),
        np.radians(np.array([0.0, 30.0, 0.0])),
        np.radians(np.array([0.0, 10.0, 45.0])),
    )
    assert planner._axis_tasks is not None
    assert planner._axis_tasks[3].kind is TaskKind.SPECIFIC
    assert planner._axis_tasks[4].kind is TaskKind.FLAT_RANGE
    assert np.isclose(planner._axis_tasks[4].lower, np.radians(-30.0))
    assert np.isclose(planner._axis_tasks[4].upper, np.radians(10.0))
    assert planner._axis_tasks[5].kind is TaskKind.FLAT_RANGE


def test_live_range_stays_centered_on_fixed_stage_target() -> None:
    planner = CartesianActionPlanner(PlannerConfig(mode="baseline_sqp"))
    assert planner._planner is not None
    initial_rotation = planner._planner.kinematics.evaluate(
        np.zeros(7),
        include_manipulability=False,
        include_link_points=False,
    ).rotation
    planner.configure_rotation_tolerance(
        initial_rotation,
        np.eye(3),
        np.radians(np.array([0.0, 0.0, 30.0])),
        np.radians(np.array([0.0, 0.0, 10.0])),
    )
    action = np.zeros(7, dtype=np.float64)
    action[5] = np.radians(20.0)
    transformed = action.copy()
    tasks = planner._axis_tasks_for_action(np.zeros(7), transformed)
    assert tasks is not None
    assert np.isclose(tasks[5].lower, np.radians(-30.0), atol=1e-10)
    assert np.isclose(tasks[5].upper, np.radians(20.0), atol=1e-10)
    assert np.isclose(tasks[5].absolute_lower, np.radians(-30.0), atol=1e-10)
    assert np.isclose(tasks[5].absolute_upper, np.radians(20.0), atol=1e-10)


def test_runtime_tasks_use_one_fixed_stage_chart_for_all_eight_masks() -> None:
    frame = rotvec_to_matrix(np.array([0.31, -0.24, 0.17]))
    stage_target = rotvec_to_matrix(np.array([-0.22, 0.09, 0.14]))
    planned_coordinates = np.array([0.12, -0.08, 0.06])
    planned = rotation_from_tolerance_coordinates(
        stage_target, frame, planned_coordinates
    )

    for values in itertools.product((False, True), repeat=3):
        mask = np.asarray(values, dtype=bool)
        state = RotationalToleranceState(
            target=stage_target,
            frame=frame,
            ranged=mask,
        )
        tasks = tuple(
            state.task(axis, planned, planned, planned) for axis in range(3)
        )
        if not np.any(mask):
            assert all(task.absolute_rotation_reference is None for task in tasks)
            assert all(task.kind is TaskKind.SPECIFIC for task in tasks)
            continue
        for axis, task in enumerate(tasks):
            np.testing.assert_allclose(
                task.absolute_rotation_reference, stage_target, atol=1e-12
            )
            if mask[axis]:
                assert task.kind is TaskKind.FLAT_RANGE
            else:
                assert task.kind is TaskKind.SPECIFIC
                assert np.isclose(task.goal, planned_coordinates[axis])


def test_release_projection_uses_one_formula_for_all_eight_masks() -> None:
    frame = rotvec_to_matrix(np.array([0.31, -0.24, 0.17]))
    reference = rotvec_to_matrix(np.array([-0.22, 0.09, 0.14]))
    planned_coordinates = np.array([0.12, -0.08, 0.06])
    optimized_coordinates = np.array([0.43, -0.36, 0.28])
    planned = rotation_from_tolerance_coordinates(
        reference, frame, planned_coordinates
    )
    optimized = rotation_from_tolerance_coordinates(
        reference, frame, optimized_coordinates
    )

    for values in itertools.product((False, True), repeat=3):
        mask = np.asarray(values, dtype=bool)
        projected, release = project_release_target(
            planned,
            optimized,
            frame,
            mask,
            reference_rotation=reference,
        )
        expected = np.where(mask, optimized_coordinates, planned_coordinates)
        np.testing.assert_allclose(
            rotation_tolerance_coordinates(projected, reference, frame),
            expected,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            release, expected - planned_coordinates, atol=1e-10
        )


def test_mask_011_releases_only_euler_pitch_and_yaw() -> None:
    frame = rotvec_to_matrix(np.array([0.0, 0.0, 0.47]))
    stage_target = frame @ np.diag((-1.0, 1.0, -1.0))
    planned = rotation_from_tolerance_coordinates(
        stage_target, frame, np.radians((18.0, 30.0, 42.0))
    )
    optimized = rotation_from_tolerance_coordinates(
        stage_target, frame, np.radians((13.0, 30.0, 42.0))
    )
    projected, _ = project_release_target(
        planned,
        optimized,
        frame,
        np.array((False, True, True)),
        reference_rotation=stage_target,
    )
    np.testing.assert_allclose(
        np.degrees(
            rotation_tolerance_coordinates(projected, stage_target, frame)
        ),
        (18.0, 30.0, 42.0),
        atol=1e-9,
    )


def test_box_tolerance_frame_keeps_world_z() -> None:
    rotation = np.array(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    frame = box_tolerance_frame(rotation)
    np.testing.assert_allclose(frame[:, 2], (0.0, 0.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)


def test_t03_full_speed_rotation_moves_from_cold_start() -> None:
    planner = CartesianActionPlanner(PlannerConfig(mode="baseline_sqp"))
    assert planner._planner is not None
    rotation = planner._planner.kinematics.evaluate(
        DEFAULT_HOME_Q,
        include_manipulability=False,
        include_link_points=False,
    ).rotation
    negative, positive = PANDA_TOLERANCE_PROFILES["T03"].bounds_rad(
        ManipulationPhase.PREGRASP,
    )
    planner.configure_rotation_tolerance(
        rotation,
        box_tolerance_frame(rotation),
        negative,
        positive,
        phase=ManipulationPhase.PREGRASP,
    )
    action = np.zeros(7)
    action[5] = np.radians(4.5)
    command = planner.plan(DEFAULT_HOME_Q, action, ActionConfig())
    assert command.telemetry is not None
    assert command.telemetry["feasible"]
    assert np.linalg.norm(command.joint_target - DEFAULT_HOME_Q) > 1e-3

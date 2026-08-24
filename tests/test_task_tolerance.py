from __future__ import annotations

import pathlib
import sys

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
    box_tolerance_frame,
)
from control.franka_env import DEFAULT_HOME_Q  # noqa: E402
from utils.control import ActionConfig  # noqa: E402


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
    assert np.isclose(tasks[5].upper, np.radians(10.0), atol=1e-10)
    assert np.isclose(tasks[5].absolute_lower, np.radians(-50.0), atol=1e-10)
    assert np.isclose(tasks[5].absolute_upper, 0.0, atol=1e-10)


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

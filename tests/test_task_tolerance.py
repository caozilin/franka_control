from __future__ import annotations

import itertools
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
    RotationalToleranceState,
    TaskKind,
    TargetPose,
    box_tolerance_frame,
)
from planning.tolerance.projection import (  # noqa: E402
    constraint_consistent_release_loss_reference,
)
from planning.tolerance.release_state import RotationReleaseState  # noqa: E402
from planning.tolerance.runtime import solve_stage_relative_target  # noqa: E402
from utils.pose import (  # noqa: E402
    rotation_from_tolerance_coordinates,
    rotation_tolerance_coordinates,
    rotvec_to_matrix,
    stage_reference_rotation,
)


def test_tolerance_ids_cover_selected_single_tasks() -> None:
    assert len(PANDA_TOLERANCE_PROFILES) == 16
    assert len(PANDA_TASK_TOLERANCE_IDS) == 27
    assert set(PANDA_TASK_TOLERANCE_IDS.values()) == set(PANDA_TOLERANCE_PROFILES)
    assert PANDA_TASK_TOLERANCE_IDS["geometry_plate_cylinder_upright"] == "T14"
    assert PANDA_TASK_TOLERANCE_IDS["geometry_region_cylinder_upright"] == "T04"
    assert PANDA_TASK_TOLERANCE_IDS["cylinder_to_narrow_box"] == "T15"
    assert PANDA_TASK_TOLERANCE_IDS["box_to_narrow_box"] == "T16"


def test_txx_profiles_store_mujoco_masks_with_uniform_thirty_degree_bounds() -> None:
    expected = {
        "T01": ((1, 1, 0), (0, 0, 1)),
        "T02": ((0, 1, 0), (0, 0, 1)),
        "T03": ((1, 1, 1), (1, 1, 1)),
        "T04": ((0, 1, 1), (0, 0, 1)),
        "T05": ((1, 1, 0), (0, 0, 1)),
        "T06": ((1, 1, 0), (1, 1, 1)),
        "T07": ((1, 1, 1), (1, 1, 1)),
        "T08": ((1, 1, 1), (1, 1, 1)),
        "T09": ((0, 1, 0), (1, 1, 1)),
        "T10": ((0, 1, 0), (1, 1, 1)),
        "T11": ((1, 1, 0), (1, 1, 1)),
        "T12": ((0, 1, 0), (0, 0, 0)),
        "T13": ((1, 1, 0), (1, 0, 0)),
        "T14": ((0, 1, 1), (1, 1, 1)),
        "T15": ((1, 1, 0), (1, 1, 0)),
        "T16": ((0, 1, 0), (0, 1, 0)),
    }
    for identifier, (pre_mask, post_mask) in expected.items():
        profile = PANDA_TOLERANCE_PROFILES[identifier]
        assert (profile.pre_mask, profile.post_mask) == (pre_mask, post_mask)
        for phase, mask in (
            (ManipulationPhase.PREGRASP, pre_mask),
            (ManipulationPhase.POSTGRASP, post_mask),
        ):
            negative, positive = profile.bounds_rad(phase)
            np.testing.assert_allclose(negative, np.radians(30 * np.asarray(mask)))
            np.testing.assert_allclose(positive, negative)
        for phase in (ManipulationPhase.GRASP, ManipulationPhase.RELEASE):
            negative, positive = profile.bounds_rad(phase)
            np.testing.assert_array_equal(negative, np.zeros(3))
            np.testing.assert_array_equal(positive, np.zeros(3))


def test_phase_classifier_matches_mujoco_stable_width_rule() -> None:
    classifier = GripperPhaseClassifier()
    assert classifier.update(0.08, False).phase is ManipulationPhase.PREGRASP
    assert classifier.update(0.08, True).phase is ManipulationPhase.GRASP
    assert classifier.update(0.04, True).phase is ManipulationPhase.GRASP
    assert classifier.update(0.04, True).phase is ManipulationPhase.GRASP
    assert classifier.update(0.04, True).phase is ManipulationPhase.POSTGRASP
    assert classifier.update(0.04, False).phase is ManipulationPhase.RELEASE


def test_stage_configuration_preserves_asymmetric_bounds() -> None:
    planner = CartesianActionPlanner(PlannerConfig(mode="ipopt"))
    planner.configure_rotation_tolerance(
        np.eye(3),
        np.eye(3),
        np.radians(np.array([0.0, 30.0, 0.0])),
        np.radians(np.array([0.0, 10.0, 45.0])),
    )
    state = planner._tolerance_state
    assert state is not None
    assert state.task(0).kind is TaskKind.SPECIFIC
    assert state.task(1).kind is TaskKind.FLAT_RANGE
    assert np.isclose(state.task(1).lower, np.radians(-30.0))
    assert np.isclose(state.task(1).upper, np.radians(10.0))
    assert state.task(2).kind is TaskKind.FLAT_RANGE


def test_runtime_tasks_share_one_release_mask_for_all_eight_masks() -> None:
    for values in itertools.product((False, True), repeat=3):
        mask = np.asarray(values, dtype=bool)
        state = RotationalToleranceState(ranged=mask)
        tasks = tuple(state.task(axis) for axis in range(3))
        for axis, task in enumerate(tasks):
            expected = TaskKind.FLAT_RANGE if mask[axis] else TaskKind.SPECIFIC
            assert task.kind is expected


def test_release_loss_projection_keeps_only_ranged_coordinates() -> None:
    frame = rotvec_to_matrix(np.array([0.31, -0.24, 0.17]))
    reference = rotvec_to_matrix(np.array([-0.22, 0.09, 0.14]))
    coordinates = np.array([0.12, -0.08, 0.06])
    carried = rotation_from_tolerance_coordinates(reference, frame, coordinates)
    for values in itertools.product((False, True), repeat=3):
        mask = np.asarray(values, dtype=bool)
        projected, projected_coordinates = (
            constraint_consistent_release_loss_reference(
                reference, carried, frame, mask
            )
        )
        expected = np.where(mask, coordinates, 0.0)
        np.testing.assert_allclose(projected_coordinates, expected, atol=1e-12)
        np.testing.assert_allclose(
            rotation_tolerance_coordinates(projected, reference, frame),
            expected,
            atol=1e-10,
        )


def test_release_state_transports_spatial_release_with_nominal_motion() -> None:
    handoff_nominal = rotvec_to_matrix(np.array([0.1, -0.2, 0.05]))
    handoff = rotvec_to_matrix(np.array([0.0, 0.0, 0.15])) @ handoff_nominal
    current_nominal = rotvec_to_matrix(np.array([-0.2, 0.1, 0.25]))
    state = RotationReleaseState()
    state.begin_stage(
        "post",
        optimized_rotation=handoff,
        nominal_rotation=handoff_nominal,
    )
    first_step = state.prepare_cycle(
        current_nominal,
        np.eye(3),
        np.array([False, False, True]),
        np.radians([-10.0, -10.0, -20.0]),
        np.radians([10.0, 10.0, 20.0]),
    )
    np.testing.assert_allclose(
        first_step.stage_reference_rotation(),
        stage_reference_rotation(handoff, handoff_nominal, current_nominal),
        atol=1e-12,
    )
    release = rotvec_to_matrix(np.array([0.0, 0.0, 0.08]))
    state.commit(release @ first_step.stage_reference_rotation(), first_step)
    next_nominal = rotvec_to_matrix(np.array([-0.1, 0.2, 0.3]))
    step = state.prepare_cycle(
        next_nominal,
        np.eye(3),
        np.array([False, False, True]),
        np.radians([-10.0, -10.0, -20.0]),
        np.radians([10.0, 10.0, 20.0]),
    )
    np.testing.assert_allclose(
        step.carried_rotation,
        release @ step.stage_reference_rotation(),
        atol=1e-12,
    )


def test_box_tolerance_frame_keeps_world_z() -> None:
    rotation = np.array(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    frame = box_tolerance_frame(rotation)
    np.testing.assert_allclose(frame[:, 2], (0.0, 0.0, 1.0), atol=1e-12)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)


def test_solver_freezes_realtime_nominal_frame_for_each_cycle() -> None:
    class Controller:
        def __init__(self) -> None:
            self.solve_frames: list[np.ndarray] = []

        def set_axis_task(self, axis: int, task: object) -> None:
            pass

        def solve(self, measured_q, target, active_dofs, frame, *, rotation_release_step):
            np.testing.assert_allclose(frame, rotation_release_step.tolerance_frame)
            self.solve_frames.append(frame.copy())
            return np.zeros(7), type("Diagnostics", (), {"feasible": False})()

    controller = Controller()
    tolerance = RotationalToleranceState(ranged=np.array((False, True, False)))
    release = RotationReleaseState()
    active_dofs = np.ones(6, dtype=bool)
    for angle_deg in (15.0, 65.0):
        nominal = rotvec_to_matrix(np.radians((0.0, 0.0, angle_deg)))
        target = TargetPose(np.zeros(3), nominal)
        _, _, _, _, step = solve_stage_relative_target(
            controller, np.zeros(7), target, tolerance, release, active_dofs,
        )
        np.testing.assert_allclose(step.tolerance_frame, box_tolerance_frame(nominal))
    assert not np.allclose(controller.solve_frames[0], controller.solve_frames[1])

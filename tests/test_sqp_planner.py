from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.franka_env import DEFAULT_HOME_Q, FrankaEnv  # noqa: E402
from planning.sqp import BaselineSQPPlanner, PandaKinematics, SQPSettings, TargetPose  # noqa: E402
from utils.pose import matrix_to_rotvec, rotvec_to_matrix  # noqa: E402


def test_panda_analytic_jacobian_matches_finite_difference() -> None:
    kinematics = PandaKinematics()
    q = DEFAULT_HOME_Q + np.array([0.08, -0.04, 0.03, 0.02, -0.05, 0.04, -0.02])
    state = kinematics.evaluate(q)
    epsilon = 1e-7
    numeric = np.empty((6, 7), dtype=np.float64)
    for index in range(7):
        probe = q.copy()
        probe[index] += epsilon
        changed = kinematics.evaluate(probe)
        numeric[:3, index] = (changed.position - state.position) / epsilon
        numeric[3:, index] = matrix_to_rotvec(changed.rotation @ state.rotation.T) / epsilon
    np.testing.assert_allclose(state.jacobian, numeric, atol=5e-7)


def test_accelerated_panda_kinematics_matches_python_fallback() -> None:
    accelerated = PandaKinematics()
    fallback = PandaKinematics(use_accelerated=False)
    assert accelerated.backend_name == "pinocchio"
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = np.clip(
            DEFAULT_HOME_Q + rng.normal(0.0, 0.12, size=7),
            accelerated.joint_lower + 0.05,
            accelerated.joint_upper - 0.05,
        )
        expected = fallback.evaluate(q)
        actual = accelerated.evaluate(q)
        np.testing.assert_allclose(actual.position, expected.position, atol=1e-12)
        np.testing.assert_allclose(actual.rotation, expected.rotation, atol=1e-12)
        np.testing.assert_allclose(actual.jacobian, expected.jacobian, atol=1e-12)
        np.testing.assert_allclose(actual.manipulability, expected.manipulability, atol=1e-12)
        np.testing.assert_allclose(actual.link_points, expected.link_points, atol=1e-12)


def test_panda_kinematics_can_skip_expensive_optional_fields() -> None:
    state = PandaKinematics().evaluate(
        DEFAULT_HOME_Q,
        include_manipulability=False,
        include_link_points=False,
    )
    assert state.manipulability is None
    assert state.link_points is None
    assert state.position.shape == (3,)
    assert state.rotation.shape == (3, 3)
    assert state.jacobian.shape == (6, 7)


def test_baseline_sqp_reaches_small_hard_pose_target() -> None:
    planner = BaselineSQPPlanner(
        solver_settings=SQPSettings(max_iterations=30, max_time_s=2.0),
    )
    initial = planner.kinematics.evaluate(DEFAULT_HOME_Q)
    target = TargetPose(
        initial.position + np.array([0.002, -0.001, 0.001]),
        rotvec_to_matrix(np.array([0.01, -0.008, 0.006])) @ initial.rotation,
    )
    plan = planner.solve_target(DEFAULT_HOME_Q, target)

    assert plan.solver.feasible, plan.solver.status
    reached = planner.kinematics.evaluate(plan.q)
    error = planner.kinematics.pose_error(reached, target.position, target.rotation)
    assert np.max(np.abs(error[:3])) <= planner.solver.settings.position_tolerance
    assert np.max(np.abs(error[3:])) <= planner.solver.settings.rotation_tolerance
    assert "self_collision" in plan.solver.task_costs


def test_no_robot_backend_accepts_absolute_joint_target() -> None:
    env = FrankaEnv(
        no_robot=True,
        no_cameras=True,
        print_events=False,
        control_mode="joint",
    )
    target = DEFAULT_HOME_Q + np.array([0.01, -0.02, 0.01, 0.0, 0.01, -0.01, 0.02])
    env.enqueue_joint_target(target)
    env.start_control_loop(max_duration=0.01)
    env.wait_control_loop()
    np.testing.assert_allclose(env.get_joint_positions(), target)
    env.stop()

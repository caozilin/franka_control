from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.franka_env import DEFAULT_HOME_Q  # noqa: E402
from planning import (  # noqa: E402
    CartesianActionPlanner,
    ConstraintValues,
    IpoptConsecutiveFailureError,
    IpoptPlanner,
    IpoptSettings,
    ObjectiveSettings,
    PlannerConfig,
    validate_candidate,
)
from planning.ipopt import IpoptAdapter  # noqa: E402
from planning.ipopt.types import TargetPose  # noqa: E402
from planning.kinematics import PandaKinematics  # noqa: E402
from planning.task_space import AxisTask, MotionState, TaskKind, TaskObjective  # noqa: E402
from utils.control import ActionConfig  # noqa: E402
from utils.pose import rotvec_to_matrix  # noqa: E402


def test_legacy_sqp_and_shadow_sources_are_removed() -> None:
    assert not (ROOT / "src" / "planning" / "shadow_reference.py").exists()
    for filename in (
        "__init__.py",
        "controller.py",
        "qp.py",
        "shadow_planner.py",
        "solver.py",
        "types.py",
    ):
        assert not (ROOT / "src" / "planning" / "sqp" / filename).exists()
    source = (ROOT / "src" / "planning" / "action_planner.py").read_text()
    assert 'PLANNER_MODE_CHOICES = ("direct", "ipopt")' in source
    assert "baseline_sqp" not in source
    assert "shadow_sqp" not in source


def _constraints(q: np.ndarray) -> ConstraintValues:
    equality = np.array((q[0] + q[1] - 1.0,))
    inequality = np.array((q[0] - 0.2,))
    return ConstraintValues(
        equality=equality,
        inequality=inequality,
        position_residual=float(abs(equality[0])),
        rotation_residual=0.0,
        inequality_violation=float(max(-inequality[0], 0.0)),
    )


def _jacobian(
    _q: np.ndarray,
    _values: ConstraintValues,
) -> tuple[np.ndarray, np.ndarray]:
    return np.array(((1.0, 1.0),)), np.array(((1.0, 0.0),))


def test_validate_candidate_recomputes_nonlinear_feasibility() -> None:
    feasible = validate_candidate(
        np.array((0.5, 0.5)),
        np.zeros(2),
        np.ones(2),
        _constraints(np.array((0.5, 0.5))),
        position_tolerance=1.0e-8,
        rotation_tolerance=1.0e-8,
        inequality_tolerance=1.0e-8,
    )
    assert feasible.feasible
    infeasible = validate_candidate(
        np.array((0.1, 0.9)),
        np.zeros(2),
        np.ones(2),
        _constraints(np.array((0.1, 0.9))),
        position_tolerance=1.0e-8,
        rotation_tolerance=1.0e-8,
        inequality_tolerance=1.0e-8,
    )
    assert not infeasible.feasible
    assert infeasible.inequality_violation == pytest.approx(0.1)


def test_dense_ipopt_adapter_solves_small_constrained_problem() -> None:
    adapter = IpoptAdapter(IpoptSettings(max_iterations=30, max_cpu_time_s=1.0))
    result = adapter.solve(
        lambda q: float((q[0] - 0.25) ** 2 + (q[1] - 0.75) ** 2),
        lambda q: np.array((2.0 * (q[0] - 0.25), 2.0 * (q[1] - 0.75))),
        _constraints,
        _jacobian,
        np.array((0.5, 0.5)),
        np.zeros(2),
        np.ones(2),
        position_tolerance=1.0e-7,
        rotation_tolerance=1.0e-7,
        inequality_tolerance=1.0e-7,
    )
    assert result.solver_succeeded
    assert result.validation.feasible
    assert result.q == pytest.approx(np.array((0.25, 0.75)), abs=1.0e-5)


def test_maintained_profile_has_only_delta_q_and_ema_release_weights() -> None:
    settings = ObjectiveSettings.delta_q_squared_conditioning_safeguard()
    nonzero = {
        name: value
        for name, value in vars(settings).items()
        if name.endswith("_weight") and value != 0.0
    }
    assert nonzero == {
        "squared_delta_q_weight": 1.0,
        "ema_release_loss_weight": 0.1,
    }


def test_squared_delta_q_objective_and_gradient_are_exact() -> None:
    kinematics = PandaKinematics()
    seed = DEFAULT_HOME_Q.copy()
    state = kinematics.evaluate(seed)
    delta = np.array((0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01))
    probe = seed + delta
    settings = ObjectiveSettings.delta_q_squared_only()
    objective = TaskObjective(
        kinematics,
        TargetPose(state.position, state.rotation),
        np.zeros(6, dtype=bool),
        MotionState(seed, seed, seed),
        settings,
        tuple(AxisTask(TaskKind.FREE) for _ in range(6)),
    )
    cost, parts = objective.evaluate(probe, include_breakdown=True)
    gradient = objective.gradient(
        probe,
        1.0e-7,
        kinematics.joint_lower,
        kinematics.joint_upper,
    )
    assert tuple(parts) == ("delta_q_squared",)
    assert cost == pytest.approx(delta @ delta)
    np.testing.assert_allclose(gradient, 2.0 * delta, atol=1.0e-12)


@pytest.mark.parametrize(
    "tolerance_frame",
    (None, rotvec_to_matrix(np.array((0.2, -0.1, 0.3)))),
)
def test_pinocchio_pose_error_jacobian_matches_finite_difference(
    tolerance_frame: np.ndarray | None,
) -> None:
    kinematics = PandaKinematics()
    q = DEFAULT_HOME_Q + np.array((0.03, -0.02, 0.01, 0.02, -0.01, 0.01, -0.02))
    state = kinematics.evaluate(q, include_manipulability=False, include_link_points=False)
    goal_rotation = rotvec_to_matrix(np.array((0.03, -0.02, 0.01))) @ state.rotation
    analytic = kinematics.pose_error_jacobian(
        state,
        goal_rotation,
        tolerance_frame,
    )
    epsilon = 1.0e-7
    numerical = np.empty((6, 7))
    for index in range(7):
        lower = q.copy()
        upper = q.copy()
        lower[index] -= epsilon
        upper[index] += epsilon
        lower_state = kinematics.evaluate(
            lower, include_manipulability=False, include_link_points=False
        )
        upper_state = kinematics.evaluate(
            upper, include_manipulability=False, include_link_points=False
        )
        numerical[:, index] = (
            kinematics.pose_error(
                upper_state, state.position, goal_rotation, tolerance_frame
            )
            - kinematics.pose_error(
                lower_state, state.position, goal_rotation, tolerance_frame
            )
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numerical, rtol=2.0e-5, atol=2.0e-6)


def test_zero_action_ipopt_plan_is_feasible_and_holds_home() -> None:
    planner = CartesianActionPlanner(PlannerConfig(mode="ipopt"))
    command = planner.plan(DEFAULT_HOME_Q, np.zeros(7), ActionConfig())
    assert command.telemetry is not None
    assert command.telemetry["feasible"]
    assert command.telemetry["planner_mode"] == "ipopt"
    np.testing.assert_allclose(command.joint_target, DEFAULT_HOME_Q, atol=1.0e-7)


def test_three_consecutive_rejections_raise_terminal_error() -> None:
    planner = IpoptPlanner(
        settings=IpoptSettings(max_iterations=2, max_cpu_time_s=0.02)
    )
    impossible = np.array((10.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    for _ in range(2):
        plan = planner.step(DEFAULT_HOME_Q, impossible)
        assert not plan.diagnostics.feasible
    with pytest.raises(
        IpoptConsecutiveFailureError,
        match=(
            r"three consecutive.*position_residual=.*"
            r"strict_rotation_residual=.*inequality_violation="
        ),
    ):
        planner.step(DEFAULT_HOME_Q, impossible)

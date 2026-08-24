from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.franka_env import FrankaEnv  # noqa: E402
from planning import CartesianActionPlanner, ControlRoute, PlannerConfig  # noqa: E402
from utils.control import ActionConfig  # noqa: E402


def test_direct_planner_preserves_cartesian_command_contract() -> None:
    planner = CartesianActionPlanner(PlannerConfig(mode="direct"))
    action = np.array([0.01, -0.02, 0.0, 0.1, 0.0, -0.1, -1.0], dtype=np.float64)
    command = planner.plan(np.zeros(7), action, ActionConfig())

    assert planner.control_mode == "cartesian"
    assert command.reference_space == "cartesian"
    np.testing.assert_allclose(command.cartesian_action, action)
    assert command.telemetry is None


def test_env_executes_baseline_sqp_through_same_cartesian_interface() -> None:
    planner = CartesianActionPlanner(PlannerConfig(mode="baseline_sqp"))
    env = FrankaEnv(
        no_robot=True,
        no_cameras=True,
        print_events=False,
        action_planner=planner,
        reference_name="cubic",
    )
    try:
        command = env.enqueue_cartesian_action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]))
        assert env.control_mode == "joint"
        assert env.reference_name == "cubic"
        assert env.tracker_mode == "joint_pid"
        assert command.reference_space == "joint"
        assert command.joint_target is not None
        assert command.telemetry is not None
        assert command.telemetry["planner_mode"] == "baseline_sqp"
        assert command.actual_pose is not None
        assert command.planned_pose is not None
        assert command.nominal_pose is not None
        assert env.get_pending_action_count() == 1
    finally:
        env.stop()


@pytest.mark.parametrize(
    "mode",
    ("direct", "baseline_sqp"),
)
def test_planner_routes_publish_the_same_10hz_event(
    mode: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = FrankaEnv(
        no_robot=True,
        no_cameras=True,
        print_events=True,
        action_planner=CartesianActionPlanner(PlannerConfig(mode=mode)),
    )
    try:
        env.enqueue_cartesian_action(np.zeros(7, dtype=np.float64))
        output = capsys.readouterr().out
        assert "10Hz tick 0001" in output
        assert "planner=" not in output
        assert "output=" not in output
        assert "reference=" not in output
        assert "tracker=" not in output
        assert "actual_plan_dxyz_m=" in output
        assert "actual_plan_drot_rad=" in output
        assert "plan_nominal_dxyz_m=" not in output
        assert "plan_nominal_drot_rad=" not in output
        if mode == "baseline_sqp":
            assert output.index("actual_plan_drot_rad=") < output.index("sqp_status=")
    finally:
        env.stop()


def test_env_defaults_to_direct_planner_for_backward_compatibility() -> None:
    env = FrankaEnv(no_robot=True, no_cameras=True, print_events=False)
    try:
        assert env.action_planner is not None
        assert env.action_planner.mode == "direct"
        command = env.enqueue_cartesian_action(np.zeros(7))
        assert command.reference_space == "cartesian"
    finally:
        env.stop()


@pytest.mark.parametrize("profile", ("min_jerk", "linear", "cubic"))
def test_joint_route_selects_reference_independently(profile: str) -> None:
    planner = CartesianActionPlanner(PlannerConfig(mode="baseline_sqp"))
    route = ControlRoute(planner, profile, "joint_impedance")

    assert route.reference_space == "joint"
    assert route.reference_profile == profile
    assert route.tracker_mode == "joint_impedance"


def test_joint_pid_is_an_independent_joint_tracker_choice() -> None:
    sqp = CartesianActionPlanner(PlannerConfig(mode="baseline_sqp"))
    route = ControlRoute(sqp, "min_jerk", "joint_pid")

    assert route.reference_space == "joint"
    assert route.tracker_mode == "joint_pid"

    env = FrankaEnv(
        no_robot=True,
        no_cameras=True,
        print_events=False,
        action_planner=sqp,
        tracker_mode="joint_pid",
    )
    try:
        assert env.tracker_mode == "joint_pid"
    finally:
        env.stop()


def test_auto_tracker_follows_reference_space() -> None:
    direct = ControlRoute(CartesianActionPlanner(PlannerConfig(mode="direct")), "linear", "auto")
    sqp = ControlRoute(CartesianActionPlanner(PlannerConfig(mode="baseline_sqp")), "linear", "auto")
    assert direct.tracker_mode == "cartesian_impedance"
    assert sqp.tracker_mode == "joint_pid"

    env = FrankaEnv(
        no_robot=True,
        no_cameras=True,
        print_events=False,
        control_mode="joint",
        tracker_mode="auto",
    )
    try:
        assert env.tracker_mode == "joint_pid"
    finally:
        env.stop()


def test_pid_cli_mode_follows_reference_space_without_converting_reference() -> None:
    direct = ControlRoute(CartesianActionPlanner(PlannerConfig(mode="direct")), "linear", "pid")
    sqp = ControlRoute(CartesianActionPlanner(PlannerConfig(mode="baseline_sqp")), "linear", "pid")
    assert direct.reference_space == "cartesian"
    assert direct.tracker_mode == "cartesian_impedance"
    assert sqp.reference_space == "joint"
    assert sqp.tracker_mode == "joint_pid"


def test_router_rejects_cross_space_tracker_and_unsupported_reference() -> None:
    direct = CartesianActionPlanner(PlannerConfig(mode="direct"))
    sqp = CartesianActionPlanner(PlannerConfig(mode="baseline_sqp"))

    with pytest.raises(ValueError, match="requires tracker"):
        ControlRoute(direct, "linear", "joint_impedance")
    with pytest.raises(ValueError, match="requires"):
        ControlRoute(direct, "linear", "joint_pid")
    with pytest.raises(ValueError, match="joint reference"):
        ControlRoute(sqp, "motion_limited", "joint_impedance")

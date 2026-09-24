#!/usr/bin/env python3
"""Bounded real-robot diagnostic for IPOPT static and fast tracking error.

The experiment keeps the commanded and measured translation inside a hard
10 cm box around the post-home pose.  Static mode applies translation-only
waypoints and waits at each waypoint.  Fast mode continuously sweeps each axis
at 10 cm/s between +/-8 cm targets.  Output is deferred until control stops.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.cli_args import parse_joint_vector  # noqa: E402
from control.franka_env import (  # noqa: E402
    DEFAULT_HOME_Q,
    DEFAULT_JOINT_DAMPING,
    DEFAULT_JOINT_STIFFNESS,
    FrankaEnv,
    ROBOT_IP,
)
from control.pid_config import add_joint_pid_arguments, joint_pid_kwargs  # noqa: E402
from planning import CartesianActionPlanner, PlannerConfig  # noqa: E402
from planning.kinematics import PandaKinematics  # noqa: E402
from utils.pose import matrix_to_rotvec, rotvec_to_matrix  # noqa: E402


POLICY_DT_S = 0.1
HARD_HOME_BOUND_M = 0.10
FAST_COMMAND_SPEED_LIMIT_M_S = 0.10
FAST_TARGET_BOUND_M = 0.08


class SafetyBoundaryError(RuntimeError):
    """Raised before or immediately after a bounded diagnostic violation."""


@dataclass(frozen=True)
class ExperimentConfig:
    offset_m: float
    step_m: float
    settle_s: float
    sample_window_s: float
    max_rotation_error_rad: float


@dataclass(frozen=True)
class FastExperimentConfig:
    offset_m: float
    speed_m_s: float
    hold_s: float


def _rotation_error(target_rotvec: np.ndarray, source_rotvec: np.ndarray) -> np.ndarray:
    target = rotvec_to_matrix(np.asarray(target_rotvec, dtype=np.float64))
    source = rotvec_to_matrix(np.asarray(source_rotvec, dtype=np.float64))
    return matrix_to_rotvec(target @ source.T)


def _rotation_errors(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_rotation_error(target_row, source_row) for target_row, source_row in zip(target, source, strict=True)],
        dtype=np.float64,
    )


def _vector_stats(values: np.ndarray) -> dict[str, object]:
    vectors = np.asarray(values, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("diagnostic vectors must be a 2D array")
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "mean": np.mean(vectors, axis=0).tolist(),
        "final": vectors[-1].tolist(),
        "mean_norm": float(np.mean(norms)),
        "p95_norm": float(np.percentile(norms, 95.0)),
        "max_norm": float(np.max(norms)),
    }


def _scalar_stats(values: np.ndarray) -> dict[str, float]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("diagnostic scalar samples must be a non-empty 1D array")
    return {
        "mean": float(np.mean(samples)),
        "final": float(samples[-1]),
        "p95": float(np.percentile(samples, 95.0)),
        "max": float(np.max(samples)),
    }


def _speed_stats(times: np.ndarray, positions: np.ndarray) -> dict[str, float]:
    timestamps = np.asarray(times, dtype=np.float64)
    xyz = np.asarray(positions, dtype=np.float64)
    if timestamps.ndim != 1 or xyz.ndim != 2 or xyz.shape != (timestamps.size, 3):
        raise ValueError("speed samples must contain matching timestamps and 3D positions")
    if timestamps.size < 2:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    dt = np.diff(timestamps)
    valid = dt > 1.0e-9
    if not np.any(valid):
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    speeds = np.linalg.norm(np.diff(xyz, axis=0)[valid], axis=1) / dt[valid]
    return {
        "mean": float(np.mean(speeds)),
        "p95": float(np.percentile(speeds, 95.0)),
        "max": float(np.max(speeds)),
    }


def _motion_trace_report(
    trace: np.ndarray,
    direction: np.ndarray,
    home_planned_position: np.ndarray,
    home_actual_position: np.ndarray,
) -> dict[str, object]:
    samples = np.asarray(trace, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 47 or samples.shape[0] == 0:
        raise RuntimeError("C++ trace contains no complete samples for the motion window")

    unit_direction = np.asarray(direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(unit_direction))
    if direction_norm <= 0.0:
        raise ValueError("motion direction must be non-zero")
    unit_direction = unit_direction / direction_norm

    goal_xyz = samples[:, 1:4]
    goal_rotvec = samples[:, 4:7]
    reference_xyz = samples[:, 7:10]
    reference_rotvec = samples[:, 10:13]
    actual_xyz = samples[:, 13:16]
    actual_rotvec = samples[:, 16:19]
    goal_actual = goal_xyz - actual_xyz

    # Joint-control pose diagnostics are refreshed at 100 Hz and repeated in
    # the 1 kHz torque trace.  A 10x decimation therefore yields real pose
    # samples without turning each refresh into a false 1 ms speed spike.
    pose_samples = samples[::10]
    if pose_samples[-1, 0] != samples[-1, 0]:
        pose_samples = np.vstack((pose_samples, samples[-1]))

    along_error = goal_actual @ unit_direction
    cross_error = goal_actual - np.outer(along_error, unit_direction)
    return {
        "trace_samples": int(samples.shape[0]),
        "duration_s": float(samples[-1, 0] - samples[0, 0]),
        "goal_minus_reference_m": _vector_stats(goal_xyz - reference_xyz),
        "reference_minus_actual_m": _vector_stats(reference_xyz - actual_xyz),
        "goal_minus_actual_m": _vector_stats(goal_actual),
        "goal_minus_actual_along_m": _scalar_stats(along_error),
        "goal_minus_actual_cross_m": _vector_stats(cross_error),
        "goal_minus_reference_rotvec_rad": _vector_stats(
            _rotation_errors(goal_rotvec, reference_rotvec)
        ),
        "reference_minus_actual_rotvec_rad": _vector_stats(
            _rotation_errors(reference_rotvec, actual_rotvec)
        ),
        "goal_minus_actual_rotvec_rad": _vector_stats(
            _rotation_errors(goal_rotvec, actual_rotvec)
        ),
        "reference_speed_m_s": _speed_stats(pose_samples[:, 0], pose_samples[:, 7:10]),
        "actual_speed_m_s": _speed_stats(pose_samples[:, 0], pose_samples[:, 13:16]),
        "goal_home_abs_max_m": np.max(
            np.abs(goal_xyz - np.asarray(home_planned_position, dtype=np.float64)), axis=0
        ).tolist(),
        "reference_home_abs_max_m": np.max(
            np.abs(reference_xyz - np.asarray(home_planned_position, dtype=np.float64)), axis=0
        ).tolist(),
        "actual_home_abs_max_m": np.max(
            np.abs(actual_xyz - np.asarray(home_actual_position, dtype=np.float64)), axis=0
        ).tolist(),
        "tau_command_abs_max_nm": np.max(np.abs(samples[:, 19:26]), axis=0).tolist(),
        "tau_desired_abs_max_nm": np.max(np.abs(samples[:, 26:33]), axis=0).tolist(),
    }


def _settled_trace_report(trace: np.ndarray, sample_window_s: float) -> dict[str, object]:
    samples = np.asarray(trace, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 47 or samples.shape[0] == 0:
        raise RuntimeError("C++ trace contains no complete samples for the settled window")
    cutoff = float(samples[-1, 0] - sample_window_s)
    settled = samples[samples[:, 0] >= cutoff]
    if settled.shape[0] == 0:
        settled = samples[-1:]

    goal_xyz = settled[:, 1:4]
    goal_rotvec = settled[:, 4:7]
    reference_xyz = settled[:, 7:10]
    reference_rotvec = settled[:, 10:13]
    actual_xyz = settled[:, 13:16]
    actual_rotvec = settled[:, 16:19]
    return {
        "trace_samples": int(settled.shape[0]),
        "window_s": float(settled[-1, 0] - settled[0, 0]),
        "goal_minus_reference_m": _vector_stats(goal_xyz - reference_xyz),
        "reference_minus_actual_m": _vector_stats(reference_xyz - actual_xyz),
        "goal_minus_actual_m": _vector_stats(goal_xyz - actual_xyz),
        "goal_minus_reference_rotvec_rad": _vector_stats(
            _rotation_errors(goal_rotvec, reference_rotvec)
        ),
        "reference_minus_actual_rotvec_rad": _vector_stats(
            _rotation_errors(reference_rotvec, actual_rotvec)
        ),
        "goal_minus_actual_rotvec_rad": _vector_stats(
            _rotation_errors(goal_rotvec, actual_rotvec)
        ),
        "tau_command_abs_max_nm": np.max(np.abs(settled[:, 19:26]), axis=0).tolist(),
        "tau_desired_abs_max_nm": np.max(np.abs(settled[:, 26:33]), axis=0).tolist(),
    }


def _assert_translation_bound(label: str, position: np.ndarray, home: np.ndarray) -> None:
    offset = np.asarray(position, dtype=np.float64) - np.asarray(home, dtype=np.float64)
    if np.any(np.abs(offset) > HARD_HOME_BOUND_M + 1.0e-9):
        raise SafetyBoundaryError(
            f"{label} left the home +/-{HARD_HOME_BOUND_M:.2f} m box: "
            f"offset={offset.tolist()}"
        )


def _assert_actual_safe(
    env: FrankaEnv,
    home_actual_position: np.ndarray,
    home_actual_rotation: np.ndarray,
    max_rotation_error_rad: float,
) -> None:
    state = env.get_robot_state_vector()
    _assert_translation_bound("measured pose", state[:3], home_actual_position)
    rotation_error = _rotation_error(home_actual_rotation, state[3:6])
    if np.linalg.norm(rotation_error) > max_rotation_error_rad:
        raise SafetyBoundaryError(
            "measured orientation exceeded the diagnostic limit: "
            f"error_rotvec={rotation_error.tolist()}, "
            f"limit_rad={max_rotation_error_rad:.6f}"
        )


def _assert_motion_trace_safe(
    trace: np.ndarray,
    home_planned_position: np.ndarray,
    home_actual_position: np.ndarray,
) -> None:
    samples = np.asarray(trace, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 19 or samples.shape[0] == 0:
        raise RuntimeError("cannot verify Cartesian safety without a complete motion trace")
    checks = (
        ("IPOPT goal trace", samples[:, 1:4], home_planned_position),
        ("reference trace", samples[:, 7:10], home_planned_position),
        ("measured trace", samples[:, 13:16], home_actual_position),
    )
    for label, positions, home in checks:
        offsets = positions - np.asarray(home, dtype=np.float64)
        max_index = np.unravel_index(np.argmax(np.abs(offsets)), offsets.shape)
        if abs(float(offsets[max_index])) > HARD_HOME_BOUND_M + 1.0e-9:
            raise SafetyBoundaryError(
                f"{label} left the home +/-{HARD_HOME_BOUND_M:.2f} m box: "
                f"offset={offsets[max_index[0]].tolist()}"
            )


def _wait_until(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining > 0.0:
        time.sleep(remaining)


def _plan_and_enqueue(
    env: FrankaEnv,
    planner: CartesianActionPlanner,
    action_xyz: np.ndarray,
    home_planned_position: np.ndarray,
    stage: str,
):
    action = np.zeros(7, dtype=np.float64)
    action[:3] = np.asarray(action_xyz, dtype=np.float64)
    action[6] = -1.0
    command = planner.plan(
        env.get_joint_positions(),
        action,
        env.action_config,
        semantic_key=("ipopt_static_tracking_diagnostic", stage),
    )
    if command.telemetry is None or not bool(command.telemetry.get("feasible", False)):
        raise RuntimeError(f"IPOPT rejected diagnostic stage {stage!r}: {command.telemetry}")
    assert command.joint_target is not None
    assert command.planned_pose is not None
    assert command.nominal_pose is not None
    _assert_translation_bound(
        "IPOPT nominal target", command.nominal_pose[:3, 3], home_planned_position
    )
    _assert_translation_bound(
        "IPOPT joint target", command.planned_pose[:3, 3], home_planned_position
    )
    env.enqueue_joint_target(command.joint_target, gripper_target=command.gripper_target)
    return command


def _move_to_offset(
    env: FrankaEnv,
    planner: CartesianActionPlanner,
    current_offset: np.ndarray,
    target_offset: np.ndarray,
    home_planned_position: np.ndarray,
    home_actual_position: np.ndarray,
    home_actual_rotation: np.ndarray,
    config: ExperimentConfig,
    stage: str,
):
    offset = np.asarray(current_offset, dtype=np.float64).copy()
    target = np.asarray(target_offset, dtype=np.float64)
    last_command = None
    next_tick = time.monotonic()
    while np.linalg.norm(target - offset, ord=np.inf) > 1.0e-12:
        delta = target - offset
        action_xyz = np.clip(delta, -config.step_m, config.step_m)
        intended = offset + action_xyz
        if np.any(np.abs(intended) > HARD_HOME_BOUND_M + 1.0e-12):
            raise SafetyBoundaryError(f"requested waypoint {stage!r} exceeds the home box")
        last_command = _plan_and_enqueue(
            env,
            planner,
            action_xyz,
            home_planned_position,
            stage,
        )
        offset = intended
        _assert_actual_safe(
            env,
            home_actual_position,
            home_actual_rotation,
            config.max_rotation_error_rad,
        )
        if not env.is_control_running():
            env.check_control_error()
            raise RuntimeError("control stopped before the diagnostic motion completed")
        next_tick += POLICY_DT_S
        _wait_until(next_tick)
    return offset, last_command


def _measure_hold(
    env: FrankaEnv,
    target_q: np.ndarray,
    home_actual_position: np.ndarray,
    home_actual_rotation: np.ndarray,
    config: ExperimentConfig,
    *,
    simulate_trace: bool = False,
) -> dict[str, object]:
    trace_head = env.read_trace_head()
    joint_errors: list[np.ndarray] = []
    deadline = time.monotonic() + config.settle_s
    next_tick = time.monotonic()
    while time.monotonic() < deadline:
        _assert_actual_safe(
            env,
            home_actual_position,
            home_actual_rotation,
            config.max_rotation_error_rad,
        )
        if not env.is_control_running():
            env.check_control_error()
            raise RuntimeError("control stopped during the diagnostic hold")
        joint_errors.append(np.asarray(target_q, dtype=np.float64) - env.get_joint_positions())
        next_tick += POLICY_DT_S
        _wait_until(min(next_tick, deadline))

    trace = env.get_trace_since(trace_head)
    if trace.shape[0] == 0 and simulate_trace:
        kinematics = PandaKinematics()
        goal = kinematics.evaluate(
            target_q, include_manipulability=False, include_link_points=False
        )
        actual = kinematics.evaluate(
            env.get_joint_positions(),
            include_manipulability=False,
            include_link_points=False,
        )
        trace = np.zeros((1, 47), dtype=np.float64)
        trace[0, 1:4] = goal.position
        trace[0, 4:7] = matrix_to_rotvec(goal.rotation)
        trace[0, 7:10] = goal.position
        trace[0, 10:13] = matrix_to_rotvec(goal.rotation)
        trace[0, 13:16] = actual.position
        trace[0, 16:19] = matrix_to_rotvec(actual.rotation)
    report = _settled_trace_report(trace, min(config.sample_window_s, config.settle_s))
    joint_array = np.asarray(joint_errors, dtype=np.float64)
    window_count = max(1, int(math.ceil(config.sample_window_s / POLICY_DT_S)))
    report["joint_target_minus_actual_rad"] = _vector_stats(joint_array[-window_count:])
    return report


def _waypoints(offset_m: float) -> tuple[tuple[str, np.ndarray], ...]:
    zero = np.zeros(3, dtype=np.float64)
    result: list[tuple[str, np.ndarray]] = [("home_initial", zero.copy())]
    for axis, name in enumerate(("x", "y", "z")):
        positive = zero.copy()
        negative = zero.copy()
        positive[axis] = offset_m
        negative[axis] = -offset_m
        result.extend(
            (
                (f"{name}_positive", positive),
                (f"{name}_negative", negative),
                (f"home_after_{name}", zero.copy()),
            )
        )
    return tuple(result)


def _fast_waypoints(offset_m: float) -> tuple[tuple[str, np.ndarray], ...]:
    zero = np.zeros(3, dtype=np.float64)
    result: list[tuple[str, np.ndarray]] = []
    for axis, name in enumerate(("x", "y", "z")):
        positive = zero.copy()
        negative = zero.copy()
        positive[axis] = offset_m
        negative[axis] = -offset_m
        result.extend(
            (
                (f"fast_{name}_positive", positive),
                (f"fast_{name}_sweep_negative", negative),
                (f"fast_{name}_return_home", zero.copy()),
            )
        )
    return tuple(result)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure stationary or fast IPOPT goal/reference/actual tracking "
            "error inside a hard home +/-10 cm Cartesian box."
        )
    )
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--confirm-motion", action="store_true")
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument(
        "--experiment",
        choices=("static", "fast", "both"),
        default="static",
        help="Run the original static test, the 10 cm/s sweep, or both.",
    )
    parser.add_argument("--offset-m", type=float, default=0.02)
    parser.add_argument("--step-m", type=float, default=0.001)
    parser.add_argument("--settle-s", type=float, default=3.0)
    parser.add_argument("--sample-window-s", type=float, default=1.0)
    parser.add_argument("--fast-offset-m", type=float, default=FAST_TARGET_BOUND_M)
    parser.add_argument(
        "--fast-speed-m-s",
        type=float,
        default=FAST_COMMAND_SPEED_LIMIT_M_S,
    )
    parser.add_argument("--fast-hold-s", type=float, default=3.0)
    parser.add_argument("--max-rotation-error-deg", type=float, default=15.0)
    parser.add_argument("--reset-duration", type=float, default=5.0)
    parser.add_argument("--reference", choices=("linear", "cubic", "min_jerk"), default="linear")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Full JSON report path (default: logs/ipopt_static_tracking_<timestamp>.json)",
    )
    parser.add_argument(
        "--joint-stiffness",
        type=float,
        nargs=7,
        default=tuple(DEFAULT_JOINT_STIFFNESS),
    )
    parser.add_argument(
        "--joint-damping",
        type=float,
        nargs=7,
        default=tuple(DEFAULT_JOINT_DAMPING),
    )
    add_joint_pid_arguments(parser)
    parser.add_argument("--home-q", type=parse_joint_vector, default=DEFAULT_HOME_Q.copy())
    return parser


def _format_vector(values: object, scale: float) -> str:
    vector = np.asarray(values, dtype=np.float64) * scale
    return "[" + ",".join(f"{value:+.3f}" for value in vector) + "]"


def _print_summary(report: dict[str, object], output_path: pathlib.Path) -> None:
    print(
        "IPOPT_STATIC_TRACKING_DIAGNOSTIC "
        f"status={report.get('status')} report={output_path.resolve()}",
        flush=True,
    )
    if report.get("status") != "ok":
        print(
            f"error_type={report.get('error_type', 'unknown')} "
            f"error={report.get('error', 'unknown')}",
            flush=True,
        )
        return
    print(
        "units: xyz=mm rotation=deg joint=deg torque=Nm; "
        "g-r=goal-reference r-a=reference-actual g-a=goal-actual",
        flush=True,
    )
    for raw_stage in report["stages"]:
        stage = raw_stage
        goal_reference = stage["goal_minus_reference_m"]["final"]
        reference_actual = stage["reference_minus_actual_m"]["final"]
        goal_actual = stage["goal_minus_actual_m"]["final"]
        goal_actual_rotation = stage["goal_minus_actual_rotvec_rad"]["final"]
        joint_error = stage["joint_target_minus_actual_rad"]["final"]
        torque = stage["tau_command_abs_max_nm"]
        print(
            f"stage={stage['stage']} "
            f"offset_mm={_format_vector(stage['target_offset_m'], 1000.0)} "
            f"g-r_xyz={_format_vector(goal_reference, 1000.0)} "
            f"r-a_xyz={_format_vector(reference_actual, 1000.0)} "
            f"g-a_xyz={_format_vector(goal_actual, 1000.0)} "
            f"g-a_rot={_format_vector(goal_actual_rotation, 180.0 / math.pi)} "
            f"joint={_format_vector(joint_error, 180.0 / math.pi)} "
            f"tau_max={_format_vector(torque, 1.0)}",
            flush=True,
        )
    for raw_stage in report.get("motion_stages", []):
        stage = raw_stage
        torque = stage["tau_command_abs_max_nm"]
        print(
            f"motion={stage['stage']} "
            f"start_mm={_format_vector(stage['start_offset_m'], 1000.0)} "
            f"target_mm={_format_vector(stage['target_offset_m'], 1000.0)} "
            f"command_cm_s={stage['command_speed_m_s'] * 100.0:.2f} "
            f"duration_s={stage['duration_s']:.3f} "
            f"g-r_p95_mm={stage['goal_minus_reference_m']['p95_norm'] * 1000.0:.3f} "
            f"r-a_p95_mm={stage['reference_minus_actual_m']['p95_norm'] * 1000.0:.3f} "
            f"g-a_p95_mm={stage['goal_minus_actual_m']['p95_norm'] * 1000.0:.3f} "
            f"g-a_max_mm={stage['goal_minus_actual_m']['max_norm'] * 1000.0:.3f} "
            f"along_mean_mm={stage['goal_minus_actual_along_m']['mean'] * 1000.0:+.3f} "
            f"cross_p95_mm={stage['goal_minus_actual_cross_m']['p95_norm'] * 1000.0:.3f} "
            f"rot_p95_deg={stage['goal_minus_actual_rotvec_rad']['p95_norm'] * 180.0 / math.pi:.3f} "
            f"ref_speed_mean_cm_s={stage['reference_speed_m_s']['mean'] * 100.0:.2f} "
            f"actual_speed_mean_cm_s={stage['actual_speed_m_s']['mean'] * 100.0:.2f} "
            f"actual_bound_mm={_format_vector(stage['actual_home_abs_max_m'], 1000.0)} "
            f"tau_max={_format_vector(torque, 1.0)}",
            flush=True,
        )
    for raw_stage in report.get("recovery_stages", []):
        stage = raw_stage
        print(
            f"recovery={stage['stage']} "
            f"g-a_xyz={_format_vector(stage['goal_minus_actual_m']['final'], 1000.0)} "
            f"g-a_rot={_format_vector(stage['goal_minus_actual_rotvec_rad']['final'], 180.0 / math.pi)} "
            f"joint={_format_vector(stage['joint_target_minus_actual_rad']['final'], 180.0 / math.pi)} "
            f"tau_max={_format_vector(stage['tau_command_abs_max_nm'], 1.0)}",
            flush=True,
        )


def main() -> int:
    args = _build_parser().parse_args()
    if not args.no_robot and not args.confirm_motion:
        raise SystemExit("Refusing real motion without --confirm-motion")
    values = np.asarray(
        (
            args.offset_m,
            args.step_m,
            args.settle_s,
            args.sample_window_s,
            args.max_rotation_error_deg,
            args.fast_offset_m,
            args.fast_speed_m_s,
            args.fast_hold_s,
        ),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise SystemExit("all motion, timing, and rotation limit values must be positive")
    if args.offset_m >= HARD_HOME_BOUND_M:
        raise SystemExit(f"--offset-m must be less than {HARD_HOME_BOUND_M:.2f}")
    if args.step_m > args.offset_m:
        raise SystemExit("--step-m must not exceed --offset-m")
    if args.fast_offset_m > FAST_TARGET_BOUND_M:
        raise SystemExit(
            f"--fast-offset-m must not exceed {FAST_TARGET_BOUND_M:.2f}; "
            "the remaining 2 cm is a mandatory safety margin"
        )
    if args.fast_speed_m_s > FAST_COMMAND_SPEED_LIMIT_M_S:
        raise SystemExit(
            f"--fast-speed-m-s must not exceed {FAST_COMMAND_SPEED_LIMIT_M_S:.2f}"
        )
    fast_step_m = float(args.fast_speed_m_s) * POLICY_DT_S
    if fast_step_m > args.fast_offset_m:
        raise SystemExit("the 10 Hz fast step must not exceed --fast-offset-m")

    config = ExperimentConfig(
        offset_m=float(args.offset_m),
        step_m=float(args.step_m),
        settle_s=float(args.settle_s),
        sample_window_s=float(args.sample_window_s),
        max_rotation_error_rad=math.radians(float(args.max_rotation_error_deg)),
    )
    fast_config = FastExperimentConfig(
        offset_m=float(args.fast_offset_m),
        speed_m_s=float(args.fast_speed_m_s),
        hold_s=float(args.fast_hold_s),
    )
    report: dict[str, object] = {
        "status": "running",
        "hard_home_bound_m": HARD_HOME_BOUND_M,
        "config": {
            "offset_m": config.offset_m,
            "step_m": config.step_m,
            "settle_s": config.settle_s,
            "sample_window_s": config.sample_window_s,
            "max_rotation_error_deg": float(args.max_rotation_error_deg),
            "experiment": args.experiment,
            "fast_offset_m": fast_config.offset_m,
            "fast_speed_m_s": fast_config.speed_m_s,
            "fast_hold_s": fast_config.hold_s,
            "reference": args.reference,
            "joint_stiffness": list(args.joint_stiffness),
            "joint_damping": list(args.joint_damping),
            **joint_pid_kwargs(args),
        },
        "stages": [],
        "motion_stages": [],
        "recovery_stages": [],
    }
    exit_code = 0
    env: FrankaEnv | None = None
    try:
        planner = CartesianActionPlanner(PlannerConfig(mode="ipopt"))
        env = FrankaEnv(
            robot_ip=args.ip,
            home_q=np.asarray(args.home_q, dtype=np.float64),
            reset_duration=float(args.reset_duration),
            max_translation_velocity=(
                fast_config.speed_m_s
                if args.experiment in {"fast", "both"}
                else config.step_m / POLICY_DT_S
            ),
            reference_name=args.reference,
            action_planner=planner,
            tracker_mode="pid",
            joint_stiffness=np.asarray(args.joint_stiffness, dtype=np.float64),
            joint_damping=np.asarray(args.joint_damping, dtype=np.float64),
            **joint_pid_kwargs(args),
            no_robot=bool(args.no_robot),
            no_cameras=True,
            use_gripper=False,
            print_events=False,
            auto_record=False,
            print_timing_summary=False,
        )
        env.reset()
        measured_home_q = env.get_joint_positions()
        if np.max(np.abs(measured_home_q - np.asarray(args.home_q, dtype=np.float64))) > 0.05:
            raise SafetyBoundaryError("robot did not reach the requested home joint pose")
        planner.reset(measured_home_q)
        home_planned_position = PandaKinematics().evaluate(
            measured_home_q,
            include_manipulability=False,
            include_link_points=False,
        ).position
        home_actual_state = env.get_robot_state_vector()
        home_actual_position = home_actual_state[:3].copy()
        home_actual_rotation = home_actual_state[3:6].copy()
        report["measured_home_q_rad"] = measured_home_q.tolist()
        report["home_actual_xyz_m"] = home_actual_position.tolist()
        report["home_planned_xyz_m"] = home_planned_position.tolist()
        zero_command = _plan_and_enqueue(
            env,
            planner,
            np.zeros(3, dtype=np.float64),
            home_planned_position,
            "home_initial",
        )
        assert zero_command.nominal_pose is not None
        assert zero_command.joint_target is not None
        home_planned_position = zero_command.nominal_pose[:3, 3].copy()
        env.clear_trace()
        env.start_control_loop(reference_name=args.reference, print_events=False)

        current_offset = np.zeros(3, dtype=np.float64)
        last_command = zero_command
        if args.experiment in {"static", "both"}:
            for stage, target_offset in _waypoints(config.offset_m):
                current_offset, moved_command = _move_to_offset(
                    env,
                    planner,
                    current_offset,
                    target_offset,
                    home_planned_position,
                    home_actual_position,
                    home_actual_rotation,
                    config,
                    stage,
                )
                if moved_command is not None:
                    last_command = moved_command
                assert last_command.joint_target is not None
                stage_report = _measure_hold(
                    env,
                    last_command.joint_target,
                    home_actual_position,
                    home_actual_rotation,
                    config,
                    simulate_trace=bool(args.no_robot),
                )
                stage_report["stage"] = stage
                stage_report["target_offset_m"] = target_offset.tolist()
                report["stages"].append(stage_report)

        if args.experiment in {"fast", "both"}:
            fast_motion_config = ExperimentConfig(
                offset_m=fast_config.offset_m,
                step_m=fast_config.speed_m_s * POLICY_DT_S,
                settle_s=fast_config.hold_s,
                sample_window_s=min(config.sample_window_s, fast_config.hold_s),
                max_rotation_error_rad=config.max_rotation_error_rad,
            )
            for stage, target_offset in _fast_waypoints(fast_config.offset_m):
                start_offset = current_offset.copy()
                trace_head = env.read_trace_head()
                current_offset, moved_command = _move_to_offset(
                    env,
                    planner,
                    current_offset,
                    target_offset,
                    home_planned_position,
                    home_actual_position,
                    home_actual_rotation,
                    fast_motion_config,
                    stage,
                )
                if moved_command is None:
                    raise RuntimeError(f"fast stage {stage!r} did not issue a command")
                last_command = moved_command
                trace = env.get_trace_since(trace_head)
                _assert_motion_trace_safe(
                    trace,
                    home_planned_position,
                    home_actual_position,
                )
                motion_report = _motion_trace_report(
                    trace,
                    target_offset - start_offset,
                    home_planned_position,
                    home_actual_position,
                )
                motion_report["stage"] = stage
                motion_report["start_offset_m"] = start_offset.tolist()
                motion_report["target_offset_m"] = target_offset.tolist()
                motion_report["command_speed_m_s"] = fast_config.speed_m_s
                report["motion_stages"].append(motion_report)

                if stage.endswith("return_home"):
                    assert last_command.joint_target is not None
                    recovery_report = _measure_hold(
                        env,
                        last_command.joint_target,
                        home_actual_position,
                        home_actual_rotation,
                        fast_motion_config,
                        simulate_trace=bool(args.no_robot),
                    )
                    recovery_report["stage"] = stage.replace("fast_", "").replace(
                        "_return_home", "_home_hold"
                    )
                    recovery_report["target_offset_m"] = target_offset.tolist()
                    report["recovery_stages"].append(recovery_report)
        report["status"] = "ok"
    except KeyboardInterrupt:
        exit_code = 130
        report["status"] = "interrupted"
        report["error_type"] = "KeyboardInterrupt"
        report["error"] = "diagnostic interrupted by operator"
    except Exception as error:
        exit_code = 1
        report["status"] = "error"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    finally:
        if env is not None:
            try:
                env.request_stop()
                env.stop()
            except Exception as stop_error:
                exit_code = 1
                report["status"] = "error"
                report["stop_error_type"] = type(stop_error).__name__
                report["stop_error"] = str(stop_error)

    output_path = args.output
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = ROOT / "logs" / f"ipopt_static_tracking_{timestamp}.json"
    output_path = pathlib.Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as output_error:
        exit_code = 1
        report["status"] = "error"
        report["error_type"] = type(output_error).__name__
        report["error"] = f"could not write report: {output_error}"
    _print_summary(report, output_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

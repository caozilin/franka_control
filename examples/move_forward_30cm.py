#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.franka_env import FrankaEnv  # noqa: E402
from control.cli_args import parse_bool, parse_joint_vector  # noqa: E402
from control.pid_config import add_joint_pid_arguments, joint_pid_kwargs  # noqa: E402
from planning import (  # noqa: E402
    CartesianActionPlanner,
    PLANNER_MODE_CHOICES,
    TRACKER_MODE_CHOICES,
    PlannerConfig,
)
from utils import POLICY_HZ  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Move Franka +X 30 cm via FrankaEnv 10Hz actions and C++ minjerk backend.")
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--distance", type=float, default=0.30)
    parser.add_argument("--step", type=float, default=0.01, help="meters per 10Hz action tick")
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--connect-only", action="store_true", help="Construct FrankaEnv and exit without starting control.")
    parser.add_argument("--planner-mode", choices=PLANNER_MODE_CHOICES, default="direct")
    parser.add_argument("--tracker-mode", choices=TRACKER_MODE_CHOICES, default="auto")
    parser.add_argument("--reference", choices=("min_jerk", "linear", "cubic", "motion_limited"), default="min_jerk")
    add_joint_pid_arguments(parser)
    parser.add_argument("--rotation-ranged-axes", type=parse_bool, nargs=3, default=(False, False, False))
    parser.add_argument("--rotation-limits-deg", type=float, nargs=3, default=(30.0, 30.0, 45.0))
    parser.add_argument("--tolerance-frame-rotvec", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--nullspace-enabled", action="store_true")
    parser.add_argument("--nullspace-pinv", choices=("plain", "damped"), default="plain")
    parser.add_argument("--nullspace-projector", choices=("kinematic", "dynamic"), default="kinematic")
    parser.add_argument("--nullspace-lambda", type=float, default=0.05)
    parser.add_argument("--nullspace-stiffness", type=float, default=10.0)
    parser.add_argument("--nullspace-damping", type=float, default=2.0)
    parser.add_argument("--nullspace-q-target", type=parse_joint_vector, default=None)
    args = parser.parse_args()

    ticks = int(round(args.distance / args.step))
    actual_distance = ticks * args.step
    print(f"This will command +X {actual_distance:.3f} m as {ticks} actions at 10Hz ({args.step:.3f} m/tick).")
    print("Keep the user stop button available.")
    if not args.yes:
        input("Press Enter to start...")

    print("[stage] constructing FrankaEnv", flush=True)
    action_planner = CartesianActionPlanner(
        PlannerConfig(
            mode=args.planner_mode,
            rotation_ranged_axes=tuple(args.rotation_ranged_axes),
            rotation_limits_deg=tuple(args.rotation_limits_deg),
            tolerance_frame_rotvec=tuple(args.tolerance_frame_rotvec),
            shadow_stage="move_forward_30cm",
        )
    )
    env = FrankaEnv(
        robot_ip=args.ip,
        max_translation_velocity=args.step * POLICY_HZ,
        reference_name=args.reference,
        action_planner=action_planner,
        tracker_mode=args.tracker_mode,
        **joint_pid_kwargs(args),
        no_robot=False,
        nullspace_enabled=args.nullspace_enabled,
        nullspace_q_target=args.nullspace_q_target,
        nullspace_stiffness=args.nullspace_stiffness,
        nullspace_damping=args.nullspace_damping,
        nullspace_pinv=args.nullspace_pinv,
        nullspace_projector=args.nullspace_projector,
        nullspace_lambda=args.nullspace_lambda,
    )
    print("[stage] FrankaEnv constructed", flush=True)
    if args.connect_only:
        env.stop()
        print("[stage] connect-only done", flush=True)
        return 0
    try:
        print("[stage] starting C++ realtime control thread", flush=True)
        # C++ consumes actions at 10Hz and runs 1kHz min-jerk Cartesian impedance torque control.
        max_duration = max(ticks * 0.1 + args.settle, abs(actual_distance) / 0.05 + args.settle)
        env.start_control_loop(max_duration=max_duration)
        print("[stage] C++ realtime control thread started", flush=True)
        next_time = time.monotonic()
        for _ in range(ticks):
            env.enqueue_cartesian_action(
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64),
                semantic_key="move_forward_30cm",
            )
            next_time += 0.1
            time.sleep(max(0.0, next_time - time.monotonic()))
        print("[stage] waiting for C++ control thread", flush=True)
        env.wait_control_loop()
        print("[stage] C++ control thread finished", flush=True)
    finally:
        env.stop()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

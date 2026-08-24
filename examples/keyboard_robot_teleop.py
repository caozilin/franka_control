from __future__ import annotations

import argparse
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_collection.key_control import KeyboardController  # noqa: E402
from control.cli_args import parse_bool, parse_joint_vector  # noqa: E402
from control.pid_config import add_joint_pid_arguments, joint_pid_kwargs  # noqa: E402
from planning import PLANNER_MODE_CHOICES, TRACKER_MODE_CHOICES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation through FrankaEnv and the C++ backend.")
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--no-home-first", action="store_true")
    parser.add_argument("--max-translation-velocity", type=float, default=0.1)
    parser.add_argument("--max-rotation-velocity", type=float, default=math.pi / 4.0)
    parser.add_argument("--reference", choices=("min_jerk", "linear", "cubic", "motion_limited"), default="linear")
    parser.add_argument("--planner-mode", choices=PLANNER_MODE_CHOICES, default="direct")
    parser.add_argument("--tracker-mode", choices=TRACKER_MODE_CHOICES, default="auto")
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

    reference = KeyboardController(
        robot_ip=args.ip,
        input_device="keyboard",
        max_translation_velocity=args.max_translation_velocity,
        max_rotation_velocity=args.max_rotation_velocity,
        reference_name=args.reference,
        planner_mode=args.planner_mode,
        tracker_mode=args.tracker_mode,
        **joint_pid_kwargs(args),
        rotation_ranged_axes=tuple(args.rotation_ranged_axes),
        rotation_limits_deg=tuple(args.rotation_limits_deg),
        tolerance_frame_rotvec=tuple(args.tolerance_frame_rotvec),
        nullspace_enabled=args.nullspace_enabled,
        nullspace_q_target=args.nullspace_q_target,
        nullspace_stiffness=args.nullspace_stiffness,
        nullspace_damping=args.nullspace_damping,
        nullspace_pinv=args.nullspace_pinv,
        nullspace_projector=args.nullspace_projector,
        nullspace_lambda=args.nullspace_lambda,
    )
    reference.run(home_first=not args.no_home_first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

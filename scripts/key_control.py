#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_collection.key_control import KeyboardController  # noqa: E402


def parse_joint_vector(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("--nullspace-q-target must contain 7 comma-separated joint values")
    return np.asarray(parts, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--input-device", choices=("keyboard", "ps4"), default="ps4")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--no-home-first", action="store_true")
    parser.add_argument("--max-translation-step", type=float, default=0.1)
    parser.add_argument("--max-rotation-step", type=float, default=math.pi / 4.0)
    parser.add_argument("--reset-duration", type=float, default=5.0)
    parser.add_argument("--reference", choices=("min_jerk", "linear", "cubic", "motion_limited"), default="linear")
    parser.add_argument("--save-recording", action="store_true", help="Save pose/timing CSVs and analysis after control exits")
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
        input_device=args.input_device,
        joystick_index=args.joystick_index,
        max_translation_step=args.max_translation_step,
        max_rotation_step=args.max_rotation_step,
        reset_duration=args.reset_duration,
        reference_name=args.reference,
        save_recording=args.save_recording,
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

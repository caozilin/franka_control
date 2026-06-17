#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


TASK_DIM_NAMES = ("x", "y", "z", "rx", "ry", "rz")
TASK_DIM_INDEX = {name: idx for idx, name in enumerate(TASK_DIM_NAMES)}

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_collection.key_control import (
    DEFAULT_MAX_ROTATION_GOAL_ERROR,
    DEFAULT_MAX_ROTATION_VELOCITY,
    DEFAULT_MAX_TRANSLATION_GOAL_ERROR,
    DEFAULT_MAX_TRANSLATION_VELOCITY,
    DEFAULT_MAX_TORQUE_RATE,
    KeyboardController,
)  # noqa: E402


def parse_joint_vector(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("--nullspace-q-target must contain 7 comma-separated joint values")
    return np.asarray(parts, dtype=np.float64)


def parse_release_task_dims(value: str) -> np.ndarray:
    mask = np.ones(6, dtype=np.float64)
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--release-task-dims requires at least one task dimension")
    for part in parts:
        if part in TASK_DIM_INDEX:
            mask[TASK_DIM_INDEX[part]] = 0.0
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--release-task-dims entries must be one of x,y,z,rx,ry,rz or indices 0-5"
            ) from exc
        if index < 0 or index >= 6:
            raise argparse.ArgumentTypeError("--release-task-dims indices must be in [0, 5]")
        mask[index] = 0.0
    if not np.any(mask):
        raise argparse.ArgumentTypeError("--release-task-dims cannot release all 6 task dimensions")
    return mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--input-device", choices=("keyboard", "ps4"), default="ps4")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--no-home-first", action="store_true")
    parser.add_argument("--max-translation-velocity", "--max-translation-step", dest="max_translation_velocity", type=float, default=DEFAULT_MAX_TRANSLATION_VELOCITY)
    parser.add_argument("--max-rotation-velocity", "--max-rotation-step", dest="max_rotation_velocity", type=float, default=DEFAULT_MAX_ROTATION_VELOCITY)
    parser.add_argument("--max-translation-goal-error", type=float, default=DEFAULT_MAX_TRANSLATION_GOAL_ERROR)
    parser.add_argument("--max-rotation-goal-error", type=float, default=DEFAULT_MAX_ROTATION_GOAL_ERROR)
    parser.add_argument("--motion-limited-max-translation-velocity", type=float, default=None)
    parser.add_argument("--motion-limited-max-rotation-velocity", type=float, default=None)
    parser.add_argument("--motion-limited-max-translation-acceleration", type=float, default=None)
    parser.add_argument("--motion-limited-max-rotation-acceleration", type=float, default=None)
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
    parser.add_argument("--release-task-dims", type=parse_release_task_dims, default=None, help="Comma-separated task dimensions to release from {x,y,z,rx,ry,rz}; released axes are excluded from task torque and nullspace projection")
    args = parser.parse_args()

    reference = KeyboardController(
        robot_ip=args.ip,
        input_device=args.input_device,
        joystick_index=args.joystick_index,
        max_translation_velocity=args.max_translation_velocity,
        max_rotation_velocity=args.max_rotation_velocity,
        max_translation_goal_error=args.max_translation_goal_error,
        max_rotation_goal_error=args.max_rotation_goal_error,
        motion_limited_max_translation_velocity=args.motion_limited_max_translation_velocity,
        motion_limited_max_rotation_velocity=args.motion_limited_max_rotation_velocity,
        motion_limited_max_translation_acceleration=args.motion_limited_max_translation_acceleration,
        motion_limited_max_rotation_acceleration=args.motion_limited_max_rotation_acceleration,
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
        task_constraint_mask=args.release_task_dims,
    )
    reference.run(home_first=not args.no_home_first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

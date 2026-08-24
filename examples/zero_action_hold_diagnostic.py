from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.cli_args import parse_joint_vector  # noqa: E402
from control.franka_env import FrankaEnv, ROBOT_IP  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal zero-action Franka torque-control hold.")
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--reference", default="min_jerk", choices=("min_jerk", "linear", "cubic", "motion_limited"))
    parser.add_argument("--no-home-first", action="store_true")
    parser.add_argument("--with-cameras", action="store_true")
    parser.add_argument("--with-gripper", action="store_true")
    parser.add_argument("--nullspace-enabled", action="store_true")
    parser.add_argument("--nullspace-pinv", choices=("plain", "damped"), default="plain")
    parser.add_argument("--nullspace-projector", choices=("kinematic", "dynamic"), default="kinematic")
    parser.add_argument("--nullspace-lambda", type=float, default=0.05)
    parser.add_argument("--nullspace-stiffness", type=float, default=10.0)
    parser.add_argument("--nullspace-damping", type=float, default=2.0)
    parser.add_argument("--nullspace-q-target", type=parse_joint_vector, default=None)
    args = parser.parse_args()

    env = FrankaEnv(
        robot_ip=args.ip,
        no_cameras=not args.with_cameras,
        use_gripper=args.with_gripper,
        reference_name=args.reference,
        nullspace_enabled=args.nullspace_enabled,
        nullspace_q_target=args.nullspace_q_target,
        nullspace_stiffness=args.nullspace_stiffness,
        nullspace_damping=args.nullspace_damping,
        nullspace_pinv=args.nullspace_pinv,
        nullspace_projector=args.nullspace_projector,
        nullspace_lambda=args.nullspace_lambda,
    )
    run_error = None
    try:
        if not args.no_home_first:
            print("Resetting robot to home pose...")
            env.reset()
        print(
            f"Starting zero-action hold: duration={args.duration:.3f}s "
            f"reference={args.reference} cameras={args.with_cameras} gripper={args.with_gripper}",
            flush=True,
        )
        env.run_action_loop(
            max_duration=args.duration,
            print_events=False,
            reference_name=args.reference,
        )
        print("Zero-action hold finished without control exception.", flush=True)
    except Exception as exc:
        run_error = exc
    finally:
        env.stop()

    if run_error is not None:
        raise run_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

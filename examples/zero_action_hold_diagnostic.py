from __future__ import annotations

import argparse
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.franka_env import FrankaEnv, ROBOT_IP  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal zero-action Franka torque-control hold.")
    parser.add_argument("--ip", default=ROBOT_IP)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--controller", default="min_jerk", choices=("min_jerk", "linear", "cubic"))
    parser.add_argument("--no-home-first", action="store_true")
    parser.add_argument("--with-cameras", action="store_true")
    parser.add_argument("--with-gripper", action="store_true")
    args = parser.parse_args()

    env = FrankaEnv(
        robot_ip=args.ip,
        no_cameras=not args.with_cameras,
        use_gripper=args.with_gripper,
        controller_name=args.controller,
    )
    run_error = None
    try:
        if not args.no_home_first:
            print("Resetting robot to home pose...")
            env.reset()
        print(
            f"Starting zero-action hold: duration={args.duration:.3f}s "
            f"controller={args.controller} cameras={args.with_cameras} gripper={args.with_gripper}",
            flush=True,
        )
        env.run_action_loop(
            max_duration=args.duration,
            print_events=False,
            controller_name=args.controller,
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

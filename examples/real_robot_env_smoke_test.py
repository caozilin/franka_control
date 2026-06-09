from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.franka_env import FrankaEnv, ROBOT_IP  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Connect to the robot and read the current end-effector state.")
    parser.add_argument("--ip", default=ROBOT_IP)
    args = parser.parse_args()

    env = FrankaEnv(robot_ip=args.ip)
    try:
        print("state:", env.get_robot_state_vector())
    finally:
        env.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

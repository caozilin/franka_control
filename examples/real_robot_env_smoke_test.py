from __future__ import annotations

import argparse
import time

from franka_control import FrankaEnv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    args = parser.parse_args()

    env = FrankaEnv(robot_ip=args.ip)
    env.start_readonly()
    try:
        print("state:", env.get_robot_state_vector())
        print("joints:", env.get_joint_state_vector())
        obs, _, _ = env.get_observation("test")
        print("observation/state shape:", obs["observation/state"].shape)
        print("observation/image shape:", obs["observation/image"].shape)
        time.sleep(0.5)
    finally:
        env.stop_control()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

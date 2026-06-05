from __future__ import annotations

import argparse
import time

from franka_control import FrankaEnv, TeleopController


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--home-first", action="store_true")
    parser.add_argument("--rotation-frame", choices=["eef", "base"], default="eef")
    args = parser.parse_args()

    env = FrankaEnv(robot_ip=args.ip)
    teleop = TeleopController(
        env=env,
        input_device="keyboard",
        rotation_input_frame=args.rotation_frame,
    )
    listener = teleop.create_keyboard_listener()

    print("keyboard robot teleop")
    print("W/S:X  A/D:Y  I/K:Z  Q/E:Roll  U/O:Pitch  J/L:Yaw")
    print(f"rotation frame: {args.rotation_frame}")
    print("G/H: gripper  +/-: speed  R: reset home  ESC: exit")
    print("keep the user stop button available before enabling motion")

    teleop.start(home_first=args.home_first)
    listener.start()
    try:
        while teleop.running:
            state, action = teleop.get_state_and_action()
            print("state:", state, "action:", action)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        teleop.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

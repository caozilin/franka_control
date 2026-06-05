from __future__ import annotations

import time

from franka_control import FrankaEnv, TeleopController


def main() -> int:
    env = FrankaEnv(no_robot=True, no_cameras=True)
    teleop = TeleopController(env=env, input_device="ps4")

    print("PS4 no-robot smoke test")
    print("Use OPTIONS to exit, or Ctrl-C in terminal")

    teleop.start(home_first=False)
    try:
        while teleop.running:
            state, action = teleop.get_state_and_action()
            print("state:", state, "action:", action)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

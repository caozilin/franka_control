from __future__ import annotations

import time

from franka_control import FrankaEnv, TeleopController


def main() -> int:
    env = FrankaEnv(no_robot=True, no_cameras=True)
    teleop = TeleopController(env=env, input_device="keyboard")
    listener = teleop.create_keyboard_listener()

    print("keyboard no-robot smoke test")
    print("W/S A/D I/K move, Q/E U/O J/L rotate, G/H gripper, +/- speed, R reset, ESC exit")

    teleop.start(home_first=False)
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

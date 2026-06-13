from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.franka_env import FrankaEnv  # noqa: E402


def main() -> int:
    env = FrankaEnv(no_robot=True, no_cameras=True, reference_name="linear")
    try:
        print("keyboard no-robot smoke: enqueueing a zero action")
        env.enqueue_action_block(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64))
        env.start_control_loop(max_duration=0.2)
        env.wait_control_loop()
        print("state:", env.get_robot_state_vector())
    finally:
        env.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_collection.key_control import KeyboardController  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Keyboard teleoperation through FrankaEnv and the C++ backend.")
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--no-home-first", action="store_true")
    parser.add_argument("--max-translation-step", type=float, default=0.1)
    parser.add_argument("--max-rotation-step", type=float, default=math.pi / 4.0)
    parser.add_argument("--controller", choices=("min_jerk", "linear", "cubic"), default="linear")
    args = parser.parse_args()

    controller = KeyboardController(
        robot_ip=args.ip,
        input_device="keyboard",
        max_translation_step=args.max_translation_step,
        max_rotation_step=args.max_rotation_step,
        controller_name=args.controller,
    )
    controller.run(home_first=not args.no_home_first)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

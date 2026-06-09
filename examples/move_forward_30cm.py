#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.franka_env import FrankaEnv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Move Franka +X 30 cm via FrankaEnv 10Hz actions and C++ minjerk backend.")
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--distance", type=float, default=0.30)
    parser.add_argument("--step", type=float, default=0.01, help="meters per 10Hz action tick")
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--connect-only", action="store_true", help="Construct FrankaEnv and exit without starting control.")
    args = parser.parse_args()

    ticks = int(round(args.distance / args.step))
    actual_distance = ticks * args.step
    print(f"This will command +X {actual_distance:.3f} m as {ticks} actions at 10Hz ({args.step:.3f} m/tick).")
    print("Keep the user stop button available.")
    if not args.yes:
        input("Press Enter to start...")

    print("[stage] constructing FrankaEnv", flush=True)
    env = FrankaEnv(robot_ip=args.ip, max_translation_step=args.step, no_robot=False)
    print("[stage] FrankaEnv constructed", flush=True)
    if args.connect_only:
        env.stop()
        print("[stage] connect-only done", flush=True)
        return 0
    try:
        print("[stage] starting C++ realtime control thread", flush=True)
        # C++ consumes actions at 10Hz and runs 1kHz min-jerk Cartesian impedance torque control.
        max_duration = max(ticks * 0.1 + args.settle, abs(actual_distance) / 0.05 + args.settle)
        env.start_control_loop(max_duration=max_duration)
        print("[stage] C++ realtime control thread started", flush=True)
        next_time = time.monotonic()
        for _ in range(ticks):
            env.enqueue_action_block(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float64))
            next_time += 0.1
            time.sleep(max(0.0, next_time - time.monotonic()))
        print("[stage] waiting for C++ control thread", flush=True)
        env.wait_control_loop()
        print("[stage] C++ control thread finished", flush=True)
    finally:
        env.stop()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

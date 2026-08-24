#!/usr/bin/env python3
"""Compatibility wrapper for the unified Cartesian teleoperation entry point."""
from __future__ import annotations

import argparse
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="172.16.0.2")
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument(
        "--control-cpu",
        type=int,
        default=2,
        help="CPU used by the Franka control process; keep it separate from the enp3s0 IRQ (currently CPU 6)",
    )
    args = parser.parse_args()
    if args.translation_scale <= 0.0 or args.rotation_scale <= 0.0:
        parser.error("translation and rotation scales must be positive")
    teleop_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "teleop.py"),
        "--input-device",
        "pico",
        "--ip",
        args.robot_ip,
        "--pico-translation-scale",
        str(args.translation_scale),
        "--pico-rotation-scale",
        str(args.rotation_scale),
        "--control-cpu",
        str(args.control_cpu),
    ]
    os.execv(sys.executable, teleop_cmd)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

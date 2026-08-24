#!/usr/bin/env python3
"""Unified Cartesian teleoperation and optional data-collection entry point."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_collection.data_recorder import main as run_teleop  # noqa: E402


def _option_value(name: str, default: str) -> str:
    prefix = f"{name}="
    for index, value in enumerate(sys.argv[1:]):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == name and index + 2 <= len(sys.argv[1:]):
            return sys.argv[1:][index + 1]
    return default


def main() -> int:
    input_device = _option_value("--input-device", "keyboard")
    bridge: subprocess.Popen | None = None
    if input_device == "pico":
        pico_port = _option_value("--pico-port", "9010")
        bridge = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "xrobotoolkit_bridge.py"),
                "--host",
                "127.0.0.1",
                "--port",
                pico_port,
            ],
            cwd=ROOT,
        )
        time.sleep(0.75)
        if bridge.poll() is not None:
            return int(bridge.returncode or 1)

    try:
        return run_teleop()
    except KeyboardInterrupt:
        return 130
    finally:
        if bridge is not None and bridge.poll() is None:
            bridge.terminate()
            try:
                bridge.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                bridge.kill()
                bridge.wait()


if __name__ == "__main__":
    raise SystemExit(main())

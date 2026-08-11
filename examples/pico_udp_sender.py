"""Send synthetic dual-controller packets for local PICO pipeline tests."""
from __future__ import annotations

import argparse
import math
import pathlib
import socket
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from devices.pico import PicoControllerState, PicoPacket  # noqa: E402


def controller(position: tuple[float, float, float], grip: float) -> PicoControllerState:
    return PicoControllerState(
        position=np.asarray(position, dtype=np.float64),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        grip=grip,
        trigger=0.0,
        thumbstick=np.zeros(2),
        primary=False,
        secondary=False,
        tracked=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9010)
    parser.add_argument("--rate", type=float, default=60.0)
    parser.add_argument("--enable-motion", action="store_true")
    args = parser.parse_args()

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.monotonic()
    sequence = 0
    try:
        while True:
            elapsed = time.monotonic() - start
            grip = 1.0 if args.enable_motion else 0.0
            packet = PicoPacket(
                sequence,
                time.time(),
                controller((-0.2 + 0.05 * math.sin(elapsed), 1.2, 0.3), grip),
                controller((0.2, 1.2, 0.3), grip),
                session_id="synthetic",
            )
            sender.sendto(packet.to_json(), (args.host, args.port))
            sequence += 1
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Bridge XRoboToolkit PC Service controller data to the Pico UDP protocol."""
from __future__ import annotations

import argparse
import pathlib
import socket
import sys
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from devices.pico.xrobotoolkit import packet_from_xrobotoolkit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Coordinator UDP host")
    parser.add_argument("--port", type=int, default=9010, help="Coordinator UDP port")
    parser.add_argument("--poll-rate", type=float, default=90.0, help="SDK polling rate in Hz")
    args = parser.parse_args()
    if args.poll_rate <= 0.0:
        parser.error("--poll-rate must be positive")

    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise SystemExit(
            "xrobotoolkit_sdk is not installed in this Python environment"
        ) from exc

    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sequence = 0
    last_timestamp_ns = 0
    period_s = 1.0 / args.poll_rate
    xrt.init()
    print(f"XRoboToolkit bridge sending to {args.host}:{args.port}", flush=True)
    try:
        while True:
            loop_start = time.monotonic()
            timestamp_ns = int(xrt.get_time_stamp_ns())
            if timestamp_ns > 0 and timestamp_ns != last_timestamp_ns:
                packet = packet_from_xrobotoolkit(xrt, sequence, timestamp_ns)
                if packet.left.tracked and packet.right.tracked:
                    sender.sendto(packet.to_json(), (args.host, args.port))
                    sequence += 1
                    last_timestamp_ns = timestamp_ns

            remaining = period_s - (time.monotonic() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        sender.close()
        xrt.close()


if __name__ == "__main__":
    main()

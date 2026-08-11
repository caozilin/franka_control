from __future__ import annotations

import pathlib
import socket
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from devices.pico import PicoControllerState, PicoPacket, PicoUdpReceiver  # noqa: E402


def _controller(x: float) -> PicoControllerState:
    return PicoControllerState(
        position=np.array([x, 0.2, -0.3]),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        grip=0.8,
        trigger=0.4,
        thumbstick=np.array([0.1, -0.2]),
        primary=True,
        secondary=False,
        tracked=True,
    )


def test_pico_packet_json_roundtrip() -> None:
    packet = PicoPacket(12, 3.5, _controller(0.1), _controller(-0.1), session_id="run-a")
    decoded = PicoPacket.from_json(packet.to_json())

    assert decoded.sequence == 12
    assert decoded.session_id == "run-a"
    np.testing.assert_allclose(decoded.left.position, [0.1, 0.2, -0.3])
    np.testing.assert_allclose(decoded.right.thumbstick, [0.1, -0.2])


def test_udp_receiver_keeps_latest_sequence() -> None:
    receiver = PicoUdpReceiver("127.0.0.1", 0)
    receiver.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for sequence in (2, 1, 3):
            packet = PicoPacket(sequence, float(sequence), _controller(0.0), _controller(float(sequence)))
            sender.sendto(packet.to_json(), ("127.0.0.1", receiver.port))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            snapshot = receiver.latest()
            if snapshot is not None and snapshot.packet.sequence == 3:
                break
            time.sleep(0.01)
        assert snapshot is not None
        assert snapshot.packet.sequence == 3
        np.testing.assert_allclose(snapshot.packet.right.position[0], 3.0)
    finally:
        sender.close()
        receiver.stop()

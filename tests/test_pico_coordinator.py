from __future__ import annotations

import pathlib
import socket
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from devices.pico import PicoControllerState, PicoPacket  # noqa: E402
from scripts.coordinator import Args, Coordinator  # noqa: E402


def _controller(position: tuple[float, float, float], grip: float = 1.0) -> PicoControllerState:
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


def _send(sender: socket.socket, port: int, sequence: int, left_x: float) -> None:
    packet = PicoPacket(
        sequence,
        time.time(),
        _controller((left_x, 0.0, 0.0)),
        _controller((0.0, 0.0, 0.0)),
        session_id="coordinator-test",
    )
    sender.sendto(packet.to_json(), ("127.0.0.1", port))


def _wait_for_sequence(coordinator: Coordinator, sequence: int) -> None:
    assert coordinator._pico_receiver is not None
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        snapshot = coordinator._pico_receiver.latest()
        if snapshot is not None and snapshot.packet.sequence == sequence:
            return
        time.sleep(0.01)
    raise AssertionError(f"PICO sequence {sequence} was not received")


def test_pico_source_runs_without_policy_server_or_robot() -> None:
    coordinator = Coordinator(
        Args(
            action_source="pico",
            pico_bind_host="127.0.0.1",
            pico_port=0,
            no_robot=True,
            no_cameras=True,
            use_gripper=False,
            startup_home=False,
        )
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        assert coordinator._client is None
        assert coordinator._pico_receiver is not None
        assert coordinator._args.pico_mapping_mode == "split"
        coordinator._run_action_step({})

        _send(sender, coordinator._pico_receiver.port, 1, 0.0)
        _wait_for_sequence(coordinator, 1)
        coordinator._run_action_step({})
        assert coordinator._latest_pico is not None
        assert coordinator._latest_pico["reanchored"] is True

        _send(sender, coordinator._pico_receiver.port, 2, 0.04)
        _wait_for_sequence(coordinator, 2)
        coordinator._run_action_step({})

        assert coordinator._latest_pico is not None
        assert coordinator._latest_pico["motion_enabled"] is True
        expected_x = min(0.02, coordinator._env.max_translation_step)
        np.testing.assert_allclose(coordinator._latest_action_transformed[:3], [expected_x, 0.0, 0.0])
    finally:
        sender.close()
        coordinator.close()

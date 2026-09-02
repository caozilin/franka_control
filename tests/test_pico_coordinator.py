from __future__ import annotations

import pathlib
import math
import socket
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from devices.pico import PicoControllerState, PicoPacket  # noqa: E402
from scripts.coordinator import Args, Coordinator, _rotation_action_ee_to_base  # noqa: E402


def _controller(
    position: tuple[float, float, float],
    grip: float = 1.0,
    orientation_xyzw: np.ndarray | None = None,
) -> PicoControllerState:
    return PicoControllerState(
        position=np.asarray(position, dtype=np.float64),
        orientation_xyzw=(
            np.array([0.0, 0.0, 0.0, 1.0])
            if orientation_xyzw is None
            else np.asarray(orientation_xyzw, dtype=np.float64)
        ),
        grip=grip,
        trigger=0.0,
        thumbstick=np.zeros(2),
        primary=False,
        secondary=False,
        tracked=True,
    )


def _send(sender: socket.socket, port: int, sequence: int, left_offset: float) -> None:
    packet = PicoPacket(
        sequence,
        time.time(),
        _controller((left_offset, 0.0, 0.0)),
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

        _send(sender, coordinator._pico_receiver.port, 2, 0.2)
        _wait_for_sequence(coordinator, 2)
        coordinator._run_action_step({})

        assert coordinator._latest_pico is not None
        assert coordinator._latest_pico["motion_enabled"] is True
        np.testing.assert_allclose(
            coordinator._latest_action_transformed[:3],
            [0.0, coordinator._env.max_translation_step, 0.0],
        )
    finally:
        sender.close()
        coordinator.close()


def test_end_effector_rotation_increment_is_converted_to_base() -> None:
    action = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0, -1.0])
    robot_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2.0])

    converted = _rotation_action_ee_to_base(action, robot_state)

    np.testing.assert_allclose(converted[:3], 0.0)
    np.testing.assert_allclose(converted[3:6], [0.0, 0.1, 0.0], atol=1e-12)
    assert converted[6] == -1.0

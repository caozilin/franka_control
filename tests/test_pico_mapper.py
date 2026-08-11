from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from devices.pico import (  # noqa: E402
    PicoControllerState,
    PicoMapperConfig,
    PicoPacket,
    PicoPoseMapper,
    PicoSnapshot,
)


def _quaternion_y(angle: float) -> np.ndarray:
    return np.array([0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle)])


def _snapshot(
    *,
    sequence: int = 0,
    right_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    right_orientation: np.ndarray | None = None,
    left_grip: float = 1.0,
    right_grip: float = 1.0,
    trigger: float = 0.0,
    tracked: bool = True,
    received: float = 10.0,
) -> PicoSnapshot:
    def controller(position, orientation, grip, controller_trigger=0.0):
        return PicoControllerState(
            np.asarray(position, dtype=np.float64),
            np.asarray(orientation, dtype=np.float64),
            grip,
            controller_trigger,
            np.zeros(2),
            False,
            False,
            tracked,
        )

    identity = np.array([0.0, 0.0, 0.0, 1.0])
    packet = PicoPacket(
        sequence,
        sequence / 60.0,
        controller((0.0, 0.0, 0.0), identity, left_grip),
        controller(right_position, identity if right_orientation is None else right_orientation, right_grip, trigger),
    )
    return PicoSnapshot(packet, received)


def test_dual_grip_reanchors_before_emitting_motion() -> None:
    mapper = PicoPoseMapper()
    first = mapper.step(_snapshot(sequence=1), now=10.0)
    moved = mapper.step(_snapshot(sequence=2, right_position=(0.02, 0.0, 0.0)), now=10.0)

    assert first is not None and first.reanchored
    np.testing.assert_allclose(first.action[:6], 0.0)
    assert moved is not None and moved.motion_enabled
    np.testing.assert_allclose(moved.action[:3], [0.01, 0.0, 0.0])


def test_mapper_clamps_without_losing_remaining_displacement() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(translation_scale=1.0, max_translation_step_m=0.05))
    mapper.step(_snapshot(sequence=1), now=10.0)
    moved = _snapshot(sequence=2, right_position=(0.2, 0.0, 0.0))
    steps = [mapper.step(moved, now=10.0).action[0] for _ in range(4)]  # type: ignore[union-attr]
    np.testing.assert_allclose(steps, [0.05, 0.05, 0.05, 0.05])


def test_unity_handedness_is_applied_to_rotation() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(max_rotation_step_rad=1.0))
    mapper.step(_snapshot(sequence=1), now=10.0)
    command = mapper.step(
        _snapshot(sequence=2, right_orientation=_quaternion_y(0.2)),
        now=10.0,
    )
    assert command is not None
    np.testing.assert_allclose(command.action[3:6], [0.0, -0.2, 0.0], atol=1e-10)


def test_release_reanchors_and_trigger_controls_gripper() -> None:
    mapper = PicoPoseMapper()
    mapper.step(_snapshot(sequence=1), now=10.0)
    released = mapper.step(_snapshot(sequence=2, left_grip=0.0, trigger=1.0), now=10.0)
    reanchored = mapper.step(
        _snapshot(sequence=3, right_position=(0.3, 0.0, 0.0), trigger=1.0),
        now=10.0,
    )

    assert released is not None and not released.motion_enabled
    assert released.action[6] == 1.0
    assert reanchored is not None and reanchored.reanchored
    np.testing.assert_allclose(reanchored.action[:6], 0.0)


def test_stale_or_invalid_tracking_emits_no_command() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(stale_timeout_s=0.1))
    assert mapper.step(_snapshot(received=1.0), now=1.2) is None
    assert mapper.step(_snapshot(tracked=False), now=10.0) is None

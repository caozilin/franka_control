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


def _quaternion_x(angle: float) -> np.ndarray:
    return np.array([math.sin(0.5 * angle), 0.0, 0.0, math.cos(0.5 * angle)])


def _quaternion_z(angle: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)])


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _snapshot(
    *,
    sequence: int = 0,
    left_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    right_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    left_orientation: np.ndarray | None = None,
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
        controller(left_position, identity if left_orientation is None else left_orientation, left_grip),
        controller(right_position, identity if right_orientation is None else right_orientation, right_grip, trigger),
    )
    return PicoSnapshot(packet, received)


def test_dual_grip_reanchors_before_emitting_motion() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(mapping_mode="single_6dof"))
    first = mapper.step(_snapshot(sequence=1), now=10.0)
    moved = mapper.step(_snapshot(sequence=2, right_position=(0.02, 0.0, 0.0)), now=10.0)

    assert first is not None and first.reanchored
    np.testing.assert_allclose(first.action[:6], 0.0)
    assert moved is not None and moved.motion_enabled
    np.testing.assert_allclose(moved.action[:3], [0.01, 0.0, 0.0])


def test_mapper_clamps_without_losing_remaining_displacement() -> None:
    mapper = PicoPoseMapper(
        PicoMapperConfig(mapping_mode="single_6dof", translation_scale=1.0, max_translation_step_m=0.05)
    )
    mapper.step(_snapshot(sequence=1), now=10.0)
    moved = _snapshot(sequence=2, right_position=(0.2, 0.0, 0.0))
    steps = [mapper.step(moved, now=10.0).action[0] for _ in range(4)]  # type: ignore[union-attr]
    np.testing.assert_allclose(steps, [0.05, 0.05, 0.05, 0.05])


def test_unity_handedness_is_applied_to_rotation() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(
        max_rotation_step_rad=0.2,
        attitude_deadzone_rad=0.0,
        attitude_full_scale_rad=0.2,
    ))
    mapper.step(_snapshot(sequence=1), now=10.0)
    command = mapper.step(
        _snapshot(sequence=2, right_orientation=_quaternion_y(0.2)),
        now=10.0,
    )
    assert command is not None
    np.testing.assert_allclose(command.action[3:6], [-0.2, 0.0, 0.0], atol=1e-10)


def test_rotation_delta_stays_in_local_controller_axes() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(
        mapping_mode="split",
        max_rotation_step_rad=0.2,
        attitude_deadzone_rad=0.0,
        attitude_full_scale_rad=0.2,
    ))
    anchor = _quaternion_z(math.pi / 2.0)
    rotated = _quaternion_multiply(anchor, _quaternion_x(0.2))

    mapper.step(_snapshot(sequence=1, right_orientation=anchor), now=10.0)
    command = mapper.step(_snapshot(sequence=2, right_orientation=rotated), now=10.0)

    assert command is not None
    # Unity-to-right-handed conversion reverses local X's sign.  A spatial
    # delta would appear on Y after the 90-degree anchor, so this also verifies
    # that the mapper emits body-fixed/local coordinates.
    np.testing.assert_allclose(command.action[3:6], [0.0, -0.2, 0.0], atol=1e-10)


def test_release_reanchors_and_trigger_controls_gripper() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(mapping_mode="single_6dof"))
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


def test_trigger_press_toggles_gripper_once_per_rising_edge() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(mapping_mode="split"))

    opened = mapper.step(_snapshot(sequence=1, trigger=0.0), now=10.0)
    closed = mapper.step(_snapshot(sequence=2, trigger=1.0), now=10.0)
    held = mapper.step(_snapshot(sequence=3, trigger=1.0), now=10.0)
    released = mapper.step(_snapshot(sequence=4, trigger=0.0), now=10.0)
    reopened = mapper.step(_snapshot(sequence=5, trigger=1.0), now=10.0)

    assert opened is not None and opened.action[6] == -1.0
    assert closed is not None and closed.action[6] == 1.0
    assert held is not None and held.action[6] == 1.0
    assert released is not None and released.action[6] == 1.0
    assert reopened is not None and reopened.action[6] == -1.0


def test_stale_or_invalid_tracking_emits_no_command() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(stale_timeout_s=0.1))
    assert mapper.step(_snapshot(received=1.0), now=1.2) is None
    assert mapper.step(_snapshot(tracked=False), now=10.0) is None


def test_split_maps_both_controller_attitudes_to_velocity_actions() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(
        mapping_mode="split",
        translation_scale=1.0,
        max_translation_step_m=0.02,
        max_rotation_step_rad=0.2,
        attitude_deadzone_rad=0.0,
        attitude_full_scale_rad=0.2,
    ))
    mapper.step(_snapshot(sequence=1), now=10.0)
    command = mapper.step(
        _snapshot(
            sequence=2,
            left_orientation=_quaternion_y(-0.2),
            right_orientation=_quaternion_y(0.2),
        ),
        now=10.0,
    )

    assert command is not None
    np.testing.assert_allclose(command.action[:3], [0.0, 0.02, 0.0])
    np.testing.assert_allclose(command.action[3:6], [-0.2, 0.0, 0.0], atol=1e-10)


def test_split_clutches_translation_and_rotation_independently() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(
        mapping_mode="split",
        max_translation_step_m=0.02,
        max_rotation_step_rad=0.1,
        attitude_deadzone_rad=0.0,
        attitude_full_scale_rad=0.1,
    ))
    translation_anchor = mapper.step(_snapshot(sequence=1, right_grip=0.0), now=10.0)
    translated = mapper.step(
        _snapshot(sequence=2, left_orientation=_quaternion_y(-0.1), right_grip=0.0),
        now=10.0,
    )
    rotation_anchor = mapper.step(
        _snapshot(sequence=3, left_grip=0.0, right_orientation=_quaternion_y(0.2)),
        now=10.0,
    )
    rotated = mapper.step(
        _snapshot(
            sequence=4,
            left_grip=0.0,
            right_orientation=_quaternion_y(0.3),
        ),
        now=10.0,
    )

    assert translation_anchor is not None and translation_anchor.reanchored
    assert translated is not None
    np.testing.assert_allclose(translated.action[:3], [0.0, 0.01, 0.0])
    assert rotation_anchor is not None and rotation_anchor.reanchored
    assert rotated is not None
    np.testing.assert_allclose(rotated.action[:3], 0.0)
    np.testing.assert_allclose(rotated.action[3:6], [-0.1, 0.0, 0.0], atol=1e-10)


def test_attitude_velocity_deadzone_and_linear_ramp() -> None:
    mapper = PicoPoseMapper(PicoMapperConfig(
        mapping_mode="split",
        translation_scale=1.0,
        max_translation_step_m=0.02,
        attitude_deadzone_rad=0.1,
        attitude_full_scale_rad=0.3,
    ))
    mapper.step(_snapshot(sequence=1, right_grip=0.0), now=10.0)

    inside = mapper.step(
        _snapshot(sequence=2, left_orientation=_quaternion_y(-0.05), right_grip=0.0),
        now=10.0,
    )
    halfway = mapper.step(
        _snapshot(sequence=3, left_orientation=_quaternion_y(-0.2), right_grip=0.0),
        now=10.0,
    )
    saturated = mapper.step(
        _snapshot(sequence=4, left_orientation=_quaternion_y(-0.5), right_grip=0.0),
        now=10.0,
    )

    assert inside is not None and halfway is not None and saturated is not None
    np.testing.assert_allclose(inside.action[:3], 0.0, atol=1e-12)
    np.testing.assert_allclose(halfway.action[:3], [0.0, 0.01, 0.0], atol=1e-10)
    np.testing.assert_allclose(saturated.action[:3], [0.0, 0.02, 0.0], atol=1e-10)

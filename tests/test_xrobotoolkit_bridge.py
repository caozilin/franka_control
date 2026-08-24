from __future__ import annotations

import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from devices.pico.mapper import (  # noqa: E402
    UNITY_LEFT_TO_RIGHT_HANDED,
    _quaternion_xyzw_to_matrix,
)
from devices.pico.xrobotoolkit import (  # noqa: E402
    openxr_pose_to_unity,
    packet_from_xrobotoolkit,
)
from scripts.coordinator import Args  # noqa: E402


class _FakeSdk:
    def get_left_controller_pose(self):
        return [1.0, 2.0, -3.0, 0.1, 0.2, 0.3, 0.9]

    def get_right_controller_pose(self):
        return [-1.0, 0.5, -2.0, 0.0, 0.0, 0.0, 1.0]

    def get_left_grip(self):
        return 0.25

    def get_right_grip(self):
        return 0.75

    def get_left_trigger(self):
        return 0.1

    def get_right_trigger(self):
        return 0.9

    def get_left_axis(self):
        return [-0.5, 0.5]

    def get_right_axis(self):
        return [0.25, -0.25]

    def get_X_button(self):
        return True

    def get_Y_button(self):
        return False

    def get_A_button(self):
        return False

    def get_B_button(self):
        return True


def test_openxr_pose_round_trips_through_existing_handedness_conversion() -> None:
    source = np.array([1.0, 2.0, -3.0, 0.1, 0.2, 0.3, 0.9])
    source[3:] /= np.linalg.norm(source[3:])

    unity_position, unity_quaternion = openxr_pose_to_unity(source)
    handedness = UNITY_LEFT_TO_RIGHT_HANDED
    recovered_position = handedness @ unity_position
    recovered_rotation = (
        handedness @ _quaternion_xyzw_to_matrix(unity_quaternion) @ handedness
    )

    np.testing.assert_allclose(recovered_position, source[:3])
    np.testing.assert_allclose(recovered_rotation, _quaternion_xyzw_to_matrix(source[3:]))


def test_packet_from_xrobotoolkit_maps_controls() -> None:
    packet = packet_from_xrobotoolkit(_FakeSdk(), 7, 2_000_000_000)

    assert packet.sequence == 7
    assert packet.timestamp_s == 2.0
    assert packet.session_id == "xrobotoolkit"
    assert packet.left.tracked and packet.right.tracked
    assert packet.left.grip == 0.25
    assert packet.right.trigger == 0.9
    assert packet.left.primary is True
    assert packet.right.secondary is True
    np.testing.assert_allclose(packet.left.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(packet.right.thumbstick, [0.25, -0.25])


def test_pico_defaults_match_observed_front_of_robot_directions() -> None:
    args = Args()
    rotation = np.asarray(args.pico_rotation_base_from_pico).reshape(3, 3)

    assert args.pico_translation_scale == 1.0
    assert args.pico_rotation_scale == 1.0
    # From the operator's position in front of the robot: forward/back is X,
    # right/left is Y, with signs calibrated from the real headset.
    np.testing.assert_allclose(rotation @ [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(rotation @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

from __future__ import annotations

from typing import Any

import numpy as np

from devices.pico.protocol import PicoControllerState, PicoPacket


def openxr_pose_to_unity(
    pose_xyz_xyzw: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an OpenXR right-handed pose to Unity's left-handed axes."""
    pose = np.asarray(pose_xyz_xyzw, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("XRoboToolkit pose must contain 7 finite values")

    position = pose[:3].copy()
    position[2] *= -1.0

    orientation = pose[3:].copy()
    orientation[:2] *= -1.0
    norm = float(np.linalg.norm(orientation))
    if norm < 1e-12:
        raise ValueError("XRoboToolkit pose contains an invalid quaternion")
    return position, orientation / norm


def _controller_state(
    pose: object,
    *,
    grip: float,
    trigger: float,
    thumbstick: object,
    primary: bool,
    secondary: bool,
) -> PicoControllerState:
    try:
        position, orientation = openxr_pose_to_unity(pose)
    except ValueError:
        return PicoControllerState(
            position=np.zeros(3, dtype=np.float64),
            orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            grip=float(grip),
            trigger=float(trigger),
            thumbstick=np.asarray(thumbstick, dtype=np.float64),
            primary=bool(primary),
            secondary=bool(secondary),
            tracked=False,
        )

    return PicoControllerState(
        position=position,
        orientation_xyzw=orientation,
        grip=float(grip),
        trigger=float(trigger),
        thumbstick=np.asarray(thumbstick, dtype=np.float64),
        primary=bool(primary),
        secondary=bool(secondary),
        tracked=True,
    )


def packet_from_xrobotoolkit(sdk: Any, sequence: int, timestamp_ns: int) -> PicoPacket:
    """Read one XRoboToolkit SDK snapshot and encode the existing Pico protocol."""
    left = _controller_state(
        sdk.get_left_controller_pose(),
        grip=sdk.get_left_grip(),
        trigger=sdk.get_left_trigger(),
        thumbstick=sdk.get_left_axis(),
        primary=sdk.get_X_button(),
        secondary=sdk.get_Y_button(),
    )
    right = _controller_state(
        sdk.get_right_controller_pose(),
        grip=sdk.get_right_grip(),
        trigger=sdk.get_right_trigger(),
        thumbstick=sdk.get_right_axis(),
        primary=sdk.get_A_button(),
        secondary=sdk.get_B_button(),
    )
    return PicoPacket(
        sequence=sequence,
        timestamp_s=timestamp_ns * 1e-9,
        left=left,
        right=right,
        session_id="xrobotoolkit",
    )

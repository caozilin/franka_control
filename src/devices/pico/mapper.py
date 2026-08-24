from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from devices.pico.receiver import PicoSnapshot
from utils.pose import matrix_to_rotvec, rotvec_to_matrix


UNITY_LEFT_TO_RIGHT_HANDED = np.diag([1.0, 1.0, -1.0])
DEFAULT_ROTATION_BASE_FROM_PICO = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class PicoMapperConfig:
    mapping_mode: str = "split"
    translation_scale: float = 0.5
    rotation_scale: float = 1.0
    max_translation_step_m: float = 0.02
    max_rotation_step_rad: float = np.pi / 40.0
    grip_threshold: float = 0.5
    trigger_threshold: float = 0.5
    stale_timeout_s: float = 0.15
    attitude_deadzone_rad: float = np.deg2rad(10.0)
    attitude_full_scale_rad: float = np.deg2rad(25.0)
    require_both_grips: bool = True
    rotation_base_from_pico: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))

    def __post_init__(self) -> None:
        mode = self.mapping_mode.lower().strip()
        if mode not in {"single_6dof", "split"}:
            raise ValueError("mapping_mode must be 'single_6dof' or 'split'")
        rotation = np.asarray(self.rotation_base_from_pico, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("rotation_base_from_pico must have shape (3, 3)")
        if not 0.0 <= self.attitude_deadzone_rad < self.attitude_full_scale_rad:
            raise ValueError("attitude angles must satisfy 0 <= deadzone < full scale")
        object.__setattr__(self, "mapping_mode", mode)
        object.__setattr__(self, "rotation_base_from_pico", rotation.copy())


@dataclass(frozen=True)
class PicoTeleopCommand:
    action: np.ndarray
    source_sequence: int
    motion_enabled: bool
    reanchored: bool


class PicoPoseMapper:
    """Map snapshots to bounded 10 Hz Base translations and EE-frame rotations."""

    def __init__(self, config: PicoMapperConfig | None = None) -> None:
        self.config = config or PicoMapperConfig()
        self._active = False
        self._anchor_position: np.ndarray | None = None
        self._anchor_translation_rotation: np.ndarray | None = None
        self._anchor_rotation: np.ndarray | None = None
        self._emitted_translation = np.zeros(3, dtype=np.float64)
        self._emitted_rotation = np.eye(3, dtype=np.float64)
        self._translation_active = False
        self._rotation_active = False
        self._gripper_closed = False
        self._trigger_pressed = False

    def _reset_motion(self) -> None:
        self._active = False
        self._anchor_position = None
        self._anchor_translation_rotation = None
        self._anchor_rotation = None
        self._emitted_translation[:] = 0.0
        self._emitted_rotation[:] = np.eye(3)
        self._translation_active = False
        self._rotation_active = False

    def reset(self) -> None:
        self._reset_motion()
        self._gripper_closed = False
        self._trigger_pressed = False

    def _gripper_action(self, trigger: float) -> float:
        pressed = trigger >= self.config.trigger_threshold
        if pressed and not self._trigger_pressed:
            self._gripper_closed = not self._gripper_closed
        self._trigger_pressed = pressed
        return 1.0 if self._gripper_closed else -1.0

    def _controller_pose_in_base_axes(self, controller) -> tuple[np.ndarray, np.ndarray]:
        handedness = UNITY_LEFT_TO_RIGHT_HANDED
        position = self.config.rotation_base_from_pico @ (handedness @ controller.position)
        rotation_unity = _quaternion_xyzw_to_matrix(controller.orientation_xyzw)
        rotation_right_handed = handedness @ rotation_unity @ handedness
        rotation = self.config.rotation_base_from_pico @ rotation_right_handed
        return position, rotation

    @staticmethod
    def _clip_norm(vector: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= limit or norm < 1e-12:
            return vector.copy()
        return vector * (limit / norm)

    @staticmethod
    def _map_controller_rotation_to_ee(rotation_step: np.ndarray) -> np.ndarray:
        """Map controller roll/pitch to the requested EE pitch/roll axes."""
        step = np.asarray(rotation_step, dtype=np.float64)
        return step[[1, 0, 2]].copy()

    @staticmethod
    def _map_left_attitude_to_translation(rotation_step: np.ndarray) -> np.ndarray:
        """Map controller roll/pitch/yaw to Base X/Y/Z translation axes."""
        return np.asarray(rotation_step, dtype=np.float64).copy()

    def _attitude_to_unit_velocity(self, rotvec: np.ndarray) -> np.ndarray:
        """Apply a component-wise angular deadzone and linear full-scale ramp."""
        value = np.asarray(rotvec, dtype=np.float64)
        magnitude = np.abs(value)
        span = self.config.attitude_full_scale_rad - self.config.attitude_deadzone_rad
        normalized = np.clip((magnitude - self.config.attitude_deadzone_rad) / span, 0.0, 1.0)
        return np.sign(value) * normalized

    def step(self, snapshot: PicoSnapshot | None, *, now: float | None = None) -> PicoTeleopCommand | None:
        if snapshot is None or snapshot.age_s(now) > self.config.stale_timeout_s:
            self._reset_motion()
            return None
        left = snapshot.packet.left
        right = snapshot.packet.right
        if not left.tracked or not right.tracked:
            self._reset_motion()
            return None

        if self.config.mapping_mode == "split":
            return self._step_split(snapshot)
        return self._step_single_6dof(snapshot)

    def _step_single_6dof(self, snapshot: PicoSnapshot) -> PicoTeleopCommand:
        left = snapshot.packet.left
        right = snapshot.packet.right

        right_grip = right.grip >= self.config.grip_threshold
        left_grip = left.grip >= self.config.grip_threshold
        enabled = right_grip and (left_grip if self.config.require_both_grips else True)
        action = np.zeros(7, dtype=np.float64)
        action[6] = self._gripper_action(right.trigger)
        position, rotation = self._controller_pose_in_base_axes(right)

        if not enabled:
            self._active = False
            return PicoTeleopCommand(action, snapshot.packet.sequence, False, False)
        if not self._active:
            self._active = True
            self._anchor_position = position
            self._anchor_rotation = rotation
            self._emitted_translation[:] = 0.0
            self._emitted_rotation[:] = np.eye(3)
            return PicoTeleopCommand(action, snapshot.packet.sequence, True, True)

        desired_translation = self.config.translation_scale * (position - self._anchor_position)
        translation_step = self._clip_norm(
            desired_translation - self._emitted_translation,
            self.config.max_translation_step_m,
        )
        self._emitted_translation += translation_step

        # Body-fixed controller delta.  Its rotvec is expressed in the local
        # controller/end-effector axes; the coordinator converts each emitted
        # increment to Base before enqueueing it.
        controller_delta = self._anchor_rotation.T @ rotation
        desired_rotation = rotvec_to_matrix(
            self.config.rotation_scale * matrix_to_rotvec(controller_delta)
        )
        rotation_step = self._clip_norm(
            matrix_to_rotvec(self._emitted_rotation.T @ desired_rotation),
            self.config.max_rotation_step_rad,
        )
        self._emitted_rotation = self._emitted_rotation @ rotvec_to_matrix(rotation_step)
        action[:3] = translation_step
        action[3:6] = self._map_controller_rotation_to_ee(rotation_step)
        return PicoTeleopCommand(action, snapshot.packet.sequence, True, False)

    def _step_split(self, snapshot: PicoSnapshot) -> PicoTeleopCommand:
        left = snapshot.packet.left
        right = snapshot.packet.right
        _, left_rotation = self._controller_pose_in_base_axes(left)
        _, right_rotation = self._controller_pose_in_base_axes(right)
        translation_enabled = left.grip >= self.config.grip_threshold
        rotation_enabled = right.grip >= self.config.grip_threshold
        action = np.zeros(7, dtype=np.float64)
        action[6] = self._gripper_action(right.trigger)
        reanchored = False

        if not translation_enabled:
            self._translation_active = False
        elif not self._translation_active:
            self._translation_active = True
            self._anchor_translation_rotation = left_rotation
            reanchored = True
        else:
            assert self._anchor_translation_rotation is not None
            controller_delta = self._anchor_translation_rotation.T @ left_rotation
            mapped_attitude = self._map_left_attitude_to_translation(
                matrix_to_rotvec(controller_delta)
            )
            unit_velocity = np.clip(
                self.config.translation_scale * self._attitude_to_unit_velocity(mapped_attitude),
                -1.0,
                1.0,
            )
            action[:3] = self.config.max_translation_step_m * unit_velocity

        if not rotation_enabled:
            self._rotation_active = False
        elif not self._rotation_active:
            self._rotation_active = True
            self._anchor_rotation = right_rotation
            self._emitted_rotation[:] = np.eye(3)
            reanchored = True
        else:
            assert self._anchor_rotation is not None
            controller_delta = self._anchor_rotation.T @ right_rotation
            mapped_attitude = self._map_controller_rotation_to_ee(
                matrix_to_rotvec(controller_delta)
            )
            unit_velocity = np.clip(
                self.config.rotation_scale * self._attitude_to_unit_velocity(mapped_attitude),
                -1.0,
                1.0,
            )
            action[3:6] = self.config.max_rotation_step_rad * unit_velocity

        return PicoTeleopCommand(
            action,
            snapshot.packet.sequence,
            translation_enabled or rotation_enabled,
            reanchored,
        )

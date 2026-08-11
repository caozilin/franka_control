"""Stable data contracts shared by policy, planning, and control layers.

The realtime backend intentionally does not import this module.  These types
define the Python-side boundary that feeds its command mailbox: units and
coordinate frames are explicit, while conversion from a policy's numerical
convention lives in :class:`PolicyActionSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


POLICY_ACTION_DIM = 7
CARTESIAN_DOF = 6
FRANKA_JOINT_DOF = 7


class CoordinateFrame(Enum):
    BASE = "base"
    END_EFFECTOR = "end_effector"


class ReferenceSpace(Enum):
    CARTESIAN = "cartesian"
    JOINT = "joint"


def _vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).copy()
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},); got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


@dataclass(frozen=True)
class ControlRates:
    """Frequency contract across the non-realtime and realtime layers."""

    policy_hz: float = 10.0
    planner_hz: float = 10.0
    servo_hz: float = 1000.0

    def __post_init__(self) -> None:
        values = np.asarray((self.policy_hz, self.planner_hz, self.servo_hz), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("control rates must be finite and positive")
        if self.servo_hz < self.policy_hz or self.servo_hz < self.planner_hz:
            raise ValueError("servo_hz must not be lower than policy_hz or planner_hz")

    @property
    def policy_period_s(self) -> float:
        return 1.0 / self.policy_hz

    @property
    def servo_period_s(self) -> float:
        return 1.0 / self.servo_hz


@dataclass(frozen=True)
class CartesianDeltaCommand:
    """One physical Cartesian command consumed by the high-level robot API.

    Translation is measured in metres, rotation is a rotation-vector increment
    in radians, and ``gripper_command`` retains the existing sign convention:
    positive closes, non-positive opens.
    """

    translation_m: np.ndarray
    rotation_vector_rad: np.ndarray
    gripper_command: float
    frame: CoordinateFrame = CoordinateFrame.BASE

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation_m", _vector(self.translation_m, 3, "translation_m"))
        object.__setattr__(
            self,
            "rotation_vector_rad",
            _vector(self.rotation_vector_rad, 3, "rotation_vector_rad"),
        )
        if not np.isfinite(self.gripper_command):
            raise ValueError("gripper_command must be finite")

    @classmethod
    def from_vector(
        cls,
        value,
        *,
        frame: CoordinateFrame = CoordinateFrame.BASE,
    ) -> "CartesianDeltaCommand":
        vector = _vector(value, POLICY_ACTION_DIM, "cartesian command")
        return cls(vector[:3], vector[3:6], float(vector[6]), frame)

    def as_vector(self) -> np.ndarray:
        return np.concatenate(
            (self.translation_m, self.rotation_vector_rad, np.asarray((self.gripper_command,)))
        )


@dataclass(frozen=True)
class JointDeltaCommand:
    """One physical joint-space delta command in radians."""

    joint_delta_rad: np.ndarray
    gripper_command: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "joint_delta_rad",
            _vector(self.joint_delta_rad, FRANKA_JOINT_DOF, "joint_delta_rad"),
        )
        if not np.isfinite(self.gripper_command):
            raise ValueError("gripper_command must be finite")

    @classmethod
    def from_vector(cls, value) -> "JointDeltaCommand":
        vector = _vector(value, FRANKA_JOINT_DOF + 1, "joint command")
        return cls(vector[:FRANKA_JOINT_DOF], float(vector[-1]))

    def as_vector(self) -> np.ndarray:
        return np.concatenate((self.joint_delta_rad, np.asarray((self.gripper_command,))))


@dataclass(frozen=True)
class PolicyActionSpec:
    """Decode a policy-specific 7-D action into physical Cartesian units."""

    translation_scale_m: float = 0.01
    rotation_scale_rad: float = 0.01
    frame: CoordinateFrame = CoordinateFrame.BASE

    def __post_init__(self) -> None:
        scales = np.asarray((self.translation_scale_m, self.rotation_scale_rad), dtype=np.float64)
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("policy action scales must be finite and positive")

    def decode_cartesian(self, action) -> CartesianDeltaCommand:
        vector = _vector(action, POLICY_ACTION_DIM, "policy action")
        return CartesianDeltaCommand(
            translation_m=vector[:3] * self.translation_scale_m,
            rotation_vector_rad=vector[3:6] * self.rotation_scale_rad,
            gripper_command=float(vector[6]),
            frame=self.frame,
        )

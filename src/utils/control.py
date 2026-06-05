from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


POLICY_HZ = 10.0
GRIPPER_WIDTH_MAX = 0.08
LOW_LEVEL_HZ = 1000.0
MAX_TORQUE_RATE = 1000.0
MAX_TRANSLATION_GOAL_ERROR = 0.03
MAX_ROTATION_GOAL_ERROR = math.radians(30.0)
MAX_REF_LINEAR_VELOCITY = 0.1
MAX_REF_LINEAR_ACCELERATION = 0.2
MAX_REF_LINEAR_JERK = 1.0
MAX_REF_ANGULAR_VELOCITY = math.radians(45.0)
MAX_REF_ANGULAR_ACCELERATION = math.radians(90.0)
MAX_REF_ANGULAR_JERK = math.radians(450.0)
REF_POSITION_EPS = 0.0005
REF_LINEAR_VELOCITY_EPS = 0.001
REF_ROTATION_EPS = 0.001
REF_ANGULAR_VELOCITY_EPS = 0.001


@dataclass
class ActionConfig:
    max_translation_step: float = 0.1
    max_rotation_step: float = math.pi / 4.0
    pos_clip: float = 1.0
    rot_clip: float = 6.0


def limit_torque_rate(tau_d: np.ndarray, tau_j_d: np.ndarray, dt: float) -> np.ndarray:
    tau_d = np.asarray(tau_d, dtype=np.float64)
    tau_j_d = np.asarray(tau_j_d, dtype=np.float64)
    dt = max(float(dt), 1.0 / LOW_LEVEL_HZ)
    max_delta = MAX_TORQUE_RATE * dt
    delta = tau_d - tau_j_d
    clipped = np.clip(delta, -max_delta, max_delta)
    if np.any(clipped != delta):
        # print(
        #     "  [扭矩限幅] "
        #     f"关节索引: {np.flatnonzero(mask)}, "
        #     f"原始delta_tau: {delta[mask]}, "
        #     f"限幅后delta_tau: {clipped[mask]}, "
        #     f"上一拍tau_J_d: {tau_j_d[mask]}, "
        #     f"本拍输出tau_cmd: {(tau_j_d + clipped)[mask]}",
        #     flush=True,
        # )
        pass
    return tau_j_d + clipped


def transform_action(action: np.ndarray, config: ActionConfig | None = None) -> np.ndarray:
    """Convert a 7D policy action into one 10Hz executable delta."""
    config = config or ActionConfig()
    action = np.asarray(action, dtype=np.float64)
    transformed = action.copy()
    transformed[:3] = np.clip(transformed[:3], -config.pos_clip, config.pos_clip)
    transformed[:3] *= config.max_translation_step / config.pos_clip
    transformed[3:6] = np.clip(transformed[3:6], -config.rot_clip, config.rot_clip)
    transformed[3:6] *= config.max_rotation_step / config.rot_clip
    transformed[6] = 0.0 if transformed[6] > 0.0 else GRIPPER_WIDTH_MAX
    return transformed

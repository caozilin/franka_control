from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from utils.pose import matrix_to_rotvec, rotvec_to_matrix


PANDA_JOINT_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64
)
PANDA_JOINT_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64
)


@dataclass(frozen=True)
class KinematicState:
    position: np.ndarray
    rotation: np.ndarray
    jacobian: np.ndarray
    manipulability: float
    link_points: np.ndarray


def _fixed_transform(position: tuple[float, float, float], rotation_vector: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotvec_to_matrix(np.asarray(rotation_vector, dtype=np.float64))
    transform[:3, 3] = position
    return transform


class PandaKinematics:
    """Small kinematic clone of the MuJoCo-menagerie Panda chain.

    The base is the real robot's ``O`` frame (no simulation table offset), and
    the output frame matches the menagerie ``ee_site`` / libfranka nominal EE.
    """

    _FIXED = (
        _fixed_transform((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
        _fixed_transform((0.0, 0.0, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
        _fixed_transform((0.0, -0.316, 0.0), (math.pi / 2.0, 0.0, 0.0)),
        _fixed_transform((0.0825, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
        _fixed_transform((-0.0825, 0.384, 0.0), (-math.pi / 2.0, 0.0, 0.0)),
        _fixed_transform((0.0, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
        _fixed_transform((0.088, 0.0, 0.0), (math.pi / 2.0, 0.0, 0.0)),
    )
    _HAND = _fixed_transform((0.0, 0.0, 0.107), (0.0, 0.0, -math.pi / 4.0))
    _EE = _fixed_transform((0.0, 0.0, 0.1034), (0.0, 0.0, 0.0))

    def __init__(self, *, joint_lower: np.ndarray = PANDA_JOINT_LOWER, joint_upper: np.ndarray = PANDA_JOINT_UPPER) -> None:
        self.joint_lower = np.asarray(joint_lower, dtype=np.float64).copy()
        self.joint_upper = np.asarray(joint_upper, dtype=np.float64).copy()

    def evaluate(self, q: np.ndarray) -> KinematicState:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(f"Panda joint vector must have shape (7,); got {q.shape}")

        transform = np.eye(4, dtype=np.float64)
        origins = []
        axes = []
        points = [transform[:3, 3].copy()]
        for fixed, angle in zip(self._FIXED, q, strict=True):
            transform = transform @ fixed
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, 2].copy())
            joint = np.eye(4, dtype=np.float64)
            joint[:3, :3] = rotvec_to_matrix(np.array([0.0, 0.0, angle], dtype=np.float64))
            transform = transform @ joint
            points.append(transform[:3, 3].copy())
        transform = transform @ self._HAND @ self._EE
        position = transform[:3, 3].copy()
        rotation = transform[:3, :3].copy()

        jacobian = np.zeros((6, 7), dtype=np.float64)
        for index, (origin, axis) in enumerate(zip(origins, axes, strict=True)):
            jacobian[:3, index] = np.cross(axis, position - origin)
            jacobian[3:, index] = axis
        determinant = float(np.linalg.det(jacobian @ jacobian.T))
        return KinematicState(
            position=position,
            rotation=rotation,
            jacobian=jacobian,
            manipulability=float(np.sqrt(np.clip(determinant, 0.0, 1.0))),
            link_points=np.asarray(points, dtype=np.float64),
        )

    @staticmethod
    def pose_error(
        state: KinematicState,
        goal_position: np.ndarray,
        goal_rotation: np.ndarray,
        tolerance_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        goal_rotation = np.asarray(goal_rotation, dtype=np.float64)
        rotation_error = matrix_to_rotvec(goal_rotation.T @ state.rotation)
        if tolerance_rotation is not None:
            rotation_error = np.asarray(tolerance_rotation, dtype=np.float64).T @ goal_rotation @ rotation_error
        return np.concatenate((state.position - goal_position, rotation_error))

    @staticmethod
    def pose_error_jacobian(
        state: KinematicState,
        goal_rotation: np.ndarray,
        tolerance_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        goal_rotation = np.asarray(goal_rotation, dtype=np.float64)
        error = matrix_to_rotvec(goal_rotation.T @ state.rotation)
        angle = float(np.linalg.norm(error))
        skew = np.array(
            ((0.0, -error[2], error[1]), (error[2], 0.0, -error[0]), (-error[1], error[0], 0.0)),
            dtype=np.float64,
        )
        coefficient = (
            1.0 / 12.0 + angle * angle / 720.0
            if angle < 1e-5
            else (1.0 - 0.5 * angle / math.tan(0.5 * angle)) / (angle * angle)
        )
        log_left_inverse = np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)
        angular_world = state.jacobian[3:]
        if tolerance_rotation is None:
            rotation_jacobian = log_left_inverse @ goal_rotation.T @ angular_world
        else:
            tolerance_rotation = np.asarray(tolerance_rotation, dtype=np.float64)
            rotation_jacobian = (
                tolerance_rotation.T
                @ goal_rotation
                @ log_left_inverse
                @ goal_rotation.T
                @ angular_world
            )
        return np.vstack((state.jacobian[:3], rotation_jacobian))

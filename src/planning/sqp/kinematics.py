from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

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
    manipulability: float | None
    link_points: np.ndarray | None


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

    def __init__(
        self,
        *,
        joint_lower: np.ndarray = PANDA_JOINT_LOWER,
        joint_upper: np.ndarray = PANDA_JOINT_UPPER,
        use_accelerated: bool = True,
    ) -> None:
        self.joint_lower = np.asarray(joint_lower, dtype=np.float64).copy()
        self.joint_upper = np.asarray(joint_upper, dtype=np.float64).copy()
        self._pin: Any | None = None
        self._pin_model: Any | None = None
        self._pin_data: Any | None = None
        self._pin_joint_ids: tuple[int, ...] = ()
        self._pin_ee_frame_id: int | None = None
        if use_accelerated:
            self._initialize_pinocchio()

    @property
    def backend_name(self) -> str:
        return "pinocchio" if self._pin_model is not None else "python"

    def _initialize_pinocchio(self) -> None:
        """Build the exact Menagerie Panda chain in Pinocchio when available."""
        try:
            import pinocchio as pin
        except ImportError:
            return

        model = pin.Model()
        parent = 0
        joint_ids: list[int] = []
        for index, fixed in enumerate(self._FIXED):
            placement = pin.SE3(fixed[:3, :3], fixed[:3, 3])
            parent = model.addJoint(parent, pin.JointModelRZ(), placement, f"panda_joint{index + 1}")
            joint_ids.append(parent)
        ee_placement = self._HAND @ self._EE
        ee_frame_id = model.addFrame(
            pin.Frame(
                "panda_ee",
                parent,
                parent,
                pin.SE3(ee_placement[:3, :3], ee_placement[:3, 3]),
                pin.FrameType.OP_FRAME,
            )
        )
        self._pin = pin
        self._pin_model = model
        self._pin_data = model.createData()
        self._pin_joint_ids = tuple(joint_ids)
        self._pin_ee_frame_id = int(ee_frame_id)

    def evaluate(
        self,
        q: np.ndarray,
        *,
        include_manipulability: bool = True,
        include_link_points: bool = True,
    ) -> KinematicState:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(f"Panda joint vector must have shape (7,); got {q.shape}")
        if self._pin_model is not None:
            return self._evaluate_pinocchio(
                q,
                include_manipulability=include_manipulability,
                include_link_points=include_link_points,
            )
        return self._evaluate_python(
            q,
            include_manipulability=include_manipulability,
            include_link_points=include_link_points,
        )

    def _evaluate_pinocchio(
        self,
        q: np.ndarray,
        *,
        include_manipulability: bool,
        include_link_points: bool,
    ) -> KinematicState:
        pin = self._pin
        model = self._pin_model
        data = self._pin_data
        frame_id = self._pin_ee_frame_id
        assert pin is not None and model is not None and data is not None and frame_id is not None

        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacement(model, data, frame_id)
        placement = data.oMf[frame_id]
        jacobian = np.asarray(
            pin.getFrameJacobian(model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED),
            dtype=np.float64,
        ).copy()
        manipulability = None
        if include_manipulability:
            determinant = float(np.linalg.det(jacobian @ jacobian.T))
            manipulability = float(np.sqrt(np.clip(determinant, 0.0, 1.0)))
        link_points = None
        if include_link_points:
            link_points = np.empty((8, 3), dtype=np.float64)
            link_points[0] = 0.0
            for index, joint_id in enumerate(self._pin_joint_ids, start=1):
                link_points[index] = data.oMi[joint_id].translation
        return KinematicState(
            position=np.asarray(placement.translation, dtype=np.float64).copy(),
            rotation=np.asarray(placement.rotation, dtype=np.float64).copy(),
            jacobian=jacobian,
            manipulability=manipulability,
            link_points=link_points,
        )

    def _evaluate_python(
        self,
        q: np.ndarray,
        *,
        include_manipulability: bool,
        include_link_points: bool,
    ) -> KinematicState:

        transform = np.eye(4, dtype=np.float64)
        origins = []
        axes = []
        points = [transform[:3, 3].copy()] if include_link_points else None
        for fixed, angle in zip(self._FIXED, q, strict=True):
            transform = transform @ fixed
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, 2].copy())
            joint = np.eye(4, dtype=np.float64)
            joint[:3, :3] = rotvec_to_matrix(np.array([0.0, 0.0, angle], dtype=np.float64))
            transform = transform @ joint
            if points is not None:
                points.append(transform[:3, 3].copy())
        transform = transform @ self._HAND @ self._EE
        position = transform[:3, 3].copy()
        rotation = transform[:3, :3].copy()

        jacobian = np.zeros((6, 7), dtype=np.float64)
        for index, (origin, axis) in enumerate(zip(origins, axes, strict=True)):
            jacobian[:3, index] = np.cross(axis, position - origin)
            jacobian[3:, index] = axis
        manipulability = None
        if include_manipulability:
            determinant = float(np.linalg.det(jacobian @ jacobian.T))
            manipulability = float(np.sqrt(np.clip(determinant, 0.0, 1.0)))
        return KinematicState(
            position=position,
            rotation=rotation,
            jacobian=jacobian,
            manipulability=manipulability,
            link_points=None if points is None else np.asarray(points, dtype=np.float64),
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

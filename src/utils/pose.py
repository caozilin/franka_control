from __future__ import annotations

import math

import numpy as np


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def fixed_xyz_rotation_matrix(angles: np.ndarray) -> np.ndarray:
    """Return the fixed-axis XYZ/RPY rotation ``Rz(yaw) Ry(pitch) Rx(roll)``.

    The three angles are ordered ``[roll_x, pitch_y, yaw_z]``. They are
    active rotations about fixed axes: X is applied first, followed by Y and
    then Z. This bounded tolerance chart is deliberately separate from
    :func:`rotvec_to_matrix`, whose input is one exponential-coordinate vector.
    """
    value = np.asarray(angles, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("Fixed-axis XYZ angles must contain three finite values")
    roll, pitch, yaw = (float(item) for item in value)
    sx, cx = math.sin(roll), math.cos(roll)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)
    return np.array(
        (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        ),
        dtype=np.float64,
    )


def fixed_xyz_rotation_angles(rotation: np.ndarray) -> np.ndarray:
    """Return principal fixed-axis XYZ/RPY angles for a rotation matrix.

    The result ``[roll_x, pitch_y, yaw_z]`` satisfies
    ``R = Rz(yaw_z) @ Ry(pitch_y) @ Rx(roll_x)``. The principal pitch lies
    in ``[-pi/2, pi/2]``. The deterministic gimbal branch chooses yaw zero.
    """
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Fixed-axis XYZ extraction requires a finite 3x3 matrix")
    horizontal = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    pitch = math.atan2(-float(matrix[2, 0]), horizontal)
    if horizontal > 1.0e-10:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        yaw = 0.0
        if matrix[2, 0] < 0.0:
            roll = math.atan2(float(matrix[0, 1]), float(matrix[0, 2]))
        else:
            roll = math.atan2(-float(matrix[0, 1]), -float(matrix[0, 2]))
    return np.array((roll, pitch, yaw), dtype=np.float64)


def rotation_tolerance_coordinates(
    actual: np.ndarray,
    target: np.ndarray,
    frame: np.ndarray,
) -> np.ndarray:
    """Return actual-minus-target fixed-axis XYZ coordinates in ``frame``."""
    actual_array = np.asarray(actual, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    frame_array = np.asarray(frame, dtype=np.float64)
    if any(
        value.shape != (3, 3)
        for value in (actual_array, target_array, frame_array)
    ):
        raise ValueError("Tolerance coordinates require three 3x3 rotations")
    return fixed_xyz_rotation_angles(
        frame_array.T @ actual_array @ target_array.T @ frame_array
    )


def rotation_from_tolerance_coordinates(
    target: np.ndarray,
    frame: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    """Apply fixed-axis XYZ tolerance coordinates to an absolute target."""
    target_array = np.asarray(target, dtype=np.float64)
    frame_array = np.asarray(frame, dtype=np.float64)
    if target_array.shape != (3, 3) or frame_array.shape != (3, 3):
        raise ValueError("Tolerance reconstruction requires two 3x3 rotations")
    return (
        frame_array
        @ fixed_xyz_rotation_matrix(coordinates)
        @ frame_array.T
        @ target_array
    )


def rotation_tolerance_coordinate_jacobian(
    actual: np.ndarray,
    target: np.ndarray,
    frame: np.ndarray,
    angular_world_jacobian: np.ndarray,
) -> np.ndarray:
    """Map a world angular-velocity Jacobian to fixed-axis XYZ rates."""
    coordinates = rotation_tolerance_coordinates(actual, target, frame)
    _, pitch, yaw = (float(item) for item in coordinates)
    sy, cy = math.sin(pitch), math.cos(pitch)
    sz, cz = math.sin(yaw), math.cos(yaw)
    if abs(cy) <= 1.0e-8:
        raise ValueError(
            "Fixed-axis XYZ tolerance Jacobian is singular at pitch +/- pi/2"
        )
    rate_to_spatial = np.array(
        (
            (cz * cy, -sz, 0.0),
            (sz * cy, cz, 0.0),
            (-sy, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    angular = np.asarray(angular_world_jacobian, dtype=np.float64)
    if angular.ndim != 2 or angular.shape[0] != 3:
        raise ValueError("Angular Jacobian must have shape (3, N)")
    return np.linalg.solve(rate_to_spatial, np.asarray(frame).T @ angular)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    cos_angle = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    angle = math.acos(cos_angle)
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)

    if abs(angle - math.pi) < 1e-6:
        diag = np.diag(matrix)
        axis = np.sqrt(np.maximum((diag + 1.0) * 0.5, 0.0)).astype(np.float64)
        if axis[0] > 1e-8:
            axis[1] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[0])
            axis[2] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[0])
        elif axis[1] > 1e-8:
            axis[0] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[1])
            axis[2] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[1])
        else:
            axis[0] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[2])
            axis[1] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[2])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return angle * axis / norm

    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def matrix_to_rotvec_continuous(matrix: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    rotvec = matrix_to_rotvec(matrix)
    if previous is None:
        return rotvec

    previous = np.asarray(previous, dtype=np.float64)
    if previous.shape != (3,):
        return rotvec

    previous_norm = float(np.linalg.norm(previous))
    angle = float(np.linalg.norm(rotvec))
    if previous_norm < 1e-12 and angle < 1e-12:
        return np.zeros(3, dtype=np.float64)

    if angle < 1e-12:
        axis = previous / previous_norm if previous_norm >= 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        center = int(round(previous_norm / (2.0 * np.pi)))
        candidates = [2.0 * np.pi * k * axis for k in range(center - 3, center + 4)]
        candidates.append(np.zeros(3, dtype=np.float64))
        return min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - previous))).copy()

    axis = rotvec / angle
    if abs(angle - math.pi) < 1e-4 and previous_norm >= 1e-12:
        previous_axis = previous / previous_norm
        if float(np.dot(axis, previous_axis)) < 0.0:
            axis = -axis
            rotvec = angle * axis
    projected_previous = float(np.dot(previous, axis))
    center = int(round((projected_previous - angle) / (2.0 * np.pi)))
    candidates = [rotvec + 2.0 * np.pi * k * axis for k in range(center - 4, center + 5)]
    return min(candidates, key=lambda candidate: float(np.linalg.norm(candidate - previous))).copy()


def end_effector_rotation_to_backend_rotation(
    current_rotation: np.ndarray,
    rotvec_ee: np.ndarray,
) -> np.ndarray:
    """Convert an end-effector-frame rotation increment to a Base-frame increment."""
    current_rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    rotvec_ee = np.asarray(rotvec_ee, dtype=np.float64)
    if rotvec_ee.shape != (3,):
        raise ValueError(f"rotvec_ee must have shape (3,); got {rotvec_ee.shape}")
    if float(np.linalg.norm(rotvec_ee)) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    delta_rotation_ee = rotvec_to_matrix(rotvec_ee)
    delta_rotation_base = current_rotation @ delta_rotation_ee @ current_rotation.T
    return matrix_to_rotvec(delta_rotation_base)


def pose_array_to_matrix(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64).reshape(4, 4, order="F")


def matrix_to_pose_array(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64).reshape(16, order="F")


def pose_error(
    current_pose: np.ndarray,
    desired_pose: np.ndarray,
    previous_rotvec: np.ndarray | None = None,
) -> np.ndarray:
    del previous_rotvec
    current = pose_array_to_matrix(current_pose)
    desired = pose_array_to_matrix(desired_pose)
    error = np.zeros(6, dtype=np.float64)
    error[:3] = current[:3, 3] - desired[:3, 3]
    error[3:] = matrix_to_rotvec(current[:3, :3] @ desired[:3, :3].T)
    return error

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


def pose_array_to_matrix(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose, dtype=np.float64).reshape(4, 4, order="F")


def matrix_to_pose_array(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64).reshape(16, order="F")


def pose_error(
    current_pose: np.ndarray,
    desired_pose: np.ndarray,
    previous_rotvec: np.ndarray | None = None,
) -> np.ndarray:
    current = pose_array_to_matrix(current_pose)
    desired = pose_array_to_matrix(desired_pose)
    error = np.zeros(6, dtype=np.float64)
    error[:3] = current[:3, 3] - desired[:3, 3]
    error[3:] = matrix_to_rotvec_continuous(current[:3, :3] @ desired[:3, :3].T, previous_rotvec)
    return error

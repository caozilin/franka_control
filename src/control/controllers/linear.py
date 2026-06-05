from __future__ import annotations

import numpy as np

from utils.control import POLICY_HZ, limit_torque_rate
from utils.pose import (
    matrix_to_pose_array,
    matrix_to_rotvec_continuous,
    pose_array_to_matrix,
    pose_error,
    rotvec_to_matrix,
)


class LinearController:
    def __init__(
        self,
        initial_pose: np.ndarray,
        *,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ):
        self.stiffness = np.asarray(stiffness, dtype=np.float64)
        self.damping = np.asarray(damping, dtype=np.float64)
        self.policy_period = 1.0 / POLICY_HZ
        self.next_policy_time = 0.0
        self.segment_start_time = 0.0
        self.segment_start_pose = np.asarray(initial_pose, dtype=np.float64).copy()
        self.segment_target_pose = self.segment_start_pose.copy()
        self.command_pose = self.segment_start_pose.copy()
        self.segment_delta_translation = np.zeros(3, dtype=np.float64)
        self.segment_delta_rotvec = np.zeros(3, dtype=np.float64)
        self.last_transformed_action = np.zeros(7, dtype=np.float64)
        self.last_error_rotvec = np.zeros(3, dtype=np.float64)
        self.policy_tick = 0

    def update_goal(self, elapsed: float, pose_goal: np.ndarray, transformed_action: np.ndarray | None = None) -> dict:
        self.segment_start_time = elapsed
        self.segment_start_pose = self.command_pose.copy()
        self.segment_target_pose = np.asarray(pose_goal, dtype=np.float64).copy()

        start_matrix = pose_array_to_matrix(self.segment_start_pose)
        target_matrix = pose_array_to_matrix(self.segment_target_pose)
        self.segment_delta_translation = target_matrix[:3, 3] - start_matrix[:3, 3]
        self.segment_delta_rotvec = matrix_to_rotvec_continuous(
            target_matrix[:3, :3] @ start_matrix[:3, :3].T,
            self.segment_delta_rotvec,
        )

        if transformed_action is not None:
            self.last_transformed_action = np.asarray(transformed_action, dtype=np.float64).copy()

        event = {
            "policy_tick": self.policy_tick,
            "elapsed": float(elapsed),
            "transformed_action": self.last_transformed_action.copy(),
            "target_xyz": self.segment_target_pose[[12, 13, 14]].copy(),
        }
        self.policy_tick += 1
        self.next_policy_time += self.policy_period
        return event

    def _compute_reference(self, elapsed: float) -> tuple[np.ndarray, np.ndarray]:
        alpha = (elapsed - self.segment_start_time) / self.policy_period
        weight, velocity_weight = self._weights(alpha)

        start_matrix = pose_array_to_matrix(self.segment_start_pose)
        command_matrix = start_matrix.copy()
        command_matrix[:3, 3] += weight * self.segment_delta_translation
        command_matrix[:3, :3] = rotvec_to_matrix(weight * self.segment_delta_rotvec) @ start_matrix[:3, :3]
        self.command_pose = matrix_to_pose_array(command_matrix)

        desired_velocity = np.zeros(6, dtype=np.float64)
        desired_velocity[:3] = velocity_weight * self.segment_delta_translation
        desired_velocity[3:6] = velocity_weight * self.segment_delta_rotvec
        return self.command_pose.copy(), desired_velocity

    def step(self, robot_state, model, pose_goal: np.ndarray, elapsed: float, dt: float) -> np.ndarray:
        desired_pose, desired_velocity = self._compute_reference(elapsed)
        coriolis = np.asarray(model.coriolis(robot_state), dtype=np.float64)
        jacobian = np.asarray(model.zero_jacobian(robot_state), dtype=np.float64).reshape(6, 7, order="F")
        dq = np.asarray(robot_state.dq, dtype=np.float64)
        error = pose_error(robot_state.O_T_EE, desired_pose, self.last_error_rotvec)
        self.last_error_rotvec = error[3:].copy()
        tau_task = jacobian.T @ (-self.stiffness @ error + self.damping @ (desired_velocity - jacobian @ dq))
        return limit_torque_rate(tau_task + coriolis, robot_state.tau_J_d, dt)

    def _weights(self, alpha: float) -> tuple[float, float]:
        raw_alpha = float(alpha)
        alpha = float(np.clip(raw_alpha, 0.0, 1.0))
        velocity = 1.0 / self.policy_period if 0.0 <= raw_alpha <= 1.0 else 0.0
        return alpha, velocity

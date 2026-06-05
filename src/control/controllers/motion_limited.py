from __future__ import annotations

import math

import numpy as np

from utils.control import (
    MAX_REF_ANGULAR_ACCELERATION,
    MAX_REF_ANGULAR_JERK,
    MAX_REF_ANGULAR_VELOCITY,
    MAX_REF_LINEAR_ACCELERATION,
    MAX_REF_LINEAR_JERK,
    MAX_REF_LINEAR_VELOCITY,
    POLICY_HZ,
    REF_ANGULAR_VELOCITY_EPS,
    REF_LINEAR_VELOCITY_EPS,
    REF_POSITION_EPS,
    REF_ROTATION_EPS,
    limit_torque_rate,
)
from utils.pose import (
    matrix_to_pose_array,
    matrix_to_rotvec_continuous,
    pose_array_to_matrix,
    pose_error,
    rotvec_to_matrix,
)


def clamp_norm(value: np.ndarray, limit: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= float(limit) or norm < 1e-12:
        return value.copy()
    return value * (float(limit) / norm)


class MotionLimitedPoseController:
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
        self.goal_pose = np.asarray(initial_pose, dtype=np.float64).copy()
        self.command_pose = self.goal_pose.copy()
        self.v_ref = np.zeros(3, dtype=np.float64)
        self.a_ref = np.zeros(3, dtype=np.float64)
        self.omega_ref = np.zeros(3, dtype=np.float64)
        self.alpha_ref = np.zeros(3, dtype=np.float64)
        self.last_transformed_action = np.zeros(7, dtype=np.float64)
        self.last_ref_rotation_error = np.zeros(3, dtype=np.float64)
        self.last_error_rotvec = np.zeros(3, dtype=np.float64)
        self.policy_tick = 0

    def update_goal(self, elapsed: float, pose_goal: np.ndarray, transformed_action: np.ndarray | None = None) -> dict:
        self.goal_pose = np.asarray(pose_goal, dtype=np.float64).copy()
        if transformed_action is not None:
            self.last_transformed_action = np.asarray(transformed_action, dtype=np.float64).copy()

        event = {
            "policy_tick": self.policy_tick,
            "elapsed": float(elapsed),
            "transformed_action": self.last_transformed_action.copy(),
            "target_xyz": self.goal_pose[[12, 13, 14]].copy(),
        }
        self.policy_tick += 1
        self.next_policy_time += self.policy_period
        return event

    def _update_position_ref(self, dt: float) -> None:
        goal_matrix = pose_array_to_matrix(self.goal_pose)
        ref_matrix = pose_array_to_matrix(self.command_pose)
        p_goal = goal_matrix[:3, 3]
        p_ref = ref_matrix[:3, 3]
        error = p_goal - p_ref
        distance = float(np.linalg.norm(error))

        if distance < REF_POSITION_EPS and float(np.linalg.norm(self.v_ref)) < REF_LINEAR_VELOCITY_EPS:
            ref_matrix[:3, 3] = p_goal
            self.v_ref.fill(0.0)
            self.a_ref.fill(0.0)
            self.command_pose = matrix_to_pose_array(ref_matrix)
            return

        if distance < 1e-12:
            v_des = np.zeros(3, dtype=np.float64)
        else:
            direction = error / distance
            v_allow = math.sqrt(max(0.0, 2.0 * MAX_REF_LINEAR_ACCELERATION * distance))
            v_des = direction * min(MAX_REF_LINEAR_VELOCITY, v_allow)

        a_cmd = clamp_norm((v_des - self.v_ref) / dt, MAX_REF_LINEAR_ACCELERATION)
        delta_a = clamp_norm(a_cmd - self.a_ref, MAX_REF_LINEAR_JERK * dt)
        self.a_ref = clamp_norm(self.a_ref + delta_a, MAX_REF_LINEAR_ACCELERATION)
        self.v_ref = clamp_norm(self.v_ref + self.a_ref * dt, MAX_REF_LINEAR_VELOCITY)
        ref_matrix[:3, 3] = p_ref + self.v_ref * dt
        self.command_pose = matrix_to_pose_array(ref_matrix)

    def _update_rotation_ref(self, dt: float) -> None:
        goal_matrix = pose_array_to_matrix(self.goal_pose)
        ref_matrix = pose_array_to_matrix(self.command_pose)
        phi = matrix_to_rotvec_continuous(goal_matrix[:3, :3] @ ref_matrix[:3, :3].T, self.last_ref_rotation_error)
        self.last_ref_rotation_error = phi.copy()
        distance = float(np.linalg.norm(phi))

        if distance < REF_ROTATION_EPS and float(np.linalg.norm(self.omega_ref)) < REF_ANGULAR_VELOCITY_EPS:
            ref_matrix[:3, :3] = goal_matrix[:3, :3]
            self.omega_ref.fill(0.0)
            self.alpha_ref.fill(0.0)
            self.command_pose = matrix_to_pose_array(ref_matrix)
            return

        if distance < 1e-12:
            omega_des = np.zeros(3, dtype=np.float64)
        else:
            direction = phi / distance
            omega_allow = math.sqrt(max(0.0, 2.0 * MAX_REF_ANGULAR_ACCELERATION * distance))
            omega_des = direction * min(MAX_REF_ANGULAR_VELOCITY, omega_allow)

        alpha_cmd = clamp_norm((omega_des - self.omega_ref) / dt, MAX_REF_ANGULAR_ACCELERATION)
        delta_alpha = clamp_norm(alpha_cmd - self.alpha_ref, MAX_REF_ANGULAR_JERK * dt)
        self.alpha_ref = clamp_norm(self.alpha_ref + delta_alpha, MAX_REF_ANGULAR_ACCELERATION)
        self.omega_ref = clamp_norm(self.omega_ref + self.alpha_ref * dt, MAX_REF_ANGULAR_VELOCITY)
        ref_matrix[:3, :3] = rotvec_to_matrix(self.omega_ref * dt) @ ref_matrix[:3, :3]
        self.command_pose = matrix_to_pose_array(ref_matrix)

    def _compute_reference(self, actual_pose: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt), 1e-6)
        self._update_position_ref(dt)
        self._update_rotation_ref(dt)
        desired_velocity = np.zeros(6, dtype=np.float64)
        desired_velocity[:3] = self.v_ref
        desired_velocity[3:6] = self.omega_ref
        return self.command_pose.copy(), desired_velocity

    def step(self, robot_state, model, pose_goal: np.ndarray, elapsed: float, dt: float) -> np.ndarray:
        desired_pose, desired_velocity = self._compute_reference(robot_state.O_T_EE, dt)
        coriolis = np.asarray(model.coriolis(robot_state), dtype=np.float64)
        jacobian = np.asarray(model.zero_jacobian(robot_state), dtype=np.float64).reshape(6, 7, order="F")
        dq = np.asarray(robot_state.dq, dtype=np.float64)
        error = pose_error(robot_state.O_T_EE, desired_pose, self.last_error_rotvec)
        self.last_error_rotvec = error[3:].copy()
        tau_task = jacobian.T @ (-self.stiffness @ error + self.damping @ (desired_velocity - jacobian @ dq))
        return limit_torque_rate(tau_task + coriolis, robot_state.tau_J_d, dt)

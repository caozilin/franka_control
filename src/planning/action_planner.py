from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Literal

import numpy as np

from planning.ipopt import IpoptPlanner, IpoptSettings
from planning.task_space import ObjectiveSettings
from planning.task_tolerance import ManipulationPhase
from planning.tolerance.state import RotationalToleranceState
from utils.control import ActionConfig, transform_action
from utils.pose import rotvec_to_matrix


PLANNER_MODE_CHOICES = ("direct", "ipopt")


@dataclass(frozen=True)
class PlannerConfig:
    mode: str = "direct"
    ipopt_settings: IpoptSettings = field(default_factory=IpoptSettings)
    objective_settings: ObjectiveSettings = field(
        default_factory=ObjectiveSettings.delta_q_squared_conditioning_safeguard
    )
    rotation_ranged_axes: tuple[bool, bool, bool] = (False, False, False)
    rotation_limits_deg: tuple[float, float, float] = (30.0, 30.0, 45.0)
    tolerance_frame_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        mode = str(self.mode).lower().strip()
        if mode not in PLANNER_MODE_CHOICES:
            raise ValueError(
                f"planner mode must be one of {PLANNER_MODE_CHOICES}; got {self.mode!r}"
            )
        object.__setattr__(self, "mode", mode)
        if len(self.rotation_ranged_axes) != 3:
            raise ValueError("rotation_ranged_axes must contain three booleans")
        if len(self.rotation_limits_deg) != 3:
            raise ValueError("rotation_limits_deg must contain three values")
        if len(self.tolerance_frame_rotvec) != 3:
            raise ValueError("tolerance_frame_rotvec must contain three values")
        limits = np.asarray(self.rotation_limits_deg, dtype=np.float64)
        if not np.all(np.isfinite(limits)) or np.any(limits < 0.0):
            raise ValueError("rotation_limits_deg must be finite and non-negative")


@dataclass(frozen=True)
class PlannedRobotCommand:
    reference_space: Literal["cartesian", "joint"]
    cartesian_action: np.ndarray | None = None
    joint_target: np.ndarray | None = None
    gripper_target: float | None = None
    telemetry: dict[str, object] | None = None
    actual_pose: np.ndarray | None = None
    planned_pose: np.ndarray | None = None
    nominal_pose: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.reference_space not in {"cartesian", "joint"}:
            raise ValueError("reference_space must be 'cartesian' or 'joint'")
        if self.reference_space == "cartesian":
            action = np.asarray(self.cartesian_action, dtype=np.float64)
            if action.shape != (7,) or self.joint_target is not None:
                raise ValueError("Cartesian planner command requires one 7D action")
            object.__setattr__(self, "cartesian_action", action.copy())
            return
        target = np.asarray(self.joint_target, dtype=np.float64)
        if target.shape != (7,) or self.cartesian_action is not None:
            raise ValueError("Joint planner command requires one 7D joint target")
        if self.gripper_target is None:
            raise ValueError("Joint planner command requires a gripper target")
        object.__setattr__(self, "joint_target", target.copy())
        for name in ("actual_pose", "planned_pose", "nominal_pose"):
            pose = getattr(self, name)
            if pose is None:
                continue
            pose_array = np.asarray(pose, dtype=np.float64)
            if pose_array.shape != (4, 4) or not np.all(np.isfinite(pose_array)):
                raise ValueError(f"{name} must be a finite 4x4 transform")
            object.__setattr__(self, name, pose_array.copy())


def _pose_transform(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


class CartesianActionPlanner:
    """Shared 10 Hz boundary from Cartesian actions to direct or IPOPT commands."""

    def __init__(self, config: PlannerConfig = PlannerConfig()) -> None:
        self.config = config
        self._planner = (
            None
            if config.mode == "direct"
            else IpoptPlanner(
                settings=config.ipopt_settings,
                objective_settings=config.objective_settings,
            )
        )
        self._tolerance_state: RotationalToleranceState | None = None
        ranged = np.asarray(config.rotation_ranged_axes, dtype=bool)
        if self._planner is not None and np.any(ranged):
            limits = np.radians(np.asarray(config.rotation_limits_deg, dtype=np.float64))
            self._tolerance_state = RotationalToleranceState(
                limits=limits,
                negative_limits=limits,
                positive_limits=limits,
                ranged=ranged,
                frame=rotvec_to_matrix(
                    np.asarray(config.tolerance_frame_rotvec, dtype=np.float64)
                ),
            )

    def configure_rotation_tolerance(
        self,
        target_rotation: np.ndarray,
        tolerance_frame: np.ndarray,
        negative_limits: np.ndarray,
        positive_limits: np.ndarray,
        *,
        phase: ManipulationPhase = ManipulationPhase.PREGRASP,
        stage_key: Hashable | None = None,
    ) -> None:
        if self._planner is None:
            raise RuntimeError("rotation tolerance requires the IPOPT planner")
        target = np.asarray(target_rotation, dtype=np.float64)
        frame = np.asarray(tolerance_frame, dtype=np.float64)
        negative = np.asarray(negative_limits, dtype=np.float64)
        positive = np.asarray(positive_limits, dtype=np.float64)
        if target.shape != (3, 3) or frame.shape != (3, 3):
            raise ValueError("rotation target and tolerance frame must be 3x3")
        if negative.shape != (3,) or positive.shape != (3,):
            raise ValueError("signed tolerance limits must have shape (3,)")
        if np.any(negative < 0.0) or np.any(positive < 0.0):
            raise ValueError("signed tolerance limits must be non-negative")
        if self._tolerance_state is None:
            self._tolerance_state = RotationalToleranceState()
        self._tolerance_state.configure_stage(
            int(phase),
            target,
            frame,
            np.maximum(negative, positive),
            negative_limits=negative,
            positive_limits=positive,
            stage_key=(int(phase) if stage_key is None else stage_key),
        )
        self._tolerance_state.ranged = (
            (negative > 0.0) | (positive > 0.0)
            if phase not in (ManipulationPhase.GRASP, ManipulationPhase.RELEASE)
            else np.zeros(3, dtype=bool)
        )

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def control_mode(self) -> str:
        return "cartesian" if self._planner is None else "joint"

    def reset(self, measured_q: np.ndarray | None = None) -> None:
        if self._tolerance_state is not None:
            self._tolerance_state.reset_subtask_history()
        if self._planner is not None:
            self._planner.reset(measured_q)

    def plan(
        self,
        measured_q: np.ndarray,
        cartesian_action: np.ndarray,
        action_config: ActionConfig,
        *,
        semantic_key: Hashable | None = None,
    ) -> PlannedRobotCommand:
        action = np.asarray(cartesian_action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Cartesian planner action must have shape (7,); got {action.shape}")
        if self._planner is None:
            return PlannedRobotCommand("cartesian", cartesian_action=action)

        transformed = transform_action(action, action_config)
        measured = np.asarray(measured_q, dtype=np.float64)
        if measured.shape != (7,):
            raise ValueError(f"measured_q must have shape (7,); got {measured.shape}")
        plan = self._planner.step(
            measured,
            transformed[:6],
            tolerance=self._tolerance_state,
            semantic_key=semantic_key,
        )
        diagnostics = plan.diagnostics
        telemetry: dict[str, object] = {
            "planner_mode": "ipopt",
            "status": diagnostics.status,
            "feasible": diagnostics.feasible,
            "converged": diagnostics.converged,
            "iterations": diagnostics.iterations,
            "optimality": diagnostics.optimality,
            "elapsed_ms": diagnostics.elapsed_ms,
            "position_residual": diagnostics.position_residual,
            "rotation_residual": diagnostics.strict_rotation_residual,
            "tolerance_violation": diagnostics.effective_constraint_violation,
            "consecutive_failures": diagnostics.consecutive_failures,
            "release_loss_multiplier": diagnostics.release_loss_multiplier,
            "stage_relative_tolerance_release_rad": diagnostics.stage_relative_tolerance_release_rad,
            "predicted_tolerance_violation_rad": diagnostics.predicted_tolerance_violation_rad,
        }
        actual_state = self._planner.kinematics.evaluate(measured)
        planned_state = self._planner.kinematics.evaluate(plan.q)
        return PlannedRobotCommand(
            "joint",
            joint_target=plan.q,
            gripper_target=float(transformed[6]),
            telemetry=telemetry,
            actual_pose=_pose_transform(actual_state.position, actual_state.rotation),
            planned_pose=_pose_transform(planned_state.position, planned_state.rotation),
            nominal_pose=_pose_transform(plan.target.position, plan.target.rotation),
        )

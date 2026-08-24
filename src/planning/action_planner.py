from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Literal

import numpy as np

from planning.sqp import (
    AxisTask,
    BaselineSQPPlanner,
    SQPObjectiveSettings,
    SQPSettings,
    ShadowSQPPlanner,
    TaskKind,
)
from utils.control import ActionConfig, transform_action
from planning.task_tolerance import ManipulationPhase, RotationalToleranceState
from utils.pose import rotvec_to_matrix


PLANNER_MODE_CHOICES = ("direct", "baseline_sqp", "shadow_sqp")
_TELEOP_DT = 0.1


@dataclass(frozen=True)
class PlannerConfig:
    mode: str = "direct"
    solver_settings: SQPSettings = field(default_factory=SQPSettings)
    objective_settings: SQPObjectiveSettings = field(default_factory=SQPObjectiveSettings)
    rotation_ranged_axes: tuple[bool, bool, bool] = (False, False, False)
    rotation_limits_deg: tuple[float, float, float] = (30.0, 30.0, 45.0)
    tolerance_frame_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    shadow_stage: str = "default"

    def __post_init__(self) -> None:
        mode = str(self.mode).lower().strip()
        if mode not in PLANNER_MODE_CHOICES:
            raise ValueError(f"planner mode must be one of {PLANNER_MODE_CHOICES}; got {self.mode!r}")
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
    """Shared 10 Hz boundary from physical Cartesian actions to robot commands."""

    def __init__(self, config: PlannerConfig = PlannerConfig()) -> None:
        self.config = config
        self._planner: BaselineSQPPlanner | ShadowSQPPlanner | None = None
        self._axis_tasks: tuple[AxisTask, ...] | None = None
        self._tolerance_frame = rotvec_to_matrix(np.asarray(config.tolerance_frame_rotvec, dtype=np.float64))
        self._ranged_axes = np.asarray(config.rotation_ranged_axes, dtype=bool)
        self._rotation_limits = np.radians(np.asarray(config.rotation_limits_deg, dtype=np.float64))
        self._tolerance_state: RotationalToleranceState | None = None
        if config.mode != "direct":
            baseline = BaselineSQPPlanner(
                solver_settings=config.solver_settings,
                objective_settings=config.objective_settings,
            )
            self._planner = ShadowSQPPlanner(baseline) if config.mode == "shadow_sqp" else baseline
            tasks = [AxisTask(TaskKind.SPECIFIC) for _ in range(6)]
            for axis, ranged in enumerate(self._ranged_axes):
                if ranged:
                    tasks[axis + 3] = AxisTask(
                        TaskKind.FLAT_RANGE,
                        lower=-self._rotation_limits[axis],
                        upper=self._rotation_limits[axis],
                    )
            self._axis_tasks = tuple(tasks)

    def configure_rotation_tolerance(
        self,
        target_rotation: np.ndarray,
        tolerance_frame: np.ndarray,
        negative_limits: np.ndarray,
        positive_limits: np.ndarray,
        *,
        phase: ManipulationPhase = ManipulationPhase.PREGRASP,
    ) -> None:
        """Replace the live SQP rotation constraints with signed stage bounds."""
        if self._planner is None:
            raise RuntimeError("rotation tolerance requires an SQP planner")
        target = np.asarray(target_rotation, dtype=np.float64)
        frame = np.asarray(tolerance_frame, dtype=np.float64)
        negative = np.asarray(negative_limits, dtype=np.float64)
        positive = np.asarray(positive_limits, dtype=np.float64)
        if target.shape != (3, 3) or frame.shape != (3, 3) or negative.shape != (3,) or positive.shape != (3,):
            raise ValueError("invalid stage rotation tolerance shape")
        if np.any(negative < 0.0) or np.any(positive < 0.0):
            raise ValueError("stage rotation tolerance bounds must be non-negative")
        previous_stage = None if self._tolerance_state is None else self._tolerance_state.active_stage
        if self._tolerance_state is None:
            self._tolerance_state = RotationalToleranceState()
        planned_rotation = target
        baseline = self._planner.baseline if isinstance(self._planner, ShadowSQPPlanner) else self._planner
        if baseline.target is not None:
            planned_rotation = baseline.target.rotation
        self._tolerance_state.configure_stage(
            int(phase),
            target,
            frame,
            np.maximum(negative, positive),
            negative_limits=negative,
            positive_limits=positive,
        )
        if previous_stage is not None and previous_stage != int(phase):
            self._tolerance_state.handle_phase_transition(
                previous_stage,
                int(phase),
                planned_rotation,
                planned_rotation,
            )
        self._tolerance_state.ranged = (
            (negative > 0.0) | (positive > 0.0)
        ) if phase not in (ManipulationPhase.GRASP, ManipulationPhase.RELEASE) else np.zeros(3, dtype=bool)
        self._tolerance_frame = frame.copy()
        self._ranged_axes = self._tolerance_state.ranged.copy()
        tasks = [AxisTask(TaskKind.SPECIFIC) for _ in range(6)]
        for axis, ranged in enumerate(self._ranged_axes):
            if ranged:
                tasks[axis + 3] = AxisTask(
                    TaskKind.FLAT_RANGE,
                    lower=-float(negative[axis]),
                    upper=float(positive[axis]),
                )
        self._axis_tasks = tuple(tasks)

    def _axis_tasks_for_target(self, target_rotation: np.ndarray) -> tuple[AxisTask, ...]:
        if self._tolerance_state is None:
            assert self._axis_tasks is not None
            return self._axis_tasks
        tasks = [AxisTask(TaskKind.SPECIFIC) for _ in range(6)]
        for axis in range(3):
            tasks[axis + 3] = self._tolerance_state.task(
                axis,
                target_rotation,
                target_rotation,
                target_rotation,
            )
        return tuple(tasks)

    def _axis_tasks_for_action(self, measured_q: np.ndarray, transformed_action: np.ndarray) -> tuple[AxisTask, ...] | None:
        """Build baseline tasks from its nominal target; retained for direct tests."""
        if self._tolerance_state is None or self._planner is None:
            return self._axis_tasks
        baseline = self._planner.baseline if isinstance(self._planner, ShadowSQPPlanner) else self._planner
        if baseline.target is None:
            current_rotation = baseline.kinematics.evaluate(
                measured_q,
                include_manipulability=False,
                include_link_points=False,
            ).rotation
        else:
            current_rotation = baseline.target.rotation
        nominal_rotation = rotvec_to_matrix(transformed_action[3:6]) @ current_rotation
        self._tolerance_state.update_intent(transformed_action[3:6], _TELEOP_DT)
        return self._axis_tasks_for_target(nominal_rotation)

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def control_mode(self) -> str:
        return "cartesian" if self._planner is None else "joint"

    def reset(self, measured_q: np.ndarray | None = None) -> None:
        if self._tolerance_state is not None:
            self._tolerance_state.reset_intent()
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
        telemetry: dict[str, object] = {"planner_mode": self.mode}
        if isinstance(self._planner, ShadowSQPPlanner):
            if self._tolerance_state is not None:
                self._tolerance_state.update_intent(transformed[3:6], _TELEOP_DT)
            shadow_plan = self._planner.step(
                measured,
                transformed[:6],
                tolerance_frame=self._tolerance_frame,
                ranged_axes=self._ranged_axes,
                rotation_limits=self._rotation_limits,
                semantic_key=(self.config.shadow_stage if semantic_key is None else semantic_key),
                axis_tasks=(self._axis_tasks if self._tolerance_state is None else None),
                axis_task_factory=(
                    None
                    if self._tolerance_state is None
                    else self._axis_tasks_for_target
                ),
            )
            plan = shadow_plan.baseline
            telemetry.update(
                shadow_anchor_reset=shadow_plan.shadow.anchor_reset,
                shadow_strict_residual_rad=shadow_plan.shadow.non_tolerance_residual_rad,
            )
        else:
            axis_tasks = self._axis_tasks_for_action(measured, transformed)
            plan = self._planner.step(
                measured,
                transformed[:6],
                axis_tasks=axis_tasks,
                tolerance_rotation=self._tolerance_frame,
            )
        telemetry.update(
            status=plan.solver.status,
            feasible=plan.solver.feasible,
            converged=plan.solver.converged,
            iterations=plan.solver.iterations,
            qp_iterations=plan.solver.qp_iterations,
            elapsed_ms=plan.solver.elapsed_ms,
            position_residual=plan.solver.constraints.position_residual,
            rotation_residual=plan.solver.constraints.rotation_residual,
            tolerance_violation=plan.solver.constraints.inequality_violation,
        )
        kinematics = (
            self._planner.baseline.kinematics
            if isinstance(self._planner, ShadowSQPPlanner)
            else self._planner.kinematics
        )
        actual_state = kinematics.evaluate(measured)
        planned_state = kinematics.evaluate(plan.q)
        return PlannedRobotCommand(
            "joint",
            joint_target=plan.q,
            gripper_target=float(transformed[6]),
            telemetry=telemetry,
            actual_pose=_pose_transform(actual_state.position, actual_state.rotation),
            planned_pose=_pose_transform(planned_state.position, planned_state.rotation),
            nominal_pose=_pose_transform(plan.nominal_target.position, plan.nominal_target.rotation),
        )

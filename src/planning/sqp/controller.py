from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from planning.sqp.kinematics import PandaKinematics
from planning.sqp.solver import ConstraintValues, FullSpaceSQPSolver, SQPSolverResult
from planning.sqp.types import SQPSettings
from planning.shadow_reference import project_release_target
from utils.pose import rotvec_to_matrix


class TaskKind(str, Enum):
    SPECIFIC = "specific"
    RANGE = "range"
    FLAT_RANGE = "flat_range"
    FREE = "free"


@dataclass(frozen=True)
class AxisTask:
    kind: TaskKind
    goal: float = 0.0
    lower: float = -1.0
    upper: float = 1.0
    preference_goal: float = 0.0
    preference_weight: float = 0.0


@dataclass(frozen=True)
class TargetPose:
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class SQPObjectiveSettings:
    velocity_weight: float = 0.7
    acceleration_weight: float = 0.5
    jerk_weight: float = 0.3
    joint_limit_weight: float = 0.1
    manipulability_weight: float = 1.0
    tolerance_weight: float = 1.0


@dataclass(frozen=True)
class SQPPlan:
    q: np.ndarray
    target: TargetPose
    solver: SQPSolverResult


def _groove(value: float, *, width: float, quadratic: float) -> float:
    return float(-np.exp(-0.5 * (value / width) ** 2) + quadratic * value * value + 1.0)


def _groove_derivative(value: float, *, width: float, quadratic: float) -> float:
    return float(np.exp(-0.5 * (value / width) ** 2) * value / (width * width) + 2.0 * quadratic * value)


class BaselineSQPPlanner:
    """10 Hz Cartesian-action planner cloned from the baseline SQP path."""

    _RANGED = {TaskKind.RANGE, TaskKind.FLAT_RANGE}

    def __init__(
        self,
        *,
        solver_settings: SQPSettings = SQPSettings(),
        objective_settings: SQPObjectiveSettings = SQPObjectiveSettings(),
        kinematics: PandaKinematics | None = None,
    ) -> None:
        self.kinematics = kinematics or PandaKinematics()
        self.solver = FullSpaceSQPSolver(solver_settings)
        self.objective_settings = objective_settings
        self._target: TargetPose | None = None
        self._xopt: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._previous2: np.ndarray | None = None

    @property
    def target(self) -> TargetPose | None:
        return self._target

    @property
    def optimized_rotation(self) -> np.ndarray | None:
        if self._xopt is None:
            return None
        return self.kinematics.evaluate(self._xopt).rotation

    def reset(self, measured_q: np.ndarray | None = None) -> None:
        self.solver.reset()
        self._xopt = None
        self._previous = None
        self._previous2 = None
        self._target = None
        if measured_q is not None:
            state = self.kinematics.evaluate(measured_q)
            self._target = TargetPose(state.position, state.rotation)

    @staticmethod
    def _tasks(active_dofs: np.ndarray, axis_tasks: tuple[AxisTask, ...] | None) -> tuple[AxisTask, ...]:
        active_dofs = np.asarray(active_dofs, dtype=bool)
        if active_dofs.shape != (6,):
            raise ValueError(f"active_dofs must have shape (6,); got {active_dofs.shape}")
        if axis_tasks is not None:
            if len(axis_tasks) != 6:
                raise ValueError("axis_tasks must contain six tasks")
            return axis_tasks
        return tuple(AxisTask(TaskKind.SPECIFIC if active else TaskKind.FREE) for active in active_dofs)

    def _constraint_evaluation(
        self,
        q: np.ndarray,
        target: TargetPose,
        tasks: tuple[AxisTask, ...],
        tolerance_rotation: np.ndarray | None,
    ) -> tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]]:
        state = self.kinematics.evaluate(q)
        error = self.kinematics.pose_error(state, target.position, target.rotation, tolerance_rotation)
        jacobian = self.kinematics.pose_error_jacobian(state, target.rotation, tolerance_rotation)
        equality_values = list(error[:3])
        equality_rows = [jacobian[index] for index in range(3)]
        strict_rotation = []
        inequality_values = []
        inequality_rows = []
        for index in range(3, 6):
            task = tasks[index]
            value = float(error[index])
            if task.kind is TaskKind.SPECIFIC:
                residual = value - task.goal
                equality_values.append(residual)
                equality_rows.append(jacobian[index])
                strict_rotation.append(residual)
            elif task.kind in self._RANGED:
                if np.isfinite(task.lower):
                    inequality_values.append(value - task.lower)
                    inequality_rows.append(jacobian[index])
                if np.isfinite(task.upper):
                    inequality_values.append(task.upper - value)
                    inequality_rows.append(-jacobian[index])
        inequality = np.asarray(inequality_values, dtype=np.float64)
        values = ConstraintValues(
            equality=np.asarray(equality_values, dtype=np.float64),
            inequality=inequality,
            position_residual=float(np.max(np.abs(error[:3]))),
            rotation_residual=float(np.max(np.abs(strict_rotation))) if strict_rotation else 0.0,
            inequality_violation=float(np.max(np.maximum(-inequality, 0.0))) if inequality.size else 0.0,
        )
        return values, (
            np.asarray(equality_rows, dtype=np.float64).reshape(-1, 7),
            np.asarray(inequality_rows, dtype=np.float64).reshape(-1, 7),
        )

    def _objective(
        self,
        q: np.ndarray,
        seed: np.ndarray,
        previous: np.ndarray,
        previous2: np.ndarray,
        target: TargetPose,
        tasks: tuple[AxisTask, ...],
        tolerance_rotation: np.ndarray | None,
        breakdown: bool,
    ) -> float | tuple[float, dict[str, float]]:
        settings = self.objective_settings
        parts = {
            "velocity": settings.velocity_weight * _groove(float(np.linalg.norm(q - seed)), width=0.1, quadratic=10.0),
            "acceleration": settings.acceleration_weight
            * _groove(float(np.linalg.norm(q - 2.0 * seed + previous)), width=0.1, quadratic=10.0),
            "jerk": settings.jerk_weight
            * _groove(float(np.linalg.norm(q - 3.0 * seed + 3.0 * previous - previous2)), width=0.1, quadratic=10.0),
        }
        centre = 0.5 * (self.kinematics.joint_lower + self.kinematics.joint_upper)
        half_range = 0.5 * (self.kinematics.joint_upper - self.kinematics.joint_lower)
        normalized = (q - centre) / half_range
        parts["joint_limits"] = settings.joint_limit_weight * float(np.sum(np.power(np.abs(normalized), 20)))
        state = self.kinematics.evaluate(q)
        parts["manipulability"] = settings.manipulability_weight * _groove(
            state.manipulability - 1.0, width=0.5, quadratic=0.1
        )
        error = self.kinematics.pose_error(state, target.position, target.rotation, tolerance_rotation)
        for index in range(3, 6):
            task = tasks[index]
            if task.kind in self._RANGED and task.preference_weight > 0.0:
                parts[f"pose_{index}_intent"] = (
                    settings.tolerance_weight
                    * task.preference_weight
                    * _groove(float(error[index] - task.preference_goal), width=0.1, quadratic=10.0)
                )
        total = float(sum(parts.values()))
        return (total, parts) if breakdown else total

    def _objective_gradient(
        self,
        q: np.ndarray,
        seed: np.ndarray,
        previous: np.ndarray,
        previous2: np.ndarray,
        target: TargetPose,
        tasks: tuple[AxisTask, ...],
        tolerance_rotation: np.ndarray | None,
    ) -> np.ndarray:
        settings = self.objective_settings
        gradient = np.zeros(7, dtype=np.float64)

        def add_radial(weight: float, vector: np.ndarray) -> None:
            norm = float(np.linalg.norm(vector))
            if weight == 0.0 or norm <= 1e-14:
                return
            gradient[:] += weight * _groove_derivative(norm, width=0.1, quadratic=10.0) * vector / norm

        add_radial(settings.velocity_weight, q - seed)
        add_radial(settings.acceleration_weight, q - 2.0 * seed + previous)
        add_radial(settings.jerk_weight, q - 3.0 * seed + 3.0 * previous - previous2)

        centre = 0.5 * (self.kinematics.joint_lower + self.kinematics.joint_upper)
        half_range = 0.5 * (self.kinematics.joint_upper - self.kinematics.joint_lower)
        normalized = (q - centre) / half_range
        gradient += (
            settings.joint_limit_weight
            * 20.0
            * np.power(np.abs(normalized), 19)
            * np.sign(normalized)
            / half_range
        )

        state = self.kinematics.evaluate(q)
        error = self.kinematics.pose_error(state, target.position, target.rotation, tolerance_rotation)
        error_jacobian = self.kinematics.pose_error_jacobian(state, target.rotation, tolerance_rotation)
        for index in range(3, 6):
            task = tasks[index]
            if task.kind in self._RANGED and task.preference_weight > 0.0:
                residual = float(error[index] - task.preference_goal)
                gradient += (
                    settings.tolerance_weight
                    * task.preference_weight
                    * _groove_derivative(residual, width=0.1, quadratic=10.0)
                    * error_jacobian[index]
                )

        if settings.manipulability_weight != 0.0:
            base = _groove(state.manipulability - 1.0, width=0.5, quadratic=0.1)
            epsilon = self.solver.settings.derivative_epsilon
            for index in range(7):
                probe = q.copy()
                probe[index] += epsilon
                changed = self.kinematics.evaluate(probe).manipulability
                gradient[index] += settings.manipulability_weight * (
                    _groove(changed - 1.0, width=0.5, quadratic=0.1) - base
                ) / epsilon
        return gradient

    def solve_target(
        self,
        measured_q: np.ndarray,
        target: TargetPose,
        *,
        active_dofs: np.ndarray | None = None,
        axis_tasks: tuple[AxisTask, ...] | None = None,
        tolerance_rotation: np.ndarray | None = None,
    ) -> SQPPlan:
        measured_q = np.asarray(measured_q, dtype=np.float64)
        active = np.ones(6, dtype=bool) if active_dofs is None else np.asarray(active_dofs, dtype=bool)
        tasks = self._tasks(active, axis_tasks)
        seed = measured_q.copy() if self._xopt is None else self._xopt.copy()
        previous = seed if self._previous is None else self._previous
        previous2 = previous if self._previous2 is None else self._previous2
        cache: dict[bytes, tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]]] = {}

        def constraints(q: np.ndarray) -> tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]]:
            key = np.ascontiguousarray(q).tobytes()
            if key not in cache:
                cache[key] = self._constraint_evaluation(q, target, tasks, tolerance_rotation)
            return cache[key]

        result = self.solver.solve(
            lambda q, breakdown: self._objective(
                q, seed, previous, previous2, target, tasks, tolerance_rotation, breakdown
            ),
            lambda q: constraints(q)[0],
            seed,
            self.kinematics.joint_lower,
            self.kinematics.joint_upper,
            lambda q, _values: constraints(q)[1],
            lambda q: self._objective_gradient(
                q, seed, previous, previous2, target, tasks, tolerance_rotation
            ),
        )
        command = result.q.copy() if result.feasible else seed
        if result.feasible:
            self._previous2 = previous.copy()
            self._previous = seed.copy()
            self._xopt = result.q.copy()
        target_rotation = np.asarray(target.rotation).copy()
        ranged = np.asarray([task.kind in self._RANGED for task in tasks[3:]], dtype=bool)
        if tolerance_rotation is not None and np.any(ranged):
            optimized_rotation = self.kinematics.evaluate(command).rotation
            lower = np.asarray([task.lower for task in tasks[3:]], dtype=np.float64)
            upper = np.asarray([task.upper for task in tasks[3:]], dtype=np.float64)
            target_rotation, _ = project_release_target(
                target_rotation,
                optimized_rotation,
                tolerance_rotation,
                ranged,
                lower,
                upper,
            )
        self._target = TargetPose(np.asarray(target.position).copy(), target_rotation)
        return SQPPlan(command, self._target, result)

    def step(
        self,
        measured_q: np.ndarray,
        cartesian_action: np.ndarray,
        *,
        active_dofs: np.ndarray | None = None,
        axis_tasks: tuple[AxisTask, ...] | None = None,
        tolerance_rotation: np.ndarray | None = None,
    ) -> SQPPlan:
        action = np.asarray(cartesian_action, dtype=np.float64)
        if action.shape != (6,):
            raise ValueError(f"Cartesian SQP action must have shape (6,); got {action.shape}")
        active = np.ones(6, dtype=bool) if active_dofs is None else np.asarray(active_dofs, dtype=bool)
        if self._target is None:
            state = self.kinematics.evaluate(measured_q)
            self._target = TargetPose(state.position, state.rotation)
        position = self._target.position + action[:3] * active[:3]
        rotation_increment = action[3:].copy()
        if tolerance_rotation is not None:
            frame = np.asarray(tolerance_rotation, dtype=np.float64)
            rotation_increment = frame @ (frame.T @ rotation_increment * active[3:])
        else:
            rotation_increment *= active[3:]
        target = TargetPose(position, rotvec_to_matrix(rotation_increment) @ self._target.rotation)
        return self.solve_target(
            measured_q,
            target,
            active_dofs=active,
            axis_tasks=axis_tasks,
            tolerance_rotation=tolerance_rotation,
        )

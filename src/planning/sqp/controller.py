from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Callable

import numpy as np

from planning.sqp.kinematics import KinematicState, PandaKinematics
from planning.sqp.solver import ConstraintValues, FullSpaceSQPSolver, SQPSolverResult
from planning.sqp.types import SQPSettings
from planning.shadow_reference import project_release_target
from utils.pose import rotvec_to_matrix


class TaskKind(str, Enum):
    SPECIFIC = "specific"
    RANGE = "range"
    PREFERRED_RANGE = "preferred_range"
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
    instantaneous_activity: float = 0.0
    instantaneous_release_goal: float | None = None
    absolute_lower: float | None = None
    absolute_upper: float | None = None
    absolute_rotation_reference: np.ndarray | None = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class TargetPose:
    position: np.ndarray
    rotation: np.ndarray


@dataclass(frozen=True)
class SQPObjectiveSettings:
    position_weight: float = 50.0
    rotation_weight: float = 10.0
    velocity_weight: float = 0.7
    acceleration_weight: float = 0.5
    jerk_weight: float = 0.3
    joint_limit_weight: float = 0.1
    manipulability_weight: float = 1.0
    self_collision_weight: float = 0.01
    tolerance_weight: float = 1.0
    tolerance_ema_offset: float = 0.05
    tolerance_ema_gain: float = 0.95
    tolerance_ema_power: float = 2.0


@dataclass(frozen=True)
class SQPPlan:
    q: np.ndarray
    target: TargetPose
    nominal_target: TargetPose
    solver: SQPSolverResult


def _groove(value: float, *, width: float, quadratic: float) -> float:
    return float(-np.exp(-0.5 * (value / width) ** 2) + quadratic * value * value + 1.0)


def _groove_derivative(value: float, *, width: float, quadratic: float) -> float:
    return float(np.exp(-0.5 * (value / width) ** 2) * value / (width * width) + 2.0 * quadratic * value)


def _swamp(
    value: np.ndarray,
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    *,
    wall_height: float,
    polynomial: float,
    power: int,
) -> np.ndarray:
    scaled = (2.0 * value - lower - upper) / (upper - lower)
    wall_scale = (-1.0 / np.log(0.05)) ** (1.0 / power)
    exponent = -np.minimum(np.power(np.abs(scaled) / wall_scale, power), 700.0)
    return (wall_height + polynomial * scaled * scaled) * (1.0 - np.exp(exponent)) - 1.0


def _swamp_derivative(
    value: np.ndarray,
    lower: np.ndarray | float,
    upper: np.ndarray | float,
    *,
    wall_height: float,
    polynomial: float,
    power: int,
) -> np.ndarray:
    width = np.asarray(upper, dtype=np.float64) - np.asarray(lower, dtype=np.float64)
    scaled = (2.0 * value - lower - upper) / width
    wall_scale = (-1.0 / np.log(0.05)) ** (1.0 / power)
    absolute = np.abs(scaled)
    raw_exponent = np.power(absolute / wall_scale, power)
    exponent = np.minimum(raw_exponent, 700.0)
    decay = np.exp(-exponent)
    exponent_derivative = (
        power
        * np.power(absolute, power - 1)
        * np.sign(scaled)
        / np.power(wall_scale, power)
    )
    exponent_derivative = np.where(raw_exponent < 700.0, exponent_derivative, 0.0)
    wall_unit = 1.0 - decay
    wall_derivative = decay * exponent_derivative
    envelope = wall_height + polynomial * scaled * scaled
    envelope_derivative = 2.0 * polynomial * scaled
    return (envelope_derivative * wall_unit + envelope * wall_derivative) * (2.0 / width)


@lru_cache(maxsize=None)
def _segment_pair_indices(segment_count: int) -> tuple[np.ndarray, np.ndarray]:
    pairs = tuple(
        (first, second)
        for first in range(segment_count)
        for second in range(first + 2, segment_count)
    )
    return (
        np.fromiter((pair[0] for pair in pairs), dtype=int),
        np.fromiter((pair[1] for pair in pairs), dtype=int),
    )


def _segment_distances(points: np.ndarray) -> np.ndarray:
    """Evaluate every non-adjacent serial-link distance in one NumPy batch."""
    segment_count = points.shape[0] - 1
    first_indices, second_indices = _segment_pair_indices(segment_count)
    if not first_indices.size:
        return np.empty(0, dtype=np.float64)
    a0 = points[first_indices]
    a1 = points[first_indices + 1]
    b0 = points[second_indices]
    b1 = points[second_indices + 1]
    u, v, w = a1 - a0, b1 - b0, a0 - b0
    uu = np.einsum("ij,ij->i", u, u)
    uv = np.einsum("ij,ij->i", u, v)
    vv = np.einsum("ij,ij->i", v, v)
    uw = np.einsum("ij,ij->i", u, w)
    vw = np.einsum("ij,ij->i", v, w)
    denominator = uu * vv - uv * uv
    epsilon = 1e-14

    def distance(s: np.ndarray, t: np.ndarray) -> np.ndarray:
        delta = w + s[:, None] * u - t[:, None] * v
        return np.sqrt(np.einsum("ij,ij->i", delta, delta))

    s_interior = np.divide(
        uv * vw - vv * uw,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > epsilon,
    )
    t_interior = np.divide(
        uu * vw - uv * uw,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > epsilon,
    )
    interior = distance(s_interior, t_interior)
    interior[
        (denominator <= epsilon)
        | (s_interior < 0.0)
        | (s_interior > 1.0)
        | (t_interior < 0.0)
        | (t_interior > 1.0)
    ] = np.inf
    t_at_start = np.clip(
        np.divide(vw, vv, out=np.zeros_like(vv), where=vv > epsilon), 0.0, 1.0
    )
    t_at_end = np.clip(
        np.divide(vw + uv, vv, out=np.zeros_like(vv), where=vv > epsilon), 0.0, 1.0
    )
    s_at_start = np.clip(
        np.divide(-uw, uu, out=np.zeros_like(uu), where=uu > epsilon), 0.0, 1.0
    )
    s_at_end = np.clip(
        np.divide(uv - uw, uu, out=np.zeros_like(uu), where=uu > epsilon), 0.0, 1.0
    )
    return np.minimum.reduce(
        (
            interior,
            distance(np.zeros_like(uu), t_at_start),
            distance(np.ones_like(uu), t_at_end),
            distance(s_at_start, np.zeros_like(vv)),
            distance(s_at_end, np.ones_like(vv)),
        )
    )


class BaselineSQPPlanner:
    """10 Hz Cartesian-action planner cloned from the baseline SQP path."""

    _RANGED = {TaskKind.RANGE, TaskKind.PREFERRED_RANGE, TaskKind.FLAT_RANGE}

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
        return self.kinematics.evaluate(
            self._xopt, include_manipulability=False, include_link_points=False
        ).rotation

    def reset(self, measured_q: np.ndarray | None = None) -> None:
        self.solver.reset()
        self._xopt = None
        self._previous = None
        self._previous2 = None
        self._target = None
        if measured_q is not None:
            state = self.kinematics.evaluate(
                measured_q, include_manipulability=False, include_link_points=False
            )
            self._target = TargetPose(state.position, state.rotation)

    @staticmethod
    def _tasks(active_dofs: np.ndarray, axis_tasks: tuple[AxisTask, ...] | None) -> tuple[AxisTask, ...]:
        active_dofs = np.asarray(active_dofs, dtype=bool)
        if active_dofs.shape != (6,):
            raise ValueError(f"active_dofs must have shape (6,); got {active_dofs.shape}")
        if axis_tasks is not None:
            if len(axis_tasks) != 6:
                raise ValueError("axis_tasks must contain six tasks")
            resolved = []
            for index, task in enumerate(axis_tasks):
                if (
                    index >= 3
                    and task.kind is TaskKind.FLAT_RANGE
                    and task.absolute_lower is not None
                    and task.absolute_upper is not None
                ):
                    task = AxisTask(
                        kind=task.kind,
                        goal=task.goal,
                        lower=task.absolute_lower,
                        upper=task.absolute_upper,
                        preference_goal=task.preference_goal,
                        preference_weight=task.preference_weight,
                        instantaneous_activity=task.instantaneous_activity,
                        instantaneous_release_goal=task.instantaneous_release_goal,
                        absolute_lower=task.absolute_lower,
                        absolute_upper=task.absolute_upper,
                        absolute_rotation_reference=task.absolute_rotation_reference,
                    )
                resolved.append(task)
            return tuple(resolved)
        return tuple(AxisTask(TaskKind.SPECIFIC if active else TaskKind.FREE) for active in active_dofs)

    def _constraint_evaluation(
        self,
        q: np.ndarray,
        target: TargetPose,
        tasks: tuple[AxisTask, ...],
        tolerance_rotation: np.ndarray | None,
        state_provider: Callable[[np.ndarray, bool, bool], KinematicState] | None = None,
    ) -> tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]]:
        state = (
            self.kinematics.evaluate(q, include_manipulability=False, include_link_points=False)
            if state_provider is None
            else state_provider(q, False, False)
        )
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
            row = jacobian[index]
            if task.absolute_rotation_reference is not None:
                absolute_reference = np.asarray(
                    task.absolute_rotation_reference, dtype=np.float64
                )
                absolute_error = self.kinematics.pose_error(
                    state,
                    target.position,
                    absolute_reference,
                    tolerance_rotation,
                )
                absolute_jacobian = self.kinematics.pose_error_jacobian(
                    state,
                    absolute_reference,
                    tolerance_rotation,
                )
                value = float(absolute_error[index])
                row = absolute_jacobian[index]
            if task.kind is TaskKind.SPECIFIC:
                residual = value - task.goal
                equality_values.append(residual)
                equality_rows.append(row)
                strict_rotation.append(residual)
            elif task.kind in self._RANGED:
                if np.isfinite(task.lower):
                    inequality_values.append(value - task.lower)
                    inequality_rows.append(row)
                if np.isfinite(task.upper):
                    inequality_values.append(task.upper - value)
                    inequality_rows.append(-row)
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
        state_provider: Callable[[np.ndarray, bool, bool], KinematicState],
    ) -> float | tuple[float, dict[str, float]]:
        settings = self.objective_settings
        parts = {
            "velocity": settings.velocity_weight * _groove(float(np.linalg.norm(q - seed)), width=0.1, quadratic=10.0),
            "acceleration": settings.acceleration_weight
            * _groove(float(np.linalg.norm(q - 2.0 * seed + previous)), width=0.1, quadratic=10.0),
            "jerk": settings.jerk_weight
            * _groove(float(np.linalg.norm(q - 3.0 * seed + 3.0 * previous - previous2)), width=0.1, quadratic=10.0),
        }
        joint_limits = _swamp(
            q,
            self.kinematics.joint_lower,
            self.kinematics.joint_upper,
            wall_height=10.0,
            polynomial=10.0,
            power=20,
        )
        parts["joint_limits"] = settings.joint_limit_weight * (float(np.sum(joint_limits)) + q.size)
        use_manipulability = settings.manipulability_weight != 0.0
        use_self_collision = settings.self_collision_weight != 0.0
        needs_pose = any(
            task.kind in self._RANGED
            and (
                task.preference_weight > 0.0
                or task.instantaneous_activity > 0.0
                or task.instantaneous_release_goal is not None
            )
            for task in tasks[3:]
        )
        state = state_provider(q, use_manipulability, use_self_collision)
        if use_manipulability:
            assert state.manipulability is not None
            parts["manipulability"] = settings.manipulability_weight * _groove(
                state.manipulability - 1.0, width=0.5, quadratic=0.1
            )
        if use_self_collision:
            assert state.link_points is not None
            clearances = _segment_distances(state.link_points) - 0.05
            collision = _swamp(
                clearances,
                0.02,
                1.5,
                wall_height=60.0,
                polynomial=0.0001,
                power=30,
            )
            parts["self_collision"] = settings.self_collision_weight * (
                float(np.sum(collision)) + collision.size
            ) - 1.70
        if needs_pose:
            error = self.kinematics.pose_error(state, target.position, target.rotation, tolerance_rotation)
            for index in range(3, 6):
                task = tasks[index]
                if task.kind in self._RANGED and (
                    task.preference_weight > 0.0
                    or task.instantaneous_activity > 0.0
                    or task.instantaneous_release_goal is not None
                ):
                    activity = float(np.clip(max(
                        task.instantaneous_activity,
                        task.preference_weight,
                    ), 0.0, 1.0))
                    dynamic_weight = (
                        settings.tolerance_ema_offset
                        + settings.tolerance_ema_gain
                        * activity ** settings.tolerance_ema_power
                    )
                    release_goal = (
                        task.preference_goal
                        if task.instantaneous_release_goal is None
                        else task.instantaneous_release_goal
                    )
                    parts[f"pose_{index}_intent"] = (
                        settings.tolerance_weight
                        * settings.rotation_weight
                        * dynamic_weight
                        * _groove(float(error[index] - release_goal), width=0.1, quadratic=10.0)
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
        state_provider: Callable[[np.ndarray, bool, bool], KinematicState],
        expensive_gradient_cache: dict[str, np.ndarray],
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

        needs_pose = any(
            task.kind in self._RANGED
            and (
                task.preference_weight > 0.0
                or task.instantaneous_activity > 0.0
                or task.instantaneous_release_goal is not None
            )
            for task in tasks[3:]
        )
        if needs_pose:
            state = state_provider(q, False, False)
            error = self.kinematics.pose_error(state, target.position, target.rotation, tolerance_rotation)
            error_jacobian = self.kinematics.pose_error_jacobian(state, target.rotation, tolerance_rotation)
            for index in range(3, 6):
                task = tasks[index]
                if task.kind in self._RANGED and (
                    task.preference_weight > 0.0
                    or task.instantaneous_activity > 0.0
                    or task.instantaneous_release_goal is not None
                ):
                    activity = float(np.clip(max(
                        task.instantaneous_activity,
                        task.preference_weight,
                    ), 0.0, 1.0))
                    dynamic_weight = (
                        settings.tolerance_ema_offset
                        + settings.tolerance_ema_gain
                        * activity ** settings.tolerance_ema_power
                    )
                    release_goal = (
                        task.preference_goal
                        if task.instantaneous_release_goal is None
                        else task.instantaneous_release_goal
                    )
                    residual = float(error[index] - release_goal)
                    gradient += (
                        settings.tolerance_weight
                        * settings.rotation_weight
                        * dynamic_weight
                        * _groove_derivative(residual, width=0.1, quadratic=10.0)
                        * error_jacobian[index]
                    )

        if settings.joint_limit_weight != 0.0:
            gradient += settings.joint_limit_weight * _swamp_derivative(
                q,
                self.kinematics.joint_lower,
                self.kinematics.joint_upper,
                wall_height=10.0,
                polynomial=10.0,
                power=20,
            )

        use_manipulability = settings.manipulability_weight != 0.0
        use_self_collision = settings.self_collision_weight != 0.0
        if use_manipulability or use_self_collision:
            anchor = expensive_gradient_cache.get("anchor")
            cached_gradient = expensive_gradient_cache.get("gradient")
            refresh_distance = self.solver.settings.expensive_gradient_refresh_rad
            if (
                refresh_distance > 0.0
                and anchor is not None
                and cached_gradient is not None
                and float(np.linalg.norm(q - anchor, ord=np.inf)) <= refresh_distance
            ):
                return gradient + cached_gradient

            def expensive_cost(evaluated: KinematicState) -> float:
                total = 0.0
                if use_manipulability:
                    assert evaluated.manipulability is not None
                    total += settings.manipulability_weight * _groove(
                        evaluated.manipulability - 1.0, width=0.5, quadratic=0.1
                    )
                if use_self_collision:
                    assert evaluated.link_points is not None
                    clearances = _segment_distances(evaluated.link_points) - 0.05
                    total += settings.self_collision_weight * float(np.sum(_swamp(
                        clearances,
                        0.02,
                        1.5,
                        wall_height=60.0,
                        polynomial=0.0001,
                        power=30,
                    )))
                return total

            base_state = state_provider(q, use_manipulability, use_self_collision)
            base = expensive_cost(base_state)
            epsilon = self.solver.settings.derivative_epsilon
            expensive_gradient = np.zeros(7, dtype=np.float64)
            for index in range(7):
                probe = q.copy()
                step = epsilon if q[index] + epsilon <= self.kinematics.joint_upper[index] else -epsilon
                if q[index] + step < self.kinematics.joint_lower[index]:
                    step = 0.5 * (self.kinematics.joint_upper[index] - self.kinematics.joint_lower[index])
                if abs(step) <= np.finfo(float).eps:
                    continue
                probe[index] += step
                probe_state = state_provider(probe, use_manipulability, use_self_collision)
                expensive_gradient[index] = (expensive_cost(probe_state) - base) / step
            expensive_gradient_cache["anchor"] = q.copy()
            expensive_gradient_cache["gradient"] = expensive_gradient.copy()
            gradient += expensive_gradient
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
        nominal_target = TargetPose(
            np.asarray(target.position, dtype=np.float64).copy(),
            np.asarray(target.rotation, dtype=np.float64).copy(),
        )
        active = np.ones(6, dtype=bool) if active_dofs is None else np.asarray(active_dofs, dtype=bool)
        tasks = self._tasks(active, axis_tasks)
        seed = measured_q.copy() if self._xopt is None else self._xopt.copy()
        previous = seed if self._previous is None else self._previous
        previous2 = previous if self._previous2 is None else self._previous2
        chart_rotation = (
            tolerance_rotation
            if any(
                task.kind is not TaskKind.SPECIFIC
                or task.absolute_rotation_reference is not None
                for task in tasks[3:]
            )
            else None
        )
        cache: dict[bytes, tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]]] = {}
        state_cache: dict[bytes, KinematicState] = {}
        expensive_gradient_cache: dict[str, np.ndarray] = {}

        def state_for(
            q: np.ndarray,
            include_manipulability: bool,
            include_link_points: bool,
        ) -> KinematicState:
            key = np.ascontiguousarray(q).tobytes()
            cached = state_cache.get(key)
            if (
                cached is not None
                and (not include_manipulability or cached.manipulability is not None)
                and (not include_link_points or cached.link_points is not None)
            ):
                return cached
            evaluated = self.kinematics.evaluate(
                q,
                include_manipulability=include_manipulability,
                include_link_points=include_link_points,
            )
            state_cache[key] = evaluated
            return evaluated

        def constraints(q: np.ndarray) -> tuple[ConstraintValues, tuple[np.ndarray, np.ndarray]]:
            key = np.ascontiguousarray(q).tobytes()
            if key not in cache:
                cache[key] = self._constraint_evaluation(
                    q, target, tasks, chart_rotation, state_for
                )
            return cache[key]

        result = self.solver.solve(
            lambda q, breakdown: self._objective(
                q,
                seed,
                previous,
                previous2,
                target,
                tasks,
                chart_rotation,
                breakdown,
                state_for,
            ),
            lambda q: constraints(q)[0],
            seed,
            self.kinematics.joint_lower,
            self.kinematics.joint_upper,
            lambda q, _values: constraints(q)[1],
            lambda q: self._objective_gradient(
                q,
                seed,
                previous,
                previous2,
                target,
                tasks,
                chart_rotation,
                state_for,
                expensive_gradient_cache,
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
            optimized_rotation = self.kinematics.evaluate(
                command, include_manipulability=False, include_link_points=False
            ).rotation
            lower = np.asarray([task.lower for task in tasks[3:]], dtype=np.float64)
            upper = np.asarray([task.upper for task in tasks[3:]], dtype=np.float64)
            target_rotation, _ = project_release_target(
                target_rotation,
                optimized_rotation,
                tolerance_rotation,
                ranged,
                lower,
                upper,
                reference_rotation=next(
                    (
                        np.asarray(task.absolute_rotation_reference, dtype=np.float64)
                        for task in tasks[3:]
                        if task.absolute_rotation_reference is not None
                    ),
                    target_rotation,
                ),
            )
        self._target = TargetPose(np.asarray(target.position).copy(), target_rotation)
        return SQPPlan(command, self._target, nominal_target, result)

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
            state = self.kinematics.evaluate(
                measured_q, include_manipulability=False, include_link_points=False
            )
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

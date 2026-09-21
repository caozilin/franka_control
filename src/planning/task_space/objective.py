from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from planning.ipopt.types import TargetPose
from utils.pose import (
    nominal_rotation_coordinate_jacobian,
    rotation_tolerance_coordinates,
)

from planning.kinematics import PandaKinematics
from .conditioning import (
    ConditioningGradientConsistency,
    ConditioningMetrics,
    ConditioningSafeguardConfig,
    bound_aware_finite_difference,
    conditioning_activation,
    conditioning_gradient_consistency,
    conditioning_metrics,
    conditioning_penalty,
    conditioning_region,
)
from .losses import (
    groove,
    groove_derivative,
    swamp,
    swamp_derivative,
    swamp_groove,
    swamp_groove_derivative,
)
from .types import AxisTask, LossParameters, TaskKind


def joint_jerk_step(
    q: np.ndarray,
    previous_q: np.ndarray,
    previous_q2: np.ndarray,
    previous_q3: np.ndarray,
) -> float:
    """Return the norm of the standard third joint-position difference."""
    jerk = q - 3.0 * previous_q + 3.0 * previous_q2 - previous_q3
    return float(np.linalg.norm(jerk))


@dataclass(frozen=True)
class ObjectiveSettings:
    # Full source-reference objective. Hard joint bounds in the PANOC
    # projection remain enabled independently of these soft-task weights.
    position_weight: float = 50.0
    rotation_weight: float = 10.0
    velocity_weight: float = 0.7
    jerk_weight: float = 0.3
    # Optional exact quadratic one-step motion loss. Unlike ``velocity_weight``,
    # this does not pass the joint increment through the groove function.
    squared_delta_q_weight: float = 0.0
    joint_limit_weight: float = 0.1
    manipulability_weight: float = 1.0
    self_collision_weight: float = 0.01
    # Current-cycle incremental-release penalty. ``a`` is the current
    # normalized intent/activity signal,
    # so the default live multiplier shown in the UI is ``0.05+0.95a^2``.
    # Released coordinates remain admissible throughout the interval, while
    # this weight controls preference for the commanded nominal increment.
    release_loss_weight: float = 1.0
    release_loss_offset: float = 0.05
    release_loss_gain: float = 0.95
    release_loss_power: float = 2.0
    # Optional cost on only this cycle's extra tolerance release, scaled by
    # the recent nominal-action EMA for that axis. Zero disables this term.
    ema_release_loss_weight: float = 0.0
    # Optional SVD extension; legacy presets remain unchanged.
    sigma_min_weight: float = 0.0
    # J_bar = [J_position / length; J_rotation]. Fixed across task masks.
    sigma_min_length_scale_m: float = 1.0
    conditioning: ConditioningSafeguardConfig = (
        ConditioningSafeguardConfig()
    )

    def __post_init__(self) -> None:
        for name in (
            "squared_delta_q_weight",
            "sigma_min_weight",
            "ema_release_loss_weight",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(self.sigma_min_length_scale_m)
            or self.sigma_min_length_scale_m <= 0
        ):
            raise ValueError("sigma_min_length_scale_m must be finite and positive")

    @classmethod
    def delta_q_squared_sigma_min_legacy(
        cls,
        *,
        motion_weight: float = 1.0,
        sigma_min_weight: float = 1.0,
        length_scale_m: float = 1.0,
        ema_release_loss_weight: float = 0.1,
    ) -> "ObjectiveSettings":
        """Extend delta-q² with negative sigma_min.

        Cost = motion_weight * ||dq||²
               - sigma_min_weight * sigma_min([Jp / length_scale_m; Jr]).
        The defaults are starting weights, not tuned values.
        """
        if not np.isfinite(motion_weight) or motion_weight < 0:
            raise ValueError("motion_weight must be finite and non-negative")
        return replace(
            cls.delta_q_squared_only(),
            squared_delta_q_weight=motion_weight,
            sigma_min_weight=sigma_min_weight,
            sigma_min_length_scale_m=length_scale_m,
            ema_release_loss_weight=ema_release_loss_weight,
        )

    @classmethod
    def delta_q_squared_sigma_min(
        cls,
        *,
        motion_weight: float = 1.0,
        sigma_min_weight: float = 1.0,
        length_scale_m: float = 1.0,
        ema_release_loss_weight: float = 0.1,
    ) -> "ObjectiveSettings":
        """Compatibility alias for the explicit legacy sigma-min profile."""
        return cls.delta_q_squared_sigma_min_legacy(
            motion_weight=motion_weight,
            sigma_min_weight=sigma_min_weight,
            length_scale_m=length_scale_m,
            ema_release_loss_weight=ema_release_loss_weight,
        )

    @classmethod
    def delta_q_squared_conditioning_safeguard(
        cls,
        *,
        motion_weight: float = 1.0,
        conditioning_weight: float = 0.0,
        eta_low: float = 0.05,
        eta_high: float = 0.10,
        sigma_epsilon: float = 1.0e-12,
        finite_difference_relative_step: float = 1.0e-5,
        ema_release_loss_weight: float = 0.1,
    ) -> "ObjectiveSettings":
        """Minimize motion with an optional conditioning safeguard.

        The maintained runtime currently disables this term by default.  A
        positive explicit weight remains available for controlled ablations.
        """
        if not np.isfinite(motion_weight) or motion_weight < 0.0:
            raise ValueError("motion_weight must be finite and non-negative")
        if not np.isfinite(conditioning_weight) or conditioning_weight < 0.0:
            raise ValueError(
                "conditioning_weight must be finite and non-negative",
            )
        return replace(
            cls.delta_q_squared_only(),
            squared_delta_q_weight=motion_weight,
            ema_release_loss_weight=ema_release_loss_weight,
            conditioning=ConditioningSafeguardConfig(
                enabled=conditioning_weight > 0.0,
                eta_low=eta_low,
                eta_high=eta_high,
                weight=conditioning_weight,
                sigma_epsilon=sigma_epsilon,
                finite_difference_relative_step=(
                    finite_difference_relative_step
                ),
            ),
        )

    @classmethod
    def reference_defaults(cls) -> "ObjectiveSettings":
        """Weights in ``ObjectiveMaster::relaxed_ik`` from the author branch."""
        return cls(
            position_weight=50.0,
            rotation_weight=10.0,
            velocity_weight=0.7,
            jerk_weight=0.3,
            joint_limit_weight=0.1,
            manipulability_weight=1.0,
            self_collision_weight=0.01,
        )

    @classmethod
    def kinematic_only(cls) -> "ObjectiveSettings":
        """Use only motion, jerk, and manipulability soft losses."""
        return cls(
            position_weight=0.0,
            rotation_weight=0.0,
            velocity_weight=0.15,
            jerk_weight=0.30,
            squared_delta_q_weight=0.0,
            joint_limit_weight=0.0,
            manipulability_weight=0.25,
            self_collision_weight=0.0,
            release_loss_weight=0.0,
            release_loss_offset=0.0,
            release_loss_gain=0.0,
        )

    @classmethod
    def delta_q_squared_only(cls) -> "ObjectiveSettings":
        """Minimize only ``||q_t - q_(t-1)||²`` under hard constraints."""
        return cls(
            position_weight=0.0,
            rotation_weight=0.0,
            velocity_weight=0.0,
            jerk_weight=0.0,
            squared_delta_q_weight=1.0,
            joint_limit_weight=0.0,
            manipulability_weight=0.0,
            self_collision_weight=0.0,
            release_loss_weight=0.0,
            release_loss_offset=0.0,
            release_loss_gain=0.0,
            ema_release_loss_weight=0.1,
        )


@dataclass(frozen=True)
class MotionState:
    xopt: np.ndarray
    previous_state: np.ndarray
    previous_state2: np.ndarray


def release_loss_multipliers_for_tasks(
    tasks: tuple[AxisTask, ...] | list[AxisTask],
    settings: ObjectiveSettings,
) -> np.ndarray:
    """Return one live multiplier per task; non-ranged tasks are NaN."""
    weights = np.full(len(tasks), np.nan, dtype=float)
    for index, task in enumerate(tasks):
        if task.kind is not TaskKind.FLAT_RANGE:
            continue
        activity = float(np.clip(max(
            task.instantaneous_activity,
            task.preference_weight,
        ), 0.0, 1.0))
        weights[index] = (
            settings.release_loss_offset
            + settings.release_loss_gain
            * activity ** settings.release_loss_power
        )
    return weights


def release_loss_multiplier_for_tasks(
    tasks: tuple[AxisTask, ...] | list[AxisTask],
    settings: ObjectiveSettings,
) -> float:
    """Return the largest live multiplier for compact diagnostics."""
    weights = release_loss_multipliers_for_tasks(tasks, settings)
    finite = weights[np.isfinite(weights)]
    return float(
        np.max(finite) if finite.size else settings.release_loss_offset
    )


class TaskObjective:
    """Weighted-sum objective F(χ(q)) from Equation (5)."""

    def __init__(self, kinematics: PandaKinematics, target: TargetPose, active_dofs: np.ndarray, motion: MotionState, settings: ObjectiveSettings, axis_tasks: tuple[AxisTask, ...] | None = None, tolerance_rotation: np.ndarray | None = None, *, dynamic_tolerance_penalty: bool = False, rotation_release_step=None) -> None:
        self.kinematics = kinematics
        self.target = target
        self.active_dofs = active_dofs
        self.motion = motion
        self.settings = settings
        if min(
            settings.release_loss_weight,
            settings.release_loss_offset,
            settings.release_loss_gain,
        ) < 0.0:
            raise ValueError("Release-loss weights must be non-negative.")
        if settings.release_loss_power <= 0.0:
            raise ValueError("Release-loss power must be positive.")
        self.axis_tasks = axis_tasks or tuple(AxisTask(TaskKind.SPECIFIC if active else TaskKind.FREE) for active in active_dofs)
        self.tolerance_rotation = None if tolerance_rotation is None else np.asarray(tolerance_rotation, dtype=float).copy()
        self.dynamic_tolerance_penalty = bool(dynamic_tolerance_penalty)
        self.rotation_release_step = rotation_release_step
        if len(self.axis_tasks) != 6:
            raise ValueError("TaskObjective requires exactly six Cartesian axis tasks.")
        self.groove_loss = LossParameters(c=0.1, a2=10.0)
        self.range_loss = LossParameters(c=0.02, a1=1.0, a2=0.01, a3=100.0, n=20)
        self.limit_loss = LossParameters(a1=10.0, a2=10.0, n=20)
        self.manip_loss = LossParameters(c=0.5, a2=0.1)
        self.self_collision_loss = LossParameters(a1=60.0, a2=0.0001, n=30)
        # Task functions are shared by all maintained task-space backends, but each task
        # is shifted by its own theoretical minimum.  The optimizer therefore
        # reports a zero-based total loss without changing task gradients.
        self._pose_minima = tuple(self._pose_task_minimum(task) for task in self.axis_tasks)
        self._cached_state_key: tuple[bytes, bool, bool] | None = None
        self._cached_state = None
        self._expensive_gradient_anchor: np.ndarray | None = None
        self._expensive_gradient: np.ndarray | None = None

    def _incremental_release_loss(
        self,
        value: float,
        task: AxisTask,
    ) -> tuple[float, float, float]:
        """Return this-frame extra-release loss and its live proportion.

        ``value`` is already the candidate's physical SO(3) displacement from
        the frozen release-loss reference. It penalizes only extra tolerance
        released by the current IPOPT waypoint, not accumulated stage release.
        At
        activity=1 and W=1 this is exactly the same groove loss and derivative
        as a Specific rotational task. Lower activity scales that same loss
        down without filtering or weakening the nominal VLA action itself.
        """
        activity = float(np.clip(max(
            task.instantaneous_activity,
            task.preference_weight,
        ), 0.0, 1.0))
        dynamic_weight = (
            self.settings.release_loss_offset
            + self.settings.release_loss_gain
            * activity ** self.settings.release_loss_power
        )
        loss = dynamic_weight * (
            float(groove(value, 0.0, self.groove_loss)) + 1.0
        )
        derivative = dynamic_weight * float(
            groove_derivative(value, 0.0, self.groove_loss)
        )
        return float(loss), float(derivative), float(dynamic_weight)

    def _ema_release_loss(
        self,
        value: float,
        task: AxisTask,
    ) -> tuple[float, float]:
        """Penalize only this cycle's extra release, scaled by action EMA."""
        ema = float(np.clip(task.ema_nominal_action, 0.0, 1.0))
        base = float(groove(value, 0.0, self.groove_loss)) + 1.0
        derivative = float(groove_derivative(value, 0.0, self.groove_loss))
        return ema * base, ema * derivative

    def _uses_dynamic_tolerance_penalty(
        self,
        index: int,
        task: AxisTask,
    ) -> bool:
        return (
            self.dynamic_tolerance_penalty
            and self.rotation_release_step is not None
            and index >= 3
            and task.kind is TaskKind.FLAT_RANGE
            and self.rotation_release_step.mask[index - 3]
        )

    @property
    def release_loss_multiplier(self) -> float:
        """Largest live release-loss multiplier among ranged axes."""
        if not self.dynamic_tolerance_penalty:
            return self.settings.release_loss_offset
        return release_loss_multiplier_for_tasks(
            self.axis_tasks[3:], self.settings,
        )

    def _state(
        self,
        q: np.ndarray,
        use_manipulability: bool,
        use_self_collision: bool,
    ):
        key = (
            np.ascontiguousarray(q).tobytes(),
            use_manipulability,
            use_self_collision,
        )
        if self._cached_state_key != key:
            self._cached_state = self.kinematics.evaluate(
                q,
                include_manipulability=use_manipulability,
                include_link_points=use_self_collision,
            )
            self._cached_state_key = key
        return self._cached_state

    def _task_rotation_error(
        self,
        state,
        index: int,
        task: AxisTask,
        planned_error: np.ndarray,
    ) -> float:
        """Return the coordinate in the task's current chart."""
        if index < 3 or task.absolute_rotation_reference is None:
            return float(planned_error[index])
        absolute = self.kinematics.pose_error(
            state,
            self.target.position,
            np.asarray(task.absolute_rotation_reference, dtype=float),
            self.tolerance_rotation,
        )
        return float(absolute[index])

    def _task_rotation_jacobian(
        self,
        state,
        index: int,
        task: AxisTask,
        planned_jacobian: np.ndarray,
    ) -> np.ndarray:
        if index < 3 or task.absolute_rotation_reference is None:
            return planned_jacobian[index]
        absolute = self.kinematics.pose_error_jacobian(
            state,
            np.asarray(task.absolute_rotation_reference, dtype=float),
            self.tolerance_rotation,
        )
        return absolute[index]

    def _pose_task_minimum(self, task: AxisTask) -> float:
        if task.kind is TaskKind.FREE:
            return 0.0
        if task.kind in (TaskKind.SPECIFIC, TaskKind.RANGE):
            return -1.0
        if task.kind is TaskKind.FLAT_RANGE:
            return 0.0
        if task.kind is TaskKind.PREFERRED_RANGE:
            return self._preferred_range_minimum(task)
        raise ValueError(f"Unsupported task kind: {task.kind}")

    def _preferred_range_minimum(self, task: AxisTask) -> float:
        """Numerically obtain the scalar task's global theoretical minimum.

        Swamp-Groove is generally not centred on its preferred goal, so its
        minimum is not always -1.  A dense global scan followed by golden
        refinement makes the shift task-specific while leaving the base loss
        function untouched.
        """
        width = task.upper - task.lower
        margin = max(
            10.0 * max(width, abs(task.goal - task.lower), abs(task.goal - task.upper), self.range_loss.c),
            np.sqrt((self.range_loss.a1 + 2.0 * self.range_loss.a3 + 1.0) / self.range_loss.a2),
        )
        lower = min(task.lower, task.goal) - margin
        upper = max(task.upper, task.goal) + margin
        grid = np.linspace(lower, upper, 2049)
        values = swamp_groove(grid, task.lower, task.upper, task.goal, self.range_loss)
        index = int(np.argmin(values))
        left = grid[max(0, index - 1)]
        right = grid[min(grid.size - 1, index + 1)]
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        for _ in range(64):
            c = right - (right - left) / phi
            d = left + (right - left) / phi
            if swamp_groove(c, task.lower, task.upper, task.goal, self.range_loss) <= swamp_groove(d, task.lower, task.upper, task.goal, self.range_loss):
                right = d
            else:
                left = c
        return float(swamp_groove((left + right) * 0.5, task.lower, task.upper, task.goal, self.range_loss))

    def evaluate(self, q: np.ndarray, include_breakdown: bool = False) -> float | tuple[float, dict[str, float]]:
        use_manipulability = self.settings.manipulability_weight != 0.0
        use_self_collision = self.settings.self_collision_weight != 0.0
        state = self._state(q, use_manipulability, use_self_collision)
        if self.rotation_release_step is None:
            pose_error = self.kinematics.pose_error(
                state, self.target.position, self.target.rotation,
                self.tolerance_rotation,
            )
        else:
            pose_error = np.zeros(6, dtype=float)
            pose_error[:3] = state.position - self.target.position
        incremental_release = None
        if self.rotation_release_step is not None:
            incremental_release = self.rotation_release_step.incremental_release_coordinates(
                state.rotation,
            )
        parts: dict[str, float] = {}
        for index in range(6):
            task = self.axis_tasks[index]
            if task.kind is TaskKind.FREE:
                continue
            if self.rotation_release_step is not None and index >= 3:
                if self._uses_dynamic_tolerance_penalty(index, task):
                    release_loss, _, _ = self._incremental_release_loss(
                        float(incremental_release[index - 3]), task,
                    )
                    parts[f"pose_{index}_incremental_release"] = (
                        self.settings.rotation_weight
                        * self.settings.release_loss_weight
                        * release_loss
                    )
                    ema_loss, _ = self._ema_release_loss(
                        float(incremental_release[index - 3]), task,
                    )
                    parts[f"pose_{index}_ema_release"] = (
                        self.settings.ema_release_loss_weight
                        * ema_loss
                    )
                continue
            weight = self.settings.position_weight if index < 3 else self.settings.rotation_weight
            # Rotational range bounds are signed coordinates in the active
            # chart. Taking abs() here destroys the distinction between the
            # lower and upper sides of a ranged/preferred task.
            nominal_error = float(pose_error[index])
            # Release-mode constraints and gradients are expressed in the
            # current cycle chart. Reusing an absolute stage wall here makes
            # the objective disagree with its
            # gradient (and assigns a large cost to an otherwise admissible
            # postgrasp pose), which can make the first release solve fail its
            # merit line search immediately.
            value_error = (
                self._task_rotation_error(state, index, task, pose_error)
            )
            if task.kind is TaskKind.SPECIFIC:
                value = groove(value_error, task.goal, self.groove_loss)
            elif task.kind is TaskKind.RANGE:
                if np.isfinite(task.lower) and np.isfinite(task.upper):
                    value = swamp(value_error, task.lower, task.upper, self.range_loss)
                elif np.isfinite(task.upper):
                    # Upper one-sided single-step tolerance: only positive
                    # deviations are discouraged.  The range is selected by
                    # the controller before this solve, so it uses the normal
                    # rotation-task weight rather than a stiff boundary boost.
                    violation = max(value_error - task.upper, 0.0)
                    value = groove(violation, 0.0, self.groove_loss)
                elif np.isfinite(task.lower):
                    # Lower one-sided tolerance mirrors the upper case.
                    violation = max(task.lower - value_error, 0.0)
                    value = groove(violation, 0.0, self.groove_loss)
                else:
                    value = None
            elif task.kind is TaskKind.PREFERRED_RANGE:
                value = swamp_groove(value_error, task.lower, task.upper, task.goal, self.range_loss)
            elif task.kind is TaskKind.FLAT_RANGE:
                # A live step-range is an admissible set, not a request to
                # return to its midpoint. Holding (error=0) and every allowed
                # direction therefore have exactly zero loss. Only a newly
                # proposed step beyond either side is discouraged.
                lower_violation = 0.0 if not np.isfinite(task.lower) else max(task.lower - value_error, 0.0)
                upper_violation = 0.0 if not np.isfinite(task.upper) else max(value_error - task.upper, 0.0)
                value = (
                    float(groove(lower_violation, 0.0, self.groove_loss)) + 1.0
                    + float(groove(upper_violation, 0.0, self.groove_loss)) + 1.0
                )
            else:  # defensive guard should a future enum member be added
                raise ValueError(f"Unsupported task kind: {task.kind}")
            if value is not None:
                parts[f"pose_{index}_{task.kind.value}"] = weight * (float(value) - self._pose_minima[index])
            if task.kind in (TaskKind.RANGE, TaskKind.FLAT_RANGE) and task.preference_weight > 0.0:
                # Dynamic VLA intent is deliberately separate from the RANGE
                # wall. It attracts this cycle toward its commanded target,
                # while weight=0 recovers Equally Valid Goals exactly.
                preference = float(groove(
                    nominal_error, task.preference_goal, self.groove_loss,
                )) + 1.0
                parts[f"pose_{index}_intent"] = (
                    weight
                    * self.settings.release_loss_weight
                    * task.preference_weight
                    * preference
                )

        if self.settings.velocity_weight != 0.0:
            parts["velocity"] = self.settings.velocity_weight * (float(groove(np.linalg.norm(q - self.motion.xopt), 0.0, self.groove_loss)) + 1.0)
        if self.settings.jerk_weight != 0.0:
            jerk = joint_jerk_step(
                q,
                self.motion.xopt,
                self.motion.previous_state,
                self.motion.previous_state2,
            )
            parts["jerk"] = self.settings.jerk_weight * (float(groove(jerk, 0.0, self.groove_loss)) + 1.0)
        if self.settings.squared_delta_q_weight != 0.0:
            delta_q = q - self.motion.xopt
            parts["delta_q_squared"] = (
                self.settings.squared_delta_q_weight
                * float(delta_q @ delta_q)
            )
        if self.settings.sigma_min_weight != 0.0:
            parts["sigma_min"] = (
                -self.settings.sigma_min_weight
                * self.minimum_singular_value(state)
            )
        if self.settings.conditioning.enabled:
            parts["conditioning"] = self.conditioning_penalty(state)
        if self.settings.joint_limit_weight != 0.0:
            parts["joint_limits"] = self.settings.joint_limit_weight * (float(np.sum(swamp(q, self.kinematics.handles.joint_lower, self.kinematics.handles.joint_upper, self.limit_loss))) + q.size)
        if use_manipulability:
            assert state.manipulability is not None
            # Constant diagnostic baseline; it leaves the objective gradient
            # and therefore the optimized configuration unchanged.
            parts["manipulability"] = (
                self.settings.manipulability_weight
                * (float(groove(state.manipulability, 1.0, self.manip_loss)) + 1.0)
                - self.kinematics.handles.spec.manipulability_display_offset
            )
        collision_loss = 0.0
        # Equivalent to the author's loops j=0..n-3, k=j+2..n-1.
        # Each link is a centreline capsule whose radius is fixed at 5 cm.
        if use_self_collision:
            assert state.link_points is not None
            clearances = self._segment_distances(state.link_points) - 0.05
            collision_loss = float(np.sum(swamp(clearances, 0.02, 1.5, self.self_collision_loss))) + clearances.size
            # This is a constant shift, hence it does not alter the optimizer
            # gradient or its solution; it only re-zeroes the reported term.
            parts["self_collision"] = (
                self.settings.self_collision_weight * collision_loss
                - self.kinematics.handles.spec.self_collision_display_offset
            )
        total = float(sum(parts.values()))
        return (total, parts) if include_breakdown else total

    def minimum_singular_value(self, state) -> float:
        """Full 6-D metric, independent of pose task masks and target charts.

        Rotating the angular rows into the tool frame is orthogonal and
        preserves the singular values under this isotropic row scaling.
        """
        jacobian = state.jacobian.copy()
        jacobian[:3] /= self.settings.sigma_min_length_scale_m
        return float(np.linalg.svd(jacobian, compute_uv=False)[-1])

    def conditioning_metrics(self, state) -> ConditioningMetrics:
        return conditioning_metrics(
            state.jacobian,
            self.kinematics.handles.spec.joint_velocity_limits,
            sigma_epsilon=self.settings.conditioning.sigma_epsilon,
        )

    def conditioning_penalty(self, state) -> float:
        return conditioning_penalty(
            self.conditioning_metrics(state),
            self.settings.conditioning,
        )

    def conditioning_activation(self, state) -> float:
        return conditioning_activation(
            self.conditioning_metrics(state),
            self.settings.conditioning,
        )

    def conditioning_region(self, state) -> str:
        return conditioning_region(
            self.conditioning_metrics(state),
            self.settings.conditioning,
        )

    def conditioning_gradient(
        self,
        q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        relative_step: float | None = None,
        current_metrics: ConditioningMetrics | None = None,
    ) -> np.ndarray:
        """Differentiate the hinge penalty at the current IPOPT iterate.

        The safeguard is exactly flat when ``eta >= eta_high``. Returning
        its exact zero derivative there avoids probing unrelated nearby
        states without freezing the metric across IPOPT major iterations.
        """
        config = self.settings.conditioning
        if not config.enabled or config.weight == 0.0:
            return np.zeros_like(q)
        metrics = current_metrics
        if metrics is None:
            current_state = self.kinematics.evaluate(
                q,
                include_manipulability=False,
                include_link_points=False,
            )
            metrics = self.conditioning_metrics(current_state)
        if metrics.inverse_condition_index >= config.eta_high:
            return np.zeros_like(q)

        def penalty_at(probe: np.ndarray) -> float:
            probe_state = self.kinematics.evaluate(
                probe,
                include_manipulability=False,
                include_link_points=False,
            )
            return self.conditioning_penalty(probe_state)

        return bound_aware_finite_difference(
            q,
            lower,
            upper,
            penalty_at,
            relative_step=(
                self.settings.conditioning.finite_difference_relative_step
                if relative_step is None else float(relative_step)
            ),
        )

    def conditioning_gradient_consistency(
        self,
        q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> ConditioningGradientConsistency:
        step = self.settings.conditioning.finite_difference_relative_step
        state = self.kinematics.evaluate(
            q,
            include_manipulability=False,
            include_link_points=False,
        )
        metrics = self.conditioning_metrics(state)
        return conditioning_gradient_consistency(
            self.conditioning_gradient(
                q, lower, upper, relative_step=0.5 * step,
                current_metrics=metrics,
            ),
            self.conditioning_gradient(
                q, lower, upper, relative_step=step,
                current_metrics=metrics,
            ),
            self.conditioning_gradient(
                q, lower, upper, relative_step=2.0 * step,
                current_metrics=metrics,
            ),
        )

    def gradient(
        self,
        q: np.ndarray,
        epsilon: float,
        lower: np.ndarray,
        upper: np.ndarray,
        expensive_refresh_distance: float = 0.0,
    ) -> np.ndarray:
        """Hybrid analytic gradient for real-time constrained optimization.

        Pose, motion-history and joint-limit terms are differentiated
        analytically. Manipulability and self-collision retain the original
        forward-difference semantics because they depend on derivatives of
        MuJoCo Jacobians and link geometry.
        """
        use_manipulability = self.settings.manipulability_weight != 0.0
        use_self_collision = self.settings.self_collision_weight != 0.0
        state = self._state(q, use_manipulability, use_self_collision)
        if self.rotation_release_step is None:
            pose_error = self.kinematics.pose_error(
                state,
                self.target.position,
                self.target.rotation,
                self.tolerance_rotation,
            )
        else:
            pose_error = np.zeros(6, dtype=float)
            pose_error[:3] = state.position - self.target.position
        gradient = np.zeros_like(q)
        incremental_release = None
        incremental_release_jacobian = None
        if self.rotation_release_step is not None:
            reference = (
                self.rotation_release_step.release_loss_reference_rotation
            )
            angular_world = state.rotation @ state.jacobian[3:]
            incremental_release = (
                self.rotation_release_step.incremental_release_coordinates(
                state.rotation,
                )
            )
            incremental_release_jacobian = nominal_rotation_coordinate_jacobian(
                state.rotation, reference, angular_world,
                self.rotation_release_step.tolerance_frame,
            )
        pose_jacobian = (
            self.kinematics.pose_error_jacobian(
                state, self.target.rotation, self.tolerance_rotation,
            )
            if self.rotation_release_step is None
            else np.vstack((state.jacobian[:3], incremental_release_jacobian))
        )

        for index, task in enumerate(self.axis_tasks):
            if task.kind is TaskKind.FREE:
                continue
            if self.rotation_release_step is not None and index >= 3:
                if self._uses_dynamic_tolerance_penalty(index, task):
                    _, release_derivative, _ = self._incremental_release_loss(
                        float(incremental_release[index - 3]), task,
                    )
                    gradient += (
                        self.settings.rotation_weight
                        * self.settings.release_loss_weight
                        * release_derivative
                        * incremental_release_jacobian[index - 3]
                    )
                    _, ema_derivative = self._ema_release_loss(
                        float(incremental_release[index - 3]),
                        task,
                    )
                    gradient += (
                        self.settings.ema_release_loss_weight
                        * ema_derivative
                        * incremental_release_jacobian[index - 3]
                    )
                continue
            weight = (
                self.settings.position_weight
                if index < 3
                else self.settings.rotation_weight
            )
            nominal_value = float(pose_error[index])
            value = self._task_rotation_error(
                state, index, task, pose_error,
            )
            task_jacobian = self._task_rotation_jacobian(
                state, index, task, pose_jacobian,
            )
            derivative = 0.0
            if task.kind is TaskKind.SPECIFIC:
                derivative = float(
                    groove_derivative(value, task.goal, self.groove_loss),
                )
            elif task.kind is TaskKind.RANGE:
                if np.isfinite(task.lower) and np.isfinite(task.upper):
                    derivative = float(swamp_derivative(
                        value, task.lower, task.upper, self.range_loss,
                    ))
                elif np.isfinite(task.upper) and value > task.upper:
                    derivative = float(groove_derivative(
                        value - task.upper, 0.0, self.groove_loss,
                    ))
                elif np.isfinite(task.lower) and value < task.lower:
                    derivative = -float(groove_derivative(
                        task.lower - value, 0.0, self.groove_loss,
                    ))
            elif task.kind is TaskKind.PREFERRED_RANGE:
                derivative = float(swamp_groove_derivative(
                    value,
                    task.lower,
                    task.upper,
                    task.goal,
                    self.range_loss,
                ))
            elif task.kind is TaskKind.FLAT_RANGE:
                if np.isfinite(task.lower) and value < task.lower:
                    derivative = -float(groove_derivative(
                        task.lower - value, 0.0, self.groove_loss,
                    ))
                if np.isfinite(task.upper) and value > task.upper:
                    derivative += float(groove_derivative(
                        value - task.upper, 0.0, self.groove_loss,
                    ))
            gradient += weight * derivative * task_jacobian
            if (
                task.kind in (TaskKind.RANGE, TaskKind.FLAT_RANGE)
                and task.preference_weight > 0.0
            ):
                preference_derivative = float(groove_derivative(
                    nominal_value, task.preference_goal, self.groove_loss,
                ))
                gradient += (
                    weight
                    * self.settings.release_loss_weight
                    * task.preference_weight
                    * preference_derivative
                    * pose_jacobian[index]
                )

        def add_radial(weight: float, vector: np.ndarray) -> None:
            norm = float(np.linalg.norm(vector))
            if weight == 0.0 or norm <= 1e-14:
                return
            derivative = float(groove_derivative(
                norm, 0.0, self.groove_loss,
            ))
            gradient[:] += weight * derivative * vector / norm

        add_radial(
            self.settings.velocity_weight,
            q - self.motion.xopt,
        )
        add_radial(
            self.settings.jerk_weight,
            q
            - 3.0 * self.motion.xopt
            + 3.0 * self.motion.previous_state
            - self.motion.previous_state2,
        )
        if self.settings.squared_delta_q_weight != 0.0:
            gradient += (
                2.0
                * self.settings.squared_delta_q_weight
                * (q - self.motion.xopt)
            )
        if self.settings.sigma_min_weight != 0.0:
            # Differentiate the actual SVD scalar at the current iterate.
            # Do not reuse the legacy expensive-gradient cache across nearby
            # q: the weakest singular direction can change there.
            for index in range(q.size):
                lo = max(lower[index], q[index] - epsilon)
                hi = min(upper[index], q[index] + epsilon)
                if hi <= lo:
                    continue
                samples = []
                for coordinate in (lo, hi):
                    probe = q.copy()
                    probe[index] = coordinate
                    probe_state = self.kinematics.evaluate(
                        probe,
                        include_manipulability=False,
                        include_link_points=False,
                    )
                    samples.append(self.minimum_singular_value(probe_state))
                gradient[index] -= self.settings.sigma_min_weight * (
                    samples[1] - samples[0]
                ) / (hi - lo)
        if self.settings.conditioning.enabled:
            gradient += self.conditioning_gradient(
                q,
                lower,
                upper,
                current_metrics=self.conditioning_metrics(state),
            )
        if self.settings.joint_limit_weight != 0.0:
            gradient += (
                self.settings.joint_limit_weight
                * swamp_derivative(
                    q,
                    self.kinematics.handles.joint_lower,
                    self.kinematics.handles.joint_upper,
                    self.limit_loss,
                )
            )

        def expensive_cost(kinematic_state) -> float:
            value = 0.0
            if use_manipulability:
                assert kinematic_state.manipulability is not None
                value += self.settings.manipulability_weight * float(
                    groove(
                        kinematic_state.manipulability,
                        1.0,
                        self.manip_loss,
                    )
                )
            if use_self_collision:
                assert kinematic_state.link_points is not None
                clearances = (
                    self._segment_distances(kinematic_state.link_points)
                    - 0.05
                )
                value += self.settings.self_collision_weight * float(
                    np.sum(swamp(
                        clearances,
                        0.02,
                        1.5,
                        self.self_collision_loss,
                    ))
                )
            return value

        if use_manipulability or use_self_collision:
            if (
                expensive_refresh_distance > 0.0
                and self._expensive_gradient_anchor is not None
                and self._expensive_gradient is not None
                and float(np.linalg.norm(
                    q - self._expensive_gradient_anchor,
                    ord=np.inf,
                )) <= expensive_refresh_distance
            ):
                return gradient + self._expensive_gradient
            base_expensive = expensive_cost(state)
            expensive_gradient = np.zeros_like(q)
            for index in range(q.size):
                step = (
                    epsilon
                    if q[index] + epsilon <= upper[index]
                    else -epsilon
                )
                if q[index] + step < lower[index]:
                    step = 0.5 * (upper[index] - lower[index])
                if abs(step) <= np.finfo(float).eps:
                    continue
                probe = q.copy()
                probe[index] += step
                probe_state = self.kinematics.evaluate(
                    probe,
                    include_manipulability=use_manipulability,
                    include_link_points=use_self_collision,
                )
                expensive_gradient[index] = (
                    expensive_cost(probe_state) - base_expensive
                ) / step
            self._expensive_gradient_anchor = q.copy()
            self._expensive_gradient = expensive_gradient.copy()
            gradient += expensive_gradient
        return gradient

    @staticmethod
    def _segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
        """Euclidean distance between two finite 3-D centreline segments."""
        u, v, w = a1 - a0, b1 - b0, a0 - b0
        uu, uv, vv = float(u @ u), float(u @ v), float(v @ v)
        uw, vw = float(u @ w), float(v @ w)
        denominator = uu * vv - uv * uv
        candidates: list[tuple[float, float]] = []
        # Interior stationary point, when the two centre lines are not parallel.
        if denominator > 1e-14:
            s = (uv * vw - vv * uw) / denominator
            t = (uu * vw - uv * uw) / denominator
            if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
                candidates.append((s, t))
        # The remaining minima are on the four boundary edges of [0,1]^2.
        if vv > 1e-14:
            candidates.extend(((0.0, np.clip(vw / vv, 0.0, 1.0)), (1.0, np.clip((vw + uv) / vv, 0.0, 1.0))))
        if uu > 1e-14:
            candidates.extend(((np.clip(-uw / uu, 0.0, 1.0), 0.0), (np.clip((uv - uw) / uu, 0.0, 1.0), 1.0)))
        if not candidates:
            candidates.append((0.0, 0.0))
        return min(float(np.linalg.norm(w + s * u - t * v)) for s, t in candidates)

    @staticmethod
    @lru_cache(maxsize=None)
    def _segment_pair_indices(
        segment_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cache the fixed serial-link pair topology across objective calls."""
        pairs = tuple(
            (first, second)
            for first in range(segment_count)
            for second in range(first + 2, segment_count)
        )
        return (
            np.fromiter((pair[0] for pair in pairs), dtype=int),
            np.fromiter((pair[1] for pair in pairs), dtype=int),
        )

    @staticmethod
    def _segment_distances(points: np.ndarray) -> np.ndarray:
        """Exact batched counterpart of :meth:`_segment_distance` for 15 pairs.

        The candidate set is identical to the scalar routine: one interior
        stationary point and the four box-boundary minima.  Keeping it in
        NumPy removes hundreds of tiny Python/NumPy calls per objective value.
        """
        segment_count = points.shape[0] - 1
        first_indices, second_indices = (
            TaskObjective._segment_pair_indices(segment_count)
        )
        if not first_indices.size:
            return np.empty(0, dtype=float)
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
        eps = 1e-14

        def distance(s: np.ndarray, t: np.ndarray) -> np.ndarray:
            delta = w + s[:, None] * u - t[:, None] * v
            return np.sqrt(np.einsum("ij,ij->i", delta, delta))

        s_interior = np.divide(uv * vw - vv * uw, denominator, out=np.zeros_like(denominator), where=denominator > eps)
        t_interior = np.divide(uu * vw - uv * uw, denominator, out=np.zeros_like(denominator), where=denominator > eps)
        interior = distance(s_interior, t_interior)
        interior[(denominator <= eps) | (s_interior < 0.0) | (s_interior > 1.0) | (t_interior < 0.0) | (t_interior > 1.0)] = np.inf

        t_at_start = np.clip(np.divide(vw, vv, out=np.zeros_like(vv), where=vv > eps), 0.0, 1.0)
        t_at_end = np.clip(np.divide(vw + uv, vv, out=np.zeros_like(vv), where=vv > eps), 0.0, 1.0)
        s_at_start = np.clip(np.divide(-uw, uu, out=np.zeros_like(uu), where=uu > eps), 0.0, 1.0)
        s_at_end = np.clip(np.divide(uv - uw, uu, out=np.zeros_like(uu), where=uu > eps), 0.0, 1.0)
        return np.minimum.reduce((
            interior,
            distance(np.zeros_like(uu), t_at_start),
            distance(np.ones_like(uu), t_at_end),
            distance(s_at_start, np.zeros_like(vv)),
            distance(s_at_end, np.ones_like(vv)),
        ))

    def penalty_breakdown(self, parts: dict[str, float]) -> dict[str, float]:
        """Group the already zero-based task losses for the percentage chart."""
        tolerance_names = {
            name for name in parts
            if (
                name.endswith("_incremental_release")
                or name.endswith("_ema_release")
                or name.endswith("_intent")
            )
        }
        pose = sum(
            max(0.0, value) for name, value in parts.items()
            if name.startswith("pose_") and name not in tolerance_names
        )
        tolerance = sum(max(0.0, parts[name]) for name in tolerance_names)
        smooth = sum(
            max(0.0, parts.get(name, 0.0))
            for name in (
                "velocity",
                "jerk",
                "delta_q_squared",
            )
        )
        joint_limits = max(0.0, parts.get("joint_limits", 0.0))
        # ``-sigma_min`` can make the raw optimizer objective negative. For
        # the non-negative percentage chart, report its improvement magnitude.
        manipulability = max(0.0, parts.get("manipulability", 0.0)) + max(
            0.0, -parts.get("sigma_min", 0.0),
        ) + max(0.0, parts.get("conditioning", 0.0))
        self_collision = max(0.0, parts.get("self_collision", 0.0))
        return {
            "Pose tracking": max(0.0, pose),
            "Incremental release": tolerance,
            "Motion smoothness": smooth,
            "Joint limits": joint_limits,
            "Manipulability": manipulability,
            "Self collision": self_collision,
        }


def make_controller_objective(
    kinematics: PandaKinematics,
    target: TargetPose,
    active_dofs: np.ndarray,
    motion: MotionState,
    settings: ObjectiveSettings,
    axis_tasks: tuple[AxisTask, ...],
    tolerance_rotation: np.ndarray | None,
    rotation_release_step=None,
) -> TaskObjective:
    """Build the identical live objective used by every optimizer backend."""
    return TaskObjective(
        kinematics,
        target,
        active_dofs,
        motion,
        settings,
        axis_tasks,
        tolerance_rotation,
        dynamic_tolerance_penalty=True,
        rotation_release_step=rotation_release_step,
    )

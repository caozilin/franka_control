from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np

from planning.shadow_reference import ShadowOrientationDiagnostics, ShadowOrientationReference
from planning.sqp.controller import AxisTask, BaselineSQPPlanner, SQPPlan, TargetPose, TaskKind


DEFAULT_ROTATION_LIMITS = np.radians(np.array([30.0, 30.0, 45.0], dtype=np.float64))


@dataclass(frozen=True)
class ShadowSQPPlan:
    baseline: SQPPlan
    shadow: ShadowOrientationDiagnostics

    @property
    def q(self) -> np.ndarray:
        return self.baseline.q

    @property
    def target(self) -> TargetPose:
        return self.baseline.target


class ShadowSQPPlanner:
    """10 Hz composition of a shadow correction and the baseline optimizer."""

    def __init__(
        self,
        baseline: BaselineSQPPlanner | None = None,
        shadow: ShadowOrientationReference | None = None,
    ) -> None:
        self.baseline = baseline or BaselineSQPPlanner()
        self.shadow = shadow or ShadowOrientationReference()

    def reset(self, measured_q: np.ndarray | None = None) -> None:
        self.baseline.reset(measured_q)
        self.shadow = ShadowOrientationReference()

    def step(
        self,
        measured_q: np.ndarray,
        cartesian_action: np.ndarray,
        *,
        tolerance_frame: np.ndarray,
        ranged_axes: np.ndarray,
        semantic_key: Hashable,
        rotation_limits: np.ndarray = DEFAULT_ROTATION_LIMITS,
        active_dofs: np.ndarray | None = None,
        axis_tasks: tuple[AxisTask, ...] | None = None,
    ) -> ShadowSQPPlan:
        action = np.asarray(cartesian_action, dtype=np.float64)
        active = np.ones(6, dtype=bool) if active_dofs is None else np.asarray(active_dofs, dtype=bool)
        ranged = np.asarray(ranged_axes, dtype=bool)
        limits = np.asarray(rotation_limits, dtype=np.float64)
        if self.baseline.target is None:
            self.baseline.reset(measured_q)
        assert self.baseline.target is not None
        optimized_rotation = self.baseline.optimized_rotation
        if optimized_rotation is None:
            optimized_rotation = self.baseline.kinematics.evaluate(measured_q).rotation
        corrected_rotation, diagnostics = self.shadow.advance(
            optimized_rotation,
            action[3:6],
            tolerance_frame,
            ranged,
            semantic_key=semantic_key,
        )
        if axis_tasks is None:
            generated = []
            for index in range(6):
                if index >= 3 and ranged[index - 3]:
                    generated.append(
                        AxisTask(TaskKind.FLAT_RANGE, lower=-limits[index - 3], upper=limits[index - 3])
                    )
                else:
                    generated.append(AxisTask(TaskKind.SPECIFIC if active[index] else TaskKind.FREE))
            axis_tasks = tuple(generated)
        target = TargetPose(
            self.baseline.target.position + action[:3] * active[:3],
            corrected_rotation,
        )
        plan = self.baseline.solve_target(
            measured_q,
            target,
            active_dofs=active,
            axis_tasks=axis_tasks,
            tolerance_rotation=tolerance_frame,
        )
        return ShadowSQPPlan(plan, diagnostics)

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np

from planning.kinematics import PandaKinematics
from planning.task_space import ObjectiveSettings
from planning.tolerance.release_state import RotationReleaseState
from planning.tolerance.runtime import advance_reference_pose, solve_stage_relative_target
from planning.tolerance.state import RotationalToleranceState

from .adapter import IpoptSettings
from .controller import MAX_CONSECUTIVE_SOLVER_FAILURES, IpoptDiagnostics, IpoptIK
from .types import TargetPose


class IpoptConsecutiveFailureError(RuntimeError):
    """Raised when the maintained optimizer rejects three consecutive solves."""


@dataclass(frozen=True)
class IpoptPlan:
    q: np.ndarray
    target: TargetPose
    diagnostics: IpoptDiagnostics


class IpoptPlanner:
    """10 Hz real-robot wrapper around the maintained single-step IPOPT IK."""

    def __init__(
        self,
        *,
        settings: IpoptSettings = IpoptSettings(),
        objective_settings: ObjectiveSettings | None = None,
        kinematics: PandaKinematics | None = None,
    ) -> None:
        self.kinematics = kinematics or PandaKinematics()
        self.controller = IpoptIK(
            self.kinematics,
            settings,
            objective_settings
            or ObjectiveSettings.delta_q_squared_conditioning_safeguard(),
        )
        self.release_state = RotationReleaseState(
            rotation_for_joint=lambda q: self.kinematics.evaluate(
                q,
                include_manipulability=False,
                include_link_points=False,
            ).rotation,
        )
        self._target: TargetPose | None = None
        self._last_command: np.ndarray | None = None

    @property
    def target(self) -> TargetPose | None:
        return self._target

    @property
    def optimized_rotation(self) -> np.ndarray | None:
        if self._last_command is None:
            return None
        return self.kinematics.evaluate(
            self._last_command,
            include_manipulability=False,
            include_link_points=False,
        ).rotation

    def reset(self, measured_q: np.ndarray | None = None) -> None:
        self.controller.reset()
        self.release_state.reset()
        self._target = None
        self._last_command = None
        if measured_q is not None:
            measured = np.asarray(measured_q, dtype=np.float64)
            state = self.kinematics.evaluate(
                measured,
                include_manipulability=False,
                include_link_points=False,
            )
            self._target = TargetPose(state.position, state.rotation)
            self._last_command = measured.copy()

    def _ensure_reference(self, measured_q: np.ndarray) -> None:
        if self._target is not None:
            return
        state = self.kinematics.evaluate(
            measured_q,
            include_manipulability=False,
            include_link_points=False,
        )
        self._target = TargetPose(state.position, state.rotation)
        self._last_command = np.asarray(measured_q, dtype=np.float64).copy()

    def step(
        self,
        measured_q: np.ndarray,
        cartesian_action: np.ndarray,
        *,
        tolerance: RotationalToleranceState | None = None,
        semantic_key: Hashable | None = None,
        dt: float = 0.1,
    ) -> IpoptPlan:
        measured = np.asarray(measured_q, dtype=np.float64)
        action = np.asarray(cartesian_action, dtype=np.float64)
        if measured.shape != (7,) or action.shape != (6,):
            raise ValueError("IPOPT planning requires measured_q (7,) and action (6,)")
        self._ensure_reference(measured)
        assert self._target is not None

        stage_key = tolerance.active_stage if semantic_key is None and tolerance is not None else semantic_key
        if tolerance is not None and self.release_state.stage_key != stage_key:
            optimized = self.optimized_rotation
            if optimized is None:
                optimized = self.kinematics.evaluate(
                    measured,
                    include_manipulability=False,
                    include_link_points=False,
                ).rotation
            self.release_state.begin_stage(
                stage_key,
                optimized_rotation=optimized,
                nominal_rotation=self._target.rotation,
            )

        position, rotation = advance_reference_pose(
            self._target.position,
            self._target.rotation,
            action,
            tolerance=tolerance,
            dt=dt if tolerance is not None else None,
        )
        self._target = TargetPose(position, rotation)

        if tolerance is None:
            for axis in range(6):
                self.controller.set_axis_task(axis, None)
            q, diagnostics = self.controller.solve(
                measured,
                self._target,
                np.ones(6, dtype=bool),
            )
        else:
            q, diagnostics, _, _, _ = solve_stage_relative_target(
                self.controller,
                measured,
                self._target,
                tolerance,
                self.release_state,
                np.ones(6, dtype=bool),
            )

        self._last_command = np.asarray(q, dtype=np.float64).copy()
        if diagnostics.consecutive_failures >= MAX_CONSECUTIVE_SOLVER_FAILURES:
            raise IpoptConsecutiveFailureError(
                "IPOPT rejected three consecutive 10 Hz solves; the last accepted "
                "joint command is being held. "
                f"Last status: {diagnostics.status}; "
                f"position_residual={diagnostics.position_residual:.3e} "
                f"(limit={self.controller.settings.position_tolerance:.3e}); "
                f"strict_rotation_residual={diagnostics.strict_rotation_residual:.3e} "
                f"(limit={self.controller.settings.rotation_tolerance:.3e}); "
                f"inequality_violation={diagnostics.effective_constraint_violation:.3e} "
                f"(limit={self.controller.settings.inequality_tolerance:.3e}); "
                f"iterations={diagnostics.iterations}; "
                f"solve_ms={diagnostics.elapsed_ms:.2f}"
            )
        return IpoptPlan(self._last_command.copy(), self._target, diagnostics)

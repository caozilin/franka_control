"""Stage-relative SO(3) rotation release state and frozen cycle snapshot."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable

import numpy as np

from utils.pose import (
    rotation_tolerance_coordinates,
    stage_relative_rotation_coordinates,
    stage_reference_rotation,
)
from .projection import constraint_consistent_release_loss_reference


@dataclass(frozen=True)
class RotationReleaseStep:
    """Immutable rotational constraint data for one outer control cycle."""

    nominal_rotation: np.ndarray
    stage_handoff_rotation: np.ndarray
    stage_handoff_nominal_rotation: np.ndarray
    tolerance_frame: np.ndarray
    mask: np.ndarray
    lower_limits: np.ndarray
    upper_limits: np.ndarray
    effective_box_baseline_error: np.ndarray
    effective_lower_limits: np.ndarray
    effective_upper_limits: np.ndarray
    carried_rotation: np.ndarray
    release_loss_reference_rotation: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "nominal_rotation",
            "stage_handoff_rotation",
            "stage_handoff_nominal_rotation",
            "tolerance_frame",
            "mask",
            "lower_limits",
            "upper_limits",
            "effective_box_baseline_error",
            "effective_lower_limits",
            "effective_upper_limits",
            "carried_rotation",
            "release_loss_reference_rotation",
        ):
            value = np.asarray(getattr(self, name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def stage_reference_rotation(self) -> np.ndarray:
        return stage_reference_rotation(
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            self.nominal_rotation,
        )

    def stage_relative_coordinates(self, rotation: np.ndarray) -> np.ndarray:
        """Return total stage release used only for hard admissibility."""
        return stage_relative_rotation_coordinates(
            rotation,
            self.nominal_rotation,
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            self.tolerance_frame,
        )

    def incremental_release_coordinates(
        self, rotation: np.ndarray,
    ) -> np.ndarray:
        """Return this cycle's additional release relative to frozen R_loss."""
        return rotation_tolerance_coordinates(
            rotation,
            self.release_loss_reference_rotation,
            self.tolerance_frame,
        )

    def predicted_violation(self, coordinates: np.ndarray) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float)
        return np.where(
            self.mask,
            np.maximum.reduce((
                self.lower_limits - values,
                np.zeros(3),
                values - self.upper_limits,
            )),
            0.0,
        )


@dataclass(frozen=True)
class RotationReleaseCandidate:
    """Diagnostics for one candidate under a frozen release step."""

    rotation: np.ndarray
    stage_relative_error_fixed_xyz_rad: np.ndarray
    stage_relative_tolerance_release_rad: np.ndarray
    predicted_tolerance_violation_rad: np.ndarray
    incremental_release_rad: np.ndarray


@dataclass
class RotationReleaseState:
    """Persistent physical state for one active manipulation stage."""

    stage_key: Hashable | None = None
    stage_handoff_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3),
    )
    stage_handoff_nominal_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3),
    )
    accepted_rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    accepted_nominal_rotation: np.ndarray = field(
        default_factory=lambda: np.eye(3),
    )
    rotation_for_joint: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def reset(self) -> None:
        self.stage_key = None
        self.stage_handoff_rotation = np.eye(3)
        self.stage_handoff_nominal_rotation = np.eye(3)
        self.accepted_rotation = np.eye(3)
        self.accepted_nominal_rotation = np.eye(3)

    def begin_stage(
        self,
        stage_key: Hashable,
        *,
        optimized_rotation: np.ndarray,
        nominal_rotation: np.ndarray,
        rotation_for_joint: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Capture one physical/nominal handoff within a subtask.

        Within a subtask, the physical handoff anchors the next stage so a
        Pre-grasp realization is transported through Post-grasp. Subtask
        boundaries must use :meth:`rebase_subtask` so the physical and nominal
        origins are changed atomically instead of restoring an old nominal
        orientation.
        """
        optimized = np.asarray(optimized_rotation, dtype=float)
        nominal = np.asarray(nominal_rotation, dtype=float)
        if optimized.shape != (3, 3) or nominal.shape != (3, 3):
            raise ValueError("Handoff rotations must have shape (3, 3)")
        self.stage_key = stage_key
        self.stage_handoff_rotation = optimized.copy()
        self.stage_handoff_nominal_rotation = nominal.copy()
        self.accepted_rotation = optimized.copy()
        self.accepted_nominal_rotation = nominal.copy()
        if rotation_for_joint is not None:
            self.rotation_for_joint = rotation_for_joint

    def rebase_subtask(
        self,
        stage_key: Hashable,
        rotation: np.ndarray,
    ) -> None:
        """Start a subtask with zero accumulated rotational release."""
        anchor = np.asarray(rotation, dtype=float)
        if anchor.shape != (3, 3) or not np.all(np.isfinite(anchor)):
            raise ValueError("Subtask rotation must be a finite 3x3 matrix")
        self.stage_key = stage_key
        self.stage_handoff_rotation = anchor.copy()
        self.stage_handoff_nominal_rotation = anchor.copy()
        self.accepted_rotation = anchor.copy()
        self.accepted_nominal_rotation = anchor.copy()

    def prepare_cycle(
        self,
        nominal_rotation: np.ndarray,
        tolerance_frame: np.ndarray,
        mask: np.ndarray,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
    ) -> RotationReleaseStep:
        """Freeze the current spatial release and effective hard box."""
        nominal = np.asarray(nominal_rotation, dtype=float)
        frame = np.asarray(tolerance_frame, dtype=float)
        ranged = np.asarray(mask, dtype=bool)
        lower = np.asarray(lower_limits, dtype=float)
        upper = np.asarray(upper_limits, dtype=float)
        if nominal.shape != (3, 3) or frame.shape != (3, 3):
            raise ValueError("Release rotations and frame must have shape (3, 3)")
        if ranged.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("Release mask and limits must have shape (3,)")
        if np.any(lower > 0.0) or np.any(upper < 0.0) or np.any(lower > upper):
            raise ValueError("Release limits must satisfy lower <= 0 <= upper")
        accepted_reference = stage_reference_rotation(
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            self.accepted_nominal_rotation,
        )
        current_reference = stage_reference_rotation(
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            nominal,
        )
        # The project defines release as the left/spatial residual
        # D = R_actual @ R_reference.T.  Carry D unchanged while nominal
        # reference motion advances; the factors do not generally commute.
        accepted_spatial_release = (
            self.accepted_rotation @ accepted_reference.T
        )
        carried_rotation = accepted_spatial_release @ current_reference
        loss_reference, _ = constraint_consistent_release_loss_reference(
            current_reference,
            carried_rotation,
            frame,
            ranged,
        )
        # Re-express the transported physical release in this cycle's frozen
        # tolerance chart; fixed-XYZ coordinates are never carried as state.
        effective_box_baseline_error = stage_relative_rotation_coordinates(
            carried_rotation,
            nominal,
            self.stage_handoff_rotation,
            self.stage_handoff_nominal_rotation,
            frame,
        )
        return RotationReleaseStep(
            nominal_rotation=nominal.copy(),
            stage_handoff_rotation=self.stage_handoff_rotation.copy(),
            stage_handoff_nominal_rotation=(
                self.stage_handoff_nominal_rotation.copy()
            ),
            tolerance_frame=frame.copy(),
            mask=ranged.copy(),
            lower_limits=lower.copy(),
            upper_limits=upper.copy(),
            effective_box_baseline_error=effective_box_baseline_error.copy(),
            effective_lower_limits=np.minimum(
                lower, effective_box_baseline_error,
            ),
            effective_upper_limits=np.maximum(
                upper, effective_box_baseline_error,
            ),
            carried_rotation=carried_rotation.copy(),
            release_loss_reference_rotation=loss_reference.copy(),
        )

    def candidate(
        self,
        q: np.ndarray,
        step: RotationReleaseStep,
    ) -> RotationReleaseCandidate:
        """Evaluate a physical or joint-space candidate in a frozen cycle."""
        command = np.asarray(q, dtype=float)
        if command.shape == (3, 3):
            rotation = command
        elif self.rotation_for_joint is not None:
            rotation = np.asarray(self.rotation_for_joint(command), dtype=float)
        else:
            raise ValueError("A joint-to-rotation evaluator is required")
        error = step.stage_relative_coordinates(rotation)
        incremental_release = step.incremental_release_coordinates(rotation)
        return RotationReleaseCandidate(
            rotation=rotation.copy(),
            stage_relative_error_fixed_xyz_rad=error.copy(),
            stage_relative_tolerance_release_rad=np.where(
                step.mask, error, 0.0,
            ),
            predicted_tolerance_violation_rad=step.predicted_violation(error),
            incremental_release_rad=incremental_release.copy(),
        )

    def commit(
        self, q_solution: np.ndarray, step: RotationReleaseStep,
    ) -> None:
        """Atomically accept a publishable physical/nominal snapshot pair."""
        command = np.asarray(q_solution, dtype=float)
        if command.shape == (3, 3):
            accepted = command.copy()
        elif self.rotation_for_joint is not None:
            accepted = np.asarray(
                self.rotation_for_joint(command), dtype=float,
            ).copy()
        else:
            raise ValueError("A joint-to-rotation evaluator is required")
        if accepted.shape != (3, 3):
            raise ValueError("Accepted rotation must have shape (3, 3)")
        self.accepted_rotation = accepted
        self.accepted_nominal_rotation = step.nominal_rotation.copy()


__all__ = (
    "RotationReleaseCandidate",
    "RotationReleaseState",
    "RotationReleaseStep",
)

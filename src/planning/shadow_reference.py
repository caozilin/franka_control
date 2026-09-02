from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from utils.pose import (
    rotation_from_tolerance_coordinates,
    rotation_tolerance_coordinates,
    rotvec_to_matrix,
)


@dataclass(frozen=True)
class ShadowOrientationDiagnostics:
    raw_bias_tolerance_frame: np.ndarray
    kept_bias_tolerance_frame: np.ndarray
    corrected_bias_tolerance_frame: np.ndarray
    non_tolerance_residual_rad: float
    anchor_reset: bool


def project_historical_tolerance_bias(
    raw_rotation: np.ndarray,
    shadow_rotation: np.ndarray,
    stage_target_rotation: np.ndarray,
    tolerance_frame: np.ndarray,
    ranged_axes: np.ndarray,
) -> tuple[np.ndarray, ShadowOrientationDiagnostics]:
    """Keep optimizer history only on released axes in one fixed XYZ chart."""
    raw = np.asarray(raw_rotation, dtype=np.float64)
    shadow = np.asarray(shadow_rotation, dtype=np.float64)
    stage_target = np.asarray(stage_target_rotation, dtype=np.float64)
    frame = np.asarray(tolerance_frame, dtype=np.float64)
    mask = np.asarray(ranged_axes, dtype=bool)
    if (
        raw.shape != (3, 3)
        or shadow.shape != (3, 3)
        or stage_target.shape != (3, 3)
        or frame.shape != (3, 3)
        or mask.shape != (3,)
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(shadow))
        or not np.all(np.isfinite(stage_target))
        or not np.all(np.isfinite(frame))
    ):
        raise ValueError("invalid shadow orientation state")

    raw_coordinates = rotation_tolerance_coordinates(raw, stage_target, frame)
    shadow_coordinates = rotation_tolerance_coordinates(shadow, stage_target, frame)
    corrected, corrected_bias_coordinates = project_release_target(
        shadow,
        raw,
        frame,
        mask,
        reference_rotation=stage_target,
    )
    raw_bias_coordinates = raw_coordinates - shadow_coordinates
    residual = (
        float(np.max(np.abs(corrected_bias_coordinates[~mask])))
        if np.any(~mask)
        else 0.0
    )
    if residual > 1.0e-9:
        raise AssertionError(
            "Zero-tolerance shadow retained a non-tolerance component: "
            f"{residual:.3e} rad"
        )
    return corrected, ShadowOrientationDiagnostics(
        raw_bias_tolerance_frame=raw_bias_coordinates.copy(),
        kept_bias_tolerance_frame=corrected_bias_coordinates.copy(),
        corrected_bias_tolerance_frame=corrected_bias_coordinates.copy(),
        non_tolerance_residual_rad=residual,
        anchor_reset=False,
    )


def project_release_target(
    planned_rotation: np.ndarray,
    optimized_rotation: np.ndarray,
    tolerance_frame: np.ndarray,
    ranged_axes: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    reference_rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge a candidate into a nominal pose using one fixed XYZ chart."""
    planned = np.asarray(planned_rotation, dtype=np.float64)
    optimized = np.asarray(optimized_rotation, dtype=np.float64)
    frame = np.asarray(tolerance_frame, dtype=np.float64)
    ranged = np.asarray(ranged_axes, dtype=bool)
    reference = (
        planned
        if reference_rotation is None
        else np.asarray(reference_rotation, dtype=np.float64)
    )
    if (
        planned.shape != (3, 3)
        or optimized.shape != (3, 3)
        or frame.shape != (3, 3)
        or reference.shape != (3, 3)
        or ranged.shape != (3,)
    ):
        raise ValueError("invalid projected release state")

    planned_coordinates = rotation_tolerance_coordinates(planned, reference, frame)
    optimized_coordinates = rotation_tolerance_coordinates(optimized, reference, frame)
    projected_coordinates = planned_coordinates.copy()
    projected_coordinates[ranged] = optimized_coordinates[ranged]
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise ValueError("projected release bounds must be supplied together")
        lower_array = np.asarray(lower, dtype=np.float64)
        upper_array = np.asarray(upper, dtype=np.float64)
        if lower_array.shape != (3,) or upper_array.shape != (3,):
            raise ValueError("projected release bounds must contain three axes")
        projected_coordinates[ranged] = np.clip(
            projected_coordinates[ranged], lower_array[ranged], upper_array[ranged]
        )
    projected = rotation_from_tolerance_coordinates(
        reference, frame, projected_coordinates
    )
    return projected, projected_coordinates - planned_coordinates


@dataclass
class ShadowOrientationReference:
    """Optimizer-independent zero-tolerance shadow state."""

    shadow_rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    _semantic_key: Hashable | None = field(default=None, init=False, repr=False)
    _ranged_axes: np.ndarray | None = field(default=None, init=False, repr=False)
    _tolerance_frame: np.ndarray | None = field(default=None, init=False, repr=False)
    _stage_target: np.ndarray | None = field(default=None, init=False, repr=False)

    def reset(
        self,
        optimized_rotation: np.ndarray,
        *,
        semantic_key: Hashable,
        ranged_axes: np.ndarray,
        stage_target_rotation: np.ndarray,
        tolerance_frame: np.ndarray,
    ) -> None:
        self.shadow_rotation = np.asarray(optimized_rotation, dtype=np.float64).copy()
        self._semantic_key = semantic_key
        self._ranged_axes = np.asarray(ranged_axes, dtype=bool).copy()
        self._stage_target = np.asarray(stage_target_rotation, dtype=np.float64).copy()
        self._tolerance_frame = np.asarray(tolerance_frame, dtype=np.float64).copy()

    def advance(
        self,
        optimized_rotation: np.ndarray,
        rotation_action: np.ndarray,
        stage_target_rotation: np.ndarray,
        tolerance_frame: np.ndarray,
        ranged_axes: np.ndarray,
        *,
        semantic_key: Hashable,
    ) -> tuple[np.ndarray, ShadowOrientationDiagnostics]:
        optimized = np.asarray(optimized_rotation, dtype=np.float64)
        action = np.asarray(rotation_action, dtype=np.float64)
        stage_target = np.asarray(stage_target_rotation, dtype=np.float64)
        frame = np.asarray(tolerance_frame, dtype=np.float64)
        mask = np.asarray(ranged_axes, dtype=bool)
        context_changed = (
            self._ranged_axes is None
            or semantic_key != self._semantic_key
            or not np.array_equal(mask, self._ranged_axes)
        )
        if context_changed:
            self.reset(
                optimized,
                semantic_key=semantic_key,
                ranged_axes=mask,
                stage_target_rotation=stage_target,
                tolerance_frame=frame,
            )
        else:
            self._stage_target = stage_target.copy()
            self._tolerance_frame = frame.copy()

        increment = rotvec_to_matrix(action)
        raw_proposal = increment @ optimized
        shadow_proposal = increment @ self.shadow_rotation
        corrected, diagnostics = project_historical_tolerance_bias(
            raw_proposal, shadow_proposal, stage_target, frame, mask
        )
        self.shadow_rotation = shadow_proposal
        return corrected, ShadowOrientationDiagnostics(
            raw_bias_tolerance_frame=diagnostics.raw_bias_tolerance_frame,
            kept_bias_tolerance_frame=diagnostics.kept_bias_tolerance_frame,
            corrected_bias_tolerance_frame=diagnostics.corrected_bias_tolerance_frame,
            non_tolerance_residual_rad=diagnostics.non_tolerance_residual_rad,
            anchor_reset=context_changed,
        )

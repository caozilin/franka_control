from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from utils.pose import matrix_to_rotvec, rotvec_to_matrix


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
    tolerance_frame: np.ndarray,
    ranged_axes: np.ndarray,
) -> tuple[np.ndarray, ShadowOrientationDiagnostics]:
    """Keep optimizer history only on axes explicitly released by the mask."""
    raw = np.asarray(raw_rotation, dtype=np.float64)
    shadow = np.asarray(shadow_rotation, dtype=np.float64)
    frame = np.asarray(tolerance_frame, dtype=np.float64)
    mask = np.asarray(ranged_axes, dtype=bool)
    if raw.shape != (3, 3) or shadow.shape != (3, 3) or frame.shape != (3, 3) or mask.shape != (3,):
        raise ValueError("invalid shadow orientation state")

    raw_coordinates = matrix_to_rotvec(frame.T @ (raw @ shadow.T) @ frame)
    kept_coordinates = mask.astype(np.float64) * raw_coordinates
    corrected = frame @ rotvec_to_matrix(kept_coordinates) @ frame.T @ shadow
    corrected_coordinates = matrix_to_rotvec(frame.T @ (corrected @ shadow.T) @ frame)
    residual = float(np.max(np.abs(corrected_coordinates[~mask]))) if np.any(~mask) else 0.0
    return corrected, ShadowOrientationDiagnostics(
        raw_bias_tolerance_frame=raw_coordinates,
        kept_bias_tolerance_frame=kept_coordinates,
        corrected_bias_tolerance_frame=corrected_coordinates,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Merge only legal optimized offsets back into the next 10 Hz target."""
    planned = np.asarray(planned_rotation, dtype=np.float64)
    optimized = np.asarray(optimized_rotation, dtype=np.float64)
    frame = np.asarray(tolerance_frame, dtype=np.float64)
    ranged = np.asarray(ranged_axes, dtype=bool)
    release = frame.T @ planned @ matrix_to_rotvec(planned.T @ optimized)
    release[~ranged] = 0.0
    if lower is not None and upper is not None:
        release[ranged] = np.clip(
            release[ranged],
            np.asarray(lower, dtype=np.float64)[ranged],
            np.asarray(upper, dtype=np.float64)[ranged],
        )
    return rotvec_to_matrix(frame @ release) @ planned, release


@dataclass
class ShadowOrientationReference:
    """Optimizer-independent zero-tolerance shadow state."""

    shadow_rotation: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    _semantic_key: Hashable | None = field(default=None, init=False, repr=False)
    _ranged_axes: np.ndarray | None = field(default=None, init=False, repr=False)
    _tolerance_frame: np.ndarray | None = field(default=None, init=False, repr=False)

    def reset(
        self,
        optimized_rotation: np.ndarray,
        *,
        semantic_key: Hashable,
        ranged_axes: np.ndarray,
        tolerance_frame: np.ndarray,
    ) -> None:
        self.shadow_rotation = np.asarray(optimized_rotation, dtype=np.float64).copy()
        self._semantic_key = semantic_key
        self._ranged_axes = np.asarray(ranged_axes, dtype=bool).copy()
        self._tolerance_frame = np.asarray(tolerance_frame, dtype=np.float64).copy()

    def advance(
        self,
        optimized_rotation: np.ndarray,
        rotation_action: np.ndarray,
        tolerance_frame: np.ndarray,
        ranged_axes: np.ndarray,
        *,
        semantic_key: Hashable,
    ) -> tuple[np.ndarray, ShadowOrientationDiagnostics]:
        optimized = np.asarray(optimized_rotation, dtype=np.float64)
        action = np.asarray(rotation_action, dtype=np.float64)
        frame = np.asarray(tolerance_frame, dtype=np.float64)
        mask = np.asarray(ranged_axes, dtype=bool)
        context_changed = (
            self._ranged_axes is None
            or semantic_key != self._semantic_key
            or not np.array_equal(mask, self._ranged_axes)
            or not np.allclose(frame, self._tolerance_frame, atol=1e-10)
        )
        if context_changed:
            self.reset(
                optimized,
                semantic_key=semantic_key,
                ranged_axes=mask,
                tolerance_frame=frame,
            )

        increment = rotvec_to_matrix(action)
        raw_proposal = increment @ optimized
        shadow_proposal = increment @ self.shadow_rotation
        corrected, diagnostics = project_historical_tolerance_bias(
            raw_proposal, shadow_proposal, frame, mask
        )
        self.shadow_rotation = shadow_proposal
        return corrected, ShadowOrientationDiagnostics(
            raw_bias_tolerance_frame=diagnostics.raw_bias_tolerance_frame,
            kept_bias_tolerance_frame=diagnostics.kept_bias_tolerance_frame,
            corrected_bias_tolerance_frame=diagnostics.corrected_bias_tolerance_frame,
            non_tolerance_residual_rad=diagnostics.non_tolerance_residual_rad,
            anchor_reset=context_changed,
        )

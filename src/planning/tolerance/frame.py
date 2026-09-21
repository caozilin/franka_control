"""SO(3) tolerance-frame filtering shared by live and offline control."""

from __future__ import annotations

from typing import Hashable

import numpy as np

from utils.pose import rotation_matrix, rotation_vector, project_to_rotation_matrix


CANONICAL_HEADING_CODE = np.array((1.0, 0.0))
HEADING_CODE_EPSILON = 1e-8


def heading_code(stage_rotation: np.ndarray) -> np.ndarray:
    """Encode a stage orientation as the tolerance frame's horizontal Y axis."""
    rotation = np.asarray(stage_rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Stage rotation must be a finite 3x3 matrix")
    z_axis = np.array((0.0, 0.0, 1.0))
    y_axis = rotation[:, 1].copy()
    y_axis -= z_axis * float(y_axis @ z_axis)
    if np.linalg.norm(y_axis) < HEADING_CODE_EPSILON:
        projected_x = rotation[:, 0] - z_axis * float(rotation[:, 0] @ z_axis)
        norm = float(np.linalg.norm(projected_x))
        if norm < HEADING_CODE_EPSILON:
            raise ValueError("Stage rotation has no stable horizontal heading")
        y_axis = projected_x / norm
    code = y_axis[:2]
    return code / np.linalg.norm(code)


def frame_from_heading(code: np.ndarray) -> np.ndarray:
    """Construct the gravity-aligned tolerance frame from ``[cos(phi), sin(phi)]``."""
    heading = np.asarray(code, dtype=float)
    if heading.shape != (2,) or not np.all(np.isfinite(heading)):
        raise ValueError("Tolerance frame code must be a finite two-vector")
    norm = float(np.linalg.norm(heading))
    if norm < HEADING_CODE_EPSILON:
        raise ValueError("Tolerance frame code norm is too small")
    heading = heading / norm
    y_axis = np.array((heading[0], heading[1], 0.0))
    z_axis = np.array((0.0, 0.0, 1.0))
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


class ToleranceFrameEMA:
    """Stage-local exponential moving average on SO(3).

    The frame is updated once per outer control cycle and is intentionally
    independent of optimizer iterations.  A stage transition starts a new
    history from the first raw frame of that stage.
    """

    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = float(alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("Tolerance prediction EMA alpha must be in (0, 1]")
        self.raw_tolerance_frame = np.eye(3)
        self.ema_tolerance_frame = np.eye(3)
        self.stage_key: Hashable | None = None
        self.initialized = False

    def reset(self) -> None:
        self.raw_tolerance_frame = np.eye(3)
        self.ema_tolerance_frame = np.eye(3)
        self.stage_key = None
        self.initialized = False

    def update(
        self,
        raw_tolerance_frame: np.ndarray,
        *,
        stage_key: Hashable,
    ) -> np.ndarray:
        raw = np.asarray(raw_tolerance_frame, dtype=float)
        if raw.shape != (3, 3):
            raise ValueError("Tolerance frame must be a 3x3 matrix")
        # Keep the accepted frame on SO(3) while allowing callers to provide
        # numerically rounded matrices from configuration or telemetry.
        raw = project_to_rotation_matrix(raw)
        new_stage = (not self.initialized) or stage_key != self.stage_key
        self.raw_tolerance_frame = raw.copy()
        if new_stage:
            self.ema_tolerance_frame = raw.copy()
        else:
            delta = rotation_vector(raw @ self.ema_tolerance_frame.T)
            self.ema_tolerance_frame = rotation_matrix(
                self.alpha * delta,
            ) @ self.ema_tolerance_frame
        self.stage_key = stage_key
        self.initialized = True
        return self.ema_tolerance_frame.copy()


class ToleranceLimitsEMA:
    """Stage-local EMA for directional tolerance magnitudes."""

    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = float(alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("Tolerance prediction EMA alpha must be in (0, 1]")
        self.raw_negative_limits = np.zeros(3)
        self.raw_positive_limits = np.zeros(3)
        self.negative_limits = np.zeros(3)
        self.positive_limits = np.zeros(3)
        self.stage_key: Hashable | None = None
        self.initialized = False

    def reset(self) -> None:
        self.raw_negative_limits = np.zeros(3)
        self.raw_positive_limits = np.zeros(3)
        self.negative_limits = np.zeros(3)
        self.positive_limits = np.zeros(3)
        self.stage_key = None
        self.initialized = False

    @staticmethod
    def _validated(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if (
            array.shape != (3,)
            or np.any(np.isnan(array))
            or np.any(np.isneginf(array))
            or np.any(array < 0.0)
        ):
            raise ValueError(f"{name} must contain three non-negative values")
        return array.copy()

    def _blend(self, previous: np.ndarray, raw: np.ndarray) -> np.ndarray:
        # Infinity denotes an explicitly unbounded tolerant axis and has no meaningful
        # interpolation with a finite trigger. Switch such axes immediately;
        # blend only pairs in the ordinary finite domain.
        blended = raw.copy()
        finite = np.isfinite(previous) & np.isfinite(raw)
        blended[finite] = (
            previous[finite]
            + self.alpha * (raw[finite] - previous[finite])
        )
        return blended

    def update(
        self,
        raw_negative_limits: np.ndarray,
        raw_positive_limits: np.ndarray,
        *,
        stage_key: Hashable,
    ) -> tuple[np.ndarray, np.ndarray]:
        negative = self._validated(
            raw_negative_limits, "Negative tolerance limits",
        )
        positive = self._validated(
            raw_positive_limits, "Positive tolerance limits",
        )
        new_stage = (not self.initialized) or stage_key != self.stage_key
        self.raw_negative_limits = negative
        self.raw_positive_limits = positive
        if new_stage:
            self.negative_limits = negative.copy()
            self.positive_limits = positive.copy()
        else:
            self.negative_limits = self._blend(
                self.negative_limits, negative,
            )
            self.positive_limits = self._blend(
                self.positive_limits, positive,
            )
        self.stage_key = stage_key
        self.initialized = True
        return (
            self.negative_limits.copy(),
            self.positive_limits.copy(),
        )


__all__ = (
    "CANONICAL_HEADING_CODE",
    "HEADING_CODE_EPSILON",
    "ToleranceFrameEMA",
    "ToleranceLimitsEMA",
    "frame_from_heading",
    "heading_code",
)

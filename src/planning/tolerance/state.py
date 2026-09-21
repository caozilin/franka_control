from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np

from planning.task_space.types import AxisTask, TaskKind
from utils.pose import (
    rotation_error_in_frame,
    rotation_matrix,
)
from .frame import (
    CANONICAL_HEADING_CODE,
    HEADING_CODE_EPSILON,
    ToleranceFrameEMA,
    ToleranceLimitsEMA,
    frame_from_heading,
)


DEFAULT_ROTATION_TOLERANCE_RAD = np.radians((30.0, 30.0, 30.0))
DEFAULT_STAGE_RANGED = np.array((
    (False, False, False),
    (False, False, False),
    (False, False, False),
    (False, False, False),
), dtype=bool)
DEFAULT_STAGE_LIMITS_DEG = np.tile(np.array((30.0, 30.0, 30.0)), (4, 1))
ROTATION_STAGE_COUNT = 4
PREGRASP_STAGE = 0
GRASP_STAGE = 1
POSTGRASP_STAGE = 2
RELEASE_STAGE = 3
STRICT_ROTATION_STAGES = (GRASP_STAGE, RELEASE_STAGE)

# At 10 Hz a full-speed keyboard command is 4.5 deg per planner step. These
# normalized thresholds also work for continuous VLA actions below full scale.
_NOMINAL_ACTION_STEP_RAD = np.radians(4.5)
_SIGNAL_TIME_CONSTANT_S = 0.35
_CONFIDENCE_ATTACK_S = 0.25
_CONFIDENCE_RELEASE_S = 0.80
_ACTIVITY_LOW = 0.05 * _NOMINAL_ACTION_STEP_RAD
_ACTIVITY_HIGH = 0.35 * _NOMINAL_ACTION_STEP_RAD
_INSTANT_PREFERENCE_FLOOR = 0.15


ORIENTATION_REFERENCE_STAGE_RELATIVE = "stage_relative_release"
ORIENTATION_REFERENCE_MODES = frozenset((
    ORIENTATION_REFERENCE_STAGE_RELATIVE,
))


@dataclass
class RotationalToleranceState:
    """Per-cycle rotational tolerance descriptor with stage-local EMA."""

    limits: np.ndarray = field(default_factory=lambda: DEFAULT_ROTATION_TOLERANCE_RAD.copy())
    negative_limits: np.ndarray | None = None
    positive_limits: np.ndarray | None = None
    ranged: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=bool))
    target: np.ndarray = field(default_factory=lambda: np.eye(3))
    frame: np.ndarray = field(default_factory=lambda: np.eye(3))
    raw_tolerance_frame: np.ndarray = field(default_factory=lambda: np.eye(3))
    raw_negative_limits: np.ndarray = field(init=False)
    raw_positive_limits: np.ndarray = field(init=False)
    tolerance_prediction_ema_alpha: float = 0.25
    active_stage: int = 0
    _intent_signed: np.ndarray = field(default_factory=lambda: np.zeros((ROTATION_STAGE_COUNT, 3)), init=False, repr=False)
    _intent_magnitude: np.ndarray = field(default_factory=lambda: np.zeros((ROTATION_STAGE_COUNT, 3)), init=False, repr=False)
    _intent_confidence: np.ndarray = field(default_factory=lambda: np.zeros((ROTATION_STAGE_COUNT, 3)), init=False, repr=False)
    _instant_activity: np.ndarray = field(default_factory=lambda: np.zeros((ROTATION_STAGE_COUNT, 3)), init=False, repr=False)
    _instant_action: np.ndarray = field(default_factory=lambda: np.zeros((ROTATION_STAGE_COUNT, 3)), init=False, repr=False)
    _frame_ema: ToleranceFrameEMA = field(init=False, repr=False)
    _limits_ema: ToleranceLimitsEMA = field(init=False, repr=False)
    _descriptor_stage_key: Hashable | None = field(
        default=None, init=False, repr=False,
    )
    _frame_code_stage_key: Hashable | None = field(
        default=None, init=False, repr=False,
    )
    _last_valid_frame_code: np.ndarray = field(
        default_factory=lambda: CANONICAL_HEADING_CODE.copy(),
        init=False,
        repr=False,
    )
    _signed_limits_are_custom: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.limits = np.asarray(self.limits, dtype=float).copy()
        self._signed_limits_are_custom = bool(
            self.negative_limits is not None or self.positive_limits is not None
        )
        self.negative_limits = np.asarray(
            self.limits if self.negative_limits is None else self.negative_limits,
            dtype=float,
        ).copy()
        self.positive_limits = np.asarray(
            self.limits if self.positive_limits is None else self.positive_limits,
            dtype=float,
        ).copy()
        self.ranged = np.asarray(self.ranged, dtype=bool).copy()
        self.target = np.asarray(self.target, dtype=float).copy()
        self.frame = np.asarray(self.frame, dtype=float).copy()
        self.raw_tolerance_frame = self.frame.copy()
        self.raw_negative_limits = self.negative_limits.copy()
        self.raw_positive_limits = self.positive_limits.copy()
        self._frame_ema = ToleranceFrameEMA(
            self.tolerance_prediction_ema_alpha,
        )
        self._limits_ema = ToleranceLimitsEMA(
            self.tolerance_prediction_ema_alpha,
        )
        if (
            self.limits.shape != (3,)
            or self.negative_limits.shape != (3,)
            or self.positive_limits.shape != (3,)
            or self.ranged.shape != (3,)
            or self.target.shape != (3, 3) or self.frame.shape != (3, 3)
        ):
            raise ValueError("Invalid rotational tolerance state shape.")
        if not 0 <= self.active_stage < ROTATION_STAGE_COUNT:
            raise ValueError("Active tolerance stage is out of range.")

    def update_tolerance_frame(
        self,
        raw_tolerance_frame: np.ndarray,
        *,
        stage_key: Hashable | None = None,
    ) -> np.ndarray:
        """Update the stage-local SO(3) EMA once per outer control cycle."""
        key = self.active_stage if stage_key is None else stage_key
        self.frame = self._frame_ema.update(
            raw_tolerance_frame, stage_key=key,
        )
        self.raw_tolerance_frame = self._frame_ema.raw_tolerance_frame.copy()
        return self.frame.copy()

    def resolve_tolerance_frame_code(
        self,
        raw_frame_code: np.ndarray,
        *,
        stage_key: Hashable | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize a raw heading code with one stage-local fallback owner."""
        code = np.asarray(raw_frame_code, dtype=float)
        if code.shape != (2,) or not np.all(np.isfinite(code)):
            raise ValueError("Tolerance frame code must be a finite two-vector")
        key = self.active_stage if stage_key is None else stage_key
        if key != self._frame_code_stage_key:
            self._last_valid_frame_code = CANONICAL_HEADING_CODE.copy()
            self._frame_code_stage_key = key
        norm = float(np.linalg.norm(code))
        if norm >= HEADING_CODE_EPSILON:
            self._last_valid_frame_code = code / norm
        normalized = self._last_valid_frame_code.copy()
        return normalized, frame_from_heading(normalized)

    @property
    def tolerance_frame_code(self) -> np.ndarray:
        return self._last_valid_frame_code.copy()

    def update_tolerance_limits(
        self,
        raw_negative_limits: np.ndarray,
        raw_positive_limits: np.ndarray,
        *,
        stage_key: Hashable | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update stage-local tolerance magnitudes once per outer cycle."""
        key = self.active_stage if stage_key is None else stage_key
        (
            self.negative_limits,
            self.positive_limits,
        ) = self._limits_ema.update(
            raw_negative_limits,
            raw_positive_limits,
            stage_key=key,
        )
        self.raw_negative_limits = (
            self._limits_ema.raw_negative_limits.copy()
        )
        self.raw_positive_limits = (
            self._limits_ema.raw_positive_limits.copy()
        )
        self.limits = np.maximum(self.negative_limits, self.positive_limits)
        return (
            self.negative_limits.copy(),
            self.positive_limits.copy(),
        )

    def configure_stage(
        self,
        stage: int,
        target: np.ndarray,
        frame: np.ndarray,
        limits: np.ndarray,
        *,
        negative_limits: np.ndarray | None = None,
        positive_limits: np.ndarray | None = None,
        stage_key: Hashable | None = None,
        update_prediction: bool = True,
    ) -> None:
        """Activate one stage and adopt its goal, axes and tolerances.

        Only a semantic stage-key change resets that stage's intent filter.
        Same-stage target, frame, bounds, and mask updates preserve intent.
        """
        if not 0 <= stage < ROTATION_STAGE_COUNT:
            raise IndexError("Tolerance stage is out of range.")
        target = np.asarray(target, dtype=float)
        frame = np.asarray(frame, dtype=float)
        limits = np.asarray(limits, dtype=float)
        negative = np.asarray(
            limits if negative_limits is None else negative_limits,
            dtype=float,
        )
        positive = np.asarray(
            limits if positive_limits is None else positive_limits,
            dtype=float,
        )
        if target.shape != (3, 3) or frame.shape != (3, 3) or limits.shape != (3,):
            raise ValueError("Invalid stage tolerance configuration shape.")
        if (
            negative.shape != (3,)
            or positive.shape != (3,)
            or np.any(negative < 0.0)
            or np.any(positive < 0.0)
        ):
            raise ValueError("Invalid signed stage tolerance limits.")
        key = stage if stage_key is None else stage_key
        stage_changed = key != self._descriptor_stage_key
        self._descriptor_stage_key = key
        if stage_changed:
            self._intent_signed[stage] = 0.0
            self._intent_magnitude[stage] = 0.0
            self._intent_confidence[stage] = 0.0
            self._instant_activity[stage] = 0.0
            self._instant_action[stage] = 0.0
        self.active_stage = stage
        self.target = target.copy()
        self._signed_limits_are_custom = bool(
            negative_limits is not None or positive_limits is not None
        )
        if not update_prediction:
            self.raw_tolerance_frame = frame.copy()
            self.raw_negative_limits = negative.copy()
            self.raw_positive_limits = positive.copy()
            return
        self.update_tolerance_limits(
            negative,
            positive,
            stage_key=key,
        )
        self.update_tolerance_frame(
            frame,
            stage_key=key,
        )

    def actual_tolerance(
        self,
        current_rotation: np.ndarray,
        target_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Actual-minus-target rotation error in the independent frame."""
        current_rotation = np.asarray(current_rotation, dtype=float)
        if current_rotation.shape != (3, 3):
            raise ValueError("Current EE rotation must be a 3x3 matrix.")
        goal = (
            self.target
            if target_rotation is None
            else np.asarray(target_rotation, dtype=float)
        )
        if goal.shape != (3, 3):
            raise ValueError("Target EE rotation must be a 3x3 matrix.")
        return rotation_error_in_frame(current_rotation, goal, self.frame)

    def target_error(
        self,
        current_rotation: np.ndarray,
        target_rotation: np.ndarray | None = None,
    ) -> np.ndarray:
        """Target-minus-actual error for telemetry and recorded statistics."""
        return -self.actual_tolerance(current_rotation, target_rotation)

    @staticmethod
    def _smoothstep(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
        scaled = np.clip((value - lower) / (upper - lower), 0.0, 1.0)
        return scaled * scaled * (3.0 - 2.0 * scaled)

    @property
    def intent_confidence(self) -> np.ndarray:
        return self._intent_confidence[self.active_stage].copy()

    @property
    def intent_preference(self) -> np.ndarray:
        # A fresh non-zero command receives an immediate but deliberately weak
        # response; only sustained coherent intent approaches full preference.
        return np.maximum(
            _INSTANT_PREFERENCE_FLOOR * self._instant_activity[self.active_stage],
            self._intent_confidence[self.active_stage],
        )

    @property
    def nominal_action_ema(self) -> np.ndarray:
        """Dead-zone-normalized stage-local EMA of the nominal action.

        Tiny planner corrections below five percent of a full-speed 10 Hz
        rotational command are treated as zero.  The smooth transition uses
        the same activity thresholds as the existing intent filter.
        """
        return self._smoothstep(
            self._intent_magnitude[self.active_stage],
            _ACTIVITY_LOW,
            _ACTIVITY_HIGH,
        ).copy()

    def reset_intent(self) -> None:
        self._intent_signed.fill(0.0)
        self._intent_magnitude.fill(0.0)
        self._intent_confidence.fill(0.0)
        self._instant_activity.fill(0.0)
        self._instant_action.fill(0.0)

    def reset_subtask_history(self) -> None:
        """Clear every stage-local history at a semantic subtask boundary."""
        self.reset_intent()
        self._frame_ema.reset()
        self._limits_ema.reset()
        self._descriptor_stage_key = None
        self._frame_code_stage_key = None
        self._last_valid_frame_code = CANONICAL_HEADING_CODE.copy()

    def update_intent(
        self,
        rotation_action: np.ndarray,
        dt: float,
    ) -> None:
        """Update intent from a base-frame nominal rotation increment."""
        action = np.asarray(rotation_action, dtype=float)
        if action.shape != (3,) or dt <= 0.0:
            raise ValueError("Rotation intent requires a 3-vector and positive dt.")
        action_tolerance = self.frame.T @ action

        signal_alpha = 1.0 - np.exp(-dt / _SIGNAL_TIME_CONSTANT_S)
        for stage in range(ROTATION_STAGE_COUNT):
            sample = action_tolerance if stage == self.active_stage else np.zeros(3)
            self._instant_action[stage] = sample
            self._intent_signed[stage] += signal_alpha * (sample - self._intent_signed[stage])
            self._intent_magnitude[stage] += signal_alpha * (np.abs(sample) - self._intent_magnitude[stage])
            activity = self._smoothstep(self._intent_magnitude[stage], _ACTIVITY_LOW, _ACTIVITY_HIGH)
            coherence = np.divide(
                np.abs(self._intent_signed[stage]),
                self._intent_magnitude[stage],
                out=np.zeros(3),
                where=self._intent_magnitude[stage] > 1e-12,
            )
            raw_confidence = activity * np.clip(coherence, 0.0, 1.0)
            rising = raw_confidence > self._intent_confidence[stage]
            alpha = np.where(
                rising,
                1.0 - np.exp(-dt / _CONFIDENCE_ATTACK_S),
                1.0 - np.exp(-dt / _CONFIDENCE_RELEASE_S),
            )
            self._intent_confidence[stage] += alpha * (raw_confidence - self._intent_confidence[stage])
            self._instant_activity[stage] = self._smoothstep(np.abs(sample), _ACTIVITY_LOW, _ACTIVITY_HIGH)

    def task(
        self,
        axis: int,
    ) -> AxisTask:
        """Build the per-axis task from stage coefficients only."""
        if self.active_stage in STRICT_ROTATION_STAGES:
            return AxisTask(TaskKind.SPECIFIC, goal=0.0)
        if self.ranged[axis]:
            # The release step interprets these limits as a stage-relative
            # SO(3) hard box. The objective separately uses its frozen
            # physical R_loss reference and never treats the box as a wall.
            negative_limit = float(
                self.negative_limits[axis]
                if self._signed_limits_are_custom
                else self.limits[axis]
            )
            positive_limit = float(
                self.positive_limits[axis]
                if self._signed_limits_are_custom
                else self.limits[axis]
            )
            # A zero coefficient must not become a degenerate flat range with
            # two dependent inequalities. Collapse it to a strict zero task.
            if negative_limit <= 0.0 and positive_limit <= 0.0:
                return AxisTask(TaskKind.SPECIFIC, goal=0.0)
            instantaneous_activity = float(
                self._instant_activity[self.active_stage, axis]
            )
            preference_weight = float(self.intent_preference[axis])
            return AxisTask(
                TaskKind.FLAT_RANGE,
                lower=-negative_limit,
                upper=positive_limit,
                preference_goal=0.0,
                preference_weight=preference_weight,
                instantaneous_activity=instantaneous_activity,
                ema_nominal_action=float(self.nominal_action_ema[axis]),
            )
        return AxisTask(TaskKind.SPECIFIC, goal=0.0)

    def error(self, current_rotation: np.ndarray, target_rotation: np.ndarray) -> np.ndarray:
        """Return current-minus-target rotation error in this stage's frame."""
        return rotation_error_in_frame(current_rotation, target_rotation, self.frame)

    def apply_increment(
        self,
        target_rotation: np.ndarray,
        rotation_action: np.ndarray,
    ) -> np.ndarray:
        """Left-multiply one base-frame increment onto the target."""
        increment = rotation_matrix(np.asarray(rotation_action, dtype=float))
        return increment @ target_rotation

    def reconcile_modes(
        self,
        target_rotation: np.ndarray,
        active_rotations: np.ndarray,
        toggled_axes: tuple[int, ...],
    ) -> np.ndarray:
        """Change Specific/Ranged modes without changing the nominal target."""
        active_rotations = np.asarray(active_rotations, dtype=bool)
        for axis in toggled_axes:
            if not 0 <= axis < 3:
                raise IndexError("Rotational axis must be in [0, 2].")
        self.ranged = ~active_rotations
        return np.asarray(target_rotation, dtype=float).copy()

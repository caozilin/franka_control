from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from planning.sqp.controller import AxisTask, TaskKind
from utils.pose import rotation_tolerance_coordinates


class ManipulationPhase(IntEnum):
    PREGRASP = 0
    GRASP = 1
    POSTGRASP = 2
    RELEASE = 3

    @property
    def key(self) -> str:
        return ("pre", "grasp", "post", "release")[int(self)]


@dataclass(frozen=True)
class PhaseObservation:
    phase: ManipulationPhase
    aperture_m: float
    motion_m: float
    commanded_closed: bool


class GripperPhaseClassifier:
    """MuJoCo-compatible four-stage classifier using the total gripper width."""

    def __init__(self, *, motion_threshold_m: float = 0.0002, stable_frames: int = 3) -> None:
        self.motion_threshold_m = float(motion_threshold_m)
        self.stable_frames = int(stable_frames)
        self.reset()

    def reset(self) -> None:
        self._previous_aperture: float | None = None
        self._previous_commanded_closed: bool | None = None
        self._previous_phase: ManipulationPhase | None = None
        self._stable_width_frames = 0

    def update(self, aperture_m: float, commanded_closed: bool) -> PhaseObservation:
        aperture = float(aperture_m)
        if not np.isfinite(aperture) or aperture < 0.0:
            raise ValueError("gripper aperture must be finite and non-negative")
        if self._previous_aperture is None:
            motion = 0.0
            command_changed = False
        else:
            motion = abs(aperture - self._previous_aperture)
            command_changed = bool(commanded_closed) != self._previous_commanded_closed
        moving = motion > self.motion_threshold_m
        if command_changed:
            self._stable_width_frames = 0
        elif moving:
            self._stable_width_frames = 1
        else:
            self._stable_width_frames += 1

        if self._previous_aperture is None and not commanded_closed:
            phase = ManipulationPhase.PREGRASP
        elif commanded_closed:
            if self._previous_phase is ManipulationPhase.POSTGRASP:
                phase = ManipulationPhase.POSTGRASP
            else:
                phase = (
                    ManipulationPhase.POSTGRASP
                    if self._stable_width_frames >= self.stable_frames
                    else ManipulationPhase.GRASP
                )
        else:
            if self._previous_phase is ManipulationPhase.PREGRASP:
                phase = ManipulationPhase.PREGRASP
            else:
                phase = (
                    ManipulationPhase.PREGRASP
                    if self._stable_width_frames >= self.stable_frames
                    else ManipulationPhase.RELEASE
                )

        self._previous_aperture = aperture
        self._previous_commanded_closed = bool(commanded_closed)
        self._previous_phase = phase
        return PhaseObservation(phase, aperture, motion, bool(commanded_closed))


@dataclass(frozen=True)
class TaskToleranceProfile:
    pre_deg: tuple[float, float, float, float, float, float]
    post_deg: tuple[float, float, float, float, float, float]

    def bounds_rad(self, phase: ManipulationPhase) -> tuple[np.ndarray, np.ndarray]:
        values = self.pre_deg if phase is ManipulationPhase.PREGRASP else self.post_deg
        bounds = np.radians(np.asarray(values, dtype=np.float64))
        return bounds[0::2], bounds[1::2]


# Unique Franka/Panda pre/post combinations extracted from the supplied CSV.
# Order per stage: Rx-, Rx+, Ry-, Ry+, Rz-, Rz+ in degrees.
PANDA_TOLERANCE_PROFILES: dict[str, TaskToleranceProfile] = {
    "T01": TaskToleranceProfile((30, 30, 30, 10, 0, 0), (0, 0, 0, 0, 45, 45)),
    "T02": TaskToleranceProfile((0, 0, 30, 10, 0, 0), (0, 0, 0, 0, 45, 45)),
    "T03": TaskToleranceProfile((30, 30, 30, 30, 45, 45), (30, 30, 30, 30, 45, 45)),
    "T04": TaskToleranceProfile((0, 0, 30, 30, 45, 45), (0, 0, 0, 0, 45, 45)),
    "T05": TaskToleranceProfile((30, 30, 30, 30, 0, 0), (0, 0, 0, 0, 45, 45)),
    "T06": TaskToleranceProfile((10, 10, 0, 0, 0, 0), (30, 30, 30, 30, 45, 45)),
    "T07": TaskToleranceProfile((20, 20, 0, 0, 0, 0), (30, 30, 30, 30, 45, 45)),
    "T08": TaskToleranceProfile((20, 20, 30, 30, 45, 45), (20, 20, 30, 30, 45, 45)),
    "T09": TaskToleranceProfile((0, 0, 30, 30, 0, 0), (30, 30, 30, 30, 45, 45)),
    "T10": TaskToleranceProfile((0, 0, 30, 30, 0, 0), (0, 0, 0, 0, 45, 45)),
    "T11": TaskToleranceProfile((5, 5, 0, 0, 0, 0), (20, 20, 30, 30, 45, 45)),
    "T12": TaskToleranceProfile((0, 0, 30, 30, 0, 0), (0, 0, 0, 0, 0, 0)),
    "T13": TaskToleranceProfile((10, 10, 30, 30, 0, 0), (30, 30, 0, 0, 0, 0)),
}

PANDA_TASK_TOLERANCE_IDS: dict[str, str] = {
    "adjust_cylindrical_bottle": "T01",
    "adjust_rectangular_bottle": "T02",
    "click_bell": "T03", "pear_to_bowl": "T03", "pear_to_plate": "T03", "press_power_strip": "T03",
    "close_cylindrical_pot_lid": "T04", "geometry_plate_cylinder_upright": "T04", "geometry_region_cylinder_upright": "T04", "open_cylindrical_pot_lid": "T04",
    "close_handle_pot_lid": "T05", "open_handle_pot_lid": "T05",
    "banana_to_plate": "T06",
    "strawberry_to_bowl": "T07", "strawberry_to_plate": "T07",
    "geometry_plate_ball": "T08",
    "geometry_plate_box_lying": "T09", "geometry_plate_cube": "T09",
    "geometry_plate_box_upright": "T10",
    "geometry_plate_cylinder_lying": "T11",
    "geometry_region_box_lying": "T12", "geometry_region_box_upright": "T12", "geometry_region_cube": "T12", "rotate_knob": "T12",
    "geometry_region_cylinder_lying": "T13",
}


def box_tolerance_frame(target_rotation: np.ndarray) -> np.ndarray:
    """MuJoCo rule: world Z and the target tool-Y projected horizontally."""
    rotation = np.asarray(target_rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("target rotation must be 3x3")
    z_axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    y_axis = rotation[:, 1].copy()
    y_axis -= z_axis * float(y_axis @ z_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        projected_x = rotation[:, 0] - z_axis * float(rotation[:, 0] @ z_axis)
        projected_x /= np.linalg.norm(projected_x)
        y_axis = np.cross(z_axis, projected_x)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


_STAGE_COUNT = 4
_STRICT_STAGES = (int(ManipulationPhase.GRASP), int(ManipulationPhase.RELEASE))
_NOMINAL_ACTION_STEP_RAD = np.radians(4.5)
_SIGNAL_TIME_CONSTANT_S = 0.35
_CONFIDENCE_ATTACK_S = 0.25
_CONFIDENCE_RELEASE_S = 0.80
_ACTIVITY_LOW = 0.05 * _NOMINAL_ACTION_STEP_RAD
_ACTIVITY_HIGH = 0.35 * _NOMINAL_ACTION_STEP_RAD
_INSTANT_PREFERENCE_FLOOR = 0.15


def _rotation_error_in_frame(actual: np.ndarray, target: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Actual-minus-target fixed-axis XYZ/RPY coordinates in ``frame``."""
    return rotation_tolerance_coordinates(actual, target, frame)


@dataclass
class RotationalToleranceState:
    """Exact live rotational-tolerance state used by franka_mujoco."""

    limits: np.ndarray = field(default_factory=lambda: np.radians((30.0, 30.0, 45.0)))
    negative_limits: np.ndarray | None = None
    positive_limits: np.ndarray | None = None
    ranged: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=bool))
    target: np.ndarray = field(default_factory=lambda: np.eye(3))
    frame: np.ndarray = field(default_factory=lambda: np.eye(3))
    active_stage: int = 0
    _intent_signed: np.ndarray = field(default_factory=lambda: np.zeros((_STAGE_COUNT, 3)), init=False)
    _intent_magnitude: np.ndarray = field(default_factory=lambda: np.zeros((_STAGE_COUNT, 3)), init=False)
    _intent_confidence: np.ndarray = field(default_factory=lambda: np.zeros((_STAGE_COUNT, 3)), init=False)
    _instant_activity: np.ndarray = field(default_factory=lambda: np.zeros((_STAGE_COUNT, 3)), init=False)
    _instant_action: np.ndarray = field(default_factory=lambda: np.zeros((_STAGE_COUNT, 3)), init=False)
    _stage_targets: np.ndarray = field(default_factory=lambda: np.tile(np.eye(3), (_STAGE_COUNT, 1, 1)), init=False)
    _stage_frames: np.ndarray = field(default_factory=lambda: np.tile(np.eye(3), (_STAGE_COUNT, 1, 1)), init=False)
    _postgrasp_error_bias: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    _postgrasp_bias_captured: bool = field(default=False, init=False)
    _signed_limits_are_custom: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.limits = np.asarray(self.limits, dtype=float).copy()
        self._signed_limits_are_custom = self.negative_limits is not None or self.positive_limits is not None
        self.negative_limits = np.asarray(self.limits if self.negative_limits is None else self.negative_limits, dtype=float).copy()
        self.positive_limits = np.asarray(self.limits if self.positive_limits is None else self.positive_limits, dtype=float).copy()
        self.ranged = np.asarray(self.ranged, dtype=bool).copy()
        self.target = np.asarray(self.target, dtype=float).copy()
        self.frame = np.asarray(self.frame, dtype=float).copy()
        self._stage_targets[:] = self.target
        self._stage_frames[:] = self.frame

    def configure_stage(
        self,
        stage: int,
        target: np.ndarray,
        frame: np.ndarray,
        limits: np.ndarray,
        *,
        negative_limits: np.ndarray | None = None,
        positive_limits: np.ndarray | None = None,
    ) -> None:
        target = np.asarray(target, dtype=float)
        frame = np.asarray(frame, dtype=float)
        limits = np.asarray(limits, dtype=float)
        negative = np.asarray(limits if negative_limits is None else negative_limits, dtype=float)
        positive = np.asarray(limits if positive_limits is None else positive_limits, dtype=float)
        changed = not np.allclose(target, self._stage_targets[stage], atol=1e-10) or not np.allclose(frame, self._stage_frames[stage], atol=1e-10)
        if changed:
            self._stage_targets[stage] = target
            self._stage_frames[stage] = frame
            self._intent_signed[stage] = 0.0
            self._intent_magnitude[stage] = 0.0
            self._intent_confidence[stage] = 0.0
            self._instant_activity[stage] = 0.0
            self._instant_action[stage] = 0.0
            if stage == int(ManipulationPhase.POSTGRASP):
                self.clear_postgrasp_bias()
        self.active_stage = stage
        self.target = target.copy()
        self.frame = frame.copy()
        self.limits = limits.copy()
        self.negative_limits = negative.copy()
        self.positive_limits = positive.copy()
        self._signed_limits_are_custom = negative_limits is not None or positive_limits is not None

    def actual_tolerance(self, current_rotation: np.ndarray, target_rotation: np.ndarray | None = None) -> np.ndarray:
        return _rotation_error_in_frame(current_rotation, self.target if target_rotation is None else target_rotation, self.frame)

    @property
    def postgrasp_bias_active(self) -> bool:
        return self.active_stage == int(ManipulationPhase.POSTGRASP) and self._postgrasp_bias_captured

    def clear_postgrasp_bias(self) -> None:
        self._postgrasp_error_bias[:] = 0.0
        self._postgrasp_bias_captured = False

    def handle_phase_transition(self, previous_stage: int, new_stage: int, current_rotation: np.ndarray, target_rotation: np.ndarray | None = None) -> None:
        if previous_stage == int(ManipulationPhase.RELEASE) and new_stage == int(ManipulationPhase.PREGRASP):
            self.clear_postgrasp_bias()
        elif previous_stage == int(ManipulationPhase.GRASP) and new_stage == int(ManipulationPhase.POSTGRASP) and not self._postgrasp_bias_captured:
            self._postgrasp_error_bias = self.actual_tolerance(current_rotation, target_rotation)
            self._postgrasp_bias_captured = True

    def boundary_tolerance(self, current_rotation: np.ndarray, target_rotation: np.ndarray | None = None) -> np.ndarray:
        actual = self.actual_tolerance(current_rotation, target_rotation)
        return actual - self._postgrasp_error_bias if self.postgrasp_bias_active else actual

    @staticmethod
    def _smoothstep(value: np.ndarray, lower: float, upper: float) -> np.ndarray:
        scaled = np.clip((value - lower) / (upper - lower), 0.0, 1.0)
        return scaled * scaled * (3.0 - 2.0 * scaled)

    @property
    def intent_preference(self) -> np.ndarray:
        return np.maximum(_INSTANT_PREFERENCE_FLOOR * self._instant_activity[self.active_stage], self._intent_confidence[self.active_stage])

    def reset_intent(self) -> None:
        self._intent_signed.fill(0.0)
        self._intent_magnitude.fill(0.0)
        self._intent_confidence.fill(0.0)
        self._instant_activity.fill(0.0)
        self._instant_action.fill(0.0)

    def update_intent(self, rotation_action: np.ndarray, dt: float) -> None:
        action_tolerance = self.frame.T @ np.asarray(rotation_action, dtype=float)
        signal_alpha = 1.0 - np.exp(-dt / _SIGNAL_TIME_CONSTANT_S)
        for stage in range(_STAGE_COUNT):
            sample = action_tolerance if stage == self.active_stage else np.zeros(3)
            self._instant_action[stage] = sample
            self._intent_signed[stage] += signal_alpha * (sample - self._intent_signed[stage])
            self._intent_magnitude[stage] += signal_alpha * (np.abs(sample) - self._intent_magnitude[stage])
            activity = self._smoothstep(self._intent_magnitude[stage], _ACTIVITY_LOW, _ACTIVITY_HIGH)
            coherence = np.divide(np.abs(self._intent_signed[stage]), self._intent_magnitude[stage], out=np.zeros(3), where=self._intent_magnitude[stage] > 1e-12)
            raw_confidence = activity * np.clip(coherence, 0.0, 1.0)
            rising = raw_confidence > self._intent_confidence[stage]
            alpha = np.where(rising, 1.0 - np.exp(-dt / _CONFIDENCE_ATTACK_S), 1.0 - np.exp(-dt / _CONFIDENCE_RELEASE_S))
            self._intent_confidence[stage] += alpha * (raw_confidence - self._intent_confidence[stage])
            self._instant_activity[stage] = self._smoothstep(np.abs(sample), _ACTIVITY_LOW, _ACTIVITY_HIGH)

    def task(self, axis: int, current_rotation: np.ndarray, target_rotation: np.ndarray | None = None, reference_rotation: np.ndarray | None = None) -> AxisTask:
        if self.active_stage in _STRICT_STAGES:
            return AxisTask(TaskKind.SPECIFIC, goal=0.0)
        fixed_stage_reference = self.target.copy() if np.any(self.ranged) else None
        planned_coordinates = (
            self.actual_tolerance(current_rotation)
            if fixed_stage_reference is not None
            else None
        )
        if self.ranged[axis]:
            previous_reference = (
                current_rotation
                if reference_rotation is None
                else np.asarray(reference_rotation, dtype=float)
            )
            if previous_reference.shape != (3, 3):
                raise ValueError("previous rotation reference must be 3x3")
            reference_value = float(
                self.actual_tolerance(previous_reference, target_rotation)[axis]
            )
            fixed_boundary_value = float(self.boundary_tolerance(current_rotation)[axis])
            fixed_value = float(self.actual_tolerance(current_rotation)[axis])
            fixed_center = fixed_value - fixed_boundary_value
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
            if negative_limit <= 0.0 and positive_limit <= 0.0:
                assert planned_coordinates is not None
                goal = float(planned_coordinates[axis])
                if abs(goal) <= 1.0e-12:
                    goal = 0.0
                return AxisTask(
                    TaskKind.SPECIFIC,
                    goal=goal,
                    absolute_rotation_reference=fixed_stage_reference,
                )
            if fixed_boundary_value > positive_limit:
                fixed_lower, fixed_upper = fixed_center - negative_limit, fixed_value
            elif fixed_boundary_value < -negative_limit:
                fixed_lower, fixed_upper = fixed_value, fixed_center + positive_limit
            else:
                fixed_lower, fixed_upper = fixed_center - negative_limit, fixed_center + positive_limit
            instantaneous_release_goal = float(
                np.clip(reference_value, -negative_limit, positive_limit)
            )
            at_fixed_boundary = (
                fixed_boundary_value >= positive_limit - 1e-6
                or fixed_boundary_value <= -negative_limit + 1e-6
            )
            instantaneous_activity = float(self._instant_activity[self.active_stage, axis])
            preference_weight = float(self.intent_preference[axis])
            if at_fixed_boundary:
                instantaneous_activity = preference_weight = 1.0
            return AxisTask(
                TaskKind.FLAT_RANGE,
                lower=fixed_lower,
                upper=fixed_upper,
                preference_goal=0.0,
                preference_weight=preference_weight,
                instantaneous_activity=instantaneous_activity,
                instantaneous_release_goal=instantaneous_release_goal,
                absolute_lower=fixed_lower,
                absolute_upper=fixed_upper,
                absolute_rotation_reference=fixed_stage_reference,
            )
        if fixed_stage_reference is None:
            return AxisTask(TaskKind.SPECIFIC, goal=0.0)
        assert planned_coordinates is not None
        return AxisTask(
            TaskKind.SPECIFIC,
            goal=float(planned_coordinates[axis]),
            absolute_rotation_reference=fixed_stage_reference,
        )

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from planning.tolerance.state import RotationalToleranceState


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

from __future__ import annotations

from dataclasses import dataclass

from planning.action_planner import CartesianActionPlanner


CARTESIAN_REFERENCE_CHOICES = ("min_jerk", "linear", "cubic", "motion_limited")
JOINT_REFERENCE_CHOICES = ("min_jerk", "linear", "cubic")
TRACKER_MODE_CHOICES = ("auto", "pid")
ROUTE_TRACKER_MODE_CHOICES = (
    "auto",
    "pid",
    "cartesian_impedance",
    "joint_impedance",
    "joint_pid",
)


@dataclass(frozen=True)
class ControlRoute:
    """Type-safe routing between independently selected control modules."""

    planner: CartesianActionPlanner
    reference_profile: str
    tracker_mode: str = "auto"

    def __post_init__(self) -> None:
        profile = str(self.reference_profile).lower().strip()
        tracker = str(self.tracker_mode).lower().strip()
        if tracker not in ROUTE_TRACKER_MODE_CHOICES:
            raise ValueError(
                f"tracker_mode must be one of {ROUTE_TRACKER_MODE_CHOICES}; got {self.tracker_mode!r}"
            )
        expected_tracker = (
            "cartesian_impedance" if self.planner.control_mode == "cartesian" else "joint_pid"
        )
        resolved_tracker = expected_tracker if tracker in TRACKER_MODE_CHOICES else tracker
        compatible_trackers = (
            ("cartesian_impedance",)
            if self.planner.control_mode == "cartesian"
            else ("joint_impedance", "joint_pid")
        )
        if resolved_tracker not in compatible_trackers:
            raise ValueError(
                f"planner {self.planner.mode!r} outputs {self.planner.control_mode} references and requires "
                f"tracker in {compatible_trackers!r}; got {resolved_tracker!r}"
            )
        choices = self.reference_choices
        if profile not in choices:
            raise ValueError(
                f"{self.reference_space} reference must be one of {choices}; got {profile!r}"
            )
        object.__setattr__(self, "reference_profile", profile)
        object.__setattr__(self, "tracker_mode", resolved_tracker)

    @property
    def reference_space(self) -> str:
        return self.planner.control_mode

    @property
    def reference_choices(self) -> tuple[str, ...]:
        return (
            CARTESIAN_REFERENCE_CHOICES
            if self.reference_space == "cartesian"
            else JOINT_REFERENCE_CHOICES
        )

    @property
    def control_mode(self) -> str:
        return self.reference_space

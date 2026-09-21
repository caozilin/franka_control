from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np


class TaskKind(str, Enum):
    """Paper task categories plus controller-only flat range and omission."""

    SPECIFIC = "specific"
    RANGE = "range"
    PREFERRED_RANGE = "preferred_range"
    # Controller-side per-cycle admissible interval. Unlike the paper's
    # swamp RANGE, every value inside this interval (including its endpoints)
    # has exactly zero loss.
    FLAT_RANGE = "flat_range"
    FREE = "free"


@dataclass(frozen=True)
class AxisTask:
    """One scalar Cartesian-error task; limits are expressed in error units."""

    kind: TaskKind
    goal: float = 0.0
    lower: float = -1.0
    upper: float = 1.0
    # Optional dynamic intent attraction layered on an equally-valid RANGE.
    # Zero preserves the paper's unmodified ranged-goal task exactly.
    preference_goal: float = 0.0
    preference_weight: float = 0.0
    # Unfiltered activity of the current nominal command in this task axis.
    # Every optimizer combines it with ``preference_weight`` (the filtered
    # memory) so a newly-started VLA command is never delayed by the filter.
    instantaneous_activity: float = 0.0
    # Stage-local EMA magnitude of the most recent nominal rotational action.
    # This is normalized to [0, 1] by RotationalToleranceState and scales the
    # optional loss on this cycle's additional tolerance release. It is
    # independent from the filtered intent preference above.
    ema_nominal_action: float = 0.0
    # Optional absolute pose-error bounds shared by every optimizer backend.
    absolute_lower: float | None = None
    absolute_upper: float | None = None
    # Fixed stage target used to evaluate an absolute ordered-XYZ tolerance
    # coordinate. In a mixed mask this is shared by both Specific axes and
    # ranged walls; nominal release preferences remain cycle-relative.
    absolute_rotation_reference: np.ndarray | None = field(
        default=None,
        compare=False,
        repr=False,
    )


def validate_axis_task(axis: int, task: AxisTask | None) -> None:
    """Validate one shared Cartesian task assignment for every backend."""
    if not 0 <= axis < 6:
        raise IndexError("Cartesian axis must be in [0, 5].")
    if (
        task is not None
        and task.kind not in (TaskKind.FREE, TaskKind.SPECIFIC)
        and task.lower >= task.upper
    ):
        raise ValueError("Range task requires lower < upper.")
    if task is None:
        return
    if not np.isfinite(task.preference_weight) or task.preference_weight < 0.0:
        raise ValueError("Task preference weight must be finite and non-negative.")
    if (
        not np.isfinite(task.instantaneous_activity)
        or not 0.0 <= task.instantaneous_activity <= 1.0
    ):
        raise ValueError("Instantaneous task activity must be in [0, 1].")
    if (
        not np.isfinite(task.ema_nominal_action)
        or not 0.0 <= task.ema_nominal_action <= 1.0
    ):
        raise ValueError("EMA nominal action must be in [0, 1].")
    if (task.absolute_lower is None) != (task.absolute_upper is None):
        raise ValueError("Absolute task bounds must be supplied together.")
    if (
        task.absolute_lower is not None
        and (
            not np.isfinite(task.absolute_lower)
            or not np.isfinite(task.absolute_upper)
            or task.absolute_lower >= task.absolute_upper
        )
    ):
        raise ValueError("Absolute task bounds require finite lower < upper.")
    if task.absolute_rotation_reference is not None:
        reference = np.asarray(task.absolute_rotation_reference, dtype=float)
        if reference.shape != (3, 3) or not np.all(np.isfinite(reference)):
            raise ValueError(
                "Absolute rotation reference must be a finite 3x3 matrix.",
            )


def resolve_axis_tasks(
    overrides: list[AxisTask | None],
    active_dofs: np.ndarray,
) -> tuple[AxisTask, ...]:
    """Resolve UI defaults and absolute flat ranges identically everywhere."""
    resolved: list[AxisTask] = []
    for index, (task, active) in enumerate(zip(
        overrides, np.asarray(active_dofs, dtype=bool), strict=True,
    )):
        task = task if task is not None else AxisTask(
            TaskKind.SPECIFIC if active else TaskKind.FREE,
        )
        if (
            index >= 3
            and task.kind is TaskKind.FLAT_RANGE
            and task.absolute_lower is not None
            and task.absolute_upper is not None
        ):
            task = replace(
                task,
                lower=task.absolute_lower,
                upper=task.absolute_upper,
            )
        resolved.append(task)
    return tuple(resolved)


@dataclass(frozen=True)
class LossParameters:
    """Parameters Ω used by the paper's parametric loss functions."""

    c: float = 0.10
    a1: float = 1.0
    a2: float = 0.05
    a3: float = 100.0
    m: int = 2
    n: int = 20
    b: float | None = None

    @property
    def wall_scale(self) -> float:
        # Equation (7)'s paper setting: 95% of wall height at |x'| = 1.
        if self.b is not None:
            return self.b
        return float((-1.0 / np.log(0.05)) ** (1.0 / self.n))

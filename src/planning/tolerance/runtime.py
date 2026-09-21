"""Shared live/headless rotational-tolerance control operations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable
import numpy as np

from planning.ipopt.controller import IpoptIK
from planning.ipopt.types import TargetPose
from planning.task_space.types import AxisTask, TaskKind
from utils.pose import rotation_matrix
from .release_state import RotationReleaseState, RotationReleaseStep
from .state import RotationalToleranceState

def advance_reference_pose(
    position: np.ndarray,
    rotation: np.ndarray,
    action: np.ndarray,
    *,
    tolerance: RotationalToleranceState | None = None,
    dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the independent nominal planner reference by one increment.

    Rotation is unconditionally left-multiplied. Absolute planner frames and
    source-frame switches are intentionally absent from this API. Runtime
    control, simulation and benchmarks must all integrate the same incremental
    action from the previous nominal reference. When tolerance is active,
    this records rotational intent before applying the nominal increment; the
    physical FK feedback remains an observation input and never replaces this
    nominal reference.
    """
    position_array = np.asarray(position, dtype=float)
    rotation_array = np.asarray(rotation, dtype=float)
    action_array = np.asarray(action, dtype=float)
    if position_array.shape != (3,) or rotation_array.shape != (3, 3):
        raise ValueError("Reference pose must contain a 3-vector and 3x3 rotation")
    if action_array.shape not in ((6,), (7,)) or not np.all(np.isfinite(action_array)):
        raise ValueError("Reference action must contain six or seven finite values")
    rotation_action = action_array[3:6]
    if tolerance is not None:
        if dt is None or not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("A positive finite dt is required with tolerance")
        tolerance.update_intent(rotation_action, float(dt))
        advanced_rotation = tolerance.apply_increment(
            rotation_array, rotation_action,
        )
    else:
        delta = rotation_matrix(rotation_action)
        advanced_rotation = delta @ rotation_array
    return position_array + action_array[:3], advanced_rotation


def configure_rotation_axis_tasks(
    tolerance: RotationalToleranceState,
    controllers: Iterable[object],
) -> tuple[tuple[AxisTask, AxisTask, AxisTask], np.ndarray]:
    """Install final tasks and derive their sole release/constraint mask."""
    tasks = (
        tolerance.task(0),
        tolerance.task(1),
        tolerance.task(2),
    )
    for controller in controllers:
        for axis, task in enumerate(tasks):
            controller.set_axis_task(axis + 3, task)
    release_mask = np.fromiter(
        (task.kind is TaskKind.FLAT_RANGE for task in tasks),
        dtype=bool,
        count=3,
    )
    return tasks, release_mask


def rebase_rotation_subtask(
    target: TargetPose,
    tolerance: RotationalToleranceState,
    release_state: RotationReleaseState,
    *,
    stage_key: Hashable,
    accepted_rotation: np.ndarray,
) -> np.ndarray:
    """Atomically make an accepted optimized rotation the new nominal zero.

    Every simulation entry point must use this operation at a real subtask
    boundary. It resets the action-integrated nominal branch, the persistent
    release coordinates, and all stage-local tolerance filter/intent history
    as one state transition.
    """
    anchor = np.asarray(accepted_rotation, dtype=float)
    if anchor.shape != (3, 3) or not np.all(np.isfinite(anchor)):
        raise ValueError("Accepted rotation must be a finite 3x3 matrix")
    tolerance.reset_subtask_history()
    release_state.rebase_subtask(stage_key, anchor)
    return anchor.copy()


def solve_stage_relative_target(
    controller: IpoptIK,
    measured_q: np.ndarray,
    target: TargetPose,
    tolerance: RotationalToleranceState,
    release_state: RotationReleaseState,
    active_dofs: np.ndarray,
):
    """Execute one canonical stage-relative rotational IPOPT cycle.

    This is the single backend used by interactive, benchmark, and
    counterfactual simulation. It owns task installation, the frozen release
    snapshot, the solve, and the feasible-state commit.
    """
    rotation_tasks, release_mask = configure_rotation_axis_tasks(
        tolerance, (controller,),
    )
    active_dofs[:3] = True
    active_dofs[3:] = ~release_mask
    release_step = release_state.prepare_cycle(
        target.rotation,
        tolerance.frame,
        release_mask,
        -tolerance.negative_limits,
        tolerance.positive_limits,
    )
    joint_target, diagnostics = controller.solve(
        measured_q,
        target,
        active_dofs,
        tolerance.frame,
        rotation_release_step=release_step,
    )
    if diagnostics.feasible:
        release_state.commit(joint_target, release_step)
    return (
        joint_target,
        diagnostics,
        rotation_tasks,
        release_mask,
        release_step,
    )


__all__ = (
    "advance_reference_pose",
    "configure_rotation_axis_tasks",
    "rebase_rotation_subtask",
    "solve_stage_relative_target",
)

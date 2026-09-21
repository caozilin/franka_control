"""Pure fixed-chart projection primitives for rotational tolerance."""

from __future__ import annotations

import numpy as np

from utils.pose import (
    rotation_from_tolerance_coordinates,
    rotation_tolerance_coordinates,
)


def constraint_consistent_release_loss_reference(
    stage_reference_rotation: np.ndarray,
    carried_rotation: np.ndarray,
    tolerance_frame: np.ndarray,
    ranged_axes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a physical carry pose only to define incremental-loss zero.

    Tolerant coordinates are retained and Specific coordinates are zeroed in
    the current frozen chart.  This stateless operation deliberately has no
    tolerance-bound inputs and never changes a joint-space solution.
    """
    reference = np.asarray(stage_reference_rotation, dtype=float)
    carried = np.asarray(carried_rotation, dtype=float)
    frame = np.asarray(tolerance_frame, dtype=float)
    ranged = np.asarray(ranged_axes, dtype=bool)
    if (
        reference.shape != (3, 3)
        or carried.shape != (3, 3)
        or frame.shape != (3, 3)
        or ranged.shape != (3,)
    ):
        raise ValueError("Invalid release-loss projection state")

    carried_coordinates = rotation_tolerance_coordinates(
        carried, reference, frame,
    )
    projected_coordinates = np.zeros(3, dtype=float)
    projected_coordinates[ranged] = carried_coordinates[ranged]

    projected = rotation_from_tolerance_coordinates(
        reference, frame, projected_coordinates,
    )
    return projected, projected_coordinates.copy()


__all__ = ("constraint_consistent_release_loss_reference",)

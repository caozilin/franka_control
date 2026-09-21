from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConstraintValues:
    """Fresh nonlinear constraint values for one candidate configuration."""

    equality: np.ndarray
    inequality: np.ndarray
    position_residual: float
    rotation_residual: float
    inequality_violation: float

    @property
    def violation_l1(self) -> float:
        return float(
            np.sum(np.abs(self.equality))
            + np.sum(np.maximum(-self.inequality, 0.0))
        )


@dataclass(frozen=True)
class CandidateValidation:
    """Solver-independent nonlinear feasibility result."""

    finite: bool
    within_joint_bounds: bool
    position_residual: float
    strict_rotation_residual: float
    inequality_violation: float
    feasible: bool


def validate_candidate(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    constraints: ConstraintValues,
    *,
    position_tolerance: float,
    rotation_tolerance: float,
    inequality_tolerance: float,
    joint_tolerance: float = 1.0e-10,
) -> CandidateValidation:
    """Validate a candidate from freshly evaluated nonlinear quantities.

    Callers must evaluate ``constraints`` at ``q`` rather than reuse a
    solver's internal residual report.
    """

    q = np.asarray(q, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    finite = bool(
        np.all(np.isfinite(q))
        and np.all(np.isfinite(constraints.equality))
        and np.all(np.isfinite(constraints.inequality))
        and np.isfinite(constraints.position_residual)
        and np.isfinite(constraints.rotation_residual)
        and np.isfinite(constraints.inequality_violation)
    )
    bounds_shape_ok = q.shape == lower.shape == upper.shape
    within_joint_bounds = bool(
        finite
        and bounds_shape_ok
        and np.all(q >= lower - joint_tolerance)
        and np.all(q <= upper + joint_tolerance)
    )
    position_residual = float(constraints.position_residual)
    strict_rotation_residual = float(constraints.rotation_residual)
    inequality_violation = float(constraints.inequality_violation)
    feasible = bool(
        within_joint_bounds
        and position_residual <= position_tolerance
        and strict_rotation_residual <= rotation_tolerance
        and inequality_violation <= inequality_tolerance
    )
    return CandidateValidation(
        finite=finite,
        within_joint_bounds=within_joint_bounds,
        position_residual=position_residual,
        strict_rotation_residual=strict_rotation_residual,
        inequality_violation=inequality_violation,
        feasible=feasible,
    )

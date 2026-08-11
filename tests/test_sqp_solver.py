from __future__ import annotations

import numpy as np

from planning.sqp import ConstraintValues, FullSpaceSQPSolver, SQPSettings
from planning.sqp.qp import SmallActiveSetQP


def test_active_set_qp_matches_baseline_equality_and_bound_case() -> None:
    result = SmallActiveSetQP().solve(
        np.eye(2),
        np.zeros(2),
        np.array(((1.0, 1.0),)),
        np.array((1.0,)),
        np.array(((-1.0, 0.0),)),
        np.array((-0.8,)),
    )
    assert result.success, result.status
    np.testing.assert_allclose(result.step, (0.8, 0.2), atol=1e-9)


def test_sqp_elastic_phase_reaches_hard_equality() -> None:
    solver = FullSpaceSQPSolver(SQPSettings(
        max_iterations=10,
        max_time_s=2.0,
        trust_region=0.2,
        position_tolerance=1e-6,
    ))

    def objective(value: np.ndarray, breakdown: bool):
        del value
        return (0.0, {}) if breakdown else 0.0

    def constraints(value: np.ndarray) -> ConstraintValues:
        error = np.array((value[0] - 1.0,))
        return ConstraintValues(
            equality=error,
            inequality=np.empty(0),
            position_residual=abs(float(error[0])),
            rotation_residual=0.0,
            inequality_violation=0.0,
        )

    result = solver.solve(
        objective,
        constraints,
        np.array((0.0,)),
        np.array((-2.0,)),
        np.array((2.0,)),
        constraint_jacobian=lambda _q, _v: (np.ones((1, 1)), np.empty((0, 1))),
        objective_gradient=lambda q: np.zeros_like(q),
        elastic_phase_one=True,
    )
    assert result.feasible, result.status
    np.testing.assert_allclose(result.q, (1.0,), atol=1e-6)

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from planning.sqp.qp import SmallActiveSetQP
from planning.sqp.types import SQPSettings


@dataclass(frozen=True)
class ConstraintValues:
    equality: np.ndarray
    inequality: np.ndarray
    position_residual: float
    rotation_residual: float
    inequality_violation: float

    @property
    def violation_l1(self) -> float:
        return float(np.sum(np.abs(self.equality)) + np.sum(np.maximum(-self.inequality, 0.0)))


@dataclass(frozen=True)
class SQPSolverResult:
    q: np.ndarray
    cost: float
    iterations: int
    qp_iterations: int
    optimality: float
    elapsed_ms: float
    converged: bool
    feasible: bool
    status: str
    constraints: ConstraintValues
    task_costs: dict[str, float]


class FullSpaceSQPSolver:
    """Original line-search SQP with damped BFGS and active-set QP."""

    def __init__(self, settings: SQPSettings = SQPSettings()) -> None:
        self.settings = settings
        self.qp = SmallActiveSetQP(settings.max_qp_iterations)
        self._hessian: np.ndarray | None = None
        self._constraint_signature: tuple[int, int, int] | None = None

    def reset(self) -> None:
        self._hessian = None
        self._constraint_signature = None

    def _gradient(
        self,
        objective: Callable[[np.ndarray], float],
        q: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        base = objective(q)
        gradient = np.zeros_like(q)
        epsilon = self.settings.derivative_epsilon
        for index in range(q.size):
            probe = q.copy()
            step = epsilon if q[index] + epsilon <= upper[index] else -epsilon
            if q[index] + step < lower[index]:
                step = 0.5 * (upper[index] - lower[index])
            if abs(step) <= np.finfo(float).eps:
                continue
            probe[index] += step
            gradient[index] = (objective(probe) - base) / step
        return gradient

    def _constraint_jacobians(
        self,
        constraints: Callable[[np.ndarray], ConstraintValues],
        q: np.ndarray,
        values: ConstraintValues,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        equality = np.empty((values.equality.size, q.size))
        inequality = np.empty((values.inequality.size, q.size))
        epsilon = self.settings.derivative_epsilon
        for index in range(q.size):
            probe = q.copy()
            step = epsilon if q[index] + epsilon <= upper[index] else -epsilon
            if q[index] + step < lower[index]:
                step = 0.5 * (upper[index] - lower[index])
            if abs(step) <= np.finfo(float).eps:
                equality[:, index] = 0.0
                inequality[:, index] = 0.0
                continue
            probe[index] += step
            perturbed = constraints(probe)
            if values.equality.size:
                equality[:, index] = (perturbed.equality - values.equality) / step
            if values.inequality.size:
                inequality[:, index] = (perturbed.inequality - values.inequality) / step
        return equality, inequality

    @staticmethod
    def _damped_bfgs(
        hessian: np.ndarray,
        step: np.ndarray,
        gradient_delta: np.ndarray,
    ) -> np.ndarray:
        if float(np.linalg.norm(step)) <= 1e-12:
            return hessian
        hessian_step = hessian @ step
        curvature_model = float(step @ hessian_step)
        curvature_observed = float(step @ gradient_delta)
        if curvature_model <= 1e-14:
            return np.eye(step.size)
        if curvature_observed < 0.2 * curvature_model:
            theta = 0.8 * curvature_model / max(curvature_model - curvature_observed, 1e-14)
            gradient_delta = theta * gradient_delta + (1.0 - theta) * hessian_step
        denominator = float(step @ gradient_delta)
        if denominator <= 1e-14:
            return hessian
        updated = (
            hessian
            - np.outer(hessian_step, hessian_step) / curvature_model
            + np.outer(gradient_delta, gradient_delta) / denominator
        )
        return 0.5 * (updated + updated.T)

    def _is_feasible(self, values: ConstraintValues) -> bool:
        return (
            values.position_residual <= self.settings.position_tolerance
            and values.rotation_residual <= self.settings.rotation_tolerance
            and values.inequality_violation <= self.settings.inequality_tolerance
        )

    def solve(
        self,
        objective_with_breakdown: Callable[
            [np.ndarray, bool], float | tuple[float, dict[str, float]]
        ],
        constraints: Callable[[np.ndarray], ConstraintValues],
        seed: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        constraint_jacobian: Callable[
            [np.ndarray, ConstraintValues], tuple[np.ndarray, np.ndarray]
        ] | None = None,
        objective_gradient: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        elastic_phase_one: bool = False,
    ) -> SQPSolverResult:
        started = time.perf_counter()
        q = np.clip(np.asarray(seed, dtype=float).copy(), lower, upper)
        qp_iterations = 0
        optimality = float("inf")
        status = "major iteration limit"
        converged = False
        objective_cache: dict[bytes, float] = {}

        def objective(value: np.ndarray) -> float:
            key = np.ascontiguousarray(value).tobytes()
            cached = objective_cache.get(key)
            if cached is None:
                cached = float(objective_with_breakdown(value, False))
                objective_cache[key] = cached
            return cached

        if objective_gradient is None:
            def gradient_at(value: np.ndarray) -> np.ndarray:
                return self._gradient(objective, value, lower, upper)
        else:
            gradient_at = objective_gradient

        cost = objective(q)
        gradient = gradient_at(q)
        values = constraints(q)
        signature = (q.size, values.equality.size, values.inequality.size)
        hessian = (
            self._hessian.copy()
            if self._hessian is not None and self._constraint_signature == signature
            else np.eye(q.size)
        )
        if constraint_jacobian is None:
            def jacobian(
                value: np.ndarray,
                evaluated: ConstraintValues,
            ) -> tuple[np.ndarray, np.ndarray]:
                return self._constraint_jacobians(constraints, value, evaluated, lower, upper)
        else:
            jacobian = constraint_jacobian
        equality_jacobian, inequality_jacobian = jacobian(q, values)
        iterations = 0

        for iterations in range(1, self.settings.max_iterations + 1):
            if time.perf_counter() - started >= self.settings.max_time_s:
                status = "time limit"
                break
            identity = np.eye(q.size)
            step_lower = np.maximum(lower - q, -self.settings.trust_region)
            step_upper = np.minimum(upper - q, self.settings.trust_region)
            inequality_matrix = np.vstack((-inequality_jacobian, identity, -identity))
            inequality_rhs = np.concatenate((values.inequality, step_upper, -step_lower))
            phase_one = elastic_phase_one and not self._is_feasible(values) and equality_jacobian.size > 0
            if phase_one:
                penalty = self.settings.merit_penalty
                qp_hessian = hessian + penalty * equality_jacobian.T @ equality_jacobian
                qp_gradient = gradient + penalty * equality_jacobian.T @ values.equality
                qp_equality_jacobian = np.empty((0, q.size))
                qp_equality_rhs = np.empty(0)
            else:
                qp_hessian = hessian
                qp_gradient = gradient
                qp_equality_jacobian = equality_jacobian
                qp_equality_rhs = -values.equality
            box_only = qp_equality_jacobian.shape[0] == 0 and inequality_jacobian.shape[0] == 0
            qp_result = (
                self.qp.solve_box(
                    qp_hessian, qp_gradient, step_lower, step_upper,
                    tolerance=self.settings.inequality_tolerance,
                )
                if box_only
                else self.qp.solve(
                    qp_hessian, qp_gradient, qp_equality_jacobian, qp_equality_rhs,
                    inequality_matrix, inequality_rhs,
                    tolerance=self.settings.inequality_tolerance,
                )
            )
            qp_iterations += qp_result.iterations
            if box_only and not qp_result.success:
                hessian = np.eye(q.size)
                qp_result = self.qp.solve_box(
                    hessian, qp_gradient, step_lower, step_upper,
                    tolerance=self.settings.inequality_tolerance,
                )
                qp_iterations += qp_result.iterations
            if (
                not qp_result.success
                and elastic_phase_one
                and not phase_one
                and equality_jacobian.size > 0
            ):
                phase_one = True
                penalty = self.settings.merit_penalty
                qp_result = self.qp.solve(
                    hessian + penalty * equality_jacobian.T @ equality_jacobian,
                    gradient + penalty * equality_jacobian.T @ values.equality,
                    np.empty((0, q.size)), np.empty(0), inequality_matrix, inequality_rhs,
                    tolerance=self.settings.inequality_tolerance,
                )
                qp_iterations += qp_result.iterations
            if not qp_result.success:
                status = f"QP failed: {qp_result.status}"
                break

            step = qp_result.step
            optimality = float(np.linalg.norm(step, ord=np.inf))
            if self._is_feasible(values) and optimality <= self.settings.step_tolerance:
                converged = True
                status = "converged"
                break

            merit = cost + self.settings.merit_penalty * values.violation_l1
            merit_direction = float(gradient @ step) - self.settings.merit_penalty * values.violation_l1
            accepted = False
            best: tuple[float, np.ndarray, float, ConstraintValues] | None = None
            alpha = 1.0
            for _ in range(self.settings.max_line_search_iterations + 1):
                candidate = q + alpha * step
                candidate_cost = objective(candidate)
                candidate_values = constraints(candidate)
                candidate_merit = candidate_cost + self.settings.merit_penalty * candidate_values.violation_l1
                if best is None or candidate_merit < best[0]:
                    best = candidate_merit, candidate, candidate_cost, candidate_values
                if candidate_merit <= merit + 1e-4 * alpha * merit_direction:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                if best is None or best[0] >= merit:
                    status = "merit line search failed"
                    break
                _, candidate, candidate_cost, candidate_values = best

            new_gradient = gradient_at(candidate)
            new_equality_jacobian, new_inequality_jacobian = jacobian(candidate, candidate_values)
            if phase_one:
                hessian = np.eye(q.size)
            else:
                equality_count = equality_jacobian.shape[0]
                equality_multipliers = qp_result.multipliers[:equality_count]
                lagrangian_gradient = gradient + equality_jacobian.T @ equality_multipliers
                new_lagrangian_gradient = new_gradient + new_equality_jacobian.T @ equality_multipliers
                active_multipliers = qp_result.multipliers[equality_count:]
                for row, multiplier in zip(qp_result.active_set, active_multipliers, strict=True):
                    if row < inequality_jacobian.shape[0]:
                        lagrangian_gradient -= multiplier * inequality_jacobian[row]
                        new_lagrangian_gradient -= multiplier * new_inequality_jacobian[row]
                hessian = self._damped_bfgs(
                    hessian, candidate - q, new_lagrangian_gradient - lagrangian_gradient,
                )
            q = candidate
            cost = candidate_cost
            values = candidate_values
            gradient = new_gradient
            equality_jacobian = new_equality_jacobian
            inequality_jacobian = new_inequality_jacobian

        feasible = self._is_feasible(values)
        if feasible and np.all(np.isfinite(hessian)):
            self._hessian = hessian.copy()
            self._constraint_signature = signature
        if feasible and not converged and status in {"major iteration limit", "time limit"}:
            status = f"feasible ({status})"
        final_cost, task_costs = objective_with_breakdown(q, True)
        return SQPSolverResult(
            q=q,
            cost=float(final_cost),
            iterations=iterations,
            qp_iterations=qp_iterations,
            optimality=optimality,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            converged=converged,
            feasible=feasible,
            status=status,
            constraints=values,
            task_costs=task_costs,
        )

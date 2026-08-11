from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QPResult:
    step: np.ndarray
    multipliers: np.ndarray
    active_set: tuple[int, ...]
    iterations: int
    success: bool
    status: str


class SmallActiveSetQP:
    """Dense active-set QP used by the original constrained SQP."""

    _REGULARIZATION = 1e-9
    _RANK_TOLERANCE = 1e-10

    def __init__(self, max_iterations: int = 32) -> None:
        self.max_iterations = max_iterations

    def solve_box(
        self,
        hessian: np.ndarray,
        gradient: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        tolerance: float = 1e-9,
    ) -> QPResult:
        n = gradient.size
        hessian = 0.5 * (hessian + hessian.T)
        hessian = hessian + self._REGULARIZATION * np.eye(n)
        diagonal = np.diag(hessian)
        if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
            return QPResult(np.zeros(n), np.empty(0), (), 0, False, "non-positive box-QP diagonal")
        step = np.clip(np.zeros(n), lower, upper)
        maximum_sweeps = max(32, 2 * self.max_iterations)
        projected_residual = float("inf")
        for iteration in range(1, maximum_sweeps + 1):
            full_gradient = hessian @ step + gradient
            projected = step - np.clip(step - full_gradient, lower, upper)
            projected_residual = float(np.linalg.norm(projected, ord=np.inf))
            if projected_residual <= max(tolerance, 1e-10):
                break
            bound_tolerance = max(tolerance, 1e-10)
            active_lower = (step <= lower + bound_tolerance) & (full_gradient > 0.0)
            active_upper = (step >= upper - bound_tolerance) & (full_gradient < 0.0)
            active_mask = active_lower | active_upper
            free = np.flatnonzero(~active_mask)
            if free.size == 0:
                break
            active_indices = np.flatnonzero(active_mask)
            rhs = -gradient[free]
            if active_indices.size:
                rhs -= hessian[np.ix_(free, active_indices)] @ step[active_indices]
            free_hessian = hessian[np.ix_(free, free)]
            try:
                free_solution = np.linalg.solve(free_hessian, rhs)
            except np.linalg.LinAlgError:
                free_solution, *_ = np.linalg.lstsq(free_hessian, rhs, rcond=1e-12)
            direction = np.zeros(n)
            direction[free] = free_solution - step[free]
            alpha = 1.0
            for index in free:
                if direction[index] > 0.0:
                    alpha = min(alpha, (upper[index] - step[index]) / direction[index])
                elif direction[index] < 0.0:
                    alpha = min(alpha, (lower[index] - step[index]) / direction[index])
            if alpha <= 1e-14:
                step = np.clip(step, lower, upper)
                continue
            step = np.clip(step + alpha * direction, lower, upper)
        if not np.all(np.isfinite(step)):
            return QPResult(np.zeros(n), np.empty(0), (), iteration, False, "non-finite box-QP solution")
        active_upper = np.flatnonzero(np.isclose(step, upper, atol=max(tolerance, 1e-10), rtol=0.0))
        active_lower = np.flatnonzero(np.isclose(step, lower, atol=max(tolerance, 1e-10), rtol=0.0))
        active = tuple(int(value) for value in np.concatenate((active_upper, n + active_lower)))
        success = projected_residual <= max(1e-7, 10.0 * tolerance)
        return QPResult(
            step, np.zeros(len(active)), active, iteration, success,
            "solved" if success else "box-QP iteration limit",
        )

    @staticmethod
    def _solve_kkt(
        hessian: np.ndarray,
        gradient: np.ndarray,
        matrix: np.ndarray,
        rhs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n = gradient.size
        if matrix.size == 0:
            return np.linalg.solve(hessian, -gradient), np.empty(0), 0.0
        kkt = np.block([
            [hessian, matrix.T],
            [matrix, np.zeros((matrix.shape[0], matrix.shape[0]))],
        ])
        solution, *_ = np.linalg.lstsq(kkt, np.concatenate((-gradient, rhs)), rcond=1e-11)
        step = solution[:n]
        multipliers = solution[n:]
        residual = float(np.linalg.norm(matrix @ step - rhs, ord=np.inf))
        return step, multipliers, residual

    def solve(
        self,
        hessian: np.ndarray,
        gradient: np.ndarray,
        equality_matrix: np.ndarray,
        equality_rhs: np.ndarray,
        inequality_matrix: np.ndarray,
        inequality_rhs: np.ndarray,
        *,
        tolerance: float = 1e-9,
    ) -> QPResult:
        n = gradient.size
        hessian = 0.5 * (hessian + hessian.T) + self._REGULARIZATION * np.eye(n)
        equality_matrix = np.asarray(equality_matrix, dtype=float).reshape(-1, n)
        equality_rhs = np.asarray(equality_rhs, dtype=float).reshape(-1)
        inequality_matrix = np.asarray(inequality_matrix, dtype=float).reshape(-1, n)
        inequality_rhs = np.asarray(inequality_rhs, dtype=float).reshape(-1)
        active: list[int] = []

        for iteration in range(1, self.max_iterations + 1):
            active_matrix = inequality_matrix[active] if active else np.empty((0, n), dtype=float)
            matrix = np.vstack((equality_matrix, active_matrix))
            rhs = np.concatenate((equality_rhs, inequality_rhs[active]))
            step, multipliers, equality_residual = self._solve_kkt(hessian, gradient, matrix, rhs)
            if not np.all(np.isfinite(step)):
                return QPResult(np.zeros(n), np.empty(0), tuple(active), iteration, False, "non-finite KKT solution")
            if equality_residual > max(1e-7, 10.0 * tolerance):
                return QPResult(step, multipliers, tuple(active), iteration, False, "inconsistent active constraints")

            violations = inequality_matrix @ step - inequality_rhs
            inactive = np.ones(inequality_rhs.size, dtype=bool)
            if active:
                inactive[active] = False
            if np.any(inactive):
                candidate = int(np.argmax(np.where(inactive, violations, -np.inf)))
                if violations[candidate] > tolerance:
                    active_multipliers = multipliers[equality_matrix.shape[0]:]
                    if active and np.min(active_multipliers) < -tolerance:
                        del active[int(np.argmin(active_multipliers))]
                        continue
                    candidate_row = inequality_matrix[candidate:candidate + 1]
                    old_rank = np.linalg.matrix_rank(matrix, tol=self._RANK_TOLERANCE)
                    new_rank = np.linalg.matrix_rank(np.vstack((matrix, candidate_row)), tol=self._RANK_TOLERANCE)
                    if new_rank == old_rank:
                        return QPResult(step, multipliers, tuple(active), iteration, False, "dependent violated constraint")
                    active.append(candidate)
                    continue

            active_multipliers = multipliers[equality_matrix.shape[0]:]
            if active and np.min(active_multipliers) < -tolerance:
                del active[int(np.argmin(active_multipliers))]
                continue
            return QPResult(step, multipliers, tuple(active), iteration, True, "solved")

        return QPResult(step, multipliers, tuple(active), self.max_iterations, False, "active-set iteration limit")

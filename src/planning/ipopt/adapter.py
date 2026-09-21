from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Callable

import numpy as np

from .validation import (
    CandidateValidation,
    ConstraintValues,
    validate_candidate,
)


class IpoptUnavailableError(RuntimeError):
    """Raised when the optional cyipopt backend cannot be imported."""


# Successful/acceptable solves and resource-limit exits may publish only after
# the independent nonlinear feasibility check passes. IPOPT status 6 is its
# explicit "feasible point found" termination.
ACCEPTABLE_IPOPT_STATUS_CODES = frozenset((0, 1, 6, -1, -4))


@dataclass(frozen=True)
class IpoptSettings:
    """Numerical settings for direct-joint IPOPT solves."""

    max_iterations: int = 20
    max_cpu_time_s: float = 0.095
    tolerance: float = 1.0e-7
    acceptable_tolerance: float = 1.0e-5
    acceptable_iterations: int = 3
    print_level: int = 0
    derivative_test: str = "none"
    derivative_epsilon: float = 1.0e-7
    expensive_gradient_refresh_rad: float = 0.015
    position_tolerance: float = 1.0e-5
    rotation_tolerance: float = 1.0e-4
    inequality_tolerance: float = 1.0e-7

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        positive = (
            self.max_cpu_time_s,
            self.tolerance,
            self.acceptable_tolerance,
            self.derivative_epsilon,
            self.position_tolerance,
            self.rotation_tolerance,
            self.inequality_tolerance,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("IPOPT time and tolerances must be finite and positive")
        if self.acceptable_iterations <= 0:
            raise ValueError("acceptable_iterations must be positive")
        if (
            not np.isfinite(self.expensive_gradient_refresh_rad)
            or self.expensive_gradient_refresh_rad < 0.0
        ):
            raise ValueError(
                "expensive_gradient_refresh_rad must be finite and non-negative",
            )
        if self.derivative_test not in ("none", "first-order"):
            raise ValueError("derivative_test must be 'none' or 'first-order'")


@dataclass(frozen=True)
class IpoptResult:
    q: np.ndarray
    cost: float
    elapsed_ms: float
    iterations: int
    optimality: float
    status_code: int
    status: str
    solver_succeeded: bool
    validation: CandidateValidation
    constraints: ConstraintValues


class _CyipoptCallbacks:
    def __init__(
        self,
        variable_count: int,
        equality_count: int,
        inequality_count: int,
        objective: Callable[[np.ndarray], float],
        gradient: Callable[[np.ndarray], np.ndarray],
        constraints: Callable[[np.ndarray], ConstraintValues],
        jacobian: Callable[
            [np.ndarray, ConstraintValues], tuple[np.ndarray, np.ndarray]
        ],
    ) -> None:
        self._variable_count = variable_count
        self._equality_count = equality_count
        self._inequality_count = inequality_count
        self._objective = objective
        self._gradient = gradient
        self._constraints = constraints
        self._jacobian = jacobian
        self._constraint_cache: dict[bytes, ConstraintValues] = {}
        self.iterations = 0

    @staticmethod
    def _key(q: np.ndarray) -> bytes:
        return np.ascontiguousarray(q, dtype=float).tobytes()

    def _constraint_values(self, q: np.ndarray) -> ConstraintValues:
        key = self._key(q)
        cached = self._constraint_cache.get(key)
        if cached is None:
            cached = self._constraints(np.asarray(q, dtype=float))
            if cached.equality.shape != (self._equality_count,):
                raise ValueError("equality constraint count changed during solve")
            if cached.inequality.shape != (self._inequality_count,):
                raise ValueError("inequality constraint count changed during solve")
            self._constraint_cache[key] = cached
        return cached

    def objective(self, q: np.ndarray) -> float:
        return float(self._objective(np.asarray(q, dtype=float)))

    def gradient(self, q: np.ndarray) -> np.ndarray:
        gradient = np.asarray(
            self._gradient(np.asarray(q, dtype=float)), dtype=float,
        )
        if gradient.shape != (self._variable_count,):
            raise ValueError("objective gradient has an unexpected shape")
        return gradient

    def constraints(self, q: np.ndarray) -> np.ndarray:
        values = self._constraint_values(q)
        return np.concatenate((values.equality, values.inequality))

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        values = self._constraint_values(q)
        equality, inequality = self._jacobian(
            np.asarray(q, dtype=float), values,
        )
        if equality.shape != (self._equality_count, self._variable_count):
            raise ValueError("equality Jacobian has an unexpected shape")
        if inequality.shape != (
            self._inequality_count, self._variable_count,
        ):
            raise ValueError("inequality Jacobian has an unexpected shape")
        dense = np.vstack((equality, inequality))
        return np.asarray(dense, dtype=float).ravel(order="C")

    def intermediate(
        self,
        _algorithm_mode,
        iteration_count,
        _objective_value,
        _primal_infeasibility,
        _dual_infeasibility,
        _barrier_parameter,
        _d_norm,
        _regularization_size,
        _alpha_dual,
        _alpha_primal,
        _line_search_trials,
    ) -> bool:
        self.iterations = max(self.iterations, int(iteration_count))
        return True


class IpoptAdapter:
    """Rebuild a small dense cyipopt problem for one single-step NLP."""

    def __init__(self, settings: IpoptSettings = IpoptSettings()) -> None:
        self.settings = settings

    @staticmethod
    def _load_cyipopt():
        try:
            import cyipopt
        except (ImportError, OSError) as error:
            raise IpoptUnavailableError(
                "IPOPT backend is unavailable. Install the project 'ipopt' "
                "extra and a native IPOPT library.",
            ) from error
        # cyipopt 1.7 logs every Python callback at INFO independently of
        # IPOPT's print_level.  Keep warnings/errors while leaving the host
        # application's INFO logging untouched.
        cyipopt.set_logging_level(logging.WARNING)
        return cyipopt

    def solve(
        self,
        objective: Callable[[np.ndarray], float],
        gradient: Callable[[np.ndarray], np.ndarray],
        constraints: Callable[[np.ndarray], ConstraintValues],
        constraint_jacobian: Callable[
            [np.ndarray, ConstraintValues], tuple[np.ndarray, np.ndarray]
        ],
        seed: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        position_tolerance: float,
        rotation_tolerance: float,
        inequality_tolerance: float,
    ) -> IpoptResult:
        cyipopt = self._load_cyipopt()
        seed = np.asarray(seed, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        if seed.ndim != 1 or seed.shape != lower.shape or seed.shape != upper.shape:
            raise ValueError("seed and joint bounds must be equal-sized vectors")
        if not all(np.all(np.isfinite(value)) for value in (seed, lower, upper)):
            raise ValueError("seed and joint bounds must be finite")
        seed = np.clip(seed, lower, upper)
        initial_constraints = constraints(seed)
        equality_count = initial_constraints.equality.size
        inequality_count = initial_constraints.inequality.size
        constraint_lower = np.concatenate((
            np.zeros(equality_count),
            np.zeros(inequality_count),
        ))
        constraint_upper = np.concatenate((
            np.zeros(equality_count),
            np.full(inequality_count, np.inf),
        ))
        callbacks = _CyipoptCallbacks(
            seed.size,
            equality_count,
            inequality_count,
            objective,
            gradient,
            constraints,
            constraint_jacobian,
        )
        problem = cyipopt.Problem(
            n=seed.size,
            m=equality_count + inequality_count,
            problem_obj=callbacks,
            lb=lower,
            ub=upper,
            cl=constraint_lower,
            cu=constraint_upper,
        )
        options = {
            "hessian_approximation": "limited-memory",
            # IPOPT otherwise relaxes variable bounds before solving.  The
            # objective's bound-aware finite differences deliberately use the
            # robot's physical joint limits, so a callback at a relaxed point
            # would violate that contract even when the excess is numerical.
            "bound_relax_factor": 0.0,
            "max_iter": self.settings.max_iterations,
            "max_cpu_time": self.settings.max_cpu_time_s,
            "tol": self.settings.tolerance,
            "acceptable_tol": self.settings.acceptable_tolerance,
            "acceptable_iter": self.settings.acceptable_iterations,
            "print_level": self.settings.print_level,
            "sb": "yes",
        }
        if self.settings.derivative_test != "none":
            options["derivative_test"] = self.settings.derivative_test
        for name, value in options.items():
            problem.add_option(name, value)

        started = time.perf_counter()
        q, info = problem.solve(seed)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        q = np.asarray(q, dtype=float)
        final_constraints = constraints(q)
        validation = validate_candidate(
            q,
            lower,
            upper,
            final_constraints,
            position_tolerance=position_tolerance,
            rotation_tolerance=rotation_tolerance,
            inequality_tolerance=inequality_tolerance,
        )
        status_code = int(info.get("status", -999))
        status_message = info.get("status_msg", "unknown IPOPT status")
        if isinstance(status_message, bytes):
            status_message = status_message.decode(errors="replace")
        objective_value = info.get("obj_val")
        if objective_value is None:
            objective_value = objective(q)
        equality_jacobian, inequality_jacobian = constraint_jacobian(
            q, final_constraints,
        )
        combined_jacobian = np.vstack((
            equality_jacobian,
            inequality_jacobian,
        ))
        constraint_multipliers = np.asarray(
            info.get("mult_g", np.zeros(combined_jacobian.shape[0])),
            dtype=float,
        )
        lower_multipliers = np.asarray(
            info.get("mult_x_L", np.zeros(q.size)), dtype=float,
        )
        upper_multipliers = np.asarray(
            info.get("mult_x_U", np.zeros(q.size)), dtype=float,
        )
        lagrangian_gradient = (
            np.asarray(gradient(q), dtype=float)
            + combined_jacobian.T @ constraint_multipliers
            - lower_multipliers
            + upper_multipliers
        )
        optimality = float(np.linalg.norm(lagrangian_gradient, ord=np.inf))
        return IpoptResult(
            q=q,
            cost=float(objective_value),
            elapsed_ms=float(elapsed_ms),
            iterations=callbacks.iterations,
            optimality=optimality,
            status_code=status_code,
            status=str(status_message),
            solver_succeeded=status_code in (0, 1),
            validation=validation,
            constraints=final_constraints,
        )

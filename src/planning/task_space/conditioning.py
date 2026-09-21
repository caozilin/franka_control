from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


TASK_VELOCITY_SCALES = np.array(
    (0.10, 0.10, 0.10, np.pi / 4.0, np.pi / 4.0, np.pi / 4.0),
    dtype=float,
)


@dataclass(frozen=True)
class ConditioningSafeguardConfig:
    """Two-threshold smooth conditioning-safeguard parameters."""

    enabled: bool = False
    eta_low: float = 0.05
    eta_high: float = 0.10
    weight: float = 0.10
    sigma_epsilon: float = 1.0e-12
    finite_difference_relative_step: float = 1.0e-5

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.eta_low,
                self.eta_high,
                self.weight,
                self.sigma_epsilon,
                self.finite_difference_relative_step,
            ),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Conditioning safeguard parameters must be finite")
        if not 0.0 <= self.eta_low < self.eta_high <= 1.0:
            raise ValueError(
                "Conditioning thresholds must satisfy "
                "0 <= eta_low < eta_high <= 1",
            )
        if self.weight < 0.0:
            raise ValueError("Conditioning weight must be non-negative")
        if self.sigma_epsilon <= 0.0:
            raise ValueError("Conditioning sigma epsilon must be positive")
        if self.finite_difference_relative_step <= 0.0:
            raise ValueError(
                "Conditioning finite-difference relative step must be positive",
            )


@dataclass(frozen=True)
class ConditioningMetrics:
    sigma_min_raw: float
    sigma_min_normalized: float
    sigma_max_normalized: float
    sigma_min_gap_normalized: float
    inverse_condition_index: float


@dataclass(frozen=True)
class ConditioningGradientConsistency:
    gradient_norm: float
    relative_error_half_step: float
    relative_error_double_step: float
    cosine_half_step: float
    cosine_double_step: float


def _direction_cosine(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1.0e-15 and second_norm <= 1.0e-15:
        return 1.0
    if first_norm <= 1.0e-15 or second_norm <= 1.0e-15:
        return 0.0
    return float(np.clip(
        float(first @ second) / (first_norm * second_norm), -1.0, 1.0,
    ))


def conditioning_gradient_consistency(
    gradient_half_step: np.ndarray,
    gradient: np.ndarray,
    gradient_double_step: np.ndarray,
) -> ConditioningGradientConsistency:
    """Compare numerical conditioning gradients across h/2, h, and 2h."""
    half = np.asarray(gradient_half_step, dtype=float)
    base = np.asarray(gradient, dtype=float)
    double = np.asarray(gradient_double_step, dtype=float)
    if half.shape != base.shape or double.shape != base.shape:
        raise ValueError("Conditioning gradient samples must have equal shape")
    if not all(np.all(np.isfinite(value)) for value in (half, base, double)):
        raise ValueError("Conditioning gradient samples must be finite")
    denominator = max(1.0, float(np.linalg.norm(base)))
    return ConditioningGradientConsistency(
        gradient_norm=float(np.linalg.norm(base)),
        relative_error_half_step=float(
            np.linalg.norm(base - half) / denominator
        ),
        relative_error_double_step=float(
            np.linalg.norm(base - double) / denominator
        ),
        cosine_half_step=_direction_cosine(base, half),
        cosine_double_step=_direction_cosine(base, double),
    )


def conditioning_metrics(
    jacobian: np.ndarray,
    joint_velocity_limits: np.ndarray,
    *,
    sigma_epsilon: float = 1.0e-12,
    task_velocity_scales: np.ndarray = TASK_VELOCITY_SCALES,
) -> ConditioningMetrics:
    """Return raw and velocity-normalized full-6D Jacobian diagnostics.

    ``jacobian`` must already be restricted to the robot arm velocity DoFs.
    This function is deliberately kinematics-free so every solver consumes
    the same metric without triggering another FK/Jacobian evaluation.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    joint_limits = np.asarray(joint_velocity_limits, dtype=float)
    task_scales = np.asarray(task_velocity_scales, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != 6:
        raise ValueError("Conditioning Jacobian must have shape (6, arm_dof)")
    if joint_limits.shape != (jacobian.shape[1],):
        raise ValueError(
            "Joint velocity limits must have shape (arm_dof,) matching the "
            "restricted Jacobian",
        )
    if task_scales.shape != (6,):
        raise ValueError("Task velocity scales must have shape (6,)")
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("Conditioning Jacobian must be finite")
    if not np.all(np.isfinite(joint_limits)) or np.any(joint_limits <= 0.0):
        raise ValueError("Joint velocity limits must be finite and positive")
    if not np.all(np.isfinite(task_scales)) or np.any(task_scales <= 0.0):
        raise ValueError("Task velocity scales must be finite and positive")
    if not np.isfinite(sigma_epsilon) or sigma_epsilon <= 0.0:
        raise ValueError("Sigma epsilon must be finite and positive")

    raw_singular_values = np.linalg.svd(jacobian, compute_uv=False)
    normalized = (
        jacobian
        * joint_limits[np.newaxis, :]
        / task_scales[:, np.newaxis]
    )
    normalized_singular_values = np.linalg.svd(
        normalized, compute_uv=False,
    )
    sigma_min_raw = float(raw_singular_values[-1])
    sigma_min_normalized = float(normalized_singular_values[-1])
    sigma_max_normalized = float(normalized_singular_values[0])
    sigma_min_gap_normalized = float(
        normalized_singular_values[-2] - normalized_singular_values[-1]
    )
    inverse_condition_index = float(
        sigma_min_normalized / max(sigma_max_normalized, sigma_epsilon)
    )
    return ConditioningMetrics(
        sigma_min_raw=sigma_min_raw,
        sigma_min_normalized=sigma_min_normalized,
        sigma_max_normalized=sigma_max_normalized,
        sigma_min_gap_normalized=sigma_min_gap_normalized,
        inverse_condition_index=inverse_condition_index,
    )


def conditioning_penalty(
    metrics: ConditioningMetrics,
    config: ConditioningSafeguardConfig,
) -> float:
    if not config.enabled or config.weight == 0.0:
        return 0.0
    eta = metrics.inverse_condition_index
    activation = conditioning_activation(metrics, config)
    recovery_residual = config.eta_high - eta
    return float(
        config.weight * activation * recovery_residual * recovery_residual
    )


def conditioning_activation(
    metrics: ConditioningMetrics,
    config: ConditioningSafeguardConfig,
) -> float:
    """Raised-cosine activation across the critical-to-healthy band."""
    if not config.enabled or config.weight == 0.0:
        return 0.0
    eta = metrics.inverse_condition_index
    if eta <= config.eta_low:
        return 1.0
    if eta >= config.eta_high:
        return 0.0
    phase = (
        (eta - config.eta_low)
        / (config.eta_high - config.eta_low)
    )
    return float(0.5 * (1.0 + np.cos(np.pi * phase)))


def conditioning_region(
    metrics: ConditioningMetrics,
    config: ConditioningSafeguardConfig,
) -> str:
    """Classify the current normalized conditioning state."""
    eta = metrics.inverse_condition_index
    if eta <= config.eta_low:
        return "critical"
    if eta < config.eta_high:
        return "transition"
    return "healthy"


def bound_aware_finite_difference(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    evaluate: Callable[[np.ndarray], float],
    *,
    relative_step: float,
) -> np.ndarray:
    """Differentiate a scalar without evaluating outside joint bounds.

    A central stencil is preferred. Near a bound the function switches to a
    forward or backward stencil; exceptionally narrow feasible intervals use
    the largest valid symmetric or one-sided step available.
    """
    q = np.asarray(q, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if q.shape != lower.shape or q.shape != upper.shape:
        raise ValueError("Finite-difference state and bounds must match")
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(lower)) or not np.all(
        np.isfinite(upper)
    ):
        raise ValueError("Finite-difference state and bounds must be finite")
    if np.any(lower > upper) or np.any(q < lower) or np.any(q > upper):
        raise ValueError("Finite-difference state must lie inside its bounds")
    if not np.isfinite(relative_step) or relative_step <= 0.0:
        raise ValueError("Finite-difference relative step must be positive")

    gradient = np.zeros_like(q)
    base_value: float | None = None
    machine_step = np.finfo(float).eps
    for index in range(q.size):
        requested = relative_step * max(1.0, abs(float(q[index])))
        lower_room = float(q[index] - lower[index])
        upper_room = float(upper[index] - q[index])

        if lower_room >= requested and upper_room >= requested:
            minus = q.copy()
            plus = q.copy()
            minus[index] -= requested
            plus[index] += requested
            gradient[index] = (
                float(evaluate(plus)) - float(evaluate(minus))
            ) / (2.0 * requested)
            continue

        symmetric_step = min(lower_room, upper_room, requested)
        if symmetric_step > machine_step:
            minus = q.copy()
            plus = q.copy()
            minus[index] -= symmetric_step
            plus[index] += symmetric_step
            gradient[index] = (
                float(evaluate(plus)) - float(evaluate(minus))
            ) / (2.0 * symmetric_step)
            continue

        if base_value is None:
            base_value = float(evaluate(q))
        forward_step = min(upper_room, requested)
        if forward_step > machine_step:
            plus = q.copy()
            plus[index] += forward_step
            gradient[index] = (
                float(evaluate(plus)) - base_value
            ) / forward_step
            continue
        backward_step = min(lower_room, requested)
        if backward_step > machine_step:
            minus = q.copy()
            minus[index] -= backward_step
            gradient[index] = (
                base_value - float(evaluate(minus))
            ) / backward_step
    return gradient

"""Gaussian, wall, and polynomial losses for task-space objectives.

Equations (6)--(11) of Wang et al., ICRA 2023.  Constant offsets are kept:
they make plots match the paper and do not affect optimisation gradients.
"""

from __future__ import annotations

import numpy as np

from .types import LossParameters


def _array(value: float | np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=float)


def gaussian(value: float | np.ndarray, goal: float | np.ndarray, c: float) -> np.ndarray:
    error = (_array(value) - _array(goal)) / c
    return -np.exp(-0.5 * error * error)


def polynomial(value: float | np.ndarray, goal: float | np.ndarray, a2: float, m: int) -> np.ndarray:
    return a2 * np.power(_array(value) - _array(goal), m)


def scaled(value: float | np.ndarray, lower: float | np.ndarray, upper: float | np.ndarray) -> np.ndarray:
    width = _array(upper) - _array(lower)
    if np.any(width <= 0.0):
        raise ValueError("A ranged task requires lower < upper.")
    return (2.0 * _array(value) - _array(lower) - _array(upper)) / width


def wall(value: float | np.ndarray, lower: float | np.ndarray, upper: float | np.ndarray, parameters: LossParameters) -> np.ndarray:
    x = scaled(value, lower, upper)
    exponent = -np.minimum(np.power(np.abs(x) / parameters.wall_scale, parameters.n), 700.0)
    return parameters.a1 * (1.0 - np.exp(exponent))


def groove(value: float | np.ndarray, goal: float | np.ndarray, parameters: LossParameters) -> np.ndarray:
    return gaussian(value, goal, parameters.c) + polynomial(value, goal, parameters.a2, parameters.m)


def groove_derivative(
    value: float | np.ndarray,
    goal: float | np.ndarray,
    parameters: LossParameters,
) -> np.ndarray:
    error = _array(value) - _array(goal)
    gaussian_part = (
        np.exp(-0.5 * np.square(error / parameters.c))
        * error
        / (parameters.c * parameters.c)
    )
    polynomial_part = (
        parameters.a2
        * parameters.m
        * np.power(error, parameters.m - 1)
    )
    return gaussian_part + polynomial_part


def _wall_unit_and_derivative(
    scaled_value: np.ndarray,
    parameters: LossParameters,
) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.abs(scaled_value)
    raw_exponent = np.power(
        absolute / parameters.wall_scale, parameters.n,
    )
    exponent = np.minimum(raw_exponent, 700.0)
    decay = np.exp(-exponent)
    derivative_exponent = (
        parameters.n
        * np.power(absolute, parameters.n - 1)
        * np.sign(scaled_value)
        / np.power(parameters.wall_scale, parameters.n)
    )
    derivative_exponent = np.where(
        raw_exponent < 700.0, derivative_exponent, 0.0,
    )
    return 1.0 - decay, decay * derivative_exponent


def swamp(value: float | np.ndarray, lower: float | np.ndarray, upper: float | np.ndarray, parameters: LossParameters) -> np.ndarray:
    x = scaled(value, lower, upper)
    exponent = -np.minimum(np.power(np.abs(x) / parameters.wall_scale, parameters.n), 700.0)
    return (parameters.a1 + parameters.a2 * np.power(x, parameters.m)) * (1.0 - np.exp(exponent)) - 1.0


def swamp_derivative(
    value: float | np.ndarray,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    parameters: LossParameters,
) -> np.ndarray:
    width = _array(upper) - _array(lower)
    x = scaled(value, lower, upper)
    wall_unit, wall_derivative = _wall_unit_and_derivative(x, parameters)
    polynomial = parameters.a1 + parameters.a2 * np.power(x, parameters.m)
    polynomial_derivative = (
        parameters.a2
        * parameters.m
        * np.power(x, parameters.m - 1)
    )
    return (
        polynomial_derivative * wall_unit
        + polynomial * wall_derivative
    ) * (2.0 / width)


def swamp_groove(value: float | np.ndarray, lower: float | np.ndarray, upper: float | np.ndarray, goal: float | np.ndarray, parameters: LossParameters) -> np.ndarray:
    # Reference implementation: f1 Gaussian + f2 polynomial + f3 wall.
    x = scaled(value, lower, upper)
    exponent = -np.minimum(np.power(np.abs(x) / parameters.wall_scale, parameters.n), 700.0)
    error = _array(value) - _array(goal)
    return -parameters.a1 * np.exp(-0.5 * np.square(error / parameters.c)) + parameters.a2 * np.power(error, parameters.m) + parameters.a3 * (1.0 - np.exp(exponent))


def swamp_groove_derivative(
    value: float | np.ndarray,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    goal: float | np.ndarray,
    parameters: LossParameters,
) -> np.ndarray:
    width = _array(upper) - _array(lower)
    x = scaled(value, lower, upper)
    _, wall_derivative = _wall_unit_and_derivative(x, parameters)
    error = _array(value) - _array(goal)
    return (
        parameters.a1
        * np.exp(-0.5 * np.square(error / parameters.c))
        * error
        / (parameters.c * parameters.c)
        + parameters.a2
        * parameters.m
        * np.power(error, parameters.m - 1)
        + parameters.a3 * wall_derivative * (2.0 / width)
    )

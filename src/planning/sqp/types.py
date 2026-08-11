from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SQPSettings:
    max_iterations: int = 20
    max_time_s: float = 0.095
    derivative_epsilon: float = 1e-7
    step_tolerance: float = 1e-7
    position_tolerance: float = 1e-5
    rotation_tolerance: float = 1e-4
    inequality_tolerance: float = 1e-7
    trust_region: float = 0.35
    merit_penalty: float = 1_000.0
    max_line_search_iterations: int = 12
    max_qp_iterations: int = 32
    expensive_gradient_refresh_rad: float = 0.015

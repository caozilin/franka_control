from __future__ import annotations

import argparse
import math


PID_DEFAULTS = {
    "pid_proportional_gain": 0.18,
    "pid_integral_gain_s": 0.30,
    "pid_velocity_gain_s": 0.04,
    "pid_maximum_correction_rad": math.radians(3.0),
    "pid_integration_error_limit_rad": math.radians(4.0),
    "pid_integral_time_constant_s": 1.0,
    "pid_stationary_integral_time_constant_s": 0.25,
    "pid_stationary_velocity_threshold_rad_s": 0.02,
}


def add_joint_pid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pid-proportional-gain", type=float, default=PID_DEFAULTS["pid_proportional_gain"])
    parser.add_argument("--pid-integral-gain-s", type=float, default=PID_DEFAULTS["pid_integral_gain_s"])
    parser.add_argument("--pid-velocity-gain-s", type=float, default=PID_DEFAULTS["pid_velocity_gain_s"])
    parser.add_argument(
        "--pid-maximum-correction-rad",
        type=float,
        default=PID_DEFAULTS["pid_maximum_correction_rad"],
    )
    parser.add_argument(
        "--pid-integration-error-limit-rad",
        type=float,
        default=PID_DEFAULTS["pid_integration_error_limit_rad"],
    )
    parser.add_argument(
        "--pid-integral-time-constant-s",
        type=float,
        default=PID_DEFAULTS["pid_integral_time_constant_s"],
    )
    parser.add_argument(
        "--pid-stationary-integral-time-constant-s",
        type=float,
        default=PID_DEFAULTS["pid_stationary_integral_time_constant_s"],
    )
    parser.add_argument(
        "--pid-stationary-velocity-threshold-rad-s",
        type=float,
        default=PID_DEFAULTS["pid_stationary_velocity_threshold_rad_s"],
    )


def joint_pid_kwargs(args: argparse.Namespace) -> dict[str, float]:
    return {name: float(getattr(args, name)) for name in PID_DEFAULTS}

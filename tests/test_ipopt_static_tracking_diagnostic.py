from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ipopt_static_tracking_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("ipopt_static_tracking_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnostic
SPEC.loader.exec_module(diagnostic)


def test_waypoints_remain_well_inside_hard_home_box() -> None:
    waypoints = diagnostic._waypoints(0.02)
    assert waypoints[0][0] == "home_initial"
    for _, waypoint in waypoints:
        assert np.max(np.abs(waypoint)) <= 0.02
        diagnostic._assert_translation_bound("test", waypoint, np.zeros(3))


def test_hard_home_box_rejects_any_axis_beyond_ten_centimeters() -> None:
    with pytest.raises(diagnostic.SafetyBoundaryError, match="left the home"):
        diagnostic._assert_translation_bound(
            "test",
            np.array((0.0, 0.100001, 0.0)),
            np.zeros(3),
        )


def test_fast_waypoints_use_eight_centimeter_margin_inside_hard_box() -> None:
    waypoints = diagnostic._fast_waypoints(0.08)
    assert [name for name, _ in waypoints] == [
        "fast_x_positive",
        "fast_x_sweep_negative",
        "fast_x_return_home",
        "fast_y_positive",
        "fast_y_sweep_negative",
        "fast_y_return_home",
        "fast_z_positive",
        "fast_z_sweep_negative",
        "fast_z_return_home",
    ]
    for _, waypoint in waypoints:
        assert np.max(np.abs(waypoint)) <= 0.08
        diagnostic._assert_translation_bound("test", waypoint, np.zeros(3))


def test_settled_trace_report_separates_reference_and_tracking_error() -> None:
    trace = np.zeros((3, 47), dtype=np.float64)
    trace[:, 0] = (0.0, 0.5, 1.0)
    trace[:, 1:4] = np.array((0.03, 0.0, 0.0))
    trace[:, 7:10] = np.array((0.02, 0.0, 0.0))
    trace[:, 13:16] = np.array((0.005, 0.0, 0.0))
    trace[:, 4:7] = np.array((0.0, 0.0, 0.03))
    trace[:, 10:13] = np.array((0.0, 0.0, 0.02))
    trace[:, 16:19] = np.array((0.0, 0.0, 0.005))
    trace[:, 19:26] = 2.0
    report = diagnostic._settled_trace_report(trace, 1.0)

    assert report["goal_minus_reference_m"]["final"] == pytest.approx((0.01, 0.0, 0.0))
    assert report["reference_minus_actual_m"]["final"] == pytest.approx((0.015, 0.0, 0.0))
    assert report["goal_minus_actual_m"]["final"] == pytest.approx((0.025, 0.0, 0.0))
    assert report["goal_minus_actual_rotvec_rad"]["final"] == pytest.approx((0.0, 0.0, 0.025))
    assert report["tau_command_abs_max_nm"] == pytest.approx(np.full(7, 2.0))


def test_motion_trace_report_separates_lag_cross_track_and_speed() -> None:
    trace = np.zeros((21, 47), dtype=np.float64)
    trace[:, 0] = np.arange(21, dtype=np.float64) * 0.001
    trace[:10, 1] = 0.01
    trace[10:, 1] = 0.02
    trace[:10, 7] = 0.005
    trace[10:, 7] = 0.015
    trace[:10, 13:16] = np.array((0.002, 0.001, 0.0))
    trace[10:, 13:16] = np.array((0.012, 0.001, 0.0))
    trace[:, 19:26] = 3.0

    report = diagnostic._motion_trace_report(
        trace,
        np.array((1.0, 0.0, 0.0)),
        np.zeros(3),
        np.zeros(3),
    )

    assert report["goal_minus_reference_m"]["final"] == pytest.approx((0.005, 0.0, 0.0))
    assert report["reference_minus_actual_m"]["final"] == pytest.approx((0.003, -0.001, 0.0))
    assert report["goal_minus_actual_along_m"]["final"] == pytest.approx(0.008)
    assert report["goal_minus_actual_cross_m"]["final"] == pytest.approx((0.0, -0.001, 0.0))
    assert report["reference_speed_m_s"]["mean"] == pytest.approx(0.5)
    assert report["actual_speed_m_s"]["mean"] == pytest.approx(0.5)
    assert report["tau_command_abs_max_nm"] == pytest.approx(np.full(7, 3.0))


def test_motion_trace_safety_checks_all_recorded_poses() -> None:
    trace = np.zeros((2, 47), dtype=np.float64)
    trace[1, 13] = 0.100001
    with pytest.raises(diagnostic.SafetyBoundaryError, match="measured trace"):
        diagnostic._assert_motion_trace_safe(trace, np.zeros(3), np.zeros(3))

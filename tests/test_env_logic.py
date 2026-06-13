from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis import analyze_trace_csv  # noqa: E402
from recording import TraceRecorder, create_run_paths  # noqa: E402
from utils.control import GRIPPER_WIDTH_MAX, transform_action  # noqa: E402
from utils.pose import matrix_to_pose_array, matrix_to_rotvec, rotvec_to_matrix  # noqa: E402


def _pose_array(x: float, y: float, z: float, rx: float, ry: float, rz: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotvec_to_matrix(np.array([rx, ry, rz], dtype=np.float64))
    matrix[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return matrix_to_pose_array(matrix)


def test_transform_action_matches_current_scaling() -> None:
    action = np.array([1.5, -2.0, 0.5, 8.0, -6.0, 3.0, -1.0], dtype=np.float64)
    transformed = transform_action(action)
    np.testing.assert_allclose(
        transformed[:6],
        np.array([0.1, -0.1, 0.05, math.pi / 4.0, -math.pi / 4.0, math.pi / 8.0], dtype=np.float64),
    )
    assert transformed[6] == GRIPPER_WIDTH_MAX


def test_teleop_rotation_conversion_matches_backend_left_multiply() -> None:
    current_rotation = rotvec_to_matrix(np.array([0.2, -0.35, 0.8], dtype=np.float64))
    rotvec_ee = np.array([0.04, -0.02, 0.06], dtype=np.float64)

    rotvec_base = matrix_to_rotvec(current_rotation @ rotvec_to_matrix(rotvec_ee) @ current_rotation.T)

    np.testing.assert_allclose(
        rotvec_to_matrix(rotvec_base) @ current_rotation,
        current_rotation @ rotvec_to_matrix(rotvec_ee),
        atol=1e-12,
    )


def test_trace_recorder_and_analysis_roundtrip(tmp_path: pathlib.Path) -> None:
    run_paths = create_run_paths(tmp_path, "min_jerk")
    recorder = TraceRecorder(reference_name="min_jerk")

    recorder.append_sample(
        {
            "time": 0.0,
            "reference": "min_jerk",
            "goal_pose": _pose_array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "ref_pose": _pose_array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            "actual_pose": _pose_array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    )
    recorder.append_sample(
        {
            "time": 0.001,
            "reference": "min_jerk",
            "goal_pose": _pose_array(0.01, 0.0, 0.0, 0.0, 0.0, 0.1),
            "ref_pose": _pose_array(0.01, 0.0, 0.0, 0.0, 0.0, 0.1),
            "actual_pose": _pose_array(0.009, 0.0, 0.0, 0.0, 0.0, 0.08),
        }
    )

    recorder.write_trace_csv(run_paths.trace_csv)
    summary = analyze_trace_csv(run_paths.trace_csv)

    assert summary["sample_count"] == 2
    assert summary["duration_sec"] == 0.001
    assert summary["axes"]["x"]["max_abs_ref_minus_actual"] > 0.0
    assert summary["axes"]["rz"]["rmse_ref_minus_actual"] > 0.0

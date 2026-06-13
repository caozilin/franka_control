from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils.pose import matrix_to_rotvec_continuous, pose_array_to_matrix


AXES = ("x", "y", "z", "rx", "ry", "rz")
SERIES = ("goal", "ref", "actual")


def _pose_to_six_axis(pose: np.ndarray, previous_rotvec: np.ndarray | None = None) -> np.ndarray:
    matrix = pose_array_to_matrix(pose)
    rotvec = matrix_to_rotvec_continuous(matrix[:3, :3], previous_rotvec)
    return np.array(
        [
            float(matrix[0, 3]),
            float(matrix[1, 3]),
            float(matrix[2, 3]),
            *rotvec.tolist(),
        ],
        dtype=np.float64,
    )


def _align_rotvec_to_reference(
    rotvec: np.ndarray,
    reference: np.ndarray,
    previous_rotvec: np.ndarray | None = None,
) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        if previous_rotvec is None:
            return rotvec.copy()
        previous_rotvec = np.asarray(previous_rotvec, dtype=np.float64)
        previous_norm = float(np.linalg.norm(previous_rotvec))
        if previous_norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        axis = previous_rotvec / previous_norm
        center = int(round(previous_norm / (2.0 * np.pi)))
        candidates = [2.0 * np.pi * k * axis for k in range(center - 3, center + 4)]
        candidates.append(np.zeros(3, dtype=np.float64))
    else:
        axis = rotvec / angle
        if abs(angle - np.pi) < 1e-4 and previous_rotvec is not None:
            previous_rotvec = np.asarray(previous_rotvec, dtype=np.float64)
            previous_norm = float(np.linalg.norm(previous_rotvec))
            if previous_norm >= 1e-12 and float(np.dot(axis, previous_rotvec / previous_norm)) < 0.0:
                axis = -axis
                rotvec = -rotvec

        if previous_rotvec is None:
            projected_previous = float(np.dot(reference, axis))
        else:
            projected_previous = float(np.dot(np.asarray(previous_rotvec, dtype=np.float64), axis))
        center = int(round((projected_previous - angle) / (2.0 * np.pi)))
        candidates = [rotvec + 2.0 * np.pi * k * axis for k in range(center - 4, center + 5)]

    def score(candidate: np.ndarray) -> float:
        continuity_penalty = 0.0
        if previous_rotvec is not None:
            continuity_penalty = float(np.linalg.norm(candidate - np.asarray(previous_rotvec, dtype=np.float64)))
        reference_penalty = float(np.linalg.norm(candidate - reference))
        return continuity_penalty + 0.7 * reference_penalty

    return min(candidates, key=score).copy()


@dataclass
class TraceRecorder:
    reference_name: str
    rows: list[dict[str, float]] = field(default_factory=list)
    _previous_rotvec_by_series: dict[str, np.ndarray] = field(default_factory=dict)

    def append_sample(self, sample: dict) -> None:
        row: dict[str, float] = {
            "time": float(sample["time"]),
            "reference": sample["reference"],
        }
        previous_rotvecs = {label: value.copy() for label, value in self._previous_rotvec_by_series.items()}
        values_by_series: dict[str, np.ndarray] = {}
        for series_name, pose_key in (("goal", "goal_pose"), ("ref", "ref_pose"), ("actual", "actual_pose")):
            values_by_series[series_name] = _pose_to_six_axis(sample[pose_key], previous_rotvecs.get(series_name))

        goal_rotvec = values_by_series["goal"][3:6].copy()
        values_by_series["ref"][3:6] = _align_rotvec_to_reference(
            values_by_series["ref"][3:6],
            goal_rotvec,
            previous_rotvecs.get("ref"),
        )
        values_by_series["actual"][3:6] = _align_rotvec_to_reference(
            values_by_series["actual"][3:6],
            goal_rotvec,
            previous_rotvecs.get("actual"),
        )

        for series_name in SERIES:
            self._previous_rotvec_by_series[series_name] = values_by_series[series_name][3:6].copy()
            for axis_name, value in zip(AXES, values_by_series[series_name]):
                row[f"{series_name}_{axis_name}"] = float(value)
        self.rows.append(row)

    def write_trace_csv(self, path: Path) -> None:
        fieldnames = ["time", "reference"]
        for series_name in SERIES:
            fieldnames.extend(f"{series_name}_{axis_name}" for axis_name in AXES)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_metadata_json(self, path: Path, metadata: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(metadata, output_file, indent=2, ensure_ascii=True)
            output_file.write("\n")

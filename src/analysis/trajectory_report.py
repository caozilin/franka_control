from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from recording.trace_recorder import AXES, SERIES
from utils.pose import matrix_to_rotvec, rotvec_to_matrix


SERIES_COLORS = {"goal": "#d62728", "ref": "#2ca02c", "actual": "#1f77b4"}


def _load_rows(trace_csv: Path) -> list[dict[str, str]]:
    with Path(trace_csv).open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        return list(reader)


def _raw_row_to_pose_values(raw_row: dict[str, str], series_name: str) -> np.ndarray:
    values = np.array([float(raw_row[f"{series_name}_{axis_name}"]) for axis_name in AXES], dtype=np.float64)
    rotation = rotvec_to_matrix(values[3:6])
    values[3:6] = matrix_to_rotvec(rotation)
    return values


def _enumerate_rotvec_candidates(rotvec: np.ndarray, previous_rotvec: np.ndarray | None = None) -> list[np.ndarray]:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        if previous_rotvec is None:
            return [rotvec.copy()]
        previous_rotvec = np.asarray(previous_rotvec, dtype=np.float64)
        previous_norm = float(np.linalg.norm(previous_rotvec))
        if previous_norm < 1e-12:
            return [np.zeros(3, dtype=np.float64)]
        axis = previous_rotvec / previous_norm
        center = int(round(previous_norm / (2.0 * np.pi)))
        candidates = [2.0 * np.pi * k * axis for k in range(center - 2, center + 3)]
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
            projected_previous = 0.0
        else:
            projected_previous = float(np.dot(np.asarray(previous_rotvec, dtype=np.float64), axis))
        center = int(round((projected_previous - angle) / (2.0 * np.pi)))
        candidates = [rotvec + 2.0 * np.pi * k * axis for k in range(center - 2, center + 3)]

    unique_candidates: list[np.ndarray] = []
    for candidate in candidates:
        if any(np.allclose(candidate, existing, atol=1e-9, rtol=0.0) for existing in unique_candidates):
            continue
        unique_candidates.append(np.asarray(candidate, dtype=np.float64).copy())
    return unique_candidates


def _jointly_align_rotvecs(
    values_by_series: dict[str, np.ndarray],
    previous_rotvecs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    candidate_sets = {
        series_name: _enumerate_rotvec_candidates(values_by_series[series_name][3:6], previous_rotvecs.get(series_name))
        for series_name in SERIES
    }

    best_cost = float("inf")
    best_rotvecs: dict[str, np.ndarray] | None = None
    for goal_candidate in candidate_sets["goal"]:
        for ref_candidate in candidate_sets["ref"]:
            for actual_candidate in candidate_sets["actual"]:
                continuity_cost = 0.0
                for series_name, candidate in (("goal", goal_candidate), ("ref", ref_candidate), ("actual", actual_candidate)):
                    previous = previous_rotvecs.get(series_name)
                    if previous is not None:
                        continuity_cost += float(np.sum(np.square(candidate - previous)))

                consistency_cost = (
                    float(np.sum(np.square(ref_candidate - goal_candidate)))
                    + float(np.sum(np.square(actual_candidate - ref_candidate)))
                    + 0.5 * float(np.sum(np.square(actual_candidate - goal_candidate)))
                )
                norm_cost = 1e-6 * (
                    float(np.sum(np.square(goal_candidate)))
                    + float(np.sum(np.square(ref_candidate)))
                    + float(np.sum(np.square(actual_candidate)))
                )
                cost = continuity_cost + 0.7 * consistency_cost + norm_cost
                if cost < best_cost:
                    best_cost = cost
                    best_rotvecs = {
                        "goal": goal_candidate.copy(),
                        "ref": ref_candidate.copy(),
                        "actual": actual_candidate.copy(),
                    }

    assert best_rotvecs is not None
    aligned_values = {series_name: values.copy() for series_name, values in values_by_series.items()}
    for series_name in SERIES:
        aligned_values[series_name][3:6] = best_rotvecs[series_name]
    return aligned_values


def _reconstruct_continuous_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    previous_rotvecs: dict[str, np.ndarray] = {}
    reconstructed_rows: list[dict[str, float | str]] = []
    for raw_row in raw_rows:
        values_by_series = {
            series_name: _raw_row_to_pose_values(raw_row, series_name)
            for series_name in SERIES
        }
        values_by_series = _jointly_align_rotvecs(values_by_series, previous_rotvecs)

        reconstructed_row: dict[str, float | str] = {
            "time": float(raw_row["time"]),
            "controller": raw_row["controller"],
        }
        for series_name in SERIES:
            previous_rotvecs[series_name] = values_by_series[series_name][3:6].copy()
            for axis_name, value in zip(AXES, values_by_series[series_name]):
                reconstructed_row[f"{series_name}_{axis_name}"] = float(value)
        reconstructed_rows.append(reconstructed_row)
    return reconstructed_rows


def analyze_trace_csv(trace_csv: Path) -> dict:
    rows = _reconstruct_continuous_rows(_load_rows(trace_csv))
    if not rows:
        return {"sample_count": 0, "duration_sec": 0.0, "axes": {}}

    time_values = np.array([float(row["time"]) for row in rows], dtype=np.float64)
    summary = {
        "sample_count": len(rows),
        "duration_sec": float(time_values[-1] - time_values[0]),
        "axes": {},
    }
    for axis_name in AXES:
        ref_values = np.array([float(row[f"ref_{axis_name}"]) for row in rows], dtype=np.float64)
        actual_values = np.array([float(row[f"actual_{axis_name}"]) for row in rows], dtype=np.float64)
        error = ref_values - actual_values
        max_index = int(np.argmax(np.abs(error)))
        summary["axes"][axis_name] = {
            "final_ref_minus_actual": float(error[-1]),
            "rmse_ref_minus_actual": float(np.sqrt(np.mean(np.square(error)))),
            "max_abs_ref_minus_actual": float(np.max(np.abs(error))),
            "max_abs_ref_minus_actual_time": float(time_values[max_index]),
        }
    return summary


def write_summary_json(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, ensure_ascii=True)
        output_file.write("\n")


def write_plot_svg(trace_csv: Path, svg_path: Path, *, max_points: int = 2000) -> None:
    rows = _reconstruct_continuous_rows(_load_rows(trace_csv))
    # Downsample for SVG: keep at most max_points evenly spaced rows
    if len(rows) > max_points:
        step = max(1, len(rows) // max_points)
        rows = rows[::step]
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        empty_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="80"><text x="20" y="45">no samples</text></svg>\n'
        svg_path.write_text(empty_svg, encoding="utf-8")
        return

    time_values = np.array([float(row["time"]) for row in rows], dtype=np.float64)
    width = 1200
    panel_height = 150
    panel_gap = 34
    margin_left = 90
    margin_right = 35
    margin_top = 58
    legend_height = 50
    plot_width = width - margin_left - margin_right
    height = margin_top + len(AXES) * panel_height + (len(AXES) - 1) * panel_gap + legend_height
    t_min = float(time_values[0])
    t_max = float(time_values[-1])
    if t_max <= t_min:
        t_max = t_min + 1e-6

    def scale_x(value: float) -> float:
        return margin_left + (float(value) - t_min) / (t_max - t_min) * plot_width

    tick_start = int(math.floor(t_min))
    tick_end = int(math.ceil(t_max))
    second_ticks = list(range(tick_start, tick_end + 1))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-size="20" font-family="sans-serif">goal / ref / actual pose tracks</text>',
    ]

    for axis_index, axis_name in enumerate(AXES):
        panel_top = margin_top + axis_index * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        data_by_series = {
            series_name: np.array([float(row[f"{series_name}_{axis_name}"]) for row in rows], dtype=np.float64)
            for series_name in SERIES
        }
        all_values = np.concatenate(list(data_by_series.values()))
        y_min = float(np.min(all_values))
        y_max = float(np.max(all_values))
        if y_max <= y_min:
            y_max = y_min + 1e-6
        padding = max((y_max - y_min) * 0.08, 1e-5)
        y_min -= padding
        y_max += padding

        def scale_y(value: float) -> float:
            return panel_bottom - (float(value) - y_min) / (y_max - y_min) * panel_height

        svg.extend([
            f'<rect x="{margin_left}" y="{panel_top}" width="{plot_width}" height="{panel_height}" fill="#fafafa" stroke="#cccccc"/>',
            f'<text x="18" y="{panel_top + panel_height / 2:.1f}" font-size="14" font-family="sans-serif">{axis_name}</text>',
            f'<text x="{margin_left - 8}" y="{panel_top + 10}" text-anchor="end" font-size="11" font-family="monospace">{y_max:.5f}</text>',
            f'<text x="{margin_left - 8}" y="{panel_bottom}" text-anchor="end" font-size="11" font-family="monospace">{y_min:.5f}</text>',
        ])
        for tick in second_ticks:
            if t_min - 1e-9 <= tick <= t_max + 1e-9:
                tick_x = scale_x(tick)
                svg.append(f'<line x1="{tick_x:.2f}" y1="{panel_top}" x2="{tick_x:.2f}" y2="{panel_bottom}" stroke="#dddddd" stroke-width="1"/>')
                if axis_index == len(AXES) - 1:
                    svg.append(f'<text x="{tick_x:.2f}" y="{panel_bottom + 18}" text-anchor="middle" font-size="11" font-family="monospace">{tick:d}s</text>')

        error_series = data_by_series["ref"] - data_by_series["actual"]
        max_error_index = int(np.argmax(np.abs(error_series)))
        max_error_time = float(time_values[max_error_index])
        max_error_value = float(error_series[max_error_index])
        max_error_x = scale_x(max_error_time)
        max_error_y = scale_y(float(data_by_series["actual"][max_error_index]))
        label_y = max(panel_top + 16, max_error_y - 10)
        label_x = min(margin_left + plot_width - 10, max_error_x + 8)
        svg.append(
            f'<text x="{margin_left + plot_width - 8}" y="{panel_top + 18}" text-anchor="end" '
            f'font-size="12" font-family="monospace" fill="#111111">final ref-actual={error_series[-1]:+.6f}</text>'
        )
        svg.append(
            f'<line x1="{max_error_x:.2f}" y1="{panel_top}" x2="{max_error_x:.2f}" y2="{panel_bottom}" '
            f'stroke="#f0c0c0" stroke-dasharray="3,3" stroke-width="1"/>'
        )
        svg.append(f'<circle cx="{max_error_x:.2f}" cy="{max_error_y:.2f}" r="4" fill="#d62728" stroke="white" stroke-width="1.5"/>')
        svg.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="start" '
            f'font-size="11" font-family="monospace" fill="#d62728">max |ref-actual|={max_error_value:+.6f}</text>'
        )

        for series_name in SERIES:
            points = " ".join(
                f"{scale_x(t):.2f},{scale_y(v):.2f}"
                for t, v in zip(time_values, data_by_series[series_name])
            )
            svg.append(f'<polyline fill="none" stroke="{SERIES_COLORS[series_name]}" stroke-width="1.5" points="{points}"/>')

    legend_y = height - 25
    legend_x = margin_left
    for series_name in SERIES:
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{SERIES_COLORS[series_name]}" stroke-width="2"/>')
        svg.append(f'<text x="{legend_x + 36}" y="{legend_y + 4}" font-size="13" font-family="sans-serif">{series_name}</text>')
        legend_x += 100
    svg.append(f'<text x="{width / 2:.1f}" y="{height - 5}" text-anchor="middle" font-size="13" font-family="sans-serif">time [s]</text>')
    svg.append("</svg>")
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")

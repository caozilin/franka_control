from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from recording.trace_recorder import AXES, SERIES


SERIES_COLORS = {"goal": "#d62728", "ref": "#2ca02c", "actual": "#1f77b4"}


def _load_rows(trace_csv: Path) -> list[dict[str, str]]:
    with Path(trace_csv).open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        return list(reader)


def _select_evenly_spaced_rows(raw_rows: list[dict[str, str]], max_points: int) -> list[dict[str, str]]:
    if max_points <= 0 or len(raw_rows) <= max_points:
        return raw_rows
    indices = np.linspace(0, len(raw_rows) - 1, num=max_points, dtype=np.int64)
    return [raw_rows[int(index)] for index in np.unique(indices)]


def _parse_continuous_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for raw_row in raw_rows:
        row: dict[str, float | str] = {
            "time": float(raw_row["time"]),
            "reference": raw_row.get("reference", ""),
        }
        for series_name in SERIES:
            for axis_name in AXES:
                row[f"{series_name}_{axis_name}"] = float(raw_row[f"{series_name}_{axis_name}"])
        rows.append(row)
    return rows


def analyze_trace_csv(trace_csv: Path) -> dict:
    rows = _parse_continuous_rows(_load_rows(trace_csv))
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
    raw_rows = _select_evenly_spaced_rows(_load_rows(trace_csv), max_points)
    rows = _parse_continuous_rows(raw_rows)
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

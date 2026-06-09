from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np


PARENT_TIMING_FIELDS = {"loop_total", "policy_total", "controller_step", "controller_reference"}

TIMING_FIELDS = (
    "loop_total",
    "read_once",
    "policy_total",
    "action_get",
    "transform_action",
    "update_pose_goal",
    "state_lock",
    "controller_update_goal",
    "gripper",
    "print_events",
    "controller_step",
    "controller_reference",
    "controller_slerp",
    "controller_model_coriolis",
    "controller_model_jacobian",
    "controller_velocity_math",
    "controller_pose_error",
    "controller_wrench_torque",
    "controller_torque_limit",
    "raw_trace_write",
    "trace_callback",
    "torques_build",
    "write_once",
)


class RealtimeTimingProfiler:
    def __init__(self, capacity: int, *, fields: tuple[str, ...] = TIMING_FIELDS):
        self.capacity = int(capacity)
        self.fields = tuple(fields)
        self.field_to_index = {name: i for i, name in enumerate(self.fields)}
        self.elapsed = np.zeros(self.capacity, dtype=np.float64)
        self.robot_dt = np.zeros(self.capacity, dtype=np.float64)
        self.data = np.zeros((self.capacity, len(self.fields)), dtype=np.float64)
        self.count = 0
        self.dropped = 0

    def now(self) -> float:
        return time.perf_counter()

    def add_elapsed(self, field: str, start: float) -> None:
        idx = self.count
        if idx >= self.capacity:
            return
        field_idx = self.field_to_index.get(field)
        if field_idx is None:
            return
        self.data[idx, field_idx] += time.perf_counter() - float(start)

    def add_duration(self, field: str, duration: float) -> None:
        idx = self.count
        if idx >= self.capacity:
            return
        field_idx = self.field_to_index.get(field)
        if field_idx is None:
            return
        self.data[idx, field_idx] += float(duration)

    def finish_frame(self, elapsed: float, robot_dt: float, loop_start: float) -> None:
        idx = self.count
        if idx >= self.capacity:
            self.dropped += 1
            return
        self.elapsed[idx] = float(elapsed)
        self.robot_dt[idx] = float(robot_dt)
        self.add_duration("loop_total", time.perf_counter() - float(loop_start))
        self.count += 1

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["frame", "elapsed", "robot_dt", *self.fields])
            for i in range(self.count):
                writer.writerow([i, self.elapsed[i], self.robot_dt[i], *self.data[i].tolist()])

    def summary_rows(self) -> list[dict[str, float | str]]:
        if self.count <= 0:
            return []
        rows = []
        values = self.data[: self.count]
        for i, field in enumerate(self.fields):
            column = values[:, i]
            rows.append(
                {
                    "field": field,
                    "total_ms": float(np.sum(column) * 1000.0),
                    "avg_us": float(np.mean(column) * 1_000_000.0),
                    "p99_us": float(np.percentile(column, 99.0) * 1_000_000.0),
                    "max_us": float(np.max(column) * 1_000_000.0),
                }
            )
        return rows

    def _slow_frame_modules(self, frame: int, *, limit: int = 6, min_us: float = 1.0) -> list[tuple[str, float]]:
        modules: list[tuple[str, float]] = []
        for field, field_idx in self.field_to_index.items():
            if field in PARENT_TIMING_FIELDS:
                continue
            value_us = float(self.data[frame, field_idx] * 1_000_000.0)
            if value_us >= float(min_us):
                modules.append((field, value_us))
        modules.sort(key=lambda item: item[1], reverse=True)
        return modules[: int(limit)]

    def format_frame_causes(self, frame: int, *, limit: int = 6, min_us: float = 1.0) -> str:
        modules = self._slow_frame_modules(int(frame), limit=limit, min_us=min_us)
        return ", ".join(f"{name}={value_us:.1f}us" for name, value_us in modules)

    def print_summary(self, *, top_frames: int = 10) -> None:
        if self.count <= 0:
            print("Realtime timing: no frames recorded")
            return
        print("\nRealtime timing summary")
        print(f"  frames: {self.count}, dropped: {self.dropped}")
        print("  module                         total_ms     avg_us     p99_us     max_us")
        for row in self.summary_rows():
            if row["total_ms"] <= 0.0 and row["max_us"] <= 0.0:
                continue
            print(
                f"  {row['field']:<28} "
                f"{row['total_ms']:>9.3f} "
                f"{row['avg_us']:>10.3f} "
                f"{row['p99_us']:>10.3f} "
                f"{row['max_us']:>10.3f}"
            )

        loop_idx = self.field_to_index["loop_total"]
        slow_indices = np.argsort(self.data[: self.count, loop_idx])[-int(top_frames) :][::-1]
        print("  slowest frames by loop_total:")
        for frame in slow_indices:
            module_text = self.format_frame_causes(int(frame))
            if module_text:
                module_text = f" causes: {module_text}"
            print(
                f"    frame={int(frame):05d} "
                f"elapsed={self.elapsed[frame]:.6f}s "
                f"robot_dt={self.robot_dt[frame] * 1000.0:.3f}ms "
                f"loop_total={self.data[frame, loop_idx] * 1000.0:.3f}ms"
                f"{module_text}"
            )

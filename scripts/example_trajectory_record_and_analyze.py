#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analysis import analyze_trace_csv, write_plot_svg, write_summary_json  # noqa: E402
from control.franka_env import FrankaEnv  # noqa: E402
from recording import TraceRecorder, create_run_paths  # noqa: E402
from utils import POLICY_HZ  # noqa: E402



ACTION_DIM = 7
BLOCK_SIZE = 60
REFERENCE_CHOICES = ("min_jerk", "linear", "cubic", "motion_limited")


def parse_joint_vector(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("--nullspace-q-target must contain 7 comma-separated joint values")
    return np.asarray(parts, dtype=np.float64)


def timed_step(label, func):
    start = time.time()
    result = func()
    print(f"  [save] {label} in {time.time() - start:.3f}s")
    return result


class TimedActionBlockSource:
    def __init__(self, *, max_translation_step: float, max_rotation_step: float, scale: float):
        self._actions = self._build_actions(max_translation_step, max_rotation_step, scale)
        self._cursor = 0

    def _build_actions(
        self,
        max_translation_step: float,
        max_rotation_step: float,
        scale: float,
    ) -> list[np.ndarray]:
        block6 = np.array(
            [
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.018, 0.008, 0.005, 0.035, -0.030, 0.040],
                [0.010, -0.012, 0.000, -0.025, 0.000, -0.030],
                [0.010, -0.000, 0.006, -0.000, 0.020, -0.030],
                [0.000, -0.012, 0.006, -0.000, 0.000, -0.000],
                [0.000, -0.000, 0.000, -0.025, 0.020, -0.030],
                [0.010, -0.000, 0.006, -0.000, 0.020, -0.000],
                [0.000, -0.012, 0.000, -0.025, 0.000, -0.000],
                [0.000, -0.000, 0.006, -0.000, 0.000, -0.030],
                [0.010, -0.012, 0.000, -0.000, 0.020, -0.000],
                [0.010, -0.000, 0.000, -0.025, 0.000, -0.030],
                [0.000, -0.012, 0.006, -0.000, 0.020, -0.000],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [-0.012, -0.014, -0.008, -0.030, 0.030, -0.040],
                [0.012, 0.018, 0.010, 0.030, -0.020, 0.035],
                [-0.012, -0.018, 0.000, -0.030, 0.020, -0.035],
                [0.010, 0.015, -0.010, 0.025, 0.000, 0.030],
                [-0.010, -0.015, 0.000, -0.025, 0.000, -0.030],
                [0.015, -0.020, 0.012, 0.020, 0.025, -0.025],
                [-0.015, 0.020, 0.000, -0.020, -0.025, 0.025],
                [0.008, 0.014, -0.012, 0.015, -0.015, 0.020],
                [-0.008, -0.014, 0.000, -0.015, 0.015, -0.020],
                [0.012, 0.018, 0.010, 0.010, 0.020, 0.015],
                [-0.012, -0.018, -0.010, -0.010, -0.020, -0.015],
                [0.008, 0.000, -0.006, 0.000, -0.018, 0.000],
                [0.000, 0.010, -0.000, 0.010, -0.000, 0.000],
                [0.008, 0.000, -0.000, 0.000, -0.018, 0.000],
                [0.000, 0.010, -0.006, 0.000, -0.000, 0.000],
                [0.008, 0.000, -0.000, 0.010, -0.000, 0.000],
                [0.000, 0.010, -0.000, 0.000, -0.018, 0.000],
                [0.008, 0.000, -0.006, 0.000, -0.000, 0.000],
                [0.000, 0.010, -0.000, 0.010, -0.018, 0.000],
                [0.008, 0.000, -0.000, 0.000, -0.000, 0.000],
                [0.000, 0.010, -0.006, 0.000, -0.000, 0.000],
            ],
            dtype=np.float64,
        )
        block6 *= float(scale)

        actions: list[np.ndarray] = []
        rotation_scale = float(max_rotation_step) / 6.0
        for delta in block6:
            action = np.zeros(ACTION_DIM, dtype=np.float64)
            action[:3] = delta[:3] / float(max_translation_step)
            action[3:6] = delta[3:6] / rotation_scale
            action[6] = -1.0
            actions.append(action)

        for _ in range(int(POLICY_HZ)):
            action = np.zeros(ACTION_DIM, dtype=np.float64)
            action[6] = -1.0
            actions.append(action)
        return actions

    @property
    def done(self) -> bool:
        return self._cursor >= len(self._actions)

    @property
    def duration(self) -> float:
        return len(self._actions) / POLICY_HZ

    def next_block(self) -> np.ndarray:
        if self.done:
            return np.zeros((BLOCK_SIZE, ACTION_DIM), dtype=np.float64)
        remaining = self._actions[self._cursor : self._cursor + BLOCK_SIZE]
        self._cursor += 1
        if len(remaining) < BLOCK_SIZE:
            padding = [np.zeros(ACTION_DIM, dtype=np.float64) for _ in range(BLOCK_SIZE - len(remaining))]
            remaining = remaining + padding
        return np.vstack(remaining)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2", help="Robot IP address")
    parser.add_argument("--reference", choices=REFERENCE_CHOICES, default="min_jerk")
    parser.add_argument("--log-dir", type=pathlib.Path, default=ROOT / "logs" / "runs")
    parser.add_argument("--max-translation-step", type=float, default=0.1)
    parser.add_argument("--max-rotation-step", type=float, default=math.pi / 4.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--reset-duration", type=float, default=5.0)
    parser.add_argument("--settle", type=float, default=4.0)
    parser.add_argument("--no-home-first", action="store_true")
    parser.add_argument("--no-robot", action="store_true", help="Run the trajectory/logging path without connecting to the robot")
    parser.add_argument("--yes", action="store_true", help="Skip interactive safety prompts")
    parser.add_argument("--nullspace-enabled", action="store_true")
    parser.add_argument("--nullspace-pinv", choices=("plain", "damped"), default="plain")
    parser.add_argument("--nullspace-projector", choices=("kinematic", "dynamic"), default="kinematic")
    parser.add_argument("--nullspace-lambda", type=float, default=0.05)
    parser.add_argument("--nullspace-stiffness", type=float, default=10.0)
    parser.add_argument("--nullspace-damping", type=float, default=2.0)
    parser.add_argument("--nullspace-q-target", type=parse_joint_vector, default=None)
    args = parser.parse_args()

    print("WARNING: This script will run a scaled 10Hz action sequence on the robot.")
    print("First 5s: mixed motion. Last 1s: zero motion. C++ may continue settling after the last action.")
    print(f"Reference: {args.reference}, scale: {args.scale}")
    if not args.yes:
        input("Press Enter to continue...")

    run_paths = create_run_paths(args.log_dir, args.reference)
    recorder = TraceRecorder(reference_name=args.reference)
    source = TimedActionBlockSource(
        max_translation_step=args.max_translation_step,
        max_rotation_step=args.max_rotation_step,
        scale=args.scale,
    )
    env = FrankaEnv(
        robot_ip=args.ip,
        reset_duration=args.reset_duration,
        max_translation_step=args.max_translation_step,
        max_rotation_step=args.max_rotation_step,
        reference_name=args.reference,
        no_robot=args.no_robot,
        auto_record=False,
        nullspace_enabled=args.nullspace_enabled,
        nullspace_q_target=args.nullspace_q_target,
        nullspace_stiffness=args.nullspace_stiffness,
        nullspace_damping=args.nullspace_damping,
        nullspace_pinv=args.nullspace_pinv,
        nullspace_projector=args.nullspace_projector,
        nullspace_lambda=args.nullspace_lambda,
    )
    print(f"Run directory: {run_paths.run_dir}")

    try:
        if not args.no_home_first:
            print("Resetting robot to home pose...")
            env.reset()
            if not args.yes:
                input("Reset complete. Press Enter to start torque control...")

        tick = 0
        first_block = source.next_block()
        env.enqueue_action_block(first_block)
        tick += 1
        print(f"10Hz tick {tick:03d}  (first block enqueued)")

        env.start_control_loop(max_duration=source.duration + float(args.settle))
        next_time = time.monotonic()
        while not source.done:
            next_time += 1.0 / POLICY_HZ
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            block = source.next_block()
            env.enqueue_action_block(block)
            tick += 1
            first_action = block[0]
            action_mm = first_action[:3] * float(args.max_translation_step) * 1000.0
            print(f"10Hz tick {tick:03d}  dxyz_mm=[{action_mm[0]:+.1f},{action_mm[1]:+.1f},{action_mm[2]:+.1f}]  "
                  f"drot=[{first_action[3]:+.3f},{first_action[4]:+.3f},{first_action[5]:+.3f}]")

        print("Waiting for C++ backend to finish settling...")
        env.wait_control_loop()
        print("Saving 1kHz trace from C++ ring buffer...")
        timed_step("trace rows", lambda: env.save_trace_to_recorder(recorder, reference_name=args.reference))
        timed_step("timing csv", lambda: env.save_timing_csv(run_paths.timing_csv))
    finally:
        env.stop()

    timed_step("trace csv", lambda: recorder.write_trace_csv(run_paths.trace_csv))
    timed_step(
        "metadata json",
        lambda: recorder.write_metadata_json(
            run_paths.metadata_json,
            {
                "reference": args.reference,
                "max_translation_step": float(args.max_translation_step),
                "max_rotation_step": float(args.max_rotation_step),
                "scale": float(args.scale),
                "reset_duration": float(args.reset_duration),
                "sample_count": len(recorder.rows),
                "trajectory_files": {
                    "pose_tracks_csv": run_paths.trace_csv.name,
                    "timing_csv": run_paths.timing_csv.name,
                    "summary_json": run_paths.summary_json.name,
                    "plot_svg": run_paths.plot_svg.name,
                },
            },
        ),
    )
    summary = timed_step("analyze trace", lambda: analyze_trace_csv(run_paths.trace_csv))
    timed_step("summary json", lambda: write_summary_json(run_paths.summary_json, summary))
    timed_step("plot svg", lambda: write_plot_svg(run_paths.trace_csv, run_paths.plot_svg))

    print(f"Saved pose tracks: {run_paths.trace_csv}")
    print(f"Saved timing: {run_paths.timing_csv}")
    print(f"Saved metadata: {run_paths.metadata_json}")
    print(f"Saved summary: {run_paths.summary_json}")
    print(f"Saved plot: {run_paths.plot_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

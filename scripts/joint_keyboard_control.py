#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import threading
import time

import numpy as np
from pynput import keyboard

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.franka_env import FrankaEnv  # noqa: E402
from control.pid_config import add_joint_pid_arguments, joint_pid_kwargs  # noqa: E402
from planning import JOINT_REFERENCE_CHOICES, TRACKER_MODE_CHOICES  # noqa: E402

INPUT_DT = 0.1
STEP_DEG = 3.0
STEP_RAD = math.radians(STEP_DEG)
DEFAULT_HOME_Q = np.array([0.0, -math.pi / 4.0, 0.0, -3.0 * math.pi / 4.0, 0.0, math.pi / 2.0, math.pi / 4.0], dtype=np.float64)
JOINT_LIMIT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
JOINT_LIMIT_HIGH = np.array([+2.8973, +1.7628, +2.8973, -0.0698, +2.8973, +3.7525, +2.8973], dtype=np.float64)

KEY_BINDINGS = {
    "q": (0, +1), "a": (0, -1),
    "w": (1, +1), "s": (1, -1),
    "e": (2, +1), "d": (2, -1),
    "r": (3, +1), "f": (3, -1),
    "t": (4, +1), "g": (4, -1),
    "y": (5, +1), "h": (5, -1),
    "u": (6, +1), "j": (6, -1),
}


def format_degrees(q: np.ndarray) -> str:
    q_deg = np.degrees(q)
    return "[" + ", ".join(f"{value:+7.2f}" for value in q_deg) + "] deg"


class JointKeyboardController:
    def __init__(
        self,
        *,
        robot_ip: str,
        home_first: bool,
        dry_run: bool,
        reference_name: str,
        tracker_mode: str,
        pid_kwargs: dict[str, float],
    ):
        self.env = FrankaEnv(
            robot_ip=robot_ip,
            control_mode="joint",
            reference_name=reference_name,
            tracker_mode=tracker_mode,
            **pid_kwargs,
            print_events=False,
            use_gripper=False,
            auto_record=False,
            no_robot=dry_run,
        )
        self.home_first = bool(home_first)
        self.dry_run = bool(dry_run)
        self.running = True
        self.keys_pressed: set[str] = set()
        self._keys_lock = threading.Lock()
        self.target_q = DEFAULT_HOME_Q.copy()
        self.gripper_close = False
        self._input_thread: threading.Thread | None = None

    def _on_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            self.env.request_stop()
            return False
        try:
            char = key.char.lower()
        except AttributeError:
            return None
        if char in KEY_BINDINGS:
            with self._keys_lock:
                self.keys_pressed.add(char)
        elif char == "z":
            self.gripper_close = False
        elif char == "x":
            self.gripper_close = True
        return None

    def _on_release(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            return None
        with self._keys_lock:
            self.keys_pressed.discard(char)
        return None

    def _build_joint_delta(self) -> np.ndarray:
        action = np.zeros(7, dtype=np.float64)
        with self._keys_lock:
            pressed = set(self.keys_pressed)
        for char in sorted(pressed):
            joint_index, direction = KEY_BINDINGS[char]
            action[joint_index] += direction * STEP_RAD
        return action

    def _input_loop(self) -> None:
        next_time = time.monotonic()
        tick = 0
        while self.running:
            joint_delta = self._build_joint_delta()
            previous_target = self.target_q.copy()
            self.target_q = np.clip(previous_target + joint_delta, JOINT_LIMIT_LOW, JOINT_LIMIT_HIGH)
            limited_delta = self.target_q - previous_target
            action = np.zeros(8, dtype=np.float64)
            action[:7] = limited_delta
            action[7] = 1.0 if self.gripper_close else -1.0
            if not self.dry_run:
                self.env.enqueue_action_block(action)
            tick += 1
            mode = "dry" if self.dry_run else "exec"
            print(f"10Hz joint tick {tick:04d}  mode={mode}  q_target={format_degrees(self.target_q)}", flush=True)
            next_time += INPUT_DT
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_time = time.monotonic()

    def run(self) -> None:
        if self.home_first and not self.dry_run:
            print("Resetting to home pose...", flush=True)
            self.env.reset()
        if self.dry_run:
            self.target_q = DEFAULT_HOME_Q.copy()
        else:
            self.target_q = np.clip(self.env.get_joint_positions(), JOINT_LIMIT_LOW, JOINT_LIMIT_HIGH)

        print(f"Keyboard joint env-control at 10Hz, step={STEP_DEG:.0f} deg.", flush=True)
        print("Keys: q/a J1 +/-  w/s J2 +/-  e/d J3 +/-  r/f J4 +/-  t/g J5 +/-  y/h J6 +/-  u/j J7 +/-  z open  x close  ESC exit", flush=True)
        print(f"Initial q_target={format_degrees(self.target_q)}", flush=True)

        listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        listener.start()
        try:
            if not self.dry_run:
                self.env.start_control_loop(print_events=False)
            self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
            self._input_thread.start()
            if not self.dry_run:
                self.env.wait_control_loop()
            else:
                while self.running:
                    time.sleep(0.1)
        finally:
            self.running = False
            listener.stop()
            if self._input_thread is not None:
                self._input_thread.join(timeout=1.0)
            if not self.dry_run:
                self.env.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="10Hz keyboard joint actions through FrankaEnv joint min-jerk mode.")
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--home-first", action="store_true", help="Move to DEFAULT_HOME_Q before accepting keyboard input")
    parser.add_argument("--dry-run", action="store_true", help="Only print q_target; do not command the robot")
    parser.add_argument("--reference", choices=JOINT_REFERENCE_CHOICES, default="min_jerk")
    parser.add_argument("--tracker-mode", choices=TRACKER_MODE_CHOICES, default="auto")
    add_joint_pid_arguments(parser)
    args = parser.parse_args()

    controller = JointKeyboardController(
        robot_ip=args.ip,
        home_first=args.home_first,
        dry_run=args.dry_run,
        reference_name=args.reference,
        tracker_mode=args.tracker_mode,
        pid_kwargs=joint_pid_kwargs(args),
    )
    controller.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

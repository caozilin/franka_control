#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

import numpy as np
from pylibfranka import CartesianPose, ControllerMode, Robot

from franka_control.math_utils import matrix_to_rotvec, rotvec_to_matrix

CONTROL_DT = 0.1
MAX_LIN_VEL = 0.1
MAX_ROT_VEL = math.pi / 4.0
MAX_DELTA_POS = MAX_LIN_VEL * CONTROL_DT
MAX_DELTA_ROT = MAX_ROT_VEL * CONTROL_DT
GRIPPER_SPEED = 0.08
GRIPPER_FORCE = 60.0
GRIPPER_OPEN_WIDTH = 0.08
SPEED_LEVELS = [0.4, 0.7, 1.0]
KEY_HOLD_TIMEOUT = 0.25
MOTION_KEYS = ["w", "s", "a", "d", "i", "k", "q", "e", "u", "o", "j", "l"]
KEY_TO_INDEX = {key: idx for idx, key in enumerate(MOTION_KEYS)}

COLLISION_TORQUE_LOWER = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
COLLISION_TORQUE_UPPER = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
COLLISION_FORCE_LOWER = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
COLLISION_FORCE_UPPER = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]


@dataclass
class KeyboardSharedState:
    timestamps: object
    speed_index: object
    gripper_target: object
    running: object


def _log_motion_keys(timestamps, last_snapshot: tuple[str, ...]) -> tuple[str, ...]:
    now = time.monotonic()
    snapshot = tuple(
        key for key, idx in KEY_TO_INDEX.items()
        if now - float(timestamps[idx]) <= KEY_HOLD_TIMEOUT
    )
    if snapshot != last_snapshot:
        print("motion keys:", "".join(snapshot), flush=True)
    return snapshot


def keyboard_process_main(shared: KeyboardSharedState) -> None:
    tty_file = open("/dev/tty", "r", buffering=1)
    fd = tty_file.fileno()
    old_settings = termios.tcgetattr(fd)
    last_snapshot: tuple[str, ...] = ()
    try:
        tty.setcbreak(fd)
        print("keyboard process ready", flush=True)
        while bool(shared.running.value):
            ready, _, _ = select.select([tty_file], [], [], 0.05)
            if not ready:
                last_snapshot = _log_motion_keys(shared.timestamps, last_snapshot)
                continue

            char = tty_file.read(1).lower()
            now = time.monotonic()
            if char == "\x1b":
                shared.running.value = False
                break
            if char in {"+", "="}:
                shared.speed_index.value = min(len(SPEED_LEVELS) - 1, int(shared.speed_index.value) + 1)
                print(f"speed: {SPEED_LEVELS[int(shared.speed_index.value)]:.1f}", flush=True)
                continue
            if char in {"-", "_"}:
                shared.speed_index.value = max(0, int(shared.speed_index.value) - 1)
                print(f"speed: {SPEED_LEVELS[int(shared.speed_index.value)]:.1f}", flush=True)
                continue
            if char == "g":
                shared.gripper_target.value = 0.0
                print("key: g -> close gripper", flush=True)
                continue
            if char == "h":
                shared.gripper_target.value = GRIPPER_OPEN_WIDTH
                print("key: h -> open gripper", flush=True)
                continue
            idx = KEY_TO_INDEX.get(char)
            if idx is not None:
                shared.timestamps[idx] = now
                last_snapshot = _log_motion_keys(shared.timestamps, last_snapshot)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        tty_file.close()


def read_keyboard_snapshot(shared: KeyboardSharedState) -> tuple[set[str], float, float, bool]:
    now = time.monotonic()
    keys = {
        key for key, idx in KEY_TO_INDEX.items()
        if now - float(shared.timestamps[idx]) <= KEY_HOLD_TIMEOUT
    }
    speed = SPEED_LEVELS[int(shared.speed_index.value)]
    gripper_target = float(shared.gripper_target.value)
    running = bool(shared.running.value)
    return keys, speed, gripper_target, running


def o_t_ee_to_matrix(o_t_ee: np.ndarray) -> np.ndarray:
    return np.asarray(o_t_ee, dtype=np.float64).reshape(4, 4, order="F")


def matrix_to_o_t_ee(matrix: np.ndarray) -> list[float]:
    return np.asarray(matrix, dtype=np.float64).reshape(16, order="F").tolist()


def apply_base_frame_delta_matrix(matrix: np.ndarray, delta: np.ndarray) -> np.ndarray:
    next_matrix = np.asarray(matrix, dtype=np.float64).copy()
    next_matrix[:3, 3] += delta[:3]
    delta_rot = np.asarray(delta[3:6], dtype=np.float64)
    if np.linalg.norm(delta_rot) > 1e-12:
        next_matrix[:3, :3] = rotvec_to_matrix(delta_rot) @ next_matrix[:3, :3]
    return next_matrix


def min_jerk(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def interpolate_pose_matrix(start_matrix: np.ndarray, target_matrix: np.ndarray, alpha: float) -> np.ndarray:
    # Straight Cartesian path with a min-jerk time law: zero velocity and acceleration at endpoints.
    alpha = min_jerk(alpha)
    matrix = np.asarray(start_matrix, dtype=np.float64).copy()
    matrix[:3, 3] = (1.0 - alpha) * start_matrix[:3, 3] + alpha * target_matrix[:3, 3]

    start_rot = start_matrix[:3, :3]
    target_rot = target_matrix[:3, :3]
    relative_rotvec = matrix_to_rotvec(target_rot @ start_rot.T)
    matrix[:3, :3] = rotvec_to_matrix(relative_rotvec * alpha) @ start_rot
    return matrix


def convert_local_rot_delta_to_base(local_rot_delta: np.ndarray, commanded_matrix: np.ndarray) -> np.ndarray:
    if np.linalg.norm(local_rot_delta) < 1e-12:
        return local_rot_delta.copy()
    current_rot = commanded_matrix[:3, :3]
    delta_local = rotvec_to_matrix(local_rot_delta)
    delta_base = current_rot @ delta_local @ current_rot.T
    return matrix_to_rotvec(delta_base)


def build_action(keys: set[str], speed: float, gripper_target: float, commanded_matrix: np.ndarray) -> np.ndarray:
    dx = (int("s" in keys) - int("w" in keys)) * MAX_DELTA_POS * speed
    dy = (int("d" in keys) - int("a" in keys)) * MAX_DELTA_POS * speed
    dz = (int("i" in keys) - int("k" in keys)) * MAX_DELTA_POS * speed
    # Keep the first hardware test translational only. Rotation can be re-enabled once the
    # position teleop is stable.
    delta_rot_base = np.zeros(3, dtype=np.float64)
    action = np.array([dx, dy, dz, *delta_rot_base.tolist(), 0.0], dtype=np.float64)
    action[6] = 1.0 if gripper_target <= 0.0 else -1.0
    return action

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="172.16.0.2")
    args = parser.parse_args()

    ctx = mp.get_context("fork")
    shared = KeyboardSharedState(
        timestamps=ctx.Array("d", [0.0] * len(MOTION_KEYS), lock=False),
        speed_index=ctx.Value("i", len(SPEED_LEVELS) - 1),
        gripper_target=ctx.Value("d", GRIPPER_OPEN_WIDTH),
        running=ctx.Value("b", True),
    )
    keyboard_process = ctx.Process(target=keyboard_process_main, args=(shared,), daemon=True)
    keyboard_process.start()

    robot = None
    try:
        robot = Robot(args.ip)

        robot.set_collision_behavior(
            COLLISION_TORQUE_LOWER,
            COLLISION_TORQUE_UPPER,
            COLLISION_FORCE_LOWER,
            COLLISION_FORCE_UPPER,
        )

        print("WARNING: This script will move the robot.")
        print("Keep the user stop button available.")
        print("Terminal must stay focused while controlling.")
        print("W/S:X  A/D:Y  I/K:Z  Q/E:Roll  U/O:Pitch  J/L:Yaw")
        print("G/H: gripper  +/-: speed  ESC: exit")

        time.sleep(0.5)
        active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)
        robot_state, _duration = active_control.readOnce()
        initial_o_t_ee = np.asarray(robot_state.O_T_EE, dtype=np.float64).copy()
        active_control.writeOnce(CartesianPose(initial_o_t_ee.tolist()))

        last_log = time.monotonic()
        interp_time = 0.0
        current_command_matrix = o_t_ee_to_matrix(initial_o_t_ee)
        interp_start_matrix = current_command_matrix.copy()
        interp_target_matrix = current_command_matrix.copy()
        pending_action = np.zeros(7, dtype=np.float64)

        while True:
            robot_state, duration = active_control.readOnce()
            now = time.monotonic()
            dt = duration.to_sec()

            keys, speed, gripper_target, running = read_keyboard_snapshot(shared)
            if not running:
                break

            interp_time += dt
            if interp_time >= CONTROL_DT:
                # Finish the previous min-jerk segment before replanning. Starting a new segment
                # from an in-flight point resets velocity and can trigger libfranka discontinuity checks.
                current_command_matrix = interp_target_matrix.copy()
                pending_action = build_action(keys, speed, gripper_target, interp_target_matrix)
                interp_start_matrix = interp_target_matrix.copy()
                interp_target_matrix = apply_base_frame_delta_matrix(interp_target_matrix, pending_action[:6])
                interp_time = 0.0

            alpha = interp_time / CONTROL_DT
            current_command_matrix = interpolate_pose_matrix(interp_start_matrix, interp_target_matrix, alpha)

            cartesian_pose = CartesianPose(matrix_to_o_t_ee(current_command_matrix))
            active_control.writeOnce(cartesian_pose)

            if now - last_log >= 1.0:
                actual_pos = [robot_state.O_T_EE[12], robot_state.O_T_EE[13], robot_state.O_T_EE[14]]
                print("actual_pos:", np.round(actual_pos, 4).tolist(), "action:", np.round(pending_action, 4).tolist())
                last_log = now
    except KeyboardInterrupt:
        shared.running.value = False
    finally:
        shared.running.value = False
        if keyboard_process is not None:
            keyboard_process.join(timeout=1.0)
            if keyboard_process.is_alive():
                keyboard_process.terminate()
                keyboard_process.join(timeout=1.0)
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

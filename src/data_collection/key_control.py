#!/usr/bin/env python3
"""
Franka Panda 键盘/手柄遥操作控制器
==================================
只做遥操作，不记录数据：
  - 10Hz 读取输入状态并生成 7 维 action
  - action 语义沿用 vla4desk: [dx, dy, dz, d_rx, d_ry, d_rz, gripper]
  - 底层通过 FrankaEnv 的 1kHz 笛卡尔扭矩控制执行
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
import pathlib
import sys
import threading
import time

import numpy as np
from pynput import keyboard

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from control.franka_env import (
    DEFAULT_MAX_ROTATION_GOAL_ERROR,
    DEFAULT_MAX_ROTATION_VELOCITY,
    DEFAULT_MAX_TRANSLATION_GOAL_ERROR,
    DEFAULT_MAX_TRANSLATION_VELOCITY,
    DEFAULT_MAX_TORQUE_RATE,
    FrankaEnv,
)  # noqa: E402
from utils import POLICY_HZ  # noqa: E402
from utils.control import transform_action  # noqa: E402
from utils.pose import matrix_to_rotvec, rotvec_to_matrix  # noqa: E402


INPUT_DT = 0.1
TELEOP_MAX_LIN_VEL = DEFAULT_MAX_TRANSLATION_VELOCITY
TELEOP_MAX_ROT_VEL = DEFAULT_MAX_ROTATION_VELOCITY
MAX_DELTA_POS = TELEOP_MAX_LIN_VEL * INPUT_DT
MAX_DELTA_ROT = TELEOP_MAX_ROT_VEL * INPUT_DT
BLOCK_SIZE = 60
JOYSTICK_DEADZONE = 0.3

PS4_AXIS_LEFT_X = 0
PS4_AXIS_LEFT_Y = 1
PS4_AXIS_L2 = 2
PS4_AXIS_RIGHT_X = 3
PS4_AXIS_RIGHT_Y = 4
PS4_AXIS_R2 = 5

PS4_BUTTON_CROSS = 0
PS4_BUTTON_CIRCLE = 1
PS4_BUTTON_TRIANGLE = 2
PS4_BUTTON_SQUARE = 3
PS4_BUTTON_L1 = 4
PS4_BUTTON_R1 = 5
PS4_BUTTON_OPTIONS = 9
PS4_BUTTON_L3 = 11
PS4_BUTTON_R3 = 12


def end_effector_rotation_to_backend_rotation(current_rotation: np.ndarray, rotvec_ee: np.ndarray) -> np.ndarray:
    current_rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
    rotvec_ee = np.asarray(rotvec_ee, dtype=np.float64)
    if float(np.linalg.norm(rotvec_ee)) < 1e-12:
        return np.zeros(3, dtype=np.float64)
    delta_rotation_ee = rotvec_to_matrix(rotvec_ee)
    delta_rotation_base = current_rotation @ delta_rotation_ee @ current_rotation.T
    return matrix_to_rotvec(delta_rotation_base)


class KeyboardController:
    def __init__(
        self,
        *,
        robot_ip: str = "172.16.0.2",
        input_device: str = "keyboard",
        joystick_index: int = 0,
        max_translation_velocity: float = DEFAULT_MAX_TRANSLATION_VELOCITY,
        max_rotation_velocity: float = DEFAULT_MAX_ROTATION_VELOCITY,
        max_translation_goal_error: float = DEFAULT_MAX_TRANSLATION_GOAL_ERROR,
        max_rotation_goal_error: float = DEFAULT_MAX_ROTATION_GOAL_ERROR,
        motion_limited_max_translation_velocity: float | None = None,
        motion_limited_max_rotation_velocity: float | None = None,
        motion_limited_max_translation_acceleration: float | None = None,
        motion_limited_max_rotation_acceleration: float | None = None,
        reset_duration: float = 5.0,
        reference_name: str = "linear",
        save_recording: bool = False,
        nullspace_enabled: bool = False,
        nullspace_q_target: np.ndarray | None = None,
        nullspace_stiffness: float = 10.0,
        nullspace_damping: float = 2.0,
        nullspace_pinv: str = "plain",
        nullspace_projector: str = "kinematic",
        nullspace_lambda: float = 0.05,
        task_constraint_mask: np.ndarray | None = None,
        no_robot: bool = False,
        no_cameras: bool = True,
        debug_ee_axes: bool = False,
    ):
        if input_device not in ("keyboard", "ps4"):
            raise ValueError(f"不支持的输入设备: {input_device}")

        self.input_device = input_device
        self.joystick_index = int(joystick_index)
        self.robot_ip = robot_ip
        self.env = FrankaEnv(
            robot_ip=robot_ip,
            reset_duration=reset_duration,
            max_translation_velocity=max_translation_velocity,
            max_rotation_velocity=max_rotation_velocity,
            max_translation_goal_error=max_translation_goal_error,
            max_rotation_goal_error=max_rotation_goal_error,
            motion_limited_max_translation_velocity=motion_limited_max_translation_velocity,
            motion_limited_max_rotation_velocity=motion_limited_max_rotation_velocity,
            motion_limited_max_translation_acceleration=motion_limited_max_translation_acceleration,
            motion_limited_max_rotation_acceleration=motion_limited_max_rotation_acceleration,
            reference_name=reference_name,
            save_recording=save_recording,
            log_subdir="teleop",
            nullspace_enabled=nullspace_enabled,
            nullspace_q_target=nullspace_q_target,
            nullspace_stiffness=nullspace_stiffness,
            nullspace_damping=nullspace_damping,
            nullspace_pinv=nullspace_pinv,
            nullspace_projector=nullspace_projector,
            nullspace_lambda=nullspace_lambda,
            task_constraint_mask=task_constraint_mask,
            no_robot=no_robot,
            no_cameras=no_cameras,
        )
        self.max_translation_velocity = float(max_translation_velocity)
        self.max_rotation_velocity = float(max_rotation_velocity)
        self.max_translation_step = self.max_translation_velocity / POLICY_HZ
        self.max_rotation_step = self.max_rotation_velocity / POLICY_HZ
        self.reference_name = str(reference_name)
        self.debug_ee_axes = bool(debug_ee_axes)

        self.gripper_target = 0.08
        self.step_size = 1.0
        self._speed_levels = [0.4, 0.7, 1.0]
        self._speed_index = 2
        self.running = True

        self.keys_pressed: set[str] = set()
        self._keys_lock = threading.Lock()
        self._motion_keys = {
            "w", "s", "a", "d",
            "i", "k",
            "q", "e",
            "u", "o",
            "j", "l",
        }

        self._reset_requested = False
        self._restart_requested = False
        self._restart_delay_sec = 2.0
        self._restart_ready_event: threading.Event | None = None
        self._reset_in_progress = threading.Event()
        self._event_callbacks: dict[str, callable] = {}
        self._action_lock = threading.Lock()
        self._latest_action = np.zeros(7, dtype=np.float64)
        self._latest_recording_action = np.zeros(7, dtype=np.float64)
        self._latest_state = np.zeros(8, dtype=np.float64)
        self._latest_joint_state = np.zeros(7, dtype=np.float64)
        self._latest_commanded_pose = np.zeros(6, dtype=np.float64)
        self._trace_lock = threading.Lock()
        self._recording_trace: deque[dict[str, np.ndarray | int | float]] = deque(maxlen=6000)
        self._recording_seq_id = 0
        self._input_thread: threading.Thread | None = None
        self._input_generation = 0
        self._input_last_tick = 0.0
        self._pygame_ready = False
        self._joystick = None
        self._prev_buttons: dict[int, bool] = {}
        self._prev_hat = (0, 0)
        self._rumble_token = 0
        if self.input_device == "ps4":
            self._init_ps4()

    def start(self, *, home_first: bool = True):
        self._print_controls()
        if home_first:
            self._reset_env()
        self._stop_rumble()
        self._sync_ps4_state()
        self.rumble(0.18, 0.7, 120)

    def _reset_env(self):
        print("  [复位] 回到初始位姿...")
        self.gripper_target = 0.08
        self._stop_rumble()
        try:
            self.env.reset()
        except Exception:
            print("  [复位] 失败")
            raise
        self._sync_ps4_state()
        print("  [复位] 完成")

    def stop(self):
        self.running = False
        self.env.request_stop()
        self.env.stop()
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
        self._shutdown_ps4()

    def bind_event(self, event_name: str, callback):
        """绑定离散事件回调，例如 record_start / record_stop。"""
        self._event_callbacks[event_name] = callback

    def _emit_event(self, event_name: str):
        callback = self._event_callbacks.get(event_name)
        if callback is not None:
            callback()

    def _print_controls(self):
        if self.input_device == "keyboard":
            print("  [控制] 键盘模式")
            print("  [控制] W/S:X  A/D:Y  I/K:Z  Q/E:Roll  U/O:Pitch  J/L:Yaw")
            print("  [控制] G/H:夹爪  +/-:速度档位  R:复位  1/2/3:开始/结束并恢复控制/作废录制  ESC:退出")
            return

        print("  [控制] PS4 手柄模式")
        print("  [控制] 左摇杆:X/Y平移  右摇杆:Z平移/Yaw  L1/R1:Roll  L2/R2:Pitch")
        print("  [控制] 三角/圆圈:打开/关闭夹爪  叉:作废并复位  方块:复位  十字键左右:速度档位")
        print("  [控制] L3/R3:开始/结束录制并恢复控制  OPTIONS:退出")

    def _on_key_press(self, key):
        if self.input_device != "keyboard":
            return
        try:
            char = key.char.lower()
        except AttributeError:
            char = key.name.lower()

        with self._keys_lock:
            if char == "escape":
                self.stop()
                return False
            if char == "r":
                self._request_reset()
                return
            if char == "1":
                self._emit_event("record_start")
                return
            if char == "2":
                self._emit_event("record_stop")
                return
            if char == "3":
                self._emit_event("record_discard")
                self.reset_to_home(open_gripper=True)
                return
            if char in ("+", "="):
                self._speed_index = min(len(self._speed_levels) - 1, self._speed_index + 1)
                self.step_size = self._speed_levels[self._speed_index]
                print(f"  [速度] 倍率: {self.step_size * 100:.0f}%")
                return
            if char in ("-", "_"):
                self._speed_index = max(0, self._speed_index - 1)
                self.step_size = self._speed_levels[self._speed_index]
                print(f"  [速度] 倍率: {self.step_size * 100:.0f}%")
                return
            if char == "g":
                self._close_gripper()
                return
            if char == "h":
                self._open_gripper()
                return
            self.keys_pressed.add(char)

    def _on_key_release(self, key):
        if self.input_device != "keyboard":
            return
        try:
            char = key.char.lower()
        except AttributeError:
            char = key.name.lower()
        with self._keys_lock:
            self.keys_pressed.discard(char)

    def _init_ps4(self):
        global pygame
        if pygame is None:
            try:
                import pygame as _pygame
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "PS4 手柄模式需要安装 pygame，请先执行 `pip install pygame`。"
                ) from exc
            pygame = _pygame

        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count <= self.joystick_index:
            raise RuntimeError(
                f"未找到 PS4 手柄。当前检测到 {count} 个手柄，"
                f"但请求使用索引 {self.joystick_index}。"
            )

        self._joystick = pygame.joystick.Joystick(self.joystick_index)
        self._joystick.init()
        self._pygame_ready = True
        self._prev_hat = (0, 0)
        self._prev_buttons = {
            idx: bool(self._joystick.get_button(idx))
            for idx in range(self._joystick.get_numbuttons())
        }
        print(f"  [手柄] 已连接: {self._joystick.get_name()} (index={self.joystick_index})")

    def _shutdown_ps4(self):
        if not self._pygame_ready or pygame is None:
            return

        self._stop_rumble()
        if self._joystick is not None and self._joystick.get_init():
            self._joystick.quit()
        self._joystick = None
        pygame.joystick.quit()
        pygame.quit()
        self._pygame_ready = False

    def _stop_rumble(self):
        self._rumble_token += 1
        if self.input_device != "ps4" or not self._pygame_ready or self._joystick is None:
            return
        try:
            if hasattr(self._joystick, "rumble"):
                self._joystick.rumble(0.0, 0.0, 0)
                self._joystick.rumble(0.0, 0.0, 1)
        except Exception:
            pass

    def _sync_ps4_state(self):
        if self.input_device != "ps4" or not self._pygame_ready or self._joystick is None or pygame is None:
            return
        try:
            pygame.event.pump()
            self._prev_hat = self._joystick.get_hat(0) if self._joystick.get_numhats() > 0 else (0, 0)
            self._prev_buttons = {
                idx: bool(self._joystick.get_button(idx))
                for idx in range(self._joystick.get_numbuttons())
            }
        except pygame.error:
            pass

    def rumble(self, low: float = 0.0, high: float = 0.0, duration_ms: int = 150):
        if self.input_device != "ps4" or not self._pygame_ready or self._joystick is None:
            return
        try:
            if hasattr(self._joystick, "rumble"):
                self._stop_rumble()
                self._joystick.rumble(low, high, duration_ms)
                self._rumble_token += 1
                token = self._rumble_token

                def _stop_later():
                    time.sleep(max(duration_ms, 1) / 1000.0)
                    if (
                        token == self._rumble_token
                        and self.input_device == "ps4"
                        and self._pygame_ready
                        and self._joystick is not None
                    ):
                        try:
                            self._joystick.rumble(0.0, 0.0, 1)
                        except Exception:
                            pass

                threading.Thread(target=_stop_later, daemon=True).start()
        except Exception:
            pass

    def _apply_deadzone(self, value: float) -> float:
        value = float(value)
        mag = abs(value)
        if mag < JOYSTICK_DEADZONE:
            return 0.0
        scaled = (mag - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
        return float(np.sign(value) * np.clip(scaled, 0.0, 1.0))

    def _read_axis(self, axis_index: int) -> float:
        if not self._pygame_ready or self._joystick is None:
            return 0.0
        try:
            if axis_index >= self._joystick.get_numaxes():
                return 0.0
            return self._apply_deadzone(self._joystick.get_axis(axis_index))
        except pygame.error:
            return 0.0

    def _read_trigger(self, axis_index: int, fallback_button: int | None = None) -> float:
        if not self._pygame_ready or self._joystick is None:
            return 0.0
        try:
            if axis_index < self._joystick.get_numaxes():
                raw = self._joystick.get_axis(axis_index)
                normalized = (raw + 1.0) / 2.0
                if normalized < JOYSTICK_DEADZONE:
                    return 0.0
                scaled = (normalized - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
                return float(np.clip(scaled, 0.0, 1.0))

            if fallback_button is not None and fallback_button < self._joystick.get_numbuttons():
                return float(self._joystick.get_button(fallback_button))
        except pygame.error:
            return 0.0
        return 0.0

    def _get_button(self, button_idx: int) -> float:
        if not self._pygame_ready or self._joystick is None:
            return 0.0
        try:
            if button_idx >= self._joystick.get_numbuttons():
                return 0.0
            return float(self._joystick.get_button(button_idx))
        except pygame.error:
            return 0.0

    def _request_reset(self):
        if self._reset_in_progress.is_set():
            print("  [复位] 已在复位中")
            return
        print("  [复位] 停止当前控制后复位")
        self._restart_requested = False
        self._reset_requested = True
        self._reset_in_progress.set()
        self.env.request_stop()

    def request_control_restart(
        self,
        delay_sec: float | None = None,
        *,
        ready_event: threading.Event | None = None,
    ):
        if self._reset_in_progress.is_set():
            print("  [控制] 已在状态切换中")
            return
        self._restart_delay_sec = float(delay_sec if delay_sec is not None else 2.0)
        self._restart_ready_event = ready_event
        print(f"  [控制] 已请求停止当前控制，满足恢复条件后约 {self._restart_delay_sec:.1f}s 恢复")
        self._reset_requested = False
        self._restart_requested = True
        self._reset_in_progress.set()
        self.env.request_stop()

    def _handle_ps4_buttons(self) -> bool:
        if not self._pygame_ready or self._joystick is None:
            return False

        try:
            for button_idx in range(self._joystick.get_numbuttons()):
                pressed = bool(self._joystick.get_button(button_idx))
                prev_pressed = self._prev_buttons.get(button_idx, False)
                self._prev_buttons[button_idx] = pressed
                if not pressed or prev_pressed:
                    continue

                if button_idx == PS4_BUTTON_OPTIONS:
                    self.rumble(0.9, 0.9, 260)
                    self.running = False
                    self.env.request_stop()
                    return False
                if button_idx == PS4_BUTTON_CROSS:
                    self._emit_event("record_discard")
                    self.reset_to_home(open_gripper=True)
                    return self.running
                if button_idx == PS4_BUTTON_SQUARE:
                    self._request_reset()
                    return self.running
                if button_idx == PS4_BUTTON_L3:
                    self._emit_event("record_start")
                elif button_idx == PS4_BUTTON_R3:
                    self._emit_event("record_stop")
                elif button_idx == PS4_BUTTON_TRIANGLE:
                    self._open_gripper()
                elif button_idx == PS4_BUTTON_CIRCLE:
                    self._close_gripper()

            hat = self._joystick.get_hat(0) if self._joystick.get_numhats() > 0 else (0, 0)
            prev_hat_x, _ = self._prev_hat
            hat_x, _ = hat
            if hat_x == 1 and prev_hat_x != 1:
                self._speed_index = min(len(self._speed_levels) - 1, self._speed_index + 1)
                self.step_size = self._speed_levels[self._speed_index]
                print(f"  [速度] 倍率: {self.step_size * 100:.0f}%")
            elif hat_x == -1 and prev_hat_x != -1:
                self._speed_index = max(0, self._speed_index - 1)
                self.step_size = self._speed_levels[self._speed_index]
                print(f"  [速度] 倍率: {self.step_size * 100:.0f}%")
            self._prev_hat = hat
            return self.running
        except pygame.error:
            return False

    def _get_keyboard_delta(self) -> tuple[float, float, float, float, float, float]:
        with self._keys_lock:
            keys = set(self.keys_pressed)

        speed = self.step_size
        dx = (int("s" in keys) - int("w" in keys)) * MAX_DELTA_POS * speed
        dy = (int("d" in keys) - int("a" in keys)) * MAX_DELTA_POS * speed
        dz = (int("i" in keys) - int("k" in keys)) * MAX_DELTA_POS * speed
        droll = (int("q" in keys) - int("e" in keys)) * MAX_DELTA_ROT * speed
        dpitch = (int("o" in keys) - int("u" in keys)) * MAX_DELTA_ROT * speed
        dyaw = (int("j" in keys) - int("l" in keys)) * MAX_DELTA_ROT * speed
        return dx, dy, dz, droll, dpitch, dyaw

    def _get_ps4_delta(self) -> tuple[float, float, float, float, float, float]:
        if not self._pygame_ready or self._joystick is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        try:
            pygame.event.pump()
        except pygame.error:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if not self._handle_ps4_buttons():
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        speed = self.step_size
        left_x = self._read_axis(PS4_AXIS_LEFT_X)
        left_y = self._read_axis(PS4_AXIS_LEFT_Y)
        right_x = self._read_axis(PS4_AXIS_RIGHT_X)
        right_y = self._read_axis(PS4_AXIS_RIGHT_Y)
        l2 = self._read_trigger(PS4_AXIS_L2, fallback_button=6)
        r2 = self._read_trigger(PS4_AXIS_R2, fallback_button=7)
        l1 = self._get_button(PS4_BUTTON_L1)
        r1 = self._get_button(PS4_BUTTON_R1)

        dx = left_y * MAX_DELTA_POS * speed
        dy = left_x * MAX_DELTA_POS * speed
        dz = -right_y * MAX_DELTA_POS * speed
        droll = (l1 - r1) * MAX_DELTA_ROT * speed
        dpitch = (r2 - l2) * MAX_DELTA_ROT * speed
        dyaw = -right_x * MAX_DELTA_ROT * speed
        return dx, dy, dz, droll, dpitch, dyaw

    def _get_input_delta(self) -> tuple[float, float, float, float, float, float]:
        if self.input_device == "ps4":
            return self._get_ps4_delta()
        return self._get_keyboard_delta()

    def _current_rotation_matrix(self, commanded_pose: np.ndarray | None = None) -> np.ndarray:
        try:
            if commanded_pose is not None and np.asarray(commanded_pose).shape[0] >= 6:
                return rotvec_to_matrix(np.asarray(commanded_pose, dtype=np.float64)[3:6])
            state = self.env.get_robot_state_vector()
            return rotvec_to_matrix(state[3:6])
        except Exception:
            return np.eye(3, dtype=np.float64)

    def _end_effector_rotation_to_backend_rotation(
        self,
        rotvec_ee: np.ndarray,
        commanded_pose: np.ndarray | None = None,
    ) -> np.ndarray:
        return end_effector_rotation_to_backend_rotation(
            self._current_rotation_matrix(commanded_pose),
            rotvec_ee,
        )

    def _build_action(
        self,
        commanded_pose: np.ndarray | None = None,
        input_delta: tuple[float, float, float, float, float, float] | None = None,
    ) -> np.ndarray:
        dx, dy, dz, droll, dpitch, dyaw = input_delta if input_delta is not None else self._get_input_delta()
        drx, dry, drz = self._end_effector_rotation_to_backend_rotation(
            np.array([droll, dpitch, dyaw], dtype=np.float64),
            commanded_pose,
        )
        return np.array(
            [
                dx,
                dy,
                dz,
                drx,
                dry,
                drz,
                1.0 if self.gripper_target <= 0.0 else -1.0,
            ],
            dtype=np.float64,
        )

    def _print_ee_debug(
        self,
        *,
        state: np.ndarray,
        commanded_pose: np.ndarray,
        input_delta: tuple[float, float, float, float, float, float],
        action: np.ndarray,
    ) -> None:
        if not self.debug_ee_axes:
            return
        rotation = self._current_rotation_matrix(commanded_pose)
        pos_ee = np.asarray(input_delta[:3], dtype=np.float64)
        rot_ee = np.asarray(input_delta[3:6], dtype=np.float64)
        rot_base = np.asarray(action[3:6], dtype=np.float64)
        x_axis = rotation[:, 0]
        y_axis = rotation[:, 1]
        z_axis = rotation[:, 2]
        print(
            "  [EE_DEBUG] "
            f"state_rpy/rv=({state[3]:+.3f},{state[4]:+.3f},{state[5]:+.3f}) "
            f"ee_x=({x_axis[0]:+.3f},{x_axis[1]:+.3f},{x_axis[2]:+.3f}) "
            f"ee_y=({y_axis[0]:+.3f},{y_axis[1]:+.3f},{y_axis[2]:+.3f}) "
            f"ee_z=({z_axis[0]:+.3f},{z_axis[1]:+.3f},{z_axis[2]:+.3f}) "
            f"dpos_ee=({pos_ee[0]:+.4f},{pos_ee[1]:+.4f},{pos_ee[2]:+.4f}) "
            f"drot_ee=({rot_ee[0]:+.4f},{rot_ee[1]:+.4f},{rot_ee[2]:+.4f}) "
            f"drot_base=({rot_base[0]:+.4f},{rot_base[1]:+.4f},{rot_base[2]:+.4f}) "
            f"action=({action[0]:+.4f},{action[1]:+.4f},{action[2]:+.4f},"
            f"{action[3]:+.4f},{action[4]:+.4f},{action[5]:+.4f},{action[6]:+.1f})",
            flush=True,
        )


    def _input_loop(self, generation: int):
        next_time = time.monotonic()
        while self.running and generation == self._input_generation:
            self._input_last_tick = time.monotonic()
            if self._reset_in_progress.is_set():
                time.sleep(INPUT_DT)
                next_time = time.monotonic()
                continue

            try:
                state = self.env.get_robot_state_vector()
                joint_state = self.env.get_joint_positions()
                commanded_pose = state[:6].copy()
                input_delta = self._get_input_delta()
                action = self._build_action(commanded_pose, input_delta)
                self._print_ee_debug(
                    state=state,
                    commanded_pose=commanded_pose,
                    input_delta=input_delta,
                    action=action,
                )
                if self._reset_in_progress.is_set():
                    next_time = time.monotonic()
                    continue
                recording_action = transform_action(action, self.env.action_config)
                recording_action[6] = action[6]
                with self._action_lock:
                    self._latest_action = action
                    self._latest_recording_action = recording_action
                    self._latest_state = state.copy()
                    self._latest_joint_state = joint_state.copy()
                    self._latest_commanded_pose = commanded_pose.copy()
                self.env.enqueue_action_block(action)
                self._append_recording_snapshot(
                    state=state,
                    joint_state=joint_state,
                    commanded_pose=commanded_pose,
                    action=recording_action,
                )
            except Exception as exc:
                if self._reset_in_progress.is_set() or self._reset_requested:
                    print(f"  [输入] 复位期间忽略输入异常: {exc}", flush=True)
                    time.sleep(INPUT_DT)
                    next_time = time.monotonic()
                    continue
                print(f"  [输入] 输入线程异常，停止控制: {exc}", flush=True)
                self.running = False
                self.env.request_stop()
                break
            next_time += INPUT_DT
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_time = time.monotonic()

    def _get_latest_action(self) -> np.ndarray:
        with self._action_lock:
            return self._latest_action.copy()

    def _start_input_thread(self, *, force_restart: bool = False) -> None:
        if self._input_thread is not None and self._input_thread.is_alive() and not force_restart:
            return
        self._input_generation += 1
        generation = self._input_generation
        with self._action_lock:
            self._latest_action = np.zeros(7, dtype=np.float64)
            self._latest_action[6] = 1.0 if self.gripper_target <= 0.0 else -1.0
            self._latest_recording_action = np.zeros(7, dtype=np.float64)
            self._latest_recording_action[6] = self._latest_action[6]
        self._input_last_tick = time.monotonic()
        self._input_thread = threading.Thread(target=self._input_loop, args=(generation,), daemon=True)
        self._input_thread.start()

    def _ensure_input_thread(self, *, reason: str) -> None:
        now = time.monotonic()
        alive = self._input_thread is not None and self._input_thread.is_alive()
        stale = alive and (now - self._input_last_tick) > max(0.5, INPUT_DT * 5.0)
        if alive and not stale:
            return
        print(f"  [输入] 重启输入线程: {reason}", flush=True)
        self._start_input_thread(force_restart=True)

    def _set_gripper_target(self, target: float) -> None:
        self.gripper_target = float(np.clip(target, 0.0, 0.08))
        with self._action_lock:
            self._latest_action[6] = 1.0 if self.gripper_target <= 0.0 else -1.0

    def _close_gripper(self):
        print("  [夹爪] 关闭")
        self._set_gripper_target(0.0)

    def _open_gripper(self):
        print("  [夹爪] 打开")
        self._set_gripper_target(0.08)


    def reset_to_home(self, open_gripper: bool = True):
        """公开复位入口，供数据录制器在保存/作废后调用。"""
        if open_gripper:
            self._open_gripper()
        self._request_reset()

    def _append_recording_snapshot(
        self,
        *,
        state: np.ndarray,
        joint_state: np.ndarray,
        commanded_pose: np.ndarray,
        action: np.ndarray,
    ) -> None:
        with self._trace_lock:
            self._recording_seq_id += 1
            snapshot = {
                "seq_id": self._recording_seq_id,
                "timestamp": time.time(),
                "state": np.asarray(state, dtype=np.float64).copy(),
                "joint_state": np.asarray(joint_state, dtype=np.float64).copy(),
                "commanded_pose": np.asarray(commanded_pose, dtype=np.float64).copy(),
                "action": np.asarray(action, dtype=np.float64).copy(),
            }
            self._recording_trace.append(snapshot)

    def get_state_and_action(self) -> tuple[np.ndarray, np.ndarray]:
        """返回最近一拍的状态和录制用 action。"""
        with self._action_lock:
            return self._latest_state.copy(), self._latest_recording_action.copy()

    def get_recording_snapshot(self) -> dict[str, np.ndarray | int | float]:
        """返回最近一拍的录制快照。"""
        with self._trace_lock:
            if self._recording_trace:
                latest = self._recording_trace[-1]
                return {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in latest.items()
                }

        state = self.env.get_robot_state_vector()
        return {
            "seq_id": self._recording_seq_id,
            "timestamp": time.time(),
            "state": state,
            "joint_state": self.env.get_joint_positions(),
            "commanded_pose": state[:6].copy(),
            "action": np.zeros(7, dtype=np.float64),
        }

    def get_recording_snapshots_since(self, last_seq_id: int) -> list[dict[str, np.ndarray | int | float]]:
        """返回所有 seq_id > last_seq_id 的录制快照。"""
        with self._trace_lock:
            snapshots = [item for item in self._recording_trace if int(item["seq_id"]) > int(last_seq_id)]
            return [
                {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in item.items()
                }
                for item in snapshots
            ]

    def run(self, *, home_first: bool = True):
        self.start(home_first=home_first)
        listener = None
        try:
            if self.input_device == "keyboard":
                listener = keyboard.Listener(
                    on_press=self._on_key_press,
                    on_release=self._on_key_release,
                )
                listener.start()
            self._start_input_thread()
            while self.running:
                self.env.start_control_loop(
                    max_duration=None,
                    print_events=True,
                    reference_name=self.reference_name,
                )
                wait_error = None
                try:
                    self.env.wait_control_loop()
                except Exception as exc:
                    wait_error = exc
                    if not (self._reset_requested or self._restart_requested):
                        raise
                    action = "复位" if self._reset_requested else "恢复控制"
                    print(f"  [控制] 停止当前控制时收到异常，继续{action}: {exc}", flush=True)
                if not (self._reset_requested or self._restart_requested):
                    break
                do_reset = self._reset_requested
                do_restart = self._restart_requested
                self._reset_requested = False
                self._restart_requested = False
                try:
                    if do_reset:
                        self._reset_env()
                        self._ensure_input_thread(reason="post-reset")
                    elif do_restart:
                        restart_ready_event = self._restart_ready_event
                        self._restart_ready_event = None
                        if restart_ready_event is not None:
                            print("  [控制] 等待保存完成...", flush=True)
                            if not restart_ready_event.wait(timeout=300.0):
                                print("  [控制] 保存完成等待超时，继续按计划恢复", flush=True)
                            else:
                                print("  [控制] 保存已完成", flush=True)
                        self._stop_rumble()
                        self._sync_ps4_state()
                        time.sleep(max(0.0, self._restart_delay_sec))
                        self._ensure_input_thread(reason="post-restart")
                        print("  [控制] 已恢复，可继续操作")
                finally:
                    if do_reset:
                        self._restart_ready_event = None
                    self._reset_in_progress.clear()
        finally:
            if listener is not None:
                listener.stop()
                listener.join()
            self.stop()

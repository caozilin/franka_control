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
import math
import os
import pathlib
import sys
import threading
import time

import numpy as np
from pynput import keyboard

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
try:
    import pygame
except ModuleNotFoundError:
    pygame = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from control.franka_env import FrankaEnv  # noqa: E402


INPUT_DT = 0.1
MAX_LIN_VEL = 0.1
MAX_ROT_VEL = math.pi / 4.0
MAX_DELTA_POS = MAX_LIN_VEL * INPUT_DT
MAX_DELTA_ROT = MAX_ROT_VEL * INPUT_DT
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
class KeyboardController:
    def __init__(
        self,
        *,
        robot_ip: str = "172.16.0.2",
        input_device: str = "keyboard",
        joystick_index: int = 0,
        max_translation_step: float = 0.1,
        max_rotation_step: float = math.pi / 4.0,
        reset_duration: float = 5.0,
        controller_name: str = "linear",
    ):
        if input_device not in ("keyboard", "ps4"):
            raise ValueError(f"不支持的输入设备: {input_device}")

        self.input_device = input_device
        self.joystick_index = int(joystick_index)
        self.robot_ip = robot_ip
        self.env = FrankaEnv(
            robot_ip=robot_ip,
            reset_duration=reset_duration,
            max_translation_step=max_translation_step,
            max_rotation_step=max_rotation_step,
            controller_name=controller_name,
        )
        self.max_translation_step = float(max_translation_step)
        self.max_rotation_step = float(max_rotation_step)
        self.controller_name = str(controller_name)

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
        self._action_lock = threading.Lock()
        self._latest_action = np.zeros(7, dtype=np.float64)
        self._input_thread: threading.Thread | None = None
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
        self.rumble(0.18, 0.7, 120)

    def _reset_env(self):
        print("  [复位] 回到初始位姿...")
        self.gripper_target = 0.08
        try:
            self.env.reset()
        except Exception:
            print("  [复位] 失败")
            raise
        print("  [复位] 完成")

    def stop(self):
        self.running = False
        self.env.request_stop()
        self.env.stop()
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
        self._shutdown_ps4()

    def _print_controls(self):
        if self.input_device == "keyboard":
            print("  [控制] 键盘模式")
            print("  [控制] W/S:X  A/D:Y  I/K:Z  Q/E:Roll  U/O:Pitch  J/L:Yaw")
            print("  [控制] G/H:夹爪  +/-:速度档位  R:复位  ESC:退出")
            return

        print("  [控制] PS4 手柄模式")
        print("  [控制] 左摇杆:X/Y平移  右摇杆:Z平移/Yaw  L1/R1:Roll  L2/R2:Pitch")
        print("  [控制] 三角/圆圈:打开/关闭夹爪  叉/方块:复位  十字键左右:速度档位  OPTIONS:退出")

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
        if pygame is None:
            raise ModuleNotFoundError(
                "PS4 手柄模式需要安装 pygame，请先执行 `pip install pygame`。"
            )

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

        if self._joystick is not None and self._joystick.get_init():
            self._joystick.quit()
        self._joystick = None
        pygame.joystick.quit()
        pygame.quit()
        self._pygame_ready = False

    def rumble(self, low: float = 0.0, high: float = 0.0, duration_ms: int = 150):
        if self.input_device != "ps4" or not self._pygame_ready or self._joystick is None:
            return
        try:
            if hasattr(self._joystick, "rumble"):
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
        print("  [复位] 停止当前控制后复位")
        self._reset_requested = True
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
                if button_idx in (PS4_BUTTON_CROSS, PS4_BUTTON_SQUARE):
                    self.rumble(0.2, 0.75, 90)
                    self._request_reset()
                    return False
                if button_idx == PS4_BUTTON_TRIANGLE:
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
        dpitch = (int("u" in keys) - int("o" in keys)) * MAX_DELTA_ROT * speed
        dyaw = (int("l" in keys) - int("j" in keys)) * MAX_DELTA_ROT * speed
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
        dpitch = (l2 - r2) * MAX_DELTA_ROT * speed
        dyaw = right_x * MAX_DELTA_ROT * speed
        return dx, dy, dz, droll, dpitch, dyaw

    def _get_input_delta(self) -> tuple[float, float, float, float, float, float]:
        if self.input_device == "ps4":
            return self._get_ps4_delta()
        return self._get_keyboard_delta()

    def _build_action(self) -> np.ndarray:
        dx, dy, dz, droll, dpitch, dyaw = self._get_input_delta()
        return np.array(
            [
                dx / self.max_translation_step,
                dy / self.max_translation_step,
                dz / self.max_translation_step,
                droll / self.max_rotation_step * 6.0,
                dpitch / self.max_rotation_step * 6.0,
                dyaw / self.max_rotation_step * 6.0,
                1.0 if self.gripper_target <= 0.0 else -1.0,
            ],
            dtype=np.float64,
        )


    def _input_loop(self):
        tick = 0
        next_time = time.monotonic()
        while self.running:
            action = self._build_action()
            with self._action_lock:
                self._latest_action = action
            try:
                self.env.enqueue_action_block(action)
            except Exception as exc:
                print(f"  [输入] action 入队失败: {exc}", flush=True)
                self.running = False
                self.env.request_stop()
                break
            tick += 1
            dx, dy, dz = action[0], action[1], action[2]
            drx, dry, drz = action[3], action[4], action[5]
            gripper = "open" if action[6] > 0 else "close"
            print(f"10Hz tick {tick:04d}  dxyz=[{dx:+.3f},{dy:+.3f},{dz:+.3f}]  "
                  f"drot=[{drx:+.3f},{dry:+.3f},{drz:+.3f}]  gripper={gripper}")
            next_time += INPUT_DT
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_time = time.monotonic()

    def _get_latest_action(self) -> np.ndarray:
        with self._action_lock:
            return self._latest_action.copy()

    def _start_input_thread(self) -> None:
        if self._input_thread is not None and self._input_thread.is_alive():
            return
        with self._action_lock:
            self._latest_action = np.zeros(7, dtype=np.float64)
            self._latest_action[6] = 1.0 if self.gripper_target <= 0.0 else -1.0
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

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
                    controller_name=self.controller_name,
                )
                self.env.wait_control_loop()
                if not self._reset_requested:
                    break
                self._reset_requested = False
                self._reset_env()
        finally:
            if listener is not None:
                listener.stop()
                listener.join()
            self.stop()

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

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
pygame = None
keyboard = None


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
from devices.pico import (  # noqa: E402
    DEFAULT_ROTATION_BASE_FROM_PICO,
    PicoMapperConfig,
    PicoPoseMapper,
    PicoSnapshot,
    PicoUdpReceiver,
)
from planning import (  # noqa: E402
    PANDA_TOLERANCE_PROFILES,
    CartesianActionPlanner,
    GripperPhaseClassifier,
    ManipulationPhase,
    PlannerConfig,
    box_tolerance_frame,
)
from planning.bottle_upright import BottleUprightPlanner
from utils import POLICY_HZ  # noqa: E402
from utils.control import transform_action  # noqa: E402
from utils.pose import end_effector_rotation_to_backend_rotation, rotvec_to_matrix  # noqa: E402


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
PS4_BUTTON_PS = 10
PS4_BUTTON_L3 = 11
PS4_BUTTON_R3 = 12


class KeyboardController:
    def __init__(
        self,
        *,
        robot_ip: str = "172.16.0.2",
        input_device: str = "keyboard",
        joystick_index: int = 0,
        rotation_frame: str = "ee",
        tolerance_id: str | None = None,
        pico_bind_host: str = "127.0.0.1",
        pico_port: int = 9010,
        pico_mapping_mode: str = "split",
        pico_translation_scale: float = 1.0,
        pico_rotation_scale: float = 1.0,
        pico_grip_threshold: float = 0.5,
        pico_trigger_threshold: float = 0.5,
        pico_stale_timeout_s: float = 0.15,
        pico_attitude_deadzone_deg: float = 10.0,
        pico_attitude_full_scale_deg: float = 25.0,
        pico_require_both_grips: bool = True,
        pico_rotation_base_from_pico: np.ndarray | None = None,
        max_translation_velocity: float = DEFAULT_MAX_TRANSLATION_VELOCITY,
        max_rotation_velocity: float = DEFAULT_MAX_ROTATION_VELOCITY,
        max_translation_goal_error: float = DEFAULT_MAX_TRANSLATION_GOAL_ERROR,
        max_rotation_goal_error: float = DEFAULT_MAX_ROTATION_GOAL_ERROR,
        max_torque_rate: float = DEFAULT_MAX_TORQUE_RATE,
        motion_limited_max_translation_velocity: float | None = None,
        motion_limited_max_rotation_velocity: float | None = None,
        motion_limited_max_translation_acceleration: float | None = None,
        motion_limited_max_rotation_acceleration: float | None = None,
        reset_duration: float = 5.0,
        reference_name: str = "linear",
        planner_mode: str = "direct",
        tracker_mode: str = "auto",
        pid_proportional_gain: float = 0.18,
        pid_integral_gain_s: float = 0.30,
        pid_velocity_gain_s: float = 0.04,
        pid_maximum_correction_rad: float = math.radians(3.0),
        pid_integration_error_limit_rad: float = math.radians(4.0),
        pid_integral_time_constant_s: float = 1.0,
        pid_stationary_integral_time_constant_s: float = 0.25,
        pid_stationary_velocity_threshold_rad_s: float = 0.02,
        rotation_ranged_axes: tuple[bool, bool, bool] = (False, False, False),
        rotation_limits_deg: tuple[float, float, float] = (30.0, 30.0, 45.0),
        tolerance_frame_rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0),
        shadow_stage: str = "teleop",
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
        if input_device not in ("keyboard", "ps4", "pico"):
            raise ValueError(f"不支持的输入设备: {input_device}")
        if rotation_frame not in ("ee", "base"):
            raise ValueError(f"不支持的旋转坐标系: {rotation_frame}")
        normalized_tolerance_id = None if tolerance_id is None else str(tolerance_id).upper()
        if normalized_tolerance_id is not None and normalized_tolerance_id not in PANDA_TOLERANCE_PROFILES:
            raise ValueError(f"未知 Franka 容差 ID: {tolerance_id}")
        if normalized_tolerance_id is not None and input_device != "ps4":
            raise ValueError("阶段容差 ID 目前只支持 PS4 手柄")
        if normalized_tolerance_id is not None and planner_mode == "direct":
            raise ValueError("阶段容差 ID 需要 --planner-mode baseline_sqp 或 shadow_sqp")

        self.input_device = input_device
        self.joystick_index = int(joystick_index)
        self.rotation_frame = rotation_frame
        self.tolerance_id = normalized_tolerance_id
        self.robot_ip = robot_ip
        action_planner = CartesianActionPlanner(
            PlannerConfig(
                mode=planner_mode,
                rotation_ranged_axes=rotation_ranged_axes,
                rotation_limits_deg=rotation_limits_deg,
                tolerance_frame_rotvec=tolerance_frame_rotvec,
                shadow_stage=shadow_stage,
            )
        )
        self.env = FrankaEnv(
            robot_ip=robot_ip,
            reset_duration=reset_duration,
            max_translation_velocity=max_translation_velocity,
            max_rotation_velocity=max_rotation_velocity,
            max_translation_goal_error=max_translation_goal_error,
            max_rotation_goal_error=max_rotation_goal_error,
            max_torque_rate=max_torque_rate,
            motion_limited_max_translation_velocity=motion_limited_max_translation_velocity,
            motion_limited_max_rotation_velocity=motion_limited_max_rotation_velocity,
            motion_limited_max_translation_acceleration=motion_limited_max_translation_acceleration,
            motion_limited_max_rotation_acceleration=motion_limited_max_rotation_acceleration,
            reference_name=reference_name,
            action_planner=action_planner,
            tracker_mode=tracker_mode,
            pid_proportional_gain=pid_proportional_gain,
            pid_integral_gain_s=pid_integral_gain_s,
            pid_velocity_gain_s=pid_velocity_gain_s,
            pid_maximum_correction_rad=pid_maximum_correction_rad,
            pid_integration_error_limit_rad=pid_integration_error_limit_rad,
            pid_integral_time_constant_s=pid_integral_time_constant_s,
            pid_stationary_integral_time_constant_s=pid_stationary_integral_time_constant_s,
            pid_stationary_velocity_threshold_rad_s=pid_stationary_velocity_threshold_rad_s,
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
        self.planner_mode = action_planner.mode
        self.tracker_mode = self.env.tracker_mode
        self.debug_ee_axes = bool(debug_ee_axes)
        self.action_planner = action_planner
        self._tolerance_profile = (
            None if self.tolerance_id is None else PANDA_TOLERANCE_PROFILES[self.tolerance_id]
        )
        self._phase_classifier = GripperPhaseClassifier()
        self._stage_target_poses: dict[ManipulationPhase, np.ndarray] = {}
        self._next_stage_capture = ManipulationPhase.PREGRASP
        self._active_manipulation_phase: ManipulationPhase | None = None
        self._tolerance_config_dirty = False
        self._bottle_planner = BottleUprightPlanner()
        self._bottle_actions: deque[np.ndarray] = deque()
        self._bottle_plan_lock = threading.Lock()
        self._bottle_planning = False
        self._bottle_plan_id = 0

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
        self.recording_active = False
        self._pico_receiver: PicoUdpReceiver | None = None
        self._pico_mapper: PicoPoseMapper | None = None
        self._pico_primary_pressed = False
        self._pico_secondary_pressed = False
        if self.input_device == "ps4":
            self._init_ps4()
        elif self.input_device == "pico":
            rotation_base_from_pico = np.asarray(
                DEFAULT_ROTATION_BASE_FROM_PICO
                if pico_rotation_base_from_pico is None
                else pico_rotation_base_from_pico,
                dtype=np.float64,
            )
            self._pico_receiver = PicoUdpReceiver(pico_bind_host, pico_port)
            self._pico_mapper = PicoPoseMapper(
                PicoMapperConfig(
                    mapping_mode=pico_mapping_mode,
                    translation_scale=pico_translation_scale,
                    rotation_scale=pico_rotation_scale,
                    max_translation_step_m=self.env.max_translation_step,
                    max_rotation_step_rad=self.env.max_rotation_step,
                    grip_threshold=pico_grip_threshold,
                    trigger_threshold=pico_trigger_threshold,
                    stale_timeout_s=pico_stale_timeout_s,
                    attitude_deadzone_rad=np.deg2rad(pico_attitude_deadzone_deg),
                    attitude_full_scale_rad=np.deg2rad(pico_attitude_full_scale_deg),
                    require_both_grips=pico_require_both_grips,
                    rotation_base_from_pico=rotation_base_from_pico,
                )
            )

    def start(self, *, home_first: bool = True):
        self._print_controls()
        if self._pico_receiver is not None:
            self._pico_receiver.start()
            print(f"  [PICO] UDP 接收: {self._pico_receiver.host}:{self._pico_receiver.port}")
        if home_first:
            self._reset_env()
        self._stop_rumble()
        self._sync_ps4_state()
        self._sync_pico_state()
        self.rumble(0.18, 0.7, 120)

    def _reset_env(self):
        print("  [复位] 回到初始位姿...")
        with self._bottle_plan_lock:
            self._bottle_plan_id += 1
            self._bottle_planning = False
            self._bottle_actions.clear()
        self.gripper_target = 0.08
        self._stop_rumble()
        try:
            self.env.reset()
        except Exception:
            print("  [复位] 失败")
            raise
        self._sync_ps4_state()
        if self._pico_mapper is not None:
            self._pico_mapper.reset()
        self._sync_pico_state()
        self._phase_classifier.reset()
        self._active_manipulation_phase = None
        self._tolerance_config_dirty = bool(self._stage_target_poses)
        print("  [复位] 完成")

    def stop(self):
        self.running = False
        self.env.request_stop()
        self.env.stop()
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
        self._shutdown_ps4()
        if self._pico_receiver is not None:
            self._pico_receiver.stop()

    def bind_event(self, event_name: str, callback):
        """绑定离散事件回调，例如 record_start / record_stop。"""
        self._event_callbacks[event_name] = callback

    def _emit_event(self, event_name: str):
        callback = self._event_callbacks.get(event_name)
        if callback is not None:
            callback()

    def set_recording_active(self, active: bool) -> None:
        self.recording_active = bool(active)

    def _print_controls(self):
        print(
            f"  [规划] planner={self.planner_mode} reference={self.reference_name} tracker={self.tracker_mode}"
        )
        if self.input_device != "pico":
            print(f"  [旋转] input_frame={self.rotation_frame} backend_frame=base")
        if self.input_device == "keyboard":
            print("  [控制] 键盘模式")
            print("  [控制] W/S:X  A/D:Y  I/K:Z  Q/E:Roll  U/O:Pitch  J/L:Yaw")
            print("  [控制] G/H:夹爪  N:相机/瓶口同边扶瓶  M:相机/瓶口异边扶瓶  +/-:速度档位  R:复位  1/2/3:开始/结束并恢复控制/作废录制  ESC:退出")
            return

        if self.input_device == "pico":
            print("  [控制] PICO 双手柄模式")
            print("  [控制] 左 Grip+相对位置:Base 平移  右 Grip+相对姿态:末端系旋转")
            print("  [控制] 手柄保持在相对目标时机械臂最终静止；右 Trigger:切换夹爪")
            print("  [控制] A:开始录制/再次按下保存  B:作废当前录制  Ctrl+C:退出")
            return

        print("  [控制] PS4 手柄模式")
        print("  [控制] 左摇杆:X/Y平移  右摇杆:Z平移/Yaw  L1/R1:Roll  L2/R2:Pitch")
        print("  [控制] 三角/圆圈:打开/关闭夹爪  叉:作废并复位  方块:复位  十字键左右:速度档位")
        print("  [控制] L3/R3:开始/结束录制并恢复控制  OPTIONS:退出")
        print("  [控制] 键盘 N:腕部相机/瓶口同边  M:异边（最终末端相差 180deg）")
        if self._tolerance_profile is not None:
            print(f"  [容差] id={self.tolerance_id}；PS键依次采集/覆盖 Pre、Post 目标姿态")
            print(
                f"  [容差] Pre={self._tolerance_profile.pre_deg} deg  "
                f"Post={self._tolerance_profile.post_deg} deg"
            )

    def _on_key_press(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            char = key.name.lower()

        if char in ("n", "m"):
            self._start_bottle_plan(camera_mouth_same_side=(char == "n"), key_name=char.upper())
            return
        if self.input_device != "keyboard":
            return

        if char == "escape":
            self.running = False
            self.env.request_stop()
            return False

        with self._keys_lock:
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

    def _start_bottle_plan(
        self,
        *,
        camera_mouth_same_side: bool,
        key_name: str,
    ) -> None:
        with self._bottle_plan_lock:
            if self._bottle_planning or self._bottle_actions:
                print("  [扶瓶] 已在规划或执行中", flush=True)
                return
            if self.gripper_target > 0.0:
                print("  [扶瓶] 拒绝启动：请先抓紧瓶子", flush=True)
                return
            self._bottle_planning = True
            self._bottle_plan_id += 1
            plan_id = self._bottle_plan_id
        print(
            f"  [扶瓶] 开始生成 {key_name} 序列："
            f"腕部相机/瓶口={'同边' if camera_mouth_same_side else '异边'}",
            flush=True,
        )

        def worker() -> None:
            try:
                state = self.env.get_robot_state_vector()
                joint = self.env.get_joint_positions()
                plan = self._bottle_planner.plan(
                    joint,
                    state[:3],
                    rotvec_to_matrix(state[3:6]),
                    camera_mouth_same_side=camera_mouth_same_side,
                )
                candidate = plan.candidate
                with self._bottle_plan_lock:
                    if plan_id != self._bottle_plan_id:
                        return
                    self._bottle_actions.extend(action.copy() for action in plan.actions)
                print(
                    f"  [扶瓶] key={key_name} "
                    f"腕部相机/瓶口={'同边' if camera_mouth_same_side else '异边'} "
                    f"候选={plan.candidate_count} 选择={candidate.label} "
                    f"目标=({candidate.final_position[0]:.6f},"
                    f"{candidate.final_position[1]:.6f},"
                    f"{candidate.final_position[2]:.6f}) "
                    f"joint_margin={math.degrees(candidate.minimum_joint_margin_rad):.1f}deg "
                    f"frames={len(plan.actions)}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  [扶瓶] {exc}", flush=True)
            finally:
                with self._bottle_plan_lock:
                    if plan_id == self._bottle_plan_id:
                        self._bottle_planning = False

        threading.Thread(target=worker, name="bottle_planner", daemon=True).start()

    def _next_bottle_action(self) -> np.ndarray | None:
        with self._bottle_plan_lock:
            if not self._bottle_actions:
                if not self._bottle_planning:
                    return None
                # Planning uses a snapshot of the current pose.  Freeze the
                # Cartesian command until the worker publishes the trajectory
                # so manual input cannot invalidate that snapshot.
                action = np.zeros(7, dtype=np.float64)
                action[6] = 1.0
                return action
            action = self._bottle_actions.popleft()
            remaining = len(self._bottle_actions)
        self._set_gripper_target(0.0 if action[6] > 0.0 else 0.08)
        if remaining == 0:
            print("  [扶瓶] 自动轨迹执行完成", flush=True)
        return action

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

    def _sync_pico_state(self) -> None:
        if self._pico_receiver is None:
            return
        snapshot = self._pico_receiver.latest()
        if snapshot is None:
            self._pico_primary_pressed = False
            self._pico_secondary_pressed = False
            return
        self._pico_primary_pressed = bool(snapshot.packet.right.primary)
        self._pico_secondary_pressed = bool(snapshot.packet.right.secondary)

    def _capture_stage_target(self) -> None:
        if self._tolerance_profile is None:
            return
        state = self.env.get_robot_state_vector()
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = state[:3]
        pose[:3, :3] = rotvec_to_matrix(state[3:6])
        stage = self._next_stage_capture
        self._stage_target_poses[stage] = pose
        self._next_stage_capture = (
            ManipulationPhase.POSTGRASP
            if stage is ManipulationPhase.PREGRASP
            else ManipulationPhase.PREGRASP
        )
        self._tolerance_config_dirty = True
        label = "Pre" if stage is ManipulationPhase.PREGRASP else "Post"
        p = pose[:3, 3]
        print(
            f"  [容差目标] 已记录 {label}: "
            f"position=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}) "
            f"rotvec=({state[3]:+.4f},{state[4]:+.4f},{state[5]:+.4f})",
            flush=True,
        )
        if len(self._stage_target_poses) < 2:
            print("  [容差目标] 请移动到 Post 目标姿态，再按一次 PS 键", flush=True)
        else:
            next_label = "Pre" if self._next_stage_capture is ManipulationPhase.PREGRASP else "Post"
            print(f"  [容差目标] Pre/Post 已就绪；下一次 PS 键将覆盖 {next_label}", flush=True)

    def _update_stage_tolerance(self, state: np.ndarray) -> None:
        if self._tolerance_profile is None or len(self._stage_target_poses) < 2:
            return
        aperture = float(abs(state[6]) + abs(state[7])) if state.shape[0] >= 8 else self.gripper_target
        observation = self._phase_classifier.update(
            aperture,
            commanded_closed=self.gripper_target <= 0.0,
        )
        phase = observation.phase
        if phase is self._active_manipulation_phase and not self._tolerance_config_dirty:
            return
        stable_phase = (
            ManipulationPhase.PREGRASP
            if phase in (ManipulationPhase.PREGRASP, ManipulationPhase.GRASP)
            else ManipulationPhase.POSTGRASP
        )
        pose = self._stage_target_poses[stable_phase]
        if phase in (ManipulationPhase.GRASP, ManipulationPhase.RELEASE):
            negative = np.zeros(3, dtype=np.float64)
            positive = np.zeros(3, dtype=np.float64)
        else:
            negative, positive = self._tolerance_profile.bounds_rad(phase)
        self.action_planner.configure_rotation_tolerance(
            pose[:3, :3],
            box_tolerance_frame(pose[:3, :3]),
            negative,
            positive,
            phase=phase,
        )
        self._active_manipulation_phase = phase
        self._tolerance_config_dirty = False
        print(
            f"  [容差阶段] {phase.key.upper()} aperture={observation.aperture_m:.4f}m "
            f"bounds_deg=(-{np.degrees(negative).round(1).tolist()},"
            f"+{np.degrees(positive).round(1).tolist()})",
            flush=True,
        )

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
                if button_idx == PS4_BUTTON_PS:
                    self._capture_stage_target()
                    continue
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

    def _handle_pico_recording_buttons(self, snapshot: PicoSnapshot) -> None:
        primary = bool(snapshot.packet.right.primary)
        secondary = bool(snapshot.packet.right.secondary)
        primary_rising = primary and not self._pico_primary_pressed
        secondary_rising = secondary and not self._pico_secondary_pressed
        self._pico_primary_pressed = primary
        self._pico_secondary_pressed = secondary

        # B wins if both buttons arrive pressed in the same packet: discard must
        # never immediately start a fresh episode by accident.
        if secondary_rising:
            self._emit_event("record_discard")
            return
        if primary_rising:
            self._emit_event("record_stop" if self.recording_active else "record_start")

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
        rotation_delta = np.array([droll, dpitch, dyaw], dtype=np.float64)
        if self.rotation_frame == "ee":
            rotation_delta = self._end_effector_rotation_to_backend_rotation(
                rotation_delta,
                commanded_pose,
            )
        drx, dry, drz = rotation_delta
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

    def _build_pico_action(self, commanded_pose: np.ndarray) -> tuple[np.ndarray, tuple[float, ...]]:
        assert self._pico_receiver is not None and self._pico_mapper is not None
        snapshot = self._pico_receiver.latest()
        command = self._pico_mapper.step(snapshot)
        if snapshot is not None and snapshot.age_s() <= self._pico_mapper.config.stale_timeout_s:
            self._handle_pico_recording_buttons(snapshot)
        if command is None:
            action = np.zeros(7, dtype=np.float64)
            action[6] = 1.0 if self.gripper_target <= 0.0 else -1.0
            return action, tuple(action[:6])

        action = command.action.copy()
        action[3:6] = self._end_effector_rotation_to_backend_rotation(
            action[3:6],
            commanded_pose,
        )
        self._set_gripper_target(0.0 if action[6] > 0.0 else 0.08)
        return action, tuple(command.action[:6])

    def _build_current_input_action(
        self,
        commanded_pose: np.ndarray,
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        if self.input_device == "pico":
            return self._build_pico_action(commanded_pose)
        input_delta = self._get_input_delta()
        return self._build_action(commanded_pose, input_delta), input_delta

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
        rot_input = np.asarray(input_delta[3:6], dtype=np.float64)
        rot_base = np.asarray(action[3:6], dtype=np.float64)
        rotation_input_frame = "ee" if self.input_device == "pico" else self.rotation_frame
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
            f"drot_{rotation_input_frame}=({rot_input[0]:+.4f},{rot_input[1]:+.4f},{rot_input[2]:+.4f}) "
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
                # Poll the active device even during autonomous execution so
                # reset/abort/exit buttons remain responsive.
                manual_action, input_delta = self._build_current_input_action(commanded_pose)
                bottle_action = self._next_bottle_action()
                if bottle_action is None:
                    action = manual_action
                else:
                    action = bottle_action
                    input_delta = tuple(float(value) for value in action[:6])
                self._update_stage_tolerance(state)
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
                self.env.enqueue_cartesian_action(
                    action,
                    semantic_key=(
                        "teleop",
                        self.input_device,
                        self.planner_mode,
                        None
                        if self._active_manipulation_phase is None
                        else self._active_manipulation_phase.key,
                    ),
                )
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
        self._input_thread = threading.Thread(
            target=self._input_loop,
            args=(generation,),
            name="franka_policy",
            daemon=True,
        )
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
        global keyboard
        self.start(home_first=home_first)
        listener = None
        try:
            if keyboard is None:
                from pynput import keyboard as pynput_keyboard

                keyboard = pynput_keyboard
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

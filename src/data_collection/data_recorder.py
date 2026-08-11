#!/usr/bin/env python3
"""
Franka Panda 数据采集脚本
========================
兼容 vla4desk/src/data_collection 的采集格式：每段轨迹保存 cam1.mp4、cam2.mp4 和 data.json。

键盘：
    1   开始录制
    2   结束录制并保存，约2秒后恢复控制
    3   作废当前轨迹并复位
    ESC 退出，不自动保存未完成轨迹

PS4：
    L3      开始录制
    R3      结束录制并保存，约2秒后恢复控制
    Cross   作废当前轨迹并复位
    OPTIONS 退出
"""

from __future__ import annotations

import argparse
import logging
import math
import pathlib
import sys
import threading
import time
from collections import deque

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_collection.key_control import (
    DEFAULT_MAX_ROTATION_VELOCITY,
    DEFAULT_MAX_ROTATION_GOAL_ERROR,
    DEFAULT_MAX_TRANSLATION_VELOCITY,
    DEFAULT_MAX_TRANSLATION_GOAL_ERROR,
    DEFAULT_MAX_TORQUE_RATE,
    KeyboardController,
)  # noqa: E402
from data_collection.episode_io import Episode, EpisodeWriter  # noqa: E402


class DataRecorder:
    COLLECTION_DIR = REPO_ROOT / "collected"
    TASK_NAME = "default"
    COLLECT_HZ = 10.0
    MAX_FRAMES = 3000
    ACTION_THRESH_POS = 0.0005
    ACTION_THRESH_ROT = 0.005
    STATE_THRESH_POS = 0.0005
    STATE_THRESH_ROT = 0.005
    STATE_THRESH_GRIPPER = 0.0005
    GRIPPER_WIDTH_OPEN = 0.08
    GRIPPER_WIDTH_CLOSED = 0.0
    GRIPPER_EXTRA_KEEP_FRAMES = 5
    ACTION_SCALE = 100.0
    PROMPT = ""

    def __init__(
        self,
        key_control: KeyboardController,
        collection_dir=None,
        task_name=None,
        collect_hz=None,
        max_frames=None,
        action_thresh_pos=None,
        action_thresh_rot=None,
        action_scale=None,
        prompt=None,
        episode_writer: EpisodeWriter | None = None,
    ) -> None:
        self.kc = key_control
        if collection_dir:
            collection_path = pathlib.Path(collection_dir)
            self.collection_dir = collection_path if collection_path.is_absolute() else self.COLLECTION_DIR / collection_path
        else:
            self.collection_dir = self.COLLECTION_DIR
        self.task_name = task_name or self.TASK_NAME
        self.collect_hz = float(collect_hz or self.COLLECT_HZ)
        self.max_frames = int(max_frames if max_frames is not None else self.MAX_FRAMES)
        self.dt = 1.0 / self.collect_hz
        self.action_scale = float(action_scale) if action_scale is not None else self.ACTION_SCALE
        self.prompt = self.PROMPT if prompt is None else str(prompt)
        self.action_thresh_pos = float(action_thresh_pos if action_thresh_pos is not None else self.ACTION_THRESH_POS)
        self.action_thresh_rot = float(action_thresh_rot if action_thresh_rot is not None else self.ACTION_THRESH_ROT)
        self.episode_writer = episode_writer or EpisodeWriter(self.collection_dir)

        self.is_recording = False
        self.recording_frames1: deque[np.ndarray] = deque(maxlen=self.max_frames)
        self.recording_frames2: deque[np.ndarray] = deque(maxlen=self.max_frames)
        self.recording_data: list[dict] = []
        self.record_lock = threading.Lock()

        self.stats = dict(trajectories_saved=0, total_frames=0, skipped_frames=0)
        self.running = True
        self._exited = False
        self._last_recorded_state: np.ndarray | None = None
        self._gripper_force_record_frames = 0
        self._last_trace_seq = 0
        self._status_peak_ee_force_torque = np.zeros(6, dtype=np.float64)
        self._stats_lock = threading.Lock()
        self._pending_save: tuple[list[np.ndarray], list[np.ndarray], list[dict], threading.Event] | None = None

    def _scale_action_for_storage(self, action: np.ndarray) -> list[float]:
        scaled_action = np.asarray(action, dtype=np.float64).copy()
        scaled_action[:6] *= self.action_scale
        return scaled_action.tolist()

    def start_recording(self) -> None:
        if self.is_recording:
            print("  [录制] 已在录制中")
            return
        with self.record_lock:
            self.is_recording = True
            self.recording_frames1.clear()
            self.recording_frames2.clear()
            self.recording_data.clear()
            self._last_recorded_state = None
            self._gripper_force_record_frames = 0
            latest_trace = self.kc.get_recording_snapshot()
            self._last_trace_seq = int(latest_trace["seq_id"])
        self.kc.rumble(0.15, 0.85, 120)
        print("  [录制] 开始 -> 按 2/R3 结束当前轨迹，停控并同步保存后立即恢复")

    def stop_recording(self) -> None:
        if not self.is_recording:
            print("  [录制] 当前没有在录制")
            return
        with self.record_lock:
            self.is_recording = False
            frames1 = list(self.recording_frames1)
            frames2 = list(self.recording_frames2)
            data = list(self.recording_data)
        self.kc.rumble(0.55, 0.25, 180)
        print("  [录制] R3/2 已触发，先停止机械臂，再同步保存当前轨迹")
        save_done = threading.Event()
        if frames1 and data:
            self._pending_save = (frames1, frames2, data, save_done)
        else:
            self._pending_save = ([], [], [], save_done)
        self.kc.request_control_restart(delay_sec=0.0, ready_event=save_done)

    def discard_recording(self) -> None:
        with self.record_lock:
            if not self.is_recording:
                print("  [录制] 当前没有在录制，跳过作废")
                return
            discarded_frames = len(self.recording_data)
            self.is_recording = False
            self.recording_frames1.clear()
            self.recording_frames2.clear()
            self.recording_data.clear()
            self._last_recorded_state = None
            self._gripper_force_record_frames = 0
            latest_trace = self.kc.get_recording_snapshot()
            self._last_trace_seq = int(latest_trace["seq_id"])
        print(f"  [录制] 当前轨迹已作废，不保存 ({discarded_frames} 帧)")

    def _save_trajectory(self, frames1: list[np.ndarray], frames2: list[np.ndarray], data: list[dict]) -> None:
        episode = Episode(
            frames1=frames1,
            frames2=frames2,
            data=data,
            task_name=self.task_name,
            collect_hz=self.collect_hz,
            max_frames=self.max_frames,
            action_scale=self.action_scale,
            prompt=self.prompt,
        )
        self.episode_writer.save(episode)

        with self._stats_lock:
            self.stats["trajectories_saved"] += 1
            self.stats["total_frames"] += len(data)
            saved = self.stats["trajectories_saved"]
            total = self.stats["total_frames"]
            skipped = self.stats["skipped_frames"]
        print(
            f"  [统计] 已保存 {saved} 段, "
            f"总帧 {total}, 跳过 {skipped} 帧"
        )



    def _is_gripper_command_active(self, action: np.ndarray, state: np.ndarray) -> bool:
        width = self._gripper_width(state)
        if action.shape[0] < 7:
            return False
        # Recording action uses +1 for close, -1 for open.
        if action[6] > 0.0:
            return width > (self.GRIPPER_WIDTH_CLOSED + self.STATE_THRESH_GRIPPER)
        if action[6] < 0.0:
            return width < (self.GRIPPER_WIDTH_OPEN - self.STATE_THRESH_GRIPPER)
        return False

    def _is_action_empty(self, action: np.ndarray, state: np.ndarray) -> bool:
        return bool(
            (np.abs(action[:3]) < self.action_thresh_pos).all()
            and (np.abs(action[3:6]) < self.action_thresh_rot).all()
            and not self._is_gripper_command_active(action, state)
        )

    def _is_state_same_as_last_recorded(self, state: np.ndarray) -> bool:
        if self._last_recorded_state is None:
            return False
        return bool(
            (np.abs(state[:3] - self._last_recorded_state[:3]) < self.STATE_THRESH_POS).all()
            and (np.abs(state[3:6] - self._last_recorded_state[3:6]) < self.STATE_THRESH_ROT).all()
            and (np.abs(state[6:8] - self._last_recorded_state[6:8]) < self.STATE_THRESH_GRIPPER).all()
        )

    def _gripper_width(self, state: np.ndarray) -> float:
        return float(state[6] - state[7])

    def _should_force_record_for_gripper(self, state: np.ndarray) -> bool:
        width = self._gripper_width(state)
        near_closed = abs(width - self.GRIPPER_WIDTH_CLOSED) < self.STATE_THRESH_GRIPPER
        near_open = abs(width - self.GRIPPER_WIDTH_OPEN) < self.STATE_THRESH_GRIPPER
        return not (near_closed or near_open)

    def run(self) -> None:
        self.running = True
        control_hint = (
            "  1: 开始录制   2: 结束录制并保存后恢复控制   3: 作废并复位   ESC: 退出"
            if self.kc.input_device == "keyboard"
            else "  L3: 开始录制   R3: 结束录制并保存后恢复控制   Cross: 作废并复位   OPTIONS: 退出"
        )
        print("\n" + "=" * 60)
        print("  数据录制器已启动")
        print(f"  保存路径: {self.collection_dir / self.task_name}")
        print(f"  采集频率: {self.collect_hz}Hz  每段最大帧: {self.max_frames}")
        print("=" * 60)
        print(control_hint)
        print("=" * 60 + "\n")

        last_t = time.time()
        frame_count = 0
        while self.kc.running:
            pending_save = self._pending_save
            if pending_save is not None and not self.kc.env.is_control_running():
                frames1, frames2, data, done_event = pending_save
                self._pending_save = None
                try:
                    if frames1 and data:
                        print("  [保存] 控制已停止，开始同步写入当前轨迹")
                        self._save_trajectory(frames1, frames2, data)
                        print("  [录制] 当前轨迹保存完成，控制将立即恢复")
                    else:
                        print("  [录制] 没有采到有效帧，控制将立即恢复")
                finally:
                    done_event.set()
            elapsed = time.time() - last_t
            if elapsed >= self.dt:
                last_t += self.dt
                self._collect_frame()
                frame_count += 1
                if frame_count % max(1, int(self.collect_hz)) == 0:
                    self._print_status()
            time.sleep(0.005)
        self._on_exit()

    def _collect_frame(self) -> None:
        current_ft = np.abs(np.asarray(self.kc.env.ee_force_torque, dtype=np.float64))
        self._status_peak_ee_force_torque = np.maximum(self._status_peak_ee_force_torque, current_ft)
        traces = self.kc.get_recording_snapshots_since(self._last_trace_seq)
        if not traces:
            return
        for trace in traces:
            self._last_trace_seq = int(trace["seq_id"])
            state = np.asarray(trace["state"], dtype=np.float64)
            joint_state = np.asarray(trace["joint_state"], dtype=np.float64)
            commanded_pose = np.asarray(trace["commanded_pose"], dtype=np.float64)
            action = np.asarray(trace["action"], dtype=np.float64)
            timestamp = float(trace["timestamp"])

            force_record = False
            if self._should_force_record_for_gripper(state):
                self._gripper_force_record_frames = self.GRIPPER_EXTRA_KEEP_FRAMES
            if self._gripper_force_record_frames > 0:
                force_record = True
            if not force_record and self._is_action_empty(action, state) and self._is_state_same_as_last_recorded(state):
                self.stats["skipped_frames"] += 1
                continue

            img1, img2 = self.kc.env.get_camera_frames()

            recording = False
            with self.record_lock:
                recording = self.is_recording
            if not recording:
                continue

            with self.record_lock:
                if not self.is_recording:
                    continue
                self.recording_frames1.append(img1.copy())
                self.recording_frames2.append(img2.copy())
                self.recording_data.append({
                    "id": len(self.recording_data),
                    "timestamp": round(timestamp, 6),
                    "state": state.tolist(),
                    "joint_state": joint_state.tolist(),
                    "action": self._scale_action_for_storage(action),
                    "commanded_pose": commanded_pose.tolist(),
                })
                self._last_recorded_state = state.copy()
                if self._gripper_force_record_frames > 0:
                    self._gripper_force_record_frames -= 1
                if len(self.recording_frames1) >= self.max_frames:
                    print(f"  [录制] 达到最大帧数 {self.max_frames}，自动结束")
                    self.stop_recording()
                    break

    def _print_status(self) -> None:
        state, _ = self.kc.get_state_and_action()
        pos = state[:3]
        gripper = state[6] - state[7]
        latest_trace = self.kc.get_recording_snapshot()
        commanded_pose = np.asarray(latest_trace["commanded_pose"], dtype=np.float64)
        pos_error_norm = float(np.linalg.norm(state[:3] - commanded_pose[:3]))
        rot_error_norm = float(np.linalg.norm(state[3:6] - commanded_pose[3:6]))
        ft = self._status_peak_ee_force_torque.copy()
        self._status_peak_ee_force_torque.fill(0.0)
        status = "REC" if self.is_recording else "IDLE"
        frames = len(self.recording_data) if self.is_recording else 0
        print(
            f"  [{status}] pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})  "
            f"gripper={gripper:.3f}m  pos_err={pos_error_norm:.6f}  rot_err={rot_error_norm:.6f}  "
            f"ft_max=({ft[0]:.2f}, {ft[1]:.2f}, {ft[2]:.2f}, {ft[3]:.2f}, {ft[4]:.2f}, {ft[5]:.2f})  "
            f"epi={self.stats['trajectories_saved']}  cur_frames={frames}"
        )

    def _on_exit(self) -> None:
        if self._exited:
            return
        self._exited = True
        print("\n  [退出] 直接结束，不自动保存未完成轨迹")
        with self.record_lock:
            self.is_recording = False
            self.recording_frames1.clear()
            self.recording_frames2.clear()
            self.recording_data.clear()
        print("  [退出] 完成")


def parse_joint_vector(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError("--nullspace-q-target must contain 7 comma-separated joint values")
    return np.asarray(parts, dtype=np.float64)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Franka 数据采集器")
    parser.add_argument("--collection_dir", "--collection-dir", default=None)
    parser.add_argument("--task_name", "--task-name", default=None)
    parser.add_argument("--collect_hz", "--collect-hz", type=float, default=None)
    parser.add_argument("--max_frames", "--max-frames", type=int, default=None)
    parser.add_argument("--action_scale", "--action-scale", type=float, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--ip", default="172.16.0.2")
    parser.add_argument("--input_device", "--input-device", choices=("keyboard", "ps4"), default="ps4")
    parser.add_argument("--joystick_index", "--joystick-index", type=int, default=0)
    parser.add_argument("--no-home-first", action="store_true")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--debug-ee-axes", action="store_true", help="Print 10Hz end-effector-frame input and base-frame action debug lines")
    parser.add_argument("--max-translation-velocity", type=float, default=DEFAULT_MAX_TRANSLATION_VELOCITY)
    parser.add_argument("--max-rotation-velocity", type=float, default=DEFAULT_MAX_ROTATION_VELOCITY)
    parser.add_argument("--max-translation-goal-error", type=float, default=DEFAULT_MAX_TRANSLATION_GOAL_ERROR)
    parser.add_argument("--max-rotation-goal-error", type=float, default=DEFAULT_MAX_ROTATION_GOAL_ERROR)
    parser.add_argument("--max-torque-rate", type=float, default=DEFAULT_MAX_TORQUE_RATE)
    parser.add_argument("--reset-duration", type=float, default=5.0)
    parser.add_argument("--reference", choices=("min_jerk", "linear", "cubic", "motion_limited"), default="linear")
    parser.add_argument("--nullspace-enabled", action="store_true")
    parser.add_argument("--nullspace-pinv", choices=("plain", "damped"), default="plain")
    parser.add_argument("--nullspace-projector", choices=("kinematic", "dynamic"), default="kinematic")
    parser.add_argument("--nullspace-lambda", type=float, default=0.05)
    parser.add_argument("--nullspace-stiffness", type=float, default=10.0)
    parser.add_argument("--nullspace-damping", type=float, default=2.0)
    parser.add_argument("--nullspace-q-target", type=parse_joint_vector, default=None)
    args = parser.parse_args()

    kc = KeyboardController(
        robot_ip=args.ip,
        input_device=args.input_device,
        joystick_index=args.joystick_index,
        max_translation_velocity=args.max_translation_velocity,
        max_rotation_velocity=args.max_rotation_velocity,
        max_translation_goal_error=args.max_translation_goal_error,
        max_rotation_goal_error=args.max_rotation_goal_error,
        reset_duration=args.reset_duration,
        reference_name=args.reference,
        save_recording=False,
        nullspace_enabled=args.nullspace_enabled,
        nullspace_q_target=args.nullspace_q_target,
        nullspace_stiffness=args.nullspace_stiffness,
        nullspace_damping=args.nullspace_damping,
        nullspace_pinv=args.nullspace_pinv,
        nullspace_projector=args.nullspace_projector,
        nullspace_lambda=args.nullspace_lambda,
        no_robot=args.no_robot,
        no_cameras=args.no_cameras,
        debug_ee_axes=args.debug_ee_axes,
    )

    recorder = DataRecorder(
        key_control=kc,
        collection_dir=args.collection_dir,
        task_name=args.task_name,
        collect_hz=args.collect_hz,
        max_frames=args.max_frames,
        action_scale=args.action_scale,
        prompt=args.prompt,
    )
    kc.bind_event("record_start", recorder.start_recording)
    kc.bind_event("record_stop", recorder.stop_recording)
    kc.bind_event("record_discard", recorder.discard_recording)

    rec_thread = threading.Thread(target=recorder.run, daemon=True)
    rec_thread.start()
    try:
        kc.run(home_first=not args.no_home_first)
    finally:
        kc.stop()
        recorder._on_exit()
        rec_thread.join(timeout=1.0)
    print("程序已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

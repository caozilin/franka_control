#!/usr/bin/env python3
"""
Franka Panda 键盘遥操作程序
==========================
基于笛卡尔阻抗力矩控制，架构完全照搬 vla4desk：
  - 独立 10Hz 输入线程（_input_loop）：读取按键 → 计算 action → 本地积分位姿
  - 主控制循环（1kHz）：读取最新 action → 应用到 target_pose → 发送力矩
  - 线程安全：_keys_lock 保护按键状态，_state_lock 保护 action/位姿

键位映射（WASD + IJKL + QEUO）：
  W/S: X轴平移
  A/D: Y轴平移
  I/K: Z轴平移
  Q/E: Roll旋转
  U/O: Pitch旋转
  J/L: Yaw旋转

控制参数：
  INPUT_DT: 100ms (10Hz) 输入采样间隔
  MAX_LIN_VEL: 0.1 m/s 最大线速度
  MAX_ROT_VEL: π/4 rad/s 最大角速度
"""

# 记录参数
MAX_TELEOP_DURATION = 120.0  # 最大记录时间秒
TRACE_CAPACITY = int(1000 * MAX_TELEOP_DURATION)  # 1kHz * 时间

import argparse
import math
import threading
import time

import numpy as np
from pynput import keyboard

from pylibfranka import Robot, Torques


# ==================================================================
# 控制参数
# ==================================================================

INPUT_DT = 0.1               # 输入采样间隔 100ms (10Hz)
MAX_LIN_VEL = 0.1             # 最大线速度 0.1 m/s
MAX_ROT_VEL = math.pi / 4     # 最大角速度 45°/s
MAX_DELTA_POS = MAX_LIN_VEL * INPUT_DT     # 0.01 m
MAX_DELTA_ROT = MAX_ROT_VEL * INPUT_DT     # ≈ 0.0785 rad

# 打印参数
PRINT_HZ = 1.0
PRINT_INTERVAL = 1.0 / PRINT_HZ

# 阻抗控制参数
DEFAULT_STIFFNESS = np.diag([600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
DEFAULT_DAMPING_RATIO = 2.0
DEFAULT_DAMPING = np.diag([DEFAULT_DAMPING_RATIO * np.sqrt(DEFAULT_STIFFNESS[i, i]) for i in range(6)])


# ==================================================================
# 旋转工具函数
# ==================================================================

def rotvec_to_matrix(rotvec):
    """将 rotvec 转换为旋转矩阵（罗德里格斯公式）"""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c,   t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c],
    ], dtype=np.float64)


def matrix_to_rotvec(matrix):
    """将旋转矩阵转换为 rotvec (轴角)"""
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    cos_angle = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    angle = math.acos(cos_angle)
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    if abs(angle - math.pi) < 1e-6:
        diag = np.diag(matrix)
        axis = np.sqrt(np.maximum((diag + 1.0) * 0.5, 0.0)).astype(np.float64)
        if axis[0] > 1e-8:
            axis[1] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[0])
            axis[2] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[0])
        elif axis[1] > 1e-8:
            axis[0] = (matrix[0, 1] + matrix[1, 0]) / (4.0 * axis[1])
            axis[2] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[1])
        else:
            axis[0] = (matrix[0, 2] + matrix[2, 0]) / (4.0 * axis[2])
            axis[1] = (matrix[1, 2] + matrix[2, 1]) / (4.0 * axis[2])
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return angle * axis / norm
    axis = np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    ) / (2.0 * math.sin(angle))
    return axis * angle


def pose_to_rotvec(pose):
    """从 4x4 位姿矩阵提取 rotvec (轴角)"""
    pose = np.asarray(pose, dtype=np.float64).reshape(4, 4, order="F")
    R = pose[:3, :3]
    return matrix_to_rotvec(R)


def rotvec_to_rpy(rotvec):
    """将 rotvec 转换为 roll, pitch, yaw"""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.zeros(3, dtype=np.float64)
    axis = rotvec / angle
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    R = np.array([
        [t*axis[0]*axis[0] + c,   t*axis[0]*axis[1] - s*axis[2], t*axis[0]*axis[2] + s*axis[1]],
        [t*axis[0]*axis[1] + s*axis[2], t*axis[1]*axis[1] + c,   t*axis[1]*axis[2] - s*axis[0]],
        [t*axis[0]*axis[2] - s*axis[1], t*axis[1]*axis[2] + s*axis[0], t*axis[2]*axis[2] + c],
    ], dtype=np.float64)
    if np.abs(R[2, 0]) < 1:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        yaw = 0
        if R[2, 0] < 0:
            roll = np.arctan2(-R[1, 2], -R[1, 1])
            pitch = np.pi / 2
        else:
            roll = np.arctan2(R[1, 2], R[1, 1])
            pitch = -np.pi / 2
    return np.array([roll, pitch, yaw], dtype=np.float64)


def pose_error(current_pose, desired_pose):
    """计算位姿误差 (6,) = [pos_err(3), rot_err(3)]"""
    current = np.asarray(current_pose, dtype=np.float64).reshape(4, 4, order="F")
    desired = np.asarray(desired_pose, dtype=np.float64).reshape(4, 4, order="F")
    error = np.zeros(6, dtype=np.float64)
    error[:3] = current[:3, 3] - desired[:3, 3]
    error[3:] = matrix_to_rotvec(current[:3, :3] @ desired[:3, :3].T)
    return error


def xyz_from_pose(pose):
    """从 4x4 位姿中提取 xyz 位置"""
    return np.asarray(pose, dtype=np.float64)[[12, 13, 14]].copy()


def limit_torque_rate(tau_d, tau_j_d, dt):
    """限制力矩变化率，防止 torque discontinuity"""
    MAX_TORQUE_RATE = 1000.0
    tau_d = np.asarray(tau_d, dtype=np.float64)
    tau_j_d = np.asarray(tau_j_d, dtype=np.float64)
    dt = max(float(dt), 1e-3)
    max_delta = MAX_TORQUE_RATE * dt
    return tau_j_d + np.clip(tau_d - tau_j_d, -max_delta, max_delta)


# ==================================================================
# 键盘遥操作控制器（照搬 vla4desk 架构）
# ==================================================================

class KeyboardTeleop:
    """键盘遥操作控制器。

    架构（完全照搬 vla4desk key_control.py）：
      - pynput 回调直接写 keys_pressed（_keys_lock 保护）
      - 独立 10Hz 输入线程 _input_loop：读按键 → 算 action → 本地积分位姿
      - 主控制循环通过 get_latest_action() 取最新 action，应用后发力矩
    """

    def __init__(self):
        self.keys_pressed: set = set()
        self._keys_lock = threading.Lock()
        self.running = True
        self._stop_event = threading.Event()

        self.step_size = 1.0          # 速度倍率
        self._speed_levels = [0.4, 0.7, 1.0]
        self._speed_index = 2

        # 线程安全的状态共享（_state_lock 保护）
        self._state_lock = threading.Lock()
        self._latest_action = np.zeros(7, dtype=np.float64)       # [dx,dy,dz,drx,dry,drz,gripper]
        self._commanded_pose = np.zeros(6, dtype=np.float64)      # [x,y,z,rx,ry,rz] rotvec
        self._robot_state = np.zeros(8, dtype=np.float64)         # 供输入线程读取

        # 输入线程
        self._input_thread: threading.Thread | None = None
        self.listener: keyboard.Listener | None = None

    # ------------------------------------------------------------------
    # 键盘回调（照搬 vla4desk _on_key_press / _on_key_release）
    # ------------------------------------------------------------------

    def _on_key_press(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            char = key.name.lower()

        if char == 'esc':
            self.running = False
            return False

        with self._keys_lock:
            self.keys_pressed.add(char)

    def _on_key_release(self, key):
        try:
            char = key.char.lower()
        except AttributeError:
            char = key.name.lower()

        with self._keys_lock:
            self.keys_pressed.discard(char)

    # ------------------------------------------------------------------
    # 输入线程（照搬 vla4desk _input_loop）
    # ------------------------------------------------------------------

    def _get_keyboard_delta(self):
        """照搬 vla4desk _get_keyboard_delta：从 keys_pressed 计算增量"""
        with self._keys_lock:
            keys = set(self.keys_pressed)

        speed = self.step_size
        dx = (int('s' in keys) - int('w' in keys)) * MAX_DELTA_POS * speed
        dy = (int('d' in keys) - int('a' in keys)) * MAX_DELTA_POS * speed
        dz = (int('i' in keys) - int('k' in keys)) * MAX_DELTA_POS * speed
        droll = (int('q' in keys) - int('e' in keys)) * MAX_DELTA_ROT * speed
        dpitch = (int('u' in keys) - int('o' in keys)) * MAX_DELTA_ROT * speed
        dyaw = (int('l' in keys) - int('j' in keys)) * MAX_DELTA_ROT * speed
        return dx, dy, dz, droll, dpitch, dyaw

    def _apply_action_to_pose_array(self, pose: np.ndarray, action: np.ndarray) -> np.ndarray:
        """照搬 vla4desk _apply_action_to_pose_array：将 action 积分到 commanded pose"""
        next_pose = np.asarray(pose, dtype=np.float64).copy()
        delta = np.asarray(action[:6], dtype=np.float64)
        next_pose[:3] += delta[:3]

        angle = np.linalg.norm(delta[3:6])
        if angle > 1e-6:
            delta_rot = rotvec_to_matrix(delta[3:6])
            current_rot = rotvec_to_matrix(next_pose[3:6])
            next_pose[3:6] = matrix_to_rotvec(delta_rot @ current_rot)
        return next_pose

    def _input_loop(self):
        """10Hz 输入采样线程（照搬 vla4desk _input_loop）。

        只负责：
        - 从 keys_pressed 读取按键并计算当前拍 step delta
        - 转为速度指令（除以 INPUT_DT）供主循环每周期 × dt 消费
        - 本地积分 commanded_pose（避免依赖主循环位姿产生漂移）
        - 通过 _state_lock 把最新速度指令暴露给主控制循环
        """
        next_tick = time.perf_counter()
        while self.running and not self._stop_event.is_set():
            loop_start = time.perf_counter()

            # 从主循环读取最新机械臂状态
            with self._state_lock:
                commanded_pose = self._commanded_pose.copy()

            # 计算键盘增量（step delta，同 vla4desk）
            dx, dy, dz, droll, dpitch, dyaw = self._get_keyboard_delta()
            step_delta = np.array([dx, dy, dz, droll, dpitch, dyaw], dtype=np.float64)

            # 本地积分 commanded_pose
            next_commanded_pose = self._apply_action_to_pose_array(
                commanded_pose,
                np.append(step_delta, 0.0),
            )

            # 转为速度指令 (m/s, rad/s) 暴露给主循环，主循环每周期 × dt 得到增量
            velocity_action = np.append(step_delta / INPUT_DT, 0.0)

            # 发布最新速度指令和积分位姿
            with self._state_lock:
                self._latest_action = velocity_action
                self._commanded_pose = next_commanded_pose

            # 维持 10Hz 节拍
            next_tick += INPUT_DT
            now = time.perf_counter()
            sleep_time = next_tick - now
            if sleep_time > 0:
                time.sleep(sleep_time)
                continue

            # 超时后按当前时刻重置 deadline，避免长期累积漂移
            if now - loop_start > INPUT_DT:
                print(f"[输入线程] 超时 {((now - next_tick) * 1000):.1f}ms")
            next_tick = now

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self, initial_pose_4x4: np.ndarray):
        """启动输入线程。

        Args:
            initial_pose_4x4: 机器人当前位姿 (16,) flat column-major 4x4 矩阵
        """
        # 初始化 commanded_pose: 从 4x4 位姿提取 xyz + rotvec
        pose_4x4 = np.asarray(initial_pose_4x4, dtype=np.float64).reshape(4, 4, order="F")
        initial_commanded = np.zeros(6, dtype=np.float64)
        initial_commanded[:3] = pose_4x4[:3, 3]
        initial_commanded[3:6] = matrix_to_rotvec(pose_4x4[:3, :3])

        with self._state_lock:
            self._commanded_pose = initial_commanded
            self._latest_action = np.zeros(7, dtype=np.float64)

        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

        # 启动 pynput 监听
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self.listener.start()

    def stop(self):
        """停止控制器"""
        self.running = False
        self._stop_event.set()
        if self._input_thread is not None and self._input_thread.is_alive():
            self._input_thread.join(timeout=2.0)
        if self.listener is not None:
            self.listener.stop()

    def get_latest_action(self) -> np.ndarray:
        """主控制循环调用：获取最新 action (7,) [dx,dy,dz,drx,dry,drz,gripper]"""
        with self._state_lock:
            return self._latest_action.copy()

    def update_robot_state(self, state):
        """主控制循环调用：更新机械臂状态供输入线程参考"""
        pass  # 当前输入线程不需要机械臂状态，保留接口

    def print_controls(self):
        print("=" * 60)
        print("键盘遥操作 - 基于笛卡尔阻抗力矩控制")
        print("=" * 60)
        print("  W/S: X轴平移  A/D: Y轴平移  I/K: Z轴平移")
        print("  Q/E: Roll旋转  U/O: Pitch旋转  J/L: Yaw旋转")
        print("  ESC: 退出程序")
        print("=" * 60)


# ==================================================================
# 轨迹记录（保持不变）
# ==================================================================

def make_xyz_trace():
    """创建 6D 位姿轨迹记录 (xyz + rpy)"""
    return {
        "count": 0,
        "time": np.zeros(TRACE_CAPACITY, dtype=np.float64),
        "actual_pos": np.zeros((TRACE_CAPACITY, 3), dtype=np.float64),
        "actual_rpy": np.zeros((TRACE_CAPACITY, 3), dtype=np.float64),
        "target_pos": np.zeros((TRACE_CAPACITY, 3), dtype=np.float64),
        "target_rpy": np.zeros((TRACE_CAPACITY, 3), dtype=np.float64),
        "command_pos": np.zeros((TRACE_CAPACITY, 3), dtype=np.float64),
        "command_rpy": np.zeros((TRACE_CAPACITY, 3), dtype=np.float64),
    }


def append_xyz_trace(trace, sample_time, robot_state, target_pose, command_pose):
    """追加 6D 位姿轨迹记录"""
    index = trace["count"]
    if index >= TRACE_CAPACITY:
        return
    trace["time"][index] = float(sample_time)

    actual_rvec = rotvec_to_rpy(pose_to_rotvec(robot_state.O_T_EE))
    target_rvec = rotvec_to_rpy(pose_to_rotvec(target_pose))
    command_rvec = rotvec_to_rpy(pose_to_rotvec(command_pose))

    trace["actual_pos"][index] = xyz_from_pose(robot_state.O_T_EE)
    trace["actual_rpy"][index] = actual_rvec
    trace["target_pos"][index] = xyz_from_pose(target_pose)
    trace["target_rpy"][index] = target_rvec
    trace["command_pos"][index] = xyz_from_pose(command_pose)
    trace["command_rpy"][index] = command_rvec
    trace["count"] = index + 1


def xyz_trace_arrays(trace):
    """获取轨迹数组"""
    count = int(trace["count"])
    return (
        trace["time"][:count],
        trace["actual_pos"][:count],
        trace["actual_rpy"][:count],
        trace["target_pos"][:count],
        trace["target_rpy"][:count],
        trace["command_pos"][:count],
        trace["command_rpy"][:count],
    )


def save_xyz_trace_csv(trace, output_path):
    """保存为 CSV (6D: xyz + rpy)"""
    time_values, actual_pos, actual_rpy, target_pos, target_rpy, command_pos, command_rpy = xyz_trace_arrays(trace)
    if len(time_values) == 0:
        print("没有记录到坐标，跳过 CSV。")
        return
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(
            "time,"
            "actual_x,actual_y,actual_z,actual_roll,actual_pitch,actual_yaw,"
            "target_x,target_y,target_z,target_roll,target_pitch,target_yaw,"
            "command_x,command_y,command_z,command_roll,command_pitch,command_yaw,"
            "target_minus_actual_x,target_minus_actual_y,target_minus_actual_z,"
            "target_minus_actual_roll,target_minus_actual_pitch,target_minus_actual_yaw,"
            "command_minus_actual_x,command_minus_actual_y,command_minus_actual_z,"
            "command_minus_actual_roll,command_minus_actual_pitch,command_minus_actual_yaw\n"
        )
        for t, ap, ar, tp, tr, cp, cr in zip(time_values, actual_pos, actual_rpy, target_pos, target_rpy, command_pos, command_rpy):
            target_delta_pos = tp - ap
            target_delta_rpy = tr - ar
            command_delta_pos = cp - ap
            command_delta_rpy = cr - ar
            values = [t, *ap, *ar, *tp, *tr, *cp, *cr,
                      *target_delta_pos, *target_delta_rpy,
                      *command_delta_pos, *command_delta_rpy]
            output_file.write(",".join(f"{value:.9f}" for value in values) + "\n")
    print(f"6D 位姿 CSV 已保存到：{output_path}")


def save_xyz_trace_plot(trace, output_path):
    """保存为 SVG 图表 (6D: xyz + rpy，共6个面板)"""
    time_values, actual_pos, actual_rpy, target_pos, target_rpy, command_pos, command_rpy = xyz_trace_arrays(trace)
    if len(time_values) == 0:
        print("没有记录到坐标，跳过绘图。")
        return

    pos_series = [("实际", actual_pos, "#1f77b4"), ("目标", target_pos, "#ff7f0e"), ("命令", command_pos, "#2ca02c")]
    rpy_series = [("实际", actual_rpy, "#1f77b4"), ("目标", target_rpy, "#ff7f0e"), ("命令", command_rpy, "#2ca02c")]

    width = 1100
    panel_width = 340
    panel_height = 130
    panel_gap = 25
    margin_left = 80
    margin_top = 50
    margin_bottom = 40

    t_min = float(time_values[0])
    t_max = float(time_values[-1])
    if t_max <= t_min:
        t_max = t_min + 1e-6

    def scale_x(t, plot_left, plot_width):
        return plot_left + (float(t) - t_min) / (t_max - t_min) * plot_width

    def polyline(values, axis_index, scale_y, plot_left, plot_width):
        return " ".join(
            f"{scale_x(sample_time, plot_left, plot_width):.2f},{scale_y(value):.2f}"
            for sample_time, value in zip(time_values, values[:, axis_index])
        )

    svg = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{margin_top + 3 * panel_height + 3 * panel_gap + margin_bottom}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        "<text x=\"" + str(width // 2) + "\" y=\"28\" text-anchor=\"middle\" font-size=\"18\" font-family=\"sans-serif\">遥操作 6D 位姿</text>",
    ]

    pos_labels = ["x [m]", "y [m]", "z [m]"]
    for axis_index, axis_label in enumerate(pos_labels):
        plot_left = margin_left + axis_index * (panel_width + panel_gap)
        panel_top = margin_top
        panel_bottom = panel_top + panel_height

        values = np.concatenate([item[1][:, axis_index] for item in pos_series])
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        if y_max <= y_min:
            y_max = y_min + 1e-6
        padding = max((y_max - y_min) * 0.1, 1e-5)
        y_min -= padding
        y_max += padding

        def scale_y_pos(value):
            return panel_bottom - (float(value) - y_min) / (y_max - y_min) * panel_height

        svg.extend([
            f"<rect x=\"{plot_left}\" y=\"{panel_top}\" width=\"{panel_width}\" height=\"{panel_height}\" fill=\"#fafafa\" stroke=\"#cccccc\"/>",
            f"<line x1=\"{plot_left}\" y1=\"{panel_bottom}\" x2=\"{plot_left + panel_width}\" y2=\"{panel_bottom}\" stroke=\"#333333\"/>",
            f"<line x1=\"{plot_left}\" y1=\"{panel_top}\" x2=\"{plot_left}\" y2=\"{panel_bottom}\" stroke=\"#333333\"/>",
            f"<text x=\"{plot_left + panel_width // 2}\" y=\"{panel_top - 8}\" text-anchor=\"middle\" font-size=\"13\" font-family=\"sans-serif\">{axis_label}</text>",
        ])
        for _, data, color in pos_series:
            svg.append(f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"1.5\" points=\"{polyline(data, axis_index, scale_y_pos, plot_left, panel_width)}\"/>")

    rpy_labels = ["roll [rad]", "pitch [rad]", "yaw [rad]"]
    for axis_index, axis_label in enumerate(rpy_labels):
        plot_left = margin_left + axis_index * (panel_width + panel_gap)
        panel_top = margin_top + 2 * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height

        values = np.concatenate([item[1][:, axis_index] for item in rpy_series])
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        if y_max <= y_min:
            y_max = y_min + 1e-6
        padding = max((y_max - y_min) * 0.1, 1e-5)
        y_min -= padding
        y_max += padding

        def scale_y_rpy(value):
            return panel_bottom - (float(value) - y_min) / (y_max - y_min) * panel_height

        svg.extend([
            f"<rect x=\"{plot_left}\" y=\"{panel_top}\" width=\"{panel_width}\" height=\"{panel_height}\" fill=\"#fafafa\" stroke=\"#cccccc\"/>",
            f"<line x1=\"{plot_left}\" y1=\"{panel_bottom}\" x2=\"{plot_left + panel_width}\" y2=\"{panel_bottom}\" stroke=\"#333333\"/>",
            f"<line x1=\"{plot_left}\" y1=\"{panel_top}\" x2=\"{plot_left}\" y2=\"{panel_bottom}\" stroke=\"#333333\"/>",
            f"<text x=\"{plot_left + panel_width // 2}\" y=\"{panel_top - 8}\" text-anchor=\"middle\" font-size=\"13\" font-family=\"sans-serif\">{axis_label}</text>",
        ])
        for _, data, color in rpy_series:
            svg.append(f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"1.5\" points=\"{polyline(data, axis_index, scale_y_rpy, plot_left, panel_width)}\"/>")

    legend_y = margin_top + 3 * panel_height + 3 * panel_gap - 5
    legend_x = margin_left
    for label, _, color in pos_series:
        svg.append(f"<line x1=\"{legend_x}\" y1=\"{legend_y}\" x2=\"{legend_x + 25}\" y2=\"{legend_y}\" stroke=\"{color}\" stroke-width=\"2\"/>")
        svg.append(f"<text x=\"{legend_x + 32}\" y=\"{legend_y + 4}\" font-size=\"12\" font-family=\"sans-serif\">{label}</text>")
        legend_x += 70

    svg.append(f"<text x=\"{width // 2}\" y=\"{margin_top + 3 * panel_height + 3 * panel_gap + margin_bottom - 8}\" text-anchor=\"middle\" font-size=\"12\" font-family=\"sans-serif\">时间 [s]</text>")
    svg.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(svg))
    print(f"6D 位姿曲线已保存到：{output_path}")


# ==================================================================
# 打印
# ==================================================================

def print_status(robot_state, desired_pose, command_pose,
                 tau_task, coriolis, gravity, friction):
    """打印状态和力矩分量"""
    actual_xyz = xyz_from_pose(robot_state.O_T_EE)
    target_xyz = xyz_from_pose(desired_pose)
    command_xyz = xyz_from_pose(command_pose)

    print(f"\n状态采样 t={time.time():.1f}s")
    print(f"  实际xyz [m]: {np.array2string(actual_xyz, precision=6, suppress_small=True)}")
    print(f"  目标-实际 xyz [m]: {np.array2string(target_xyz - actual_xyz, precision=6, suppress_small=True)}")
    print(f"  命令-实际 xyz [m]: {np.array2string(command_xyz - actual_xyz, precision=6, suppress_small=True)}")
    print(f"  力矩分量 [Nm]:")
    print(f"    主控制 tau_task: {np.array2string(tau_task, precision=2, suppress_small=True)}")
    print(f"    科里奥利 coriolis: {np.array2string(coriolis, precision=2, suppress_small=True)}")
    print(f"    重力 gravity: {np.array2string(gravity, precision=2, suppress_small=True)}")
    print(f"    摩擦 friction: {np.array2string(friction, precision=2, suppress_small=True)}", flush=True)


# ==================================================================
# 主控制循环
# ==================================================================

def run_teleop(robot_ip):
    from pylibfranka import Robot, Torques

    robot = Robot(robot_ip)
    teleop = KeyboardTeleop()

    try:
        # 设置碰撞行为
        robot.set_collision_behavior(
            [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
            [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
            [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
            [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
        )

        teleop.print_controls()
        input("按 Enter 开始控制...")

        # 启动力矩控制
        active_control = robot.start_torque_control()
        model = robot.load_model()

        stiffness = DEFAULT_STIFFNESS
        damping = DEFAULT_DAMPING

        # 读取初始状态
        initial_state, _ = active_control.readOnce()
        initial_pose = np.asarray(initial_state.O_T_EE, dtype=np.float64).copy()

        # 启动键盘遥操作（输入线程 + pynput 监听）
        teleop.start(initial_pose)

        target_pose = initial_pose.copy()  # 目标位姿（4x4 平面数组）
        elapsed = 0.0
        next_print_time = 0.0

        # xyz 轨迹记录
        xyz_trace = make_xyz_trace()

        # 生成时间戳文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = f"teleop_{timestamp}.csv"
        plot_path = f"teleop_{timestamp}.svg"

        start_time = time.time()
        print(f"开始遥操作！按 ESC 退出（超时 {MAX_TELEOP_DURATION:.0f}s 自动退出）。")

        while teleop.running and elapsed < MAX_TELEOP_DURATION:
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()

            # 读取输入线程产出的最新速度指令 (m/s, rad/s)，× dt 得到增量
            action = teleop.get_latest_action()
            delta_xyz = action[:3] * dt
            delta_rotvec = action[3:6] * dt

            if np.any(delta_xyz) or np.linalg.norm(delta_rotvec) > 1e-12:
                target_pose = np.asarray(target_pose, dtype=np.float64).reshape(4, 4, order="F")
                target_pose[:3, 3] += delta_xyz
                if np.linalg.norm(delta_rotvec) > 1e-12:
                    delta_rot = np.eye(4, dtype=np.float64)
                    delta_rot[:3, :3] = rotvec_to_matrix(delta_rotvec)
                    target_pose = target_pose @ delta_rot
                target_pose = np.asarray(target_pose, dtype=np.float64).reshape(16, order="F")

            elapsed += dt

            # 记录 xyz 轨迹
            append_xyz_trace(xyz_trace, elapsed, robot_state, target_pose, target_pose)

            # 计算各力矩分量
            coriolis = np.asarray(model.coriolis(robot_state), dtype=np.float64)
            gravity = np.asarray(model.gravity(robot_state), dtype=np.float64)
            jacobian = np.asarray(model.zero_jacobian(robot_state), dtype=np.float64).reshape(6, 7, order="F")
            dq = np.asarray(robot_state.dq, dtype=np.float64)
            error = pose_error(robot_state.O_T_EE, target_pose)
            tau_task = jacobian.T @ (-stiffness @ error - damping @ (jacobian @ dq))
            tau_friction = np.zeros(7, dtype=np.float64)  # 仅用于打印

            # 1Hz 打印状态
            if elapsed >= next_print_time:
                print_status(robot_state, target_pose, target_pose,
                           tau_task, coriolis, gravity, tau_friction)
                next_print_time += PRINT_INTERVAL

            # 实际发送：tau_task + coriolis（重力/摩擦由控制器内置补偿）
            tau_d_raw = tau_task + coriolis
            # 限制力矩变化率，避免 torque discontinuity 报错
            tau_d = limit_torque_rate(tau_d_raw, robot_state.tau_J_d, dt)
            torque_command = Torques(tau_d.tolist())
            active_control.writeOnce(torque_command)

        print("正在停止...")

    except KeyboardInterrupt:
        print("正在停止...")
        teleop.running = False
    except Exception as e:
        print(f"错误: {e}")
        return 1
    finally:
        teleop.stop()
        try:
            robot.stop()
        except Exception:
            pass
        # 保存 6D 位姿轨迹
        if 'xyz_trace' in locals() and xyz_trace["count"] > 0:
            save_xyz_trace_csv(xyz_trace, csv_path)
            save_xyz_trace_plot(xyz_trace, plot_path)
        else:
            print("没有轨迹数据可保存。")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Franka Panda 键盘遥操作")
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()

    return run_teleop(args.ip)


if __name__ == "__main__":
    raise SystemExit(main())

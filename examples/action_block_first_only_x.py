#!/usr/bin/env python3

import argparse
import math
import queue
import threading
import time

import numpy as np



LOW_LEVEL_HZ = 1000.0
POLICY_HZ = 10.0
BLOCK_SIZE = 60
MOVE_STEPS = 30
STOP_STEPS = 10
X_STEP_METERS = 0.01
MAX_TORQUE_RATE = 1000.0  # Nm/s, equivalent to libfranka kMaxTorqueRate
PRINT_HZ = 5.0
PRINT_INTERVAL = 1.0 / PRINT_HZ
TRACE_CAPACITY = int((MOVE_STEPS + STOP_STEPS + 5) / POLICY_HZ * LOW_LEVEL_HZ)
CSV_PATH = "action_block_first_only_x_xyz.csv"
XYZ_PLOT_PATH = "action_block_first_only_x_xyz.svg"

# 原始阻抗控制参数（默认）
DEFAULT_STIFFNESS = np.diag([600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
DEFAULT_DAMPING_RATIO = 2.0
DEFAULT_DAMPING = np.diag([DEFAULT_DAMPING_RATIO * np.sqrt(DEFAULT_STIFFNESS[i, i]) for i in range(6)])

# 可选的摩擦力补偿参数
FRICTION_STATIC = np.array([2.0, 2.0, 1.5, 1.5, 1.0, 0.8, 0.6])  # 静摩擦 Nm
FRICTION_VISCOUS = np.array([0.5, 0.5, 0.4, 0.4, 0.3, 0.2, 0.1])  # 粘性摩擦 Nm/(rad/s)
FRICTION_DEADBAND = np.array([0.05, 0.05, 0.04, 0.04, 0.03, 0.02, 0.01])  # 静摩擦死区 rad/s

# 可选的积分控制参数
INTEGRAL_GAIN = np.diag([50.0, 50.0, 50.0, 10.0, 10.0, 10.0])
INTEGRAL_LIMIT = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])


def matrix_to_rotvec(matrix):
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


def min_jerk_weight(alpha):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def min_jerk_velocity_weight(alpha, duration):
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if duration <= 0.0:
        return 0.0
    return (30.0 * alpha**2 - 60.0 * alpha**3 + 30.0 * alpha**4) / duration


def make_action_sequence():
    move = [np.array([X_STEP_METERS, 0.0, 0.0], dtype=np.float64) for _ in range(MOVE_STEPS)]
    stop = [np.zeros(3, dtype=np.float64) for _ in range(STOP_STEPS)]
    return move + stop


class ActionBlockSource:
    """Upper-level policy that returns action blocks, not single actions."""

    def __init__(self):
        self._actions = make_action_sequence()
        self._cursor = 0

    @property
    def done(self):
        return self._cursor >= len(self._actions)

    def next_block(self):
        if self.done:
            first_action = np.zeros(3, dtype=np.float64)
            return np.tile(first_action, (BLOCK_SIZE, 1))

        remaining = self._actions[self._cursor : self._cursor + BLOCK_SIZE]
        self._cursor += 1
        if len(remaining) < BLOCK_SIZE:
            remaining = remaining + [np.zeros(3, dtype=np.float64)] * (BLOCK_SIZE - len(remaining))
        return np.vstack(remaining)


class FirstOnlyBlockFollower:
    """1 kHz lower-level follower. It is only allowed to inspect block[0]."""

    def __init__(self, initial_pose, action_source):
        self.action_source = action_source
        self.policy_period = 1.0 / POLICY_HZ
        self.next_policy_time = 0.0
        self.segment_start_time = 0.0
        self.command_pose = np.asarray(initial_pose, dtype=np.float64).copy()
        self.segment_start_pose = self.command_pose.copy()
        self.segment_target_pose = self.command_pose.copy()
        self.policy_tick = 0

    def update_policy_if_needed(self, elapsed):
        if elapsed + 1e-12 < self.next_policy_time:
            return None

        action_block = self.action_source.next_block()
        visible_action = action_block[0].copy()

        self.segment_start_time = elapsed
        self.segment_start_pose = self.segment_target_pose.copy()
        self.segment_target_pose = self.segment_start_pose.copy()
        self.segment_target_pose[12] += visible_action[0]
        self.segment_target_pose[13] += visible_action[1]
        self.segment_target_pose[14] += visible_action[2]

        event = {
            "policy_tick": self.policy_tick,
            "elapsed": elapsed,
            "visible_action": visible_action,
            "target_xyz": self.segment_target_pose[[12, 13, 14]].copy(),
        }
        self.policy_tick += 1
        self.next_policy_time += self.policy_period
        return event

    def step(self, elapsed):
        event = self.update_policy_if_needed(elapsed)
        alpha = (elapsed - self.segment_start_time) / self.policy_period
        weight = min_jerk_weight(alpha)
        self.command_pose = self.segment_start_pose + weight * (
            self.segment_target_pose - self.segment_start_pose
        )
        # 速度前馈：minimum-jerk 轨迹的一阶导数
        s_dot = min_jerk_velocity_weight(alpha, self.policy_period)
        delta_xyz = self.segment_target_pose[[12, 13, 14]] - self.segment_start_pose[[12, 13, 14]]
        desired_velocity = np.zeros(6, dtype=np.float64)
        desired_velocity[:3] = s_dot * delta_xyz
        return self.command_pose.copy(), desired_velocity, event

    def finished(self, elapsed):
        return self.action_source.done


def configure_collision_behavior(robot):
    robot.set_collision_behavior(
        [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
        [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
        [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
        [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
    )


def print_policy_event(event):
    action_mm = event["visible_action"] * 1000.0
    target = event["target_xyz"]
    print(
        f"10Hz tick {event['policy_tick']:02d} t={event['elapsed']:.3f}s "
        f"block[0]=[{action_mm[0]:.1f}, {action_mm[1]:.1f}, {action_mm[2]:.1f}] mm "
        f"target_xyz=[{target[0]:.6f}, {target[1]:.6f}, {target[2]:.6f}] m",
        flush=True,
    )


def xyz_from_pose(pose):
    return np.asarray(pose, dtype=np.float64)[[12, 13, 14]].copy()


def make_xyz_sample(robot_state, sample_time, target_pose, command_pose,
                    tau_task=None, coriolis=None, gravity=None, friction=None):
    actual_xyz = xyz_from_pose(robot_state.O_T_EE)
    target_xyz = xyz_from_pose(target_pose)
    command_xyz = xyz_from_pose(command_pose)
    return {
        "time": float(sample_time),
        "actual_xyz": actual_xyz,
        "target_delta_xyz": target_xyz - actual_xyz,
        "command_delta_xyz": command_xyz - actual_xyz,
        "tau_task": tau_task,
        "coriolis": coriolis,
        "gravity": gravity,
        "friction": friction,
    }


def print_xyz_sample(sample):
    print(f"\n5Hz状态采样，运动时间 t={sample['time']:.3f}s")
    print("  实际xyz [m]:", np.array2string(sample["actual_xyz"], precision=6, suppress_small=True))
    print("  目标-实际 xyz [m]:", np.array2string(sample["target_delta_xyz"], precision=6, suppress_small=True))
    print("  命令-实际 xyz [m]:", np.array2string(sample["command_delta_xyz"], precision=6, suppress_small=True), flush=True)
    if sample["tau_task"] is not None:
        print("  力矩分量 [Nm]:")
        print("    主控制 tau_task:", np.array2string(sample["tau_task"], precision=2, suppress_small=True))
        print("    科里奥利 coriolis:", np.array2string(sample["coriolis"], precision=2, suppress_small=True) if sample["coriolis"] is not None else " N/A")
        print("    重力 gravity:", np.array2string(sample["gravity"], precision=2, suppress_small=True) if sample["gravity"] is not None else " N/A")
        print("    摩擦 friction:", np.array2string(sample["friction"], precision=2, suppress_small=True) if sample["friction"] is not None else " N/A")


def print_worker(sample_queue, stop_event):
    while not stop_event.is_set() or not sample_queue.empty():
        try:
            sample = sample_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        print_xyz_sample(sample)
        sample_queue.task_done()


def make_xyz_trace():
    return {
        "count": 0,
        "time": np.zeros(TRACE_CAPACITY, dtype=np.float64),
        "actual": np.zeros((TRACE_CAPACITY, 3), dtype=np.float64),
        "target": np.zeros((TRACE_CAPACITY, 3), dtype=np.float64),
        "command": np.zeros((TRACE_CAPACITY, 3), dtype=np.float64),
    }


def append_xyz_trace(trace, sample_time, robot_state, target_pose, command_pose):
    index = trace["count"]
    if index >= TRACE_CAPACITY:
        return
    trace["time"][index] = float(sample_time)
    trace["actual"][index] = xyz_from_pose(robot_state.O_T_EE)
    trace["target"][index] = xyz_from_pose(target_pose)
    trace["command"][index] = xyz_from_pose(command_pose)
    trace["count"] = index + 1


def xyz_trace_arrays(trace):
    count = int(trace["count"])
    return (
        trace["time"][:count],
        trace["actual"][:count],
        trace["target"][:count],
        trace["command"][:count],
    )


def save_xyz_trace_csv(trace, output_path):
    time_values, actual, target, command = xyz_trace_arrays(trace)
    if len(time_values) == 0:
        print("没有记录到 xyz 坐标，跳过 CSV。")
        return
    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(
            "time,actual_x,actual_y,actual_z,target_x,target_y,target_z,"
            "command_x,command_y,command_z,target_minus_actual_x,target_minus_actual_y,"
            "target_minus_actual_z,command_minus_actual_x,command_minus_actual_y,command_minus_actual_z\n"
        )
        for t, a, target_xyz, command_xyz in zip(time_values, actual, target, command):
            target_delta = target_xyz - a
            command_delta = command_xyz - a
            values = [t, *a, *target_xyz, *command_xyz, *target_delta, *command_delta]
            output_file.write(",".join(f"{value:.9f}" for value in values) + "\n")
    print(f"xyz 坐标 CSV 已保存到：{output_path}")


def save_xyz_trace_plot(trace, output_path):
    time_values, actual, target, command = xyz_trace_arrays(trace)
    if len(time_values) == 0:
        print("没有记录到 xyz 坐标，跳过绘图。")
        return

    series = [("实际", actual, "#1f77b4"), ("目标", target, "#ff7f0e"), ("命令", command, "#2ca02c")]
    axis_labels = ["x", "y", "z"]
    width = 1100
    height = 820
    margin_left = 90
    margin_right = 30
    plot_width = width - margin_left - margin_right
    panel_height = 220
    panel_gap = 35
    margin_top = 55
    t_min = float(time_values[0])
    t_max = float(time_values[-1])
    if t_max <= t_min:
        t_max = t_min + 1e-6

    def scale_x(t):
        return margin_left + (float(t) - t_min) / (t_max - t_min) * plot_width

    def polyline(values, axis_index, scale_y):
        return " ".join(
            f"{scale_x(sample_time):.2f},{scale_y(value):.2f}"
            for sample_time, value in zip(time_values, values[:, axis_index])
        )

    svg = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        "<text x=\"550\" y=\"28\" text-anchor=\"middle\" font-size=\"20\" font-family=\"sans-serif\">action_block_first_only_x 末端 xyz</text>",
    ]
    for axis_index, axis_label in enumerate(axis_labels):
        panel_top = margin_top + axis_index * (panel_height + panel_gap)
        panel_bottom = panel_top + panel_height
        values = np.concatenate([item[1][:, axis_index] for item in series])
        y_min = float(np.min(values))
        y_max = float(np.max(values))
        if y_max <= y_min:
            y_max = y_min + 1e-6
        padding = max((y_max - y_min) * 0.08, 1e-5)
        y_min -= padding
        y_max += padding

        def scale_y(value):
            return panel_bottom - (float(value) - y_min) / (y_max - y_min) * panel_height

        svg.extend([
            f"<rect x=\"{margin_left}\" y=\"{panel_top}\" width=\"{plot_width}\" height=\"{panel_height}\" fill=\"#fafafa\" stroke=\"#cccccc\"/>",
            f"<line x1=\"{margin_left}\" y1=\"{panel_bottom}\" x2=\"{margin_left + plot_width}\" y2=\"{panel_bottom}\" stroke=\"#333333\"/>",
            f"<line x1=\"{margin_left}\" y1=\"{panel_top}\" x2=\"{margin_left}\" y2=\"{panel_bottom}\" stroke=\"#333333\"/>",
            f"<text x=\"18\" y=\"{panel_top + panel_height / 2:.1f}\" font-size=\"15\" font-family=\"sans-serif\">{axis_label} [m]</text>",
            f"<text x=\"{margin_left - 8}\" y=\"{panel_top + 5}\" text-anchor=\"end\" font-size=\"11\" font-family=\"monospace\">{y_max:.5f}</text>",
            f"<text x=\"{margin_left - 8}\" y=\"{panel_bottom}\" text-anchor=\"end\" font-size=\"11\" font-family=\"monospace\">{y_min:.5f}</text>",
        ])
        for _, data, color in series:
            svg.append(f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"1.6\" points=\"{polyline(data, axis_index, scale_y)}\"/>")

    legend_y = height - 25
    legend_x = margin_left
    for label, _, color in series:
        svg.append(f"<line x1=\"{legend_x}\" y1=\"{legend_y}\" x2=\"{legend_x + 28}\" y2=\"{legend_y}\" stroke=\"{color}\" stroke-width=\"2\"/>")
        svg.append(f"<text x=\"{legend_x + 36}\" y=\"{legend_y + 4}\" font-size=\"13\" font-family=\"sans-serif\">{label}</text>")
        legend_x += 95
    svg.append(f"<text x=\"{width / 2:.1f}\" y=\"{height - 5}\" text-anchor=\"middle\" font-size=\"13\" font-family=\"sans-serif\">运动时间 [s]</text>")
    svg.append("</svg>")

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(svg))
    print(f"xyz 坐标曲线已保存到：{output_path}")


def run_dry():
    initial_pose = np.eye(4, dtype=np.float64).reshape(16, order="F")
    follower = FirstOnlyBlockFollower(initial_pose, ActionBlockSource())
    elapsed = 0.0
    dt = 1.0 / LOW_LEVEL_HZ
    loop_count = 0

    while not follower.finished(elapsed):
        _, _, event = follower.step(elapsed)
        if event is not None:
            print_policy_event(event)
        elapsed += dt
        loop_count += 1

    final_x_mm = (follower.command_pose[12] - initial_pose[12]) * 1000.0
    print(f"dry-run done: {loop_count} low-level cycles, final x delta={final_x_mm:.1f} mm")


def pose_array_to_matrix(pose):
    return np.asarray(pose, dtype=np.float64).reshape(4, 4, order="F")


def matrix_to_pose_array(matrix):
    return np.asarray(matrix, dtype=np.float64).reshape(16, order="F")


def pose_error(current_pose, desired_pose):
    current = pose_array_to_matrix(current_pose)
    desired = pose_array_to_matrix(desired_pose)
    error = np.zeros(6, dtype=np.float64)
    error[:3] = current[:3, 3] - desired[:3, 3]
    error[3:] = matrix_to_rotvec(current[:3, :3] @ desired[:3, :3].T)
    return error


def limit_torque_rate(tau_d, tau_j_d, dt):
    tau_d = np.asarray(tau_d, dtype=np.float64)
    tau_j_d = np.asarray(tau_j_d, dtype=np.float64)
    dt = max(float(dt), 1e-3)
    max_delta = MAX_TORQUE_RATE * dt
    return tau_j_d + np.clip(tau_d - tau_j_d, -max_delta, max_delta)


def compute_friction_compensation(dq):
    """计算摩擦力补偿（静摩擦 + 粘性摩擦）"""
    dq = np.asarray(dq, dtype=np.float64)
    friction = np.zeros(7, dtype=np.float64)
    for i in range(7):
        abs_dq = abs(dq[i])
        if abs_dq > FRICTION_DEADBAND[i]:
            friction[i] = np.sign(dq[i]) * FRICTION_VISCOUS[i] * abs_dq
            if abs_dq < 0.1:
                friction[i] += np.sign(dq[i]) * FRICTION_STATIC[i]
    return friction


def compute_cartesian_impedance_torque(robot_state, model, desired_pose, desired_velocity, stiffness, damping, dt,
                                        use_gravity=False, use_friction=False, use_integral=False,
                                        integral_error=None):
    """笛卡尔阻抗控制，可选启用重力补偿、摩擦补偿、积分控制"""
    coriolis = np.asarray(model.coriolis(robot_state), dtype=np.float64)
    jacobian = np.asarray(model.zero_jacobian(robot_state), dtype=np.float64).reshape(6, 7, order="F")
    dq = np.asarray(robot_state.dq, dtype=np.float64)
    error = pose_error(robot_state.O_T_EE, desired_pose)

    # 比例-阻尼项（含速度前馈）
    tau_task = jacobian.T @ (-stiffness @ error + damping @ (desired_velocity - jacobian @ dq))
    tau_d = tau_task + coriolis

    # 可选：重力补偿
    if use_gravity:
        gravity = np.asarray(model.gravity(robot_state), dtype=np.float64)
        tau_d += gravity

    # 可选：积分项
    if use_integral and integral_error is not None:
        integral_error = integral_error + error * dt
        integral_error = np.clip(integral_error, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        tau_integral = jacobian.T @ (INTEGRAL_GAIN @ integral_error)
        tau_d += tau_integral

    # 可选：摩擦力补偿
    if use_friction:
        tau_friction = compute_friction_compensation(dq)
        tau_d += tau_friction

    return limit_torque_rate(tau_d, robot_state.tau_J_d, dt), integral_error if use_integral else None, error


def run_robot(robot_ip, use_gravity=False, use_friction=False, use_integral=False):
    from pylibfranka import Robot, Torques

    robot = Robot(robot_ip)
    try:
        configure_collision_behavior(robot)
        print("WARNING: This example will move the robot +300 mm along base-frame x.")
        print("This path uses torque control with Cartesian impedance.")
        print(f"Options: gravity_compensation={use_gravity}, friction_compensation={use_friction}, integral_control={use_integral}")
        print("The lower-level loop runs at the robot control rate and only uses action_block[0].")
        input("Press Enter to start, with the user stop button ready...")

        initial_state = robot.read_once()
        follower = FirstOnlyBlockFollower(initial_state.O_T_EE, ActionBlockSource())
        active_control = robot.start_torque_control()
        model = robot.load_model()

        stiffness = DEFAULT_STIFFNESS
        damping = DEFAULT_DAMPING
        integral_error = np.zeros(6, dtype=np.float64) if use_integral else None

        elapsed = 0.0
        motion_finished = False
        next_print_time = 0.0
        xyz_trace = make_xyz_trace()
        print_queue = queue.Queue(maxsize=32)
        print_stop_event = threading.Event()
        printer_thread = threading.Thread(target=print_worker, args=(print_queue, print_stop_event), daemon=True)
        printer_thread.start()

        while not motion_finished:
            robot_state, duration = active_control.readOnce()
            elapsed += duration.to_sec()

            desired_pose, desired_velocity, event = follower.step(elapsed)
            append_xyz_trace(xyz_trace, elapsed, robot_state, follower.segment_target_pose, desired_pose)

            # 分别计算各力矩分量（用于诊断打印，始终计算）
            coriolis = np.asarray(model.coriolis(robot_state), dtype=np.float64)
            gravity = np.asarray(model.gravity(robot_state), dtype=np.float64)  # 始终计算
            jacobian = np.asarray(model.zero_jacobian(robot_state), dtype=np.float64).reshape(6, 7, order="F")
            dq = np.asarray(robot_state.dq, dtype=np.float64)
            error = pose_error(robot_state.O_T_EE, desired_pose)
            tau_task = jacobian.T @ (-stiffness @ error + damping @ (desired_velocity - jacobian @ dq))
            tau_friction = compute_friction_compensation(dq)  # 始终计算

            if elapsed >= next_print_time:
                sample = make_xyz_sample(robot_state, elapsed, follower.segment_target_pose, desired_pose,
                                        tau_task=tau_task, coriolis=coriolis,
                                        gravity=gravity, friction=tau_friction)
                try:
                    print_queue.put_nowait(sample)
                except queue.Full:
                    pass
                next_print_time += PRINT_INTERVAL

            # 实际运行只用 tau_task + coriolis（不加重力和摩擦）
            tau_d = tau_task + coriolis
            torque_command = Torques(tau_d.tolist())
            if follower.finished(elapsed):
                torque_command.motion_finished = True
                motion_finished = True
            active_control.writeOnce(torque_command)

        print_stop_event.set()
        printer_thread.join(timeout=2.0)
        save_xyz_trace_csv(xyz_trace, CSV_PATH)
        save_xyz_trace_plot(xyz_trace, XYZ_PLOT_PATH)
        print("motion finished")
        return 0
    except Exception as exc:
        print(f"Error occurred: {exc}")
        if "print_stop_event" in locals():
            print_stop_event.set()
        if "printer_thread" in locals():
            printer_thread.join(timeout=2.0)
        if "xyz_trace" in locals() and xyz_trace["count"] > 0:
            save_xyz_trace_csv(xyz_trace, CSV_PATH)
            save_xyz_trace_plot(xyz_trace, XYZ_PLOT_PATH)
        robot.stop()
        return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually command the robot. Without this flag, only run the scheduler simulation.",
    )
    parser.add_argument(
        "--gravity",
        action="store_true",
        help="Enable gravity compensation (reduces static error from gravity load).",
    )
    parser.add_argument(
        "--friction",
        action="store_true",
        help="Enable friction compensation (reduces friction-induced error).",
    )
    parser.add_argument(
        "--integral",
        action="store_true",
        help="Enable integral control (eliminates steady-state error, may cause oscillation).",
    )
    args = parser.parse_args()

    if args.execute:
        return run_robot(args.ip, use_gravity=args.gravity, use_friction=args.friction, use_integral=args.integral)
    run_dry()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

import argparse

import numpy as np

from pylibfranka import CartesianPose, ControllerMode, JointPositions, Robot


HOME_JOINTS = [0.0, -np.pi / 4.0, 0.0, -3.0 * np.pi / 4.0, 0.0, np.pi / 2.0, np.pi / 4.0]
HOME_DURATION = 5.0
HOME_SETTLE_DURATION = 0.3
ACTION_RATE_HZ = 10.0
ACTION_INTERVAL = 1.0 / ACTION_RATE_HZ
ACTION_COUNT = 50
ACTION_DELTA_X = 0.005
EMPTY_ACTION_COUNT = 10
TOTAL_ACTION_COUNT = ACTION_COUNT + EMPTY_ACTION_COUNT
MOTION_DURATION = TOTAL_ACTION_COUNT * ACTION_INTERVAL
XYZ_PLOT_PATH = "cartesian_pose_key_xyz.svg"
INTERPOLATION_MODE = "block_smoothstep"  # "block_smoothstep" or "block_linear"
RECORD_INTERVAL = 1


def move_to_home(robot):
    active_control = robot.start_joint_position_control(ControllerMode.CartesianImpedance)
    initial_q = None
    time_elapsed = 0.0
    settle_elapsed = 0.0
    home_q = np.asarray(HOME_JOINTS, dtype=np.float64)

    while True:
        robot_state, duration = active_control.readOnce()
        dt = duration.to_sec()
        if initial_q is None:
            initial_q = np.asarray(robot_state.q, dtype=np.float64)

        time_elapsed += dt
        alpha = min(1.0, time_elapsed / HOME_DURATION)
        smooth = 0.5 - 0.5 * np.cos(np.pi * alpha)
        q = initial_q + (home_q - initial_q) * smooth

        if alpha >= 1.0:
            q = home_q
            settle_elapsed += dt

        joint_positions = JointPositions(q.tolist())
        if settle_elapsed >= HOME_SETTLE_DURATION:
            joint_positions.motion_finished = True
            active_control.writeOnce(joint_positions)
            break

        active_control.writeOnce(joint_positions)

    robot.stop()


def matrix4_from_state(value):
    return np.asarray(value, dtype=np.float64).reshape(4, 4, order="F")

def xyz_from_pose(pose):
    return np.asarray(pose, dtype=np.float64)[[12, 13, 14]].copy()


def record_force_contact_sample(robot_state, sample_time, target_pose, command_pose):
    return {
        "time": float(sample_time),
        "actual_xyz": xyz_from_pose(np.asarray(robot_state.O_T_EE, dtype=np.float64)),
        "target_xyz": xyz_from_pose(target_pose),
        "command_xyz": xyz_from_pose(command_pose),
    }

def print_force_contact_sample(sample):
    print(f"\n5Hz状态采样，运动时间 t={sample['time']:.3f}s")
    print("  实际xyz [m]:", np.array2string(sample["actual_xyz"], precision=6, suppress_small=True))
    print("  目标xyz [m]:", np.array2string(sample["target_xyz"], precision=6, suppress_small=True))
    print("  命令xyz [m]:", np.array2string(sample["command_xyz"], precision=6, suppress_small=True), flush=True)

def append_xyz_trace(trace, sample_time, robot_state, target_pose, command_pose):
    actual_pose = np.asarray(robot_state.O_T_EE, dtype=np.float64)
    trace["time"].append(float(sample_time))
    trace["actual"].append(xyz_from_pose(actual_pose))
    trace["target"].append(xyz_from_pose(target_pose))
    trace["command"].append(xyz_from_pose(command_pose))


def save_xyz_trace_plot(trace, output_path):
    if not trace["time"]:
        print("没有记录到 xyz 坐标，跳过绘图。")
        return
    time_values = np.asarray(trace["time"], dtype=np.float64)
    actual = np.asarray(trace["actual"], dtype=np.float64)
    target = np.asarray(trace["target"], dtype=np.float64)
    command = np.asarray(trace["command"], dtype=np.float64)
    if actual.ndim != 2 or target.ndim != 2 or command.ndim != 2 or actual.shape[1] != 3 or target.shape[1] != 3 or command.shape[1] != 3:
        print("xyz 记录不完整，跳过绘图。")
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
        points = []
        for sample_time, value in zip(time_values, values[:, axis_index]):
            points.append(f"{scale_x(sample_time):.2f},{scale_y(value):.2f}")
        return " ".join(points)

    svg = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        "<text x=\"550\" y=\"28\" text-anchor=\"middle\" font-size=\"20\" font-family=\"sans-serif\">5秒内末端 xyz 实时坐标</text>",
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
        for label, data, color in series:
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


def quintic_smoothstep(alpha):
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()

    # Connect to robot
    robot = Robot(args.ip)
    xyz_trace = {"time": [], "actual": [], "target": [], "command": []}

    try:
        # First move the robot to a suitable joint configuration
        print("警告：这个示例会移动机械臂！")
        print("请确认用户停止按钮在手边。")
        input("按 Enter 继续...")

        print("正在移动到 home 关节位姿...")
        move_to_home(robot)
        print("已完成 home 关节位姿复位。")

        # Start cartesian pose control with external control loop
        active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)

        time_elapsed = 0.0
        motion_finished = False
        settle_elapsed = 0.0
        next_record_time = 0.0
        if INTERPOLATION_MODE not in {"block_smoothstep", "block_linear"}:
            raise ValueError(f"未知插值模式：{INTERPOLATION_MODE}")

        robot_state, duration = active_control.readOnce()
        command_pose = np.asarray(robot_state.O_T_EE, dtype=np.float64).copy()
        target_pose = command_pose.copy()
        next_action_time = 0.0
        action_index = 0
        action_deltas = np.concatenate((
            np.full(ACTION_COUNT, ACTION_DELTA_X, dtype=np.float64),
            np.zeros(EMPTY_ACTION_COUNT, dtype=np.float64),
        ))
        block_interp_elapsed = ACTION_INTERVAL
        block_start_pose = command_pose.copy()
        block_target_pose = target_pose.copy()

        # External control loop
        while not motion_finished:
            # Read robot state and duration
            robot_state, duration = active_control.readOnce()

            dt = duration.to_sec()
            time_elapsed += dt

            if time_elapsed >= next_action_time and action_index < TOTAL_ACTION_COUNT:
                block_start_pose = command_pose.copy()
                block_target_pose = target_pose.copy()
                target_pose[12] += action_deltas[action_index]
                block_target_pose = target_pose.copy()
                block_interp_elapsed = 0.0
                action_index += 1
                next_action_time += ACTION_INTERVAL

            block_interp_elapsed = min(block_interp_elapsed + dt, ACTION_INTERVAL)
            alpha = block_interp_elapsed / ACTION_INTERVAL
            if INTERPOLATION_MODE == "block_smoothstep":
                ratio = quintic_smoothstep(alpha)
            else:
                ratio = alpha
            command_pose = block_start_pose + (block_target_pose - block_start_pose) * ratio

            if time_elapsed <= MOTION_DURATION:
                append_xyz_trace(xyz_trace, time_elapsed, robot_state, target_pose, command_pose)

            if action_index >= TOTAL_ACTION_COUNT and block_interp_elapsed >= ACTION_INTERVAL:
                settle_elapsed += dt

            if time_elapsed >= next_record_time and next_record_time <= MOTION_DURATION:
                sample = record_force_contact_sample(robot_state, time_elapsed, target_pose, command_pose)
                print_force_contact_sample(sample)
                next_record_time += RECORD_INTERVAL

            cartesian_pose = CartesianPose(command_pose.copy())

            if settle_elapsed >= ACTION_INTERVAL:
                cartesian_pose.motion_finished = True
                motion_finished = True
                print("主运动完成，正在结束示例。")

            active_control.writeOnce(cartesian_pose)
        save_xyz_trace_plot(xyz_trace, XYZ_PLOT_PATH)

    except Exception as e:
        print(f"发生错误：{e}")
        if robot is not None:
            robot.stop()
            if xyz_trace["time"]:
                save_xyz_trace_plot(xyz_trace, XYZ_PLOT_PATH)
        return -1


if __name__ == "__main__":
    main()

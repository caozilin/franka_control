#!/usr/bin/env python3

# Copyright (c) 2025 Franka Robotics GmbH
# Use of this source code is governed by the Apache-2.0 license, see LICENSE

import argparse
import os
import queue
import threading

import numpy as np

from pylibfranka import CartesianPose, ControllerMode, Robot

MOTION_DURATION = 5.0
RECORD_INTERVAL = 1.0
ENABLE_REALTIME_PRIORITY = True
REALTIME_PRIORITY = 80
CPU_AFFINITY = None  # e.g. {2}; None means keep current affinity

TRACE_SAMPLE_CAPACITY = int(MOTION_DURATION * 1200) + 200
def configure_process_timing():
    if CPU_AFFINITY is not None:
        try:
            os.sched_setaffinity(0, CPU_AFFINITY)
            print(f"已设置当前控制线程 CPU 亲和性：{sorted(CPU_AFFINITY)}")
        except Exception as exc:
            print(f"设置 CPU 亲和性失败：{exc}")

    if ENABLE_REALTIME_PRIORITY:
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(REALTIME_PRIORITY))
            print(f"已为当前控制线程启用实时调度 SCHED_FIFO，优先级 {REALTIME_PRIORITY}")
        except PermissionError:
            print("无法启用实时调度：需要 sudo 或 CAP_SYS_NICE 权限")
        except Exception as exc:
            print(f"启用实时调度失败：{exc}")

XYZ_PLOT_PATH = "cartesian_pose_origin_xyz.svg"


def xyz_from_pose(pose):
    return np.asarray(pose, dtype=np.float64)[[12, 13, 14]].copy()


def record_xyz_sample(robot_state, sample_time, target_pose, command_pose):
    actual_xyz = xyz_from_pose(np.asarray(robot_state.O_T_EE, dtype=np.float64))
    target_xyz = xyz_from_pose(target_pose)
    command_xyz = xyz_from_pose(command_pose)
    return {
        "time": float(sample_time),
        "actual_xyz": actual_xyz,
        "target_delta_xyz": target_xyz - actual_xyz,
        "command_delta_xyz": command_xyz - actual_xyz,
    }

def print_xyz_sample(sample):
    print(f"\n5Hz状态采样，运动时间 t={sample['time']:.3f}s")
    print("  实际xyz [m]:", np.array2string(sample["actual_xyz"], precision=6, suppress_small=True))
    print("  目标-实际 xyz [m]:", np.array2string(sample["target_delta_xyz"], precision=6, suppress_small=True))
    print("  命令-实际 xyz [m]:", np.array2string(sample["command_delta_xyz"], precision=6, suppress_small=True), flush=True)


def print_worker(sample_queue, stop_event):
    while not stop_event.is_set() or not sample_queue.empty():
        try:
            sample = sample_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        print_xyz_sample(sample)
        sample_queue.task_done()


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
        return " ".join(f"{scale_x(sample_time):.2f},{scale_y(value):.2f}" for sample_time, value in zip(time_values, values[:, axis_index]))

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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="172.16.0.2", help="Robot IP address")
    args = parser.parse_args()


    # Connect to robot
    robot = Robot(args.ip)

    try:
        # Set collision behavior
        lower_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        upper_torque_thresholds = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
        lower_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]
        upper_force_thresholds = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

        robot.set_collision_behavior(
            lower_torque_thresholds,
            upper_torque_thresholds,
            lower_force_thresholds,
            upper_force_thresholds,
        )

        # First move the robot to a suitable joint configuration
        print("WARNING: This example will move the robot!")
        print("Please make sure to have the user stop button at hand!")
        input("Press Enter to continue...")

        # Start cartesian pose control with external control loop
        active_control = robot.start_cartesian_pose_control(ControllerMode.JointImpedance)

        time_elapsed = 0.0
        motion_finished = False
        next_record_time = 0.0
        xyz_trace = {"time": [], "actual": [], "target": [], "command": []}
        print_queue = queue.Queue(maxsize=32)
        print_stop_event = threading.Event()
        printer_thread = threading.Thread(target=print_worker, args=(print_queue, print_stop_event), daemon=True)
        printer_thread.start()
        configure_process_timing()

        robot_state, duration = active_control.readOnce()
        initial_cartesian_pose = robot_state.O_T_EE

        # External control loop
        while not motion_finished:
            # Read robot state and duration
            robot_state, duration = active_control.readOnce()

            kRadius = 0.3
            angle = np.pi / 4 * (1 - np.cos(np.pi / 5.0 * time_elapsed))
            delta_x = kRadius * np.sin(angle)
            delta_z = kRadius * (np.cos(angle) - 1)

            # Update time
            time_elapsed += duration.to_sec()

            # Update joint positions
            new_cartesian_pose = initial_cartesian_pose.copy()
            new_cartesian_pose[12] += delta_x  # x position
            new_cartesian_pose[14] += delta_z  # z position

            # Set joint positions
            if time_elapsed <= MOTION_DURATION:
                append_xyz_trace(xyz_trace, time_elapsed, robot_state, new_cartesian_pose, new_cartesian_pose)


            if time_elapsed >= next_record_time and next_record_time <= MOTION_DURATION:
                sample = record_xyz_sample(robot_state, time_elapsed, new_cartesian_pose, new_cartesian_pose)
                try:
                    print_queue.put_nowait(sample)
                except queue.Full:
                    pass
                next_record_time += RECORD_INTERVAL

            cartesian_pose = CartesianPose(new_cartesian_pose)

            # Set motion_finished flag to True on the last update
            if time_elapsed >= MOTION_DURATION:
                cartesian_pose.motion_finished = True
                motion_finished = True
                print("Finished motion, shutting down example")

            # Send command to robot
            active_control.writeOnce(cartesian_pose)

        print_stop_event.set()
        printer_thread.join(timeout=2.0)
        save_xyz_trace_plot(xyz_trace, XYZ_PLOT_PATH)

    except Exception as e:
        print(f"Error occurred: {e}")
        if "print_stop_event" in locals():
            print_stop_event.set()
        if "printer_thread" in locals():
            printer_thread.join(timeout=2.0)
        if "xyz_trace" in locals() and xyz_trace["time"]:
            save_xyz_trace_plot(xyz_trace, XYZ_PLOT_PATH)
        if robot is not None:
            robot.stop()
        return -1


if __name__ == "__main__":
    main()

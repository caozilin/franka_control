from __future__ import annotations

import math
import multiprocessing as mp
import queue

import numpy as np
from pylibfranka import ControllerMode, JointPositions, Robot, Torques

from control.controllers import CubicController, LinearController, MinJerkController, MotionLimitedPoseController
from utils.control import (
    ActionConfig,
    GRIPPER_WIDTH_MAX,
    MAX_ROTATION_GOAL_ERROR,
    MAX_TRANSLATION_GOAL_ERROR,
    transform_action,
)
from utils.pose import (
    matrix_to_pose_array,
    matrix_to_rotvec_continuous,
    pose_array_to_matrix,
    rotvec_to_matrix,
)


ROBOT_IP = "172.16.0.2"

DEFAULT_HOME_Q = np.array(
    [0.0, -math.pi / 4.0, 0.0, -3.0 * math.pi / 4.0, 0.0, math.pi / 2.0, math.pi / 4.0],
    dtype=np.float64,
)

DEFAULT_STIFFNESS = np.diag([600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
DEFAULT_DAMPING = np.diag([2.0 * math.sqrt(DEFAULT_STIFFNESS[i, i]) for i in range(6)])
GRIPPER_SPEED = 0.08
GRIPPER_FORCE = 60.0
GRIPPER_CONNECT_TIMEOUT = 3.0


def _gripper_worker(robot_ip: str, command_queue, status_queue, speed: float, force: float) -> None:
    from pylibfranka import Gripper

    width_tolerance = 0.003
    gripper = None

    def drain_latest(first_target):
        target = first_target
        while True:
            try:
                latest = command_queue.get_nowait()
            except queue.Empty:
                return target
            if latest is None:
                return None
            target = latest

    def target_satisfied(target: float) -> bool:
        try:
            state = gripper.read_once()
        except Exception as exc:
            print(f"  [夹爪] read_once 异常: {exc}", flush=True)
            return False

        if target <= 1e-6:
            return bool(state.is_grasped) or float(state.width) <= width_tolerance
        return abs(float(state.width) - target) <= width_tolerance

    try:
        gripper = Gripper(robot_ip)
        status_queue.put((True, ""))

        while True:
            target = drain_latest(command_queue.get())
            if target is None:
                break

            target = float(np.clip(float(target), 0.0, GRIPPER_WIDTH_MAX))
            if target_satisfied(target):
                continue

            try:
                if target <= 1e-6:
                    ok = gripper.grasp(0.0, float(speed), float(force), 0.08, 0.08)
                else:
                    ok = gripper.move(target, float(speed))
                if ok is False:
                    print(f"  [夹爪] command({target:.3f}) 返回失败", flush=True)
            except Exception as exc:
                print(f"  [夹爪] command({target:.3f}) 异常: {exc}", flush=True)
    except Exception as exc:
        status_queue.put((False, str(exc)))
    finally:
        if gripper is not None:
            try:
                gripper.stop()
            except Exception:
                pass


CONTROLLERS = {
    "min_jerk": MinJerkController,
    "linear": LinearController,
    "cubic": CubicController,
    "motion_limited": MotionLimitedPoseController,
    "smooth_pose": MotionLimitedPoseController,
    "action_block": MinJerkController,
    "first_only": MinJerkController,
}


def _min_jerk_weight(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def clamp_norm(value: np.ndarray, limit: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= limit or norm < 1e-12:
        return value.copy()
    return value * (float(limit) / norm)


class FrankaEnv:
    """Minimal pylibfranka env: connect, reset, and run selectable torque control."""

    def __init__(
        self,
        robot_ip: str = ROBOT_IP,
        *,
        home_q: np.ndarray | None = None,
        reset_duration: float = 5.0,
        max_translation_step: float = 0.1,
        max_rotation_step: float = math.pi / 4.0,
        stiffness: np.ndarray | None = None,
        damping: np.ndarray | None = None,
        use_gripper: bool = True,
        gripper_speed: float = GRIPPER_SPEED,
        gripper_force: float = GRIPPER_FORCE,
    ):
        self.robot_ip = robot_ip
        self.robot = Robot(robot_ip)
        self.model = self.robot.load_model()
        self.home_q = np.asarray(home_q if home_q is not None else DEFAULT_HOME_Q, dtype=np.float64)
        self.reset_duration = float(reset_duration)
        self.action_config = ActionConfig(
            max_translation_step=float(max_translation_step),
            max_rotation_step=float(max_rotation_step),
        )
        self.stiffness = np.asarray(stiffness if stiffness is not None else DEFAULT_STIFFNESS, dtype=np.float64)
        self.damping = np.asarray(damping if damping is not None else DEFAULT_DAMPING, dtype=np.float64)
        self.action_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_requested = False
        self._last_gripper_target: float | None = GRIPPER_WIDTH_MAX
        self._gripper_enabled = False
        self._gripper_ctx = mp.get_context("spawn")
        self._gripper_command_queue = None
        self._gripper_status_queue = None
        self._gripper_process = None
        if use_gripper:
            self._start_gripper_worker(float(gripper_speed), float(gripper_force))
        self.configure_collision_behavior()


    def _start_gripper_worker(self, speed: float, force: float) -> None:
        self._gripper_command_queue = self._gripper_ctx.Queue(maxsize=1)
        self._gripper_status_queue = self._gripper_ctx.Queue(maxsize=1)
        self._gripper_process = self._gripper_ctx.Process(
            target=_gripper_worker,
            args=(self.robot_ip, self._gripper_command_queue, self._gripper_status_queue, speed, force),
            daemon=True,
        )
        self._gripper_process.start()
        try:
            ok, message = self._gripper_status_queue.get(timeout=GRIPPER_CONNECT_TIMEOUT)
        except queue.Empty:
            ok, message = False, "startup timeout"

        self._gripper_enabled = bool(ok)
        if self._gripper_enabled:
            print("  [夹爪] worker 已启动")
        else:
            print(f"  [夹爪] worker 启动失败: {message}")
            self._stop_gripper_worker()

    def _stop_gripper_worker(self) -> None:
        if self._gripper_process is None:
            return
        if self._gripper_command_queue is not None:
            try:
                self._gripper_command_queue.put_nowait(None)
            except Exception:
                pass
        self._gripper_process.join(timeout=2.0)
        if self._gripper_process.is_alive():
            self._gripper_process.terminate()
            self._gripper_process.join(timeout=1.0)
        self._gripper_process = None
        self._gripper_enabled = False

    def _set_gripper_target(self, target: float) -> None:
        target = float(np.clip(target, 0.0, GRIPPER_WIDTH_MAX))
        self._last_gripper_target = target
        if not self._gripper_enabled or self._gripper_command_queue is None:
            return
        try:
            while True:
                self._gripper_command_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._gripper_command_queue.put_nowait(target)
        except queue.Full:
            pass

    def configure_collision_behavior(self) -> None:
        self.robot.set_collision_behavior(
            [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
            [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
            [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
            [20.0, 20.0, 20.0, 25.0, 25.0, 25.0],
        )

    def reset(self) -> None:
        """Move the robot to home_q with pylibfranka joint position control."""
        active_control = self.robot.start_joint_position_control(ControllerMode.CartesianImpedance)
        elapsed = 0.0
        initial_q: np.ndarray | None = None

        while True:
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()
            elapsed += dt
            if initial_q is None:
                initial_q = np.asarray(robot_state.q_d if hasattr(robot_state, "q_d") else robot_state.q, dtype=np.float64)

            alpha = min(elapsed / self.reset_duration, 1.0)
            weight = _min_jerk_weight(alpha)
            q_cmd = initial_q + weight * (self.home_q - initial_q)
            command = JointPositions(q_cmd.tolist())
            if alpha >= 1.0:
                command.motion_finished = True
                active_control.writeOnce(command)
                break
            active_control.writeOnce(command)

    def enqueue_action_block(self, action_block: np.ndarray) -> None:
        block = np.asarray(action_block, dtype=np.float64)
        self.action_queue.put(block.copy())

    def request_stop(self) -> None:
        self._stop_requested = True

    def _pop_first_action(self) -> np.ndarray:
        try:
            action_block = self.action_queue.get_nowait()
        except queue.Empty:
            return np.zeros(7, dtype=np.float64)
        return action_block[0].copy()

    def _update_pose_goal(self, pose_goal: np.ndarray, transformed_action: np.ndarray, actual_pose: np.ndarray) -> np.ndarray:
        goal_matrix = pose_array_to_matrix(pose_goal)
        actual_matrix = pose_array_to_matrix(actual_pose)
        candidate = goal_matrix.copy()
        candidate[:3, 3] += transformed_action[:3]
        candidate[:3, :3] = rotvec_to_matrix(transformed_action[3:6]) @ goal_matrix[:3, :3]

        limited = candidate.copy()
        position_error = candidate[:3, 3] - actual_matrix[:3, 3]
        limited[:3, 3] = actual_matrix[:3, 3] + clamp_norm(position_error, MAX_TRANSLATION_GOAL_ERROR)

        previous_rotation_error = getattr(self, "_last_goal_rotation_error", None)
        rotation_error = matrix_to_rotvec_continuous(candidate[:3, :3] @ actual_matrix[:3, :3].T, previous_rotation_error)
        self._last_goal_rotation_error = rotation_error.copy()
        limited_rotation_error = clamp_norm(rotation_error, MAX_ROTATION_GOAL_ERROR)
        limited[:3, :3] = rotvec_to_matrix(limited_rotation_error) @ actual_matrix[:3, :3]
        return matrix_to_pose_array(limited)

    def run_action_loop(
        self,
        *,
        max_duration: float | None = None,
        print_events: bool = True,
        action_source=None,
        interpolation: str | None = None,
        controller_name: str = "min_jerk",
        trace_callback=None,
    ) -> None:
        """Run 1kHz torque loop and consume action blocks at 10Hz.

        At every 10Hz tick the loop either samples action_source() directly,
        or pops one action block in FIFO order and executes only block[0].
        Empty queue means zero action. controller_name is "min_jerk", "linear", "cubic", or "motion_limited".
        """
        if interpolation is not None:
            controller_name = interpolation
        self._stop_requested = False
        try:
            controller_cls = CONTROLLERS[controller_name]
        except KeyError as exc:
            choices = ", ".join(sorted(CONTROLLERS))
            raise ValueError(f"unsupported controller: {controller_name}; choices: {choices}") from exc

        initial_state = self.robot.read_once()
        pose_goal = np.asarray(initial_state.O_T_EE, dtype=np.float64).copy()
        controller = controller_cls(
            initial_state.O_T_EE,
            stiffness=self.stiffness,
            damping=self.damping,
        )
        active_control = self.robot.start_torque_control()

        elapsed = 0.0
        self._last_goal_rotation_error = np.zeros(3, dtype=np.float64)
        while True:
            robot_state, duration = active_control.readOnce()
            dt = duration.to_sec()
            elapsed += dt

            if elapsed + 1e-12 >= controller.next_policy_time:
                action = np.asarray(action_source(), dtype=np.float64) if action_source is not None else self._pop_first_action()
                transformed_action = transform_action(action, self.action_config)
                pose_goal = self._update_pose_goal(pose_goal, transformed_action, robot_state.O_T_EE)
                event = controller.update_goal(elapsed, pose_goal, transformed_action)
                self._set_gripper_target(transformed_action[6])
                if print_events:
                    action_mm = event["transformed_action"][:3] * 1000.0
                    action_rot = event["transformed_action"][3:6]
                    target = event["target_xyz"]
                    print(
                        f"10Hz tick {event['policy_tick']:03d} t={event['elapsed']:.3f}s "
                        f"dxyz_mm=[{action_mm[0]:.1f}, {action_mm[1]:.1f}, {action_mm[2]:.1f}] "
                        f"drot=[{action_rot[0]:.4f}, {action_rot[1]:.4f}, {action_rot[2]:.4f}] "
                        f"target_xyz=[{target[0]:.6f}, {target[1]:.6f}, {target[2]:.6f}]",
                    )

            tau_d = controller.step(robot_state, self.model, pose_goal, elapsed, dt)
            if trace_callback is not None:
                trace_callback(
                    {
                        "time": float(elapsed),
                        "goal_pose": getattr(controller, "goal_pose", pose_goal).copy(),
                        "ref_pose": controller.command_pose.copy(),
                        "actual_pose": np.asarray(robot_state.O_T_EE, dtype=np.float64).copy(),
                        "controller": controller_name,
                    }
                )
            command = Torques(tau_d.tolist())

            if self._stop_requested or (max_duration is not None and elapsed >= max_duration):
                command.motion_finished = True
                self._stop_requested = True
                active_control.writeOnce(command)
                break
            active_control.writeOnce(command)

    def stop(self) -> None:
        self._stop_requested = True
        self._stop_gripper_worker()
        self.robot.stop()


"""Realtime OpenPI/Cosmos coordinator for franka_control.

This script provides a FastAPI control panel, captures observations, requests
policy actions over the existing websocket client, and queues actions into the
Franka torque loop. Replanning is synchronous: the next inference request is
issued only after the current local action plan has been consumed.
"""
from __future__ import annotations

import asyncio
import collections
import dataclasses
import enum
import json
import logging
import pathlib
import sys
import threading
import time
from typing import Any

import imageio
import numpy as np
import tyro
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from client.websocket_client_policy import WebsocketClientPolicy
from orchestration import ActionPlanScheduler, RTCConfig
from recording import RealtimeTimingProfiler
from control.contracts import ControlRates, PolicyActionSpec
from control.franka_env import DEFAULT_CAM1_SERIAL, DEFAULT_CAM2_SERIAL, FrankaEnv, ROBOT_IP
from client.realtime_utils import (
    coerce_rgb_frame,
    encode_jpeg_b64,
    extract_first_present,
    jsonable,
    shape_list,
)
from utils.control import transform_action

LOGS_BASE_DIR = pathlib.Path(__file__).resolve().parents[1] / "logs"
STATIC_DIR_BASE = ROOT / "src" / "static"
STREAM_DT = 0.05

logger = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    HOMING = "homing"


@dataclasses.dataclass
class Args:
    host: str = "100.96.2.67"
    port: int = 8000
    web_port: int = 8080
    prompt: str = "put the yellow cube on the red plate"
    log_subdir: str = ""
    policy_type: str = "cosmos"
    replan_steps: int | None = None
    api_key: str | None = None
    robot_ip: str = ROBOT_IP
    cam1_serial: str | None = DEFAULT_CAM1_SERIAL
    cam2_serial: str | None = DEFAULT_CAM2_SERIAL
    no_robot: bool = False
    no_cameras: bool = False
    use_gripper: bool = True
    save_recording: bool = False
    startup_home: bool = True
    control_hz: float = 10.0
    policy_translation_scale_m: float = 0.01
    policy_rotation_scale_rad: float = 0.01
    reference: str | None = None
    reference_name: str = "min_jerk"
    nullspace_enabled: bool = False
    nullspace_q_target: tuple[float, float, float, float, float, float, float] | None = None
    nullspace_stiffness: float = 10.0
    nullspace_damping: float = 2.0
    nullspace_pinv: str = "plain"
    nullspace_projector: str = "kinematic"
    nullspace_lambda: float = 0.05
    step_warn_ms: float = 20.0
    rtc_enabled: bool = False
    rtc_execution_horizon: int = 5
    rtc_inference_delay: int = 3

    def __post_init__(self) -> None:
        self.policy_type = self.policy_type.lower().strip()
        if self.policy_type not in {"openpi", "cosmos"}:
            raise ValueError("policy_type must be 'openpi' or 'cosmos'")
        if not str(self.log_subdir or "").strip():
            self.log_subdir = self.policy_type
        if self.reference is not None:
            self.reference_name = str(self.reference)
        self.reference_name = self.reference_name.lower().strip()
        if self.reference_name not in {"min_jerk", "linear", "cubic", "motion_limited"}:
            raise ValueError("reference must be one of 'min_jerk', 'linear', 'cubic', or 'motion_limited'")
        self.nullspace_pinv = self.nullspace_pinv.lower().strip()
        if self.nullspace_pinv not in {"plain", "damped"}:
            raise ValueError("nullspace_pinv must be 'plain' or 'damped'")
        self.nullspace_projector = self.nullspace_projector.lower().strip()
        if self.nullspace_projector not in {"kinematic", "dynamic"}:
            raise ValueError("nullspace_projector must be 'kinematic' or 'dynamic'")
        if self.replan_steps is None:
            self.replan_steps = 16 if self.policy_type == "cosmos" else 5
        self.rtc_execution_horizon = max(1, int(self.rtc_execution_horizon))
        self.rtc_inference_delay = max(0, int(self.rtc_inference_delay))
        ControlRates(policy_hz=float(self.control_hz), planner_hz=float(self.control_hz))


class Coordinator:
    def __init__(self, args: Args):
        self._args = args
        self._policy_action_spec = PolicyActionSpec(
            translation_scale_m=args.policy_translation_scale_m,
            rotation_scale_rad=args.policy_rotation_scale_rad,
        )
        self._state = State.IDLE
        self._state_lock = threading.Lock()
        self._prompt = args.prompt
        self._prompt_lock = threading.Lock()
        self._log_subdir = normalize_log_subdir(args.log_subdir)
        self._log_config_lock = threading.Lock()

        self._action_scheduler = ActionPlanScheduler(
            int(args.replan_steps or 1),
            RTCConfig(
                enabled=args.policy_type == "openpi" and bool(args.rtc_enabled),
                execution_horizon=args.rtc_execution_horizon,
                inference_delay=args.rtc_inference_delay,
            ),
        )
        self._client_lock = threading.Lock()
        self._client: WebsocketClientPolicy | None = None
        self._infer_lock = threading.Lock()
        self._infer_thread: threading.Thread | None = None
        self._infer_running = False
        self._infer_generation = 0

        self._recording = False
        self._record_frames1: list[np.ndarray] = []
        self._record_frames2: list[np.ndarray] = []
        self._record_frames3: list[np.ndarray] = []
        self._record_frames4: list[np.ndarray] = []
        self._telemetry_log: collections.deque[dict[str, Any]] = collections.deque(maxlen=10800)
        self._session_dir: pathlib.Path | None = None
        self._record_start_time = 0.0
        self._record_lock = threading.Lock()

        self._latest_img1: np.ndarray | None = None
        self._latest_img2: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        self._latest_state: list | None = None
        self._latest_joints: list | None = None
        self._latest_target_pose: list | None = None
        self._latest_action: list | None = None
        self._latest_action_transformed: list | None = None
        self._latest_infer_ms: float | None = None
        self._latest_total_ms: float | None = None
        self._latest_ee_force_torque: list | None = None
        self._latest_value_prediction: float | None = None
        self._latest_proprio: Any = None
        self._latest_future_proprio: Any = None
        self._latest_future_wrist: np.ndarray | None = None
        self._latest_future_primary: np.ndarray | None = None
        self._latest_infer_keys: list[str] | None = None
        self._latest_future_wrist_shape: list[int] | None = None
        self._latest_future_primary_shape: list[int] | None = None
        self._telemetry_lock = threading.Lock()

        self._ws_clients: set[WebSocket] = set()
        self._ws_lock = asyncio.Lock()

        self._env = FrankaEnv(
            robot_ip=args.robot_ip,
            cam1_serial=args.cam1_serial,
            cam2_serial=args.cam2_serial,
            no_robot=args.no_robot,
            no_cameras=args.no_cameras,
            use_gripper=args.use_gripper,
            control_hz=args.control_hz,
            save_recording=args.save_recording,
            reference_name=args.reference_name,
            nullspace_enabled=args.nullspace_enabled,
            nullspace_q_target=None if args.nullspace_q_target is None else np.asarray(args.nullspace_q_target, dtype=np.float64),
            nullspace_stiffness=args.nullspace_stiffness,
            nullspace_damping=args.nullspace_damping,
            nullspace_pinv=args.nullspace_pinv,
            nullspace_projector=args.nullspace_projector,
            nullspace_lambda=args.nullspace_lambda,
        )
        if args.startup_home and not args.no_robot:
            logger.info("Startup homing Franka before serving UI")
            self._env.reset()
        self._control_timing_profiler: RealtimeTimingProfiler | None = None
        self._connect_client()

    @property
    def state(self) -> State:
        with self._state_lock:
            return self._state

    @property
    def log_subdir(self) -> str:
        with self._log_config_lock:
            return self._log_subdir

    def _connect_client(self) -> None:
        try:
            self._client = WebsocketClientPolicy(self._args.host, self._args.port, self._args.api_key)
        except Exception as exc:
            logger.warning("推理服务连接失败，推理功能暂不可用: %s", exc)
            self._client = None

    def _reset_policy_client(self, reason: str) -> None:
        if self._client is None:
            self._connect_client()
            return
        try:
            logger.info("Resetting inference client: %s", reason)
            with self._client_lock:
                self._client.reset()
        except Exception as exc:
            logger.warning("推理客户端 reset 失败: %s", exc)
            self._client = None

    def cmd_start(self) -> None:
        with self._state_lock:
            if self._state == State.IDLE:
                self._action_scheduler.clear()
                self._bump_infer_generation()
                self._state = State.RUNNING
                logger.info("State -> RUNNING")

    def cmd_stop(self) -> None:
        with self._state_lock:
            if self._state == State.RUNNING:
                self._action_scheduler.clear()
                self._bump_infer_generation()
                self._env.clear_actions()
                self._env.stop_control()
                self._state = State.IDLE
                logger.info("State -> IDLE")
        self._clear_action_runtime()

    def cmd_home(self) -> None:
        with self._state_lock:
            self._action_scheduler.clear()
            self._bump_infer_generation()
            self._env.clear_actions()
            self._state = State.HOMING
            logger.info("State -> HOMING")
        self._clear_action_runtime()
        self._reset_policy_client("home")

    def cmd_set_prompt(self, prompt: str) -> None:
        with self._prompt_lock:
            self._prompt = str(prompt)
        self._action_scheduler.clear()
        self._bump_infer_generation()

    def cmd_set_log_subdir(self, log_subdir: str) -> str:
        normalized = normalize_log_subdir(log_subdir)
        with self._log_config_lock:
            self._log_subdir = normalized
        return normalized

    def _clear_action_runtime(self) -> None:
        with self._telemetry_lock:
            self._latest_action = None
            self._latest_action_transformed = None
            self._latest_infer_ms = None
            self._latest_total_ms = None

    def _bump_infer_generation(self) -> None:
        with self._infer_lock:
            self._infer_generation += 1

    def _current_infer_generation(self) -> int:
        with self._infer_lock:
            return self._infer_generation

    def _set_infer_running(self, running: bool) -> None:
        with self._infer_lock:
            self._infer_running = bool(running)

    def _is_infer_running(self) -> bool:
        with self._infer_lock:
            return self._infer_running

    def _allocate_session_dir(self) -> pathlib.Path:
        log_subdir = self.log_subdir
        if not log_subdir:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for idx in range(100):
                suffix = "" if idx == 0 else f"_{idx:02d}"
                session_dir = LOGS_BASE_DIR / f"{stamp}{suffix}"
                try:
                    session_dir.mkdir(parents=True, exist_ok=False)
                    return session_dir
                except FileExistsError:
                    continue
            session_dir = LOGS_BASE_DIR / f"{stamp}_{time.time_ns()}"
            session_dir.mkdir(parents=True, exist_ok=False)
            return session_dir

        base_dir = LOGS_BASE_DIR / log_subdir
        base_dir.mkdir(parents=True, exist_ok=True)
        nums = []
        for path in base_dir.iterdir():
            if path.is_dir() and path.name.startswith("epo_"):
                try:
                    nums.append(int(path.name[4:]))
                except ValueError:
                    pass
        next_epo = max(nums) + 1 if nums else 1
        while True:
            session_dir = base_dir / f"epo_{next_epo}"
            try:
                session_dir.mkdir(parents=True, exist_ok=False)
                return session_dir
            except FileExistsError:
                next_epo += 1

    def _start_session_logging(self) -> None:
        with self._record_lock:
            self._record_frames1.clear()
            self._record_frames2.clear()
            self._record_frames3.clear()
            self._record_frames4.clear()
            self._telemetry_log.clear()
            self._session_dir = self._allocate_session_dir()
            self._recording = True
            self._record_start_time = time.time()
            session_dir = self._session_dir
        if session_dir is not None:
            self._write_session_metadata(session_dir, status="recording")
        logger.info("Auto-recording started: %s", self._session_dir)

    def _stop_session_logging(self) -> None:
        with self._record_lock:
            self._recording = False
            frames1 = self._record_frames1[:]
            frames2 = self._record_frames2[:]
            frames3 = self._record_frames3[:]
            frames4 = self._record_frames4[:]
            tele_log = list(self._telemetry_log)
            self._record_frames1.clear()
            self._record_frames2.clear()
            self._record_frames3.clear()
            self._record_frames4.clear()
            self._telemetry_log.clear()
            session_dir = self._session_dir
            self._session_dir = None

        if session_dir is None:
            logger.warning("Session logging stopped without an allocated session directory")
            return
        save_thread = threading.Thread(
            target=self._save_session_logging,
            args=(session_dir, frames1, frames2, frames3, frames4, tele_log),
            daemon=False,
        )
        save_thread.start()
        logger.info("Session save queued: %s frames=%d telemetry=%d", session_dir, len(frames1), len(tele_log))

    def _write_session_metadata(self, session_dir: pathlib.Path, *, status: str) -> None:
        metadata = {
            "status": status,
            "policy_type": self._args.policy_type,
            "prompt": self._prompt,
            "log_subdir": self.log_subdir,
            "control_hz": self._args.control_hz,
            "reference_name": self._args.reference_name,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(session_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def _save_session_logging(
        self,
        session_dir: pathlib.Path,
        frames1: list[np.ndarray],
        frames2: list[np.ndarray],
        frames3: list[np.ndarray],
        frames4: list[np.ndarray],
        tele_log: list[dict[str, Any]],
    ) -> None:
        try:
            self._write_session_metadata(session_dir, status="saved")
            with open(session_dir / "telemetry.jsonl", "w", encoding="utf-8") as f:
                for entry in tele_log:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fps = max(1.0, float(self._args.control_hz))
            if frames1:
                imageio.mimwrite(str(session_dir / "cam1.mp4"), frames1, fps=fps, codec="libx264", pixelformat="yuv420p")
            if frames2:
                imageio.mimwrite(str(session_dir / "cam2.mp4"), frames2, fps=fps, codec="libx264", pixelformat="yuv420p")
            if frames3:
                imageio.mimwrite(str(session_dir / "cam3.mp4"), frames3, fps=fps, codec="libx264", pixelformat="yuv420p")
            if frames4:
                imageio.mimwrite(str(session_dir / "cam4.mp4"), frames4, fps=fps, codec="libx264", pixelformat="yuv420p")
            logger.info("Session saved to %s", session_dir)
        except Exception as exc:
            logger.exception("Failed to save session %s: %s", session_dir, exc)

    def run_control_loop(self) -> None:
        dt = 1.0 / max(1e-6, float(self._args.control_hz))
        last_state = State.IDLE
        next_tick = time.perf_counter()
        while True:
            with self._state_lock:
                current_state = self._state
            self._sync_recording_state(current_state, last_state)
            last_state = current_state
            try:
                self._step(current_state)
            except Exception:
                logger.exception("Control loop error")
            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    def _sync_recording_state(self, current_state: State, last_state: State) -> None:
        if current_state == State.RUNNING and last_state != State.RUNNING:
            self._start_session_logging()
        elif current_state != State.RUNNING and last_state == State.RUNNING:
            self._stop_session_logging()

    def _step(self, current_state: State) -> None:
        step_start = time.perf_counter()
        with self._prompt_lock:
            prompt = self._prompt
        obs_start = time.perf_counter()
        obs, img1_raw, img2_raw = self._env.get_observation(prompt)
        obs_ms = (time.perf_counter() - obs_start) * 1000.0
        with self._frame_lock:
            self._latest_img1 = img1_raw
            self._latest_img2 = img2_raw
        telemetry_start = time.perf_counter()
        self._update_latest_telemetry(obs)
        telemetry_ms = (time.perf_counter() - telemetry_start) * 1000.0
        record_start = time.perf_counter()
        self._record_frames(img1_raw, img2_raw)
        self._record_tick_telemetry()
        record_ms = (time.perf_counter() - record_start) * 1000.0

        action_ms = 0.0
        if current_state == State.RUNNING:
            action_start = time.perf_counter()
            self._run_action_step(obs)
            action_ms = (time.perf_counter() - action_start) * 1000.0
        elif current_state == State.HOMING:
            self._env.stop_control()
            self._env.clear_actions()
            self._env.reset()
            self._clear_action_runtime()
            with self._state_lock:
                self._state = State.IDLE
            logger.info("Homing complete, State -> IDLE")
        else:
            if self._env.is_control_running():
                self._env.stop_control()
            self._env.clear_actions()

        total_ms = (time.perf_counter() - step_start) * 1000.0
        if total_ms >= float(self._args.step_warn_ms):
            logger.warning(
                "Coordinator step slow: state=%s total=%.1fms obs=%.1fms telemetry=%.1fms record=%.1fms action=%.1fms",
                current_state.value,
                total_ms,
                obs_ms,
                telemetry_ms,
                record_ms,
                action_ms,
            )

    def _update_latest_telemetry(self, obs: dict) -> None:
        with self._telemetry_lock:
            self._latest_state = obs["observation/state"].tolist()
            self._latest_joints = obs.get("observation/joints", np.zeros(7)).tolist()
            self._latest_target_pose = self._env.commanded_pose_array.tolist()
            self._latest_ee_force_torque = self._env.ee_force_torque.tolist()

    def _record_frames(self, img1_raw: np.ndarray, img2_raw: np.ndarray) -> None:
        with self._record_lock:
            if not self._recording:
                return
            self._record_frames1.append(img1_raw.copy())
            self._record_frames2.append(img2_raw.copy())
            if self._args.policy_type == "cosmos":
                with self._telemetry_lock:
                    fw = self._latest_future_wrist
                    fp = self._latest_future_primary
                if fw is not None:
                    self._record_frames3.append(np.asarray(fw).copy())
                if fp is not None:
                    self._record_frames4.append(np.asarray(fp).copy())

    def _run_action_step(self, obs: dict) -> None:
        if self._env.robot is not None and not self._env.is_control_running():
            try:
                self._env.check_control_error()
            except Exception:
                logger.exception("Franka torque control exited unexpectedly; State -> IDLE")
                self._action_scheduler.clear()
                self._clear_action_runtime()
                with self._state_lock:
                    self._state = State.IDLE
                return
            logger.info("Starting Franka torque control on RUNNING state")
            self._control_timing_profiler = RealtimeTimingProfiler(capacity=12000)
            self._env.start_control(
                home_first=False,
                reference_name=self._args.reference_name,
                timing_profiler=self._control_timing_profiler,
            )
            return

        if self._action_scheduler.should_prefetch():
            self._request_infer_async(obs)

        action = self._action_scheduler.pop_next()
        if action is None:
            return
        raw_action = action.copy()
        exec_action = self._policy_action_spec.decode_cartesian(action).as_vector()
        self._env.enqueue_action(exec_action)
        self._action_scheduler.consume_executed_step()
        transformed = transform_action(exec_action, self._env.action_config)
        with self._telemetry_lock:
            self._latest_action = raw_action.tolist()
            self._latest_action_transformed = transformed.tolist()
        self._record_action_telemetry()

    def _copy_obs_for_inference(self, obs: dict) -> dict:
        copied = {}
        for key, value in obs.items():
            copied[key] = value.copy() if isinstance(value, np.ndarray) else value
        return copied

    def _use_rtc(self) -> bool:
        return self._action_scheduler.rtc.enabled

    def _request_infer_async(self, obs: dict) -> bool:
        with self._infer_lock:
            if self._infer_running:
                return False
            self._infer_running = True
            generation = self._infer_generation
        infer_obs = self._copy_obs_for_inference(obs)
        self._attach_rtc_request(infer_obs)
        self._infer_thread = threading.Thread(
            target=self._infer_action_plan_worker,
            args=(infer_obs, generation),
            daemon=True,
        )
        self._infer_thread.start()
        return True

    def _attach_rtc_request(self, obs: dict) -> None:
        rtc_request = self._action_scheduler.rtc_request()
        if rtc_request is not None:
            obs["rtc"] = rtc_request

    def _infer_action_plan_worker(self, obs: dict, generation: int) -> None:
        try:
            chunk, rtc_model_chunk = self._infer_action_chunk(obs)
            if not chunk:
                return
            if generation != self._current_infer_generation() or self.state != State.RUNNING:
                return
            self._action_scheduler.append_inference_result(chunk, rtc_model_chunk)
        except Exception as exc:
            logger.warning("Rejected inference action schedule, State -> IDLE: %s", exc)
            self._action_scheduler.clear()
            self._clear_action_runtime()
            self._reset_policy_client("schedule_error")
            with self._state_lock:
                self._state = State.IDLE
        finally:
            self._set_infer_running(False)

    def _infer_action_chunk(self, obs: dict) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
        if self._client is None:
            self._connect_client()
        if self._client is None:
            logger.warning("推理服务不可用，切回 IDLE")
            with self._state_lock:
                self._state = State.IDLE
            return [], None
        try:
            t0 = time.time()
            with self._client_lock:
                result = self._client.infer(obs)
            total_ms = (time.time() - t0) * 1000.0
            chunk = self._coerce_action_chunk(result["actions"])
            rtc_model_chunk = self._coerce_rtc_model_chunk(result)
            timing = result.get("server_timing", {})
            infer_ms = timing.get("infer_ms")
            with self._telemetry_lock:
                self._latest_total_ms = total_ms
                self._latest_infer_ms = infer_ms
                self._latest_infer_keys = sorted(str(k) for k in result.keys())
                self._update_policy_specific_outputs(result)
            return chunk, rtc_model_chunk
        except Exception as exc:
            logger.warning("推理请求失败，切回 IDLE: %s", exc)
            self._clear_action_runtime()
            self._reset_policy_client("infer_error")
            with self._state_lock:
                self._state = State.IDLE
            return [], None

    def _coerce_action_chunk(self, actions: Any) -> list[np.ndarray]:
        arr = np.asarray(actions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError(f"Expected actions with shape (H, 7), got {arr.shape}")
        target_len = max(1, int(self._args.replan_steps or 1))
        return [row.astype(np.float64).copy() for row in arr[:target_len]]

    def _coerce_rtc_model_chunk(self, result: dict) -> list[np.ndarray] | None:
        rtc_payload = result.get("rtc")
        if not isinstance(rtc_payload, dict):
            return None
        model_actions = rtc_payload.get("model_actions")
        if model_actions is None:
            return None
        arr = np.asarray(model_actions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] <= 0:
            raise ValueError(f"Expected rtc.model_actions with shape (H, A), got {arr.shape}")
        return [row.astype(np.float64).copy() for row in arr]

    def _update_policy_specific_outputs(self, result: dict) -> None:
        if self._args.policy_type != "cosmos":
            self._latest_value_prediction = None
            self._latest_proprio = None
            self._latest_future_proprio = None
            self._latest_future_wrist = None
            self._latest_future_primary = None
            self._latest_future_wrist_shape = None
            self._latest_future_primary_shape = None
            return
        self._latest_value_prediction = jsonable(result.get("value_prediction"))
        self._latest_proprio = jsonable(result.get("proprio"))
        self._latest_future_proprio = jsonable(result.get("future_proprio"))
        future_wrist = extract_first_present(
            result,
            (
                "future_images.wrist",
                "future_images.wrist_image",
                "future_images.future_wrist_image",
                "future_images.observation/wrist_image",
                "future_wrist",
                "future_wrist_image",
                "predicted_wrist",
                "predicted_wrist_image",
            ),
        )
        future_primary = extract_first_present(
            result,
            (
                "future_images.primary",
                "future_images.primary_image",
                "future_images.image",
                "future_images.future_image",
                "future_images.observation/image",
                "future_primary",
                "future_primary_image",
                "predicted_primary",
                "predicted_primary_image",
                "predicted_image",
            ),
        )
        self._latest_future_wrist_shape = shape_list(future_wrist)
        self._latest_future_primary_shape = shape_list(future_primary)
        self._latest_future_wrist = coerce_rgb_frame(future_wrist)
        self._latest_future_primary = coerce_rgb_frame(future_primary)

    def _record_tick_telemetry(self) -> None:
        with self._record_lock:
            if not self._recording:
                return
            schedule = self._action_scheduler.snapshot()
            with self._prompt_lock:
                prompt = self._prompt
            control_status = self._env.get_control_status()
            entry = {
                "timestamp": round(time.time() - self._record_start_time, 3),
                "event": "tick",
                "reference_generator_state": self.state.value,
                "prompt": prompt,
                "state": self._latest_state,
                "joint_state": self._latest_joints,
                "target_pose": self._latest_target_pose,
                "ee_force_torque": self._latest_ee_force_torque,
                "action_raw": self._latest_action,
                "action_transformed": self._latest_action_transformed,
                "inference_time_ms": self._latest_infer_ms,
                "total_time_ms": self._latest_total_ms,
                "pending_action_count": self._env.get_pending_action_count(),
                "queued_plan_count": schedule.action_count,
                "rtc_model_tail_count": schedule.rtc_model_count,
                **control_status,
            }
            if self._args.policy_type == "cosmos":
                entry["value_prediction"] = self._latest_value_prediction
                entry["proprio"] = self._latest_proprio
                entry["future_proprio"] = self._latest_future_proprio
            self._telemetry_log.append(entry)

    def _record_action_telemetry(self) -> None:
        with self._record_lock:
            if not self._recording:
                return
            schedule = self._action_scheduler.snapshot()
            with self._prompt_lock:
                prompt = self._prompt
            control_status = self._env.get_control_status()
            entry = {
                "timestamp": round(time.time() - self._record_start_time, 3),
                "event": "action",
                "reference_generator_state": self.state.value,
                "prompt": prompt,
                "state": self._latest_state,
                "joint_state": self._latest_joints,
                "target_pose": self._latest_target_pose,
                "ee_force_torque": self._latest_ee_force_torque,
                "action_raw": self._latest_action,
                "action_transformed": self._latest_action_transformed,
                "inference_time_ms": self._latest_infer_ms,
                "total_time_ms": self._latest_total_ms,
                "pending_action_count": self._env.get_pending_action_count(),
                "queued_plan_count": schedule.action_count,
                "rtc_model_tail_count": schedule.rtc_model_count,
                **control_status,
            }
            if self._args.policy_type == "cosmos":
                entry["value_prediction"] = self._latest_value_prediction
                entry["proprio"] = self._latest_proprio
                entry["future_proprio"] = self._latest_future_proprio
            self._telemetry_log.append(entry)

    def _build_stream_payload(self, img1_b64: str, img2_b64: str) -> str:
        with self._prompt_lock:
            prompt = self._prompt
        with self._record_lock:
            session_dir = self._session_dir
            recording = self._recording
        with self._telemetry_lock:
            schedule = self._action_scheduler.snapshot()
            future_wrist = self._latest_future_wrist
            future_primary = self._latest_future_primary
            control_status = self._env.get_control_status()
            payload = {
                "reference_generator_state": self.state.value,
                "recording": recording,
                "prompt": prompt,
                "log_subdir": self.log_subdir,
                "log_session_dir": display_log_path(session_dir, LOGS_BASE_DIR),
                "img1": img1_b64,
                "img2": img2_b64,
                "state": self._latest_state,
                "joint_state": self._latest_joints,
                "target_pose": self._latest_target_pose,
                "ee_force_torque": self._latest_ee_force_torque,
                "action_raw": self._latest_action,
                "action_transformed": self._latest_action_transformed,
                "infer_ms": self._latest_infer_ms,
                "total_ms": self._latest_total_ms,
                "robot_connected": self._env.robot is not None,
                "server_connected": self._client is not None,
                "pending_action_count": self._env.get_pending_action_count(),
                "queued_plan_count": schedule.action_count,
                "rtc_model_tail_count": schedule.rtc_model_count,
                "rtc_enabled": self._use_rtc(),
                "rtc_inference_delay": self._args.rtc_inference_delay if self._use_rtc() else None,
                "rtc_execution_horizon": self._args.rtc_execution_horizon if self._use_rtc() else None,
                "rtc_model_tail_shape": list(schedule.rtc_model_shape),
                **control_status,
                "async_chunk_enabled": True,
                "async_infer_running": self._is_infer_running(),
            }
            if self._args.policy_type == "cosmos":
                payload["value_prediction"] = self._latest_value_prediction
                payload["proprio"] = self._latest_proprio
                payload["future_proprio"] = self._latest_future_proprio
                payload["future_wrist"] = encode_jpeg_b64(future_wrist) if future_wrist is not None else None
                payload["future_primary"] = encode_jpeg_b64(future_primary) if future_primary is not None else None
        return json.dumps(payload)

    async def stream_to_clients(self) -> None:
        while True:
            await asyncio.sleep(STREAM_DT)
            with self._frame_lock:
                img1 = self._latest_img1
                img2 = self._latest_img2
            if img1 is None or img2 is None:
                continue
            msg = self._build_stream_payload(encode_jpeg_b64(img1), encode_jpeg_b64(img2))
            async with self._ws_lock:
                dead = set()
                for ws in self._ws_clients:
                    try:
                        await ws.send_text(msg)
                    except Exception:
                        dead.add(ws)
                self._ws_clients -= dead


def normalize_log_subdir(value: str | None) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    if pathlib.PurePosixPath(raw).is_absolute():
        raise ValueError("log_subdir must be relative to logs/")
    parts = [part for part in raw.split("/") if part]
    if any(part in {".", ".."} for part in parts):
        raise ValueError("log_subdir cannot contain '.' or '..'")
    return "/".join(parts)


def display_log_path(path: pathlib.Path | None, logs_base_dir: pathlib.Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(logs_base_dir.parent))
    except ValueError:
        return str(path)


def build_app(coordinator: Coordinator) -> FastAPI:
    static_dir = STATIC_DIR_BASE / coordinator._args.policy_type
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse((static_dir / "index.html").read_text(encoding="utf-8"))

    @app.post("/cmd/start")
    async def cmd_start() -> dict:
        coordinator.cmd_start()
        return {"status": coordinator.state.value}

    @app.post("/cmd/stop")
    async def cmd_stop() -> dict:
        coordinator.cmd_stop()
        return {"status": coordinator.state.value}

    @app.post("/cmd/home")
    async def cmd_home() -> dict:
        coordinator.cmd_home()
        return {"status": coordinator.state.value}

    @app.post("/cmd/prompt")
    async def cmd_prompt(body: dict) -> dict:
        coordinator.cmd_set_prompt(body.get("prompt", ""))
        return {"ok": True}

    @app.post("/cmd/log_subdir")
    async def cmd_log_subdir(body: dict) -> dict:
        try:
            log_subdir = coordinator.cmd_set_log_subdir(body.get("log_subdir", ""))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "log_subdir": log_subdir, "log_session_dir": ""}

    @app.get("/status")
    async def status() -> dict:
        with coordinator._telemetry_lock:
            schedule = coordinator._action_scheduler.snapshot()
            control_status = coordinator._env.get_control_status()
            data = {
                "reference_generator_state": coordinator.state.value,
                "recording": coordinator._recording,
                "prompt": coordinator._prompt,
                "log_subdir": coordinator.log_subdir,
                "log_session_dir": display_log_path(coordinator._session_dir, LOGS_BASE_DIR),
                "robot_connected": coordinator._env.robot is not None,
                "server_connected": coordinator._client is not None,
                "last_infer_keys": coordinator._latest_infer_keys,
                "future_wrist_shape": coordinator._latest_future_wrist_shape,
                "future_primary_shape": coordinator._latest_future_primary_shape,
                "state": coordinator._latest_state,
                "joint_state": coordinator._latest_joints,
                "target_pose": coordinator._latest_target_pose,
                "action_raw": coordinator._latest_action,
                "action_transformed": coordinator._latest_action_transformed,
                "infer_ms": coordinator._latest_infer_ms,
                "total_ms": coordinator._latest_total_ms,
                "pending_action_count": coordinator._env.get_pending_action_count(),
                "queued_plan_count": schedule.action_count,
                "rtc_model_tail_count": schedule.rtc_model_count,
                "rtc_enabled": coordinator._use_rtc(),
                "rtc_inference_delay": coordinator._args.rtc_inference_delay if coordinator._use_rtc() else None,
                "rtc_execution_horizon": coordinator._args.rtc_execution_horizon if coordinator._use_rtc() else None,
                "rtc_model_tail_shape": list(schedule.rtc_model_shape),
                **control_status,
                "async_chunk_enabled": True,
                "async_infer_running": coordinator._is_infer_running(),
            }
        return data

    @app.websocket("/ws/frames")
    async def ws_frames(websocket: WebSocket) -> None:
        await websocket.accept()
        async with coordinator._ws_lock:
            coordinator._ws_clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            async with coordinator._ws_lock:
                coordinator._ws_clients.discard(websocket)

    @app.on_event("startup")
    async def startup() -> None:
        asyncio.create_task(coordinator.stream_to_clients())

    return app


def main(args: Args) -> None:
    import uvicorn

    coordinator = Coordinator(args)
    ctrl_thread = threading.Thread(target=coordinator.run_control_loop, daemon=True)
    ctrl_thread.start()
    app = build_app(coordinator)
    uvicorn.run(app, host="0.0.0.0", port=args.web_port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))

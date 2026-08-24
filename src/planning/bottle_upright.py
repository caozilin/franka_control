from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from planning.sqp.kinematics import PandaKinematics
from utils.pose import matrix_to_rotvec, rotvec_to_matrix


FINAL_POSITION_M = np.array((0.42928630, 0.0, 0.31502816), dtype=np.float64)
FINAL_BASE_YAWS_RAD = (-0.5 * math.pi, 0.5 * math.pi)
LIFT_CLEARANCE_M = 0.115
TURN_CLEARANCE_M = 0.140
IK_MAX_ITERATIONS = 100
IK_POSITION_TOLERANCE_M = 0.004
IK_ROTATION_TOLERANCE_RAD = math.radians(3.0)
IK_MIN_JOINT_MARGIN_RAD = math.radians(3.0)
CANDIDATE_MIN_JOINT_MARGIN_RAD = math.radians(15.0)
CANDIDATE_JOINT_MARGIN_BAND_RAD = math.radians(5.0)
EDGE_RESOLUTION_RAD = 0.045
COMMAND_RATE_HZ = 10.0
CARTESIAN_AXIS_SPEED_M_S = 0.100
CARTESIAN_ACCELERATION_TIME_S = 0.500
CARTESIAN_DECELERATION_TIME_S = 0.500
BOTTLE_TRANSFER_SPEED_M_S = 0.100
BOTTLE_LOWER_HANDOFF_SPEED_M_S = 0.045
BOTTLE_TRANSFER_DECELERATION_TIME_S = 1.000
BOTTLE_LOWER_DECELERATION_TIME_S = 2.000
DEFAULT_ANGULAR_SPEED_RAD_S = math.radians(45.0)
TRANSFER_ANGULAR_SPEED_RAD_S = math.radians(30.0)
PRECISION_ANGULAR_SPEED_RAD_S = math.radians(18.0)
SCURVE_RAMP_FRACTION = 0.20
SCURVE_DURATION_FACTOR = 1.0 / (1.0 - SCURVE_RAMP_FRACTION)


@dataclass(frozen=True)
class BottleCandidate:
    label: str
    final_position: np.ndarray
    final_rotation: np.ndarray
    lift_q: np.ndarray
    turn_q: np.ndarray
    final_q: np.ndarray
    minimum_joint_margin_rad: float
    manipulability: float
    prescore: float


@dataclass(frozen=True)
class BottlePlan:
    actions: tuple[np.ndarray, ...]
    candidate: BottleCandidate
    candidate_count: int


class BottleUprightPlanner:
    """Post-grasp half of MuJoCo Adjust Bottle for the calibrated real cell."""

    def __init__(self, kinematics: PandaKinematics | None = None) -> None:
        self.kinematics = kinematics or PandaKinematics()

    def _valid(self, q: np.ndarray) -> bool:
        lower = self.kinematics.joint_lower
        upper = self.kinematics.joint_upper
        if np.any(q <= lower + math.radians(1.0)) or np.any(q >= upper - math.radians(1.0)):
            return False
        state = self.kinematics.evaluate(q, include_manipulability=False, include_link_points=True)
        assert state.link_points is not None
        points = state.link_points
        segments = [
            (points[index], points[index + 1])
            for index in range(points.shape[0] - 1)
            if np.linalg.norm(points[index + 1] - points[index]) >= 1e-8
        ]
        for first in range(len(segments)):
            for second in range(first + 2, len(segments)):
                # Same 5 cm serial-link capsule used by the live SQP objective.
                if self._segment_distance(
                    segments[first][0], segments[first][1],
                    segments[second][0], segments[second][1],
                ) < 0.05:
                    return False
        return True

    @staticmethod
    def _segment_distance(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> float:
        u, v, w = a1 - a0, b1 - b0, a0 - b0
        uu, uv, vv = float(u @ u), float(u @ v), float(v @ v)
        uw, vw = float(u @ w), float(v @ w)
        denominator = uu * vv - uv * uv
        candidates: list[tuple[float, float]] = []
        if denominator > 1e-14:
            s = (uv * vw - vv * uw) / denominator
            t = (uu * vw - uv * uw) / denominator
            if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
                candidates.append((s, t))
        if vv > 1e-14:
            candidates.extend(((0.0, np.clip(vw / vv, 0.0, 1.0)), (1.0, np.clip((vw + uv) / vv, 0.0, 1.0))))
        if uu > 1e-14:
            candidates.extend(((np.clip(-uw / uu, 0.0, 1.0), 0.0), (np.clip((uv - uw) / uu, 0.0, 1.0), 1.0)))
        if not candidates:
            return min(
                float(np.linalg.norm(a - b))
                for a in (a0, a1) for b in (b0, b1)
            )
        return min(float(np.linalg.norm(w + s * u - t * v)) for s, t in candidates)

    def _solve_ik(self, position: np.ndarray, rotation: np.ndarray, seeds: tuple[np.ndarray, ...]):
        best = None
        for raw_seed in seeds:
            q = np.clip(raw_seed, self.kinematics.joint_lower, self.kinematics.joint_upper).copy()
            for _ in range(IK_MAX_ITERATIONS):
                state = self.kinematics.evaluate(q, include_manipulability=False, include_link_points=False)
                position_error = position - state.position
                rotation_error = matrix_to_rotvec(rotation @ state.rotation.T)
                if np.linalg.norm(position_error) <= IK_POSITION_TOLERANCE_M and np.linalg.norm(rotation_error) <= IK_ROTATION_TOLERANCE_RAD:
                    break
                error = np.concatenate((position_error, rotation_error))
                damping = 0.025 + 0.075 * min(1.0, float(np.linalg.norm(error)))
                jacobian = state.jacobian
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * damping * np.eye(6), error)
                largest = float(np.max(np.abs(delta)))
                if largest > 0.10:
                    delta *= 0.10 / largest
                q = np.clip(q + delta, self.kinematics.joint_lower, self.kinematics.joint_upper)
            else:
                continue
            if not self._valid(q):
                continue
            state = self.kinematics.evaluate(q, include_manipulability=True, include_link_points=False)
            margin = float(np.min(np.minimum(q - self.kinematics.joint_lower, self.kinematics.joint_upper - q)))
            if margin < IK_MIN_JOINT_MARGIN_RAD:
                continue
            score = float(state.manipulability or 0.0) + 0.2 * margin
            if best is None or score > best[3]:
                best = (q.copy(), float(state.manipulability or 0.0), margin, score)
        return best

    def _edge_valid(self, start: np.ndarray, goal: np.ndarray) -> bool:
        count = max(
            1,
            int(math.ceil(float(np.max(np.abs(goal - start))) / EDGE_RESOLUTION_RAD)),
        )
        return all(
            self._valid(start + (goal - start) * (index / count))
            for index in range(1, count + 1)
        )

    @staticmethod
    def _redundant_seeds(*references: np.ndarray) -> tuple[np.ndarray, ...]:
        seeds: list[np.ndarray] = []
        for reference in references:
            seeds.append(reference.copy())
            for elbow, wrist in ((0.0, -math.pi / 2), (0.0, math.pi / 2), (-0.45, 0.0), (0.45, 0.0), (-0.45, -math.pi / 2), (0.45, math.pi / 2)):
                seed = reference.copy()
                seed[2] += elbow
                seed[-1] += wrist
                seeds.append(seed)
        return tuple(seeds)

    @staticmethod
    def _final_rotation(
        yaw: float,
        *,
        camera_mouth_same_side: bool,
    ) -> np.ndarray:
        # The only observable needed for an axisymmetric bottle is whether its
        # mouth is on the wrist-camera side.  The calibrated original N action
        # used MuJoCo's negative fixed-grasp branch, so keep it as M's canonical
        # gripper frame; N flips that relative bottle pose by 180 degrees.
        canonical_grasp_rotation = np.diag((1.0, -1.0, -1.0))
        # Real-cell calibration: the same-side case is the original +90 deg
        # relative bottle pose; the opposite-side case is its 180 deg flip.
        initial_pitch = math.pi / 2.0 if camera_mouth_same_side else -math.pi / 2.0
        initial_body_rotation = rotvec_to_matrix(np.array((0.0, initial_pitch, 0.0)))
        body_in_gripper = canonical_grasp_rotation.T @ initial_body_rotation
        final_body_rotation = rotvec_to_matrix(np.array((0.0, 0.0, yaw)))
        return final_body_rotation @ body_in_gripper.T

    def candidates(
        self,
        current_q: np.ndarray,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        *,
        camera_mouth_same_side: bool = False,
    ) -> list[BottleCandidate]:
        lift_position = current_position + np.array((0.0, 0.0, LIFT_CLEARANCE_M))
        lift = self._solve_ik(lift_position, current_rotation, self._redundant_seeds(current_q))
        if lift is None or not self._edge_valid(current_q, lift[0]):
            raise RuntimeError("扶瓶规划失败：抬升姿态无可行 IK")
        result: list[BottleCandidate] = []
        for yaw in FINAL_BASE_YAWS_RAD:
            label = "left" if yaw > 0.0 else "right"
            final_rotation = self._final_rotation(
                yaw,
                camera_mouth_same_side=camera_mouth_same_side,
            )
            turn_position = FINAL_POSITION_M + np.array((0.0, 0.0, TURN_CLEARANCE_M))
            turn = self._solve_ik(turn_position, final_rotation, self._redundant_seeds(lift[0], current_q))
            if turn is None:
                continue
            final = self._solve_ik(FINAL_POSITION_M, final_rotation, self._redundant_seeds(turn[0], lift[0]))
            if final is None or not self._edge_valid(turn[0], final[0]):
                continue
            current_margin = float(np.min(np.minimum(
                current_q - self.kinematics.joint_lower,
                self.kinematics.joint_upper - current_q,
            )))
            margin = min(current_margin, lift[2], turn[2], final[2])
            motion = float(np.linalg.norm(turn[0] - lift[0]))
            descent = float(np.linalg.norm(final[0] - turn[0]))
            manipulability = min(lift[1], turn[1], final[1])
            prescore = 2.0 * motion + 0.35 * descent + 0.08 / max(manipulability, 1e-4) + 0.06 / max(margin, math.radians(1.0))
            result.append(BottleCandidate(label, FINAL_POSITION_M.copy(), final_rotation, lift[0], turn[0], final[0], margin, manipulability, prescore))
        safe = [item for item in result if item.minimum_joint_margin_rad >= CANDIDATE_MIN_JOINT_MARGIN_RAD]
        if safe:
            best_margin = max(item.minimum_joint_margin_rad for item in safe)
            result = [item for item in safe if item.minimum_joint_margin_rad >= best_margin - CANDIDATE_JOINT_MARGIN_BAND_RAD]
        return sorted(result, key=lambda item: (item.prescore, item.label))

    @staticmethod
    def _raised_cosine_integral(elapsed: float, duration: float) -> float:
        if duration <= 0.0:
            return elapsed
        clamped = float(np.clip(elapsed, 0.0, duration))
        return 0.5 * clamped - duration / (2.0 * math.pi) * math.sin(
            math.pi * clamped / duration
        )

    @staticmethod
    def _quintic_ramp_integral(fraction: float) -> float:
        u = float(np.clip(fraction, 0.0, 1.0))
        return u**4 * (2.5 - 3.0 * u + u**2)

    @classmethod
    def _s_curve_progress(cls, fraction: float) -> float:
        u = float(np.clip(fraction, 0.0, 1.0))
        ramp = SCURVE_RAMP_FRACTION
        raw_total = 1.0 - ramp
        if u < ramp:
            raw = ramp * cls._quintic_ramp_integral(u / ramp)
        elif u <= 1.0 - ramp:
            raw = 0.5 * ramp + (u - ramp)
        else:
            remaining = ramp * cls._quintic_ramp_integral((1.0 - u) / ramp)
            raw = raw_total - remaining
        return raw / raw_total

    @classmethod
    def _append_pose_segment(
        cls,
        actions: list[np.ndarray],
        start_position: np.ndarray,
        start_rotation: np.ndarray,
        goal_position: np.ndarray,
        goal_rotation: np.ndarray,
        *,
        start_speed_m_s: float,
        end_speed_m_s: float,
        angular_speed_rad_s: float,
        deceleration_time_s: float = CARTESIAN_DECELERATION_TIME_S,
    ) -> None:
        """MuJoCo append_blended_motion timing, emitted as base-frame deltas."""
        delta = goal_position - start_position
        axis_distance = float(np.max(np.abs(delta)))
        total_rotation = matrix_to_rotvec(goal_rotation @ start_rotation.T)
        rotation_angle = float(np.linalg.norm(total_rotation))
        initial_speed = min(max(float(start_speed_m_s), 0.0), CARTESIAN_AXIS_SPEED_M_S)
        final_speed = min(max(float(end_speed_m_s), 0.0), CARTESIAN_AXIS_SPEED_M_S)

        def ramp_distance(candidate_peak: float) -> float:
            result = 0.0
            if candidate_peak > initial_speed + 1e-12:
                result += 0.5 * (initial_speed + candidate_peak) * CARTESIAN_ACCELERATION_TIME_S
            if candidate_peak > final_speed + 1e-12:
                result += 0.5 * (candidate_peak + final_speed) * deceleration_time_s
            return result

        peak_floor = max(initial_speed, final_speed)
        if axis_distance >= ramp_distance(CARTESIAN_AXIS_SPEED_M_S) - 1e-12:
            peak_speed = CARTESIAN_AXIS_SPEED_M_S
        elif ramp_distance(peak_floor) > axis_distance + 1e-12:
            peak_speed = peak_floor
        else:
            lower, upper = peak_floor, CARTESIAN_AXIS_SPEED_M_S
            for _ in range(64):
                middle = 0.5 * (lower + upper)
                if ramp_distance(middle) <= axis_distance:
                    lower = middle
                else:
                    upper = middle
            peak_speed = lower

        acceleration_time = CARTESIAN_ACCELERATION_TIME_S if peak_speed > initial_speed + 1e-12 else 0.0
        deceleration_time = deceleration_time_s if peak_speed > final_speed + 1e-12 else 0.0
        acceleration_distance = 0.5 * (initial_speed + peak_speed) * acceleration_time
        deceleration_distance = 0.5 * (peak_speed + final_speed) * deceleration_time
        if acceleration_distance + deceleration_distance > axis_distance + 1e-12:
            scale = axis_distance / max(acceleration_distance + deceleration_distance, 1e-12)
            acceleration_time *= scale
            deceleration_time *= scale
            acceleration_distance *= scale
            deceleration_distance *= scale
        cruise_distance = max(axis_distance - acceleration_distance - deceleration_distance, 0.0)
        cruise_time = cruise_distance / max(peak_speed, 1e-12)
        translation_duration = acceleration_time + cruise_time + deceleration_time
        rotation_duration = (
            SCURVE_DURATION_FACTOR * rotation_angle / angular_speed_rad_s
            if rotation_angle > 1e-12 else 0.0
        )
        duration = max(translation_duration, rotation_duration, 1.0 / COMMAND_RATE_HZ)
        count = max(1, int(math.ceil(duration * COMMAND_RATE_HZ)))
        sampled_duration = count / COMMAND_RATE_HZ
        cruise_end_time = acceleration_time + cruise_time

        previous_position = start_position.copy()
        previous_rotation = start_rotation.copy()
        for index in range(1, count + 1):
            elapsed = index / COMMAND_RATE_HZ
            profile_elapsed = elapsed * translation_duration / sampled_duration
            if axis_distance <= 1e-12:
                position_fraction = 1.0
            elif profile_elapsed <= acceleration_time:
                travelled = initial_speed * profile_elapsed + (peak_speed - initial_speed) * cls._raised_cosine_integral(profile_elapsed, acceleration_time)
                position_fraction = travelled / axis_distance
            elif profile_elapsed <= cruise_end_time:
                travelled = acceleration_distance + peak_speed * (profile_elapsed - acceleration_time)
                position_fraction = travelled / axis_distance
            else:
                deceleration_elapsed = profile_elapsed - cruise_end_time
                travelled = acceleration_distance + cruise_distance + peak_speed * deceleration_elapsed + (final_speed - peak_speed) * cls._raised_cosine_integral(deceleration_elapsed, deceleration_time)
                position_fraction = travelled / axis_distance
            position = start_position + float(np.clip(position_fraction, 0.0, 1.0)) * delta
            rotation_fraction = cls._s_curve_progress(index / count)
            rotation = rotvec_to_matrix(rotation_fraction * total_rotation) @ start_rotation
            translation = position - previous_position
            rotation_delta = matrix_to_rotvec(rotation @ previous_rotation.T)
            action = np.zeros(7, dtype=np.float64)
            action[:3] = translation
            action[3:6] = rotation_delta
            action[6] = 1.0
            actions.append(action)
            previous_position = position
            previous_rotation = rotation

    def plan(
        self,
        current_q: np.ndarray,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        *,
        camera_mouth_same_side: bool = False,
    ) -> BottlePlan:
        candidates = self.candidates(
            current_q,
            current_position,
            current_rotation,
            camera_mouth_same_side=camera_mouth_same_side,
        )
        if not candidates:
            raise RuntimeError("扶瓶规划失败：左右两个固定末端候选均未通过筛选")
        chosen = candidates[0]
        actions: list[np.ndarray] = []
        lift_position = current_position + np.array((0.0, 0.0, LIFT_CLEARANCE_M))
        turn_position = FINAL_POSITION_M + np.array((0.0, 0.0, TURN_CLEARANCE_M))
        self._append_pose_segment(
            actions, current_position, current_rotation,
            lift_position, current_rotation,
            start_speed_m_s=0.0,
            end_speed_m_s=BOTTLE_TRANSFER_SPEED_M_S,
            angular_speed_rad_s=DEFAULT_ANGULAR_SPEED_RAD_S,
        )
        self._append_pose_segment(
            actions, lift_position, current_rotation,
            turn_position, chosen.final_rotation,
            start_speed_m_s=BOTTLE_TRANSFER_SPEED_M_S,
            end_speed_m_s=BOTTLE_LOWER_HANDOFF_SPEED_M_S,
            angular_speed_rad_s=TRANSFER_ANGULAR_SPEED_RAD_S,
            deceleration_time_s=BOTTLE_TRANSFER_DECELERATION_TIME_S,
        )
        self._append_pose_segment(
            actions, turn_position, chosen.final_rotation,
            chosen.final_position, chosen.final_rotation,
            start_speed_m_s=BOTTLE_LOWER_HANDOFF_SPEED_M_S,
            end_speed_m_s=0.0,
            angular_speed_rad_s=PRECISION_ANGULAR_SPEED_RAD_S,
            deceleration_time_s=BOTTLE_LOWER_DECELERATION_TIME_S,
        )
        for _ in range(10):
            actions.append(np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))
        for _ in range(10):
            actions.append(np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)))
        return BottlePlan(tuple(actions), chosen, len(candidates))

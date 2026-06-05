from __future__ import annotations

from franka_control import FrankaEnv


def main() -> int:
    env = FrankaEnv(no_robot=True, no_cameras=True)
    print("initial state:", env.get_robot_state_vector())
    obs, _, _ = env.get_observation("test")
    print("observation/state shape:", obs["observation/state"].shape)
    print("observation/image shape:", obs["observation/image"].shape)

    env.enqueue_action([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], transform=True)
    print("commanded pose after one action:", env.commanded_pose_array)
    print("latest trace:", env.get_latest_control_trace())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

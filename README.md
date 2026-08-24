# franka_control

基于进程内 C++ 实时后端的 Python 高级 Franka 控制接口。

Python 层以 10Hz 运行策略/输入更新，并通过 pybind11 扩展发送动作。C++ 后端拥有 libfranka 连接和实时控制回路；控制回路不回调 Python。

## Python 环境

必须使用 `uv` 为本项目创建专用虚拟环境。不要用系统 Python、系统 `pip`，也不要混用其他虚拟环境；编译和运行都应使用本项目的 `.venv`。

```bash
cd franka_control
uv venv --python 3.12
uv sync --extra build --extra dev
```

日常检查建议先确认解释器来源：

```bash
which python
python --version
```

期望项目解释器位于：

```bash
.venv/bin/python
```

默认目录结构如下，`libfranka` 与本项目位于同一父目录：

```text
<workspace>/
├── libfranka/
└── franka_control/
```

构建 libfranka 0.21.2 后构建 C++ 扩展：

```bash
CMEEL_PREFIX="$PWD/.venv/lib/python3.12/site-packages/cmeel.prefix"
uv run --extra build cmake -S ../libfranka -B ../libfranka/build-openrobots \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DGENERATE_PYLIBFRANKA=OFF \
  -DCMAKE_PREFIX_PATH="$CMEEL_PREFIX" \
  -Dfmt_DIR=/usr/lib/x86_64-linux-gnu/cmake/fmt \
  -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python" \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
uv run --extra build cmake --build ../libfranka/build-openrobots -j2
uv run --extra build cmake -S . -B build -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python"
uv run --extra build cmake --build build -j2
```

如果只想重编译当前 pybind11 扩展，也应继续使用同一个 `uv` 环境：

```bash
uv run --extra build cmake -S . -B build -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python"
uv run --extra build cmake --build build -j2
uv run python -c "import sys; sys.path.insert(0, 'src'); from control._franka_backend import RealtimeFrankaBackend; print('import-ok')"
```

## 最小机器人验证

最简单的机器人验证是将末端执行器沿基坐标系 +X 方向移动 30cm。它使用 `FrankaEnv`；Python 每 0.1s 发送一个动作，C++ 以 1kHz 的频率输出笛卡尔阻抗力矩指令流。支持 C++ 参考轨迹类型包括 `min_jerk`、`linear`、`cubic` 和 `motion_limited`。

```bash
uv run python examples/move_forward_30cm.py --ip 172.16.0.2
```

使用仅连接模式验证 C++ 扩展可以连接但不启动运动：

```bash
uv run python examples/move_forward_30cm.py --ip 172.16.0.2 --yes --connect-only
```

在运行主动运动前，请确保用户急停按钮可用。

## 统一运动规划接口

所有笛卡尔动作源都通过同一个 `CartesianActionPlanner` 接口，可在启动时选择
`direct`、`baseline_sqp` 或 `shadow_sqp`。键盘、PS4 和 PICO 遥操作及数据采集统一
使用 `scripts/teleop.py`；Policy 和固定轨迹也使用相同规划边界。

```bash
.venv/bin/python scripts/teleop.py --input-device keyboard
.venv/bin/python scripts/teleop.py --input-device ps4 --planner-mode baseline_sqp
.venv/bin/python scripts/teleop.py --input-device pico --with-cameras
.venv/bin/python scripts/teleop.py --input-device ps4 --planner-mode shadow_sqp \
  --rotation-ranged-axes false false true --rotation-limits-deg 30 30 45
```

统一入口默认 `direct + linear` 且关闭相机；只有显式传入 `--with-cameras` 才启用双相机和片段录制。详细按键见 [统一笛卡尔遥操作文档](docs/cartesian_teleop.md)。

Planner、Reference 和 Torque Tracker 分别选择，再由类型化 Router 按输出空间检查兼容性。
`direct` 输出笛卡尔目标，可配 `min_jerk`、`linear`、`cubic` 或 `motion_limited`
笛卡尔 Reference，并使用内部的 `cartesian_impedance`。两种 SQP 输出绝对关节目标，可配
`min_jerk`、`linear` 或 `cubic` 关节 Reference，并使用内部的 `joint_pid`。
`joint_pid` 移植自 `franka_mujoco` 的有界泄漏 PID 关节参考外环，
修正后的关节参考仍由关节阻抗安全转换为扭矩。关节 Reference 固定在一个 10 Hz
控制周期内走完。CLI 只提供 `--tracker-mode {auto,pid}`；两者都按 Reference 空间
选择内部实现：笛卡尔 Reference 保持笛卡尔并使用 `cartesian_impedance`，关节
Reference 使用 `joint_pid`，不会把笛卡尔 Reference 转成关节 Reference。

## 6秒轨迹复现

通过相同的 `FrankaEnv` 和 C++ 后端路径复现 60 拍、10Hz 的轨迹。`--scale` 用于缩放笛卡尔平移和旋转增量，然后再转换为归一化动作。

```bash
uv run python scripts/example_trajectory_record_and_analyze.py --ip 172.16.0.2 --reference linear --scale 1.0
```

不连接机器人地试运行 Python/日志路径：

```bash
uv run python scripts/example_trajectory_record_and_analyze.py --no-robot --yes --scale 1.0 --settle 0.2
```

使用 `--reference min_jerk`、`--reference linear`、`--reference cubic` 或 `--reference motion_limited` 来选择 C++ 实时参考轨迹。要使 nullspace 中仅 J2 趋近于 0，请使用 `--nullspace-enabled --nullspace-q-target "nan,0,nan,nan,nan,nan,nan"`；`nan` 表示该关节没有姿态参考。

默认情况下，机器人脚本首先调用 C++ 关节位置复位。只有在明确希望从当前位置开始时才使用 `--no-home-first`。


！！！TODO
可以迁，而且方向已经基本明确了。

  结论
  franka-interface 里真正可迁移的不是“双连接轮询”，而是 franka_ros2 的这套模式：

  - 单个 franka::Gripper 对象
  - move/grasp 放到异步线程执行
  - 同时继续在另一条线程/定时器里 readOnce()
  - 用缓存的 current_gripper_state_ 提供实时位置/feedback

  我已经在你当前项目里做了同模型验证，而且成功了：

  - 单个 RealtimeGripperBackend 对象
  - 一个线程里执行阻塞 command(0.0, 0.08, 60.0)
  - 主线程每 0.1s 调 read_once()
  - 实际拿到了连续中间值：

  0.0677
  0.0516
  0.0356
  0.0197
  0.00516
  0.00094

  而且最终：

  - command 返回 True
  - final read_once() 正常
  - 没有 Net Exception / I/O error

  这说明：

  - 双连接方案不行
  - 单连接同对象“命令异步 + 状态轮询”是可行的
  - 这正是你应该迁移到本项目的实现路线

  franka-interface 里怎么做
  有两套值得参考：

  1. franka-interface 自己的 run loop

  - 单个 FrankaGripper
  - 100Hz 后台线程轮询 getGripperState()
  - 位置在：
      - ../franka-interface/franka-interface/include/franka-interface/run_loop.h:63
      - ../franka-interface/franka-interface/src/run_loop.cpp:333

  2. franka_ros2 的 franka_gripper

  - 单个 std::unique_ptr<franka::Gripper>
  - move/grasp 用 std::async(...)
  - 定时器继续 publishGripperState() 调 readOnce()
  - 位置在：
      - ../franka-interface/ros2_ws/src/franka_ros2/franka_gripper/src/gripper_action_server.cpp:141
      - ../franka-interface/ros2_ws/src/franka_ros2/franka_gripper/src/gripper_action_server.cpp:253
      - ../franka-interface/ros2_ws/src/franka_ros2/franka_gripper/include/franka_gripper/gripper_action_server.hpp:84

  迁移到本项目的具体方案
  你这个项目里应该改成：

  1. RealtimeGripperBackend 只保留一个实例
     在夹爪 worker 线程里创建一次，不再创建第二个 backend。

  2. 命令不要在 worker 主循环里直接阻塞执行
     现在是：

  - command_queue.get()
  - gripper.command(...)
  - 期间整个 worker 卡住

  要改成：

  - worker 主循环持续 10Hz read_once()
  - 收到新目标时，如果当前没有命令线程在跑，就启动一个命令线程
  - 命令线程里执行 gripper.command(...)
  - worker 主循环继续轮询 read_once()，实时更新 _last_gripper_width

  3. 命令合并策略
     因为你的输入是开/关离散命令：

  - 新目标来了，更新 desired_target
  - 如果当前命令线程空闲，就按最新目标启动一次
  - 如果命令线程还在跑，不打断；等它结束后检查 desired_target 是否变化，再决定是否立刻补发下一条

  4. state[6:8] 继续从 _last_gripper_width 出
     这部分你现有结构已经对了：

  - src/control/franka_env.py:1057

  为什么这是正确路线
  因为它同时满足三件事：

  - 不阻塞机械臂主控制线程
  - 不需要双 gripper 连接
  - 能拿到夹爪运动过程中的连续位置

  当前实现
  夹爪 worker 已经按这套结构重构，具体线程与命令合并契约见 `docs/gripper_driver.md`：

  - 单 backend
  - 后台 10Hz 持续 read_once()
  - 命令异步线程执行 move/grasp
  - 状态连续刷新

  这是现在最值得落地的修复。

# franka_control

基于进程内 C++ 实时后端的 Python 高级 Franka 控制接口。

Python 层以 10Hz 运行策略/输入更新，并通过 pybind11 扩展发送动作。C++ 后端拥有 libfranka 连接和实时控制回路；控制回路不回调 Python。

## Python 环境

推荐使用 `uv` 为本项目创建专用虚拟环境。

```bash
cd /home/k324/franka_my_code/franka_control
$HOME/.local/bin/uv venv --python 3.12
source .venv/bin/activate
$HOME/.local/bin/uv pip install -U pip setuptools wheel pybind11 cmake
$HOME/.local/bin/uv pip install -e .
```

构建 libfranka 0.21.2 后构建 C++ 扩展：

```bash
cd /home/k324/franka_my_code/franka_control
source .venv/bin/activate
cmake -S /home/k324/franka_my_code/libfranka-0.21.2 -B /home/k324/franka_my_code/libfranka-0.21.2/build-openrobots -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DGENERATE_PYLIBFRANKA=OFF
cmake --build /home/k324/franka_my_code/libfranka-0.21.2/build-openrobots -j2
cmake -S . -B build -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j2
```

## 最小机器人验证

最简单的机器人验证是将末端执行器沿基坐标系 +X 方向移动 30cm。它使用 `FrankaEnv`；Python 每 0.1s 发送一个动作，C++ 以 1kHz 的频率输出笛卡尔阻抗力矩指令流。支持 C++ 参考轨迹类型包括 `min_jerk`、`linear`、`cubic` 和 `motion_limited`。

```bash
.venv/bin/python examples/move_forward_30cm.py --ip 172.16.0.2
```

使用仅连接模式验证 C++ 扩展可以连接但不启动运动：

```bash
.venv/bin/python examples/move_forward_30cm.py --ip 172.16.0.2 --yes --connect-only
```

在运行主动运动前，请确保用户急停按钮可用。

## 6秒轨迹复现

通过相同的 `FrankaEnv` 和 C++ 后端路径复现 60 拍、10Hz 的轨迹。`--scale` 用于缩放笛卡尔平移和旋转增量，然后再转换为归一化动作。

```bash
.venv/bin/python scripts/example_trajectory_record_and_analyze.py --ip 172.16.0.2 --reference linear --scale 1.0
```

不连接机器人地试运行 Python/日志路径：

```bash
.venv/bin/python scripts/example_trajectory_record_and_analyze.py --no-robot --yes --scale 1.0 --settle 0.2
```

使用 `--reference min_jerk`、`--reference linear`、`--reference cubic` 或 `--reference motion_limited` 来选择 C++ 实时参考轨迹。要使 nullspace 中仅 J2 趋近于 0，请使用 `--nullspace-enabled --nullspace-q-target "nan,0,nan,nan,nan,nan,nan"`；`nan` 表示该关节没有姿态参考。

默认情况下，机器人脚本首先调用 C++ 关节位置复位。只有在明确希望从当前位置开始时才使用 `--no-home-first`。

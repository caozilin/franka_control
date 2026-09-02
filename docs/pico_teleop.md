# PICO 双手柄遥操作

PICO 端只负责发送左右手柄的 tracking pose 和按键状态。本项目通过 UDP 接收最新状态，以 10Hz 生成笛卡尔动作，再选择 `direct`、`baseline_sqp` 或 `shadow_sqp` 路径执行。

## 当前映射

默认 `split` 模式：

- 左手柄位置控制末端 XYZ 平移，左手柄姿态被忽略。
- 右手柄姿态控制末端旋转，右手柄位置被忽略。
- 左 Grip 是平移离合，右 Grip 是旋转离合；两个通道分别重锚，可以单独或同时操作。
- 每个通道首次按下只重设自己的锚点，不产生跳变。
- 右 Trigger 使用上升沿切换夹爪：每按下一次，在张开和闭合之间切换一次；持续按住不会重复切换。
- Primary、Secondary 和摇杆随协议接收并保留，当前不绑定机器人行为。
- 超过 `pico_stale_timeout_s` 没有新包，或任一手柄丢失 tracking 时，不再生成动作。

`single_6dof` baseline 使用右手柄 grip pose 控制全部六维增量，并沿用左右 Grip 同时使能的行为。两种模式都应使用 grip pose，而不是 aim/ray pose。

通过命令行切换：

```bash
# 主方法，默认值
--pico-mapping-mode split

# 单手6D baseline
--pico-mapping-mode single_6dof
```

PICO/Unity 通常以 60Hz 或 90Hz 发包。接收线程只保存最新包，coordinator 固定以 `control_hz=10` 消费，因此不会堆积旧位姿。

## XRoboToolkit 真机入口

PICO、键盘和 PS4 现已统一使用笛卡尔遥操作/数采入口。已安装 XRoboToolkit PC Service 和 `xrobotoolkit_sdk` 时：

```bash
.venv/bin/python scripts/teleop.py --input-device pico
```

该入口会管理 SDK bridge，先自动错误恢复并回零，再进入控制；不需要浏览器。相机默认关闭，传 `--with-cameras` 后启用数采：A 开始、再次按 A 保存，B 作废。旧命令 `scripts/run_xrobotoolkit_teleop.py` 仍兼容，但只作为该统一入口的转发器。

在当前电脑上，Franka 网口 `enp3s0` 的中断固定在 CPU 6，因此入口默认将控制进程固定到独立的 CPU 2，避免 1 kHz 实时控制线程与网卡中断争用同一 CPU。需要修改时可传 `--control-cpu N`。

默认按操作者站在机械臂正前方标定：左手柄相对位置的前/后、左/右、上/下分别映射机器人 Base 平移。平移和旋转比例默认都是 `1.0`。右手柄相对姿态在末端执行器局部坐标系中解释，每一拍再使用当前末端姿态转换成 Base 系增量动作。

- 左 Grip：按下时锚定左手柄位置；之后的相对位移就是末端 Base 平移目标。
- 右 Grip：按下时锚定右手柄姿态；之后的相对姿态就是末端旋转目标，右手 X/Y 互换、Z 不变。
- 这是相对位姿控制，不是速度控制：保持手柄不动时，机械臂走到对应目标后会静止。松开再按 Grip 会在当前位姿重新锚定。
- 右 Trigger：每按一下切换一次夹爪张开/闭合状态。

## UDP 数据

默认监听 `0.0.0.0:9010`，每个 UDP datagram 是一个 JSON 对象：

```json
{
  "version": 1,
  "session_id": "pico-session-1",
  "sequence": 42,
  "timestamp_s": 1780000000.0,
  "left": {
    "position": [-0.2, 1.2, 0.3],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "grip": 1.0,
    "trigger": 0.0,
    "thumbstick": [0.0, 0.0],
    "primary": false,
    "secondary": false,
    "tracked": true
  },
  "right": {
    "position": [0.2, 1.2, 0.3],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    "grip": 1.0,
    "trigger": 0.0,
    "thumbstick": [0.0, 0.0],
    "primary": false,
    "secondary": false,
    "tracked": true
  }
}
```

位姿使用 Unity/PICO 的左手系（x 向右、y 向上、z 向前）。项目内部先转换为右手系，再应用 `pico_rotation_base_from_pico` 标定旋转矩阵。

## 离线联调

先启动无真机、无相机 coordinator：

```bash
uv run python scripts/coordinator.py \
  --action-source pico \
  --pico-mapping-mode split \
  --pico-bind-host 127.0.0.1 \
  --pico-port 9010 \
  --no-robot --no-cameras --no-use-gripper --no-startup-home
```

另一个终端发送静止数据：

```bash
uv run python examples/pico_udp_sender.py
```

加 `--enable-motion` 会让示例左手柄沿 x 方向往复运动。即使是离线 backend，也应先在 coordinator 页面执行 Start；状态接口 `/status` 的 `pico` 字段会显示当前映射模式、序号、包龄、tracking、Grip、Trigger 和重锚状态。

## 接入规划器

只需切换 Python 参数，不改 C++：

```bash
# 直接笛卡尔跟踪
uv run python scripts/coordinator.py --action-source pico --planner-mode direct

# baseline SQP
uv run python scripts/coordinator.py --action-source pico --planner-mode baseline_sqp

# shadow 容差修正 + baseline SQP
uv run python scripts/coordinator.py --action-source pico --planner-mode shadow_sqp
```

主要可调参数都在 coordinator 的 Python CLI：`pico_mapping_mode`、`pico_translation_scale`、`pico_rotation_scale`、Grip/Trigger 阈值、超时、坐标标定矩阵，以及现有全部 SQP、reference 和 1kHz tracker 参数。

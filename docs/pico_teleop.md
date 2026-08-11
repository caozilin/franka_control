# PICO 双手柄遥操作

PICO 端只负责发送左右手柄的 tracking pose 和按键状态。本项目通过 UDP 接收最新状态，以 10Hz 生成笛卡尔动作，再选择 `direct`、`baseline_sqp` 或 `shadow_sqp` 路径执行。

## 当前映射

- 使用右手柄的 grip pose（不是 aim/ray pose）控制末端六维增量。
- 左右 Grip 同时按下才允许运动；首次按下只重设锚点，不产生跳变。
- 松开任一 Grip 后停止运动，下次按下重新锚定。
- 右 Trigger 控制夹爪：超过阈值闭合，否则张开。
- Primary、Secondary 和摇杆随协议接收并保留，当前不绑定机器人行为。
- 超过 `pico_stale_timeout_s` 没有新包，或任一手柄丢失 tracking 时，不再生成动作。

PICO/Unity 通常以 60Hz 或 90Hz 发包。接收线程只保存最新包，coordinator 固定以 `control_hz=10` 消费，因此不会堆积旧位姿。

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
  --pico-bind-host 127.0.0.1 \
  --pico-port 9010 \
  --no-robot --no-cameras --no-use-gripper --no-startup-home
```

另一个终端发送静止数据：

```bash
uv run python examples/pico_udp_sender.py
```

加 `--enable-motion` 会让示例右手柄沿 x 方向往复运动。即使是离线 backend，也应先在 coordinator 页面执行 Start；状态接口 `/status` 的 `pico` 字段会显示序号、包龄、tracking、Grip、Trigger 和重锚状态。

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

主要可调参数都在 coordinator 的 Python CLI：`pico_translation_scale`、`pico_rotation_scale`、Grip/Trigger 阈值、超时、坐标标定矩阵，以及现有全部 SQP、reference 和 1kHz tracker 参数。

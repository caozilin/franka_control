# 统一笛卡尔遥操作与数采入口

键盘、PS4 手柄和 PICO 头显统一通过 `scripts/teleop.py` 进入同一套笛卡尔控制、规划、末端系旋转转换、夹爪和数据录制逻辑。

默认配置：

- `planner-mode=direct`
- `reference=linear`
- 单轴平移速度上限为 `0.1 m/s`（10 Hz 下每帧每轴最大 `0.01 m`）
- 碰撞阈值使用 Franka 官方示例：末端 XYZ `30 N` 报告接触、`40 N` 反射停止；RPY 为 `25/35 Nm`
- 键盘/PS4 默认按末端系旋转；可用 `--rotation-frame base` 改为绕 Base 固定 XYZ 轴旋转
- 启动时自动错误恢复并回零
- 相机关闭，不录制图像
- 控制进程固定到 CPU 2，避开当前 Franka 网口所在的 CPU 6 中断

## 启动

```bash
# 键盘
.venv/bin/python scripts/teleop.py --input-device keyboard

# PS4 手柄
.venv/bin/python scripts/teleop.py --input-device ps4

# PS4 手柄，绕 Base 固定 XYZ 轴旋转
.venv/bin/python scripts/teleop.py --input-device ps4 --rotation-frame base

# PS4 + Shadow SQP + 任务容差组合 T09
.venv/bin/python scripts/teleop.py \
  --input-device ps4 \
  --planner-mode shadow_sqp \
  --tolerance-id T09

# PICO；入口会同时管理 XRoboToolkit SDK bridge
.venv/bin/python scripts/teleop.py --input-device pico
```

容差 ID 与任务的对应关系见 [franka_tolerance_ids.md](franka_tolerance_ids.md)。启用后按 PS 键依次记录 Pre、Post 当前姿态；第三次起循环覆盖 Pre、Post。夹爪阶段按照 MuJoCo 的 Pre/Grasp/Post/Release 稳定开度规则切换，其中 Grasp/Release 为严格零容差。

需要数据采集时显式打开相机：

```bash
.venv/bin/python scripts/teleop.py \
  --input-device pico \
  --with-cameras \
  --task-name default
```

录制保存到 `collected/<task-name>/epo_N/`，包含 `cam1.mp4`、`cam2.mp4` 和 `data.json`。相机关闭时触发录制键只会给出提示，不会生成黑色视频。

## 录制按键

| 输入设备 | 开始 | 结束并保存 | 作废 |
| --- | --- | --- | --- |
| 键盘 | `1` | `2` | `3`（同时复位） |
| PS4 | `L3` | `R3` | `Cross`（同时复位） |
| PICO | `A` | 再按一次 `A` | `B` |

PICO 的 B 只作废当前片段，不保存，也不复位机械臂。A/B 均按上升沿处理，持续按住不会重复触发。

## 扶瓶自动后半程

手动抓紧瓶子后，按键盘 `M` 或 `N` 从当前姿态开始执行 MuJoCo Adjust Bottle 的抓取后流程：保持抓取姿态抬升、转移并转正、下降、停留、释放。腕部相机与瓶口在同一边时按 `N`，在相反两边时按 `M`；这个相对关系已同时包含瓶子世界朝向和腕部抓取分支，不再额外从当前末端姿态猜测 positive/negative 分支。N/M 对应的最终末端姿态相差 `180°`。左/右两个末端姿态候选使用同样的 IK、自碰撞、边可行性、可操作度和关节余量筛选。规划和执行时会屏蔽手动位移，但复位、作废和退出按键仍会轮询。
该功能只提供预设的笛卡尔动作序列，不改变、收紧或绕过当前的阶段容差；Direct/Baseline SQP/Shadow SQP 仍按启动时的现有设置执行。
序列使用 MuJoCo 的 10 Hz 时间参数化：单轴平移峰值 `0.1 m/s`，抬升/转移/下降保持非零边界速度连续衔接，下降用 `2 s` raised-cosine 减速。按标定姿态生成的标称序列约 `6.5 s`，其中停留和夹爪张开各 `1 s`。

当前真机标定的固定末端位置为 `(0.429286, 0.000000, 0.315028) m`，其中 x/z 来自标定时读取的真机末端位置，y 按要求固定为 0；姿态由 MuJoCo 同款 Base yaw `-90°/+90°` 候选决定。必须先关闭夹爪，否则 `M/N` 会拒绝启动。

## PICO 操作

- 左 Grip：按下时锚定左手柄位置；左手之后的相对位移映射到 Base XYZ 平移目标。
- 右 Grip：按下时锚定右手柄姿态；右手之后的相对姿态映射到末端系旋转目标，X/Y 互换、Z 保持不变，再转换为 Base 系增量执行。
- 左右均为相对位姿而非速度控制：手柄保持不动后，机械臂到达相对目标即停止；松开再按 Grip 会重新锚定。
- 右 Trigger：每按一次切换夹爪张开/闭合。
- A：开始录制；再次按 A 结束并保存。
- B：作废当前录制。

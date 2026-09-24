# Franka IPOPT 容差 ID

Txx 尽量保留原有 ID 与任务对应关系；MuJoCo 当前对直立圆柱的盘子与区域目标使用不同的 Post mask，因此新增 `T14`；窄盒放置任务使用 `T15`、`T16`。每组 mask 按容差坐标系的 `(Rx, Ry, Rz)` 排列：`1` 表示允许该轴在 `±30°` 内优化，`0` 表示严格约束。不同 ID 现在可能有相同 mask。长任务的子任务容差暂未纳入。

| ID | Pre mask | Post mask | 对应任务 ID |
| --- | --- | --- | --- |
| `T01` | `1 1 0` | `0 0 1` | `adjust_cylindrical_bottle` |
| `T02` | `0 1 0` | `0 0 1` | `adjust_rectangular_bottle` |
| `T03` | `1 1 1` | `1 1 1` | `click_bell`, `pear_to_bowl`, `pear_to_plate`, `press_power_strip` |
| `T04` | `0 1 1` | `0 0 1` | `close_cylindrical_pot_lid`, `geometry_region_cylinder_upright`, `open_cylindrical_pot_lid` |
| `T05` | `1 1 0` | `0 0 1` | `close_handle_pot_lid`, `open_handle_pot_lid` |
| `T06` | `1 1 0` | `1 1 1` | `banana_to_plate` |
| `T07` | `1 1 1` | `1 1 1` | `strawberry_to_bowl`, `strawberry_to_plate` |
| `T08` | `1 1 1` | `1 1 1` | `geometry_plate_ball` |
| `T09` | `0 1 0` | `1 1 1` | `geometry_plate_box_lying`, `geometry_plate_cube` |
| `T10` | `0 1 0` | `1 1 1` | `geometry_plate_box_upright` |
| `T11` | `1 1 0` | `1 1 1` | `geometry_plate_cylinder_lying` |
| `T12` | `0 1 0` | `0 0 0` | `geometry_region_box_lying`, `geometry_region_box_upright`, `geometry_region_cube`, `rotate_knob` |
| `T13` | `1 1 0` | `1 0 0` | `geometry_region_cylinder_lying` |
| `T14` | `0 1 1` | `1 1 1` | `geometry_plate_cylinder_upright` |
| `T15` | `1 1 0` | `1 1 0` | `cylinder_to_narrow_box` |
| `T16` | `0 1 0` | `0 1 0` | `box_to_narrow_box` |

旋转容差使用固定轴 XYZ/RPY 欧拉坐标 `[roll, pitch, yaw]`，组合矩阵为 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。每个 10 Hz 规划周期从动作积分后的当前标称末端姿态构造重力对齐容差系：Z 轴为世界竖直方向，Y 轴沿末端 Y 轴的水平投影。该周期的 IPOPT 求解与释放约束共用这一个固定容差系，不经过阶段目标容差系的 EMA。普通末端位姿及 Base 系旋转增量仍使用旋转向量。

启动示例：

```bash
.venv/bin/python scripts/teleop.py \
  --input-device ps4 \
  --planner-mode ipopt \
  --tolerance-id T09
```

启用后无需按 PS 键标注容差系或 Pre/Post 目标姿态。Pre/Post mask 随夹爪阶段自动切换，Grasp/Release 始终严格零容差。阶段由夹爪开合命令与连续 3 帧宽度稳定共同判定。各阶段的物理/标称 handoff 与 stage-relative 释放状态继续由同一 IPOPT 求解路径处理。

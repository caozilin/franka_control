# Franka SQP 容差 ID

数据来源：`90833e0f-ffaf-4fb7-983c-11ec2db5cc77.csv` 中 `robot_uid=panda` 的 25 条记录。
具有完全相同 Pre/Post 十二个边界值的任务共用一个 ID。

每组数值顺序均为：`Rx- Rx+ Ry- Ry+ Rz- Rz+`，单位为度。`-`/`+` 表示目标姿态两侧可接受的非对称容差幅值，并不是带符号的输入值。

| ID | Pre | Post | 对应任务 ID |
| --- | --- | --- | --- |
| `T01` | `30 30 30 10 0 0` | `0 0 0 0 45 45` | `adjust_cylindrical_bottle` |
| `T02` | `0 0 30 10 0 0` | `0 0 0 0 45 45` | `adjust_rectangular_bottle` |
| `T03` | `30 30 30 30 45 45` | `30 30 30 30 45 45` | `click_bell`, `pear_to_bowl`, `pear_to_plate`, `press_power_strip` |
| `T04` | `0 0 30 30 45 45` | `0 0 0 0 45 45` | `close_cylindrical_pot_lid`, `geometry_plate_cylinder_upright`, `geometry_region_cylinder_upright`, `open_cylindrical_pot_lid` |
| `T05` | `30 30 30 30 0 0` | `0 0 0 0 45 45` | `close_handle_pot_lid`, `open_handle_pot_lid` |
| `T06` | `10 10 0 0 0 0` | `30 30 30 30 45 45` | `banana_to_plate` |
| `T07` | `20 20 0 0 0 0` | `30 30 30 30 45 45` | `strawberry_to_bowl`, `strawberry_to_plate` |
| `T08` | `20 20 30 30 45 45` | `20 20 30 30 45 45` | `geometry_plate_ball` |
| `T09` | `0 0 30 30 0 0` | `30 30 30 30 45 45` | `geometry_plate_box_lying`, `geometry_plate_cube` |
| `T10` | `0 0 30 30 0 0` | `0 0 0 0 45 45` | `geometry_plate_box_upright` |
| `T11` | `5 5 0 0 0 0` | `20 20 30 30 45 45` | `geometry_plate_cylinder_lying` |
| `T12` | `0 0 30 30 0 0` | `0 0 0 0 0 0` | `geometry_region_box_lying`, `geometry_region_box_upright`, `geometry_region_cube`, `rotate_knob` |
| `T13` | `10 10 30 30 0 0` | `30 30 0 0 0 0` | `geometry_region_cylinder_lying` |

启动示例：

```bash
.venv/bin/python scripts/teleop.py \
  --input-device ps4 \
  --planner-mode shadow_sqp \
  --tolerance-id T09
```

启用后按 PS 键依次采集 Pre、Post 目标姿态，之后继续按 PS 键会依次覆盖 Pre、Post。

控制规则与 `franka_mujoco` 相同：Pre/Post 使用表中的非对称旋转范围，Grasp/Release 使用严格姿态；范围轴采用固定任务目标坐标、即时释放目标和意图 EMA，SQP 约束使用 `absolute_lower/absolute_upper`。阶段由夹爪开合命令与连续 3 帧宽度稳定共同判定。
运行阶段采用 MuJoCo 的四阶段夹爪规则：Pre/Post 使用表中对应容差；Grasp/Release 使用严格零容差。

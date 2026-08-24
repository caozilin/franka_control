# SQP planning modes

All Cartesian action sources now use `planning.CartesianActionPlanner` and the
same `FrankaEnv.enqueue_cartesian_action()` execution boundary. The planner is
selected before constructing `FrankaEnv`; input devices and policies do not
contain planner-specific branches.

- `direct`: Cartesian action to the existing Cartesian reference path.
- `baseline_sqp`: 10 Hz Cartesian action to baseline SQP, then an absolute
  joint waypoint to the 1 kHz joint reference and tracker.
- `shadow_sqp`: the same baseline SQP with an optimizer-independent shadow
  orientation correction before each solve.

Examples on the Linux workstation:

```bash
.venv/bin/python scripts/coordinator.py \
  --policy-type openpi \
  --planner-mode baseline_sqp

.venv/bin/python scripts/coordinator.py \
  --policy-type openpi \
  --planner-mode shadow_sqp \
  --rotation-ranged-axes False False True \
  --rotation-limits-deg 30 30 45

.venv/bin/python scripts/teleop.py \
  --input-device ps4 \
  --planner-mode shadow_sqp \
  --rotation-ranged-axes false false true \
  --rotation-limits-deg 30 30 45
```

The same planner selection is exposed by the unified `scripts/teleop.py`,
`scripts/example_trajectory_record_and_analyze.py`, and
`examples/move_forward_30cm.py`. Direct mode uses the selected Cartesian
`reference` (`min_jerk`, `linear`, `cubic`, or `motion_limited`) and the
internal `cartesian_impedance` tracker. Both SQP modes produce absolute joint targets;
their joint reference can independently select `min_jerk`, `linear`, or
`cubic`, followed by the bounded leaky internal `joint_pid` reference-correction
tracker plus joint impedance. Joint `motion_limited` is not implemented and is
rejected before control starts. The CLI exposes only `--tracker-mode {auto,pid}`.
Both names preserve the reference space and select its internal implementation:
`cartesian_impedance` for Cartesian references and `joint_pid` for joint references.
No Cartesian reference is converted into a joint reference. Every joint reference
segment spans exactly one low-frequency
control period (`1 / control_hz`).

The `Args` dataclass contains the SQP iteration/time/tolerance/trust-region/QP
settings, objective weights, shadow mask/frame/limits,
joint impedance/PID tracker gains, and torque-rate limit. They are passed from Python at
startup and do not require rebuilding C++. Session metadata records these
arguments (except the API key), while each action telemetry row records SQP
status, residuals, iteration counts, solve time, and shadow diagnostics.

The default SQP time budget is 95 ms because planning runs at 10 Hz. Reference
generation and torque tracking remain separate C++ components executed in the
same 1 kHz libfranka callback.

## Real-time SQP implementation

The real-robot planner keeps the MuJoCo baseline problem and solver semantics,
but evaluates the hard-coded Menagerie Panda chain through Pinocchio's C++
bindings in the robot base/O frame. A numerically equivalent pure-Python
kinematics implementation remains as an automatic fallback. Manipulability and
link-centre points are computed only when their objective weights require them.

The objective path mirrors the optimized simulation implementation:

- joint-limit loss gradients are analytic;
- kinematic states are shared by objective, gradient, and constraint queries at
  the same SQP iterate;
- finite-difference manipulability and self-collision gradients are reused
  inside one solve while the iterate remains within
  `SQPSettings.expensive_gradient_refresh_rad` (default `0.015` rad);
- all non-adjacent link-segment distances are evaluated in one NumPy batch.

On the development workstation, a deterministic 100-frame strict 6D Panda
comparison measured approximately 4.6 ms mean and 6.1 ms p95 for the real-robot
planner, versus approximately 6.4 ms mean for the matching MuJoCo controller.
These are offline solver timings rather than guarantees for every robot pose;
runtime telemetry remains the authoritative measurement for a live session.

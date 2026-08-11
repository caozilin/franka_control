# SQP planning modes

`scripts/coordinator.py` is the Python entrypoint for the existing OpenPI or
Cosmos client and all three execution modes:

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
```

The `Args` dataclass contains the SQP iteration/time/tolerance/trust-region/QP
settings, objective weights, shadow mask/frame/limits, joint min-jerk duration,
joint tracker gains, and torque-rate limit. They are passed from Python at
startup and do not require rebuilding C++. Session metadata records these
arguments (except the API key), while each action telemetry row records SQP
status, residuals, iteration counts, solve time, and shadow diagnostics.

The default SQP time budget is 95 ms because planning runs at 10 Hz. Reference
generation and torque tracking remain separate C++ components executed in the
same 1 kHz libfranka callback.

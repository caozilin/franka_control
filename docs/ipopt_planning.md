# IPOPT planning mode

All Cartesian action sources use `planning.CartesianActionPlanner` and the
same `FrankaEnv.enqueue_cartesian_action()` execution boundary.

- `direct`: Cartesian action to the Cartesian reference path.
- `ipopt`: 10 Hz action to the maintained single-step constrained IPOPT
  optimizer, then an absolute joint waypoint to the 1 kHz joint reference and
  tracker.

Example:

```bash
uv run python scripts/coordinator.py \
  --policy-type openpi \
  --planner-mode ipopt
```

## Maintained optimization problem

The implementation is ported from `franka_mujoco` revision
`5d78c7989b2b64aea71a792a1a20f400c0758ba5`. It optimizes the seven Panda
joint angles directly. Position and Specific rotation axes are hard equality
constraints; ranged rotation axes and joint limits are hard inequalities.
Every result is independently re-evaluated before publication.

The active secondary objective contains exactly:

- squared joint increment `Δq²`, default weight `1.0`;
- current-cycle incremental tolerance release scaled by nominal-action EMA,
  default weight `0.1`.

Acceleration and jerk are evaluation quantities, not optimizer losses. The
archived whole-stage Block optimizer and the former custom SQP/QP/shadow paths
are not present in this project.

The default limit is 20 iterations and 95 ms CPU time. A rejected solve holds
the last accepted joint command. Three consecutive rejected solves raise
`IpoptConsecutiveFailureError`; the calling control loop prints the exception
and requests a stop.

## Real-robot adaptation

The optimizer, objective, constraint validation, stage-relative release state,
and failure semantics match the simulator implementation. Only the kinematic
backend differs: the real project evaluates the Panda chain with Pinocchio
(with its existing NumPy fallback) in the robot O/base frame.

IPOPT remains outside the libfranka callback:

```text
10 Hz Cartesian action
  -> IpoptPlanner
  -> absolute 7D joint target
  -> bounded command queue
  -> 1 kHz joint reference + tracker + torque-rate limiter
```

## Dependency runtime

`cyipopt==1.7.0` is locked by uv. Native IPOPT and its numerical libraries
are installed locally under `.runtime/ipopt`, which is ignored by Git.
The installed extension contains an RPATH to that local runtime, so ordinary
`.venv/bin/python` and `uv run` invocations need no manual
`LD_LIBRARY_PATH`.

If the environment must be rebuilt on another workstation, install a native
IPOPT development package first or recreate the local runtime, then run
`uv sync --frozen --extra dev` with the matching pkg-config and linker paths.

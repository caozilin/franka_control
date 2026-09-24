# Realtime control pipeline

The Linux workstation executes reference generation, tracking, and safety in
one libfranka 1 kHz callback.  They are separate code components, not separate
threads:

```text
10 Hz action -> Planner -> typed ControlRoute
          |
          v
CartesianReferenceGenerator / JointReferenceGenerator
          |  CartesianReferenceSample / JointReferenceSample
          v
CartesianImpedanceTracker / JointImpedanceTracker / JointPidTracker
          |  desired joint torque
          v
TorqueRateLimiter
          |  safe joint torque
          v
libfranka robot.control() at 1 kHz
```

## Component contracts

- A reference generator owns trajectory time parameterization and continuity.
  It does not read tracker gains or calculate torque.
- A tracker consumes measured robot state and one immutable reference sample.
  It does not consume policy chunks or interpolate waypoints.
- A safety component consumes desired actuator commands.  It does not modify
  the planner target or the reference generator's internal trajectory.
- All three components execute sequentially against the same robot state and
  `franka::Duration`; no queue or scheduler is inserted between them.

The shared 10 Hz event reports `actual_plan_dxyz_m` / `actual_plan_drot_rad`
for every planner. IPOPT nominal-target residuals are not printed because the
nominal Cartesian target is enforced as a hard constraint.

Every realtime route publishes the same 1 kHz trace and timing records. For
joint-output planners, goal and reference poses are obtained from their joint
vectors with the libfranka model before recording.

Joint and Cartesian reference samples are intentionally different types.  A
joint reference can only be paired with a joint tracker, and a Cartesian
reference can only be paired with a Cartesian tracker.

Planner, reference profile, and tracker are selected independently. The Python
`ControlRoute` does not implement planning, interpolation, or torque control;
it only resolves `auto` and rejects cross-space combinations before hardware is
opened. Cartesian references support `min_jerk`, `linear`, `cubic`, and
`motion_limited`; joint references currently support `min_jerk`, `linear`, and
`cubic`.

## Non-realtime boundary

Python inference and IPOPT planning publish bounded low-frequency
commands through the backend command boundary.  They must never be called by a
reference generator or tracker.  If the non-realtime producer is delayed, the
reference generator finishes or holds its current safe target while the 1 kHz
pipeline continues to run.

The IPOPT path is therefore:

```text
Python Cartesian action -> IpoptPlanner at 10 Hz
                        -> absolute q target
                        -> FrankaEnv.enqueue_joint_target()
                        -> 1 kHz JointReferenceGenerator + tracker
```

Joint deltas and absolute IPOPT waypoints use distinct backend queues. This keeps
the existing joint teleoperation contract intact and prevents optimizer output
from being accumulated as a delta.

Rotational tolerance uses the same stage-relative release state as
`franka_mujoco`: each stage captures paired physical and nominal handoffs,
transports accepted spatial release with nominal motion, freezes one tolerance
chart per 10 Hz cycle, and commits state only after an independently validated
IPOPT result is publishable.
The chart frame comes from the action-integrated nominal EE rotation on every
10 Hz cycle: world Z stays upright and the tool Y heading is projected into
the horizontal plane. The frame is frozen for that cycle's solve.

## Linux verification

The C++ extension is built and exercised on the Franka workstation because it
depends on the workstation's libfranka 0.21.2 build and realtime network
interface.  The minimum verification sequence is:

```bash
source .venv/bin/activate
cmake -S . -B build -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j2
.venv/bin/python examples/zero_action_hold_diagnostic.py --ip 172.16.0.2
```

Active-motion verification must only follow a successful zero-action hold and
requires an available user stop.

## Python-side tuning

Experiment parameters are passed when constructing `FrankaEnv`; changing them
does not require rebuilding the C++ extension. This includes `control_hz`,
Cartesian and joint stiffness/damping, reference limits and convergence
thresholds, torque-rate limits, collision thresholds,
nullspace settings, and gripper command tolerances. C++ keeps only numerical
constants and Franka hardware limits fixed.

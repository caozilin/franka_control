# Gripper driver

`devices.AsyncGripperDriver` owns the only `RealtimeGripperBackend` instance for
the gripper. Its worker polls `read_once()` at 10 Hz while a separate command
thread may block in `move` or `grasp` on that same backend.

The driver exposes a latest-value command contract:

- `set_target(width)` replaces the desired target instead of building an
  unbounded queue.
- An active command is never interrupted by a newer target.
- When it completes, only the newest target is dispatched next.
- `width` is a cached observation and never waits for a gripper network read.

The C++ `read_once()` binding releases the Python GIL during network I/O, so
camera, policy, and orchestration threads are not stalled by the state poll.

On Linux, verify concurrent `read_once()` plus `move`/`grasp` against the
installed libfranka version before deploying on hardware.

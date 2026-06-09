# franka_control

Python high-level Franka control interface with an in-process C++ realtime backend.

The Python layer runs policy/input updates at 10Hz and sends actions into a pybind11
extension. The C++ backend owns the libfranka connection and realtime control loop;
the control loop does not call back into Python.

## Python Environment

Recommended: use `uv` and one dedicated virtual environment for this project.

```bash
cd /home/k324/franka_my_code/franka_control
$HOME/.local/bin/uv venv --python 3.12
source .venv/bin/activate
$HOME/.local/bin/uv pip install -U pip setuptools wheel pybind11 cmake
$HOME/.local/bin/uv pip install -e .
```

Build the C++ extension after building libfranka 0.21.2:

```bash
cd /home/k324/franka_my_code/franka_control
source .venv/bin/activate
cmake -S /home/k324/franka_my_code/libfranka-0.21.2 -B /home/k324/franka_my_code/libfranka-0.21.2/build-openrobots -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DGENERATE_PYLIBFRANKA=OFF
cmake --build /home/k324/franka_my_code/libfranka-0.21.2/build-openrobots -j2
cmake -S . -B build -DPYTHON_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build -j2
```

## Minimal Robot Check

The shortest current robot validation moves the end effector in base-frame +X by
30cm. It uses `FrankaEnv`; Python sends one action every 0.1s, and C++ emits the
1kHz Cartesian impedance torque command stream. Supported C++ reference profiles
are `min_jerk`, `linear`, and `cubic`.

```bash
.venv/bin/python examples/move_forward_30cm.py --ip 172.16.0.2
```

Use connect-only mode to verify the C++ extension can connect without starting
motion:

```bash
.venv/bin/python examples/move_forward_30cm.py --ip 172.16.0.2 --yes --connect-only
```

Keep the user stop button available before running active motion.

## 6s Trajectory Reproduction

This reproduces the 60-tick, 10Hz trajectory through the same `FrankaEnv` and C++
backend path. `--scale` scales the Cartesian translation and rotation deltas before
they are converted to normalized actions.

```bash
.venv/bin/python scripts/example_trajectory_record_and_analyze.py --ip 172.16.0.2 --controller linear --scale 1.0
```

Dry-run the Python/logging path without connecting to the robot:

```bash
.venv/bin/python scripts/example_trajectory_record_and_analyze.py --no-robot --yes --scale 1.0 --settle 0.2
```

Use `--controller min_jerk`, `--controller linear`, or `--controller cubic` to
select the C++ realtime reference profile.

By default, robot scripts call C++ joint-position reset first. Use
`--no-home-first` only when you explicitly want to start from the current pose.

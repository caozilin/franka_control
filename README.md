# franka_control

Ubuntu 24.04 workspace for running official `pylibfranka` examples first.

## Python Environment

Recommended: use `uv` and one dedicated virtual environment for this project.

```bash
cd /home/k324/franka_my_code/franka_control
$HOME/.local/bin/uv venv --python 3.12
source .venv/bin/activate
$HOME/.local/bin/uv pip install -U pip setuptools wheel
$HOME/.local/bin/uv pip install -e .
```

If PyPI does not provide the exact `pylibfranka` version you need, install from the cloned source:

```bash
cd /home/k324/franka_my_code/franka_control
source .venv/bin/activate
$HOME/.local/bin/uv pip install -U pip setuptools wheel pybind11 cmake patchelf build
$HOME/.local/bin/uv pip install /home/k324/franka_my_code/pylibfranka-libfranka-0.21.2
$HOME/.local/bin/uv pip install -e .
```

## Official Examples

Current tested environment:

```text
pylibfranka==0.21.2
numpy==2.4.6
Python 3.12
```

The official Python examples are copied under:

```text
examples/pylibfranka/
```

Robot IP extracted from `vla4desk`: `172.16.0.2`. All copied examples default to this IP, while `--ip` can still override it.

Read-only state example:

```bash
python examples/pylibfranka/print_robot_state.py --count 1
```

Motion examples can move the robot. Keep the user stop button available and start with the official read-only state example before running active control scripts.


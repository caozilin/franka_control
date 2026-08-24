from __future__ import annotations

import argparse

import numpy as np


def parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected 'true' or 'false'")


def parse_joint_vector(value: str) -> np.ndarray:
    try:
        parts = [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--nullspace-q-target must contain 7 comma-separated joint values"
        ) from exc
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(
            "--nullspace-q-target must contain 7 comma-separated joint values"
        )
    return np.asarray(parts, dtype=np.float64)

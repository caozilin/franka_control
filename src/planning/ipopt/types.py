from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetPose:
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError("Target pose requires a 3-vector and a 3x3 rotation")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            raise ValueError("Target pose must be finite")
        object.__setattr__(self, "position", position.copy())
        object.__setattr__(self, "rotation", rotation.copy())

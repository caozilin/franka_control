from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np


PICO_PROTOCOL_VERSION = 1


def _vector(value: object, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector.copy()


@dataclass(frozen=True)
class PicoControllerState:
    position: np.ndarray
    orientation_xyzw: np.ndarray
    grip: float
    trigger: float
    thumbstick: np.ndarray
    primary: bool
    secondary: bool
    tracked: bool

    @classmethod
    def from_dict(cls, value: dict) -> "PicoControllerState":
        orientation = _vector(value["orientation_xyzw"], 4, "orientation_xyzw")
        norm = float(np.linalg.norm(orientation))
        if norm < 1e-12:
            raise ValueError("orientation_xyzw must be a valid quaternion")
        return cls(
            position=_vector(value["position"], 3, "position"),
            orientation_xyzw=orientation / norm,
            grip=float(value.get("grip", 0.0)),
            trigger=float(value.get("trigger", 0.0)),
            thumbstick=_vector(value.get("thumbstick", [0.0, 0.0]), 2, "thumbstick"),
            primary=bool(value.get("primary", False)),
            secondary=bool(value.get("secondary", False)),
            tracked=bool(value.get("tracked", False)),
        )

    def as_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "orientation_xyzw": self.orientation_xyzw.tolist(),
            "grip": self.grip,
            "trigger": self.trigger,
            "thumbstick": self.thumbstick.tolist(),
            "primary": self.primary,
            "secondary": self.secondary,
            "tracked": self.tracked,
        }


@dataclass(frozen=True)
class PicoPacket:
    sequence: int
    timestamp_s: float
    left: PicoControllerState
    right: PicoControllerState
    session_id: str = "default"
    version: int = PICO_PROTOCOL_VERSION

    @classmethod
    def from_dict(cls, value: dict) -> "PicoPacket":
        version = int(value.get("version", -1))
        if version != PICO_PROTOCOL_VERSION:
            raise ValueError(f"unsupported PICO protocol version: {version}")
        return cls(
            sequence=int(value["sequence"]),
            timestamp_s=float(value["timestamp_s"]),
            left=PicoControllerState.from_dict(value["left"]),
            right=PicoControllerState.from_dict(value["right"]),
            session_id=str(value.get("session_id", "default")),
            version=version,
        )

    @classmethod
    def from_json(cls, payload: bytes | str) -> "PicoPacket":
        return cls.from_dict(json.loads(payload))

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "timestamp_s": self.timestamp_s,
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
        }

    def to_json(self) -> bytes:
        return json.dumps(self.as_dict(), separators=(",", ":")).encode("utf-8")

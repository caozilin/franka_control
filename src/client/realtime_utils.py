from __future__ import annotations

import base64
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def encode_jpeg_b64(img: np.ndarray) -> str:
    if cv2 is None:
        raise ImportError("cv2 is required for JPEG encoding; install opencv-python")
    arr = coerce_rgb_frame(img)
    if arr is None:
        raise ValueError("Cannot encode empty image")
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return base64.b64encode(buf).decode("ascii")


def extract_first_present(data: dict, paths: tuple[str, ...]) -> Any:
    for path in paths:
        cur = data
        found = True
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                found = False
                break
            cur = cur[part]
        if found and cur is not None:
            return cur
    return None


def coerce_rgb_frame(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    arr = np.squeeze(arr)
    while arr.ndim > 3:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim != 3:
        return None
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] != 3:
        return None
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32, copy=False)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        min_value = float(finite.min())
        max_value = float(finite.max())
        if -1.0 <= min_value and max_value <= 1.0:
            arr = (arr + 1.0) * 127.5 if min_value < 0.0 else arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def shape_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    try:
        return [int(dim) for dim in np.asarray(value).shape]
    except Exception:
        return None


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

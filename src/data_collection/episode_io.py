from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np


@dataclass
class Episode:
    frames1: list[np.ndarray]
    frames2: list[np.ndarray]
    data: list[dict]
    task_name: str
    collect_hz: float
    max_frames: int
    action_scale: float
    prompt: str


class EpisodeWriter:
    """Write the existing cam1.mp4, cam2.mp4 and data.json layout."""

    def __init__(self, collection_dir: Path) -> None:
        self.collection_dir = Path(collection_dir)

    def _next_dir(self, task_name: str) -> Path:
        task_dir = self.collection_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        numbers = []
        for path in task_dir.iterdir():
            if path.is_dir() and path.name.startswith("epo_"):
                try:
                    numbers.append(int(path.name[4:]))
                except ValueError:
                    pass
        episode_dir = task_dir / f"epo_{max(numbers, default=0) + 1}"
        episode_dir.mkdir()
        return episode_dir

    def save(self, episode: Episode) -> Path:
        episode_dir = self._next_dir(episode.task_name)
        try:
            imageio.mimwrite(
                str(episode_dir / "cam1.mp4"), episode.frames1,
                fps=episode.collect_hz, codec="libx264", pixelformat="yuv420p",
            )
            imageio.mimwrite(
                str(episode_dir / "cam2.mp4"), episode.frames2,
                fps=episode.collect_hz, codec="libx264", pixelformat="yuv420p",
            )
            print(f"  [保存] 视频: {episode_dir}")
        except Exception as exc:
            logging.error("视频保存失败: %s", exc)

        metadata = {
            "task_name": episode.task_name,
            "collect_hz": episode.collect_hz,
            "max_frames": episode.max_frames,
            "num_frames": len(episode.frames1),
            "action_scale": episode.action_scale,
            "prompt": episode.prompt,
            "frames": episode.data,
        }
        json_path = episode_dir / "data.json"
        with json_path.open("w", encoding="utf-8") as output_file:
            json.dump(metadata, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
        print(f"  [保存] JSON: {json_path} ({len(episode.data)} 帧)")
        return episode_dir

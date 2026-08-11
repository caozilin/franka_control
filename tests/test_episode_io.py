from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_collection.episode_io import Episode, EpisodeWriter


def test_episode_writer_keeps_existing_layout(tmp_path, monkeypatch) -> None:
    written_videos = []
    monkeypatch.setattr("data_collection.episode_io.imageio.mimwrite", lambda path, *_args, **_kwargs: written_videos.append(path))
    writer = EpisodeWriter(tmp_path)
    episode = Episode(
        frames1=[np.zeros((2, 2, 3), dtype=np.uint8)],
        frames2=[np.zeros((2, 2, 3), dtype=np.uint8)],
        data=[{"id": 0, "action": [1.0] * 7}],
        task_name="pick",
        collect_hz=10.0,
        max_frames=3000,
        action_scale=100.0,
        prompt="pick object",
    )

    episode_dir = writer.save(episode)
    assert episode_dir == tmp_path / "pick" / "epo_1"
    assert [Path(path).name for path in written_videos] == ["cam1.mp4", "cam2.mp4"]
    metadata = json.loads((episode_dir / "data.json").read_text(encoding="utf-8"))
    assert metadata["task_name"] == "pick"
    assert metadata["num_frames"] == 1
    assert metadata["frames"] == episode.data

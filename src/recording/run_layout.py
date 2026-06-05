from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    trace_csv: Path
    metadata_json: Path
    summary_json: Path
    plot_svg: Path


def create_run_paths(log_root: Path, controller_name: str) -> RunPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_root) / f"{timestamp}_{controller_name}"
    return RunPaths(
        run_dir=run_dir,
        trace_csv=run_dir / "pose_tracks.csv",
        metadata_json=run_dir / "metadata.json",
        summary_json=run_dir / "summary.json",
        plot_svg=run_dir / "pose_tracks.svg",
    )

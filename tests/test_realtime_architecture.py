from __future__ import annotations

import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
CPP = ROOT / "src" / "control" / "cpp"


def test_realtime_reference_tracker_and_safety_are_separate_components() -> None:
    expected = (
        CPP / "reference" / "cartesian_reference.hpp",
        CPP / "reference" / "joint_reference.hpp",
        CPP / "tracker" / "cartesian_impedance_tracker.hpp",
        CPP / "tracker" / "joint_impedance_tracker.hpp",
        CPP / "safety" / "torque_rate_limiter.hpp",
    )
    assert all(path.is_file() for path in expected)

    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    assert "reference.sample(" in backend
    assert "tracker.compute(" in backend
    assert "safety.apply(" in backend
    assert "updateMotionLimitedReference" not in backend
    assert "startNewSegment" not in backend


def test_realtime_ring_publishes_head_after_frame_write() -> None:
    backend = (CPP / "franka_backend.cpp").read_text(encoding="utf-8")
    timing = (CPP / "utils" / "timing.hpp").read_text(encoding="utf-8")
    for source in (backend, timing):
        write_index = source.index("ring_[idx % capacity_] = frame;")
        publish_index = source.index("write_head_.store(idx + 1, std::memory_order_release);", write_index)
        assert write_index < publish_index

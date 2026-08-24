from __future__ import annotations

import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_collection.key_control import KeyboardController  # noqa: E402
from data_collection.data_recorder import DataRecorder  # noqa: E402
from devices.pico import PicoControllerState, PicoPacket, PicoSnapshot  # noqa: E402


def _controller(*, primary: bool = False, secondary: bool = False) -> PicoControllerState:
    return PicoControllerState(
        position=np.zeros(3),
        orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        grip=0.0,
        trigger=0.0,
        thumbstick=np.zeros(2),
        primary=primary,
        secondary=secondary,
        tracked=True,
    )


def _snapshot(sequence: int, *, a: bool = False, b: bool = False) -> PicoSnapshot:
    packet = PicoPacket(
        sequence=sequence,
        timestamp_s=sequence / 90.0,
        left=_controller(),
        right=_controller(primary=a, secondary=b),
        session_id="unified-teleop-test",
    )
    return PicoSnapshot(packet, 10.0)


def test_pico_a_toggles_record_save_and_b_discards() -> None:
    controller = KeyboardController(
        input_device="pico",
        pico_port=0,
        no_robot=True,
        no_cameras=True,
    )
    events: list[str] = []

    def start() -> None:
        events.append("start")
        controller.set_recording_active(True)

    def stop() -> None:
        events.append("stop")
        controller.set_recording_active(False)

    def discard() -> None:
        events.append("discard")
        controller.set_recording_active(False)

    controller.bind_event("record_start", start)
    controller.bind_event("record_stop", stop)
    controller.bind_event("record_discard", discard)

    controller._handle_pico_recording_buttons(_snapshot(1, a=True))
    controller._handle_pico_recording_buttons(_snapshot(2, a=True))
    controller._handle_pico_recording_buttons(_snapshot(3))
    controller._handle_pico_recording_buttons(_snapshot(4, a=True))
    controller._handle_pico_recording_buttons(_snapshot(5))
    controller._handle_pico_recording_buttons(_snapshot(6, a=True, b=True))

    assert events == ["start", "stop", "discard"]


def test_all_cartesian_inputs_share_controller_class() -> None:
    controllers = [
        KeyboardController(input_device="keyboard", no_robot=True, no_cameras=True),
        KeyboardController(input_device="pico", pico_port=0, no_robot=True, no_cameras=True),
    ]
    assert all(controller.planner_mode == "direct" for controller in controllers)
    assert all(controller.reference_name == "linear" for controller in controllers)


def test_recording_is_disabled_when_cameras_are_off(tmp_path: pathlib.Path) -> None:
    controller = KeyboardController(input_device="pico", pico_port=0, no_robot=True, no_cameras=True)
    recorder = DataRecorder(controller, collection_dir=tmp_path)

    recorder.start_recording()

    assert recorder.is_recording is False
    assert controller.recording_active is False

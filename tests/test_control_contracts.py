from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from control.contracts import (  # noqa: E402
    CartesianDeltaCommand,
    ControlRates,
    CoordinateFrame,
    JointDeltaCommand,
    PolicyActionSpec,
)


def test_policy_action_spec_decodes_existing_openpi_scale() -> None:
    action = np.array([1.0, -2.0, 0.5, 3.0, -4.0, 5.0, 1.0])
    command = PolicyActionSpec().decode_cartesian(action)

    np.testing.assert_allclose(command.translation_m, [0.01, -0.02, 0.005])
    np.testing.assert_allclose(command.rotation_vector_rad, [0.03, -0.04, 0.05])
    assert command.gripper_command == 1.0
    assert command.frame is CoordinateFrame.BASE
    np.testing.assert_allclose(
        command.as_vector(),
        [0.01, -0.02, 0.005, 0.03, -0.04, 0.05, 1.0],
        atol=0.0,
    )
    assert command.as_vector()[6] == 1.0


def test_command_contracts_copy_input_vectors() -> None:
    source = np.arange(7, dtype=np.float64)
    command = CartesianDeltaCommand.from_vector(source)
    source[:] = -1.0

    np.testing.assert_allclose(command.as_vector(), np.arange(7, dtype=np.float64))

    joint = JointDeltaCommand.from_vector(np.arange(8, dtype=np.float64))
    np.testing.assert_allclose(joint.as_vector(), np.arange(8, dtype=np.float64))


@pytest.mark.parametrize(
    "factory,value",
    [
        (CartesianDeltaCommand.from_vector, np.zeros(6)),
        (CartesianDeltaCommand.from_vector, np.full(7, np.nan)),
        (JointDeltaCommand.from_vector, np.zeros(7)),
        (JointDeltaCommand.from_vector, np.full(8, np.inf)),
    ],
)
def test_command_contracts_reject_invalid_vectors(factory, value) -> None:
    with pytest.raises(ValueError):
        factory(value)


def test_control_rates_expose_explicit_periods() -> None:
    rates = ControlRates(policy_hz=10.0, planner_hz=20.0, servo_hz=1000.0)
    assert rates.policy_period_s == 0.1
    assert rates.servo_period_s == 0.001

    with pytest.raises(ValueError):
        ControlRates(policy_hz=2000.0, servo_hz=1000.0)

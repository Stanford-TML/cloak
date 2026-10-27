from unittest.mock import Mock
import pytest
from scripts import check_umi_gripper as check

def test_waits_through_transit_and_requires_settled_samples(monkeypatch):
    positions = iter([.75, .68, .2, .01, .01, .01])
    robot = Mock()
    robot.get_robot_state.side_effect = lambda: ({"gripper_position": next(positions)}, {})
    monkeypatch.setattr(check.time, "sleep", lambda _: None)
    assert check.wait_for_position(robot, 0.) == .01
    assert robot.get_robot_state.call_count == 6
    robot.update_gripper.assert_not_called()

def test_timeout_does_not_issue_another_command():
    robot = Mock()
    with pytest.raises(RuntimeError, match="did not settle"):
        check.wait_for_position(robot, 0., timeout=0)
    robot.update_gripper.assert_not_called()

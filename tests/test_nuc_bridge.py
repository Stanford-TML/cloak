"""Exercise process selection and failure propagation without Polymetis or hardware."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock
import numpy as np
import pytest


@pytest.fixture
def bridge(monkeypatch):
    for name in (
        "torch",
        "polymetis",
        "droid",
        "droid.misc",
        "droid.misc.subprocess_utils",
        "droid.misc.transformations",
        "droid.robot_ik",
        "droid.robot_ik.robot_ik_solver",
    ):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["torch"].Tensor = np.asarray
    sys.modules["polymetis"].GripperInterface = Mock()
    sys.modules["polymetis"].RobotInterface = Mock()
    sys.modules["droid.misc.subprocess_utils"].run_threaded_command = lambda f: f()
    for name in ("add_poses", "euler_to_quat", "pose_diff", "quat_to_euler"):
        setattr(sys.modules["droid.misc.transformations"], name, Mock())
    sys.modules["droid.robot_ik.robot_ik_solver"].RobotIKSolver = Mock()
    path = Path(__file__).resolve().parents[1] / "deployment/nuc/robot.py"
    spec = importlib.util.spec_from_file_location("nuc_bridge_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sharpa_does_not_spawn_gripper(bridge, monkeypatch):
    popen = Mock()
    monkeypatch.setattr(bridge.subprocess, "Popen", popen)
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    robot = bridge.FrankaRobot()
    robot.launch_controller("robotiq_2f", False)
    assert popen.call_count == 1
    assert robot._gripper_process is None


def test_impedance_start_is_bounded_and_failure_surfaces(bridge, monkeypatch):
    robot = bridge.FrankaRobot()
    robot._robot = Mock()
    robot._robot.is_running_policy.return_value = False
    robot._controller_not_loaded = False
    tick = iter(range(0, 1000, 6))
    monkeypatch.setattr(bridge.time, "monotonic", lambda: next(tick))
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    robot.update_joints([0] * 7, blocking=False)
    assert robot._robot.start_cartesian_impedance.call_count == 3
    with pytest.raises(RuntimeError, match="three attempts"):
        robot.get_robot_state()
    robot._robot.update_desired_joint_positions.assert_not_called()


def test_interrupted_policy_is_not_restarted(bridge):
    robot = bridge.FrankaRobot()
    robot._robot = Mock()
    robot._robot.is_running_policy.return_value = False
    robot._controller_not_loaded = False
    robot._impedance_started = True
    robot.update_joints([0] * 7, blocking=False)
    robot._robot.start_cartesian_impedance.assert_not_called()
    robot._robot.update_desired_joint_positions.assert_not_called()
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        robot.get_robot_state()


def test_gripper_metadata_waits_for_homing(bridge, monkeypatch):
    from types import SimpleNamespace
    responses = iter([SimpleNamespace(max_width=0.), SimpleNamespace(max_width=.0808)])
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    assert bridge.wait_gripper_metadata(lambda: next(responses)).max_width == .0808


def test_gripper_metadata_timeout_rejects_fallback(bridge):
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        bridge.wait_gripper_metadata(lambda: SimpleNamespace(max_width=0.), timeout=0)


def test_native_contact_uses_tested_grasp_once_per_transition(bridge):
    robot = bridge.FrankaRobot()
    robot._gripper = Mock()
    robot._gripper_kind = "franka_hand"
    robot._max_gripper_width = .08
    robot.update_gripper(.9, velocity=False)
    robot.update_gripper(.9, velocity=False)
    robot._gripper.grasp.assert_called_once_with(speed=.03, force=20., grasp_width=0.,
        epsilon_inner=0., epsilon_outer=.08, blocking=False)
    robot.update_gripper(0., velocity=False)
    robot._gripper.goto.assert_called_once_with(width=.08, blocking=False, speed=.03, force=20.)

def test_robotiq_retains_position_commands(bridge):
    robot = bridge.FrankaRobot()
    robot._gripper = Mock()
    robot._gripper_kind = "robotiq_2f"
    robot._max_gripper_width = .085
    robot.update_gripper(.5, velocity=False)
    robot._gripper.grasp.assert_not_called()
    assert robot._gripper.goto.call_args.kwargs["width"] == .0425

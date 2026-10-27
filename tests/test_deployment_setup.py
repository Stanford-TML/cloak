from unittest.mock import Mock
import numpy as np
import pytest

from deployment.calibration import calibration_utils
from deployment.misc.server_interface import ServerInterface
from deployment.robot_env import RobotEnv


def test_missing_calibration_prevents_controller_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration_utils, "dir_path", str(tmp_path))
    controller = Mock(side_effect=AssertionError("Hardware must not be initialized"))
    monkeypatch.setattr("deployment.robot_env.ServerInterface", controller)
    with pytest.raises(FileNotFoundError):
        RobotEnv(hand_camera_id="123", debug=True)
    controller.assert_not_called()


def test_sharpa_launch_disables_parallel_gripper():
    client = ServerInterface.__new__(ServerInterface)
    client.server = Mock()
    client.use_gripper = False
    client.launch_controller("robotiq_2f")
    client.server.launch_controller.assert_called_once_with("robotiq_2f", False)


def test_umi_reset_pose_and_actions(tmp_path, monkeypatch):
    from deployment.franka_umi_env import FrankaUMIEnv
    from openpi.constants import UMI_ARM_QPOS

    monkeypatch.setattr(calibration_utils, "dir_path", str(tmp_path))
    calibration_utils.save_calibration_info("umi", "123", [0] * 6, initial_pose=[0] * 6)
    env = FrankaUMIEnv(do_reset=False, debug=True, hand_camera_id="123")
    np.testing.assert_allclose(env.reset_joints, UMI_ARM_QPOS)
    assert env._robot.gripper_kind == "franka_hand"
    env._robot.update_command = Mock(return_value={})
    env.step(np.r_[np.zeros(7), 0.8])
    assert env._robot.update_command.call_args.args[0][-1] == 0.9
    with pytest.raises(ValueError):
        env.step(np.zeros(29))

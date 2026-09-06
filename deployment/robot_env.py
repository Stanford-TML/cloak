import time
from copy import deepcopy

import gym
import numpy as np

from deployment.calibration.calibration_utils import load_calibration_info
from openpi.constants import DROID_ARM_QPOS
from deployment.misc.parameters import nuc_ip
from deployment.misc.server_interface import ServerInterface
from deployment.misc.time import time_ms


class _NoCameraReader:
    """Minimal camera reader for replay-only mode when cameras are not needed."""
    camera_dict = {}

    def read_cameras(self):
        return {}, {}


class RobotEnv(gym.Env):
    def __init__(self, action_space="joint_position", gripper_action_space=None, camera_kwargs={}, do_reset=True, use_cameras=True, use_gripper=True, gripper_kind="robotiq_2f", calibration_name="robotiq", hand_camera_id=None, debug=False, debug_camera_ids=None):
        # Initialize Gym Environment
        super().__init__()

        # Debug/mock mode: ServerInterface + MultiCameraWrapper return default
        # state / blank images, so the client↔server loop runs with no hardware.
        self.debug = debug

        # Serial number of the wrist ("hand") camera — the single source of truth,
        # passed down from the deploy script. Used by get_camera_extrinsics to pick
        # which calibration entry gets the gripper-relative extrinsic treatment.
        self.hand_camera_id = hand_camera_id

        # Define Action Space #
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity"]
        self.action_space = action_space
        self.gripper_action_space = gripper_action_space
        self.check_action_range = "velocity" in action_space

        # Robot Configuration
        # DROID_ARM_QPOS is imported from openpi.constants (the single source of
        # truth shared with the server, which seeds its IK warm-start from it).
        self.reset_joints = DROID_ARM_QPOS
        self.randomize_low = np.array([-0.1, -0.2, -0.1, -0.3, -0.3, -0.3])
        self.randomize_high = np.array([0.1, 0.2, 0.1, 0.3, 0.3, 0.3])
        self.DoF = 7 if ("cartesian" in action_space) else 8

        if nuc_ip is None and not debug:
            from franka.robot import FrankaRobot

            self._robot = FrankaRobot(use_gripper=use_gripper)
        else:
            self._robot = ServerInterface(
                ip_address=nuc_ip, use_gripper=use_gripper, gripper_kind=gripper_kind, debug=debug
            )

        # Create Cameras. Import lazily so a real (non-debug) run without the ZED
        # SDK still fails loudly rather than at import time.
        if use_cameras or debug:
            from deployment.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper

            self.camera_reader = MultiCameraWrapper(
                camera_kwargs, debug=debug, debug_camera_ids=debug_camera_ids or [hand_camera_id]
            )
        else:
            self.camera_reader = _NoCameraReader()
        self.calibration_dict = load_calibration_info(name=calibration_name)

        # Reset Robot
        if do_reset:
            self.reset()

    def step(self, action):
        # Check Action
        assert len(action) == self.DoF
        if self.check_action_range:
            assert (action.max() <= 1) and (action.min() >= -1)

        # Update Robot
        action_info = self.update_robot(
            action,
            action_space=self.action_space,
            gripper_action_space=self.gripper_action_space,
        )

        # Return Action Info
        return action_info

    def cycle_gripper(self) -> bool:
        """Visible close → open cycle so the operator can confirm the gripper
        responds before a rollout (the 'g' option in the startup prompts).
        Robotiq position space: 0 = open, 1 = closed; final state = open.
        FrankaUMIEnv overrides this with hardware-specific failure detection,
        and FrankaSharpaEnv overrides it to close→open the 22-DOF hand."""
        print("[RobotEnv] cycle gripper: close → open")
        self._robot.update_gripper(1, velocity=False, blocking=True)
        time.sleep(0.6)
        self._robot.update_gripper(0, velocity=False, blocking=True)
        time.sleep(0.6)
        return True

    def reset(self, randomize=False):
        self._robot.update_gripper(0, velocity=False, blocking=True)

        if randomize:
            noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high)
        else:
            noise = None

        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)

    def update_robot(self, action, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
        action_info = self._robot.update_command(
            action,
            action_space=action_space,
            gripper_action_space=gripper_action_space,
            blocking=blocking
        )
        return action_info

    def read_cameras(self):
        return self.camera_reader.read_cameras()

    def get_state(self):
        read_start = time_ms()
        state_dict, timestamp_dict = self._robot.get_robot_state()
        timestamp_dict["read_start"] = read_start
        timestamp_dict["read_end"] = time_ms()
        return state_dict, timestamp_dict

    def get_camera_extrinsics(self, state_dict):
        # Expose the wrist camera's static cam-to-gripper calibration under the
        # '_gripper_offset' suffix the server expects. The server FK-composes
        # cam-to-base itself from joint_position, so we no longer derive it here.
        extrinsics = deepcopy(self.calibration_dict)
        for cam_id in self.calibration_dict:
            if self.hand_camera_id not in cam_id:
                continue
            extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
        return extrinsics

    def get_observation(self):
        obs_dict = {"timestamp": {}}

        # Robot State #
        state_dict, timestamp_dict = self.get_state()
        obs_dict["robot_state"] = state_dict
        obs_dict["timestamp"]["robot_state"] = timestamp_dict

        # Camera Readings #
        camera_obs, camera_timestamp = self.read_cameras()
        obs_dict.update(camera_obs)
        obs_dict["timestamp"]["cameras"] = camera_timestamp

        # Camera Info #
        extrinsics = self.get_camera_extrinsics(state_dict)
        obs_dict["camera_extrinsics"] = extrinsics

        intrinsics = {}
        for cam in self.camera_reader.camera_dict.values():
            cam_intr_info = cam.get_intrinsics()
            for (full_cam_id, info) in cam_intr_info.items():
                intrinsics[full_cam_id] = info["cameraMatrix"]
        obs_dict["camera_intrinsics"] = intrinsics

        return obs_dict

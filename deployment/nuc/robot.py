# ROBOT SPECIFIC IMPORTS
import os
import socket
import signal
import subprocess
import time

import grpc
import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface

from droid.misc.subprocess_utils import run_threaded_command

from droid.misc.transformations import add_poses, euler_to_quat, pose_diff, quat_to_euler
from droid.robot_ik.robot_ik_solver import RobotIKSolver

# Per-gripper goto kwargs (speed in m/s, force in N).
# Robotiq values match the historical defaults; franka_hand values are tuned
# for the native franka_gripper with UMI fingertips (~20-40 N typical range).
GRIPPER_GOTO_KWARGS = {
    "robotiq_2f": {"speed": 0.05, "force": 0.1},
    "franka_hand": {"speed": 0.03, "force": 20.0},
}

def wait_gripper_metadata(fetch, timeout=10.0):
    """Wait for homing to publish a real stroke width; never normalize with a guess."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            metadata = fetch()
            width = float(metadata.max_width)
            if np.isfinite(width) and width > 0:
                return metadata
        except grpc.RpcError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Gripper metadata unavailable after homing; controller startup stopped")
        time.sleep(0.1)


# UTILITY SPECIFIC IMPORTS


class FrankaRobot:
    def __init__(self, use_gripper=True):
        self.use_gripper = use_gripper
        self._motion_error = None
        self._impedance_started = False

    def launch_controller(self, gripper_kind="robotiq_2f", use_gripper=None):
        if gripper_kind not in GRIPPER_GOTO_KWARGS:
            raise ValueError("Unsupported gripper kind")
        if use_gripper is None:
            use_gripper = self.use_gripper
        self.kill_controller()
        directory = os.path.dirname(os.path.realpath(__file__))
        self._gripper_kind = gripper_kind
        self._robot_process = subprocess.Popen(
            ["bash", os.path.join(directory, "launch_robot.sh")], start_new_session=True
        )
        self._gripper_process = (
            subprocess.Popen(
                ["bash", os.path.join(directory, "launch_gripper.sh"), gripper_kind], start_new_session=True
            )
            if use_gripper
            else None
        )
        self._server_launched = True
        time.sleep(5)

    # Franka Hand max stroke (m). Used when polymetis metadata isn't available yet
    # (franka_hand_client only calls InitRobotClient after gripper_->homing(), which
    # can finish after our GripperInterface() has already queried the server).
    DEFAULT_MAX_GRIPPER_WIDTH = 0.077

    def launch_robot(self, use_gripper=None):
        if use_gripper is None:
            use_gripper = self.use_gripper
        # Use 127.0.0.1 to avoid IPv6 localhost resolution. Wait for polymetis server.
        for _ in range(150):  # ~15s at 0.1s interval
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", 50051)) == 0:
                    break
            time.sleep(0.1)
        # Server is up but robot context is valid only after franka_panda_client connects and calls InitRobotClient.
        for attempt in range(60):  # up to ~30s
            try:
                self._robot = RobotInterface(ip_address="127.0.0.1", enforce_version=False)
                break
            except grpc.RpcError as e:
                if "Robot context not valid" in str(e.details()) and attempt < 59:
                    time.sleep(0.5)
                    continue
                raise
        self._last_native_close = None
        self._gripper = None
        self._max_gripper_width = self.DEFAULT_MAX_GRIPPER_WIDTH
        if use_gripper:
            self._gripper = GripperInterface(ip_address="127.0.0.1")
            from polymetis.gripper_interface import EMPTY
            md = wait_gripper_metadata(
                lambda: self._gripper.grpc_connection.GetRobotClientMetadata(EMPTY, timeout=1.0)
            )
            self._gripper.metadata = md
            self._max_gripper_width = float(md.max_width)
        self._ik_solver = RobotIKSolver()
        self._controller_not_loaded = False

    def kill_controller(self):
        # Stop only process groups launched by this bridge.
        for name in ("_robot_process", "_gripper_process"):
            process = getattr(self, name, None)
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)

    def update_command(self, command, action_space="cartesian_velocity", gripper_action_space=None, blocking=False):
        action_dict = self.create_action_dict(
            command, action_space=action_space, gripper_action_space=gripper_action_space
        )

        self.update_joints(action_dict["joint_position"], velocity=False, blocking=blocking)
        self.update_gripper(action_dict["gripper_position"], velocity=False, blocking=blocking)

        return action_dict

    def update_pose(self, command, velocity=False, blocking=False):
        if blocking:
            if velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(command)
                command = add_poses(cartesian_delta, curr_pose)

            pos = torch.Tensor(command[:3])
            quat = torch.Tensor(euler_to_quat(command[3:6]))
            curr_joints = self._robot.get_joint_positions()
            desired_joints, success = self._robot.solve_inverse_kinematics(pos, quat, curr_joints)
            if not success:
                raise RuntimeError("Inverse kinematics failed")
            self.update_joints(desired_joints, velocity=False, blocking=True)
        else:
            if not velocity:
                curr_pose = self.get_ee_pose()
                cartesian_delta = pose_diff(command, curr_pose)
                command = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)

            robot_state = self.get_robot_state()[0]
            joint_velocity = self._ik_solver.cartesian_velocity_to_joint_velocity(command, robot_state=robot_state)

            self.update_joints(joint_velocity, velocity=True, blocking=False)

    def update_joints(self, command, velocity=False, blocking=False, cartesian_noise=None):
        if self._motion_error:
            raise RuntimeError(f"Controller failure: {self._motion_error}")
        if cartesian_noise is not None:
            command = self.add_noise_to_joints(command, cartesian_noise)
        command = torch.Tensor(command)

        if velocity:
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            command = joint_delta + self._robot.get_joint_positions()

        def helper_non_blocking():
            try:
                if self._impedance_started and not self._robot.is_running_policy():
                    raise RuntimeError("Active control policy stopped unexpectedly; end the episode and inspect the NUC logs")
                for attempt in range(3):
                    if self._robot.is_running_policy():
                        break
                    self._robot.start_cartesian_impedance()
                    deadline = time.monotonic() + 5
                    while not self._robot.is_running_policy() and time.monotonic() < deadline:
                        time.sleep(0.01)
                if not self._robot.is_running_policy():
                    raise RuntimeError("Impedance controller did not start after three attempts")
                self._impedance_started = True
                self._robot.update_desired_joint_positions(command)
            except Exception as error:
                self._motion_error = str(error)
            finally:
                self._controller_not_loaded = False

        if blocking:
            if self._robot.is_running_policy():
                self._robot.terminate_current_policy()
            time_to_go = self.adaptive_time_to_go(command)
            self._robot.move_to_joint_positions(command, time_to_go=time_to_go)

            self._robot.start_cartesian_impedance()
            self._impedance_started = True
        else:
            if not self._controller_not_loaded:
                self._controller_not_loaded = True
                run_threaded_command(helper_non_blocking)

    def update_gripper(self, command, velocity=True, blocking=False):
        if self._gripper is None:
            return
        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            command = gripper_delta + self.get_gripper_position()

        command = np.clip(command, 0, 1)
        gripper_kind = getattr(self, "_gripper_kind", None) or "robotiq_2f"
        kw = GRIPPER_GOTO_KWARGS[gripper_kind]
        target_width = self._max_gripper_width * (1 - command)

        if gripper_kind == "franka_hand":
            # UMI deployment uses binary open/close. Send once per transition,
            # rather than restarting a grasp for every policy timestep.
            close = bool(command >= 0.5)
            if getattr(self, "_last_native_close", None) == close:
                return
            if close:
                self._gripper.grasp(speed=kw["speed"], force=kw["force"],
                    grasp_width=0.0, epsilon_inner=0.0,
                    epsilon_outer=self._max_gripper_width, blocking=blocking)
            else:
                self._gripper.goto(width=self._max_gripper_width, blocking=blocking, **kw)
            self._last_native_close = close
        else:
            self._gripper.goto(width=target_width, blocking=blocking, **kw)

    def add_noise_to_joints(self, original_joints, cartesian_noise):
        original_joints = torch.Tensor(original_joints)

        pos, quat = self._robot.robot_model.forward_kinematics(original_joints)
        curr_pose = pos.tolist() + quat_to_euler(quat).tolist()
        new_pose = add_poses(cartesian_noise, curr_pose)

        new_pos = torch.Tensor(new_pose[:3])
        new_quat = torch.Tensor(euler_to_quat(new_pose[3:]))

        noisy_joints, success = self._robot.solve_inverse_kinematics(new_pos, new_quat, original_joints)

        if success:
            desired_joints = noisy_joints
        else:
            desired_joints = original_joints

        return desired_joints.tolist()

    def get_joint_positions(self):
        return self._robot.get_joint_positions().tolist()

    def get_joint_velocities(self):
        return self._robot.get_joint_velocities().tolist()

    def get_gripper_position(self):
        if self._gripper is None:
            return 0.0
        return 1 - (self._gripper.get_state().width / self._max_gripper_width)

    def get_ee_pose(self):
        pos, quat = self._robot.get_ee_pose()
        angle = quat_to_euler(quat.numpy())
        return np.concatenate([pos, angle]).tolist()

    def get_robot_state(self):
        if self._motion_error:
            raise RuntimeError(f"Controller failure; inspect NUC before restarting: {self._motion_error}")
        robot_state = self._robot.get_robot_state()
        gripper_position = self.get_gripper_position()
        pos, quat = self._robot.robot_model.forward_kinematics(torch.Tensor(robot_state.joint_positions))
        cartesian_position = pos.tolist() + quat_to_euler(quat.numpy()).tolist()

        state_dict = {
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position,
            "joint_positions": list(robot_state.joint_positions),
            "joint_velocities": list(robot_state.joint_velocities),
            "joint_torques_computed": list(robot_state.joint_torques_computed),
            "prev_joint_torques_computed": list(robot_state.prev_joint_torques_computed),
            "prev_joint_torques_computed_safened": list(robot_state.prev_joint_torques_computed_safened),
            "motor_torques_measured": list(robot_state.motor_torques_measured),
            "prev_controller_latency_ms": robot_state.prev_controller_latency_ms,
            "prev_command_successful": robot_state.prev_command_successful,
        }

        timestamp_dict = {
            "robot_timestamp_seconds": robot_state.timestamp.seconds,
            "robot_timestamp_nanos": robot_state.timestamp.nanos,
        }

        return state_dict, timestamp_dict

    def adaptive_time_to_go(self, desired_joint_position, t_min=0, t_max=4):
        curr_joint_position = self._robot.get_joint_positions()
        displacement = desired_joint_position - curr_joint_position
        time_to_go = self._robot._adaptive_time_to_go(displacement)
        clamped_time_to_go = min(t_max, max(time_to_go, t_min))
        return clamped_time_to_go

    def create_action_dict(self, action, action_space, gripper_action_space=None, robot_state=None):
        assert action_space in ["cartesian_position", "joint_position", "cartesian_velocity", "joint_velocity"]
        if robot_state is None:
            robot_state = self.get_robot_state()[0]
        action_dict = {"robot_state": robot_state}
        velocity = "velocity" in action_space

        if gripper_action_space is None:
            gripper_action_space = "velocity" if velocity else "position"
        assert gripper_action_space in ["velocity", "position"]

        if gripper_action_space == "velocity":
            action_dict["gripper_velocity"] = action[-1]
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(action[-1])
            gripper_position = robot_state["gripper_position"] + gripper_delta
            action_dict["gripper_position"] = float(np.clip(gripper_position, 0, 1))
        else:
            action_dict["gripper_position"] = float(np.clip(action[-1], 0, 1))
            gripper_delta = action_dict["gripper_position"] - robot_state["gripper_position"]
            gripper_velocity = self._ik_solver.gripper_delta_to_velocity(gripper_delta)
            action_dict["gripper_delta"] = gripper_velocity

        if "cartesian" in action_space:
            if velocity:
                action_dict["cartesian_velocity"] = action[:-1]
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(action[:-1])
                action_dict["cartesian_position"] = add_poses(
                    cartesian_delta, robot_state["cartesian_position"]
                ).tolist()
            else:
                action_dict["cartesian_position"] = action[:-1]
                cartesian_delta = pose_diff(action[:-1], robot_state["cartesian_position"])
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()

            action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                action_dict["cartesian_velocity"], robot_state=robot_state
            ).tolist()
            joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
            action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()

        if "joint" in action_space:
            # NOTE: Joint to Cartesian has undefined dynamics due to IK
            if velocity:
                action_dict["joint_velocity"] = action[:-1]
                joint_delta = self._ik_solver.joint_velocity_to_delta(action[:-1])
                action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()
            else:
                action_dict["joint_position"] = action[:-1]
                joint_delta = np.array(action[:-1]) - np.array(robot_state["joint_positions"])
                joint_velocity = self._ik_solver.joint_delta_to_velocity(joint_delta)
                action_dict["joint_velocity"] = joint_velocity.tolist()

        return action_dict

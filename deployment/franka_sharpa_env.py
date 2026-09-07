"""Franka arm + Sharpa hand env. Position control only. Replaces gripper with Sharpa hand."""

import os
import sys
import time

import numpy as np

# Sharpa SDK path
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_sdk_path = os.path.join(_proj_root, "SharpaWaveSDK_4.3.4", "python")
if _sdk_path not in sys.path:
    sys.path.insert(0, _sdk_path)

# The Sharpa SDK is only needed to drive the real hand. Import it lazily so
# debug/mock mode (which leaves self._hand = None) runs with no SDK installed.
try:
    from sharpa import SharpaWaveManager, ControlMode, ControlSource, HandSide
    _SHARPA_SDK_AVAILABLE = True
except Exception:
    SharpaWaveManager = ControlMode = ControlSource = HandSide = None
    _SHARPA_SDK_AVAILABLE = False

from openpi.constants import SHARPA_ARM_QPOS, SHARPA_HAND_QPOS_OPEN, lerp_sharpa_hand_qpos
from deployment.robot_env import RobotEnv

HAND_SPEC_TO_SIDE = (
    {"sharpa_L": HandSide.LEFT, "sharpa_R": HandSide.RIGHT} if _SHARPA_SDK_AVAILABLE else {}
)
SHARPA_JOINTS = 22

# Arm + hand reset poses live in droid/constants.py (the central source). The
# server seeds its IK warm-start from the matching server-side constants at
# startup; if SHARPA_ARM_QPOS / SHARPA_HAND_QPOS_OPEN diverge from those, the
# first action chunk lands in the wrong null-space branch and jumps the arm.
# SHARPA_HAND_QPOS_OPEN is also what --fixed-hand-sharpa-ik emits on the first
# chunk (lerp(grip≈0)), so resetting the physical hand there avoids a jump.


def _connect_hand(required_side: HandSide):
    """Connect to device matching hand_side. Returns (hand, None) or (None, error_msg)."""
    manager = SharpaWaveManager.get_instance()
    time.sleep(1)
    device_sns = manager.get_all_device_sn()
    if not device_sns:
        return None, "No devices found"
    for device_sn in device_sns:
        hand = manager.connect(device_sn)
        if hand.get_device_info().hand_side == required_side:
            return hand, None
        manager.disconnect(device_sn)
    return None, f"No device with hand side {required_side} found"


def _init_hand(hand) -> bool:
    for setter, val in [
        (hand.set_control_mode, ControlMode.POSITION),
        (hand.set_speed_coeff, 0.3),
        (hand.set_current_coeff, 0.6),
        (hand.set_control_source, ControlSource.SDK),
    ]:
        err = setter(val)
        if err.code != 0:
            return False
    return True


class FrankaSharpaEnv(RobotEnv):
    """RobotEnv with Sharpa hand instead of gripper."""

    def __init__(self, hand_spec="sharpa_R", action_space="joint_position", use_cameras=False, do_reset=True, hand_camera_id=None, debug=False, debug_camera_ids=None):
        assert action_space in ("joint_position", "cartesian_position"), (
            f"FrankaSharpaEnv only supports 'joint_position' or 'cartesian_position', "
            f"got '{action_space}'"
        )
        super().__init__(
            action_space=action_space,
            gripper_action_space="position",
            camera_kwargs={},
            do_reset=False,
            use_cameras=use_cameras,
            use_gripper=False,
            calibration_name="robotiq",  # Sharpa reuses the Robotiq calibration
            hand_camera_id=hand_camera_id,
            debug=debug,
            debug_camera_ids=debug_camera_ids,
        )
        self.reset_joints = SHARPA_ARM_QPOS
        self.hand_spec = hand_spec
        # In debug/mock mode self._hand stays None; every hand call below is guarded.
        self._hand = None
        if not debug:
            self._connect_and_init_hand()
        if do_reset:
            self.reset()

    def _connect_and_init_hand(self):
        if not _SHARPA_SDK_AVAILABLE:
            raise RuntimeError(
                "Sharpa SDK not available — install SharpaWaveSDK_4.3.4 at the repo "
                "root, or run with debug=True for the mock hand."
            )
        side = HAND_SPEC_TO_SIDE.get(self.hand_spec)
        if side is None:
            raise ValueError(f"Unknown hand_spec '{self.hand_spec}'. Use 'sharpa_L' or 'sharpa_R'.")
        self._hand, err = _connect_hand(side)
        if self._hand is None:
            raise RuntimeError(f"Sharpa hand connection failed: {err}")
        if not _init_hand(self._hand):
            raise RuntimeError("Sharpa hand init failed")
        self._hand.start()

    def reset(self, randomize=False):
        # Arm only (no gripper)
        if randomize:
            noise = np.random.uniform(low=self.randomize_low, high=self.randomize_high)
        else:
            noise = None
        self._robot.update_joints(self.reset_joints, velocity=False, blocking=True, cartesian_noise=noise)
        if self._hand is not None:
            self._hand.set_joint_position(SHARPA_HAND_QPOS_OPEN.tolist(), True)

    def cycle_gripper(self) -> bool:
        """Visible close → open cycle on the Sharpa hand so the operator can
        confirm it responds before a rollout (the 'g' option in the startup
        prompts). Drives the 22 hand joints between the open (grip=0) and closed
        (grip=1) poses via lerp_sharpa_hand_qpos; final state = open, matching
        reset(). The arm is not touched. The Sharpa hand has no parallel-jaw
        failure mode to detect, so this always succeeds (last_cycle_ok=True)."""
        print("[FrankaSharpaEnv] cycle hand: close → open")
        if self._hand is not None:
            self._hand.set_joint_position(lerp_sharpa_hand_qpos(1.0).tolist(), True)
            time.sleep(0.8)
            self._hand.set_joint_position(lerp_sharpa_hand_qpos(0.0).tolist(), True)
            time.sleep(0.8)
        self.last_cycle_ok = True
        return True

    def get_observation(self):
        obs_dict = super().get_observation()
        if self._hand is None:  # debug/mock: report the open hand pose
            angles_rad = SHARPA_HAND_QPOS_OPEN.tolist()
        else:
            err, angles_rad = self._hand.get_joint_position_rad()
            if err.code != 0:
                raise RuntimeError(f"Sharpa get_joint_position_rad failed: {err.message}")
        obs_dict["hand_state"] = {
            "joint_positions": np.array(angles_rad, dtype=np.float64)[:SHARPA_JOINTS],
        }
        return obs_dict

    def step(self, action):
        """Dispatch on action_space + action length:
        - joint_*    | 8 = legacy DROID action (7 arm + 1 gripper). Hand is held at zero.
        - joint_*    | 29 = arm joints + Sharpa qpos (7 arm + 22 hand).
        - cartesian_position | 29 = arm cart (3 pos + 3 euler + 1 gripper-slot)
                               + 22 hand qpos. Cart 7-tuple is routed through
                               Polymetis's cartesian controller (same path
                               RobotEnv / FrankaUMIEnv use); hand qpos goes to
                               the Sharpa hand. Gripper slot is ignored by the
                               Sharpa env — the policy's gripper scalar should
                               already be reflected in the hand qpos via a
                               client-side lerp between OPEN/CLOSED.
        """
        action = np.asarray(action)
        if self.action_space == "cartesian_position":
            if len(action) != 7 + SHARPA_JOINTS:
                raise ValueError(
                    f"FrankaSharpaEnv.step (cartesian_position): expected action of "
                    f"length {7 + SHARPA_JOINTS} (7 cart + 22 hand), got {len(action)}"
                )
            cart_arm = action[:7]
            hand_joints = action[7:7 + SHARPA_JOINTS]
            return self.update_robot_cartesian(cart_arm, hand_joints)

        if len(action) == self.DoF:  # legacy 8-dim
            arm_joints = action[:7]
            hand_joints = np.zeros(SHARPA_JOINTS, dtype=np.float64)
        elif len(action) == 7 + SHARPA_JOINTS:  # 29-dim arm + sharpa
            arm_joints = action[:7]
            hand_joints = action[7:7 + SHARPA_JOINTS]
        else:
            raise ValueError(
                f"FrankaSharpaEnv.step: expected action of length {self.DoF} (arm+gripper) "
                f"or {7 + SHARPA_JOINTS} (arm+sharpa), got {len(action)}"
            )
        return self.update_robot(arm_joints, hand_joints)

    def update_robot(self, arm_joints, hand_joints, blocking=False):
        """Send arm_joints (7) and hand_joints (22) to the robot."""
        self._robot.update_joints(arm_joints, velocity=False, blocking=blocking)
        if self._hand is not None:
            self._hand.set_joint_position(hand_joints, interpolate=True)
        return {"joint_position": arm_joints, "hand_position": hand_joints}

    def update_robot_cartesian(self, cart_arm, hand_joints, blocking=False):
        """Send cart_arm (7 = pos+euler+gripper-slot) via Polymetis's cartesian
        controller and hand_joints (22) to the Sharpa hand. The gripper slot is
        forwarded with ``gripper_action_space=None`` since this env was created
        with ``use_gripper=False`` — Polymetis ignores it."""
        self._robot.update_command(
            cart_arm,
            action_space="cartesian_position",
            gripper_action_space=None,
            blocking=blocking,
        )
        if self._hand is not None:
            self._hand.set_joint_position(hand_joints, interpolate=True)
        return {"cartesian_position": cart_arm, "hand_position": hand_joints}

    def disconnect_hand(self):
        if self._hand:
            self._hand.stop()
            SharpaWaveManager.get_instance().disconnect_all()
            self._hand = None

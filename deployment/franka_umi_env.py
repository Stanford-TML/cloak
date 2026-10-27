"""Native Franka gripper with UMI fingertips."""

import time
import numpy as np

from deployment.robot_env import RobotEnv
from openpi.constants import UMI_ARM_QPOS


class FrankaUMIEnv(RobotEnv):
    def __init__(self, *, do_reset=True, **kwargs):
        super().__init__(do_reset=False, gripper_kind="franka_hand", calibration_name="umi", **kwargs)
        self.reset_joints = UMI_ARM_QPOS
        if do_reset:
            self.reset()

    def step(self, action):
        action = np.asarray(action, dtype=float).copy()
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("UMI expects seven arm joints and one finite gripper scalar")
        # Preserve binary control, with a conservative closing cap for soft fingers.
        action[-1] = 0.9 if action[-1] >= 0.5 else 0.0
        return super().step(action)

    def cycle_gripper(self):
        """Operator-requested check; do not restart controllers automatically."""
        self._robot.update_gripper(0, velocity=False, blocking=True)
        time.sleep(0.8)
        before = float(self.get_state()[0]["gripper_position"])
        self._robot.update_gripper(0.85, velocity=False, blocking=True)
        time.sleep(0.8)
        after = float(self.get_state()[0]["gripper_position"])
        self._robot.update_gripper(0, velocity=False, blocking=True)
        time.sleep(0.8)
        ok = before < 0.2 and after >= 0.3
        print(f"UMI gripper check: {'passed' if ok else 'FAILED'} (open={before:.3f}, closed={after:.3f})")
        return ok

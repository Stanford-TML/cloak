"""The manual overlay must match the mask actually used during inference."""

import numpy as np
import pytest
from openpi.calibration.silhouette import Sim
from openpi.constants import (
    DEFAULT_CAM_TO_GRIPPER,
    DROID_ARM_QPOS,
    UMI_ARM_QPOS,
    SHARPA_ARM_QPOS,
    SHARPA_HAND_QPOS_OPEN,
)
from openpi.policies.gripper_mask_renderer import GripperMaskRenderer, UmiMaskRenderer, SharpaMaskRenderer
from deployment.calibration.hand_eye import pose_vector


@pytest.mark.parametrize(
    "name,cls,joints,gripper",
    [
        ("robotiq", GripperMaskRenderer, DROID_ARM_QPOS, 0.0),
        ("umi", UmiMaskRenderer, UMI_ARM_QPOS, 0.0),
        ("sharpa", SharpaMaskRenderer, SHARPA_ARM_QPOS, SHARPA_HAND_QPOS_OPEN),
    ],
)
def test_manual_and_deployment_masks_match(name, cls, joints, gripper):
    sim = Sim(name)
    renderer = cls()
    try:
        ee = sim.set_pose(joints, 0.0)
        pose = [-0.0973, -0.1003, 0.022, -0.3038, 0.4428, -1.3705] if name == "sharpa" else DEFAULT_CAM_TO_GRIPPER
        world_camera = ee @ sim.t_c2g_from_6vec(pose)
        manual = sim.render_gripper_mask(world_camera, 210.0)
        assert manual.any(), "Test camera must see the end effector"
        renderer.set_intrinsics(210.0)
        deployed, _ = renderer.render_mask(joints, gripper, pose_vector(world_camera))
        np.testing.assert_array_equal(manual, deployed)
    finally:
        sim.renderer.close()
        renderer._seg_renderer.close()


def test_deployment_uses_sdk_intrinsics_and_exact_camera_id():
    from types import SimpleNamespace
    from deployment.deploy_policy import _extract_observation

    args = SimpleNamespace(external_camera_id="99", wrist_camera_id="123", wrist_cam_lens="left", debug=False)
    frame = np.zeros((720, 1280, 4), dtype=np.uint8)
    obs = {
        "image": {"99_left": frame, "123_left": frame, "9123_left": np.ones_like(frame)},
        "robot_state": {"joint_positions": [0] * 7, "gripper_position": 0.0},
        "camera_extrinsics": {"123_left_gripper_offset": [0] * 6, "9123_left_gripper_offset": [1] * 6},
        "camera_intrinsics": {"123_left": np.array([[800.0, 0, 640], [0, 840.0, 360], [0, 0, 1.0]])},
    }
    result = _extract_observation(args, obs)
    np.testing.assert_allclose(result["camera_intrinsic"], [200, 210, 160, 90])
    np.testing.assert_allclose(result["camera_extrinsic_cam_to_gripper"], 0)
    assert not result["wrist_image"].any()

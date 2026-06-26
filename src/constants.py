"""Physical, calibration, and path constants for the Robotiq 2F-85 wrist-extrinsics pipeline.

Hardcoded here (from openpi.constants / scene_paths / the DROID preprocessing
assets) so the cloak code stands alone.
"""

from pathlib import Path

import numpy as np

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
ROBOTIQ_SCENE_XML = _ASSETS / "franka_fr3_robotiq" / "scene.xml"

# RLDS wrist-stream size and key.
RLDS_H, RLDS_W = 180, 320
WRIST_CAM_KEY = "wrist_image_left"

# Settled qpos for the 8 Robotiq 2F-85 joints (qpos[7:] in fr3_robotiq.xml) at
# fully open (ctrl=0) and closed (ctrl=0.8); the 1-DOF command lerps between them.
GRIPPER_QPOS_OPEN = np.array([
    0.05685287, 0.0001478, 0.05684168, -0.05472553,
    0.05685287, 0.0001478, 0.05684095, -0.0547242,
])
GRIPPER_QPOS_CLOSED = np.array([
    0.77526133, 0.00017553, 0.77537824, -0.74554769,
    0.77526154, 0.00017356, 0.77538077, -0.74552788,
])


def lerp_gripper_qpos(gripper_position: float) -> np.ndarray:
    """8 Robotiq joint angles for `gripper_position` in [0, 1] (0=open, 1=closed)."""
    g = float(gripper_position)
    return (1.0 - g) * GRIPPER_QPOS_OPEN + g * GRIPPER_QPOS_CLOSED


# Mean wrist-camera parameters across the DROID dataset, used as the starting
# point for the per-episode optimization: cam_to_gripper [tx,ty,tz,rx,ry,rz]
# (m + XYZ euler rad) in the EE frame, and [fx,fy,cx,cy] intrinsics at RLDS 320x180.
DEFAULT_CAM_TO_GRIPPER = np.array([
    -0.07603768464487827, 0.030755540645682176, -0.005156207813252746,
    -0.33089674351839693, 0.0052405001986228815, -1.5305427085990393,
])
DEFAULT_WRIST_INTRINSICS = np.array([
    182.7151540905966, 182.7151540905966, 160.14853881286072, 89.9061105794428,
])

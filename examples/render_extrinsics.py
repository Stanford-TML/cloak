#!/usr/bin/env python3
"""Render the Franka + Robotiq 2F-85 gripper from the wrist camera using our
calibrated DROID wrist extrinsics.

DATA (`assets/droid_wrist_extrinsics.json`)
    { "<LAB>|<timestamp>": [tx, ty, tz, rx, ry, rz], ... }   # 77,425 DROID episodes
Each value is the wrist camera's pose in the end-effector frame ("cam_to_gripper"):
    - tx,ty,tz : translation in meters
    - rx,ry,rz : rotation as XYZ Tait-Bryan euler angles in radians
The end-effector frame is `attachment_site` (Franka link7 + 0.107 m on z), the
frame DROID's `cartesian_position` represents. The extrinsics were silhouette-
calibrated per episode; the worst 1% (drift from the DROID mean) are dropped.

USAGE
    cam_to_world = FK(joint_position, gripper) @ cam_to_gripper
where FK is MuJoCo forward kinematics at `attachment_site`. This script mounts a
camera there and renders the gripper as the wrist camera would see it.

    python examples/render_extrinsics.py --key "AUTOLab|2023-07-07-09h-42m-23s"
"""

import dataclasses
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import tyro

ASSETS = Path(__file__).resolve().parent.parent / "assets"
SCENE_XML = ASSETS / "franka_fr3_robotiq" / "scene.xml"
EXTRINSICS_JSON = ASSETS / "droid_wrist_extrinsics.json"

# Settled qpos for the 8 Robotiq 2F-85 joints (qpos[7:] in fr3_robotiq.xml) at
# fully open / closed; the 1-DOF gripper command lerps between them.
GRIPPER_QPOS_OPEN = np.array([0.05685, 0.00015, 0.05684, -0.05473, 0.05685, 0.00015, 0.05684, -0.05472])
GRIPPER_QPOS_CLOSED = np.array([0.77526, 0.00018, 0.77538, -0.74555, 0.77526, 0.00017, 0.77538, -0.74553])

# Wrist-camera focal length (fy) and render size of the DROID wrist stream.
DEFAULT_FY = 182.715
RENDER_H, RENDER_W = 180, 320

# A Franka "ready" pose, just for the demo (any 7-DOF joint config works).
SAMPLE_JOINTS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
SAMPLE_GRIPPER = 0.0  # 0 = open, 1 = closed


def cam_to_gripper_matrix(vec6: np.ndarray) -> np.ndarray:
    """4x4 cam-to-gripper from [tx,ty,tz, rx,ry,rz] (meters + XYZ euler radians)."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", vec6[3:]).as_matrix()
    T[:3, 3] = vec6[:3]
    return T


def build_model() -> mujoco.MjModel:
    """Load the scene and add a wrist camera we can place each frame."""
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    spec.worldbody.add_camera().name = "wrist_cam"
    return spec.compile()


def forward_kinematics(model, data, joints: np.ndarray, gripper: float) -> np.ndarray:
    """Set arm + gripper qpos, run FK, return the 4x4 attachment_site (EE) pose in world."""
    g = float(gripper)
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = joints
    data.qpos[7:] = (1.0 - g) * GRIPPER_QPOS_OPEN + g * GRIPPER_QPOS_CLOSED
    mujoco.mj_forward(model, data)
    site = model.site("attachment_site").id
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[site]
    T[:3, :3] = data.site_xmat[site].reshape(3, 3)
    return T


def render(model, data, renderer, cam_to_world: np.ndarray, fy: float) -> np.ndarray:
    """Render from a camera at `cam_to_world` (OpenCV convention: x right, y down, z forward)."""
    cam_id = model.cam("wrist_cam").id
    flip = np.diag([1.0, -1.0, -1.0])  # OpenCV -> MuJoCo/OpenGL camera axes
    data.cam_xpos[cam_id] = cam_to_world[:3, 3]
    data.cam_xmat[cam_id] = (cam_to_world[:3, :3] @ flip).flatten()
    model.cam_fovy[cam_id] = np.degrees(2.0 * np.arctan(RENDER_H / (2.0 * fy)))
    renderer.update_scene(data, camera="wrist_cam")
    return renderer.render()


@dataclasses.dataclass
class Args:
    key: str | None = None
    """Episode key 'LAB|timestamp'; defaults to the first entry."""
    json: Path = EXTRINSICS_JSON
    """Extrinsics JSON to read."""
    out: Path = Path("wrist_view.png")


def main(args: Args) -> None:
    extrinsics = json.loads(args.json.read_text())
    key = args.key or next(iter(extrinsics))
    if key not in extrinsics:
        raise SystemExit(f"key {key!r} not found in {args.json.name}")
    vec6 = np.asarray(extrinsics[key])

    model = build_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)

    T_ee = forward_kinematics(model, data, SAMPLE_JOINTS, SAMPLE_GRIPPER)
    T_cam = T_ee @ cam_to_gripper_matrix(vec6)
    rgb = render(model, data, renderer, T_cam, DEFAULT_FY)

    Image.fromarray(rgb).save(args.out)
    print(f"key={key}\ncam_to_gripper={vec6.tolist()}\nwrote {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))

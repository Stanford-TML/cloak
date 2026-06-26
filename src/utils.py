"""Minimal MuJoCo FK/mask Sim and DROID RLDS helpers for Silhouette Calibration.

Stripped from the full DROID preprocessing pipeline to the Robotiq-only subset
the wrist-extrinsics optimizer needs: forward kinematics at the end-effector and
a wrist-camera gripper-segmentation render.
"""

import math
import re

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import tensorflow as tf

from constants import RLDS_H, RLDS_W, ROBOTIQ_SCENE_XML, lerp_gripper_qpos

# Geom group of the floor collision plane; hidden in the segmentation render.
_COLLISION_GROUP = 3
_SEG_HIDDEN_GROUP = 5

# Robotiq 2F-85 bodies whose geoms form the gripper silhouette.
_GRIPPER_BODY_NAMES = [
    "base_mount", "2f85_base",
    "right_driver", "right_coupler", "right_spring_link", "right_follower", "right_pad", "right_silicone_pad",
    "left_driver", "left_coupler", "left_spring_link", "left_follower", "left_pad", "left_silicone_pad",
]


# ---------------------------------------------------------------------------
# DROID RLDS helpers
# ---------------------------------------------------------------------------

KNOWN_LABS = {
    "AUTOLab", "CLVR", "GuptaLab", "ILIAD", "IPRL", "IRIS",
    "PennPAL", "RAD", "RAIL", "REAL", "RPL", "TRI", "WEIRD",
}
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
_LAB_BY_LOWER = {lab.lower(): lab for lab in KNOWN_LABS}


def find_dataset_dir(data_dir) -> "Path":
    """Find the TFDS dataset directory (the one holding features.json) under `data_dir`."""
    hits = list(data_dir.rglob("features.json"))
    if not hits:
        raise FileNotFoundError(f"No TFDS dataset (features.json) found under {data_dir}")
    return hits[0].parent


def _parse_lab(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("success", "failure") and i > 0:
            return parts[i - 1]
    raise ValueError(f"file_path missing 'success'/'failure' segment: {file_path!r}")


def _parse_timestamp_dir(ts_dir: str):
    """DROID path TIMESTAMP_DIR -> (year, month, day, hh, mm, ss); handles ':' and '_' separators."""
    tokens = [t for t in re.split(r"[_:]", ts_dir) if t]
    if len(tokens) != 7:
        return None
    _dow, mon, day, hh, mm, ss, year = tokens
    if mon not in _MONTHS:
        return None
    try:
        return int(year), _MONTHS[mon], int(day), int(hh), int(mm), int(ss)
    except ValueError:
        return None


def file_path_to_lang_key(file_path: str):
    """DROID episode path -> canonical 'LAB|YYYY-MM-DD-HHh-MMm-SSs' key, or None."""
    parts = file_path.replace("\\", "/").rstrip("/").split("/")
    if parts and parts[-1] == "trajectory.h5":
        parts = parts[:-1]
    if len(parts) < 4:
        return None
    parsed = _parse_timestamp_dir(parts[-1])
    if parsed is None:
        return None
    lab = _parse_lab(file_path)
    lab = _LAB_BY_LOWER.get(lab.lower(), lab)
    y, mo, d, hh, mm, ss = parsed
    return f"{lab}|{y:04d}-{mo:02d}-{d:02d}-{hh:02d}h-{mm:02d}m-{ss:02d}s"


def decode_camera_frames(steps, camera_key: str) -> np.ndarray:
    """Decode all frames for one camera. Returns uint8 (N, H, W, 3)."""
    frames = []
    for step in steps:
        img = step["observation"][camera_key]
        if img.dtype == tf.string:
            img = tf.io.decode_jpeg(img)
        frames.append(img.numpy())
    return np.stack(frames)


# ---------------------------------------------------------------------------
# MuJoCo Sim: FK at the end-effector + wrist-camera gripper mask
# ---------------------------------------------------------------------------

class Sim:
    """Franka FR3 + Robotiq 2F-85 forward kinematics and wrist-cam segmentation."""

    EE_SITE = "attachment_site"  # DROID end-effector frame (link7 + 0.107 m on z).

    def __init__(self) -> None:
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=RLDS_H, width=RLDS_W)
        self.renderer.enable_segmentation_rendering()
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = 1
        self._opt.geomgroup[_COLLISION_GROUP] = 0
        self._opt.sitegroup[:] = 0  # sites would punch holes in the mask
        try:  # hide the floor so it never occludes the gripper in the mask
            self.model.geom_group[self.model.geom("floor").id] = _SEG_HIDDEN_GROUP
            self._opt.geomgroup[_SEG_HIDDEN_GROUP] = 0
        except KeyError:
            pass
        ids = {self.model.body(n).id for n in _GRIPPER_BODY_NAMES}
        self.gripper_geom_ids = np.array(
            [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in ids])
        self._cam_id = self.model.cam("wrist_cam").id
        self._ee_site_id = self.model.site(self.EE_SITE).id

    def _build_model(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(ROBOTIQ_SCENE_XML))
        cam = spec.worldbody.add_camera()
        cam.name = "wrist_cam"
        cam.fovy = 60.0  # placeholder; overwritten per render
        return spec.compile()

    def set_pose(self, joint_position: np.ndarray, gripper_position: float) -> np.ndarray:
        """Set arm + gripper qpos, run FK, return the 4x4 end-effector pose in world."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = joint_position
        self.data.qpos[7:] = lerp_gripper_qpos(gripper_position)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        T = np.eye(4)
        T[:3, 3] = self.data.site_xpos[self._ee_site_id]
        T[:3, :3] = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        return T

    @staticmethod
    def t_c2g_from_6vec(cam_to_gripper: np.ndarray) -> np.ndarray:
        """4x4 cam-to-gripper from [tx,ty,tz, rx,ry,rz] (meters + XYZ euler radians)."""
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", cam_to_gripper[3:]).as_matrix()
        T[:3, 3] = cam_to_gripper[:3]
        return T

    def render_gripper_mask(self, T_cam: np.ndarray, fy: float) -> np.ndarray:
        """Render the gripper silhouette from a camera at `T_cam` (OpenCV convention).

        Assumes `set_pose` was called this step. Returns an (H, W) bool mask.
        """
        self.model.cam_fovy[self._cam_id] = math.degrees(2.0 * math.atan(RLDS_H / (2.0 * fy)))
        flip = np.diag([1.0, -1.0, -1.0])  # OpenCV -> MuJoCo/OpenGL camera axes
        self.data.cam_xpos[self._cam_id] = T_cam[:3, 3]
        self.data.cam_xmat[self._cam_id] = (T_cam[:3, :3] @ flip).flatten()
        self.renderer.update_scene(self.data, camera="wrist_cam", scene_option=self._opt)
        seg = self.renderer.render()
        return np.isin(seg[:, :, 0], self.gripper_geom_ids)

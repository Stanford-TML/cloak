"""Silhouette Calibration (Algorithm 1): recover the wrist camera's pose in the
end-effector frame ("cam_to_gripper") by matching a rendered gripper silhouette
to the gripper in the wrist image.

  1. Build a target mask of the gripper from wrist frames — the intersection of
     low intensity (the gripper is dark) and low temporal std (the gripper is
     rigidly mounted, so its pixels barely move while the background does) over
     the gripper-open frames.
  2. Nelder-Mead optimize the 6-DOF cam_to_gripper so the rendered gripper mask
     maximizes IoU with the target.

Operates on plain numpy arrays (no TensorFlow), so it is shared by the DROID
batch job (preprocessing/preprocess_wrist_extrinsics.py) and the live-robot
calibration (deployment/calibrate_wrist_camera.py).
"""

import dataclasses
import math
from typing import Literal

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from openpi.constants import RLDS_H, RLDS_W, ROBOTIQ_SCENE_XML, lerp_gripper_qpos

# Target-mask construction.
OPEN_TOL = 0.05         # gripper_position <= this counts as "open"
ROI_TOP_FRAC = 0.50     # gripper lives in the bottom rows / right cols of the wrist frame
ROI_LEFT_FRAC = 0.20
BIN_KEEP_FRAC = 0.50    # keep the darkest / most-rigid half of the ROI
MIN_KEEP_FRAMES = 1

# Nelder-Mead settings.
NM_MAXITER = 600
NM_INIT_T_STEP = 0.005   # 5 mm initial simplex spread (translation)
NM_INIT_R_STEP = 0.0087  # ~0.5 deg initial simplex spread (rotation)

# Geom group of the floor collision plane; hidden in the segmentation render.
_COLLISION_GROUP = 3
_SEG_HIDDEN_GROUP = 5

# Robotiq 2F-85 bodies whose geoms form the gripper silhouette.
_GRIPPER_BODY_NAMES = [
    "base_mount", "2f85_base",
    "right_driver", "right_coupler", "right_spring_link", "right_follower", "right_pad", "right_silicone_pad",
    "left_driver", "left_coupler", "left_spring_link", "left_follower", "left_pad", "left_silicone_pad",
]


@dataclasses.dataclass(frozen=True)
class CalibOptions:
    cue: Literal["intersection", "intensity", "std"] = "intersection"
    """Target-mask cue: low intensity, low temporal std, or their intersection."""
    keep_frac: float = BIN_KEEP_FRAC
    """Fraction of ROI pixels kept by each cue's percentile threshold."""


@dataclasses.dataclass(frozen=True)
class Target:
    """A pseudo-GT silhouette plus the robot state to render its match from."""
    mask: np.ndarray
    joint_position: np.ndarray
    gripper_position: float
    step: int


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


# ---------------------------------------------------------------------------
# Target mask
# ---------------------------------------------------------------------------

def _select_kept_gray(gripper_positions: np.ndarray, frames: np.ndarray):
    """Grayscale (K, H, W) of the gripper-open frames, or None if too few."""
    keep = np.asarray(gripper_positions) <= OPEN_TOL
    if int(keep.sum()) < MIN_KEEP_FRAMES:
        return None
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames[keep]]).astype(np.float32)


def _gripper_roi(H: int, W: int) -> np.ndarray:
    roi = np.ones((H, W), dtype=bool)
    roi[: int(H * ROI_TOP_FRAC), :] = False
    roi[:, : int(W * ROI_LEFT_FRAC)] = False
    return roi


def _threshold_mask(score: np.ndarray, roi: np.ndarray, keep_frac: float) -> np.ndarray:
    """Keep the bottom `keep_frac` of `score` inside `roi`."""
    thr = float(np.percentile(score[roi], keep_frac * 100))
    return (score < thr) & roi


def build_target_mask(gripper_positions: np.ndarray, frames: np.ndarray, opts: CalibOptions = CalibOptions()):
    """Gripper target mask from the gripper-open RGB `frames` (N, H, W, 3), in the
    gripper ROI. None if too few open frames."""
    gray = _select_kept_gray(gripper_positions, frames)
    if gray is None:
        return None
    roi = _gripper_roi(*gray.shape[1:])
    if opts.cue == "intensity":
        return _threshold_mask(np.median(gray, axis=0), roi, opts.keep_frac)
    if opts.cue == "std":
        return _threshold_mask(gray.std(axis=0), roi, opts.keep_frac)
    intensity = _threshold_mask(np.median(gray, axis=0), roi, opts.keep_frac)
    std = _threshold_mask(gray.std(axis=0), roi, opts.keep_frac)
    return intensity & std


def first_open_step(gripper_positions: np.ndarray) -> int:
    return int(np.argmax(np.asarray(gripper_positions) <= OPEN_TOL))


def build_target(
    frames: np.ndarray,
    joint_positions: np.ndarray,
    gripper_positions: np.ndarray,
    opts: CalibOptions = CalibOptions(),
) -> Target | None:
    """Open-frame target mask with the robot state of the first gripper-open step."""
    mask = build_target_mask(gripper_positions, frames, opts)
    if mask is None:
        return None
    t = first_open_step(gripper_positions)
    return Target(mask, np.asarray(joint_positions[t]), float(gripper_positions[t]), t)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / max(union, 1)


def pose_distance(pose: np.ndarray, ref: np.ndarray):
    """L2 translation (m) and axis-angle rotation magnitude (rad) between two 6-DOF poses."""
    R = Rotation.from_euler("xyz", pose[3:]) * Rotation.from_euler("xyz", ref[3:]).inv()
    return float(np.linalg.norm(pose[:3] - ref[:3])), float(R.magnitude())


def render_target_pose(target: Target, params: np.ndarray, fy: float, sim: Sim) -> np.ndarray:
    T_ee = sim.set_pose(target.joint_position, target.gripper_position)
    return sim.render_gripper_mask(T_ee @ sim.t_c2g_from_6vec(params), fy)


def optimize_pose(target: Target, fy: float, sim: Sim, init: np.ndarray):
    """Nelder-Mead the 6-DOF cam_to_gripper from `init` to maximize IoU with `target`.

    Returns (cam_to_gripper, iou_init, iou_opt, n_iter).
    """

    def loss(params):
        return 1.0 - iou(render_target_pose(target, params, fy, sim), target.mask)

    init = np.asarray(init, dtype=np.float64)
    init_step = np.array([NM_INIT_T_STEP] * 3 + [NM_INIT_R_STEP] * 3)
    init_simplex = np.vstack([init] + [init + init_step * e for e in np.eye(6)])
    iou_init = 1.0 - loss(init)
    result = minimize(
        loss, x0=init, method="Nelder-Mead",
        options={"initial_simplex": init_simplex, "maxiter": NM_MAXITER,
                 "xatol": 1e-4, "fatol": 1e-4, "adaptive": True, "disp": False},
    )
    return np.asarray(result.x, dtype=np.float64), float(iou_init), float(1.0 - result.fun), int(result.nit)

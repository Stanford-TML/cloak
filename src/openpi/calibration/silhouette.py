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

from collections.abc import Callable
import dataclasses
import math
from pathlib import Path
from typing import Literal

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from openpi.constants import (
    RLDS_H, RLDS_W, ROBOTIQ_SCENE_XML, SHARPA_SCENE_XML, UMI_SCENE_XML, YAM_LINEAR_SCENE_XML,
    lerp_gripper_qpos, lerp_sharpa_hand_qpos,
)
from openpi.policies import gripper_mask_renderer as _renderers

# Target-mask construction.
OPEN_TOL = 0.05         # gripper_position <= this counts as "open"
ROI_TOP_FRAC = 0.50     # gripper lives in the bottom rows / right cols of the wrist frame
ROI_LEFT_FRAC = 0.20
BIN_KEEP_FRAC = 0.50    # keep the darkest / most-rigid half of the ROI
MIN_KEEP_FRAMES = 1
RIGID_FRAC = 0.05       # most-rigid fraction of pixels that locates the end effector
RIGID_CROP_MARGIN = 10  # px of padding around the most-rigid pixels' bounding box ("rigid" crop)

# Nelder-Mead settings.
NM_MAXITER = 600
NM_INIT_T_STEP = 0.005   # 5 mm initial simplex spread (translation)
NM_INIT_R_STEP = 0.0087  # ~0.5 deg initial simplex spread (rotation)

# Geom group of the floor collision plane; hidden in the segmentation render.
_COLLISION_GROUP = 3
_SEG_HIDDEN_GROUP = 5


@dataclasses.dataclass(frozen=True)
class Embodiment:
    scene_xml: Path
    ee_bodies: list[str]
    """End-effector bodies whose geoms form the silhouette (same as its mask renderer)."""
    ee_qpos: Callable[[float], np.ndarray]
    """Gripper scalar (0 = open, 1 = closed) -> end-effector qpos (qpos[arm_dof:])."""
    arm_dof: int = 7


EMBODIMENTS = {
    "robotiq": Embodiment(ROBOTIQ_SCENE_XML, _renderers._GRIPPER_BODY_NAMES, lerp_gripper_qpos),
    "sharpa": Embodiment(SHARPA_SCENE_XML, _renderers._HAND_BODY_NAMES, lerp_sharpa_hand_qpos),
    "umi": Embodiment(UMI_SCENE_XML, _renderers._UMI_GRIPPER_BODY_NAMES, _renderers._umi_finger_qpos),
    "yam": Embodiment(YAM_LINEAR_SCENE_XML, _renderers._YAM_GRIPPER_BODY_NAMES, _renderers._yam_finger_qpos, arm_dof=6),
}


@dataclasses.dataclass(frozen=True)
class CalibOptions:
    cue: Literal["intersection", "intensity", "std"] = "intersection"
    """Target-mask cue: low intensity, low temporal std, or their intersection."""
    keep_frac: float = BIN_KEEP_FRAC
    """Fraction of ROI pixels kept by each cue's percentile threshold."""
    ee_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """End-effector RGB (0-255), e.g. from `estimate_ee_color`. The intensity cue keeps the
    pixels whose per-pixel median RGB is closest to it; the default black keeps the
    darkest pixels."""
    crop: Literal["fixed", "rigid"] = "fixed"
    """Region the target mask is restricted to: "fixed" = the bottom-right of the wrist
    view, where a Robotiq sits; "rigid" = the bounding box of the most rigid pixels
    (e.g. Sharpa, which is mounted off to the side)."""


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
    """Arm + end-effector forward kinematics and wrist-cam segmentation of the end effector."""

    EE_SITE = "attachment_site"  # DROID end-effector frame (link7 + 0.107 m on z).

    def __init__(self, embodiment: str = "robotiq") -> None:
        self.embodiment = EMBODIMENTS[embodiment]
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
        ids = {self.model.body(n).id for n in self.embodiment.ee_bodies}
        self.gripper_geom_ids = np.array(
            [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in ids])
        self._cam_id = self.model.cam("wrist_cam").id
        self._ee_site_id = self.model.site(self.EE_SITE).id

    def _build_model(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(str(self.embodiment.scene_xml))
        cam = spec.worldbody.add_camera()
        cam.name = "wrist_cam"
        cam.fovy = 60.0  # placeholder; overwritten per render
        return spec.compile()

    def set_pose(self, joint_position: np.ndarray, gripper_position: float) -> np.ndarray:
        """Set arm + end-effector qpos, run FK, return the 4x4 end-effector pose in world."""
        mujoco.mj_resetData(self.model, self.data)
        a = self.embodiment.arm_dof
        self.data.qpos[:a] = joint_position
        self.data.qpos[a:] = self.embodiment.ee_qpos(gripper_position)
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

def _select_kept(gripper_positions: np.ndarray, frames: np.ndarray):
    """RGB frames (K, H, W, 3) of the gripper-open steps, or None if too few."""
    keep = np.asarray(gripper_positions) <= OPEN_TOL
    if int(keep.sum()) < MIN_KEEP_FRAMES:
        return None
    return frames[keep]


def _temporal_std(kept: np.ndarray) -> np.ndarray:
    """Per-pixel grayscale std (H, W) across the RGB frames `kept`."""
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in kept]).std(axis=0)


def _most_rigid(std: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The most rigid RIGID_FRAC of `region`'s pixels. The end effector moves with the
    camera, so its pixels barely change across frames while the background does."""
    return (std <= np.percentile(std[region], 100 * RIGID_FRAC)) & region


def _gripper_roi(std: np.ndarray, crop: str) -> np.ndarray:
    """Where the end effector can be: "fixed" = the bottom-right of the wrist view (where
    a Robotiq sits); "rigid" = the padded bounding box of the most rigid pixels."""
    H, W = std.shape
    roi = np.zeros((H, W), dtype=bool)
    if crop == "fixed":
        roi[int(H * ROI_TOP_FRAC):, int(W * ROI_LEFT_FRAC):] = True
    else:
        ys, xs = np.nonzero(_most_rigid(std, np.ones((H, W), dtype=bool)))
        (y0, y1), (x0, x1) = np.percentile(ys, [1, 99]).astype(int), np.percentile(xs, [1, 99]).astype(int)
        m = RIGID_CROP_MARGIN
        roi[max(y0 - m, 0):y1 + m + 1, max(x0 - m, 0):x1 + m + 1] = True
    return roi


def _threshold_mask(score: np.ndarray, roi: np.ndarray, keep_frac: float) -> np.ndarray:
    """Keep the bottom `keep_frac` of `score` inside `roi`."""
    thr = float(np.percentile(score[roi], keep_frac * 100))
    return (score < thr) & roi


def build_target_mask(gripper_positions: np.ndarray, frames: np.ndarray, opts: CalibOptions = CalibOptions()):
    """Pseudo-GT gripper mask from the gripper-open RGB `frames` (N, H, W, 3): pixels in
    the gripper ROI whose median color is close to `opts.ee_color` (intensity cue)
    and/or that barely change across frames (std cue). None if too few open frames."""
    kept = _select_kept(gripper_positions, frames)
    if kept is None:
        return None
    std = _temporal_std(kept)
    roi = _gripper_roi(std, opts.crop)

    color_dist = np.linalg.norm(np.median(kept, axis=0) - opts.ee_color, axis=-1)
    intensity = _threshold_mask(color_dist, roi, opts.keep_frac)
    rigid = _threshold_mask(std, roi, opts.keep_frac)

    return {"intensity": intensity, "std": rigid, "intersection": intensity & rigid}[opts.cue]


def estimate_ee_color(gripper_positions: np.ndarray, frames: np.ndarray, crop: str = "fixed") -> np.ndarray:
    """End-effector RGB (0-255) measured from the gripper-open RGB `frames`: the median
    color of the most rigid pixels in the gripper ROI."""
    kept = _select_kept(gripper_positions, frames)
    std = _temporal_std(kept)
    rigid = _most_rigid(std, _gripper_roi(std, crop))
    return np.median(np.median(kept, axis=0)[rigid], axis=0)


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

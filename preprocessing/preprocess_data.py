#!/usr/bin/env python3
"""Preprocess DROID RLDS TFRecord data to add gripper masks and affordances.

For each episode, runs MuJoCo FK to render a wrist-camera gripper mask for
every robot in MASK_ROBOTS (Robotiq + UMI + Sharpa, each with its own camera
offset) plus a Robotiq-only EE-pose affordance at the 2F-85 pinch midpoint,
then writes the new features back into the existing RLDS TFRecord shards in
data_dir (in-place, via atomic temp-file rename).

New features added under steps/observation/:
    wrist_image_left_gripper_mask         : bytes (N,) — per-frame PNG-encoded bool
                                            mask (H, W) uint8 — Robotiq (legacy key)
    wrist_image_left_gripper_mask_umi      : same, UMI hand
    wrist_image_left_gripper_mask_sharpa   : same, Sharpa hand
    affordance_pos     : float (N, 3) — pinch-midpoint position in base frame (Robotiq)
    affordance_rot     : float (N, 3, 3) — gripper rotation matrix in base frame.
                                            Convert to 6D continuous representation at
                                            train time via R[:, :2] for the network.
    affordance_pixels  : float (N, 2) — pinch-midpoint pixel coords in wrist image
"""

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import dataclasses
import datetime
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # suppress TF C++ INFO/WARNING/ERROR startup noise

import cv2
import mediapy
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import tensorflow as tf  # noqa: E402
import tensorflow_datasets as tfds  # noqa: E402
from tqdm import tqdm
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openpi.constants import DEFAULT_CAM_TO_GRIPPER, DEFAULT_WRIST_INTRINSICS, RLDS_H, RLDS_W, WRIST_CAM_KEY, lerp_gripper_qpos  # noqa: E402
from utils import decode_camera_frames, file_path_to_lang_key, find_dataset_dir, parse_lab  # noqa: E402

from openpi.constants import YAM_GRIPPER_QPOS_CLOSED  # noqa: E402
from openpi.constants import YAM_GRIPPER_QPOS_OPEN  # noqa: E402
from openpi.constants import lerp_sharpa_hand_qpos  # noqa: E402

# All robot renders use the scene XML (floor + light) for clearer
# visualization; the floor geom has contype/conaffinity=0 in scene.xml so it
# never collides (and the segmentation pass hides it so it can't occlude the
# gripper mask).
from openpi.constants import ROBOTIQ_SCENE_XML as ROBOTIQ_XML  # noqa: E402
from openpi.constants import SHARPA_SCENE_XML as SHARPA_XML  # noqa: E402
from openpi.constants import UMI_SCENE_XML as UMI_XML  # noqa: E402
from openpi.constants import YAM_LINEAR_SCENE_XML as YAM_XML  # noqa: E402

# DROID data is recorded at 15 Hz.
DATA_FREQ = 15.0

_GRIPPER_BODY_NAMES = [
    "base_mount",
    "2f85_base",
    "right_driver",
    "right_coupler",
    "right_spring_link",
    "right_follower",
    "right_pad",
    "right_silicone_pad",
    "left_driver",
    "left_coupler",
    "left_spring_link",
    "left_follower",
    "left_pad",
    "left_silicone_pad",
]

_GRIPPER_BASE_NAMES = [
    "base_mount",
    "2f85_base",
    "right_driver",
    "right_coupler",
    "left_driver",
    "left_coupler",
]

_GRIPPER_BODY_NAMES_UMI = [
    "panda_hand",
    "left_finger",
    "right_finger",
]

# UMI/Sharpa intentionally have no separate "base" geoms: the whole hand goes
# into the gripper mask and base_masks is empty (compute_mask's dilated-base
# OR then reduces to just `masks`). Only Robotiq splits a dilated base out.
_GRIPPER_BASE_NAMES_UMI: list[str] = []

# UMI tendon ctrlrange [0, 0.04] m; wire `gripper_position` convention is
# 0=open, 1=closed but UMI's actuator semantics is inverted (ctrl=0 -> closed,
# ctrl=0.04 -> open), so qpos[7:9] = 0.04 * (1 - g).
_UMI_FINGER_RANGE = 0.04

_GRIPPER_BODY_NAMES_SHARPA = [
    "mount_right_sharpa",
    "right_hand_C_MC",
    "right_thumb_CMC_VL",
    "right_thumb_MC",
    "right_thumb_MCP_VL",
    "right_thumb_PP",
    "right_thumb_DP",
    "right_index_MCP_VL",
    "right_index_PP",
    "right_index_MP",
    "right_index_DP",
    "right_middle_MCP_VL",
    "right_middle_PP",
    "right_middle_MP",
    "right_middle_DP",
    "right_ring_MCP_VL",
    "right_ring_PP",
    "right_ring_MP",
    "right_ring_DP",
    "right_pinky_MC",
    "right_pinky_MCP_VL",
    "right_pinky_PP",
    "right_pinky_MP",
    "right_pinky_DP",
]

# Empty by design — see the _GRIPPER_BASE_NAMES_UMI note. The mount/wrist
# bodies above already carry the whole Sharpa hand into the gripper mask.
_GRIPPER_BASE_NAMES_SHARPA: list[str] = []

# YAM linear gripper: palm/motor body + the two slide fingers. No separate base
# (same convention as UMI/Sharpa — base_masks stays empty).
_GRIPPER_BODY_NAMES_YAM = [
    "gripper",
    "tip_left",
    "tip_right",
]
_GRIPPER_BASE_NAMES_YAM: list[str] = []

# Robots whose gripper mask is rendered into every episode. Robotiq keeps the
# legacy key (empty suffix) so existing dataloaders/configs/trained runs are
# unaffected; UMI/Sharpa get a `_<robot>` suffix. Only Robotiq produces the
# affordance (its 2F-85 pinch site). Order = column/render order.
MASK_ROBOTS = ("robotiq", "umi", "sharpa")
_MASK_KEY_SUFFIX = {"robotiq": "", "umi": "_umi", "sharpa": "_sharpa"}


def _mask_key(robot: str) -> str:
    """tfrecord context-feature key (sans `steps/observation/`) for a robot's
    per-frame gripper mask."""
    return f"{WRIST_CAM_KEY}_gripper_mask{_MASK_KEY_SUFFIX[robot]}"


INDENT = "  "


# ---------------------------------------------------------------------------
# MuJoCo FK helpers
# ---------------------------------------------------------------------------


# Geom group index reserved for the scene floor; disabled on segmentation
# renders so it can't occlude the gripper. Franka + gripper geoms sit in
# groups 0-3, so 5 is unused.
_SEG_HIDDEN_GROUP = 5

# Collision geoms live in group 3 across all three robot XMLs (visual is
# group 2). Both the RGB and segmentation renders show visual geoms only, so
# collision proxies (which can differ from the visual mesh — e.g. the UMI
# finger uses tri_finger.obj for visual but tri_finger_14cm.obj for collision)
# never enter the image or the mask.
_COLLISION_GROUP = 3

# Per-robot wrist-camera offset T_off. T_c2g is the DROID Robotiq-rig
# cam_to_gripper; T_off is the fixed offset into the rendered end effector's
# camera frame. Each entry is a 6-DOF [tx, ty, tz, rx, ry, rz] (meters +
# xyz-euler radians) — the same format as the stored cam_to_gripper, built
# into a 4x4 via t_c2g_from_6vec.
#
# The frame the offset is applied in differs per robot (_CAM_OFFSET_FRAME):
#   "ee"  -> T_cam = T_ee @ T_off @ T_c2g  (offset about the EE axes, then c2g)
#   "cam" -> T_cam = T_ee @ T_c2g @ T_off  (c2g first, then offset in cam frame)
# UMI's panda hand carries a -45 deg z roll vs the Robotiq mount, applied in
# the EE frame. Sharpa applies its offset in the camera frame (post-c2g).
# Robotiq is the reference (zero offset); its frame is irrelevant.
_CAM_OFFSET_6DOF = {
    "robotiq": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "umi": [0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(-45.0)],
    # "sharpa": [0.175, -0.005, 0.0, np.deg2rad(-5.0), 0.0, 0.0],
    "sharpa": np.array([0.134, -0.022, 0.006, *np.deg2rad([2.0, 20.5, 19.5])]),
    # YAM attachment_site is placed to coincide with the Robotiq EE at the
    # default pose, so a zero offset already matches the rig camera; tune from here.
    "yam": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
_CAM_OFFSET_FRAME = {"robotiq": "ee", "umi": "ee", "sharpa": "cam", "yam": "cam"}

# The Sharpa mask is rendered against a single FIXED Robotiq-rig
# cam_to_gripper (the nominal reference extrinsic), NOT the per-episode
# optimized one, so the Sharpa hand projection is consistent across episodes
# regardless of per-episode extrinsics noise. Same value as the reference
# _T_ROBOTIQ in scripts/adjust_camera_extrinsics.py.
_SHARPA_FIXED_C2G = np.array(
    [
        -0.07751704452973845,
        0.033548654422299276,
        0.0088600793817621,
        -0.32293863318387617,
        -0.0032319651694419083,
        -1.5771158951055568,
    ]
)

# Per-trajectory random camera jitter added on top of the fixed Sharpa T_off
# (composed in the camera frame via compose_T_cam's extra_offset). One delta
# is sampled per episode and held fixed across all its steps: each translation
# axis ~ U(-_SHARPA_AUG_T_M, +_SHARPA_AUG_T_M) metres and each rotation axis
# ~ U(-_SHARPA_AUG_RAD, +_SHARPA_AUG_RAD) radians.
_SHARPA_AUG_T_M = 0.01  # +/- 1 cm per translation axis
_SHARPA_AUG_RAD = np.deg2rad(1.0)  # +/- 1 deg per rotation axis

# Sharpa per-trajectory finger-closing augmentation: each finger gets a random
# weight in [0, 1] scaling how far it may close (see lerp_sharpa_hand_qpos).
# Indices into the length-5 [thumb, index, middle, ring, pinky] weight vector
# (== SHARPA_FINGER_NAMES order). The thumb and middle MUST always close fully
# (weight pinned to 1.0); the rest are sampled.
_SHARPA_FULL_CLOSE_FINGERS = (0, 2)  # thumb, middle


@dataclasses.dataclass
class EpisodeResult:
    """Per-episode FK + segmentation output from ``Sim.run_episode``.

    ``masks``/``base_masks`` are always populated (``base_masks`` is all-False
    for UMI/Sharpa — see ``_GRIPPER_BASE_NAMES_UMI``). The affordance fields are
    only computed for Robotiq (``compute_affordances=True``) and are ``None``
    otherwise.
    """

    masks: np.ndarray  # (N, H, W) bool
    base_masks: np.ndarray  # (N, H, W) bool
    affordance_pos: np.ndarray | None = None  # (N, 3) f32 — pinch midpoint, base frame
    affordance_rot: np.ndarray | None = None  # (N, 3, 3) f32 — pinch rotation, base frame
    affordance_pixels: np.ndarray | None = None  # (N, 2) i32 — pinch px in wrist image


class Sim:
    """MuJoCo simulator for FK and wrist-camera segmentation rendering."""

    # `attachment_site` (fr3_link7 + 0.107m on z) is the DROID EE frame: the
    # frame that the dataset's `cartesian_position` represents and that
    # `cam_to_gripper` is optimized against in preprocess_wrist_extrinsics.py.
    EE_SITE = "attachment_site"

    def __init__(self, robot: str = "robotiq") -> None:
        if robot not in ("robotiq", "umi", "sharpa", "yam"):
            raise ValueError(f"unknown robot: {robot!r} (expected 'robotiq', 'umi', 'sharpa', or 'yam')")
        self.robot = robot
        # Arm DOF preceding the gripper joints in qpos. Franka arms are 7-DOF;
        # the I2RT YAM is 6-DOF. Drives set_pose / _set_gripper_qpos slicing.
        self._arm_dof = {"robotiq": 7, "umi": 7, "sharpa": 7, "yam": 6}[robot]
        self.model = self._build_model()
        self.data = mujoco.MjData(self.model)
        self.seg_renderer = mujoco.Renderer(self.model, height=RLDS_H, width=RLDS_W)
        self.seg_renderer.enable_segmentation_rendering()
        self.rgb_renderer = mujoco.Renderer(self.model, height=RLDS_H, width=RLDS_W)
        # Move the floor to a dedicated geom group. RGB renders with every
        # group enabled so the floor stays visible; segmentation renders with
        # that group disabled so the floor never occludes the gripper in the
        # mask depth buffer. All robots now use the scene XML, so the `floor`
        # geom is present; the KeyError guard stays as a safety net.
        self._rgb_opt = mujoco.MjvOption()
        self._rgb_opt.geomgroup[:] = 1
        self._rgb_opt.geomgroup[_COLLISION_GROUP] = 0
        self._seg_opt = mujoco.MjvOption()
        self._seg_opt.geomgroup[:] = 1
        self._seg_opt.geomgroup[_COLLISION_GROUP] = 0
        # Site markers (e.g. UMI's left/right_gripper_site at the fingertips)
        # are drawn as spheres and get their own segid, which would punch a
        # hole in the gripper mask. Sites are only read via data.site_xpos
        # (FK, independent of MjvOption), so hiding them here is safe.
        self._rgb_opt.sitegroup[:] = 0
        self._seg_opt.sitegroup[:] = 0
        try:
            self.model.geom_group[self.model.geom("floor").id] = _SEG_HIDDEN_GROUP
            self._seg_opt.geomgroup[_SEG_HIDDEN_GROUP] = 0
        except KeyError:
            pass
        self.gripper_geom_ids, self.base_geom_ids = self._get_gripper_geom_ids()
        self._wrist_cam_id = self.model.cam("wrist_cam").id
        self._ee_site_id = self.model.site(self.EE_SITE).id
        # `pinch` is a Robotiq-only site; UMI uses `left_gripper_site`/`right_gripper_site`.
        if self.robot == "robotiq":
            self._pinch_site_id = self.model.site("pinch").id
        else:
            self._pinch_site_id = None
        # Robot base body (only used for the Robotiq affordance frame). Franka
        # robots use fr3_link0; YAM has no Franka base, so resolve its root link.
        base_name = "link1" if self.robot == "yam" else "fr3_link0"
        self._base_body_id = self.model.body(base_name).id
        # Fixed per-robot wrist-camera offset (T_off in compose_T_cam) and the
        # frame it is applied in ("ee" -> pre-c2g, "cam" -> post-c2g).
        self._cam_offset = self.t_c2g_from_6vec(np.asarray(_CAM_OFFSET_6DOF[self.robot], dtype=np.float64))
        self._cam_offset_frame = _CAM_OFFSET_FRAME[self.robot]

    def _build_model(self) -> mujoco.MjModel:
        xml = {"robotiq": ROBOTIQ_XML, "umi": UMI_XML, "sharpa": SHARPA_XML, "yam": YAM_XML}[self.robot]
        spec = mujoco.MjSpec.from_file(str(xml))
        cam = spec.worldbody.add_camera()
        cam.name = "wrist_cam"
        cam.fovy = 60.0  # placeholder; overwritten per episode
        return spec.compile()

    def _get_gripper_geom_ids(self) -> tuple[np.ndarray, np.ndarray]:
        body_names, base_names = {
            "robotiq": (_GRIPPER_BODY_NAMES, _GRIPPER_BASE_NAMES),
            "umi": (_GRIPPER_BODY_NAMES_UMI, _GRIPPER_BASE_NAMES_UMI),
            "sharpa": (_GRIPPER_BODY_NAMES_SHARPA, _GRIPPER_BASE_NAMES_SHARPA),
            "yam": (_GRIPPER_BODY_NAMES_YAM, _GRIPPER_BASE_NAMES_YAM),
        }[self.robot]
        gripper_body_ids = {self.model.body(name).id for name in body_names}
        base_body_ids = {self.model.body(name).id for name in base_names}
        all_geom_ids = np.array([i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in gripper_body_ids])
        base_geom_ids = np.array([i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] in base_body_ids])
        return all_geom_ids, base_geom_ids

    def _set_gripper_qpos(self, gripper_position: float, sharpa_finger_weights: np.ndarray | None = None) -> None:
        """Set the gripper qpos (after the arm DOFs) from the 1-DOF DROID
        `gripper_position` (0=open, 1=closed).

        Robotiq: lerp between settled open/closed configs over 8 coupled joints.
        UMI: two symmetric prismatic fingers, qpos = 0.04 * (1 - g) on each
        (no settle step — slides are linear in the actuator coordinate).
        Sharpa: lerp between settled open/closed 22-DOF hand configs (thumb +
        index curl on close, other fingers extended — emulates a Robotiq pinch).
        YAM: two coupled prismatic slide fingers, lerped open<->closed.
        `sharpa_finger_weights` (length-5, Sharpa only) scales per-finger
        closing; see `lerp_sharpa_hand_qpos`. Ignored for the others.
        """
        a = self._arm_dof
        if self.robot == "robotiq":
            self.data.qpos[a:] = lerp_gripper_qpos(gripper_position)
        elif self.robot == "umi":
            self.data.qpos[a : a + 2] = _UMI_FINGER_RANGE * (1.0 - float(gripper_position))
        elif self.robot == "yam":
            g = float(gripper_position)
            self.data.qpos[a : a + 2] = (1.0 - g) * YAM_GRIPPER_QPOS_OPEN + g * YAM_GRIPPER_QPOS_CLOSED
        else:  # sharpa
            # The G2D constants are 22-DOF (full hand); this repo's XML
            # comments out middle/ring/pinky joints, so the active model has
            # only thumb+index (9 DOFs). Slice to whatever the model has so
            # we tolerate either configuration without code changes.
            n_hand = self.model.nq - a
            self.data.qpos[a:] = lerp_sharpa_hand_qpos(gripper_position, sharpa_finger_weights)[:n_hand]

    def _set_camera_from_T_cam(self, T_cam: np.ndarray) -> None:
        """Update wrist_cam pose from a 4x4 cam-to-world matrix (CV convention)."""
        flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        self.data.cam_xpos[self._wrist_cam_id] = T_cam[:3, 3]
        self.data.cam_xmat[self._wrist_cam_id] = (T_cam[:3, :3] @ flip).flatten()

    def _set_wrist_cam_fovy(self, fy: float) -> None:
        """Vertical FOV (degrees) from focal length `fy` at render height ``RLDS_H``."""
        self.model.cam_fovy[self._wrist_cam_id] = math.degrees(2.0 * math.atan(RLDS_H / (2.0 * fy)))

    def set_pose(
        self,
        joint_position: np.ndarray,
        gripper_position: float,
        sharpa_finger_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reset MjData, set qpos (arm + gripper via robot-specific rule), run
        mj_forward, return T_ee (4x4 attachment_site pose in world).

        Pure FK — no physics. Bit-identical with downstream consumers that re-run
        mj_forward at the same joint_position + gripper_position.
        `sharpa_finger_weights` (length-5, Sharpa only) is forwarded to
        `_set_gripper_qpos`; see `lerp_sharpa_hand_qpos`.
        """
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: self._arm_dof] = joint_position
        self._set_gripper_qpos(gripper_position, sharpa_finger_weights)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        T_ee = np.eye(4)
        T_ee[:3, 3] = self.data.site_xpos[self._ee_site_id]
        T_ee[:3, :3] = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        return T_ee

    @staticmethod
    def t_c2g_from_6vec(cam_to_gripper: np.ndarray) -> np.ndarray:
        """Build a 4x4 ``T_c2g`` from the 6-vector ``[tx, ty, tz, rx, ry, rz]``
        (xyz translation + xyz Tait-Bryan euler) — the stored cam_to_gripper format."""
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler("xyz", cam_to_gripper[3:]).as_matrix()
        T[:3, 3] = cam_to_gripper[:3]
        return T

    def render_mask(self, T_cam: np.ndarray, fy: float) -> tuple[np.ndarray, np.ndarray]:
        """Render the wrist-cam segmentation at the given camera pose ``T_cam``
        (4x4 cam-to-world matrix, CV convention) with vertical FOV from ``fy``.

        Assumes ``set_pose`` has been called this step so qpos / FK are current.
        Returns ``(gripper_mask, base_mask)`` — both ``(H, W)`` bool arrays.
        """
        self._set_wrist_cam_fovy(fy)
        self._set_camera_from_T_cam(T_cam)
        self.seg_renderer.update_scene(self.data, camera="wrist_cam", scene_option=self._seg_opt)
        seg = self.seg_renderer.render()
        geom_ids = seg[:, :, 0]
        return (
            np.isin(geom_ids, self.gripper_geom_ids),
            np.isin(geom_ids, self.base_geom_ids),
        )

    def render_rgb(self, T_cam: np.ndarray, fy: float) -> np.ndarray:
        """Render the wrist-cam RGB scene at the given camera pose ``T_cam`` with
        vertical FOV from ``fy``. Assumes ``set_pose`` has been called this step.
        Returns ``(H, W, 3)`` uint8.
        """
        self._set_wrist_cam_fovy(fy)
        self._set_camera_from_T_cam(T_cam)
        self.rgb_renderer.update_scene(self.data, camera="wrist_cam", scene_option=self._rgb_opt)
        return self.rgb_renderer.render()

    def compose_T_cam(
        self,
        T_ee: np.ndarray,
        cam_to_gripper: np.ndarray,
        *,
        cam_offset: np.ndarray | None = None,
        cam_offset_frame: str | None = None,
        extra_offset: np.ndarray | None = None,
    ) -> np.ndarray:
        """4x4 ``T_cam`` (cam-to-world, CV convention) with the per-robot offset.

        ``T_c2g`` is built from the stored DROID Robotiq-rig ``cam_to_gripper``
        6-vector; ``T_off`` is the per-robot offset (identity for Robotiq). The
        frame it is applied in is per-robot:
          - ``"ee"``  -> ``T_ee @ T_off @ T_c2g`` (offset about the EE axes,
            then c2g; UMI / Robotiq).
          - ``"cam"`` -> ``T_ee @ T_c2g @ T_off`` (c2g first, then offset in
            the camera frame; Sharpa).
        Single source of truth for the composition so the visualization test
        and any preprocessing path can't drift.

        ``cam_offset`` / ``cam_offset_frame`` override the instance defaults
        (``self._cam_offset`` / ``self._cam_offset_frame``) when given —
        used by the interactive extrinsics tuner to sweep an offset without
        mutating the Sim. ``cam_offset`` may be a 6-vector ``[tx, ty, tz, rx,
        ry, rz]`` (built via :meth:`t_c2g_from_6vec`) or a 4x4 matrix.

        ``extra_offset`` (6-vector or 4x4) is an additional delta composed in
        ``T_off``'s own frame, right after it (``T_off @ T_extra``) — used by
        ``run_episode`` to apply a per-trajectory random Sharpa camera jitter
        on top of the fixed ``T_off``. Identity / no-op when omitted.
        """
        T_c2g = self.t_c2g_from_6vec(np.asarray(cam_to_gripper, dtype=np.float64))
        if cam_offset is None:
            T_off = self._cam_offset
        else:
            cam_offset = np.asarray(cam_offset, dtype=np.float64)
            T_off = cam_offset if cam_offset.shape == (4, 4) else self.t_c2g_from_6vec(cam_offset)
        if extra_offset is not None:
            extra_offset = np.asarray(extra_offset, dtype=np.float64)
            T_extra = extra_offset if extra_offset.shape == (4, 4) else self.t_c2g_from_6vec(extra_offset)
            T_off = T_off @ T_extra
        frame = self._cam_offset_frame if cam_offset_frame is None else cam_offset_frame
        if frame == "ee":
            return T_ee @ T_off @ T_c2g
        return T_ee @ T_c2g @ T_off

    def render_ee(
        self,
        joint_position: np.ndarray,
        gripper_position: float,
        cam_to_gripper: np.ndarray,
        fy: float,
        *,
        rgb: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """FK + per-robot camera offset + render in one call.

        Sets qpos from ``(joint_position, gripper_position)``, composes
        ``T_cam`` via the per-robot offset, and renders the gripper
        segmentation mask (plus RGB unless ``rgb=False``). Returns
        ``(gripper_mask, rgb_or_None)``. Callers needing the base mask use
        ``set_pose`` + ``compose_T_cam`` + ``render_mask`` directly.
        """
        T_ee = self.set_pose(joint_position, gripper_position)
        T_cam = self.compose_T_cam(T_ee, cam_to_gripper)
        mask, _ = self.render_mask(T_cam, fy)
        img = self.render_rgb(T_cam, fy) if rgb else None
        return mask, img

    def run_episode(
        self,
        steps: list,
        intrinsics: np.ndarray,
        cam_to_gripper: np.ndarray,
        *,
        compute_affordances: bool,
    ) -> EpisodeResult:
        """Run FK + segmentation for one episode, for any robot.

        Per step the standard Sim pipeline is used so every consumer composes
        the camera identically: ``set_pose`` (arm qpos direct, gripper qpos
        lerped between settled open/closed configs, ``mj_forward``, return the
        EE site pose) -> ``compose_T_cam`` (apply this robot's fixed
        ``_CAM_OFFSET_6DOF`` offset to the DROID-rig ``cam_to_gripper``;
        identity for Robotiq, so ``T_cam = T_ee @ T_c2g`` exactly as before)
        -> ``render_mask``. Pure FK means downstream consumers re-running
        ``mj_forward`` at the same joint/gripper position get bit-identical
        ``site_xpos``.

        ``masks``/``base_masks`` are always returned. ``base_masks`` is
        all-False for UMI/Sharpa (no separate base bodies — see
        ``_GRIPPER_BASE_NAMES_UMI``).

        Sharpa per-trajectory augmentations (sampled once per call, held fixed
        across all steps):
          - ``cam_to_gripper`` is replaced by the fixed ``_SHARPA_FIXED_C2G``
            (consistent projection across episodes, ignores the per-episode one).
          - a random camera-frame jitter (translation axis ~ U(+/-
            ``_SHARPA_AUG_T_M``) m, rotation axis ~ U(+/- ``_SHARPA_AUG_R_DEG``)
            deg) applied via ``compose_T_cam(extra_offset=...)``.
          - a per-finger closing-weight vector (thumb and middle pinned to 1.0
            so they always close fully; the rest ~ U(0, 1)). See
            ``lerp_sharpa_hand_qpos`` / ``_SHARPA_FULL_CLOSE_FINGERS``.

        ``compute_affordances`` (Robotiq only): also read the 2F-85 ``pinch``
        site (gripper midpoint, on ``2f85_base``) in the robot base frame
        (``fr3_link0``) and its pixel projection through the same per-step
        ``T_cam``. The full 3x3 rotation is stored; conversion to the 6D
        continuous representation (Zhou et al. 2019, ``R[:, :2]``) happens at
        train time. Raises if requested for a robot without a pinch site.
        """
        if compute_affordances and self._pinch_site_id is None:
            raise NotImplementedError(
                "compute_affordances needs the 2F-85 pinch site (Robotiq only); " f"got robot={self.robot!r}"
            )
        fy = float(intrinsics[1])
        fx, _, cx, cy = (float(v) for v in intrinsics)

        # Sharpa per-trajectory augmentations, all held fixed across steps:
        # fixed reference c2g, a +/- translation-only camera jitter, and a
        # finger-closing weight vector (thumb & middle MUST close fully).
        sharpa_finger_weights = None
        extra_offset = None
        if self.robot == "sharpa":
            cam_to_gripper = _SHARPA_FIXED_C2G
            extra_offset = np.zeros(6)
            extra_offset[:3] = np.random.uniform(-_SHARPA_AUG_T_M, _SHARPA_AUG_T_M, 3)
            extra_offset[3:] = np.random.uniform(-_SHARPA_AUG_RAD, _SHARPA_AUG_RAD, 3)
            sharpa_finger_weights = np.random.uniform(0.0, 1.0, 5)
            sharpa_finger_weights[list(_SHARPA_FULL_CLOSE_FINGERS)] = 1.0

        positions, rotations, masks, base_masks, pixels = [], [], [], [], []
        for step in steps:
            arm_pos = step["observation"]["joint_position"].numpy()
            # NOTE: It is important to use gripper_position here, not gripper_action.
            grip = step["observation"]["gripper_position"].numpy().item()
            T_ee = self.set_pose(arm_pos, grip, sharpa_finger_weights)
            T_cam = self.compose_T_cam(T_ee, cam_to_gripper, extra_offset=extra_offset)

            mask, base_mask = self.render_mask(T_cam, fy)
            masks.append(mask)
            base_masks.append(base_mask)

            if not compute_affordances:
                continue

            # World → base transform (base = fr3_link0). fr3_link0 is fixed at the world
            # origin today, but resolving the transform explicitly keeps things correct
            # if the base ever gets moved.
            R_wb = self.data.xmat[self._base_body_id].reshape(3, 3)
            t_wb = self.data.xpos[self._base_body_id]
            R_bw = R_wb.T
            t_bw = -R_bw @ t_wb

            pinch_pos_w = self.data.site_xpos[self._pinch_site_id]
            pinch_mat_w = self.data.site_xmat[self._pinch_site_id].reshape(3, 3)
            pinch_pos_b = R_bw @ pinch_pos_w + t_bw
            pinch_mat_b = R_bw @ pinch_mat_w
            positions.append(pinch_pos_b.copy())
            rotations.append(pinch_mat_b.copy())

            # Pixel projection uses the *world*-frame pinch position since T_cam is in world.
            world_to_cam = np.linalg.inv(T_cam)
            p_cam = world_to_cam @ np.append(pinch_pos_w, 1.0)
            x, y, z = p_cam[:3]
            pixels.append([int(round(fx * x / z + cx)), int(round(fy * y / z + cy))])

        result = EpisodeResult(
            masks=np.stack(masks),  # (N, H, W)
            base_masks=np.stack(base_masks),  # (N, H, W)
        )
        if compute_affordances:
            result.affordance_pos = np.stack(positions).astype(np.float32)  # (N, 3)
            result.affordance_rot = np.stack(rotations).astype(np.float32)  # (N, 3, 3)
            result.affordance_pixels = np.array(pixels, dtype=np.int32)  # (N, 2)
        return result


# ---------------------------------------------------------------------------
# TFRecord / RLDS helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Camera metadata helpers (JSON cache lookups)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MissFlags:
    """Per-episode tracking of which JSON caches missed (and we fell back to defaults)."""

    serial: bool = False
    intrinsics: bool = False
    cam_to_gripper: bool = False
    lang: bool = False


def build_episode_metadata(
    nfs_path: str,
    lang_key: str | None,
    serials_lookup: dict[str, str],
    intrinsics_lookup: dict[str, dict[str, np.ndarray]],
    cam_to_gripper_lookup: dict[str, np.ndarray],
    lang_lookup: dict[str, tuple[str, str, str]],
) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str] | None, MissFlags, list[str]]:
    """JSON-only metadata lookup for one episode.

    Each of the four caches is consulted by `lang_key`; misses fall back to
    `DEFAULT_WRIST_INTRINSICS` / `DEFAULT_CAM_TO_GRIPPER` (or None for language
    annotations, in which case the original instructions are kept) and are
    recorded in the returned `MissFlags` for later aggregation.

    Returns (intrinsics, cam_to_gripper, lang_annotations, miss_flags, log_lines).
    """
    log_lines: list[str] = []
    flags = MissFlags()

    serial = serials_lookup.get(lang_key) if lang_key else None
    if serial is None:
        flags.serial = True
        log_lines.append(f"{INDENT}WARN: no wrist serial for {lang_key or nfs_path}")

    intrinsics = intrinsics_lookup.get(lang_key, {}).get(serial) if (lang_key and serial) else None
    if intrinsics is None:
        flags.intrinsics = True
        log_lines.append(f"{INDENT}WARN: no intrinsics for {lang_key or nfs_path}, using DEFAULT_WRIST_INTRINSICS")
        intrinsics = DEFAULT_WRIST_INTRINSICS

    cam_to_gripper = cam_to_gripper_lookup.get(lang_key) if lang_key else None
    if cam_to_gripper is None:
        flags.cam_to_gripper = True
        log_lines.append(f"{INDENT}WARN: no cam_to_gripper for {lang_key or nfs_path}, using DEFAULT_CAM_TO_GRIPPER")
        cam_to_gripper = DEFAULT_CAM_TO_GRIPPER

    lang_annotations = lang_lookup.get(lang_key) if lang_key else None
    if lang_annotations is None:
        flags.lang = True
        log_lines.append(f"{INDENT}WARN: no language annotations for {lang_key or nfs_path}, keeping original")

    return (
        np.asarray(intrinsics, dtype=np.float64),
        np.asarray(cam_to_gripper, dtype=np.float64),
        lang_annotations,
        flags,
        log_lines,
    )


_BG_COLOR = np.array([0, 255, 0], dtype=np.uint8)  # background mask

# Opacity of the mask fill in the visualization video: <1 blends the fill with
# the underlying wrist RGB at masked pixels so the gripper shows through.
_VIS_MASK_ALPHA = 0.5


def dilate_masks(masks: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Dilate each mask with a square kernel to better cover the gripper silhouette."""
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = np.array([cv2.dilate(mask.astype(np.uint8), kernel) for mask in masks])
    return dilated.astype(bool)


# TODO: Probably want to make this online as well.
def _fill_shifted(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked pixels using a randomly shifted copy of the frame."""
    h, w = frame.shape[:2]
    dy = np.random.randint(h // 2, h)
    dx = np.random.randint(w // 2, w)
    shifted = np.roll(np.roll(frame, dy, axis=0), dx, axis=1)
    result = frame.copy()
    result[mask] = shifted[mask]
    return result


def apply_masks(
    frames: np.ndarray,
    masks: np.ndarray,
    fill_mask: bool = False,
    alpha: float = 1.0,
) -> np.ndarray:
    """Replace background pixels (outside mask) in each frame. Returns uint8 (N, H, W, 3).

    masks: (N, H, W) bool array — True where the gripper is.
    fill_mask=True fills with a randomly shifted copy; False fills with solid _BG_COLOR.
    alpha: fill opacity at masked pixels. 1.0 (default) = opaque replacement
    (original behaviour, bit-identical); <1 blends the fill with the original
    frame so the gripper RGB shows through.
    """
    result = frames.copy()
    for i, mask in enumerate(masks):
        fill = _fill_shifted(frames[i], mask) if fill_mask else _BG_COLOR
        if alpha >= 1.0:
            result[i][mask] = fill[mask] if fill_mask else fill
        else:
            orig = frames[i][mask].astype(np.float32)
            fill_px = (fill[mask] if fill_mask else fill).astype(np.float32)
            result[i][mask] = ((1.0 - alpha) * orig + alpha * fill_px).round().astype(np.uint8)
    return result


def compute_mask(masks: np.ndarray, base_masks: np.ndarray) -> np.ndarray:
    """Combine full and base masks into a single (N, H, W) bool mask per frame."""
    base_masks = dilate_masks(base_masks, kernel_size=16)
    return np.logical_or(masks, base_masks)


# ---------------------------------------------------------------------------
# RLDS / TFRecord helpers
# ---------------------------------------------------------------------------


def update_features_json(dataset_dir: Path) -> None:
    """Add preprocessed observation keys to features.json so TFDS exposes them."""
    import json

    features_path = dataset_dir / "features.json"
    if not features_path.exists():
        print(f"  Warning: features.json not found at {features_path}, skipping update.")
        return

    d = json.loads(features_path.read_text())
    obs = d["featuresDict"]["features"]["steps"]["sequence"]["feature"]["featuresDict"]["features"]["observation"][
        "featuresDict"
    ]["features"]
    ep_meta = d["featuresDict"]["features"]["episode_metadata"]["featuresDict"]["features"]

    changed = False

    episode_metadata_flag_specs = {
        "extrinsics_found": {
            "description": "1 if the per-episode cam_to_gripper extrinsics came from the "
            "preprocess_wrist_extrinsics.py JSON cache; 0 if we fell back to DEFAULT_CAM_TO_GRIPPER.",
            "tensor": {"dtype": "int64", "encoding": "none", "shape": {}},
            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
        },
    }
    for key, spec in episode_metadata_flag_specs.items():
        if ep_meta.get(key) != spec:
            ep_meta[key] = spec
            changed = True

    for robot in MASK_ROBOTS:
        key = _mask_key(robot)
        if key in obs:
            continue
        desc = "Binary gripper segmentation mask for wrist camera image, " "PNG-encoded uint8."
        if robot != "robotiq":
            desc = (
                f"Binary {robot} hand segmentation mask for the wrist camera "
                f"image (rendered with the {robot} Sim's camera offset), "
                "PNG-encoded uint8."
            )
        obs[key] = {
            "description": desc,
            "image": {
                "dtype": "uint8",
                "encodingFormat": "png",
                "shape": {"dimensions": ["180", "320", "1"]},
            },
            "pythonClassName": "tensorflow_datasets.core.features.image_feature.Image",
        }
        changed = True

    affordance_specs = {
        "affordance_pos": {
            "description": "Gripper pinch-midpoint 3-D position in robot base frame, shape (3,).",
            "tensor": {
                "dtype": "float32",
                "shape": {"dimensions": ["3"]},
            },
            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
        },
        "affordance_rot": {
            "description": "Gripper rotation in base frame as a full 3x3 rotation matrix. "
            "Convert to 6D continuous representation (Zhou et al. 2019) at "
            "train time via R[:, :2] before feeding the network.",
            "tensor": {
                "dtype": "float32",
                "shape": {"dimensions": ["3", "3"]},
            },
            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
        },
        "affordance_pixels": {
            "description": "Gripper pinch-midpoint pixel coordinates in wrist image, shape (2,).",
            "tensor": {
                "dtype": "float32",
                "shape": {"dimensions": ["2"]},
            },
            "pythonClassName": "tensorflow_datasets.core.features.tensor_feature.Tensor",
        },
    }
    for key, spec in affordance_specs.items():
        if obs.get(key) != spec:
            obs[key] = spec
            changed = True

    if changed:
        features_path.write_text(json.dumps(d, indent=2))
        print(f"  Updated features.json at {features_path}")
    else:
        print(f"  features.json already up to date at {features_path}")


def write_readme(dataset_dir: Path) -> None:
    """Write a README.md to dataset_dir describing the added features."""
    readme = dataset_dir / "README.md"
    last_processed = datetime.date.today().strftime("%Y-%m-%d")
    readme.write_text(f"""\
**Last processed:** {last_processed}

# DROID Dataset — Preprocessed Extension

This directory contains the original [DROID](https://droid-dataset.github.io/) RLDS/TFDS
dataset with additional per-episode features added in-place.

The original DROID data is unchanged, including both exterior camera views.
New features have been appended to each episode's TFRecord using a custom preprocessing script.

If you have any questions, contact Michael (mpiseno@stanford.edu).

---

## Added features

All new features are stored in `context.feature` of each episode's `SequenceExample`
proto under the `steps/observation/` namespace.

### `steps/observation/exterior_image_1_left`, `steps/observation/exterior_image_2_left`
- **Type:** `bytes_list` — one entry per timestep
- **Encoding:** JPEG-encoded `uint8` RGB image
- **Description:** The two original DROID exterior camera views, both retained unchanged.
  Select or sample among them downstream at train time.

### `steps/observation/wrist_image_left_gripper_mask[_umi|_sharpa]`
- **Type:** `bytes_list` — one entry per timestep, one key per robot
- **Encoding:** PNG-encoded `uint8` (0 = background, 1 = gripper), decode with `cv2.imdecode` or `tf.io.decode_png`
- **Description:** Binary segmentation mask of the gripper/hand pixels in the wrist
  camera image, computed via MuJoCo forward kinematics. One key per robot in
  `MASK_ROBOTS`, each rendered with that robot's own camera offset against the
  shared DROID-rig `cam_to_gripper`:
  - `wrist_image_left_gripper_mask` — Franka FR3 + Robotiq 2F-85 (legacy key). Covers
    the full gripper assembly plus a dilated gripper base for model/reality misalignment.
  - `wrist_image_left_gripper_mask_umi` — Franka FR3 + UMI hand (whole hand, no base).
  - `wrist_image_left_gripper_mask_sharpa` — Franka FR3 + Sharpa hand (whole hand, no base).

### `steps/observation/affordance_pixels`
- **Type:** `float_list` — `N * 2` values, reshape to `(N, 2)`
- **Shape:** `(N, 2)` — `[px_x, px_y]` per frame
- **Description:** Projected pixel coordinates of the gripper pinch-midpoint site in
  the wrist camera image, obtained by running FK and projecting the 3-D site position
  through the per-step calibrated wrist camera intrinsics and extrinsics.

### `steps/observation/affordance_pos`
- **Type:** `float_list` — `N * 3` values, reshape to `(N, 3)`
- **Shape:** `(N, 3)` — `[x, y, z]` per frame
- **Description:** 3-D position of the gripper pinch-midpoint site (the standard Robotiq
  2F-85 TCP, on the `2f85_base` body, midway between the two fingers) in the robot base
  frame, computed via MuJoCo FK from the recorded joint positions.

### `steps/observation/affordance_rot`
- **Type:** `float_list` — `N * 9` values, reshape to `(N, 3, 3)`
- **Shape:** `(N, 3, 3)` — full rotation matrix per frame
- **Description:** Gripper orientation in the robot base frame, stored as the full 3×3
  rotation matrix for ease of visualization, sanity checks, and pose composition. At
  train time, convert to the 6D continuous representation (Zhou et al. 2019) by taking
  the first two columns: `rot6d = R[:, :2].reshape(-1, 6)` — row-major flatten gives
  `[r00, r01, r10, r11, r20, r21]` (interleaved between cols 0 and 1, NOT `[col0; col1]`).

---

## Potential Issues

- The gripper masks are generated from a MuJoCo simulation using the same robot model as DROID,
  but they may not perfectly align with the real gripper in the videos due to inaccuracies in the camera extrinsics.
- The same potential inaccuracies may affect the affordance pixel coordinates, since these are computed by projecting the FK-computed fingertip positions through the camera intrinsics and extrinsics.
""")
    print(f"{INDENT}Wrote {readme}")


# ---------------------------------------------------------------------------
# Language annotation helpers
# ---------------------------------------------------------------------------

LANG_INSTR_KEYS = (
    "steps/language_instruction",
    "steps/language_instruction_2",
    "steps/language_instruction_3",
)

# Project-level processed metadata assets, all keyed by '<LAB>|<timestamp>'
# (e.g. 'IRIS|2023-04-25-11h-42m-28s'). Built offline by:
#   - preprocess_metadata.py         → lang annotations + intrinsics + ZED serials
#   - preprocess_wrist_extrinsics.py → optimized wrist cam_to_gripper poses
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
LANG_ANNOTATIONS_ASSET = _ASSETS_DIR / "droid_lang_annotations_processed.json"
INTRINSICS_ASSET = _ASSETS_DIR / "droid_intrinsics_processed.json"
ZED_SERIALS_ASSET = _ASSETS_DIR / "droid_zed_serials.json"
CAM_TO_GRIPPER_ASSET = _ASSETS_DIR / "droid_wrist_extrinsics.json"


def load_language_annotations() -> dict[str, tuple[str, str, str]]:
    """Load the processed (LAB, timestamp)-keyed annotations asset."""
    raw = json.loads(LANG_ANNOTATIONS_ASSET.read_text())
    return {k: tuple(v) for k, v in raw.items()}


def load_intrinsics() -> dict[str, dict[str, np.ndarray]]:
    """Load the processed (LAB, timestamp) -> {serial: [fx, fy, cx, cy]} intrinsics asset.

    Values are pre-scaled to RLDS resolution (CAM_SCALE applied at build time).
    """
    raw = json.loads(INTRINSICS_ASSET.read_text())
    return {k: {s: np.asarray(v, dtype=np.float64) for s, v in cams.items()} for k, cams in raw.items()}


def load_wrist_serials() -> dict[str, str]:
    """Load (LAB, timestamp) -> wrist ZED serial from droid_zed_serials.json."""
    raw = json.loads(ZED_SERIALS_ASSET.read_text())
    return {k: v["wrist"] for k, v in raw.items() if "wrist" in v}


def load_cam_to_gripper() -> dict[str, np.ndarray]:
    """Load (LAB, timestamp) -> 6-DoF cam_to_gripper [dx,dy,dz,drx,dry,drz]; on-disk format `{key: [6 floats]}`."""
    raw = json.loads(CAM_TO_GRIPPER_ASSET.read_text())
    return {k: np.asarray(v) for k, v in raw.items()}


def record_has_masks(raw_record: bytes) -> bool:
    """Return True if this TFRecord was produced by the *current* preprocess schema.

    Old runs (fingertip-pair affordance) wrote 6 floats per step under
    affordance_pos; the current schema writes 3 (pinch midpoint). Requiring
    the mask key for *every* robot in MASK_ROBOTS (Robotiq + UMI + Sharpa, all
    with the same per-step count) AND a 3-floats-per-step affordance_pos
    prevents a stale shard — including a Robotiq-only one from before the
    multi-robot masks landed — from being skipped after a schema change.
    """
    seq_ex = tf.train.SequenceExample()
    seq_ex.ParseFromString(raw_record)
    feats = seq_ex.context.feature
    if "steps/observation/affordance_rot" not in feats:
        return False
    if "steps/observation/affordance_pos" not in feats:
        return False
    n_steps = None
    for robot in MASK_ROBOTS:
        key = f"steps/observation/{_mask_key(robot)}"
        if key not in feats:
            return False
        n = len(feats[key].bytes_list.value)
        if n_steps is None:
            n_steps = n
        elif n != n_steps:
            return False
    n_pos_floats = len(feats["steps/observation/affordance_pos"].float_list.value)
    return bool(n_steps) and n_pos_floats == 3 * n_steps


def build_episode_record(
    raw_record: bytes,
    frame_masks_by_key: dict[str, np.ndarray],
    extrinsics_found: bool,
    extra_obs: dict[str, np.ndarray] | None = None,
    language_annotations: tuple[str, str, str] | None = None,
) -> bytes:
    """Return modified episode bytes by adding new features to the proto.

    Parses the original SequenceExample, writes each per-robot gripper mask as
    a PNG-encoded bytes feature list under steps/observation/<key> (keys from
    `frame_masks_by_key`, e.g. wrist_image_left_gripper_mask[_umi|_sharpa]),
    appends extra_obs as new float feature lists under steps/observation/<key>,
    then serializes. All other fields are preserved verbatim.

    frame_masks_by_key   : {feature_key: (N, H, W) mask array} — one entry per robot
    extra_obs            : per-step float arrays, shape (N, ...) — stored flat as N*... floats
    extrinsics_found : True if the per-episode cam_to_gripper extrinsics came from the
                           JSON cache; False if we fell back to DEFAULT_CAM_TO_GRIPPER.
                           Stored as int64 under episode_metadata/extrinsics_found.
    """
    seq_ex = tf.train.SequenceExample()
    seq_ex.ParseFromString(raw_record)

    for key, masks in frame_masks_by_key.items():
        feat = seq_ex.context.feature[f"steps/observation/{key}"]
        del feat.bytes_list.value[:]
        for mask in masks:
            _, png_bytes = cv2.imencode(".png", mask.astype(np.uint8))
            feat.bytes_list.value.append(png_bytes.tobytes())

    for obs_key, values in (extra_obs or {}).items():
        feat = seq_ex.context.feature[f"steps/observation/{obs_key}"]
        del feat.float_list.value[:]
        for row in values:
            feat.float_list.value.extend(row.flatten().tolist())

    feat = seq_ex.context.feature["episode_metadata/extrinsics_found"]
    del feat.int64_list.value[:]
    feat.int64_list.value.append(int(extrinsics_found))

    # Overwrite the three per-step language instruction features with the new annotations.
    # If no annotation was found for this episode, leave the existing values alone.
    if language_annotations is not None:
        n_steps = len(next(iter(frame_masks_by_key.values())))
        for key, instr in zip(LANG_INSTR_KEYS, language_annotations):
            feat = seq_ex.context.feature[key]
            del feat.bytes_list.value[:]
            encoded = instr.encode("utf-8")
            for _ in range(n_steps):
                feat.bytes_list.value.append(encoded)

    # Both exterior views (exterior_image_1_left / exterior_image_2_left) are kept as-is.
    return seq_ex.SerializeToString()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def overlay_affordances(frames: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Overlay projected gripper pinch-midpoint dot on wrist frames.

    pixels : (N, 2) int — pinch-midpoint [px_x, px_y] per frame.
    """
    result = frames.copy()
    for i, px in enumerate(pixels):
        cv2.circle(result[i], tuple(int(v) for v in px), 5, (255, 255, 255), -1)
    return result


def save_vis_video(
    video_path: Path,
    steps: list,
    gripper_mask: np.ndarray,
    affordance_pixels: np.ndarray,
    fill_mask: bool,
    umi_mask: np.ndarray | None = None,
    sharpa_mask: np.ndarray | None = None,
) -> None:
    """Row 1: [wrist rgb | robotiq-masked + affordance | exterior].
    Row 2 (when umi_mask/sharpa_mask given): [umi-masked | sharpa-masked |
    black placeholder] — the UMI/Sharpa masks applied to the wrist rgb the
    same way as row 1's robotiq panel, padded with black so both rows have
    the same column count.
    """
    frames = decode_camera_frames(steps, WRIST_CAM_KEY)
    masked_vis = apply_masks(frames, gripper_mask, fill_mask=fill_mask, alpha=_VIS_MASK_ALPHA)
    masked_vis = overlay_affordances(masked_vis, affordance_pixels)
    exterior_frames = decode_camera_frames(steps, "exterior_image_1_left")
    combined = np.concatenate([frames, masked_vis, exterior_frames], axis=2)
    if umi_mask is not None and sharpa_mask is not None:
        umi_vis = apply_masks(frames, umi_mask, fill_mask=fill_mask, alpha=_VIS_MASK_ALPHA)
        sharpa_vis = apply_masks(frames, sharpa_mask, fill_mask=fill_mask, alpha=_VIS_MASK_ALPHA)
        row2 = np.concatenate([umi_vis, sharpa_vis, np.zeros_like(frames)], axis=2)
        combined = np.concatenate([combined, row2], axis=1)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(video_path), combined, fps=int(DATA_FREQ))


_REQUIRED_OBS_KEYS = {
    "joint_position",
    "gripper_position",
    WRIST_CAM_KEY,
    "exterior_image_1_left",
    "exterior_image_2_left",
}


def validate_dataset(
    ds: tf.data.Dataset,
    shard_files: list[Path],
) -> None:
    """Sanity-check dataset structure before the main processing loop."""
    assert len(shard_files) > 0, "No shard files found"

    # Observation keys and step count on the first episode
    first_episode = next(iter(ds))
    first_steps = list(first_episode["steps"])
    assert len(first_steps) > 0, "First episode has no steps"
    obs_keys = set(first_steps[0]["observation"].keys())
    missing = _REQUIRED_OBS_KEYS - obs_keys
    assert not missing, f"Missing observation keys in dataset: {missing}"

    # Metadata assets are loaded lazily by each worker, so check upfront that
    # they exist rather than hitting a per-worker error mid-run.
    for asset in (LANG_ANNOTATIONS_ASSET, INTRINSICS_ASSET, ZED_SERIALS_ASSET, CAM_TO_GRIPPER_ASSET):
        assert asset.exists(), f"Metadata asset not found at {asset}."

    print(f"{INDENT}Validation passed")


@dataclasses.dataclass
class ShardStats:
    """Per-shard counters and buffered log lines returned by process_shard.

    Log lines are buffered (rather than printed inline) so the main process can
    emit them via tqdm.write without racing with the progress bar.

    `*_misses` are per-lab dicts counting episodes for which the corresponding
    JSON cache lookup missed and we fell back to the default in utils.py.
    """

    n_processed: int = 0
    n_failed: int = 0
    serial_misses: dict[str, int] = dataclasses.field(default_factory=dict)
    serial_misses_by_outcome: dict[str, int] = dataclasses.field(default_factory=dict)
    intrinsics_misses: dict[str, int] = dataclasses.field(default_factory=dict)
    intrinsics_misses_by_outcome: dict[str, int] = dataclasses.field(default_factory=dict)
    cam_to_gripper_misses: dict[str, int] = dataclasses.field(default_factory=dict)
    cam_to_gripper_misses_by_outcome: dict[str, int] = dataclasses.field(default_factory=dict)
    lang_no_annotation: dict[str, int] = dataclasses.field(default_factory=dict)
    lang_no_annotation_by_outcome: dict[str, int] = dataclasses.field(default_factory=dict)
    logs: list[str] = dataclasses.field(default_factory=list)

    def merge(self, other: "ShardStats") -> None:
        """Accumulate counters from another ShardStats. Logs are not merged."""
        self.n_processed += other.n_processed
        self.n_failed += other.n_failed
        for src, dst in (
            (other.serial_misses, self.serial_misses),
            (other.serial_misses_by_outcome, self.serial_misses_by_outcome),
            (other.intrinsics_misses, self.intrinsics_misses),
            (other.intrinsics_misses_by_outcome, self.intrinsics_misses_by_outcome),
            (other.cam_to_gripper_misses, self.cam_to_gripper_misses),
            (other.cam_to_gripper_misses_by_outcome, self.cam_to_gripper_misses_by_outcome),
            (other.lang_no_annotation, self.lang_no_annotation),
            (other.lang_no_annotation_by_outcome, self.lang_no_annotation_by_outcome),
        ):
            for k, count in src.items():
                dst[k] = dst.get(k, 0) + count


def _episode_outcome(file_path: str) -> str:
    """Return 'success', 'failure', or 'unknown' from the NFS path segments."""
    for part in file_path.replace("\\", "/").split("/"):
        if part in ("success", "failure"):
            return part
    return "unknown"


def process_shard(
    shard_idx: int,
    shard_path: Path,
    dataset_dir: Path,
    write_videos: bool,
    video_dir: Path,
    fill_mask: bool,
    skip_existing: bool,
) -> ShardStats:
    """Worker: process one TFRecord shard. Returns a ShardStats with counters and buffered logs."""
    stats = ShardStats()

    raw_bytes_list = [r.numpy() for r in tf.data.TFRecordDataset([str(shard_path)])]

    # Skip shard if every episode already has masks. We don't have miss stats
    # for already-processed records (no longer stored in records), so the stats
    # written for a fully-resumed run will only reflect newly-processed shards.
    if skip_existing and all(record_has_masks(rb) for rb in raw_bytes_list):
        stats.logs.append(
            f"{INDENT}[shard {shard_idx:04d}] All {len(raw_bytes_list)} episodes already processed, skipping"
        )
        return stats

    builder = tfds.builder_from_directory(str(dataset_dir))
    # One Sim per robot, reused across every episode in the shard. Robotiq
    # also yields the affordance; UMI/Sharpa contribute their own gripper mask
    # under separate keys. Each Sim applies its own _CAM_OFFSET_6DOF via
    # compose_T_cam, so all masks share the same DROID-rig cam_to_gripper.
    sims = {robot: Sim(robot=robot) for robot in MASK_ROBOTS}
    lang_lookup = load_language_annotations()
    intrinsics_lookup = load_intrinsics()
    serials_lookup = load_wrist_serials()
    cam_to_gripper_lookup = load_cam_to_gripper()

    output_records: list[bytes] = []
    shard_modified = False

    for ep_local_idx, raw_bytes in enumerate(raw_bytes_list):
        episode = builder.info.features.deserialize_example(raw_bytes)
        steps = list(episode["steps"])
        nfs_path = episode["episode_metadata"]["file_path"].numpy().decode()

        try:
            lang_key = file_path_to_lang_key(nfs_path)
            lab = parse_lab(nfs_path)
            outcome = _episode_outcome(nfs_path)

            intrinsics, cam_to_gripper, lang_annotations, miss_flags, meta_logs = build_episode_metadata(
                nfs_path, lang_key, serials_lookup, intrinsics_lookup, cam_to_gripper_lookup, lang_lookup
            )
            for line in meta_logs:
                stats.logs.append(f"{INDENT}[shard {shard_idx:04d} ep {ep_local_idx:05d}] {line.lstrip()}")

            results = {
                robot: sims[robot].run_episode(
                    steps, intrinsics, cam_to_gripper, compute_affordances=(robot == "robotiq")
                )
                for robot in MASK_ROBOTS
            }
            frame_masks_by_key = {
                _mask_key(robot): compute_mask(res.masks, res.base_masks) for robot, res in results.items()
            }
            robotiq_res = results["robotiq"]

            if write_videos:
                video_path = video_dir / f"shard{shard_idx:04d}_ep{ep_local_idx:05d}.mp4"
                save_vis_video(
                    video_path,
                    steps,
                    frame_masks_by_key[_mask_key("robotiq")],
                    robotiq_res.affordance_pixels,
                    fill_mask,
                    umi_mask=frame_masks_by_key[_mask_key("umi")],
                    sharpa_mask=frame_masks_by_key[_mask_key("sharpa")],
                )
                stats.logs.append(f"{INDENT}Wrote {video_path}")

            modified_bytes = build_episode_record(
                raw_bytes,
                frame_masks_by_key=frame_masks_by_key,
                extrinsics_found=not miss_flags.cam_to_gripper,
                extra_obs={
                    "affordance_pos": robotiq_res.affordance_pos,
                    "affordance_rot": robotiq_res.affordance_rot,
                    "affordance_pixels": robotiq_res.affordance_pixels,
                },
                language_annotations=lang_annotations,
            )
            output_records.append(modified_bytes)
            shard_modified = True
            stats.n_processed += 1
            for hit, by_lab, by_outcome in (
                (miss_flags.serial, stats.serial_misses, stats.serial_misses_by_outcome),
                (miss_flags.intrinsics, stats.intrinsics_misses, stats.intrinsics_misses_by_outcome),
                (miss_flags.cam_to_gripper, stats.cam_to_gripper_misses, stats.cam_to_gripper_misses_by_outcome),
                (miss_flags.lang, stats.lang_no_annotation, stats.lang_no_annotation_by_outcome),
            ):
                if hit:
                    by_lab[lab] = by_lab.get(lab, 0) + 1
                    by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

        except Exception as e:
            stats.logs.append(f"{INDENT}[shard {shard_idx:04d} ep {ep_local_idx:05d}] {type(e).__name__}: {e}")
            stats.logs.append(traceback.format_exc())
            output_records.append(raw_bytes)  # keep original on error
            stats.n_failed += 1

    if shard_modified:
        tmp_path = shard_path.parent / f"{shard_path.name}.{shard_idx}.tmp"
        with tf.io.TFRecordWriter(str(tmp_path)) as writer:
            for rec in output_records:
                writer.write(rec)
        tmp_path.rename(shard_path)
        stats.logs.append(
            f"{INDENT}[shard {shard_idx:04d}] Updated {shard_path.name} "
            f"({stats.n_processed} processed, {stats.n_failed} failed)"
        )

    return stats


def _write_stats(data_dir: Path, stats: ShardStats) -> None:
    stats_path = Path("scratch/preprocess_stats.txt")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    total = stats.n_processed

    def _section(
        title: str,
        by_lab: dict[str, int],
        by_outcome: dict[str, int],
        miss_label: str = "Misses (used default)",
    ) -> list[str]:
        n_miss = sum(by_lab.values())
        out = [title, "=" * len(title)]
        if total:
            pct = n_miss / total * 100
            out.append(f"Total episodes processed : {total}")
            out.append(f"{miss_label:<24} : {n_miss}  ({pct:.1f}%)")
        else:
            out.append("Total episodes processed : 0")
            out.append(f"{miss_label:<24} : {n_miss}")
        out.append("Misses by lab:")
        if not by_lab:
            out.append("  (none)")
        else:
            for lab, count in sorted(by_lab.items(), key=lambda x: -x[1]):
                out.append(f"  {lab:<12} {count}")
        out.append("Misses by outcome:")
        if not by_outcome:
            out.append("  (none)")
        else:
            for outcome in ("success", "failure", "unknown"):
                count = by_outcome.get(outcome, 0)
                if count:
                    out.append(f"  {outcome:<12} {count}")
        out.append("")
        return out

    lines: list[str] = []
    lines += _section(
        "Wrist Serial Cache (droid_zed_serials.json)",
        stats.serial_misses,
        stats.serial_misses_by_outcome,
    )
    lines += _section(
        "Wrist Intrinsics Cache (droid_intrinsics_processed.json)",
        stats.intrinsics_misses,
        stats.intrinsics_misses_by_outcome,
    )
    lines += _section(
        "Cam-to-Gripper Cache (droid_cam_to_gripper_processed.json)",
        stats.cam_to_gripper_misses,
        stats.cam_to_gripper_misses_by_outcome,
    )
    lines += _section(
        "Language Annotation Cache (droid_lang_annotations_processed.json)",
        stats.lang_no_annotation,
        stats.lang_no_annotation_by_outcome,
        miss_label="Misses (kept original)",
    )
    lines.append(f"Data dir: {data_dir}")
    stats_path.write_text("\n".join(lines) + "\n")
    print(f"Stats written to {stats_path}")


class _InlineFuture:
    """A future whose work is deferred until inline_as_completed iterates over it."""

    def __init__(self, fn: callable, kwargs: dict):
        self._fn = fn
        self._kwargs = kwargs
        self._future: Future = Future()

    def run(self):
        try:
            self._future.set_result(self._fn(**self._kwargs))
        except Exception as e:
            self._future.set_exception(e)

    def result(self):
        return self._future.result()


class _InlineExecutor:
    """Drop-in for ProcessPoolExecutor that runs work in the calling process (for debugging)."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def submit(self, fn, **kwargs) -> _InlineFuture:
        return _InlineFuture(fn, kwargs)


def inline_as_completed(fs):
    """Like as_completed, but runs _InlineFuture work lazily as each is yielded."""
    for f in fs:
        if isinstance(f, _InlineFuture):
            f.run()
        yield f


@dataclasses.dataclass
class Args:
    data_dir: Path
    """Root DROID data directory."""
    video_dir: Path = Path("scratch/processed_droid_videos")
    """Directory for output visualization videos."""
    skip_existing: bool = False
    """Skip shards whose every episode is already in the *current* schema. Defaults
    to False so a re-run after a schema change reprocesses everything; opt in
    explicitly when resuming a partially completed run on the same schema."""
    fill_mask: bool = False
    """Fill masked regions with randomly shifted image content instead of solid green."""
    n_workers: int = 8
    """Number of parallel worker processes. Set to 0 for single-process debugging."""
    debug: bool = False
    """Process only a few shards (for debugging)."""


def main(args: Args) -> None:
    args.data_dir = args.data_dir.expanduser()
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    print("Finding TFDS dataset...")
    dataset_dir = find_dataset_dir(args.data_dir)
    video_dir = args.video_dir.expanduser()
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"{INDENT}Found: {dataset_dir}")
    print(f"{INDENT}Video dir: {video_dir}")

    builder = tfds.builder_from_directory(builder_dir=str(dataset_dir))
    shard_files = sorted(p for p in dataset_dir.glob("*.tfrecord*") if not p.name.endswith(".tmp"))

    n_debug_shards = 3
    if args.debug:
        shard_files = shard_files[:n_debug_shards]
        n_eps_total = sum(sum(1 for _ in tf.data.TFRecordDataset([str(sf)])) for sf in shard_files)
        print(f"WARN: debug mode — processing {n_debug_shards} shards ({n_eps_total} episodes)")

    print(f"{INDENT}{len(shard_files)} shards found")

    print("Validating dataset...")
    # Validate against the *existing* features.json — workers also use this schema
    # to deserialize, so it must still match what's on disk. We update features.json
    # only after processing completes (see end of main).
    ds = tf.data.TFRecordDataset([str(sf) for sf in shard_files[:n_debug_shards]])
    ds = ds.map(builder.info.features.deserialize_example)
    validate_dataset(ds, shard_files)

    print(f"Processing {len(shard_files)} shards with {args.n_workers} workers...")
    if args.n_workers > 0:
        mp_ctx = multiprocessing.get_context("spawn")
        executor = ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp_ctx)
        completer = as_completed
    else:
        executor = _InlineExecutor()
        completer = inline_as_completed

    totals = ShardStats()
    with executor:
        futures = {
            executor.submit(
                process_shard,
                shard_idx=shard_idx,
                shard_path=shard_path,
                dataset_dir=dataset_dir,
                write_videos=shard_idx < n_debug_shards,  # write_videos for the first few shards
                video_dir=video_dir,
                fill_mask=args.fill_mask,
                skip_existing=args.skip_existing,
            ): shard_idx
            for shard_idx, shard_path in enumerate(shard_files)
        }

        is_tty = sys.stdout.isatty()
        _log_print = tqdm.write if is_tty else print

        def log_print(msg: str) -> None:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            _log_print(f"[{ts}] {msg}")

        for future in tqdm(
            completer(futures),
            total=len(futures),
            desc="Shards",
            file=sys.stdout,
            mininterval=0.1 if is_tty else 60,
            miniters=1 if is_tty else 10,
            dynamic_ncols=is_tty,
        ):
            shard_idx = futures[future]
            try:
                shard_stats = future.result()
                totals.merge(shard_stats)
                for line in shard_stats.logs:
                    log_print(line)
            except Exception as e:
                log_print(f"Shard {shard_idx:04d} raised exception: {e}")
                log_print(traceback.format_exc())

    update_features_json(dataset_dir)
    write_readme(dataset_dir)
    print(f"Done. Processed {totals.n_processed} episodes ({totals.n_failed} failed). Videos saved to {video_dir}.")

    _write_stats(args.data_dir, totals)


if __name__ == "__main__":
    main(tyro.cli(Args))

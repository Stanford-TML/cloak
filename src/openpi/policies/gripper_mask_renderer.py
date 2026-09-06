"""Single-frame MuJoCo gripper mask renderer for real-time inference.

Extracts the mask generation logic from examples/droid/preprocess_data.py
into a reusable class that renders one frame at a time instead of full episodes.
"""

from collections.abc import Callable
import math
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.constants import lerp_gripper_qpos

# Scene XML (with floor) so debug RGB renders show the ground plane. The floor is
# non-colliding (contype/conaffinity=0) and is hidden from the segmentation pass
# below, so the mask is unaffected even if the gripper passes under it.
from openpi.constants import ROBOTIQ_SCENE_XML as ROBOTIQ_XML
from openpi.constants import SHARPA_SCENE_XML as SHARPA_XML
from openpi.constants import UMI_SCENE_XML as UMI_XML
from openpi.constants import YAM_LINEAR_SCENE_XML as YAM_XML

# Geom group reserved for a scene floor; disabled on the segmentation render so
# the floor can't occlude the gripper in the depth buffer (a low/under-floor
# reach would otherwise punch a hole in the mask). Group 5 is unused by the
# Franka + gripper geoms, which sit in 0-3. Collision proxies live in group 3
# (visual in group 2) and are hidden from the mask too. Mirrors
# examples/droid/preprocess_data.py.
_SEG_HIDDEN_GROUP = 5
_COLLISION_GROUP = 3

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


class GripperMaskRenderer:
    """MuJoCo-based single-frame gripper segmentation mask renderer."""

    def __init__(self, scene_xml: str | Path = ROBOTIQ_XML, render_h: int = 180, render_w: int = 320,
                 *, gripper_qpos_fn: Callable[..., np.ndarray] = lerp_gripper_qpos, arm_dof: int = 7) -> None:
        self._render_h = render_h
        self._render_w = render_w
        # Arm-DOF split for forward_qpos: 7 for Franka scenes (Robotiq/UMI/Sharpa),
        # 6 for YAM. The gripper qpos block starts at qpos[arm_dof].
        self._arm_dof = arm_dof
        # Maps the client's gripper command to the gripper-DOF qpos block (qpos[7:]).
        # Default = lerp_gripper_qpos (Robotiq scalar -> 8 coupled joints); the UMI
        # and Sharpa subclasses pass their own. Mirrors
        # preprocess_data.Sim._set_gripper_qpos so inference masks match the
        # training masks (pure FK at the given qpos, no physics settle).
        self._gripper_qpos_fn = gripper_qpos_fn
        self._model = self._build_model(str(scene_xml))
        self._data = mujoco.MjData(self._model)
        self._seg_renderer = mujoco.Renderer(self._model, height=render_h, width=render_w)
        self._seg_renderer.enable_segmentation_rendering()
        self._gripper_geom_ids, self._base_geom_ids = self._get_gripper_geom_ids()
        self._wrist_cam_id = self._model.cam("wrist_cam").id
        # Segmentation render option: hide collision proxies, FK-only site
        # markers, and (when the scene XML carries one) the floor — moved to a
        # dedicated group this option disables so it can't occlude the gripper.
        # See _SEG_HIDDEN_GROUP / _COLLISION_GROUP above; mirrors Sim's _seg_opt.
        self._seg_opt = mujoco.MjvOption()
        self._seg_opt.geomgroup[:] = 1
        self._seg_opt.geomgroup[_COLLISION_GROUP] = 0
        self._seg_opt.sitegroup[:] = 0
        try:
            self._model.geom_group[self._model.geom("floor").id] = _SEG_HIDDEN_GROUP
            self._seg_opt.geomgroup[_SEG_HIDDEN_GROUP] = 0
        except KeyError:
            pass

    def _build_model(self, scene_xml: str) -> mujoco.MjModel:
        spec = mujoco.MjSpec.from_file(scene_xml)
        cam = spec.worldbody.add_camera()
        cam.name = "wrist_cam"
        cam.fovy = 60.0  # placeholder; overwritten by set_intrinsics
        return spec.compile()

    def _get_gripper_geom_ids(self) -> tuple[np.ndarray, np.ndarray]:
        # Check if robotiq body names exist; if not, auto-detect from fr3_link7 descendants.
        has_robotiq = all(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name) != -1
            for name in _GRIPPER_BODY_NAMES
        )
        if has_robotiq:
            gripper_body_ids = {self._model.body(name).id for name in _GRIPPER_BODY_NAMES}
            base_body_ids = {self._model.body(name).id for name in _GRIPPER_BASE_NAMES}
        else:
            link7_id = self._model.body("fr3_link7").id
            gripper_body_ids: set[int] = set()
            queue = [i for i in range(self._model.nbody) if self._model.body_parentid[i] == link7_id]
            while queue:
                bid = queue.pop(0)
                gripper_body_ids.add(bid)
                for i in range(self._model.nbody):
                    if self._model.body_parentid[i] == bid:
                        queue.append(i)
            base_body_ids = set()
        all_geom_ids = np.array([i for i in range(self._model.ngeom) if self._model.geom_bodyid[i] in gripper_body_ids])
        base_geom_ids = np.array([i for i in range(self._model.ngeom) if self._model.geom_bodyid[i] in base_body_ids])
        return all_geom_ids, base_geom_ids

    def set_intrinsics(self, fy: float) -> None:
        """Set camera vertical FOV from focal length fy (in pixels at render resolution)."""
        self._model.cam_fovy[self._wrist_cam_id] = math.degrees(2.0 * math.atan(self._render_h / (2.0 * fy)))

    def _set_camera_extrinsic(self, extrinsic: np.ndarray) -> None:
        """Update wrist_cam pose from [x,y,z,rx,ry,rz] cam-to-base (CV convention)."""
        R_cv = Rotation.from_euler("xyz", extrinsic[3:]).as_matrix()
        flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        self._data.cam_xpos[self._wrist_cam_id] = extrinsic[:3]
        self._data.cam_xmat[self._wrist_cam_id] = (R_cv @ flip).flatten()

    def render_mask(
        self,
        joint_position: np.ndarray,
        gripper_position: float,
        camera_extrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate gripper masks for a single frame.

        Args:
            joint_position: 7-DOF arm joint angles.
            gripper_position: Gripper opening (0-1 scalar).
            camera_extrinsic: [x, y, z, rx, ry, rz] cam-to-base transform.

        Returns:
            (gripper_mask, base_mask) — each (H, W) bool arrays where True = gripper pixel.
        """
        qpos = self.forward_qpos(joint_position, gripper_position)
        return self.render_mask_at(qpos, camera_extrinsic)

    def forward_qpos(self, joint_position: np.ndarray, gripper_command) -> np.ndarray:
        """Reset, set arm + gripper qpos directly, run mj_forward, return the qpos.

        Pure FK — no physics settle. The client sends the robot's *current*
        proprioceptive state, so we already know the qpos; we only need FK to
        resolve body/site/geom poses for rendering. ``gripper_command`` is
        whatever ``gripper_qpos_fn`` consumes (a scalar opening in [0, 1] for
        Robotiq/UMI, or the full hand-joint vector for Sharpa); its output fills
        qpos[7:]. Matches the training-time mask geometry in
        preprocess_data.Sim.set_pose.
        """
        mujoco.mj_resetData(self._model, self._data)
        a = self._arm_dof
        self._data.qpos[:a] = np.asarray(joint_position, dtype=float).reshape(-1)[:a]
        gripper_qpos = np.asarray(self._gripper_qpos_fn(gripper_command), dtype=float).reshape(-1)
        n = self._model.nq - a
        self._data.qpos[a:a + n] = gripper_qpos[:n]
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        return self._data.qpos.copy()

    def render_mask_at(
        self, qpos: np.ndarray, camera_extrinsic: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render masks at a pre-computed qpos. Cheap path for interactive tuners."""
        self._data.qpos[:] = qpos
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

        self._set_camera_extrinsic(camera_extrinsic)
        self._seg_renderer.update_scene(self._data, camera="wrist_cam", scene_option=self._seg_opt)
        seg = self._seg_renderer.render()  # (H, W, 2): ch0=geom ID, ch1=geom type
        geom_ids = seg[:, :, 0]

        gripper_mask = np.isin(geom_ids, self._gripper_geom_ids)
        base_mask = np.isin(geom_ids, self._base_geom_ids)
        return gripper_mask, base_mask

    @staticmethod
    def compose_cam_to_base_from_ee(
        ee_xpos: np.ndarray, ee_xmat: np.ndarray, cam_to_gripper: np.ndarray
    ) -> np.ndarray:
        """Compose ``T_cam_to_base = T_ee @ T_cam_to_gripper`` -> 6-DOF [x,y,z,rx,ry,rz].

        Args:
            ee_xpos: (3,) base-frame position of the EE attachment site.
            ee_xmat: (9,) row-major rotation of the EE attachment site.
            cam_to_gripper: (6,) cam-relative-to-gripper [x,y,z,rx,ry,rz], scipy
                euler ``"xyz"`` radians.

        Returns the same euler convention used by ``_set_camera_extrinsic``.
        """
        R_ee = np.asarray(ee_xmat, dtype=np.float64).reshape(3, 3)
        R_cg = Rotation.from_euler("xyz", cam_to_gripper[3:]).as_matrix()
        R_cb = R_ee @ R_cg
        t_cb = R_ee @ cam_to_gripper[:3] + np.asarray(ee_xpos, dtype=np.float64)
        return np.concatenate([t_cb, Rotation.from_matrix(R_cb).as_euler("xyz")])

    def compose_cam_to_base(self, cam_to_gripper: np.ndarray) -> np.ndarray:
        """Compose cam-to-base from this renderer's *current* attachment_site pose.

        Reads the EE attachment site posed by the most recent ``forward_qpos`` /
        ``render_mask_at``, so call this after one of those. Used by the
        Inject*Mask transforms when the client sends cam-to-gripper, keeping the
        MuJoCo site query inside the renderer.
        """
        ee_site = self._data.site("attachment_site")
        return self.compose_cam_to_base_from_ee(ee_site.xpos, ee_site.xmat, cam_to_gripper)

    def compute_fingertip_positions(
        self,
        joint_position: np.ndarray,
        gripper_position: float,
    ) -> np.ndarray:
        """FK only: returns (2, 3) array of [left, right] fingertip xpos in robot base frame."""
        self.forward_qpos(joint_position, gripper_position)
        left = self._data.site("left_gripper_site").xpos.copy()
        right = self._data.site("right_gripper_site").xpos.copy()
        return np.stack([left, right])  # (2, 3)


# ---------------------------------------------------------------------------
# Lazy singleton for server reuse
# ---------------------------------------------------------------------------

_renderer_instance: GripperMaskRenderer | None = None


def get_renderer(
    scene_xml: str | Path = ROBOTIQ_XML,
    render_h: int = 180,
    render_w: int = 320,
) -> GripperMaskRenderer:
    """Return a shared GripperMaskRenderer instance (created on first call)."""
    global _renderer_instance
    if _renderer_instance is None:
        _renderer_instance = GripperMaskRenderer(scene_xml=scene_xml, render_h=render_h, render_w=render_w)
    return _renderer_instance


# ===========================================================================
# Sharpa hand mask renderer
# ===========================================================================
#
# Subclasses GripperMaskRenderer — only overrides body-name detection and passes
# an identity ``gripper_qpos_fn``, since the Sharpa client sends the full
# hand-joint vector (which is the gripper qpos block directly), not a scalar.

_HAND_BODY_NAMES = [
    # Mount + palm
    "mount_right_sharpa",
    "right_hand_C_MC",
    # Thumb
    "right_thumb_CMC_VL",
    "right_thumb_MC",
    "right_thumb_MCP_VL",
    "right_thumb_PP",
    "right_thumb_DP",
    # Index
    "right_index_MCP_VL",
    "right_index_PP",
    "right_index_MP",
    "right_index_DP",
    # Middle
    "right_middle_MCP_VL",
    "right_middle_PP",
    "right_middle_MP",
    "right_middle_DP",
    # Ring
    "right_ring_MCP_VL",
    "right_ring_PP",
    "right_ring_MP",
    "right_ring_DP",
    # Pinky
    "right_pinky_MC",
    "right_pinky_MCP_VL",
    "right_pinky_PP",
    "right_pinky_MP",
    "right_pinky_DP",
]

_BASE_BODY_NAMES = []


class SharpaMaskRenderer(GripperMaskRenderer):
    """GripperMaskRenderer variant for the Franka FR3 + Sharpa hand."""

    def __init__(self, scene_xml: str | Path = SHARPA_XML, render_h: int = 180, render_w: int = 320) -> None:
        # The Sharpa client sends the full hand-joint vector, so the gripper qpos
        # IS the command — identity mapping. forward_qpos slices it to the
        # model's hand-DOF count (handles the thumb+index 9-DOF scene too).
        super().__init__(
            scene_xml=scene_xml,
            render_h=render_h,
            render_w=render_w,
            gripper_qpos_fn=lambda hand: np.asarray(hand, dtype=float),
        )

    def _get_gripper_geom_ids(self) -> tuple[np.ndarray, np.ndarray]:
        hand_body_ids = {self._model.body(name).id for name in _HAND_BODY_NAMES}
        base_body_ids = {self._model.body(name).id for name in _BASE_BODY_NAMES}
        all_geom_ids = np.array([i for i in range(self._model.ngeom) if self._model.geom_bodyid[i] in hand_body_ids])
        base_geom_ids = np.array([i for i in range(self._model.ngeom) if self._model.geom_bodyid[i] in base_body_ids])
        return all_geom_ids, base_geom_ids


_sharpa_renderer_instance: SharpaMaskRenderer | None = None


def get_sharpa_renderer(
    scene_xml: str | Path = SHARPA_XML,
    render_h: int = 180,
    render_w: int = 320,
) -> SharpaMaskRenderer:
    """Return a shared SharpaMaskRenderer instance (created on first call)."""
    global _sharpa_renderer_instance
    if _sharpa_renderer_instance is None:
        _sharpa_renderer_instance = SharpaMaskRenderer(scene_xml=scene_xml, render_h=render_h, render_w=render_w)
    return _sharpa_renderer_instance


# ===========================================================================
# UMI parallel-jaw mask renderer
# ===========================================================================
#
# Subclasses GripperMaskRenderer — overrides body-name detection and passes a
# UMI ``gripper_qpos_fn``: the wire scalar (0=open, 1=closed) maps to the two
# prismatic finger qpos as ``0.04 * (1 - g)`` metres, written straight into
# qpos[7:9] by the base ``forward_qpos`` (Franka 7-DOF arm). The base name list
# is empty by design — the whole jaw is one mask, no separate dilated base.

_UMI_GRIPPER_BODY_NAMES = [
    "panda_hand",
    "left_finger",
    "right_finger",
]
_UMI_FINGER_RANGE = 0.04


def _umi_finger_qpos(gripper_position: float) -> np.ndarray:
    """UMI scalar opening [0, 1] -> the two prismatic finger qpos (metres)."""
    return np.full(2, _UMI_FINGER_RANGE * (1.0 - float(gripper_position)))


class UmiMaskRenderer(GripperMaskRenderer):
    """``GripperMaskRenderer`` variant for the Franka FR3 + UMI parallel jaw."""

    def __init__(self, scene_xml: str | Path = UMI_XML, render_h: int = 180, render_w: int = 320) -> None:
        super().__init__(
            scene_xml=scene_xml,
            render_h=render_h,
            render_w=render_w,
            gripper_qpos_fn=_umi_finger_qpos,
        )

    def _get_gripper_geom_ids(self) -> tuple[np.ndarray, np.ndarray]:
        gripper_body_ids = {self._model.body(name).id for name in _UMI_GRIPPER_BODY_NAMES}
        all_geom_ids = np.array([i for i in range(self._model.ngeom) if self._model.geom_bodyid[i] in gripper_body_ids])
        return all_geom_ids, np.array([], dtype=int)


_umi_renderer_instance: UmiMaskRenderer | None = None


def get_umi_renderer(
    scene_xml: str | Path = UMI_XML,
    render_h: int = 180,
    render_w: int = 320,
) -> UmiMaskRenderer:
    """Return a shared UmiMaskRenderer instance (created on first call)."""
    global _umi_renderer_instance
    if _umi_renderer_instance is None:
        _umi_renderer_instance = UmiMaskRenderer(scene_xml=scene_xml, render_h=render_h, render_w=render_w)
    return _umi_renderer_instance


# ===========================================================================
# YAM linear-gripper mask renderer
# ===========================================================================
#
# YAM is a 6-DOF arm (not a Franka), so ``forward_qpos`` is overridden to write
# the arm into qpos[:6] and the two slide fingers into qpos[6:8]. The wire scalar
# (0=open, 1=closed) maps to the slide range ``0.0475 * (1 - g)`` metres. Gripper
# bodies (gripper + two tips) mirror preprocess_data's _GRIPPER_BODY_NAMES_YAM;
# empty base like UMI.

_YAM_GRIPPER_BODY_NAMES = [
    "gripper",
    "tip_left",
    "tip_right",
]
_YAM_ARM_DOF = 6
_YAM_FINGER_RANGE = 0.0475


def _yam_finger_qpos(gripper_position: float) -> np.ndarray:
    """YAM scalar opening [0, 1] -> the two slide-finger qpos (metres)."""
    return np.full(2, _YAM_FINGER_RANGE * (1.0 - float(gripper_position)))


class YamMaskRenderer(GripperMaskRenderer):
    """``GripperMaskRenderer`` variant for the I2RT YAM 6-DOF arm + linear jaw."""

    def __init__(self, scene_xml: str | Path = YAM_XML, render_h: int = 180, render_w: int = 320) -> None:
        super().__init__(
            scene_xml=scene_xml,
            render_h=render_h,
            render_w=render_w,
            gripper_qpos_fn=_yam_finger_qpos,
            arm_dof=_YAM_ARM_DOF,  # YAM is a 6-DOF arm; slide fingers land in qpos[6:8]
        )

    def _get_gripper_geom_ids(self) -> tuple[np.ndarray, np.ndarray]:
        gripper_body_ids = {self._model.body(name).id for name in _YAM_GRIPPER_BODY_NAMES}
        all_geom_ids = np.array([i for i in range(self._model.ngeom) if self._model.geom_bodyid[i] in gripper_body_ids])
        return all_geom_ids, np.array([], dtype=int)


_yam_renderer_instance: YamMaskRenderer | None = None


def get_yam_renderer(
    scene_xml: str | Path = YAM_XML,
    render_h: int = 180,
    render_w: int = 320,
) -> YamMaskRenderer:
    """Return a shared YamMaskRenderer instance (created on first call)."""
    global _yam_renderer_instance
    if _yam_renderer_instance is None:
        _yam_renderer_instance = YamMaskRenderer(scene_xml=scene_xml, render_h=render_h, render_w=render_w)
    return _yam_renderer_instance

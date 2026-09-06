"""IK transform mapping (T, 8) DROID actions to (T, 29) Franka+Sharpa qpos.

Per step: Robotiq FK -> L/R gripper-site poses -> mink IK on the Sharpa scene
(thumb<-right, middle<-left, index/ring/pinky<-left + pinch-frame offset). The
config warm-starts across steps and infer() calls. ``fixed_hand_ik=True`` lerps
the hand between OPEN/CLOSED by the gripper scalar and solves only the arm.

Also runs the reverse direction (``solve_robotiq``): affordance points -> a
Robotiq (arm, gripper) state, used by ``SharpaToRobotiqRewrite``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading

import mink
from mink.lie.so3 import SO3
import mujoco
import numpy as np

from openpi import transforms as _transforms
from openpi.constants import (
    DROID_ARM_QPOS,
    GRIPPER_OPEN_SCALAR,
    SHARPA_ARM_QPOS,
    SHARPA_HAND_QPOS_OPEN,
    lerp_gripper_qpos,
    lerp_sharpa_hand_qpos,
)

# Use the robot XMLs (no floor) for FK/IK.
from openpi.constants import ROBOTIQ_FR3_XML as ROBOTIQ_XML
from openpi.constants import SHARPA_FR3_XML as SHARPA_XML

logger = logging.getLogger("openpi")

# IK parameters (match preprocess_ik.py).
_SOLVER = "daqp"
_POS_THRESHOLD = 1e-3
_ORI_THRESHOLD = 1e-2
_MAX_ITERS = 20
_DT = 0.005
_HAND_DOF = 22
# Secondary-scene arm DOF (7 = Franka; 6 = YAM). FK/reverse Robotiq scene is always 7.
_ARM_DOF = 7
_MOTION_FPS = 15  # DROID action-chunk fps

# Secondary-scene FrameTask costs (low finger orientation -> better pinch).
_POSITION_COST = 2.0
_ORIENTATION_COST = 0.05
# Wrist-yoke costs (target set per call in _set_targets / solve_robotiq).
_FORWARD_WRIST_ORIENTATION_COST = 0.01
_REVERSE_WRIST_ORIENTATION_COST = 0.01

# Extra-finger IK targets in the Robotiq pinch frame (Sharpa has these sites,
# UMI/YAM don't): pulled to left_gripper_site_pos + R_pinch @ offset.
_FINGER_OFFSETS: dict[str, np.ndarray] = {
    "index_fingertip": np.array([-0.02, 0.0, 0.0]),
    "ring_fingertip": np.array([0.02, 0.0, 0.0]),
    # "pinky_fingertip": np.array([0.04, 0.0, 0.0]),
}

def _site_pose(data: mujoco.MjData, site_name: str) -> np.ndarray:
    """Return site pose as (pos[3], quat_wxyz[4]) in the world/base frame."""
    pos = data.site(site_name).xpos.copy()
    rotmat = data.site(site_name).xmat.copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotmat)
    return np.concatenate([pos, quat])


class SharpaIKTransform(_transforms.DataTransformFn):
    """Maps (T, 8) DROID actions to (T, 29) Franka+Sharpa qpos.

    ``chunk_advance`` = the client's executed-actions-per-chunk (open_loop_horizon).
    The warm-start ``_last_q`` is recorded at index ``chunk_advance - 1`` so the next
    chunk seeds time-aligned with the robot's real pose (else the arm pulses at
    chunk boundaries).
    """

    def __init__(
        self,
        robotiq_xml: str | Path = ROBOTIQ_XML,
        sharpa_xml: str | Path = SHARPA_XML,
        chunk_advance: int | None = None,
        *,
        arm_dof: int = _ARM_DOF,
        hand_dof: int = _HAND_DOF,
        action_hand_dof: int | None = None,
        secondary_target_site_left: str = "middle_fingertip",
        secondary_target_site_right: str = "thumb_fingertip",
        secondary_limit_buffer: float = 0.1,
        grasp_z_offset: float = -0.0,
        gripper_open_clip_threshold: float = 0.2,
        motion_fps: int = _MOTION_FPS,
        fixed_hand_ik: bool = False,
        arm_seed: np.ndarray = SHARPA_ARM_QPOS,
        ee_seed: np.ndarray = SHARPA_HAND_QPOS_OPEN,
    ) -> None:
        # Render-on-EGL safe (no display required for FK/IK).
        if os.environ.get("DISPLAY") is None:
            os.environ.setdefault("MUJOCO_GL", "egl")

        # Secondary-scene DOFs (Sharpa: 22/22; UMI/YAM: 2 internal, 1 on the wire).
        self._arm_dof = int(arm_dof)
        self._hand_dof = int(hand_dof)
        self._action_hand_dof = int(action_hand_dof) if action_hand_dof is not None else int(hand_dof)
        self._secondary_target_site_left = secondary_target_site_left
        self._secondary_target_site_right = secondary_target_site_right
        # If True, lerp the hand by the gripper scalar and solve only the arm.
        self._fixed_hand_ik = bool(fixed_hand_ik)
        # World-frame Z added to L/R targets in _set_targets (negative = deeper grasp).
        self._grasp_z_offset = float(grasp_z_offset)
        # L/R separation -> Robotiq scalar values below this snap to 0 (open).
        self._gripper_open_clip_threshold = float(gripper_open_clip_threshold)

        # FK model (Franka + Robotiq 2F-85) for gripper-site poses.
        self._fk_xml_path = Path(robotiq_xml)
        self._fk_model = mujoco.MjModel.from_xml_path(str(robotiq_xml))
        self._disable_floor_contact(self._fk_model)
        self._fk_model.opt.timestep = _DT
        self._fk_data = mujoco.MjData(self._fk_model)

        # IK model (Franka + secondary hand/gripper); nq depends on the gripper.
        self._ik_xml_path = Path(sharpa_xml)
        self._ik_model = mujoco.MjModel.from_xml_path(str(sharpa_xml))
        self._disable_floor_contact(self._ik_model)
        self._ik_model.opt.timestep = _DT
        self._ik_data = mujoco.MjData(self._ik_model)

        # Extra-finger sites present on Sharpa (middle/index/ring/pinky) but not
        # UMI/YAM; missing ones are skipped. Pinch site (FK scene) frames the offsets.
        self._extra_finger_offsets: dict[str, np.ndarray] = {
            name: offset
            for name, offset in _FINGER_OFFSETS.items()
            if mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_SITE, name) != -1
        }
        self._has_extra_finger_sites = len(self._extra_finger_offsets) > 0
        self._has_pinch_site = (
            self._has_extra_finger_sites
            and mujoco.mj_name2id(self._fk_model, mujoco.mjtObj.mjOBJ_SITE, "pinch") != -1
        )

        self._configuration = mink.Configuration(self._ik_model)

        # Agreed reset pose (secondary frame); first-call _init_last_q builds
        # _last_q and the posture target from it. _home_q is the reference pose.
        self._arm_seed = np.asarray(arm_seed, dtype=np.float64).reshape(-1)[: self._arm_dof].copy()
        self._ee_seed = np.asarray(ee_seed, dtype=np.float64).reshape(-1)[: self._hand_dof].copy()
        self._home_q = self._secondary_home_qpos()

        self._tasks = self._build_tasks(self._ik_model)
        self._limits = [mink.ConfigurationLimit(model=self._ik_model, min_distance_from_limits=secondary_limit_buffer)]

        self._last_q: np.ndarray | None = None
        self._chunk_advance = chunk_advance
        self._motion_fps = int(motion_fps)

        # Reverse IK (Robotiq scene): solve_robotiq maps affordance points ->
        # (arm_qpos, gripper) Robotiq state.
        self._robotiq_configuration = mink.Configuration(self._fk_model)
        self._robotiq_tasks = self._build_robotiq_tasks(self._fk_model)
        self._robotiq_limits = [mink.ConfigurationLimit(model=self._fk_model, min_distance_from_limits=0.1)]
        self._robotiq_home_q = self._fk_model.key("home").qpos.copy()
        self._robotiq_last_q: np.ndarray | None = None

        # True: pin the reverse Robotiq wrist to its rest quat (safe fallback).
        # False: delta-track the opposite-side wrist. Flip manually.
        self._use_rest_wrist_pose: bool = True

        # Body-frame rotation delta between Robotiq and secondary attachment_site
        # frames (FK at reset). Also stores self._robotiq_rest_quat.
        self._rest_wrist_delta = self._compute_rest_wrist_delta()

        # Thumb-index distances at open/closed, for separation -> [0,1] gripper.
        self._gripper_distance_calibration = self._calibrate_gripper_distances()

        # mink Configuration is mutable shared state; singleton is shared.
        self._lock = threading.Lock()

        self._logged_first_call = False

    def action_to_qpos(self, actions: np.ndarray) -> np.ndarray:
        """Expand a (..., arm_dof + action_hand_dof) wire chunk to (..., arm_dof + hand_dof) qpos."""
        actions = np.asarray(actions)
        expected = self._arm_dof + self._action_hand_dof
        if actions.shape[-1] != expected:
            raise ValueError(f"action_to_qpos expects last dim {expected}, got shape {actions.shape}")
        if self._action_hand_dof == self._hand_dof:
            return actions
        flat = actions.reshape(-1, expected)
        out = np.zeros((flat.shape[0], self._arm_dof + self._hand_dof))
        out[:, : self._arm_dof] = flat[:, : self._arm_dof]
        for i, g in enumerate(flat[:, self._arm_dof]):
            out[i, self._arm_dof :] = self._hand_qpos_from_gripper(float(g))
        return out.reshape(actions.shape[:-1] + (self._arm_dof + self._hand_dof,))

    def _disable_floor_contact(self, model: mujoco.MjModel) -> None:
        """Disable contact for the floor geom if present (no-op on floorless XMLs)."""
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor_id == -1:
            return
        model.geom_contype[floor_id] = 0
        model.geom_conaffinity[floor_id] = 0

    def _secondary_home_qpos(self) -> np.ndarray:
        """Reference qpos for the secondary scene: ``home`` keyframe, else the seed."""
        if mujoco.mj_name2id(self._ik_model, mujoco.mjtObj.mjOBJ_KEY, "home") != -1:
            return self._ik_model.key("home").qpos.copy()
        q = np.zeros(self._ik_model.nq, dtype=np.float64)
        q[: self._arm_dof] = self._arm_seed
        q[self._arm_dof : self._arm_dof + self._hand_dof] = self._ee_seed
        return q

    def _build_tasks(self, model: mujoco.MjModel) -> dict:
        # More arm damping than fingers -> solver prefers moving fingers.
        damping_cost = np.where(np.arange(model.nv) < self._arm_dof, 1.0, 0.1)
        # Posture bias on the arm only (fingers free of the home key's curl).
        posture_cost = np.where(np.arange(model.nv) < self._arm_dof, 1e-2, 0.0)
        damping_task = mink.DampingTask(model=model, cost=damping_cost)
        posture_task = mink.PostureTask(model=model, cost=posture_cost)
        posture_task.set_target(self._home_q)
        tasks: dict = {
            self._secondary_target_site_right: mink.FrameTask(
                frame_name=self._secondary_target_site_right,
                frame_type="site",
                position_cost=_POSITION_COST,
                orientation_cost=_ORIENTATION_COST,
                lm_damping=1.0,
            ),
            self._secondary_target_site_left: mink.FrameTask(
                frame_name=self._secondary_target_site_left,
                frame_type="site",
                position_cost=_POSITION_COST,
                orientation_cost=_ORIENTATION_COST,
                lm_damping=1.0,
            ),
            # Orientation-only wrist yoke (target set per call in _set_targets).
            "attachment_site": mink.FrameTask(
                frame_name="attachment_site",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=_FORWARD_WRIST_ORIENTATION_COST,
                lm_damping=1.0,
            ),
            "damping": damping_task,
            "posture": posture_task,
        }
        # Extra-finger targets — present iff the secondary scene has the sites.
        for name in self._extra_finger_offsets:
            tasks[name] = mink.FrameTask(
                frame_name=name,
                frame_type="site",
                position_cost=_POSITION_COST,
                orientation_cost=_ORIENTATION_COST,
                lm_damping=1.0,
            )
        return tasks

    def _build_robotiq_tasks(self, model: mujoco.MjModel) -> dict:
        """mink tasks fitting the Robotiq L/R gripper sites to two 3D points.

        Position-only (affordance gives xyz, not orientation); roll redundancy is
        absorbed by the ``attachment_site`` wrist yoke (target set in solve_robotiq).
        """
        # Heavy damping on the gripper DOFs (locked to a lerp post-solve anyway).
        damping_cost = np.where(np.arange(model.nv) < 7, 1.0, 1e3)
        damping_task = mink.DampingTask(model=model, cost=damping_cost)
        posture_task = mink.PostureTask(model=model, cost=1e-2)
        posture_task.set_target(model.key("home").qpos)
        return {
            "left_gripper_site": mink.FrameTask(
                frame_name="left_gripper_site",
                frame_type="site",
                position_cost=_POSITION_COST,
                orientation_cost=0.0,
                lm_damping=1.0,
            ),
            "right_gripper_site": mink.FrameTask(
                frame_name="right_gripper_site",
                frame_type="site",
                position_cost=_POSITION_COST,
                orientation_cost=0.0,
                lm_damping=1.0,
            ),
            "damping": damping_task,
            "posture": posture_task,
            "attachment_site": mink.FrameTask(
                frame_name="attachment_site",
                frame_type="site",
                position_cost=0.0,
                orientation_cost=_REVERSE_WRIST_ORIENTATION_COST,
                lm_damping=1.0,
            ),
        }

    def _compute_rest_wrist_delta(self) -> SO3:
        """``inv(R_robotiq_rest) @ R_secondary_rest`` from FK at the reset poses.

        Captures the fixed mount-geometry offset between the two attachment_site
        frames. Side effect: stores ``self._robotiq_rest_quat`` /
        ``self._secondary_rest_quat``.
        """
        robotiq_scratch = mujoco.MjData(self._fk_model)
        robotiq_scratch.qpos[:] = 0
        robotiq_scratch.qpos[:7] = DROID_ARM_QPOS
        robotiq_scratch.qpos[7:7 + 8] = lerp_gripper_qpos(0.0)
        robotiq_scratch.qvel[:] = 0
        mujoco.mj_forward(self._fk_model, robotiq_scratch)
        robotiq_quat = _site_pose(robotiq_scratch, "attachment_site")[3:]

        secondary_scratch = mujoco.MjData(self._ik_model)
        secondary_scratch.qpos[:] = 0
        secondary_scratch.qpos[:self._arm_dof] = self._arm_seed
        secondary_scratch.qpos[self._arm_dof : self._arm_dof + self._hand_dof] = self._ee_seed
        secondary_scratch.qvel[:] = 0
        mujoco.mj_forward(self._ik_model, secondary_scratch)
        secondary_quat = _site_pose(secondary_scratch, "attachment_site")[3:]

        self._secondary_rest_quat = secondary_quat
        self._robotiq_rest_quat = robotiq_quat

        robotiq_so3 = SO3(wxyz=robotiq_quat)
        secondary_so3 = SO3(wxyz=secondary_quat)
        delta = robotiq_so3.inverse() @ secondary_so3

        # Sanity: delta composed back onto the Robotiq rest quat recovers secondary.
        residual = (robotiq_so3 @ delta).inverse() @ secondary_so3
        residual_log = np.asarray(SO3.identity().minus(residual))
        assert np.linalg.norm(residual_log) < 1e-6, (
            f"Rest-wrist delta failed roundtrip: |log(residual)|={np.linalg.norm(residual_log):.2e}"
        )
        return delta

    def _fk_gripper_sites(self, arm_qpos: np.ndarray, gripper: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (left_gripper_pose, right_gripper_pose) as (7,) [pos, quat_wxyz]."""
        gripper = float(np.clip(gripper, 0.0, 1.0))
        finger_qpos = lerp_gripper_qpos(gripper)

        self._fk_data.qpos[:7] = arm_qpos
        self._fk_data.qpos[7 : 7 + 8] = finger_qpos
        self._fk_data.qvel[:] = 0.0
        mujoco.mj_forward(self._fk_model, self._fk_data)

        left = _site_pose(self._fk_data, "left_gripper_site")
        right = _site_pose(self._fk_data, "right_gripper_site")
        return left, right

    def _read_pinch_quat(self) -> np.ndarray:
        """Pinch-site quat from the last FK on ``_fk_data`` (gate on _has_pinch_site)."""
        rotmat = self._fk_data.site("pinch").xmat.copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rotmat)
        return quat

    def _read_attachment_quat(self) -> np.ndarray:
        """Secondary ``attachment_site`` quat from the last FK on ``_ik_data``."""
        return _site_pose(self._ik_data, "attachment_site")[3:]

    def _set_targets(
        self,
        left_pose: np.ndarray,
        right_pose: np.ndarray,
        pinch_quat: np.ndarray | None = None,
    ) -> None:
        if self._has_extra_finger_sites:
            assert pinch_quat is not None, "Pinch quat is required when extra finger sites are present"

        world_offset = np.array([0.0, 0.0, self._grasp_z_offset])

        T_left = mink.SE3.from_rotation_and_translation(
            rotation=SO3(wxyz=left_pose[3:]),
            translation=left_pose[:3] + world_offset,
        )
        T_right = mink.SE3.from_rotation_and_translation(
            rotation=SO3(wxyz=right_pose[3:]),
            translation=right_pose[:3] + world_offset,
        )
        self._tasks[self._secondary_target_site_left].set_target(T_left)
        self._tasks[self._secondary_target_site_right].set_target(T_right)

        if self._use_rest_wrist_pose:
            target_rotation = SO3(wxyz=self._secondary_rest_quat)  # no delta composition, pins to rest pose
            self._tasks["attachment_site"].set_target(
                mink.SE3.from_rotation_and_translation(
                    rotation=target_rotation,
                    # Catch bugs easier. Should never have target position cost.
                    translation=np.ones(3),
                )
            )

        # Sharpa-only extra fingers: pos = left + R_pinch @ offset, ori copies left.
        if self._has_extra_finger_sites:
            R_local = SO3(wxyz=pinch_quat).as_matrix()
            R_target = SO3(wxyz=left_pose[3:])
            for name, offset in self._extra_finger_offsets.items():
                target_pos = left_pose[:3] + world_offset + R_local @ offset
                self._tasks[name].set_target(
                    mink.SE3.from_rotation_and_translation(
                        rotation=R_target,
                        translation=target_pos,
                    )
                )

    def _iterate_ik(
        self,
        configuration: mink.Configuration,
        tasks: dict,
        limits: list,
        max_iters: int,
        *,
        convergence_sites: list[str],
        lock_slice: slice | None = None,
        check_orientation: bool = False,
    ) -> None:
        """Run up to ``max_iters`` mink IK steps; early-exit on convergence.

        ``lock_slice`` zeroes those velocity DOFs each step so a pre-set qpos
        there survives ``integrate_inplace`` (pinned hand / locked gripper).
        """
        for _ in range(max_iters):
            vel = mink.solve_ik(
                configuration,
                tasks.values(),
                _DT,
                _SOLVER,
                damping=1e-3,
                limits=limits,
            )
            if lock_slice is not None:
                vel = np.asarray(vel).copy()
                vel[lock_slice] = 0.0
            configuration.integrate_inplace(vel, _DT)

            converged = True
            for site in convergence_sites:
                err = tasks[site].compute_error(configuration)
                if np.linalg.norm(err[:3]) > _POS_THRESHOLD:
                    converged = False
                    break
                if check_orientation and np.linalg.norm(err[3:]) > _ORI_THRESHOLD:
                    converged = False
                    break
            if converged:
                break

    def _apply_locked_hand(self, gripper: float) -> None:
        """Pin cfg.q + posture-target hand to lerp(gripper) (fixed_hand_ik only).

        Call after warm-start restore and before the IK loop.
        """
        hand = self._hand_qpos_from_gripper(gripper)
        q = np.asarray(self._configuration.q, dtype=np.float64).copy()
        q[self._arm_dof : self._arm_dof + self._hand_dof] = hand
        self._configuration.update(q)
        posture_target = self._tasks["posture"].target_q.copy()
        posture_target[self._arm_dof : self._arm_dof + self._hand_dof] = hand
        self._tasks["posture"].set_target(posture_target)

    def _wire_hand_action(self, q: np.ndarray, grip: float) -> np.ndarray:
        """Hand portion of the wire output (Sharpa: the IK-solved hand qpos)."""
        return q[self._arm_dof : self._arm_dof + self._hand_dof].copy()

    def _hand_qpos_from_gripper(self, gripper: float) -> np.ndarray:
        """Gripper scalar [0, 1] -> hand qpos, length ``_hand_dof`` (Sharpa: 22-DOF lerp)."""
        return lerp_sharpa_hand_qpos(float(gripper))

    def _init_last_q(self) -> None:
        """Seed ``_last_q`` and the posture target to the reset pose (first call)."""
        seed = self._home_q.copy()
        seed[:self._arm_dof] = self._arm_seed
        seed[self._arm_dof : self._arm_dof + self._hand_dof] = self._ee_seed
        self._last_q = seed
        self._tasks["posture"].set_target(seed)

    def reset(self) -> None:
        """Reset warm-start state (e.g., on a new episode)."""
        self._last_q = None

    def _calibrate_gripper_distances(self) -> tuple[float, float]:
        """``(d_open, d_closed)``: secondary L/R site separation at open/closed (m)."""
        scratch = mujoco.MjData(self._ik_model)
        scratch.qpos[:] = 0
        out = []
        for g in (0.0, 1.0):  # 0 = open, 1 = closed
            scratch.qpos[:self._arm_dof] = 0  # arm pose doesn't affect site-to-site distance
            scratch.qpos[self._arm_dof : self._arm_dof + self._hand_dof] = self._hand_qpos_from_gripper(g)
            mujoco.mj_forward(self._ik_model, scratch)
            left = scratch.site(self._secondary_target_site_left).xpos.copy()
            right = scratch.site(self._secondary_target_site_right).xpos.copy()
            out.append(float(np.linalg.norm(left - right)))
        return out[0], out[1]

    def virtual_gripper_from_hand(self, hand_qpos: np.ndarray) -> float:
        """Map observed Sharpa hand qpos to a synthetic Robotiq gripper_position.

        Runs Sharpa FK to get thumb_fingertip and index_fingertip xpos, computes
        their distance, and maps linearly between the calibrated open/closed
        Robotiq fingertip distances:
          ``distance == d_open  -> 0``
          ``distance == d_closed -> 1``
        Result is clipped to [0, 1].
        """
        hand_qpos = hand_qpos.squeeze()[: self._hand_dof]
        # Use the IK model's data scratch buffer; arm doesn't affect relative
        # finger geometry but we set it to 0 to match the calibration frame.
        self._ik_data.qpos[:] = 0
        self._ik_data.qpos[self._arm_dof : self._arm_dof + self._hand_dof] = hand_qpos
        mujoco.mj_forward(self._ik_model, self._ik_data)
        thumb = self._ik_data.site("thumb_fingertip").xpos.copy()
        index = self._ik_data.site("index_fingertip").xpos.copy()
        d = float(np.linalg.norm(thumb - index))
        return self._robotiq_gripper_value_from_separation(d)

    def _robotiq_gripper_value_from_separation(self, separation: float) -> float:
        """Site separation -> gripper [0, 1]; weak closes snap to open."""
        d_open, d_closed = self._gripper_distance_calibration
        if d_open == d_closed:
            return 0.0
        g = (d_open - separation) / (d_open - d_closed)
        g = float(np.clip(g, 0.0, 1.0))
        if g < self._gripper_open_clip_threshold:
            g = 0.0
        return g

    def compute_target_affordance(
        self, arm_qpos: np.ndarray, hand_qpos: np.ndarray
    ) -> np.ndarray:
        """Embodiment FK -> the two ``[left, right]`` IK-target xyz (2, 3), base frame.

        This (Sharpa) base impl maps the 22-DOF hand to ``[middle, thumb]``
        fingertips. UMI/YAM override it to read a scalar gripper and return their
        ``[left, right]`` gripper sites, so the shared ``SharpaToRobotiqRewrite``
        needs no per-embodiment ``__call__`` override.
        """
        arm_qpos = arm_qpos.squeeze()[: self._arm_dof]
        hand_qpos = hand_qpos.squeeze()[: self._hand_dof]
        with self._lock:
            self._ik_data.qpos[:] = 0
            self._ik_data.qpos[:self._arm_dof] = arm_qpos
            self._ik_data.qpos[self._arm_dof : self._arm_dof + self._hand_dof] = hand_qpos
            self._ik_data.qvel[:] = 0
            mujoco.mj_forward(self._ik_model, self._ik_data)
            middle_pos = self._ik_data.site("middle_fingertip").xpos.copy()
            thumb_pos = self._ik_data.site("thumb_fingertip").xpos.copy()
        return np.stack([middle_pos, thumb_pos], axis=0)

    def compute_g(
        self, target_left: np.ndarray, target_right: np.ndarray, hand: np.ndarray | None = None
    ) -> float:
        """Robotiq gripper [0, 1] for ``solve_robotiq``.

        This (Sharpa) base impl derives it from the affordance L/R separation.
        UMI/YAM override to return the client's true gripper scalar directly.
        """
        return self._robotiq_gripper_value_from_separation(
            float(np.linalg.norm(target_left - target_right))
        )

    def _init_robotiq_last_q(self) -> None:
        """Seed the reverse-IK warm-start + posture target to DROID rest (first call)."""
        seed = self._robotiq_home_q.copy()
        seed[:7] = DROID_ARM_QPOS
        seed[7 : 7 + 8] = lerp_gripper_qpos(GRIPPER_OPEN_SCALAR)
        self._robotiq_last_q = seed
        self._robotiq_tasks["posture"].set_target(seed)

    def solve_robotiq(
        self,
        target_left: np.ndarray,
        target_right: np.ndarray,
        *,
        attachment_quat: np.ndarray | None = None,
        g: float = 0.0,
    ) -> tuple[np.ndarray, float]:
        """Robotiq IK fitting the L/R gripper sites to two 3D points.

        Gripper opening ``g`` comes from ``compute_g`` (Sharpa: affordance
        separation; UMI/YAM: the true client gripper scalar). Fingers are locked
        to the lerp; the arm is solved by position-only mink IK, warm-started
        across calls. Returns ``(arm_qpos[7], gripper_value)``.
        """
        assert attachment_quat is not None, "attachment_quat is required to set the IK target orientation"

        target_left = target_left.squeeze()
        target_right = target_right.squeeze()
        finger_qpos = lerp_gripper_qpos(g)

        T_left = mink.SE3.from_rotation_and_translation(rotation=SO3.identity(), translation=target_left)
        T_right = mink.SE3.from_rotation_and_translation(rotation=SO3.identity(), translation=target_right)

        with self._lock:
            if self._robotiq_last_q is None:
                self._init_robotiq_last_q()

            init_q = self._robotiq_last_q.copy()
            # Lock the 8 gripper DOFs to this call's opening (g is recomputed
            # every call from the affordance separation); the IK velocity is
            # zeroed on these DOFs below so they hold while the arm solves.
            # Mirrors _apply_locked_hand on the forward path.
            init_q[7 : 7 + 8] = finger_qpos  # seed the fingers to the target opening
            self._robotiq_configuration.update(init_q)

            self._robotiq_tasks["left_gripper_site"].set_target(T_left)
            self._robotiq_tasks["right_gripper_site"].set_target(T_right)
            if self._use_rest_wrist_pose:
                target_rotation = SO3(wxyz=self._robotiq_rest_quat)  # no delta composition, pins to rest pose
                self._robotiq_tasks["attachment_site"].set_target(
                    mink.SE3.from_rotation_and_translation(
                        rotation=target_rotation,
                        # Catch bugs easier if robot reaches upward if wrist position cost
                        # is ever > 0, which it should never be.
                        translation=np.ones(3),
                    )
                )

            self._iterate_ik(
                self._robotiq_configuration,
                self._robotiq_tasks,
                self._robotiq_limits,
                _MAX_ITERS,
                convergence_sites=["left_gripper_site", "right_gripper_site"],
                lock_slice=slice(7, 7 + 8),  # lock the 8 Robotiq gripper joints
                check_orientation=False,
            )

            q = self._robotiq_configuration.q.copy()
            self._robotiq_last_q = q

        return q[:7].copy(), float(g)

    def reset_robotiq_warmstart(self) -> None:
        """Drop the Robotiq IK warm-start (e.g., on a new episode)."""
        self._robotiq_last_q = None

    def __call__(self, data: dict) -> dict:
        """Solve IK on the (T, 8) chunk -> (T, arm_dof + action_hand_dof) qpos chunk."""
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < 8:
            raise ValueError(
                f"SharpaIKTransform expects actions of shape (T, >=8), got {actions.shape}"
            )

        T = actions.shape[0]
        out = np.zeros((T, self._arm_dof + self._action_hand_dof), dtype=np.float64)

        # Solve only the first n_solve targets (the client executes chunk_advance
        # per chunk); warm-start at n_solve - 1 (the executed step).
        n_solve = self._chunk_advance if self._chunk_advance is not None else T
        n_solve = max(1, min(n_solve, T))
        warmstart_idx = n_solve - 1

        with self._lock:
            if self._last_q is None:
                self._init_last_q()
            self._configuration.update(self._last_q)

            for motion_t in range(n_solve):
                arm = actions[motion_t, :7]
                grip = float(actions[motion_t, 7])
                left_pose, right_pose = self._fk_gripper_sites(arm, grip)
                pinch_quat = self._read_pinch_quat() if self._has_pinch_site else None

                self._set_targets(left_pose, right_pose, pinch_quat=pinch_quat)
                if self._fixed_hand_ik:
                    self._apply_locked_hand(grip)

                self._iterate_ik(
                    self._configuration,
                    self._tasks,
                    self._limits,
                    _MAX_ITERS,
                    convergence_sites=[self._secondary_target_site_left, self._secondary_target_site_right],
                    lock_slice=slice(self._arm_dof, self._arm_dof + self._hand_dof) if self._fixed_hand_ik else None,
                    check_orientation=False,
                )

                q = self._configuration.q.copy()
                out[motion_t, :self._arm_dof] = q[:self._arm_dof]
                out[motion_t, self._arm_dof : self._arm_dof + self._action_hand_dof] = self._wire_hand_action(q, grip)

                if motion_t == warmstart_idx:
                    self._last_q = q.copy()

            # Tail (never executed by the client): replicate the last solved step.
            out[n_solve:] = out[n_solve - 1]

        if not self._logged_first_call:
            self._logged_first_call = True
            logger.info(
                "%s active — input actions (T=%d, 8) -> output (T=%d, %d). " "FK XML: %s, IK XML: %s, motion_fps=%d",
                type(self).__name__,
                T,
                T,
                self._arm_dof + self._action_hand_dof,
                self._fk_xml_path.name,
                self._ik_xml_path.name,
                self._motion_fps,
            )

        return {**data, "actions": out.astype(np.float32)}


class SharpaToRobotiqRewrite(_transforms.DataTransformFn):
    """Replace Sharpa proprio with the equivalent Franka+Robotiq state.

    Per obs: Sharpa FK -> affordance -> Robotiq IK; overwrite
    ``observation/{joint_position, gripper_position, affordance_pos}``.

    No-op if ``observation/joint_position`` or the hand key is missing (e.g.,
    training data or vanilla Robotiq clients).

    Embodiment-generic: the per-robot logic lives in the IK transform's
    ``compute_target_affordance`` / ``compute_g``, so UMI/YAM only set
    ``_SECONDARY_LABEL`` + ``hand_input_key`` and point ``_get_ik_transform`` at
    their singleton — no ``__call__`` override.
    """

    # Display label in the log line; subclasses override (UMI/YAM).
    _SECONDARY_LABEL: str = "Sharpa"

    def __init__(
        self,
        *,
        hand_input_key: str = "observation/hand_joint_position",
    ) -> None:
        self._logged_first_call = False
        self._hand_input_key = hand_input_key

    def _get_ik_transform(self) -> "SharpaIKTransform":
        """Matching IK singleton (subclasses override)."""
        return get_ik_transform()

    def __call__(self, data: dict) -> dict:
        arm = data.get("observation/joint_position", None)
        ee = data.get(self._hand_input_key, None)
        if arm is None or ee is None:
            return data  # no-op (training data / vanilla Robotiq client)

        ik = self._get_ik_transform()
        arm = np.asarray(arm, dtype=np.float64).reshape(-1)[: ik._arm_dof]
        ee = np.asarray(ee, dtype=np.float64).reshape(-1)[: ik._hand_dof]

        affordance = ik.compute_target_affordance(arm, ee)  # (2, 3)
        arm_rq, gripper = ik.solve_robotiq(
            affordance[0],
            affordance[1],
            attachment_quat=ik._read_attachment_quat(),
            g=ik.compute_g(affordance[0], affordance[1], ee),
        )

        if not self._logged_first_call:
            self._logged_first_call = True
            sep = float(np.linalg.norm(affordance[0] - affordance[1]))
            # |Δarm| is only meaningful when the secondary arm is also 7-DOF
            # (Sharpa/UMI); skip it for the 6-DOF YAM arm.
            extra = (
                f", |Δarm vs observed|={np.linalg.norm(arm - arm_rq):.3f}rad"
                if arm.shape[0] == arm_rq.shape[0]
                else ""
            )
            logger.info(
                "%s->Robotiq input rewrite active — affordance |L-R|=%.4fm -> gripper=%.3f%s",
                self._SECONDARY_LABEL,
                sep,
                gripper,
                extra,
            )

        return {
            **data,
            "observation/joint_position": arm_rq.astype(np.float32),
            "observation/gripper_position": np.array([gripper], dtype=np.float32),
        }


_input_rewrite_instance: SharpaToRobotiqRewrite | None = None


def get_input_rewrite_transform() -> SharpaToRobotiqRewrite:
    """Return a shared SharpaToRobotiqRewrite instance (created on first call)."""
    global _input_rewrite_instance
    if _input_rewrite_instance is None:
        _input_rewrite_instance = SharpaToRobotiqRewrite()
    return _input_rewrite_instance


_ik_transform_instance: SharpaIKTransform | None = None


def get_ik_transform(
    *,
    chunk_advance: int | None = None,
    robotiq_xml: str | Path | None = None,
    sharpa_xml: str | Path | None = None,
    fixed_hand_ik: bool = False,
) -> SharpaIKTransform:
    """Shared ``SharpaIKTransform`` singleton (args used on first call only)."""
    global _ik_transform_instance
    if _ik_transform_instance is None:
        _ik_transform_instance = SharpaIKTransform(
            robotiq_xml=robotiq_xml if robotiq_xml is not None else ROBOTIQ_XML,
            sharpa_xml=sharpa_xml if sharpa_xml is not None else SHARPA_XML,
            chunk_advance=chunk_advance,
            fixed_hand_ik=fixed_hand_ik,
        )
        logger.info("SharpaIKTransform singleton built (fixed_hand_ik=%s).", fixed_hand_ik)
    return _ik_transform_instance


def reset_ik_singleton() -> None:
    """Drop forward + reverse IK warm-start state (no-op if singleton not built)."""
    if _ik_transform_instance is not None:
        _ik_transform_instance.reset()
        _ik_transform_instance.reset_robotiq_warmstart()

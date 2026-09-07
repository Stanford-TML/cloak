"""IK transform for the I2RT YAM linear gripper.

Like ``UmiIKTransform`` (a parallel jaw on a scalar-gripper wire) but YAM is not
a Franka: 6-DOF arm, so ``arm_dof=6``. Input (T, 8) Robotiq actions -> output
(T, 7) YAM wire ([:6] arm, [6] gripper [0, 1]).

Only the per-embodiment hooks differ from the Sharpa base
(``compute_target_affordance`` / ``compute_g`` / finger mapping); the shared
``SharpaToRobotiqRewrite.__call__`` drives the reverse rewrite unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np

from openpi.constants import YAM_ARM_QPOS, YAM_GRIPPER_QPOS_OPEN
from openpi.constants import YAM_LINEAR_XML as YAM_XML
from openpi.policies.sharpa_ik_transform import (
    ROBOTIQ_XML,
    SharpaIKTransform,
    SharpaToRobotiqRewrite,
)

logger = logging.getLogger("openpi")

# Per-finger slide range [0, 0.0475] m; larger = more open (0 = closed wire scalar).
_YAM_FINGER_RANGE = 0.0475
_YAM_ARM_DOF = 6


def yam_finger_qpos_from_gripper(g: float) -> np.ndarray:
    """Scalar gripper [0, 1] -> YAM's two finger slides (m). 0=open, 1=closed."""
    g = float(np.clip(g, 0.0, 1.0))
    val = (1.0 - g) * _YAM_FINGER_RANGE
    return np.array([val, val], dtype=np.float64)


class YamIKTransform(SharpaIKTransform):
    """``SharpaIKTransform`` for the YAM scene (arm_dof=6, parallel jaw)."""

    def __init__(
        self,
        *,
        robotiq_xml: str | Path = ROBOTIQ_XML,
        yam_xml: str | Path = YAM_XML,
        chunk_advance: int | None = None,
    ) -> None:
        super().__init__(
            robotiq_xml=robotiq_xml,
            sharpa_xml=yam_xml,  # secondary scene = YAM
            chunk_advance=chunk_advance,
            arm_dof=_YAM_ARM_DOF,
            hand_dof=2,
            action_hand_dof=1,
            secondary_target_site_left="left_gripper_site",
            secondary_target_site_right="right_gripper_site",
            secondary_limit_buffer=1e-4,  # 0.0475 m slide range; Sharpa's 0.1 is infeasible
            grasp_z_offset=0.0,
            gripper_open_clip_threshold=0.0,
            arm_seed=YAM_ARM_QPOS,
            ee_seed=YAM_GRIPPER_QPOS_OPEN,
        )

    # No ``_iterate_ik`` override (cf. UMI): the solved finger qpos is discarded
    # on the wire and in the render, and the slide asymmetry mink introduces is
    # absorbed by the fingers, not the arm (deployed site error is unchanged).

    def _wire_hand_action(self, q: np.ndarray, grip: float) -> np.ndarray:  # type: ignore[override]
        return np.array([float(grip)])

    def _hand_qpos_from_gripper(self, gripper: float) -> np.ndarray:  # type: ignore[override]
        return yam_finger_qpos_from_gripper(float(gripper))

    def compute_target_affordance(  # type: ignore[override]
        self, arm_qpos: np.ndarray, hand: np.ndarray | float
    ) -> np.ndarray:
        """``[left, right]`` gripper-site xpos (2, 3) from YAM FK (scalar gripper)."""
        arm_qpos = np.asarray(arm_qpos).squeeze()[: self._arm_dof]
        gripper = float(np.asarray(hand).reshape(-1)[0])
        finger = yam_finger_qpos_from_gripper(gripper)
        with self._lock:
            self._ik_data.qpos[:] = 0
            self._ik_data.qpos[: self._arm_dof] = arm_qpos
            self._ik_data.qpos[self._arm_dof : self._arm_dof + self._hand_dof] = finger
            self._ik_data.qvel[:] = 0
            mujoco.mj_forward(self._ik_model, self._ik_data)
            left = self._ik_data.site("left_gripper_site").xpos.copy()
            right = self._ik_data.site("right_gripper_site").xpos.copy()
        return np.stack([left, right], axis=0).astype(np.float64)

    def compute_g(self, target_left, target_right, hand=None) -> float:  # type: ignore[override]
        """YAM sends a true gripper scalar; use it directly (else fall back to separation)."""
        if hand is None:
            return super().compute_g(target_left, target_right)
        return float(np.clip(np.asarray(hand).reshape(-1)[0], 0.0, 1.0))

    def virtual_gripper_from_hand(self, hand_qpos: np.ndarray) -> float:  # type: ignore[override]
        raise NotImplementedError(
            "YAM client sends observation/gripper_position scalar directly; "
            "no virtual-gripper inversion needed."
        )


class YamToRobotiqRewrite(SharpaToRobotiqRewrite):
    """Rewrite YAM proprio to the equivalent Franka+Robotiq state via reverse IK.

    Reuses the base ``__call__`` (reads ``observation/{joint_position,
    gripper_position}`` — YAM's 6-DOF arm + scalar gripper — computes the YAM
    affordance, solves Robotiq IK, and overwrites the proprio with the
    Robotiq-equivalent 7-DOF arm + gripper the policy was trained on).
    """

    _SECONDARY_LABEL = "YAM"

    def __init__(self) -> None:
        super().__init__(hand_input_key="observation/gripper_position")

    def _get_ik_transform(self) -> YamIKTransform:  # type: ignore[override]
        return get_ik_transform()


_ik_transform_instance: YamIKTransform | None = None


def get_ik_transform(
    *,
    chunk_advance: int | None = None,
    robotiq_xml: str | Path | None = None,
    yam_xml: str | Path | None = None,
) -> YamIKTransform:
    """Shared ``YamIKTransform`` singleton (args used on first call only)."""
    global _ik_transform_instance
    if _ik_transform_instance is None:
        _ik_transform_instance = YamIKTransform(
            robotiq_xml=robotiq_xml if robotiq_xml is not None else ROBOTIQ_XML,
            yam_xml=yam_xml if yam_xml is not None else YAM_XML,
            chunk_advance=chunk_advance,
        )
        logger.info("YamIKTransform singleton built.")
    return _ik_transform_instance


_input_rewrite_instance: YamToRobotiqRewrite | None = None


def get_input_rewrite_transform() -> YamToRobotiqRewrite:
    """Return a shared ``YamToRobotiqRewrite`` instance (created on first call)."""
    global _input_rewrite_instance
    if _input_rewrite_instance is None:
        _input_rewrite_instance = YamToRobotiqRewrite()
    return _input_rewrite_instance


def reset_ik_singleton() -> None:
    """Drop forward + reverse IK warm-start state (no-op if singleton not built)."""
    if _ik_transform_instance is not None:
        _ik_transform_instance.reset()
        _ik_transform_instance.reset_robotiq_warmstart()

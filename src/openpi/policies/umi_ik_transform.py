"""IK transform for the Franka FR3 + UMI parallel jaw.

``SharpaIKTransform`` subclass: hand_dof=2 (tendon-coupled slides), action_hand_dof=1
(scalar gripper wire, same (T, 8) shape as Robotiq), L/R gripper-site IK targets.

Only the per-embodiment hooks differ from the Sharpa base
(``compute_target_affordance`` / ``compute_g`` / finger mapping); the shared
``SharpaToRobotiqRewrite.__call__`` drives the reverse rewrite unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import numpy as np

from openpi.constants import UMI_ARM_QPOS, UMI_HAND_QPOS_OPEN
from openpi.constants import UMI_FR3_XML as UMI_XML
from openpi.policies.sharpa_ik_transform import (
    ROBOTIQ_XML,
    SharpaIKTransform,
    SharpaToRobotiqRewrite,
)

logger = logging.getLogger("openpi")

# Per-finger range [0, 0.04] m; qpos=0 open, qpos=0.04 closed (wire scalar 0=open, 1=closed).
_UMI_FINGER_RANGE = 0.04


def umi_finger_qpos_from_gripper(g: float) -> np.ndarray:
    """Scalar gripper [0, 1] -> UMI's two finger qpos (m). 0=open, 1=closed."""
    g = float(np.clip(g, 0.0, 1.0))
    val = (1.0 - g) * _UMI_FINGER_RANGE
    return np.array([val, val], dtype=np.float64)


def umi_gripper_from_finger_qpos(finger_qpos: float | np.ndarray) -> float:
    """Inverse of ``umi_finger_qpos_from_gripper`` (mean of a length-2 array)."""
    arr = np.asarray(finger_qpos, dtype=np.float64).reshape(-1)
    val = float(arr.mean()) if arr.size else 0.0
    g = 1.0 - float(np.clip(val, 0.0, _UMI_FINGER_RANGE)) / _UMI_FINGER_RANGE
    return float(np.clip(g, 0.0, 1.0))


class UmiIKTransform(SharpaIKTransform):
    """``SharpaIKTransform`` for the Franka+UMI scene (parallel jaw)."""

    def __init__(
        self,
        *,
        robotiq_xml: str | Path = ROBOTIQ_XML,
        umi_xml: str | Path = UMI_XML,
        chunk_advance: int | None = None,
    ) -> None:
        super().__init__(
            robotiq_xml=robotiq_xml,
            sharpa_xml=umi_xml,  # secondary scene = UMI
            chunk_advance=chunk_advance,
            hand_dof=2,
            action_hand_dof=1,
            secondary_target_site_left="left_gripper_site",
            secondary_target_site_right="right_gripper_site",
            secondary_limit_buffer=1e-4,  # 0.04 m finger range; Sharpa's 0.1 is infeasible
            grasp_z_offset=0.0,
            gripper_open_clip_threshold=0.0,
            arm_seed=UMI_ARM_QPOS,  # bakes in the +pi/4 mount-roll vs Robotiq
            ee_seed=UMI_HAND_QPOS_OPEN,
        )

    def _iterate_ik(self, configuration, tasks, limits, max_iters, **kwargs) -> None:  # type: ignore[override]
        super()._iterate_ik(configuration, tasks, limits, max_iters, **kwargs)
        # Enforce the finger tendon equality (q[7]==q[8]) that mink's
        # integrate_inplace bypasses. Secondary scene only; solve_robotiq's
        # config locks its own fingers.
        if configuration is self._configuration:
            q = np.asarray(configuration.q, dtype=np.float64).copy()
            m = 0.5 * (q[7] + q[8])
            q[7] = q[8] = m
            configuration.update(q)

    def _wire_hand_action(self, q: np.ndarray, grip: float) -> np.ndarray:  # type: ignore[override]
        return np.array([float(grip)], dtype=np.float64)

    def _hand_qpos_from_gripper(self, gripper: float) -> np.ndarray:  # type: ignore[override]
        return umi_finger_qpos_from_gripper(float(gripper))

    def compute_target_affordance(  # type: ignore[override]
        self, arm_qpos: np.ndarray, hand: np.ndarray | float
    ) -> np.ndarray:
        """``[left, right]`` gripper-site xpos (2, 3) from UMI FK (scalar gripper)."""
        arm_qpos = np.asarray(arm_qpos).squeeze()[:7]
        gripper = float(np.asarray(hand).reshape(-1)[0])
        finger = umi_finger_qpos_from_gripper(gripper)
        with self._lock:
            self._ik_data.qpos[:] = 0
            self._ik_data.qpos[:7] = arm_qpos
            self._ik_data.qpos[7:9] = finger
            self._ik_data.qvel[:] = 0
            mujoco.mj_forward(self._ik_model, self._ik_data)
            left = self._ik_data.site("left_gripper_site").xpos.copy()
            right = self._ik_data.site("right_gripper_site").xpos.copy()
        return np.stack([left, right], axis=0).astype(np.float64)  # (2, 3)

    def compute_g(self, target_left, target_right, hand=None) -> float:  # type: ignore[override]
        """UMI sends a true gripper scalar; use it directly (else fall back to separation)."""
        if hand is None:
            return super().compute_g(target_left, target_right)
        return float(np.clip(np.asarray(hand).reshape(-1)[0], 0.0, 1.0))

    def virtual_gripper_from_hand(self, hand_qpos: np.ndarray) -> float:  # type: ignore[override]
        raise NotImplementedError(
            "UMI client sends observation/gripper_position scalar directly; "
            "no virtual-gripper inversion needed."
        )


class UmiToRobotiqRewrite(SharpaToRobotiqRewrite):
    """Rewrite UMI proprio to the equivalent Franka+Robotiq state via reverse IK.

    Reuses the base ``__call__`` (reads ``observation/{joint_position,
    gripper_position}``, computes the UMI affordance, solves Robotiq IK, and
    overwrites the proprio), making the UMI pose in-distribution for the
    Robotiq-trained model.
    """

    _SECONDARY_LABEL = "UMI"

    def __init__(self) -> None:
        super().__init__(hand_input_key="observation/gripper_position")

    def _get_ik_transform(self) -> UmiIKTransform:  # type: ignore[override]
        return get_ik_transform()


_ik_transform_instance: UmiIKTransform | None = None


def get_ik_transform(
    *,
    chunk_advance: int | None = None,
    robotiq_xml: str | Path | None = None,
    umi_xml: str | Path | None = None,
) -> UmiIKTransform:
    """Shared ``UmiIKTransform`` singleton (args used on first call only)."""
    global _ik_transform_instance
    if _ik_transform_instance is None:
        _ik_transform_instance = UmiIKTransform(
            robotiq_xml=robotiq_xml if robotiq_xml is not None else ROBOTIQ_XML,
            umi_xml=umi_xml if umi_xml is not None else UMI_XML,
            chunk_advance=chunk_advance,
        )
        logger.info("UmiIKTransform singleton built.")
    return _ik_transform_instance


_input_rewrite_instance: UmiToRobotiqRewrite | None = None


def get_input_rewrite_transform() -> UmiToRobotiqRewrite:
    """Return a shared ``UmiToRobotiqRewrite`` instance (created on first call)."""
    global _input_rewrite_instance
    if _input_rewrite_instance is None:
        _input_rewrite_instance = UmiToRobotiqRewrite()
    return _input_rewrite_instance


def reset_ik_singleton() -> None:
    """Drop forward + reverse IK warm-start state (no-op if singleton not built)."""
    if _ik_transform_instance is not None:
        _ik_transform_instance.reset()
        _ik_transform_instance.reset_robotiq_warmstart()

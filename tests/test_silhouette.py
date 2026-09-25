"""Silhouette Calibration: pose parameterization, pseudo-GT mask, and optimizer self-consistency."""
import numpy as np
import pytest

from openpi.calibration import silhouette
from openpi.constants import (
    DEFAULT_CAM_TO_GRIPPER, DEFAULT_WRIST_INTRINSICS, DROID_ARM_QPOS, lerp_sharpa_hand_qpos,
    sharpa_gripper_from_hand_qpos,
)


def test_t_c2g_from_6vec_and_pose_distance():
    pose = np.array([0.01, -0.02, 0.03, 0.0, 0.0, 0.0])
    T = silhouette.Sim.t_c2g_from_6vec(pose)
    np.testing.assert_allclose(T[:3, :3], np.eye(3))
    np.testing.assert_allclose(T[:3, 3], pose[:3])
    assert silhouette.pose_distance(pose, pose) == (0.0, 0.0)
    t, r = silhouette.pose_distance(pose, np.zeros(6))
    assert np.isclose(t, np.linalg.norm(pose[:3])) and r == 0.0


@pytest.mark.parametrize("g", [0.0, 0.3, 1.0])
def test_sharpa_gripper_scalar_round_trip(g):
    assert np.isclose(sharpa_gripper_from_hand_qpos(lerp_sharpa_hand_qpos(g)), g)


def _synthetic_frames(rng, rect, color, n=8, shape=(60, 80)):
    """Random background that changes every frame plus one rigid `color` rectangle at `rect`."""
    frames = rng.integers(0, 256, (n, *shape, 3))
    (y0, y1), (x0, x1) = rect
    frames[:, y0:y1, x0:x1] = np.clip(color + rng.normal(0, 2, (n, y1 - y0, x1 - x0, 3)), 0, 255)
    return frames.astype(np.uint8)


# The rectangle fills ~45% of the ROI, so the 50% keep fraction selects it with little slack.
@pytest.mark.parametrize("crop, rect", [("fixed", ((36, 60), (44, 80))), ("rigid", ((4, 28), (4, 40)))])
def test_build_target_mask_finds_rigid_end_effector(crop, rect):
    rng = np.random.default_rng(0)
    color = np.array([30, 40, 50])
    frames = _synthetic_frames(rng, rect, color)
    gripper = np.zeros(len(frames))

    ee_color = silhouette.estimate_ee_color(gripper, frames, crop)
    np.testing.assert_allclose(ee_color, color, atol=5)

    opts = silhouette.CalibOptions(ee_color=tuple(ee_color), crop=crop)
    mask = silhouette.build_target_mask(gripper, frames, opts)
    truth = np.zeros(frames.shape[1:3], dtype=bool)
    (y0, y1), (x0, x1) = rect
    truth[y0:y1, x0:x1] = True
    assert (mask & truth).sum() / mask.sum() > 0.9  # precision
    assert (mask & truth).sum() / truth.sum() > 0.9  # recall


def test_build_target_needs_open_frames():
    frames = np.zeros((3, 60, 80, 3), dtype=np.uint8)
    assert silhouette.build_target(frames, np.zeros((3, 7)), np.ones(3)) is None


@pytest.fixture
def sim():
    try:
        sim = silhouette.Sim("robotiq")
    except Exception as e:  # no EGL/GPU on this machine
        pytest.skip(f"MuJoCo renderer unavailable: {e!r}")
    yield sim
    sim.renderer.close()


def test_optimize_pose_recovers_rendered_target(sim):
    fy = float(DEFAULT_WRIST_INTRINSICS[1])
    T_ee = sim.set_pose(DROID_ARM_QPOS, 0.0)
    mask = sim.render_gripper_mask(T_ee @ sim.t_c2g_from_6vec(DEFAULT_CAM_TO_GRIPPER), fy)
    assert mask.sum() > 0 and mask[: mask.shape[0] // 2].sum() == 0  # gripper sits in the bottom half

    target = silhouette.Target(mask, DROID_ARM_QPOS, 0.0, 0)
    init = DEFAULT_CAM_TO_GRIPPER + np.array([0.003, -0.002, 0.004, 0.02, -0.015, 0.01])
    pose, iou_init, iou_opt, _ = silhouette.optimize_pose(target, fy, sim, init)
    assert iou_init < 0.9
    assert iou_opt > 0.95 and iou_opt > iou_init
    t, r = silhouette.pose_distance(pose, DEFAULT_CAM_TO_GRIPPER)
    assert t < 0.005 and r < np.radians(2)

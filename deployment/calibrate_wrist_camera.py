"""Silhouette-calibrate the wrist camera on the real Franka + Robotiq robot.

Moves the arm to the home (DROID reset) qpos, opens the gripper, sweeps the arm
through a few poses around home grabbing one wrist frame (left lens) per pose,
then runs Silhouette Calibration (openpi.calibration.silhouette) on those
frames. Writes the wrist cam-to-gripper pose to
deployment/calibration/calibration_info_<embodiment>.json, and the captured
frames plus a mask overlay (pseudo-GT vs calibrated) to
scratch/wrist_calibration/<embodiment>/.

Run: bash scripts/calibrate_wrist_camera.sh
Re-fit a saved capture without the robot: add --from-capture scratch/wrist_calibration/robotiq/frames.npz
"""

import dataclasses
import json
import os
from pathlib import Path
import sys
import time

# Make the vendored `deployment/` package (repo root) and `src/` importable
# when run as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

# Off-screen MuJoCo rendering for the silhouette Sim (must be set before mujoco loads).
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image
import tyro

from deployment.calibration.calibration_utils import calib_info_path
from deployment.deploy_policy import _resize_no_pad
from deployment.deploy_policy import fetch_intrinsics
from deployment.misc.parameters import nuc_ip
from openpi.calibration import silhouette
from openpi.constants import DEFAULT_CAM_TO_GRIPPER, DROID_ARM_QPOS, RLDS_H, RLDS_W

# Per-embodiment outputs land in scratch/wrist_calibration/<embodiment>/.
OUTPUT_ROOT = REPO_ROOT / "scratch" / "wrist_calibration"

# Joint offsets (rad) from the home qpos (DROID_ARM_QPOS): a j1 (base yaw) x j7 (wrist
# roll) grid, which keeps the gripper at a constant height, so the background changes
# while the gripper stays fixed in the wrist image. One frame per pose.
SWEEP_OFFSETS = [np.array([a, 0, 0, 0, 0, 0, b]) for a in (-0.4, 0.0, 0.4) for b in (-0.6, 0.6)]
MIN_FRAMES = 5  # Silhouette Calibration needs at least this many gripper-open frames

# Visualization: each panel is the wrist frame upscaled 2x.
PANEL_H, PANEL_W = 2 * RLDS_H, 2 * RLDS_W
AXIS_LENGTH, AXIS_WIDTH = 0.05, 0.004  # camera-frame axes (m)


@dataclasses.dataclass
class Args:
    # Wrist ZED serial (the SN<serial> file under /usr/local/zed/settings/).
    wrist_camera_id: str
    # End effector being calibrated; selects calibration_info_<embodiment>.json.
    embodiment: str = "robotiq"
    # Skip the robot and re-fit a capture saved by a previous run.
    from_capture: Path | None = None


def capture(serial: str) -> dict:
    """Sweep the arm with the gripper open and record wrist frames + robot state."""
    from deployment.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper

    if nuc_ip is None:
        from franka.robot import FrankaRobot

        robot = FrankaRobot(use_gripper=True)
    else:
        from deployment.misc.server_interface import ServerInterface

        robot = ServerInterface(ip_address=nuc_ip, use_gripper=True, gripper_kind="robotiq_2f")
    camera = MultiCameraWrapper().camera_dict[serial]

    if input(f"Move to home, then sweep the arm through {len(SWEEP_OFFSETS)} poses? [y/N]: ").strip().lower() != "y":
        raise SystemExit("Aborted.")
    # Blocking joint moves are smooth interpolated moves (same call as RobotEnv.reset).
    print("Moving to the home qpos.")
    robot.update_joints(DROID_ARM_QPOS, velocity=False, blocking=True)
    robot.update_gripper(0, velocity=False, blocking=True)  # 0 = open

    frames, joint_positions, gripper_positions = [], [], []
    for offset in SWEEP_OFFSETS:
        robot.update_joints(DROID_ARM_QPOS + offset, velocity=False, blocking=True)
        time.sleep(0.5)
        state, _ = robot.get_robot_state()
        for _ in range(3):  # flush stale frames buffered during the move
            image = camera.read_camera()[0]["image"][f"{serial}_left"]
        rgb = image[..., :3][..., ::-1]  # BGRA -> RGB, then the same resize the deploy client uses
        frames.append(_resize_no_pad(rgb, RLDS_H, RLDS_W))
        joint_positions.append(state["joint_positions"])
        gripper_positions.append(state["gripper_position"])
    print("Returning to the home qpos.")
    robot.update_joints(DROID_ARM_QPOS, velocity=False, blocking=True)

    return {
        "frames": np.stack(frames),
        "joint_positions": np.array(joint_positions),
        "gripper_positions": np.array(gripper_positions),
        "fy": fetch_intrinsics(serial, "left")[1],
    }


def overlay(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.6) -> np.ndarray:
    """Alpha-blend `color` onto `frame` inside `mask`."""
    out = frame.copy()
    out[mask] = ((1 - alpha) * out[mask] + alpha * np.asarray(color)).astype(np.uint8)
    return out


def _add_camera_axes(scene: mujoco.MjvScene, T_cam: np.ndarray) -> None:
    """Draw the camera frame as x/y/z capsules (red/green/blue) into `scene`."""
    for axis, rgba in enumerate([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, AXIS_WIDTH,
                             T_cam[:3, 3], T_cam[:3, 3] + AXIS_LENGTH * T_cam[:3, axis])
        scene.ngeom += 1


def render_external(sim: silhouette.Sim, target: silhouette.Target, pose: np.ndarray) -> np.ndarray:
    """Third-person sim render of the gripper with the calibrated wrist-camera frame
    drawn as axes (x red, y green, z blue; OpenCV convention, z = optical axis)."""
    T_ee = sim.set_pose(target.joint_position, target.gripper_position)
    T_cam = T_ee @ sim.t_c2g_from_6vec(pose)

    # View from the camera's side of the gripper (-x in the EE frame), 45 deg around
    # toward +y and from above (the gripper points along +z), centered between the
    # gripper body and the camera so both are in frame.
    ex, ey, ez = T_ee[:3, 0], T_ee[:3, 1], T_ee[:3, 2]
    view_dir = -ex + ey - 0.5 * ez
    view_dir /= np.linalg.norm(view_dir)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = 0.5 * (T_ee[:3, 3] + 0.07 * ez) + 0.5 * T_cam[:3, 3]
    cam.distance = 0.3
    cam.azimuth = np.degrees(np.arctan2(-view_dir[1], -view_dir[0]))
    cam.elevation = np.degrees(np.arcsin(-view_dir[2]))

    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    with mujoco.Renderer(sim.model, height=PANEL_H, width=PANEL_W) as renderer:
        renderer.update_scene(sim.data, camera=cam, scene_option=opt)
        _add_camera_axes(renderer.scene, T_cam)
        rgb = renderer.render()
        # Empty background renders black (no skybox; Sim hides the floor): paint it
        # white wherever the depth pass hit nothing (the far clipping plane).
        renderer.enable_depth_rendering()
        renderer.update_scene(sim.data, camera=cam, scene_option=opt)
        _add_camera_axes(renderer.scene, T_cam)
        depth = renderer.render()
    rgb[depth >= 0.99 * sim.model.vis.map.zfar * sim.model.stat.extent] = 255
    return rgb


def main(args: Args) -> None:
    if args.embodiment != "robotiq":
        raise NotImplementedError(f"Silhouette Calibration only has a Robotiq renderer so far, got {args.embodiment!r}")
    output_dir = OUTPUT_ROOT / args.embodiment
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = output_dir / "frames.npz"
    overlay_path = output_dir / "mask_overlay.png"

    if args.from_capture is not None:
        data = dict(np.load(args.from_capture))
    else:
        data = capture(args.wrist_camera_id)
        np.savez_compressed(capture_path, **data)
        print(f"Saved capture to {capture_path}")

    fy = float(data["fy"])
    n_open = int((data["gripper_positions"] <= silhouette.OPEN_TOL).sum())
    assert n_open >= MIN_FRAMES, f"Only {n_open} gripper-open frames; need at least {MIN_FRAMES}."
    target = silhouette.build_target(data["frames"], data["joint_positions"], data["gripper_positions"])
    sim = silhouette.Sim()
    pose, iou_init, iou_opt, _ = silhouette.optimize_pose(target, fy, sim, DEFAULT_CAM_TO_GRIPPER)
    print(f"IoU {iou_init:.3f} (DROID default) -> {iou_opt:.3f} (calibrated)")
    print(f"cam_to_gripper: {np.round(pose, 4).tolist()}")

    # Pseudo-GT mask (white on black) | calibrated gripper mask (green) over the wrist
    # RGB frame | external sim view with the calibrated camera frame.
    frame = data["frames"][target.step]
    gt = Image.fromarray(target.mask.astype(np.uint8) * 255).convert("RGB").resize((PANEL_W, PANEL_H), Image.NEAREST)
    calibrated = silhouette.render_target_pose(target, pose, fy, sim)
    rgb = Image.fromarray(overlay(frame, calibrated)).resize((PANEL_W, PANEL_H), Image.BILINEAR)
    external = render_external(sim, target, pose)
    sim.renderer.close()  # free the EGL context now; at interpreter exit it errors noisily
    Image.fromarray(np.concatenate([np.asarray(gt), np.asarray(rgb), external], axis=1)).save(overlay_path)
    print(f"Mask overlay (pseudo-GT | calibrated on RGB | external view): {overlay_path}")

    calib_path = Path(calib_info_path(args.embodiment))
    calib = {f"{args.wrist_camera_id}_left": {"pose": pose.tolist(), "timestamp": time.time()}}
    calib_path.write_text(json.dumps(calib, indent=2) + "\n")
    print(f"Wrote {calib_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))

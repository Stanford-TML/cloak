"""Silhouette-calibrate the wrist camera on the real Franka (Robotiq or Sharpa hand).

Moves the arm to the embodiment's home qpos, opens the gripper/hand, sweeps the
arm through a few poses around home grabbing one wrist frame (left lens) per
pose, then runs Silhouette Calibration (openpi.calibration.silhouette) on those
frames. Writes the wrist cam-to-gripper pose to
deployment/calibration/calibration_info_<embodiment>.json, and the captured
frames plus a mask overlay (pseudo-GT vs calibrated) to
scratch/wrist_calibration/<embodiment>/.

Run: bash scripts/calibrate_wrist_camera.sh
Re-fit a saved capture without the robot: add --from-capture scratch/wrist_calibration/<embodiment>/frames_<embodiment>.npz
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

import cv2
import mujoco
import numpy as np
from PIL import Image
import tyro

from deployment.calibration.calibration_utils import calib_info_path
from deployment.deploy_policy import _resize_no_pad
from deployment.deploy_policy import fetch_intrinsics
from deployment.misc.parameters import nuc_ip
from openpi.calibration import silhouette
from openpi.constants import (
    DEFAULT_CAM_TO_GRIPPER, DROID_ARM_QPOS, RLDS_H, RLDS_W, SHARPA_ARM_QPOS, SHARPA_HAND_QPOS_OPEN,
    sharpa_gripper_from_hand_qpos,
)

# Per-embodiment outputs land in scratch/wrist_calibration/<embodiment>/.
OUTPUT_ROOT = REPO_ROOT / "scratch" / "wrist_calibration"

# Home arm qpos per embodiment: the pose its deploy client resets to.
HOME_QPOS = {"robotiq": DROID_ARM_QPOS, "sharpa": SHARPA_ARM_QPOS}



# Nelder-Mead start per embodiment; it must put the end effector in view, or the IoU
# is 0 everywhere nearby and the optimizer never moves. Sharpa: the mask-rendering
# extrinsic from hand_openpi (examples/droid/preprocess_data.py), i.e. the Robotiq-rig
# cam_to_gripper _SHARPA_FIXED_C2G composed with the camera-frame offset
# _CAM_OFFSET_6DOF["sharpa"] = [0.134, -0.022, 0.006, deg(2.0, 20.5, 19.5)].
INIT_CAM_TO_GRIPPER = {
    "robotiq": DEFAULT_CAM_TO_GRIPPER,
    "sharpa": np.array([-0.0973, -0.1003, 0.022, -0.3038, 0.4428, -1.3705]),
}

# Region the pseudo-GT is cropped to: the bottom-right of the wrist view, where a
# Robotiq sits, or (Sharpa, mounted off to the side) the box around the most rigid pixels.
CROP = {"robotiq": "fixed", "sharpa": "rigid"}

# Joint offsets (rad) from the home qpos: a j1 (base yaw) x j7 (wrist
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


def _connect_sharpa_hand():
    """Connect to and start the right Sharpa hand (same SDK calls as FrankaSharpaEnv)."""
    from deployment.franka_sharpa_env import _SHARPA_SDK_AVAILABLE, HAND_SPEC_TO_SIDE, _connect_hand, _init_hand

    if not _SHARPA_SDK_AVAILABLE:
        raise RuntimeError("Sharpa SDK not available — install SharpaWaveSDK_4.3.4 at the repo root.")
    hand, err = _connect_hand(HAND_SPEC_TO_SIDE["sharpa_R"])
    if hand is None or not _init_hand(hand):
        raise RuntimeError(f"Sharpa hand connection failed: {err}")
    hand.start()
    return hand


def capture(serial: str, embodiment: str) -> dict:
    """Sweep the arm with the gripper/hand open and record wrist frames + robot state.

    Sharpa's 22-D hand qpos is stored as the gripper scalar of the OPEN -> CLOSED
    lerp, so the same open-frame filter and sim rendering apply to both.
    """
    from deployment.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper

    use_gripper = embodiment == "robotiq"  # the Sharpa hand is driven here, not by the NUC
    if nuc_ip is None:
        from franka.robot import FrankaRobot

        robot = FrankaRobot(use_gripper=use_gripper)
    else:
        from deployment.misc.server_interface import ServerInterface

        robot = ServerInterface(ip_address=nuc_ip, use_gripper=use_gripper, gripper_kind="robotiq_2f")
    hand = _connect_sharpa_hand() if embodiment == "sharpa" else None
    camera = MultiCameraWrapper().camera_dict[serial]
    home = HOME_QPOS[embodiment]

    if input(f"Move to home, then sweep the arm through {len(SWEEP_OFFSETS)} poses? [y/N]: ").strip().lower() != "y":
        raise SystemExit("Aborted.")
    # Blocking joint moves are smooth interpolated moves (same call as RobotEnv.reset).
    print("Moving to the home qpos.")
    robot.update_joints(home, velocity=False, blocking=True)
    if hand is None:
        robot.update_gripper(0, velocity=False, blocking=True)  # 0 = open
    else:
        hand.set_joint_position(SHARPA_HAND_QPOS_OPEN.tolist(), True)
        time.sleep(1.0)

    frames, joint_positions, gripper_positions = [], [], []
    for offset in SWEEP_OFFSETS:
        robot.update_joints(home + offset, velocity=False, blocking=True)
        time.sleep(0.5)
        state, _ = robot.get_robot_state()
        for _ in range(3):  # flush stale frames buffered during the move
            image = camera.read_camera()[0]["image"][f"{serial}_left"]
        rgb = image[..., :3][..., ::-1]  # BGRA -> RGB, then the same resize the deploy client uses
        frames.append(_resize_no_pad(rgb, RLDS_H, RLDS_W))
        joint_positions.append(state["joint_positions"])
        if hand is None:
            gripper_positions.append(state["gripper_position"])
        else:
            _, hand_qpos = hand.get_joint_position_rad()
            gripper_positions.append(sharpa_gripper_from_hand_qpos(np.asarray(hand_qpos)[:22]))
    print("Returning to the home qpos.")
    robot.update_joints(home, velocity=False, blocking=True)
    if hand is not None:
        hand.stop()

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


def label(img: np.ndarray, lines: list[str]) -> np.ndarray:
    """Write `lines` in the top-left corner of `img` (white with a black outline)."""
    out = np.array(img)  # writable copy (arrays from PIL images are read-only)
    for i, line in enumerate(lines):
        org = (8, 22 + 20 * i)
        cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _add_camera_axes(scene: mujoco.MjvScene, T_cam: np.ndarray, alpha: float = 1.0, width: float = AXIS_WIDTH) -> None:
    """Draw the camera frame as x/y/z capsules (red/green/blue) into `scene`."""
    for axis, rgb in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.asarray((*rgb, alpha), dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             T_cam[:3, 3], T_cam[:3, 3] + AXIS_LENGTH * T_cam[:3, axis])
        scene.ngeom += 1


def render_external(sim: silhouette.Sim, target: silhouette.Target, pose: np.ndarray, init: np.ndarray) -> np.ndarray:
    """Third-person sim render of the end effector with the calibrated wrist-camera frame
    drawn as axes (x red, y green, z blue; OpenCV convention, z = optical axis) and the
    optimizer's initial camera frame as thin, faded axes."""
    T_ee = sim.set_pose(target.joint_position, target.gripper_position)
    T_cam = T_ee @ sim.t_c2g_from_6vec(pose)
    T_init = T_ee @ sim.t_c2g_from_6vec(init)

    def draw_axes(scene: mujoco.MjvScene) -> None:
        _add_camera_axes(scene, T_init, alpha=0.5, width=0.6 * AXIS_WIDTH)
        _add_camera_axes(scene, T_cam)

    # View from the camera's side of the gripper (-x in the EE frame), 45 deg around
    # toward +y and from above (the gripper points along +z), centered on the gripper
    # body and both camera frames, far enough back to fit all three.
    ex, ey, ez = T_ee[:3, 0], T_ee[:3, 1], T_ee[:3, 2]
    view_dir = -ex + ey - 0.5 * ez
    view_dir /= np.linalg.norm(view_dir)
    points = np.stack([T_ee[:3, 3] + 0.07 * ez, T_cam[:3, 3], T_init[:3, 3]])
    center = points.mean(axis=0)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = 0.25 + np.linalg.norm(points - center, axis=1).max()
    cam.azimuth = np.degrees(np.arctan2(-view_dir[1], -view_dir[0]))
    cam.elevation = np.degrees(np.arcsin(-view_dir[2]))

    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    with mujoco.Renderer(sim.model, height=PANEL_H, width=PANEL_W) as renderer:
        renderer.update_scene(sim.data, camera=cam, scene_option=opt)
        draw_axes(renderer.scene)
        rgb = renderer.render()
        # Empty background renders black (no skybox; Sim hides the floor): paint it
        # white wherever the depth pass hit nothing (the far clipping plane).
        renderer.enable_depth_rendering()
        renderer.update_scene(sim.data, camera=cam, scene_option=opt)
        draw_axes(renderer.scene)
        depth = renderer.render()
    rgb[depth >= 0.99 * sim.model.vis.map.zfar * sim.model.stat.extent] = 255
    return rgb


def main(args: Args) -> None:
    output_dir = OUTPUT_ROOT / args.embodiment
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = output_dir / f"frames_{args.embodiment}.npz"
    overlay_path = output_dir / f"mask_overlay_{args.embodiment}.png"

    if args.from_capture is not None:
        data = dict(np.load(args.from_capture))
    else:
        assert args.embodiment in HOME_QPOS, f"Live capture supports {list(HOME_QPOS)}, got {args.embodiment!r}"
        data = capture(args.wrist_camera_id, args.embodiment)
        np.savez_compressed(capture_path, **data)
        print(f"Saved capture to {capture_path}")

    fy = float(data["fy"])
    n_open = int((data["gripper_positions"] <= silhouette.OPEN_TOL).sum())
    assert n_open >= MIN_FRAMES, f"Only {n_open} gripper-open frames; need at least {MIN_FRAMES}."
    # Pseudo-GT intensity cue: pixels closest to the end effector's color, measured from
    # the frames (the model's display colors can be far off under real lighting).
    crop = CROP[args.embodiment]
    ee_color = silhouette.estimate_ee_color(data["gripper_positions"], data["frames"], crop)
    print(f"End-effector color (estimated): RGB {np.round(ee_color).astype(int).tolist()}")
    opts = silhouette.CalibOptions(ee_color=tuple(ee_color), crop=crop)
    target = silhouette.build_target(data["frames"], data["joint_positions"], data["gripper_positions"], opts)
    sim = silhouette.Sim(args.embodiment)
    init = INIT_CAM_TO_GRIPPER[args.embodiment]
    pose, iou_init, iou_opt, _ = silhouette.optimize_pose(target, fy, sim, init)
    print(f"IoU {iou_init:.3f} (init) -> {iou_opt:.3f} (calibrated)")
    print(f"cam_to_gripper: {np.round(pose, 4).tolist()}")

    # Pseudo-GT mask (white on black) | calibrated gripper mask (green) over the wrist
    # RGB frame | external sim view with the calibrated (and faded initial) camera frame.
    frame = data["frames"][target.step]
    gt = Image.fromarray(target.mask.astype(np.uint8) * 255).convert("RGB").resize((PANEL_W, PANEL_H), Image.NEAREST)
    calibrated = silhouette.render_target_pose(target, pose, fy, sim)
    rgb = Image.fromarray(overlay(frame, calibrated)).resize((PANEL_W, PANEL_H), Image.BILINEAR)
    external = render_external(sim, target, pose, init)
    sim.renderer.close()  # free the EGL context now; at interpreter exit it errors noisily
    panels = [
        label(np.asarray(gt), ["Pseudo-GT mask"]),
        label(np.asarray(rgb), [f"Calibrated mask (green), IoU {iou_opt:.2f}"]),
        label(external, ["Camera frames (x red, y green, z blue)", "thick: calibrated", "thin, faded: initial"]),
    ]
    Image.fromarray(np.concatenate(panels, axis=1)).save(overlay_path)
    print(f"Mask overlay (pseudo-GT | calibrated on RGB | external view): {overlay_path}")

    calib_path = Path(calib_info_path(args.embodiment))
    calib = {f"{args.wrist_camera_id}_left": {"pose": pose.tolist(), "timestamp": time.time()}}
    calib_path.write_text(json.dumps(calib, indent=2) + "\n")
    print(f"Wrote {calib_path}")


if __name__ == "__main__":
    main(tyro.cli(Args))

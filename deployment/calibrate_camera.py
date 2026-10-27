# ruff: noqa: E402
"""Checkerboard calibration followed by required manual refinement."""

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np

from deployment.calibration.calibration_utils import EMBODIMENTS, save_calibration_info
from deployment.calibration.hand_eye import calibration_offset, detect_board, pose_vector, solve_hand_eye, transform
from deployment.calibration.manual_tuning import tune


def capture(args):
    # Confirm before controller initialization, which itself can start hardware.
    if not os.environ.get("CLOAK_NUC_IP"):
        raise ValueError("Set CLOAK_NUC_IP before connecting to the robot")
    import tkinter as tk

    probe = tk.Tk()
    probe.withdraw()
    probe.destroy()
    if input("Start controllers and move through the DROID calibration trajectory? [y/N]: ").lower() != "y":
        raise SystemExit("Cancelled; no hardware connected.")
    from deployment.camera_utils.camera_readers.zed_camera import gather_zed_cameras
    from deployment.misc.server_interface import ServerInterface
    from openpi.calibration.silhouette import Sim

    # Open only the requested camera; do not require existing calibration.
    cameras = {camera.serial_number: camera for camera in gather_zed_cameras()}
    if args.wrist_camera_id not in cameras:
        raise ValueError(f"Camera {args.wrist_camera_id} not found; detected {list(cameras)}")
    camera = cameras[args.wrist_camera_id]
    camera.set_reading_parameters()
    camera.set_trajectory_mode()
    sim, hand = None, None
    try:
        sim = Sim(args.embodiment)
        robot = ServerInterface(
            os.environ["CLOAK_NUC_IP"],
            use_gripper=args.embodiment != "sharpa",
            gripper_kind="franka_hand" if args.embodiment == "umi" else "robotiq_2f",
        )
        if args.embodiment == "sharpa":
            from deployment.franka_sharpa_env import _connect_hand, _init_hand, HAND_SPEC_TO_SIDE
            from openpi.constants import SHARPA_HAND_QPOS_OPEN

            hand, error = _connect_hand(HAND_SPEC_TO_SIDE["sharpa_R"])
            if hand is None or not _init_hand(hand):
                raise RuntimeError(f"Sharpa initialization failed: {error}")
            hand.start()
            hand.set_joint_position(SHARPA_HAND_QPOS_OPEN.tolist(), True)
        else:
            robot.update_gripper(0, velocity=False, blocking=True)
        key = f"{args.wrist_camera_id}_{args.lens}"
        intrinsics = camera.get_intrinsics()[key]["cameraMatrix"]

        def image():
            reading = None
            for _ in range(3):
                reading = camera.read_camera()
                if reading is None:
                    raise RuntimeError("Camera frame acquisition failed")
            return reading[0]["image"][key][..., :3][..., ::-1].copy()

        first = image()
        preview = ROOT / "scratch" / "wrist_calibration" / args.embodiment / "board_preview.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview), cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        if detect_board(first, intrinsics) is None:
            raise ValueError(f"Board not visible. Inspect {preview}; position the fixed board and retry.")
        print(f"Board detected. Preview: {preview}")
        if input("Keep board fixed. Begin arm motion from the CURRENT pose? [y/N]: ").lower() != "y":
            raise SystemExit("Cancelled before arm trajectory.")
        state, _ = robot.get_robot_state()
        origin = transform(state["cartesian_position"])
        frames, joints, grippers, ee_poses = [], [], [], []
        # Match DROID's small nonblocking Cartesian steps. Its bridge converts
        # Cartesian targets to bounded joint increments, so sparse blocking
        # waypoints would not actually trace the requested path.
        for i, t in enumerate(np.arange(0, 2 * np.pi + 0.01, 0.01)):
            desired = pose_vector(origin @ transform(calibration_offset(t)))
            command = np.r_[desired, 0.0]
            robot.update_command(
                command, action_space="cartesian_position", gripper_action_space="position", blocking=False
            )
            time.sleep(1 / 15)
            if i % 10:
                continue
            time.sleep(0.5)
            frame = image()
            state, _ = robot.get_robot_state()
            joint = np.asarray(state["joint_positions"])
            grip = float(state["gripper_position"]) if args.embodiment != "sharpa" else 0.0
            frames.append(frame)
            joints.append(joint)
            grippers.append(grip)
            # Fit in exactly the attachment frame used downstream, not a
            # potentially different Desk-configured Cartesian tool frame.
            ee_poses.append(sim.set_pose(joint, grip).copy())
            print(f"Captured {len(frames)} board views")
        return dict(
            frames=np.asarray(frames),
            joint_positions=np.asarray(joints),
            gripper_positions=np.asarray(grippers),
            ee_to_base=np.asarray(ee_poses),
            intrinsics=intrinsics,
            camera_id=args.wrist_camera_id,
            lens=args.lens,
            embodiment=args.embodiment,
        )
    finally:
        camera.disable_camera()
        if hand is not None:
            hand.stop()
        if sim is not None:
            sim.renderer.close()


def main(args):
    capture_path = (
        ROOT / "scratch" / "wrist_calibration" / args.embodiment / f"capture_{args.wrist_camera_id}_{args.lens}.npz"
    )
    if args.resume:
        with np.load(capture_path, allow_pickle=False) as data:
            record = dict(data)
    else:
        record = capture(args)
        capture_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(capture_path, **record)
    for field, expected in (("embodiment", args.embodiment), ("camera_id", args.wrist_camera_id), ("lens", args.lens)):
        if str(record[field]) != expected:
            raise ValueError(f"Capture {field} does not match requested rig")
    ee, board, errors = [], [], []
    for frame, pose in zip(record["frames"], record["ee_to_base"], strict=True):
        detected = detect_board(frame, record["intrinsics"])
        if detected is not None and detected[1] <= 3:
            ee.append(pose)
            board.append(detected[0])
            errors.append(detected[1])
    initial, quality = solve_hand_eye(ee, board)
    quality["max_reprojection_error_px"] = max(errors)
    print(f"Automatic calibration: {quality}")
    final = tune(
        initial,
        record["frames"],
        record["joint_positions"],
        record["gripper_positions"],
        record["intrinsics"],
        args.embodiment,
    )
    if final is None:
        print("Cancelled. Existing accepted calibration was not changed.")
        return
    path = save_calibration_info(
        args.embodiment, args.wrist_camera_id, final, initial_pose=initial, lens=args.lens, quality=quality
    )
    print(f"Saved {path}. Deployment loads this file automatically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embodiment", choices=EMBODIMENTS, required=True)
    parser.add_argument("--wrist-camera-id", required=True)
    parser.add_argument("--lens", choices=("left", "right"), default="left")
    parser.add_argument(
        "--resume", action="store_true", help="Reuse this rig's saved capture without connecting to hardware"
    )
    main(parser.parse_args())

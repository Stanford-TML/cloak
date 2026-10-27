"""DROID ChArUco target and hand-eye calibration in the public renderer's EE frame."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def transform(pose):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def pose_vector(matrix):
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")]


def calibration_offset(t):
    """Original DROID hand-camera trajectory, in the starting tool frame."""
    return np.array(
        [
            0.05 * np.sin(4 * t),
            -0.08 * np.sin(2 * t),
            0.1 * abs(np.sin(3 * t)),
            0.2 * np.sin(2 * t) / 1.5,
            0.2 * np.sin(3 * t) / 1.5,
            0.2 * np.sin(4 * t) / 1.5,
        ]
    )


def detect_board(image, intrinsics):
    """Return target-to-camera and reprojection RMS for a rectified RGB image."""
    aruco = cv2.aruco
    board = aruco.CharucoBoard((7, 5), 0.037, 0.030, aruco.getPredefinedDictionary(aruco.DICT_5X5_100))
    params = aruco.CharucoParameters()
    params.cameraMatrix = np.asarray(intrinsics, dtype=float)
    params.distCoeffs = np.zeros(5)  # ZED VIEW.LEFT/RIGHT are rectified.
    detector = aruco.CharucoDetector(board, params)
    corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
    if ids is None or len(ids) < 10:
        return None
    objects, pixels = board.matchImagePoints(corners, ids)
    ok, rvec, tvec = cv2.solvePnP(objects, pixels, intrinsics, np.zeros(5))
    if not ok:
        return None
    projected, _ = cv2.projectPoints(objects, rvec, tvec, intrinsics, np.zeros(5))
    rms = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=-1))))
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(rvec)[0]
    matrix[:3, 3] = tvec.ravel()
    return matrix, rms


def solve_hand_eye(ee_to_base, target_to_camera):
    """Solve AX=XB and reject degenerate motion or an inconsistent fixed board."""
    ee, target = np.asarray(ee_to_base), np.asarray(target_to_camera)
    if ee.shape != target.shape or ee.ndim != 3 or ee.shape[1:] != (4, 4) or len(ee) < 10:
        raise ValueError("Need at least 10 paired 4x4 poses with visible board corners")
    if not np.isfinite(ee).all() or not np.isfinite(target).all():
        raise ValueError("Calibration poses must be finite")
    rotations = Rotation.from_matrix(ee[0, :3, :3].T @ ee[:, :3, :3]).as_rotvec()
    if np.linalg.svd(rotations, compute_uv=False)[1] < 0.1:
        raise ValueError("Insufficient rotation about independent axes; recapture the board")
    r, t = cv2.calibrateHandEye(
        ee[:, :3, :3], ee[:, :3, 3], target[:, :3, :3], target[:, :3, 3], method=cv2.CALIB_HAND_EYE_DANIILIDIS
    )
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = r, t.ravel()
    if not np.isfinite(result).all():
        raise ValueError("Hand-eye solve failed")
    board_world = ee @ result @ target
    center = np.median(board_world[:, :3, 3], axis=0)
    position_error = np.linalg.norm(board_world[:, :3, 3] - center, axis=1)
    rotation = Rotation.from_matrix(board_world[:, :3, :3])
    angle_error = (rotation.mean().inv() * rotation).magnitude()
    quality = {
        "samples": len(ee),
        "max_board_error_m": float(position_error.max()),
        "max_board_error_deg": float(np.rad2deg(angle_error.max())),
    }
    if quality["max_board_error_m"] > 0.01 or quality["max_board_error_deg"] > 3:
        raise ValueError(f"Board consistency check failed; keep board fixed and recapture: {quality}")
    return pose_vector(result), quality

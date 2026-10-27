"""Per-embodiment calibration storage, independent of hardware SDKs."""

import json
import math
import os
from pathlib import Path
import time

EMBODIMENTS = ("robotiq", "umi", "sharpa")
dir_path = str(Path(__file__).resolve().parent)
FRAME = "attachment_site"


def calib_info_path(name=None):
    if name not in EMBODIMENTS:
        raise ValueError(f"Select an embodiment from {EMBODIMENTS}, got {name!r}")
    return str(Path(dir_path) / f"calibration_info_{name}.json")


def _validate_pose(pose):
    if not isinstance(pose, (list, tuple)) or len(pose) != 6:
        raise ValueError("Pose must contain six values: xyz meters, xyz Euler radians")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pose):
        raise ValueError("Pose must contain finite numbers")


def load_calibration_info(keep_time=False, name=None, camera_id=None, lens="left"):
    filepath = calib_info_path(name)
    if lens not in ("left", "right"):
        raise ValueError("Camera lens must be left or right")
    if not Path(filepath).is_file():
        raise FileNotFoundError(
            f"No calibration at {filepath}. Run scripts/calibrate_wrist_camera.sh "
            f"with EMBODIMENT={name} and WRIST_CAMERA_ID set to your camera."
        )
    with open(filepath) as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Empty or invalid calibration: {filepath}")
    for key, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid calibration entry: {key}")
        _validate_pose(entry.get("pose"))
        # Existing manually renamed files remain readable.
        if "schema_version" in entry:
            if entry["schema_version"] != 1:
                raise ValueError("Unsupported calibration schema")
            if entry.get("embodiment") != name or entry.get("frame") != FRAME:
                raise ValueError("Calibration embodiment or coordinate frame does not match")
            if entry.get("method") not in ("charuco_hand_eye_manual", "silhouette"):
                raise ValueError("Unsupported calibration method")
            if entry.get("method") == "charuco_hand_eye_manual" and entry.get("manual_accepted") is not True:
                raise ValueError("Calibration has not completed manual tuning")
            if key != f"{entry.get('camera_id')}_{entry.get('lens')}":
                raise ValueError("Calibration camera identity does not match its key")
    if camera_id is not None and f"{camera_id}_{lens}" not in data:
        raise ValueError(f"No {lens}-lens calibration for camera {camera_id} in {filepath}")
    return data if keep_time else {key: entry["pose"] for key, entry in data.items()}


def save_calibration_info(name, camera_id, pose, *, initial_pose, lens="left", quality=None, method="charuco_hand_eye_manual"):
    """Save a completed fit atomically, retaining other camera entries."""
    if method not in ("charuco_hand_eye_manual", "silhouette"):
        raise ValueError("Unsupported calibration method")
    filepath = Path(calib_info_path(name))
    if lens not in ("left", "right") or not str(camera_id).isdigit():
        raise ValueError("Expected numeric camera serial and left/right lens")
    pose, initial_pose = list(map(float, pose)), list(map(float, initial_pose))
    _validate_pose(pose)
    _validate_pose(initial_pose)
    data = load_calibration_info(keep_time=True, name=name) if filepath.exists() else {}
    data[f"{camera_id}_{lens}"] = {
        "schema_version": 1,
        "embodiment": name,
        "camera_id": str(camera_id),
        "lens": lens,
        "frame": FRAME,
        "pose": pose,
        "initial_pose": initial_pose,
        "manual_accepted": method == "charuco_hand_eye_manual",
        "method": method,
        "timestamp": time.time(),
        "quality": quality or {},
    }
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    pending = filepath.with_suffix(".json.pending")
    with pending.open("w") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, filepath)
    return filepath

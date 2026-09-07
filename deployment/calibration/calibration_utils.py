import json
import os

# Prepare Calibration Info #
dir_path = os.path.dirname(os.path.realpath(__file__))


def calib_info_path(name=None):
    """Path to the calibration JSON. ``name`` selects a per-embodiment file
    (e.g. "robotiq" -> calibration_info_robotiq.json); ``None`` is the base
    calibration_info.json."""
    fname = f"calibration_info_{name}.json" if name else "calibration_info.json"
    return os.path.join(dir_path, fname)


def load_calibration_info(keep_time=False, name=None):
    filepath = calib_info_path(name)
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"No camera calibration found at {filepath}. "
            f"Place the robot's calibration_info_{name}.json there."
        )
    with open(filepath, "r") as jsonFile:
        calibration_info = json.load(jsonFile)
    if not keep_time:
        calibration_info = {key: data["pose"] for key, data in calibration_info.items()}
    return calibration_info

"""DROID RLDS helpers shared by the preprocessing scripts.

The Silhouette Calibration Sim (MuJoCo FK + wrist-camera gripper mask) lives in
``openpi.calibration.silhouette``.
"""

import re

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------------
# DROID RLDS helpers
# ---------------------------------------------------------------------------

KNOWN_LABS = {
    "AUTOLab", "CLVR", "GuptaLab", "ILIAD", "IPRL", "IRIS",
    "PennPAL", "RAD", "RAIL", "REAL", "RPL", "TRI", "WEIRD",
}
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
_LAB_BY_LOWER = {lab.lower(): lab for lab in KNOWN_LABS}


def find_dataset_dir(data_dir) -> "Path":
    """Find the TFDS dataset directory (the one holding features.json) under `data_dir`."""
    hits = list(data_dir.rglob("features.json"))
    if not hits:
        raise FileNotFoundError(f"No TFDS dataset (features.json) found under {data_dir}")
    return hits[0].parent


def parse_lab(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("success", "failure") and i > 0:
            return parts[i - 1]
    raise ValueError(f"file_path missing 'success'/'failure' segment: {file_path!r}")


def _parse_timestamp_dir(ts_dir: str):
    """DROID path TIMESTAMP_DIR -> (year, month, day, hh, mm, ss); handles ':' and '_' separators."""
    tokens = [t for t in re.split(r"[_:]", ts_dir) if t]
    if len(tokens) != 7:
        return None
    _dow, mon, day, hh, mm, ss, year = tokens
    if mon not in _MONTHS:
        return None
    try:
        return int(year), _MONTHS[mon], int(day), int(hh), int(mm), int(ss)
    except ValueError:
        return None


def make_lang_annotation_key(lab: str, year: int, month: int, day: int, hh: int, mm: int, ss: int) -> str:
    """Canonical '<LAB>|<timestamp>' lookup key shared by the metadata builder and preprocess script."""
    lab = _LAB_BY_LOWER.get(lab.lower(), lab)
    return f"{lab}|{year:04d}-{month:02d}-{day:02d}-{hh:02d}h-{mm:02d}m-{ss:02d}s"


def parse_annotation_episode_id(episode_id: str):
    """Parse a HF annotation episode_id '<LAB>+<hash>+<YYYY-MM-DD-HHh-MMm-SSs>' into a 7-tuple, or None."""
    parts = episode_id.split("+")
    if len(parts) != 3:
        return None
    lab, _hash, ts = parts
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})h-(\d{1,2})m-(\d{1,2})s", ts)
    if not m:
        return None
    try:
        year, month, day, hh, mm, ss = (int(x) for x in m.groups())
    except ValueError:
        return None
    return lab, year, month, day, hh, mm, ss


def file_path_to_lang_key(file_path: str):
    """DROID episode path -> canonical 'LAB|YYYY-MM-DD-HHh-MMm-SSs' key, or None."""
    parts = file_path.replace("\\", "/").rstrip("/").split("/")
    if parts and parts[-1] == "trajectory.h5":
        parts = parts[:-1]
    if len(parts) < 4:
        return None
    parsed = _parse_timestamp_dir(parts[-1])
    if parsed is None:
        return None
    return make_lang_annotation_key(parse_lab(file_path), *parsed)


def decode_camera_frames(steps, camera_key: str) -> np.ndarray:
    """Decode all frames for one camera. Returns uint8 (N, H, W, 3)."""
    frames = []
    for step in steps:
        img = step["observation"][camera_key]
        if img.dtype == tf.string:
            img = tf.io.decode_jpeg(img)
        frames.append(img.numpy())
    return np.stack(frames)


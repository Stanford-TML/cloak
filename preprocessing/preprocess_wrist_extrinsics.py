#!/usr/bin/env python3
"""Silhouette Calibration of DROID wrist-camera extrinsics (Algorithm 1).

For each DROID episode we recover the wrist camera's pose in the end-effector
frame ("cam_to_gripper") by matching a rendered gripper silhouette to the
gripper in the wrist image:

  1. Build a target mask of the gripper from the wrist video — the intersection
     of low-intensity (the gripper is dark) and low temporal-std (the gripper is
     rigidly mounted, so its pixels barely move) over the gripper-open frames.
  2. Forward-kinematics the end-effector, then Nelder-Mead optimize the 6-DOF
     cam_to_gripper so the rendered gripper mask maximizes IoU with the target.

Outliers (the worst 1% by drift from the DROID mean) are dropped, and the
results are written as { "<LAB>|<timestamp>": [tx,ty,tz, rx,ry,rz] }.

    python preprocessing/preprocess_wrist_extrinsics.py --data-dir /path/to/DROID --output out.json
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import json
import multiprocessing
import os
from pathlib import Path
import sys
from typing import Literal

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import tensorflow as tf
import tensorflow_datasets as tfds
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openpi.constants import DEFAULT_CAM_TO_GRIPPER, DEFAULT_WRIST_INTRINSICS, WRIST_CAM_KEY  # noqa: E402
from utils import Sim, decode_camera_frames, file_path_to_lang_key, find_dataset_dir  # noqa: E402

# Target-mask construction.
OPEN_TOL = 0.05         # gripper_position <= this counts as "open"
ROI_TOP_FRAC = 0.50     # gripper lives in the bottom rows / right cols of the wrist frame
ROI_LEFT_FRAC = 0.20
BIN_KEEP_FRAC = 0.50    # keep the darkest / most-rigid half of the ROI
MIN_KEEP_FRAMES = 1

# Nelder-Mead settings.
NM_MAXITER = 600
NM_INIT_T_STEP = 0.005   # 5 mm initial simplex spread (translation)
NM_INIT_R_STEP = 0.0087  # ~0.5 deg initial simplex spread (rotation)

# Drop episodes past this percentile of translation/rotation drift from the mean.
EXTRINSICS_OUTLIER_PERCENTILE = 99.0


@dataclasses.dataclass(frozen=True)
class CalibOptions:
    cue: Literal["intersection", "intensity", "std"] = "intersection"
    """Target-mask cue: low intensity, low temporal std, or their intersection."""
    keep_frac: float = BIN_KEEP_FRAC
    """Fraction of ROI pixels kept by each cue's percentile threshold."""


@dataclasses.dataclass(frozen=True)
class Target:
    """A pseudo-GT silhouette plus the robot state to render its match from."""
    mask: np.ndarray
    joint_position: np.ndarray
    gripper_position: float
    step: int


def _gripper_positions(steps) -> np.ndarray:
    return np.array([s["observation"]["gripper_position"].numpy().item() for s in steps])


def _select_kept_gray(steps, frames):
    """Grayscale (K, H, W) of the gripper-open frames, or None if too few."""
    keep = _gripper_positions(steps) <= OPEN_TOL
    if int(keep.sum()) < MIN_KEEP_FRAMES:
        return None
    return np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames[keep]]).astype(np.float32)


def _gripper_roi(H: int, W: int) -> np.ndarray:
    roi = np.ones((H, W), dtype=bool)
    roi[: int(H * ROI_TOP_FRAC), :] = False
    roi[:, : int(W * ROI_LEFT_FRAC)] = False
    return roi


def _threshold_mask(score: np.ndarray, roi: np.ndarray, keep_frac: float) -> np.ndarray:
    """Keep the bottom `keep_frac` of `score` inside `roi`."""
    thr = float(np.percentile(score[roi], keep_frac * 100))
    return (score < thr) & roi


def build_target_mask(steps, frames, opts: CalibOptions = CalibOptions()):
    """Gripper target mask from the gripper-open frames, in the gripper ROI. None if too few frames."""
    gray = _select_kept_gray(steps, frames)
    if gray is None:
        return None
    roi = _gripper_roi(*gray.shape[1:])
    if opts.cue == "intensity":
        return _threshold_mask(np.median(gray, axis=0), roi, opts.keep_frac)
    if opts.cue == "std":
        return _threshold_mask(gray.std(axis=0), roi, opts.keep_frac)
    intensity = _threshold_mask(np.median(gray, axis=0), roi, opts.keep_frac)
    std = _threshold_mask(gray.std(axis=0), roi, opts.keep_frac)
    return intensity & std


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / max(union, 1)


def _pose_distance(pose: np.ndarray, ref: np.ndarray):
    """L2 translation (m) and axis-angle rotation magnitude (rad) between two 6-DOF poses."""
    R = Rotation.from_euler("xyz", pose[3:]) * Rotation.from_euler("xyz", ref[3:]).inv()
    return float(np.linalg.norm(pose[:3] - ref[:3])), float(R.magnitude())


def first_open_step(steps) -> int:
    return int(np.argmax(_gripper_positions(steps) <= OPEN_TOL))


def build_target(steps, frames, opts: CalibOptions = CalibOptions()) -> Target | None:
    """Open-frame target mask with the robot state of the first gripper-open step."""
    mask = build_target_mask(steps, frames, opts)
    if mask is None:
        return None
    t = first_open_step(steps)
    return Target(mask, steps[t]["observation"]["joint_position"].numpy(),
                  steps[t]["observation"]["gripper_position"].numpy().item(), t)


def render_target_pose(target: Target, params: np.ndarray, fy: float, sim: Sim) -> np.ndarray:
    T_ee = sim.set_pose(target.joint_position, target.gripper_position)
    return sim.render_gripper_mask(T_ee @ sim.t_c2g_from_6vec(params), fy)


def optimize_pose(target: Target, fy: float, sim: Sim, init: np.ndarray):
    """Nelder-Mead the 6-DOF cam_to_gripper from `init` to maximize IoU with `target`.

    Returns (cam_to_gripper, iou_init, iou_opt, n_iter).
    """

    def loss(params):
        return 1.0 - _iou(render_target_pose(target, params, fy, sim), target.mask)

    init = np.asarray(init, dtype=np.float64)
    init_step = np.array([NM_INIT_T_STEP] * 3 + [NM_INIT_R_STEP] * 3)
    init_simplex = np.vstack([init] + [init + init_step * e for e in np.eye(6)])
    iou_init = 1.0 - loss(init)
    result = minimize(
        loss, x0=init, method="Nelder-Mead",
        options={"initial_simplex": init_simplex, "maxiter": NM_MAXITER,
                 "xatol": 1e-4, "fatol": 1e-4, "adaptive": True, "disp": False},
    )
    return np.asarray(result.x, dtype=np.float64), float(iou_init), float(1.0 - result.fun), int(result.nit)


def optimize_episode(steps, frames, fy: float, sim: Sim, opts: CalibOptions = CalibOptions()):
    """Return (cam_to_gripper, iou_init, iou_opt). Raises RuntimeError if unoptimizable."""
    target = build_target(steps, frames, opts)
    if target is None:
        raise RuntimeError("no gripper-open frames for target mask")
    pose, iou_init, iou_opt, _ = optimize_pose(target, fy, sim, DEFAULT_CAM_TO_GRIPPER)
    return pose, iou_init, iou_opt


def filter_outliers(entries: dict, percentile: float = EXTRINSICS_OUTLIER_PERCENTILE) -> dict:
    """Drop entries past `percentile` of translation or rotation drift from DEFAULT_CAM_TO_GRIPPER."""
    keys = list(entries)
    if not keys:
        return entries
    dist = {k: _pose_distance(np.asarray(entries[k]), DEFAULT_CAM_TO_GRIPPER) for k in keys}
    t_thr = float(np.percentile([dist[k][0] for k in keys], percentile))
    r_thr = float(np.percentile([dist[k][1] for k in keys], percentile))
    return {k: entries[k] for k in keys if dist[k][0] < t_thr and dist[k][1] < r_thr}


def write_json(entries: dict, path: Path) -> None:
    """Write { key: [6 floats] } sorted, one human-readable entry per line."""
    items = sorted(entries.items())
    lines = ["{"]
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        lines.append(f"  {json.dumps(k)}: {json.dumps(v)}{comma}")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


@dataclasses.dataclass
class Args:
    data_dir: Path
    """Path to the DROID RLDS dataset."""
    output: Path = Path("droid_wrist_extrinsics.json")
    limit: int | None = None
    """Cap the number of episodes per shard (for testing)."""
    n_workers: int = 24
    """Parallel worker processes, one shard at a time each (each holds its own Sim)."""
    opts: CalibOptions = dataclasses.field(default_factory=CalibOptions)


_WORKER: dict = {}


def process_shard(shard: Path, dataset_dir: Path, opts: CalibOptions, limit: int | None):
    """Calibrate every episode in one shard. Returns (entries, n_fail, log lines)."""
    if not _WORKER:
        _WORKER["sim"] = Sim()
        _WORKER["builder"] = tfds.builder_from_directory(str(dataset_dir))
    sim, builder = _WORKER["sim"], _WORKER["builder"]
    # One representative wrist focal length; per-episode intrinsics would tighten accuracy.
    fy = float(DEFAULT_WRIST_INTRINSICS[1])
    entries, n_fail, logs = {}, 0, []
    for raw in tf.data.TFRecordDataset([str(shard)]):
        ep = builder.info.features.deserialize_example(raw.numpy())
        nfs = ep["episode_metadata"]["file_path"].numpy().decode()
        if "/success/" not in nfs:
            continue
        key = file_path_to_lang_key(nfs)
        if key is None:
            continue
        steps = list(ep["steps"])
        frames = decode_camera_frames(steps, WRIST_CAM_KEY)
        try:
            pose, iou_init, iou_opt = optimize_episode(steps, frames, fy, sim, opts)
        except RuntimeError:
            n_fail += 1
            continue
        entries[key] = pose.tolist()
        logs.append(f"{key}  IoU {iou_init:.2f} -> {iou_opt:.2f}")
        if limit and len(entries) >= limit:
            break
    return entries, n_fail, logs


def main(args: Args) -> None:
    dataset_dir = find_dataset_dir(args.data_dir)
    shards = sorted(p for p in dataset_dir.glob("*.tfrecord*") if not p.name.endswith(".tmp"))
    print(f"Found {len(shards)} shards in {dataset_dir}")

    entries: dict[str, list] = {}
    n_fail = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(process_shard, s, dataset_dir, args.opts, args.limit): s for s in shards}
        for i, fut in enumerate(as_completed(futures), 1):
            shard_entries, shard_fail, logs = fut.result()
            entries.update(shard_entries)
            n_fail += shard_fail
            print("\n".join(logs))
            print(f"[{i}/{len(shards)}] {futures[fut].name}  ok={len(entries)} fail={n_fail}", flush=True)
            # Checkpoint so a crash does not lose finished shards.
            write_json(entries, args.output)

    n_before = len(entries)
    entries = filter_outliers(entries)
    write_json(entries, args.output)
    print(f"\nOptimized {n_before} ({n_fail} skipped); dropped {n_before - len(entries)} outliers; "
          f"wrote {len(entries)} -> {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))

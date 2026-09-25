#!/usr/bin/env python3
"""Silhouette Calibration of DROID wrist-camera extrinsics (Algorithm 1).

For each DROID episode we recover the wrist camera's pose in the end-effector
frame ("cam_to_gripper") by matching a rendered gripper silhouette to the
gripper in the wrist image. The algorithm itself lives in
``openpi.calibration.silhouette`` (shared with the live-robot calibration in
deployment/calibrate_wrist_camera.py); this script runs it over DROID RLDS.

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

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openpi.calibration.silhouette import CalibOptions, Sim, build_target, optimize_pose, pose_distance  # noqa: E402
from openpi.constants import DEFAULT_CAM_TO_GRIPPER, DEFAULT_WRIST_INTRINSICS, WRIST_CAM_KEY  # noqa: E402
from utils import decode_camera_frames, file_path_to_lang_key, find_dataset_dir  # noqa: E402

# Drop episodes past this percentile of translation/rotation drift from the mean.
EXTRINSICS_OUTLIER_PERCENTILE = 99.0

SHARD_INDEX_PATH = Path(__file__).resolve().parent.parent / "assets" / "droid_shard_index.json"


def optimize_episode(steps, frames, fy: float, sim: Sim, opts: CalibOptions = CalibOptions()):
    """Return (cam_to_gripper, iou_init, iou_opt). Raises RuntimeError if unoptimizable."""
    joint_positions = np.stack([s["observation"]["joint_position"].numpy() for s in steps])
    gripper_positions = np.array([s["observation"]["gripper_position"].numpy().item() for s in steps])
    target = build_target(frames, joint_positions, gripper_positions, opts)
    if target is None:
        raise RuntimeError("no gripper-open frames for target mask")
    pose, iou_init, iou_opt, _ = optimize_pose(target, fy, sim, DEFAULT_CAM_TO_GRIPPER)
    return pose, iou_init, iou_opt


def filter_outliers(entries: dict, percentile: float = EXTRINSICS_OUTLIER_PERCENTILE) -> dict:
    """Drop entries past `percentile` of translation or rotation drift from DEFAULT_CAM_TO_GRIPPER."""
    keys = list(entries)
    if not keys:
        return entries
    dist = {k: pose_distance(np.asarray(entries[k]), DEFAULT_CAM_TO_GRIPPER) for k in keys}
    t_thr = float(np.percentile([dist[k][0] for k in keys], percentile))
    r_thr = float(np.percentile([dist[k][1] for k in keys], percentile))
    return {k: entries[k] for k in keys if dist[k][0] < t_thr and dist[k][1] < r_thr}


def shards_for_keys(shards: list[Path], keys: set[str]) -> list[Path]:
    """The shards holding `keys`, looked up in the shard-index asset."""
    shard_index = json.loads(SHARD_INDEX_PATH.read_text())
    wanted = {f"{shard_index[k][0]:05d}" for k in keys}
    return [s for s in shards if s.name.split(".tfrecord-")[1].split("-of-")[0] in wanted]


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
    keys: Path | None = None
    """Only calibrate the episode keys (LAB|timestamp, one per line) in this file; shards
    are looked up in assets/droid_shard_index.json."""
    n_workers: int = 24
    """Parallel worker processes, one shard at a time each (each holds its own Sim)."""
    opts: CalibOptions = dataclasses.field(default_factory=CalibOptions)


_WORKER: dict = {}


def process_shard(shard: Path, dataset_dir: Path, opts: CalibOptions, limit: int | None, keys: set | None = None):
    """Calibrate every episode in one shard (or only those in `keys`). Returns (entries, n_fail, log lines)."""
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
        if key is None or (keys is not None and key not in keys):
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
    keys = None
    if args.keys is not None:
        keys = set(args.keys.read_text().split())
        shards = shards_for_keys(shards, keys)
        print(f"Restricting to {len(keys)} episodes in {len(shards)} shards")

    entries: dict[str, list] = {}
    n_fail = 0
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(process_shard, s, dataset_dir, args.opts, args.limit, keys): s for s in shards}
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

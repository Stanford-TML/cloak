"""DROID batch driver: shard lookup for a key subset, outlier filter, and JSON output."""
import json
from pathlib import Path

import numpy as np

import preprocess_wrist_extrinsics as pwe
from openpi.constants import DEFAULT_CAM_TO_GRIPPER


def test_shards_for_keys_uses_shard_index(tmp_path, monkeypatch):
    index = {"LAB|a": [3, 0], "LAB|b": [3, 5], "LAB|c": [17, 1], "LAB|d": [200, 2]}
    (tmp_path / "index.json").write_text(json.dumps(index))
    monkeypatch.setattr(pwe, "SHARD_INDEX_PATH", tmp_path / "index.json")
    shards = [Path(f"droid_101-train.tfrecord-{i:05d}-of-02048") for i in range(2048)]

    out = pwe.shards_for_keys(shards, {"LAB|a", "LAB|b", "LAB|c"})
    assert [s.name for s in out] == [shards[3].name, shards[17].name]


def test_filter_outliers_drops_far_pose():
    base = DEFAULT_CAM_TO_GRIPPER
    rng = np.random.default_rng(0)
    entries = {f"LAB|{i}": (base + rng.normal(0, 1e-4, 6)).tolist() for i in range(20)}
    entries["LAB|far"] = (base + np.array([0.05, 0, 0, 0, 0, 0])).tolist()
    kept = pwe.filter_outliers(entries)
    assert "LAB|far" not in kept and len(kept) >= 19


def test_write_json_round_trips(tmp_path):
    entries = {"B|2": [1.0, 2.0, 3.0, 0.1, 0.2, 0.3], "A|1": [0.0] * 6}
    out = tmp_path / "out.json"
    pwe.write_json(entries, out)
    assert json.loads(out.read_text()) == entries
    assert out.read_text().startswith('{\n  "A|1"')  # sorted, one entry per line

import json
import numpy as np
import pytest
from deployment.calibration import calibration_utils as storage


@pytest.fixture(autouse=True)
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "dir_path", str(tmp_path))


def test_embodiment_and_lens_roundtrip():
    for name in storage.EMBODIMENTS:
        pose = np.arange(6) / 100
        path = storage.save_calibration_info(name, "123", pose, initial_pose=np.zeros(6))
        assert path.name == f"calibration_info_{name}.json"
        assert storage.load_calibration_info(name=name, camera_id="123") == {"123_left": pose.tolist()}
        with pytest.raises(ValueError, match="No right-lens"):
            storage.load_calibration_info(name=name, camera_id="123", lens="right")
        with pytest.raises(ValueError, match="camera 456"):
            storage.load_calibration_info(name=name, camera_id="456")


@pytest.mark.parametrize("name", [None, "../umi", "yam"])
def test_invalid_embodiment(name):
    with pytest.raises(ValueError):
        storage.calib_info_path(name)


def test_reject_nonfinite_without_overwriting():
    path = storage.save_calibration_info("umi", "123", [0] * 6, initial_pose=[0] * 6)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="finite"):
        storage.save_calibration_info("umi", "123", [float("nan")] * 6, initial_pose=[0] * 6)
    assert path.read_bytes() == before


def test_reject_wrong_frame_and_unaccepted():
    path = storage.save_calibration_info("sharpa", "123", [0] * 6, initial_pose=[0] * 6)
    data = json.loads(path.read_text())
    data["123_left"]["frame"] = "base"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="coordinate frame"):
        storage.load_calibration_info(name="sharpa")
    data["123_left"]["frame"] = storage.FRAME
    data["123_left"]["manual_accepted"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="manual tuning"):
        storage.load_calibration_info(name="sharpa")


def test_silhouette_result_loads_without_manual_acceptance():
    path = storage.save_calibration_info(
        "robotiq", "123", [0] * 6, initial_pose=[0] * 6, method="silhouette"
    )
    entry = json.loads(path.read_text())["123_left"]
    assert entry["manual_accepted"] is False
    assert entry["method"] == "silhouette"
    assert storage.load_calibration_info(name="robotiq", camera_id="123")["123_left"] == [0] * 6

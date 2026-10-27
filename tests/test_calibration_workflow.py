from types import SimpleNamespace
import numpy as np
import pytest
from deployment import calibrate_camera as workflow
from deployment.calibration import calibration_utils as storage
from deployment.calibration.hand_eye import transform


@pytest.mark.parametrize("accept", [False, True])
def test_resume_requires_acceptance_and_never_connects_hardware(tmp_path, monkeypatch, accept):
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(storage, "dir_path", str(tmp_path))
    monkeypatch.setattr(workflow, "capture", lambda _: pytest.fail("Resume must not connect to hardware"))
    args = SimpleNamespace(embodiment="umi", wrist_camera_id="123", lens="left", resume=True)
    rng = np.random.default_rng(9)
    ee = np.array([transform(np.r_[rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.5, 0.5, 3)]) for _ in range(20)])
    camera = transform([0.03, -0.04, 0.02, 0.1, 0.2, -0.5])
    target = iter(np.linalg.inv(ee @ camera) @ transform([0.5, 0, 0.3, 0.2, -0.4, 0.1]))
    monkeypatch.setattr(workflow, "detect_board", lambda *args: (next(target), 0.2))
    final_pose = np.array([0.031, -0.04, 0.02, 0.1, 0.2, -0.5])
    monkeypatch.setattr(workflow, "tune", lambda *args: final_pose if accept else None)
    folder = tmp_path / "scratch" / "wrist_calibration" / "umi"
    folder.mkdir(parents=True)
    np.savez(
        folder / "capture_123_left.npz",
        frames=np.zeros((20, 10, 10, 3), dtype=np.uint8),
        joint_positions=np.zeros((20, 7)),
        gripper_positions=np.zeros(20),
        ee_to_base=ee,
        intrinsics=np.eye(3),
        embodiment="umi",
        camera_id="123",
        lens="left",
    )
    path = storage.save_calibration_info("umi", "123", [0] * 6, initial_pose=[0] * 6)
    original = path.read_bytes()
    workflow.main(args)
    if accept:
        np.testing.assert_allclose(storage.load_calibration_info(name="umi")["123_left"], final_pose)
    else:
        assert path.read_bytes() == original

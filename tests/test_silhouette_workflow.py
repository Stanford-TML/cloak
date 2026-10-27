"""Offline integration of the default silhouette command and deployment storage."""
import numpy as np
import pytest
from deployment import calibrate_wrist_camera as workflow
from deployment.calibration import calibration_utils as storage
from openpi.constants import DEFAULT_CAM_TO_GRIPPER, DROID_ARM_QPOS

@pytest.mark.parametrize("embodiment,lens", [("unknown", "left"), ("robotiq", "right")])
def test_unsupported_rig_rejected_before_capture(monkeypatch, embodiment, lens):
    monkeypatch.setattr(workflow, "capture", lambda *a: pytest.fail("hardware accessed"))
    with pytest.raises(ValueError):
        workflow.main(workflow.Args("123", embodiment=embodiment, lens=lens))

@pytest.mark.parametrize("mode", ["resume", "captures", "from_capture"])
@pytest.mark.parametrize("iou", [0.8, 0.0, float("nan")])
def test_resume_saves_fit_or_preserves_previous_calibration(tmp_path, monkeypatch, iou, mode):
    monkeypatch.setattr(storage, "dir_path", str(tmp_path))
    monkeypatch.setattr(workflow, "OUTPUT_ROOT", tmp_path / "captures")
    monkeypatch.setattr(workflow, "capture", lambda *a: pytest.fail("hardware accessed"))
    path = storage.save_calibration_info("robotiq", "123", [0] * 6, initial_pose=[0] * 6)
    before = path.read_bytes()
    out = workflow.OUTPUT_ROOT / "robotiq"
    out.mkdir(parents=True)
    np.savez(out / "frames_robotiq.npz",
        frames=np.zeros((6, 180, 320, 3), dtype=np.uint8),
        joint_positions=np.tile(DROID_ARM_QPOS, (6, 1)),
        gripper_positions=np.zeros(6), fy=150.,
        camera_id="123", lens="left", embodiment="robotiq")
    target = workflow.silhouette.Target(np.ones((180, 320), dtype=bool), DROID_ARM_QPOS, 0., 0)
    monkeypatch.setattr(workflow.silhouette, "estimate_ee_color", lambda *a: np.zeros(3))
    monkeypatch.setattr(workflow.silhouette, "build_target", lambda *a: target)
    monkeypatch.setattr(workflow.silhouette, "optimize_pose",
        lambda *a: (DEFAULT_CAM_TO_GRIPPER.copy(), 0.2, iou, None))
    kwargs = {"resume": True} if mode == "resume" else {"captures": out} if mode == "captures" else {"from_capture": out / "frames_robotiq.npz"}
    if iou == 0.8:
        workflow.main(workflow.Args("123", **kwargs))
        loaded = storage.load_calibration_info(keep_time=True, name="robotiq", camera_id="123")
        assert loaded["123_left"]["method"] == "silhouette"
        assert (out / "mask_overlay_robotiq.png").is_file()
    else:
        with pytest.raises(ValueError, match="fit failed"):
            workflow.main(workflow.Args("123", **kwargs))
        assert path.read_bytes() == before

def test_resume_rejects_wrong_camera_before_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(workflow, "capture", lambda *a: pytest.fail("hardware accessed"))
    out = tmp_path / "robotiq"
    out.mkdir()
    np.savez(out / "frames_robotiq.npz", camera_id="999", lens="left", embodiment="robotiq")
    with pytest.raises(ValueError, match="camera_id"):
        workflow.main(workflow.Args("123", resume=True))


@pytest.mark.parametrize("position", [1.0, float("nan"), -1.0])
def test_failed_gripper_open_blocks_arm_motion(position):
    class Robot:
        def get_robot_state(self):
            return {"gripper_position": position}, {}
    with pytest.raises(RuntimeError, match="before arm motion"):
        workflow.require_open_gripper(Robot(), timeout=0)

def test_gripper_open_waits_for_feedback(monkeypatch):
    positions = iter([1.0, 0.3, 0.0])
    monkeypatch.setattr(workflow.time, "sleep", lambda _: None)
    class Robot:
        def get_robot_state(self):
            return {"gripper_position": next(positions)}, {}
    workflow.require_open_gripper(Robot())

def test_umi_seed_renders_visible_gripper():
    target = workflow.silhouette.Target(np.ones((180, 320), dtype=bool),
        workflow.UMI_ARM_QPOS, 0., 0)
    sim = workflow.silhouette.Sim("umi")
    mask = workflow.silhouette.render_target_pose(
        target, workflow.INIT_CAM_TO_GRIPPER["umi"], 150., sim)
    assert 100 < np.count_nonzero(mask) < mask.size * .8

@pytest.mark.parametrize('kind', ['folder', 'file', 'resume'])
def test_saved_capture_paths_preserve_camera_identity(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(workflow, 'OUTPUT_ROOT', tmp_path)
    monkeypatch.setattr(workflow, 'capture', lambda *a: pytest.fail('hardware accessed'))
    folder=tmp_path/'robotiq'
    folder.mkdir()
    path=folder/'frames_robotiq.npz'
    np.savez(path, camera_id='999', lens='left', embodiment='robotiq')
    kwargs={'captures':folder} if kind=='folder' else {'from_capture':path} if kind=='file' else {'resume':True}
    with pytest.raises(ValueError, match='camera_id'):
        workflow.main(workflow.Args('123', **kwargs))

@pytest.mark.parametrize('kwargs', [{'resume':True,'captures':__import__('pathlib').Path('.')}, {'captures':__import__('pathlib').Path('.'),'from_capture':__import__('pathlib').Path('x')}])
def test_capture_sources_conflict_before_hardware(monkeypatch,kwargs):
    monkeypatch.setattr(workflow, 'capture', lambda *a: pytest.fail('hardware accessed'))
    with pytest.raises(ValueError, match='only one'):
        workflow.main(workflow.Args('123', **kwargs))

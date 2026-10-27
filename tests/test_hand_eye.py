import numpy as np
import pytest
from deployment.calibration.hand_eye import solve_hand_eye, transform
from deployment.calibration.manual_tuning import adjust


def test_hand_eye_recovers_camera_in_attachment_frame():
    rng = np.random.default_rng(8)
    expected = transform([0.03, -0.08, 0.02, 0.1, 0.2, -0.5])
    board = transform([0.5, 0, 0.3, 0.2, -0.4, 0.1])
    ee = np.array([transform(np.r_[rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.5, 0.5, 3)]) for _ in range(25)])
    target = np.linalg.inv(ee @ expected) @ board
    pose, quality = solve_hand_eye(ee, target)
    np.testing.assert_allclose(transform(pose), expected, atol=1e-8)
    assert quality["max_board_error_m"] < 1e-8
    with pytest.raises(ValueError, match="Insufficient rotation"):
        solve_hand_eye(np.repeat(ee[:1], 25, axis=0), target)


def test_local_camera_offset_not_world_translation():
    pose = [0, 0, 0, 0, 0, np.pi / 2]
    moved = adjust(pose, 0, 0.01)
    np.testing.assert_allclose(moved[:3], [0, 0.01, 0], atol=1e-12)

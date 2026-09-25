#!/usr/bin/env bash
# Silhouette-calibrate the wrist camera on the real Franka + Robotiq robot and
# write deployment/calibration/calibration_info_<embodiment>.json (used by deploy_policy.sh).
# Moves the arm: keep the workspace clear and the e-stop in hand.
# Run: bash scripts/calibrate_wrist_camera.sh
# Re-fit the last capture without the robot: bash scripts/calibrate_wrist_camera.sh --from-capture scratch/wrist_calibration/robotiq/frames_robotiq.npz
set -euo pipefail

# End effector being calibrated: robotiq | sharpa.
EMBODIMENT=sharpa

# Wrist ZED serial — must match WRIST_CAMERA_ID in deploy_policy.sh.
WRIST_CAMERA_ID=14056440

# Set CLOAK_NUC_IP to the Franka NUC's IP; if unset, the controller runs locally.
uv run --group client python deployment/calibrate_wrist_camera.py \
    --embodiment "${EMBODIMENT}" \
    --wrist-camera-id "${WRIST_CAMERA_ID}" \
    "$@"

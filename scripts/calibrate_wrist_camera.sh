#!/usr/bin/env bash
# Silhouette-calibrate the wrist camera on the real Franka + Robotiq robot and
# write deployment/calibration/calibration_info_<embodiment>.json (used by deploy_policy.sh).
# Moves the arm: keep the workspace clear and the e-stop in hand.
# Run: bash scripts/calibrate_wrist_camera.sh
# Re-fit the last capture without the robot: bash scripts/calibrate_wrist_camera.sh --from-capture scratch/wrist_calibration/robotiq/frames_robotiq.npz
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/deployment_options.sh" "$@"

# The selected calibration implementation determines supported embodiments.


# Wrist ZED serial — must match WRIST_CAMERA_ID in deploy_policy.sh.
WRIST_CAMERA_ID="${WRIST_CAMERA_ID:-}"


OFFLINE_CAPTURE=false
for arg in ${CLOAK_EXTRA_ARGS[@]+"${CLOAK_EXTRA_ARGS[@]}"}; do
    case "$arg" in --captures|--captures=*|--from-capture|--from-capture=*|--resume) OFFLINE_CAPTURE=true ;; esac
done
if [[ "$OFFLINE_CAPTURE" != true && ! "$WRIST_CAMERA_ID" =~ ^[0-9]+$ ]]; then
    echo 'Set numeric WRIST_CAMERA_ID for live capture' >&2; exit 2
fi

# Set CLOAK_NUC_IP to the Franka NUC's IP; if unset, the controller runs locally.
uv run --group client python deployment/calibrate_wrist_camera.py \
    --embodiment "${EMBODIMENT}" \
    --wrist-camera-id "${WRIST_CAMERA_ID}" \
    --lens "${WRIST_CAM_LENS:-left}" \
    ${CLOAK_EXTRA_ARGS[@]+"${CLOAK_EXTRA_ARGS[@]}"}

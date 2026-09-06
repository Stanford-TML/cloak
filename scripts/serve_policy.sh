#!/usr/bin/env bash
# Deploy a policy server. Edit the variables below for any robot / config.
# Comment out CHECKPOINT_DIR to serve a RANDOM-WEIGHT policy (testing only —
# actions are meaningless, but the full transform + IK + serve pipeline runs).
set -euo pipefail

# Off-screen MuJoCo rendering for the gripper-mask transforms.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# Robots (CONFIG):
#   robotiq gripper — pi05_full_droid_finetune_{v0,v3}
#   sharpa hand     — pi05_full_droid_finetune_{v0,v3}_sharpa_ik
CONFIG=pi05_full_droid_finetune_v3_sharpa_ik
CHECKPOINT_DIR=/home/michael/src/hand_openpi/checkpoints/pi05_full_droid_finetune_v3-2/

if [[ "${CONFIG}" == *sharpa* ]]; then
    FIXED_HAND_FLAG="--fixed_hand_sharpa_ik"
else
    FIXED_HAND_FLAG=""
fi

POLICY_ARGS=(policy:checkpoint --policy.config="${CONFIG}" --policy.dir="${CHECKPOINT_DIR}")
uv run scripts/serve_policy.py ${FIXED_HAND_FLAG} "${POLICY_ARGS[@]}"

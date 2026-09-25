#!/usr/bin/env bash
# Deploy a policy server. Edit the variables below for any robot / config.
# Comment out CHECKPOINT_DIR to serve a RANDOM-WEIGHT policy (testing only —
# actions are meaningless, but the full transform + IK + serve pipeline runs).
set -euo pipefail

# Off-screen MuJoCo rendering for the gripper-mask transforms.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# One deployment config (the full-cloak v3-1 checkpoint) serves every robot; the
# robot is chosen with EMBODIMENT: robotiq | sharpa | umi | yam. use_mask renders
# that robot's wrist mask and use_ik retargets the action chunk to it (no-op for
# robotiq). EMBODIMENT must match the client (deploy_policy.sh).
# Use pi05_full_droid_finetune_v3-1_ik for the Cloak-VLA policy from the paper (uses ik for tip pose retargeting). Used for unseen embodiments (sharpa, umi, or yam)
# Use pi05_full_droid_finetune_v3-1 for Cloak-VLA without an IK layer. Works for seen embodiments (i.e. robotiq)
# Use pi05_full_droid_finetune_v0 for the Pi05_droid finetune without IK. Intended for robotiq embodiment 
CONFIG=pi05_full_droid_finetune_v0
EMBODIMENT=robotiq
CHECKPOINT_DIR=checkpoints/pi05_full_droid_finetune_v0/pi05_full_droid_finetune_v0/100000

# Sharpa can optionally lock the hand and solve only the arm (see serve_policy.py).
if [[ "${EMBODIMENT}" == "sharpa" ]]; then
    FIXED_HAND_FLAG="--fixed_hand_sharpa_ik"
else
    FIXED_HAND_FLAG=""
fi

POLICY_ARGS=(policy:checkpoint --policy.config="${CONFIG}" --policy.dir="${CHECKPOINT_DIR}")
uv run deployment/serve_policy.py --embodiment "${EMBODIMENT}" ${FIXED_HAND_FLAG} "${POLICY_ARGS[@]}"

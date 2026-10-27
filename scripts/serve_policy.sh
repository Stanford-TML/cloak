#!/usr/bin/env bash
# Deploy a trained policy. Configure CHECKPOINT_DIR in .env or the environment.
# Pass --embodiment robotiq|umi|sharpa|yam; CLI selection overrides .env.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/deployment_options.sh" "$@"

# Off-screen MuJoCo rendering for the gripper-mask transforms.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# One deployment config (the full-cloak v3-1 checkpoint) serves every robot; the
# robot is chosen with EMBODIMENT: robotiq | sharpa | umi | yam. use_mask renders
# that robot's wrist mask and use_ik retargets the action chunk to it (no-op for
# robotiq). EMBODIMENT must match the client (deploy_policy.sh).
# Use pi05_full_droid_finetune_v3-1_ik for the Cloak-VLA policy from the paper (uses ik for tip pose retargeting). Used for unseen embodiments (sharpa, umi, or yam)
# Use pi05_full_droid_finetune_v3-1 for Cloak-VLA without an IK layer. Works for seen embodiments (i.e. robotiq)
# Use pi05_full_droid_finetune_v0 for the Pi05_droid finetune without IK. Intended for robotiq embodiment
CONFIG="${CONFIG:-pi05_full_droid_finetune_v3-1_ik}"

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a trained checkpoint}"

# Sharpa can optionally lock the hand and solve only the arm (see serve_policy.py).
if [[ "${EMBODIMENT}" == "sharpa" ]]; then
    FIXED_HAND_FLAG="--fixed_hand_sharpa_ik"
else
    FIXED_HAND_FLAG=""
fi

case "${DEBUG_MASKS:-false}" in true|false) ;; *) echo 'DEBUG_MASKS must be true or false' >&2; exit 2 ;; esac
MASK_ARGS=()
if [[ "${DEBUG_MASKS:-false}" == true ]]; then MASK_ARGS=(--debug-masks); fi
POLICY_ARGS=(policy:checkpoint --policy.config="${CONFIG}" --policy.dir="${CHECKPOINT_DIR}")
uv run deployment/serve_policy.py --port "${POLICY_PORT:-8000}" --embodiment "${EMBODIMENT}" ${FIXED_HAND_FLAG} ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} ${CLOAK_EXTRA_ARGS[@]+"${CLOAK_EXTRA_ARGS[@]}"} "${POLICY_ARGS[@]}"

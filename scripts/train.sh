#!/usr/bin/env bash
# Run JAX training for a DROID config. Edit the variables below.
# Run: bash scripts/train.sh
set -euo pipefail

# Config to train (see README_train.md). v0/v3 = Robotiq base configs; the
# *_debug variants use a tiny model + droid_100 for a fast smoke test.
CONFIG=pi05_full_droid_finetune_v0_debug

# Experiment name (required) — checkpoints go under
# <checkpoint_base_dir>/<config>/<exp_name>.
EXP_NAME=${CONFIG}

# DROID dataset in RLDS format (parent of the tfds dataset dir). The *_debug
# configs load droid_100 from here (/home/michael/data/DROID/droid_100/1.0.0).
RLDS_DATA_DIR=/home/michael/data/DROID

# Weights & Biases logging: "true" to log, "false" to disable.
WANDB=true

# wandb_enabled defaults True; only pass a flag to turn it off.
WANDB_FLAG=$([ "${WANDB}" = "true" ] && echo "" || echo "--no-wandb-enabled")

# ${WANDB_FLAG} is intentionally unquoted so the empty case adds no argument.
uv run --group train python src/train.py "${CONFIG}" \
    --exp-name "${EXP_NAME}" \
    --data.rlds-data-dir "${RLDS_DATA_DIR}" \
    ${WANDB_FLAG}

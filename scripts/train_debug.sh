#!/bin/bash
#SBATCH --job-name=cloak_train_debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=scratch/slurm/cloak_train_debug_%j.log
#SBATCH --account=move
#SBATCH --partition=move
#SBATCH --time=0-01:00:00

# Load the user's shell env (uv / module setup) on SLURM compute nodes. Done
# before `set -u` so unbound vars in .bashrc don't trip the strict mode below.
source ~/.bashrc 2>/dev/null || true
set -euo pipefail

echo "===== SLURM Job Info ====="
echo "Job ID:   ${SLURM_JOB_ID:-N/A}"
echo "Job Name: ${SLURM_JOB_NAME:-N/A}"
echo "Node:     ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs:     ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "=========================="

# Off-screen MuJoCo rendering (the first-batch sanity-check images render the mask).
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Let JAX preallocate most of the GPU for training.
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

CONFIG=pi05_full_droid_finetune_v0_debug
RLDS_DATA_DIR=/move/data/DROID

# Small + fast, no side effects: a few steps, tiny batch, wandb off, and
# --save-interval 0 disables ALL checkpoint saving (including the final step).
# Bookkeeping (config.json, logs) goes to a throwaway, gitignored scratch dir that
# --overwrite reuses across runs.
uv run --group train python src/train.py "${CONFIG}" \
    --exp-name debug \
    --data.rlds-data-dir "${RLDS_DATA_DIR}" \
    --data.use-droid-100 \
    --num-train-steps 10 \
    --batch-size 8 \
    --save-interval 0 \
    --no-wandb-enabled \
    --checkpoint-base-dir scratch/debug_checkpoints \
    --overwrite

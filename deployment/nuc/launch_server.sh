#!/usr/bin/env bash
set -euo pipefail
# Conda activation hooks may inspect unset variables.
set +u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate polymetis-local
set -u
export PYTHONPATH="/app:${PYTHONPATH:-}"
exec python /cloak-nuc/run_server.py

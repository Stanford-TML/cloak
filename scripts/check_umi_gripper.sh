#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/rig_env.sh
[[ "$EMBODIMENT" == umi ]] || { echo "Set EMBODIMENT=umi first"; exit 1; }
: "${CLOAK_NUC_IP:?Set CLOAK_NUC_IP}"
uv run --group client python scripts/check_umi_gripper.py

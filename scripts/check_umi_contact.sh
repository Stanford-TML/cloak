#!/usr/bin/env bash
# Optional supervised contact check. No arm commands; native hand must be running.
set -euo pipefail
: "${NUC_SSH_HOST:?Set NUC_SSH_HOST to your NUC SSH host or user@host}"
NUC_CONTAINER="${NUC_CONTAINER:-cloak-nuc-controller-1}"
[[ "$NUC_CONTAINER" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'Invalid NUC_CONTAINER' >&2; exit 2; }
ssh -t "$NUC_SSH_HOST" "docker exec -it $NUC_CONTAINER /bin/bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate polymetis-local && exec python /cloak-nuc/check_hand_contact.py'"

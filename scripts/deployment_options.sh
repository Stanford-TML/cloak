#!/usr/bin/env bash
# Shared by calibration, server and client launchers. CLI overrides .env.
CLOAK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLOAK_REPO_ROOT"
# Preserve explicit environment settings over repository defaults.
CLOAK_SAVED_NAMES=()
CLOAK_SAVED_VALUES=()
for name in EMBODIMENT WRIST_CAMERA_ID EXTERNAL_CAMERA_ID WRIST_CAM_LENS CLOAK_NUC_IP POLICY_HOST POLICY_PORT CONFIG CHECKPOINT_DIR DEBUG DEBUG_MASKS TIMESTEPS; do
    if declare -p "$name" >/dev/null 2>&1; then
        CLOAK_SAVED_NAMES+=("$name")
        CLOAK_SAVED_VALUES+=("${!name}")
    fi
done
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi
for ((i=0; i<${#CLOAK_SAVED_NAMES[@]}; i++)); do
    printf -v "${CLOAK_SAVED_NAMES[i]}" '%s' "${CLOAK_SAVED_VALUES[i]}"
    export "${CLOAK_SAVED_NAMES[i]}"
done
CLOAK_EXTRA_ARGS=()
while (($#)); do
    case "$1" in
        --embodiment)
            [[ $# -ge 2 && "$2" != --* ]] || { echo 'Missing --embodiment value' >&2; exit 2; }
            EMBODIMENT="$2"; shift 2 ;;
        --embodiment=*) EMBODIMENT="${1#*=}"; shift ;;
        *) CLOAK_EXTRA_ARGS+=("$1"); shift ;;
    esac
done
case "${EMBODIMENT:-}" in
    robotiq|umi|sharpa|yam) ;;
    *) echo 'Set --embodiment robotiq|umi|sharpa|yam (or EMBODIMENT in .env).' >&2; exit 2 ;;
esac

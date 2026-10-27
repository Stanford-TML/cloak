#!/usr/bin/env bash
# Deploy the robot client against a running cloak policy server (scripts/serve_policy.sh).
# Test mode only: the operator types instructions at the prompt.
# Run: bash scripts/deploy_policy.sh
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/deployment_options.sh" "$@"

# Debug/mock mode: "true" runs with a mock robot + blank cameras (no hardware),
# to test the client↔server loop against a running server. "false" for real robots.
DEBUG="${DEBUG:-false}"

# Which robot to drive. Must match the server's --embodiment (robotiq | sharpa |
# umi | yam). "robotiq"/"sharpa" have client robot envs in this repo's deployment/;
# umi/yam additionally require their client-side robot env.


# Rollout length cap; 0 = run until Ctrl+C (900 ≈ 1 minute at 15 Hz).
TIMESTEPS="${TIMESTEPS:-0}"

# ZED camera serial numbers — set these to your cameras (SN<serial> files under
# /usr/local/zed/settings/). Ignored in debug mode.
EXTERNAL_CAMERA_ID="${EXTERNAL_CAMERA_ID:-}"
WRIST_CAMERA_ID="${WRIST_CAMERA_ID:-}"

case "$DEBUG" in true|false) ;; *) echo 'DEBUG must be true or false' >&2; exit 2 ;; esac
[[ "$TIMESTEPS" =~ ^[0-9]+$ ]] || { echo 'TIMESTEPS must be a nonnegative integer' >&2; exit 2; }
if [[ "$DEBUG" != true ]]; then
    : "${CLOAK_NUC_IP:?Set CLOAK_NUC_IP}"
    [[ "$WRIST_CAMERA_ID" =~ ^[0-9]+$ && "$EXTERNAL_CAMERA_ID" =~ ^[0-9]+$ ]] || { echo 'Set numeric WRIST_CAMERA_ID and EXTERNAL_CAMERA_ID' >&2; exit 2; }
fi
# "--debug" when DEBUG=true, else empty (tyro leaves debug at its default False).
DEBUG_FLAG=$([ "${DEBUG}" = "true" ] && echo "--debug" || echo "")

# Server host/port use the defaults in deploy_policy.py (edit its Args to change
# them). Client-side hardware deps live in the `client` dependency group.
# ${DEBUG_FLAG} is intentionally unquoted so the empty case adds no argument.
uv run --group client python deployment/deploy_policy.py \
    ${DEBUG_FLAG} \
    --embodiment "${EMBODIMENT}" \
    --max-timesteps "${TIMESTEPS}" \
    --remote-host "${POLICY_HOST:-127.0.0.1}" \
    --remote-port "${POLICY_PORT:-8000}" \
    --wrist-cam-lens "${WRIST_CAM_LENS:-left}" \
    --external-camera-id "${EXTERNAL_CAMERA_ID}" \
    --wrist-camera-id "${WRIST_CAMERA_ID}" \
    ${CLOAK_EXTRA_ARGS[@]+"${CLOAK_EXTRA_ARGS[@]}"}

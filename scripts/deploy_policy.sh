#!/usr/bin/env bash
# Deploy the robot client against a running cloak policy server (scripts/serve_policy.sh).
# Test mode only: the operator types instructions at the prompt.
# Run: bash scripts/deploy_policy.sh
set -euo pipefail

# Debug/mock mode: "true" runs with a mock robot + blank cameras (no hardware),
# to test the client↔server loop against a running server. "false" for real robots.
DEBUG=true

# Which robot to drive. Must match the server's --embodiment (robotiq | sharpa |
# umi | yam). "robotiq"/"sharpa" have client robot envs in this repo's deployment/;
# umi/yam additionally require their client-side robot env.
EMBODIMENT=sharpa

# Rollout length cap; 0 = run until Ctrl+C (900 ≈ 1 minute at 15 Hz).
TIMESTEPS=0

# ZED camera serial numbers — set these to your cameras (SN<serial> files under
# /usr/local/zed/settings/). Ignored in debug mode.
EXTERNAL_CAMERA_ID=32114480
WRIST_CAMERA_ID=14056440

# "--debug" when DEBUG=true, else empty (tyro leaves debug at its default False).
DEBUG_FLAG=$([ "${DEBUG}" = "true" ] && echo "--debug" || echo "")

# Server host/port use the defaults in deploy_policy.py (edit its Args to change
# them). Client-side hardware deps live in the `client` dependency group.
# ${DEBUG_FLAG} is intentionally unquoted so the empty case adds no argument.
uv run --group client python deployment/deploy_policy.py \
    ${DEBUG_FLAG} \
    --embodiment "${EMBODIMENT}" \
    --max-timesteps "${TIMESTEPS}" \
    --external-camera-id "${EXTERNAL_CAMERA_ID}" \
    --wrist-camera-id "${WRIST_CAMERA_ID}"

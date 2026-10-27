#!/usr/bin/env bash
set -euo pipefail
: "${ROBOT_IP:?Set ROBOT_IP}"
exec launch_robot.py robot_client=franka_hardware \
    "robot_client.executable_cfg.robot_ip=${ROBOT_IP}"

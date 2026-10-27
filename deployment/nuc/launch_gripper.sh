#!/usr/bin/env bash
set -euo pipefail
case "${1:?Select robotiq_2f or franka_hand}" in
    robotiq_2f)
        exec launch_gripper.py gripper=robotiq_2f "gripper.comport=${ROBOTIQ_PORT:-/dev/ttyUSB0}" ;;
    franka_hand)
        exec launch_gripper.py gripper=franka_hand \
            "gripper.executable_cfg.robot_ip=${ROBOT_IP:?Set ROBOT_IP}" ;;
    *) echo "Unsupported gripper: $1" >&2; exit 2 ;;
esac

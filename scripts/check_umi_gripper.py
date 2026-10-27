"""Operator-run empty native-hand check; connects without launching or moving the arm."""
import math
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.misc.server_interface import ServerInterface

def wait_for_position(robot, target, timeout=8.0):
    """Require three consecutive in-tolerance samples before advancing."""
    deadline = time.monotonic() + timeout
    settled = 0
    measured = float("nan")
    while time.monotonic() < deadline:
        state, _ = robot.get_robot_state()
        measured = float(state["gripper_position"])
        settled = settled + 1 if math.isfinite(measured) and abs(measured - target) <= 0.03 else 0
        if settled >= 3:
            return measured
        time.sleep(0.1)
    raise RuntimeError(f"Gripper did not settle at {target:.2f}; last position={measured:.4f}. Stop and inspect feedback.")

def main():
    robot = ServerInterface(ip_address=os.environ["CLOAK_NUC_IP"], launch=False,
                            use_gripper=True, gripper_kind="franka_hand")
    answer = input("Empty fingertips and clear of the hand? Close fully, then reopen [y/N]: ")
    if answer.strip().lower() != "y":
        print("Cancelled; no commands sent.")
        return
    values = []
    for label, position in [("open", 0.0), ("closed", 1.0), ("reopened", 0.0)]:
        robot.update_gripper(position, velocity=False, blocking=True)
        measured = wait_for_position(robot, position)
        values.append(measured)
        print(f"{label}: measured position={measured:.4f}", flush=True)
    if not (-0.01 <= values[0] <= 0.05 and 0.5 <= values[1] <= 1.01
            and -0.01 <= values[2] <= 0.05):
        raise RuntimeError("Gripper feedback check failed; do not start rollout.")
    print("PASS: empty-hand open/close/open feedback. Object-contact behavior not tested.")

if __name__ == "__main__":
    main()

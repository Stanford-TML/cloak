"""Operator-run native-hand contact check. No arm interface or controller startup."""
import time
from polymetis import GripperInterface

def wait_for(hand, condition, label, timeout=10):
    deadline = time.monotonic() + timeout
    previous = hand.get_state().timestamp
    previous_stamp = (previous.seconds, previous.nanos)
    good = 0
    while time.monotonic() < deadline:
        state = hand.get_state()
        stamp = (state.timestamp.seconds, state.timestamp.nanos)
        if state.error_code:
            raise RuntimeError(f"{label}: driver error {state.error_code}")
        if stamp != previous_stamp:
            good = good + 1 if condition(state) and not state.is_moving else 0
            previous_stamp = stamp
            if good >= 3:
                print(f"{label}: width={state.width:.4f} m, grasped={state.is_grasped}, command_ok={state.prev_command_successful}", flush=True)
                return
        time.sleep(.1)
    raise RuntimeError(f"{label} did not complete with fresh feedback; do not start rollout.")

def main():
    hand = GripperInterface(ip_address="127.0.0.1")
    maximum = float(hand.metadata.max_width)
    if not 0 < maximum < .2:
        raise RuntimeError("Invalid native-hand metadata")
    if input("Hand empty and hands clear? Open fingers [y/N]: ").strip().lower() != "y":
        return
    hand.goto(width=maximum, speed=.03, force=20, blocking=True)
    wait_for(hand, lambda s: abs(s.width-maximum)<.001 and s.prev_command_successful, "Open")
    print("Place a lightweight, non-fragile object securely supported between fingertips.")
    print("Do not hold the object by hand. Remove your hands before confirming.")
    if input("Object supported, fingers clear of support, hands clear? Grasp at 20 N [y/N]: ").strip().lower() != "y":
        return
    hand.grasp(speed=.03, force=20, grasp_width=0., epsilon_inner=0.,
               epsilon_outer=maximum, blocking=True)
    wait_for(hand, lambda s: s.is_grasped and s.prev_command_successful, "Contact")
    input("Keep the object supported and hands clear. Press Enter to reopen: ")
    hand.goto(width=maximum, speed=.03, force=20, blocking=True)
    wait_for(hand, lambda s: abs(s.width-maximum)<.001 and s.prev_command_successful, "Reopened")
    print("PASS: native-hand contact and release. No arm commands sent.")

if __name__ == "__main__":
    main()

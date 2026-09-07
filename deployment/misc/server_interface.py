import time

import numpy as np
import zerorpc

from openpi.constants import DROID_ARM_QPOS


def attempt_n_times(function_list, max_attempts, sleep_time=0.1):
    if type(function_list) is not list:
        function_list = list(function_list)

    for i in range(max_attempts):
        try:
            [f() for f in function_list]
            return
        except zerorpc.exceptions.RemoteError as err:
            last_attempt = i == (max_attempts - 1)
            if last_attempt:
                raise err
            else:
                time.sleep(sleep_time)


class ServerInterface:
    def __init__(self, ip_address="127.0.0.1", launch=True, use_gripper=True, gripper_kind="robotiq_2f", debug=False):
        self.use_gripper = use_gripper
        self.gripper_kind = gripper_kind
        self.ip_address = ip_address
        # Debug/mock mode: don't dial the NUC — return a default state and swallow
        # commands so the client↔server loop runs with no robot.
        self.debug = debug
        if debug:
            return
        self.establish_connection()

        if launch:
            func_list = [
                lambda: self.launch_controller(self.gripper_kind),
                lambda: self.launch_robot(),
            ]
            attempt_n_times(func_list, max_attempts=2)

    def establish_connection(self):
        # heartbeat=60 -> the client gives up on a call after ~120s of server
        # silence (LostRemote). This is DETECTION, not the bug: a stalled call
        # SHOULD surface rather than hang forever. 60s is large enough to ride
        # out the legitimately-long launch_robot()/launch_controller() calls
        # (~45s) without false positives. A genuine stall is then recovered by
        # the reconnect-on-LostRemote path on the read methods below.
        # NOTE: do NOT set heartbeat=None — that removes the only client-side
        # stall detector and turns a server freeze into an unkillable hang.
        self.server = zerorpc.Client(heartbeat=60)
        self.server.connect("tcp://" + self.ip_address + ":4242")

    def reconnect(self):
        """Tear down a dead zerorpc client and dial a fresh one. Used to recover
        from LostRemote (a heartbeat lapse): the Polymetis controller runs in its
        own process, so a lost heartbeat means the zerorpc bridge stalled, not
        that the robot is gone — a fresh client picks back up once the server's
        gevent hub frees up."""
        try:
            self.server.close()
        except Exception:
            pass
        self.establish_connection()

    def _read_with_reconnect(self, method_name, *args, max_attempts=5, sleep_time=1.0):
        """Call a read-only server method, surviving transient LostRemote by
        reconnecting and retrying instead of crashing the rollout. Safe only for
        idempotent reads — do NOT route command/update calls through here, since
        a blind retry could double-apply a motion."""
        for attempt in range(max_attempts):
            try:
                return getattr(self.server, method_name)(*args)
            except zerorpc.exceptions.LostRemote:
                if attempt == max_attempts - 1:
                    raise
                print(
                    f"[ServerInterface] LostRemote on {method_name}; reconnecting "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                self.reconnect()
                time.sleep(sleep_time)

    def launch_controller(self, gripper_kind="robotiq_2f"):
        # NUC server signature: launch_controller(gripper_kind="robotiq_2f").
        self.server.launch_controller(gripper_kind)

    def launch_robot(self):
        self.server.launch_robot()

    def update_command(self, command, action_space="cartesian_velocity", gripper_action_space="velocity", blocking=False):
        if self.debug:
            return {}
        action_dict = self.server.update_command(command.tolist(), action_space, gripper_action_space, blocking)
        return action_dict

    def update_joints(self, command, velocity=True, blocking=False, cartesian_noise=None):
        if self.debug:
            return
        cmd_list = np.asarray(command).tolist()
        if cartesian_noise is not None:
            cartesian_noise = np.asarray(cartesian_noise).tolist()
        self.server.update_joints(cmd_list, velocity, blocking, cartesian_noise)

    def update_gripper(self, command, velocity=True, blocking=False):
        if self.debug:
            return
        self.server.update_gripper(command, velocity, blocking)

    def get_robot_state(self):
        if self.debug:
            return {
                "cartesian_position": [0.4, 0.0, 0.3, np.pi, 0.0, 0.0],
                "joint_positions": DROID_ARM_QPOS.tolist(),
                "gripper_position": 0.0,
            }, {}
        return self._read_with_reconnect("get_robot_state")

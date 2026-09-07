import random
from collections import defaultdict

import numpy as np

from deployment.camera_utils.camera_readers.zed_camera import gather_zed_cameras


class MultiCameraWrapper:
    def __init__(self, camera_kwargs={}, debug=False, debug_camera_ids=None):
        # Debug/mock mode: no ZED hardware — read_cameras() returns blank frames
        # keyed by the given serials so the client can find its cameras.
        self.debug = debug
        if debug:
            self._debug_camera_ids = [str(s) for s in (debug_camera_ids or [])]
            self.camera_dict = {}
            return

        # Open Cameras #
        zed_cameras = gather_zed_cameras()
        self.camera_dict = {cam.serial_number: cam for cam in zed_cameras}

        # Set Correct Parameters #
        for cam_id in self.camera_dict.keys():
            curr_cam_kwargs = camera_kwargs.get(cam_id, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

        # Launch Camera #
        self.set_trajectory_mode()

    def set_trajectory_mode(self):
        for cam in self.camera_dict.values():
            cam.set_trajectory_mode()

    ### Basic Camera Functions ###
    def read_cameras(self):
        if self.debug:
            blank = np.zeros((720, 1280, 4), dtype=np.uint8)
            images = {
                f"{serial}_{lens}": blank
                for serial in self._debug_camera_ids
                for lens in ("left", "right")
            }
            return {"image": images}, {}

        full_obs_dict = defaultdict(dict)
        full_timestamp_dict = {}

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if not self.camera_dict[cam_id].is_running():
                continue
            data_dict, timestamp_dict = self.camera_dict[cam_id].read_camera()

            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])
            full_timestamp_dict.update(timestamp_dict)

        return full_obs_dict, full_timestamp_dict

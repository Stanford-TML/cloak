from copy import deepcopy

import cv2
import numpy as np

from droid.misc.time import time_ms

try:
    import pyzed.sl as sl
except ModuleNotFoundError:
    print("WARNING: You have not setup the ZED cameras, and currently cannot use them")
    sl = None


def gather_zed_cameras():
    if sl is None:
        return []
    all_zed_cameras = []
    for cam in sl.Camera.get_device_list():
        cam = ZedCamera(cam)
        all_zed_cameras.append(cam)

    return all_zed_cameras


resize_func_map = {"cv2": cv2.resize, None: None}

# Guarded so the module imports without the ZED SDK (e.g. in debug/mock mode);
# only the real-camera paths below dereference sl.
standard_params = (
    dict(
        depth_minimum_distance=0.1, camera_resolution=sl.RESOLUTION.HD720,
        depth_stabilization=False, camera_fps=60, camera_image_flip=sl.FLIP_MODE.OFF,
    )
    if sl is not None
    else {}
)


class ZedCamera:
    def __init__(self, camera):
        # Save Parameters #
        self.serial_number = str(camera.serial_number)
        self.current_mode = None
        self._current_params = None

        # Open Camera #
        print("Opening Zed: ", self.serial_number)

    def set_reading_parameters(
        self,
        image=True,
        depth=False,
        pointcloud=False,
        concatenate_images=False,
        resolution=(0, 0),
        resize_func=None,
    ):
        # Non-Permenant Values #
        self.traj_image = image
        self.traj_concatenate_images = concatenate_images
        self.traj_resolution = resolution

        # Permenant Values #
        self.depth = depth
        self.pointcloud = pointcloud
        self.resize_func = resize_func_map[resize_func]

    ### Camera Modes ###
    def set_trajectory_mode(self):
        # Set Parameters #
        self.image = self.traj_image
        self.concatenate_images = self.traj_concatenate_images
        self.skip_reading = not any([self.image, self.depth, self.pointcloud])

        if self.resize_func is None:
            self.zed_resolution = sl.Resolution(*self.traj_resolution)
            self.resizer_resolution = (0, 0)
        else:
            self.zed_resolution = sl.Resolution(0, 0)
            self.resizer_resolution = self.traj_resolution

        # Set Mode #
        change_settings = self._current_params != standard_params
        if change_settings:
            self._configure_camera(standard_params)
        self.current_mode = "trajectory"

    def _configure_camera(self, init_params):
        # Close Existing Camera #
        self.disable_camera()

        # Initialize Readers #
        self._cam = sl.Camera()
        self._sbs_img = sl.Mat()
        self._left_img = sl.Mat()
        self._right_img = sl.Mat()
        self._runtime = sl.RuntimeParameters()

        # Open Camera #
        self._current_params = init_params
        sl_params = sl.InitParameters(**init_params)
        sl_params.set_from_serial_number(int(self.serial_number))
        sl_params.camera_image_flip = sl.FLIP_MODE.OFF
        status = self._cam.open(sl_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Camera Failed To Open")

        # Save Intrinsics #
        self.latency = int(2.5 * (1e3 / sl_params.camera_fps))
        calib_params = self._cam.get_camera_information().camera_configuration.calibration_parameters
        self._intrinsics = {
            self.serial_number + "_left": self._process_intrinsics(calib_params.left_cam),
            self.serial_number + "_right": self._process_intrinsics(calib_params.right_cam),
        }

    ### Calibration Utilities ###
    def _process_intrinsics(self, params):
        intrinsics = {}
        intrinsics["cameraMatrix"] = np.array([[params.fx, 0, params.cx], [0, params.fy, params.cy], [0, 0, 1]])
        intrinsics["distCoeffs"] = np.array(list(params.disto))
        return intrinsics

    def get_intrinsics(self):
        return deepcopy(self._intrinsics)

    ### Basic Camera Utilities ###
    def _process_frame(self, frame):
        frame = deepcopy(frame.get_data())
        if self.resizer_resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resizer_resolution)

    def read_camera(self):
        # Skip if Read Unnecesary #
        if self.skip_reading:
            return {}, {}

        # Read Camera #
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        err = self._cam.grab(self._runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            return None
        timestamp_dict[self.serial_number + "_read_end"] = time_ms()

        # Benchmark Latency #
        received_time = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        timestamp_dict[self.serial_number + "_frame_received"] = received_time
        timestamp_dict[self.serial_number + "_estimated_capture"] = received_time - self.latency

        # Return Data #
        data_dict = {}

        if self.image:
            if self.concatenate_images:
                self._cam.retrieve_image(self._sbs_img, sl.VIEW.SIDE_BY_SIDE, resolution=self.zed_resolution)
                data_dict["image"] = {self.serial_number: self._process_frame(self._sbs_img)}
            else:
                self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT, resolution=self.zed_resolution)
                self._cam.retrieve_image(self._right_img, sl.VIEW.RIGHT, resolution=self.zed_resolution)
                data_dict["image"] = {
                    self.serial_number + "_left": self._process_frame(self._left_img),
                    self.serial_number + "_right": self._process_frame(self._right_img),
                }
        # if self.depth:
        # 	self._cam.retrieve_measure(self._left_depth, sl.MEASURE.DEPTH, resolution=self.resolution)
        # 	self._cam.retrieve_measure(self._right_depth, sl.MEASURE.DEPTH_RIGHT, resolution=self.resolution)
        # 	data_dict['depth'] = {
        # 		self.serial_number + '_left': self._left_depth.get_data().copy(),
        # 		self.serial_number + '_right': self._right_depth.get_data().copy()}
        # if self.pointcloud:
        # 	self._cam.retrieve_measure(self._left_pointcloud, sl.MEASURE.XYZRGBA, resolution=self.resolution)
        # 	self._cam.retrieve_measure(self._right_pointcloud, sl.MEASURE.XYZRGBA_RIGHT, resolution=self.resolution)
        # 	data_dict['pointcloud'] = {
        # 		self.serial_number + '_left': self._left_pointcloud.get_data().copy(),
        # 		self.serial_number + '_right': self._right_pointcloud.get_data().copy()}

        return data_dict, timestamp_dict

    def disable_camera(self):
        if self.current_mode == "disabled":
            return
        if hasattr(self, "_cam"):
            self._current_params = None
            self._cam.close()
        self.current_mode = "disabled"

    def is_running(self):
        return self.current_mode != "disabled"

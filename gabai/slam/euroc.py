"""Write a recording in the EuRoC folder layout and a calibration in Basalt's JSON format, so Basalt can replay it.

Layout (what Basalt's EuRoC loader reads):
  <root>/mav0/cam0/data.csv        "t_ns,filename" per frame, images in <root>/mav0/cam0/data/
  <root>/mav0/cam1/data.csv        same for the right camera
  <root>/mav0/imu0/data.csv        "t_ns,gx,gy,gz,ax,ay,az" (rad/s, m/s^2)

Images are the RECTIFIED pair (gabai.tools.calibrate_stereo.Rectifier), so the camera model is a plain pinhole and the
right camera sits +B along the left camera's x axis. The camera frames are stamped late (the picture shows the motion
from `cam_delay_s` earlier), so the writer moves their timestamps back and writes cam_time_offset_ns = 0.
"""
import csv
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

G = 9.80665
DEG = np.pi / 180.0
HEADER_IMG = "#timestamp [ns],filename"
HEADER_IMU = "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]"


def pose_dict(R, t):
    x, y, z, w = Rotation.from_matrix(np.asarray(R, float)).as_quat()
    return {"px": float(t[0]), "py": float(t[1]), "pz": float(t[2]), "qx": float(x), "qy": float(y), "qz": float(z), "qw": float(w)}


def basalt_calib(f, cx, cy, size, B, R_ic, rate_hz=200.0, gyro_noise=None, accel_noise=None, accel_params=None):
    """Basalt calibration (value0 layout of its data/euroc/euroc_ds_calib.json).

    R_ic: rotation from camera axes to IMU axes. The lever arm between IMU and left camera is NOT known: it is written as
    zero. gyro_noise / accel_noise are the discrete standard deviations (rad/s, m/s^2); the defaults are the BMI160
    datasheet noise densities (0.007 deg/s/rtHz, 180 ug/rtHz) times sqrt(rate), not measured on this unit.
    accel_params: the 9 Basalt accelerometer parameters from gabai.slam.accel_cal (zeros when the accelerometer is uncalibrated).
    """
    gyro_noise = gyro_noise if gyro_noise is not None else 0.007 * DEG * np.sqrt(rate_hz)
    accel_noise = accel_noise if accel_noise is not None else 180e-6 * G * np.sqrt(rate_hz)
    t0 = pose_dict(R_ic, [0.0, 0.0, 0.0])
    # right camera = left camera moved +B along the left camera's x, same orientation (rectified)
    t1 = pose_dict(np.asarray(R_ic, float), np.asarray(R_ic, float) @ np.array([B, 0.0, 0.0]))
    cam = {"camera_type": "pinhole", "intrinsics": {"fx": float(f), "fy": float(f), "cx": float(cx), "cy": float(cy)}}
    return {"value0": {
        "T_imu_cam": [t0, t1],
        "intrinsics": [cam, cam],
        "resolution": [[int(size[0]), int(size[1])]] * 2,
        "vignette": [],
        "calib_accel_bias": [float(v) for v in accel_params] if accel_params is not None else [0.0] * 9,
        "calib_gyro_bias": [0.0] * 12,
        "imu_update_rate": float(rate_hz),
        "accel_noise_std": [float(accel_noise)] * 3,
        "gyro_noise_std": [float(gyro_noise)] * 3,
        "accel_bias_std": [0.001] * 3,
        "gyro_bias_std": [0.0001] * 3,
        "T_mocap_world": pose_dict(np.eye(3), [0, 0, 0]),
        "T_imu_marker": pose_dict(np.eye(3), [0, 0, 0]),
        "mocap_time_offset_ns": 0,
        "mocap_to_imu_offset_ns": 0,
        "cam_time_offset_ns": 0,
    }}


class EurocWriter:
    def __init__(self, root, cam_delay_s=0.0):
        self.root, self.delay = root, cam_delay_s
        self.dirs = {k: os.path.join(root, "mav0", k) for k in ("cam0", "cam1", "imu0")}
        for k in ("cam0", "cam1"):
            os.makedirs(os.path.join(self.dirs[k], "data"), exist_ok=True)
        os.makedirs(self.dirs["imu0"], exist_ok=True)
        self.f = {k: open(os.path.join(d, "data.csv"), "w", newline="") for k, d in self.dirs.items()}
        for k in ("cam0", "cam1"):
            self.f[k].write(HEADER_IMG + "\n")
        self.f["imu0"].write(HEADER_IMU + "\n")
        self.n_frames = self.n_imu = 0
        self._last_cam_ns = self._last_imu_ns = -1

    def add_frame(self, t, left_gray, right_gray):
        """t: capture time in seconds on the Pi clock. Returns False (and writes nothing) if the stamp does not advance."""
        t_ns = int(round((t - self.delay) * 1e9))
        if t_ns <= self._last_cam_ns:
            return False
        self._last_cam_ns = t_ns
        name = f"{t_ns}.png"
        for k, img in (("cam0", left_gray), ("cam1", right_gray)):
            cv2.imwrite(os.path.join(self.dirs[k], "data", name), img)
            self.f[k].write(f"{t_ns},{name}\n")
        self.n_frames += 1
        return True

    def add_imu(self, samples):
        """samples: [t, gx, gy, gz, ax, ay, az] with the gyro in deg/s and the accelerometer in g (what the Pi sends)."""
        for t, gx, gy, gz, ax, ay, az in samples:
            t_ns = int(round(t * 1e9))
            if t_ns <= self._last_imu_ns:
                continue
            self._last_imu_ns = t_ns
            w = np.array([gx, gy, gz]) * DEG
            a = np.array([ax, ay, az]) * G
            self.f["imu0"].write(f"{t_ns},{w[0]:.9f},{w[1]:.9f},{w[2]:.9f},{a[0]:.9f},{a[1]:.9f},{a[2]:.9f}\n")
            self.n_imu += 1

    def close(self):
        for f in self.f.values():
            f.close()


def load_imu_extrinsics(path, default_delay_s=0.045):
    """Camera-to-IMU rotation and camera delay from data/imu_sync.json (written by gabai.tools.sync_imu_camera).
    Without that file the rotation is the identity and the delay a guess: calibrated is False and Basalt will not track."""
    import json
    cfg = {"R_ic": np.eye(3), "delay_s": default_delay_s, "calibrated": False}
    if path and os.path.exists(path):
        d = json.load(open(path))
        cfg.update(R_ic=np.array(d["R_ci"], float).T, delay_s=float(d["delay_s"]), calibrated=True)
    return cfg


def load_accel_calib(path="data/accel_calib.json"):
    """The 9 Basalt accelerometer parameters saved by gabai.tools.calibrate_accel, or None if that calibration was never done.
    A Basalt run without it trusts a raw accelerometer that can be off by 2 to 4 percent, which makes the result unstable."""
    import json
    if path and os.path.exists(path):
        return json.load(open(path))["basalt_params"]
    return None

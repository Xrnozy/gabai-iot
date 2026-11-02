"""One definition of the coordinate frames, so the compass, the IMU, the camera and the SLAM all mean the same thing by x, y, z.

FRAMES
  device (D)   the LEFT CAMERA frame, x right, y down, z forward (OpenCV / RDF). The stereo depth, the visual SLAM and Basalt already use it. Everything is
               expressed in it: "forward" is +z, "up" for a person wearing the device on the chest is -y.
  IMU  (I)     the BMI160 axes as wired. Not aligned with D. D <- I is R_dev_imu (v_dev = R_dev_imu @ v_imu); the file is data/imu_sync.json (R_ci).
  compass (M)  the HMC5883L axes as wired. I <- M is R_imu_mag (v_imu = R_imu_mag @ v_mag); the file is data/mag_sync.json (R_im).
  world (W)    x magnetic north, y west, z up (what the orientation filter in gabai.fusion uses). A heading is the angle of the device's forward axis
               around z, 0 when it points at magnetic north, growing towards west (as in gabai.orient.HeadingTracker).

The sensors are fixed to the device, so the matrices above never change. Putting the device flat, or on the chest, changes only the rotation device -> world,
which the filter measures from gravity (tilt) and the compass (heading). Nothing has to be remapped when the pose changes; what these functions add is
recognising WHICH pose it is (classify_mount) and the pitch of the camera in it, which the SLAM needs (tracking.initial_pitch_deg).

Calibration poses used by gabai.tools.frame_calibrate: lying on its back with the front (camera side) facing UP: gravity is +z in D; standing upright as worn on
the chest: gravity is -y in D (up = -y). Two such gravity directions measured in the IMU axes fix R_dev_imu (triad), and spinning the device about the
vertical in each pose lets the compass axes be matched to the IMU axes.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

UP_WORN = np.array([0.0, -1.0, 0.0])        # gravity's 'up' in the device frame when it hangs upright on the chest
UP_FRONT_UP = np.array([0.0, 0.0, 1.0])     # ... when it lies on its back with the camera side up


def nearest_rotation(M):
    """The closest proper rotation to M (a calibration file can hold a matrix that is a rotation only to a few digits)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, float))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def triad(up_front_up_imu, up_upright_imu):
    """R_dev_imu from two still gravity readings in the IMU axes: one with the device lying on its back (front up), one standing upright.
    The first gives the device z axis exactly (a table is flat). The second gives the y axis from its part perpendicular to z: leaning the device forward or
    back only changes the z part, so the perpendicular part is the device's vertical axis whatever the lean (a sideways lean is not allowed for).
    Limit: a SIDEWAYS lean of the upright pose (rolling about the forward axis) rotates that perpendicular part and goes straight into the result as an error of
    the same size about z; gravity alone cannot see it. Hold the device left-right level; the cross-check against imu_sync.json in load_frames shows it."""
    zb = np.asarray(up_front_up_imu, float)
    zb = zb / np.linalg.norm(zb)
    ua = np.asarray(up_upright_imu, float)
    ua = ua / np.linalg.norm(ua)
    perp = ua - (ua @ zb) * zb
    n = np.linalg.norm(perp)
    if n < 0.2:
        raise ValueError("the two poses give nearly the same gravity direction: lay it on its back, then stand it up, so the readings differ by about 90 degrees")
    y_up = perp / n                                          # 'up' axis in the plane perpendicular to z
    y_dev = -y_up                                            # device y is down
    x_dev = np.cross(y_dev, zb)                              # right-handed: x = y cross z
    R = np.stack([x_dev, y_dev, zb])                         # rows: device axes expressed in IMU axes, so v_dev = R @ v_imu
    return R


def classify_mount(up_dev, tol_deg=25.0):
    """Which pose is the device in, from gravity's 'up' measured in the device frame. Returns (name, details).
    names: chest_upright, flat_front_up, flat_front_down, upside_down, on_its_side, tilted. details: pitch_down_deg (how far the camera looks below the
    horizon, negative = above), roll_deg (sideways lean), tilt_from_pose_deg (distance from the named pose)."""
    u = np.asarray(up_dev, float)
    u = u / np.linalg.norm(u)
    ang = lambda a: float(np.degrees(np.arccos(np.clip(u @ a, -1, 1))))
    pitch_down = float(-np.degrees(np.arcsin(np.clip(u[2], -1, 1))))                # forward axis above the horizon when u.z > 0
    roll = float(np.degrees(np.arctan2(u[0], -u[1])))
    poses = {"chest_upright": np.array([0, -1.0, 0]), "flat_front_up": np.array([0, 0, 1.0]), "flat_front_down": np.array([0, 0, -1.0]),
             "upside_down": np.array([0, 1.0, 0]), "on_its_side": None}
    best = min(((n, ang(a)) for n, a in poses.items() if a is not None), key=lambda x: x[1])
    side = min(ang(np.array([1.0, 0, 0])), ang(np.array([-1.0, 0, 0])))
    if side < best[1]:
        best = ("on_its_side", side)
    name = best[0] if best[1] <= tol_deg else "tilted"
    return name, {"pitch_down_deg": pitch_down, "roll_deg": roll, "tilt_from_pose_deg": best[1]}


@dataclass
class Frames:
    R_dev_imu: np.ndarray = field(default_factory=lambda: np.eye(3))
    R_imu_mag: np.ndarray = field(default_factory=lambda: np.eye(3))
    sources: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def R_dev_mag(self):
        return self.R_dev_imu @ self.R_imu_mag

    @property
    def R_imu_dev(self):
        return self.R_dev_imu.T

    def forward_in_imu(self):
        """The device's forward axis (camera z) written in IMU axes: the axis whose heading is the device's heading."""
        return self.R_dev_imu.T @ np.array([0.0, 0.0, 1.0])

    def up_in_device(self, up_imu):
        u = np.asarray(up_imu, float)
        return self.R_dev_imu @ (u / np.linalg.norm(u))

    def mount(self, up_imu):
        return classify_mount(self.up_in_device(up_imu))


def load_frames(imu_sync="data/imu_sync.json", mag_sync="data/mag_sync.json", device="data/device_frame.json"):
    """The device <- IMU and IMU <- compass rotations from their files. Order for D <- I: data/device_frame.json made by frame_calibrate (from gravity in two
    known poses), then data/imu_sync.json (from the camera and gyro). When both exist they are compared and a disagreement over 10 degrees is noted."""
    f = Frames()
    from_cam = from_grav = None
    if os.path.exists(imu_sync):
        from_cam = nearest_rotation(json.load(open(imu_sync))["R_ci"])
    if os.path.exists(device):
        from_grav = nearest_rotation(json.load(open(device))["R_dev_imu"])
    if from_cam is not None and from_grav is not None:
        a = float(np.degrees(np.arccos(np.clip((np.trace(from_cam.T @ from_grav) - 1) / 2, -1, 1))))
        f.notes.append(f"camera-gyro axis map and gravity-pose axis map differ by {a:.1f} deg" + ("  <- check which one is right (py -m gabai.tools.frame_check)" if a > 10 else ""))
    f.R_dev_imu, f.sources["imu_to_device"] = (from_cam, imu_sync) if from_cam is not None else ((from_grav, device) if from_grav is not None else (np.eye(3), "none (identity)"))
    if from_cam is None and from_grav is None:
        f.notes.append("no IMU-to-camera axis map: run py -m gabai.tools.frame_calibrate (or sync_imu_camera)")
    if os.path.exists(mag_sync):
        f.R_imu_mag = nearest_rotation(json.load(open(mag_sync))["R_im"])
        f.sources["compass_to_imu"] = mag_sync
    else:
        f.sources["compass_to_imu"] = "none (identity)"
        f.notes.append("no compass axis map: run py -m gabai.tools.frame_calibrate (or sync_sensors)")
    return f

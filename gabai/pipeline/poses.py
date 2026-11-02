"""Camera pose providers. Both modes give the same PoseResult, so the map and the Rerun views do not change when the mode changes.

PoseResult: t_ns, R_wc, t_wc (camera pose in the world; camera axes x right, y down, z forward; world z up), ok, info, source.

StereoOnlyProvider  our own stereo visual odometry (gabai.kfvo.KeyframeVO, or the original frame-to-frame StereoVO). Reads no IMU. World frame = the camera's first pose with
                    z up ASSUMED (the camera is taken to start upright), so a camera that starts tilted gives a tilted world. Drifts; no loop closure.
VisualInertialProvider (gabai/pipeline/vio.py, mode stereo_imu)  the same visual odometry plus the IMU: gyro prior and bias, accelerometer gravity alignment, rest hold.
                    Independent of StereoOnlyProvider: the two modes share only KeyframeVO and PoseResult.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.diag import log as dlog

log = dlog.get("pose")


@dataclass
class PoseResult:
    t_ns: int
    R_wc: object
    t_wc: object
    ok: bool
    source: str
    info: dict = field(default_factory=dict)


class StereoOnlyProvider:
    name, uses_imu = "stereo_only", False
    world = "first camera pose, z up assumed (no gravity sensor in this mode)"

    def __init__(self, cal, vo="keyframe", fuse_depth=True, R_ic=None, **kfvo_options):
        """R_ic: rotation of the camera axes in the IMU axes. With it, and IMU samples fed in, the gyro rotation between two frames starts the feature search where the picture
        really went (a prior only: the pose still comes from the pictures, and a gross disagreement is the only thing that holds it). Without it nothing here reads an IMU."""
        self.R_ic = None if R_ic is None or vo != "keyframe" else np.asarray(R_ic, float)
        self.wants_gyro = self.R_ic is not None
        self._rot, self._t_imu, self._n_imu, self._t_prev = R.identity(), None, 0, None
        if vo == "keyframe":
            from gabai.kfvo import KeyframeVO
            self.vo = KeyframeVO(cal["f"], cal["f"], cal["cx"], cal["cy"], fuse_depth=fuse_depth, **kfvo_options)
        else:
            from gabai.tools.video_cloud_view import StereoVO
            self.vo = StereoVO(cal["f"], cal["f"], cal["cx"], cal["cy"])
        self.vo.set_baseline(cal["B"])
        self.can_correct = hasattr(self.vo, "correct")

    def correct(self, R_wc, t_wc):
        """The map refiner moved the pose: the odometry continues from the corrected pose."""
        if self.can_correct:
            self.vo.correct(R_wc, t_wc)

    def feed_imu(self, samples):
        """(t s, gyro rad/s, accel) rows in IMU axes, oldest first. Without R_ic they are ignored: this mode does not depend on the IMU."""
        if not self.wants_gyro:
            return
        for t_s, gyr, _ in samples:
            if self._t_imu is not None and 0 < t_s - self._t_imu < 0.1:
                self._rot = self._rot * R.from_rotvec(np.asarray(gyr, float) * (t_s - self._t_imu))
                self._n_imu += 1
            self._t_imu = t_s

    def push(self, t_ns, gray_left, gray_right, disp, trusted, dR_prior=None):
        """dR_prior: camera rotation since the previous frame from an outside sensor; when none is given and a gyro was fed, it is the gyro's, turned into camera axes."""
        kw = {}
        if hasattr(self.vo, "vel_s"):
            kw["frame_s"] = None if self._t_prev is None else (t_ns - self._t_prev) * 1e-9
        if dR_prior is None and self.wants_gyro and self._n_imu:
            dR_prior = self.R_ic.T @ self._rot.as_matrix() @ self.R_ic
        self._rot, self._n_imu, self._t_prev = R.identity(), 0, t_ns
        if dR_prior is not None:
            kw["dR_prior"] = dR_prior
        Rm, tm, info = self.vo.step(gray_left, disp, trusted, **kw)
        if not info["ok"]:
            log.warning_every("vo_lost", 5.0, "pose.vo_held", reason=info["reason"], inliers=info["inliers"], tracked=info["tracked"])
        return [PoseResult(t_ns, Rm, tm, info["ok"], self.name, info)]

    def close(self):
        pass


def camera_in_imu(folder):
    """Rotation of the camera axes in the IMU axes from folder/calib.json (the file Basalt reads), or None when there is none."""
    p = os.path.join(folder or "", "calib.json")
    if not os.path.exists(p):
        return None
    T0 = json.load(open(p))["value0"]["T_imu_cam"][0]
    return R.from_quat([T0["qx"], T0["qy"], T0["qz"], T0["qw"]]).as_matrix()


def make_provider(cfg, cal):
    """The one place the mode decides which estimator is built. stereo_only opens no IMU file."""
    t = cfg.tracking
    opts = {} if t.vo != "keyframe" else dict(predict=t.predict, refine=t.refine, relocalize=t.relocalize, initial_pitch_deg=t.initial_pitch_deg, window_ba=t.window_ba, ba_every=t.ba_every, ba_window=t.ba_window, ba_max_landmarks=t.ba_max_landmarks, loop_closure=t.loop_closure, loop_apply=t.loop_apply and t.loop_closure,
                                               loop_radius_m=t.loop_radius_m, loop_min_path_m=t.loop_min_path_m, loop_appearance=t.loop_appearance, loop_every=t.loop_every, loop_max_drift_m=t.loop_max_drift_m)
    if cfg.pipeline.mode == "stereo_imu":
        from gabai.pipeline.vio import VisualInertialProvider
        R_ic = camera_in_imu(cfg.source.folder)
        if R_ic is None:
            raise RuntimeError(f"stereo_imu needs the camera-IMU rotation: {cfg.source.folder}/calib.json not found (run gabai.tools.sync_imu_camera, then check_extrinsics --write)")
        return VisualInertialProvider(cal, R_ic, t.fuse_depth, klt=None if cfg.filter.enabled else cfg.klt, filt=cfg.filter, **opts)
    R_ic = None
    if t.gyro_prior and t.vo == "keyframe" and (cfg.source.kind != "live" or os.path.exists(cfg.source.imu_sync)):       # a live rig with no saved camera-IMU calibration has no usable rotation
        R_ic = camera_in_imu(cfg.source.folder)
        if R_ic is not None:
            opts["prior_gate_deg"] = 25.0                                    # the gyro only steers the search here: only a gross disagreement (wrong axes, no sync) may hold the pose
    return StereoOnlyProvider(cal, t.vo, t.fuse_depth, R_ic=R_ic, **opts)

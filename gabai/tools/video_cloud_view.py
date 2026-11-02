"""Point cloud and camera path from two plain video files (no IMU, no sensors), shown in Rerun.

    py -m gabai.tools.video_cloud_view --left data/left.mp4 --right data/right.mp4 --start 30 --max-frames 300 --skip 2
    py -m gabai.tools.video_cloud_view --max-frames 0            # the whole video (slow: about 2 frames per second of processing)

There is no IMU here, so Basalt cannot run. The camera path comes from a SIMPLE stereo visual odometry written for this tool (StereoVO): corners
are tracked from frame to frame with optical flow, each corner has a 3D position from the stereo depth, and the camera motion is the
rigid transform that best explains the move (OpenCV solvePnPRansac). It has no IMU, no loop closure, no map optimisation. It drifts, it can
lose tracking (the pose is then held and the frame is flagged), and it is not Basalt.

The video has no calibration, so focal length and baseline are ASSUMED (--f-frac, --baseline). The scale of the whole path and map is therefore
unknown, and the shape is only as good as that guess. Treat what you see as a way to look at the footage, not a measurement.
"""
import argparse
import os
import time

import cv2
import numpy as np

from gabai.sim import look_pose
from gabai.tools.slam_cloud_view import RerunSink, SlamCloud

MIN_INLIERS = 15
MIN_FEATURES = 25
DEPTH_RANGE = (0.3, 8.0)


class StereoVO:
    def __init__(self, fx, fy, cx, cy, max_corners=400):
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        self.fx, self.fy, self.cx, self.cy, self.max_corners = fx, fy, cx, cy, max_corners
        R0, t0 = look_pose(np.zeros(3))                      # start at the origin looking along +x, z up (world frame of the view)
        self.R_wc, self.t_wc = R0, t0
        self.prev_gray, self.pts2d, self.pts3d, self.n = None, None, None, 0
        self.last_flow = None                                # (points before, points after) of the features that agreed on the last motion

    def _seed(self, gray, disp, trusted):
        c = cv2.goodFeaturesToTrack(gray, self.max_corners, 0.01, 10)
        if c is None:
            return None, None
        c = c.reshape(-1, 2)
        u, v = c[:, 0].astype(int), c[:, 1].astype(int)
        d = disp[np.clip(v, 0, disp.shape[0] - 1), np.clip(u, 0, disp.shape[1] - 1)]
        ok = np.isfinite(d) & (d > 0.5) & trusted[np.clip(v, 0, trusted.shape[0] - 1), np.clip(u, 0, trusted.shape[1] - 1)]
        z = np.where(ok, self.fx * self._B / np.where(ok, d, 1), np.nan)
        ok &= (z > DEPTH_RANGE[0]) & (z < DEPTH_RANGE[1])
        c, z = c[ok], z[ok]
        return c.astype(np.float32), np.stack([(c[:, 0] - self.cx) / self.fx * z, (c[:, 1] - self.cy) / self.fy * z, z], 1)

    _B = 0.0507

    def set_baseline(self, B):
        self._B = B

    def step(self, gray, disp, trusted):
        """Returns (R_wc, t_wc, info). info["ok"] False = the pose was held, with the reason."""
        self.n += 1
        info = {"ok": False, "reason": "", "inliers": 0, "tracked": 0}
        if self.prev_gray is not None and self.pts2d is not None and len(self.pts2d) >= MIN_FEATURES:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.pts2d.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
            good = st.reshape(-1).astype(bool)
            info["tracked"] = int(good.sum())
            if good.sum() >= MIN_FEATURES:
                ok, rvec, tvec, inl = cv2.solvePnPRansac(self.pts3d[good].astype(np.float64), nxt.reshape(-1, 2)[good].astype(np.float64), self.K, None,
                                                         iterationsCount=200, reprojectionError=2.5, confidence=0.99, flags=cv2.SOLVEPNP_SQPNP)       # EPnP gave a wrong depth shift on near-planar scenes
                n_in = 0 if inl is None else len(inl)
                info["inliers"] = n_in
                if ok and n_in >= MIN_INLIERS:
                    sel = inl.reshape(-1)
                    self.last_flow = (self.pts2d[good][sel].copy(), nxt.reshape(-1, 2)[good][sel].copy())
                    Rm, _ = cv2.Rodrigues(rvec)                       # X_new = Rm X_prev + t: the previous camera seen from the new one
                    Rm_inv, t_inv = Rm.T, -Rm.T @ tvec.reshape(3)     # new camera in the previous camera's frame
                    self.t_wc = self.t_wc + self.R_wc @ t_inv
                    self.R_wc = self.R_wc @ Rm_inv
                    info["ok"] = True
                else:
                    info["reason"] = f"only {n_in} tracked features agree on a camera motion (need {MIN_INLIERS})"
            else:
                info["reason"] = f"only {int(good.sum())} features survived tracking (need {MIN_FEATURES})"
        elif self.prev_gray is not None:
            info["reason"] = "no features with trusted depth in the previous frame"
        else:
            info["ok"], info["reason"] = True, "first frame"
        self.prev_gray = gray
        self.pts2d, self.pts3d = self._seed(gray, disp, trusted)
        if self.pts2d is None or len(self.pts2d) < MIN_FEATURES:
            if info["ok"] and info["reason"] == "first frame":
                info["ok"], info["reason"] = False, "no features with trusted depth to start from"
            elif not info["reason"]:
                info["reason"] = "no features"
        return self.R_wc.copy(), self.t_wc.copy(), info


def main():
    from gabai.spike.demo_viewer import load_rectification, video_frames
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", default="data/left.mp4")
    ap.add_argument("--right", default="data/right.mp4")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--skip", type=int, default=2, help="use every Nth frame")
    ap.add_argument("--max-frames", type=int, default=300, help="0 = the whole video")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--f-frac", type=float, default=0.6, help="ASSUMED focal length as a fraction of image width")
    ap.add_argument("--baseline", type=float, default=0.065, help="ASSUMED camera spacing in metres")
    ap.add_argument("--model", default="s", choices=["s", "m", "h"])
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--no-semi-dense", action="store_true")
    a = ap.parse_args()
    from gabai.spike.las2_runner import Las2Depth
    las = Las2Depth(a.model)
    rect, *_ = load_rectification(a.left, a.right, a.width, os.path.join("data", f"autorect_{a.width}.npz"))
    vo = view = sink = None
    n = lost = 0
    t0 = time.time()
    for label, t, l, r, f, B, cy in video_frames(a.left, a.right, a.width, a.start, a.skip, a.f_frac, a.baseline, rect):
        if vo is None:
            h, w = l.shape[:2]
            vo = StereoVO(f, f, w / 2, cy)
            vo.set_baseline(B)
            sink = RerunSink(f, w / 2, cy, (w, h), a.save, not a.no_viewer, t0_ns=int(a.start * 1e9))
            view = SlamCloud(f, B, w / 2, cy, sink=sink, semi_dense=not a.no_semi_dense)
            print(label, "| f=%.0f px, B=%.3f m assumed: scale unknown, no IMU, simple visual odometry (drifts)" % (f, B))
        gray = cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)
        disp, trusted, _ = las.disparity_lr_batched(cv2.cvtColor(l, cv2.COLOR_BGR2RGB), cv2.cvtColor(r, cv2.COLOR_BGR2RGB))
        disp = np.where(trusted, disp, np.nan)
        R_wc, t_wc, info = vo.step(gray, disp, trusted)
        sink.image(t, l)
        sink.scalar("vision/inliers", t, info["inliers"])
        if not info["ok"]:
            lost += 1
            sink.rr.log("events", sink.rr.TextLog(f"t={t:.1f}s pose held: {info['reason']}"))
            print(f"t={t:6.1f}s  pose held: {info['reason']}")
        view.on_pose(int(t * 1e9), R_wc, t_wc, gray, disp, trusted)
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
    path = float(np.linalg.norm(np.diff(np.array(view.traj), axis=0), axis=1).sum()) if view and len(view.traj) > 1 else 0.0
    print(f"{n} frames in {time.time() - t0:.0f} s; pose held on {lost}; path {path:.2f} (scale unknown); map voxels {len(view.map) if view else 0}")


if __name__ == "__main__":
    main()

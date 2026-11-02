"""Watch a recorded sequence being tracked and mapped: camera path, a semi-dense point cloud placed with the tracked pose, the IMU, and how well
the IMU and the visual-inertial tracker agree. Shown in Rerun; scrub the timeline or let it play.

    py -m gabai.tools.slam_cloud_view data/slam/rec_20261026_155811                   whole recording, as fast as it processes
    py -m gabai.tools.slam_cloud_view data/slam/rec_20261026_155811 --realtime        paced like the original (1x)
    py -m gabai.tools.slam_cloud_view data/slam/rec_20261026_155811 --basalt-ui       also open Basalt's own window (its real tracked points)
    py -m gabai.tools.slam_cloud_view data/slam/rec_20261026_155811 --save data/slam_view.rrd --no-viewer

What is on screen, and what each thing really is:
  world/trajectory, world/camera       Basalt's pose (position + orientation) of the camera, from running Basalt on the recording's IMU and pictures.
  world/live                           This frame's depth points (LAS2 stereo depth, left-right checked), placed in the world with that pose.
  world/map                            All frames fused into 5 cm voxels. If the pose is good the walls stay thin; if it drifts they smear.
  The "semi-dense" selection           Only pixels with strong image gradient are kept (edges and texture), as in semi-dense SLAM. Depth there is the
                                       stereo model's, not Basalt's: Basalt's own map points are not exposed by its C interface.
  camera/tracks                        Image feature tracks from OUR OWN optical-flow tracker (cv2 KLT). They show what a feature tracker sees in this
                                       footage; they are NOT Basalt's tracked landmarks. For Basalt's real tracked points use --basalt-ui.
  imu/*                                Gyro rate and accelerometer magnitude from the recording.
  imu/rotation_disagreement_deg        Between consecutive poses: the angle between the rotation Basalt reports and the rotation obtained by integrating the
                                       gyro (a constant gyro bias, learned from Basalt's own rotation, is removed first and plotted separately in imu/gyro_bias_estimate_dps). Small = the IMU and the visual-inertial estimate agree and are in sync.
                                       It shows consistency, not how Basalt weights the IMU inside its optimiser; that is not exposed.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.cloud import VoxelMap, depth_std, depth_weight, map_points, transform


def cam_pose(p_wi, q_wi, R_ic, t_ic):
    """Camera pose in the world from Basalt's IMU pose (position, xyzw quaternion) and the camera-in-IMU extrinsics."""
    R_wi = R.from_quat(q_wi).as_matrix()
    return R_wi @ np.asarray(R_ic, float), np.asarray(p_wi, float) + R_wi @ np.asarray(t_ic, float)


def gyro_residual_deg(t, w, bias, t0, t1, q0, q1):
    """Angle in degrees between Basalt's rotation from pose 0 to pose 1 and the rotation made by integrating the gyro over [t0, t1]."""
    t, w = np.asarray(t, float), np.asarray(w, float) - np.asarray(bias, float)
    sel = (t >= t0) & (t <= t1)
    if sel.sum() < 2:
        return float("nan")
    Rg = np.eye(3)
    ts, ws = t[sel], w[sel]
    for i in range(len(ts) - 1):
        Rg = Rg @ R.from_rotvec(0.5 * (ws[i] + ws[i + 1]) * (ts[i + 1] - ts[i])).as_matrix()
    Rv = R.from_quat(q0).inv() * R.from_quat(q1)
    return float(np.degrees((R.from_matrix(Rg).inv() * Rv).magnitude()))


def gyro_bias_sample(t, w, t0, t1, q0, q1):
    """What constant gyro bias (rad/s, in the IMU frame) would make the gyro agree with Basalt's rotation over [t0, t1]; None if too little data."""
    t, w = np.asarray(t, float), np.asarray(w, float)
    sel = (t >= t0) & (t <= t1)
    if sel.sum() < 2 or t1 <= t0:
        return None
    Rg = np.eye(3)
    ts, ws = t[sel], w[sel]
    for i in range(len(ts) - 1):
        Rg = Rg @ R.from_rotvec(0.5 * (ws[i] + ws[i + 1]) * (ts[i + 1] - ts[i])).as_matrix()
    rv_g = R.from_matrix(Rg).as_rotvec()
    rv_v = (R.from_quat(q0).inv() * R.from_quat(q1)).as_rotvec()
    return (rv_g - rv_v) / (ts[-1] - ts[0])


def semi_dense_mask(gray, thr=14.0):
    """Pixels with a strong local gradient (edges, texture) where stereo depth is most reliable; blank areas are dropped."""
    g = cv2.GaussianBlur(np.asarray(gray, np.float32), (3, 3), 0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 4.0
    return cv2.dilate((mag > thr).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


class ListSink:
    def __init__(self):
        self.n_frames, self.last = 0, None

    def log(self, t, **kw):
        self.n_frames += 1
        self.last = dict(kw, t=t)

    def scalar(self, name, t, v):
        pass


class SlamCloud:
    """Semi-dense stereo points fused into a voxel map that lasts as the camera moves, and that can be re-laid when the trajectory is corrected.

    Persistence. Voxels are kept until the camera is farther than `window_m` from them (`prune`), or until the camera itself passes through them (`carve_m`: a camera
    cannot be inside a solid, so such a voxel is a moved object or an early error). Nothing else removes a voxel except the hard cap `max_voxels`. When the pose and
    the map disagree for too long, `forget_near` drops the voxels around the camera, not the whole map.

    Re-laying. With an `anchor` (keyframe id and its pose, from the keyframe odometry) each frame also keeps a thinned copy of its points in the camera frame together
    with the camera's pose RELATIVE to that keyframe. A loop closure moves the keyframes; `rebuild(node_pose)` then puts every stored frame back at the corrected pose and
    fuses the voxel map again, so the revisited wall is one layer, not two. `max_frames` and the window bound that store the same way they bound the map.

    Change. A moved or removed thing must leave the map. `carve` projects the stored voxels into the current stereo depth (`carve_decay` < 1 turns it on): a voxel whose
    pixel, and every pixel around it, was measured farther away than the voxel by more than the depth noise and a pose allowance has been looked through, so it loses part
    of its evidence and goes when none is left. A voxel behind something nearer, or next to a hole, is not touched (it may still be there). New things need no code: they are
    integrated like anything else, and the voxels a moved object leaves behind are carved by the first looks past them.

    Mono fill. `fill` = (monocular disparity on the stereo scale, mask of the pixels it supplies) goes into a SECOND voxel layer, `fill_map`, with `fill_weight` of a stereo
    point's weight. `map` (the pose refiner, the alignment check, the saved map) never sees it, and `shown_points` adds a fill voxel only where no stereo voxel exists, so a
    stereo measurement is never moved or averaged with a guess. The fill is not thinned by the semi-dense gradient mask: a blank wall is exactly what stereo could not measure.
    """

    def __init__(self, f, B, cx, cy, sink=None, voxel=0.05, window_m=30.0, semi_dense=True, stride=3, max_range=4.0, map_every=5, weighted=True, max_depth_std=0.0,
                 refiner=None, core=0.2, core_faces_only=False, max_voxels=0, carve_m=0.0, fill_weight=0.3, fill_stride=None, max_frames=600, frame_points=1500, carve_decay=0.0):
        self.f, self.B, self.cx, self.cy, self.sink = f, B, cx, cy, sink
        self.weighted, self.max_depth_std, self.refiner, self.core, self.core_faces_only = weighted, max_depth_std, refiner, core, core_faces_only
        self.max_voxels, self.resets, self.carve_m, self.fill_weight = max_voxels, 0, carve_m, fill_weight
        self.fill_stride = fill_stride or stride + 1
        self.max_frames, self.frame_points, self.carve_decay, self.carved = max_frames, frame_points, carve_decay, 0
        self.pose, self.refine_info = None, None            # the pose the last frame was placed with (the input pose, or the map-corrected one)
        self.map, self.semi, self.stride, self.max_range = VoxelMap(voxel, window_m), semi_dense, stride, max_range
        self.fill_map = VoxelMap(voxel, window_m)           # points supplied by the monocular map; kept apart from the stereo measurements in `map`
        self.traj, self.n, self.map_every = [], 0, max(int(map_every), 1)
        self.frames = []                                    # per integrated frame: {"anchor", "R", "t" (camera in the anchor keyframe), "p"/"w" (stereo points in the camera frame), "pf"/"wf" (filled), "pos" (world)}
        self.rebuilds = 0
        self._track = []                                    # per anchored frame: [anchor id, camera position in that keyframe, world position]; the path, re-laid with the map
        self._rng = np.random.default_rng(0)

    def reset(self):
        """Forget the whole map (tracking and map disagree for too long, so adding more frames would only stack a second copy). The path is kept."""
        self.map = VoxelMap(self.map.voxel, self.map.window_m)
        self.fill_map = VoxelMap(self.map.voxel, self.map.window_m)
        self.frames = []
        self.resets += 1
        if self.refiner is not None:
            self.refiner.reset()

    def forget_near(self, center, radius):
        """Drop the map around `center` (and the stored frames taken there), keep the rest. What a long pose/map disagreement deserves: the part the camera is looking at
        is the part that is wrong, the part far away was built while the pose was still good."""
        n = self.map.clear_sphere(center, radius)
        self.fill_map.clear_sphere(center, radius)
        c = np.asarray(center, float)
        self.frames = [fr for fr in self.frames if np.linalg.norm(fr["pos"] - c) > radius]
        self.resets += 1
        if self.refiner is not None:
            self.refiner.reset()
        return n

    def forget_in_view(self, R_wc, t_wc, near_m=0.5):
        """forget_near for a camera that only turns: drop the part of the map inside the camera's view now (and the frames stored from about here), keep what is behind it."""
        n = self.map.clear_frustum(R_wc, t_wc, self.f, self.cx, self.cy, 1.2 * self.max_range)
        self.fill_map.clear_frustum(R_wc, t_wc, self.f, self.cx, self.cy, 1.2 * self.max_range)
        c = np.asarray(t_wc, float)
        self.frames = [fr for fr in self.frames if np.linalg.norm(fr["pos"] - c) > near_m]
        self.resets += 1
        if self.refiner is not None:
            self.refiner.reset()
        return n

    def view_points(self, left_gray, disp, trusted):
        """This frame's map points in the camera frame (what on_pose would add), for registering the frame to the map before deciding to add it."""
        excl = None if not self.semi else ~semi_dense_mask(left_gray)
        pts, _ = map_points(disp, trusted, self.f, self.B, self.cx, self.cy, max_range=self.max_range, exclude=excl, stride=self.stride, max_std_m=self.max_depth_std)
        return pts

    def _thin(self, p, w):
        if len(p) <= self.frame_points:
            return p.astype(np.float32), w.astype(np.float32)
        i = self._rng.choice(len(p), self.frame_points, replace=False)
        return p[i].astype(np.float32), w[i].astype(np.float32)

    def carve(self, R_wc, t_wc, disp, trusted, pose_slack_m=0.10, sigma_k=3.0):
        """Weaken the voxels (stereo and filled layers) that the current frame looked through; returns how many were removed. See the class note."""
        d = np.asarray(disp)
        valid = np.isfinite(d) & (d > 0.5) & np.asarray(trusted, bool)
        if not valid.any():
            return 0
        z_obs = np.where(valid, self.f * self.B / np.where(valid, d, 1.0), 0.0).astype(np.float32)
        z_min = cv2.erode(z_obs, np.ones((5, 5), np.uint8))                     # smallest depth in 5x5; a hole or a nearer pixel in it makes it 0 or small, so nothing is carved there
        h, w = d.shape
        removed = 0
        for m in (self.map, self.fill_map):
            if not len(m):
                continue
            rel = m.sum / m.w[:, None] - np.asarray(t_wc, float)
            near = np.flatnonzero((rel ** 2).sum(1) < (1.2 * self.max_range) ** 2)
            if not len(near):
                continue
            Xc = rel[near] @ np.asarray(R_wc, float)
            z = Xc[:, 2]
            ok = z > 0.3
            u, v = self.f * Xc[:, 0] / np.where(ok, z, 1.0) + self.cx, self.f * Xc[:, 1] / np.where(ok, z, 1.0) + self.cy
            ok &= (u >= 6) & (u < w - 6) & (v >= 6) & (v < h - 6)
            near, z, u, v = near[ok], z[ok], u[ok].astype(int), v[ok].astype(int)
            through = z_min[v, u] > z + pose_slack_m + sigma_k * depth_std(z, self.f, self.B)
            removed += m.contradict(near[through], self.carve_decay)
        self.carved += removed
        return removed

    def shown_points(self, min_obs=3):
        """The map as it is drawn: the stereo voxels, plus the filled voxels that no stereo voxel occupies."""
        p = self.map.points(min_obs=min_obs, core=self.core, faces_only=self.core_faces_only)
        if not len(self.fill_map):
            return p
        ok = self.fill_map.n >= min_obs
        ok &= ~np.isin(self.fill_map.keys, self.map.keys)
        return np.concatenate([p, self.fill_map.sum[ok] / self.fill_map.w[ok, None]]) if ok.any() else p

    def on_pose(self, t_ns, R_wc, t_wc, left_gray, disp, trusted, extra_exclude=None, fill=None, anchor=None):
        self.n += 1
        excl = None if not self.semi else ~semi_dense_mask(left_gray)
        if extra_exclude is not None:                      # e.g. hands, people, a held object: they move, so they must not become map
            excl = extra_exclude if excl is None else (excl | extra_exclude)
        pts, info = map_points(disp, trusted, self.f, self.B, self.cx, self.cy, max_range=self.max_range, exclude=excl, stride=self.stride,
                                max_std_m=self.max_depth_std)
        if self.carve_decay:
            info["carved"] = self.carve(R_wc, t_wc, disp, trusted)
        self.refine_info = None
        if self.refiner is not None:
            R_wc, t_wc, self.refine_info = self.refiner.refine(self.map, R_wc, t_wc, pts, self.n)
        self.pose = (R_wc, t_wc)
        self.traj.append(np.asarray(t_wc, float))
        w = depth_weight(pts[:, 2]) if self.weighted else np.ones(len(pts))
        pf = np.zeros((0, 3))
        if fill is not None:
            fd, fm = fill[0], fill[1]
            if extra_exclude is not None:
                fm = fm & ~extra_exclude
            both = np.where(fm, fd, np.where(np.isfinite(disp) & np.asarray(trusted, bool), disp, np.nan))          # stereo and filled together, so a jump between them is seen as an edge
            pf, _ = map_points(both, fm, self.f, self.B, self.cx, self.cy, max_range=self.max_range, stride=self.fill_stride, max_std_m=self.max_depth_std)
            info["filled_points"] = int(len(pf))
        wf = (depth_weight(pf[:, 2]) if self.weighted else np.ones(len(pf))) * self.fill_weight
        self.map.integrate(transform(pts, R_wc, t_wc), w)
        if len(pf):
            self.fill_map.integrate(transform(pf, R_wc, t_wc), wf)
        for m in (self.map, self.fill_map):
            m.prune(t_wc, m.window_m, self.carve_m)
            m.cap(self.max_voxels, t_wc)
        if anchor is not None and (len(pts) or len(pf)):
            kf_id, R_k, t_k = anchor
            R_k, t_k = np.asarray(R_k, float), np.asarray(t_k, float)
            p, wt = self._thin(pts, w)
            pff, wff = self._thin(pf, wf)
            self._track.append([kf_id, R_k.T @ (np.asarray(t_wc, float) - t_k), np.asarray(t_wc, float).copy()])
            self.frames.append({"anchor": kf_id, "R": R_k.T @ np.asarray(R_wc, float), "t": R_k.T @ (np.asarray(t_wc, float) - t_k), "p": p, "w": wt, "pf": pff, "wf": wff,
                                "pos": np.asarray(t_wc, float).copy()})
            self._prune_frames(t_wc)
        colors = None
        if len(pts):
            uv = np.clip(np.stack([pts[:, 0] / pts[:, 2] * self.f + self.cx, pts[:, 1] / pts[:, 2] * self.f + self.cy], 1).astype(int), [0, 0],
                         [left_gray.shape[1] - 1, left_gray.shape[0] - 1])
            g = left_gray[uv[:, 1], uv[:, 0]]
            colors = np.stack([g, g, g], 1)
        if self.sink is not None:
            self.sink.log(t_ns * 1e-9, R_wc=R_wc, t_wc=t_wc, live_points=transform(pts, R_wc, t_wc), live_colors=colors, trajectory=np.array(self.traj), info=info,
                          map_points=self.shown_points() if self.n % self.map_every == 0 else None)
        return info

    def _prune_frames(self, center):
        r = self.map.window_m
        if r:
            c = np.asarray(center, float)
            self.frames = [fr for fr in self.frames if np.linalg.norm(fr["pos"] - c) <= r]
        del self.frames[:-self.max_frames]

    def rebuild(self, node_pose):
        """Put every stored frame at the pose its keyframe has NOW and fuse the map again. node_pose(anchor id) -> (R_wk, t_wk), or None for a keyframe the odometry no
        longer holds (that frame is dropped). Returns (frames re-laid, frames dropped, largest move of a frame in metres)."""
        vm, fm, kept, moved, dropped = VoxelMap(self.map.voxel, self.map.window_m), VoxelMap(self.map.voxel, self.map.window_m), [], 0.0, 0
        for fr in self.frames:
            node = node_pose(fr["anchor"])
            if node is None:
                dropped += 1
                continue
            R_wk, t_wk = np.asarray(node[0], float), np.asarray(node[1], float)
            R_wc, t_wc = R_wk @ fr["R"], R_wk @ fr["t"] + t_wk
            moved = max(moved, float(np.linalg.norm(t_wc - fr["pos"])))
            fr["pos"] = t_wc
            vm.integrate(transform(fr["p"], R_wc, t_wc), fr["w"])
            fm.integrate(transform(fr["pf"], R_wc, t_wc), fr["wf"])
            kept.append(fr)
        for e in self._track:                                # the path moves with the keyframes it hangs on; a forgotten keyframe leaves its points where they last were
            node = node_pose(e[0])
            if node is not None:
                e[2] = np.asarray(node[0], float) @ e[1] + np.asarray(node[1], float)
        self.traj = [e[2] for e in self._track]
        self.frames, self.map, self.fill_map, self.rebuilds = kept, vm, fm, self.rebuilds + 1
        if self.refiner is not None:
            self.refiner.reset()
        return len(kept), dropped, moved


class RerunSink:
    def __init__(self, f, cx, cy, size, save=None, viewer=True, t0_ns=0):
        import rerun as rr
        self.rr, self.f, self.cx, self.cy, self.size, self.t0 = rr, f, cx, cy, size, t0_ns
        rr.init("gabai_slam_cloud", spawn=viewer)
        if save:
            rr.save(save)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world/camera", rr.Pinhole(focal_length=f, principal_point=(cx, cy), resolution=size, camera_xyz=rr.ViewCoordinates.RDF), static=True)

    def _t(self, t):
        self.rr.set_time("t", duration=float(t - self.t0 * 1e-9))

    def image(self, t, bgr, name="camera/image"):
        self._t(t)
        self.rr.log(name, self.rr.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=70))

    def scalar(self, name, t, v):
        if np.isfinite(v):
            self._t(t)
            self.rr.log(name, self.rr.Scalars(float(v)))

    def log(self, t, R_wc=None, t_wc=None, live_points=None, live_colors=None, trajectory=None, map_points=None, **_):
        rr = self.rr
        self._t(t)
        rr.log("world/camera", rr.Transform3D(translation=t_wc, mat3x3=R_wc))
        if trajectory is not None and len(trajectory) > 1:
            rr.log("world/trajectory", rr.LineStrips3D([trajectory], colors=[255, 200, 0], radii=0.01))
        if live_points is not None and len(live_points):
            rr.log("world/live", rr.Points3D(live_points, colors=live_colors, radii=0.008))
        if map_points is not None and len(map_points):
            rr.log("world/map", rr.Points3D(map_points, radii=0.012, colors=[110, 170, 255]))


def run(folder, model="s", stride=1, realtime=False, basalt_ui=False, save=None, viewer=True, max_frames=0, semi_dense=True, imu_every=4, no_gate=False):
    from gabai.slam.euroc import G
    from gabai.slam.still import StillGate
    from gabai.slam.vit import BasaltTracker
    from gabai.spike.las2_runner import Las2Depth
    from gabai.tools.live_basalt import replay_events, safe_toml_path, write_toml
    c = json.load(open(os.path.join(folder, "calib.json")))["value0"]
    k = c["intrinsics"][0]["intrinsics"]
    T0, T1 = c["T_imu_cam"][0], c["T_imu_cam"][1]
    R_ic = R.from_quat([T0["qx"], T0["qy"], T0["qz"], T0["qw"]]).as_matrix()
    t_ic = np.array([T0["px"], T0["py"], T0["pz"]])
    B = float(np.linalg.norm([T1["px"] - T0["px"], T1["py"] - T0["py"], T1["pz"] - T0["pz"]]))
    size = tuple(c["resolution"][0])
    first = next(l for l in open(os.path.join(folder, "mav0", "cam0", "data.csv")) if not l.startswith("#")).split(",")[0]
    t0_ns = int(first)
    sink = RerunSink(k["fx"], k["cx"], k["cy"], size, save, viewer, t0_ns)
    toml = safe_toml_path(os.path.abspath(folder))
    write_toml(toml, os.path.join(os.path.abspath(folder), "calib.json").replace("\\", "/"), show_ui=basalt_ui)
    tracker = BasaltTracker(toml, show_ui=basalt_ui)
    las = Las2Depth(model)
    view = SlamCloud(k["fx"], B, k["cx"], k["cy"], sink=sink, semi_dense=semi_dense)
    gate = StillGate(1.0)
    if no_gate or not _gate_opens(folder):
        gate = _OpenGate()
        print("NOTE: this recording never holds still long enough for the still-gate (gyro stays above 3 deg/s, probably a resting gyro bias): "
              "Basalt starts at once, so its first orientation and the run-to-run result are less reliable." if not no_gate else "still-gate off")
    imu_t, imu_w, bias, bias_samples = [], [], np.zeros(3), []
    pushed, prev, prev_gray, tracks = {}, None, None, None
    residuals, n_frames, n_imu, n_poses, t_start = [], 0, 0, 0, time.time()
    for kind, t_ns, x, y in replay_events(folder, 1.0 if realtime else 1e6):
        if kind == "imu":
            for ts, acc, gyr in gate.feed(t_ns, x, y):
                if not imu_t and gate.ready:
                    pass
                if tracker.push_imu(ts, acc, gyr):
                    n_imu += 1
                    imu_t.append(ts * 1e-9)
                    imu_w.append(np.asarray(gyr, float))
                    if n_imu % imu_every == 0:
                        sink.scalar("imu/gyro_rate_dps", ts * 1e-9, np.degrees(np.linalg.norm(gyr)))
                        sink.scalar("imu/accel_norm_g", ts * 1e-9, np.linalg.norm(acc) / G)
        else:
            if n_imu < 20 or not gate.ready:
                continue
            if tracker.push_frame(t_ns, x, y):
                n_frames += 1
                pushed[t_ns] = (x, y)
                bgr = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
                sink.image(t_ns * 1e-9, bgr)
                tr_img, n_tr = _klt(prev_gray, x, tracks)
                tracks = n_tr[0]
                if tr_img is not None:
                    sink.image(t_ns * 1e-9, tr_img, "camera/tracks")
                    sink.scalar("vision/klt_tracks", t_ns * 1e-9, n_tr[1])
                prev_gray = x
        while True:
            p = tracker.pop_pose()
            if p is None:
                break
            n_poses += 1
            pair = pushed.pop(p["t_ns"], None)
            for old in [t for t in pushed if t < p["t_ns"] - 2_000_000_000]:
                pushed.pop(old)
            R_wc, t_wc = cam_pose(p["p"], p["q"], R_ic, t_ic)
            if prev is not None:
                # The resting gyro bias is learned from Basalt's own rotation (running mean over the earlier intervals), so what is left in the
                # disagreement is noise and timing error, not a constant offset. The estimate itself is logged so the offset is still visible.
                if len(bias_samples) >= 20:
                    bias = np.mean(bias_samples, axis=0)
                    sink.scalar("imu/gyro_bias_estimate_dps", p["t_ns"] * 1e-9, np.degrees(np.linalg.norm(bias)))
                    r = gyro_residual_deg(imu_t, imu_w, bias, prev[0] * 1e-9, p["t_ns"] * 1e-9, prev[1], p["q"])
                else:
                    r = float("nan")
                bs = gyro_bias_sample(imu_t, imu_w, prev[0] * 1e-9, p["t_ns"] * 1e-9, prev[1], p["q"])
                if bs is not None:
                    bias_samples.append(bs)
                sink.scalar("imu/rotation_disagreement_deg", p["t_ns"] * 1e-9, r)
                dt = (p["t_ns"] - prev[0]) * 1e-9
                sink.scalar("vision/rotation_rate_dps", p["t_ns"] * 1e-9, np.degrees((R.from_quat(prev[1]).inv() * R.from_quat(p["q"])).magnitude()) / max(dt, 1e-6))
                if np.isfinite(r):
                    residuals.append(r)
            prev = (p["t_ns"], p["q"])
            if pair is not None and (n_poses % stride == 0):
                l, rr_ = pair
                disp, trusted, _ = las.disparity_lr_batched(cv2.cvtColor(l, cv2.COLOR_GRAY2RGB), cv2.cvtColor(rr_, cv2.COLOR_GRAY2RGB))
                view.on_pose(p["t_ns"], R_wc, t_wc, l, np.where(trusted, disp, np.nan), trusted)
        if max_frames and n_frames >= max_frames:
            break
    tracker.close()
    tr = np.array(view.traj) if view.traj else np.zeros((0, 3))
    path = float(np.linalg.norm(np.diff(tr, axis=0), axis=1).sum()) if len(tr) > 1 else 0.0
    s = {"folder": folder, "frames_pushed": n_frames, "poses": n_poses, "map_voxels": len(view.map), "path_m": round(path, 2),
         "trajectory_box_m": None if not len(tr) else np.round(tr.max(0) - tr.min(0), 2).tolist(), "map_voxels_evicted": view.map.evicted, "seconds": round(time.time() - t_start, 1),
         "rotation_disagreement_deg": None if not residuals else {"median": float(np.median(residuals)), "p95": float(np.percentile(residuals, 95))},
         "note": "KLT tracks are this tool's own, not Basalt's landmarks; the disagreement is consistency between gyro and Basalt, not accuracy"}
    return s


class _OpenGate:
    """Stands in for the still-gate when it is off: every sample passes straight through."""
    ready = True

    def feed(self, t_ns, accel, gyro):
        return [(int(t_ns), np.asarray(accel, float), np.asarray(gyro, float))]


def _gate_opens(folder):
    from gabai.slam.still import StillGate
    d = np.loadtxt(os.path.join(folder, "mav0", "imu0", "data.csv"), delimiter=",", comments="#")
    g = StillGate(1.0)
    return any(g.feed(int(r[0]), r[4:7], r[1:4]) and g.ready for r in d)


def _klt(prev_gray, gray, pts):
    """Our own feature tracker (Shi-Tomasi corners + pyramidal Lucas-Kanade), drawn on the picture. Not Basalt's landmarks."""
    if prev_gray is None or pts is None or len(pts) < 60:
        pts = cv2.goodFeaturesToTrack(gray, 300, 0.01, 12)
        return None, (pts, 0 if pts is None else len(pts))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.reshape(-1).astype(bool)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for a, b in zip(pts[ok].reshape(-1, 2), nxt[ok].reshape(-1, 2)):
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 255), 1)
        cv2.circle(img, tuple(int(v) for v in b), 2, (0, 200, 0), -1)
    return img, (nxt[ok].reshape(-1, 1, 2), int(ok.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--model", default="s", choices=["s", "m", "h"])
    ap.add_argument("--stride", type=int, default=1, help="make depth for every Nth pose (1 = all)")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--basalt-ui", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-still-gate", action="store_true", help="start Basalt at once (done automatically when the recording never holds still)")
    ap.add_argument("--no-semi-dense", action="store_true", help="keep every trusted pixel instead of only textured ones")
    a = ap.parse_args()
    s = run(a.folder, a.model, a.stride, a.realtime, a.basalt_ui, a.save, not a.no_viewer, a.max_frames, not a.no_semi_dense, no_gate=a.no_still_gate)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()

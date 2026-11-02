"""Simple live stereo SLAM: LAS2 depth + keyframe visual odometry + a small rolling map, all in Rerun. Nothing is stored: no point cloud on disk, no recording.

Per picture pair: LAS2 disparity (left-right checked) -> gabai.kfvo.KeyframeVO pose (features seeded once per keyframe with a metric 3D position from the depth, tracked with LK, RANSAC PnP,
motion-only refinement, its own relocalisation and keyframe pose graph) -> trusted depth pixels placed in the world with that pose and merged into a voxel map.

The map is bounded: at most --max-voxels voxels (default 60000, about 2 MB); when it is full the voxels farthest from the camera are dropped, so it follows you like a local map and its size never
grows. A voxel is drawn once it has been seen --min-hits times. A pair whose pose could not be solved is never added (a lost track cannot smear geometry).

Live from the Pi (the sender running, same layout and calibration as the main pipeline, config/pipeline.toml):
    py -m gabai.tools.simple_slam
From a recording written by record_pairs:
    py -m gabai.tools.simple_slam --replay data/recordings/vio_XXXX

What it is not: no loop closure on the map (the voxels are placed once with the pose of their time; when the keyframe graph later corrects the path the old voxels stay where they were),
no IMU, no persistent map. For the full system use `py -m gabai.pipeline.run`.
"""
import argparse
import time

import cv2
import numpy as np

KEY_BITS = 21
KEY_OFF = 1 << (KEY_BITS - 1)


class VoxelMap:
    """Bounded voxel map: per voxel the running mean of position and colour and the number of hits."""

    def __init__(self, voxel=0.04, max_voxels=60000, min_hits=2):
        self.v, self.max, self.min_hits = voxel, max_voxels, min_hits
        self.keys = np.empty(0, np.int64)
        self.sp = np.empty((0, 3), np.float64)          # sum of positions
        self.sc = np.empty((0, 3), np.float64)          # sum of colours
        self.n = np.empty(0, np.int32)

    def _key(self, p):
        g = np.floor(p / self.v).astype(np.int64) + KEY_OFF
        return (g[:, 0] << (2 * KEY_BITS)) | (g[:, 1] << KEY_BITS) | g[:, 2]

    def add(self, pts, cols, cam):
        """pts (N,3) world metres, cols (N,3) uint8. cam: the camera position, to drop the far voxels when full."""
        if len(pts) == 0:
            return
        k = self._key(pts)
        u, inv = np.unique(k, return_inverse=True)
        cnt = np.bincount(inv).astype(np.int32)
        sp = np.stack([np.bincount(inv, pts[:, i]) for i in range(3)], 1)
        sc = np.stack([np.bincount(inv, cols[:, i].astype(np.float64)) for i in range(3)], 1)
        allk = np.concatenate([self.keys, u])
        uk, inv2 = np.unique(allk, return_inverse=True)
        n = np.bincount(inv2, np.concatenate([self.n, cnt])).astype(np.int32)
        self.sp = np.stack([np.bincount(inv2, np.concatenate([self.sp[:, i], sp[:, i]])) for i in range(3)], 1)
        self.sc = np.stack([np.bincount(inv2, np.concatenate([self.sc[:, i], sc[:, i]])) for i in range(3)], 1)
        self.keys, self.n = uk, n
        if len(self.keys) > self.max:
            d = np.linalg.norm(self.sp / self.n[:, None] - cam, axis=1)
            keep = np.sort(np.argpartition(d, self.max)[:self.max])
            self.keys, self.sp, self.sc, self.n = self.keys[keep], self.sp[keep], self.sc[keep], self.n[keep]

    def points(self):
        m = self.n >= self.min_hits
        return (self.sp[m] / self.n[m, None]).astype(np.float32), np.clip(self.sc[m] / self.n[m, None], 0, 255).astype(np.uint8)

    def __len__(self):
        return len(self.keys)


def depth_points(left_bgr, disp, trusted, cal, stride, zmin, zmax):
    """Camera-frame points (N,3) and colours (N,3, RGB) of every `stride`-th trusted pixel between zmin and zmax metres."""
    ys, xs = np.mgrid[0:disp.shape[0]:stride, 0:disp.shape[1]:stride]
    d = disp[ys, xs]
    ok = trusted[ys, xs] & np.isfinite(d) & (d > 0.5)
    z = np.where(ok, cal["f"] * cal["B"] / np.where(ok, d, 1.0), 0)
    ok &= (z >= zmin) & (z <= zmax)
    z, x, y = z[ok], xs[ok], ys[ok]
    pts = np.stack([(x - cal["cx"]) * z / cal["f"], (y - cal["cy"]) * z / cal["f"], z], 1)
    return pts, left_bgr[ys[ok], xs[ok], ::-1]


def blueprint():
    import rerun.blueprint as rrb
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="map and path"),
            rrb.Vertical(rrb.Spatial2DView(origin="camera/left", name="left picture"), rrb.Spatial2DView(origin="camera/depth", name="LAS2 depth"),
                         rrb.TimeSeriesView(origin="stats", name="stats"), row_shares=[3, 3, 2]),
            column_shares=[3, 2]),
        rrb.TimePanel(timeline="frame"))


def run(a):
    import rerun as rr
    from scipy.spatial.transform import Rotation
    from gabai.kfvo import KeyframeVO
    from gabai.pipeline.config import PipelineConfig
    from gabai.pipeline.sources import open_source
    from gabai.spike.las2_runner import Las2Depth

    cfg = PipelineConfig.load(a.config)
    cfg.source.kind, cfg.source.max_frames, cfg.source.skip = ("replay" if a.replay else "live"), a.max_frames, 1
    if a.replay:
        cfg.source.folder = a.replay
    if a.no_flip180:
        cfg.source.flip180 = False
    print("loading LAS2 ...", flush=True)
    las = Las2Depth(cfg.stereo.model)
    rr.init("gabai_simple_slam", spawn=not a.no_viewer)
    if a.save_rrd:
        rr.save(a.save_rrd)
    rr.send_blueprint(blueprint())
    rr.log("world", rr.ViewCoordinates.RDF, static=True)              # the world is the first camera pose: x right, y down, z forward

    vo, vmap, path, last_draw = None, VoxelMap(a.voxel, a.max_voxels, a.min_hits), [], 0.0
    try:
        _loop(a, cfg, las, vo, vmap, path, last_draw, rr, Rotation, KeyframeVO, open_source)
    finally:
        if a.save_traj and path:
            np.savetxt(a.save_traj, np.asarray(path), fmt="%.5f", header="x y z (m), world = first camera pose")
            print(f"path ({len(path)} positions) saved to {a.save_traj}")
        print(f"done: {len(path)} poses used, {len(vmap)} voxels held")


def _loop(a, cfg, las, vo, vmap, path, last_draw, rr, Rotation, KeyframeVO, open_source):
    t_start = tail = None
    lost = 0
    for fr in open_source(cfg):
        cal = fr.cal
        if vo is None:
            vo = KeyframeVO(cal["f"], cal["f"], cal["cx"], cal["cy"], fuse_depth=True)
            vo.set_baseline(cal["B"])
        t0 = time.perf_counter()
        d, trusted, dt = las.disparity_lr_batched(cv2.cvtColor(fr.left, cv2.COLOR_BGR2RGB), cv2.cvtColor(fr.right, cv2.COLOR_BGR2RGB))
        disp = np.where(trusted, d, np.nan).astype(np.float32)
        gl = cv2.cvtColor(fr.left, cv2.COLOR_BGR2GRAY)
        R, t, info = vo.step(gl, disp, trusted, frame_s=None if tail is None else fr.t - tail)
        tail = fr.t
        t_start = fr.t if t_start is None else t_start
        ok = bool(info["ok"])
        lost = 0 if ok else lost + 1
        if ok:
            pts, cols = depth_points(fr.left, disp, trusted, cal, a.stride, a.min_z, a.max_z)
            if len(pts):
                vmap.add(pts @ np.asarray(R).T + np.asarray(t).ravel(), cols, np.asarray(t).ravel())
            path.append(np.asarray(t, np.float32).ravel())
        ms = (time.perf_counter() - t0) * 1000.0
        rr.set_time("frame", sequence=fr.idx)
        rr.set_time("t", duration=fr.t - t_start)
        q = Rotation.from_matrix(R).as_quat()
        rr.log("world/camera", rr.Transform3D(translation=np.asarray(t).ravel(), mat3x3=np.asarray(R)))
        rr.log("world/camera/image", rr.Pinhole(focal_length=cal["f"], principal_point=[cal["cx"], cal["cy"]], resolution=[fr.left.shape[1], fr.left.shape[0]], image_plane_distance=0.15))
        rr.log("camera/left", rr.Image(fr.left[:, :, ::-1]).compress(jpeg_quality=75))
        z = np.where(trusted & (d > 0.5), cal["f"] * cal["B"] / np.maximum(d, 0.5), 0)
        col = cv2.applyColorMap(np.clip(z * 255.0 / 5.0, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        col[z <= 0] = (40, 40, 40)
        rr.log("camera/depth", rr.Image(col[:, :, ::-1]).compress(jpeg_quality=70))
        if len(path) > 1:
            rr.log("world/path", rr.LineStrips3D([np.asarray(path[-a.path_len:])], colors=[(255, 200, 40)], radii=0.004))
        now = time.perf_counter()
        if now - last_draw >= a.map_every_s and len(vmap):
            p, c = vmap.points()
            rr.log("world/map", rr.Points3D(p, colors=c, radii=a.voxel * 0.5))
            last_draw = now
        rr.log("stats/frame_ms", rr.Scalars(ms))
        rr.log("stats/las2_ms", rr.Scalars(dt * 1000.0))
        rr.log("stats/tracked", rr.Scalars(float(info.get("tracked", 0))))
        rr.log("stats/map_voxels", rr.Scalars(float(len(vmap))))
        rr.log("stats/pose_ok", rr.Scalars(1.0 if ok else 0.0))
        if not ok and lost in (1, 30):
            print(f"pose held ({info.get('reason', '')}): {lost} pair(s) not used", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", help="folder written by record_pairs, instead of the live Pi stream")
    ap.add_argument("--config", default="config/pipeline.toml", help="calibration, flip180 and stereo model are read from here")
    ap.add_argument("--no-flip180", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N pairs (0 = until Ctrl+C)")
    ap.add_argument("--voxel", type=float, default=0.04, help="map voxel size, metres")
    ap.add_argument("--max-voxels", type=int, default=60000, help="the map never holds more; the voxels farthest from the camera are dropped first")
    ap.add_argument("--min-hits", type=int, default=2, help="a voxel is drawn once it was seen this many times")
    ap.add_argument("--stride", type=int, default=6, help="use every Nth pixel of the depth map")
    ap.add_argument("--min-z", type=float, default=0.3)
    ap.add_argument("--max-z", type=float, default=4.0, help="depth beyond this is too noisy for a map (m)")
    ap.add_argument("--map-every-s", type=float, default=0.5, help="how often the map is redrawn in Rerun")
    ap.add_argument("--path-len", type=int, default=3000, help="positions of the path drawn")
    ap.add_argument("--save-traj", help="optional: write the path (x y z per line) when it ends; the point map is never saved")
    ap.add_argument("--save-rrd", help="optional: record the Rerun session to a file")
    ap.add_argument("--no-viewer", action="store_true")
    try:
        run(ap.parse_args(argv))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()

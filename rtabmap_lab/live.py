"""Runs the RTAB-Map experiment: stereo pairs in, RTAB-Map's own odometry / depth / loop closure / map, a Rerun dashboard out.

Nothing of the GabAi pipeline's estimators is used: no IMU, no KeyframeVO, no Depth Anything, no LAS2. Only its plumbing is reused read-only, so the pictures are exactly as
good as in the main system: the Pi receiver, the stereo calibration, rectification, the left/right layout check and the 180 deg turn (gabai.pipeline.sources).

  py -m rtabmap_lab live                               the Pi's stream (plain sender; leave its IMU on so the start-up orientation check can run, RTAB-Map never uses it)
  py -m rtabmap_lab replay --folder data/recordings/vio_20261101_014854
  py -m rtabmap_lab live --db rtabmap_lab/maps/office.db           continue that map (it is shown again at start)
  py -m rtabmap_lab live --db rtabmap_lab/maps/office.db --localize     only find the camera in that map, do not add to it
Ctrl+C stops it and saves the map. py -m rtabmap_lab stop  kills a bridge that was left running.
"""
import argparse
import os
import time

import cv2
import numpy as np

from rtabmap_lab.bridge_client import Bridge, LAB
from rtabmap_lab import orientation
from rtabmap_lab.dashboard import Dashboard
from rtabmap_lab.surfel import SurfelFusion
from rtabmap_lab.tsdf import TsdfMap

DEFAULT_CONFIG = os.path.join(LAB, "config", "rtabmap_stereo.ini")


def frames(a):
    """Rectified pairs from the existing pipeline plumbing (read-only use). Yields gabai.pipeline.sources.Frame."""
    from gabai.pipeline.config import PipelineConfig
    from gabai.pipeline.sources import live_source, prepare_replay, replay_source
    from gabai.tools.calibrate_stereo import Rectifier
    ov = {"source.kind": "live" if a.source == "live" else "replay", "source.skew_comp": False, "source.flip180": False, "klt.enabled": False, "filter.enabled": False}
    if a.source == "live":
        ov["source.port"] = a.port
    else:
        ov.update({"source.folder": a.folder, "source.max_frames": 0})
    if a.calib:
        ov["source.calib"] = a.calib
    elif a.source == "replay":                                                # a recording is only rectified right by the calibration it was made with (the cameras move in their mounts between sessions)
        import json
        rc = json.load(open(os.path.join(a.folder, "meta.json"))).get("calibration", {}).get("file")
        if rc and os.path.exists(rc):
            ov["source.calib"] = rc
    cfg = PipelineConfig.load(a.pipeline_config, ov)
    if a.source == "live":
        rect = Rectifier(cfg.source.calib)
        return cfg, live_source(cfg, rect, 0.0, min(1.0, cfg.source.live_width / rect.size[0]))
    rect, delay, k = prepare_replay(cfg)
    return cfg, replay_source(cfg, rect, delay, k)


def run(a):
    from gabai.diag import log as dlog
    dlog.configure(run_name="rtabmap_lab", level="WARNING", console_level="ERROR")
    db = a.db
    if not db and not a.no_db:
        os.makedirs(os.path.join(LAB, "maps"), exist_ok=True)
        db = os.path.join(LAB, "maps", time.strftime("session_%Y%m%d_%H%M%S.db"))
    if a.localize and not (db and os.path.exists(db)):
        raise SystemExit("--localize needs an existing map: --db path.db")
    cfg, source = frames(a)
    flip, judged = a.orientation == "flip", False
    if a.orientation == "auto" and a.source == "replay":
        v = orientation.from_recording(a.folder, imu_sync=cfg.source.imu_sync)
        flip, judged = v["decision"] == "flip", True
        print("orientation check (accelerometer as an outside judge only):", v["reason"], flush=True)
    imu_rows, gyro_rows, R_ic = [], [], orientation.load_R_ic(cfg.source.imu_sync)
    from rtabmap_lab.rolling_shutter import ShutterCorrection
    shutter = ShutterCorrection.load(a.shutter_file) if a.shutter_fix else None
    if a.shutter_fix:
        print(f"rolling-shutter correction (gyro): {'from ' + a.shutter_file if shutter else 'off, ' + a.shutter_file + ' not found'}", flush=True)
    rec_imu, rec_t = None, -np.inf                                         # replay: the recording's imu.csv (the replay source gives no IMU rows); same clock as t_capture
    if a.source == "replay" and os.path.exists(os.path.join(a.folder, "imu.csv")):
        rec_imu = np.loadtxt(os.path.join(a.folder, "imu.csv"), delimiter=",", skiprows=1)
    bridge = Bridge(db=db or "", port=a.bridge_port, localize=a.localize, config=a.config, extra=a.rtabmap_arg)
    print(f"RTAB-Map bridge up. map database: {db or '(memory only)'}{'  [localization only]' if a.localize else ''}", flush=True)
    dash, n_sent, started, last_pair, tsdf, upgrader, surfels = None, 0, None, None, None, None, None
    try:
        for fr in source:
            h, w = fr.left.shape[:2]
            if a.orientation == "auto" and not judged:                    # live: the first 2 s of accelerometer rows judge the picture, those frames are not used
                imu_rows += [np.asarray(r[2], float) / 9.80665 for r in fr.imu]
                if len(imu_rows) < 150:
                    continue
                v = orientation.judge(imu_rows, orientation.load_R_ic(cfg.source.imu_sync))
                flip, judged = v["decision"] == "flip", True
                print("orientation check (accelerometer as an outside judge only):", v["reason"], flush=True)
            if flip:                                                       # the camera turned 180 deg about its optical axis: the pictures trade places, the principal point moves
                fr.left, fr.right = cv2.rotate(fr.right, cv2.ROTATE_180), cv2.rotate(fr.left, cv2.ROTATE_180)
            if dash is None:
                c = dict(fr.cal)
                if flip:
                    c["cx"], c["cy"] = (w - 1) - c["cx"], (h - 1) - c["cy"]
                bridge.init(c["f"], c["f"], c["cx"], c["cy"], c["B"], w, h, a.cloud_stride, a.min_range, a.max_range, a.disparity_every, a.lost_reset, a.max_depth_std)
                dash = Dashboard(c, (w, h), viewer=not a.no_viewer, save=a.save)
                dash.raw_points, dash.flatten = a.raw_points, a.flatten
                if a.depth != "rtabmap":
                    from rtabmap_lab.gabai_depth import DepthUpgrader, GabaiDepth
                    print({"gabai": "loading the GabAi depth chain (LAS2 + Depth Anything V2 on both pictures)", "ffs": "loading Fast-FoundationStereo", "sgbm": "SGBM + texture mask + WLS (CPU, no model)", "fused": "loading FFS, LAS2 and Depth Anything V2 for the fusion"}[a.depth] + " ...", flush=True)
                    if a.depth == "gabai":
                        depth_model = GabaiDepth(cfg)
                    elif a.depth == "sgbm":
                        from rtabmap_lab.sgbm_depth import SgbmDepth
                        depth_model = SgbmDepth(max_std=a.max_depth_std)
                    else:
                        from rtabmap_lab.fused_source import FusionDepth
                        depth_model = FusionDepth(a.depth)
                    dash.upgrader = upgrader = DepthUpgrader(depth_model, c, (w, h), dash.add_depth, dash.set_points,
                                                             dash.fused_depth, stride=a.cloud_stride, min_range=a.min_range, max_range=a.max_range, wait_pose_s=a.depth_wait_pose)
                if not a.no_surfels:
                    dash.sinks.append(surfels := SurfelFusion((c["f"], c["f"], c["cx"], c["cy"]), c["f"] * c["B"], on_update=dash.surfels, min_views=a.min_views, max_range=a.max_range))
                if not a.no_tsdf:
                    if a.tsdf_backend == "gpu":
                        from rtabmap_lab.gpu_tsdf import GpuTsdfMap
                        tsdf = GpuTsdfMap((c["f"], c["f"], c["cx"], c["cy"]), (w, h), voxel=a.tsdf_voxel, trunc=a.tsdf_trunc, max_range=a.max_range, fb=c["f"] * c["B"], on_mesh=dash.mesh, blur_keep=a.blur_keep)
                    else:
                        tsdf = TsdfMap((c["f"], c["f"], c["cx"], c["cy"]), (w, h), voxel=a.tsdf_voxel, trunc=a.tsdf_trunc, max_range=a.max_range, on_mesh=dash.mesh, blur_keep=a.blur_keep)
                    dash.sinks.append(tsdf)
                    dash.tsdf = tsdf
                if not a.no_nav:
                    from rtabmap_lab.navmap import NavMap, NavSink
                    dash.sinks.append(NavSink(NavMap(cell_m=a.nav_cell), c["f"], c["cx"], c["cy"], on_update=dash.nav))
                print(f"stereo {w}x{h}, f={c['f']:.1f}px, baseline={c['B'] * 1000:.1f} mm ({'metric' if c.get('metric') else 'ASSUMED'}); pictures turned 180 deg: {flip}", flush=True)

            if rec_imu is not None and not fr.imu:
                sel = (rec_imu[:, 0] > rec_t) & (rec_imu[:, 0] <= fr.t)
                fr.imu = [(r[0], np.radians(r[1:4]), r[4:7] * 9.80665) for r in rec_imu[sel]]
                rec_t = fr.t
            if shutter is not None and R_ic is not None:                   # opposite rolling shutters: the right picture resampled at the left picture's time, row by row (rtabmap_lab/rolling_shutter.py)
                recent = [r[1] for r in fr.imu if fr.t - r[0] < 0.02]
                if recent:
                    F = np.diag([-1.0, -1.0, 1.0]) if flip else np.eye(3)
                    dts = fr.sync.get("pair_dt_signed_ms")
                    fr.right = shutter.correct(fr.right, F @ R_ic.T @ np.mean(recent, 0), c["f"], c["cx"], c["cy"], flip, None if dts is None else (-dts if flip else dts))
            for m in bridge.poll():
                handle(m, dash, fr.left)
            right_gray = cv2.cvtColor(fr.right, cv2.COLOR_BGR2GRAY)
            if a.source == "replay":
                bridge.wait_idle()                                            # a recording: every pair is used, in order
            if upgrader is not None:
                upgrader.remember(fr.idx, fr.left, fr.right)
            gyro_rows += fr.imu                                            # the gyro since the last SENT pair: the bridge keeps the heading with it when the odometry is lost
            gyro = orientation.gyro_rotvec([r[0] for r in gyro_rows], [r[1] for r in gyro_rows], R_ic, flip, fr.t) if a.gyro and gyro_rows and n_sent else None
            if bridge.send_frame(fr.idx, fr.t, fr.left, right_gray, gyro):
                gyro_rows = []
                n_sent += 1
                dash.at(fr.idx, fr.t)
                dash.pair(fr.left, fr.right)
                last_pair = fr.left
            if a.source == "replay":
                bridge.wait_idle()
                for m in bridge.poll():
                    dash.at(fr.idx, fr.t)
                    handle(m, dash, last_pair, bridge.skipped)
                if upgrader is not None:
                    upgrader.wait_idle()                                           # a recording: every node gets the chosen depth, however long the models take
            else:
                for m in bridge.poll():
                    handle(m, dash, last_pair, bridge.skipped)
            if not bridge.alive:
                print("the bridge stopped:\n" + "\n".join(bridge.log[-8:]), flush=True)
                break
            if a.max_frames and n_sent >= a.max_frames:
                break
    except KeyboardInterrupt:
        print("\nstopping ...", flush=True)
    finally:
        if dash is not None:
            for m in bridge.poll():
                handle(m, dash, last_pair, bridge.skipped)
            dash.log_map(force=True)
        print(f"frames sent {bridge.sent}, skipped {bridge.skipped}; saving the map ...", flush=True)
        if upgrader is not None:
            upgrader.close()
            u = upgrader.stats
            print(f"depth source {a.depth}: {u['gabai']} nodes with it, {u['fallback']} with RTAB-Map's own" + (f"  [{u['error']}]" if u.get("error") else ""), flush=True)
            if u["fallback"] > 0.3 * max(u["gabai"] + u["fallback"], 1) and a.source == "live":
                print("WARNING: the depth models could not keep up with the nodes: many nodes fell back to RTAB-Map's depth. Try --depth ffs (one model) or a larger --max-depth-std / smaller picture rate.", flush=True)
        if surfels is not None:
            surfels.flush()
            st = surfels.stats
            print(f"surfels: {st['stable']} confirmed by >= {a.min_views} pictures of {st['surfels']} seen, {st['rebuilds']} rebuilds after loop closures" + (f"  [ERROR {st['error']}]" if st.get("error") else ""), flush=True)
            surfels.close()
        if tsdf is not None:
            stem = os.path.splitext(db)[0] if db else os.path.join(LAB, "maps", time.strftime("tsdf_%Y%m%d_%H%M%S"))
            os.makedirs(os.path.dirname(stem), exist_ok=True)
            tsdf.flush()
            if a.dump_nodes:                                               # the exact input of the TSDF of this run, for tools/surface_lab.py --npz
                ids = sorted(i for i in tsdf.frames if i in tsdf.poses)
                np.savez_compressed(a.dump_nodes, ids=ids, depth=np.stack([tsdf.frames[i][0] for i in ids]), bgr=np.stack([tsdf.frames[i][1] for i in ids]), poses=np.stack([tsdf.poses[i] for i in ids]),
                                    cls=np.stack([tsdf.frames[i][2] if tsdf.frames[i][2] is not None else np.zeros(tsdf.frames[i][0].shape, np.uint8) for i in ids]))
            print(f"TSDF: {len(tsdf.frames)} nodes held, {len(tsdf.poses)} with a pose, {tsdf.stats['integrated']} fused", flush=True)
            if tsdf.save_mesh(stem + "_tsdf.ply"):
                print(f"TSDF mesh saved: {stem}_tsdf.ply  ({tsdf.stats['vertices']} vertices, {tsdf.stats['rebuilds']} rebuilds after loop closures)", flush=True)
            tsdf.close()
        bridge.close(save=not a.localize)
        for line in bridge.log[-5:]:
            print("bridge:", line, flush=True)
        if db and os.path.exists(db):
            print(f"map saved: {db}  ({os.path.getsize(db) / 1e6:.1f} MB). Export: py -m rtabmap_lab export {db}", flush=True)


def handle(m, dash, left, skipped=0):
    t = m["type"]
    if t == "status":
        dash.status(m, left, skipped)
        if m["loop_id"] > 0 or m["proximity_id"] > 0:
            dash.event(f"loop closure: frame {m['id']} matched node {m['loop_id'] or m['proximity_id']} ({'appearance' if m['loop_id'] > 0 else 'proximity'})", "INFO")
    elif t == "disparity":
        dash.disparity(m["image"])
    elif t == "node":
        dash.node(m)
    elif t == "depth":
        dash.depth(m)
    elif t == "graph":
        dash.graph(m)
    elif t == "log":
        dash.event(m["text"], "WARN")
        print("bridge:", m["text"], flush=True)
    elif t == "ready" and m["nodes"]:
        dash.event(f"loaded a map with {m['nodes']} nodes", "INFO")
        print(f"loaded a map with {m['nodes']} nodes (their points are drawn again)", flush=True)


def add_args(ap):
    ap.add_argument("--source", default="live", choices=["live", "replay"])
    ap.add_argument("--folder", help="replay: a folder written by gabai.tools.record_pairs")
    ap.add_argument("--port", type=int, default=5600, help="live: the port the Pi sender connects to (the GabAi pipeline uses the same one: run one at a time)")
    ap.add_argument("--bridge-port", type=int, default=5710)
    ap.add_argument("--db", help="map database to create or continue; default rtabmap_lab/maps/session_<time>.db")
    ap.add_argument("--no-db", action="store_true", help="keep the map in memory only")
    ap.add_argument("--localize", action="store_true", help="with --db: only localise in that map")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="RTAB-Map parameter file (.ini)")
    ap.add_argument("--rtabmap-arg", nargs=2, action="append", default=[], metavar=("--Group/Name", "value"), help="any RTAB-Map parameter, e.g. --rtabmap-arg --Vis/MinInliers 25")
    ap.add_argument("--pipeline-config", default="config/pipeline.toml", help="read-only: where the calibration path and the Pi port defaults come from")
    ap.add_argument("--orientation", default="auto", choices=["auto", "as-is", "flip"], help="auto: the accelerometer judges once at the start whether the pictures are upright (it is not used for anything else); "
                    "as-is: use the pictures as they are; flip: turn them 180 deg. An upside-down picture gives an upside-down map")
    ap.add_argument("--flip180", dest="orientation", action="store_const", const="flip", help="same as --orientation flip (the flag the GabAi pipeline uses)")
    ap.add_argument("--no-flip180", dest="orientation", action="store_const", const="as-is", help="same as --orientation as-is")
    ap.add_argument("--cloud-stride", type=int, default=4, help="every Nth pixel of the disparity becomes a map point")
    ap.add_argument("--min-range", type=float, default=0.3)
    ap.add_argument("--max-range", type=float, default=2.5, help="metres of depth used: the real disparity noise is about 1 px (14 cm at 2 m); on walk_20261102 2.5 m gave the map closest to held-out pictures (4 m: 4.02 px, 2.5 m: 3.72, 2 m: 3.84)")
    ap.add_argument("--blur-keep", type=float, default=0.75, help="fuse a node's depth into the TSDF only if its picture is among the sharpest this share of recent nodes (1 = all)")
    ap.add_argument("--max-depth-std", type=float, default=0.08, help="drop depth whose expected noise z^2 * 0.3 px / (f * baseline) is above this many metres (0.08 keeps about 2.6 m with this rig); 0 = keep all")
    ap.add_argument("--disparity-every", type=int, default=3, help="show RTAB-Map's disparity every Nth frame")
    ap.add_argument("--gyro", action="store_true", help="give the bridge the gyro rotation between pairs, used only to keep the heading when the odometry restarts after being lost (off: pictures only)")
    ap.add_argument("--shutter-fix", action="store_true", help="resample the right picture at the left picture's time from the gyro (rtabmap_lab/rolling_shutter.py; off by default)")
    ap.add_argument("--shutter-file", default="data/rolling_shutter.json", help="the rig's measured rolling-shutter readout (py -m rtabmap_lab.rolling_shutter <recording> --save ...)")
    ap.add_argument("--calib", help="stereo calibration (.npz). Default: replay uses the one the recording was made with (its meta.json), live the one in --pipeline-config")
    ap.add_argument("--lost-reset", type=int, default=15, help="restart the odometry at its last pose after this many lost frames in a row")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--depth", default="rtabmap", choices=["rtabmap", "gabai", "ffs", "sgbm", "fused"], help="rtabmap: RTAB-Map's own stereo matching (SGBM) for the TSDF and the points; sgbm: OpenCV SGBM + texture mask + WLS, failed pixels stay empty (rtabmap_lab/sgbm_depth.py); gabai: the main system's depth (LAS2 stereo + Depth Anything V2 on both pictures, "
                    "fused), computed on the GPU for each new node. The poses are RTAB-Map's either way")
    ap.add_argument("--no-surfels", action="store_true", help="skip the surfel map (the main 3D view)")
    ap.add_argument("--min-views", type=int, default=2, help="a surfel is drawn only when this many pictures agree on it (1 = no check; 3 = cleaner and sparser)")
    ap.add_argument("--flatten", action=argparse.BooleanOptionalAction, default=True, help="snap the big planes (walls, floor, ceiling) of the confirmed surfels flat: points already within 2.5 cm of a plane move onto it, nothing else moves")
    ap.add_argument("--raw-points", action="store_true", help="also draw every node's raw stereo points (the cloud with the fans), off by default")
    ap.add_argument("--no-tsdf", action="store_true", help="skip the Open3D TSDF surface (the stereo point map stays)")
    ap.add_argument("--depth-wait-pose", type=float, default=8.0, help="with --depth gabai/ffs/fused: run the depth models only for nodes RTAB-Map's graph has a pose for (60%% of the nodes it announces never get one); a node still unposed after this many seconds is dropped. 0 = every node, as before")
    ap.add_argument("--no-nav", action="store_true", help="skip the egocentric navigation map (rtabmap_lab/navmap.py: levelled obstacle grid, distance field, route)")
    ap.add_argument("--nav-cell", type=float, default=0.05, help="navigation grid cell size in metres (the grid is 6 x 6 m around the camera)")
    ap.add_argument("--tsdf-backend", default="gpu", choices=["gpu", "open3d"], help="gpu: nvblox-style TSDF on the GPU (rtabmap_lab/gpu_tsdf.py: free-space carving, weight-gated mesh, no floating debris); open3d: the original CPU volume")
    ap.add_argument("--tsdf-voxel", type=float, default=0.02, help="TSDF voxel size in metres (finer = more sensitive to depth noise)")
    ap.add_argument("--tsdf-trunc", type=float, default=0.08, help="TSDF truncation distance in metres (3 to 4 voxels, about the depth noise at the working range)")
    ap.add_argument("--dump-nodes", help="write every node's depth, picture and pose to this .npz at the end (for tools/surface_lab.py --npz)")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--save", help="also write the Rerun recording to this .rrd")
    return ap


def main(argv=None):
    ap = add_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    a = ap.parse_args(argv)
    if a.source == "replay" and not a.folder:
        ap.error("--source replay needs --folder")
    a.rtabmap_arg = [x for pair in a.rtabmap_arg for x in pair]
    run(a)


if __name__ == "__main__":
    main()

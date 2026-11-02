"""The pipeline: frames in, one Rerun recording and one summary out. The mode (stereo_only or stereo_imu) only changes which pose provider is built.

Per frame: depth (stereo model, left-right checked) -> pose (provider) -> map (semi-dense points placed with the pose) -> hands and objects ->
diagnostics -> Rerun. A pose that is not trustworthy (tracking lost) is shown but its frame is NOT added to the map, so a lost track cannot stack or
duplicate geometry. Basalt returns poses a few frames late; each pose is paired with the stored frame it belongs to by its timestamp.
"""
import json
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from gabai.diag import checks as C
from gabai import depthq as Q
from gabai.diag import log as dlog
from gabai.mapref import MapRefiner
from gabai.navguard import nav_active
from gabai.povmesh import fused_for_view, surface_mask
from gabai.pipeline import diagnostics as D
from gabai.pipeline.config import PipelineConfig, validate
from gabai.pipeline.perf import AsyncViz, Prefetch, StageWorker, Stopwatch, limit_threads, lower_priority
from gabai.pipeline.poses import make_provider
from gabai.pipeline.sources import open_source
from gabai.tools.slam_cloud_view import SlamCloud

log = dlog.get("pipeline")
MAX_PENDING = 150


def remove_speckles(disp, trusted, max_px, max_diff):
    """Drop small isolated islands of trusted disparity (blobs under max_px pixels whose disparity steps by more than max_diff from their neighbours).
    Only removes pixels; the values that stay are untouched. Returns the new trusted mask."""
    if max_px <= 0:
        return trusted
    d16 = np.where(trusted, np.nan_to_num(disp) * 16.0, -16.0).astype(np.int16)
    cv2.filterSpeckles(d16, -16, int(max_px), int(round(max_diff * 16)))
    return trusted & (d16 != -16)


MONO_FB = 30.0   # f*B stand-in (px*m) to turn monocular metres into disparity; the affine fit to stereo absorbs the real value


def default_depth(model, speckle_px=0, speckle_diff=1.0, mono="", mono_mode="blend", scale=1.0, depth_correction="", mono_pair=False, pair_tol_px=2.0):
    from concurrent.futures import ThreadPoolExecutor
    from gabai import depthcal
    from gabai.spike.las2_runner import Las2Depth

    def load_mono():                                                     # importing transformers alone takes several seconds: do it while the stereo model loads
        from gabai.fuse_depth import DepthFuser
        from gabai.spike.mono_depth import MonoDepth
        return MonoDepth(mono), DepthFuser()
    with ThreadPoolExecutor(1, "mono-load") as ex:
        job = ex.submit(load_mono) if mono else None
        las = Las2Depth(model)
        mono_model, fuser = job.result() if job else (None, None)
    correct = depthcal.load(depth_correction, model) if depth_correction else None       # distance correction from the calibration board pairs (gabai/depthcal.py)
    if mono_model is not None:
        mono_pool = ThreadPoolExecutor(1, "mono-depth")                  # the monocular model runs on its own CUDA stream and thread, at the same time as the stereo model

    def fn(left_bgr, right_bgr):
        rgb_l = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb_l.shape[:2]
        pair = mono_model is not None and mono_pair and mono_mode == "fill"
        mono_job = None if mono_model is None else mono_pool.submit(mono_model.depth_pair, rgb_l, rgb_r) if pair else mono_pool.submit(mono_model.depth, rgb_l)
        if scale != 1.0:                                  # a bar 2 px wide is 3 px wide at 1.5x: the model's coarse feature grid sees it better (scripts/bench_input_scale.py)
            big = (int(round(w * scale)), int(round(h * scale)))
            d, trusted, dt = las.disparity_lr_batched(cv2.resize(rgb_l, big, interpolation=cv2.INTER_CUBIC), cv2.resize(rgb_r, big, interpolation=cv2.INTER_CUBIC))
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA) / scale
            trusted = cv2.resize(trusted.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.99      # every enlarged pixel behind it was trusted
        else:
            d, trusted, dt = las.disparity_lr_batched(rgb_l, rgb_r)
        ms = dt * 1000.0
        if correct is not None:
            d = correct(d)
        lr_ok = trusted
        fill = None
        if mono_model is not None and mono_mode == "fill":     # stereo stays the only source of every measured pixel; the monocular map only supplies the holes
            mono_m, mono_r = mono_job.result() if pair else (mono_job.result(), None)
            dm = MONO_FB / np.clip(mono_m, 0.2, 30.0)
            aligned, fill_mask, finfo = fuser.fill(np.where(trusted, d, 0), trusted, dm)
            if pair and finfo.get("ok"):                       # the right picture's monocular map, on the same scale and put into the left view, must agree with the left one
                from gabai.fuse_depth import combine_mono_pair, fit_affine, warp_right_to_left
                good = trusted & (d > 1)
                right_in_left = warp_right_to_left(MONO_FB / np.clip(mono_r, 0.2, 30.0), np.where(good, d, aligned))       # raw right monocular disparity, in the left view
                ok_r = good & np.isfinite(right_in_left)
                if ok_r.sum() >= fuser.min_pixels:                  # its own scale and offset from the stereo pixels (the right view is not exactly the left one)
                    a_r, b_r = fit_affine(right_in_left, d, ok_r)
                    r_in_left = a_r * right_in_left + b_r
                else:
                    r_in_left = np.full(aligned.shape, np.nan, np.float32)
                combined, agree = combine_mono_pair(aligned, r_in_left, pair_tol_px)
                finfo["pair_agree"] = float((agree & fill_mask).sum() / max(fill_mask.sum(), 1))
                aligned = np.where(fill_mask & agree, combined, aligned).astype(np.float32)      # where the right map agrees the two are averaged; where not, the left one still fills the hole
            fill = (aligned, fill_mask, finfo)
            ms += mono_model.last_ms
        elif mono_model is not None:
            from gabai.fuse_depth import mono_veto
            mono_m = mono_job.result()
            dm = MONO_FB / np.clip(mono_m, 0.2, 30.0)
            if mono_mode == "check":
                trusted, _, _ = mono_veto(d, trusted, dm)
            else:                                              # holes stay holes: only pixels stereo measured are blended, nothing is invented
                fused, _ = fuser.fuse(np.where(trusted, d, 0), trusted, dm)
                trusted = trusted & (fused > 0.5)
                d = np.where(trusted, fused, d)
            ms += mono_model.last_ms
        after_mono = trusted
        trusted = remove_speckles(d, trusted, speckle_px, speckle_diff)
        fn.filled = None if fill is None or not fill[1].any() else (fill[0].astype(np.float32), fill[1] & ~trusted, fill[2])      # never over a trusted stereo pixel
        fn.reasons = Q.reject_reasons(d, lr_ok, after_mono, trusted, cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY), None if fn.filled is None else fn.filled[1])
        fn.fill_info = None if fill is None else fill[2]
        fn.mono_view = None if fill is None else (fill[0], fill[2])
        fn.mono_raw = None if mono_job is None else mono_m             # what the monocular model itself says (metres), whether or not it was usable
        fn.mono_raw_r = mono_r if pair and mono_mode == "fill" else None
        return np.where(trusted, d, np.nan).astype(np.float32), trusted, ms
    def warm_up(h=360, w=640):
        """One throw-away call on a blank pair, so the first real frame does not pay for the GPU's first-call set-up (about 1.5 s); the fuser forgets what it saw."""
        blank = np.zeros((h, w, 3), np.uint8)
        fn(blank, blank)
        if fuser is not None:
            fuser.ab = fuser.low = None
        fn.filled = fn.mono_view = fn.mono_raw = fn.mono_raw_r = fn.fill_info = None
    fn.warm_up = warm_up
    fn.reasons = None                                                  # why each pixel was kept or dropped (gabai/depthq.py), refreshed by every call
    fn.filled = None                                                   # (monocular disparity on the stereo scale, mask of the pixels it supplies, fit info) in mono_mode "fill"
    fn.fill_info = None
    fn.mono_view = None
    fn.mono_raw = None
    fn.mono_raw_r = None
    return fn


def load_held_models(cfg):
    """The hand tracker and the object detector (they need no calibration, so they can load while the depth models do). Returns (hands, detector, notes)."""
    hands = det = None
    notes = {}
    if cfg.hands.enabled:
        try:
            from gabai.hands import HandTracker
            hands = HandTracker()
        except Exception as e:
            notes["hands"] = str(e)
    if cfg.detector.enabled:
        try:
            from gabai.detect import UltralyticsDetector
            det = UltralyticsDetector()
        except Exception as e:
            notes["objects"] = str(e)
    return hands, det, notes


def build_held(cfg, cal, models=None):
    from gabai.pipeline.held import HeldAnalyzer
    hands, det, notes = models() if models is not None else load_held_models(cfg)
    for k, v in notes.items():
        log.warning("pipeline.model_off", model=k, reason=v)
    if hands is None and det is None:
        return None, notes
    return HeldAnalyzer(cal["f"], cal["B"], cal["cx"], cal["cy"], det, hands, metric=cal["metric"], detector_every=cfg.perf.detector_every), notes


def sample_correspondences(disp, trusted, n=40, margin=8):
    ys, xs = np.nonzero(np.isfinite(disp) & (disp > 0.5) & np.asarray(trusted, bool))
    if len(xs) == 0:
        return []
    sel = np.random.default_rng(0).choice(len(xs), min(n, len(xs)), replace=False)
    return [((xs[i], ys[i]), (xs[i] - disp[ys[i], xs[i]], ys[i])) for i in sel]


def run(cfg, depth_fn=None, held=None, viz_factory=None, log_dir=None, quiet=False):
    errs, warns = validate(cfg)
    if errs:
        raise SystemExit("cannot run in mode '%s':\n  - %s" % (cfg.pipeline.mode, "\n  - ".join(errs)))
    run_dir = dlog.configure(log_dir=log_dir, run_name=f"pipeline_{cfg.pipeline.mode}", level=os.environ.get("GABAI_LOG_LEVEL", "INFO"))
    log.info("pipeline.start", mode=cfg.pipeline.mode, source=cfg.source.kind, config={k: vars(v) for k, v in vars(cfg).items()}, warnings=warns)
    for w in warns:
        if not quiet:
            print("WARNING:", w)
    perf_notes = []
    if cfg.perf.low_priority:
        perf_notes.append(lower_priority())
    perf_notes.append(limit_threads(cfg.perf.threads))
    staged = bool(cfg.perf.staged and cfg.perf.async_viz)          # two stages need the drawing thread to keep each stage's frame position apart
    if cfg.perf.staged and not staged:
        perf_notes.append("staged execution needs async_viz: running in one thread")
    log.info("pipeline.perf", notes=perf_notes, perf=vars(cfg.perf), staged=staged)
    sw = Stopwatch()
    t_start = time.perf_counter()
    held_job = None
    if held is None and (cfg.hands.enabled or cfg.detector.enabled):
        from concurrent.futures import ThreadPoolExecutor
        held_job = ThreadPoolExecutor(1, "held-load").submit(load_held_models, cfg)     # loads while the depth models do
    depth_fn = depth_fn or default_depth(cfg.stereo.model, cfg.stereo.speckle_px, cfg.stereo.speckle_diff, cfg.stereo.mono, cfg.stereo.mono_mode, cfg.stereo.scale, cfg.stereo.depth_correction, cfg.stereo.mono_pair)
    if hasattr(depth_fn, "warm_up"):
        depth_fn.warm_up()
    log.info("pipeline.startup", models_ready_s=round(time.perf_counter() - t_start, 1), hint="the depth models are loaded and warmed up; the viewer opens with the first picture")
    if not quiet:
        print(f"depth models ready after {time.perf_counter() - t_start:.1f} s; waiting for the first picture (the viewer opens with it)", flush=True)
    if held_job is not None:                                          # everything is loaded BEFORE the source (and so the live server) opens: no picture arrives while a model is still loading
        held_job.result()
        if not quiet:
            print("hand and object models ready", flush=True)
    obj3d = None
    if cfg.objects3d.enabled:
        from gabai.boxer_lift import make_lifter, preload
        from gabai.objects3d import Objects3D
        preload()
        o3 = cfg.objects3d
        obj3d = Objects3D(lambda: make_lifter([x.strip() for x in o3.labels.split(",")], o3.thresh2d, o3.thresh3d), o3.every_s, o3.max_range, o3.gate, o3.min_looks)
        obj3d.wait_ready()
        if not quiet:
            print("Boxer 3D object models " + (f"NOT loaded ({obj3d.error}); objects are off" if obj3d.error else "ready and warmed up"), flush=True)
    src = open_source(cfg)
    if cfg.perf.prefetch:
        src = Prefetch(src, 4)

    class St:                                                         # state shared by the two stages; stage A sets up the first part once, stage B owns the rest
        provider = viz = cloud = None
        held_status = {"hands": "off", "objects": "off"}
        map_cache, map_cache_n = np.zeros((0, 3)), -99
        prev_gray = None
        last_cal = None
        refiner = None

    st = St()
    st.held_obj = held
    st.pending, st.traj, st.check_tally = {}, [], {}
    st.counts = {"frames": 0, "poses_ok": 0, "poses_held": 0, "map_frames": 0, "frames_skipped_inconsistent": 0, "map_resets": 0, "pose_jumps": 0}
    st.last_pose, st.incons_run, st.vox_peak, st.vox_trace = None, 0, 0, []
    st.loop_seen = None
    st.tsdf, st.obj3d = None, None
    st.ms = {"depth": [], "frame": []}
    st.why_sum, st.why_n = {}, 0
    st.prev_t, st.gaps = None, []
    st.starts, st.ends = [], []
    st.held_on, st.nav_throttled, st.nav_checked, st.nav_seconds = True, False, 0.0, 0.0
    st.hands_drawn = False
    st.refiner = None
    st.align_hist, st.last_imu, st.hist = [], {}, deque(maxlen=3)
    st.hs = {"hand_detections": 0, "hand_3d_ok": 0, "hand_3d_refused": 0, "joints_stereo": 0, "joints_model": 0, "frames_with_held_model": 0,
             "frames_object_seen_in_hand": 0, "grabs": 0, "releases": 0}

    def tally(cs):
        for c in cs:
            st.check_tally.setdefault(c.name, {"pass": 0, "fail": 0, "suspect": 0, "unknown": 0})[c.status] += 1

    def stage_b(bundle):
        """Pose, map, hands and objects, diagnostics. Needs the depth of its own frame only, so it overlaps with the next frame's depth in stage A."""
        fr, gl, gr, disp, trusted, t_ns = bundle["fr"], bundle["gl"], bundle["gr"], bundle["disp"], bundle["trusted"], bundle["t_ns"]
        b0 = bundle
        viz, provider, cloud, held_obj, cal_fr = st.viz, st.provider, st.cloud, (st.held_obj if st.held_on else None), fr.cal
        heavy_draw = bundle["heavy_draw"]
        viz.at(fr.idx, fr.t)
        rect_px = None
        if fr.idx % 30 == 0:                                          # row alignment of the pair: on half-size copies, because this check is slow
            from gabai.spike.autorect import vertical_residual
            with sw("rectification_check"):
                rect_px = vertical_residual(cv2.resize(gl, None, fx=0.5, fy=0.5), cv2.resize(gr, None, fx=0.5, fy=0.5))
            viz.scalar("depth/row_misalignment_px", rect_px)
        frame_checks = list(bundle["checks"])
        if rect_px is not None:
            frame_checks.append(D.check_rectification(rect_px))
        st.pending[t_ns] = {"fr": fr, "disp": disp, "trusted": trusted, "gl": gl, "checks": frame_checks, "t0": bundle["t0"], "rect": rect_px, "fill": bundle.get("fill"), "fused": bundle.get("fused")}
        while len(st.pending) > MAX_PENDING:
            old = min(st.pending)
            st.pending.pop(old)
            log.warning("pipeline.pending_dropped", t_ns=old, hint="a pose never arrived for this frame")
        if (provider.uses_imu or getattr(provider, "wants_gyro", False)) and fr.imu:
            gy = np.array([g for _, g, _ in fr.imu])
            ac = np.array([a for _, _, a in fr.imu])
            st.last_imu = {"gyro_dps": float(np.degrees(np.linalg.norm(gy, axis=1).mean())), "accel_g": float(np.linalg.norm(ac, axis=1).mean() / 9.80665), "n": len(fr.imu),
                           "rate_hz": len(fr.imu) / max(fr.imu[-1][0] - fr.imu[0][0], 1e-3) if len(fr.imu) > 1 else None}
            viz.scalar("imu/gyro_rate_dps", st.last_imu["gyro_dps"])
            viz.scalar("imu/accel_norm_g", st.last_imu["accel_g"])
            viz.scalar("imu/samples_per_frame", st.last_imu["n"])
            provider.feed_imu(fr.imu)
        with sw("pose"):
            if cfg.tracking.vo_uses_fill and b0["fill"] is not None:        # stereo pixels first; the monocular fill only where stereo has no depth
                vo_disp, vo_trusted = np.where(b0["fill"][1], b0["fill"][0], disp), trusted | b0["fill"][1]
            else:
                vo_disp, vo_trusted = disp, trusted
            poses = provider.push(t_ns, gl, gr, vo_disp, vo_trusted, **({"klt": fr.klt} if getattr(provider, "wants_klt", False) else {}))
        if getattr(provider, "ready", True) is False:           # no pose before the IMU has delivered: without this line the cloud just stays empty
            msg = "; ".join(provider.waiting_lines()) + ". No pose means no 3D cloud (the depth panels still work)"
            log.warning_every("wait_pose", 5.0, "pipeline.waiting_for_pose", imu_samples=provider.n_imu, hint=msg)
            if fr.idx % 60 == 0:
                viz.event(msg, "WARNING")
        for p in poses:
            b = st.pending.pop(p.t_ns, None)
            if b is None:
                log.warning("pipeline.pose_without_frame", t_ns=p.t_ns, hint="no stored frame has this timestamp: timestamps between the pose provider and the source disagree")
                continue
            f = b["fr"]
            viz.at(f.idx, f.t)
            if p.ok and cfg.map.max_pose_step_m and st.last_pose is not None and float(np.linalg.norm(np.asarray(p.t_wc, float) - st.last_pose)) > cfg.map.max_pose_step_m:
                p.ok = False
                p.info["reason"] = f"pose jumped {np.linalg.norm(np.asarray(p.t_wc, float) - st.last_pose):.2f} m in one frame (limit {cfg.map.max_pose_step_m} m): not trusted"
                st.counts["pose_jumps"] += 1
            st.last_pose = np.asarray(p.t_wc, float).copy()
            checks = list(b["checks"])
            st.hist.append((f.t, np.asarray(p.t_wc, float), R.from_matrix(p.R_wc)))
            checks.append(D.check_pose_stability(list(st.hist)))
            if "inliers" in p.info:
                checks.append(D.check_tracking_quality(p.info))
                viz.scalar("tracking/inliers", p.info["inliers"])
            if f.sync.get("buf"):
                viz.sync(f.sync["buf"], fr.idx - f.idx)
            for k_ in ("ekf_speed_mps", "ekf_gyro_bias_dps", "ekf_acc_bias", "ekf_sigma_pos_cm", "ekf_sigma_att_deg", "innov_pos_cm", "innov_att_deg", "ekf_rejected", "ekf_resets", "ekf_imu_rejected", "ekf_diverged", "ekf_dev_cm", "ekf_dev_deg", "ekf_using_filter"):
                if p.info.get(k_) is not None:
                    viz.scalar(f"filter/{k_}", p.info[k_])
            if p.info.get("klt_inliers") is not None:
                viz.scalar("tracking/klt_inliers", p.info["klt_inliers"])
                viz.scalar("tracking/klt_weight_used", p.info["klt_weight"])
                if p.info.get("klt_vs_vo_m") is not None:
                    viz.scalar("tracking/klt_vs_vo_cm", 100 * p.info["klt_vs_vo_m"])
            if len(st.traj):
                viz.scalar("tracking/pose_step_m", float(np.linalg.norm(p.t_wc - st.traj[-1])))
            if "gyro_disagreement_deg" in p.info:
                viz.scalar("imu/gyro_vs_tracker_disagreement_deg", p.info["gyro_disagreement_deg"])
                viz.scalar("imu/gyro_bias_estimate_dps", p.info["gyro_bias_dps"])
                viz.scalar("imu/tracker_rotation_rate_dps", p.info["rotation_rate_dps"])
            if p.info.get("relocalized"):
                viz.event(f"t={f.t:.1f}s tracking was lost and found again: {p.info['reason']}", "INFO")
                st.counts["relocalized"] = st.counts.get("relocalized", 0) + 1
            if p.info.get("loop"):
                lp = p.info["loop"]
                viz.event(f"t={f.t:.1f}s revisit of keyframe {lp['keyframe']}: the pose had drifted {lp['drift_cm']:.1f} cm / {lp['drift_deg']:.1f} deg since then "
                          f"({lp['inliers']} features agree){' - pose corrected' if lp['applied'] else ' - not applied'}", "INFO")
                st.counts["loops_found"] = st.counts.get("loops_found", 0) + 1
            held_out = None
            lv = p.info.get("loop_version")
            if lv is not None and lv != st.loop_seen and p.ok:                          # a loop closure moved the keyframes: put the stored frames and the path back where they now belong
                if st.loop_seen is not None and getattr(provider, "vo", None) is not None:
                    n_fr, n_drop, moved = cloud.rebuild(getattr(provider, "node_pose", None) or provider.vo.node_pose)
                    st.traj = [np.asarray(x, float) for x in cloud.traj]
                    st.map_cache, st.map_cache_n, st.incons_run = np.zeros((0, 3)), -99, 0
                    st.counts["loop_rebuilds"] = st.counts.get("loop_rebuilds", 0) + 1
                    if st.tsdf is not None:                                # frames fused with the old poses: start the volume again
                        st.tsdf.reset()
                        viz.tsdf_clear()
                    if st.obj3d is not None:                               # the keyframes moved: every look of every object is placed again with the corrected poses, and the objects take their solved positions
                        if hasattr(provider, "node_pose"):
                            st.obj3d.apply_graph(provider.node_pose, provider.landmark_positions())
                        else:
                            st.obj3d.reset()
                            viz.objects3d_clear()
                    viz.clear_tracked() if st.hands_drawn else None
                    viz.cloud(None, None, cloud.shown_points())
                    viz.event(f"t={f.t:.1f}s loop closed: the trajectory was corrected and {n_fr} stored frames were laid down again (largest move {moved * 100:.0f} cm)", "INFO")
                    log.info("pipeline.loop_rebuild", fid=f.fid, frames=n_fr, dropped=n_drop, moved_cm=round(moved * 100, 1), graph=getattr(provider.vo, "last_graph", None))
                st.loop_seen = lv
            if p.ok:
                st.counts["poses_ok"] += 1
                st.traj.append(np.asarray(p.t_wc, float))
                with sw("alignment"):
                    if cloud.n - st.map_cache_n >= 2:
                        st.map_cache, st.map_cache_n = cloud.map.points(min_obs=2), cloud.n
                    al = D.map_alignment(st.map_cache, p.R_wc, p.t_wc, b["disp"], f.cal["f"], f.cal["B"], f.cal["cx"], f.cal["cy"]) if len(st.map_cache) else None
                checks.append(D.check_alignment(al))
                if al:
                    viz.scalar("tracking/map_vs_depth_median_cm", al["median_abs_m"] * 100)
                    viz.scalar("tracking/map_vs_depth_signed_cm", al["median_signed_m"] * 100)
                    st.align_hist.append(al["median_abs_m"])
                if held_obj is not None:
                    with sw("hands_objects"):
                        held_out = held_obj.frame(f.t, f.left, b["disp"], b["trusted"], p.R_wc, p.t_wc)
                    with sw("viz_annotate"):
                        if heavy_draw or held_out["events"]:
                            viz.annotate(f.left, held_out)
                            st.hands_drawn = True
                    hs = st.hs
                    for hd_ in held_out["hands"]:
                        hs["hand_detections"] += 1
                        if hd_["solve"]["ok"]:
                            hs["hand_3d_ok"] += 1
                            hs["joints_stereo"] += hd_["solve"]["n_stereo"]
                            hs["joints_model"] += hd_["solve"]["n_model"]
                        else:
                            hs["hand_3d_refused"] += 1
                            log.warning_every("hand3d_refused", 5.0, "pipeline.hand_3d_refused", fid=f.fid, reason=hd_["solve"]["reason"], scale=hd_["solve"]["scale"])
                    hs["frames_with_held_model"] += int(any(m.get("n_points") for m in held_out["models"]))
                    hs["frames_object_seen_in_hand"] += int(any(m.get("seen") for m in held_out["models"]))
                    hs["grabs"] += sum(e["type"] == "grab" for e in held_out["events"])
                    hs["releases"] += sum(e["type"] in ("release", "lost") for e in held_out["events"])
                    for e in held_out["events"]:
                        viz.event(f"{e['type']}: {e.get('label')} {e.get('reason') or ''}", "INFO")
                        log.info("pipeline.hand_event", fid=f.fid, type=e["type"], label=e.get("label"), reason=e.get("reason"))
                if held_obj is None and st.hands_drawn:                  # the hand/object analysis was switched off: take its drawings out of the scene
                    viz.clear_tracked()
                    st.hands_drawn = False
                inconsistent = bool(cfg.map.max_misalign_cm and al is not None and al["median_abs_m"] * 100 > cfg.map.max_misalign_cm)
                if inconsistent:
                    st.incons_run += 1
                    st.counts["frames_skipped_inconsistent"] += 1
                    viz.pose(p.R_wc, p.t_wc, np.array(st.traj))
                    log.warning_every("map_inconsistent", 5.0, "pipeline.map_inconsistent", fid=f.fid, misalign_cm=round(al["median_abs_m"] * 100, 1), run=st.incons_run,
                                      hint="the pose and the map disagree; this frame is not added to the map so it cannot stack a second copy")
                    if cfg.map.recover_every and cloud.refiner is not None and st.incons_run % cfg.map.recover_every == 0:
                        pts_cam = cloud.view_points(b["gl"], b["disp"], b["trusted"])
                        R_fix, t_fix, rinfo = cloud.refiner.recover(cloud.map, p.R_wc, p.t_wc, pts_cam, cloud.n)
                        if rinfo["applied"]:                                  # the frame fits the map from a different pose: the odometry was off, not the map
                            provider.correct(R_fix, t_fix)
                            st.counts["map_recoveries"] = st.counts.get("map_recoveries", 0) + 1
                            viz.event(f"t={f.t:.1f}s the pose disagreed with the map and was fitted back to it ({rinfo['shift_cm']:.0f} cm, {rinfo['rot_deg']:.1f} deg, "
                                      f"{rinfo['inlier_frac']:.0%} of the points within 8 cm): the map is kept", "INFO")
                            st.incons_run = 0
                    if cfg.map.reset_after_frames and st.incons_run >= cfg.map.reset_after_frames:
                        if cfg.map.forget_in_view:
                            gone = cloud.forget_in_view(p.R_wc, p.t_wc)
                            scope = f"the part of the map in the camera's view now ({gone} voxels) was dropped, everything behind and beside it is kept"
                        elif cfg.map.forget_radius_m:
                            gone = cloud.forget_near(p.t_wc, cfg.map.forget_radius_m)
                            scope = f"the map within {cfg.map.forget_radius_m:g} m of the camera ({gone} voxels) was dropped, the rest is kept"
                        else:
                            cloud.reset()
                            scope = "the map was dropped"
                        viz.reset_map()
                        if st.tsdf is not None:                              # the volume cannot forget one area: it starts again from here
                            st.tsdf.reset()
                            viz.tsdf_clear()
                        viz.cloud(None, None, cloud.shown_points())
                        st.map_cache, st.map_cache_n, st.incons_run = np.zeros((0, 3)), -99, 0
                        st.counts["map_resets"] += 1
                        viz.event(f"t={f.t:.1f}s the map and the camera pose disagreed for {cfg.map.reset_after_frames} frames: {scope} and rebuilt from here", "WARN")
                        log.warning("pipeline.map_reset", fid=f.fid, scope=scope)
                else:
                    st.incons_run = 0
                    with sw("map"):
                        cloud.on_pose(p.t_ns, p.R_wc, p.t_wc, b["gl"], b["disp"], b["trusted"], extra_exclude=None if held_out is None else held_out["exclude_mask"],
                                      fill=b["fill"], anchor=p.info.get("anchor"))
                    if st.tsdf is not None:
                        with sw("tsdf"):
                            st.tsdf.add(f.left, b["disp"], b["trusted"], p.R_wc, p.t_wc)
                            if st.tsdf.n % cfg.map.tsdf_every == 0 and heavy_draw:
                                m = st.tsdf.mesh()
                                if m is not None:
                                    viz.tsdf_mesh(*m)
                    if st.obj3d is not None:                                # never waits: the frame is handed over or dropped, results are drawn when they are ready
                        if not st.nav_throttled:
                            st.obj3d.submit(f.t, f.left, b["disp"], b["trusted"], f.cal, p.R_wc, p.t_wc, p.info.get("anchor"))
                        snap = st.obj3d.poll()
                        if snap is not None:
                            viz.objects3d(snap)
                    if b["fill"] is not None:
                        st.counts["frames_filled"] = st.counts.get("frames_filled", 0) + 1
                    st.counts["map_frames"] += 1
                    st.vox_peak = max(st.vox_peak, len(cloud.map))
                    if st.counts["map_frames"] % 25 == 0:
                        st.vox_trace.append(len(cloud.map))
                ri = None if inconsistent else cloud.refine_info
                if ri is not None and ri.get("applied"):
                    provider.correct(*cloud.pose)                         # the odometry continues from the corrected pose
                    st.traj[-1] = np.asarray(cloud.pose[1], float)
                    viz.scalar("tracking/map_correction_cm", ri["shift_cm"])
                viz.scalar("tracking/map_voxels", len(cloud.map))
                if heavy_draw and cfg.viz.pov_mesh and hasattr(viz, "pov_mesh"):
                    viz.pov_mesh(f.left, b["fused"], p.R_wc, p.t_wc, f.cal)
            else:
                st.counts["poses_held"] += 1
                if hasattr(viz, "pov_clear"):
                    viz.pov_clear()                         # the pose is not trusted: a surface placed with it would float in the wrong place
                if st.hands_drawn:                                       # this frame's pose cannot be trusted: do not leave hands placed with the old one
                    viz.clear_tracked()
                    st.hands_drawn = False
                viz.pose(p.R_wc, p.t_wc, np.array(st.traj) if st.traj else None)
                viz.event(f"t={f.t:.1f}s pose held, this frame is NOT added to the map: {p.info.get('reason')}", "WARN")
            with sw("viz_features"):
                if heavy_draw:
                    _features(viz, provider, f, b, st.prev_gray, p)
            st.prev_gray = b["gl"]
            tally(checks)
            with sw("status"):
                viz.status(cfg.pipeline.mode, _status_lines(cfg, provider, cal_fr, f, st.last_imu, st.held_status, st.counts), checks)
            st.ms["frame"].append((time.perf_counter() - b["t0"]) * 1000)
            st.ends.append(time.perf_counter())
            for c in checks:
                if c.status == "fail":
                    log.warning_every(f"chk:{c.name}", 5.0, "pipeline.check_fail", fid=f.fid, check=c.name, value=c.value, detail=c.detail, hint=c.hint)

    worker = None
    t_prev_start = None
    period = 1.0 / cfg.perf.fps_cap if cfg.perf.fps_cap > 0 else 0.0
    src_it = iter(src)
    try:
        while True:
            t_wait0 = time.perf_counter()
            fr = next(src_it, None)
            if fr is None:
                break
            sw.add("source", (time.perf_counter() - t_wait0) * 1000)
            t_frame0 = time.perf_counter()
            if cfg.perf.yield_to_navigation and t_frame0 - st.nav_checked >= 1.0:
                st.nav_checked = t_frame0
                active = nav_active()
                if active != st.nav_throttled:
                    st.nav_throttled = active
                    st.held_on = not active
                    msg = (f"navigation viewer is running: this pipeline slows to {cfg.perf.nav_fps_cap:g} fps and the hand/object analysis is off" if active else
                           "navigation viewer has stopped: this pipeline is back to full speed")
                    log.warning("pipeline.nav_yield", active=active, nav_fps_cap=cfg.perf.nav_fps_cap, message=msg)
                    if st.viz is not None:
                        st.viz.event(msg, "WARN")
            eff_period = (1.0 / cfg.perf.nav_fps_cap if st.nav_throttled and cfg.perf.nav_fps_cap > 0 else period)
            if eff_period and t_prev_start is not None:
                spare = eff_period - (t_frame0 - t_prev_start)
                if spare > 0:
                    time.sleep(spare)                                     # frame-rate cap: leave the CPU and GPU to the navigation viewer
                    t_frame0 = time.perf_counter()
            t_prev_start = t_frame0
            st.starts.append(t_frame0)
            st.counts["frames"] += 1
            if st.provider is None:
                cal, h, w = fr.cal, *fr.left.shape[:2]
                st.last_cal = cal
                st.provider = make_provider(cfg, cal)
                factory = viz_factory or (lambda c, s_, sv, vw: Viz_default(c, s_, sv, vw, scale=cfg.perf.viz_scale, map_max=cfg.perf.map_log_max_points))
                st.viz = factory(cal, (w, h), cfg.viz.save or None, cfg.viz.viewer)
                if cfg.perf.async_viz:
                    st.viz = AsyncViz(st.viz, cfg.perf.queue)
                refiner = MapRefiner(max_shift_m=cfg.map.refine_max_shift_m, every=cfg.map.refine_every) if cfg.map.refine_pose and not st.provider.uses_imu else None
                st.refiner = refiner
                if cfg.map.tsdf:
                    from gabai.tsdf import TsdfMap
                    st.tsdf = TsdfMap(cal, (w, h), cfg.map.tsdf_voxel, cfg.map.tsdf_trunc, cfg.map.max_range, cfg.map.max_depth_std)
                st.obj3d = obj3d                                       # loaded and warmed up before the source was opened
                if obj3d is not None and cfg.objects3d.graph and getattr(st.provider, "vo", None) is not None:
                    st.provider.vo.landmark_source = obj3d.graph_obs      # objects seen again from far-apart keyframes become factors of the pose graph
                st.cloud = SlamCloud(cal["f"], cal["B"], cal["cx"], cal["cy"], sink=st.viz, voxel=cfg.map.voxel, semi_dense=cfg.map.semi_dense, stride=cfg.map.stride,
                                     max_range=cfg.map.max_range, map_every=cfg.perf.map_log_every, weighted=cfg.map.weighted, max_depth_std=cfg.map.max_depth_std,
                                     refiner=refiner, core=cfg.map.core, core_faces_only=True, max_voxels=cfg.map.max_voxels, carve_m=cfg.map.carve_m, fill_weight=cfg.map.fill_weight, carve_decay=cfg.map.carve_decay)
                st.cloud.map.window_m = cfg.map.window_m or None
                if st.held_obj is None and (cfg.hands.enabled or cfg.detector.enabled):
                    st.held_obj, notes = build_held(cfg, cal, held_job.result if held_job is not None else None)
                ho = st.held_obj
                st.held_status = {"hands": "on" if ho is not None and ho.hands is not None else "off", "objects": "on" if ho is not None and ho.det is not None else "off"}
                geo = D.stereo_geometry_report(cal["f"], cal["B"], cal["metric"], cal["source"])
                log.info("pipeline.geometry", **geo, mode=cfg.pipeline.mode, world=st.provider.world, sync=fr.sync, held=st.held_status)
                st.viz.at(0, fr.t)
                st.viz.event(f"mode {cfg.pipeline.mode}; {geo['note']}", "INFO")
                st.viz.event("'valid / rejected' colours: green = depth kept, blue = outside the right picture (closer = wider band), grey = no texture to match, "
                             "red = left/right disagree (occlusion, repeated pattern, motion between the two exposures), orange = small island removed, violet = monocular veto", "INFO")
                for wmsg in warns:
                    st.viz.event(wmsg, "WARN")
                if staged:
                    worker = StageWorker(stage_b, cfg.perf.stage_queue, cfg.perf.drop_when_busy)
            viz = st.viz
            viz.at(fr.idx, fr.t)
            with sw("gray"):
                gl, gr = cv2.cvtColor(fr.left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr.right, cv2.COLOR_BGR2GRAY)
            with sw("depth"):
                disp, trusted, depth_ms = depth_fn(fr.left, fr.right)
            st.ms["depth"].append(depth_ms)
            valid = np.isfinite(disp) & (disp > 0.5)
            vf = float(valid.mean())
            tf = float((valid & trusted).sum() / max(valid.sum(), 1))
            heavy_draw = fr.idx % max(cfg.perf.viz_every, 1) == 0
            with sw("viz_stereo"):
                if heavy_draw:
                    viz.stereo(fr.left, fr.right, disp, trusted, fr.cal, sample_correspondences(disp, trusted) if fr.idx % 5 == 0 else None)
            fill = getattr(depth_fn, "filled", None)
            if fill is not None and cfg.map.fill_reach_px:                  # a hole far from any measured pixel stays empty
                reach = cv2.dilate(trusted.astype(np.uint8), np.ones((2 * cfg.map.fill_reach_px + 1,) * 2, np.uint8)).astype(bool)
                fill = (fill[0], fill[1] & reach, fill[2])
            fused = fused_for_view(disp, fill)
            fused_view = fused                                                # the panel shows stereo + monocular before the depth-edge mask, so its gaps are only real ones
            surf = surface_mask(fused, fr.cal) if cfg.map.surface_filter else None      # the same pixels feed the map and the POV mesh
            if surf is not None:
                fused = np.where(surf, fused, np.nan).astype(np.float32)
                if fill is not None:
                    fill = (fill[0], fill[1] & surf, fill[2])
            if heavy_draw and hasattr(viz, "fusion"):
                viz.fusion(fused_view, getattr(depth_fn, "mono_view", None), fr.cal, getattr(depth_fn, "mono_raw", None), getattr(depth_fn, "mono_raw_r", None))
            why = getattr(depth_fn, "reasons", None)
            if why is not None:
                fr_why = Q.reason_fractions(why)
                for name, v in fr_why.items():
                    viz.scalar("depth/frac_" + name.replace(" ", "_").replace("/", "_"), v)
                    st.why_sum[name] = st.why_sum.get(name, 0.0) + v
                st.why_n += 1
                if heavy_draw and hasattr(viz, "rejects"):
                    viz.rejects(why)
            if st.prev_t is not None:
                gap_ms = (fr.t - st.prev_t) * 1000.0
                viz.scalar("timing/pair_gap_ms", gap_ms)
                st.gaps.append(gap_ms)
                if gap_ms > 300:
                    log.warning_every("stutter", 5.0, "pipeline.stutter", gap_ms=round(gap_ms), fid=fr.fid,
                                      hint="no pair for this long: the Pi or the link stalled (power: see pi.power in the Pi log; Wi-Fi: send_slow). Tracking across the gap is unreliable")
            st.prev_t = fr.t
            viz.scalar("depth/picture_mean_left", float(gl.mean()))
            viz.scalar("depth/picture_mean_right", float(gr.mean()))
            if fr.sync.get("pair_dt_ms") is not None:
                viz.scalar("depth/pair_offset_ms", float(fr.sync["pair_dt_ms"]))
            if (fr.sync.get("skew_comp") or {}).get("shift_px") is not None:
                viz.scalar("depth/offset_correction_px", float(fr.sync["skew_comp"]["shift_px"]))
            viz.scalar("depth/valid_fraction", vf)
            viz.scalar("depth/trusted_fraction_of_valid", tf)
            viz.scalar("depth/model_ms", depth_ms)
            if valid.any():
                viz.scalar("depth/median_m", float(np.median(fr.cal["f"] * fr.cal["B"] / disp[valid])))
            checks_a = [c for c in (C.check_depth_coverage(vf), D.check_stereo_sync(fr.sync, None, fr.cal["f"])) if c is not None]
            bundle = {"fr": fr, "gl": gl, "gr": gr, "disp": disp, "trusted": trusted if surf is None else trusted & surf, "fused": fused, "t_ns": int(round(fr.t * 1e9)), "checks": checks_a, "t0": t_frame0, "heavy_draw": heavy_draw,
                      "fill": fill}
            if worker is not None:
                worker.submit(bundle)
            else:
                stage_b(bundle)
    finally:
        if worker is not None:
            worker.close()
    if isinstance(src, Prefetch):
        src.close()
    if st.provider is not None:
        closer = threading.Thread(target=st.provider.close, name="provider-close", daemon=True)       # Basalt's native stop can block; waiting for it must not make the run unstoppable
        closer.start()
        closer.join(5.0)
        if closer.is_alive():
            log.warning("pipeline.provider_close_timeout", hint="the pose estimator did not stop within 5 s; leaving it")
    viz_stats = None
    if isinstance(st.viz, AsyncViz):
        st.viz.close()
        viz_stats = st.viz.stats()
    counts, ms, align_hist, traj = st.counts, st.ms, st.align_hist, st.traj
    if st.cloud is not None and cfg.map.save:
        from gabai.relocalize import save_map
        save_map(cfg.map.save, st.cloud.map, {"mode": cfg.pipeline.mode, "depth_is_metric": bool(st.last_cal["metric"])})
    rs = None if st.refiner is None else {k: (v if not isinstance(v, list) else (None if not v else {"median": round(float(np.median(v)), 3), "max": round(float(np.max(v)), 3)}))
                                           for k, v in st.refiner.stats.items()}
    path = float(np.linalg.norm(np.diff(np.array(traj), axis=0), axis=1).sum()) if len(traj) > 1 else 0.0
    summary = {"mode": cfg.pipeline.mode, "source": cfg.source.kind, **counts, "poses_without_frame_pending": len(st.pending), "path_m": round(path, 3),
               "path_is_metric": bool(st.last_cal["metric"]) if st.last_cal else None, "map_voxels": len(st.cloud.map) if st.cloud else 0, "map_voxels_peak": st.vox_peak, "map_voxels_every_25_frames": st.vox_trace, "map_voxels_evicted": st.cloud.map.evicted if st.cloud else 0, "map_voxels_carved": st.cloud.carved if st.cloud else 0,
               "map_solid_voxels": int((st.cloud.map.n >= 3).sum()) if st.cloud else 0, "map_obs_per_voxel_median": float(np.median(st.cloud.map.n)) if st.cloud and len(st.cloud.map) else None,
               "map_alignment_median_cm": None if not align_hist else {"median": round(float(np.median(align_hist)) * 100, 2), "last": round(align_hist[-1] * 100, 2),
                                                                      "first": round(align_hist[0] * 100, 2)},
               "checks": st.check_tally, "depth_ms_median": None if not ms["depth"] else round(float(np.median(ms["depth"])), 1),
               "frame_ms_median": None if not ms["frame"] else round(float(np.median(ms["frame"])), 1),
               "fps_steady": None if len(st.ends) < 3 else round((len(st.ends) - 1) / (st.ends[-1] - st.ends[0]), 2),
               "frame_ms_p95": None if not ms["frame"] else round(float(np.percentile(ms["frame"], 95)), 1), "warnings": warns, "models": st.held_status, "log": run_dir,
               "imu_used": st.provider.uses_imu if st.provider else None, "hands3d": st.hs, "map_refine": rs, "vo": getattr(getattr(st.provider, "vo", None), "n_keyframes", None), "vo_stats": getattr(getattr(st.provider, "vo", None), "stats", None), "stage_ms": sw.report(), "viz": viz_stats,
               "pair_timing_ms": None if not st.gaps else {"gap_median": round(float(np.median(st.gaps)), 1), "gap_p95": round(float(np.percentile(st.gaps, 95)), 1), "gap_max": round(float(max(st.gaps)), 1),
                                                           "gaps_over_200ms": int(sum(g > 200 for g in st.gaps)), "gaps_over_500ms": int(sum(g > 500 for g in st.gaps))},
               "depth_pixels_by_cause": None if not st.why_n else {k: round(v / st.why_n, 4) for k, v in st.why_sum.items()},
               "perf": {"settings": vars(cfg.perf), "notes": perf_notes, "staged": staged, "frames_dropped_stage_b": worker.dropped if worker else 0,
                        "navigation_seen_running": st.nav_throttled}}
    log.info("pipeline.summary", **summary)
    json.dump(summary, open(os.path.join(run_dir, "pipeline_summary.json"), "w", encoding="utf-8"), indent=2, default=str)
    dlog.shutdown()
    return summary


def Viz_default(cal, size, save, viewer, scale=0.5, map_max=60000):
    from gabai.pipeline.viz import Viz
    return Viz(cal, size, save, viewer, scale=scale, map_max_points=map_max)


def _features(viz, provider, f, b, prev_gray, p):
    if getattr(getattr(provider, "vo", None), "last_flow", None) is not None:
        a, c = provider.vo.last_flow
        viz.features(f.left, c, [np.array([x, y]) for x, y in zip(a, c)])
    else:
        viz.features(f.left)                                            # no features were tracked in this frame: nothing is drawn rather than something invented
    if getattr(f, "klt", None) is not None or "buf" in f.sync:
        viz.klt(f.left, f.klt, f.right)


def _status_lines(cfg, provider, cal, f, last_imu, held_status, counts):
    if provider.uses_imu:
        imu_line = "fused with the pictures (gyro prior, gravity-aligned world, rest hold), " + (str(last_imu.get("n", 0)) + " samples this frame" if last_imu else "no samples yet")
    elif getattr(provider, "wants_gyro", False):
        imu_line = "gyro only: its rotation between frames steers the feature search; the pose comes from the pictures"
    else:
        imu_line = "NOT used in this mode (not opened)"
    L = [f"- frame {f.idx} (id {f.fid}), t = {f.t:.2f} s",
         f"- depth: {'METRIC (calibrated)' if cal['metric'] else 'NOT metric, scale assumed'}; {cal['source']}",
         f"- pose: {provider.name}; world: {provider.world}",
         "- IMU: " + imu_line,
         f"- sync: {f.sync.get('kind')} - {f.sync.get('note')}",
         f"- poses ok {counts['poses_ok']}, held {counts['poses_held']}, frames in map {counts['map_frames']}; hands {held_status['hands']}, objects {held_status['objects']}"]
    cam = f.sync.get("cam") or {}
    if f.sync.get("kind") == "live":
        L.append("- camera exposure: " + (f"{cam['exposure_ms']:g} ms (cap {cam.get('exposure_cap_ms')} ms), gain {cam.get('gain')}, picture mean {cam.get('picture_mean')}" if cam.get("exposure_ms") is not None
                                          else "NOT reported: older sender or --exposure auto (auto runs long in a dim room: a moving scene smears)"))
    if f.sync.get("power"):
        L.append("- " + f.sync["power"])
    if f.sync.get("row_alignment_px_after") is not None:
        L.append(f"- plain video row alignment fix: residual {f.sync['row_alignment_px_before']:.1f} px before, {f.sync['row_alignment_px_after']:.1f} px after")
    return L

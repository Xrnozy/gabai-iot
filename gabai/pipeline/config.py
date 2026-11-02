"""Pipeline configuration and the checks that decide whether a chosen mode can run.

Two modes, one output shape (see gabai/pipeline/poses.py):
  stereo_only   no IMU is opened, read or needed
  stereo_imu    the stereo pictures as in stereo_only plus the IMU (gyro prior, gravity-aligned world, rest hold; gabai/pipeline/vio.py)

validate() returns (errors, warnings). An error means the mode cannot run as asked (the run stops and says why); a warning means it can run but
a result may be poor. It checks what can be checked from files: the IMU file exists and covers the camera time, its rate and gaps, that the
camera-IMU extrinsics are present and marked calibrated. It cannot check that the extrinsics are RIGHT; that needs the
motion check in docs/rerun-pipeline.md.
"""
import os
import tomllib
from dataclasses import dataclass, field, fields

import numpy as np

MODES = ("stereo_only", "stereo_imu")
SOURCES = ("video", "recording", "live", "replay")


@dataclass
class PipelineSection:
    mode: str = "stereo_only"


@dataclass
class SourceSection:
    kind: str = "video"
    left: str = "data/left.mp4"
    right: str = "data/right.mp4"
    folder: str = ""
    start: float = 0.0
    skip: int = 2
    replay_skip: int = 1             # replay of a recorded pair folder: use every n-th pair (1 = all); n = 4 stands in for a link that delivers a quarter of the frames
    max_frames: int = 300
    port: int = 5600                 # live only: where the Pi sender connects
    token: str = "gabai"
    calib: str = "data/stereo_calib_new.npz"
    imu_sync: str = "data/imu_sync.json"
    accel_calib: str = "data/accel_calib.json"
    rotate_left: int = -1            # live only: -1 = the rotation stored in the stereo calibration; 0/90/180/270 overrides it
    rotate_right: int = -1
    auto_layout: bool = True         # live / replay: if the pictures only line up (rows within 0.5 px) after both are turned 180 deg and swapped, the two cameras came out in the other order (the /dev/video numbers moved): do that
    flip180: bool = False            # live only: turn the rectified pictures 180 deg (the two pictures trade places, so the geometry stays valid and nothing is recalibrated); use it when the view is upside down
    skew_comp: bool = False          # undo the left/right capture-time offset the Pi measures (gabai/skewcomp.py). OFF: on your own recordings it raised the valid share a little but doubled the frame-to-frame depth noise and did not steady the map; try it with scripts/evaluate_recording.py
    live_width: int = 640            # live only: rectified pairs wider than this are shrunk to it (INTER_AREA averages noise); the camera parameters are scaled to match


@dataclass
class StereoSection:
    model: str = "s"
    width: int = 640
    f_frac: float = 0.6
    baseline: float = 0.065
    rig_baseline_mm: float = 42.0    # live only: lens-centre spacing of the printed case (Fusion 360 model); a calibration more than 5% away from it is reported; 0 = no check
    scale: float = 1.0               # the pictures are enlarged by this factor before LAS2 and the result brought back (disparity divided by it): 1.5 costs about 40 ms more with LAS2-H and halves the large errors; the nearest measurable distance grows by the same factor
    speckle_px: int = 100            # trusted disparity islands smaller than this (px) that jump from their surroundings are dropped; 0 = off
    speckle_diff: float = 1.0        # disparity step (px) that separates one island from its neighbours
    mono: str = ""                   # monocular metric depth model: "" = off, "dav2" = Depth Anything V2 Metric Indoor Small
    mono_mode: str = "blend"         # fill: the monocular map, scaled to the trusted stereo pixels, supplies ONLY the pixels stereo did not measure (marked, down-weighted in the map; stereo pixels are untouched); blend: trusted stereo pixels are averaged with the aligned monocular map by each one's noise (holes stay unknown); check: it only vetoes stereo pixels that grossly disagree
    mono_pair: bool = True           # fill mode: the monocular model also runs on the RIGHT picture; the right map, put into the left view, must agree with the left one (within 2 px) for a hole pixel to be filled, and where they agree the two are averaged
    depth_correction: str = "data/calibration/depth_correction.json"   # distance correction of the stereo model fitted on the calibration board pairs (py -m gabai.tools.fit_depth_correction); a missing file or model entry = no correction; "" = off


@dataclass
class ImuSection:
    still_gate: str = "auto"


@dataclass
class MapSection:
    voxel: float = 0.05
    max_range: float = 4.0
    semi_dense: bool = True
    stride: int = 3
    weighted: bool = True            # a precise (near) point counts for more than a noisy (far) one in each voxel's mean: weight 1 / depth variance
    max_depth_std: float = 0.0       # drop points whose depth standard deviation (z^2 * 0.3 px / (f * B)) exceeds this many metres; 0 = no limit
    max_voxels: int = 300000         # hard cap on map size (about 14 MB): voxels seen under 3 times go first, then the farthest from the camera; 0 = no cap
    surface_filter: bool = True      # the map and the camera-POV mesh take only pixels on a real connected surface (valid depth, not at a depth edge, not edge-on, not a small island): no flying points between surfaces
    fill_reach_px: int = 40          # the monocular fill only supplies holes within this many pixels of a trusted stereo pixel: a big empty area (sky, dark corner) stays empty; 0 = no limit
    max_pose_step_m: float = 0.5     # a pose that jumps farther than this between two frames is not trusted and its frame is not added to the map; 0 = off
    max_misalign_cm: float = 15.0    # a frame whose pose puts the map more than this far (median) from its own measured depth is NOT added to the map: stacking it would make a second copy; 0 = off
    reset_after_frames: int = 30     # this many such frames in a row: the map around the camera (forget_radius_m) is dropped and rebuilt from the current pose (tracking and map no longer agree); 0 = never
    recover_every: int = 3           # while frames are being kept out of the map for disagreeing with it, every Nth such frame tries to register itself to the map from far away (up to 1 m / 35 deg) and fix the pose; 0 = never
    forget_in_view: bool = True      # if that fails for reset_after_frames frames: drop only the map inside the camera's view now (False: the sphere of forget_radius_m, which is the whole room indoors)
    forget_radius_m: float = 3.0     # that drop covers only this radius around the camera, the rest of the map stays; 0 = the whole map (the old behaviour)
    core: float = 0.2                # shown map: drop a voxel seen under this fraction as often as its busiest neighbour (smeared layers). 0.7 kept only 7% of a real map
    window_m: float = 8.0            # the map keeps voxels within this distance of the camera: once the camera is farther than this from a point, the point is forgotten (bounds memory on a long walk); 0 = keep everything
    carve_m: float = 0.3             # a voxel the camera itself passes through (closer than this) is removed: a camera cannot be inside a solid; 0 = off
    carve_decay: float = 0.6         # a stored voxel the camera now sees THROUGH (the depth around its pixel is farther, by more than the noise) keeps this fraction of its evidence per look and goes when none is left: a moved or removed thing leaves the map; 0 = off (points then stay until the camera is far away)
    fill_weight: float = 0.3         # a point filled in from the monocular map counts this much of a stereo point of the same distance in a voxel's mean (stereo.mono_mode = "fill")
    refine_pose: bool = True         # stereo_only: register each frame to the map and apply a small correction (see gabai/mapref.py); ignored in stereo_imu
    refine_max_shift_m: float = 0.06 # a correction larger than this is not trusted and not applied
    refine_every: int = 2            # try the registration every Nth map frame
    tsdf: bool = False               # also fuse the depth into a TSDF volume (Open3D, gabai/tsdf.py) and draw its mesh in Rerun as world/tsdf_mesh: repeated looks average out depth noise. Needs a good pose
    tsdf_voxel: float = 0.02         # metres per TSDF voxel (finer = more sensitive to depth noise)
    tsdf_trunc: float = 0.08         # truncation distance of the signed distance (at least 3 to 4 voxels, and about the depth noise at the working range)
    tsdf_every: int = 30             # redraw the mesh every Nth map frame (marching cubes is the slow part)
    save: str = ""                   # write the fused map to this .npz at the end (gabai.relocalize.load_map reads it); empty = do not


@dataclass
class TrackingSection:
    vo: str = "keyframe"             # stereo_only pose: keyframe = gabai.kfvo.KeyframeVO (less drift, faster); frame = gabai.tools.video_cloud_view.StereoVO (the original)
    fuse_depth: bool = True          # keyframe VO: average each feature's stereo depth over the frames that see it
    predict: bool = True             # keyframe VO: start the feature search at the position the last motion predicts (guided search; keeps tracking through fast turns)
    refine: bool = True              # keyframe VO: refine each pose with Levenberg-Marquardt on the inliers (motion-only bundle adjustment)
    relocalize: bool = True          # keyframe VO: after lost tracking, find the pose again from ORB features against the stored keyframes
    initial_pitch_deg: float = 0.0   # keyframe VO: how far the camera looks BELOW the horizon at the start (a chest mount is usually 5-15). The SLAM world has z up only if this is right; py -m gabai.tools.frame_check prints it from the IMU
    vo_uses_fill: bool = False       # keyframe VO: the monocular fill (stereo.mono_mode = fill) also gives depth to features where stereo has none, so the tracker has more points to follow; the pose estimator drops the ones that disagree
    gyro_prior: bool = False         # keyframe VO (stereo_only otherwise reads no IMU): when the source brings gyro samples (a live Pi stream, or a recording with imu0) and the camera-IMU rotation is known (calib.json), the gyro rotation between two frames starts the feature search: the picture is followed through fast turns. Off = pictures only
    window_ba: bool = True           # keyframe VO: sliding-window bundle adjustment on a worker thread (gabai/localba.py); landmarks seen from several keyframes get parallax-based depth
    ba_every: int = 3                # run it every Nth keyframe (every keyframe: more accurate, but the worker cannot keep up)
    ba_window: int = 6
    ba_max_landmarks: int = 250
    loop_closure: bool = False       # keyframe VO: detect a return to an old keyframe and measure the drift (reported as an event)
    loop_apply: bool = False         # ...and correct the trajectory: a verified, confirmed loop becomes an edge of the pose graph, ALL keyframe poses are re-optimised, and the map is re-laid from them (SlamCloud.rebuild). Needs loop_closure
    loop_radius_m: float = 1.0       # a loop is looked for only among keyframes this close to where the camera is believed to be
    loop_min_path_m: float = 2.0     # ...and at least this much travel earlier
    loop_appearance: bool = True     # also look among ALL old keyframes for the place by its ORB features, wherever the odometry thinks they are (verified by PnP before it is used)
    loop_every: int = 2              # check every Nth new keyframe
    loop_max_drift_m: float = 2.0    # a loop that would move the pose farther than this (or 0.3 m + 15% of the path since that keyframe, whichever is less) is a look-alike and is refused


@dataclass
class KltSection:
    enabled: bool = False            # stereo_imu + live: also use the feature tracks the Pi sends (sender --klt) to correct the TRANSLATION (gabai/kltfuse.py). The rotation stays the IMU's (the picture rotation only where the IMU has none). Off = the pose is exactly what it was
    weight: float = 0.75             # share of the KLT translation in the fused translation while it agrees with the PC tracker (0.5 equal, 1 KLT only)
    min_inliers: int = 15            # a KLT translation needs this many tracked points that agree on it...
    max_reproj_px: float = 1.5       # ...within this many pixels (RMS)
    agree_m: float = 0.08            # the two translations agree within this distance; farther apart, the pose still moves toward KLT, but at most max_step_m per frame
    max_step_m: float = 0.02         # the most one frame's KLT correction may move the pose (no sudden jump: a larger difference is closed over several frames)


@dataclass
class FilterSection:
    enabled: bool = False            # stereo_imu: run the error-state Kalman filter (gabai/vioekf.py) on the pose: IMU prediction, the pictures' pose as a gated measurement, the Pi's own motion (sender --klt-motion) as velocity and angular-rate measurements. Smooths the pose and refuses outliers; the pictures and the IMU are used exactly as before, the filter only shapes the reported pose. Replaces klt.enabled's direct fusion (both would use the same tracks)
    pos_sigma_m: float = 0.02        # noise of one picture pose, position
    att_sigma_deg: float = 1.0       # ...and orientation
    klt_sigma_m: float = 0.012       # noise of the Pi's motion over one pair interval (smaller = KLT trusted more)
    klt_sigma_deg: float = 0.3
    gyro_noise: float = 0.004        # rad/s per IMU sample: small, so the IMU stays the main source of rotation
    acc_noise: float = 0.1           # m/s^2 per IMU sample
    max_dev_cm: float = 15.0         # the reported pose is the filter's only while it is within this distance...
    max_dev_deg: float = 5.0         # ...and this angle of the raw picture pose of the same frame; past that the raw pose is reported (the filter can then never spoil the map by more than this)
    gate_chi2: float = 16.27         # a picture pose farther than this (3 dof, 99.9%) from the prediction is refused as an outlier; five refusals in a row reset the filter to the pictures


@dataclass
class SyncSection:
    enabled: bool = True             # live: hand the pipeline complete, ordered measurement groups (pair + its KLT points + every IMU sample of its interval; gabai/pc/syncbuf.py). Off = the newest pair, IMU read before it
    max_lag_s: float = 1.0           # the pipeline is allowed to fall this far behind the newest pair; past it the oldest PICTURES are skipped (counted; their IMU stays and is integrated). 0 = never skip
    keep_lag_s: float = 0.3          # ...down to this much backlog
    imu_wait_ms: float = 150.0       # a pair waits at most this long for the IMU samples up to its time (they normally arrive first); then it goes out marked incomplete
    klt_wait_ms: float = 150.0       # ...and for its KLT points (only when the sender sends them)
    max_groups: int = 120            # hard bound of the queue (about 16 MB of JPEG)


@dataclass
class HandsSection:
    enabled: bool = True


@dataclass
class DetectorSection:
    enabled: bool = True


@dataclass
class Objects3dSection:
    enabled: bool = False            # static-object landmarks from Boxer (gabai/objects3d.py): drawn in Rerun as world/boxer/*, never used to correct the pose. Needs third_party/boxer weights; costs GPU time on a background thread
    every_s: float = 0.3             # seconds between frames handed to Boxer (it takes about 135 ms a frame; the frame loop never waits for it)
    max_range: float = 2.0           # ignore objects farther than this (stereo depth noise at 41.7 mm baseline is about 10 cm at 3 m)
    gate: float = 0.25               # a look joins an object of the same class whose median is within this many metres
    min_looks: int = 5               # an object is drawn as a landmark after this many looks
    labels: str = "cabinet,refrigerator,table,desk,sofa,chair,tv,monitor,shelf,bookshelf,bed,sink,oven,microwave,laptop,keyboard,plant"     # text prompts for OWLv2, comma separated
    thresh2d: float = 0.25
    thresh3d: float = 0.5
    graph: bool = False              # objects seen again from far-apart keyframes become factors of the pose graph (poses, map and objects are corrected together). OFF: on the desk clip it tightened the objects but did not reduce the independently measured rotation drift, and it re-lays the whole map up to 57 times


@dataclass
class PerfSection:
    async_viz: bool = True           # draw in Rerun on its own thread; the compute loop never waits for it
    queue: int = 6                   # heavy picture calls waiting for the drawing thread before new ones are dropped (and counted)
    viz_scale: float = 0.5           # the disparity, depth, valid and feature panels are drawn at this fraction of the picture size
    viz_every: int = 1               # draw the 2D panels every Nth frame (1 = every frame)
    map_log_every: int = 10          # send the fused map to Rerun every Nth map frame
    map_log_max_points: int = 250000 # at most this many map points per send
    detector_every: int = 3          # run the object detector every Nth frame (the last detections are reused, marked stale)
    low_priority: bool = True        # run below normal priority so a navigation viewer on the same PC gets the CPU first
    threads: int = 0                 # cap OpenCV and torch CPU threads (0 = leave alone)
    fps_cap: float = 0.0             # process at most this many frames per second (0 = as fast as possible)
    prefetch: bool = True            # read and decode the source on its own thread
    staged: bool = True              # stage A (depth) and stage B (pose, map, hands, drawing) on separate threads, so they overlap; needs async_viz
    stage_queue: int = 1             # frames waiting between the two stages
    yield_to_navigation: bool = True # while the navigation viewer is running on this PC (it writes data/nav_active.json), cap to nav_fps_cap and switch hands/objects off
    nav_fps_cap: float = 3.0         # the frame rate while navigation is running
    drop_when_busy: bool = False     # True for a live stream: when stage B is behind, skip the frame instead of waiting (counted). False for recordings: every frame is processed


@dataclass
class VizSection:
    save: str = ""
    viewer: bool = True
    pov_mesh: bool = False           # also draw the textured mesh of the CURRENT frame (world/pov_mesh) in the POV view: a second surface over the fused one, and one more thing to draw every frame


@dataclass
class PipelineConfig:
    pipeline: PipelineSection = field(default_factory=PipelineSection)
    source: SourceSection = field(default_factory=SourceSection)
    stereo: StereoSection = field(default_factory=StereoSection)
    imu: ImuSection = field(default_factory=ImuSection)
    map: MapSection = field(default_factory=MapSection)
    tracking: TrackingSection = field(default_factory=TrackingSection)
    klt: KltSection = field(default_factory=KltSection)
    sync: SyncSection = field(default_factory=SyncSection)
    filter: FilterSection = field(default_factory=FilterSection)
    hands: HandsSection = field(default_factory=HandsSection)
    detector: DetectorSection = field(default_factory=DetectorSection)
    objects3d: Objects3dSection = field(default_factory=Objects3dSection)
    viz: VizSection = field(default_factory=VizSection)
    perf: PerfSection = field(default_factory=PerfSection)

    @classmethod
    def load(cls, path=None, overrides=None):
        c = cls()
        data = {}
        if path:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        for sec, vals in data.items():
            c._set_section(sec, vals)
        for dotted, v in (overrides or {}).items():
            sec, key = dotted.split(".", 1)
            c._set_section(sec, {key: v})
        return c

    def _set_section(self, sec, vals):
        if not hasattr(self, sec):
            raise ValueError(f"unknown section [{sec}]; sections are: {', '.join(f.name for f in fields(self))}")
        obj = getattr(self, sec)
        names = [f.name for f in fields(obj)]
        for k, v in vals.items():
            if k not in names:
                raise ValueError(f"unknown key '{k}' in [{sec}]; keys are: {', '.join(names)}")
            setattr(obj, k, type(getattr(obj, k))(v) if not isinstance(getattr(obj, k), bool) else bool(v))


def _imu_stats(path):
    d = np.loadtxt(path, delimiter=",", comments="#")
    if d.ndim != 2 or len(d) < 2:
        return None
    t = d[:, 0] * 1e-9
    dt = np.diff(t)
    return {"n": len(t), "t0": float(t[0]), "t1": float(t[-1]), "rate_hz": float(1.0 / np.median(dt)), "max_gap_ms": float(dt.max() * 1000)}


def _validate_live(cfg, errs, warns):
    src = cfg.source
    for name in ("rotate_left", "rotate_right"):
        if getattr(src, name) not in (-1, 0, 90, 180, 270):
            errs.append(f"source.{name} = {getattr(src, name)} is not -1 (as calibrated), 0, 90, 180 or 270")
    if not os.path.exists(src.calib):
        errs.append(f"stereo calibration not found: {src.calib} (run gabai.tools.calibrate_stereo)")
    else:
        z = np.load(src.calib)
        if float(z["T"].ravel()[0]) > 0:
            errs.append(f"{src.calib} is SWAPPED: its 'right' camera sits on the left, so depth would be wrong. Swap the sender's --left/--right devices (and their rotations) and recalibrate")
        elif cfg.stereo.rig_baseline_mm > 0:
            from gabai.tools.calibrate_stereo import check_baseline
            msg = check_baseline({"baseline": float(z["baseline"])}, float(z["square_mm"]) if "square_mm" in z.files else 0.0, cfg.stereo.rig_baseline_mm)
            if "DOES NOT AGREE" in msg:
                warns.append(f"{src.calib}: {msg}")
    if cfg.pipeline.mode == "stereo_imu":
        if not os.path.exists(src.imu_sync):
            warns.append(f"{src.imu_sync} not found: the camera-IMU rotation is a guess: the gyro would turn the wrong way (run gabai.tools.sync_imu_camera)")
        elif os.path.exists(src.calib) and os.path.getmtime(src.imu_sync) < os.path.getmtime(src.calib):
            warns.append(f"{src.imu_sync} is older than {src.calib}: the cameras were calibrated again since the camera-IMU rotation was measured, so it may no longer fit this rig. "
                         "Re-run gabai.tools.sync_imu_camera (then check_extrinsics --write); stereo_only does not use it")
    return errs, warns


def validate(cfg, check_files=True):
    errs, warns = [], []
    mode, src = cfg.pipeline.mode, cfg.source
    if mode not in MODES:
        errs.append(f"mode '{mode}' is not one of {MODES}")
        return errs, warns
    if src.kind not in SOURCES:
        errs.append(f"source.kind '{src.kind}' is not one of {SOURCES}")
        return errs, warns
    if not 1.0 <= cfg.stereo.scale <= 2.0:
        errs.append(f"stereo.scale {cfg.stereo.scale} is not between 1.0 and 2.0")
    if cfg.stereo.mono not in ("", "dav2"):
        errs.append(f"stereo.mono '{cfg.stereo.mono}' is not '' or 'dav2'")
    if cfg.stereo.mono_mode not in ("blend", "check", "fill"):
        errs.append(f"stereo.mono_mode '{cfg.stereo.mono_mode}' is not 'blend', 'check' or 'fill'")
    if cfg.tracking.vo not in ("keyframe", "frame"):
        errs.append(f"tracking.vo '{cfg.tracking.vo}' is not 'keyframe' or 'frame'")
    if cfg.klt.enabled:
        if mode != "stereo_imu" or src.kind != "live":
            errs.append("klt.enabled needs pipeline.mode = \"stereo_imu\" and source.kind = \"live\" (the points come from a Pi sender started with --klt)")
        if not 0.0 <= cfg.klt.weight <= 1.0:
            errs.append(f"klt.weight {cfg.klt.weight} is not between 0 and 1")
    if cfg.filter.enabled:
        if mode != "stereo_imu":
            errs.append("filter.enabled needs pipeline.mode = \"stereo_imu\" (the filter integrates the IMU)")
        if cfg.klt.enabled:
            warns.append("filter.enabled and klt.enabled both use the Pi's tracks: the filter uses the Pi's motion estimate and klt.enabled's direct translation fusion is switched off, so the same pictures are not counted twice")
        if src.kind == "live":
            warns.append("filter.enabled: for the Pi's motion to reach the filter start the sender with --klt-motion (without it the filter runs on the IMU and the pictures only)")
    if mode == "stereo_imu" and cfg.map.refine_pose:
        warns.append("map.refine_pose is ignored in stereo_imu: the gravity-aligned poses are used as they are")
    if mode == "stereo_imu" and src.kind == "video":
        errs.append("stereo_imu needs an IMU, and plain video files have none. Use source.kind = \"recording\" (a folder with imu0), or switch to stereo_only.")
    if src.kind == "video":
        warns.append(f"plain video has no calibration: focal length ({cfg.stereo.f_frac} x width) and baseline ({cfg.stereo.baseline} m) are assumed, so depth is not metric "
                     "and the scale of the path and map is unknown")
    if not check_files:
        return errs, warns
    if src.kind == "replay":
        if mode == "stereo_imu":
            errs.append("a recording made by gabai.tools.record_pairs holds no IMU data: replay it with mode stereo_only")
        for name in ("meta.json", "left.avi", "right.avi", "pairs.csv"):
            if not os.path.exists(os.path.join(src.folder or "", name)):
                errs.append(f"{os.path.join(src.folder or '', name)} not found: source.folder must be a folder written by gabai.tools.record_pairs")
        return _validate_live(cfg, errs, warns) if not errs else (errs, warns)
    if src.kind == "live":
        return _validate_live(cfg, errs, warns)
    if src.kind == "video":
        for p in (src.left, src.right):
            if not os.path.exists(p):
                errs.append(f"video file not found: {p}")
        return errs, warns
    if not src.folder or not os.path.isdir(src.folder):
        errs.append(f"recording folder not found: '{src.folder}'")
        return errs, warns
    cal = os.path.join(src.folder, "calib.json")
    if not os.path.exists(cal):
        errs.append(f"{cal} not found")
        return errs, warns
    if mode == "stereo_only":
        return errs, warns                                           # the IMU is not touched in this mode
    from json import load
    c = load(open(cal))["value0"]
    if len(c.get("T_imu_cam") or []) < 2:
        errs.append(f"{cal} has no T_imu_cam (camera-IMU extrinsics) for both cameras; run gabai.tools.sync_imu_camera and check_extrinsics")
    info_p = os.path.join(src.folder, "recording_info.json")
    info = load(open(info_p)) if os.path.exists(info_p) else {}
    if not info.get("imu_extrinsics_calibrated", False):
        warns.append("camera-IMU extrinsics are not calibrated (recording_info.json says so, or is missing): the gyro prior will turn the wrong way. Run gabai.tools.sync_imu_camera, then check_extrinsics --write")
    imu_p = os.path.join(src.folder, "mav0", "imu0", "data.csv")
    cam_p = os.path.join(src.folder, "mav0", "cam0", "data.csv")
    if not os.path.exists(imu_p):
        errs.append(f"IMU file not found: {imu_p}")
        return errs, warns
    st = _imu_stats(imu_p)
    if st is None:
        errs.append(f"IMU file {imu_p} has no samples")
        return errs, warns
    if st["rate_hz"] < 100:
        warns.append(f"IMU rate is {st['rate_hz']:.0f} Hz (expected 100 Hz or more); the gyro rotation between frames gets coarser")
    if st["max_gap_ms"] > 100:
        warns.append(f"IMU has a gap of {st['max_gap_ms']:.0f} ms; integration across it is unreliable")
    cam_t = [int(l.split(",")[0]) * 1e-9 for l in open(cam_p) if not l.startswith("#")]
    if cam_t and (cam_t[0] > st["t1"] or cam_t[-1] < st["t0"]):
        errs.append(f"IMU and camera time ranges do not overlap (IMU {st['t0']:.1f}-{st['t1']:.1f} s, camera {cam_t[0]:.1f}-{cam_t[-1]:.1f} s): the clocks are not shared")
    elif cam_t and (st["t0"] > cam_t[0] + 0.05 or st["t1"] < cam_t[-1] - 0.05):
        warns.append("the IMU does not cover the whole camera time range; frames outside it have no inertial data")
    return errs, warns

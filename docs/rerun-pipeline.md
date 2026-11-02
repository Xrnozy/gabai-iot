# Rerun pipeline: stereo RGB-D, semi-dense map, visual-inertial tracking, hands and held objects

One command, one Rerun window, two modes. Written 2026-10-27. Everything below says what was run and what was not.

```bash
py -m gabai.tools.rerun_pipeline --check                                  # can the configured mode run? says why not
py -m gabai.tools.rerun_pipeline                                          # config/pipeline.toml as it is
py -m gabai.tools.rerun_pipeline --mode stereo_only --source video --left data/left.mp4 --right data/right.mp4 --max-frames 0
py -m gabai.tools.rerun_pipeline --mode stereo_imu --source recording --folder data/slam/live_20261026_181526
py -m gabai.tools.rerun_pipeline --source live --mode stereo_imu                 # live from the Pi, everything in one Rerun window
py -m gabai.tools.rerun_pipeline --save data/run.rrd --no-viewer          # write a recording, open later with: rerun data/run.rrd
py -m gabai.tools.stereo_alignment_check data/slam/<folder>               # are left and right rows aligned? (see "Depth accuracy")
```

## 1. Choosing the mode

Set `mode` in `config/pipeline.toml` or pass `--mode`. The command line wins over the file.

| | `stereo_only` | `stereo_imu` |
|---|---|---|
| Pose from | our own stereo visual odometry (`StereoVO`): tracked corners with stereo depth, rigid motion by PnP | Basalt's stereo-inertial estimator; IMU samples and both pictures are pushed into it |
| IMU | not opened, not read, not needed (a test fails if it is opened, even when the file exists) | required: file, extrinsics, calibration, still start |
| Sources | plain video, or a recording | a recording, or the live Pi stream (`--source live`) |
| World frame | first camera pose, z up **assumed** (a camera that starts tilted gives a tilted world) | gravity-aligned from the IMU |
| Scale | metric only when the stereo calibration is real (recording); **not metric for plain video** | metric (calibrated stereo) |
| Fails by | losing tracking on blur / no texture (the pose is held and the frame is not added to the map), and drifting | diverging when the extrinsics, accelerometer calibration or still start are wrong; a pose may arrive a few frames late |
| Init | immediately | waits for 20 IMU samples and, if the recording holds still at the start, for 1 s of stillness (`imu.still_gate`) |

Both modes produce the same `PoseResult` (camera-to-world rotation and position, ok flag, info) and feed the same map and the same Rerun panels, so nothing in the viewer changes when you switch.

**What is real IMU fusion.** Only `stereo_imu`, and only because Basalt does it: the IMU samples are pushed into its estimator. The tool does not blend an IMU into the stereo-only odometry and does not fake it. What the tool adds on top are consistency checks computed outside Basalt: the angle between Basalt's rotation between two poses and the rotation from integrating the gyro, after removing a constant gyro bias learned from Basalt's own rotation (the bias is plotted separately). Small = IMU and tracker agree and are in sync. It does not show how Basalt weights the IMU inside its optimiser; the C interface does not expose that. Basalt's own tracked landmarks are also not exposed, so the "tracked features" panel shows our odometry's features in `stereo_only` and nothing in `stereo_imu` (open Basalt's own window with the older `slam_cloud_view --basalt-ui` to see its tracks).

### What `validate` checks before a run

- mode names, source kind, files exist
- `stereo_imu` on plain video: error (video has no IMU)
- `stereo_imu` on a recording: `imu0/data.csv` exists and has samples, `T_imu_cam` is in `calib.json` for both cameras, the IMU time range overlaps the camera's (error if not), IMU rate under 100 Hz and gaps over 100 ms (warnings), extrinsics marked calibrated in `recording_info.json` (warning if not or if the file is missing), Basalt library present
- `stereo_only`: never touches the IMU
- unknown config keys are rejected with the list of valid keys

It cannot prove the extrinsics are right. A wrong camera-IMU rotation shows up as `imu/gyro_vs_tracker_disagreement_deg` staying large and as `map_alignment` failing.

## 2. The viewer layout

`rerun_pipeline` opens Rerun with a blueprint, so the panels are laid out for you:

| Panel | Shows |
|---|---|
| 3D | `world/map` (fused semi-dense points), `world/live` (this frame), `world/camera` with its picture in the frustum, `world/trajectory`, `world/hands/*` (3D keypoints and bones), `world/objects/*` |
| left camera, right camera | the raw pair, same frame index |
| disparity, depth | disparity colour map (grey = no data); metric depth image (0.2 to 6 m, unit metres) |
| valid / rejected | green = measured and left-right consistent; red = the model gave a value but the left-right check rejected it; grey = no data |
| stereo matches | left and right side by side with lines for 40 sampled correspondences (every 5th frame): lines should be horizontal |
| hands and objects | boxes with label, model score and relation (held, held_unconfirmed, contact, near, overlap_not_touching, uncertain, apart), hand keypoints and bones |
| tracked features | our odometry's tracked corners and their motion (stereo_only) |
| status and checks | markdown: mode, what is assumed, every check with its status and what to look at |
| time series | `imu/*`, `tracking/*` (inliers, pose step, map-vs-depth residual, voxel count), `depth/*` (valid fraction, trusted fraction, median depth, row misalignment, model ms) |
| events | pose held, grab, release, warnings |

Two timelines are logged: `frame` (sequence number) and `t` (seconds). A delayed Basalt pose is logged at the frame it belongs to, so the camera, points and pictures of one frame line up when you scrub.

Metric or not: the status panel states `METRIC (calibrated)` or `NOT metric, scale assumed`. The depth image is in "metres" only in the first case; for plain video f and baseline are assumed (`stereo.f_frac`, `stereo.baseline`), so the numbers are proportional, not metres.

## 3. Reading the diagnostics

| Check | Meaning | Look at |
|---|---|---|
| `map_alignment` | the map seen from the current pose against the depth measured now: median centimetres apart (pass under 5 cm, fail over 12 cm). A constant offset = the pose is wrong now; an offset growing over the run = drift | `tracking/map_vs_depth_median_cm`, `..._signed_cm` |
| `pose_stability` | step and rotation of the newest pose (jump over 0.5 m or 4 m/s, rotation over 45 deg in one step) | `tracking/pose_step_m` |
| `tracking_quality` | features agreeing on the motion (pass 40 or more, fail under 15) | `tracking/inliers` |
| `rectification` | vertical offset of matched features between left and right (pass under 1 px) | `depth/row_misalignment_px` |
| `stereo_sync` | left/right capture offset against the false disparity it causes; **unknown** whenever the offset was never measured (plain video and recordings do not store it) | |
| `depth_coverage` | share of valid depth | `depth/valid_fraction` |

A lost track (stereo_only) holds the last pose, logs an event, and the frame is **not** added to the map, so lost tracking cannot stack or duplicate geometry. Hands, people and a held object are masked out of the map.

Held-object relations come from the grasp rules plus picture and depth overlap: `held` only when a grab was confirmed and the object was seen moving with the hand; `held_unconfirmed` when it was hidden by the hand; `contact` needs picture overlap and 3D closeness; overlap without trusted depth is `uncertain`. Scores shown are the models' own and are not probabilities.

## 3a. Persistent map and stereo-only tracking (added 2026-10-28)

Keyframe visual odometry, an uncertainty-weighted map, and a damped scan-to-map pose correction: see [stereo-mapping.md](stereo-mapping.md) for the design,
the stereo-only versus IMU differences, measured speed and accuracy, and the limits.

## 3b. The 3D hand and the object it holds (added 2026-10-27)

Code: `gabai/pipeline/hand3d.py` (solver, hand frame, object model), wired in `gabai/pipeline/held.py`, drawn by `Viz.hands3d` in `gabai/pipeline/viz.py`.

**The hand.** MediaPipe's 21 keypoints are placed in 3D one by one. Each joint's depth comes from stereo where it is trusted (median trusted depth in a 5x5 window) and otherwise from MediaPipe's own relative depth scaled to metres; a stereo depth that disagrees with the hand's own shape by more than 6 cm (background, flying pixel) is overruled. Every joint carries its source and Rerun colours it: **green = stereo**, **orange = model estimate**, grey = flat. The hand is drawn as joints, finger bones and a palm surface, and smoothed over frames (a move over 10 cm is followed at once). When the depth is metric, a hand whose wrist-to-middle-knuckle length is not 5.5 to 14 cm is refused, because wrong depth or a wrong detection would put a wrongly sized hand in the scene; the reason is in the 2D label and the log (`pipeline.hand_3d_refused`). For plain video the depth scale is assumed, so that test is skipped and the factor that would make the hand 9.5 cm is recorded instead (`scale_factor_if_hand_is_9p5cm`).

**The hand frame.** A rigid frame at the palm centre (x from the little-finger knuckle to the index knuckle, z the palm normal), drawn as three coloured arrows at `world/hands/<id>/frame`. It moves and turns with the hand.

**The held object.** Once the grasp rules confirm a grab (`held`), the object's points are taken from the detector's mask (box if there is no mask), minus the hand, trusted, and within 12 cm of the object's own depth. They are converted into the **hand frame** and fused into 1 cm voxels, so each new view makes the object more complete. Rerun draws them under `world/hands/<id>/frame/held/<object>` in hand coordinates, with a fitted oriented box. Because they are logged as children of the hand-frame transform, the object and the hand move and turn together exactly, every frame, including when the hand covers the object or the detector loses it (the panel then says "hidden: shown from earlier views, attached to the hand", orange instead of blue). When the hand opens and stops (`release`), the object is frozen in the world where it was put down (`world/objects/<key>/model`, green) and the hand-attached copy is removed; if the hand is lost while holding (`lost`), it is frozen in orange as unobserved.

**One hand, one 3D model (fixed 2026-10-27).** Earlier, copies of the hand stayed behind in the 3D view as the real hand moved. Two causes. (1) The id of a hand was MediaPipe's left/right label, which flips from frame to frame (on your video it changed 9 times in 55 single-hand frames), so each flip drew the hand under a new Rerun path and the old path kept its last value. (2) Nothing was ever removed: Rerun keeps the last value logged at a path until it is cleared, and a lost hand, a hand without 3D, a vanished object, the old position of an object that became held, and the 2D hand and object overlays were never cleared. Now: `HandIdentity` (`gabai/pipeline/hand3d.py`) gives each real hand a track id (`hand1`, `hand2`, ...) by matching its palm in 3D (or its picture position when there is no depth) to the nearest earlier track, with a gate that widens while the track is unseen; the id survives a short loss (8 frames) and is retired after that. `Viz` remembers what it has drawn and, each frame, clears every hand that is not tracked now (recursively: joints, bones, palm, frame, the held object under the frame), clears the hand frame when there is none, clears 2D overlays when nothing is detected, and clears object positions that disappeared or became held. `clear_tracked()` removes everything when the hand analysis is switched off (navigation running) or the frame's pose cannot be trusted. Objects put down and frozen in the world stay on purpose. Checked on the real video (200 frames): 2 track ids for the 2 hands, never more than 2 hands in the scene, and in every frame the scene held exactly as many hands as were detected. The new tests fail on the old behaviour (checked).

**What this does not do.** The object model is only the surface seen so far (one face of a block gives a flat patch); the box is a fit to those points, not the true shape. Nothing is generated for unseen sides. The grab decision still comes from the grasp rules, not from a score. If the hand cannot be placed in 3D in a frame, the object is not attached in that frame and the status says why.

**Verified, and how.** Synthetic scenes with known geometry (`tests/test_hand3d.py`, `tests/test_held3d.py`): joints recovered within 2 cm with full stereo depth and 3 cm with the fingertips missing; a hand frame that follows a known rotation; a block rigid in a hand that moves 20 cm and turns 40 degrees stays within 2 cm of the truth after the detector loses it; the frozen object after release within 2 cm; the Rerun entities (object under the hand frame, hand-attached copy cleared on release) and a real `.rrd` written without errors. **On your real videos** (`left.mp4`/`right.mp4`, first 400 frames at every 3rd): 657 hand detections, all 657 placed in 3D, 13,604 joints from stereo and 193 from the model, 0 refused (with the scale test skipped because the scale is assumed). No grab occurred in that stretch: the footage is a top-down kitchen view, and the detector's "cup" boxes are mostly pans and bowls, so the object-in-hand path has not been seen on real footage. It needs a recording where a hand clearly picks something up, with a metric calibration so the size test and the object size are real.

## 3c. Speed, lag, and keeping navigation first (added 2026-10-27)

**What was slow.** Per frame on this PC (100 real frames, hands and objects on): depth 36 ms, hands and objects 42 ms, source decode 12 ms, pose 8 ms, map 11 ms, one row-alignment check of 165 ms every 15 frames (a visible stutter), and every Rerun call on the same thread. The recording written was 1.24 MB per frame (raw float32 depth, raw colour images), which also made the Rerun viewer itself lag.

**What was changed** (`gabai/pipeline/perf.py`, settings in `[perf]` of `config/pipeline.toml`):

| Change | Effect |
|---|---|
| Drawing on its own thread (`async_viz`), bounded queue; heavy picture calls are dropped and counted if the viewer cannot keep up, scalars and events never | the compute thread spends under 1 ms per frame on drawing; viewer lag median 6 to 14 ms, 0 dropped in the runs below |
| Smaller Rerun payload: depth as 16-bit PNG in millimetres, disparity/depth/valid/feature panels at half size and compressed, camera pictures at JPEG 70, map sent every 10th frame and capped at 60,000 points, long trajectory thinned | recording size per frame 1.24 MB to about 0.27 MB (4.5 times smaller); the Rerun viewer scrubs and plays much more smoothly |
| Video decode on its own thread (`prefetch`) | source time leaves the critical path |
| Two stages that overlap (`staged`): stage A = depth, stage B = pose, map, hands, objects, checks; one frame of queue (`stage_queue`) | steady-state frame rate 7.8 fps to 14.6 fps (1.9 times), same poses, path and map (a test compares staged against single-thread) |
| Object detector every 3rd frame, always on while an object is held; last detections reused and marked stale | hands and objects 42 ms to 33 ms median |
| Edge mask on disparity in float32 instead of depth in float64 (same answer, tested) | 8.0 ms to 3.9 ms per frame |
| Row-alignment check on half-size pictures every 30 frames | the 165 ms stutter becomes about 40 ms every 30th frame |
| Per-stage timers in every summary (`stage_ms`), end-to-end latency, steady fps | you can see where time goes: `pipeline_summary.json` |

**Trade-off to know.** The two stages raise throughput but also the delay between a picture arriving and its full result (end-to-end median 114 ms single-thread against 192 ms staged). The depth itself is ready after stage A, about 55 ms. For a live stream set `drop_when_busy = true`: when stage B is behind, the frame is skipped (counted in `frames_dropped_stage_b`), so the delay stays bounded; the odometry then sees fewer frames and can lose tracking sooner (a test shows this). Recordings use `false`, so every frame is processed.

**Navigation first.** The navigation viewer (`demo_viewer`) and this pipeline are separate programs that can run on the same GPU. Measured with the viewer running headless on the video (per-frame loop time, median / 95th percentile, runs vary by about 5 ms):

| Situation | Viewer loop ms |
|---|---|
| viewer alone | 42 to 48 / 49 to 64 |
| pipeline flat out beside it (new, 14 fps) | 53 to 58 / 62 to 69 |
| pipeline capped at 7 fps | 47 / 58 |
| **pipeline with the heartbeat guard (default)** | **44 / 56 (the same as alone)** |

So an unconstrained pipeline does slow the navigation loop, by about 10 to 16 ms (20 to 40 percent) here. Priority alone does not help, because it cannot prioritise the GPU. The guard (`gabai/navguard.py`): the viewer writes `data/nav_active.json` every second while it runs; the pipeline checks it each second and, while it is fresh and its process alive, caps itself to `nav_fps_cap` (3 fps) and turns hands and objects off (a status message says so). A crashed viewer stops refreshing the file and the pipeline returns to full speed within about 3 seconds. Other protections: `low_priority` (below-normal CPU priority, costs about 20 percent of the pipeline's own speed when the PC is busy), `fps_cap`, `threads`.

**What is not covered.** The guard and caps were measured on this PC with the video source, not with the live Pi stream. A PC with a weaker GPU may need a lower `nav_fps_cap` or to run the pipeline only offline. The viewer's own time with the Rerun *viewer window* open was not measured (the runs were headless). Nothing here makes the pipeline itself safe to rely on for navigation: the alerts still come only from `demo_viewer`.

## 4. Results on this machine (2026-10-27)

All with LAS2-s, `--skip 2` or 3. Numbers are what the tool reported, not accuracy measurements.

**Plain video, stereo_only** (`data/left.mp4`, `data/right.mp4`, 150 frames from second 30): 150 of 150 poses accepted, path 1.40 in assumed-scale units, map alignment median 2.7 cm (1.2 cm at the start, 3.3 cm at the end), hands and YOLO on, 37 s for 150 frames. Row alignment of plain video before the automatic fix: 11 px; the pipeline applies the fix and its `rectification` check passed on all 10 sampled frames. Stereo sync: unknown.

**Recording `live_20261026_181526`, both modes** (384 frames):

| | stereo_only | stereo_imu |
|---|---|---|
| poses accepted / held | 342 / 42 | 382 / 0 |
| path reported | 1.1 m | **373.7 m (diverged)** |
| map alignment | median 23.5 cm, rising from 0.5 to 25.6 cm | median 17.7 cm |
| `pose_stability` | 383 pass | 223 pass, 143 fail |
| warning | | extrinsics not calibrated (no `recording_info.json`) |

Reading it: stereo_only drifted away from the map it built (alignment 0.5 cm to 25 cm over the run) and lost tracking on 42 frames; stereo_imu diverged, which the extrinsics warning predicted. Neither is a good result on this recording.

**A confirmed cause for bad depth in the recordings: the left and right pictures are not row-aligned.** `stereo_alignment_check` on `live_20261026_181526`, `rec_20261026_155811` and `rec_20261026_155458`: matched features sit about 17 px apart vertically (median, range roughly 4 to 40 px) even on frames where the camera is nearly still, and one pair measured 37 px. Measured as a static offset, not tied to rotation (correlation with gyro rate about 0), so it is not a capture-timing problem. A rectified pair needs about 0 px. The saved calibration does rectify its own calibration pairs (14 px raw, 0.0 px rectified on `data/calib_pairs_swapped`), so the rectifier is right for the scene it was calibrated on and wrong for these recordings. Possible causes, not distinguished yet: the cameras moved or were remounted since calibration, left/right device order or rotation differs from the calibration, a calibration done on too few or too similar views. Next step: recalibrate (`docs/calibration-guide.md`), record again, run `stereo_alignment_check`; it should read `aligned`. Until then depth, Basalt and any map built from these recordings are suspect.

## 5. Depth accuracy: what to check, in order

1. `py -m gabai.tools.stereo_alignment_check <recording>`: verdict `aligned`, `static_misalignment` (calibration/mount), `timing_offset` (offset grows with rotation), `misaligned`, or `unknown` (too little texture).
2. Baseline and focal length: the pipeline logs `pipeline.geometry` with f, B, whether they are calibrated, and the depth error caused by 0.5 px of disparity error (for 881 px and 5.07 cm: 0.01 m at 1 m, 0.04 m at 2 m, 0.10 m at 3 m, 0.28 m at 5 m). Verify metric depth against a tape measure with `target_test` and `range_test`.
3. Synchronisation: not measured for plain video or stored in recordings; `stereo_sync` stays `unknown` rather than passing. `pair_dt_ms` is only known on the live Pi link.
4. A content-based search for a frame offset between the two videos (valid-disparity fraction against offset) did not peak at any offset on `left.mp4` / `right.mp4`, so it could not confirm or rule out a desynchronisation there.

## 6. Debugging recipes

| Problem | Do |
|---|---|
| Poor depth | steps in section 5; open `valid / rejected` (lots of red = the left-right check disagrees: rows, texture, blur); `stereo matches` lines should be horizontal; `depth/row_misalignment_px` |
| Tracking drift | `tracking/map_vs_depth_median_cm` rising over the run; `tracking/pose_step_m` spikes; in stereo_imu `imu/gyro_vs_tracker_disagreement_deg` and `imu/gyro_bias_estimate_dps` (a resting bias over about 3 deg/s keeps Basalt's still-gate shut) |
| Unstable poses | `pose_stability` fails; `tracking/inliers` low in stereo_only; events panel shows each held frame and why |
| Wrong coordinate transform | `map_alignment` fails from the first frames (not growing): camera-IMU rotation, stereo calibration, or axis convention. Check the 3D view: the camera frustum picture should sit where the map points it sees are |
| Point-cloud stacking at the origin | cannot happen for lost tracking (frame not added); if it appears, check `world/live` is placed with `world/camera` and that the status shows pose ok |
| Hands/objects wrong | status panel says hands/objects `off` when a model file is missing; `annotated` boxes show relation and why |
| Sync / IMU timing | `validate` warnings; `docs/imu-sync-guide.md`; the `still_gate` setting |

All diagnostics are also in `logs/pipeline_<mode>_<time>/gabai.jsonl` (`pipeline.geometry`, `pipeline.check_fail`, `pipeline.hand_event`, `pose.vo_held`, `pipeline.summary`) and in `pipeline_summary.json` next to it; `py -m gabai.tools.diag_report` and `collect_diag` read them like any other run.

## 7. Dependencies and calibration

- Python packages: `rerun-sdk` 0.40, `mediapipe` (hands), `ultralytics` (objects), `open3d`, `scipy`, `opencv`, `torch`. Model files: `checkpoints/hand_landmarker.task`, `models/yolo11n-seg.pt`, LAS2 weights. A missing hand or object model turns that part off and the status panel says so.
- `stereo_only` on a recording needs `calib.json` (intrinsics and the baseline from `T_imu_cam`). On plain video nothing is calibrated.
- `stereo_imu` needs, in the recording: `calib.json` with `T_imu_cam` for both cameras (from `sync_imu_camera` + `check_extrinsics --write`), `recording_info.json` marking them calibrated, accelerometer calibration (`calibrate_accel`), a still start (gyro under 3 deg/s for 1 s), IMU rate above 100 Hz, and Basalt built at `D:\slam\basalt`.

## 8. Tests

`py scripts/run_tests.py pipeline held stereo_alignment` (pipeline config, diagnostics, held-object logic, orchestrator with injected depth on synthetic recordings, alignment check). One orchestrator test runs both modes on the real recording `data/slam/live_20261026_181526` with the real depth model and Basalt (skipped when the folder is absent).

What the automated tests cover: mode validation and the no-IMU guarantee, sideways motion recovered in metres by the odometry on a synthetic scene, lost-tracking frames kept out of the map, map-vs-depth alignment passing for a good pose and failing for a 25 cm pose error and for a drifting map, pose jump and rotation detection, rectification and sync checks, held vs contact vs near vs uncertain relations, every panel receiving data for every frame, the real two-mode run producing the same entities with IMU panels only in `stereo_imu`.

## 9. Not done, not verified

- Nothing was run on the live rig. A live Pi stream is not a source of this tool (record with `record_basalt` or `live_basalt`, then run this on the folder).
- Metric accuracy of depth, trajectory and map has not been measured; on the available recordings the pictures are not row-aligned, so the numbers above are not accuracy.
- No recording holds still long enough and has calibrated extrinsics at the same time, so a healthy `stereo_imu` run has not been seen here.
- The semi-dense points are stereo depth at textured pixels, not Basalt's own semi-dense map (not exposed). No loop closure or global optimisation in either mode.
- Held-object detection uses COCO classes (YOLO11n-seg); "keys" and other small items are not classes. Grab thresholds are starting values, not measured against real pick-ups.
- The ARKitScenes repository and the three Rerun example recordings you linked were not opened; the layout follows their idea (images under the camera, depth and RGB side by side, trajectory with the cloud) from the entity design, not from their code.

## Live from the Pi (added 2026-10-29)

`--source live` takes the two cameras and the IMU from the Pi sender (port 5600, token `gabai`; `--port` changes it), runs the same pipeline and draws everything in one Rerun window. Start the command first, then the sender on the Pi, and hold the rig still for the first seconds (the still gate). It writes `calib.json` into a new `data/slam/live_<time>/` folder for Basalt, but does not save the pictures or the IMU. Frames are taken newest first (`perf.drop_when_busy`), so a slow PC skips frames instead of falling behind. Closing the Rerun window or pressing Ctrl-C ends the run. Tested with a fake sender that checks calibration writing and that each frame carries the IMU samples up to its stamp; not yet run against the real Pi.

## Less depth noise (added 2026-10-29)

Two optional steps on the stereo disparity, set under `[stereo]` in `config/pipeline.toml` or on the command line:

- **Speckle filter** (`speckle_px = 100`, `speckle_diff = 1.0`, on by default): trusted islands smaller than `speckle_px` pixels that jump more than `speckle_diff` px from their neighbours are dropped. On three frames of `data/left.mp4` / `right.mp4` it removed 3 to 6% of the trusted pixels. It only removes pixels.
- **Monocular metric depth** (`--mono dav2`, off by default): Depth Anything V2 Metric Indoor Small, already in the local Hugging Face cache. `--mono-mode blend` (default) averages each trusted stereo pixel with the monocular map, put on the stereo scale, weighted by each one's own noise; pixels stereo did not measure stay unknown, nothing is filled in. `--mono-mode check` only drops stereo pixels that disagree grossly with it. On six frames of the video pairs the high-frequency roughness of the disparity (a noise proxy, no ground truth) went from 0.021 to 0.015 px with blend, and stayed about 0.021 with check. Roughness cannot tell noise from lost detail, so blend may soften fine structure; accuracy was not measured. Cost: about 13 ms per frame on this PC.

## Depth holes and noise while the camera moves

What was checked, with LAS2-H kept as the depth model:

- Baseline and calibration. `data/stereo_calib.npz` holds a baseline of 42.10 mm against the 42 mm of the Fusion 360 case (+0.25%), a stereo reprojection error of 0.41 px and a row error of 0.105 px, and `P2[0,3] / -f` equals the baseline. The calibration is correct and was not changed. `stereo.rig_baseline_mm = 42` in `config/pipeline.toml` makes the live run warn when a calibration is more than 5% away from the case; `calibrate_stereo` prints the same check and names the square size that would match (the baseline scales with `--square-mm`).
- `data/imu_sync.json` (camera-IMU rotation) was measured on 2026-10-26, before the cameras were swapped and recalibrated. `stereo_imu` now warns when it is older than the stereo calibration. It does not match `data/device_frame.json` by any simple flip, so re-run `gabai.tools.sync_imu_camera` on the current rig. `stereo_only` does not use it.
- Left/right capture offset is the main cause of noise while moving. The two webcams are paired by nearest capture time, so the right picture is up to about 17 ms off the left one, and the model reads the scene motion in that gap as depth. The Pi now sends the signed offset with every pair (new sender and receiver; an old sender still works, it just sends none), and `gabai/skewcomp.py` winds the right picture back to the left camera's instant using dense optical flow of the last two left pictures (about 2 ms). Switch off with `source.skew_comp = false`.
- Holes: the 'valid / rejected' panel now colours every missing pixel by cause (`gabai/depthq.py`): outside the right picture (a band as wide as the disparity, wider when closer), no texture, left/right disagree, small island removed, monocular veto. The share of each cause is plotted under `depth/frac_*` and written to the run summary as `depth_pixels_by_cause`. Run on your live scene to see which cause dominates before changing anything else.

Tried and not kept, because the measurement showed no gain on the recorded sequence: sub-pixel left-right check (only 0.5% of pixels are near misses), a flying-pixel filter (removed 1.7% of good pixels, floating points were already 0.01%), exposure matching and denoising of the right picture (LAS2-H lost nothing under a 55% exposure mismatch or heavy noise), per-frame filling from earlier frames (the voxel map already fuses earlier frames in 3D, so it would add nothing to the map).

Compare before and after on a recording:

    py scripts/bench_depth_quality.py --start 30 --frames 60 --model h

It simulates a 33 ms offset on `data/left.mp4` / `data/right.mp4`, runs the depth model and the whole pipeline on the clean, offset and corrected pairs, and writes numbers and pictures to `data/bench/depth_quality/`. Errors are measured against the clean pair's own LAS2 result; the video has no ground-truth depth.

## Noise while the camera moves: live against the recorded video

`data/left.mp4` / `data/right.mp4` come from one stereo camera with a shared shutter, in daylight (short exposures). The live rig is two separate USB webcams in a dim room. Three differences, in the order of how much they cost:

1. The right picture is captured up to 17 ms (30 fps capture) or 33 ms (15 fps capture) away from the left one, and the stereo model reads the scene motion in that gap as depth: false disparity = focal length (618 px) x turn rate x offset, 10 px at 60 deg/s and 16 ms. Measured on the video with a 33 ms offset: median depth error 4.0 px, 74% of the pixels more than 2 px off (clean pair: 0). Changing `--fps` or `--capture-fps` only moves the size of the offset. The fix is the offset correction (`gabai/skewcomp.py`), but it works only when the Pi sends the offset: the 15 calibration pairs show the Pi still runs the old sender (`camera_offset_ms` is null in every pair). The live run now prints `WARNING: the sender does not send the left/right capture offset ...` when that is the case.
2. In a dim room the cameras' own auto exposure runs as long as the frame period (66 ms at `--capture-fps 15`), and the two cameras choose different values. The longer the exposure the more a moving scene smears, and different exposures also shift the instants the two pictures really show. `--exposure fast` (now the sender's default) gives both cameras the same manual exposure, never above `--max-exposure-ms` (16 ms), and lets the gain make up the light. The sender reports the exposure and gain it runs with, and the live status shows them (`camera exposure ...`). Smear of a point: focal length x turn rate x exposure, so 16 ms gives 10 px at 60 deg/s where 66 ms gives 43 px.
3. Smear and noise hurt much less than the offset. Measured on the same video (`py scripts/bench_motion_blur.py`): 32 px of smear on both pictures: valid depth 89%, median error 0.65 px, 10% of pixels over 2 px off; noise of sigma 24 grey levels: 0.49 px, 9%; an 8 ms exposure with the extra gain (smear 8 px, noise 4): 0.29 px, 2%; 16 ms sits between this and the 32 px case. So trading exposure for gain is right, and fixing the offset matters more than either.

Run it:

    Pi:  python3 -m gabai.pi.sender --host <PC_IP> --width 640 --height 360 --fps 15 --capture-fps 30
    PC:  py -m gabai.tools.rerun_pipeline --source live --mode stereo_only --model h

With the new sender the PC prints, before the first picture, the camera exposure line and no offset warning; the Rerun status panel keeps `camera exposure N ms, gain G` and `sync: right camera is +x ms from the left; corrected for the motion (k px)`. If the Pi log shows `cam.too_dark`, the room is too dark for 16 ms: add light (or raise `--max-exposure-ms`, which smears more). Use `--exposure auto` to get the old behaviour back for a comparison.

Not measured on the real rig: the true left/right offset (the sender can only time when each frame arrives; a constant difference in the two cameras' USB delay would remain), and whether your cameras accept `exposure_time_absolute` while streaming (the sender reads the controls back and logs `cam.exposure`; a camera that ignores them shows it there). While moving the rig up and down quickly, watch `depth/row_misalignment_px`: more than about 1 px with the correction on means a constant offset is left.

## One stable exposure for your room

Why the pictures stayed dark at 16 ms and gain 100: between the 8 ms and the 16 ms run the picture median went from about 40 to about 42, while doubling the exposure should have doubled it. So either the cameras do not apply `exposure_time_absolute` while streaming, or the gain and exposure act far less than their numbers say. The calibration pairs taken with the cameras' own auto exposure have a median of 96 to 102, so auto finds much more light than the manual values do. The probe tells which case you have:

    python3 -m gabai.pi.exposure_probe --left /dev/video2 --right /dev/video0

Hold the rig still, aimed at your usual scene with your usual lights. It sets both cameras to the same values over a grid (4 to 33 ms, three gains), prints brightness, clipping and noise for each, tries the cameras' auto exposure, says whether a longer exposure really makes a brighter picture, prints the Pi's power state, and ends with the shortest exposure that gives a bright (median 70 or more), unclipped and matching picture, plus the sender command that uses it as FIXED values (`--exposure <0.1 ms units> --gain <n>`). A fixed value is the most stable choice for stereo: nothing changes while you move.

If a fixed value cannot cover both a dim and a bright room, the simplest alternative is `--exposure fast` (the sender default): it adapts for up to 10 s (`--settle-s`), then LOCKS, so the brightness cannot drift while you move; restart the sender when the lighting changes. The cost of any exposure cap is smear (turn rate x focal length x exposure: 10 px at 60 deg/s and 16 ms), the cost of gain is noise; LAS2 tolerated both in the simulation (see above), but a dark picture with the gain at its maximum leaves no room to improve, so add light first.

The sender prints the controls twice: before the cameras open and again with the stream open, so a camera that resets them shows up. In the Rerun 'depth quality' plot, `picture_mean_left` and `picture_mean_right` should be above about 70 and close to each other, `pair_offset_ms` is how far apart the two cameras were captured, and `offset_correction_px` is what the correction moved the right picture; when the rig moves quickly these two should rise together while `depth/valid_fraction` stays up.

The Pi's under-voltage is not a side issue: `0x50005` means the supply sagged and the CPU was capped at that moment, which is when frames stall (`pi.loop_gap`), arrive late (`pi.send_slow`) and get time-stamped late, and this also reduces how well the offset correction can work. Fix the supply first: `vcgencmd get_throttled` should print `0x0` while both cameras stream.

## Is it image noise? (`py scripts/bench_denoise.py`)

No denoiser was added to the pipeline: the measurements say image noise is not what makes the depth bad, and the candidates that remove it either do nothing for depth or cost detail or time.

- Your cameras' noise, from the 30-frame still recordings: sigma 4.5 grey levels (picture median 73). With noise of sigma 8 added to sharp video pairs, LAS2-H's median error is 0.21 px (clean pair = 0), 1.7% of pixels more than 2 px off, and valid depth does not change (86.7% against 87%). Even at sigma 24 it is 0.48 px.
- Candidates, same test (sigma 8 / 16 / 24 median error in px): none 0.209 / 0.336 / 0.483; 3x3 Gaussian 0.196 / 0.308 / 0.409 (0.1 ms, but only 83% of the sharpest edges survive and a 2 px line drops from 255 to 201); bilateral 0.189 / 0.331 / 0.472 (0.8 ms); median 0.195 / 0.327 / 0.441 (eats thin objects); non-local means 0.205 / 0.332 / 0.484 (270 ms, far over the budget). The best case is 0.07 px at sigma 24, far above the noise you have.
- Moving rig, 33 ms left/right offset: as before 4.0 px median error, 77% of pixels over 2 px off; with the offset correction 1.1 px, 28%; correction plus smoothing 1.1 px, 28% (with sigma 12 noise: 1.14 against 1.22). Smoothing alone, without the correction: 3.97 px. The offset, not the noise, is what breaks the depth when the rig moves; the top views in `data/bench/denoise_clouds.png` show the offset pair bending the scene and the corrected pair matching the clean one.
- A noise estimate from a single picture is not reliable on these cameras (the MJPEG compression smooths the noise: the estimate read 1.0 where the still-frame measurement is 4.5), so an automatic switch that smooths only noisy pictures could not be built honestly.

## Turning the view 180 degrees (`--flip180`)

    py -m gabai.tools.rerun_pipeline --source live --mode stereo_only --model h --flip180

Both rectified pictures are turned 180 deg and the two trade places (the turned right picture becomes the left one), so disparities stay positive and the calibration stays valid: no recalibration, no change on the Pi. The principal point moves to (W-1-cx, H-1-cy), the left/right capture offset changes sign, and for `stereo_imu` Basalt gets the camera turned 180 deg about its optical axis (camera-IMU rotation times diag(-1,-1,1), the principal point moved the same way). Also available as `source.flip180 = true` in `config/pipeline.toml`.

Checked on LAS2-H with three of your saved pairs: valid depth 92.4 / 86.4 / 89.5% before and 92.4 / 88.0 / 89.3% after the turn; the 3D points of the turned view, mapped back into the original camera frame, lie a median 2.5 to 8 mm from the original cloud. Not checked: Basalt with the turned calibration (needs a live run), and the camera-IMU rotation was already stale (see above).

Which way up is right: with the sender at `--left /dev/video2 --right /dev/video0` and `data/stereo_calib_new.npz` the two cameras' pictures should already come out upright (the camera order line printed at start confirms that order). If the Rerun pictures are still upside down, use `--flip180`. Do not use `--rotate-left/--rotate-right` for this: they change one picture only and break the rectification (the row-alignment line then warns).

## Larger pictures for the depth model (`--scale`)

    py -m gabai.tools.rerun_pipeline --source live --mode stereo_only --model h --scale 1.5

The 640x360 pictures are enlarged (cubic) by the factor before LAS2, and the disparity and the left-right mask are brought back to 640x360 (disparity divided by the factor). Upscaling adds no information, but the model matches on a coarse feature grid, so thin bars and wires gain accuracy. Measured with `py scripts/bench_input_scale.py` on `data/left.mp4` (1280x720 natively, shrunk to 640x360 to mimic the Pi) against LAS2-H run on the native 1280x720 pair, GPU time per pair:

| model, size | median error | pixels over 1 px off | time |
|---|---|---|---|
| LAS2-S, 640 | 0.415 px | 19.2% | 36 ms |
| LAS2-M, 640 | 0.274 px | 7.4% | 41 ms |
| LAS2-H, 640 (default) | 0.185 px | 5.6% | 64 ms |
| LAS2-M, 1280 | 0.161 px | 3.2% | 94 ms |
| LAS2-H, 960 (`--scale 1.5`) | 0.118 px | 3.5% | 105 ms |
| LAS2-H, 1280 (`--scale 2`) | 0.062 px | 1.7% | 186 ms |

The share of valid pixels does not change (86 to 87% in every row), so this reduces wrong depth, not holes. In the pipeline `--scale 1.5` costs 105 ms per frame against 62 ms (the CPU enlarging takes the rest), so expect about 9 fps of depth instead of 15; `--scale 2` is too slow for real time on this GPU. The nearest measurable distance grows by the same factor (model limit 192 px of disparity: 0.13 m at 1.0, 0.20 m at 1.5). Default stays 1.0.

What the remaining holes in the live screenshots are: the 'valid / rejected' panel shows red (left/right disagree) over the fan grill and along the bunk-bed bars, and grey (no texture) on the plain walls. A grill or a thin bar in front of a far background looks different from each camera (different parallax), so the two views cannot agree on one depth; a blank wall has nothing to match. Those pixels are left empty on purpose, not filled.

## Recording your own test footage and replaying it

The earlier simulations in this file used `data/left.mp4` and `data/right.mp4`, which come from a different stereo camera. Judge the changes on recordings from your own cameras:

    Pi:  python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video2 --right /dev/video0 --width 640 --height 360 --fps 15 --capture-fps 30
    PC:  py -m gabai.tools.record_pairs --name still --seconds 20      hold the rig still
         py -m gabai.tools.record_pairs --name normal --seconds 30     walk or turn at an ordinary pace
         py -m gabai.tools.record_pairs --name rapid  --seconds 30     fast turns and sweeps

Stop the viewer first (same port). The recording starts 3 s after the first pair arrives, ends after `--seconds` or on q, and is written to `data/recordings/<name>_<date_time>/`: `left.avi` and `right.avi` (640x360, lossless FFV1, frame n of one belongs to frame n of the other), `pairs.csv` (Pi capture time, the signed left/right offset, exposure, gain, brightness, sequence number per pair) and `meta.json` (devices, calibration, counts, how many pairs the receiver missed). The pictures are stored as the live pipeline receives them, before rectification, so a replay does everything the live run does, with the recorded times and offsets. Aim at something with texture 1 to 3 m away; in the 'rapid' take, turn the rig the way you do in real use.

Replay it in Rerun:

    py -m gabai.tools.rerun_pipeline --source replay --folder data/recordings/<folder> --mode stereo_only --model h [--scale 1.5] [--flip180]

Add `--no-viewer --save run.rrd` to write a file instead. Recordings hold no IMU data, so only `stereo_only` replays. Compare settings on the same footage:

    py scripts/evaluate_recording.py data/recordings/<folder> --variants before,after[,after+scale1.5]

Every variant replays the same pairs. `before` has the left/right offset correction off, `after` on. Each frame is put in a motion class from how fast the picture moves (still under 30 px/s, normal 30 to 200, rapid over 200; 100 px/s is about 9 deg/s). Per class it reports the valid share, the causes of the rest, and depth noise (how far a pixel's disparity moves from one frame to the next after following the picture motion); for the whole recording it reports the 3D map (voxels, voxels seen 3 or more times, median distance between the map and each new frame's depth, path length) and the speed. It writes `evaluation/results.json`, `depth_compare.png` (depth of a rapid and a typical frame for each variant) and `cloud_compare.png` (top and side views of the maps) inside the recording folder. There is no ground-truth depth: noise and stability are measured, accuracy is not. The offset correction can only differ from `before` when the recording has offsets (`pairs.csv` column dt_ms filled; the sender must be the current one).

## What the first recordings from your own cameras showed (2026-10-29)

Recordings: `still_*`, `normal_*` (the rig actually stayed still, 1 px/s), `rapid_*` (picture speed up to 1700 px/s), all from `/dev/video2` left and `/dev/video0` right, 640x360.

1. The calibration did not fit the pictures. Matched room features should sit on the same row after rectification; with `data/stereo_calib_new.npz` they were 2.6 px apart (still recording) and 2.5 px (normal), and 1.2 to 13 px in the rapid one. The recorded pictures are upside down, which is the layout of the 15 calibration pairs (`stereo_calib_new_c1.npz` fits them at 0.3 px). So the two cameras came out in the other order than the `/dev/video` names say: after a replug or reboot the numbers can trade places, the sender still reports `/dev/video2` as left, and the device-name check passes. Turning both pictures 180 deg and swapping them first fits `stereo_calib_new.npz` at 0.3 px (0.13 px on the calm frames of the normal recording, as good as the board calibration), and the pictures come out upright. No new board calibration was needed: the numbers in the file are right for the cameras, the pictures were being fed to it the wrong way round.
2. The fix is `source.auto_layout` (on): at the start the first pairs are rectified both ways and the layout whose rows line up is kept (a message `CAMERA LAYOUT: ... swapped layout` says so). It works for live and replay. To stop it happening, give the sender fixed names, e.g. `ls -l /dev/v4l/by-path/` on the Pi and pass those paths as `--left` / `--right`.
3. Results on the same footage (`py scripts/evaluate_recording.py <folder> --variants before,layout,layout+offset`):

| recording | | valid depth | depth noise (px frame to frame) | map |
|---|---|---|---|---|
| normal (still, 404 pairs) | before | 66.1% | 0.329 | 12430 voxels, map vs frame depth 10.3 cm, 36 frames refused |
| | layout fixed | 81.2% | 0.215 | 4284 voxels, 4.8 cm, 0 refused |
| rapid, still frames | before / fixed | 70.6% / 84.1% | 0.322 / 0.236 | |
| rapid, normal speed | before / fixed | 74.2% / 84.7% | 0.858 / 0.640 | |
| rapid, rapid | before / fixed | 60.7% / 63.6% | 1.454 / 1.106 | 2493 / 4797 voxels, 19 cm both |

4. The left/right offset correction (`source.skew_comp`) does not help on your footage, so it is now off by default. The Pi's offset is right: regressing the vertical row error against the vertical picture speed on the rapid recording gives the right camera 13.3 ms after the left, the Pi says 14.3 ms. But with the correction on, valid depth at rapid speed rose 63.6% to 67.8% while the frame-to-frame noise went 1.11 to 2.11 px, at normal speed it was worse on both (84.7% to 82.8%, 0.64 to 1.02), and the map got no steadier (261 to 290 frames refused as inconsistent). The correction guesses the 14 ms motion from the 70 ms between two pictures and from smeared pictures, which is not good enough; it worked in the earlier simulation because that used an exact one-frame offset.
5. What still limits the rapid recording: the pictures themselves. At 500 to 1700 px/s the frames are smeared over tens of pixels, so 24% of the pixels have no texture to match and the map refuses most frames (the pose cannot be tracked on smeared pictures). The row error stays at 3.5 px on fast frames even with the layout right. That is the exposure and the rolling shutter of the cameras, not the calibration; shorter exposures (`--exposure fast`, enough light) are what can change it.
6. Not verified: the absolute scale of the depth (focal length and baseline). A recording has no known distances; check it with a tape measure using `py -m gabai.tools.target_test`.

## Stutter, power and synchronization (analysis of the five recordings and the Pi log)

Run `py scripts/analyze_timing.py` for the numbers below (per recording: pairs per second, gaps between pairs, arrival delay, the left/right offset, the Pi's power flags).

| recording | pairs/s | gaps over 200 ms | time spent in gaps over 150 ms | arrival delay above the fastest pair (median / p95 / max) | left/right offset |
|---|---|---|---|---|---|
| normal_235431 | 4.3 | 27 | 76% | 106 / 1185 / 4079 ms | -14.4 ms (std 4.6) |
| normal_235458 | 13.3 | 5 | 7.5% | 26 / 88 / 414 ms | -14.4 ms |
| rapid_235536 | 14.2 | 0 | 0% | 22 / 54 / 185 ms | -14.3 ms (std 0.1) |
| still_235327 | 14.6 | 0 | 0% | 11 / 31 / 41 ms | -10.5 ms |
| still_235352 | 12.5 | 9 | 16% | 35 / 181 / 598 ms | -14.4 ms |

- Power: every recording ends with `throttled=0x50005` (under-voltage and throttled right now), as does your Pi log. The Pi log shows 135 KB pairs taking 225 to 1688 ms to send (about 0.6 Mbit/s at worst) and loop gaps of 0.7 and 1.7 s. In the recordings the gaps and the arrival delay rise together (a stall in the link, not a slow camera), and the runs with the worst stalls have the highest round-trip time (0.20 and 0.23 s). A sagging supply and a weak or power-saving Wi-Fi link both do this and the logs cannot tell them apart; the recordings now log the throttle flags, round-trip time, temperature and camera rates for every pair (`pairs.csv`), so the next ones can.
- Stutter does not make the depth of a pair worse: the still recording with 76% of its time in gaps has the same valid depth (85.8%) and noise (0.21 px) as the smooth ones. It hurts when the rig moves: tracking across a 0.5 to 4 s gap loses the pose, which is what the 'jump' and 'map inconsistent' warnings show.
- Synchronization: the offset between the two cameras is a constant (std 0.04 to 0.3 ms in the smooth runs), 14.3 ms in four runs and 10.5 ms in one. It is the phase of two free-running 30 fps streams, fixed when each stream starts and different after a restart. That is close to the worst case of 16.7 ms, and it is what makes a moving scene give false disparity. The sender now restarts the right camera's stream until the offset is within 4 ms (`--phase-target-ms`, `--no-align-phase` to skip): it measures the offset, waits (-offset mod 33.3 ms) and restarts the right stream, and repeats up to 6 times. It prints `camera offset (right minus left): -14.3 ms -> +1.2 ms  (aligned)`. This replaces the flow-based correction that did not help.
- The sender slows its send rate while sends are slow (`pi.rate` in the log, `send rate 15.0 -> 11.2 pairs/s` on the console) and speeds up again slowly, so it does not keep a weak link saturated; `--no-adapt-rate` turns it off. It warns when Wi-Fi power saving is on (`sudo iw dev wlan0 set power_save off`; `scripts/pi_power_save.sh` does it).
- Calibration verified on the recordings: with the right layout the rows of matched room features differ by 0.275 px (median over 4438 matches from the calm frames of three recordings; this is the matching noise level), signed mean -0.11 px, and a plane fitted to the vertical error over the picture changes it by 0.02 px (0.35 px across the picture height), so the baseline (41.72 mm), focal length, distortion and rectification in `stereo_calib_new.npz` need no correction. A new calibration would need new board pairs, which only helps if the cameras moved.

Not verified (needs the Pi): the phase alignment and the rate control run on the Pi hardware; their logic is tested here with simulated cameras. To check them, record again with the new sender and compare:

    py scripts/analyze_timing.py data/recordings/<new folder>

The offset should be within 4 ms, and with the supply fixed (`vcgencmd get_throttled` printing `0x0`) the gaps over 200 ms should disappear. A wired Ethernet cable to the Pi removes the Wi-Fi from the question.

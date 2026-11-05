# `mapper/`: clean semi-dense mapping pipeline (plan, not built)

Written 2026-11-03. Status: awaiting approval. Nothing in `rtabmap_lab/`, `gabai/` or the Pi code is changed by this plan; the new code lives in `mapper/` and imports the existing pieces read-only.

## 1. What the measurements already say (why a rewrite alone will not fix the map)

From `docs/stereo-depth-diagnosis.md`, `docs/rtabmap-noise.md`, `docs/room-reconstruction.md`, `docs/calibration-audit.md`:

| cause | size | owner | fixed by software? |
|---|---|---|---|
| Opposite rolling shutters (left camera mounted upside down) | up to 6 px row shift while turning, depth off up to 40% top vs bottom | rig | partly: `rolling_shutter.py` halves it (2.49 -> 1.12 px). Real fix: mount both cameras the same way up |
| Cameras move in their mounts | right camera pitched 1.3 deg in one day; wrong file = 7 px row error | rig | no. Needs a rigid mount + a per-session row check (exists) |
| Wrong metre: config uses the 42 mm file, boards say 46.8-47.1 mm | every depth 10% short | config | yes, use `stereo_calib_charuco_45.npz`; settle with tape-measured walls |
| Dim light, blur, JPEG, Pi under-voltage (`throttled=0x50005`) | disparity noise 0.89 px sharp -> 1.21 px blurry | capture | no. Light, `--exposure fast`, 5 V 2.5 A supply |
| Short baseline (47 mm) | sigma_z ~3.5 cm at 1 m, ~14 cm at 2 m at 1 px | physics | no. Keep range <= 2.5 m |
| Odometry lost while turning, restart loses heading | 36 deg worst error (13 after gyro restart fix) | software | yes: IMU inside odometry (planned below) |
| Neural vs classic matcher | SGBM, LAS2, FoundationStereo all measured the same floor | - | the floor is the rig, not the matcher |

**Push-back.** A from-scratch pipeline on the same rig will measure the same floor. Fast-FoundationStereo already ran here and scored 0.479 px against SGBM's 0.511 px. The plan below is worth building, but its gains are bounded until three rig items are done (same-way-up cameras, rigid mount, stable power). I recommend doing those first, then recalibrating, then building M1.

**Push-back 2.** "Rebuild RTAB-Map" should mean rebuild the pipeline *around* RTAB-Map, not the RTAB-Map library. Your own requirement 3 says to use RTAB-Map's odometry, loop closure and graph. Re-implementing those is months of work with no measured benefit.

**Missing input.** You said you would provide a reference implementation. None is in the repo (`Aria Research/` holds talk videos and `Gabai Lite.md`, not code). I need it, or its name, before M3 and M5 are final.

## 2. What already exists and is reused, not rewritten

| diagram box | existing code | status |
|---|---|---|
| Pi: cameras, IMU, timestamp sync | `gabai/pi/sender.py`, `cams.py`, `imu.py` | done; frame counters, IMU sample numbers, resend on reconnect |
| Pi: KLT + forward-backward check | `gabai/pi/klt.py` (`--klt`, FB check option) | done; 200 pts, ~215 kbit/s |
| Server: synchronized frame buffer | `gabai/pc/syncbuf.py` `SyncBuffer` | done; pair + its KLT + full IMU interval per group |
| Rectification, calibration | `gabai` `Rectifier`, `data/stereo_calib_charuco_45.npz` | done |
| Rolling-shutter correction | `rtabmap_lab/rolling_shutter.py` | done |
| FoundationStereo | `gabai/spike/ffs_runner.py` (PyTorch fp16, 76-86 ms); `third_party/Fast-FoundationStereo/scripts/make_onnx.py`, `run_demo_tensorrt.py` | PyTorch only; TensorRT not wired |
| RTAB-Map odometry + SLAM | `rtabmap_lab/bridge/main.cpp` | done for stereo; no IMU input, no external depth |
| Object memory with fixed IDs | `gabai/objects3d.py` `ObjectMap` (association, label hysteresis, merge, forgetting) | done |
| Detector | `gabai/detect.py` `UltralyticsDetector`, `models/yolo11n-seg.pt` | done |
| Held-out scoring | `rtabmap_lab/tools/recon_compare.py` | done; reused as the acceptance metric |

Installed: torch 2.11 + CUDA, TensorRT, ONNX, Open3D, Rerun, Ultralytics.

## 3. New code: `mapper/`

```
mapper/
  __main__.py       live | replay | build-engine
  depth_trt.py      M1  FFS TensorRT engine: disparity + confidence
  confidence.py     M1  per-pixel confidence from LR check, texture, edges, sigma_z
  motion_check.py   M2  KLT + gyro vs odometry consistency score
  bridge_v2.cpp     M2/M3  RTAB-Map: stereo odometry with IMU, RGB-D nodes with our depth
  cloud.py          M3  semi-dense persistent point store anchored to nodes
  scene.py          M4  objects -> 3D -> ObjectMap -> relations
  run.py            M5  threads and queues, joins by frame counter
tests/test_mapper_*.py
```

### M0. Rig gate (before anything else, ~0 new code)
- Start-up refuses to map when the first pairs' rectified row error is above 1 px (`rectify_check` logic, exists) and says to recalibrate.
- Calibration file comes from the recording's `meta.json` / one config key; the 42 mm file is not a default anywhere in `mapper/`.
- Acceptance: tape-measured walls at 1, 1.5, 2, 2.5 m (`tools/known_distance.py`, exists) give the scale. Mean depth error target under 2% at 2 m.

### M1. Depth + confidence (GPU)
- Export FFS once to ONNX at the rectified size padded to 32 (640x384) with the repo's `make_onnx.py`, build an fp16 TensorRT engine, cache it under `models/`.
- One engine call with batch 2: (L, R) and the mirrored (R', L'). Gives left and right disparity for the cost of about one call on the GPU.
- Confidence per pixel, in [0, 1], product of:
  - left-right agreement: `exp(-(dL - dR_warped)^2 / (2 * 0.5^2))`;
  - texture: local gradient energy, so blank walls get low confidence instead of invented depth;
  - edge: low within 1 px of a disparity jump larger than the noise explains (kills flying pixels);
  - range: `sigma_z = z^2 * sigma_d / (f B)` with the measured sigma_d; zero past `max_range` (2.5 m default).
- Output per frame: metric depth (float32 m), confidence (float16), sigma_z.
- Acceptance (each vs the current RTAB-Map SGBM path, same recording, same scoring): latency under 40 ms per pair on the RTX 5060 Ti; held-out error and wall error lower; confidence calibrated (pixels with conf > 0.8 have half the disparity error of pixels with conf < 0.3).

**M1 result (2026-11-03, 24 pairs of `walk_20261102_230115`, 45 mm calibration, RTX 5060 Ti).** Built: `mapper/depth_trt.py`, `confidence.py`, `export_onnx.py`, `bench_depth.py`, `tests/test_mapper_confidence.py`. Engines in `models/trt/` (`py -m mapper.depth_trt build --iters N --max-disp D`).

| engine | ms per pair (L-R + R-L) | median / p90 disparity diff vs PyTorch FFS | plane scatter conf>0.8 / <=0.3 |
|---|---|---|---|
| 8 iters, 192 px (default) | 98 | 0.044 / 0.85 px | 0.64 / 1.05 px |
| 4 iters, 192 px | 85 | 0.68 / 9.2 px | 0.59 / 1.10 px |
| 4 iters, 96 px | 52 | 2.6 / 38 px | 0.73 / 1.17 px |

The 40 ms target is missed: the 192-shift cost volume dominates, not the GRU loop, and every faster engine loses accuracy. Default stays 8 iters / 192 px (about 1.6x faster per image than PyTorch, and it includes the left-right check that PyTorch needs a second call for). Depth is only needed for map nodes (a few per second), so 98 ms does not block odometry at 15 pairs/s. Confidence is meaningful (high-confidence pixels scatter 40% less than low). Not yet done: real-scene calibration of the thresholds, and tape-measured scale.

### M2. Odometry with IMU, KLT as an independent check
- `bridge_v2.cpp` (copy of the current bridge, not an edit): each FRAME carries the IMU samples of its interval; the bridge passes them as `IMU` objects in `SensorData` so RTAB-Map uses orientation as a motion guess and gravity in the graph (`Optimizer/GravitySigma`). No separate VIO.
- `motion_check.py`: from the pair's KLT tracks (already FB-checked on the Pi), fit the rotation between frames (essential matrix, RANSAC) and compare with the gyro and with the odometry delta. Output per frame: `track_ok`, `rot_disagree_deg`, `inlier_ratio`.
- Use: a frame whose odometry disagrees with both gyro and KLT by more than a threshold is not made a map node and its depth is not fused. KLT never sets a pose.
- Acceptance: lost-odometry frames on `walk_20261102_230115` down from 84; heading error vs gyro over the walk; start-to-end gap on a closed-loop walk.

**M2 result (2026-11-03, `walk_20261102_230115`, pairs 0..300, 45 mm calibration, pictures rectified, one RTAB-Map run per row; the same run repeated varies by about +-15 lost pairs, so differences under that are noise).**
Built: `mapper/bridge/main.cpp` (a copy of the lab bridge, which is untouched; `py -m mapper.build_bridge` -> `mapper/bin/mapper_bridge.exe`; `--imu 1` hands the gyro's attitude to RTAB-Map's odometry as IMU data, `--guess 1` hands the gyro rotation + last translation as `Odometry::process`'s relative motion guess), `mapper/odom_eval.py` (A/B harness), `mapper/motion_check.py` (KLT rotation vs gyro vs odometry), `tests/test_mapper_motion.py` (4 tests).

| run | lost pairs | restarts | inliers | start-end gap | heading error vs gyro (median / sum over the clip) |
|---|---|---|---|---|---|
| baseline (pictures only) | 75 | 5 | 136 | 0.38 m | 0.38 deg / 239 deg |
| gyro turns the pose at restarts (lab bridge) | 75-90 | 5-6 | 137 | 0.23-0.36 m | 0.36 deg / 118 deg |
| `--imu 1` (RTAB-Map IMU orientation) | 75 | 5 | 135 | 0.30 m | 0.38 deg / 129 deg |
| `--guess 1` (relative motion guess) | 75 | 5 | 135 | 0.30 m | 0.38 deg / 122 deg |

Findings:
- **IMU assistance does not reduce lost pairs.** IMU orientation and motion guess both left the count unchanged (75). What the gyro does buy is the restart: the summed heading error over the clip halves (239 -> 118-129 deg) because a restarted odometry keeps its heading. That is already in the lab bridge (`gyro=True`); `mapper_bridge --imu/--guess` adds nothing measurable and stays off by default.
- **Why pairs are lost.** All five loss runs are exactly 15 pairs (the restart threshold). Lost pairs have a sharpness (Laplacian variance) of 111 against 551 for tracked pairs, and 67% of pairs at 0-10 deg/s are lost against 7-15% at 10-60 deg/s: blank or blurred views, not fast motion. A better motion guess cannot create matches that are not in the picture; more light, faster exposure and a stable supply can.
- **KLT as an independent signal works.** From plain LK tracks with a forward-backward check, a rotation-only fit (bearing vectors, RANSAC Kabsch) gives a usable rotation on 81% of pairs and agrees with the gyro to 0.48 deg median / 1.31 deg at the 90th percentile. Pairs where KLT has too few tracks: 56, of which 37 are pairs the odometry lost (66% precision, 49% recall of the 75). Its role is the flag in `motion_check.verdict`: the odometry is blamed only when it disagrees with a measurement AND the other independent one does not contradict that measurement.
- An essential-matrix fit was tried first and dropped: at walking parallax `recoverPose`'s depth cut-off kept only 16% of pairs.
- Not done: gravity alignment (`Mem/UseOdomGravity`, `Optimizer/GravitySigma`) needs a gravity-referenced orientation; the gyro-only attitude has none. Hooking `verdict` into node creation waits for M5.

### M3. Mapping and the semi-dense cloud
- Node input to `Rtabmap::process`: left RGB + our metric depth + odometry pose. RTAB-Map's features, loop closure and graph optimisation are unchanged.
- `cloud.py`, the persistent store:
  - Semi-dense selection: per keyframe, keep pixels with confidence > tau AND image gradient above a threshold (edges, texture, corners), as in semi-dense direct SLAM. Blank-wall pixels are not stored as points.
  - Points are stored in their node's camera frame, keyed by node id. A loop closure only changes node poses, so the world cloud is re-posed by a matrix per node, with no re-fusion and no rebuild.
  - Duplicates: a 5 mm voxel hash in world space; a new point that lands on a voxel with a compatible normal fuses by inverse variance (weight = confidence / sigma_z^2) and increments its view count. Re-hashed after each graph update.
  - Shown and exported only when seen by at least 2 nodes. This removes the single-view fans along viewing rays.
  - Point size is a viewer argument only (`--point-size`, Rerun radius); the stored geometry has no size.
- Outputs: `cloud.ply` (x y z rgb confidence views), `trajectory.txt` (optimised, TUM format), the RTAB-Map `.db`.
- Acceptance via `recon_compare`-style held-out scoring: error and wall error lower than the TSDF baseline (2.86 / 1.86 px after the rolling-shutter fix); roughness on walls; no doubled wall after a loop closure (pose_check disagreement).

**M3 result (2026-11-03, `walk_20261102_230115`, all 860 pairs, 45 mm calibration, gyro restart on, our TensorRT depth gated at confidence 0.5; every 5th node held out; same nodes for every method).**
Built: `mapper/cloud.py` (`PointMap`: node-anchored points, confidence / sigma_z^2 weighted voxel fusion with the voxel size following the depth noise, 5-30 mm, a point shown only when 2+ nodes saw it, PLY export), `mapper/run_offline.py` (bridge + TensorRT depth -> cloud.ply, trajectory.txt, nodes.npz, `--compare`), `tests/test_mapper_cloud.py` (4 tests). Outputs of this run: `data/mapper/walk/`, `data/mapper/walk_noshutter/`.

| method (disparity error vs held-out nodes) | coverage | error median / 90th px | walls px | points or vertices |
|---|---|---|---|---|
| **semi-dense cloud (ours), rolling-shutter fix on** | 0.76 | 4.02 / 21.0 | 3.98 | 75k |
| raw points, 1 cm voxel | 0.95 | 9.01 / 40.1 | 8.99 | 154k |
| surfel (>=2 views) | 0.75 | 4.02 / 21.2 | 4.00 | 249k |
| TSDF 2 cm | 0.79 | **2.23 / 16.3** | **2.16** | 56k vertices |
| ours, shutter fix off | 0.69 | 6.83 / 31.7 | 6.25 | 51k |
| TSDF 2 cm, shutter fix off | 0.75 | 4.89 / 28.4 | 4.56 | 63k vertices |

Findings:
- **The cloud does not beat the TSDF on the full walk.** It equals the surfel map with 3.3x fewer points, and halves the raw-point error, but the TSDF's averaging stays 1.8x closer to the held-out pictures. On the first 200 pairs (little drift) the cloud was best: 0.69 px against 0.72 (surfel), 0.95 (TSDF), 1.05 (raw points).
- **The error on a long walk is pose consistency, not the cloud.** 60 loop closures and 91 lost pairs move poses by more than the depth noise; every method inherits that. A TSDF smooths disagreeing nodes together; a point cloud keeps each node's sheet and hides single-view ones.
- **The rolling-shutter correction matters more than any choice here:** it cuts the cloud's error 6.83 -> 4.02 px and the TSDF's 4.89 -> 2.23 px. It stays on by default in `run_offline`.
- **Persistence works as designed:** a loop closure re-poses nodes without re-fusing (test: moved poses rebuild, unchanged poses return the cached cloud).
- **Speed:** 860 pairs in 129 s (6.7 pairs/s) with a 100 ms depth call per node, in one thread; the 15 pairs/s live rate needs depth in its own thread (M5).
- Not done: normals check on voxel merge; a mesh or TSDF layer under the cloud; incremental (not full) rebuild on a pose change.

### M4. Egocentric perception
- Detection: `UltralyticsDetector` with `yolo11n-seg` on keyframes, its own thread, newest frame only.
- 3D position: median of the mask's depth, weighted by confidence, back-projected with the frame's own pose; an object with too few confident mask pixels is not placed.
- Identity: `ObjectMap` from `gabai/objects3d.py` as-is (it already handles association, label hysteresis, merging, forgetting). Observations are stored per node, so loop closures re-place objects like points.
- Relations (`scene.py`), each a plain geometric rule:
  - `on(a, b)`: a's bottom within 5 cm above b's top surface and a's footprint inside b's;
  - `near(a, b)`: closest box distance under 0.5 m;
  - `left_of / in_front_of` relative to the camera, for spoken output.
- Acceptance: on a recording walked twice past the same desk, each object keeps one ID; its position spread under 5 cm; `on(cup, table)` correct on hand-labelled frames.

### M5. Asynchronous run
- Threads: receive (`SyncBuffer`) -> odometry (bridge) and depth (GPU) in parallel, joined by frame counter -> mapping (node + cloud) -> perception. Depth and perception take the newest frame only; odometry never waits for depth.
- Depth reaches the map only for frames that became nodes (same idea as `--depth-wait-pose`).
- Per-stage latency logged; acceptance: odometry keeps 15 pairs/s while depth and perception run.

## 4. Order and checks

1. Rig: same-way-up cameras, rigid mount, 5 V 2.5 A supply, recalibrate, tape-measured wall clips. (You)
2. M1 + its tests (synthetic plane, LR confidence on a known disparity).
3. M2 + tests (synthetic rotation, KLT vs gyro).
4. M3 + tests (loop-closure re-posing keeps points; duplicates fuse; single-view points hidden).
5. M4 + tests (relation rules on synthetic boxes; ID kept across two passes).
6. M5 integration; one table comparing new vs current pipeline on the same recordings.

Each step is measured against the current pipeline on `walk_20261102_230115` and on the new tape-measured clips before the next starts.

## 5. Not planned
- Rewriting RTAB-Map's odometry, loop closure or optimiser.
- A separate VIO (Basalt etc.), by requirement.
- TSDF or mesh in the new pipeline (the cloud is the product; the old TSDF stays in `rtabmap_lab`).
- Monocular depth fill: it invents depth in exactly the places confidence says not to trust.

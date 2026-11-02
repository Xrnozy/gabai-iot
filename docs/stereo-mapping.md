# Persistent stereo map and camera tracking (added 2026-10-28)

This describes how the Rerun pipeline keeps one map in one world frame while the camera moves, how it works with stereo only, and what changes when an IMU
is added. Numbers are measured on this PC; the section "Results" says exactly what each number is and is not.

Rough-motion tracking, relocalisation, loop detection and the IMU seam: see [visual-slam-research.md](visual-slam-research.md).

## 1. What Aria does, and what was taken from it

Read for this work (Project Aria documentation, "MPS semi-dense point cloud" page, and the Aria Gen 2 announcement):

- The semi-dense cloud comes from tracks that start at pixels with a strong image gradient and are followed through later frames. Each track is a 3D point
  stored as an inverse distance along a ray from its first observation.
- Every point carries an uncertainty (`inv_dist_std`, `dist_std`). The documentation says every track is output whatever its quality, and the user is
  expected to threshold on these values (nominal limits 0.005 1/m and 0.01 m) or weight by them.
- Points are expressed in the world frame of the closed-loop trajectory, so they are consistent with each other across the whole recording.
- Aria Gen 2 runs SLAM on the glasses themselves (Meta announcement).

I did not manage to read the Aria trajectory page (the address returned 404), so statements below about open-loop versus closed-loop trajectories come from
general knowledge of the Aria tools, not from a page read in this session.

What was adopted, in a much smaller form:

| Aria idea | Here |
|---|---|
| Semi-dense points where the image has gradient | `semi_dense_mask` (already present) |
| Per-point depth uncertainty, thresholded or used as a weight | `depth_std` and `depth_weight` in `gabai/cloud.py`: a point's depth standard deviation is `z^2 * 0.3 px / (f * B)`; the voxel mean is weighted by 1 / variance; `map.max_depth_std` drops points above a limit |
| One world frame for all points | Every frame is placed with its pose into one `VoxelMap`; a voxel is one entry however many frames see it, so overlapping views fuse and do not duplicate |
| Poses made consistent with what was seen before (closed loop) | `gabai/mapref.py`: each frame is registered to the map and the pose gets a small damped correction |
| Keyframes | `gabai/kfvo.py` |

What was NOT adopted: bundle adjustment or a pose graph, place recognition (loop closure from a distant earlier visit), inverse-distance parametrisation,
online calibration, and any inertial term. The system is a small stereo-only odometry with a map-registration correction. It is not Aria's SLAM.

## 2. Stereo-only pipeline

```
frame pair --> stereo depth (LAS2, left-right checked)          stage A, GPU
          --> KeyframeVO: pose from the keyframe's features     stage B
          --> semi-dense points, uncertainty weights
          --> MapRefiner: register to the map, damped correction
          --> VoxelMap: fuse in the world frame
          --> Rerun
```

**Keyframe visual odometry** (`gabai/kfvo.py`). The original `StereoVO` re-seeds its features every frame and chains frame-to-frame motions, so each frame
adds its own error. `KeyframeVO` seeds features once per keyframe with 3D positions from the stereo depth, tracks them with Lucas-Kanade from the keyframe
picture (started from the previous positions so tracks do not slide), and solves the camera pose against the keyframe's 3D points with RANSAC PnP. A new
keyframe is made when fewer than half of the features survive, or the camera moved 12 cm, turned 12 degrees, or 15 frames passed. Each frame also sees every
tracked feature with its own stereo depth; these are averaged (inverse variance, capped weight) into the keyframe's landmarks, so one frame's depth noise does
not stay for the keyframe's life. Lost tracking holds the pose and says why, and starts a new keyframe at the held pose. The world frame is the first
camera pose with z up ASSUMED, as before.

**Map refinement** (`gabai/mapref.py`). Visual odometry knows motion since the keyframe, so errors add up and a surface seen again can land beside its first
copy. The refiner matches 800 of the current frame's points to the nearest map points (k-d tree, rebuilt when the map grows 10 percent or every 20 frames) and
solves the rigid motion (iterative closest point, matches farther than 8 cm rejected, stops when the step is under 0.5 mm). It applies a correction only
when: at least 200 points and 40 percent of them matched, the matched points are not almost flat or in a line, the fit improved by at least 10 percent, the correction is under 6 cm and 1 degree, and the corrections applied so far in total stay under 3 cm + 5 percent of the distance travelled (and 2 degrees + 3 degrees per metre). Otherwise
the pose is exactly what the odometry gave and the reason is counted. An accepted correction is applied at half strength (a single noisy registration must
not make the path jump), and passed to `KeyframeVO.correct()`, which moves the keyframe by the same rigid change so later frames continue from the corrected
pose instead of drifting back.

**Map** (`VoxelMap`, 5 cm voxels). One entry per voxel: weighted sum of points, sum of weights, observation count. Memory is about 48 bytes per voxel and is
bounded by `map.window_m` (voxels farther than 30 m from the camera are forgotten). `map.save` writes the map at the end (`gabai.relocalize.load_map` reads
it). A saved map is not loaded into a later run.

## 3. Stereo-only versus stereo plus IMU

Both modes return the same `PoseResult` and use the same map and Rerun views; only the provider changes (`gabai/pipeline/poses.py`).

| | stereo_only | stereo_imu (Basalt) |
|---|---|---|
| Sensors read | two cameras only; the IMU is never opened | two cameras and the IMU |
| World frame | first camera pose, z up ASSUMED; a camera that starts tilted gives a tilted world | gravity-aligned from the accelerometer |
| Scale | metric only if the stereo calibration is metric (plain video: assumed, not metric) | metric from the stereo baseline, and the IMU constrains it |
| Fast rotation, blur, a blank wall | features lost, pose held, then a new keyframe starts: the path has a gap | the gyro carries the rotation through short visual dropouts |
| Drift | grows with keyframes; the map refiner removes only small drift where the map is in view | lower and bounded in roll and pitch by gravity; yaw still drifts |
| Needs | nothing beyond the cameras | still start, camera-IMU extrinsics, accelerometer calibration, synchronised clocks |
| Map refinement | on | off: Basalt's poses are used as given |

Prepared for the IMU, not done: `KeyframeVO.step(..., dR_prior=R)` accepts the camera rotation between two frames from a gyro. It is used only as a gate:
when the visual rotation differs from it by more than 6 degrees the pose is held and the frame flagged. It does not enter the estimate. Tested with a
supplied rotation; no gyro is connected to it. Real visual-inertial fusion in stereo_only mode would need a gyro and accelerometer model with bias states in
the estimator; none exists here, and stereo_only does not claim it. The only inertial path in this project is Basalt in `stereo_imu` mode.

## 4. Results

All timings: this PC, RTX 5060 Ti, LAS2 `s`, 640 px wide, `data/left.mp4` + `data/right.mp4`, 150 frames from second 30, every 2nd frame, viewer off.
The video has no calibration, so depth is not metric. Tool: `py -m gabai.tools.pipeline_bench` (add `--hands` to leave hands and objects on).

**Speed** (median of one run each; runs on this PC vary by about 1 fps):

| Variant | fps, no hands | fps, with hands | pose ms | map ms | frame p95 ms (no hands) |
|---|---|---|---|---|---|
| Before (frame-to-frame VO, unweighted map) | 17.9 | 14.6 | 8.9 | 7.5 | 95 |
| Keyframe VO | 18.3 | 14.7 | 6.0 | 9.2 | 86 |
| + weighted map | 18.1 | 14.9 | 6.1 | 9.9 | 91 |
| + map refinement (the default now) | 18.0 | 14.7 | 5.6 | 11.3 | 93 |

Throughput is set by the depth model in stage A (about 60 ms per frame). The pose, map and refinement work runs in stage B and overlaps with it, so the
extra map time (about 4 ms) does not lower the frame rate. The refinement costs about 4 ms per call at 13 000 map points and runs every 2nd frame. No
slowdown was measured; a difference under about 1 fps is inside run-to-run noise.

**Accuracy on a rendered wall with known path** (`py -m gabai.tools.vo_bench`, 120 frames, 4 seeds, disparity noise 0.3 px of which 70 percent is the same
every frame): final position error over a 1.16 m path was 1.26 cm (rms 1.61 cm) for the frame-to-frame odometry and 1.20 cm (rms 0.76 cm) for the keyframe odometry, at
3.5 ms per frame instead of 8.4. The depth fusion matters: without it the keyframe odometry had 4.2 cm. A depth error that is the same in every frame (here 70 percent of
the noise) is a scale error that no odometry removes.

**Map consistency on a rendered room, out and back to the start, with a drift of about 6 cm and 2 degrees added to the odometry**
(`py -m gabai.tools.vo_bench --room --noise 0.1 --drift`, 200 frames, 2 seeds):

| | final position error | map points' median / 90th percentile distance to the true walls | voxels |
|---|---|---|---|
| frame-to-frame VO | 7.8 cm | 2.9 / 11.0 cm | 19 916 |
| keyframe VO | 6.3 cm | 2.9 / 10.5 cm | 19 549 |
| keyframe VO + map refinement | 3.2 cm | 1.9 / 5.9 cm | 16 884 |

Fewer voxels for the same room means fewer shifted second copies. Without the added drift the odometry error was already about 1 cm and the refiner
corrected nothing (it only acts when it improves the fit). With 0.3 px of persistent depth noise the refiner did not fire either: the depth noise floor
(about 7 cm at 3 m) is above the drift it could correct.

**Real video** (no ground truth), 700 frames from second 20, refiner as shipped (see section 8 for why it applies almost nothing here): map-vs-depth
alignment median 3.87 cm without the refiner and 3.87 cm with it, last frame 3.95 cm in both; 1572 voxels against 1574; the refiner applied 1 correction in
324 tries. It is harmless on this clip and untested for benefit on real footage.

## 5. Fixed on the way

- The shown map used `core = 0.7`, which dropped voxels having a neighbour seen 1.4 times as often. Observation counts on real video range from 2 to
  15 000 (near surfaces and a still camera are sampled far more), so only 74 of 1041 solid voxels were drawn. Default is now `map.core = 0.2`, and the
  pipeline checks only the 6 face neighbours (4x faster: 7 ms against 29 ms at 13 000 voxels).
- `VoxelMap.integrate` used `np.add.at`; it uses `np.bincount` now, which is faster.

## 6. Limits

- No loop closure. A return to a place after a long walk with more than 6 cm of accumulated drift is not recognised and not corrected; the map then holds
  two copies. Large-scale consistency needs place recognition and a pose graph, which do not exist here.
- Map refinement works only where the existing map is in view and has enough structure. On a single flat wall nothing says where along the wall you are;
  the shift limit keeps this from becoming a random walk (tested), but sliding along a wall is not corrected.
- The depth noise (assumed 0.3 px) is an assumption, not measured on this rig. The weights and the `max_depth_std` limit inherit it.
- Plain video has no calibration: focal length and baseline are assumed, so the path and map scale are unknown.
- Moving objects: hands and held objects are excluded from the map when the hand analysis is on. Other moving things (people) are not, except through
  the registration's far-match rejection.
- Fast motion, blur, low texture and glare lose the features; the pose is then held, and the gap is not filled.
- The synthetic benchmarks use a flat, well-textured scene and a simple noise model. They measure the method, not the real world.
- Stereo_imu does not use the map refiner or `KeyframeVO`; their benefit there is unmeasured. No real IMU recording was run in this work.
- Run-to-run speed noise on this PC is about 1 fps; the speed table is single runs.

## 7. Settings

`config/pipeline.toml`: `[tracking] vo` (`keyframe` or `frame`), `fuse_depth`; `[map] weighted`, `max_depth_std`, `core`, `window_m`, `refine_pose`,
`refine_max_shift_m`, `refine_every`, `save`. To get the old behaviour: `vo = "frame"`, `refine_pose = false`, `weighted = false`, `core = 0.7`.

The summary (`pipeline_summary.json`) has `map_refine` (tries, applied, rejected with reasons, size of corrections), `map_solid_voxels`, and
`map_obs_per_voxel_median`.

## Sources

- Project Aria documentation, MPS semi-dense point cloud: https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_pointcloud
- Aria Gen 2 announcement: https://www.meta.com/blog/project-aria-gen-2-next-generation-egocentric-research-glasses-reality-labs-ai-robotics/

## 8. Runaway map growth (found and fixed 2026-10-28)

**Symptom.** On a longer stretch of the video the voxel count climbed past 500 000, the map-vs-depth alignment went from 3 cm to 250-500 cm, and the video and
3D views broke.

**Reproduced** with `data/left.mp4` from second 20, 450 frames: 1 300 voxels at frame 400, 14 266 at frame 450; alignment 3.5 cm at frame 405, 41 cm at 431,
250 cm at 432. The pose translation looked calm (about 1 cm per frame) while the map walked away, so it was a rotation error. With `map.refine_pose = false`
the same stretch stays at 1 444 voxels, so the cause was the map refiner, not the odometry or the depth.

**Root causes** (three, all mine):
1. The refiner moved the pose by half of each registration's correction, but computed the partial translation from the world origin
   (`Rd_part @ t + gain * td`) instead of interpolating between the old and the new pose. Far from the origin a small rotation became metres of shift (the
   path read 34 m on a 3 m walk). Fixed: the partial correction is now an interpolation between the two poses.
2. The refiner's corrections fed on themselves. On a close, nearly flat scene a registration can always find a small rotation that lowers the residual (it is
   fitting noise); the corrected frame is added to the map; the next frame disagrees with that map, and so on. The limits were per correction (6 cm, 3
   degrees), nothing limited the total. Fixed: a registration whose matched points are almost flat or linear is refused (spread check); the total of the
   corrections is limited by the distance travelled; the per-correction rotation limit is 1 degree and the required improvement 10 percent.
3. Nothing stopped a bad pose from adding frames. Whatever the cause, a frame placed away from the map made a second copy, and the map had no size limit
   (the configuration listed `max_voxels` and `max_pose_step_m` but the code did not read them).

**Guards now in place** (each reported in `pipeline_summary.json`; each one has a test in `tests/test_map_growth.py`):
- `map.max_voxels` (300 000, about 14 MB): hard cap; voxels seen under 3 times go first, farthest first, then the farthest solid ones.
- `map.max_misalign_cm` (15): a frame whose pose puts the map more than this from its own measured depth is NOT added (`frames_skipped_inconsistent`). After
  `map.reset_after_frames` (30) such frames in a row the map is dropped, removed from the Rerun scene, and rebuilt from the current pose (`map_resets`); a
  warning event says so.
- `map.max_pose_step_m` (0.5) in the pipeline and `max_step_m` / `max_step_deg` (0.5 m, 45 degrees) in `KeyframeVO`: a pose that jumps in one frame is not
  trusted, the pose is held, a new keyframe starts, and the frame is not mapped (`pose_jumps`).
- The depth fusion in `KeyframeVO` is applied only to accepted poses, so a rejected pose cannot corrupt the landmarks.

**Points.** The map is drawn with 4 mm points (was 12 mm) and the live frame with 3 mm (was 6 mm). The map itself stays one entry per 5 cm voxel (a centre of
mass, weighted by depth precision), so repeated views refine an existing voxel and do not add points.

**Check after the fix.** `py -m gabai.tools.map_growth_check --start 20 --frames 700` (about 40 s; add `--hands`): 700 frames, 1 704 voxels at the end with
the earlier refiner limits and 1 572 with the final ones, no growth step over 50 percent, no reset. This tool exits with 1 when the map passes the cap,
grows by more than 50 percent and 2 000 voxels within 25 frames, or is reset more than once.

**What this does not prove.** The scenes that grew were close, flat, and handheld, from plain video with assumed geometry. The guards bound the damage; they
do not make a wrong pose right. A real drift above 15 cm is handled by dropping and rebuilding the map, which loses the old map.

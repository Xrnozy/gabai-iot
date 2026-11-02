# Persistent semi-dense map, monocular fill, depth correction, loop closure (2026-10-31)

## What the map does now
- **Persistence.** A map point stays until the camera is farther than `map.window_m` (8 m) from it, or the camera itself passes through it (`map.carve_m`, 0.3 m).
  Nothing else removes a point except the hard cap `map.max_voxels`. When the pose and the map disagree for `map.reset_after_frames` frames, only the map within
  `map.forget_radius_m` (3 m) of the camera is dropped. Before, the whole map was dropped.
- **Finer points.** `map.voxel` 0.03 (was 0.05), `map.stride` 2 (was 3). Rerun draws points a fixed number of screen pixels wide, not a fixed size in metres, and
  sends up to 250000 map points with an even stride (it sent a random 60000, which shimmered).
- **Monocular fill** (`stereo.mono = "dav2"`, `stereo.mono_mode = "fill"`). Depth Anything V2 is scaled to the trusted stereo pixels (`gabai/fuse_depth.py`, `DepthFuser.fill`) and
  supplies only the pixels stereo did not measure. They live in a second voxel layer (`SlamCloud.fill_map`) at 0.3 of a stereo point's weight. The stereo map, the pose
  refiner, the alignment check and the saved map never see them; the display adds a fill voxel only where no stereo voxel exists. The fill is refused for a frame when the scaled
  monocular map is more than 1.5 px from the stereo pixels, when it is uncorrelated with them, or when fewer than 300 trusted pixels support the fit.
- **Distance correction** (`gabai/depthcal.py`, `data/calibration/depth_correction.json`, fitted by `py -m gabai.tools.fit_depth_correction --write` on `data/calib_pairs_15`).
  A ratio curve of true / predicted disparity per LAS2 size. The curve is known only where the board was seen, 0.4 to 0.7 m, and fades to nothing outside it.
- **Loop closure** (`tracking.loop_closure`, `tracking.loop_apply`, now on). ORB descriptors of a new keyframe vote for the old keyframes they match (any distance from where the
  odometry thinks they are), PnP on the old keyframe's 3D points verifies (at least 60 inliers; the drift must be below min(2 m, 0.3 m + 15% of the path since)), and the
  same old place with the same drift vector must be found on the next check. Then the loop is an edge of the pose graph (`gabai/posegraph.py`), every keyframe pose is re-optimised,
  and `SlamCloud.rebuild` lays every stored frame back at its keyframe's new pose and fuses the voxel map again. The path is re-laid the same way.

## Measured
| | before | after |
|---|---|---|
| LAS2-H distance error on the board pairs (leave one pair out) | 6.55% | 3.34% (0.4 to 0.7 m only) |
| Monocular fill vs hidden stereo pixels (recordings, median / p90 distance error) | | 5.1% / 16.5% (normal), 5.8% / 25.7% (rapid) |
| Fused cloud, `normal_20261029_235458` replay: points / plane thickness / spacing | 1743 / 1.80 cm / 3.69 cm | 32382 / 1.40 cm / 2.41 cm |
| Synthetic textured room, one lap: gap between start and end of the path | 36 cm | 9 cm |
| Same room: thickness of the wall seen at the start and at the end (middle half of its points) | 21.4 cm | 10.8 cm (`tests/test_loop_closure.py` asserts at most 0.6x) |

Frame time on the replay with model `h` and fill on: median 159 ms (normal), 226 ms (rapid). The old settings were not timed on the same run.

## Not established
- Distance accuracy beyond 0.75 m is not calibrated: no board was seen there. Capture a larger board with `gabai.tools.range_test`, extend the fit, refit.
- Loop closure is proven on a synthetic room with a scale drift, not on a real recording that returns to its start. The two real replays found three loop candidates in the fast
  recording and applied none (all failed the drift, confirmation or inlier checks); so no false closure moved the map.
- The map is persistent inside `window_m`. A place the camera left by more than 8 m is forgotten, as asked; a loop back to it re-adds it, but cannot correct points already dropped.
- The visual-odometry failures at low frame rate (see the stall investigation) are not fixed by any of this.

## Dashboard: depth fusion and camera-POV mesh (2026-10-31)
- **Monocular and fused depth panels** (`fusion/mono`, `fusion/fused`, in the middle column). Mono is the monocular map after scaling onto the stereo pixels (drawn also when the fit was refused, so you can see why);
  fused is stereo plus the monocular pixels in the holes only. Scalars under `fusion/`: `accepted` (1 = fill used this frame), `fit_residual_px`, `fit_correlation`, `filled_px`.
- **Camera POV 3D reconstruction** (own 3D view, entity `world/pov_mesh`, `gabai/povmesh.py`). A mesh of the current fused depth, textured with the left picture, rebuilt every drawn frame and placed with the current pose.
  Triangles are dropped where a corner has no depth, at depth jumps and at edge-on surfaces, so holes stay holes. It is cleared while the pose is held. It is drawn in the main 3D view (placed with the same pose as the
  camera and the cloud) and again in its own view. The depth is smoothed first (bilateral, keeps real steps), 4 px grid. Not done: vertex normals (Rerun's converter costs 50 ms per frame), so there is no smooth shading.
- Checked: 5 unit tests and a 40-frame replay of the rapid recording (all panels and the mesh logged each frame, median 10119 triangles). Not checked: how it looks to you in the viewer, and the cost of the extra drawing at the full frame rate (it follows `perf.viz_every`).

## Only real surfaces go into the map (`map.surface_filter`, `map.fill_reach_px`)
`gabai/povmesh.py: surface_mask` keeps a pixel only if it has valid depth, is not within 1 px (half size) of a depth jump, is not seen edge-on (slope > 16 per metre sideways) and belongs to a connected patch of at least 50 px (half size).
The map, the pose refiner, the alignment check and the mesh all take these pixels only, so they show the same geometry. The monocular fill also stops `fill_reach_px` (40) px from the nearest trusted stereo pixel.
Measured on `normal_20261029_235458` (model h, 404 frames): 87-90% of valid pixels survive per frame; solid map voxels 5733 -> 3510, thickness 0.99 -> 0.97 cm; the long streaks along depth edges in the front view are gone
(`data/reports/cloud_compare_*.png`). Not shown: that every removed voxel was a ghost, or that the points no longer move when the camera moves (the pose is unchanged by this).

## Monocular on both pictures, fused with stereo (`stereo.mono_pair`)
The monocular model runs on the left and the right picture (one batched call, own CUDA stream and thread, at the same time as the stereo model). The right map is fitted to the stereo pixels with its own scale and offset and put into
the left view; a hole is filled only where it agrees with the left map within 2 px, and the two are averaged there. Panels: `fusion/mono` (left, model's own metres), `fusion/mono_right`, `fusion/fused`; plot `fusion/left_right_mono_agree`.
Hold-out test (`scripts/evaluate_mono_pair.py`, 24 frames of `normal_20261029_235458`, hidden stereo patches): error vs the hidden stereo distance, median / p90: left only 5.6% / 23.2%; with agreement within 2 px 5.8% / 16.5%,
keeping 60% of the pixels (1 px: 5.0% / 15.2%, 33%; 4 px: 6.9% / 18.4%, 88%). So the check removes the worst fills; it does not improve the median. In the real holes of a replay frame only about 30% agreed
(real holes are harder than hidden patches), so the fill is sparser than before. Cost: the monocular model adds about 17-21 ms per frame to a 62-70 ms stereo frame (one GPU).

## The map no longer collapses when the pose and the map disagree (`map.recover_every`, `map.forget_in_view`)
Before: `reset_after_frames` (30) frames in a row that disagreed with the map dropped everything within `forget_radius_m` (3 m) of the camera - indoors, the whole room (map voxels 17,109 -> 80 on `rapid_20261029_235536`).
Now: every `recover_every`-th such frame registers itself to the map from far away (`MapRefiner.recover`: match radius 0.6 m -> 0.08 m; applied only if at least 50% of the points land within 8 cm, rms under 4 cm, not a flat sheet,
at most 1 m / 35 deg) and the odometry is corrected; if that keeps failing, only the map inside the camera's current view is dropped (`SlamCloud.forget_in_view`), not the part behind it.
Measured on the rapid recording (427 frames, model h): old 1 drop, 17,109 -> 80 voxels, end 14,705; view-only forgetting alone: end 30,289 (never below its peak); with recovery: 0 drops, the pose fitted back 33 times, end 24,022.
Not changed: 273 of 427 frames are still skipped as inconsistent and the median map-vs-depth gap is 19.8 cm, so the disagreement itself (fast rotation, poor depth) is not fixed, only its cost to the map. A recovered pose has not been checked against ground truth.

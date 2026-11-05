# Reconstruction failures, causes, and a SemanticFusion-style replacement for the TSDF (design, not built)

Written 2026-11-03. Evidence: `walk_20261102_230115`, the Rerun screenshots of the live run, `mapper/odom_eval.py` sweeps, `docs/mapper-plan.md`. Rule from the request: RTAB-Map's visual odometry only, no IMU, no gyro, no separate VIO. `mapper/run_offline.py` now does exactly that by default (gyro and rolling-shutter correction are opt-in flags).

## 1. What the failures are (measured)

| symptom | cause | evidence | fixable in software? |
|---|---|---|---|
| Odometry lost in runs of exactly 15 pairs (the restart threshold) | The pictures hold nothing to track. A lost picture is near black (mean brightness 44, Laplacian variance 6) facing a blank wall. Tracked pictures have variance 558. | Lost pairs: 22 stereo-consistent ORB matches against 289 tracked; odometry inliers 0 against 136. 67% of pairs at 0-10 deg/s are lost, so it is not fast motion. Looked at the frames: black wall vs a bright monitor. | No. Light, a longer exposure budget or a different view. Visual-only odometry cannot track a black wall. |
| Fanned, doubled walls; path knotted into a ball | Every lost run ended in a manual restart that assumed the camera had not rotated, with a confident link across the gap. After the first restart all later headings are wrong, and loop closure then forces inconsistent places together (113-123 closures on a mostly turning-in-place walk). | Heading error against the recorded gyro (judge only, never an input): 239 deg summed over 300 pairs without any heading help, against 118 with. | Partly. Fixed the made-up link: the bridge now uses RTAB-Map's own `Odom/ResetCountdown` (large covariance, new map session, loop closure joins sessions) and shows every session in the graph. The unknown rotation during a blind gap cannot be recovered without another sensor; only a revisit (loop closure) can. |
| The monitor is unrecognisable | The monitor is text on a bright screen, heavily motion-blurred into horizontal streaks (`data/_lost_vs_good.jpg`). Horizontal streaks are parallel to the epipolar line, so stereo matching along the row is ambiguous there: the disparity on the screen is poor whatever the matcher. | Sharp frame variance 914 and still smeared; matches on the screen are repeated lines. | Partly: confidence already marks it low; more exposure control and a vertical-gradient check would help. Blur is a capture problem. |
| Walls not flat in the TSDF | The TSDF averages whatever poses it is given; with poses that disagree by degrees it averages two walls into a thick one and carves holes. | Held-out error 2.2-3.8 px, run to run (RTAB-Map is not deterministic; one run varies by about 1-2 px, so single-run comparisons under that are noise). | Not by the reconstruction. Fix the poses first. |

Gating nodes by odometry inliers and sharpness (`--min-inliers`, `--min-sharp`) was tried: the held-out error moved from 3.4 to 2.8-4.0 px depending on the threshold, inside the noise. It stays an option, off by default.

**Consequence for the request.** A better fusion method cannot make walls flat if node poses disagree by more than the depth noise. `rtabmap_lab/surfel.py` is already an ElasticFusion-style surfel map with projective association and confidence; it scores the same as my cloud on this walk. Replacing the TSDF with another fusion scheme changes how pose errors are averaged, not the pose errors.

## 2. SemanticFusion, as it applies here

The paper (McCormac et al., ICRA 2017) has three parts:
1. **ElasticFusion SLAM** supplies a global surfel map. Each frame is tracked (ICP + colour), every surfel is projected into the camera, and a measurement that lands on a surfel with compatible depth and normal updates it (confidence-weighted mean of position, normal, colour); otherwise it creates a new surfel. Loop closure bends the whole map with a deformation graph, so surfels move but are never re-created.
2. **A CNN** gives per-pixel class probabilities, run on every few frames (51 ms of the 67 ms budget).
3. **Recursive Bayesian label fusion.** Each surfel keeps a class distribution, initialised uniform. For every surfel visible in a frame, the pixel it projects onto multiplies in that pixel's class probability and the result is renormalised: P(l | frames 1..k) = Z^-1 P(l | frames 1..k-1) P(O_u = l | I_k). The SLAM correspondences decide which pixel updates which surfel. An optional fully-connected CRF over surfel position and colour regularises labels; the paper found it gave small gains and could drown the evidence if run too often.

### Mapping to our system

| SemanticFusion | here |
|---|---|
| ElasticFusion tracking | RTAB-Map visual odometry + pose graph (required) |
| Surfel map, projective association | `rtabmap_lab/surfel.py` already does this; reuse it as the geometry layer |
| Deformation graph for loop closure | Not available. Surfels are anchored to RTAB-Map nodes (as `mapper/cloud.py` does) and re-associated after each graph update, so a loop closure re-poses and then re-merges overlapping surfels |
| Depth confidence | FoundationStereo confidence already in `mapper/confidence.py`: weight = confidence / sigma_z^2 |
| CNN, RGB-D | YOLO11n-seg (`models/yolo11n-seg.pt`, already in the repo): per-pixel probabilities from instance masks and scores, one "background" class for unmasked pixels. A dedicated semantic net is a later swap, not needed to prove the fusion |
| Bayesian class update | Implement as written: a probability vector per surfel, multiplied by the pixel's class probability each time the surfel is visible, floored to avoid a zero killing a class for good |
| CRF | Left out first (the paper's own gain was small); revisit if labels flicker |
| Object identity | Surfel labels give regions; `gabai/objects3d.py` `ObjectMap` already keeps persistent object IDs. Connect labelled surfels into instances and associate them with `ObjectMap` (M4) |

### What would be new code
1. `mapper/surfel_fuse.py`: node-anchored surfels with normals, projective association on the node poses, confidence-weighted update, stable/unstable status, free-space removal, plane-aware merge after a graph update. Built from `rtabmap_lab/surfel.py` ideas, written separately so both can be compared.
2. `mapper/semantics.py`: per-pixel class probabilities from YOLO masks, the Bayesian surfel update, label readout, and instance grouping into `ObjectMap`.
3. Rerun layers: surfels coloured by class, per-class confidence, object boxes and IDs.
4. A comparison on the same nodes with the same held-out score, plus a flatness check on walls, table tops and the monitor face (plane fit residual and edge sharpness), so "flat" and "looks like a monitor" are measured, not eyeballed.

## 3. What is displayed from RTAB-Map's odometry today, and the gap
Shown live: the camera pose and path, odometry features, inliers and lost flag over time, loop closures, map nodes, per-stage milliseconds, and events (odometry lost and reset, new map session). **Not shown:** the individual feature tracks on the image, because the bridge does not send RTAB-Map's keypoints back. That needs a new bridge message (keypoints with inlier flags from `OdometryInfo`); not built yet.

## 4. Decision needed before building section 2
The reconstruction cannot get better than the pose graph, and the pose graph is limited by the pictures (dark blank walls, blur). The SemanticFusion-style layer is worth building (it gives persistent merged geometry and consistent labels), but it will not make this particular recording's walls flat. A new recording with light, a short exposure and some texture on the walls is the fastest single improvement, and the same pipeline can be measured on it unchanged.

## 5. Built (2026-11-04): `mapper/surfel_fuse.py`, `mapper/semantics.py`, `tests/test_mapper_surfel.py` (6 tests)

Run: `py -m mapper.run_offline <recording> --out <dir> --semantic [--compare] [--live | --save-rrd f.rrd]`. Outputs `surfels.ply` (x y z rgb label prob). RTAB-Map visual odometry only; no IMU.
- **Geometry:** `SemanticSurfelMap` is `rtabmap_lab.surfel.SurfelMap` (untouched) with the same projective association, normal check, confidence weighting (FoundationStereo confidence / sigma_z^2), see-through removal and probation, plus a class distribution per surfel.
- **Labels:** the paper's recursive Bayesian update, tempered (alpha 0.5) and floored; new surfels start uniform. YOLO11n-seg masks give the pixel evidence. `tv` and `laptop` are one class (`screen`): with two classes the detector split one monitor into two objects.
- **Loop closure:** when the optimiser moves a pose, all nodes are re-integrated with the new poses (no deformation graph), so overlapping surfels merge; the map rebuilt 25 times on the walk.
- **Objects:** DBSCAN groups of same-class surfels -> `gabai/objects3d.ObjectMap` (persistent ids), with the plane-fit residual and extent of each.

Result, full walk (860 pairs, 326 nodes, 164 loop closures, RTAB-Map only, 65 nodes held out):

| method | held-out error px (median / 90th) | coverage |
|---|---|---|
| TSDF 2 cm | **3.75 / 18.7** | 0.83 |
| semi-dense cloud | 8.26 / 28.7 | 0.78 |
| old surfel map | 8.45 / 28.9 | 0.71 |
| semantic surfels (v2) | 8.46 / 28.9 | 0.71 |

Semantics worked: 72% of surfels labelled, one `screen` object (0.76 x 0.30 m in-plane, 55k surfels), no duplicate monitor. Geometry did not improve: v2 equals the old surfel map because it is the same association, and neither beats the TSDF here. The screen's plane-fit rms is 44 mm and its depth extent 0.33 m: poses of the nodes that saw it disagree by more than a monitor's thickness, so no merge rule can flatten it. That is the pose problem of section 1, unchanged. On the first 300 pairs (20 closures) the same code gave rms 28 mm and 0.14 m.

## 6. Plane flattening that drops stacked layers (2026-11-04): `mapper/flatten.py`, `tests/test_mapper_flatten.py` (3 tests)

`--semantic --flatten`. `rtabmap_lab/planes.py` only snaps surfels already within 2.5 cm of a plane, which does nothing when the same wall or monitor exists as parallel sheets 5-30 cm apart (poses disagree). `flatten` instead decides which sheet is the surface: a weighted RANSAC whose score is the summed surfel weight (confidence / sigma_z^2), so the most reliable sheet wins; weighted least-squares refinement; surfels on the sheet (3 cm) are projected onto it; surfels inside the sheet's footprint, facing the same way, 3-20 cm off it, are dropped as a second copy (never moved: that would invent geometry). Planar classes (screen, keyboard, table, book, bed, couch) get their own planes first; then walls, floor and ceiling from unlabelled surfels. People, cups, plants are never touched; surfels outside a footprint (edges, corners, small objects) keep their position.

Full walk, 287 nodes, 143 loop closures, RTAB-Map only, 57 nodes held out (RTAB-Map is not deterministic: compare rows within one run):

| method | held-out error px (median / 90th) | roughness mm | points |
|---|---|---|---|
| TSDF 2 cm | **4.96 / 20.0** | 7.3 | 49k vertices |
| semantic surfels + flatten | 5.96 / 26.6 | **2.4** | 178k |
| semantic surfels (no flatten) | 7.04 / 29.2 | 3.8 | 234k |
| semi-dense cloud | 7.22 / 27.4 | 5.9 | 62k |

Flattening cut the surfel map's held-out error by 15% and its roughness by 37%, and dropped 72k of 278k surfels (26%) as stacked copies. It found two screen planes (0.68 x 0.28 m and 0.41 x 0.33 m): the monitor is seen as two sheets with different tilt, so the rotation error between nodes is not just an offset a single plane can absorb. The "scatter after = 0.0 mm" column in the log is true by construction (members are projected); the held-out error is the real measure. The TSDF is still the closest to held-out pictures.

## 7. Depth source: retinify (2026-11-04)
`--depth retinify` is the default of `mapper.run_offline` (`--depth las2` and `--depth ffs` remain). retinify's library is Linux-only (Ubuntu, a hard-coded /usr/local/cuda, Debian packages, no pip), so `mapper/depth_retinify.py` runs its released ONNX (`third_party/retinify/stereo-matching.onnx`, retinify-models v0.1.7, 10 MB, Apache-2.0) through this machine's TensorRT (fp16 engine cached in `models/trt/`) with retinify's own pre/post-processing (gray float 0-255, resize to 640x480 "balanced", disparity rescaled by the width ratio), plus the same mirrored-pair confidence as the other sources. Not the retinify library itself: its calibration/rectification, occlusion filter and point-cloud code are not used (rectification stays ours, the occlusion filter is replaced by the left-right agreement).

Same 24 pairs of the walk, 45 mm calibration (pair = left-right + mirrored pass + confidence):

| source | ms | plane scatter conf>0.8 / 0.3-0.8 / <=0.3 (px) | depth kept |
|---|---|---|---|
| Fast-FoundationStereo (TensorRT, 8 iters) | 92 | 0.64 / 0.75 / 1.05 | 68% |
| LAS2-S | 33 | 0.83 / 0.89 / 1.17 | 69% |
| **retinify (balanced)** | **11** | 0.75 / 0.82 / 1.12 | 61% |

Full walk with retinify (217 nodes, 98 loop closures, 43 held out; depth 9 ms per node, 860 pairs in 81 s): TSDF 3.88 px, semi-dense cloud 4.23, semantic surfels 4.37, + flatten 4.32 (roughness 1.4 mm against 2.7 unflattened); monitor class 34.8k surfels. RTAB-Map varies between runs, so only the rows within this run compare. Not measured: retinify's accuracy modes (fast, accurate), fp32 against fp16, and anything on a non-rectified pair.

## 8. Why the monitor faced up, and what was and was not broken in the tracking (2026-11-04)
Measured with the recorded gyro and accelerometer as JUDGES only (never inputs; no IMU in any estimate):
- **The odometry and loop closure are mostly right.** On the full walk (24 odometry resets, 195 neighbour + 47 global + 82 local-space loop links) the rotation between linked nodes agrees with the gyro's to 1.0 deg median for neighbour links and 1.6-1.7 deg for loop closures, including 112 closures spanning 100+ frames; none of the loop closures is off by more than 12 deg (3 neighbour links, the reset jumps, are off by 15-31 deg).
- **What is wrong is the frame of the whole map.** RTAB-Map takes the first camera pose as level. This walk starts with the rig looking UP 26 deg and at 56 deg within 3 s. The map's z axis is therefore 48-55 deg from gravity, and a vertical monitor face comes out 52-72 deg from vertical, "facing up". It is a rotation of the world, not a tracking error, and no odometry can see it.
- **Rotation lost at the start and at resets is real but unfixable by vision alone.** Frames 3 to 7: gyro 27 deg, odometry 9 deg (the rig turns at 60-90 deg/s, frames are blurred, 3 are lost, `Odom/ResetCountdown 3` restarts with no relative motion); frames 10 to 68 (58 lost frames, a dark blank wall): gyro 21 deg, odometry 5 deg. Resets policies tried on the first 300 pairs (absolute orientation error against the gyro since the first frame, median): reset after 3 lost: 29 deg, after 15: 36, after 40: 37; never reset: odometry never recovers (296 of 300 frames lost, with or without the motion-guess window and with a bigger local map or fewer required inliers). A visual-only system cannot know a rotation it did not see; a later loop closure is the only repair, and those links are present and consistent.
- **Fix: `mapper/align.py`.** The world is aligned from the mean of the cameras' own up directions (vision only). Against the accelerometer as judge, map up vs gravity: RTAB-Map frame 48 deg median, **aligned 6.5-6.8 deg** (90th percentile 9.2). Refining up with plane normals was built and measured and is OFF: it ended 67 deg away in one run and 20 deg in the next (the monitor face and its stacked sheets dominate this scene's planes). Monitor planes (from `flatten`, 18k and 4k surfels): face-normal elevation 59 and 52 deg in RTAB-Map's frame, 24 and 15 deg after alignment (0 = vertical; a monitor leans back 10-15 deg). The remaining tilt is the 6.5 deg up error plus the pose inconsistency of the nodes that saw it.
- Written next to the usual outputs: `surfels_aligned.ply`, `surfels_flat_aligned.ply`, `trajectory_aligned.txt`, `alignment.json`. In `--live` the Rerun "world" is re-rotated every 5 s from the same estimate (the view origin is "/" so that transform is applied), so the display stays upright as the map grows.

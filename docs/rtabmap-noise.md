# Why the RTAB-Map map is noisy

Written 2026-11-02. Evidence: `rtabmap_lab/README.md`, and the replay of `data/recordings/walk_20261102_230115` (860 pairs, `data/stereo_calib_charuco_45.npz`, baseline 46.84 mm, f 617 px) with `--depth rtabmap` (SGBM) and `--depth gabai` (LAS2 + Depth Anything V2).
No ground truth exists for any of this: no tape-measured distances have been recorded yet, so "error" below means disagreement between measurements, not distance from the truth.

There are three separate causes. They stack, and they need different fixes.

## 1. Depth noise: physics of a short baseline

Stereo depth error grows with the square of distance: `sigma_z = z^2 * sigma_d / (f * B)`. With sigma_d = 0.3 px (what the code assumes), f*B = 617 * 0.04684 = 28.9 px*m:

| distance | depth noise |
|---|---|
| 1 m | 1 cm |
| 2 m | 4 cm |
| 3 m | 9 cm |
| 4 m | 17 cm |

Fusing a wall seen at 3-4 m therefore gives a surface that is centimetres thick, and points streak along the viewing ray. The bridge drops pixels whose expected noise is above `--max-depth-std` (0.08 m, about 2.8 m with this rig) and caps range at `--max-range` 4 m.
Single-picture planes measure 15-20 mm thick at close range (`depth_bench`: LAS2 16.6 mm, SGBM 20.6 mm). That is the floor, whatever the later stages do. A wider baseline is the only real fix; fusion only averages it down.

**Measured, 2026-11-03** (`py -m rtabmap_lab.tools.stereo_stages <recording> --config config/pipeline_calib45.toml --models`): a flat wall is a plane in disparity space, so the scatter around fitted planes is the matcher's own disparity noise, no tape needed.
Walk of 2 Nov, 32 pairs:

| variant | ms | valid | sigma_d | noise at 2 m / 4 m |
|---|---|---|---|---|
| SGBM, bench settings | 87 | 53% | 0.511 px | 7.1 / 28 cm |
| SGBM block 5, 3-way | 15 | 51% | 0.522 px | 7.2 / 29 cm |
| + texture mask | 15 | 34% | 0.520 px | 7.2 / 29 cm |
| + texture mask + WLS | 29 | 33% | 0.505 px | 7.0 / 28 cm |
| LAS2 + left-right check | 36 | 76% | 0.491 px | 6.8 / 27 cm |
| Fast-FoundationStereo | 86 | 100% | 0.479 px | 6.6 / 27 cm |

The real disparity noise is about **0.5 px, not the 0.3 px assumed above**, so the table above is optimistic by about 1.7x (about 7 cm at 2 m, about 28 cm at 4 m). It is the same for every matcher, even the neural one, and no filter moves it: texture masking and WLS remove pixels or smooth them but the remaining scatter stays.
That points at something shared by all of them, not at matching. Candidates, none tested: lens distortion not fully modelled at the picture edges, rectification (row error 0.19 px at calibration), non-planar surfaces counted as planes (inflates), and motion blur. The flat-wall clips (1, 2, 3 m) separate these: a true wall at a known distance gives both the noise and, from the mean, the scale.
The +-1 px inlier band slightly flatters noisy matchers, so compare the 'on plane' share too (not in this table).

## 2. Depth coverage: SGBM leaves holes

RTAB-Map's own dense matcher gives a depth for only 48% of pixels on this walk (LAS2 71%, FFS 91%) and is the slowest of the three (86 ms per pair). On blank walls and in dim light it gives almost nothing, so the surface is patchy there and the few pixels it does keep are the noisiest ones.
Monocular depth (`--depth gabai`) fills the holes, but raw it is not metric: its ratio to stereo ranges from 0.63x to 5.7x between frames. Scaled from neighbouring stereo pixels it is within 5% of stereo for 55% of pixels (median error 4.3%, 90th percentile 16%).
More coverage is not more accuracy: the stereo + mono run has 3.4x the surfels (749k against 218k) and the same pose error.

## 3. Pose error: SMALLER than first claimed (corrected 2026-11-03, see the bottom)

Each node's depth is placed with RTAB-Map's pose. When two nodes see the same wall and their poses disagree, the wall appears twice. `tools/pose_check` projects each node's depth into overlapping nodes:

| run | depth disagreement between overlapping nodes | expected from depth noise alone | added by poses / calibration |
|---|---|---|---|
| SGBM | 6.8% (90th pct 8.8%) | 2.5% | 4.4 points |
| stereo + mono | 6.9% (90th pct 8.8%) | 2.1% | 4.8 points |

The depth source made no difference, so swapping depth sources cannot fix this. The earlier recording (`vio_20261101_014854`) gave 5.5% against 0.7% expected.
Start-to-end gap on the walk: 0.27 m (SGBM) and 0.33 m (stereo + mono), about 5-6% of a 5.4 m path. The pose extent was only 0.34 m per axis, so the walk was mostly turning in place and that drift figure is rough.

Likely contributors, in order of how well they are supported:
- **Stereo-only odometry.** No IMU is used; it matches image features. Blur and blank views lose tracking: the bridge restarts it at its last pose after 15 lost frames (it happened once in the stereo + mono run), and a restart is a pose jump.
- **Frame rate and blur.** The recording has 14.3 pairs per second, 6 pairs missed, and the Pi reports `throttled=0x50005` (under-voltage): exposure is not reported, so motion blur is possible. Not measured per frame.
- **Calibration.** A camera tilted about 1.3 degrees between the 02 Nov morning and evening calibrations. If a camera shifts mid-session the rectified rows no longer match and depth bends. Row error on this walk is about 0.0 px median (spread 0.85 px with ORB), so the current file fits, but nothing checks it during a run.
- **Baseline scale.** If 46.84 mm is wrong, metres are wrong but the shape is not. It does not add noise, only a scale error (see the 42 vs 47 mm discussion; verify with a tape-measured wall).

## What does not help (measured)

- A different depth source: same pose error with SGBM or stereo + mono.
- A finer or coarser TSDF: voxel size did not make it flatter, larger voxels made it worse.
- The GPU TSDF: removes floating debris (10-11% of area to 0%) but plane thickness is 20-22 mm against 16-17 mm for the CPU volume.

## What to try, in order

1. **Record tape-measured wall clips** (1, 1.5, 2, 3, 4 m, plus a blank wall). This turns "disagreement" into real error and settles the baseline.
2. **Fix the pose source, not the depth.** Options: Basalt visual-inertial odometry (the recordings already contain `imu.csv`), or pose refinement against the fused volume (depth ICP, listed as "next step" in the README, not built).
3. **Shorten the working range.** Keep `--max-range` at 2.5-3 m; the noise table shows nothing further out is worth fusing.
4. **Fix the capture.** `--exposure fast` on the sender, a proper 5 V 2.5 A supply for the Pi, and a rig that does not flex between calibration and use.
5. **A longer test walk** that ends on a marked start spot and actually moves several metres, so drift is measured rather than guessed.

## New: `--depth sgbm` and the stage tool (2026-11-03)

- `py -m rtabmap_lab live --depth sgbm` (also `replay`): OpenCV SGBM with a texture mask, speckle removal and WLS refinement (`rtabmap_lab/sgbm_depth.py`). A pixel that fails a check stays empty, nothing is filled. The noise gate uses the measured 0.5 px, so with `--max-depth-std 0.08` it keeps depth to about 2.2 m.
- `py -m rtabmap_lab.tools.stereo_stages <recording> --config ... [--models]` measures sigma_d per variant; `--dump <folder>` saves left, right, raw, no-LR, texture mask, WLS, final disparity, depth and a point cloud per pair.
- Result on the walk: the WLS/texture filters did not lower sigma_d (0.52 -> 0.505 px). They make the depth sparser, not more accurate. The RTAB-Map replay with `--depth sgbm` gave 6.2% pose-limited disagreement against 6.8% for RTAB-Map's own SGBM, but it lost odometry 6 times and had only 10 overlapping pairs to measure, so that comparison is weak.
- Gotcha found: `cv2.ximgproc.createDisparityWLSFilter` / `createRightMatcher` zero the left matcher's left-right check, uniqueness ratio and speckle window (valid pixels 50% -> 84%). `SgbmDepth` sets them back.

## TSDF investigation, 2026-11-03: where the spikes come from, and what did not help

**Correction.** Section 3 above said pose error is the main cause of doubled walls. That was wrong, because the check compared nodes by translation only and used 0.25 px of depth noise. Redone on the 2 Nov walk (45 mm calibration, `pose_check --calib`, 0.5 px noise):
depth disagreement between overlapping nodes is already **5.6%** when the two nodes differ by 1-3 degrees of rotation and **6.8%** at 12-30 degrees (same pattern for SGBM and for LAS2 + mono depth), and 6.1% when they are under 5 cm apart. So poses add about 1 point; the rest is the depth maps themselves
(0.5 px of disparity noise is 4-5% of depth at 3 m, and edges and non-flat pixels are noisier than the plane fit). The walk is mostly turning in place, so this says little about drift over a long walk.

**What the TSDF already does** (`gpu_tsdf.py`): per-pixel truncation from the stereo noise, weight falling with distance, free-space carving, flying-pixel and grazing-angle gates, minimum weight before meshing, small-fragment removal.

**Tried, no useful gain** (local roughness = distance of a mesh vertex from the plane of its 5 cm neighbours, `flatness.roughness`; about 30% of vertices are over 1 cm in every variant, partly real clutter on the desk):

| change | roughness median / spiky | verdict |
|---|---|---|
| current defaults | 6.8 mm / 32% (SGBM dump), 7.3 mm / 34% (LAS2+mono dump) | baseline |
| noise 0.5 px (measured) in the volume | 6.5 / 30%, 6.8 / 31% | within 0.5 mm; fewer pieces but 63% more area on the SGBM dump, so not a clean win; not adopted |
| + isolated-outlier gate (5x5 median) | 6.5 / 29%, 6.7 / 31% | no effect, removed |
| + minimum weight 0.5 | 7.7 / 37%, 7.0 / 33% | worse, loses 60% of the area |
| + noise limit 6 cm | 7.3 / 36%, 7.2 / 34% | no better, loses area |
| pose refinement against the volume (`GpuVolume(refine=True)`, `tools/refine_poses.py`) | disagreement 6.8% -> 6.9% | no gain; 92 of 220 nodes had too few surface points to align; off by default |

**The thickness number in `surface_lab`/`flatness.report` is saturated** (every plane reads about 28 mm, or 17 mm at a 1.5 cm threshold): it measures the width of the band it looks in, not the surface. Use `roughness`.

**Conclusion.** The spikes follow the depth noise, and nothing in the integration or mesh stages removes noise that is in the measurements; only fewer, better measurements (longer baseline, more light, shorter range) or measurements averaged over many views do. Not found or fixed: the actual source of the 0.5 px (see section 1).
`tests/test_gpu_tsdf.py::test_wall_and_sphere...` fails ("second skin" behind the wall, 11k vertices past 2.08 m). It fails with the default 0.3 px as well; I did not run it before my edits, but my edits do not touch that path.

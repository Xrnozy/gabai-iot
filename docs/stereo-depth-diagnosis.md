# Why the DIY stereo depth was bad: root causes, measured

Written 2026-11-03. Recordings: `walk_20261102_230115` (860 pairs, gyro), `vio_20261101_014854` (815 pairs, gyro). Calibration data: `data/charuco_pairs` (25 mm board), `data/charuco_pairs_45` (45 mm board).
No tape-measured distance exists yet, so "error" is disagreement with measurements a method did not use. Every number below can be reproduced with the commands at the end.

## The pipeline, as it runs

Pi (`gabai/pi`): two USB webcams at 640x360, the left camera mounted upside down and turned 180 deg in software, JPEG, sent with the capture times and the left/right offset (`dt_ms`) and the IMU.
PC: `gabai/pipeline/sources.py` PairPrep (layout check, rectification with `Rectifier`, `config/pipeline.toml` `source.calib`) -> `rtabmap_lab/live.py` (orientation, now the rolling-shutter
correction and the gyro) -> RTAB-Map bridge (`rtabmap_lab/bridge/main.cpp`: odometry, loop closure, its own dense disparity -> depth -> `depth` messages) -> `gpu_tsdf.py` / `surfel.py` / `navmap.py`.
Depth is RTAB-Map's own: `Z = f B / d`, f and B from the rectified calibration, d from its matcher.

## Current defaults: pictures only (updated 2026-11-03, gyro features off)

The gyro-based fixes below (rolling-shutter correction, gyro heading at odometry restarts) are **off by default**: `--shutter-fix` / `--gyro` turn them on. RTAB-Map depth is improved with the pictures alone:

| walk_20261102, RTAB-Map's own depth, TSDF vs held-out nodes (gyro off) | error px | walls px | coverage | pieces |
|---|---|---|---|---|
| 42 file (what `replay` used: `config/pipeline.toml`) | 6.02 | 2.24 | 29% | 46 |
| 45 file (the one the walk was recorded with) | 4.02 | 2.21 | 73% | 81 |
| + depth to 2.5 m (same nodes) | 3.72 | 1.96 | 78% | 59 |
| + skip the blurriest 25% of nodes (same nodes) | **3.41** | **1.84** | 73% | 50 |
| all defaults, new RTAB-Map run (its held-out nodes also end at 2.5 m) | 3.74 | 2.18 | 74% | 54 |

Changes: `replay` uses the calibration named in the recording's `meta.json` (`--calib` overrides; with the 42 file only 14% of the walk's pixels got depth against 44%, and odometry inliers halved);
`--max-range` 4 -> 2.5 m; `TsdfMap(blur_keep=0.75)` / `--blur-keep`. RTAB-Map's SGBM settings were left as they are: block 5/7/11, 8-direction mode, uniqueness 15 and doubled P1/P2 all measure 1.0-1.2 px
(robust spread on fitted planes, +-3 px band; the older 0.5 px figure came from a +-1 px band that cannot read much above 0.58). Real per-frame disparity noise is about **1 px**: 3.5 cm at 1 m,
14 cm at 2 m. It follows blur (0.89 px sharpest quarter, 1.21 px blurriest) and, while turning, the opposite rolling shutters below; neither is a matcher setting.

## Root causes, largest first

### 1. Opposite rolling shutters (new, the main one)
The left camera is upside down, so it reads its rows bottom-to-top while the right camera reads top-to-bottom. A pair is not one instant: the right row v is exposed `dt(v) = a + b (v/h - 0.5)` after
the left row v. Measured from SIFT matches of rectified pairs against the recorded gyro (`py -m rtabmap_lab.rolling_shutter`):

| recording | a (middle row) | b (top -> bottom) | gyro explains | row mismatch per pair, before -> after correction |
|---|---|---|---|---|
| walk, 2 Nov | +1.3 ms | **+56 ms** | 57% | 0.97 -> 0.62 px |
| vio, 1 Nov | -1.2 ms | **+45 ms** | 54% | 1.01 -> 0.51 px |

b is the hardware (two readouts of ~25 ms in opposite directions); a is the Pi's grab order (it flipped sign when the device numbers swapped). While the rig turns at its usual 24 deg/s, the right picture
shifts by up to `f * w * b/2` = **6 px** at the top and bottom rows, against 14 px of disparity for a wall at 2 m: depth wrong by up to 40%, opposite at the top and the bottom, and different every time the
turn changes. Rows misalign the same way (pitch), which breaks matching. Every matcher sees the same distorted pair, which is why SGBM, LAS2 and FoundationStereo all measured the same "0.5 px noise"
(`docs/rtabmap-noise.md`), and why depth maps of nodes taken while turning disagreed. Within one wall the error is close to linear in the row, so it tilts walls rather than roughening them: that
is why per-picture filtering never helped.

**Fix implemented** (`rtabmap_lab/rolling_shutter.py`): the right rectified picture is resampled at the left picture's time, row by row, from the gyro rate (first-order rotational image motion,
one `cv2.remap`, about 1 ms). a = each pair's measured `dt_ms`, b = 51 ms from `data/rolling_shutter.json`. Used by `live` and `replay` (`--no-shutter-fix` to compare), the calibration lab and
`recon_compare`. Result on the walk, separate RTAB-Map runs, same scoring:

| | before | after |
|---|---|---|
| depth disagreement between nodes 3-8 deg apart | 1.55 px | **0.92 px** |
| same, 8-20 deg apart | 2.49 px | **1.12 px** |
| same, 20-90 deg apart | 3.54 px | **1.48 px** |
| odometry mean inliers / nodes | 127 / 201 | 137 / 230 |
| TSDF vs held-out pictures, all / walls | 4.20 / 2.99 px | **2.86 / 1.86 px** |
| TSDF coverage of held-out pixels | 73% | 87% |
| TSDF pieces | 94 | 48 |

What it cannot fix: translation during the readout (small at walking speed), motion blur, and a gyro error. **The real fix is mechanical**: mount both cameras the same way up (or, if the left
camera's driver offers `vertical_flip` / `horizontal_flip` — check with `v4l2-ctl -d /dev/videoN --list-ctrls` on the Pi — flip it in the sensor, which normally also reverses the readout, and drop
the software 180 deg turn). Then a constant ~1 ms remains, which the correction also handles. Recalibrate after any such change.

### 2. The cameras move in their mounts, and the pipeline uses an old calibration
`py -m gabai.tools.calib_audit` (report: `docs/calibration-audit.md`). Each dataset fits itself (rows 0.19-0.20 px) but the other one by **13.6 px**: the right camera pitched 1.3 deg between 16:09 and
22:14 on 2 Nov. `py -m rtabmap_lab.tools.rectify_check` on the recordings agrees:

| recording | with the 42 file (morning) | with the 45 file (evening) |
|---|---|---|
| vio, 1 Nov | 1.4 px (0.7 corrected) | 4.8 px |
| walk, 2 Nov evening | **7 px** | 1.05 px (0.75 corrected) |

`config/pipeline.toml` (and so `live`) uses the morning file. If the rig is still as it was that evening, live pairs are 7 px off their rows and RTAB-Map's matcher fails on most of the picture:
the horizontal streaks and holes in its disparity. The pipeline prints the row error of the first live pair ("OK: ..." / "WARNING: ..."): read it at every start. The rig needs a rigid mount;
until then, recheck with the calibration lab's automatic check (pitch and roll from natural scenes) or recalibrate.

### 3. The metre: baseline 47 mm, not 42
Both boards, at their printed size, give **47.10 and 46.84 mm** (0.5% apart; bootstrap +-0.1 mm each). `stereo_calib_charuco_42.npz` is the 25 mm board solved with 22.29 mm squares so that the baseline
comes out at 42 mm (the CAD lens spacing): every depth from it is 0.897 of what the boards say, 10% short. Two independent prints agreeing makes a printing error unlikely; the optical centres are
not where the CAD lenses are. Settle it with `py -m rtabmap_lab.tools.known_distance` (tape-measured walls); not done, no such clips exist.

### 4. The pictures themselves
Dim room, motion blur, a blown-out monitor, blank walls, JPEG from the Pi, and an under-voltage flag (`throttled=0x50005`). Blank walls give no stereo depth with any matcher; that is the
remaining coverage limit. Light, slower turns and a 5 V 2.5 A supply help more than any filter.

## What was checked and is fine
- Calibration resolution = capture resolution (640x360), same rotations (left 180, right 0).
- Corner detection: 30/30 pairs usable in both sets, single-camera RMS 0.17-0.24 px, worst view 0.55 px.
- `Z = f B / d` with the rectified f, disparity in px (RTAB-Map's own); the depth RTAB-Map hands over scales exactly with the baseline it is given (calibration lab test).
- Rectification maps (`initUndistortRectifyMap`, CV_16SC2) match `stereoRectify` exactly (calibration lab test).

Weak points, not fixed: board distances only 0.4-0.9 m and corners reach ~70% of the way to the picture corners, so k2/k3 are poorly determined (k3 1.36 vs 0.63 between sets) and the
picture edges are extrapolated; the left and right focal lengths differ by 3% (different modules, handled by rectification).

## Matchers and reconstruction
Matchers were compared earlier on the same pairs (`docs/rtabmap-noise.md`, `tools/stereo_stages.py --dump` saves left, right, raw, LR check, texture mask, WLS, final disparity, depth and points
per pair): SGBM, SGBM+WLS, LAS2, Fast-FoundationStereo all at 0.48-0.52 px. That shared floor is explained by causes 1 and 2; re-run the comparison after a rigid, same-way-up rig.
Reconstruction methods: `docs/room-reconstruction.md`. TSDF stays: with the correction it is the most accurate on held-out pictures.

## Reproduce
```
py -m gabai.tools.calib_audit                                                         # calibration report -> docs/calibration-audit.md
py -m rtabmap_lab.tools.rectify_check data/recordings/<rec> --calib data/stereo_calib_charuco_45.npz data/stereo_calib_charuco_42.npz --out rect.png
py -m rtabmap_lab.rolling_shutter data/recordings/<rec> --calib <calib> [--save data/rolling_shutter.json]
py -m rtabmap_lab.tools.recon_compare data/recordings/<rec> --calib <calib> --out data/recon/<name> --view
py -m rtabmap_lab.tools.known_distance --calib <calib> data/recordings/wall_100=1.00 data/recordings/wall_200=2.00 ...
py -m rtabmap_lab replay --folder data/recordings/<rec> [--no-shutter-fix]
```
Tests: `tests/test_rolling_shutter.py`, `tests/test_planar_recon.py`, `tests/test_calib_lab.py`.

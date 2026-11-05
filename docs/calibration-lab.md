# Calibration lab: edit the stereo calibration while RTAB-Map replays a recording

Written 2026-11-03. Code: `rtabmap_lab/calib_lab/` (params, recording, runner, autocalib, lab, server, static/index.html). Tests: `tests/test_calib_lab.py` (8, all pass).

```
py -m rtabmap_lab calib-lab                                   # opens http://127.0.0.1:8765
py -m rtabmap_lab calib-lab --folder data/recordings/<folder> --calib data/stereo_calib_charuco_45.npz --clip 0 300
```
`--folder` is the recorded pair folder (read only), `--calib` the saved calibration the session starts from (what "reset" returns to), `--clip` the pairs RTAB-Map processes on every (re)start.
Everything the lab writes goes to `data/calib_lab/` (configs, experiments, measurements) and `data/stereo_calib_lab_<name>.npz` (only when you press export). The recording and `config/pipeline.toml` are never modified.

## What it does

| | |
|---|---|
| Recording | `Recording` reads `left.avi` / `right.avi` / `pairs.csv` with random access. Left and right come from the same pair index, so they cannot drift apart. With the saved calibration the rectified pairs are **bit-identical** to what `live.py replay` produces (checked on the first four pairs). The files are fingerprinted at load and checked after every run. |
| Manual calibration | 25 editable numbers: `K1/K2` fx fy cx cy, `D1/D2` k1 k2 k3 p1 p2, `R` as a rotation vector in degrees, `T` in mm, `baseline_mm` (scales T), `rect.alpha`. Each has a slider, a number box, a reset button and a `?` panel with seven parts (what it does, what it changes, why it matters, what uses it, too high/low/wrong, what to watch). The derived numbers (`rect.f`, `rect.cx`, `rect.cy`, `rect.baseline_mm`, R1/R2 angles, `P1`, `P2`, `Q`) are shown read-only with the same explanations. |
| Instant preview | Every edit rebuilds the rectification maps and redraws the current pair at once: rectified left/right, stereo disparity, row alignment of that pair, valid disparity, disparity noise on flat planes. Click the disparity picture for the distance at a pixel (`f B / d`). |
| RTAB-Map | A `Run` owns a fresh RTAB-Map bridge and an empty GPU TSDF, built from the calibration it was given. Every calibration edit makes the old map **stale** (banner, chip, "old map" on measured points); while playing, a new run starts from the clip start 0.8 s after the last edit, while paused or finished it starts at Play / Restart. Old maps are never reused. |
| Playback | Play, pause, restart, seek bar (pair number and seconds), clip start/end, loop, "restart RTAB-Map on every change" on/off, position of RTAB-Map and of the picture on screen. Seeking while paused only moves the picture; pressing Play at a different pair starts RTAB-Map again there (a new map starts at that pair, RTAB-Map cannot jump). |
| 3D view | TSDF mesh of the run, camera frustum and path, pinned BEFORE mesh (orange wireframe). **Measure**: click the mesh; the point is a position on the metric mesh (no 2D estimate). Shown: x y z, distance from the camera now (follows the live pose), from the camera at the moment of the pick, from the map origin, point-to-point distances. Export JSON + CSV. |
| Before / after | "Pin current as BEFORE" keeps the calibration, its metrics and its mesh. A wipe slider on the left picture and the disparity compares the same pair under both calibrations; the quality and results tables show both. |
| Automatic check | See below. |
| Save / load / compare | Save the working copy (kind `manual`) or the automatic result (kind `auto`, name prefix `auto_`): separate files. Load applies at once (RTAB-Map restarts with it). Compare two calibrations: changed numbers and stored metrics. Replay the clip once per ticked calibration, one after the other. Export as `.npz` for the real pipeline (`Rectifier` reads it identically, tested). |
| Experiments | Each finished run is saved in `data/calib_lab/experiments/<time>_<fingerprint>/`: calibration, recording path and file fingerprint, clip, settings, metrics, `tsdf.ply`. |

## What can change while it runs, and what restarts RTAB-Map

| change | takes effect | why |
|---|---|---|
| any of the 25 calibration numbers | preview at once; **RTAB-Map restarts** from the clip start with an empty map | `f, cx, cy, baseline` are given to the bridge once at start and enter the odometry; every depth in the TSDF is `f B / d`. Test: scaling the baseline by 1.10 scales RTAB-Map's own depth for the same pair by 1.10. A map mixing two calibrations has no meaning, so none is ever kept. |
| play / pause / seek / loop / clip | immediately | no geometry changes |
| view, measure, pin, wipe | immediately | display only |

A restart costs about 8 s per 120 pairs (about 22 s for 300) on this PC. There is no calibration change that is safe to apply to a running map, so none is offered.

## Automatic check (no board)

Matches natural features (SIFT, ratio test, essential-matrix RANSAC) **inside each left/right pair** and fits the relative geometry. Only pitch and roll of the right camera (`R.rx_deg`, `R.rz_deg`) are ever applied. Pairs are skipped when the rig turns fast enough to shift the right picture by more than 0.25 px between the two exposures (`f * omega * dt` with the recorded gyro and the left/right offset; the walk's median is 0.33 px, 10th percentile 0.11), when blurred, or when there are too few consistent matches.
It reports a verdict and these checks, each pass or fail: enough pairs, enough matches, covers the picture (12 regions), near and far both seen, model fits the matches, each applied parameter is well determined (standard deviation from the cost curvature, and agreement between even and odd pairs), the change improves the rows of **held-out pairs measured on the real rectified pictures**, the correction is small. A failed check gives `INSUFFICIENT DATA` or `UNRELIABLE (not applied)` and nothing is proposed.

**It cannot know, and never changes:** the baseline length (a room has no known size: set it from a tape-measured distance with the measure tool, "set baseline"), the yaw `R.ry_deg` (it shifts disparity, not rows) and the principal point x. The baseline direction (`ty`, `tz`) and the right focal ratio are estimated but only shown: on this footage they were unstable between even and odd pairs and applying them made rows worse (0.61 -> 0.96 px), so they are not applied.

Measured on `walk_20261102_230115`:

| test | result |
|---|---|
| saved calibration `stereo_calib_charuco_45.npz` | `CONFIRMED`: 28 usable pairs, rows 0.61 px, no change helps |
| the same calibration damaged by hand (+0.4 deg pitch, -0.3 deg roll) | `RELIABLE`: estimates -0.358 deg and +0.303 deg (true -0.348, +0.349, within 0.05 deg); rows of held-out pairs 4.18 -> 0.72 px |
| synthetic correspondences with a known error (test) | pitch and roll recovered to 0.03 deg; yaw has a larger standard deviation than pitch |
| one corner of the picture only, or 3 matches | `INSUFFICIENT DATA`, nothing proposed (tests) |

On the first 300 pairs only (the rig mostly turning) the check found 8 usable pairs of 40 and refused: that is why it defaults to the whole recording.

## A/B on identical input (first 200 pairs of the walk)

| run | lost frames | depth disagreement between overlapping nodes | TSDF roughness |
|---|---|---|---|
| saved calibration | 60 | 4.7% | 8.0 mm |
| `K2.cx` + 3 px | 60 | 6.3% | 7.4 mm |

A wrong principal point makes the map less consistent (depth disagreement is the number to watch); roughness did not move, because it mostly measures depth noise. One run each; the lab feeds the same pairs in the same order every time.

## Not done, and cautions

- **Recorded video only.** The live Pi stream is not wired in. To tune a calibration for live use, record with `gabai.tools.record_pairs`, tune here, then press "Export as .npz" and point `calib` in a pipeline config at it.
- The rows metric is not calibration alone: pairs exposed while turning have row error from the motion. The page says so per pair and the run metrics only sample pairs under 0.25 px of smear.
- "Depth disagreement" and roughness are consistency measures. **No tape-measured distance has been recorded for this rig**, so none of this proves metric accuracy: use the measure tool on a wall at a known distance and set the baseline from it (this assumes the focal length is right).
- The 0.5 px disparity noise floor found earlier (`docs/rtabmap-noise.md`) is not touched by calibration edits; a better calibration removes bias and inconsistency, not that noise.
- The preview disparity is OpenCV SGBM (block 9, no WLS); RTAB-Map's own disparity is shown beside it, and they differ.
- The odometry loses tracking in the first 120 pairs of this walk whatever the calibration (60 of 120 pairs lost): compare runs on a clip where it tracks if you want the odometry rows to mean something.
- Leftovers from my testing are in `data/calib_lab/` (one test config `test_baseline_plus5`, one measurement export, a few experiments). I could not delete them (the shell refused the removal); delete the folder's contents if you want a clean start.

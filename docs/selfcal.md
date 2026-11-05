# Checkerboard-free stereo self-calibration (`mapper/selfcal/`)

Written 2026-11-04. Nothing from a previous calibration is used as input. Nothing existing is written or changed: output goes to `data/selfcal/run_<time>/`, created fresh (`exist_ok=False`, files opened `"x"`); the only shared file is `data/selfcal/active.json`, written by `activate --confirm`.

## What is estimated
Per camera: focal length (square pixels), principal point, k1, k2. Stereo: the rotation between the cameras and the **direction** of the baseline. Per frame: the left camera's pose. Per feature track: a 3D point. Tangential distortion and k3 are not observable enough from natural scenes and stay 0.
**Scale is not observable from pictures.** The result is in baseline units; `--baseline-mm` (a caliper measurement of the lens centres) makes it metric. Without it `calib.npz` is saved with `metric=False` and cannot be activated.

## Method (and why this combination)
1. **Frames:** only rig-still frames (median flow < 0.5 px), left/right capture offset <= 3 ms, sharp and lit, spread in time. The cameras read their rows in opposite directions, so any motion becomes a row error that would be learned as calibration. A walk gives zero usable frames (flow 18-23 px on every recording of the walk and vio sets).
2. **Correspondences:** SIFT on a contrast-equalised picture, mutual ratio-test matches, one fundamental-matrix RANSAC per picture pair (rejects moving objects and wrong matches that break rigid two-view geometry), tracks by union-find with ambiguous tracks dropped.
3. **Focal length:** essential-matrix criterion (Mendonca-Cipolla) over all left-left, right-right and left-right fundamental matrices, no target. A flat cost curve means the views do not constrain it: **refused**.
4. **Stereo init:** robust rotation-only fit of the bearing vectors, refined by the Sampson error with the baseline on the rig's x axis (a free direction slides toward forward motion while f is wrong); the side is fixed by cheirality.
5. **Bundle adjustment:** sparse trust-region on all parameters with a graduated Huber loss (30, 8, 3, 1.5 px: a tight loss on a wrong model discards the focal-length error as outliers, measured), first with the weakly observed principal point and distortion held, then free; outlier observations dropped between rounds.
6. **Uncertainty:** covariance from the Schur complement of J^T J (points marginalised) scaled by the residual variance, reported per parameter.

## Refusals (nothing is guessed)
Fewer than 10 usable frames; fewer than 5 fundamental matrices; focal length not determined; fewer than 3 frames placed; reprojection rms > 1 px after the solve; held-out validation failures (below).

## Validation, on frames the solve never saw
25% of the chosen frames are held out. The new calibration rectifies them and SIFT matches between the rectified left and right are measured: median row error <= 0.40 px, 90th percentile <= 1.0 px, >= 150 matches, >= 95% positive disparity (catches swapped or rotated cameras). `--compare file.npz` evaluates an existing calibration read-only on the same frames, so old and new can be compared. Not included yet: a depth-consistency check against a measured distance (needs a tape-measured wall: `rtabmap_lab/tools/known_distance.py`).

## Online refinement gate
`output.gate_update(old, new)`: a newer solve replaces the current one only if its focal length is determined to 1%, its reprojection error did not rise, no parameter moved by more than 4 sigma, and it is not less certain. Otherwise the reasons are reported and the current estimate stays. Not wired to a live stream yet; it needs still frames at varied orientations to arrive, which a walking rig does not provide.

## Integration
`py -m mapper.run_offline <recording> --calib data/selfcal/<run>/calib.npz` (or `--calib active`) feeds the new focal length, principal point, baseline and rectification to the RTAB-Map bridge, FoundationStereo depth (`StereoTrt(f, B)`) and the reconstruction. The default stays the existing file until `activate --confirm`.

## Tests
`tests/test_selfcal.py` (5): a synthetic room with a known rig (f 612/598, k1 0.12/0.10, 3% outliers, 0.3 px noise) is recovered to focal 0.03%, principal point 0.3 px, rotation < 0.1 deg; refuses with no rotation between views; the npz loads in the pipeline and the estimated rig puts true points on the same rows (median 0.02 px); the update gate; the CLI refuses on a static recording without writing a calibration.

## What the existing recordings give: nothing, and the tool says so
Every walk and vio recording has no still frame. The still recordings are one fixed view each (shift between chosen frames 0.1-0.2 px), so the focal length is unobservable (cost curve flat; estimate wandered 243-268 px, flagged POORLY DETERMINED) and the run is refused. No real-rig calibration was produced; no number from this tool should be used on real data until a capture like the one below is made.

## Capture protocol (no board)
In a textured, well-lit room with things 0.7-4 m away, hold the rig still for about a second at 20-30 different orientations (pan, tilt and roll by 15-30 degrees between holds). Record with the normal recorder; keep the exposure fast. Then:
```
py -m mapper.selfcal calibrate data/recordings/<rec> --baseline-mm <caliper value> --compare data/stereo_calib_charuco_45.npz
py -m mapper.selfcal activate data/selfcal/<run> --confirm      # only after the report shows validated and activatable
```

# One-time stereo calibration

Run this once (and again only if you move or re-aim a camera, or change the 640x480 size). It captures checkerboard pictures from both cameras, computes the lens and stereo geometry, and saves `data/stereo_calib.npz`. The live viewer then loads it automatically and gives real distances in metres.

## What you need
- The board: `data/checkerboard_9x6.png` (10x7 squares, 9x6 inner corners). Best printed on A4/A3 paper and glued flat to a stiff board. A laptop or monitor screen also works if you turn brightness up, avoid glare, and keep it full size.
- The square size in mm, measured with a ruler (one black square's side). The whole result scales with this number, so measure carefully. Do not use the pixel size of the PNG.
- Good light. The cameras should show the board clearly, not dark.
- The Pi streaming (see live-link-guide.md). Keep the cameras fixed on their bar; do not move them after calibrating.

## Run
PC, in the project folder (replace 25 with your measured square size):
```powershell
py -m gabai.tools.calibrate_stereo --square-mm 25
```
Pi:
```bash
cd ~
python3 -m gabai.pi.sender --host 192.168.1.64
```
A window shows both cameras with detected corners drawn. A pair is saved automatically when the board is seen by BOTH cameras, is still for a moment, and is in a new position. A counter shows `n/25`. Green dots mark where you have already covered the image.

## How to move the board (this decides the quality)
- Cover the whole image: all corners and edges of both cameras, not just the middle.
- Vary distance: close (about 40 cm) through about 1.5 m.
- Tilt it left, right, up and down by up to about 40 degrees. Include some flat-on views.
- Keep the board flat and the whole pattern visible to both cameras every time.
- Hold still until it says CAPTURED, then move to a new spot.

At 25 pairs it calibrates and prints the result. You can also press `c` earlier (minimum 10 good pairs) or `q` to quit.

## Reading the result
```
baseline 6.50 cm   focal 419 px
stereo reprojection error 0.13 px   epipolar (row) error 0.03 px
verdict: GOOD
```
- GOOD: reprojection and row error both under 0.5 px. OK: under 1 px. POOR: recapture with a flatter board, more light, more varied angles.
- Compare the baseline with a ruler measurement between the two lens centres. If it is more than about 3% off, your square size is wrong.
- Pairs are kept in `data/calib_pairs/` and the result in `data/stereo_calib.npz` and `.json`.

## Use it
```powershell
py -m gabai.spike.demo_viewer --source live --model m
```
It prints `using calibration data/stereo_calib.npz` and the panel shows `CALIBRATED: B=... cm f=... px (metric)`. Use `--no-calib` to ignore it.

## Redo from saved pictures without the Pi
```powershell
py -m gabai.tools.calibrate_stereo --images data/calib_pairs --square-mm 25
```

## Verified vs not
Verified on synthetic board images (offline and through the live network path with automatic capture): the true 6.50 cm baseline was recovered to within 0.01 cm and the verdict was GOOD. Not verified: your real cameras, real lens distortion, printed-board flatness.

## Camera order matters (found on your rig)
Your first calibration had the cameras swapped: `/dev/video0` is physically on the RIGHT and `/dev/video2` on the LEFT, so every disparity came out negative and depth was wrong by 20% to 150%. The tool now prints a SWAPPED warning when that happens.
Your current `data/stereo_calib.npz` is the corrected one (left = `/dev/video2` turned 180 degrees, right = `/dev/video0`). Run the Pi like this:
```bash
python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0
```
The old calibration is kept in `data/backup/` in case you need it.

## Accuracy measured on your rig (`py -m gabai.tools.benchmark_depth --pairs-dir data/calib_pairs_swapped`)
Ground truth = the checkerboard's true distance from solvePnP, 21 pictures at 0.42 to 0.85 m. Board area only, a flat high-contrast target, so this is the easy case.
| Model | median error | 90th percentile | model time |
|---|---|---|---|
| LAS2-S | 0.5% | 1.3% | 27 ms |
| LAS2-M | 0.5% | 1.2% | 36 ms |
| LAS2-H | 0.4% | 1.1% | 55 ms |
| Depth Anything V2 Metric Small | 77% (absolute scale wrong, shape fine) | 216% | 16 ms |
Not measured: distances beyond 0.85 m, textureless or shiny surfaces, moving subjects. The scale rests on the 25 mm square size you entered.

## Capturing 15 stereo pairs at 640x360

    Pi:  python3 -m gabai.pi.sender --host <PC_IP> --width 640 --height 360 --fps 10 --capture-fps 15
    PC:  py -m gabai.tools.capture_calib_pairs --square-mm 19.4

Measure one printed square with a ruler first and pass that number; the calibrated baseline scales with it. The tool shows both pictures side by side with the detected corners, a counter (`3/15 pairs saved`) and the pose to take next. Press SPACE to save, U to undo, Q to quit. A pair is saved only when the whole board is found in both pictures, the board is steady, the two cameras were captured less than 10 ms apart (`--max-offset-ms`), and the board is in a new place (`--min-shift`). The reason a pair is refused is printed on screen.

Take the 15 poses in this order. Keep the whole board inside both pictures every time, and hold it still while the status says READY.

1. centred, flat, medium distance (about half the picture wide)
2. centred, flat, close (about 80% of the picture wide)
3. centred, flat, far (about 25% of the picture wide)
4. flat, at the left edge
5. flat, at the right edge
6. flat, at the top edge
7. flat, at the bottom edge
8. turned about 30 deg around the vertical axis, left side nearer
9. turned the other way, right side nearer
10. tilted about 30 deg, top edge away from the cameras
11. tilted about 30 deg, bottom edge away
12. rotated about 30 deg in its own plane, centred
13. top-left corner of the picture, tilted
14. bottom-right corner of the picture, tilted
15. close and strongly tilted (about 45 deg)

Pairs go to `data/calib_pairs_15` as `pair_01_L.png` / `pair_01_R.png` ... `pair_15_L.png` / `pair_15_R.png`, with `capture_log.json` (offset between the cameras, board position and size per pair) and `rotation.json`. The pictures are turned by `--rotate-left 180 --rotate-right 0`, the same turns the live pipeline uses; pass other values only if the cameras are mounted differently. After the 15th pair the calibration runs into `data/stereo_calib_15.npz` and prints its baseline next to the 42 mm of the printed case. `data/stereo_calib.npz` is not touched, and the depth pipeline is unchanged. Run the calibration again from the saved pairs with:

    py -m gabai.tools.calibrate_stereo --images data/calib_pairs_15 --square-mm 19.4 --out data/stereo_calib_15

## The calibration made from the 15 pairs (2026-10-29)

Commands, in this order:

    py scripts/analyze_stereo_pairs.py --square-mm 19.4 --layout capture --out data/stereo_calib_new_c1 --devices /dev/video0,/dev/video2
    py scripts/analyze_stereo_pairs.py --square-mm 19.4 --layout upright --out data/stereo_calib_new --devices /dev/video2,/dev/video0
    py scripts/compare_calibrations.py

The first file is the pairs exactly as saved (the Pi sender ran with `--left /dev/video0 --right /dev/video2`, the old default, pictures upside down). The second is the same cameras read in the other order (every picture turned 180 deg and left and right swapped), which is what the sender's new default `--left /dev/video2 --right /dev/video0` gives: upright pictures. A calibration is only valid for the order it was made with, and the turns (180 / 0) are the same for both, so the file now records the cameras (`left_dev`, `right_dev`) and the live run checks them. The pipeline uses `data/stereo_calib_new.npz` (`source.calib`). Neither command reads or writes `data/stereo_calib.npz`.

Start the live system:

    Pi:  python3 -m gabai.pi.sender --host <PC_IP> --width 640 --height 360 --fps 10 --capture-fps 15
    PC:  py -m gabai.tools.rerun_pipeline --source live --mode stereo_only --model h

(The Pi needs the current `gabai/pi` files; the defaults there are the cameras of `stereo_calib_new.npz`.) The PC prints, before the first picture:

    stereo calibration: data/stereo_calib_new.npz
      made 2026-10-29 ...  layout upright  lens model k1k2  640x360  pictures turned 180 / 0 deg
      cameras: left /dev/video2, right /dev/video0
      baseline 41.72 mm (T = -41.7 -0.0 -0.6 mm), focal 618.0 px, reprojection 0.535 px, row error 0.127 px, 15 board views, square 19.4 mm
    OK: camera order confirmed: the sender reads left /dev/video2, right /dev/video0, as the calibration was made
    OK: row alignment of the first pair is 0.0x px: the calibration fits the pictures

If the sender reads the other order the run stops with `WRONG CAMERA ORDER` and the flags to use. An older sender cannot say which cameras it reads; then only the row-alignment line tells you: about 0.1 px is right, 1 px or more means the calibration does not fit (`WARNING: row alignment ... other order`).

What the 15 pairs showed (full numbers in `data/stereo_calib_new_report.json`):

- The board was found in both pictures of all 15 pairs, none dropped. Reprojection 0.53 px, row error 0.12 px, baseline 41.72 mm against the designed 42 mm (-0.7%; a 19.52 mm square would give exactly 42 mm, so the ruler value 19.4 is within 0.12 mm). Leave-one-out prediction of the held-out pair: 0.49 px; the plain k1 k2 lens model is as good as k1 k2 k3 and was kept because the k3 fit gives large values (0.5 and 1.3) in corners no board reached. The rational model diverges with 15 views.
- Weak points of the capture, not of the result: the paper board is not flat (its corners sit 0.35 to 0.63 px off a plane, which sets the 0.5 px floor of the reprojection error); the left picture is softer than the right in every pair (Laplacian variance 2 to 4 times lower: check the focus ring of the left camera); the board was always 39 to 70 cm away and the lower-left sixth of the picture holds no corner, so use a board glued to something rigid and add far and corner views next time.
- The earlier `data/stereo_calib.npz` was made for the upright order. Used on pairs in that order it gives 0.70 px row error; used with the sender in the other order, 2.8 px (worst 7.6 px), 75% valid depth instead of 89%, and board depth 6.8% off.
- Same 15 pairs, same LAS2-H: row error 0.70 -> 0.13 px, rows of the whole picture 0.36 -> 0.06 px, valid depth 89.4% -> 89.9%, board thickness 0.51 -> 0.43 mm, board depth error 0.91% -> 0.70%. On a still rig the dense depth was already clean; the new calibration mainly makes the rows exact.

## Check that the whole system uses the calibration

    py scripts/check_calibration_use.py

It runs the real live pipeline for 8 frames on a fake Pi (the saved pairs, sent the way the two cameras deliver them) and prints, for each stage, the numbers in the calibration file next to the numbers the stage was given: the rectifier, the picture size the depth model receives, the visual odometry (fx, cx, cy, baseline), the 3D map, the Rerun views (depth in metres), the Basalt calibration file (focal length, principal point, baseline, size), the camera-order check, and the row alignment of the rectified first pair (about 0.1 px when the file fits the pictures). It also lists every default in `gabai/` that still names `data/stereo_calib.npz`; there are none now: the pipeline and all the live tools (`noise_view`, `range_test`, `target_test`, `demo_viewer`, `record_basalt`, `live_basalt`, `sync_imu_camera`, `sync_sensors`, the benchmarks) default to `data/stereo_calib_new.npz`, and the self-check and diagnostics bundle read it when it exists. `calibrate_stereo` still writes `data/stereo_calib` by default; pass `--out data/stereo_calib_new` (or `--calibration` / `source.calib` on the tools) when you calibrate again.

> Newer: ChArUco calibration tool (`gabai/tools/charuco_calibrate.py`) and the current calibration state are described in `docs/rtabmap-lab-and-mapping-update.md`.

# Monocular metric depth fused with LAS2 stereo

## Idea
Stereo gives real distances from geometry but is noisy at range and fails on blank or shiny areas. A monocular metric model gives clean, sharp structure but its overall scale is only roughly right. `gabai/fuse_depth.py` puts the monocular map on the stereo scale (a robust scale and offset, plus a slowly varying correction), then blends the two per pixel by how noisy each is, and fills areas where stereo failed (flagged "mono only").

## What was measured (and the honest result)
Checkerboard pairs, 21 pairs at 0.42 to 0.85 m, LAS2-M stereo with Depth Anything V2 Metric Indoor Small (the only monocular model installed so far):
| | median error | 90th pct | noise (std) | roughness |
|---|---|---|---|---|
| stereo | 0.44% | 1.25% | 0.70% | 0.008 px |
| fused | 0.43% | 1.27% | 0.71% | 0.009 px |
| mono aligned | 0.63% | 1.74% | 1.02% | 0.027 px |
Static noisy fake rig (noise added to the pictures): stereo flicker 0.11 px, mono 0.20 px, fused 0.13 px.
So in everything I could measure, LAS2 depth is already smooth (almost no spatial roughness) and steady frame to frame, and the monocular model flickers MORE than stereo. Fusion did not improve these numbers. I have not seen the noise you describe. It may be flicker or error at distance on your live stream, which these tests do not cover, so the next step is to measure it on your rig.

Physical limit worth knowing: with a 5.1 cm baseline and 881 px focal length, a 0.5 px disparity error costs about 1 cm at 1 m, 4.5 cm at 2 m, 10 cm at 3 m and 28 cm at 5 m. A monocular model cannot remove that at range (it is typically 5 to 10% off in metres); it can smooth and fill.

## Measure your noise (needs the Pi streaming; rig held still, 10 s)
Point the rig at a room with a flat wall or the floor in the middle of the picture.
```powershell
py -m gabai.tools.noise_test --mono dav2
```
It prints temporal noise (flicker), frame-to-frame change, bumpiness of a flat patch (plane residual), usable pixel share, and the depth jitter in cm at 1, 2 and 3 m, for raw stereo, aligned monocular and fused. Send me the table.

## See it live
```powershell
py -m gabai.spike.demo_viewer --source live --model m --fuse dav2
```
The panel adds a line `fused with dav2: mono NN ms, scale a=.. b=.. px, NN% of pixels filled by mono only`. Pixels filled by mono only come from a model, not from geometry, so treat them as lower confidence.

## Better monocular model (pending your approval)
MoGe-2 (Microsoft, MIT licence) predicts metric scale and keeps fine detail. Variants and download sizes: ViT-S 35 M parameters 141 MB, ViT-B 104 M parameters 419 MB, ViT-L 326 M parameters 1.3 GB (about 60 ms on an RTX 3090 in FP16 per its README). Needs `pip install git+https://github.com/microsoft/MoGe.git` plus the weights from Hugging Face. Not installed yet. After install: `--mono moge-s` or `--fuse moge-s`.

## About SLAM
No SLAM exists yet. The project has per-frame depth, a calibrated stereo rig, a synced IMU and compass, but no camera tracking between frames (visual-inertial odometry) and no map. The plan left that out as too risky. Cleaner depth helps a later map, but the first thing a map needs is pose tracking, which is a separate piece of work.

## Live noise video (new)
```powershell
py -m gabai.tools.noise_view --mono dav2
```
Four pictures: left camera, raw stereo depth, fused depth, and a NOISE MAP (per pixel, how much its disparity jumped over the last 30 frames: blue steady, red jumpy, black often untrusted). Below them a rolling table of flicker in pixels and its cost in cm at 1, 2 and 3 m for raw stereo, the aligned monocular map and the fused map. A banner says whether the picture was STEADY (numbers are sensor noise) or MOVING (numbers include your motion).
Keys: `n` switches the noise map between stereo, mono and fused; `r` resets the window; `s` saves a screenshot in `data/`; `q` quits. `--no-lr` skips the left-right check for about twice the speed (4 to 8 FPS on this PC); `--out noise.mp4` records what you see.
Checked with a simulated noisy static rig only, at about 4.6 FPS with the left-right check.

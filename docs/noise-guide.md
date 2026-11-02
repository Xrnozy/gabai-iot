# Depth noise: what your screenshot showed and what to do

## What your live noise view showed
A dim, cluttered desk with a bright monitor, rig still ("STEADY"):
| | flicker | cost in depth at 1 / 2 / 3 m |
|---|---|---|
| raw stereo | 1.01 px (frame to frame 1.37 px) | 2.3 / 9.1 / 20.4 cm |
| monocular (aligned) | 5.5 px | 12 / 49 / 111 cm |
Most of the stereo depth was black (rejected by the left-right check) and the noise map showed jumpy pixels only on the few trusted areas. In my simulation, with sensor noise added to a well-lit scene, the same stereo flicker was 0.1 to 0.2 px. So your noise is about 5 to 10 times worse than that, and the likely cause is the picture itself: a dark scene with a bright monitor makes the camera pick a short exposure with no gain, and the images come out noisy and low-contrast.

## What I found about the monocular model
On a still, noisy sequence the monocular depth (Depth Anything V2 Metric Small) flickered MORE than stereo, and blending it in made the result worse, not better:
| strategy (noisy still scene) | flicker |
|---|---|
| raw stereo, trusted pixels | 0.14 to 0.20 px |
| fused with monocular, per-frame scale fit | 0.36 px (worse) |
| fused, scale fit smoothed over time | 0.40 px (worse) |
| fused, monocular map averaged over time first | 0.21 to 0.23 px (no better) |
| **stereo averaged over time (new)** | **0.05 px** |
So monocular fusion is not the cure for this noise. What works is averaging the stereo itself over time on pixels that are steady, while letting real changes through at once. It is in the viewer now, on by default (`--no-smooth` turns it off, `--smooth-alpha 0.3` sets how fast it follows). On the recorded sequence it cut flicker from 0.14 to 0.05 px, and 0.105 to 0.039 px in the live view test. It resets while you turn fast (60 deg/s or more), so a turning rig is not smeared. It cannot help much with the other noise source, which is wrong depth on blank or dark areas; those pixels are rejected and left unknown.

## Fix the input too (the biggest lever)
1. Light the scene. A lamp pointed at the wall or floor in front of the rig is the cheapest improvement. Avoid a bright monitor or window in the picture, because auto exposure then darkens everything else.
2. Raise the camera gain and brightness (they were 0). On the Pi, `v4l2-ctl -d /dev/video2 --list-ctrls` shows the allowed ranges; then start the sender with, for example, `--gain 40 --brightness 20` (stay inside the range printed). Gain also amplifies noise, so light is better than gain.
3. In dim light you can lengthen the exposure by lowering the capture rate: `--capture-fps 15 --exposure 600` (60 ms; this smears a moving scene: see the section on noise while moving in rerun-pipeline.md). This costs sync accuracy (pairing offset up to about 33 ms instead of 17 ms), so it is a trade-off.
4. In the noise view, press `g` (brighten), `c` (contrast boost) and `t` (average 3 frames, still rig only) and watch the "raw stereo" flicker number change after the window refills. Press `w` to save the last 30 picture pairs to `data/noise_rec_<time>.npz` so I can study your real noisy frames offline.

## Commands
```powershell
py -m gabai.tools.noise_view --mono dav2        # live noise video, new row "stereo + time avg"
py -m gabai.spike.demo_viewer --source live --model m          # smoothing is on by default
py -m gabai.spike.demo_viewer --source live --model m --no-smooth   # to compare
```
Pi, with camera gain (replace the numbers with values from your camera's range):
```bash
python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0 --gain 40 --brightness 20
```

## Not verified
All filter numbers are from a simulated noisy still scene built from your calibration pairs, not from your dark desk scene. I have not seen how the filter or the input fixes behave on your real noisy frames; `w` in the noise view gives me exactly that data. MoGe-2 (a better monocular model) is not installed; given how unstable the monocular map was, I would not expect it to beat the temporal average for noise, though it might help fill holes.

## Update: your second screenshot, the "two small depth models" idea, and speed
Your second screenshot was taken while the rig was MOVING (the banner said so: the picture changed by 18.5 grey levels), so its flicker numbers describe your motion, not the sensor noise. Only 32% of the stereo pixels were trusted. Hold the rig still on a table to measure noise.

**Idea: run a small depth model on both cameras and use it to fill the stereo.** Tested on a still, noisy sequence (the monocular map is aligned to stereo separately for each camera, the right one is warped into the left view, and the two are averaged where they agree):
| | flicker of the monocular map |
|---|---|
| left camera only | 1.38 px |
| both cameras combined | 1.19 px (only 14% better; ideal would be 29%) |
| for comparison: raw stereo | 0.14 px, stereo averaged over time 0.05 px |
The two views agreed on only 49% of pixels, so half the pixels gave no averaging at all. Costs 14 ms extra. The code exists and is tested (`warp_right_to_left`, `combine_mono_pair` in `gabai/fuse_depth.py`) but I did not wire it in, because the result is still about ten times noisier than stereo. Monocular depth is good for filling holes and for sharp structure; it is not good for lowering noise. Keep in mind that filled pixels are model guesses, not measurements.

**Speed (headless live viewer, 640x480, this PC, same sender; before to after):**
| change | effect |
|---|---|
| left-right check as ONE batched model call instead of two | 79 to 48 ms (LAS2-M), 61 to 43 ms (S), 123 to 74 ms (H); same result (99.8% of the trust mask identical) |
| monocular pre-processing on the GPU instead of the CPU | 21 to 14 ms |
| fusion noise estimates at half resolution, faster scale fit | fuse 47 to 16 ms |
| mono in a second thread overlapping stereo | only 5 ms: the GPU is the limit, not used |
| viewer frame rates now | stereo + temporal filter: 14.2 fps (the simulated Pi sends 15, so this is the cap); with left-right check: 12.8 to 14.1 fps; fused with the monocular map: 9.1 fps (LAS2-M), 10.1 fps (LAS2-S) |
The noise view itself (which also computes the statistics and four picture panels) is slower than the viewer, about 4 to 6 fps; use `--no-lr` there for more.

**Recommendation:** for less noise, light the scene, keep the temporal filter on (default), and skip the monocular fusion unless you want holes filled. Fastest good setting: `py -m gabai.spike.demo_viewer --source live --model m` (14 fps). To see filled holes: add `--fuse dav2` (about 9 fps with M, 10 with S).

## Update: temporal smoothing is now OFF by default
On the hand-held rig the temporal average ghosted and made the depth wiggle; `--no-smooth` looked better in your three screenshots. The viewer now runs without it. To turn it back on for a still rig, add `--smooth`. The filter was tightened (gate 1.5%, motion detection, no held pixels while moving) but I have not seen it behave well on your live video, so it stays opt-in until you test it.

## Update: monocular "referee" and the close-range pass
**`--mono-check dav2` (new, off by default):** the monocular model is put on the stereo scale and used only to REJECT stereo pixels that disagree with it by more than max(6 px, 25% of the disparity). Surviving pixels keep their pure stereo value, so the monocular map's own flicker never reaches the output (that was what made blending worse). It targets gross errors (repeating bars, a wrong near object), not small wiggle. Tested on a synthetic mismatch only: 100% of a +40 px bad patch vetoed, under 2% of good pixels. NOT yet seen on your real frames.
Measured loop time on the fake rig, headless, LAS2-H with `--lr-check`: 80 ms (12.5 fps) without it, 103 ms (9.7 fps) with it; LAS2-M: 75 ms (13.3 fps) with it.

**`--near` (new, off by default):** LAS2 cannot match a disparity above 192 px, which is any object closer than 0.23 m. On a test pair with a 220 px shift the full-size pass returned 144 px and trusted 69% of it (wrong). `--near` adds a half-resolution pass that reaches 0.12 m (it returned 220.1 px). It costs about 55 ms with H (loop 80 to 140 ms, 7 fps) because H's time does not shrink with image size, so it is opt-in.

**Not done:** the PC latency (about 70 ms) is Wi-Fi plus the model plus queuing; I did not change it.

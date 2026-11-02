# Architecture

What exists today and how data moves. Everything here is implemented unless it says "not implemented". Status of each part (verified in software, verified on hardware, not done) is in section 9.

## 1. The two machines

```
 Raspberry Pi 3B                                         PC (Windows, RTX 5060 Ti)
 ---------------                                         -------------------------
 2 x EMEET C100 (v4l2, MJPEG)                            gabai.pc.receiver     TCP server, decodes messages
 BMI160 gyro+accel (I2C)        --- TCP, one socket --->   |
 HMC5883L compass (I2C)         (JPEG pairs + IMU +        +-- gabai.spike.demo_viewer   rectify -> LAS2 depth -> zones -> alerts
 gabai.pi.sender                 compass + status)         |        +-- gabai.diag.*       traces, checks, logs, artifacts
                                                           |        +-- gabai.pc.telemetry web page + CSV
 logs: ~/gabai_logs/pi_<time>/                             +-- gabai.tools.live_basalt / record_basalt   SLAM (Basalt) from the same stream
                                                           logs: logs/<run>/gabai.jsonl
```

The Pi only captures and sends. All depth, SLAM and decisions run on the PC. The speaker is the PC's.

## 2. Data flow of one frame (live viewer)

1. **Pi capture** (`gabai/pi/cams.py`). Two reader threads grab the left and right camera. A pair is formed when the two capture times are close. The frame id `fid` is the Pi monotonic capture time in whole milliseconds (`frame_id(t)` in `gabai/diag/trace.py`). The same number is used on the Pi and on the PC, so a Pi log line and a PC log line for the same picture can be matched.
2. **Pi send** (`gabai/pi/sender.py`). JPEG pair, IMU samples (gyro deg/s, accel g), compass, and a status message (CPU temperature, throttle flags) go over one TCP socket, prefixed by the shared token. The sender reconnects with back-off.
3. **PC receive** (`gabai/pc/receiver.py`). Decodes, keeps only the newest pair (older ones that were never read are counted as `overwritten`), counts decode errors, gaps and inter-arrival time, logs `link.*` events.
4. **Orientation** (`gabai/fusion.py`, `gabai/orient.py`). Gyro + gravity (and compass if enabled) give a camera orientation. The camera-IMU rotation and delay come from `data/imu_sync.json`.
5. **Rectify** (`gabai/tools/calibrate_stereo.py` Rectifier). Uses `data/stereo_calib.npz`. Output is a pair where a point has the same row in both images.
6. **Depth** (`gabai/spike/las2_runner.py`). LAS2 (size s/m/l/h) gives a disparity map in pixels. Left-right consistency marks pixels that disagree as untrusted (NaN). Options: `--near` (second half-resolution pass for close range), `--mono-check` (monocular veto), `--fuse` (monocular metric fusion).
7. **Zones** (`gabai/zones.py`). Disparity to depth (`Z = f*B/d`), floor removal with a floor model (camera height, IMU up-vector), three bands (left/ahead/right), result is an alert key and a distance rounded to 0.5 m, or `drop_ahead`, or `cant_see_ahead` when too little depth is trusted. Unknown is never treated as free.
8. **Alerts** (`gabai/alerts.py`, `gabai/live_alerts.py`, `gabai/speaker.py`). Debounce, repeat rules and priority, then one of 23 pre-rendered WAV clips is played. Every non-spoken decision has a reason (`AlertController.decision`).
9. **Diagnostics** (`gabai/diag/session.py` `PerceptionDiag`). Builds a `FrameTrace` for the frame: stages with timings, quantities tagged by kind, checks, decisions. Written to the log as `frame.trace` (sampled, plus always for interesting frames).
10. **Telemetry** (`gabai/pc/telemetry.py`). Web page on port 8000 and a per-frame CSV in `data/telemetry_log.csv`.

## 3. Optional parts that are wired in as advice only

- `--grid`: a rolling occupancy grid (`gabai/grid.py`) built from the same depth. It is a cross-check: if the zone alert and the grid disagree, a `check.suspect` is logged. The grid does not change what is spoken.
- `--governor`: thermal/lane governor (`gabai/governor.py`). When the GPU is hot it switches off the optional extras `--near` and `--mono-check`. It never switches off the zone check or the alerts.
- Detection (`gabai/detect.py`): adapter and depth-for-detection code exist and are unit-tested, but no detector weights are installed, so nothing runs it. Not wired into the viewer.

## 4. SLAM path (Basalt)

`gabai/tools/record_basalt.py` records a EuRoC-format sequence from the live stream. `gabai/tools/live_basalt.py` runs Basalt (through the VIT C API, `gabai/slam/vit.py`) live or on a recording (`--replay`). A still-gate (`gabai/slam/still.py`) starts Basalt only when the rig is held still, so the gyro bias is learned. Accelerometer calibration (`gabai/slam/accel_cal.py`, `data/accel_calib.json`) is applied to the IMU stream before it reaches Basalt. `gabai/slam/diag.py` summarizes a trajectory and lists findings. SLAM does not feed the obstacle alerts today.

## 5. Coordinate systems and units

| Frame | Axes | Used by |
|---|---|---|
| Image | u right, v down, pixels, origin top-left of the rectified image | zones, depth |
| Camera | x right, y down, z forward (optical axis), metres | zones, sim, `explain_point` |
| World (sim, `gabai.sim`) | x forward, y left, z up, metres, floor at z = 0 | simulator, replay scenarios |
| IMU (BMI160) | axes as mounted; mapped to camera by the rotation in `imu_sync.json` | fusion, SLAM |

- Disparity: pixels. Depth: `Z = f*B/d`, metres along the optical axis, not along the ground. Current values: f = 881 px, B = 5.07 cm, cx = 326.6, cy = 222.9 (from `data/stereo_calib.npz`; check yours with `py -m gabai.tools.selfcheck`).
- Gyro: deg/s on the wire, rad/s in the EuRoC files. Accel: g on the wire, m/s^2 in EuRoC files.
- Up-vector: the world up direction in camera coordinates, from the IMU. `floor_disparity` uses it to model a tilted floor.
- The distance the alert speaks is the **optical depth**. On a camera tilted down by an angle, the distance to walk (horizontal range) is shorter. The evidence in the log records both (`p5_depth_m`, `p5_horizontal_range_m`). The spoken value has not been changed. In the replay scenario with a 35 degree tilt the difference is about 0.15 m.

## 6. Model input and output

- LAS2 input: rectified left and right RGB, resized so both sides are multiples of the model stride. Output: disparity in pixels at the input scale, maximum 192 px (minimum range about 0.23 m at f = 881, B = 5.07 cm). Pixels at the cap are suspect (`check_disparity_cap`).
- The model output is a **model prediction**, not a measurement. The code marks it that way (`trace.MODEL`). A pixel becomes `validated` only through a named independent check; today that is left-right consistency, which shows the two views agree, not that the metres are right.
- Metric accuracy of the whole chain has been measured only in `docs/accuracy-guide.md` style runs on the bench. It has not been established across conditions.

## 7. Timestamps, sync, latency, stale data

- Pi time and PC time are different clocks. The code never subtracts them. The frame id is the Pi capture time, and the PC measures age on its own clock from the moment the message arrives.
- Pair sync: left/right capture time difference is logged (`check_pair_sync` compares the motion implied by the gyro rate to a pixel limit).
- Camera-IMU delay is measured by `sync_imu_camera` and stored in `imu_sync.json`. The recorder removes it from camera stamps, so EuRoC stamps are already aligned.
- Stale: a frame older than `STALE_FRAME_S` (checks.py) is reported as `stale` and the alert path says why. Latency budget for the loop is 400 ms (`check_latency`).
- Dropped frames: the receiver keeps only the newest pair; the number of skipped pairs is `overwritten` in `link.stats`.

## 8. Logging and diagnostics

See `docs/debugging.md`. Short version: JSON lines, one per event, in `logs/<run>/gabai.jsonl`; asynchronous, bounded, rotated, old runs pruned; no raw footage unless you ask.

## 9. Status

| Part | Verified in software | Verified on the real rig | Not done |
|---|---|---|---|
| Pi sender, cams, IMU, compass | unit tests with fake Pi | earlier sessions (live link works) | new logging code not run on a real Pi yet |
| Receiver, viewer, zones, alerts | unit + integration tests, fake Pi, replay on a real recording | live runs in earlier sessions | diagnostics layer not run on live hardware |
| Diagnostics, self-check, collect_diag, report | tests (see `docs/testing.md`) | no | |
| Basalt live / recording | tests of wrappers, real recording replay | recording and live runs earlier in the session | long-run drift not measured |
| Grid, governor | unit tests, simulator | no | not allowed to change alerts |
| Detection (YOLO) | adapter tests with a fake model | no | no weights; not wired |
| Rerun pipeline, stereo-only and stereo+IMU modes (`docs/rerun-pipeline.md`) | tests incl. a two-mode run on a real recording | no | recordings are not row-aligned, so accuracy is unmeasured; not a live source |
| Phases 3-6 of `spec/Gabai Lite.md` | modules and tests, a real-video preview run | no | not wired into the live viewer; SceneScript, ShapeR, nvblox, YOLOE not available. See `docs/phases-3-6.md` |

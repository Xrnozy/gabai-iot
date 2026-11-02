# Hardware test checklist

For when the rig is back. Nothing in this file has been run on the new logging and diagnostics code with real hardware; the steps below are what to do and what a pass or a failure looks like. IP addresses are the ones used in earlier sessions (PC 192.168.1.64, Pi 192.168.1.63, Pi user `xrnozy`); change them if yours differ.

Work in order. Each step isolates one component, so a failure tells you where to look. Stop at the first failure and send the package (section 12).

## 0. Hardware setup

- Pi 3B with the two EMEET C100 on separate USB ports, BMI160 on I2C bus 1, HMC5883L on the software bus 3 (`i2c-gpio` overlay), both cameras rigid on the same bar (baseline about 5 cm, calibration assumes it does not change).
- Power: a supply rated 2.5 A or more with a short cable. A power bank gave under-voltage (`docs/pi-power-guide.md`); results taken while `pi.stats` reports under-voltage or throttling are suspect.
- PC and Pi on the same network (cable is best). PC allows Python through the firewall for TCP 5600.
- Pi: `sudo apt install python3-opencv v4l-utils`, then copy the code: `scp -r gabai xrnozy@192.168.1.63:~/` (PowerShell in the project folder).
- Printed checkerboard 9x6 inner corners, square size measured with a ruler. A tape measure. A textured scene (shelves). A box about 40 cm wide and tall.

## 1. PC software check (no hardware)

```bash
py scripts/run_tests.py
py -m gabai.tools.selfcheck --deep
```
Expect: all modules pass (skips listed); self-check has no FAIL. WARN lines say what to fix. Failure sign: any FAIL. Send back: the selfcheck output.

## 2. Pi alone: sensors and cameras

On the Pi:
```bash
i2cdetect -y 1          # BMI160 at 0x68 or 0x69
i2cdetect -y 3          # HMC5883L at 0x1e
v4l2-ctl --list-devices
```
Expect: the addresses show; two C100 devices listed. Failure sign: nothing at the address (wiring), one camera missing (USB).

## 3. Link without cameras

PC: `py -m gabai.pc.receiver --port 5600`   Pi: `python3 -m gabai.pi.sender --host 192.168.1.64 --test-pattern --log-level DEBUG`
Expect: PC prints frames at about 15 per second; `link.up` in `logs/<run>/gabai.jsonl`; no `link.lost`. Pi log in `~/gabai_logs/pi_<time>/`. Failure sign: `link.rejected` (token), no `link.accept` (network/firewall). Isolates: network and protocol from cameras.

## 4. Link with cameras and sensors

Pi: `python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0`
PC: `py -m gabai.spike.demo_viewer --source live --model s --no-audio`
Expect (first 30 s): `cam.first_frame` for both cameras, no `cam.mode_mismatch`, `pi.stats` with CPU temperature and no throttle flag, `link.stats` rate near the set FPS, an IMU rate near 100-200 Hz with no gaps. Run `py -m gabai.tools.diag_report`. Failure signs: `cam.mode_mismatch`, throttle flags, IMU gaps, left/right order swapped (`docs/calibration-guide.md`).

## 5. Stereo calibration (one time, or when the cameras moved)

```bash
py -m gabai.tools.calibrate_stereo --square-mm <measured>
```
Follow `docs/calibration-guide.md`. Expect: reprojection and row error both under 0.5 px (GOOD) or under 1 px (OK), as in the guide. Then `py -m gabai.tools.selfcheck`: calibration lines PASS. Send back: console output and `data/stereo_calib.json`.

## 6. IMU and compass sync

```bash
py -m gabai.tools.sync_imu_camera        # or sync_sensors with compass, see docs/sync-all-guide.md
py -m gabai.tools.calibrate_accel        # 8 orientations
```
Expect: delay stable and match score GOOD; accel calibration saved. Check gravity with `check_gravity` in the report: still accel magnitude close to 1 g. Failure sign: BAD match, delay different on repeats (check cables, run again).

## 7. Depth quality, component isolated (no SLAM, no alerts)

Tape-measured targets at 0.5, 1, 1.5, 2, 3 m, lit and dim:
```bash
py -m gabai.tools.target_test --label box --true-m 1.0 --light lit --model m
py -m gabai.tools.range_test --square-mm <measured> --model m
py -m gabai.tools.make_report --targets data/target_tests.csv --range data/range_test.csv   # summary of the saved tests
```
Expect: the error numbers in the report. There is no pass threshold in the code for these; compare to the table in `docs/accuracy-guide.md` and `docs/calibration-guide.md`. Failure signs: error grows quadratically with distance (normal for stereo), constant offset (calibration or baseline), no depth on a blank wall (known limit). Send back: the CSVs in `data/` and the report.

## 8. Obstacle alerts, indoors

```bash
py -m gabai.spike.demo_viewer --source live --model m --cam-h <measured camera height in m> --log-level DEBUG
```
Walk toward the box from 4 m. Press `f` when it speaks about empty floor, `x` when it stays silent for a real obstacle (each saves the evidence). Expect: a spoken distance that shrinks; on a tilted camera it will read a little long (optical depth, see `docs/architecture.md` 5). Check in the report: floor model says `tilted by IMU up-vector`, not `level camera`. Failure signs: floor false alarms (wrong `--cam-h`), `cant_see_ahead` in good light, alerts later than 2 m. Measure the real distance at the first alert with a tape and write it down with the log time.

## 9. Latency

In the same run read the report: loop time, link age, `check_latency` (budget 400 ms). To split it: Pi `pi.stats`, `link.stats` inter-arrival, trace stage timings (`receive`, `rectify`, `depth`, `zones`). Add `--no-audio` to exclude speech. Failure signs: depth stage over 100 ms on `--model m` (try `s`), receive stage large (network), queue growing (`check_queue`).

## 10. SLAM

```bash
py -m gabai.tools.record_basalt --seconds 60        # lights on, 10 s still, then a slow walk loop
py -m gabai.tools.live_basalt --replay data/slam/rec_<time>
py -m gabai.tools.live_basalt                         # live from the Pi
```
Expect: `slam.gate_open`, `slam.tracker_ready`, `slam.summary` findings with no `confirmed` failures. Drift test: end the loop at the start point; the summary reports the end-to-start gap. Failure signs: `slam.accel_calib_missing`, pose steps, still drift above what `docs/slam-plan.md` states. Isolate IMU vs camera: `py -m gabai.tools.check_extrinsics` and the replay with the recorded folder. Send back: the whole `data/slam/rec_<time>` except pictures (the package leaves pictures out).

## 11. Mapping / grid cross-check

Add `--grid` to the viewer in step 8. Expect: no `check.suspect` for grid disagreement while walking toward the box. The grid is advisory; the zone alert is what speaks. Failure sign: repeated disagreement, which means one of them is wrong; the evidence saved with `--diag-save grid,depth` shows which.

## 12. Procedures when something is poor

| Problem | Do |
|---|---|
| Poor accuracy | Step 7 at known distances; if error is a constant factor suspect baseline/focal in calibration (redo step 5); if it varies with scene, texture or light. Send the CSVs and the package. |
| Depth errors in one place | Reproduce with `--diag-save depth,frame`; open the saved pictures; the trace gives pixel, disparity, depth and checks for the frame. |
| Tracking drift | Step 10 replay of the recording; check still-gate, accel calibration, extrinsics in order; compare drift still vs walking. |
| Mapping problems | Step 11; confirm orientation source in the trace (`IMU orientation` vs none). |
| Latency | Step 9 split by stage. |
| Anything unexplained | `py -m gabai.tools.collect_diag`, send the zip. |

## Package to send back

```bash
py -m gabai.tools.collect_diag
```
Zip in `data/diag/`. Add: one sentence on what looked wrong and when, the tape-measured truth where you have it, the Pi's `~/gabai_logs` folder, and which power supply was used. Pictures are not included unless you add `--with-images` (they are camera footage).

## What this checklist cannot promise

It does not establish accuracy by itself. Metric error is whatever step 7 measures on your scenes. A pass in steps 1-3 says nothing about steps 4-11.

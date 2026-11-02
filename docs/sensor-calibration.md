# Guided sensor calibration: HMC5883L compass, then BMI160 IMU (2026-10-29)

`py -m gabai.tools.calibrate_sensors` walks you through calibrating the two sensors one after the other, with live graphs, a 3D compass-sample plot, instructions and
progress in Rerun. You do the physical movements; the script reads the real sensors through the existing Pi link.

## Install and run
Nothing new to install: it uses packages the project already needs (`numpy`, `scipy`, `rerun-sdk`). Check with `py -c "import rerun, scipy, numpy"`.

1. Copy the current code to the Pi (PowerShell, project folder): `scp -r gabai xrnozy@<PI_IP>:~/`
2. Pi, device resting FACE-DOWN on the table (camera side down) and still: `python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video2 --right /dev/video0`
   It must print `IMU found` and `compass found`. Keep the device still for its first 3 s (it measures the gyro bias then).
3. PC: `py -m gabai.tools.calibrate_sensors`. Rerun opens. Instructions are in the **What to do now** panel (and printed in the terminal); press Enter in the terminal to start each
   step and answer `y`/`n` when asked to save.

Options: `--phase mag` or `--phase imu`, `--redo-mag`, `--redo-imu`, `--no-viewer --save-rrd run.rrd` (record instead of opening a window), `--log-level DEBUG`, `--data-dir`.

## What happens
**Phase 1, HMC5883L only.** (1) face-down rest, still 4 s, which checks that gravity is on the front/back axis (IMU X); (2) lift the device and draw big slow *figure-eights* in
every orientation, like painting the inside of a ball, including upside down and on each side, until the coverage reaches 90%. The step ends only when **all** its requirements are
met, and no timer ends it: 90% coverage of 26 directions, at least 400 compass readings, at least 6 s, and a compass that really changes (a device lying still only jitters by a few
counts, which must not count as coverage). The Progress panel shows `STILL NEEDED: ...` (missing directions, readings left). Ctrl+C is the only way out of the step; it is not kept
when stopped half way, because the one step is the whole capture.

**Phase 2, hold detection.** A hold counts when the gyro stays below 5 deg/s and the accelerometer is steadier than 0.15 m/s2 (std) for 3 s. These limits are looser than the library
defaults (3 deg/s, 0.08) because a hand trembles; they were not measured on your hands. The panel shows the live gyro and shaking numbers next to the limits, so you can see why a hold
is not counted. The gyro direction check turns the device clockwise about the camera axis, taken from where gravity was face-down.

**What Rerun shows.** Top: instructions, progress and still-needed list, and a small panel that says **● RECORDING** (a calibration capture is running, samples are kept) or
**○ LIVE VIEW ONLY** (the sensors are drawn but nothing is stored; this is the state while the script waits for your Enter, and after each step). Middle: the compass cloud (raw counts;
grey points = live view, colored = recorded) and the **device model**: the BMI160 board (red x, green y, blue z) and the device it sits in (grey box with arrows right / down / forward,
drawn with the saved IMU-to-device axes, or the reference mapping when none is saved yet, and the box label says which). It turns with the gyro + gravity orientation of the BMI160; the
compass is not used for it, so only tilt is reliable and heading can drift slowly. Below: compass X/Y/Z, gyro and accelerometer curves, and the progress bars (`recording` is 1 while a
capture runs).
Then it computes the hard-iron offset, the soft-iron matrix and the compass-to-IMU axis map, runs the checks (table below), and shows them. **You review and answer `y` to save.**

**Phase 2, BMI160 only.** It starts only after phase 1 is saved *and* you answer `y` to "Continue with the BMI160 IMU calibration now?". (1) face-down rest, still 10 s: gyro noise and
residual, accelerometer at rest, gravity direction; (2) hold it still, 3 s each, on all six sides (captured automatically; the panel lists the faces still missing); (3) upright as worn
on the chest, still 4 s; (4) a clockwise turn seen from the camera side, which checks the gyro's direction against the axis mapping. Then the checks and your `y` to save.

Stop with Ctrl+C at any time. Saved results stay. An unfinished capture is kept (`data/calibration/*_partial.npz`) and offered the next time. Running again skips a phase whose file is
valid (`--redo-mag` / `--redo-imu` repeat it).

## Checks and what they mean (a FAIL is never offered for saving)
| check | pass when | if not |
|---|---|---|
| coverage of directions | 80% or more of 26 directions have readings | turn through the missing orientations named in the panel |
| sphere quality | corrected field strength varies under 3% (fail over 6%) | more rotation; something magnetic moving relative to the sensor |
| field strength | 20 to 70 uT after correction | strong soft iron, a magnet, current close by |
| hard iron | offset under twice the Earth's field | a magnet or iron next to the sensor (your raw 89 uT) |
| abnormal readings | under 2% of readings 30% off | spikes: a phone or a moving cable |
| angle to gravity constant | spread under 6 degrees | wrong axis map or offset |
| compass-to-IMU axis map | clearly better than the next candidate | not enough orientations |
| accelerometer bias and scale | all six faces held and a well-conditioned fit | "NOT calibrated" and the reason; nothing is saved for it |
| gyro | noise under 0.4 deg/s, residual under 1 deg/s | the sender was started while the device moved: restart it |

## Axes: reference mapping, and what is verified
The mounting, for the device face-down: IMU X front/back (the camera axis), Y left/right, Z up/down; compass X left/right, Y front/back, Z up/down. That is a statement of which physical
direction each sensor axis carries, so it fixes which axes pair up, not their signs. The compass map measured on your unit (compass X<->IMU Y, Y<->IMU X, Z<->IMU Z) agrees with it. Paired by physical direction:

| | compass | IMU | device (camera) |
|---|---|---|---|
| left/right | X | Y | x (right) |
| up/down | Z | Z | y (down) |
| front/back | Y | X | z (forward) |

Nothing is assumed beyond the permutation, and the permutation itself is checked: the script measures the compass-to-IMU matrix (all 24 axis-aligned rotations are tried; the one that keeps
the field at a constant angle to gravity over your movements wins, with a margin to the runner-up) and the IMU-to-device matrix (two gravity directions: face-down and upright). Signs come from
those measurements. If the measured permutation differs from the table above, the check says so, and the measured matrix, not the table, is what is saved.

## Coordinate frames and the chest mount
Three frames are kept apart (`gabai/frames.py`): **raw sensor** axes (compass X/Y/Z, IMU X/Y/Z as wired; the files store parameters for these), **device** axes (the left camera: x right, y down, z
forward; the files store the matrices `device_from_imu` and `device_from_mag` separately) and the **SLAM/world** frame (z up). Nothing is converted when it is saved.
Worn on the chest the device is upright with the camera forward, so the device axes are the wearer's right, down and forward: **no extra axis transformation is needed, only the tilt**: a chest
slopes, so the camera usually looks 5 to 15 degrees below the horizon. `py -m gabai.tools.frame_check` (wear it, stand still) prints that pitch; put it in `[tracking] initial_pitch_deg` so the
SLAM world z is truly up. The orientation filter handles every other pose by itself from gravity and the compass.

**Applied once.** The saved numbers refer to raw readings (`applies_to: raw`; a file that says otherwise is rejected). The running system applies them in exactly one place: `gabai.mag.apply_cal`
for the compass (`corrected = W @ (raw - offset)`, loaded from `data/mag_cal.json` by `load_mag_cfg`) and the accelerometer model in `gabai.fusion` (from `data/accel_calib.json`). The calibration
module itself never corrects a reading. **Gyro bias is deliberately not saved for reuse**: the Pi measures it at every start and subtracts it before sending, so a stored bias would be removed
twice. What is stored is the gyro noise and the residual left after the Pi's correction, as a health check; `bias_policy` in the file must be `not_applied`.

## Files
`data/calibration/hmc5883l.json`, `data/calibration/bmi160.json`: schema `gabai.calibration/1`, sensor identification (kind, bus, address, units), the parameters, axis mappings with the reference
and the measured matrices, transformation matrices, validation results and timestamps. Written atomically; the previous file is kept as `.bak`. Checked on every load; an invalid file is reported and
ignored (never half-used). For the running system the same numbers are written to `data/mag_cal.json`, `data/mag_sync.json` (compass delay already there is kept), `data/accel_calib.json` (only if
the accelerometer fit was supported) and `data/device_frame.json`; `imu_3d`, `demo_viewer` and the fusion load them automatically.

## Expected output (compass phase, abridged)
```
=== HMC5883L compass, step 2/2: Figure-eights over the whole sphere ===
HMC5883L calibration result
PASS  coverage of directions: 92% of 26 directions covered
PASS  sphere quality: corrected field strength varies by 0.9% (good)
PASS  field strength: 44 uT after calibration (Earth: 25-65 uT)
PASS  angle to gravity stays constant: spread 1.4 deg (dip about +55 deg); consistent
PASS  compass-to-IMU axis map: measured [[0, -1, 0], [1, 0, 0], [0, 0, 1]] (score 2.1, next best 544.0)
Save this compass calibration? [y/N] y
Compass calibration SAVED at ...: data/calibration/hmc5883l.json, and for the running system data/mag_cal.json and mag_sync.json.
```
(That is the output on the simulated device used in the tests.)

## Troubleshooting
- **"no IMU or compass data for 8 s"**: the sender is not running, or it printed `IMU not used` / `compass not used` (check wiring and `i2cdetect -y 1` / `-y 3`), or the port or token differ (`--port`, `--token`).
- **Face-down check says gravity is not on axis X**: the device was not flat and face-down, or the sensor mounting differs from the reference; the measured mapping is what gets saved.
- **Coverage stuck below 80%**: the panel names the missing directions in the compass's own axes (+X, -Y+Z, ...); turn the device so those sensor axes point along the Earth's field. Slow, large movements, including upside down.
- **Quality fails or abnormal readings**: keep phones, laptops, magnets and steel away; keep the Pi, battery and cables exactly as they are worn; do not move cables while turning.
- **Field far above 65 uT even after correction, or a hard-iron offset over 100 uT**: a magnet or iron next to the compass (the Pi, USB cables, a speaker); move the compass on a short pole 10 cm or more away.
- **Axis map "not clearly better"**: you turned about one axis only; use the figure-eights and every side.
- **Accelerometer "NOT calibrated"**: not all six faces were held still for 3 s each; that is deliberate, bias and scale are not guessed from fewer.
- **Gyro residual warning**: restart the Pi sender and keep the device still for its first 3 s.
- **No compass points in Rerun**: the grey cloud and the compass curves draw from the moment the script starts, even before Enter. Nothing at all, and after 8 s the Progress panel says "No IMU or compass data": the Pi is not sending (sender not running, `compass not used`). IMU curves but no compass: the sender could not start the compass (`i2cdetect -y 3` must show `1e`).
- **Step does not finish**: read `STILL NEEDED` in the Progress panel; it names what is missing (degrees about which axis, readings, directions).
- **Is it recording?**: the "Recording or live view?" panel, the `recording` curve (1 = capture running), and the terminal lines `[RECORDING] <capture>` / `[live view only] ...`. After a capture ends it prints how many samples it kept.
- **Debug logs**: `--log-level DEBUG`; events `cal.rec.start` / `cal.rec.stop` (capture, outcome, samples kept) / `cal.rec.discard` (samples thrown away between captures), `cal.live` (live view reads), `cal.mag.progress` (step, degrees, coverage, still_needed), `cal.mag.step_done`, `cal.mag.rest`, `cal.mag.result`, `cal.model.axes` (which IMU-to-device axes the 3D model uses), `cal.imu.rest`, `cal.imu.result`, `cal.no_data`, in `logs/calibrate_*/gabai.jsonl`.

## Limits
- Tested end to end on a simulated device with known truth (offset recovered to 0.1 count, all 24 compass axis maps recovered, accelerometer bias and scale recovered, device axes recovered, stop and resume,
  corrupt files, no-data, real Rerun front end). **Not run on your hardware**; the first real run is the first real test, and the expected-output block above is the simulation's.
- Soft iron is only as good as the directions you cover; the fit falls back to hard iron only (sphere) when they are not enough, and says so.
- The gyro direction check relies on you turning clockwise as seen from the camera side; a wrong-way turn shows as a WARN, not a FAIL.
- A sideways lean of the upright pose is invisible to gravity and goes into `device_from_imu` as an error of the same size; hold it left-right level. `load_frames` compares the result with `data/imu_sync.json`.
- Compass heading is relative to magnetic north and unreliable near steel and wiring.

# IMU and compass check, then fast-turn flagging

## Test 0: do the sensors answer (Pi)
Your earlier chat export shows the BMI160 already answering at address 0x68 on I2C bus 1. The compass (HMC5883L) is wired to pins 35/37 (GPIO19/GPIO26), which are not hardware I2C, so it needs a software I2C bus.

1. Copy the files (PC, project folder): `scp -r gabai xrnozy@192.168.1.63:~/`
2. Add the compass bus (Pi), then reboot:
```bash
echo "dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=26,i2c_gpio_scl=19" | sudo tee -a /boot/firmware/config.txt
sudo reboot
```
3. After reboot (Pi):
```bash
i2cdetect -y 1      # expect 68
i2cdetect -y 3      # expect 1e  (0d instead means a QMC5883L clone: tell me)
cd ~ && python3 -m gabai.spike.pi_imu_check
```
Expected: `BMI160 ... chip id=0xD1 -> OK`, accel magnitude about 1.0 g at rest, gyro raw near 0; `HMC5883L ... id=b'H43' -> OK` with non-zero field values (rotate the board and run again: they should change).
The compass is only checked here. Nothing reads it live yet.

## Fast-turn flagging (live)
The sender reads the gyro at 200 Hz and sends the recent peak turn speed with every frame pair. The viewer shows it with the sync error that turning causes.

Pi (keep the rig STILL for the first 1.5 s: it measures the gyro bias):
```bash
cd ~
python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0
```
It prints `IMU found ...`, then `gyro bias [...] deg/s, movement during calibration X`. X should be under about 1.5; more means the rig moved, so restart it. If it prints `IMU not used (...)`, the stream still works without turn flagging; send me the message.
PC:
```powershell
py -m gabai.spike.demo_viewer --source live --model s
```
New panel line:
`TURN: 95 deg/s (FAST) gyro [x, y, z] sync-induced disparity error: typical 5.8 px, worst 32.1 px = 64% of the nearest disparity`
- STEADY under 25 deg/s, TURNING 25 to 60, FAST above 60 (starting values to tune).
- The error estimate is focal length x angular speed x the left/right offset. It uses your measured typical offset (4 ms) and worst offset (22 ms). Units: pixels of extra disparity that turning adds.
- Above 60 deg/s an extra orange line says `"reduced accuracy"`, as the spec asks.

## What to try and what to expect
1. Hold still: TURN about 0 to 3 deg/s, STEADY. A steady non-zero value means the bias is off; restart the sender while still.
2. Turn your head left and right slowly (about 30 deg/s): TURNING appears.
3. Turn quickly (about 100 deg/s): FAST, the "reduced accuracy" line, and visible depth wobble.
4. Report: the numbers you saw at rest, slow turn and fast turn, and whether the depth really degrades when it says FAST.

## Not verified
Gyro axes are shown raw because the mounting orientation is unknown; the flag uses the vector magnitude. The thresholds and the pixel-error estimate have not been compared with measured depth errors during turning.

## Heading (yaw) stability (fixed 2026-10-28)

Symptom: with the board standing still the yaw curve ran away and kept jumping between +180 and -180. Causes, found by simulating a board on its side (gravity on the
sensor x axis, as your accelerometer plot showed):

1. The orientation filter started from the identity and needed 15-30 s to settle onto gravity. On a board that does not start flat it passed through pitch +-90 degrees on the way.
2. The yaw shown was the Euler yaw of the sensor x axis, which is undefined when x points up or down (gimbal lock). Near it, a few thousandths of noise swung the
   angle through the whole circle.
3. Underneath, the gyro's small leftover bias (the Pi measures it in the first 1.5 s; yours was taken while the rig moved) integrates into a slow real drift, because yaw
   has no reference unless the compass is fused.

Changes: the filter starts level from the first accelerometer sample (`q_from_accel`); the heading is the azimuth of the most horizontal sensor axis (`HeadingTracker`,
wrapped and continuous, held and flagged invalid if that axis nears vertical); the gyro's residual bias is followed while the board is still (`GyroBias`; it ignores
anything that moves or changes rate, but a perfectly steady turn slower than 1.5 deg/s would be partly absorbed). Debug line `fusion.yaw` (run with `GABAI_LOG_LEVEL=DEBUG`
or `--log-level DEBUG`) once a second: raw gyro mean, bias, heading wrapped and continuous, gyro-only heading, correction. Test: `py scripts/run_tests.py test_heading`.
Real yaw drift without a compass remains: a few tenths of a degree per minute after the bias is found, more if the board is warming up.

## Stable orientation: what was wrong and what changed (2026-10-30)
Symptom: tilt and heading wobbled and wandered with small hand movements. Measured on your own recordings (`py scripts/evaluate_orientation.py`, three 60 s IMU
recordings from 2026-10-26, and the compass run in `data/calibration/mag_partial.npz`):

| finding | measured | consequence |
|---|---|---|
| the Madgwick correction was a fixed length | the filter added **4.8 to 5.1 deg/s to the gyro all the time**, resting or turning, in a direction that followed accelerometer noise and shaking | never settled; every shake steered the tilt |
| the compass gate flapped | on your compass run the gate switched the compass on and off **6.5 times a second** (449 switches in 69 s), because the angle to gravity spreads 14 to 16 degrees while the board is waved about (accelerometer is not gravity then) | a heading that jumps every time the gate flips |
| the compass could tilt the board | the old 6-term step mixed compass and gravity | a disturbed compass moved roll/pitch |
| compass noise is NOT the problem | 0.5 to 1.5 counts (0.3 deg) when still; calibration quality 1.2%; axis map is the best of all 24, a continuous refit gains nothing | no extra smoothing of the compass |
| gyro is fine | noise 0.1 to 0.35 deg/s, residual bias 0.01 to 0.09 deg/s, no lag against the accelerometer (best shift 0 samples) | no gyro scale or delay correction added: the data do not show one |
| Pi sampling | IMU arrives at 161 to 163 Hz of the 200 Hz asked, 99th percentile spacing 10.5 ms (nominal 6) | not changed: needs a hardware test (see below) |

Changes: `gabai/orient.py` `Madgwick` keeps its name and `update`/`update_marg`, but (1) the tilt correction is proportional to the error (dies out at rest), weighted down
when the accelerometer length is not 1 g or has been varying in the last 0.15 s, and capped at 4 deg/s once the tilt is known (a gyro holds tilt for minutes; a sudden
disagreement is a push); it is lifted at start and after 2 s of steady disagreement; (2) the compass corrects only the heading (a rotation about the vertical), never the
tilt, with less weight while turning, a rate cap of 3 deg/s + 0.2 x turn rate once the heading is known, and a small integral term for steady drift. `gabai/mag.py`
`MagGate.check(..., trusted)` does not judge or learn the angle to gravity while the accelerometer is not gravity alone, and needs 5 good readings in a row after a reject.
`gabai/fusion.py` passes `trusted` (|a| within 5% of 1 g and turn rate under 150 deg/s). `beta` keeps its meaning as "higher follows the sensors faster": tilt time constant
1/(20 beta) s, heading 1/(6 beta) s.

Result on the three real recordings (`scripts/evaluate_orientation.py`): correction added at rest **4.9 -> 0.3 to 0.6 deg/s**, while turning **5.0 -> 0.7 to 1.4 deg/s**; tilt error
against the accelerometer where it reads gravity alone is unchanged within its own noise (still: 0.3 -> 0.4 to 0.6 deg median). Gate: 6.5 -> 2.8 switches a second on your
compass run, using the accelerometer length only (the live gate also uses the turn rate, so it should switch less). A 1.5 s, 0.3 g push tilts the estimate 5.4 deg instead of 8.6;
a zero-mean shake 0.3 instead of 0.7. Not measured: gyro + accelerometer + compass **together** on real data (none was recorded at the same time).

Check it on your sensors: Pi sender running, then
`py -m gabai.tools.record_sensors --name still --seconds 30` (still), `--name small --seconds 40` (small hand movements), `--name turn --seconds 40` (slow turns), and
`py scripts/evaluate_orientation.py --sensors data/sensors/<file>.npz` prints before/after per motion class, including the heading step while still. Live: `py -m gabai.tools.imu_3d`
now also draws `filter/accel_trust`, `filter/tilt_error_deg` and `filter/compass_weight`. Tests: `py scripts/run_tests.py test_orientation_stability test_orient test_fusion test_compass_fusion`.
Not done, and why: chip-clock (BMI160 sensortime) timestamps on the Pi and a faster poll would cut the sampling jitter, but the chip clock's drift against the Pi's was not measured and a
wrong mapping would break the camera sync that is tuned today; the compass delay (`mag_sync.json` says 0.0, never measured) is worth a `py -m gabai.tools.sync_sensors` run.

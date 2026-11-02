# Checking the HMC5883L compass

A magnetometer measures the magnetic field in three directions. There is no picture to look at, so the check has three parts: does it answer, do the readings behave like a real field, and can the readings be corrected (calibrated) into a trustworthy heading.

## Run
1. Copy the files to the Pi (PowerShell, project folder): `scp -r gabai xrnozy@192.168.1.63:~/`
2. Pi (the compass bus overlay and reboot from imu-guide.md must already be done):
```bash
cd ~
python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0
```
It should print `compass found (bus 3, 0x1E)` next to the IMU lines. If it prints `compass not used (...)`, send me that line.
3. PC:
```powershell
py -m gabai.tools.compass_check
```

## The window
Three scatter plots (X-Y, X-Z, Y-Z) of the readings, a compass dial, and status lines: raw counts, field strength in microtesla, readings per second, wobble, overflows, how many directions you covered, and the quality of a calibration if you made one now.

## Tests, in order
| Step | What to do | Pass when |
|---|---|---|
| 1 Still | Put the board down, hands off | numbers steady (wobble under about 0.5 uT); field strength 25 to 65 uT; 40+ readings per second |
| 2 Real field | Hold a phone or small magnet 5 cm from it, then take it away | strength jumps while it is near, and returns to the old value |
| 3 Turn flat | Keep the board flat and make one slow full circle | dial arrow sweeps all the way round once; "heading sweep" reaches about 360 |
| 4 Tumble | Rotate it through every direction for 20 s, including upside down | three plots fill out into rings/ellipses; "directions covered" near 100% |
| 5 Calibrate | Press `c` | message shows quality under 0.03 (good) or under 0.06 (usable); saved to `data/mag_cal.json` |
| 6 After calibration | Repeat step 3 | the rings are centred on the cross, and the arrow follows your turning smoothly |

## What the failures mean
- **No data / "compass not used"**: overlay line missing in `/boot/firmware/config.txt`, no reboot, wiring, or a QMC5883L clone (answers at 0x0D, different chip).
- **Overflows keep rising**: the field is stronger than the sensor's range; there is a magnet or speaker very close.
- **Strength far from 25-65 uT**: metal or magnets near the sensor (the Pi, power bank, screws, a speaker). Move it away from them or mount it on a short pole.
- **Rings are off-centre**: that is normal; calibration (step 5) corrects it.
- **Quality stays high after tumbling**: it did not visit every direction, or something nearby is moving or changing (a laptop, a phone). Repeat away from them.
- **Heading wrong by a fixed amount**: normal. The compass has no idea where north is until you point it; the dial's 0 means the sensor's +x axis points at magnetic north. Which way the sensor's x axis faces on your rig is not known yet.

## Limits
- Heading assumes the board lies flat. Tilting changes it unless it is tilt-compensated, which is not done.
- The compass board's axes are not aligned with the IMU or the camera; that mapping has not been measured.
- Verified so far with synthetic data and a fake Pi (the calibration recovered a known offset of 120, -80, 40 counts to within 3 counts, quality 0.018). Not verified on your real sensor.

## Why the yaw drifted and did not return after the rig was lifted (found 2026-10-28)

Evidence from your recordings: `data/mag_cal.json` and `data/mag_sync.json` did not exist, so the compass was never calibrated, never mapped onto the IMU axes, and `imu_3d` never fused it. The heading was gyro only. A gyro cannot know where it is after a lift: its heading integrates whatever bias is left, so it can never return. Your raw field was about 965 counts, roughly 89 uT; the Earth's field is 25 to 65 uT, so something magnetic sits close to the sensor (hard iron of about 50 uT), and the raw readings stepped between plateaus (x between -100 and -210 counts) when you moved things.

What changed (2026-10-28):
- `gabai.mag.calibrate_ellipsoid`: hard-iron offset plus a full 3x3 soft-iron matrix (the old per-axis scale cannot undo a tilted stretch). It refuses to claim an ellipsoid when the sensor was turned through too few directions (flat turn only) and says so in `coverage`.
- `gabai.mag.MagGate`: a reading whose strength is more than 25 percent from the calibrated one, or whose angle to gravity differs more than 15 degrees from what this place has been giving, is not used for the heading (counted and logged). A lasting change is accepted after 5 s.
- `gabai.mag.tilt_heading_deg`: tilt-compensated magnetic heading; the filter now starts pointing at magnetic north when a good compass reading exists, so the heading is absolute.
- The accelerometer calibration in `data/accel_calib.json` (made for Basalt) is now applied to the accelerometer before the orientation filter (your board read 0.95 g).
- Pi sender: the gyro bias is measured for 3 s and repeated (up to 5 times) if the rig moved, instead of once for 1.5 s.

What to do (each step once, then the heading is tied to magnetic north):
1. Pi: `python3 -m gabai.pi.sender --host <PC_IP> --left /dev/video2 --right /dev/video0` (keep the rig still for the first 3 s).
2. PC: `py -m gabai.tools.sync_sensors --seconds 30`, and while it runs tumble the whole rig through every direction (figure eights, upside down). It writes `mag_cal.json` and `mag_sync.json` and prints the quality and the field strength after calibration; quality under 0.03 is good.
3. PC: `py -m gabai.tools.imu_3d`. It says whether the compass is used. Watch `compass/heading_deg`, `compass/gate_ok`, `compass/field_vs_calibrated` and `heading/continuous_deg`. Lift the rig, put it back: the heading should return to within a few degrees.
4. Debug: add `--log-level DEBUG` to see `fusion.yaw` once a second, with `compass_minus_filter_deg` (the compass against the filter), the gate reason and counters.

Limits, said plainly:
- The calibration removes only what is fixed to the sensor. A field that changes with the Pi's or cameras' current, a cable, or something that moves relative to the sensor cannot be calibrated away; the gate throws such readings out, and the heading falls back to the gyro for the time. If the field stays at 89 uT after calibration (check `field_uT` in `imu_3d`; it should be 25 to 65 uT after calibration of `sync_sensors`), mount the compass on a short pole 10 cm or more from the Pi, USB cables, cameras and any speaker.
- Heading is relative to MAGNETIC north, not true north (no declination), and it is wrong indoors near steel, wiring and monitors. Returns to the same spot repeat well; two rooms do not agree.
- The axis map is one of the 24 axis-aligned rotations. A compass that is mounted tilted by some other angle is not corrected.
- Tested on simulated sensors only (known orientation, field, distortion). Not tested on your hardware.

# Coordinate frames and the one-time frame calibration (2026-10-28)

All sensors are fixed to the device. Their axes relative to the device never change, whatever pose the device is in (flat on a table, worn on the chest). What changes
with the pose is the rotation device -> world, and the orientation filter measures that from gravity (tilt) and the compass (heading). So nothing has to be remapped
by hand when you put it on. What the software must know once is: how the IMU axes and the compass axes sit in the device. That is what `frame_calibrate` measures.

## The frames (defined in `gabai/frames.py`)
| frame | axes | where it is used |
|---|---|---|
| device (D) = left camera | x right, y down, z forward | stereo depth, visual SLAM, Basalt, everything is expressed in it |
| IMU (I) | the BMI160 axes as wired | gyro, accelerometer |
| compass (M) | the HMC5883L axes as wired | magnetometer |
| world (W) | x magnetic north, y west, z up | orientation filter; heading = angle of the device's forward axis, 0 = north, growing towards west |

Maps: `D <- I` is `R_dev_imu` (`data/imu_sync.json` R_ci from the camera and gyro, or `data/device_frame.json` from gravity), `I <- M` is `R_imu_mag` (`data/mag_sync.json` R_im).
Your measured camera-IMU map (data/imu_sync.json) says the IMU's x axis points up when worn, its z axis backwards, and the camera forward is IMU -z. Your description of
the compass axes (x left-right, y front-back, z up-down flat; y up-down, z front-back worn) is a statement about one pose, not the fixed mounting; the mounting, including
signs, is measured by the calibration below, so no sign is guessed.

## One-time calibration (`py -m gabai.tools.frame_calibrate`)
Four captures, each started with Enter, no tumbling:
1. lay it on its back, front (camera side) UP, still 4 s: gravity gives the device z axis;
2. flat, one slow full turn about the vertical in 20 s: a ring of compass readings;
3. stand it upright as worn on the chest, left-right level, still 4 s: gravity gives the vertical axis (device -y);
4. same pose, one slow turn on the spot: a second ring about a different axis.
It solves: `R_dev_imu` (two gravity directions), the compass hard-iron offset (a sphere through both rings), and the compass->IMU axis map (the one of the 24 axis-aligned
rotations under which the field keeps a constant angle to gravity while turning, in BOTH poses; one pose cannot tell the axes apart). It refuses to save and says why when a turn was
too short, the tilt changed during a turn, the axis map is not clearly the best, or the field after calibration is not Earth-sized. Files written (old ones copied to `.bak`):
`data/device_frame.json`, `data/mag_cal.json`, `data/mag_sync.json`. The raw captures are kept in `data/frame_calibration_raw.npz`; `--from-npz` solves again without hardware.
`imu_3d`, `demo_viewer` and the fusion load these files on every start: no calibration at start-up.

Run it with the device and battery wired as they are worn: the Pi, the battery and the cables are part of the field the compass sees.

## Every time you wear it (automatic) and the check
- The fusion recognises the pose from gravity after 1 s of rest (`chest_upright`, `flat_front_up`, `flat_front_down`, `upside_down`, `on_its_side`, `tilted`), prints it in `imu_3d`,
  and uses the device's forward axis, not a guessed axis, for the heading.
- `py -m gabai.tools.frame_check` (wear it, stand still, 4 s) prints the pose, gravity and the Earth's field in the device frame, the camera pitch below the horizon, whether the
  field is Earth-sized with a plausible dip, and where the camera points relative to magnetic north.
- The SLAM world: `stereo_only` assumes the camera starts level, so its world z is up only if that is true. Put the pitch printed by `frame_check` into `[tracking] initial_pitch_deg`
  (a chest mount usually looks 5 to 15 degrees down) and the SLAM world z is truly up. `stereo_imu` (Basalt) is gravity-aligned already. The SLAM yaw (which way is x) is arbitrary
  in `stereo_only` (the camera starts looking along +x); only the IMU-and-compass heading is tied to magnetic north.

## Limits
- Sideways lean of the upright pose (rolling about the forward axis) is invisible to gravity and goes into `R_dev_imu` as an error of the same size; hold it left-right level. When
  `data/imu_sync.json` also exists the two maps are compared and a difference over 10 degrees is printed (`load_frames`). When both exist `imu_sync.json` wins, because it
  comes from the camera itself.
- Two rings give the hard-iron offset but not the soft-iron stretch. For that run `py -m gabai.tools.sync_sensors` once with a tumble; it overwrites `mag_cal.json` with the full fit.
- The compass is only as good as its surroundings: the 89 uT seen on your board means magnetic material close to it, and anything that moves relative to the sensor cannot be calibrated.
- Tested on simulated sensors with a known truth (all 24 axis maps recovered, device frame within 1 degree with a 15 degree chest lean). Not tested on your hardware.

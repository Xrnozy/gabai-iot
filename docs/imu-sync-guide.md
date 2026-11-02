# 3D camera view and IMU-camera sync

## What is new
- The IMU is now streamed as a timestamped series (about 200 samples per second, stamped on the same Pi clock as the camera frames), not as one reading per frame.
- The PC runs an orientation filter (gyro + gravity from the accelerometer) and keeps a history, so for every camera frame it looks up the orientation at the moment that frame was captured, not the latest one.
- `sync_imu_camera` measures two things from your own rig, with no board and no flashing light: the delay between the camera and the IMU, and which gyro axis is which camera axis.
- The live viewer draws a 3D camera (box, lens cone, axes, ground grid) in the top-left corner of the left picture, turning with the rig, with roll, pitch and yaw printed under it.

## One-time setup
1. Copy the changed files to the Pi (PowerShell, project folder): `scp -r gabai xrnozy@192.168.1.63:~/`
2. Start the sender on the Pi and keep the rig still for the first 1.5 s (gyro bias):
```bash
cd ~
python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0
```
3. PC, with a textured scene in front of the cameras (shelves, posters; not a blank wall):
```powershell
py -m gabai.tools.sync_imu_camera
```
When it says "recording", pick up the rig and rotate it by hand for 25 seconds: sweep left-right and up-down, quickly and smoothly (aim for 60 deg/s or more). It then prints:
- `camera-IMU delay: +NN ms`: how much later the camera frame is stamped than the motion it shows. A typical value is 30 to 100 ms.
- `match quality`: the correlation between the motion seen in the pictures and the gyro. Above 0.7 is GOOD. If it says WEAK, move more and use a more textured scene.
- the axis map and the correlation table. It saves `data/imu_sync.json`.

## Use it
```powershell
py -m gabai.spike.demo_viewer --source live --model m
```
The panel shows `IMU sync: orientation taken NN ms before the frame (calibrated)`. The inset title says `IMU axes OK` once `data/imu_sync.json` exists; before that it says `axes NOT calibrated` and the camera will appear tilted in a wrong direction. You can override the delay with `--imu-delay-ms 60`.

## What to check
1. Hold still: the 3D camera should sit still and level. Roll and pitch near 0 when the rig is level.
2. Turn left: the 3D camera turns left, the same direction as the picture.
3. Tilt the rig up: pitch goes positive.
4. Turn quickly and stop: the 3D camera should stop at the same moment the picture stops. A visible lead or lag means the delay is off; rerun `sync_imu_camera` or adjust `--imu-delay-ms`.

## Known limits
- Yaw drifts slowly, because no magnetometer is used (the compass is wired but not read yet). Roll and pitch stay correct because gravity corrects them.
- If the rig is accelerated hard (shaken), the accelerometer is fooled and roll and pitch wobble briefly.
- The delay and axis estimate assume mostly rotation. Moving the rig sideways while turning confuses it a little.
- The axis map is a signed permutation (it cannot represent a camera mounted at a tilted angle to the IMU).
- Verified so far: the filter, interpolation, estimator and 3D drawing with synthetic data (27 automated tests), and the whole path with a fake Pi (the filter reached the expected 119 degrees of yaw after a 60 deg/s turn for 2 s). Not verified: on your real IMU and cameras.

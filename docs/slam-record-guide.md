# Recording a sequence from your rig for Basalt

Basalt replays a recorded sequence, so you record first and replay after. This guide covers the recording. Basalt itself is built in `D:\slam\basalt` (native Windows, clang); see `docs/slam-plan.md`.

## Before you record: the camera-to-IMU rotation
Basalt needs to know how the IMU axes sit relative to the left camera and how late the camera frames are stamped. Without that, it cannot track. The viewer has been running with the DEFAULT (identity rotation, 45 ms guess), because `data/imu_sync.json` does not exist yet.

Make it once (about 1 minute, see `docs/imu-sync-guide.md`): start the sender, then tumble the rig by hand in all directions while looking at a textured scene, and run
```powershell
py -m gabai.tools.sync_imu_camera
```
It writes `data/imu_sync.json`. The recorder warns loudly if the file is missing.

## Record
1. Close the viewer (the recorder uses the same port, 5600).
2. On the PC:
```powershell
py -m gabai.tools.record_basalt --seconds 60
```
3. Start the sender on the Pi (the normal command, without `--low-power`, so frames pair as closely as possible).
4. Hold the rig still for the first 3 seconds, then walk slowly with the cameras on textured things 1 to 5 m away. Avoid blank walls, bright windows in the picture, and fast turns. Walk a loop and end where you started: the spec test is the end point matching the start within 0.5 m on a 30 m loop.

It prints the frame rate, the IMU rate and the longest gap every 5 seconds, then writes `data/slam/rec_<time>/`:
- `mav0/` the EuRoC-layout images (rectified, grayscale) and IMU data,
- `calib.json` the Basalt calibration (camera model from your stereo calibration, the camera-IMU rotation, datasheet noise values),
- `recording_info.json` rates and gaps,
- `run_basalt.cmd` the replay command.

## What is estimated, not measured
- The lever arm between the IMU and the left camera is written as zero. The real offset is a few centimetres.
- IMU noise values come from the BMI160 datasheet scaled to the sample rate, not from this unit.
- The camera delay is applied by moving the camera timestamps back; it is correct only if `imu_sync.json` holds a measured delay.
- The rig's stereo calibration is trusted as it is: the same data the depth uses.

## What a recording cannot show
A good Basalt trajectory on a sequence with the real drift visible still needs a path whose true end point you know. For the spec's loop test, mark the start on the floor and measure the gap at the end with a tape.

## Replay with Basalt (native Windows build)
Basalt is built at `D:\slam\basalt` (clang, vcpkg classic mode, no WSL). Each recording folder has `run_basalt.cmd`, which opens Basalt's window; or run it headless and save the trajectory:
```powershell
set PATH=D:\slam\vcpkg\installed\x64-windows\bin;%PATH%
cd /d D:\slam\basalt
.\build\basalt_vio.exe --show-gui 0 --dataset-path "<recording folder>" --dataset-type euroc --cam-calib "<recording folder>\calib.json" --config-path data\euroc\euroc_config.json --save-trajectory tum --save-trajectory-fn "<recording folder>\trajectory.txt"
```
Two source edits were needed for the Windows build: `#include <chrono>` in `thirdparty/ros/roscpp_core/rostime/src/time.cpp`, and the ROS bag branch removed from `src/io/dataset_io.cpp` (the shipped `basalt.patch` no longer applies to the current source). Build with `cmake --build build -j 4`; more parallel jobs ran the compiler out of memory.

## Check the camera-IMU rotation from the recording
```powershell
py -m gabai.tools.check_extrinsics data/slam/rec_<time>
```
It compares the rotation seen in the images with the gyro. Add `--write` to save the fitted rotation into `data/imu_sync.json` (the old file is copied to `data/backup/`). The sync tool snaps the rotation to a perfect axis permutation; on your rig the fit was 9.8 degrees away from that, and Basalt needs the exact rotation.

## What happened on your two recordings (2026-10-26)
| Recording | What it was | Basalt result |
|---|---|---|
| `rec_20261026_155458` | fast tumbling (median 53 deg/s), dark, blurred frames | not usable; 32 m of path in a 3 to 6 m box |
| `rec_20261026_155811` | slow (10 to 20 deg/s), sharp, 12.9 frames/s, not still at the start | with the snapped rotation: diverged to 12.5 km. With the fitted rotation: 20.5 m of path inside a 3.2 x 3.2 x 1.2 m box, end point 1.09 m from the start, 4 jumps of 0.3 m in the first 6 s |

Not validated: there is no ground truth for these walks, the start was not still (Basalt reported 1.8 m of motion in the first 3 s), and the camera delay is only known to a 30 ms range (about 2 to 32 ms). The spec test (30 m loop ending within 0.5 m, 60 s still drift under 5 cm) has not been run.

## Live Basalt from the Pi (step S6)
```powershell
py -m gabai.tools.live_basalt
```
Close the depth viewer first (same port 5600), then start the sender on the Pi. Two windows open: Basalt's own (camera tracks, map points, 3D path) and ours (left camera, a top-down path with the start marked, pose count, pose delay, and a red NO POSE warning). The run is also saved to `data/slam/live_<time>/` as an ordinary recording plus `trajectory.txt`, so it can be replayed. `py -m gabai.tools.live_basalt --replay <recording folder>` runs the same code on a recorded sequence at real time (`--speed 3` for faster).

The tool waits until the rig has been **still for 1 second** before it starts Basalt ("HOLD STILL" bar in our window). Without that, Basalt's result depends on thread timing: one recording gave 20 m of path on one run and 8.8 km on another. `--no-still-gate` turns the wait off.

How it talks to Basalt: `gabai/slam/vit.py` loads `D:\slam\basalt\build\basalt.dll` and calls Basalt's tracker interface (VIT 2.0). That needed one more edit to the Basalt build: the interface functions had no Windows export, so `src/vit/vit_exports.def` and a link option in `CMakeLists.txt` were added.

Measured with a recording replayed through this path: pose delay 16 ms median (31 ms p95 at real time). That is the delay between pushing a frame and getting its pose, not the delay from the real world.

## Accelerometer calibration (needed before Basalt is reliable)
Your accelerometer reads gravity about 2 to 4 percent low (9.42 and 9.60 m/s^2 against 9.807 on the two recordings). Basalt integrates that into metres. Test on a still sequence: starting Basalt at frame 5 instead of frame 0 made it drift 15.6 m while sitting still; with the accelerometer rescaled by 4.1 percent the same data drifted 0.3 m. Basalt's calibration file had zero accelerometer correction.
```powershell
py -m gabai.tools.calibrate_accel
```
Start the sender first. Put the rig down with each of its six sides facing the ground in turn, and hold each one dead still for 2 seconds. It prints which sides it still needs; add a few tilted holds for a better fit. It writes `data/accel_calib.json`, and `record_basalt` and `live_basalt` pass those numbers to Basalt (they print a warning while the file is missing). Your two existing recordings cover only two orientations, so the tool refuses to fit from them.

## What is tested and what is not (2026-10-26)
- Tested on recordings only, not yet on the live Pi stream: the live path matches the offline program exactly on the first 20 poses, and on a still sequence (0.09 m start to end offline and through the live path).
- Not tested: a live run from the Pi; any run with a measured accelerometer calibration (only a crude 4.1 percent scale); the spec drift and loop tests.
- `rec_20261026_155811` (the slow walk) is not a reliable benchmark: it did not start still, and Basalt's result on it varies between runs.

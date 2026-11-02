# Better VIO and a cleaner 3D cloud: findings and plan

Status of every claim: **[measured]** from your logs or code, **[source]** from the official page named, **[assumption]** not checked.

## 1. Biggest causes (in order of evidence)

| # | Cause | Evidence |
|---|---|---|
| 1 | Pi under-voltage and link stalls | `link.gap` of 0.3 to 3.9 s and `under_voltage_now`, `throttled_now` in every run of `logs/pipeline_stereo_only_20261031_220948` [measured]. The pose cannot be tracked across a stall. |
| 2 | Lost visual tracking | Many frames track 100 to 400 points but only 0 to 14 agree. From 22:15:06 there were no features with trusted depth, and depth coverage fell to 7 to 40 % [measured]. |
| 3 | Pose jumps accepted | Steps of 0.4 to 0.7 m in one frame (up to 4 m/s) in the same log [measured]. `stereo_imu` now rejects steps above 3 m/s (+5 cm). |
| 4 | Wrong camera-IMU calibration | `data/imu_sync.json` was fitted on the wrong picture layout [measured earlier]. Not yet redone. |
| 5 | Depth noise from the short baseline | Baseline 41.7 mm, f = 618 px. Depth std = z² · 0.3 px / (f · B) = 0.0116 · z² m: 1.2 cm at 1 m, 4.7 cm at 2 m, 10.5 cm at 3 m, 19 cm at 4 m [calculated from the config model]. The map takes points to `max_range = 4.0` with `max_depth_std = 0.0` (no limit) [measured in `config/pipeline.toml`]. |
| 6 | Loose map consistency gate | `max_misalign_cm = 15`: a frame whose pose is up to 15 cm off the map is still added. With 3 cm voxels that smears surfaces [measured config; the effect on your cloud is an assumption]. |
| 7 | No map correction in `stereo_imu` | `map.refine_pose` is ignored in this mode, so nothing pulls the pose back to the map [measured]. |

Point-cloud filtering cannot fix a wrong pose. Fix 1 to 4 before judging the cloud.

## 2. What the cloud already does (do not add twice)

[measured, config and code] left-right consistency check, speckle removal, `surface_filter` (no flying points, no small islands), per-voxel mean weighted by 1 / depth variance, `carve_m` and `carve_decay` (forget what the camera sees through), `core` (drop weakly seen voxels), `window_m`, `max_voxels`.
So Open3D statistical and radius outlier removal would repeat most of this. Open3D 0.19.0 is already installed here [measured].

## 3. VIO software compared

Where it runs: the Pi only captures and streams. All estimation runs on the Windows PC [measured, `gabai/pi/sender.py`, `gabai/pipeline`]. Pi performance does not limit the choice. PC speed and Windows support do.

| | License | Stereo + IMU | Needs | Fit for you |
|---|---|---|---|---|
| Basalt [repo](https://gitlab.com/VladyslavUsenko/basalt) | BSD-3 [source] | Yes [source: project] | Already built here (`BASALT_DLL`) [measured] | Cheapest benchmark: same recordings, no new install. It drifted on your rig, but with a wrong calibration and a bare scene, so that run does not prove the estimator is bad. |
| ORB-SLAM3 [repo](https://github.com/UZ-SLAMLab/ORB_SLAM3) | GPLv3 [source] | Yes, stereo-inertial, with loop closure and multi-map [source] | Ubuntu, Pangolin, OpenCV, g2o. Authors advise a strong PC (e.g. i7) [source] | Heavy, Linux only. Run-to-run variance reported [source: [HybVIO paper](https://arxiv.org/pdf/2106.11857)]. |
| VINS-Fusion [repo](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) | GPLv3 [source] | Yes, with online camera-IMU rotation and time-offset estimation [source] | ROS Kinetic or Melodic, Ceres. Authors recommend global-shutter cameras and hardware sync [source] | The time-offset estimate is what you lack. Your cameras are USB webcams without hardware sync [assumption: rolling shutter]. |
| Kimera-VIO [repo](https://github.com/MIT-SPARK/Kimera-VIO) | BSD-2 [source] | Yes [source] | GTSAM, OpenGV, OpenCV, Ubuntu 20.04 [source] | Makes a 3D mesh, but a heavy dependency chain on Windows. |

No Raspberry Pi benchmark of any of these was found [source: search, nothing found]. No claim here about accuracy on your rig. That needs a test on your recordings.

Recommendation: keep `KeyframeVO` for now and run the section 5 tests. If stationary drift or pure-rotation tests fail and the cause is the estimator, not power or calibration, test Basalt on the same recordings next. Do not move to ORB-SLAM3, VINS-Fusion or Kimera until Basalt has been tried: each needs a Linux or ROS setup.

## 4. Point cloud: what to do

1. **Limit noisy points (no code).** Set `max_depth_std = 0.05` (keeps points to about 2.1 m) or `0.08` (about 2.6 m) and `max_range = 3.0` in `config/pipeline.toml`. Loses far points on purpose. Not yet tested on a recording.
2. **Tighten the map gate.** Try `max_misalign_cm = 6`. More frames are dropped when the pose is poor. Test before keeping.
3. **TSDF** ([Open3D `ScalableTSDFVolume`](https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html), [source]): fuses each depth image into a signed-distance volume with the camera pose, and averages frames, which gives a smoother surface. Your weighted voxel mean plus carving already do part of this. TSDF adds sub-voxel surfaces and a mesh. The page notes that finer voxels are more sensitive to depth noise [source]. Speed at 640×360 and 15 fps on your PC is not measured. Worth an offline test (recorded poses + depth, 1 to 2 cm voxels) after poses are stable, not before. It cannot fix drift: a wrong pose puts the surface in the wrong place.
4. **Open3D filters** ([statistical and radius outlier removal](https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html), [source]): use only for the exported cloud, not live. Parameters depend on the data [source].

## 5. Test order and what each result means

```
1 Pi power ok?  vcgencmd get_throttled = 0x0, no link.gap > 0.5 s in 5 min
   no  -> new 5 V supply and short cable, retest. Nothing below is trustworthy.
   yes -> 2
2 Calibration: sync_imu_camera -> record_basalt -> check_extrinsics --write
   "camera layout" lines present, delay and R_ci saved -> 3
3 Stationary 60 s: wander max cm / deg  (scripts/evaluate_still.py)
   > 5 cm -> look at tracking inliers and the rest-hold flag; fix depth coverage or scene first
   ok     -> 4
4 Pure rotation: turn on the spot, position change
   > 5 cm -> check camera-IMU delay and R_ic; try tracking.gyro_prior=false to isolate
   ok     -> 5
5 Cover the cameras 3 s, then restore
   position held while covered, then jump on return > 10 cm -> relocalisation or keyframe restart
   ok     -> 6
6 Cloud: scan one flat wall, then fit a plane and read the thickness
   thick or doubled -> pose (back to 3 to 5); thick but single -> depth noise (section 4.1)
   ok     -> 7
7 30 m loop, end error < 0.5 m
   fails with 3 to 6 ok -> estimator limit: try Basalt on the same recording
```

## 6. Plan, easiest first

1. Fix Pi power. Check `vcgencmd get_throttled` on the Pi.
2. Section 4.1 and 4.2 config changes. Replay one recording before and after, look at the cloud.
3. Redo the calibration (steps in my earlier message).
4. Tests 3 to 5 above.
5. Add a wall-flatness number (plane-fit thickness) so cloud quality is a measurement. Small script, not written yet.
6. Only if the tests show an estimator limit: Basalt benchmark on your recordings.
7. TSDF offline test.

## 7. Hardware and performance

- Pi 3B: captures two USB cameras and the IMU, streams at 10 to 15 pairs/s. No estimation runs there. Under-voltage is its main risk (USB cameras draw power).
- PC: runs depth (LAS2), KeyframeVO and the map. Frame rate and GPU load are not measured in this document.
- Assumption: no extra hardware is needed. A global-shutter or hardware-synced camera pair would help every VIO tool above, and you said no extra hardware.

## 8. Stutter test and the fix (2026-11-01)

Test: `scripts/evaluate_stutter.py data/recordings/vio_20261101_014854 --stutter log` plays the recording through the live code path in virtual time, with the stalls of the 01:19 to 01:23 log (170 of 815 pairs get through, up to 3.8 s apart, the IMU samples of a stall arrive with the next pair). The reference is the same estimator on all 815 pairs, so the numbers show what the stutter costs, not absolute accuracy. Run it with `--baseline` to see the behaviour before the fix. Model `s`, one recording, 60 s.

| | trusted poses (of 170) | position error vs reference, rms / max | rotation step error vs reference, median / p90 |
|---|---|---|---|
| before | 123 | 15.5 / 63.6 cm | 0.99 / 6.05 deg |
| after | 147 to 148 | 11.5 to 14.3 / 17.9 to 21.8 cm | 0.85 to 0.90 / 4.5 to 5.4 deg |

Before: 3 runs gave identical numbers. After: 3 runs gave 14.3, 11.5 and 11.5 cm (a worker thread makes it vary a little).

What changed (`gabai/pipeline/vio.py`, `gabai/kfvo.py`): when the IMU samples cover the whole stall, the gyro rotation is kept as the prior over the stall (the tracker used to throw it away after 0.4 s), and for pairs more than 0.15 s after the previous one the camera rotation is taken from the gyro and only the translation is searched, with a 2-point RANSAC (`_t_only`) that works when most feature tracks are wrong. A full pose still wins when it has clearly more support. Full-rate behaviour is unchanged by construction (pairs under 0.15 s apart take the old path).

Tried and dropped: larger Lucas-Kanade window and one more pyramid level on stalls (no gain), fixing the rotation only when the full pose fails (less gain), and accepting 10 instead of 15 inliers for the rotation-fixed solution (more trusted poses, but 25 cm rms error: wrong poses got through).

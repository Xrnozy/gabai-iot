# Seeing the tracking and the map grow: `slam_cloud_view`

```bash
py -m gabai.tools.slam_cloud_view data/slam/live_20261026_181526                   # opens Rerun, plays the whole recording as fast as it processes
py -m gabai.tools.slam_cloud_view data/slam/live_20261026_181526 --realtime        # paced like the original
py -m gabai.tools.slam_cloud_view data/slam/live_20261026_181526 --basalt-ui       # also Basalt's own window with its real tracked points
py -m gabai.tools.slam_cloud_view <folder> --save data/view.rrd --no-viewer        # record only; open later with: rerun data/view.rrd
```

It runs Basalt on a recording's IMU and pictures, and for each pose it makes stereo depth (LAS2), keeps the textured pixels (semi-dense), places the points in the world with the pose and fuses them into 5 cm voxels. Drag the timeline in Rerun to replay; the cloud and the map update as the camera moves.

## What you see, and what each part really is

| Rerun entity | What it is |
|---|---|
| `world/camera`, `world/trajectory` | Basalt's pose of the camera (IMU pose combined with the camera-IMU extrinsics). |
| `world/live` | This frame's stereo depth points in the world. Depth is LAS2's, not Basalt's. |
| `world/map` | All frames fused. Walls stay thin when the pose is good and smear when it drifts (shown in a test with a deliberately drifting pose). |
| `camera/image` | Every frame of the recording (773 of 773 in the 60 s recordings), JPEG-compressed. |
| `camera/tracks`, `vision/klt_tracks` | Feature tracks from this tool's own optical-flow tracker. **Not Basalt's tracked landmarks.** Basalt's C interface returns poses only; its landmarks are visible only in its own window (`--basalt-ui`). |
| `imu/gyro_rate_dps`, `imu/accel_norm_g` | The recorded IMU. |
| `imu/rotation_disagreement_deg` | Angle between Basalt's rotation between two poses and the rotation from integrating the gyro over the same time. Small = IMU and tracker agree and are in sync. It shows consistency; it does not show how Basalt weights the IMU inside its optimiser, which is not exposed. |
| `imu/gyro_bias_estimate_dps` | The constant gyro offset that would make the gyro match Basalt, learned as a running mean from Basalt's own rotation. It is removed before the disagreement is computed, so a bias does not appear in the disagreement plot but does appear here. |
| `vision/rotation_rate_dps` | Rotation rate Basalt reports, to compare with the gyro rate. |

## Why the earlier preview showed only a short piece

`preview_3d` (the hands and objects preview) defaults to `--max-frames 150 --skip 3`, about 15 seconds of a 10-minute file, and it only logs frames it processed. For a recording, `slam_cloud_view` logs every frame. For the long `data/left.mp4`, raise `--max-frames` and lower `--skip` in `preview_3d` (about 1.7 frames per second of processing on this PC with `--model s`).

## Results on the recordings in `data/slam/` (2026-10-27, Basalt run with no determinism, stride 2)

These show the tool working, and they show that tracking on these recordings is not trustworthy. Nothing here is a measurement of accuracy.

| Recording | Still-gate | Poses | Path Basalt reports | Notes |
|---|---|---|---|---|
| `rec_20261026_155811` | never opens (gyro stays above 3 deg/s) | 772 of 773 frames | 19.8 m in one run, 10870 m in the next | Two runs of the same input give different results: Basalt diverges. The saved `trajectory.txt` in that folder also reports 12503 m. |
| `rec_20261026_155458` | opens at 1.0 s | 810 of 811 frames | 1504 m | Diverged. |
| `live_20261026_181526` | opens | 766 of 767 frames | 53 m | Not diverged to kilometres, but 53 m is still not what a slow walk in a room looks like. Unverified. |

Reading these: the map keeps only voxels within 30 m of the camera, so when the pose runs away the map is mostly evicted (`map_voxels_evicted` in the summary). A map that is small while the path is huge means tracking was lost, not that the room is empty. The known causes in this project: a rig that is not still at the start (gyro bias above the gate's 3 deg/s), accelerometer calibration, camera-IMU rotation, and fast motion. `docs/slam-plan.md` and `docs/troubleshooting.md` list the checks.

## Limits

- Rerun's timeline replays what was processed. It is not a live window onto the Pi: for the live rig, `live_basalt` saves the run as a recording, which this tool then plays. Putting the same view on the live stream is not done.
- Speed: about 35 to 60 s to process a 60 s recording with LAS2-s and stride 2 on this PC (Basalt included).
- Semi-dense here means "stereo depth only where the image has gradient". It is not Basalt's own semi-dense map, which its interface does not expose.

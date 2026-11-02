# KLT feature tracking on the Pi (optional)

Off by default. Without `--klt` the sender runs exactly as before: no tracker is built, no extra message is sent, the `pi.stats` line has no `klt_*` fields.

```bash
# Pi
python3 -m gabai.pi.sender --host 192.168.1.11 --left /dev/video2 --right /dev/video0 --width 640 --height 360 --fps 15 --capture-fps 30 --exposure auto --klt
# PC (instead of the pipeline: one program per port)
py -m gabai.tools.klt_view --port 5600            # --ids writes feature ids, --points-only skips the pictures
```

The sender prints `tracking mode: ...` at start and logs `pi.tracking_mode`.

## What runs on the Pi

Per sent pair, after the pair and the IMU message are already out: decode the left JPEG straight to a 320x180 grayscale picture (the JPEG decoder skips work; no
BGR picture, no resize), follow the live points with `calcOpticalFlowPyrLK`, drop tracks that fail, move more than `--klt-max-err` grey levels, leave the picture
or fail the optional forward-backward check, and run `goodFeaturesToTrack` only when fewer than `--klt-min-frac` x `--klt-max` points are left (and at most every 5 frames),
masked so it does not pick points next to existing ones. Single-threaded (`cv2.setNumThreads(1)`), no extra thread. A tracking error is logged and skipped; the video is unaffected.
`--klt-stereo` adds the right picture's decode and one LK from left to right.

Pairs are not dropped for tracking. If tracking plus sending exceeds the frame period, the existing newest-pair logic and rate governor apply as they do for a slow link; `pi.klt_slow` warns when
tracking alone takes over half a frame period.

## Message (type 7, `KLT`)

Same `t_capture` as the FRAME it describes. Per point: id (u32), x, y (u16, pixels x 16 of the full 640x360 picture), status (0 tracked, 1 new, 2 lost = last known position, sent once), and with
`--klt-stereo` the right x, y (0xFFFF = not found). 9 bytes a point (13 with stereo): 200 points at 15 pairs/s is about 215 kbit/s (310 with stereo), against about 16 Mbit/s of video.
Coordinates use the OpenCV convention (pixel centre at the integer); `klt_view` adds 0.5 because Rerun puts pixel centres at half integers.

## Measure it

`python3 -m gabai.pi.klt` times decode / track / detect on synthetic frames with no cameras. `python3 -m gabai.pi.klt_bench --seconds 60 -- <your sender arguments>` runs the real sender
without and with `--klt` and prints CPU, memory, temperature, throttle flags, camera and send fps, pairing offset, KLT time and bandwidth side by side. The Pi has no power sensor; use a USB meter.

## Limits

- Shown points are in the raw sent pictures. The pipeline's panels show rectified pictures, whose coordinates differ, so use `klt_view` rather than drawing these points on those panels.
- The LK error cut-off (`--klt-max-err`, 20) was chosen on a synthetic scene (good tracks 1-12, wrong ones 25-38). Check it on real footage; a camera with a sudden exposure change loses tracks sooner.
- `scale` must be 1, 0.5, 0.25 or 0.125 (sizes the JPEG decoder can produce directly). At 0.25 a corner is placed to a few pixels only.

## Measurement groups: frames, KLT and IMU kept together (live)

TCP already delivers in order without loss while a link is up, so the losses were elsewhere. Where measurements were lost before: the Pi's IMU buffer kept only the newest 120 samples per frame (a 0.6 s
stall lost the rest); a reconnect lost everything in flight; the receiver kept only the newest pair; KLT was matched to a pair by float equality of its time and dropped if it came late; the IMU was read
before the newest pair, so its interval could end short of the pair's time.

What the link carries now (all optional trailers/fields, older peers still work):

| stream | numbered by | sent |
|---|---|---|
| stereo pair | the cameras' own frame counters (left, right): a gap is a pair the Pi skipped | after its IMU |
| KLT | the counter of the pair it was computed from, plus per point id, status, patch error | after its pair |
| IMU | a sample number counted on the Pi | before the pair, oldest first, nothing skipped (30 s ring, 400 samples per message) |
| ACK (PC to Pi) | the first IMU sample number the PC still needs | right after a reconnect: the Pi sends the lost samples again |

The Pi still takes the newest pair when its link is slow (a queue of old 135 KB pairs on an undervolted Pi is what made the delay), but the gap is now visible (`pi.stats` `cam_skipped`, `sync/total_pi_skipped`) and the IMU record
across it is complete, which is what the visual odometry integrates over a gap.

The PC's `SyncBuffer` (`gabai/pc/syncbuf.py`, setting `[sync]`) releases one group at a time to the pipeline: the pair, the KLT points with the same counter (a counter match with a different time is refused), and every IMU sample
since the previous group. It releases at once when the IMU has reached the pair's time and the points are here; otherwise it waits at most `imu_wait_ms` / `klt_wait_ms` and releases the group marked incomplete. Repeated pairs and
repeated or out-of-order IMU samples are dropped; a hole in the sample numbers is counted and its length reported with the group that spans it. Pictures are skipped only when the pipeline is more than `max_lag_s` behind,
counted, and the IMU interval of the skipped ones goes out with the next pair. `max_lag_s = 0` never skips. The video viewer (`LiveReceiver.latest()`) is a separate path that always jumps to the newest picture: skipping a display picture
removes nothing from the SLAM record.

Not done, on purpose: the Pi does not queue or resend old pictures or KLT sets (VO needs the IMU across a gap, not the old pictures, and resending them over a link that just failed is what makes a call lag); the IMU shares the one TCP
connection with the pictures, so a blocked link delays both (a second connection would separate them, at the cost of two streams to keep in step).

## Fusing the tracks into the pose (stereo_imu, live)

```bash
py -m gabai.tools.rerun_pipeline --source live --mode stereo_imu --klt            # sender on the Pi started with --klt; --klt-weight 0.9 leans harder on KLT
```

Off by default (`[klt] enabled = false`); without it the pose is what it was. With it, `gabai/kltfuse.py` does this after each `KeyframeVO` step:

1. **Points.** The Pi's points are mapped into the pipeline's left picture (calibration rotation, rectification, shrink, swapped layout, `flip180`), checked against the real rectifier in `tests/test_kltfuse.py`.
2. **Keyframe points.** At each `KeyframeVO` keyframe, the tracked points with trusted stereo depth become 3D points (re-seeded from the current frame when under 40% of them are still tracked).
3. **KLT translation.** Each frame, with the rotation FIXED to the one `KeyframeVO` already has (the IMU's wherever the gyro covers the frame), the translation that best explains the points the Pi still tracks (by id) is solved.
   It counts only with at least `min_inliers` points within `max_reproj_px` RMS; a rotation that the tracks contradict by more than about a degree makes the fit poor and the KLT is not used.
4. **Combine.** Blend `weight` toward KLT (less with few points); weights sum to 1, so it is one estimate chosen between two, not a second constraint added. A difference larger than `max_step_m` is closed over several frames. When the PC tracker lost the frame, the KLT translation holds the position (rotation from the IMU).
5. **Written back.** The fused translation goes into `KeyframeVO` (its keyframe too when one was just made). The rotation is never changed by KLT. Gravity alignment, rest hold and the speed check then apply as before.

Rerun: panel "Pi KLT tracks" (points coloured by patch error, ids, trails, motion arrows, lost points, new points; `klt/features/left` for hover coordinates), `klt/count/*`, `tracking/klt_*`, and `sync/*`:
`frame_seq`, `klt_frame_seq`, `klt_seq_minus_frame_seq` (0 = matched), `imu_samples_in_interval`, `imu_last_minus_frame_ms` (positive = the IMU was already there), `imu_gap_s`, `backlog_pairs`, `imu_queue`, `latency_ms`,
`wait_for_parts_ms`, `skipped_before`, `pose_lag_frames`, `ok`, and running totals `total_*` (pi_skipped, imu_missing, dup_frames, recovery_skipped, incomplete, klt_missing, ...). A warning goes to the events log when a group is incomplete.

### What is not shown to be true

"KLT is more reliable than the PC tracker" is an instruction, not a measured fact. The PC tracker follows each keyframe point directly at full size with a gyro-guided search; the Pi tracks frame to frame on half-size pictures, which can drift and lose points in blur. The weight is therefore a setting. Compare `--klt-weight 0` (PC tracker only, same code path) with 0.75 and 1.0 on the same motion before trusting it. The tests prove the mapping, the combining rules and the buffering on known data, not that the fused pose is more accurate on real footage, and the buffering has not run against a real undervolted Pi.


## Kalman filter and the Pi's own motion estimate

```bash
# Pi: stereo tracks from both cameras and the motion between two sent pairs computed ON the Pi (copy data/stereo_calib_new.npz to the Pi first)
python3 -m gabai.pi.sender --host 192.168.1.11 --left /dev/video2 --right /dev/video0 --width 640 --height 360 --fps 15 --capture-fps 30 --exposure auto --klt --klt-motion --klt-calib ~/gabai/data/stereo_calib_new.npz
# PC
py -m gabai.tools.rerun_pipeline --source live --mode stereo_imu --filter
```

**On the Pi (`--klt-motion`, implies `--klt-stereo`).** The left tracks are followed into the right picture (`--klt-stereo`, one LK of the same points, not a second detection). The previous pair's points get a depth from
their left/right disparity (rectified with the calibration, same-row check), the current pair's tracked points are their new pixels, and one PnP of about 100 points gives the camera's motion between the two pairs. The message carries it
(rotation vector, translation, interval in seconds, inliers, reprojection RMS: 38 bytes). It costs a point rectification, a triangulation and one PnP per pair: measure it with `klt_bench` (the `klt_ms` it logs now includes it).
It needs the Pi's camera order to match the calibration; otherwise the same-row check fails and no motion is sent, never a wrong one.

**On the server (`--filter`, `[filter]`).** `gabai/vioekf.py` is an error-state Kalman filter in the gravity-aligned world: state position, velocity, orientation, gyro bias and accelerometer bias.
- The IMU predicts at every sample. Its rotation noise is small, so the IMU stays the main source of orientation.
- KeyframeVO's pose of each frame is a measurement of position and orientation. It is refused when it is farther from the prediction than the chi-square gate (an outlier, not a correction); five refusals in a row reset the filter to the pictures.
- The Pi's motion is a velocity measurement (its translation over its interval) and an angular-rate measurement (which learns the gyro bias). It does not measure orientation or position, so it cannot pull the rotation away from the IMU.
- At rest: velocity and angular rate are zero.
- KeyframeVO and the keyframes are untouched; the filter shapes the reported pose (smoother, outliers refused, half a second of coasting through a lost picture, then the pose holds). `--filter` switches `--klt`'s direct translation fusion off, so the same tracks are not used twice.

Tested on a simulated trajectory with IMU bias and noise, picture noise, 8% position outliers and a one-second picture dropout (`tests/test_vioekf.py`): the filtered position error is under 0.6x the raw picture noise, outliers are refused, the gyro bias is learned from a start of zero. The tuning has not been checked on real recordings:
watch `filter/innov_pos_cm`, `filter/innov_att_deg` and `filter/ekf_rejected` in Rerun. Innovations that stay large mean the noise settings do not match your rig. The VO pose and the Pi motion come from the same pixels, so their errors are partly shared; the filter treats them as independent and the defaults are widened to compensate, not removed.

**Not a filter problem.** Many object boxes for two monitors and a noisy cloud come mainly from pose drift: the same monitor seen later lands somewhere else in the world, so it is a new object and a second layer of points. A steadier pose is the fix this adds. The settings that act on the symptoms directly are `objects3d.min_looks` (a landmark is drawn after this many looks; 5 shows one-off clusters), `objects3d.gate` (how far apart two looks of one object may be before they are two objects) and `map.max_depth_std` (drops the noisiest far points).

> Overview of the RTAB-Map lab, GPU TSDF, Gaussian map and navigation map: `docs/rtabmap-lab-and-mapping-update.md`.

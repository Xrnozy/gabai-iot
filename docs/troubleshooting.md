# Troubleshooting

Find the symptom, run the command, read the log event. The report command first, always:

```bash
py -m gabai.tools.diag_report
```

Findings marked `confirmed` have measured evidence; `possible` ones list causes to check in order. Do not treat a `possible` cause as proven.

## Link and Pi

| Symptom | Look for | Likely cause and next step |
|---|---|---|
| Viewer waits, no picture | `link.listening` but no `link.accept` | Pi not sending, wrong PC IP, firewall. On the Pi run `python3 -m gabai.pi.sender --host <PC_IP> --test-pattern` (no cameras). Allow Python through the Windows firewall for port 5600. |
| `link.rejected` | reason `token mismatch` or `first message was not a HELLO` | Token differs. Same `--token` on both sides. |
| `pi.connect_failed` repeating | `pi.connect_failed` with the error | Network, wrong host. The sender retries with back-off. |
| `link.lost` often | `link.lost` times, `link.stats` gaps | Wi-Fi quality; try the Ethernet cable; `--sndbuf` smaller. |
| Low frame rate | `link.stats` rate, `pi.stats` dropped, `check.suspect` latency | Pi CPU (`pi.stats` CPU temp), Wi-Fi, JPEG size. Try `--fps 10`. |
| `pi.stats` throttle flags | under-voltage / throttled | Power supply: see `docs/pi-power-guide.md`. Results taken while under-voltage are suspect. |
| `cam.mode_mismatch` | camera did not give 640x480 MJPEG | USB bandwidth or camera. Unplug other USB devices; use different ports. |
| `cam.open_failed`, `cam.reader_error` | device path | `v4l2-ctl --list-devices`; use `--left /dev/video2 --right /dev/video0` as in your guides. |
| `pi.imu_init_failed` / `pi.mag_init_failed` | I2C error | Wiring, `i2cdetect -y 1` (BMI160) and `-y 3` (compass with the gpio overlay). The system continues without that sensor and says so; orientation then falls back to a level camera. |

## Depth and alerts

| Symptom | Look for | Likely cause and next step |
|---|---|---|
| Alerts about the floor ("obstacle ahead" on empty floor) | `zones.floor_model`, `cam_h_m`, `n_obstacle_px` in the trace | Wrong `--cam-h`, uncalibrated IMU (floor model level instead of tilted), a bad calibration. Reproduce with `replay_pipeline --sim clear_floor_wrong_height`. Fix the height, recalibrate the IMU sync. |
| Obstacle not announced | `n_beyond_range`, `n_removed_as_floor`, valid fraction | Object beyond 3 m optical depth, removed as floor, or no texture (blank wall) so depth is untrusted. Blank surfaces are a known weakness of stereo. |
| Alert comes late on a tilted camera | `p5_depth_m` vs `p5_horizontal_range_m` | The spoken distance is optical depth, which is longer than the horizontal distance when tilted. Known; behaviour unchanged. |
| `cant_see_ahead` too often | `valid_fraction_below_horizon`, `check_trusted_fraction` | Low light, motion blur, textureless floor, bad rectification. Look at the saved frame (`--diag-save frame,depth`). |
| Depth flickers frame to frame | `check_pair_sync` suspect, `pair_dt_ms` | Left/right captured at different times. Raise `--capture-fps`, avoid fast motion. |
| Disparity pinned at the cap | `check_disparity_cap` | Object closer than about 0.23 m. Not a failure of the model; the range limit. |
| No sound | `alert.suppressed` events carry a `reason` | Audio off (`--no-audio`), clip missing (`py -m gabai.tools.make_alert_audio`), debounce or repeat rule. `alert.spoken` is logged for every phrase actually played. |
| Bad calibration | self-check, `docs/calibration-guide.md` | Reprojection error, rows not aligned; recalibrate with more varied board poses. |

## SLAM

| Symptom | Look for | Likely cause and next step |
|---|---|---|
| Basalt never starts | `slam.gate_open` missing | Rig not still for `--still-s` seconds; or `--no-still-gate` (results then vary). |
| Pose jumps | `check_pose_step` suspect, `slam.summary` findings | Fast rotation, poor lighting, accel calibration missing (`slam.accel_calib_missing`), wrong camera-IMU rotation (`check_extrinsics`). |
| Drift while still | `slam.summary` still-drift finding | Check the message: it says whether the number came from extrapolation. Re-run `calibrate_accel`. |
| `slam.crash` | traceback in the log | Send the package (below). |
| Config error with spaces in the path | | Basalt cannot parse paths with spaces; the tools copy config to `D:\slam\cfg`. |

## Logging itself

| Symptom | Cause |
|---|---|
| `log.dropped` in the log | The queue filled (slow disk or very high DEBUG volume). The log has gaps; lower the level or point `GABAI_LOG_DIR` to a faster disk. |
| `suppressed` field on a line | Repeats of that line were limited to one per period; the field is how many were skipped. `trace.rate_limited` means interesting frame traces were skipped: the total is in the event. |
| Run folders missing | Only the newest `GABAI_LOG_KEEP_RUNS` (10) are kept; use `collect_diag` right after a problem. |
| Disk filling | Pictures from `--diag-save`; check `artifacts.budget_reached`, lower `GABAI_DIAG_MAX_MB`. |

## Send a package to someone (or an AI agent) for help

```bash
py -m gabai.tools.collect_diag
```

Send `data/diag/diag_<time>.zip` and one sentence: what looked wrong, and at what time. If the Pi was involved copy `~/gabai_logs` from the Pi too (`scp -r pi@<PI_IP>:~/gabai_logs .`). No camera pictures are inside unless you used `--with-images`. Tokens are redacted.

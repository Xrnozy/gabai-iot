# Debugging and diagnostics

How to turn on logs, where they go, what the events mean, and how to find why a result was wrong.

## 1. Where the logs are

| Machine | Folder | Made by |
|---|---|---|
| PC | `logs/<time>_<name>/` | viewer, receiver, live_basalt, record_basalt, replay_pipeline, selfcheck |
| Pi | `~/gabai_logs/pi_<time>/` | `python3 -m gabai.pi.sender` |

Each run folder holds `gabai.jsonl` (the log), `run.json` (command line without secrets, versions, start time), and optional `artifacts/` (pictures, only when you ask). `gabai.jsonl` is rotated at `GABAI_LOG_MAX_MB` (default 10) with `GABAI_LOG_BACKUPS` (default 3) old files. Only the newest `GABAI_LOG_KEEP_RUNS` (default 10) run folders are kept.

One line is one JSON object: `ts` (wall clock), `mono` (monotonic seconds), `run`, `comp` (component), `lvl`, `ev` (event name), then fields. `fid` is the frame id (Pi capture time in ms). The same `fid` appears in the Pi log, the PC receiver, the frame trace and the alert, so one picture can be followed through every component.

## 2. Settings

| Setting | Env var | Command line | Default |
|---|---|---|---|
| Level | `GABAI_LOG_LEVEL` | `--log-level DEBUG` | INFO |
| Per component level | `GABAI_LOG_COMPONENTS` (e.g. `depth=DEBUG,slam=DEBUG`) | | |
| Console level | `GABAI_LOG_CONSOLE` | | WARNING |
| Folder | `GABAI_LOG_DIR` | `--log-dir` | `logs/` (Pi: `~/gabai_logs`) |
| File size (MB) / backups / runs kept | `GABAI_LOG_MAX_MB`, `GABAI_LOG_BACKUPS`, `GABAI_LOG_KEEP_RUNS` | | 10 / 3 / 10 |
| Queue size (records) | `GABAI_LOG_QUEUE` | | 20000 |
| Per-frame diagnostics off | `GABAI_DIAG=0` | `--no-diag` | on |
| Save pictures | `GABAI_DIAG_SAVE=depth,frame,disp,grid` | `--diag-save` | none |
| Save a picture every N frames | `GABAI_DIAG_EVERY` | | 30 |
| Picture budget (MB) | `GABAI_DIAG_MAX_MB` | | 200 |

Levels: DEBUG adds one line per frame / per pose / raw sensor values. INFO has start-up, link state, periodic stats, decisions that are spoken, summaries. WARNING has things that may make results wrong. ERROR has failures. Repeated lines from the per-frame code (`info_every` / `warning_every`) are limited to one per period; the next line that does get through carries a `suppressed` field with the number skipped. Errors are not limited. If more interesting frame traces arrive per second than the limit allows, a `trace.rate_limited` warning gives the total skipped. The logger counts any line it had to drop because its queue was full (`log.dropped`). If you see `log.dropped`, the log is incomplete and says so.

Logging is asynchronous (a queue and a writer thread), so a slow disk does not slow the frame loop. Measured cost of the per-frame diagnostics in the viewer on this PC: about +4 ms per frame, inside the +/-3 ms run-to-run noise. Not measured on a long run.

## 3. What gets logged

- Cameras (Pi): `cam.open`, `cam.first_frame`, `cam.controls`, `cam.mode_mismatch` (asked for 640x480 MJPEG, got something else), `cam.open_failed`, `cam.reader_error`, `cam.recovered`.
- Pi link: `pi.connecting`, `pi.connected`, `pi.connect_failed`, `pi.link_down`, `pi.stats` (frames sent, bytes, dropped, queue, CPU temp, throttle flags), `pi.low_power`, `pi.imu_init_failed`, `pi.mag_init_failed`, `pi.imu_raw`/`pi.mag_raw` (DEBUG).
- PC link: `link.listening`, `link.accept`, `link.hello`, `link.up`, `link.lost`, `link.rejected` (wrong token, reason only), `link.stats` (rates, gaps, overwritten, decode errors).
- Per frame: `frame.trace` with stages (`receive`, `rectify`, `depth`, `zones`, `alert`), timings, quantities tagged by kind, `checks` (pass/fail/suspect/unknown), `decisions`, and `notes`.
- Alerts: `alert.spoken` with distance and the evidence; `alert.suppressed` with a `reason` when a decision was made not to speak (debounce, repeat rule, clip missing, audio off). Checks: `check.fail` (WARNING, rate-limited per check) and `check.suspect` (DEBUG).
- Resources: CPU, RAM, GPU temperature and use, queue sizes (GPU data comes from a background `nvidia-smi` poll).
- SLAM: `slam.start`, `slam.gate_open`, `slam.tracker_ready`, `slam.pose` (DEBUG), `slam.stats`, `slam.summary`, `slam.crash`, missing-calibration warnings.
- Replay: `replay.start`, `replay.frame_missing`, `replay.frame_error`, `replay.verdict`.

## 4. Value kinds (what a number is)

Every quantity in a trace says what it is. Do not read more into it than that.

| Kind | Meaning |
|---|---|
| `measured` | read from a sensor (e.g. IMU rate, capture time) |
| `model_prediction` | output of a neural network (disparity, any depth from it) |
| `estimated` | computed from other numbers with assumptions (e.g. floor model from camera height) |
| `stale` | was valid, now older than its limit |
| `unavailable` | not produced (sensor missing, model failed) |
| `validated` | passed a named independent check; the check is named in the record |

Checks use `pass`, `fail` (confirmed: a measured value is beyond a limit), `suspect` (a possible problem, cause not confirmed) and `unknown` (could not be evaluated). Nothing is reported as confident because a model returned a number.

## 5. Reading a run

```bash
py -m gabai.tools.diag_report
```

Prints a report for the newest run: link health, frame rate and latency, depth coverage, alert counts, check failure rates (from the regular sample only, so interesting frames do not inflate them), and **findings**. A finding is `confirmed` (with the measured evidence) or `possible` (with possible causes and next steps). Use `--json out.json` or `--md out.md` to save it. Give it a folder or a `.jsonl` to read another run.

## 6. Tracing a wrong result back to its source

A bad alert (false alarm, missed obstacle, wrong distance):

1. Press `d` in the viewer window right after it happens: writes the last frame traces to the run folder (`traces_ring_<fid>.jsonl`). Press `f` (false alarm) or `x` (missed obstacle) to also dump the evidence for that frame to `marks/`.
2. Run `diag_report` on the run. Read the findings first.
3. Open the trace for the frame (`frame.trace`, search by `fid` or time). The zone stage has the evidence:
   - `bands.<name>.n_obstacle_px`, `nearest_px` (x, y of the closest obstacle pixel), `min_depth_m`, `p5_depth_m`, `p5_horizontal_range_m`, `depth_uncertainty_m` (half a pixel of disparity error, an assumption, not a measured error)
   - `n_removed_as_floor`, `n_beyond_range` (what was not counted and why)
   - `floor_model` (`level camera`, `tilted by IMU up-vector`, or none), `pitch_down_deg`, `cam_h_m`
   - `decision_path` (the rule that fired, in words)
4. For one pixel: `gabai.diag.explain.explain_point(...)` gives the chain pixel -> disparity -> depth -> camera xyz -> world xyz, with the uncertainty at each step.
5. Wrong camera height (`--cam-h`) is a common cause of floor false alarms. Reproduce it in software: `py -m gabai.tools.replay_pipeline --sim clear_floor_wrong_height`.

## 7. Saved intermediate outputs

Off by default. `--diag-save depth,frame,disp,grid` writes JPEG/PNG into `artifacts/` of the run folder, within a size budget (`GABAI_DIAG_MAX_MB`); a `artifacts.budget_reached` event is logged when it stops. Saved pictures are camera footage: treat them as private. `collect_diag` leaves them out unless `--with-images`.

## 8. Offline replay

```bash
py -m gabai.tools.replay_pipeline --list
py -m gabai.tools.replay_pipeline --sim box_2m
py -m gabai.tools.replay_pipeline data/slam/rec_<time> --model s --cam-h 1.2 --max-frames 50 --skip 2
```

- `--sim <scenario>`: a rendered floor and boxes from a known camera pose. Perfect disparity goes through the same zone and alert code and the same diagnostics. Verdicts: `PASS`, `FALSE_ALARM`, `MISSED`, `WRONG_DISTANCE`. This tests the decision logic and the floor model. It does **not** test the depth network.
- A recording folder (EuRoC layout from `record_basalt`): rectified pairs go through the real LAS2 model and the real zone/alert chain. No audio. Writes `data/replay/replay_summary.json` and a normal run log you can feed to `diag_report`.
- A recording has no ground truth. The replay shows what the system would have said; it does not say whether it was right.

## 9. Self-check, report, package

```bash
py -m gabai.tools.selfcheck            # a few seconds
py -m gabai.tools.selfcheck --deep     # also runs the depth model once
py -m gabai.tools.selfcheck --pi 192.168.1.63
py -m gabai.tools.collect_diag         # one zip with everything needed to debug
```

Self-check lines are PASS / WARN / FAIL / SKIP; each WARN and FAIL says what it means and the command to fix it. `collect_diag` makes `data/diag/diag_<time>.zip` with the newest 3 run logs, calibration files, summaries, the self-check, a report, environment info and `MANIFEST.json` (SHA-256 of each file and what was left out). Tokens and passwords are replaced by `<redacted>`. No camera pictures unless `--with-images`.

## 10. What the logs cannot tell you

- Whether a distance is right in metres. Without a measured tape distance for the same scene there is only internal consistency (`validated` means the two views agree).
- Anything on the Pi that happened before its logger started, or while the Pi had no power (see `docs/pi-power-guide.md`: under-voltage was seen on the power bank).
- CPU temperature of the PC: Windows does not expose it to Python without extra tools; it is reported as unavailable, and the governor treats unknown as not cool.

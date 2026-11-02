# Testing

Four kinds of test. Only the first three can be run without the rig.

| Kind | Needs | What it shows | What it cannot show |
|---|---|---|---|
| Unit tests | PC only | one function or class behaves as written | that the parts work together on real data |
| Integration tests | PC only (fake Pi on localhost) | sender-receiver-viewer chain handles messages, drops, bad input | real camera, I2C, Wi-Fi, USB behaviour |
| Simulation / replay | PC only (GPU for a recording) | decision logic and floor model against known geometry; the real depth model on a recorded sequence | metric accuracy of the depth model in a new scene, real sensor noise |
| Hardware tests | the rig | everything real | (see `docs/hardware-test-checklist.md`) |

## Focused set (2026-10-28)

```
py scripts/run_tests.py --priority     # 213 tests in 25 modules, about 4 minutes: map, voxel growth, depth validity, sync, obstacle warnings
py scripts/run_tests.py --deferred     # the other 331 tests in 46 modules: hands and objects, IMU, spec phases, diagnostics plumbing, Pi tools
py scripts/run_tests.py test_kfvo_robust      # camera-only robustness (rough motion, lost pictures, loops): about 90 s, deferred
py -m gabai.tools.map_growth_check --start 20 --frames 700     # real video: does the map stay bounded (needs the GPU, about 40 s)
```

The list is `tests/priority_modules.txt`. No test was deleted; the full run (`py scripts/run_tests.py`) is unchanged.

## Run the software tests

No pytest needed.

```bash
py scripts/run_tests.py                 # every module in tests/, one process each
py scripts/run_tests.py zones grid      # only modules whose name contains one of these words
py scripts/run_tests.py -v              # also print the traceback of each failure
```

Output is one line per module: passed, failed, skipped, time. Exit code 0 only if nothing failed. A skipped test is not a pass: it is listed as skipped and says why (for example the real recording or the model weights are not on this machine). If a module crashes or hangs, only that module is reported as crashed.

Tests also run under pytest (`py -m pytest tests`) if it is installed; tests that use pytest fixtures such as `tmp_path` are skipped by `run_tests.py` and the tests in this repo avoid them.

## What the tests cover

| Area | Test modules |
|---|---|
| Logging core (queue, drop count, rotation, retention, rate limit, secrets) | `test_diag_log` |
| Checks and value kinds | `test_diag_checks`, `test_diag_sink_explain` |
| Frame trace per frame, link watcher | `test_diag_session`, `test_diag_watch` |
| Receiver counters, bad input, token not logged | `test_receiver_diag` |
| Pi sender and cameras (fake devices) | `test_pi_diag`, `test_pi_power` |
| Zone check and its evidence | `test_zones_evidence` (+ existing zone tests) |
| Alerts and "why not spoken" | `test_live_alerts` |
| Simulator, grid, governor, detector adapter | `test_sim`, `test_grid`, `test_governor`, `test_detect` |
| SLAM wrappers, calibration, still gate, trajectory findings | `test_slam_diag`, `test_slam_run_diag`, `test_still_gate`, `test_accel_cal`, `test_vit_structs`, `test_euroc_writer`, `test_check_extrinsics` |
| Self-check, report, package | `test_selfcheck`, `test_diag_report`, `test_collect_diag` |
| Replay and scenarios | `test_replay` (one test runs the real depth model on `data/slam/rec_20261026_155811`, skipped if the folder is absent) |
| Phases 3-6 (hands, grasp, cloud and voxel map, preview, relocaliser, memory, rooms, background runner) | `test_hands`, `test_grasp`, `test_cloud`, `test_preview3d`, `test_relocalize`, `test_memory`, `test_phase6` |
| Rerun pipeline (modes, diagnostics, held objects, orchestrator, alignment check) | `test_pipeline_config`, `test_pipeline_diag`, `test_held`, `test_hand3d`, `test_held3d`, `test_hand_identity`, `test_perf`, `test_navguard`, `test_pipeline_run`, `test_stereo_alignment_check`, `test_video_cloud`, `test_slam_cloud`, `test_kfvo` (keyframe odometry), `test_mapref` (map refiner, weighted map, uncertainty) |
| Failure behaviour: no depth, zero disparity, blind-then-recover, no IMU, no camera height, missing picture | `test_failure_modes` |

## Integration test with a fake Pi

`tests/fake_pi_sensors.py` plays the Pi: a textured scene seen from a rotating camera, gyro, accelerometer and compass, with chosen delays.

```bash
py tests/fake_pi_sensors.py 5600 30        # terminal 1: fake Pi sends to port 5600 for 30 s
py -m gabai.spike.demo_viewer --source live --model s --no-window --max-frames 60 --no-audio   # terminal 2
py -m gabai.tools.diag_report              # then read the report
```

The viewer connects to the fake exactly as it does to the real Pi, so link, decode, trace and report code are all exercised.

## Replay tests

```bash
py -m gabai.tools.replay_pipeline --list
py -m gabai.tools.replay_pipeline --sim box_2m
py -m gabai.tools.replay_pipeline data/slam/rec_20261026_155811 --model s --cam-h 1.2 --max-frames 30 --skip 3
```

See `docs/debugging.md` section 8. Scenario verdicts: `PASS`, `FALSE_ALARM`, `MISSED`, `WRONG_DISTANCE`. `clear_floor_wrong_height` is expected to return `FALSE_ALARM`: that is the signature of a wrong camera height, kept as a regression check that the logs show the believed height.

## Self-check before a hardware session

```bash
py -m gabai.tools.selfcheck --deep
```

## Rules for adding tests

- A test that asserts something false by mistake must fail. Assert the positive first (an obstacle is found) before asserting the absence of something else.
- Use the simulator for geometry with a known answer; do not hard-code numbers from a run.
- Do not put a real token in a test; tests check that the token is not in the log.
- Phases are not marked complete by a passing unit test alone: the status table in `docs/architecture.md` names what has been run on hardware.

## Known gaps in testing

- No test drives a real Pi. The Pi code is tested with fake devices; I2C, V4L2 and power behaviour are untested in software.
- No test measures metric depth accuracy of LAS2 in a new scene. Use the hardware procedures.
- No long-duration test (hours) of log rotation under real load or of SLAM drift.
- The replay uses recorded frames that were rectified at record time; it does not test the rectifier against the current calibration.

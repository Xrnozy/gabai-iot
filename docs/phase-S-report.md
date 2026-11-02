# Phase S report: hardware spike

Date written: 2026-10-26. Everything below comes from measurements on this rig unless marked "not measured".

## Verdict
**GO, with conditions.** The hardware can deliver stereo depth to a PC over Wi-Fi at about 15 FPS. Left/right camera sync is good enough going straight and degraded during fast turns. It is not a clean pass of the original spec (10 ms sync, depth error below 5% out to 3 m); several spec numbers are unmeasured.

## Results against the plan's checks
| Check | Result | Status |
|---|---|---|
| Camera sync (spec: 10 ms or less) | Edge method, 47 edges, fixed 30 ms exposure, 30 FPS pairing: median 3.8 ms, quartiles 2.1 and 9.9 ms, 90th percentile 17.9 ms, worst 22.4 ms; 28% of edges above 10 ms. The cameras cannot exceed 30 fps at 640x480 (MJPG 30, YUYV 25), so software cannot tighten it. | Typical passes, worst case fails. Verdict: usable straight ahead, degraded in fast turns. Mitigation built: gyro turn flag and "reduced accuracy". |
| Camera order and mounting | `/dev/video2` is the left camera and is mounted upside down (turned 180 deg on the PC); `/dev/video0` is the right. The first calibration had them swapped. | Fixed; the calibration tool now warns on swapped cameras. |
| Stereo calibration | 21 of 25 views used, baseline 5.07 cm, reprojection 0.64 px, row error 0.19 px, verdict OK. | Done. More varied views would tighten it. |
| Depth model | LAS2 on your rig, checkerboard truth, 21 pairs, 0.42 to 0.85 m: median error 0.4 to 0.5%, 90th percentile about 1.1 to 1.3%, model 22 to 55 ms (S, M, H). Colour and grayscale made no difference. | Measured close range only. |
| Depth beyond 0.85 m, glass, thin pole, dark bag, low light, drop-off | Not measured. A range test tool exists (`py -m gabai.tools.range_test`). | Open |
| Link Pi to PC | Wi-Fi, about 15 FPS received, capture to PC latency about 170 ms (uncertainty about 180 ms) after the send-buffer fix; 1520 ms before it. Tailscale not tried. | Works; latency is not capture to speech. |
| Camera controls | Auto exposure and fixed exposure both took effect. Locked white balance and focus not verified. | Partly verified |
| Pi power, load, free RAM over 10 minutes (spec: 900 MB free) | Not run as a 10-minute test. A 60 s two-camera run gave 14.4 pairs per second with no drops. Power flag showed under-voltage events under load (you chose to ignore it). RAM not measured. | Open |
| IMU (BMI160) and compass (HMC5883L) | Both answer on the check script; a live turn flag, 3D camera view, and one-recording sync of camera, gyro and compass are built. Sync results so far come from simulation only. | Hardware answers; sync unverified on real data |
| Pi 3B Wi-Fi bandwidth | Real camera frames fit at about 15 FPS at 640x480. | OK |

## Decision
Proceed to Phase V (done in this session, see phase-V-manual-test.md). Conditions before relying on any number: run the open checks above during the Phase V supervised tests, in particular range accuracy, floor/drop behaviour, and the 10-minute stability run.

# SLAM plan: Basalt stereo-inertial tracking (spec phase 1c)

Chosen by the user on 2026-10-26: build Basalt natively on Windows (first tried WSL, then switched on the user's request). Basalt is the spec's choice for tracking (`spec/Gabai Lite.md`). The Phase V and Phase M hardware tests were skipped on purpose and will be exercised through these phases, so a failed SLAM test may expose a fault that those tests would have found (power, link stalls, sync).

## Spec pass test for phase 1c
- A 30 m indoor walking loop ends within 0.5 m of the start.
- No lost tracking on normal head turns.
- Standing still for 60 s drifts under 5 cm.

No part of this is claimed until it is measured on the user's rig. The EuRoC runs below only show that the Basalt build works.

## Steps
| Step | What | Pass when | Needs the rig? |
|---|---|---|---|
| S1 | Build Basalt (`basalt_vio`) natively on Windows (clang + vcpkg) | `basalt_vio --help` runs | no. DONE 2026-10-26 |
| S2 | Run Basalt on EuRoC MH_01_easy with its shipped calibration | trajectory file is written and its error against the dataset's ground truth is reported (ATE) | no. SKIPPED by the user (download blocked, then chose to go straight to the rig) |
| S3 | Write our rig's calibration in Basalt's format: stereo from `data/stereo_calib.npz`, IMU extrinsics and camera-IMU delay from the sync tools, IMU noise values from a still recording | a file Basalt loads without error; the values are listed with where each came from | DONE for the file format and rotation (`check_extrinsics`); noise values still from the datasheet |
| S4 | Record a sequence from the Pi (frames, IMU) and convert it to the EuRoC folder format so Basalt can replay it | `basalt_vio` runs on it and prints a trajectory | DONE 2026-10-26 (`record_basalt`, two recordings); results not validated, see `docs/slam-record-guide.md` |
| S5 | Spec test: 60 s still drift, then the 30 m loop closure | numbers from the spec above | yes, and a 30 m measured loop |
| S6 | Live: feed Basalt from the receiver instead of a file, publish pose to the viewer | pose in the 3D inset follows the rig | BUILT 2026-10-26 (`gabai.tools.live_basalt`); tested on recordings only. Needs the accelerometer calibration and a live run on your rig |

## What each step can and cannot show
- S1 to S3 prove the software and the file format. They say nothing about accuracy on this rig.
- The time offset between the cameras and the IMU must be measured once and then kept fixed (fixed exposure, fixed frame rate). Basalt does not estimate it while running.
- Webcams with auto-exposure and a software sync give worse tracking than the EuRoC cameras. A poor S5 result may come from the rig, not from Basalt.
- Pi under-voltage and link stalls (see `docs/pi-power-guide.md`) will show up as lost tracking.

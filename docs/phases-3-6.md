# Phases 3 to 6: what exists, what was run, what was not

Spec: `spec/Gabai Lite.md` (phase table near the end). Written 2026-10-26. Every line below says whether it was run and on what. "Simulated" means a rendered or generated scene with a known answer. "Real video" means the two files `data/left.mp4` and `data/right.mp4` (no calibration, no IMU, no pose). Nothing was run on the rig. None of the spec's pass thresholds (3 cm hand error, 9 of 10 pick-ups, relocalise in 5 s from 5 spots on a real home map, 8 of 10 items found within 30 cm) has been measured on real data.

## Installed for this work (with the user's permission)

`mediapipe` 1.1.1, `rerun-sdk` 0.40.0 (pip); `checkpoints/hand_landmarker.task` (7.8 MB, Google MediaPipe models); `models/yolo11n-seg.pt` (6 MB, Ultralytics). The pip install also pulled `opencv-contrib-python` 5.0.0.93 next to `opencv-python`; the full test suite still passes. Gabai code never downloads models by itself.

## Phase 3: hands, 3D lift, grab and release, Rerun view

| Part | File | Status |
|---|---|---|
| Hand keypoints (21 per hand) | `gabai/hands.py` `HandTracker` | Ran on real video: on 74 of 118 sampled frames of `left.mp4` at least one hand was found. Whether the keypoints are right was not measured. |
| 3D lift with stereo depth | `lift_hand` | Simulated only. Median trusted disparity in a 5x5 window; keypoints far in depth from the rest of the hand are dropped (NaN), never guessed. |
| Open / closed | `finger_state` | Rule of thumb from fingertip to wrist over knuckle to wrist (thresholds 1.6 open, 1.2 closed, between is neither). Not a trained classifier; foreshortened hands can read wrong. |
| Grab / release rules | `gabai/grasp.py` | Simulated trajectories (9 tests): pick-up and put-down, fist passing by, open hand resting, object that stays put, hidden object, two objects, lost hand. Constants are starting values. |
| Hand and person masks removed from the map | `hand_mask`, `person_mask`, `map_points(exclude=)` | Simulated and run on real video. |
| Rerun view and video preview | `gabai/tools/preview_3d.py` | Ran end to end on 40 frames of the real videos with real MediaPipe, real YOLO and LAS2-s; wrote a 37 MB `.rrd`. Looked at by nobody yet. |
| "Boxer at 1 Hz" (3D boxes from 2D boxes) | not implemented | Objects are located by the median depth inside the 2D box (`det_depth`), as a point, not a 3D box. |
| Grab events spoken ("holding: mug") | `grasp.phrase` | Text only. Not connected to the alert queue or the speaker. |

How the grab rule works (`gabai/grasp.py`): a closed hand within radius + 6 cm of an object for 0.2 s, then moves 8 cm with it. While it moves, the object must be seen following (within 3 cm of where the hand's motion predicts) or not seen at all (hidden by the hand); if it is seen staying put the candidate is cancelled with that reason. A grab says whether the object was actually seen following. Release: hand open and slower than 0.15 m/s for 0.3 s. Lost: hand not seen for 1 s while holding; the object keeps its last position, marked unobserved.

Limits that matter:
- The preview treats the camera as fixed. With a moving head, the hand path and the fused map include the camera's motion. With the rig, pass the Basalt pose to `Preview3D.frame(pose=...)`; that path was only tested with identity poses.
- MediaPipe's left/right label assumes a mirrored image; this camera is not mirrored, so no rule uses the label.
- The video has no calibration, so f and B are assumed (`--f-frac 0.6`, `--baseline 0.065`); distances in the preview are not metric.

## Phase 4: clean points, voxel map, saved map, relocaliser

| Part | File | Status |
|---|---|---|
| Keep only trusted pixels, drop depth edges, cut past 4 m, cut moving things | `gabai/cloud.py` `map_points` | Simulated and real video. Every drop is counted. |
| Fuse many views into 5 cm voxels | `VoxelMap` | CPU, simulated. 50 noisy views of a wall: spread of the fused surface 0.027 m against 0.081 m for one view, using `points(min_obs=10, core=0.9)`. **nvblox and the Open3D CUDA grid named in the spec are not used**; speed on the rig is unknown. |
| Rolling window (flat memory on a long walk) | `VoxelMap.trim` | Simulated 100 m walk: voxel count stays flat. This is CPU memory, not the GPU memory the spec's target is about. |
| Saved map | `gabai/relocalize.py` `save_map` / `load_map` | Round trip tested. |
| Relocaliser | `relocalize` | Simulated asymmetric room, 5 start spots with known yaw and shift: pose error under 8 cm and under 2 degrees, each under 5 s. A view of a different place and a symmetric room are rejected, with the reason. Never run on a real saved map. |
| Using the correction | `apply_correction` | Math tested. Not wired into the Basalt live tool, so poses are not corrected automatically. |
| Home map built from a real walk | not done | Needs the rig, Basalt poses and a walk; no saved real map exists. |

The relocaliser accepts a result only if at least 55% of the current points match within 12 cm, the average match error is under 6 cm, and no other pose fits almost as well. Otherwise `ok` is False and the caller must keep the uncorrected pose.

## Phase 5: object memory

| Part | File | Status |
|---|---|---|
| SQLite store, sightings, place events, absent checks | `gabai/memory.py` `ObjectMemory` | Tested (11 tests). Survives reopening. An object is never deleted for not being seen; "not seen there since ..." is said instead. |
| "Where are my keys?" answer with direction, distance and age | `ObjectMemory.answer`, `heading`, `bearing` | Tested with generated poses. A camera pointing straight up or down has no ground heading and raises instead of guessing. |
| Grab / release into memory | `apply_event` | Tested with `gabai.grasp` event dicts. |
| Finding "keys" and similar items | not done | The installed YOLO model (COCO) has no "keys" class. The spec's YOLOE open-vocabulary model needs a text encoder download and was not set up. Today memory stores whatever the detector's labels are (cup, bottle, phone). |
| Voice query and speaking the answer | not done | `answer()` returns text; nothing listens or speaks it. |
| 8 of 10 items within 30 cm | not measured | |

## Phase 6: room shape in the background

| Part | File | Status |
|---|---|---|
| RANSAC planes (the spec's backup) | `gabai/rooms.py` `layout_from_planes` | Simulated room: floor, ceiling, 4 walls found, height and size within 5 cm and 15 cm; furniture is not called a wall; tilted clouds are reported as `other`, not forced. |
| SceneScript | not available | `scenescript_layout()` raises `RoomUnavailable`. The checkpoint in `models/` has no loader and the code is not installed. |
| ShapeR | not available | `shaper_mesh()` raises `RoomUnavailable`. |
| Run only when idle, never raise a lane | `gabai/background.py` `BackgroundRunner` | Tested with fake jobs: does not start while busy, stops a running job within 1 s when the rig moves or a lane exceeds 100 ms, requeues it, records failures, reports a job that ignores stop, never blocks the caller. Not connected to the live viewer. |

## Commands

```bash
py -m gabai.tools.preview_3d --left data/left.mp4 --right data/right.mp4 --start 30 --max-frames 150 --skip 3 --model s
py -m gabai.tools.preview_3d --save data/preview.rrd --no-viewer        # record, open later with: rerun data/preview.rrd
py scripts/run_tests.py hands grasp cloud preview3d relocalize memory phase6 slam_cloud
```

## Wiring that is NOT done

None of these modules is called by `demo_viewer` or the alert queue yet. The preview tool is the only place that runs hands, objects, the cloud and the map together. Connecting them to the live rig (pose from Basalt, voice output, the governor's `background_ok` into `BackgroundRunner.tick`, correction into Basalt poses) is future work and needs the rig to verify.

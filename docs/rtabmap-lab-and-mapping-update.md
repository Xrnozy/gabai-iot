# Update 2026-11: ChArUco calibration, RTAB-Map lab, GPU mapping, navigation map

State of the system after this round of work. Everything below was run on recorded data (`data/recordings/vio_20261101_014854`) or synthetic scenes; **nothing here was run live on the Pi end to end** unless a line says so.
Details and measurement tables for the lab are also in `rtabmap_lab/README.md`; the Pi KLT / sync / Kalman work is in `docs/pi-klt.md`.

## 1. Stereo calibration with a ChArUco board

Tool: `gabai/tools/charuco_calibrate.py` (tests: `tests/test_charuco_calibrate.py`).

```bash
py -m gabai.tools.charuco_calibrate --make-board                        # writes data/charuco_board_letter.png: 7x10 squares of 25 mm, markers 18 mm, letter paper, print at 100%
py -m gabai.tools.charuco_calibrate --square-mm <measured> --marker-mm <measured> --baseline-mm 42 --rotate-left 180          # live from the Pi
py -m gabai.tools.charuco_calibrate --images data/charuco_pairs --square-mm <measured> --marker-mm <measured> --baseline-mm 42 # again from saved pairs
py -m gabai.tools.charuco_calibrate --recording data/recordings/<folder> ...                                                  # from a recording
```

- Rerun shows both pictures with their capture timestamps (left = `t_capture`, right = `t_capture + dt_ms`), detected corners, the right-minus-left offset, board coverage, and after the fit both cameras, every board position, the rectified pair and the report.
- A pair is kept when both cameras see the board, it is steady, it is in a new place, and (while the board moves) the two pictures are less than `--max-skew-ms` apart. A still board is accepted at any offset.
- Output: `data/stereo_calib_charuco.npz/.json/_charuco.json` (same format as `calibrate_stereo`; the `_charuco.json` holds every kept pair's timestamps). Kept pairs are saved in `data/charuco_pairs` so a recalibration needs no recapture.
- `--flip180` exists but does not fix swapped cameras (flipping a swapped pair leaves it swapped). The pipeline detects the other camera order by itself (`CAMERA LAYOUT: ... swapped`).
- **Current calibration** (`source.calib` in `config/pipeline.toml`): 30 pairs, reprojection 0.246 px, row error 0.204 px, focal 618.9 px, **baseline 47.10 mm with `--square-mm 25`**. The case was designed for 42 mm.
  The baseline scales linearly with the square size typed in: a printed square of about 22.3 mm would give exactly 42 mm. **Distances are about 12% too long until the printed square is measured with the caliper and the calibration is rerun.** The previous file is kept as `data/stereo_calib_charuco_prev.npz`.
  If the caliper says 25 mm, the real lens spacing is about 47 mm.
- The old calibration `data/stereo_calib_new.npz` needs the opposite camera order (left `/dev/video2`); the ChArUco one was made with left `/dev/video0`, right `/dev/video2`, and records no order, so the order check is skipped.

## 2. Pi sender fixes and power

- `gabai/pi/cams.py`: the sender crashed with `RuntimeError: deque mutated during iteration` while looking up frame counters; it now iterates over a snapshot. **Copy the file to the Pi** (`scp gabai/pi/cams.py xrnozy@<pi>:~/gabai/pi/cams.py`).
- The Pi log shows `throttled=0x50005` (under-voltage and throttling now) and 1-1.6 s `pi.loop_gap` stalls: the supply is too weak. Use `--low-power --capture-fps 20`, and a 5 V supply of at least 2.5 A on a short thick cable (`docs/pi-power-guide.md`).
- The left/right capture offset was about 0.8 ms right after the startup phase alignment and 2.5 ms in the last calibration; 12 ms appeared once, after the power stalls. For mapping the offset should be near 0.

## 3. `gabai/tools/simple_slam.py`: simple live LAS2 SLAM

LAS2 depth (left-right checked) + `KeyframeVO` pose + a bounded voxel map (`--max-voxels`, default 60000; farthest voxels dropped when full) shown in Rerun. Nothing is stored unless `--save-traj` / `--save-rrd` is given.
`py -m gabai.tools.simple_slam` (live) or `--replay <folder>`. Uses `config/pipeline.toml` (calibration, flip180, LAS2 size). No loop closure on the map, no IMU. Tests: `tests/test_simple_slam.py`. Shares port 5600 with everything else: run one program at a time.

## 4. RTAB-Map lab (`rtabmap_lab/`)

Stereo-only RTAB-Map (C++ bridge over a local socket) with its poses feeding the map builders, shown in Rerun. Run: `py -m rtabmap_lab live --depth ffs` or `replay --folder <recording>`. `py -m rtabmap_lab stop` kills a stale bridge.

| Option | Meaning |
|---|---|
| `--depth rtabmap/gabai/ffs/fused` | depth for the map: RTAB-Map's own SGBM; LAS2 + 2 monocular; Fast-FoundationStereo alone (**recommended**, best measured); FFS + LAS2 + RTAB-Map + monocular fill with confidence |
| `--tsdf-backend gpu/open3d` | nvblox-style GPU TSDF (default) or the original Open3D volume |
| `--gaussians`, `--gs-max`, `--gs-iters` | incremental Gaussian map (see 4.3); off by default |
| `--no-tsdf`, `--no-surfels`, `--no-nav` | switch a map off |
| `--depth-wait-pose S` | run the depth models only for nodes the graph has a pose for (default 8 s) |
| `--orientation auto/as-is/flip` | accelerometer judges once at start whether pictures are upright |
| `--dump-nodes f.npz` | save every node's depth, picture, pose for the offline tools |

### 4.1 Depth sources and fusion
`depthfusion.py` / `fused_source.py`: inverse-variance voting between stereo sources, monocular depth used only for arbitration and for filling holes (class 3, confidence at most 0.3, never over a measurement). Measured: **FFS alone beat the fusion** (plane thickness 21.7 mm vs 23.4 mm), hence the recommendation. Depth Anything is only in `--depth fused/gabai`.

### 4.2 TSDF backends
- Open3D `tsdf.py`: rebuilt from stored depth when a loop closure moves poses.
- `gpu_tsdf.py` (after nvblox, in PyTorch, native Windows): block-sparse voxel pool, projective integration, free-space carving, weight-gated mesh, noise-scaled truncation/weight, flying-pixel and grazing handling, small-piece cleanup.
  Result on the recording: floating debris 10-11% -> 0% of the mesh area; **not flatter** (20-22 mm vs 16-17 mm thick planes) and **not faster** (integrate 11.7 vs 5.6 ms per node, mesh 78 vs 40 ms). nvblox itself is not used (RGB-D library, no documented Windows support, takes poses from outside).
- Tools: `tools/surface_lab.py` (`--gpu`, `--gs`, `--fusion`, `--calib`), `tools/mesh_quality.py` (debris share), `tools/pose_check.py`.

### 4.3 Incremental Gaussian map (`gsmap.py`)
Depth-supervised Gaussians placed only where the map does not already explain the measured depth, refined by a few optimiser steps per node, pruned when transparent/oversized/never confirmed, moved with their keyframe on loop closures. gsplat compiles on this PC only through `gsplat_win.py` (vcvars, `WIN32_LEAN_AND_MEAN`, skipped CUDA 12.8/13.1 check, MSVC flags; first start about 3 minutes).
**Measured and judged not good enough as the default:** planes 25 mm thick, 1.1% isolated specks, 80-220 ms per node, and in the live view (user screenshot) it shows fans of floating points and runs slowly. It stays opt-in. Not tied to the user's request to make it the default: that request was superseded by "the Gaussian splat is bad and slow, I want faster 3D reconstruction".

### 4.4 Poses are the limit
`tools/pose_check.py`: overlapping nodes disagree by about 5.5% of depth against 0.7% expected from depth noise alone (RTAB-Map poses beat a static pose, 5.7% vs 8.0%, but are not exact). This, not the TSDF or depth model, causes the doubled walls. Not fixed. Next step: align each node's depth to the volume (depth ICP) before integrating; also correct the baseline scale (section 1).

### 4.5 Speed (recorded run, 640x360, RTX 5060 Ti)
Depth models per node: FFS 76 ms, LAS2 about 40 ms, two monocular maps about 26 ms (the dominant cost). RTAB-Map announces about 3 nodes for each one that gets a pose (44 of 115): `--depth-wait-pose` cut model runs from 117 to 37 nodes and the 120-frame replay from 84 s to 66 s with a similar mesh.
A timing run to find what else dominates the live frame loop was interrupted before it ran; **the "faster reconstruction" request is therefore still open** (next: time the replay with surfels / TSDF / nav switched off one by one).

### 4.6 Egocentric navigation map (`navmap.py`, Rerun tab "Navigation")
Rolling 6x6 m grid (`gabai/grid.py` log-odds) levelled on a floor plane fitted per picture (the odometry world has no gravity), distance field, traversable map (unknown ground never counts as safe, the blind strip in front of the camera is crossable at high cost and reported), least-cost route and steering angle. 5.8 ms median per update, CPU. Validated on a synthetic tilted room (`tests/test_navmap.py`); on the recording (a close board) no route was found. It updates at node rate (about 2 per second); a walker needs camera rate (feed it per-picture LAS2 depth + odometry pose): not built.

## 5. Research notes (no code)
- SimpleRecon / DoubleTake / MVSAnywhere (Niantic): multi-view depth that needs accurate poses; non-commercial licences; the transferable idea is rendering the current map into the new view as a depth hint. SPZ (MIT) only matters if splats are kept; PlaceIt3D needs ScanNet-style scenes and a CUDA 11.8 stack and is a last, offline step; ACE-Zero/MicKey/map-free-reloc compete with RTAB-Map's loop closure.
- The "Stereo-GS with FoundationStereo + DROID-SLAM (Electronics 2025)" paper could not be found; no code was built to it.

## 6. Tests
`tests/test_charuco_calibrate.py`, `test_simple_slam.py`, `test_gpu_tsdf.py`, `test_gsmap.py`, `test_navmap.py`, `test_rtabmap_lab.py` (including the pose-wait upgrader), `test_depthfusion.py`. Known unrelated failures: `tests/test_loop_closure.py` (borderline), missing `gabai.refframe` modules, an occasional `test_receiver_diag` flake.

## 7. Open items
1. Measure the printed square, rerun the calibration (baseline 47.1 mm vs 42 mm).
2. Pose refinement against the volume (the doubled walls).
3. Faster reconstruction: profile the live loop, then cut (models per node are the main cost).
4. Per-picture navigation feed; live test of the navigation map on a corridor.
5. Power supply for the Pi; copy `cams.py` to the Pi.

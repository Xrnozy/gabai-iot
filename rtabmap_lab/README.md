# rtabmap_lab: stereo-only RTAB-Map, on its own

A separate experiment. It does not touch the GabAi pipeline, its config or its modules (it only imports the Pi receiver, the calibration and the rectification from `gabai`, read-only).
RTAB-Map does the odometry, the stereo depth, the loop closure and the persistent map; Open3D fuses its keyframes into a TSDF surface; Rerun shows everything. No IMU, no KeyframeVO, no Depth Anything, no LAS2 are used by it.

```bash
py -m rtabmap_lab setup                    # once: headers + import libs + build rtabmap_bridge.exe against the installed RTAB-Map 0.23
py -m rtabmap_lab live                     # the Pi's stream (port 5600; the GabAi pipeline uses the same port, run one at a time)
py -m rtabmap_lab replay --folder data/recordings/vio_20261101_014854
py -m rtabmap_lab live --db rtabmap_lab/maps/office.db            # continue a saved map (drawn again at start)
py -m rtabmap_lab live --db rtabmap_lab/maps/office.db --localize     # only find the camera in it
py -m rtabmap_lab export rtabmap_lab/maps/office.db                # RTAB-Map's own exporter: .ply cloud + poses
py -m rtabmap_lab stop                                              # kill a bridge left running
```
Pi side: the plain sender, nothing extra, no `--klt`. Leave its IMU on: it is read once at the start by the orientation check (below) and by nothing else.

## How it is built
The installed RTAB-Map has no Python bindings, so `rtabmap_lab/bridge/main.cpp` is a small C++ program linked against the installed `rtabmap_core` and driven over a local socket by `bridge_client.py`.
It compiles with the OpenCV 4.7.0 headers (the version inside the installed RTAB-Map) and Eigen 3.4.0 that `setup` downloads into `deps/` (about 100 MB), import libraries made from RTAB-Map's own OpenCV DLLs, and empty stand-ins for the PCL headers (nothing of PCL is used).
Per frame: odometry (`OdometryF2M` on the stereo pair, RTAB-Map's own feature + stereo matching), `Rtabmap::process` (loop closure by appearance and by proximity, graph optimisation, the database),
RTAB-Map's stereo disparity for the display and for each new node's points. The Pi does no extra work; everything runs on the PC.

## Coordinates (why the map is not upside down)
RTAB-Map's base frame is x forward, y left, z up; the camera is the base frame turned by `CameraModel::opticalRotation()`. A level, upright camera at the start gives a z-up map. There is no gravity sensor in this experiment,
so an UPSIDE-DOWN PICTURE GIVES AN UPSIDE-DOWN MAP. At the start `--orientation auto` (default) uses the accelerometer once, as an outside judge, to check which way up the pictures are (needs `data/imu_sync_corrected.json`),
and turns them 180 deg if needed (`--orientation as-is` / `flip` override it). On `data/recordings/vio_20261101_014854` and `vio_20261101_014427` the pictures as the pipeline delivers them are upright (3 to 6 deg) and a 180 deg turn makes them upside down (175 deg).
`py -m rtabmap_lab.tools.check_upright <recording>` prints that check. If the verdict contradicts what you see in the camera panel, trust your eyes and set `--orientation`.
`tests/test_rtabmap_lab.py` checks the transform: a point above the camera is above in the world, ahead is +x.

## TSDF surface (Open3D)
`rtabmap_lab/tsdf.py`: every node RTAB-Map adds comes with its dense depth (from RTAB-Map's own stereo matching) and colour; Open3D's `ScalableTSDFVolume` integrates them with the node's OPTIMISED pose
(voxel 0.02 m, truncation 0.08 m, `--tsdf-voxel` / `--tsdf-trunc`) and extracts a coloured mesh every ~2 s. A TSDF cannot take a frame back out, so when the graph moves (loop closure) the volume is built again
from the stored depth and colour of all nodes with the new poses, on a worker thread (the frame loop never waits; the mesh shown is the last finished one). Nodes that moved less than 2 cm / 1 deg do not trigger it.
At the end the mesh is written next to the database as `<name>_tsdf.ply`; loading a saved map rebuilds the surface from its nodes. `--no-tsdf` turns it off.
Tested on synthetic planes (a ceiling 1.5 m above comes out at z = +1.5, a wall 3 m ahead at x = +3, a 15 cm pose change rebuilds the volume, a 4 mm one does not) and on your recording (43k-vertex mesh, 5 rebuilds after loop closures, extent 5.6 x 7.3 x 4.1 m).

## Rerun dashboard
Left/right pictures, RTAB-Map's disparity, the camera and its path, the optimised node graph, loop closures (red lines, plus a line in the events log), the accumulated point map (laid again from the optimised poses after every loop closure),
odometry features / inliers / lost, map nodes and loop count, milliseconds per stage, a status panel.

## If the map looks stretched or smeared
Two causes were found and handled, and `py -m rtabmap_lab.tools.diagnose <recording> --skip 3` measures both without a picture (lost frames, restarts, path, depth validity):
- Depth noise: block matching at a 42 mm baseline has an error of about 1 px, which is +-1.4 m at 6 m. Raw points then form long streaks along the viewing rays. The bridge now drops speckles, pixels that disagree with their 5x5 median,
  and pixels whose expected noise z^2 * 0.3 px / (f * B) is above `--max-depth-std` (0.08 m, about 2.6 m with this rig); `--max-range` is 4 m. With this baseline the map is a close-range one by physics, not by choice.
- A bridge that cannot keep up skips pairs, and stereo odometry loses tracking over big steps. The map update now runs on its own thread (odometry alone takes about 42 ms a pair, 150 frames in 10 s instead of 20), and
  `Vis/MinInliers 12` / `Vis/MaxFeatures 1500` cut lost frames on the fast recording from 84 to 38 of 150 when every second pair was skipped.
The camera calibration is the main pipeline's (`config/pipeline.toml` -> `data/stereo_calib_new.npz`: baseline 41.72 mm, f 618 px); the start-up line prints it.

## Stereo depth quality
RTAB-Map's default dense matcher is block matching (`Stereo/DenseStrategy 0`), which gives scattered blobs on dim, blurred or low-texture pictures. The experiment uses its other matcher, `cv::StereoSGBM` (`Stereo/DenseStrategy 1`, tuned in
`config/rtabmap_stereo.ini`): on your recording 33% of pixels get a depth against 13%, and the disparity follows the monitor, wall and floor. Both run RTAB-Map's own code on the pictures rectified with your calibration (`data/stereo_calib_new.npz`);
the Pi's pictures are not touched. Compare matchers yourself: `py -m rtabmap_lab.tools.diagnose <recording> --dump out.png --arg Stereo/DenseStrategy 0` (then 1) writes the left picture next to the disparity. The odometry does not use this dense disparity
(it matches features), so this only changes the depth, the point map and the TSDF.

## Depth from the main system (`--depth gabai`)
`py -m rtabmap_lab live --depth gabai`: the TSDF and the point map use the main GabAi depth instead of RTAB-Map's stereo matching: LAS2 stereo on the rectified pair, Depth Anything V2 on the left AND the right picture,
fused in the main pipeline's `fill` mode (stereo stays the only source of measured pixels; the two monocular maps, scaled to the stereo pixels, fill the holes where they agree), with the same speckle removal, depth
correction, fill reach and depth-edge mask. It is the main pipeline's `default_depth`, used read-only (`rtabmap_lab/gabai_depth.py`). RTAB-Map still does the odometry, loop closure and graph optimisation from the stereo
pictures (it matches features, it does not use this depth), so the poses are the same either way. The depth is computed on the GPU in a worker thread, only for the pairs RTAB-Map turns into nodes; a node whose pair is gone or
that arrives while the worker is behind takes RTAB-Map's own depth at once, so no node is left out of the surface. `Stereo/DenseStrategy` SGBM stays the default (no GPU or models needed).
On your recording (120 frames): SGBM 34k mesh vertices, GabAi depth 89k, with a wider spread (2.8 x 4.2 x 2.7 m against 1.7 x 2.9 x 1.5 m). More coverage is not more accuracy: the monocular maps invent plausible depth in the holes,
and the TSDF here weighs it like stereo (the main pipeline down-weights it to 0.3). Judge it in the 3D view and in the `GabAi depth` panel; there is no ground truth.

## Flat walls: surfels, not just a TSDF
A TSDF averages everything it is given, so noise and small pose disagreements become thick, wavy walls. The main 3D view is now a surfel map (`rtabmap_lab/surfel.py`): every new depth picture is checked against the map before it is
believed (same depth and same normal as an existing surfel: it updates it; a confirmed surface in front of it: rejected; a surface it clearly sees through: that surfel is removed), and only surfels at least `--min-views` (2) pictures
agree on are drawn, so the fan of points along a viewing ray that only one picture saw never appears. Then the big planes (walls, floor, ceiling) of the confirmed surfels are snapped flat (`rtabmap_lab/planes.py`, `--no-flatten` to
switch off): points already within 2.5 cm of a plane move onto it, nothing else moves. The TSDF mesh stays in the other 3D view; `--raw-points` brings back the raw cloud.
`py -m rtabmap_lab.tools.surface_lab map.db` rebuilds a saved map with each method and measures flatness (RANSAC planes at 2 cm; "planar" = share of points on a plane, thickness = RMS distance from it). Your recording, 45 nodes:

| surface | points | planar | plane thickness |
|---|---|---|---|
| one depth picture alone (no poses) | | 76% | 19 mm |
| raw stereo points of all nodes | 143k | 70% | 25 mm |
| TSDF 2 cm / 8 cm (what was shown) | 38k | 48% | 27 mm |
| TSDF 3 cm / 12 cm | 16k | 41% | 28 mm |
| surfels, no check | 120k | 81% | 23 mm |
| **surfels, confirmed by 2+ pictures** | 26k | **89%** | **17 mm** |
| surfels confirmed by 3+ | 7k | 89% | 15 mm |
| Poisson mesh of those surfels | 15k | 66% | 24 mm |

The TSDF was the flattest-looking only to the eye: by measurement it is no flatter than the raw points, and larger voxels made it worse. Depth noise per picture (19 mm) is the floor; fusion across nodes adds a little from pose disagreement, and
the surfel check takes that back. Feeding the TSDF only the depth the surfels confirm was tried and left out: it keeps too little (a few thousand vertices).

## Limits
- Not run on the live Pi here: tested on your recorded sessions (150 frames: tracked throughout, 50 nodes, many loop closures, a 36 MB database saved and loaded again). Accuracy against a ground truth has not been measured: there is none.
- Stereo-only odometry loses tracking in blur and textureless views; the bridge restarts it at its last pose after `--lost-reset` lost frames (default 15) and says so.
- RTAB-Map 0.23's dense stereo matching is block matching (StereoBM): on a textureless desk or wall only a few percent of pixels get a depth (9% in the first node of the recording), so the TSDF is patchy there and fills in as the camera moves. The surface is only as good as that depth.
- The map is a point cloud from stereo disparity (strided, voxel-thinned for the display); `export` gives RTAB-Map's own cloud.
- `rtabmap_lab/deps/dl` holds the two downloaded zips and can be deleted after `setup`.

## nvblox-style GPU TSDF (`--tsdf-backend gpu`, the default; `open3d` = the original CPU volume)

[nvblox](https://github.com/nvidia-isaac/nvblox) is a C++/CUDA library for RGB-D robots (Docker/Bazel/CMake, ROS 2; its docs do not list Windows), and it takes poses from outside, so it is not linked in. `gpu_tsdf.py` ports its mapping
techniques to PyTorch: block-sparse voxel pool on the GPU, projective integration, weighted running average with a maximum weight, free-space carving, a minimum weight before a voxel is meshed. Stereo-specific changes: truncation and
weight follow the stereo noise (sigma_z = z^2 * 0.3 px / (f*B)); pixels next to a depth jump larger than the noise explains are dropped; grazing pixels count less; small disconnected mesh pieces are removed.
Not ported: ESDF (nothing here plans a path), dynamic layers, occupancy decay.

Measured on three live runs of `vio_20261101_014854` (`surface_lab --npz --gpu --calib data/stereo_calib_new.npz`, same nodes and poses for both columns; `tools/mesh_quality.py`):

| | Open3D volume | GPU volume |
|---|---|---|
| mesh area | 4.63-4.71 m2 | 4.40-4.61 m2 |
| separate pieces | 169-222 | 3-4 |
| area in pieces under 5 cm2 (floating debris) | 10-11% | 0% |
| plane thickness (flatness.report, 4 planes, noisy: +-3 mm) | 16-17 mm | 20-22 mm |

The GPU volume removes the debris and keeps the far walls but is not flatter. The remaining doubled walls come from the poses: overlapping nodes disagree by about 5.5% of depth against 0.7% expected
(`tools/pose_check.py`). Pose refinement against the volume (depth ICP) is the next step; nothing here fixes it.
Compare more settings: `py -m rtabmap_lab.tools.surface_lab nodes.npz --npz --gpu --calib <the calibration of that run> --out folder`, then `py -m rtabmap_lab.tools.mesh_quality folder`.

## Speed, and the egocentric navigation map

Measured on the recorded run (44 posed nodes, 640x360, RTX 5060 Ti):

| stage | cost |
|---|---|
| Fast-FoundationStereo / LAS2 / two monocular maps, per node | 76 ms / about 40 ms / about 26 ms: the depth models are the expensive part |
| TSDF integrate, Open3D CPU / GPU volume | 5.6 ms / 11.7 ms per node (the GPU volume is NOT faster here; its gain is the cleaner mesh) |
| TSDF mesh, Open3D / GPU volume | 40 ms / 78 ms (the GPU one grows with the size of the mapped area) |
| `NavMap.update` (grid, floor, distance field, route) | 5.8 ms median, 16 ms worst, CPU |

`--depth-wait-pose S` (default 8): RTAB-Map announces about 3 nodes for every one that ends up with a pose in its graph (44 of 115 on this run). The depth models now wait until the graph holds the node
(`DepthUpgrader.posed`), so 37 instead of 117 nodes went through them, with a mesh of about the same size (9.9k against 10.6k vertices). `0` restores the old behaviour.

`navmap.py` (on by default, `--no-nav` to skip, `--nav-cell` for the cell size) turns the same depth + pose into what obstacle avoidance needs, without waiting for the TSDF or a mesh:
a rolling 6 x 6 m grid levelled on the floor plane found in each picture (the odometry world has no gravity), a distance field (metres to the nearest obstacle), a traversable map (seen floor, at least 0.30 m
from an obstacle; unknown ground is never counted as safe), and a least-cost route to a goal or straight ahead, with the steering angle. Rerun tab "Navigation". It crosses the blind strip in front of the
camera (a camera 1.3 m up pitched 15 deg down first sees floor 1.9 m ahead) at a high cost and reports how much of the route is over ground not yet seen.
Offline: `py -m rtabmap_lab.tools.nav_replay nodes.npz --calib <the run's calibration> --out nav.png`.
It runs at node rate (about 2 per second at most). A walker needs the grid at camera rate: feed `NavMap.update` from a per-picture depth (LAS2 alone, 40 ms) and the odometry pose, bypassing the nodes.

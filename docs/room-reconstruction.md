# Why the room does not look like a room, and what changed

Written 2026-11-03. Recording: `data/recordings/walk_20261102_230115` (860 pairs), calibration `data/stereo_calib_charuco_45.npz`, RTAB-Map run exactly as the calibration lab runs it.
No tape-measured distance exists for this rig, so "error" here is disagreement with measurements the method did not use, not distance from the truth.

## Where the distortion comes from (measured, in order of size)

| cause | how it was measured | size | fixed? |
|---|---|---|---|
| **Heading lost when odometry restarts.** The rig loses tracking while turning (blur, blank walls); after 15 lost pairs the odometry restarted at the last good pose, so every degree turned in between was lost. Walls seen before and after land at different angles: the "same wall twice" and the fanned-out slabs. | rotation between consecutive nodes from the poses vs from the recorded gyro (which agrees with the odometry to 0.9 deg median where tracking is good) | pairs 3 -> 87: pose 4 deg, gyro 40 deg. Pairs 520 -> 539: 2.5 vs 34 deg. Pairs 267 -> 285: 3 vs 28 deg | **yes**: the bridge now turns the restart pose by the gyro. Worst error 36 -> 13 deg, lost pairs 97 -> 84 |
| **Noisy depth from RTAB-Map's own matcher on this scene.** Dim, blurred pictures, blank walls, a dark monitor: 46% of pixels have depth, with horizontal streaks. | scatter of each picture's depth around its own largest plane, in disparity | 0.97 px median (twice the 0.5 px of the bench), 5.4 px at the 90th percentile | no (needs better pictures or another matcher: see below) |
| **Depth maps of neighbouring nodes disagree, and more so the further the view turns.** | each node's depth projected into the nodes 1-5 later, compared in disparity (only smooth, textured parts of both) | 1.3 px on smooth textured surfaces (0.7 expected from the noise); 1.65 px at 3-8 deg of turn, 2.5 px at 8-20, 3.5 px past 20 | partly (the heading fix); the rest is not explained |
| **Global drift over the walk.** | TSDF of the 6, 20 and all nodes nearest in time, rendered into a held-out node | 1.9 / 2.4 / 4.0 px median | partly (heading fix) |

Ruled out by measurement (each changed the disagreement by less than 0.03 px):
- a constant disparity offset (the yaw / principal-point error the automatic calibration cannot see): tried -1 to +1 px, no minimum;
- the left/right exposure offset while turning (1.1 ms x the gyro rate, 0.3 px median): correcting it per pair, 1.89 -> 1.87 px;
- per-pixel noise: replacing every flat pixel by its picture's own fitted plane before the TSDF changed nothing (3.82 -> 3.83 px, roughness 6.8 mm both). **The waviness is not single-picture noise; it is pictures that do not agree with each other.** Smoothing cannot fix that, and would only hide it.

## Methods compared on identical input

`py -m rtabmap_lab.tools.recon_compare <recording> --calib <file> --out <folder> --view` runs RTAB-Map once, keeps every node (depth, picture, pose) in `<folder>/nodes.npz`, builds every method from the same nodes and scores each one against every 5th node, which it did not use:
**error** = median disparity difference between the surface rendered into that node's view and what that node measured (px, lower is better), **walls** = the same on that node's large flat parts, **coverage** = share of its measured pixels the surface explains.
A surface that is smooth but in the wrong place scores worse; one that keeps noise also scores worse.

Run of 3 Nov, gyro bridge on; left = before the rolling-shutter correction (`data/recon/walk_20261102`), right = after (`data/recon/walk_20261102_rs`). Separate RTAB-Map runs, so the held-out nodes differ; the 90th percentile is not comparable between them (the corrected run covers more of each picture, including its hardest parts).

| method | error px | walls px | coverage | roughness mm | seconds | MB |
|---|---|---|---|---|---|---|
| points | 15.89 -> **15.89** | 14.93 -> **15.53** | 0.99 -> **0.99** | 4.6 -> **3.7** | 0.9 | 16.7 |
| tsdf | 4.20 -> **2.86** | 2.99 -> **1.86** | 0.73 -> **0.87** | 7.1 -> **6.4** | 3.5 | 9.4 |
| tsdf+frameplanes | 4.18 | 2.97 | 0.76 | 7.1 | 10.4 | 9.6 |
| surfel | 4.27 -> **4.75** | 2.83 -> **2.47** | 0.70 -> **0.85** | 4.7 -> **3.9** | 5.2 | 3.7 |
| occupancy | 6.85 | 5.90 | 0.84 | 3.0 | 5.2 | 2.3 |
| poisson | 16.74 | 15.09 | 0.99 | 3.6 | 15.9 | 83.4 |
| planes | 5.76 -> **5.17** | 4.76 -> **3.01** | 0.90 -> **0.91** | 2.5 -> **3.3** | 18.2 | 7.7 |
| hybrid | 5.83 | 4.80 | 0.96 | 4.8 | 21.9 | 12.8 |
| hybrid+snap | 5.82 | 4.78 | 0.96 | 4.5 | 25.0 | 12.8 |

## What the plane method is (`rtabmap_lab/planar_recon.py`)

1. Per picture, planes are found in (u, v, disparity), where a flat surface is exactly a plane and the noise is the same for every pixel (sequential RANSAC, largest connected piece, least squares). Then every pixel is given to the plane it fits best and each plane is refitted: otherwise the first plane found takes the neighbouring wall's pixels along their shared corner and tilts (synthetic floor: 3.9 cm off before, under 2 cm after).
2. The pictures' planes join global planes by normal (8 deg) and offset (5 cm); a global plane is the inverse-variance fit of all its points.
3. Every pixel of every picture is then tested against the global planes (within 2 sigma_z, inside the plane's measured extent); a plane's surface is its 2 cm cells seen by 2+ pictures. Its outline is the plane boundary.
4. Hybrid: pixels on no plane go to the GPU TSDF; residual triangles lying on a plane are dropped (no wall twice). Optional Manhattan snap (planes within 4 deg of the room's main axes are turned onto them).

On a synthetic room with 0.5 px of noise (`tests/test_planar_recon.py`) every wall is found within 2 cm and the room width within 2 cm; the surfaces lie on the walls (99% of vertices within 2.5 cm).

Known limits: beyond about 2 m two walls meeting at a corner differ by less than 1 px of disparity near the corner and one picture cannot separate them; a floor seen only as a thin strip at the picture's edge is weakly determined; the planes are only as consistent as the poses.

## Decision

**Keep the TSDF.** On identical input it is the closest to held-out pictures (walls 3.0 px, 1.9 px after the rolling-shutter correction); surfels are close on walls but cover less; planes and the hybrid look flatter (roughness 2.5 mm) but sit further from what the cameras measured, because the depth they fit was biased (see `docs/stereo-depth-diagnosis.md`: opposite rolling shutters). Poisson and raw points hide the surface behind noise; occupancy voxels look clean but are 6-7 px off.
`planar_recon` stays available in the comparison viewer; it is not wired into the live map.

## What changed

- `rtabmap_lab/bridge/main.cpp` (rebuilt): a FRAME may carry the gyro rotation since the previous frame; the restart after lost odometry is turned by it. Without it the bridge behaves as before.
- `rtabmap_lab/bridge_client.py`: `send_frame(..., gyro=None)`.
- `rtabmap_lab/orientation.py`: `gyro_rotvec` (IMU samples -> camera rotation, flip-aware). `rtabmap_lab/live.py` sends it (live and replay, when the frames carry IMU rows); the calibration lab sends it from the recording's `imu.csv` (`Recording.gyro_rotvec`).
- `rtabmap_lab/planar_recon.py`: plane detection, global planes, plane surfaces with boundaries, hybrid, per-picture plane denoising.
- `rtabmap_lab/tools/recon_compare.py` + `recon_view.html`: the comparison and the viewer.
- Tests: `tests/test_planar_recon.py` (4), `tests/test_calib_lab.py::test_gyro_keeps_the_heading_when_odometry_is_lost`.

## What would help most next (not done)

1. **Better pictures** (the largest remaining cause): light, `--exposure fast`, a stable 5 V supply. Blank walls give no stereo depth whatever the reconstruction.
2. **Use the gyro inside the odometry**, not only at restarts (RTAB-Map can take IMU data): fewer lost pairs in the first place.
3. **Tape-measured walls** to turn these consistency numbers into real error.

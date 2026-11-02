# Camera-only visual SLAM under fast and rough motion: research and what was built (2026-10-28)

Scope: software techniques that keep visual tracking accurate with the camera alone (no IMU), and how the code is arranged so an IMU can be added later.
The search was shallow: a few web searches and the pages they returned, several of them secondary sources (preprints, summaries, slides). I name the
source for each claim and say where I am relying on general knowledge. The papers themselves (ORB-SLAM, ORB-SLAM3, Basalt, DSO) were not read in full.

## 1. What the literature says

**Constant-velocity prediction and guided search.** ORB-SLAM-style trackers predict the new pose from the last motion, project the map points with it, and
search for each feature in a small window around the projection; the matches then feed a motion-only bundle adjustment that refines the pose
([ORB-SLAM, arXiv 1502.00956](https://arxiv.org/pdf/1502.00956); described for ORB-SLAM3 in [arXiv 2607.06023](https://arxiv.org/pdf/2607.06023)).
This is the main reason such trackers do not lose a feature that moved far between frames.

**Recovery after loss.** ORB-SLAM3 has states for a recently lost track (still using the motion model) and a lost track (relocalise against the stored
keyframes) ([ORB-SLAM3 overview](https://www.mintlify.com/UZ-SLAMLab/ORB_SLAM3/concepts/slam-overview)). Recovery is by matching features of the current
picture against a keyframe database, which does not depend on the previous frame at all.

**Fast motion and blur.** Several sources agree that fast motion with rotation is the weak point of camera-only systems. A semi-direct odometry paper reports
that LSD-SLAM, ORB-SLAM and S-PTAM "often perform poorly at fast motions combined with rotations" and tend to lose track temporarily
([arXiv 1810.07768](https://arxiv.org/pdf/1810.07768)). An ECCV 2008 keyframe SLAM paper argues that edges are more resilient to blur than corner points
([Klein and Murray](https://cs.gmu.edu/~kosecka/eccv08/eccv08_klein.pdf)). Pure rotation gives no parallax, so a monocular system cannot triangulate during
it (a forum thread, anecdotal); a stereo camera does not have this problem because depth comes from the two cameras at one instant.

**The honest limit.** A 2026 gaussian-splatting SLAM paper reports that its camera-only (monocular) mode fails on all but slow sequences and that gyroscope
data alone gives great results on most of them ([arXiv 2603.20443](https://arxiv.org/pdf/2603.20443)). A 2025 paper estimates camera velocity from the blur in a
single image, which shows blur carries motion information but is not a replacement for a gyro ([ICCV 2025](https://arxiv.org/html/2503.17358v3)). Reading these
together: software can make camera-only tracking much more tolerant, but under real blur the picture stops containing the information, and only another
sensor supplies it. That is why the seam for an IMU matters (section 4).

**Drift and loop closure.** Basalt runs KLT optical flow in a fixed-lag smoother (with IMU terms) and a mapping layer that adds keyframe bundle adjustment
with ORB features for loop detection across large time gaps ([Usenko et al., arXiv 1904.06504](https://arxiv.org/pdf/1904.06504)). ORB-SLAM-style systems
detect a revisit by appearance (a bag of words over ORB features), verify it geometrically, and spread the correction over the whole path with a pose graph.
(The pose graph is general knowledge of those systems; the search results did not confirm it for Basalt.)

## 2. What is in the code now (camera only)

| Technique | Source of the idea | Here | Measured effect (rendered scenes, section 3) |
|---|---|---|---|
| Keyframes, track against the keyframe, not the last frame | sliding keyframe windows (Basalt, ORB-SLAM) | `KeyframeVO` (earlier work) | error grows per keyframe, not per frame |
| Constant-velocity prediction, search starts at the projection | ORB-SLAM guided search | `_predict`, `_project`, LK started at the projection; falls back to the previous positions if the prediction finds too few agreeing features | rotation error under rough motion 1.56 to 1.09 degrees rms |
| Motion-only refinement of the pose | ORB-SLAM motion-only BA | `cv2.solvePnPRefineLM` on the RANSAC inliers | small: 1.15 to 1.09 degrees rms |
| Depth averaged over the frames that see a feature | filtering of landmark depth | `_fuse_depth` (earlier work) | position error 4.2 to 1.2 cm on a wall |
| Relocalisation after lost tracking | ORB-SLAM3 relocalisation | ORB descriptors with 3D stored per keyframe; matched against the last 6 keyframes, PnP, LM; tried after 2 lost frames in a row, or at once if the keyframe has no features left | after 5 lost pictures: final rotation error 21 degrees without, 1.2 degrees with |
| Reject a pose that jumps in one frame | standard sanity gate | `max_step_m`, `max_step_deg` | prevents one wrong frame from moving the map |
| Do not crash on degenerate point sets | robustness | `_pnp` turns an OpenCV exception into a failed pose | a pipeline test hit this and crashed before the fix |
| Loop detection with drift measurement | ORB-SLAM, Basalt | new keyframes checked against old ones near where the camera is believed to be (`loop_radius_m`) and at least `loop_min_path_m` of travel earlier; verified by PnP; the drift is reported | measured an injected 6 cm drift as 6.0 to 6.7 cm on the first matches |
| Apply the loop correction | | `loop_apply`: the current pose is moved to the loop pose, forward only | final error 5.9 cm to 0.1 cm on the rendered out-and-back; OFF by default |

**Not done, and why it matters.**
- No bundle adjustment over several keyframes and no pose graph: a loop correction changes the pose from now on, and the path and map built before it keep their
  drift. The map therefore holds the old, drifted copy of the loop and new geometry consistent with the first visit.
- No appearance-based place recognition: a loop is found only when the drift is smaller than `loop_radius_m` (1 m), because candidates are chosen by position.
- No edge or direct (photometric) tracking, no rolling-shutter model, no deblurring. Heavy blur still removes features.
- Loop measurements are noisy: on the rendered scene the same 6 cm drift was measured between 3.9 and 11.5 cm by different keyframes. That is why
  `loop_apply` is off by default; look at the reported drift on your own footage first (events in Rerun: "revisit of keyframe ...").
- Nothing here claims real-world accuracy. All accuracy numbers are from a rendered flat textured wall (and a rendered room) with a known path.

## 3. Measurements

Tool: `py -m gabai.tools.vo_bench` (wall scene; `--room` for the room scene). Rough motion = `truth_path_fast`: yaw swing up to about 6 degrees per frame, pitch
wobble, 0.4 degree random shake and 4 mm random translation per frame, one sudden 8 degree jerk, and motion blur (a horizontal box of half the per-frame motion, up
to 25 px). 150 frames, 6 seeds, disparity noise 0.3 px.

| Case | tracker | lost frames | position final / rms (cm) | rotation rms / final (degrees) |
|---|---|---|---|---|
| rough motion | before this work (no prediction, no refinement, no relocaliser) | 0.5 | 5.57 / 4.04 | 1.21 / 1.62 |
| rough motion | prediction + refinement | 0.8 | 3.77 / 2.76 | 0.88 / 1.00 |
| rough motion | all on (relocaliser too) | 0.7 | 4.47 / 3.24 | 0.94 / 1.16 |
| rough motion + 5 lost pictures | before | 10.0 | 10.79 / 6.12 | 15.26 / 21.88 |
| rough motion + 5 lost pictures | all on | 5.0 | 4.30 / 3.25 | 1.95 / 1.16 |
| rough motion, a true rotation handed in as the prior (stand-in for a gyro) | all on | 0.0 | 2.28 / 1.72 | 0.45 / 0.62 |

Remarks: the three techniques differ little on rough motion alone with this scene (the differences between "prediction + refinement" and "all on" are within the
noise of 6 seeds); the relocaliser is what matters when pictures are lost. The last row uses the TRUE rotation, so it shows the size of the gain a perfect
rotation prior would give the same code path, not what a real gyro would give (a real one has bias, noise and a time offset).

Real video (`data/left.mp4`, 900 frames, every 2nd): the old and the new tracker are indistinguishable (no frame lost by either; map alignment 4.16 and 4.15 cm
median; 18.1 and 18.5 fps; pose 5.2 and 5.9 ms). The video holds no sudden motion, so it does not exercise the new code. With loop detection on, 48
revisits were found and reported drifts of 1.3 to 1.8 cm and 1 to 1.7 degrees (the camera stays in one small area).

## 4. Architecture for adding an IMU later

The structure already separates what an IMU would change from what it would not.

- `gabai/pipeline/poses.py`: both modes return the same `PoseResult`; `make_provider` is the only place the mode picks an estimator. `stereo_only` never opens
  the IMU. An inertial mode is another provider (Basalt already is one).
- `KeyframeVO.step(..., dR_prior=R)`: the camera rotation since the previous frame from an outside sensor. It replaces the constant-velocity rotation in the
  prediction step (so the feature search starts in the right place through a sudden spin) and gates the result. `StereoOnlyProvider.push(..., dR_prior)` passes it through;
  nothing supplies it in `stereo_only`. Tested with a true rotation as a stand-in, so the seam works; no gyro was connected.
- `KeyframeVO.correct(R, t)`: any outside source of a better pose (the map refiner, a loop correction, a filter) can replace the pose and the keyframe follows.
- The relocaliser, the loop detector, the depth fusion and the map do not depend on how the rotation was predicted.

What adding an IMU for real would still need, none of which exists: a gyro and accelerometer model with bias states, a camera-IMU extrinsic and time offset
(the project has the tools to estimate them: `sync_imu_camera`, `check_extrinsics`), pre-integration between frames to produce `dR_prior` and a translation
prior, and then, for a tightly coupled result, an optimiser in place of the PnP step with the inertial residual in it. Basalt in `stereo_imu` mode already does
that and stays the reference.

## 5. Settings

`config/pipeline.toml`, section `[tracking]`: `predict`, `refine`, `relocalize` (all on), `loop_closure`, `loop_apply` (off), `loop_radius_m`, `loop_min_path_m`.
The summary has `vo_stats` (relocalisations, loops found and applied, how often the predicted start worked). Tests: `tests/test_kfvo.py` (interface, fast) and
`tests/test_kfvo_robust.py` (rough motion, lost pictures, prior, loops; about 90 s, deferred).

## 6. Second pass: drift (2026-10-28)

**What the papers say about camera-only versus visual-inertial** (secondary sources; I did not read the full papers). ORB-SLAM3 reports, on EuRoC, stereo-inertial
about 3.5-4.7 cm ATE and pure stereo about 2.7-4.0 cm on the easy groups but 16.3 cm on the hardest Vicon 2 group, with stereo-inertial scale error 0.3-0.8 percent
([arXiv 2007.11898](https://arxiv.org/pdf/2007.11898); the extracted table was garbled, check the PDF). An independent comparison says inertial fusion helps avoid lost tracks,
especially on V203 ([arXiv 2107.07589](https://arxiv.org/pdf/2107.07589)). Visual-only methods are described as weak under fast motion and in textureless areas, because the
IMU gives accurate short-term motion that vision lacks ([VI-DSO evaluation](https://alphaxiv.org/abs/1804.05625)). So: with good texture and a stereo baseline, camera-only
can match visual-inertial accuracy; on fast or blurred or textureless motion it cannot, because the information is absent from the pictures. I did not find evidence that
camera-only beats a well-calibrated IMU system there, and I do not claim it. LDSO shows loop closure as a Sim(3) pose graph beside a sliding-window estimator
([arXiv 1808.01111](https://arxiv.org/pdf/1808.01111)); that is the structure used here.

**Where the drift came from.** On the rendered room benchmark (`py -m gabai.tools.slam_bench`), error with no depth noise was 0.8 cm rms, with fresh noise each frame 1.8 cm,
with the persistent noise (70 percent of the error the same in every frame, like a stereo network's) 3.9 cm, fully persistent 6.5 cm. Persistent stereo error is not averaged out
by re-measuring from the same place. Parallax from several keyframes does not share it.

**Built:**
- `gabai/localba.py`: sliding-window bundle adjustment (6 keyframes, up to 250 landmarks). Residuals: reprojection and measured disparity. Two oldest poses held. Outliers dropped
  after a first pass (a Huber loss stalled). Runs on a worker thread every 3rd keyframe. Needs landmarks that live across keyframes (`carry`), which on its own made things worse
  (3.9 to 5.6 cm) until the adjustment refined them.
- `gabai/posegraph.py`: pose graph over keyframes (odometry edges plus loop edges), plain least squares (Huber stalled here too), 5-40 evaluations, 0.2-0.7 s for 100-300 keyframes,
  on a worker thread. Wired to loop detection (`loop_apply`, off by default). `KeyframeVO.trajectory()` rebuilds the whole path from the corrected keyframes.

**Measured** (rendered room, 3 laps, 300 frames, 5 seeds, same renders for each row):

| Setting | online rms / final (cm) | frame step median / p95 (ms) |
|---|---|---|
| before | 3.89 / 5.72 | 4.3 / 22.6 |
| window BA, every keyframe, same thread | 1.95 / 1.71 | 4.0 / 255 (stalls tracking) |
| window BA, every keyframe, worker thread | 4.31 / 3.48 (worker cannot keep up, 35 of 48 runs skipped) | 5.5 / 29.6 |
| window BA, every 3rd keyframe, worker (default now) | 2.77 / 3.11 (4 seeds) | 4.1 / 23.6 |

So most of the gain survives asynchronous running at no measurable cost per frame. The synchronous variant is more accurate but unusable. Real pipeline (150 frames of video):
pose stage 6.2 vs 7.5 ms median, p95 17-21 ms with or without it; end-to-end fps and p95 varied between runs by more than the difference (BA off also showed a p95 of 135 ms once).

**Pose graph with loop closure on the 3-lap room: not a win yet.** With `loop_apply` the error went UP (3.94 to 7.09 cm rms, 42 loop edges, 42 graph runs). The edges come from many
noisy single matches (the same 6 cm drift was measured as 4-11 cm), and there were no drift-free anchor edges between laps in this scene. It stays off by default. Needed before it is
trustworthy: edge covariances from the real inlier geometry, rejecting loops whose measured drift disagrees with their neighbours, and one edge per revisit not one per keyframe.
The pose graph itself is tested to close a drifted circle (ATE 0.33 to 0.02 m).

**Limits.** All numbers are from rendered scenes with an assumed noise model. The real-video clip has no ground truth and no fast motion. The adjustment's disparity weight
(`SIGMA_D`) was picked by looking at this benchmark, so the benchmark flatters it. Landmarks are only refined inside the 6-keyframe window; nothing is marginalised.

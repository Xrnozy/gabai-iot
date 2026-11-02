# Phase S spike report (in progress)

## PC-side depth model (done, measured on this PC)
- LiteAnyStereo cloned to `third_party/LiteAnyStereo` (MIT). LAS2_S.pth (24 MB) downloaded from Hugging Face `tomtomtommi/LiteAnyStereoV2` with the repo's own `download_checkpoint.py`.
- Extra pip packages installed to the global Python: transformations, thop, imageio, joblib.
- LAS2-S loads and runs on the RTX 5060 Ti (Blackwell) with plain PyTorch 2.11+cu128. Demo on a sample ZED pair (1242x2208) produced disparity, point cloud and GLB in `docs/las2_demo/`.
- Speed at 640x480, FP16 autocast, `--max_disp 192`: mean 22.84 ms, median 22.25 ms (43.8 FPS), 50 timed iterations, random weights (runtime does not depend on weights). Spec target is under 25 ms. This is model-only time on random input, not including image rectification, host-to-GPU copy or post-processing. `--max_disp 128` crashes (shape error); keep 192.
- Decision: TensorRT/ONNX is skipped for now since PyTorch FP16 already meets the spec number. Revisit only if the full pipeline misses the latency budget.
- Not yet measured: accuracy on your webcam images (needs calibrated pairs from the Pi).

## Pi-side go/no-go (waiting on your tests)
Camera sync, camera controls, Pi power/load/RAM, IMU and compass bring-up, stereo calibration, depth scenes: all scripts written, none run. See `phase-S-manual-test.md`.

## Camera sync, measured (2026-10-25, recording sync30c)
Method: phone strobe, `pi_capture sync` (both cameras 30 FPS, paired by arrival time, 15 pairs/s), fixed identical exposure 30 ms, edge method in `analyze_sync` (sub-frame, from the partly lit frame at each light edge). 47 light edges seen partly lit by both cameras.
- Right-minus-left exposure-window offset: median +3.8 ms; quartiles 2.1 and 9.9 ms; 90th percentile 17.9 ms; worst 22.4 ms.
- 28% of edges exceed 10 ms. Verdict against the 10 ms spec: the typical offset meets it, the worst case does not (matches the design expectation of about half a frame period, 17 ms, at 30 FPS).
- Caveats: only 47 edges; only edges where both cameras were partly lit are counted; fixed exposure (auto exposure would likely be worse); pairing logic only, no hardware sync; the right camera's light position came from the grid finder, not a manual check.
- An earlier result of 3.9 ms (sync30b) was invalid (slow brightness drift, not flashes) and is withdrawn.

## Camera frame rates (checked on the Pi)
EMEET C100 at 640x480: MJPG 30 fps maximum, YUYV 25 fps. Every MJPG size listed also tops out at 30 fps. 60 FPS pairing is not available, so the worst-case pairing offset stays near half a 30 FPS frame period (about 17 ms). Options left: announce reduced accuracy during fast head turns (needs the IMU gyro), or use a hardware-synchronised stereo camera.

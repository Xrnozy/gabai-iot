> **STATUS (updated 2026-10-26).** This file is a copy of the approved plan. The original is at
> `C:\Users\MY PC\.claude\plans\pasted-content-id-a514-based-on-fluffy-firefly.md`.
>
> | Phase | Status |
> |---|---|
> | S hardware spike | Done enough to decide: GO with conditions, see `docs/phase-S-report.md`. Open: 10-minute Pi stability run, depth beyond 0.85 m, glass and thin objects, drop-off |
> | V vertical slice | Built and tested without hardware. Supervised hardware tests (`docs/phase-V-manual-test.md`) SKIPPED by the user on 2026-10-26; they will be exercised during the SLAM phases. NOT approved by a test report |
> | M measure and graph | Tools built and tested without hardware. Partial real data exists (runs M2, M4, M6, P1, P2). Missing: false/missed-alarm marks, M3 sound-delay check, M5 depth on real targets. Pi under-voltage on the power bank is an open fault. Skipped by the user on 2026-10-26 |
> | SLAM (spec phase 1c) | Basalt built natively on Windows (VIT C API), recording, offline replay and live view tools built, still-gate and accelerometer calibration done. Real recordings exist; live runs worked in earlier sessions. Drift not measured over long runs. Plan: `docs/slam-plan.md` |
> | G conditional | IMU orientation done, plus compass fusion and sync tools. YOLO detector adapter and depth-for-detection written and unit-tested, but no weights installed and not wired into the viewer |
> | Diagnostics and testing (2026-10-26) | Structured logging, per-frame evidence traces, checks, self-check, diagnostic report and package, offline replay with simulated scenarios, rolling grid and governor (advisory only), failure-mode tests. Verified in software only; not yet run on the live rig. See `docs/architecture.md`, `docs/debugging.md`, `docs/testing.md`, `docs/troubleshooting.md`, `docs/hardware-test-checklist.md` |
> | Spec phases 3-6 | Modules built and tested in simulation, hands/YOLO/LAS2 preview run on real video. Not wired into the live viewer, not run on the rig, spec thresholds not measured. SceneScript, ShapeR, nvblox and YOLOE not available. See `docs/phases-3-6.md` |
> | Freeze, rehearsal, backup video | Not started |
>
> Phase V and M hardware results are owed, not done: the numbers they would give are not in any report. Freeze and rehearsal have not been started.

# Gabai Lite: Revised Plan (risk-first, 2-week deadline)

## Context
Spec: `spec/Gabai Lite.md`. Wearable Pi 3B + 2 EMEET C100 webcams + BMI160 + HMC5883L -> Ethernet -> PC (RTX 5060 Ti). Goal for the competition: stereo depth -> obstacle check -> spoken warning, with honest failure announcements, and measured graphs (latency, depth error, alert timing) as the IoT evidence. Deadline is **under 2 weeks**, so this plan commits in detail only to Phases S, V, M and a conditional G. Everything else is deferred and not planned.

The earlier plan is replaced. Reason (your review): it built harness/schemas first and left the physics/hardware risks last, put Basalt VIO on the critical path of a basic obstacle warning, and over-scoped Phases 3-6.

## Current state (audited)
- Only code: `testing/pose_viewer.py` (110 lines, unrun; TCP viewer + Madgwick/slerp; hardcoded IP; defines an ad-hoc wire format). No Pi-side sender, tests, git, manifests.
- `models/scenescript_model_ase.ckpt`, `Aria Research/`: unused, deferred.
- PC: Python 3.11.6, torch 2.11+cu128, numpy, opencv 5.0, open3d, scipy, pyserial installed. Missing: pytest, pyzmq, msgpack, onnx/tensorrt, ultralytics, a TTS engine.
- Hardware you have: Pi + 2 webcams, BMI160, HMC5883L wired, PC speakers. **I wait for your wiring photo before touching IMU/mag code.**

## Ground rules kept from before
- Gating: you test and explicitly approve each phase before the next starts.
- "Not performed" is reported as not performed; nothing unverified is claimed.
- Spec, `models/`, `Aria Research/`, `pose_viewer.py` stay untouched; new work is additive under `gabai/`, `tests/`, `docs/`.
- Test discipline is applied where it pays: alert logic, staleness, framing, drop-off detection, governor tiers. Not uniform ceremony per phase; test reports stay short (one page with a table and graphs).

## Phase S: Hardware spike (days 1-2). Decides whether the project exists
Crude scripts in `gabai/spike/`, no architecture.
1. **Camera sync**: a screen stopwatch refreshes at ~60 Hz and cannot resolve 10 ms, so it is NOT used. Method: a flashing LED on a fixed known pattern (or a fast-moving high-contrast target) seen by both cameras; offset = pixel shift / LED-phase difference between left and right frames, still and moving. Also record kernel v4l2 timestamps. Expect the 10 ms spec to fail (free-running webcams at 15 FPS can be 0-66 ms apart). Before giving up, also test **30 FPS with nearest-frame pairing** (worst case ~33 ms). Report a graded result, not binary: measured offset distribution, plus the use-case impact (walking straight: small error; fast head turns: false disparity, estimated from measured focal length and turn rate, e.g. 100 deg/s x 33 ms ~ 3 deg). Verdict forms: "meets spec", "usable straight-ahead, degraded when turning", or "no-go: buy synchronized stereo camera".
1b. **Pi power**: run `vcgencmd get_throttled` before/during/after a 10-min two-camera run; use a power bank/cable rated >=2.5 A; any under-voltage flag is recorded and fixed before continuing.
2. **Camera controls**: `v4l2-ctl --list-ctrls` on both; verify manual exposure, gain, WB, focus lock actually change the image and hold. Record which controls are ignored.
3. **Pi load**: 2x 640x480 MJPEG @15 FPS + Ethernet on the shared USB 2.0 bus; check dropped frames over 10 min, CPU, and real free RAM (the 900 MB criterion on a 1 GB board is likely unreachable; I measure and revise it).
4. **Depth on real images**: capture ~20 stereo pairs (wall, chair, dark bag, thin pole, glass, low light). Verify LiteAnyStereo (https://github.com/TomTomTommi/LiteAnyStereo) exists with usable weights and licence, whether the "LAS2-S" variant runs on Blackwell sm_120 with plain PyTorch FP16 (TensorRT/ONNX is **timeboxed to half a day, and skipped entirely if PyTorch FP16 meets latency**), and run it against OpenCV SGBM on the same pairs. Needs a quick stereo calibration (OpenCV checkerboard, not Kalibr); baseline and focal length are recorded so error targets are computed from physics, not guessed.
- **Go/no-go output** (`docs/spike-report.md`): if sync fails, stop and recommend a hardware-synchronized stereo camera before any further software. If LiteAnyStereo is unusable, fall back to SGM and report it.
- Manual: I give exact Pi/PC setup, commands, and a pass/fail sheet; you run it.

## Phase V: Vertical slice (days 3-6). Crude, end to end
Pi frames -> PC -> depth -> front-zone obstacle check -> spoken word.
- Pi: capture + minimal sender (framed TCP or ZeroMQ, JPEG bytes, Pi monotonic `t_capture`). PC: receiver, rectify, depth (LAS2 if spike passes, else SGM), simple front-zone check (left/ahead/right bands, nearest valid distance).
- **Negative obstacles**: ground-plane estimate (use IMU gravity/pitch if wiring ready, else fit floor from depth); flag floor-missing / depth-drop zones ahead (stairs down, curbs, holes) as "drop ahead"; unknown ahead => "can't see ahead".
- **Pre-rendered alert audio**: the alert vocabulary is a small fixed set ("left", "ahead", "right", distance phrases, "drop ahead", "can't see ahead", "tracking delayed", "Gabai stopped"). Render once to WAV with Piper (offline, build-time) and play clips at runtime; no live TTS in the warning path.
- **Required live telemetry page** (small FastAPI/WebSocket page on the PC, fastapi already installed): FPS, latency, nearest distance per zone, link status, Pi health (CPU/temp/throttle), scrolling alert log. This is the main IoT evidence.
- Tests first (pytest, small): alert queue priority/dedup/expiry; staleness (>300 ms "tracking delayed", >1 s "Gabai stopped" repeated every 5 s); unknown-is-not-free; front-zone logic on synthetic depth maps incl. drop-off; framing/loss detection on loopback sim.
- Drop-off honesty: learned stereo at 640x480 from head height may not show missing floor on textureless/shiny floors. Drop detection is not presented as working until tested on a real curb/step with the actual hit rate, and the "can't see ahead" rate on normal floors is measured (if it fires constantly users tune it out, which is itself a safety problem).
- Done when: you can walk toward a wall/chair/drop at the desk and hear a correct, rounded, honest warning; unplugging the cable is announced.

## Phase M: Measure and graph (days 7-9). Your competition material
- Latency clocks: Pi and PC clocks differ, so Pi `t_capture` is only used with a measured clock offset (round-trip ping over the cable, offset+uncertainty reported). Otherwise latency is measured from PC receipt and labeled so. Cross-check externally: record screen+audio and compare an LED flash to speech onset.
- `tools/` scripts + a log file per run: end-to-end latency (capture->speech, p50/p95; spec target <400 ms is a target not a result), depth error vs measured distance at 1/2/3 m on wall, chair, dark bag, thin pole, **glass and low-light cases**, false/missed alarm counts per run, sync offset, Pi CPU/RAM, dropped frames, GPU temp.
- Output: graphs (matplotlib) and a one-page report. Optional small web/log dashboard only if time remains (helps the IoT angle).
- Honest framing doc: tethered proof of concept (Ethernet to PC; audio from PC; not yet wearable), listing known weaknesses (glass, thin poles, dark objects, low light, negative obstacles).

## Phase G (conditional, days 10-11 only, if V and M are approved with slack)
At most: IMU orientation (Madgwick, reuse logic from `pose_viewer.py`) to stabilise the ground plane. YOLO person detection only if V and M finish early. Grid, governor and **VIO (Basalt) are not planned**; their spec targets would be hopes to measure, not requirements.

## Days 12-14: freeze and rehearsal (reserved, no new features)
Feature freeze, demo rehearsal, and a pre-recorded backup video in case the venue network, cable or lighting misbehaves. Each hardware round trip with you costs real time, so no phase is allowed to eat this buffer.

## Demo safety
Supervised tests only, extended to the demo: no judge or volunteer walks blindfolded relying on this. Presented as a proof of concept with measured limits.

## Deferred (not planned in detail)
Hands/grab (spec Phase 3), maps and relocalisation (4), object memory (5), SceneScript/ShapeR (6), Kalibr full calibration, voice input. Spec-clarification notes (frame transport, ASR, thresholds) go in `docs/spec-clarifications.md` when relevant.

## Honest limits to state up front
- Wired to PC and speaker on PC, so a tethered proof of concept. Headphones on the PC are the interim audio route.
- Safety: supervised tests only; no claim of reliability for real blind-user use.

## Verification
After approval I start Phase S only, with exact run instructions, then stop for your results and the go/no-go decision.

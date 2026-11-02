# Phase V manual test: vertical slice (frames, depth, obstacle check, spoken alerts, telemetry page)

Supervised tests only: do these at a desk or in a clear room with a helper. Do not rely on the output to walk safely; this is a proof of concept.

## What was built
- Spoken alerts from 23 pre-rendered clips (made once with the built-in Windows voice, offline): "left/ahead/right, about <half a metre ... three metres>", "drop ahead", "can't see ahead", "tracking delayed", "Gabai stopped, cable or camera lost", "reduced accuracy". No speech engine runs while walking.
- Priority queue: system failure outranks degraded accuracy, which outranks an obstacle within 1 m or a drop, which outranks an obstacle at 1 to 3 m. A higher alert cuts in; a lower one waits; a repeat within 3 s is dropped unless the obstacle got closer.
- Link monitor: no frames for 1 s says "Gabai stopped, cable or camera lost" and repeats every 5 s; frames older than 300 ms say "tracking delayed"; a fast turn says "reduced accuracy" (once per 10 s). A drop or "can't see ahead" must hold for two frames in a row.
- Floor model that follows the camera's real tilt (from the IMU), so the floor is not mistaken for an obstacle. Needs your camera height (`--cam-h`). Without `--cam-h`, obstacle speech is OFF (system alerts still speak), because the floor would look like an obstacle.
- Telemetry web page at port 8000 and a per-frame CSV log (`data/telemetry_log.csv`).

## Before you start (once)
1. Measure the height of the cameras above the floor in metres, as you will hold or wear the rig (for example 1.4).
2. For the tilt-aware floor model, run `py -m gabai.tools.sync_sensors` once (docs/sync-all-guide.md). Without it, the floor model assumes a level camera and a tilted rig will give false alerts.
3. Copy files to the Pi if you have not since the last session: `scp -r gabai xrnozy@192.168.1.63:~/` (PowerShell, project folder).

## Run
Pi:
```bash
cd ~
python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0
```
PC (replace 1.4 with your measured height; speakers or headphones on):
```powershell
py -m gabai.spike.demo_viewer --source live --model m --cam-h 1.4
```
It prints `spoken alerts: on | obstacle speech: on` and the telemetry address. Open `http://127.0.0.1:8000/` in a browser (or `http://<PC IP>:8000/` on a phone on the same Wi-Fi). Stop with `q` in the window.

## Tests
| # | Do | Expect | Record |
|---|---|---|---|
| 1 Sound | `py -c "import winsound;winsound.PlaySound('gabai/audio/alerts/ahead_1_0.wav', winsound.SND_FILENAME)"` | You hear "ahead, about one metre" | clear / too quiet / garbled |
| 2 Quiet scene | Point at an empty floor ahead with nothing in the way, 30 s | Mostly silent. Any speech counts as a false alarm | number of spoken alerts in 30 s, and which |
| 3 Wall | Walk slowly toward a wall from 3 m | "ahead, about N metres", N shrinking, each phrase repeating only when closer or after 3 s | distance spoken vs tape distance at 3 points |
| 4 Left/right | Put a box at your left, then right, about 1 m | "left, about one metre" then "right, about one metre" | correct side? |
| 5 Unplug | Press Ctrl+C on the Pi sender while running | Within about 1 s: "Gabai stopped, cable or camera lost", repeated every 5 s; telemetry link turns red | seconds until spoken |
| 6 Reconnect | Start the sender again | Alerts stop repeating, link green again | works / not |
| 7 Fast turn | Turn the rig quickly (about 100 deg/s) | "reduced accuracy" once, then not again for 10 s | spoken? |
| 8 Step down (supervised, a real kerb or single step; hold the rig well back) | Approach the edge from 1.5 m | "drop ahead" before the edge | spoken in how many of 10 tries; false "drop ahead" on a normal floor in 30 s |
| 9 Plain wall or dark corner | Look at a blank wall at 1 m | Either a distance or "can't see ahead", never silence-as-clear when you are touching it | what it said |
| 10 Telemetry | Watch the page while you do 3 and 5 | FPS, latency, zones, power flag and the spoken log update about 5 times a second | does it match the window |

## Report format
```
T1: pass/fail  notes:
T2: spoken alerts in 30 s = __ (which: __)
T3: spoken __ at tape __ m; __ at __ m; __ at __ m
T4: left ok? __  right ok? __
T5: seconds until "Gabai stopped" = __
T6: recovered? __
T7: reduced accuracy spoken? __
T8: drop ahead __/10; false drops in 30 s = __
T9: said: __
T10: page matches window? __
Attach: data/telemetry_log.csv, a screenshot of the page, any console errors.
```

## Known limits
- Obstacle speech needs `--cam-h`; the floor model assumes a flat floor and a correct height. Stairs, slopes, glass, shiny floors and thin poles are not tested.
- Alert phrases take about 1.3 s to say. Obstacle to speech latency is not measured yet (Phase M).
- Spoken distances are rounded to the nearest half metre and capped at 3 m.
- Uses the Windows voice and `winsound`, so sound works on Windows only.

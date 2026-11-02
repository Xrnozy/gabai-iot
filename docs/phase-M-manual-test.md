# Phase M: manual test sheet (measure and graph)

Everything here is supervised, at the desk. Nobody walks blindfolded or relies on the system. The tools and the report are tested
without hardware; **none of the numbers in the report exist until you run these tests.**

## What was built
| Piece | What it does |
|---|---|
| `demo_viewer --log <file>` | one CSV row per frame: latency, FPS, model and loop time, nearest distance per zone, alert, capture-to-speech time, Pi temperature, memory, load, under-voltage flag, round-trip time, PC GPU temperature |
| `F` and `X` keys in the viewer window | you mark a **false alarm** (it warned, nothing was there) or a **missed obstacle** (it stayed silent, something was there). Saved next to the log as `<file>.marks.csv` |
| `gabai.tools.target_test` | depth error on a real target (wall, chair, dark bag, thin pole, glass, low light) against a tape measure, plus how much of the target got any depth at all |
| `gabai.tools.range_test` | (already existed) depth error on the checkerboard |
| `gabai.tools.make_report` | graphs and a one-page report (`report.html`, one self-contained file) from the logs; anything not measured is printed as "not measured" |

## Setup (every run)
Pi (replace the IP with your PC's):
```bash
python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0
```
PC (a new log name for every run):
```powershell
py -m gabai.spike.demo_viewer --source live --model m --lr-check --cam-h 1.4 --log data/runs/M1_latency.csv
```
`--cam-h` is the lens height in metres; without it obstacle speech is off, and Phase M needs the speech. Measure it. Use the model and options you will demo with, and keep them the same for all runs.

## M1. Clock check (2 min)
Let it run for 1 minute. The window must show `capture->PC latency about N ms (+-M)`, not `latency: clock sync pending`.
- Record N and M. M is the uncertainty of the clock offset (half the round-trip time). If M is larger than about 20 ms, the latency numbers are only that accurate.
- If it stays "pending", stop and send me the screenshot: every latency graph depends on it.

## M2. Latency run (5 min)
Log: `data/runs/M2_latency.csv`. Move the rig slowly toward and away from a wall and a chair at the desk about 15 times, so alerts fire from several distances.
- Record: the speed you moved at, anything odd (stutter, delayed audio).
- Do **not** press F or X in this run.

## M3. Latency cross-check with a recording (10 min)
The log measures to the moment the clip is handed to the sound card, so it misses the sound card and speaker delay. Check it independently:
1. Start a screen recording of the PC screen **with desktop audio** (OBS, or Win+Alt+R).
2. Hold your hand 2 m from the rig, then move it quickly to 0.5 m. Do this 10 times, a few seconds apart.
3. In the recording, count frames (60 fps = 16.7 ms each) from the first frame where the depth picture shows your hand to the first frame where the speech sound starts in the audio track (an editor's waveform view is enough).
4. Record the 10 values and their median. That is a lower bound on the real hand-to-ear delay, with the camera exposure (about 33 ms) added on top.
- Compare with the log's capture-to-speech median from M2/M4. A gap of more than about 100 ms means the sound path has a delay the log cannot see; report it.

## M4. Alert run with marks (3 runs of 3 min)
Logs: `data/runs/M4_a.csv`, `M4_b.csv`, `M4_c.csv`.
- **a, clear room:** point the rig at a clear floor and wall for 3 minutes with nothing within 3 m. Any alert is false: press **F** each time you hear one.
- **b, obstacles:** present a chair, a box and a wall at 0.5, 1 and 2 m, 10 presentations in total. If it stays silent for a presentation, press **X**.
- **c, mixed:** slowly pan the room for 3 minutes. Press F and X as in a and b.
- Record: the counts you pressed, and the room (what is in the field of view).

## M5. Depth accuracy on real targets (about 45 min)
Tape-measure from the lens to the target. For each row, run the command, hold still, press SPACE 3 times (it records the median of 15 frames each time), press `q`.
```powershell
py -m gabai.tools.target_test --label wall --true-m 1.0
py -m gabai.tools.target_test --label wall --true-m 2.0
py -m gabai.tools.target_test --label wall --true-m 3.0
```
Repeat at 1, 2 and 3 m for each target, changing `--label`:
| Target | Notes |
|---|---|
| `wall` | plain wall, normal light |
| `chair` | any chair, side on |
| `dark bag` | black bag or black jacket |
| `thin pole` | broom handle or similar; add `--box 30` |
| `glass` | a window or glass door; add `--light normal` |
| `wall dim` | same wall, lights low: add `--light dim` |
The tool prints the error and the share of the box that got depth at all. **A 0% share is a valid result:** it means the system cannot see that target, which is what the report must say.
A 3 m target needs a big enough object; if you cannot place one, skip that row and say so.
Optionally also run `py -m gabai.tools.range_test --square-mm <real square size>` as in `docs/accuracy-guide.md`.

## M6. 10-minute stability and power (also closes a Phase S item)
Log: `data/runs/M6_stability.csv`. Use the power supply you will demo with, rig still, normal light, nothing pressed.
- Record: whether the stream stayed up, the throttle value shown in the window, the Pi temperature at the end.
- Any under-voltage must be fixed (shorter or thicker cable, supply rated 3 A) and the run repeated. The report states under-voltage if the log has it.

## M7. Make the report
```powershell
py -m gabai.tools.make_report --run data/runs/M2_latency.csv --run data/runs/M4_a.csv --run data/runs/M4_b.csv --run data/runs/M4_c.csv --run data/runs/M6_stability.csv --targets data/target_tests.csv --title "Gabai Lite: measured results"
```
Open `data/reports/<time>/report.html`. Send me the folder path, or the report and your notes below. I read it for problems; I do not edit numbers.

## Results sheet (fill in and send back)
| Test | Result | Notes |
|---|---|---|
| M1 latency N +- M ms | | |
| M2 capture to PC receipt p50 / p95 (ms) | | from the report |
| M2 capture to speech p50 / p95 (ms) | | |
| M3 hand-to-sound median in the recording (ms) | | |
| M4a false alarms in 3 min (clear room) | | |
| M4b missed of 10 presentations | | |
| M4c false / missed | | |
| M5 targets completed | | list; note any skipped |
| M5 worst target (name, error or "no depth") | | |
| M6 stream stayed up 10 min? under-voltage? | | |
| FPS in (p50), model ms, loop ms | | from the report |
| Anything that behaved oddly | | |

## What is NOT claimed
- No pass or fail thresholds are set here except what the spec states (400 ms target). Each number is reported as measured and compared with the target honestly, whether or not it passes.
- Capture-to-speech in the log stops when the clip is handed to the sound card (M3 checks the rest).
- The Pi's clock offset has an uncertainty of about half the round-trip time (M1 shows it).
- False and missed counts come from your key presses, so they depend on your judgement; say what you counted as a false alarm.

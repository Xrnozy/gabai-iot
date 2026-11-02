# LAS2 accuracy: what was measured, what helps, how to test at distance

## Measured on your rig (checkerboard truth, 25 pairs at 0.42-0.85 m, correct camera order)
| Variant | median error | 90th pct | 99th pct | scatter (std) |
|---|---|---|---|---|
| LAS2-S, native 640x480 | 0.49% | 1.28% | 2.28% | 6.1% (a few wild pixels) |
| LAS2-M, native | 0.46% | 1.21% | 1.79% | 0.72% |
| LAS2-H, native | 0.40% | 1.08% | 1.91% | 0.66% |
| M or H with left-right fusion (average of the two passes) | 0.41-0.45% | 1.06-1.17% | about the same | about the same |
| Input upscaled 1.5x | no better; S and M at 2x break | | | |
Findings:
- The model is already close to the floor of what this board test can measure (about 0.4%). Calibration and board-pose precision limit it, not LAS2.
- LAS2-S produces occasional wild pixels (scatter 6%). Use M (now the viewer's default) or H. H is slightly better and costs about 55 ms against 36 ms.
- Upscaling the input does not help, and at close range it pushes disparities past the model's maximum (192 px), so it fails.
- Averaging the two passes gives almost nothing. The left-right check (`--lr-check`) is still worth using to hide unreliable pixels.
- A stronger lens model (`--rational`) made the calibration diverge with 21 views. Do not use it.
- Not measured: anything beyond 0.85 m, glossy, dark or textureless surfaces, moving subjects.

## What actually limits accuracy now
1. Calibration (reprojection 0.64 px, verdict OK). More varied views at several distances would tighten it.
2. Camera sync (typical 4 ms, worst 22 ms): turning blurs depth.
3. Real scenes: textureless walls and distance, which the board test cannot show.

## Measure accuracy at distance (new)
Hold a flat checkerboard at distances you can tape-measure. The tool computes the board's true distance from its size and compares it with the depth over the board. The 25 mm board is too small beyond about 1.5 m, so print a bigger one (60 to 100 mm squares, 9x6 inner corners, glued flat) and pass its REAL square size.
Pi: `python3 -m gabai.pi.sender --host 192.168.1.64 --left /dev/video2 --right /dev/video0`
PC: `py -m gabai.tools.range_test --square-mm 60 --model m`
SPACE records a sample; take 3 or more at each of about 0.5, 1, 1.5, 2, 3 m. Press q: it prints median error, worst error and scatter by distance, and saves data/range_test.csv. Compare the displayed true distance with your tape; if they disagree, your square size entry is wrong.
Checked with replayed pairs: it reported -0.3% median error and 0.8% error spread at 0.4-0.8 m, matching the benchmark.

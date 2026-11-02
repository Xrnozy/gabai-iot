"""Estimate left/right camera exposure-window offset from a flashing light.

A short light pulse is visible to a camera only if it falls inside that camera's exposure
window (length E). If the two windows are offset by d, a pulse seen by one camera is also
seen by the other with probability max(0, 1 - |d|/E). So |d| ~= E * (1 - both/seen).
The estimate saturates at E: a result near E means "offset >= E", not "offset == E".
"""
import argparse
import csv
import statistics


def detect_spikes(times, brightness, k=6.0, min_jump=8.0):
    base = statistics.median(brightness)
    mad = statistics.median(abs(b - base) for b in brightness) or 1.0
    thr = base + max(k * 1.4826 * mad, min_jump)
    return [t for t, b in zip(times, brightness) if b > thr]


def detect_pulses(times, brightness, k=6.0, min_jump=8.0, max_run=3):
    """Isolated flashes only: bright runs longer than `max_run` frames (a hand, a lamp, a slow drift) are ignored."""
    base = statistics.median(brightness)
    mad = statistics.median(abs(b - base) for b in brightness) or 1.0
    thr = base + max(k * 1.4826 * mad, min_jump)
    events, run = [], []
    for t, b in list(zip(times, brightness)) + [(None, -1e9)]:
        if b > thr:
            run.append(t)
        else:
            if run and len(run) <= max_run:
                events.append(run[0])
            run = []
    return events


def edge_offsets(b_left, b_right, exposure_s, min_contrast=20.0):
    """Sub-frame offset between the two exposure windows from the partly-lit frame at each light edge.

    When the light switches on or off during a frame's exposure, that frame's brightness is the fraction of the
    window that was lit. Left and right see the same edge, so the difference in those fractions, times the
    exposure length, is how far apart the two windows are. Returns signed offsets in seconds (right minus left).
    Needs a light that switches slowly enough that each edge falls in one frame (a few flashes per second).
    """
    import numpy as np
    out = []
    fr = []
    for b in (np.asarray(b_left, float), np.asarray(b_right, float)):
        lo, hi = np.percentile(b, 10), np.percentile(b, 95)
        if hi - lo < min_contrast:
            return []
        fr.append(np.clip((b - lo) / (hi - lo), 0, 1))
    fl, fr_ = fr
    for i in range(1, len(fl) - 1):
        for sign, before, after in ((+1, lambda f, j: f[j - 1] < 0.15, lambda f, j: f[j + 1] > 0.85),
                                    (-1, lambda f, j: f[j - 1] > 0.85, lambda f, j: f[j + 1] < 0.15)):
            if all(0.1 < f[i] < 0.9 and before(f, i) and after(f, i) for f in (fl, fr_)):
                out.append(sign * exposure_s * (fr_[i] - fl[i]))
    return out


def best_cell(grid, min_range=10.0):
    """grid: (n_frames, rows, cols) mean brightness per block. Returns ((row, col), series) of the block that
    looks most like a flashing light: a big swing between its dark and bright levels AND many sharp jumps
    (a strobe has hundreds, a passing hand or a lamp has a handful). Works for any duty cycle."""
    import numpy as np
    g = np.asarray(grid, dtype=float)
    p5, p95 = np.percentile(g, 5, axis=0), np.percentile(g, 95, axis=0)
    swing = np.maximum(p95 - p5, (g.max(axis=0) - np.median(g, axis=0)) / 2)  # also catches sparse short flashes
    jumps = (np.abs(np.diff(g, axis=0)) > 0.4 * np.maximum(swing, 1e-9)).sum(axis=0)
    score = np.where(swing >= min_range, swing * np.minimum(jumps, 200), 0.0)
    r, c = np.unravel_index(np.argmax(score), score.shape)
    return (int(r), int(c)), g[:, r, c]


def match_spikes(left, right, tol):
    used = set()
    both = 0
    for tl in left:
        for j, tr in enumerate(right):
            if j not in used and abs(tl - tr) <= tol:
                used.add(j)
                both += 1
                break
    return both, len(left), len(right)


def estimate_offset_ms(both, n_left, n_right, exposure_s):
    seen = (n_left + n_right) / 2
    if seen == 0:
        raise ValueError("no light pulses detected in either camera")
    ratio = min(1.0, both / seen)
    return exposure_s * (1 - ratio) * 1000


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="frames.csv written by pi_capture.py")
    ap.add_argument("--exposure-ms", type=float, required=True,
                    help="exposure time actually used (exposure_time_absolute * 0.1)")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--grid", help="grid.npz from pi_capture: find the flashing light automatically in each camera")
    a = ap.parse_args()

    t, bl, br = [], [], []
    with open(a.csv) as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_left"]))
            bl.append(float(row["roi_left"]))
            br.append(float(row["roi_right"]))
    if a.grid:
        import numpy as np
        z = np.load(a.grid)
        (rl, cl), bl = best_cell(z["left"])
        (rr, cr), br = best_cell(z["right"])
        t = list(z["t"])
        print(f"light found at block row {rl} col {cl} in the left camera, row {rr} col {cr} in the right "
              f"(image split 6 rows x 8 cols)")
    import numpy as np
    off = edge_offsets(bl, br, a.exposure_ms / 1000)
    if len(off) >= 20:
        o = np.array(off) * 1000
        print(f"EDGE METHOD: {len(o)} light edges seen by both cameras")
        print(f"  right-minus-left window offset: median {np.median(o):+.1f} ms, median |offset| {np.median(np.abs(o)):.1f} ms, "
              f"90th percentile |offset| {np.percentile(np.abs(o), 90):.1f} ms")
        m = float(np.median(np.abs(o)))
        print(f"  share of edges with |offset| above 10 ms: {np.mean(np.abs(o) > 10) * 100:.0f}%   (worst seen {np.max(np.abs(o)):.1f} ms)")
        print("  verdict:", "meets the 10 ms spec" if m <= 10 else "does NOT meet the 10 ms spec")
        return
    print(f"edge method: only {len(off)} usable edges (need 20): no clear flashing light, or it switches too fast")
    sl = detect_pulses(t, bl)
    sr = detect_pulses(t, br)
    both, nl, nr = match_spikes(sl, sr, tol=0.5 / a.fps)
    print(f"isolated flashes seen: left={nl} right={nr} both={both}")
    if min(nl, nr) < 20:
        print("NO VALID RESULT: fewer than 20 isolated flashes per camera, so the light was not seen flashing. "
              "Aim a strobing light straight into both lenses and rerun.")
        return
    est = estimate_offset_ms(both, nl, nr, a.exposure_ms / 1000)
    print(f"estimated exposure-window offset: {est:.1f} ms (saturates at {a.exposure_ms:.0f} ms)")
    print("verdict:", "meets 10 ms spec" if est <= 10 else
          "does NOT meet 10 ms spec" + (" (offset may be >= exposure; lower bound only)"
                                         if est >= 0.9 * a.exposure_ms else ""))


if __name__ == "__main__":
    main()

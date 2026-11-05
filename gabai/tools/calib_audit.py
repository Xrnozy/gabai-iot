"""Audit of stereo calibration DATA (not just the RMS OpenCV prints): detections, coverage, per-camera and stereo errors, parameter uncertainty, and whether two boards agree.

  py -m gabai.tools.calib_audit [--out docs/calibration-audit.md]

Datasets (the saved pairs, re-detected here from the PNGs):
  25 mm board  data/charuco_pairs     7x10, 4x4 markers, 25 mm squares (as printed) / 18 mm markers
  45 mm board  data/charuco_pairs_45  7x5, 6x6 markers, 45 mm squares / 32 mm markers
What it checks, per dataset: corners found per pair; where the corners fall in each picture (4x3 regions) and at which distances / tilts; per-camera RMS and per-view errors; the standard deviation
OpenCV gives for every intrinsic; the stereo fit (baseline, focal, row error after rectification); the baseline's spread when views are left out (bootstrap: is it pinned down by the data?).
Across datasets: dataset A's calibration rectifies dataset B's corners (row error: does it generalise?) and triangulates B's board (square size: do the two boards agree on the metre?).
Nothing here can tell the true metre: both boards are trusted at their printed size. Only a tape-measured distance can (rtabmap_lab/tools/known_distance.py).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from gabai.tools.calibrate_stereo import calibrate_from_corners
from gabai.tools.charuco_calibrate import Board

DATASETS = {
    "25mm": dict(folder="data/charuco_pairs", squares_x=7, squares_y=10, square_mm=25.0, marker_mm=18.0, dictionary="4x4"),
    "45mm": dict(folder="data/charuco_pairs_45", squares_x=7, squares_y=5, square_mm=45.0, marker_mm=32.0, dictionary="6x6"),
}
FILES = {"stereo_calib_charuco_42.npz": "25 mm board solved with squares of 22.29 mm (scaled so the baseline is 42 mm)", "stereo_calib_charuco_45.npz": "45 mm board",
         "stereo_calib_charuco.npz": "25 mm board at 25 mm"}
NAMES = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]


def load(spec):
    b = Board(spec["squares_x"], spec["squares_y"], spec["square_mm"], spec["marker_mm"], spec["dictionary"])
    views, size, found = [], None, []
    for lf in sorted(glob.glob(os.path.join(spec["folder"], "*_L.png"))):
        l, r = cv2.imread(lf), cv2.imread(lf[:-6] + "_R.png")
        size = (l.shape[1], l.shape[0])
        dl, dr = b.detect(cv2.cvtColor(l, cv2.COLOR_BGR2GRAY)), b.detect(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY))
        found.append((0 if dl is None else len(dl[0]), 0 if dr is None else len(dr[0])))
        if dl is None or dr is None:
            continue
        ids = np.intersect1d(dl[0], dr[0])
        if len(ids) < 8:
            continue
        il, ir = [np.searchsorted(dl[0], ids)], [np.searchsorted(dr[0], ids)]
        views.append((os.path.basename(lf)[:-6], b.obj[ids].astype(np.float32), dl[1][np.argsort(dl[0])][np.searchsorted(np.sort(dl[0]), ids)], dr[1][np.argsort(dr[0])][np.searchsorted(np.sort(dr[0]), ids)]))
    return b, views, size, found


def single(views, size, k):
    o = [v[1] for v in views]
    p = [v[2 + k] for v in views]
    rms, K, D, rv, tv, sd, _, pve = cv2.calibrateCameraExtended(o, p, size, None, None, flags=cv2.CALIB_ZERO_TANGENT_DIST)
    pts = np.concatenate(p)
    grid = np.zeros((3, 4), int)
    np.add.at(grid, (np.clip((pts[:, 1] / size[1] * 3).astype(int), 0, 2), np.clip((pts[:, 0] / size[0] * 4).astype(int), 0, 3)), 1)
    dist = [float(np.linalg.norm(t)) for t in tv]
    tilt = [float(np.degrees(np.arccos(abs(cv2.Rodrigues(r)[0][2, 2])))) for r in rv]
    corner_reach = float(np.max(np.hypot((pts[:, 0] - K[0, 2]) / size[0], (pts[:, 1] - K[1, 2]) / size[1])) / np.hypot(0.5, 0.5))
    return {"rms": float(rms), "K": K, "D": D.ravel(), "std": {n: float(s) for n, s in zip(NAMES, np.asarray(sd).ravel()[:9])}, "per_view": np.asarray(pve).ravel(),
            "grid": grid, "dist_m": (min(dist), max(dist)), "tilt_deg": (min(tilt), max(tilt)), "reach": corner_reach}


def stereo(views, size):
    c = calibrate_from_corners([v[1] for v in views], [v[2] for v in views], [v[3] for v in views], size)
    return c


def bootstrap(views, size, n=12, seed=0):
    rng = np.random.default_rng(seed)
    B, f = [], []
    for _ in range(n):
        pick = rng.choice(len(views), len(views), replace=True)
        try:
            c = stereo([views[i] for i in pick], size)
        except Exception:
            continue
        B.append(c["baseline"] * 1000)
        f.append(c["focal"])
    return np.array(B), np.array(f)


def rows_with(c, views):
    """Mean |row difference| of a dataset's corners rectified with calibration c."""
    e = []
    for _, o, a, b in views:
        ua = cv2.undistortPoints(a.reshape(-1, 1, 2), c["K1"], c["D1"], R=c["R1"], P=c["P1"]).reshape(-1, 2)
        ub = cv2.undistortPoints(b.reshape(-1, 1, 2), c["K2"], c["D2"], R=c["R2"], P=c["P2"]).reshape(-1, 2)
        e.append(np.abs(ua[:, 1] - ub[:, 1]))
    return float(np.mean(np.concatenate(e)))


def square_scale(c, views, board):
    """Median (triangulated distance between neighbouring corners) / (printed distance): 1.0 when calibration c and this board agree on the metre."""
    rat = []
    for _, o, a, b in views:
        ua = cv2.undistortPoints(a.reshape(-1, 1, 2), c["K1"], c["D1"], R=c["R1"], P=c["P1"]).reshape(-1, 2)
        ub = cv2.undistortPoints(b.reshape(-1, 1, 2), c["K2"], c["D2"], R=c["R2"], P=c["P2"]).reshape(-1, 2)
        X = cv2.triangulatePoints(c["P1"], c["P2"], ua.T, ub.T)
        X = (X[:3] / X[3]).T
        d3 = np.linalg.norm(X[:, None] - X[None], axis=2)
        d0 = np.linalg.norm(o[:, None] - o[None], axis=2)
        m = (d0 > 1e-6) & (d0 < board.spec["square_mm"] / 1000 * 1.5)
        rat.append(np.median(d3[m] / d0[m]))
    return float(np.median(rat))


def npz(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/calibration-audit.md")
    ap.add_argument("--measured-baseline-mm", type=float, default=42.0, help="the lens-centre spacing you measured / the CAD value")
    a = ap.parse_args(argv)
    D, L = {}, []
    for name, spec in DATASETS.items():
        board, views, size, found = load(spec)
        s = [single(views, size, k) for k in (0, 1)]
        c = stereo(views, size)
        bB, bf = bootstrap(views, size)
        D[name] = dict(board=board, views=views, size=size, found=found, single=s, stereo=c, boot=(bB, bf))
        print(f"{name}: {len(views)} usable pairs, baseline {c['baseline']*1000:.2f} mm (bootstrap {bB.mean():.2f} +- {bB.std():.2f}), f {c['focal']:.1f}, rows {c['epipolar_err_px']:.3f} px", flush=True)
    L.append("# Stereo calibration audit\n")
    L.append("Generated by `py -m gabai.tools.calib_audit`. Both datasets are re-detected from the saved PNGs (640x360, the capture size: the stream is 640x360, so no rescaling).\n")
    L.append("## Per dataset\n")
    L.append("| | " + " | ".join(DATASETS) + " |")
    L.append("|---|" + "---|" * len(DATASETS))
    row = lambda label, fn: L.append(f"| {label} | " + " | ".join(fn(D[n]) for n in DATASETS) + " |")
    row("pairs saved / usable (both cameras, 8+ shared corners)", lambda d: f"{len(d['found'])} / {len(d['views'])}")
    row("corners per pair (median L / R)", lambda d: f"{np.median([x[0] for x in d['found']]):.0f} / {np.median([x[1] for x in d['found']]):.0f} of {(d['board'].spec['squares_x'] - 1) * (d['board'].spec['squares_y'] - 1)}")
    row("board distance (m)", lambda d: "{:.2f} - {:.2f}".format(*d["single"][0]["dist_m"]))
    row("board tilt (deg)", lambda d: "{:.0f} - {:.0f}".format(*d["single"][0]["tilt_deg"]))
    row("picture regions with < 20 corners (L / R, of 12)", lambda d: f"{int((d['single'][0]['grid'] < 20).sum())} / {int((d['single'][1]['grid'] < 20).sum())}")
    row("corners reach toward the picture corners (L / R, 1 = corner)", lambda d: f"{d['single'][0]['reach']:.2f} / {d['single'][1]['reach']:.2f}")
    row("single-camera RMS px (L / R)", lambda d: f"{d['single'][0]['rms']:.3f} / {d['single'][1]['rms']:.3f}")
    row("worst view px (L / R)", lambda d: f"{d['single'][0]['per_view'].max():.2f} / {d['single'][1]['per_view'].max():.2f}")
    for k in ("fx", "cx", "cy", "k1", "k2", "k3"):
        row(f"{k} L / R (+- OpenCV std)", lambda d, k=k: " / ".join(f"{(s['K'][0,0] if k=='fx' else s['K'][0,2] if k=='cx' else s['K'][1,2] if k=='cy' else s['D'][{'k1':0,'k2':1,'k3':4}[k]]):.3f} +- {s['std'][k]:.3f}" for s in d["single"]))
    row("stereo RMS px / pairs kept", lambda d: f"{d['stereo']['rms_stereo']:.3f} / {d['stereo']['n_views']}")
    row("rows after rectification px", lambda d: f"{d['stereo']['epipolar_err_px']:.3f}")
    row("baseline mm (bootstrap mean +- std)", lambda d: f"{d['stereo']['baseline']*1000:.2f} ({d['boot'][0].mean():.2f} +- {d['boot'][0].std():.2f})")
    row("rectified focal px (bootstrap std)", lambda d: f"{d['stereo']['focal']:.1f} (+- {d['boot'][1].std():.1f})")
    row("right camera rotation vs left (deg, x y z)", lambda d: " ".join(f"{v:+.2f}" for v in np.degrees(cv2.Rodrigues(d['stereo']['R'])[0].ravel())))
    row("translation direction (ty/tx, tz/tx)", lambda d: f"{d['stereo']['T'].ravel()[1]/d['stereo']['T'].ravel()[0]:+.3f}, {d['stereo']['T'].ravel()[2]/d['stereo']['T'].ravel()[0]:+.3f}")
    L.append("\n## Across datasets (calibration from the row, board of the column)\n")
    L.append("| calibration \\ corners of | " + " | ".join(f"{n}: rows px / square size ratio" for n in DATASETS) + " |")
    L.append("|---|" + "---|" * len(DATASETS))
    for cn in DATASETS:
        L.append(f"| {cn} | " + " | ".join(f"{rows_with(D[cn]['stereo'], D[bn]['views']):.3f} / {square_scale(D[cn]['stereo'], D[bn]['views'], D[bn]['board']):.4f}" for bn in DATASETS) + " |")
    for fn, what in FILES.items():
        p = os.path.join("data", fn)
        if os.path.exists(p):
            c = npz(p)
            L.append(f"| file `{fn}` ({what}) | " + " | ".join(f"{rows_with(c, D[bn]['views']):.3f} / {square_scale(c, D[bn]['views'], D[bn]['board']):.4f}" for bn in DATASETS) + " |")
    L.append("\nSquare size ratio 1.000 = this calibration measures that board at its printed size; 0.89 = it measures everything 11% short.\n")
    B25, B45 = D["25mm"]["stereo"]["baseline"] * 1000, D["45mm"]["stereo"]["baseline"] * 1000
    L.append("## Findings\n")
    L.append(f"- The two boards, each at its printed size, give baselines of **{B25:.2f} mm and {B45:.2f} mm** ({abs(B25 - B45) / B45 * 100:.1f}% apart). The value typed in as measured is {a.measured_baseline_mm:.0f} mm, "
             f"{(B45 - a.measured_baseline_mm) / a.measured_baseline_mm * 100:+.0f}% from them. Two independent prints agreeing makes a printing scale error unlikely (it would have to be the same on both); "
             f"the {a.measured_baseline_mm:.0f} mm is the lens spacing of the CAD model, not of the optical centres. Measure the lens centres with a caliper, or a wall at a tape-measured distance (known_distance.py).")
    L.append("- `stereo_calib_charuco_42.npz` (what `config/pipeline.toml` uses) is the 25 mm board solved with 22.29 mm squares so that the baseline comes out at 42 mm: every depth from it is "
             f"{a.measured_baseline_mm / B45:.3f} times what the boards say.")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print("written", a.out)


if __name__ == "__main__":
    main()

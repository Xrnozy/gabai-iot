"""How thick is a flat surface in the saved map? Scan one wall (or the floor), save the map (map.save in config/pipeline.toml), then:

py -m gabai.tools.wall_flatness data/world_map.npz [--band 0.10] [--min-obs 3]

Finds the biggest plane (RANSAC), then reports the spread of the map points within +-band of it. A good pose and clean depth give a few cm; a drifting pose gives a thick or
doubled wall (two peaks) that no filter removes. Compare the number before and after a change. Not a ground-truth accuracy: it measures flatness, not position.
"""
import argparse

import numpy as np


def flatness(points, band=0.10, thresh=0.03, iters=500, seed=0):
    """Largest plane in `points` (n,3). Returns dict: n_plane, thickness_cm (std of the signed distance of the points within band), p95_cm (95th percentile of |distance|), normal, two_layers."""
    pts = np.asarray(points, float)
    if len(pts) < 50:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((pts - a) @ n) < thresh).sum())
        if k > best_n:
            best, best_n = (n, a), k
    n, a = best
    near = pts[np.abs((pts - a) @ n) < thresh]                       # refit on the inliers: least squares plane
    c = near.mean(axis=0)
    n = np.linalg.svd(near - c)[2][2]
    d = (pts - c) @ n
    d = d[np.abs(d) < band]
    hist, _ = np.histogram(d, bins=np.linspace(-band, band, 21))
    peaks = int(sum(1 for i in range(1, 19) if hist[i] > hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > 0.25 * hist.max()))
    return {"n_plane": int(len(d)), "thickness_cm": float(d.std() * 100), "p95_cm": float(np.percentile(np.abs(d), 95) * 100), "normal": n.round(3).tolist(), "two_layers": peaks >= 2}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map")
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("--min-obs", type=int, default=3)
    a = ap.parse_args()
    from gabai.relocalize import load_map
    vm, meta = load_map(a.map)
    r = flatness(vm.points(min_obs=a.min_obs), a.band)
    if r is None:
        print("too few map points")
        return
    print(f"{a.map}: {len(vm)} voxels, biggest plane {r['n_plane']} points within +-{a.band * 100:.0f} cm, thickness (std) {r['thickness_cm']:.1f} cm, 95% within {r['p95_cm']:.1f} cm"
          + ("  <-- TWO LAYERS: the same wall seen from two poses" if r["two_layers"] else ""))


if __name__ == "__main__":
    main()

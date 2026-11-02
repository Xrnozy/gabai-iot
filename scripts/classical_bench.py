"""Classical stereo matchers scored on the ChArUco board pairs. On a rectified pair a plane's disparity is affine in (x, y), so the board's
disparity is fitted from its detected corners and every board pixel has a ground truth. Prints valid share, MAE and outliers per matcher.
  PYTHONPATH=. py scripts/classical_bench.py [calib.npz] [pairs_dir]"""
import glob, os, sys, time
import cv2, numpy as np
from gabai.tools.calibrate_stereo import Rectifier
from gabai.tools.charuco_calibrate import Board
from rtabmap_lab.census_sgm import CensusSGM

calib = sys.argv[1] if len(sys.argv) > 1 else "data/stereo_calib_charuco_42.npz"
folder = sys.argv[2] if len(sys.argv) > 2 else "data/charuco_pairs"
rect = Rectifier(calib)
board = Board(square_mm=22.29, marker_mm=22.29 * 18 / 25)


def gt_pairs():
    for lf in sorted(glob.glob(os.path.join(folder, "*_L.png"))):
        L, R = rect.apply(cv2.imread(lf), cv2.imread(lf.replace("_L", "_R")))
        gl, gr = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)
        a, b = board.detect(gl), board.detect(gr)
        if a is None or b is None: continue
        ids = np.intersect1d(a[0], b[0])
        if len(ids) < 20: continue
        pa = dict(zip(a[0], a[1])); pb = dict(zip(b[0], b[1]))
        xl = np.array([pa[i] for i in ids]); xr = np.array([pb[i] for i in ids])
        d = xl[:, 0] - xr[:, 0]
        A = np.c_[xl, np.ones(len(ids))]
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        res = d - A @ coef
        if np.abs(res).max() > 0.6: continue                  # corners that do not lie on one plane: skip the view
        hull = cv2.convexHull(xl.astype(np.float32)).astype(np.int32)
        m = np.zeros(gl.shape, np.uint8); cv2.fillConvexPoly(m, hull, 1)
        m = cv2.erode(m, np.ones((15, 15), np.uint8)).astype(bool)
        yy, xx = np.mgrid[:gl.shape[0], :gl.shape[1]]
        yield gl, gr, m, (coef[0] * xx + coef[1] * yy + coef[2]).astype(np.float32), float(d.mean())


ND = 128
def sgbm_cv(bs=9, p1k=8, p2k=32, uniq=10, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, cap=31, d12=1, speckle=(100, 4)):
    return cv2.StereoSGBM_create(0, ND, bs, p1k * bs * bs, p2k * bs * bs, d12, cap, uniq, speckle[0], speckle[1], mode)

def run_sgbm(s):
    return lambda l, r: s.compute(l, r).astype(np.float32) / 16.0

def run_wls(s, lam=8000, sig=1.5):
    right = cv2.ximgproc.createRightMatcher(s); f = cv2.ximgproc.createDisparityWLSFilter(s); f.setLambda(lam); f.setSigmaColor(sig); f.setLRCthresh(1)
    def go(l, r):
        dl = s.compute(l, r); dr = right.compute(r, l)
        out = f.filter(dl, l, disparity_map_right=dr).astype(np.float32) / 16.0
        conf = f.getConfidenceMap()
        out[conf < 100] = -1                                       # WLS fills holes by smearing; keep only pixels it was confident about
        return out
    return go

def run_lr(s, tol=1.0):
    def go(l, r):
        dl = s.compute(l, r).astype(np.float32) / 16.0
        fl, fr = cv2.flip(l, 1), cv2.flip(r, 1)                    # right-view disparity: match the mirrored pair
        dr = cv2.flip(s.compute(fr, fl).astype(np.float32) / 16.0, 1)
        h, w = dl.shape; xx = np.arange(w)[None].repeat(h, 0)
        xr = np.clip(np.round(xx - dl).astype(int), 0, w - 1)
        back = np.take_along_axis(dr, xr, 1)
        dl[(np.abs(dl - back) > tol) | (back <= 0)] = -1
        return dl
    return go

def preproc(f, clahe=2.0):
    c = cv2.createCLAHE(clahe, (8, 8))
    return lambda l, r: f(c.apply(l), c.apply(r))

def up2(f):                                                         # match at 2x, disparity back to 1x
    def go(l, r):
        L = cv2.resize(l, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC); R = cv2.resize(r, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        d = f(L, R); d = cv2.resize(d, (l.shape[1], l.shape[0]), interpolation=cv2.INTER_NEAREST); return np.where(d > 0, d / 2, -1).astype(np.float32)
    return go

METHODS = {
    "A SGBM current (bs9 3way)": run_sgbm(sgbm_cv()),
    "B SGBM bs5 3way": run_sgbm(sgbm_cv(5)),
    "C SGBM bs7 HH": run_sgbm(sgbm_cv(7, mode=cv2.STEREO_SGBM_MODE_HH)),
    "D SGBM bs7 + LR check": run_lr(sgbm_cv(7, d12=-1, uniq=5)),
    "E SGBM bs7 + WLS": run_wls(sgbm_cv(7, d12=1, uniq=5)),
    "F CensusSGM 8 paths": (lambda m: m.disparity)(CensusSGM(ND)),
    "G CensusSGM 4 paths": (lambda m: m.disparity)(CensusSGM(ND, paths=4)),
    "J SGBM bs7 CLAHE": preproc(run_sgbm(sgbm_cv(7))),
    "K CensusSGM, CLAHE": preproc((lambda m: m.disparity)(CensusSGM(ND))),
}

if __name__ == "__main__":
    pairs = list(gt_pairs()); print(f"{len(pairs)} board pairs with ground truth, disparity {np.mean([p[4] for p in pairs]):.1f}px mean")
    print(f"{'method':28s} {'ms':>5s} {'valid%':>7s} {'MAE px':>7s} {'RMSE':>6s} {'out>1px%':>8s} {'bias':>6s}")
    for name, fn in METHODS.items():
        v, e, t = [], [], []
        for l, r, m, gt, _ in pairs:
            t0 = time.perf_counter(); d = fn(l, r); t.append(time.perf_counter() - t0)
            ok = m & (d > 0.5); v.append(ok.sum() / m.sum()); e.append((d - gt)[ok])
        e = np.concatenate(e); good = np.abs(e) < 3
        print(f"{name:28s} {np.median(t)*1000:5.0f} {np.mean(v)*100:7.1f} {np.abs(e[good]).mean():7.3f} {np.sqrt((e[good]**2).mean()):6.3f} {(np.abs(e) > 1).mean()*100:8.1f} {np.median(e):6.3f}", flush=True)

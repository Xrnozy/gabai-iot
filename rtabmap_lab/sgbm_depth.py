"""Stereo depth from OpenCV SGBM with the checks RTAB-Map's own matcher lacks: texture rejection, speckle removal and an edge-aware WLS refinement. A pixel that fails a check STAYS EMPTY (no hole filling),
so wrong matches on blank walls do not become geometry.  py -m rtabmap_lab live --depth sgbm

Per pair: left matcher (SGBM, with its built-in left-right check disp12MaxDiff and speckle filter), optional right matcher for the WLS confidence, WLS refinement of the pixels that passed, texture mask
(local grey-level std of the left picture), then Z = f * B / d, the range gate and the noise gate sigma_z = z^2 * sigma_px / (f * B) <= max_std. sigma_px = 0.5 is MEASURED (tools/stereo_stages.py: every matcher tried sits at 0.48-0.52 px on the 2 Nov walk), not the 0.3 the bridge assumes;
with max_std 0.08 that keeps depth to about 2.2 m on this rig. Same depth_mm() contract as GabaiDepth / FusionDepth.
stages() returns every intermediate picture for rtabmap_lab/tools/stereo_stages.py.
"""
import time

import cv2
import numpy as np


class SgbmDepth:
    def __init__(self, num_disp=96, block=5, uniq=10, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY, texture=3.0, wls=True, lam=8000.0, sigma_color=1.5,
                 speckle_win=100, speckle_range=2, sigma_px=0.5, max_std=0.08):
        self.texture, self.sigma_px, self.max_std, self.use_wls = texture, sigma_px, max_std, wls
        self.m = cv2.StereoSGBM_create(0, num_disp, block, 8 * block * block, 32 * block * block, 1, 31, uniq, speckle_win, speckle_range, mode)
        if wls:
            self.mr = cv2.ximgproc.createRightMatcher(self.m)
            self.wls = cv2.ximgproc.createDisparityWLSFilter(self.m)
            self.wls.setLambda(lam)
            self.wls.setSigmaColor(sigma_color)
            self.m.setDisp12MaxDiff(1), self.m.setUniquenessRatio(uniq), self.m.setSpeckleWindowSize(speckle_win)   # building the filter zeroes these three on the left matcher (valid pixels 50% -> 84%): put them back

    def stages(self, left_bgr, right_bgr, debug=False):
        """Dict of pictures: gray_l, gray_r, raw (SGBM, px, NaN = none), tex (bool mask), wls (px or None), disp (final, NaN = none). debug=True adds raw_nolr, the matcher with its left-right check switched off."""
        gl, gr = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i for i in (left_bgr, right_bgr))
        d16 = self.m.compute(gl, gr)
        raw = np.where(d16 > 0, d16 / 16.0, np.nan).astype(np.float32)
        g = gl.astype(np.float32)
        mean, sq = cv2.boxFilter(g, -1, (7, 7)), cv2.boxFilter(g * g, -1, (7, 7))
        tex = np.sqrt(np.maximum(sq - mean * mean, 0)) >= self.texture
        out = {"gray_l": gl, "gray_r": gr, "raw": raw, "tex": tex, "wls": None}
        ok = np.isfinite(raw) & tex
        disp = raw
        if self.use_wls:
            f16 = self.wls.filter(d16, gl, disparity_map_right=self.mr.compute(gr, gl))
            out["wls"] = np.where(f16 > 0, f16 / 16.0, np.nan).astype(np.float32)
            disp = out["wls"]
        out["disp"] = np.where(ok, disp, np.nan).astype(np.float32)
        if debug:
            m = cv2.StereoSGBM_create(0, self.m.getNumDisparities(), self.m.getBlockSize(), self.m.getP1(), self.m.getP2(), -1, 31, self.m.getUniquenessRatio(), self.m.getSpeckleWindowSize(), self.m.getSpeckleRange(), self.m.getMode())
            r = m.compute(gl, gr)
            out["raw_nolr"] = np.where(r > 0, r / 16.0, np.nan).astype(np.float32)
        return out

    def depth_mm(self, left_bgr, right_bgr, cal, min_range=0.3, max_range=4.0, rtab_mm=None):
        """(uint16 depth in millimetres, 0 = none; milliseconds) for one rectified pair."""
        t0 = time.perf_counter()
        d = self.stages(left_bgr, right_bgr)["disp"]
        fb = cal["f"] * cal["B"]
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(d > 0.5, fb / d, 0.0)
        ok = (z >= min_range) & (z <= max_range)
        if self.max_std > 0:
            ok &= z * z * self.sigma_px / fb <= self.max_std
        return (np.where(ok, z, 0.0) * 1000.0).astype(np.uint16), (time.perf_counter() - t0) * 1000.0

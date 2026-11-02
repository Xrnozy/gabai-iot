"""Stereo calibration with a ChArUco board, watched in Rerun: both pictures with the camera timestamps, the detected corners, where the board has been, and after the fit the two cameras in 3D.

  1. Make the board and print it on letter paper at 100% / "actual size" (no fit-to-page):
       py -m gabai.tools.charuco_calibrate --make-board
     Measure one printed square and one marker with the caliper; give THOSE numbers below (--square-mm, --marker-mm).
  2. Calibrate live from the Pi (sender running, see docs/live-link-guide.md):
       py -m gabai.tools.charuco_calibrate --square-mm 25 --marker-mm 18 --rotate-left 180 --baseline-mm 42
     Or from a recording / a folder of NNN_L.png + NNN_R.png pairs:
       py -m gabai.tools.charuco_calibrate --recording data/recordings/vio_XXXX --square-mm 25 --marker-mm 18
       py -m gabai.tools.charuco_calibrate --images data/charuco_pairs --square-mm 25 --marker-mm 18
     The 7x5-square board with 6x6 markers (45 mm squares, 32 mm markers):
       py -m gabai.tools.charuco_calibrate --squares-x 7 --squares-y 5 --dict 6x6 --square-mm 45 --marker-mm 32 --rotate-left 180 --baseline-mm 42 --pairs-dir data/charuco_pairs_45 --out data/stereo_calib_charuco_45

A pair is kept when both cameras see the board, it is steady, it is in a new place, and the two pictures were taken less than --max-skew-ms apart while it moves (the Pi stamps each camera; a still board is fine at any offset). Cover the whole picture, near and far, tilted
both ways, until --pairs are kept (or press Ctrl+C to calibrate with what there is). The result goes to data/stereo_calib_charuco.npz (+ .json, + _charuco.json with the timestamps of every kept pair): same format as
gabai.tools.calibrate_stereo, so set source.calib in config/pipeline.toml to use it.

Timestamps: left = the pair's t_capture (the Pi's monotonic clock), right = t_capture + dt_ms (the Pi's measured right-minus-left offset). Rerun shows both on each picture, in the "stamps" panel, as the skew plot,
and as two timelines (capture = the Pi's clock, recv = this PC's clock).
"""
import argparse
import csv
import glob
import json
import os
import time

import cv2
import numpy as np

from gabai.tools.calibrate_stereo import calibrate_from_corners, check_baseline, save, verdict

DICTS = {"4x4": cv2.aruco.DICT_4X4_50, "5x5": cv2.aruco.DICT_5X5_250, "6x6": cv2.aruco.DICT_6X6_250}
LETTER_MM = (215.9, 279.4)


class Board:
    def __init__(self, squares_x=7, squares_y=10, square_mm=25.0, marker_mm=18.0, dictionary="4x4", min_corners=12):
        self.spec = dict(squares_x=squares_x, squares_y=squares_y, square_mm=square_mm, marker_mm=marker_mm, dictionary=dictionary)
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm / 1000.0, marker_mm / 1000.0, cv2.aruco.getPredefinedDictionary(DICTS[dictionary]))
        self.det = cv2.aruco.CharucoDetector(self.board)
        self.obj = np.asarray(self.board.getChessboardCorners(), np.float32).reshape(-1, 3)        # metres, by corner id
        self.size_m = (squares_x * square_mm / 1000.0, squares_y * square_mm / 1000.0)
        self.min_corners = min_corners

    def detect(self, gray):
        """(ids, xy) of the chessboard corners found, or None when fewer than min_corners."""
        cc, ci, _, mi = self.det.detectBoard(gray)
        self.markers = 0 if mi is None else len(mi)              # markers but no corners = the layout typed in is not the printed board
        if ci is None or len(ci) < self.min_corners:
            return None
        return ci.ravel().astype(int), cc.reshape(-1, 2).astype(np.float32)

    def printable(self, dpi=300):
        """The board centred on a letter page, pixels per mm exact so a 100% print has the nominal square size."""
        s = self.spec
        px = dpi / 25.4
        w, h = round(s["squares_x"] * s["square_mm"] * px), round(s["squares_y"] * s["square_mm"] * px)
        img = self.board.generateImage((w, h), marginSize=0, borderBits=1)
        page = np.full((round(LETTER_MM[1] * px), round(LETTER_MM[0] * px)), 255, np.uint8)
        if img.shape[0] > page.shape[0] or img.shape[1] > page.shape[1]:
            raise ValueError(f"the board ({s['squares_x'] * s['square_mm']:g} x {s['squares_y'] * s['square_mm']:g} mm) does not fit letter paper; use fewer or smaller squares")
        y0, x0 = (page.shape[0] - img.shape[0]) // 2, (page.shape[1] - img.shape[1]) // 2
        page[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        return page


def shared(dl, dr):
    """The corners both cameras found: (ids, left xy, right xy)."""
    ids, il, ir = np.intersect1d(dl[0], dr[0], return_indices=True)
    return ids, dl[1][il], dr[1][ir]


def mean_move(ids_a, xy_a, ids_b, xy_b, need=6):
    """Mean pixel displacement of the corners two views have in common; inf when they share fewer than `need` (then they count as different places)."""
    ids, ia, ib = np.intersect1d(ids_a, ids_b, return_indices=True)
    return float(np.linalg.norm(xy_a[ia] - xy_b[ib], axis=1).mean()) if len(ids) >= need else float("inf")


class Session:
    """Decides which pairs are kept. offer() detects the board in both pictures and applies the rules."""

    def __init__(self, board, pairs=30, min_shift=40.0, max_skew_ms=8.0, min_shared=10, need_stable=3):
        self.need_stable = need_stable           # steady frames in a row; 0 for saved pictures, which are not a video
        self.board, self.pairs, self.min_shift, self.max_skew, self.min_shared = board, pairs, min_shift, max_skew_ms, min_shared
        self.views, self.prev, self.stable, self.size = [], None, 0, None

    def offer(self, left, right, skew_ms=None):
        """Returns (detection left, detection right, status text, kept: bool)."""
        gl, gr = (cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i for i in (left, right))
        self.size = gl.shape[::-1]
        dl = self.board.detect(gl)
        ml = self.board.markers
        dr = self.board.detect(gr)
        if dl is None or dr is None:
            self.stable, self.prev = 0, None
            msg = "board not seen by " + ("both cameras" if dl is None and dr is None else "the left camera" if dl is None else "the right camera")
            other = [k for k in DICTS if k != self.board.spec["dictionary"] and max(ml, self.board.markers) == 0
                     and len(cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTS[k])).detectMarkers(gl)[0]) >= 4]
            if other and dl is None and dr is None:
                msg += f" (the markers are {other[0]}, not {self.board.spec['dictionary']}: run with --dict {other[0]} and the printed --squares-x/--squares-y)"
            elif max(ml, self.board.markers) >= 4 and dl is None and dr is None:
                s = self.board.spec
                msg += (f" ({max(ml, self.board.markers)} markers found but no board corners: the printed board is not {s['squares_x']}x{s['squares_y']} squares "
                        f"with {s['dictionary']} markers; check --squares-x/--squares-y/--dict)")
            return dl, dr, msg, False
        ids, xl, xr = shared(dl, dr)
        if len(ids) < self.min_shared:
            self.stable, self.prev = 0, None
            return dl, dr, f"only {len(ids)} corners seen by both cameras (need {self.min_shared}): show more of the board to both", False
        moved = mean_move(ids, xl, *self.prev) if self.prev is not None else float("inf")
        self.prev = (ids, xl)
        self.stable = self.stable + 1 if moved < 1.5 else 0
        if skew_ms is not None and abs(skew_ms) > self.max_skew and self.stable == 0:       # a still board makes the offset harmless; a moving one would smear it into the two pictures
            return dl, dr, f"board moving while the pictures are {skew_ms:+.1f} ms apart (limit {self.max_skew:g}): hold still", False
        if any(mean_move(ids, xl, v["ids"], v["l"]) < self.min_shift for v in self.views):
            return dl, dr, "move the board to a NEW position / angle / distance", False
        if self.stable < self.need_stable:
            return dl, dr, "hold still...", False
        self.stable = 0
        return dl, dr, f"KEPT pair {len(self.views) + 1}/{self.pairs}", True

    def keep(self, left, right, dl, dr, **stamps):
        ids, xl, xr = shared(dl, dr)
        self.views.append(dict(ids=ids, l=xl, r=xr, left=left, right=right, **stamps))

    def calibrate(self, rational=False, fix_k3=False, min_views=10):
        obj = [self.board.obj[v["ids"]] for v in self.views]
        lp = [v["l"].reshape(-1, 1, 2) for v in self.views]
        rp = [v["r"].reshape(-1, 1, 2) for v in self.views]
        return calibrate_from_corners(obj, lp, rp, self.size, rational, min_views, fix_k3)


def board_extent_check(c):
    """Turned-around cameras show up as a big rotation between the two; a wrong square size as a baseline off by the same factor (check_baseline)."""
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(c["R"])[0])))


def depth_check(c, board, views):
    """Per kept pair: the board's distance from stereo disparity of its corners against the distance solvePnP gives from the board size alone. Both come from this calibration,
    so it tests that baseline, focal length and square size agree with each other, not the absolute scale."""
    errs = []
    for v in views:
        obj = board.obj[v["ids"]]
        ok, rv, tv = cv2.solvePnP(obj, v["l"], c["K1"], c["D1"])
        if not ok:
            continue
        z_pnp = (c["R1"] @ (cv2.Rodrigues(rv)[0] @ obj.T + tv)).T[:, 2]
        ul = cv2.undistortPoints(v["l"].reshape(-1, 1, 2), c["K1"], c["D1"], R=c["R1"], P=c["P1"]).reshape(-1, 2)
        ur = cv2.undistortPoints(v["r"].reshape(-1, 1, 2), c["K2"], c["D2"], R=c["R2"], P=c["P2"]).reshape(-1, 2)
        d = ul[:, 0] - ur[:, 0]
        z_st = c["focal"] * c["baseline"] / np.where(d > 0.1, d, np.nan)
        e = np.abs(z_st - z_pnp) / z_pnp
        if np.isfinite(e).any():                                  # all NaN = no positive disparity: the cameras are swapped
            errs.append(float(np.nanmedian(e)))
    return errs


def coverage(views, size, grid=3):
    cells = np.zeros((grid, grid), int)
    for v in views:
        for x, y in v["l"]:
            cells[min(int(y / size[1] * grid), grid - 1), min(int(x / size[0] * grid), grid - 1)] += 1
    return cells


def stamps_text(seq, tl, tr, t_recv, skew, t0):
    f = lambda t: "n/a" if t is None else f"{t:.4f} s (+{t - t0:.3f})"
    return (f"pair {seq}\nleft  capture {f(tl)}\nright capture {f(tr)}\nright - left {('n/a' if skew is None else f'{skew:+.2f} ms')}\n"
            f"received on the PC {('n/a' if t_recv is None else f'{t_recv:.4f} s')}")


# ---------------------------------------------------------------- sources: (seq, left, right, t_left, t_right, t_recv, skew_ms)
def live_pairs(a):
    from gabai.pc.receiver import LiveReceiver
    rx = LiveReceiver(a.port, a.token, rotate_left=a.rotate_left, rotate_right=a.rotate_right)
    rx.start()
    print(f"waiting for the Pi on port {rx.port} ... start: python3 -m gabai.pi.sender --host <PC_IP>", flush=True)
    last = None
    try:
        while True:
            p = rx.latest()
            if p is None or p.seq == last:
                time.sleep(0.01)
                continue
            last = p.seq
            yield p.seq, p.left, p.right, p.t_capture, (None if p.dt_ms is None else p.t_capture + p.dt_ms / 1000.0), p.t_recv, p.dt_ms
    finally:
        rx.stop()


def recording_pairs(a):
    rows = list(csv.DictReader(open(os.path.join(a.recording, "pairs.csv"))))
    capl, capr = cv2.VideoCapture(os.path.join(a.recording, "left.avi")), cv2.VideoCapture(os.path.join(a.recording, "right.avi"))
    try:
        for n, row in enumerate(rows):                             # the stored pictures are already turned the way the Pi's layout needs
            okl, left = capl.read()
            okr, right = capr.read()
            if not (okl and okr):
                break
            dt = None if row["dt_ms"] == "" else float(row["dt_ms"])
            tc = float(row["t_capture"])
            yield n, left, right, tc, (None if dt is None else tc + dt / 1000.0), float(row["t_recv"]), dt
    finally:
        capl.release(), capr.release()


def image_pairs(a):
    for n, lf in enumerate(sorted(glob.glob(os.path.join(a.images, "*_L.png")))):
        rf = lf[:-6] + "_R.png"
        if os.path.exists(rf):
            yield n, cv2.imread(lf), cv2.imread(rf), None, None, None, None


# ---------------------------------------------------------------- rerun
class View:
    def __init__(self, a):
        import rerun as rr
        import rerun.blueprint as rrb
        self.rr = rr
        rr.init("gabai_charuco_calibration", spawn=not a.no_viewer)
        if a.save_rrd:
            rr.save(a.save_rrd)
        sp = lambda o, n: rrb.Spatial2DView(origin=o, name=n)
        rr.send_blueprint(rrb.Blueprint(
            rrb.Tabs(
                rrb.Vertical(
                    rrb.Horizontal(sp("cameras/left", "left picture"), sp("cameras/right", "right picture")),
                    rrb.Horizontal(rrb.TimeSeriesView(origin="timing", name="timing (ms)"), rrb.TextDocumentView(origin="stamps", name="timestamps"),
                                   sp("coverage", "where the board has been"), rrb.TextDocumentView(origin="status", name="status")),
                    row_shares=[3, 2], name="Capture"),
                rrb.Vertical(
                    rrb.Spatial3DView(origin="world", name="cameras and boards"),
                    rrb.Horizontal(sp("rectified", "rectified pair, rows must line up"), rrb.BarChartView(origin="calibration", name="reprojection error per pair (px)"),
                                   rrb.TextDocumentView(origin="report", name="report")),
                    row_shares=[3, 2], name="Result")),
            rrb.TimePanel(timeline="frame")))
        self.t0 = self.r0 = self.prev = None

    def times(self, n, tl, tr, t_recv):
        rr = self.rr
        rr.set_time("frame", sequence=n)
        if tl is not None:
            self.t0 = tl if self.t0 is None else self.t0
            rr.set_time("capture", duration=tl - self.t0)
        if t_recv is not None:
            self.r0 = t_recv if self.r0 is None else self.r0
            rr.set_time("recv", duration=t_recv - self.r0)

    def pair(self, n, left, right, dl, dr, tl, tr, t_recv, skew, status, k_views):
        rr = self.rr
        self.times(n, tl, tr, t_recv)
        t0 = self.t0 if self.t0 is not None else 0.0
        ids = set(shared(dl, dr)[0].tolist()) if dl is not None and dr is not None else set()
        for name, img, det, t in (("left", left, dl, tl), ("right", right, dr, tr)):
            pic = img.copy()
            label = f"{name.upper()}  t={'n/a' if t is None else f'{t:.4f} s'}"
            cv2.putText(pic, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(pic, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            if name == "right" and tr is not None:                                  # the right camera's own clock reading on its picture; its offset in the row below
                self.times(n, tr, tr, t_recv)
            rr.log(f"cameras/{name}", rr.Image(pic[:, :, ::-1]).compress(jpeg_quality=80))
            if det is not None:
                col = np.array([(60, 220, 90) if i in ids else (240, 190, 40) for i in det[0]], np.uint8)
                rr.log(f"cameras/{name}/corners", rr.Points2D(det[1], colors=col, radii=3.0, labels=[str(i) for i in det[0]]))
            else:
                rr.log(f"cameras/{name}/corners", rr.Clear(recursive=False))
        self.times(n, tl, tr, t_recv)
        if skew is not None:
            rr.log("timing/right_minus_left_ms", rr.Scalars(float(skew)))
        if self.prev is not None and tl is not None and self.prev[0] is not None:
            rr.log("timing/left_interval_ms", rr.Scalars((tl - self.prev[0]) * 1000.0))
            if tr is not None and self.prev[1] is not None:
                rr.log("timing/right_interval_ms", rr.Scalars((tr - self.prev[1]) * 1000.0))
        self.prev = (tl, tr)
        rr.log("stamps", rr.TextDocument(stamps_text(n, tl, tr, t_recv, skew, t0)))
        rr.log("status", rr.TextDocument(f"{status}\n{k_views} pairs kept"))

    def coverage(self, views, size):
        rr = self.rr
        h = np.zeros((size[1], size[0]), np.float32)
        for v in views:
            for x, y in v["l"]:
                cv2.circle(h, (int(x), int(y)), 14, 1.0, -1)
        col = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        for i in (1, 2):
            cv2.line(col, (size[0] * i // 3, 0), (size[0] * i // 3, size[1]), (255, 255, 255), 1)
            cv2.line(col, (0, size[1] * i // 3), (size[0], size[1] * i // 3), (255, 255, 255), 1)
        rr.log("coverage", rr.Image(col[:, :, ::-1]))

    def result(self, c, board, session, report_text):
        rr = self.rr
        rr.set_time("frame", sequence=len(session.views) + 1)
        rr.log("report", rr.TextDocument(report_text))
        rr.log("calibration/view_errors_px", rr.BarChart(np.array(c["view_errors"], np.float32)))
        rr.log("world", rr.ViewCoordinates.RDF, static=True)
        size = (int(c["size"][0]), int(c["size"][1]))
        last = session.views[-1]
        cen_r = (-c["R"].T @ c["T"]).ravel()
        for name, K, tr_, rot, img in (("left", c["K1"], np.zeros(3), np.eye(3), last["left"]), ("right", c["K2"], cen_r, c["R"].T, last["right"])):
            rr.log(f"world/{name}", rr.Transform3D(translation=tr_, mat3x3=rot), static=True)
            rr.log(f"world/{name}", rr.TransformAxes3D(0.03), static=True)
            rr.log(f"world/{name}/camera", rr.Pinhole(image_from_camera=K, resolution=list(size), image_plane_distance=0.08), static=True)
            rr.log(f"world/{name}/camera", rr.Image(img[:, :, ::-1]).compress(jpeg_quality=70), static=True)
        rr.log("world/baseline", rr.LineStrips3D([[np.zeros(3), cen_r]], colors=[(255, 160, 0)], radii=0.001, labels=[f"{c['baseline'] * 1000:.2f} mm"]), static=True)
        W, H = board.size_m
        outline = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0], [0, 0, 0]], np.float32)
        for i, v in enumerate(session.views):
            ok, rv, tv = cv2.solvePnP(board.obj[v["ids"]], v["l"], c["K1"], c["D1"])
            if ok:
                pts = (cv2.Rodrigues(rv)[0] @ outline.T + tv).T
                rr.log(f"world/boards/view_{i:02d}", rr.LineStrips3D([pts], colors=[(80, 200, 120)], radii=0.0008), static=True)
        m1 = cv2.initUndistortRectifyMap(c["K1"], c["D1"], c["R1"], c["P1"], size, cv2.CV_16SC2)
        m2 = cv2.initUndistortRectifyMap(c["K2"], c["D2"], c["R2"], c["P2"], size, cv2.CV_16SC2)
        rl, rrr = cv2.remap(last["left"], *m1, cv2.INTER_LINEAR), cv2.remap(last["right"], *m2, cv2.INTER_LINEAR)
        both = np.hstack([rl, rrr])
        for y in range(0, size[1], 30):
            cv2.line(both, (0, y), (both.shape[1], y), (0, 255, 0), 1)
        rr.log("rectified", rr.Image(both[:, :, ::-1]).compress(jpeg_quality=85), static=True)


# ---------------------------------------------------------------- main
def make_report(c, a, session, depth_errs, skews):
    lines = [f"calibration from {c['n_views']} of {len(session.views)} kept pairs",
             f"baseline {c['baseline'] * 1000:.2f} mm   focal {c['focal']:.1f} px   principal point ({c['cx']:.1f}, {c['cy']:.1f})",
             f"stereo reprojection error {c['rms_stereo']:.3f} px   row error after rectification {c['epipolar_err_px']:.3f} px   verdict {verdict(c)}",
             check_baseline(c, a.square_mm, a.baseline_mm)]
    x, y, z = (-c["R"].T @ c["T"]).ravel() * 1000.0
    lines.append(f"right lens seen from the left lens: {x:+.1f} mm sideways, {y:+.1f} mm down, {z:+.1f} mm forward "
                 f"(the case puts the left lens about 2-3 mm deeper, so expect about +2 to +3 mm forward; this number is only good to about +-1 mm with close, tilted views and +-2 mm without)")
    ang = board_extent_check(c)
    if ang > 10:
        lines.append(f"WARNING: the right camera is turned {ang:.0f} deg against the left. Cameras mounted upside down? Retry with --rotate-left 180 or --rotate-right 180.")
    if c["swapped"]:
        lines.append("WARNING: the cameras are SWAPPED: the picture called 'right' is physically on the left. The rig is probably mounted turned 180 deg. Run again with --flip180 "
                     "(turns both pictures 180 deg and swaps them, exactly what source.flip180 does in the pipeline), keeping your --rotate-left/--rotate-right.")
    if depth_errs:
        lines.append(f"board distance, stereo against board size: median difference {np.median(depth_errs) * 100:.1f}% (worst pair {np.max(depth_errs) * 100:.1f}%)")
    cells = coverage(session.views, tuple(int(v) for v in c["size"]))
    lines.append(f"coverage (corners per 3x3 area of the picture):\n{cells}")
    if (cells == 0).sum() >= 3:
        lines.append("WARNING: the board never reached " + f"{int((cells == 0).sum())} of 9 picture areas; the lens edges are not constrained. Capture more pairs in those areas.")
    if skews:
        lines.append(f"left/right timestamp offset of the kept pairs: median {np.median(skews):+.2f} ms, worst {max(skews, key=abs):+.2f} ms")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-board", action="store_true", help="write data/charuco_board_letter.png (print at 100%%) and exit")
    ap.add_argument("--squares-x", type=int, default=7)
    ap.add_argument("--squares-y", type=int, default=10)
    ap.add_argument("--square-mm", type=float, default=25.0, help="MEASURED side of a printed square")
    ap.add_argument("--marker-mm", type=float, default=18.0, help="MEASURED side of a printed marker")
    ap.add_argument("--dict", default="4x4", choices=sorted(DICTS), dest="dictionary")
    ap.add_argument("--baseline-mm", type=float, default=42.0, help="distance between the two lens centres, measured with the caliper: the result is checked against it")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--min-corners", type=int, default=12, help="board corners one camera must see")
    ap.add_argument("--min-shift", type=float, default=40.0, help="px the board must move between kept pairs")
    ap.add_argument("--max-skew-ms", type=float, default=8.0, help="largest left/right capture-time difference accepted while the board moves (a still board is accepted at any offset)")
    ap.add_argument("--rational", action="store_true")
    ap.add_argument("--fix-k3", action="store_true")
    ap.add_argument("--recording", help="calibrate from a folder written by record_pairs")
    ap.add_argument("--images", help="calibrate from a folder of NNN_L.png / NNN_R.png")
    ap.add_argument("--rotate-left", type=int, default=0, choices=[0, 90, 180, 270], help="live only: turn the left picture (180 if that camera is upside down)")
    ap.add_argument("--rotate-right", type=int, default=0, choices=[0, 90, 180, 270])
    ap.add_argument("--flip180", action="store_true", help="turn both pictures 180 deg and swap left/right (after --rotate-*), like source.flip180 in config/pipeline.toml; use it when the report says the cameras are swapped")
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--pairs-dir", default="data/charuco_pairs", help="live/recording: the kept pairs are saved here (re-run later with --images)")
    ap.add_argument("--out", default="data/stereo_calib_charuco")
    ap.add_argument("--save-rrd", help="also write the Rerun session to this file")
    ap.add_argument("--no-viewer", action="store_true", help="do not open the Rerun viewer")
    a = ap.parse_args(argv)

    board = Board(a.squares_x, a.squares_y, a.square_mm, a.marker_mm, a.dictionary, a.min_corners)
    if a.make_board:
        os.makedirs("data", exist_ok=True)
        from PIL import Image
        Image.fromarray(board.printable()).save("data/charuco_board_letter.png", dpi=(300, 300))
        print(f"data/charuco_board_letter.png: {a.squares_x}x{a.squares_y} squares of {a.square_mm:g} mm, markers {a.marker_mm:g} mm, on letter paper. Print at 100% (actual size), "
              "glue it flat on a rigid board, then measure one square and one marker with the caliper and pass those numbers.")
        return
    session = Session(board, a.pairs, a.min_shift, a.max_skew_ms, need_stable=0 if a.images else 3)
    view = View(a)
    live = not (a.recording or a.images)
    src = image_pairs(a) if a.images else recording_pairs(a) if a.recording else live_pairs(a)
    if not a.images:
        os.makedirs(a.pairs_dir, exist_ok=True)
        for f in glob.glob(os.path.join(a.pairs_dir, "*.png")):
            os.remove(f)
    rotation = (a.rotate_left, a.rotate_right) if live else (0, 0)
    if a.images and os.path.exists(os.path.join(a.images, "rotation.json")):
        rotation = tuple(json.load(open(os.path.join(a.images, "rotation.json"))))
    skews = []
    try:
        for n, left, right, tl, tr, t_recv, skew in src:
            if a.flip180:                                               # the pipeline does the same before rectifying; the right-minus-left offset changes sign
                left, right, tl, tr, skew = cv2.rotate(right, cv2.ROTATE_180), cv2.rotate(left, cv2.ROTATE_180), tr, tl, (None if skew is None else -skew)
            dl, dr, status, kept = session.offer(left, right, skew)
            if kept:
                session.keep(left, right, dl, dr, seq=n, t_left=tl, t_right=tr, skew_ms=skew)
                k = len(session.views) - 1
                if not a.images:
                    cv2.imwrite(os.path.join(a.pairs_dir, f"{k:03d}_L.png"), left)
                    cv2.imwrite(os.path.join(a.pairs_dir, f"{k:03d}_R.png"), right)
                if skew is not None:
                    skews.append(skew)
                print(status, flush=True)
            view.pair(n, left, right, dl, dr, tl, tr, t_recv, skew, status, len(session.views))
            if kept:
                view.coverage(session.views, session.size)
            if len(session.views) >= a.pairs:
                break
    except KeyboardInterrupt:
        print("\nstopped: calibrating with the pairs kept so far", flush=True)
    if not a.images and live:
        json.dump(list(rotation), open(os.path.join(a.pairs_dir, "rotation.json"), "w"))
    try:
        c = session.calibrate(a.rational, a.fix_k3)
    except ValueError as e:
        print(f"cannot calibrate: {e}")
        return
    c["view_errors"] = [float(np.mean(np.linalg.norm(
        cv2.projectPoints(board.obj[v["ids"]], *cv2.solvePnP(board.obj[v["ids"]], v["l"], c["K1"], c["D1"])[1:], c["K1"], c["D1"])[0].reshape(-1, 2) - v["l"], axis=1)))
        for v in session.views]
    meta = save(c, a.out, a.square_mm, rotation)
    depth_errs = depth_check(c, board, session.views)
    text = make_report(c, a, session, depth_errs, skews)
    print("\n" + text)
    json.dump({"board": board.spec, "measured_baseline_mm": a.baseline_mm, "meta": meta,
               "pairs": [{"seq": v["seq"], "t_left": v["t_left"], "t_right": v["t_right"], "right_minus_left_ms": v["skew_ms"], "corners": int(len(v["ids"]))} for v in session.views]},
              open(a.out + "_charuco.json", "w"), indent=2)
    view.result(c, board, session, text)
    print(f"\nsaved {a.out}.npz / .json / _charuco.json. To use it: source.calib = \"{a.out}.npz\" in config/pipeline.toml (keep the --rotate-left/right you used in mind).")


if __name__ == "__main__":
    main()

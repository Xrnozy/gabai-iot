"""Shows the feature points the Pi tracks (`sender --klt`) on top of the pair they belong to, in Rerun.

  py -m gabai.tools.klt_view --port 5600
  py -m gabai.tools.klt_view --ids --points-only      (labels with feature ids; no pictures, only the points and counts)

It listens like any other Gabai viewer, so close the pipeline or other viewers first (one program per port). The pictures are the raw sent pictures, which is
the frame the Pi's coordinates are in; the rectified pictures of the pipeline have other coordinates.

Entities:  klt/left, klt/right          the picture, with klt/left/tracked, /new, /lost drawn over it (green, yellow, red)
           klt/features/left, /right    every point as its own entity (class 0 tracked, 1 new, 2 lost), also without the picture
           klt/count/*                  tracked, new, lost, in the right picture, and the Pi's tracking time per pair
Time:      "frame" (sequence number) and "t" (the Pi's capture time), the same for the picture and its points.
"""
import argparse
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from gabai.net import protocol as P
from gabai.pc.receiver import LiveReceiver

COLOR = {P.KLT_TRACKED: (60, 220, 60), P.KLT_NEW: (255, 220, 0), P.KLT_LOST: (255, 60, 60)}
NAME = {P.KLT_TRACKED: "tracked", P.KLT_NEW: "new", P.KLT_LOST: "lost"}
WAIT_S = 0.25                                       # the points follow their pair by a few ms; longer than this means the sender has no --klt


def blueprint(images):
    left, right = (rrb.Spatial2DView(origin="klt/left" if images else "klt/features/left", name="left"),
                   rrb.Spatial2DView(origin="klt/right" if images else "klt/features/right", name="right"))
    return rrb.Blueprint(rrb.Vertical(rrb.Horizontal(left, right), rrb.TimeSeriesView(origin="klt/count", name="feature counts and Pi tracking time"), row_shares=[3, 1]),
                         rrb.TimePanel(timeline="frame"))


def log_points(path, class_path, xy, status, ids, labels):
    """xy in OpenCV convention; Rerun puts pixel centres at half integers. Points of one status share a child entity under the picture."""
    ok = np.isfinite(xy[:, 0])
    pos = xy + 0.5
    for s, name in NAME.items():
        m = ok & (status == s)
        sub = f"{path}/{name}"
        if m.any():
            rr.log(sub, rr.Points2D(pos[m], colors=COLOR[s], radii=2.5 if s != P.KLT_LOST else 3.5, labels=[str(i) for i in ids[m]] if labels else None))
        else:
            rr.log(sub, rr.Clear(recursive=False))
    if ok.any():
        rr.log(class_path, rr.Points2D(pos[ok], class_ids=status[ok].astype(int), colors=[COLOR[int(s)] for s in status[ok]], radii=2.5,
                                       labels=[str(i) for i in ids[ok]] if labels else None))
    else:
        rr.log(class_path, rr.Clear(recursive=False))


def log_pair(pair, k, a, t0):
    rr.set_time("frame", sequence=pair.seq)
    rr.set_time("t", duration=pair.t_capture - t0)
    if not a.points_only:
        for side, img in (("left", pair.left), ("right", pair.right)):
            rr.log(f"klt/{side}", rr.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=a.quality))
    if k is None:
        for side in ("left", "right"):
            for name in NAME.values():
                rr.log(f"klt/{side}/{name}", rr.Clear(recursive=False))
            rr.log(f"klt/features/{side}", rr.Clear(recursive=False))
        return
    log_points("klt/left", "klt/features/left", k.xy, k.status, k.ids, a.ids)
    live = k.status != P.KLT_LOST
    if k.xy_right is not None:
        log_points("klt/right", "klt/features/right", np.where(live[:, None], k.xy_right, np.nan), k.status, k.ids, a.ids)
    n = {NAME[s]: int((k.status == s).sum()) for s in NAME}
    for name, v in (("tracked", n["tracked"] + n["new"]), ("new", n["new"]), ("lost", n["lost"]), ("pi_ms", k.klt_ms),
                    ("in_right", int(np.isfinite(k.xy_right[live, 0]).sum()) if k.xy_right is not None else None)):
        if v is not None:
            rr.log(f"klt/count/{name}", rr.Scalars(float(v)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--token", default="gabai")
    ap.add_argument("--ids", action="store_true", help="write the feature id next to each point")
    ap.add_argument("--points-only", action="store_true", help="do not log the pictures, only points and counts (lighter for Rerun)")
    ap.add_argument("--quality", type=int, default=70, help="JPEG quality of the pictures sent to Rerun")
    a = ap.parse_args()
    rr.init("gabai_klt", spawn=True)
    rr.send_blueprint(blueprint(not a.points_only))
    rx = LiveReceiver(a.port, a.token)
    rx.start()
    print(f"waiting for the Pi on port {rx.port} (start it with --klt) ...", flush=True)
    seq, t0, last_msg, seen = 0, None, time.monotonic(), False
    try:
        while True:
            pair = rx.latest()
            if pair is None or pair.seq == seq:
                time.sleep(0.003)
                continue
            seq = pair.seq
            t0 = pair.t_capture if t0 is None else t0
            end = time.monotonic() + (WAIT_S if seen else 0.05)       # until a first point set has arrived, do not hold pictures back for a sender without --klt
            k = rx.klt_for(pair.t_capture)
            while k is None and time.monotonic() < end:
                time.sleep(0.002)
                k = rx.klt_for(pair.t_capture)
            seen = seen or k is not None
            log_pair(pair, k, a, t0)
            if k is None and time.monotonic() - last_msg > 5:
                print("no feature points arrive with the pairs: is the sender running with --klt?", flush=True)
                last_msg = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        rx.stop()


if __name__ == "__main__":
    main()

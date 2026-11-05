"""The TSDF skips the depth of nodes whose picture is blurrier than most recent ones, and keeps them stored."""
import cv2
import numpy as np

from rtabmap_lab.tsdf import TsdfMap


def test_blurred_node_is_kept_but_not_fused():
    rng = np.random.default_rng(0)
    t = TsdfMap((300.0, 300.0, 80.0, 60.0), (160, 120), threaded=False)
    depth = np.full((120, 160), 1500, np.uint16)
    for i in range(20):
        t.add_node(i, depth, rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    sharp = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    t.add_node(20, depth, cv2.GaussianBlur(sharp, (0, 0), 4))
    assert 20 in t.blurred and 20 in t.frames and len(t.blurred) <= 6                # the clearly blurred one is skipped, most are fused
    t.close()
    off = TsdfMap((300.0, 300.0, 80.0, 60.0), (160, 120), threaded=False, blur_keep=1.0)
    off.add_node(0, depth, cv2.GaussianBlur(sharp, (0, 0), 4))
    assert not off.blurred
    off.close()

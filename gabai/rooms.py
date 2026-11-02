"""Room layout (spec phase 6). Only the spec's backup is implemented: RANSAC planes.

layout_from_planes() finds the floor, ceiling and walls in a gravity-aligned cloud (z up, metres) by repeated RANSAC plane fits (Open3D). It
reports each plane's kind from its normal (horizontal plane at the bottom = floor, at the top = ceiling, vertical plane = wall, anything leaning
more than `tolerance_deg` = other), its height or offset, point count and extent, and the overall room size. It does not guess a door, a window
or a box; those need SceneScript.

SceneScript (Meta, non-commercial) and ShapeR are NOT available here: the research code is not installed and the only artefact present
(models/scenescript_model_ase.ckpt) has no loader in this project. scenescript_layout() and shaper_mesh() raise RoomUnavailable saying so, so
nothing downstream can mistake a missing model for an empty room. Both were also built around Aria glasses data and are expected to be weaker on
webcam stereo.
"""
import numpy as np

MIN_POINTS = 200
PLANE_DIST_M = 0.04
MIN_PLANE_POINTS = 300
MAX_PLANES = 10


class RoomUnavailable(RuntimeError):
    pass


def scenescript_layout(points):
    raise RoomUnavailable("SceneScript is not available: its code is not installed in this project and no loader exists for the checkpoint in models/. "
                          "Use layout_from_planes() (RANSAC planes) instead.")


def shaper_mesh(images, poses=None):
    raise RoomUnavailable("ShapeR is not available: its code and weights are not installed. No object meshes are produced.")


def layout_from_planes(points, tolerance_deg=10.0, seed=0):
    import open3d as o3d
    p = np.asarray(points, float)
    out = {"source": "RANSAC planes", "scenescript": "not used", "planes": [], "note": "", "floor_z": None, "height_m": None, "size_xy_m": None}
    if len(p) < MIN_POINTS:
        out["note"] = f"too few points ({len(p)}; need {MIN_POINTS})"
        return out
    o3d.utility.random.seed(seed)
    rest = p.copy()
    for _ in range(MAX_PLANES):
        if len(rest) < MIN_PLANE_POINTS:
            break
        c = o3d.geometry.PointCloud()
        c.points = o3d.utility.Vector3dVector(rest)
        model, inl = c.segment_plane(PLANE_DIST_M, 3, 800)
        inl = np.asarray(inl)
        if len(inl) < MIN_PLANE_POINTS:
            break
        a, b, cz, d = model
        n = np.array([a, b, cz])
        n /= np.linalg.norm(n)
        tilt = np.degrees(np.arccos(abs(n[2])))                      # 0 = horizontal plane, 90 = vertical plane
        pts = rest[inl]
        if tilt <= tolerance_deg:
            kind = "horizontal"
        elif tilt >= 90 - tolerance_deg:
            kind = "wall"
        else:
            kind = "other"
        out["planes"].append({"kind": kind, "normal": n.tolist(), "point_count": int(len(inl)), "z": float(np.median(pts[:, 2])),
                              "extent_min": pts.min(0).tolist(), "extent_max": pts.max(0).tolist(), "tilt_deg": round(float(tilt), 1)})
        keep = np.ones(len(rest), bool)
        keep[inl] = False
        rest = rest[keep]
    hz = [q for q in out["planes"] if q["kind"] == "horizontal"]
    if hz:
        low, high = min(hz, key=lambda q: q["z"]), max(hz, key=lambda q: q["z"])
        low["kind"] = "floor"
        if high is not low and high["z"] - low["z"] > 1.8:
            high["kind"] = "ceiling"
            out["height_m"] = float(high["z"] - low["z"])
        for q in hz:
            if q["kind"] == "horizontal":
                q["kind"] = "other"                                     # a table top or a shelf, not part of the room shell
        out["floor_z"] = float(low["z"])
    walls = [q for q in out["planes"] if q["kind"] == "wall"]
    if walls:
        lo = np.min([q["extent_min"] for q in walls], axis=0)
        hi = np.max([q["extent_max"] for q in walls], axis=0)
        out["size_xy_m"] = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
    if not out["planes"]:
        out["note"] = "no plane with enough points was found"
    return out

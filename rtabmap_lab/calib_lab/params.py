"""The editable stereo calibration of the lab, and the plain-language explanation of every number in it.

CalibParams is a flat dict of named numbers taken from a calibration .npz (K1/D1/K2/D2/R/T), in the units a person reads (pixels, degrees, millimetres). Rect turns a CalibParams into the
rectification maps and the camera model RTAB-Map is given (f, cx, cy, baseline); with the unchanged numbers it is exactly gabai.tools.calibrate_stereo.Rectifier (tests/test_calib_lab.py).
Nothing here touches a file on disk except from_npz (read) and save/load of the lab's own JSON.

Names follow the arrays in the .npz: K1 = left camera matrix, D1 = left distortion, K2/D2 = right, R and T = pose of the RIGHT camera relative to the LEFT (R as a rotation vector in degrees,
T in mm). `baseline_mm` is |T| as a convenience (setting it scales T). Derived numbers (rect.*) are what the rectification makes of them: read-only.
"""
import hashlib
import json
import time

import cv2
import numpy as np

FORMAT = "gabai-calib-lab-1"
INTRINSIC, DISTORTION, EXTRINSIC, RECT = "intrinsic", "distortion", "extrinsic", "rectification"
GROUPS = {INTRINSIC: "Intrinsics (what each lens does)", DISTORTION: "Lens distortion", EXTRINSIC: "Extrinsics (how the two cameras sit)", RECT: "Rectification choice"}

# ---- the explanations --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# what: what it is. changes: what moving it does. why: why it matters. used: which stage reads it. behavior: too high / too low / wrong. guidance: what to watch while you tune it.
DOC = {
    "fx": dict(
        label="focal length x (px)",
        what="How strongly the lens magnifies, measured in pixels along the picture's width. A longer focal length means a narrower view and a bigger image of a distant thing.",
        changes="Scales every point's horizontal distance from the picture centre: the picture is stretched or squeezed sideways. After rectification the shared focal length f (below) is built from the left and right values, and depth Z = f x baseline / disparity uses it.",
        why="A wrong focal length gives a wrong scale of ALL distances (depth is proportional to it) and bends flat walls into a slight curve, because the sideways position of a point is Z x (u - cx) / f.",
        used="initUndistortRectifyMap (rectification), then the depth conversion, the point cloud, the TSDF volume and RTAB-Map's own stereo model (fx, fy).",
        behavior="Too high: the scene looks smaller than it is, distances come out too LARGE and rooms look stretched in depth. Too low: distances come out too small. A left/right mismatch shows up as rows that no longer line up towards the picture edges.",
        guidance="Look at the rows: matched features should sit on the same row everywhere (the 'row error' number goes toward 0 and its slope with height flattens). Look at a flat wall in the TSDF: it should be flat, not bowed. Check a tape-measured distance in the 3D view."),
    "fy": dict(
        label="focal length y (px)",
        what="The same magnification, measured along the picture's height. For square pixels it equals fx within a fraction of a percent.",
        changes="Scales every point's vertical distance from the picture centre. Together with fx it sets the picture's aspect ratio.",
        why="If fy differs from fx the picture is stretched one way. A left/right difference in fy puts the same point on different rows in the two pictures, which breaks stereo matching.",
        used="Rectification maps, then RTAB-Map (fy) and the point cloud (y = Z (v - cy) / fy).",
        behavior="Too high or low on ONE camera: vertical misalignment that grows away from the centre row (matches stop lying on one row at the top and bottom). On both: floors and ceilings tilt in the 3D view.",
        guidance="Watch the 'row error vs height' slope in the quality panel: it should be 0. A tilting floor in the TSDF that is level in reality points here or at the rotation."),
    "cx": dict(
        label="principal point x (px)",
        what="The pixel column where the lens axis meets the sensor: the optical centre. Not necessarily the middle of the picture.",
        changes="Slides the whole picture sideways inside the rectified view. The left and right values together set where zero-disparity (infinity) sits, because disparity is the difference between them.",
        why="An error of one pixel in cx of ONE camera adds a constant error to every disparity, which is a large depth error far away (at 3 m one pixel is about 31 cm with this rig).",
        used="Rectification maps; the rectified principal point (rect.cx) is what the depth, point cloud, TSDF and RTAB-Map use to turn a pixel into a ray.",
        behavior="Left cx and right cx disagree by +1 px: every disparity is 1 px off, so a flat wall is at the wrong distance and tilted sideways. Both wrong together: the map is shifted sideways a little, which is rarely noticed.",
        guidance="Compare a distance you know (a wall at a measured distance) in the 3D view and in the disparity image; adjust the RIGHT cx (or yaw) until it agrees. The row error does not react to cx, so the tape measure is the judge."),
    "cy": dict(
        label="principal point y (px)",
        what="The pixel row where the lens axis meets the sensor.",
        changes="Slides the picture up or down inside the rectified view. A left/right difference moves the right picture vertically against the left one.",
        why="Stereo matching searches along one row. A vertical offset of the right picture is the most damaging mistake: the matcher then finds the wrong match or none.",
        used="Rectification maps; rect.cy goes into depth, the point cloud, the TSDF and RTAB-Map.",
        behavior="Right cy off by 1 px: matches shift 1 row, valid disparity falls and noise rises. Both off together: the horizon in the map tilts slightly.",
        guidance="The 'row error (median)' in the quality panel is the direct judge: move right cy until it is 0. It should also raise the share of valid disparity."),
    "k1": dict(
        label="radial distortion k1",
        what="First term of the barrel/pincushion model: how much straight lines bend, growing with the square of the distance from the centre.",
        changes="Bends the picture radially: positive k1 pulls the edges in (barrel look in the raw picture); the rectification undoes it. Straightens walls and door frames.",
        why="This lens is wide: without a good model the edges are several pixels off, which is exactly where row alignment and depth go wrong first.",
        used="initUndistortRectifyMap (undistort + rectify in one step).",
        behavior="Wrong: straight lines curve in the rectified picture, walls bow in the TSDF, and row error grows toward the picture edges but stays small in the middle.",
        guidance="Look at a straight door frame or table edge in the rectified left picture: it must be straight up to the edge. Watch row error at the left and right ends of the picture (quality panel, 'edges vs centre')."),
    "k2": dict(
        label="radial distortion k2",
        what="Second radial term (distance to the 4th power): acts mostly near the picture corners.",
        changes="Fine-tunes the bend at the edges and corners without changing the middle.",
        why="With k1 it describes the lens curve. It is strongly linked to k1 and k3: moving one without the others trades one error for another.",
        used="initUndistortRectifyMap: the per-pixel lookup table that straightens this camera's picture before stereo matching.",
        behavior="Wrong: the corners are off by pixels while the centre stays right; wavy lines near the edges.",
        guidance="Change it in small steps and watch only the corners of the rectified picture and the row error at the picture edges."),
    "k3": dict(
        label="radial distortion k3",
        what="Third radial term (6th power): only matters at the very edge of wide lenses. Easily over-fitted.",
        changes="Changes the outermost ring of the picture strongly and the middle almost not at all.",
        why="Calibrations often give it a large value (here about 0.5 to 0.7) to fit the edge. A large k3 that the data does not support makes the corners swing wildly.",
        used="initUndistortRectifyMap: the per-pixel lookup table that straightens this camera's picture before stereo matching.",
        behavior="Too large: the extreme corners stretch or fold, rectified edges look smeared. Setting it to 0 trades edge accuracy for stability.",
        guidance="If the corners look stretched, try lowering it and check that the row error at the edges does not get worse."),
    "p1": dict(
        label="tangential distortion p1",
        what="Distortion from the lens not being exactly parallel to the sensor, in one direction (y-ish).",
        changes="Shears the picture slightly, more at the edges. It is 0 in this calibration (the calibration script holds it at 0, because a good lens-sensor mount does not need it).",
        why="Real tilt of lens against sensor is possible in a 3D-printed case; if present it shows as a one-sided edge error.",
        used="initUndistortRectifyMap: the per-pixel lookup table that straightens this camera's picture before stereo matching.",
        behavior="Non-zero values you did not measure are almost always wrong: they trade one edge error for another. Small values (under 0.001) have visible effect.",
        guidance="Leave at 0 unless row error is one-sided (e.g. only the right edge is off); then change in steps of 0.0002."),
    "p2": dict(
        label="tangential distortion p2",
        what="The same lens-against-sensor tilt effect as p1, in the other direction (mostly horizontal shear of the picture).",
        changes="Shifts points sideways by an amount that grows toward the picture edges, differently on the left and the right. It is 0 in this calibration.",
        why="A one-sided horizontal edge error moves the matched point of ONE camera only, which is a disparity error at that edge: a wall looks bent there.",
        used="The distortion model inside initUndistortRectifyMap, for the left or the right picture.",
        behavior="A non-zero value that the lens does not need makes the left and right edges disagree in opposite ways: one side of a flat wall comes out nearer than the other.",
        guidance="Leave at 0 unless a flat wall bends only on one side in the 3D view; then try steps of 0.0002 and watch the disparity at that edge."),
    "rot_x": dict(
        label="right camera rotation about x (deg), 'pitch'",
        what="How much the right camera is tilted up or down relative to the left camera (nodding), as the x part of the rotation vector R.",
        changes="Moves the right picture up or down as a whole. The rectification rotates both pictures to undo it.",
        why="Stereo needs matching rows. A pitch error of 0.1 deg is about 1 px of vertical offset at this focal length: enough to ruin matching.",
        used="stereoRectify (R1/R2), so every rectified picture; RTAB-Map, the TSDF and depth via the rectified pictures.",
        behavior="Wrong: a constant vertical offset between matched features (median row error is not 0), fewer valid disparities.",
        guidance="The median row error in the quality panel is the direct judge, and it should be 0 when this is right. Step 0.02 deg at a time."),
    "rot_y": dict(
        label="right camera rotation about y (deg), 'yaw / toe-in'",
        what="How much the right camera is turned left or right relative to the left camera (toe-in).",
        changes="Shifts every disparity by the same amount (about 10 px per degree here). It does not show up as a row error.",
        why="This is the hardest angle to calibrate without a target: it only changes how far things appear to be. A 0.1 deg error is about 1 px of disparity, about 31 cm at 3 m.",
        used="stereoRectify, then depth: Z = f x B / disparity.",
        behavior="Wrong: all distances are too large or too small, with an error that grows with distance; flat walls come out tilted about the vertical axis (one end nearer).",
        guidance="Row error cannot show this. Use a wall you measured: set the 'known distance' tool on a point of the wall, or adjust until a wall that is square to the camera shows no tilt in the TSDF."),
    "rot_z": dict(
        label="right camera rotation about z (deg), 'roll'",
        what="How much the right camera is twisted about its viewing direction relative to the left (a rolled horizon).",
        changes="Rotates the right picture. Matches move up on one side and down on the other: the row error depends on where you look sideways.",
        why="A 0.1 deg roll gives about 0.5 px vertical offset at the picture edge.",
        used="stereoRectify, so every rectified picture.",
        behavior="Wrong: row error that changes sign from left to right across the picture (the slope vs x in the quality panel is not 0).",
        guidance="Watch the quality panel's 'rows drift per 100 px sideways' go to 0 while you move this by 0.02 deg at a time; the sign tells you which way to go."),
    "tx": dict(
        label="translation x (mm)",
        what="The sideways distance from the left lens to the right lens, along the line joining them. This is the baseline (negative here because the right lens is on the right in the left camera's frame).",
        changes="Scales depth: Z = f x B / disparity. 1% more baseline gives 1% more distance everywhere, with no change to the shape.",
        why="It sets the real-world scale. Everything else can be perfect and the map still has the wrong size if this is wrong.",
        used="stereoRectify (P2 and the disparity-to-depth matrix Q), RTAB-Map's baseline, the depth conversion, the TSDF.",
        behavior="Too large: everything farther than reality. Too small: nearer. No effect on row error, on matching or on flatness, only on scale.",
        guidance="Only a known length can set it: pick a point on a wall at a tape-measured distance in the 3D view and use 'scale to this distance'. This rig measured 46.8 mm with two different boards (CAD says 42 mm)."),
    "ty": dict(
        label="translation y (mm)",
        what="How far the right lens sits above or below the left one. For a rig with both lenses level it is about 0 (here under 1 mm).",
        changes="Rectification turns the pair so this vanishes; the error that remains is a small tilt of the rectified pictures.",
        why="A real 1 mm height offset is harmless after rectification. A wrong value tilts the rectified pair.",
        used="stereoRectify, which turns both pictures so their rows line up (R1/R2); every rectified picture, and so the depth, depends on it.",
        behavior="Wrong by several mm: the rectified rows tilt and a slight vertical error appears that depends on distance.",
        guidance="Change it only with a reason (a measured offset); watch row error vs distance."),
    "tz": dict(
        label="translation z (mm)",
        what="How far the right lens sits forward or back of the left. In this rig the left camera is recessed about 2 to 3 mm in the case, so a small value is real.",
        changes="Rectification absorbs it; a wrong value rotates the rectified pair slightly.",
        why="A recessed lens is real here; ignoring it costs a little accuracy for near objects.",
        used="stereoRectify, which turns both pictures so their rows line up (R1/R2); every rectified picture, and so the depth, depends on it.",
        behavior="Wrong: close objects have a small row error that far objects do not.",
        guidance="Compare row error on near features (under 0.5 m) with far ones."),
    "baseline_mm": dict(
        label="baseline (mm), |T|",
        what="The distance between the two lens centres: the length of T. Setting it scales T and keeps its direction.",
        changes="Scales depth and the whole map by new/old, nothing else. A map built with another baseline is a different size, so RTAB-Map restarts.",
        why="Depth = f x baseline / disparity: the baseline is the ruler of the whole system. It cannot be recovered from the pictures alone.",
        used="stereoRectify (Q, P2), RTAB-Map's stereo model, depth, the TSDF noise model.",
        behavior="Too large: everything too far. Too small: too near. Depth noise (cm) falls as the baseline rises.",
        guidance="Set it from a distance you measured with a tape (3D view, 'scale to this distance'). Your two boards agree on about 46.8 mm, the CAD model says 42 mm."),
    "alpha": dict(
        label="rectification crop alpha (0..1)",
        what="After rectification the picture has curved black borders. alpha = 0 crops to the largest all-valid rectangle (no black); alpha = 1 keeps every source pixel (with black borders).",
        changes="Changes the rectified picture's focal length and principal point (rect.f, rect.cx, rect.cy) and how much of the field of view is kept. Not a physical parameter.",
        why="A smaller picture means a larger rect.f, and everything that uses f follows. The saved calibration was made with alpha = 0.",
        used="stereoRectify -> P1/P2 -> RTAB-Map, depth, TSDF.",
        behavior="Higher alpha: a wider view with black corners that give no depth. Lower: sharper, less field of view.",
        guidance="Keep 0 unless you need the corners. Compare the 'valid disparity share' and the size of the view."),
    # derived, read only
    "rect.f": dict(
        label="rectified focal length (px)", what="The one focal length both rectified pictures share (P1[0,0] = P2[0,0]): after rectification the two cameras are treated as identical.",
        changes="Read-only: it is built from fx, fy, the distortion and the crop alpha. Changing any of those moves it.",
        why="Z = f x B / disparity and every ray (x = Z (u - cx) / f) use this value, so a wrong f scales every distance by the same factor.", used="The depth conversion, the point cloud, the TSDF volume and RTAB-Map's camera model (fx, fy).",
        behavior="Too high: distances come out too large. Too low: too small. It is not edited directly because it only follows from the inputs.", guidance="Edit K1.fx / K2.fx or rect.alpha and watch this value follow; compare a tape-measured distance in the 3D view."),
    "rect.cx": dict(
        label="rectified principal point x (px)", what="The column where the lens axis lands in the rectified picture (P1[0,2]).", changes="Read-only: follows cx of both cameras, the rotation and alpha.",
        why="It turns a pixel column into a sideways direction, so every sideways coordinate in the map depends on it.", used="The depth conversion, the point cloud, the TSDF volume and RTAB-Map's camera model.",
        behavior="Wrong: the whole map is shifted sideways a little, and slightly tilted about the vertical axis for far points.", guidance="Edit K1.cx (and K2.cx) and watch it follow; it normally sits near the middle of the picture (320)."),
    "rect.cy": dict(
        label="rectified principal point y (px)", what="The row where the lens axis lands in the rectified picture (P1[1,2]).", changes="Read-only: follows cy of both cameras, the rotation and alpha.",
        why="It turns a pixel row into a vertical direction, so the height of everything in the map depends on it.", used="The depth conversion, the point cloud, the TSDF volume and RTAB-Map's camera model.",
        behavior="Wrong: floors and ceilings tilt slightly in the 3D view; the horizon of the map is off.", guidance="Edit K1.cy (and K2.cy) and watch it follow; it normally sits near the middle row (180)."),
    "rect.baseline_mm": dict(
        label="rectified baseline (mm)", what="The baseline after rectification, -P2[0,3] / P2[0,0]: the same length as |T|, but laid exactly along the picture's x axis.", changes="Read-only: it follows the translation (T.x_mm, T.y_mm, T.z_mm).",
        why="It is the ruler of the depth: Z = f x B / disparity, so 1% of baseline error is 1% of distance error everywhere.", used="The depth conversion, the TSDF noise model and RTAB-Map's stereo model (its odometry scale).",
        behavior="Too large: all distances too far; too small: too near. The map keeps its shape, only its size changes.", guidance="Change baseline_mm and check it matches here; set it from a tape-measured distance in the 3D view."),
    "rect.R1_deg": dict(
        label="left rectification rotation R1 (deg)", what="How far the rectification virtually turns the LEFT camera so that both image planes become parallel (the angle of R1).", changes="Read-only: it follows R and T.",
        why="Large values (over a few degrees) mean the cameras are not parallel or the calibration is suspect, and the rectified picture loses field of view.", used="initUndistortRectifyMap for the left picture.",
        behavior="A big value is not an error by itself, but it costs usable field of view and shows how strongly the calibration relies on the rotation.", guidance="Compare it with R2: together they cancel the relative rotation R between the cameras."),
    "rect.R2_deg": dict(
        label="right rectification rotation R2 (deg)", what="How far the rectification virtually turns the RIGHT camera so that both image planes become parallel (the angle of R2).", changes="Read-only: it follows R and T.",
        why="Together with R1 it splits the relative rotation between the cameras; a very uneven split can point at an off-centre rotation error.", used="initUndistortRectifyMap for the right picture.",
        behavior="A big value costs field of view; it is not wrong by itself.", guidance="Edit R.rx_deg / R.ry_deg / R.rz_deg and watch how R1 and R2 share the change."),
    "rect.P1": dict(
        label="projection matrix P1", what="The 3x4 matrix of the rectified LEFT camera: [f 0 cx 0; 0 f cy 0; 0 0 1 0].", changes="Read-only: it is rebuilt by stereoRectify from every intrinsic and extrinsic number.",
        why="It is the camera model of the rectified left picture, which is exactly what RTAB-Map and the TSDF are told.", used="Everything after rectification: depth, point cloud, TSDF, RTAB-Map's camera model.",
        behavior="Wrong entries mean wrong rays and a wrong scale in the whole map.", guidance="Read f, cx and cy off it at a glance and compare with the rect.* values above."),
    "rect.P2": dict(
        label="projection matrix P2", what="The 3x4 matrix of the rectified RIGHT camera: the same as P1, plus P2[0,3] = -f x B, which carries the baseline.", changes="Read-only: rebuilt by stereoRectify.",
        why="Its last entry is what makes disparity-to-depth possible: B = -P2[0,3] / f.", used="The Q matrix, and with it the disparity-to-depth conversion.",
        behavior="A wrong P2[0,3] is a wrong scale of every distance.", guidance="Divide P2[0,3] by -f: the result must equal the baseline you set, in metres."),
    "rect.Q": dict(
        label="disparity-to-depth matrix Q", what="The 4x4 matrix that turns (u, v, disparity, 1) into a homogeneous 3D point: reprojectImageTo3D uses it.", changes="Read-only: rebuilt by stereoRectify.",
        why="It is the single matrix that expresses 'depth from disparity' for this calibration: Q[2,3] is the focal length and Q[3,2] is -1 / baseline.", used="The depth-from-disparity step in OpenCV; RTAB-Map and the lab use the equivalent z = f B / d.",
        behavior="A wrong Q gives a wrong depth everywhere.", guidance="Check Q[2,3] equals rect.f and 1 / |Q[3,2]| equals the baseline in metres."),
}

# ---- the editable numbers -------------------------------------------------------------------------------------------------------------------------------------------------------------
# name, doc key, group, unit, slider span (+/- around the saved value), step. Left/right pairs share the text.
EDITABLE = []
for _cam, _k, _d in (("left", "K1", "D1"), ("right", "K2", "D2")):
    EDITABLE += [(f"{_k}.fx", "fx", INTRINSIC, "px", 60.0, 0.1, _cam), (f"{_k}.fy", "fy", INTRINSIC, "px", 60.0, 0.1, _cam),
                 (f"{_k}.cx", "cx", INTRINSIC, "px", 25.0, 0.05, _cam), (f"{_k}.cy", "cy", INTRINSIC, "px", 25.0, 0.05, _cam),
                 (f"{_d}.k1", "k1", DISTORTION, "", 0.3, 0.001, _cam), (f"{_d}.k2", "k2", DISTORTION, "", 0.6, 0.002, _cam), (f"{_d}.k3", "k3", DISTORTION, "", 1.0, 0.005, _cam),
                 (f"{_d}.p1", "p1", DISTORTION, "", 0.01, 0.0001, _cam), (f"{_d}.p2", "p2", DISTORTION, "", 0.01, 0.0001, _cam)]
EDITABLE += [("R.rx_deg", "rot_x", EXTRINSIC, "deg", 3.0, 0.01, "right vs left"), ("R.ry_deg", "rot_y", EXTRINSIC, "deg", 3.0, 0.01, "right vs left"), ("R.rz_deg", "rot_z", EXTRINSIC, "deg", 3.0, 0.01, "right vs left"),
             ("T.x_mm", "tx", EXTRINSIC, "mm", 15.0, 0.05, "right vs left"), ("T.y_mm", "ty", EXTRINSIC, "mm", 8.0, 0.05, "right vs left"), ("T.z_mm", "tz", EXTRINSIC, "mm", 8.0, 0.05, "right vs left"),
             ("baseline_mm", "baseline_mm", EXTRINSIC, "mm", 15.0, 0.05, "right vs left"), ("rect.alpha", "alpha", RECT, "", 1.0, 0.01, "both")]
NAMES = [e[0] for e in EDITABLE]
SPEC = {e[0]: dict(name=e[0], doc=e[1], group=e[2], unit=e[3], span=e[4], step=e[5], camera=e[6]) for e in EDITABLE}
DERIVED = ["rect.f", "rect.cx", "rect.cy", "rect.baseline_mm", "rect.R1_deg", "rect.R2_deg", "rect.P1", "rect.P2", "rect.Q"]


def explain(name):
    """The seven-part explanation of one parameter (editable or derived)."""
    key = SPEC[name]["doc"] if name in SPEC else name
    d = dict(DOC[key])
    d["name"] = name
    d["group"] = SPEC[name]["group"] if name in SPEC else RECT
    d["camera"] = SPEC[name]["camera"] if name in SPEC else "derived"
    d["editable"] = name in SPEC
    d["unit"] = SPEC[name]["unit"] if name in SPEC else ""
    if name in SPEC and SPEC[name]["camera"] in ("left", "right"):
        d["label"] = SPEC[name]["camera"] + " " + d["label"]
    return d


# ---- the model ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class CalibParams:
    def __init__(self, values, original=None, size=(640, 360), meta=None):
        self.values = {k: float(values[k]) for k in NAMES}
        self.original = dict(original) if original is not None else dict(self.values)
        self.size, self.meta = tuple(int(v) for v in size), dict(meta or {})
        self._rect = None

    # -- construction ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_npz(cls, path):
        z = np.load(path)
        v = {}
        for cam, K, D in (("K1", z["K1"], z["D1"]), ("K2", z["K2"], z["D2"])):
            v.update({f"{cam}.fx": K[0, 0], f"{cam}.fy": K[1, 1], f"{cam}.cx": K[0, 2], f"{cam}.cy": K[1, 2]})
            d = np.asarray(D, float).ravel()
            dn = cam.replace("K", "D")
            v.update({f"{dn}.k1": d[0], f"{dn}.k2": d[1], f"{dn}.p1": d[2], f"{dn}.p2": d[3], f"{dn}.k3": d[4] if len(d) > 4 else 0.0})
        rv = cv2.Rodrigues(np.asarray(z["R"], float))[0].ravel() * 180.0 / np.pi
        T = np.asarray(z["T"], float).ravel() * 1000.0
        v.update({"R.rx_deg": rv[0], "R.ry_deg": rv[1], "R.rz_deg": rv[2], "T.x_mm": T[0], "T.y_mm": T[1], "T.z_mm": T[2], "baseline_mm": float(np.linalg.norm(T)), "rect.alpha": 0.0})
        meta = {"npz": str(path), "tail1": np.asarray(z["D1"], float).ravel()[5:].tolist(), "tail2": np.asarray(z["D2"], float).ravel()[5:].tolist(),
                "rotate_left": int(z["rotate_left"]) if "rotate_left" in z.files else 0, "rotate_right": int(z["rotate_right"]) if "rotate_right" in z.files else 0}
        return cls(v, size=tuple(z["size"]), meta=meta)

    def copy(self):
        c = CalibParams(self.values, self.original, self.size, self.meta)
        return c

    def with_(self, **kv):
        """A copy with some values changed. Keys are parameter names with '.' written as '__' (K1__cx=...) for keyword use; set() takes the real names."""
        c = self.copy()
        for k, v in kv.items():
            c.set(k.replace("__", "."), v)
        return c

    def set(self, name, value):
        """Change one value in place. baseline_mm scales T (its direction stays)."""
        if name not in self.values:
            raise KeyError(f"unknown calibration parameter {name!r}")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name}: {value} is not a number")
        if name == "baseline_mm":
            if value <= 1.0:
                raise ValueError("the baseline must be above 1 mm")
            T = np.array([self.values["T.x_mm"], self.values["T.y_mm"], self.values["T.z_mm"]])
            T = T * value / np.linalg.norm(T)
            self.values.update({"T.x_mm": T[0], "T.y_mm": T[1], "T.z_mm": T[2], "baseline_mm": value})
        else:
            self.values[name] = value
            if name.startswith("T."):
                self.values["baseline_mm"] = float(np.linalg.norm([self.values["T.x_mm"], self.values["T.y_mm"], self.values["T.z_mm"]]))
        if name.startswith("K") and name.endswith(("fx", "fy")) and value < 50:
            raise ValueError(f"{name}: a focal length under 50 px is not a camera")
        if name == "rect.alpha":
            self.values[name] = float(np.clip(value, 0.0, 1.0))
        self._rect = None
        return self

    def reset(self, name=None):
        for n in ([name] if name else NAMES):
            if n == "baseline_mm":
                continue
            self.values[n] = self.original[n]
        self.values["baseline_mm"] = float(np.linalg.norm([self.values["T.x_mm"], self.values["T.y_mm"], self.values["T.z_mm"]]))
        self._rect = None
        return self

    def changed(self, other=None, tol=1e-9):
        ref = other.values if other is not None else self.original
        return [n for n in NAMES if abs(self.values[n] - ref[n]) > tol]

    def fingerprint(self):
        """Identifies the exact numbers: two runs with the same fingerprint used the same calibration."""
        return hashlib.sha1(json.dumps([round(self.values[n], 9) for n in NAMES]).encode()).hexdigest()[:10]

    # -- the geometry ------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def arrays(self):
        v, m = self.values, self.meta
        K = lambda c: np.array([[v[f"{c}.fx"], 0, v[f"{c}.cx"]], [0, v[f"{c}.fy"], v[f"{c}.cy"]], [0, 0, 1.0]])
        D = lambda c, tail: np.array([v[f"{c}.k1"], v[f"{c}.k2"], v[f"{c}.p1"], v[f"{c}.p2"], v[f"{c}.k3"]] + list(tail))
        R = cv2.Rodrigues(np.array([v["R.rx_deg"], v["R.ry_deg"], v["R.rz_deg"]]) * np.pi / 180.0)[0]
        T = np.array([[v["T.x_mm"]], [v["T.y_mm"]], [v["T.z_mm"]]]) / 1000.0
        return K("K1"), D("D1", m.get("tail1", [])), K("K2"), D("D2", m.get("tail2", [])), R, T

    def rect(self):
        if self._rect is None:
            self._rect = Rect(self)
        return self._rect

    def derived(self):
        r = self.rect()
        return {"rect.f": r.f, "rect.cx": r.cx, "rect.cy": r.cy, "rect.baseline_mm": r.B * 1000.0,
                "rect.R1_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(r.R1)[0]))), "rect.R2_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(r.R2)[0]))),
                "rect.P1": r.P1.round(4).tolist(), "rect.P2": r.P2.round(4).tolist(), "rect.Q": r.Q.round(5).tolist()}

    # -- files -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def to_json(self, kind="manual", name="", notes=""):
        return {"format": FORMAT, "kind": kind, "name": name, "notes": notes, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "fingerprint": self.fingerprint(),
                "base": {"npz": self.meta.get("npz"), "size": list(self.size), "tail1": self.meta.get("tail1", []), "tail2": self.meta.get("tail2", [])},
                "values": self.values, "original": self.original}

    def to_npz(self, path, template=None):
        """Write this calibration as a .npz the GabAi pipeline reads (gabai.tools.calibrate_stereo.Rectifier, config source.calib): the template file's other entries (size, rotate_left/right,
        reprojection error ...) are kept, the geometry is replaced by these numbers. The template defaults to the file the session started from."""
        z = dict(np.load(template or self.meta["npz"]))
        K1, D1, K2, D2, R, T = self.arrays()
        r = self.rect()
        z.update(K1=K1, D1=D1.reshape(1, -1), K2=K2, D2=D2.reshape(1, -1), R=R, T=T, R1=r.R1, R2=r.R2, P1=r.P1, P2=r.P2, Q=r.Q, baseline=np.array(r.B), focal=np.array(r.f), cx=np.array(r.cx), cy=np.array(r.cy),
                 lab_fingerprint=np.array(self.fingerprint()), lab_changed=np.array(",".join(self.changed()) or "none"))
        np.savez(path, **z)
        return path

    @classmethod
    def from_json(cls, d):
        if d.get("format") != FORMAT:
            raise ValueError("not a calibration-lab file")
        missing = [n for n in NAMES if n not in d["values"]]
        if missing:
            raise ValueError(f"the file lacks {missing[:3]}")
        b = d.get("base", {})
        return cls(d["values"], d.get("original"), b.get("size", (640, 360)), {"npz": b.get("npz"), "tail1": b.get("tail1", []), "tail2": b.get("tail2", [])})


class Rect:
    """Rectification of a CalibParams: maps, projection matrices and the camera model RTAB-Map is given. apply() has the interface of gabai.tools.calibrate_stereo.Rectifier."""

    def __init__(self, p):
        K1, D1, K2, D2, R, T = p.arrays()
        self.size = p.size
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, self.size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=p.values["rect.alpha"])
        self.f, self.cx, self.cy = float(self.P1[0, 0]), float(self.P1[0, 2]), float(self.P1[1, 2])
        self.B = float(np.linalg.norm(T))
        self.m1 = cv2.initUndistortRectifyMap(K1, D1, self.R1, self.P1, self.size, cv2.CV_16SC2)
        self.m2 = cv2.initUndistortRectifyMap(K2, D2, self.R2, self.P2, self.size, cv2.CV_16SC2)

    def apply(self, left, right):
        return cv2.remap(left, *self.m1, cv2.INTER_LINEAR), cv2.remap(right, *self.m2, cv2.INTER_LINEAR)

    def camera(self, flip=False, k=1.0):
        """The camera model of the pictures RTAB-Map gets (what live.py puts into Bridge.init): f, B, cx, cy, size; turned 180 deg when flip."""
        w, h = int(round(self.size[0] * k)), int(round(self.size[1] * k))
        cx, cy = self.cx * k, self.cy * k
        if flip:
            cx, cy = (w - 1) - cx, (h - 1) - cy
        return {"f": self.f * k, "B": self.B, "cx": cx, "cy": cy, "size": (w, h), "metric": True}

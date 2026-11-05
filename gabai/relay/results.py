"""One small result per processed frame, in the shape both the phone and the caregiver read.

`frame_id` is the Pi frame sequence number, the same number the relay stores the picture under, so a viewer can tell which
picture a result describes. `hazards` and `speech` use the field names the phone's NavigationModule already understands.
"""
from gabai.speaker import phrase_for

ZONE_WIRE = {"left": "left", "ahead": "forward", "right": "right"}
SYSTEM_KEYS = ("gabai_stopped", "tracking_delayed", "reduced_accuracy")


def build_result(frame_id, t, alert, said):
    """alert: the zone result (key, distance_m or None) or None. said: phrases the pipeline spoke on this frame."""
    key, dist = (alert[0], alert[1]) if alert else (None, None)
    hazards = []
    if key in ZONE_WIRE:
        hazards.append({"zone": ZONE_WIRE[key], "distance_m": dist, "kind": "obstacle"})
    elif key in ("drop_ahead", "cant_see_ahead"):
        hazards.append({"zone": "forward", "distance_m": dist, "kind": key})
    if key is None:
        summary = "Path clear"
    elif key in ZONE_WIRE:
        summary = phrase_for(f"{key}_{_label(dist)}") if dist else f"Obstacle {key}"
    else:
        summary = phrase_for(key)
    out = {"mode": "pc", "t": t, "frame_id": frame_id, "alert": key, "hazards": hazards, "summary": summary, "detections": []}
    if said:
        out["speech"] = {"text": said[-1], "kind": "emergency" if key in ("drop_ahead", "cant_see_ahead") else "obstacle",
                         "interrupt": key in ("drop_ahead", "cant_see_ahead")}
    return out


def _label(dist):
    from gabai.speaker import distance_label
    return distance_label(dist)


# ---- SLAM walking guidance (mapper/navigate.py) -> the same result shape the phone already reads ----
SLAM_SPEECH_KIND = {"stop": "emergency", "path is clear": "clear", "careful": "caution", "lost tracking": "caution", "slow down": "steer",
                    "path is blocked": "emergency"}
REPEAT_S = {"stop": 3.0, "path is blocked": 4.0, "lost tracking": 6.0}          # a danger cue is repeated while it lasts; steering cues are said once


def _zone(heading_deg):
    a = abs(heading_deg)
    side = "left" if heading_deg > 0 else "right"
    return "forward" if a <= 15 else "slight_" + side if a <= 35 else side


class SlamGuide:
    """Turns each Navigator result into a relay result. Keeps the last spoken cue so the phone is told a cue once, not on every frame."""

    def __init__(self, hazard_m=1.5):
        self.hazard_m, self._cue, self._said_at = hazard_m, None, -1e9

    def result(self, frame_id, t, res, T_wc=None):
        import numpy as np
        from mapper.navigate import HEADINGS
        cue, free = res["cue"], res.get("free")
        hazards = []
        if free is not None:
            for th, d in zip(np.degrees(HEADINGS), free):
                if 0 <= d < self.hazard_m:
                    z = _zone(float(th))
                    if not any(h["zone"] == z and h["distance_m"] <= d for h in hazards):
                        hazards = [h for h in hazards if h["zone"] != z] + [{"zone": z, "distance_m": round(float(d), 1), "kind": "obstacle"}]
        dr = res.get("drops")
        if dr is not None and len(dr) >= 8:                          # a step down / hole ahead (mapper/navigate.py); a handful of low points is depth noise
            near = dr[np.argmin(np.hypot(dr[:, 0], dr[:, 1]))]
            d = float(np.hypot(near[0], near[1]))
            if d < self.hazard_m:
                hazards.insert(0, {"zone": _zone(float(np.degrees(np.arctan2(near[1], near[0])))), "distance_m": round(d, 1), "kind": "dropoff", "priority": True})
        hazards = res.get("person_hazards", []) + hazards           # people tracked in the video (mapper/dynamic.py), first so the phone says "Person approaching" rather than "Obstacle"; ghosts are excluded
        out = {"mode": "slam", "t": t, "frame_id": frame_id, "alert": cue, "hazards": hazards, "summary": cue.capitalize(), "detections": [],
               "heading_deg": res.get("heading_deg")}
        if T_wc is not None:
            out["pose"] = [round(float(v), 2) for v in np.asarray(T_wc)[:3, 3]]
        speak = cue != self._cue or (cue in REPEAT_S and t - self._said_at >= REPEAT_S[cue])
        if speak:
            self._cue, self._said_at = cue, t
            kind = SLAM_SPEECH_KIND.get(cue, "steer")
            out["speech"] = {"text": cue.capitalize(), "kind": kind, "interrupt": kind == "emergency"}
        return out

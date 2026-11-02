"""Per-frame evidence: what each pipeline stage produced, how long it took, which checks passed, and why a decision was made.

Every value that matters is tagged with what it IS, so a model's guess is never mistaken for a measurement:

    measured     read from a sensor or a clock (an IMU sample, a frame timestamp)
    model_prediction   output of a learned model (stereo disparity, monocular depth, a detector box)
    estimated    computed by our own geometry or filters from the above (metres from disparity, a Basalt pose)
    stale        was valid once, now older than its freshness limit
    unavailable  not there (no value is stored: never a made-up zero)
    validated    confirmed by an INDEPENDENT check that is named (left-right consistency, a tape measure, a second sensor)

A frame id (`fid`) ties the Pi log, the PC log and the trace together: it is the Pi's capture time in milliseconds (Pi monotonic
clock). Both machines can compute it without any extra message.
"""
import time
from contextlib import contextmanager

MEASURED, MODEL, ESTIMATED, STALE, UNAVAILABLE, VALIDATED = ("measured", "model_prediction", "estimated", "stale", "unavailable",
                                                             "validated")


def frame_id(t_capture):
    """Pi capture time (seconds, Pi monotonic clock) -> integer milliseconds. None stays None."""
    return None if t_capture is None else int(round(float(t_capture) * 1000))


def q(value, kind, **info):
    """A tagged value: {"v": value, "kind": kind, ...info}. A missing value is always `unavailable`, whatever kind was asked for."""
    if value is None:
        return {"v": None, "kind": UNAVAILABLE, "wanted": kind, **info}
    return {"v": value, "kind": kind, **info}


def validated(tagged, by):
    """Promote a tagged value to `validated`. Needs the name of the independent check; the original is not modified."""
    if not by or not str(by).strip():
        raise ValueError("a value can only be called validated by naming the independent check that confirmed it")
    out = dict(tagged)
    out.update(kind=VALIDATED, by=str(by), was=tagged.get("kind"))
    return out


class FrameTrace:
    """Everything known about one frame. Cheap to build: plain dicts, no copies of images or depth maps."""

    def __init__(self, fid, t_capture=None, t_recv=None, source="live"):
        self.fid, self.t_capture, self.t_recv, self.source = fid, t_capture, t_recv, source
        self.stages, self.checks, self.decisions, self.notes = {}, [], [], {}
        self.t0 = time.perf_counter()

    @contextmanager
    def stage(self, name):
        """with trace.stage("depth") as s: s["model"] = "las2-m"   ->  stages["depth"] = {"model": ..., "dur_ms": ...}"""
        data = self.stages.setdefault(name, {})
        t = time.perf_counter()
        try:
            yield data
        finally:
            data["dur_ms"] = round((time.perf_counter() - t) * 1000, 3)

    def put(self, name, **data):
        self.stages.setdefault(name, {}).update(data)

    def add_check(self, chk):
        self.checks.append(chk)
        return chk

    def decision(self, name, outcome, reasons=(), accepted=None, **evidence):
        self.decisions.append({"name": name, "outcome": outcome, "accepted": accepted, "reasons": list(reasons), **evidence})

    def failed(self):
        return [c.name for c in self.checks if c.status == "fail"]

    def interesting(self):
        """Worth writing even between the regular samples: a failed or suspect check, or any alert that was decided."""
        return any(c.status in ("fail", "suspect") for c in self.checks) or any(d.get("outcome") not in (None, "none") for d in self.decisions)

    def to_dict(self):
        return {"fid": self.fid, "source": self.source, "t_capture": self.t_capture, "t_recv": self.t_recv,
                "total_ms": round((time.perf_counter() - self.t0) * 1000, 3), "stages": self.stages,
                "checks": [c.as_dict() for c in self.checks], "failed_checks": self.failed(),
                "suspect_checks": [c.name for c in self.checks if c.status == "suspect"], "decisions": self.decisions, **self.notes}

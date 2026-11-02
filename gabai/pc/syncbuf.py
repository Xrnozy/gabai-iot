"""Server side of the Pi link: the three streams (stereo pairs, KLT points, IMU samples) held apart, handed to VIO/SLAM as complete, ordered measurement groups.

The display path (LiveReceiver.latest) keeps only the newest picture and skips the rest, like a live call. This buffer is the SLAM path and is separate from it: skipping a
picture on screen removes nothing here.

One group = one stereo pair + the KLT points the Pi computed from exactly that pair (matched by the pair's frame counter, never by time) + every IMU sample in the interval
since the previous group (not the nearest sample: the whole interval, in order). A group is released when
  - the IMU stream has reached the pair's time (the Pi sends IMU before the pair, so this is normally true on arrival), and
  - its KLT points are here (only when the sender sends KLT at all),
or when `imu_wait_s` / `klt_wait_s` have passed since the pair arrived: then it goes out marked incomplete and the pipeline knows what is missing. Nothing waits longer than that,
and a group whose parts are all here is released at once.

IMU samples are never dropped to catch up. Pictures are, but only past `max_lag_s` of backlog (the SLAM consumer is slower than the Pi sends): then the oldest pairs are skipped,
counted, and their IMU interval is handed over with the next pair, so the rotation across the skipped stretch is still integrated. max_lag_s <= 0 never skips (the queue is then
bounded by max_groups only, for offline or burst replay).
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Group:
    seq: int
    seq_r: object
    t: float                       # Pi capture time of the pair
    left: bytes                    # JPEG bytes, decoded by the consumer
    right: bytes
    dt_ms: object
    t_recv: float                  # this PC's clock
    klt: object = None             # KltPoints of this pair, or None
    imu: list = field(default_factory=list)      # rows (t, gx, gy, gz, ax, ay, az) with previous boundary < t <= this boundary, oldest first
    info: dict = field(default_factory=dict)


class SyncBuffer:
    def __init__(self, max_groups=120, max_lag_s=1.0, keep_lag_s=0.3, imu_wait_s=0.15, klt_wait_s=0.15, imu_max=20000, clock=time.monotonic):
        self.max_groups, self.max_lag, self.keep_lag, self.imu_wait, self.klt_wait, self.imu_max, self.clock = max_groups, max_lag_s, keep_lag_s, imu_wait_s, klt_wait_s, imu_max, clock
        self.lock, self.on = threading.Lock(), False
        self.q, self.imu, self.klt, self.gaps = deque(), deque(), {}, deque(maxlen=64)
        self.imu_last_t, self.imu_seen, self.klt_seen, self.last_seq, self.bound, self.pending_skip = None, False, False, None, None, 0
        self.c = dict(frames=0, dup_frames=0, pi_skipped=0, imu_rows=0, imu_missing=0, imu_dup=0, imu_dropped=0, overflow=0, recoveries=0, recovery_skipped=0,
                      released=0, incomplete=0, klt_missing=0, klt_orphans=0, klt_mismatch=0)

    def enable(self, **kw):
        """Start keeping groups (the standalone viewers never call this, so they keep no backlog)."""
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)
            self.on = True

    # -- producers: the receiver thread ------------------------------------------------------------------------------------------------------------------
    def add_frame(self, seq, seq_r, t, left, right, dt_ms, now=None):
        if not self.on:
            return
        now = self.clock() if now is None else now
        with self.lock:
            if seq is None:                                              # a sender without camera counters: count the pairs here (no gap can be seen)
                seq = 0 if self.last_seq is None else self.last_seq + 1
            if self.last_seq is not None and seq <= self.last_seq:       # the same pair twice (a resend) or an old one: never integrated twice
                self.c["dup_frames"] += 1
                return
            if self.last_seq is not None and seq > self.last_seq + 1:
                self.c["pi_skipped"] += seq - self.last_seq - 1          # pictures the Pi never sent (its newest-picture rule on a slow link)
            self.last_seq = seq
            self.q.append(Group(seq, seq_r, t, left, right, dt_ms, now))
            self.c["frames"] += 1
            while len(self.q) > self.max_groups:
                self.q.popleft()
                self.c["overflow"] += 1
                self.pending_skip += 1

    def add_klt(self, kp):
        if not self.on:
            return
        with self.lock:
            self.klt_seen = True
            self.klt[kp.frame_seq] = kp
            while len(self.klt) > 64:                                    # points for pairs that never came
                self.klt.pop(min(self.klt))
                self.c["klt_orphans"] += 1

    def add_imu(self, rows, missing=0):
        """rows: (t, gx, gy, gz, ax, ay, az), already free of repeated sample numbers. missing: samples the sequence numbers say were never received before these."""
        if not self.on:
            return
        with self.lock:
            self.imu_seen = True
            if missing:
                self.c["imu_missing"] += missing
                if rows and self.imu_last_t is not None:
                    self.gaps.append((self.imu_last_t, float(rows[0][0])))
            for r in rows:
                if self.imu_last_t is not None and r[0] <= self.imu_last_t:     # out of order or repeated by time: a sample is integrated once, in time order
                    self.c["imu_dup"] += 1
                    continue
                self.imu.append(r)
                self.imu_last_t = float(r[0])
                self.c["imu_rows"] += 1
            while len(self.imu) > self.imu_max:
                self.imu.popleft()
                self.c["imu_dropped"] += 1

    # -- consumer: the SLAM pipeline ---------------------------------------------------------------------------------------------------------------------
    def pop(self, delay_s=0.0, now=None):
        """The next group, or None when nothing is ready yet. delay_s: camera latency to remove from the pair's time when cutting its IMU interval (PairPrep's delay)."""
        now = self.clock() if now is None else now
        with self.lock:
            if not self.q:
                return None
            if self.max_lag > 0 and now - self.q[0].t_recv > self.max_lag:       # too far behind: skip the oldest PICTURES; the IMU interval stays and goes out with the next pair
                n = 0
                while len(self.q) > 1 and now - self.q[0].t_recv > self.keep_lag:
                    self.q.popleft()
                    n += 1
                if n:
                    self.c["recoveries"] += 1
                    self.c["recovery_skipped"] += n
                    self.pending_skip += n
            g = self.q[0]
            bound, waited = g.t - delay_s, now - g.t_recv
            imu_ok = (not self.imu_seen) or (self.imu_last_t is not None and self.imu_last_t >= bound)
            kp = self.klt.get(g.seq)
            if kp is not None and abs(kp.t_capture - g.t) > 1e-6:                  # the counter matched but the time does not: not this pair's points
                self.klt.pop(g.seq)
                self.c["klt_mismatch"] += 1
                kp = None
            klt_ok = (not self.klt_seen) or kp is not None
            if not ((imu_ok or waited >= self.imu_wait) and (klt_ok or waited >= self.klt_wait)):
                return None
            self.q.popleft()
            rows = []
            while self.imu and self.imu[0][0] <= bound:
                rows.append(self.imu.popleft())
            g.imu, g.klt = rows, self.klt.pop(g.seq, None) if kp is not None else None
            for k in [k for k in self.klt if k < g.seq]:                           # points for pairs older than this one will never be used
                del self.klt[k]
                self.c["klt_orphans"] += 1
            t0 = self.bound if self.bound is not None else (rows[0][0] if rows else bound)
            gap_s = sum(max(0.0, min(b, bound) - max(a, t0)) for a, b in self.gaps)
            if self.klt_seen and g.klt is None:
                self.c["klt_missing"] += 1
            if not imu_ok:
                self.c["incomplete"] += 1
            g.info = {"seq": g.seq, "seq_r": g.seq_r, "klt_seq": None if g.klt is None else g.klt.frame_seq, "klt_missing": self.klt_seen and g.klt is None,
                      "imu_n": len(rows), "imu_complete": imu_ok, "imu_gap_s": round(gap_s, 4),
                      "imu_margin_ms": None if self.imu_last_t is None else round((self.imu_last_t - bound) * 1000, 1),
                      "interval_s": None if self.bound is None else round(bound - self.bound, 4), "skipped_before": self.pending_skip,
                      "imu_span_ms": round((rows[-1][0] - rows[0][0]) * 1000, 1) if rows else 0.0,
                      "backlog": len(self.q), "imu_queue": len(self.imu), "wait_ms": round(waited * 1000, 1), "totals": dict(self.c)}
            self.bound, self.pending_skip = bound, 0
            self.c["released"] += 1
            return g

    def stats(self):
        with self.lock:
            return dict(self.c, groups=len(self.q), imu_queue=len(self.imu), klt_waiting=len(self.klt), last_seq=self.last_seq)

"""Which frame traces reach the log. Writing all of them would drown the real signal and cost disk, so:

  * every `every`-th frame is written (a regular sample, for latency and stage-time statistics),
  * every frame with a failed or suspect check, or an alert decision, is written too, at most `max_interesting_per_s` per second
    (the number of skipped ones is counted in `skipped_interesting` and reported in the log),
  * the last `ring` traces are always kept in memory so they can be dumped after something goes wrong (dump_ring).
"""
import json
import time
from collections import deque

from gabai.diag import log as dlog


class TraceSink:
    def __init__(self, every=15, max_interesting_per_s=5, ring=120):
        self.every, self.max_per_s = every, max_interesting_per_s
        self.ring = deque(maxlen=ring)
        self.n = 0
        self.skipped_interesting = 0
        self._win_start, self._win_n = time.monotonic(), 0
        self._log = dlog.get("trace")

    def offer(self, trace):
        """Give a finished FrameTrace. Returns True if it was written to the log."""
        d = trace.to_dict()
        self.ring.append(d)
        regular = self.n % self.every == 0
        self.n += 1
        interesting = trace.interesting()
        if not (regular or interesting):
            return False
        now = time.monotonic()
        if now - self._win_start >= 1.0:
            if self.skipped_interesting and self._win_n >= self.max_per_s:
                self._log.warning("trace.rate_limited", skipped_total=self.skipped_interesting, per_s=self.max_per_s)
            self._win_start, self._win_n = now, 0
        if not regular:
            if self._win_n >= self.max_per_s:
                self.skipped_interesting += 1
                return False
            self._win_n += 1
        self._log.info("frame.trace", sampled=regular, **d)
        return True

    def dump_ring(self, path):
        """Write the last traces (oldest first) as JSON lines. Use after an error, an alert you doubt, or on a key press."""
        with open(path, "w", encoding="utf-8") as f:
            for d in self.ring:
                f.write(json.dumps(dlog._clean(d), separators=(",", ":")) + "\n")
        return path

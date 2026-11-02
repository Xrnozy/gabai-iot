"""One alert queue with priorities, repeat suppression, and a link monitor for stale/dead input.

Priority 1 = system failure ... 5 = object info. A higher priority cuts off a lower one while it is active.
The same alert is not repeated within DEDUP_S unless it got closer.
"""
ACTIVE_S = 2.0
DEDUP_S = 3.0
STALE_AGE_S = 0.300
DEAD_AFTER_S = 1.0
DEAD_REPEAT_S = 5.0


class AlertQueue:
    def __init__(self):
        self._last = {}  # key -> (time, distance)
        self._active = (99, -1e9)  # (priority, time spoken)
        self.last_reason = None    # why the last push was refused (None when it was accepted)

    def push(self, kind, priority, key, distance, now):
        act_p, act_t = self._active
        if priority > act_p and now - act_t < ACTIVE_S:
            self.last_reason = f"lower priority ({priority}) than the alert still active ({act_p}, {now - act_t:.1f} s ago, active for {ACTIVE_S:.0f} s)"
            return None
        prev = self._last.get(key)
        if prev is not None and now - prev[0] < DEDUP_S:
            closer = distance is not None and prev[1] is not None and distance < prev[1]
            if not closer:
                self.last_reason = f"same alert '{key}' already said {now - prev[0]:.1f} s ago (repeat window {DEDUP_S:.0f} s) and not closer"
                return None
        self._last[key] = (now, distance)
        self._active = (priority, now)
        self.last_reason = None
        return key


class LinkMonitor:
    def __init__(self):
        self._last_dead = -1e9
        self._last_stale = -1e9

    def check(self, now, last_frame_t, frame_age):
        if now - last_frame_t > DEAD_AFTER_S:
            if now - self._last_dead >= DEAD_REPEAT_S:
                self._last_dead = now
                return "gabai_stopped"
            return None
        if frame_age > STALE_AGE_S and now - self._last_stale >= DEDUP_S:
            self._last_stale = now
            return "tracking_delayed"
        return None

"""Decides what to say, using the safety rules in the spec:
  * no frames for 1 s: "Gabai stopped, cable or camera lost", repeated every 5 s (works even though no frame arrives);
  * frames older than 300 ms: "tracking delayed";
  * turning fast: "reduced accuracy" (once per 10 s);
  * otherwise the front-zone obstacle, drop or "can't see ahead" result.
Obstacle speech can be switched off (obstacle_audio=False) when there is no floor model, because without one the floor
itself looks like an obstacle. System alerts are always spoken.
"""
from gabai.alerts import LinkMonitor
from gabai.speaker import phrase_for

REDUCED_REPEAT_S = 10.0


class AlertController:
    def __init__(self, speaker, obstacle_audio=True):
        self.speaker = speaker
        self.obstacle_audio = obstacle_audio
        self.monitor = LinkMonitor()
        self.last_frame_t = None
        self._last_reduced = -1e9
        self._prev_key = None
        self.decision = {}        # what the last on_frame() decided and why (read by the diagnostics)
        self._notes = []

    def _say(self, key, dist, now):
        clip = self.speaker.say(key, dist, now)
        if not clip:
            self._notes.append({"key": key, "distance": dist, "reason": getattr(self.speaker, "last_reason", None) or "not spoken"})
        return [phrase_for(clip)] if clip else []

    def on_frame(self, now, alert, turn_flag, latency_s):
        """Call for every processed frame. `alert` is the zone result tuple (key, distance) or None."""
        self.last_frame_t = now
        said = []
        self._notes = []
        link = self.monitor.check(now, now, latency_s or 0.0)
        if link:
            said += self._say(link, None, now)
        if turn_flag == "fast" and now - self._last_reduced >= REDUCED_REPEAT_S:
            self._last_reduced = now
            said += self._say("reduced_accuracy", None, now)
        key = alert[0] if alert else None
        # Obstacles speak at once (waiting would cost distance). 'Drop ahead' and "can't see ahead" are noisier, so
        # they must hold for two frames in a row; one stray frame is not worth interrupting someone for.
        confirmed = key in ("left", "ahead", "right") or key == self._prev_key
        self._prev_key = key
        if self.obstacle_audio and alert and confirmed:
            said += self._say(key, alert[1], now)
        why_silent = None
        if alert and not self.obstacle_audio:
            why_silent = "obstacle speech is off (no floor model: start with --cam-h)"
        elif alert and not confirmed:
            why_silent = f"'{key}' must hold for two frames in a row before it is spoken (first frame)"
        self.decision = {"zone_alert": list(alert) if alert else None, "link_alert": link, "turn_flag": turn_flag,
                         "confirmed": bool(alert and confirmed), "obstacle_audio": self.obstacle_audio, "spoken": list(said),
                         "not_spoken": self._notes + ([{"key": key, "reason": why_silent}] if why_silent else [])}
        return said

    def on_idle(self, now):
        """Call whenever no new frame arrived, so a dead stream is noticed."""
        if self.last_frame_t is None:
            return []
        link = self.monitor.check(now, self.last_frame_t, 0.0)
        return self._say(link, None, now) if link else []

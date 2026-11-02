"""Frame pacing on the Pi: the phase between the two cameras, the send rate on a slow link, and the Wi-Fi power-saving flag.

Phase. Two free-running 30 fps webcams each deliver a frame every 33.3 ms, but the moment inside that period is fixed by when each stream started, so the left and right
frames of a pair are a constant offset apart (14.3 ms in four of five of your recordings, 10.5 ms in the fifth; the offset changes only when a stream is restarted).
Closing the right stream and starting it again `phase_delay_ms` later moves that constant toward zero. A scene that moves during the offset gives false disparity of
focal length x turn rate x offset, so a smaller constant offset is a better picture, with nothing to guess.

Rate. A pair is about 135 KB. Sending 15 of them a second is 16 Mbit/s, more than a Raspberry Pi 3's Wi-Fi sustains when the supply is sagging; the send then blocks for
hundreds of milliseconds to seconds (pi.send_slow) and no pairs are made meanwhile (pi.loop_gap). RateGovernor lowers the send rate while sends are slow and raises it back
slowly when they are quick, so the link is not kept saturated.
"""
import subprocess
from collections import deque


def phase_delay_ms(dt_ms, period_ms):
    """How long to wait before restarting the RIGHT stream so that right minus left moves to zero. dt_ms = right capture time minus left, the median over some frames.
    The right frames shift by the wait, modulo the frame period: wait = (-dt) mod period."""
    return (-float(dt_ms)) % float(period_ms)


def nearest_offsets_ms(left_times, right_times):
    """Signed offsets (right minus left, ms) of the right frame nearest to each left frame."""
    if not left_times or not right_times:
        return []
    return [(min(right_times, key=lambda x: abs(x - tl)) - tl) * 1000.0 for tl in left_times]


class RateGovernor:
    """Send rate between min_fps and target_fps from how long each send took (ms)."""

    def __init__(self, target_fps, min_fps=5.0, slow_ms=150.0, quick_ms=80.0, cooldown_s=3.0, calm_s=10.0):
        self.target, self.min, self.slow, self.quick, self.cooldown, self.calm = float(target_fps), float(min_fps), slow_ms, quick_ms, cooldown_s, calm_s
        self.fps, self.recent, self.last_change, self.calm_since = float(target_fps), deque(maxlen=20), 0.0, None

    def update(self, send_ms, now):
        self.recent.append(send_ms)
        s = sorted(self.recent)
        med, p90 = s[len(s) // 2], s[int(0.9 * (len(s) - 1))]
        if (med > self.slow or (len(s) >= 10 and p90 > 2 * self.slow)) and len(s) >= 3 and now - self.last_change >= self.cooldown and self.fps > self.min:
            self.fps, self.last_change, self.calm_since = max(self.min, self.fps * 0.75), now, None
            self.recent.clear()                                  # decide the next step on sends made at the new rate, not on the old slow ones
        elif p90 < self.quick and len(s) >= 10:
            self.calm_since = self.calm_since if self.calm_since is not None else now
            if now - self.calm_since >= self.calm and self.fps < self.target:
                self.fps, self.last_change, self.calm_since = min(self.target, self.fps * 1.15), now, now
        else:
            self.calm_since = None
        return self.fps


def parse_power_save(text):
    """'on' / 'off' / None from the output of `iw dev wlan0 get power_save`."""
    t = text.lower()
    if "power save: on" in t:
        return "on"
    if "power save: off" in t:
        return "off"
    return None


def wifi_power_save(iface="wlan0"):
    try:
        return parse_power_save(subprocess.run(["iw", "dev", iface, "get", "power_save"], capture_output=True, text=True, timeout=3).stdout)
    except (OSError, subprocess.SubprocessError):
        return None

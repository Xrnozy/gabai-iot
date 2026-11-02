"""Spoken alerts from a small fixed vocabulary of pre-rendered WAV clips (no speech engine runs while walking).

Clips are made once by `py -m gabai.tools.make_alert_audio` (offline, Windows built-in voice). At run time an alert
picks a clip, the priority queue decides whether it may play (a higher priority cuts in, a lower one waits, repeats
within 3 s are dropped unless the obstacle got closer), and the player plays it.
"""
import os

from gabai.alerts import AlertQueue

ZONES = ("left", "ahead", "right")
DISTANCES = ("0_5", "1_0", "1_5", "2_0", "2_5", "3_0")
SYSTEM = ("drop_ahead", "cant_see_ahead", "tracking_delayed", "gabai_stopped", "reduced_accuracy")
PHRASES = {
    "drop_ahead": "drop ahead", "cant_see_ahead": "can't see ahead", "tracking_delayed": "tracking delayed",
    "gabai_stopped": "Gabai stopped, cable or camera lost", "reduced_accuracy": "reduced accuracy",
}
DIST_PHRASE = {"0_5": "half a metre", "1_0": "one metre", "1_5": "one and a half metres", "2_0": "two metres",
               "2_5": "two and a half metres", "3_0": "three metres"}


def distance_label(m):
    half = min(max(round(m * 2) / 2, 0.5), 3.0)
    return f"{half:.1f}".replace(".", "_")


def clip_for(key, distance):
    return f"{key}_{distance_label(distance)}" if key in ZONES else key


def all_clips():
    return [f"{z}_{d}" for z in ZONES for d in DISTANCES] + list(SYSTEM)


def phrase_for(clip):
    if clip in PHRASES:
        return PHRASES[clip]
    zone, dist = clip.split("_", 1)
    return f"{zone}, about {DIST_PHRASE[dist]}"


def priority_of(key, distance):
    """1 system failure, 2 degraded accuracy, 3 obstacle within 1 m or a drop or no view ahead, 4 obstacle 1 to 3 m."""
    if key in ("gabai_stopped", "tracking_delayed"):
        return 1
    if key == "reduced_accuracy":
        return 2
    if key in ("drop_ahead", "cant_see_ahead"):
        return 3
    return 3 if distance is not None and distance <= 1.0 else 4


def _winsound_player(path):
    import winsound
    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)   # async: a new clip replaces the old


class AlertSpeaker:
    def __init__(self, clips_dir="gabai/audio/alerts", player=None, queue=None, must_exist=True):
        self.dir, self.must_exist = clips_dir, must_exist
        self.queue = queue or AlertQueue()
        self.player = player or _winsound_player
        self.missing = []
        self.log = []                 # (time, text) of everything spoken, for the telemetry page
        self.last_reason = None       # why the last say() produced no sound (None when it did)

    def say(self, key, distance, now):
        clip = clip_for(key, distance if distance is not None else 0.5)
        if self.queue.push(key, priority_of(key, distance), clip if key not in ZONES else key, distance, now) is None:
            self.last_reason = self.queue.last_reason
            return None
        path = os.path.join(self.dir, clip + ".wav")
        if self.must_exist and not os.path.exists(path):
            if clip not in self.missing:
                self.missing.append(clip)
            self.last_reason = f"clip file missing: {path} (run: py -m gabai.tools.make_alert_audio)"
            return None
        self.last_reason = None
        self.player(path)
        self.log.append((now, phrase_for(clip)))
        return clip

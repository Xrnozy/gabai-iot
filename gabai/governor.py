"""Thermal and lane-time governor (spec phase 2b). Pure logic: it is given temperatures and lane times and returns the rates to use.

Every 2 s it picks a tier (spec, starting values to tune):
    normal    GPU < 70 C and CPU < 80 C and every lane under 100 ms   depth every frame, detector every 2nd, Boxer 1 Hz
    hot       GPU 70-80 C, or CPU 80-90 C, or any lane over 100 ms     depth every 2nd frame (7.5 Hz at 15 FPS, the floor), detector every 4th, Boxer 0.3 Hz
    critical  GPU > 80 C or CPU > 90 C                                 cheap fallback depth, Boxer off, say "reduced accuracy"
Never slowed: Basalt and the IMU (tracking comes first), and depth never goes below 7.5 Hz. Only optional work is cut.
Going back to a cooler tier needs 3 C of margin and 10 s of calm, so it does not flap. A missing temperature is NOT read as cool: the
tier is kept, `temps_unknown` is set, and a warning is logged.
This module decides; wiring a tier to real actions is the caller's job (see docs/architecture.md for what is wired and what is not).
"""
from gabai.diag import log as dlog

EVERY_S = 2.0
GPU_HOT, GPU_CRIT, CPU_HOT, CPU_CRIT = 70.0, 80.0, 80.0, 90.0
LANE_SLOW_MS = 100.0
MARGIN_C, CALM_S = 3.0, 10.0
UNKNOWN_WARN_S = 10.0
BACKGROUND_GPU_MAX = 65.0
RANK = {"normal": 0, "hot": 1, "critical": 2}


def _settings(tier):
    s = {"normal": dict(depth_every=1, detect_every=2, boxer_hz=1.0, use_fallback_depth=False, say_reduced_accuracy=False),
         "hot": dict(depth_every=2, detect_every=4, boxer_hz=0.3, use_fallback_depth=False, say_reduced_accuracy=False),
         "critical": dict(depth_every=1, detect_every=4, boxer_hz=0.0, use_fallback_depth=True, say_reduced_accuracy=True)}[tier]
    return dict(s, tier=tier, basalt_every=1, depth_hz_floor=7.5)


class Governor:
    def __init__(self):
        self.log = dlog.get("governor")
        self.tier = "normal"
        self.settings = _settings("normal")
        self.settings.update(reason="start", temps_unknown=False, background_ok=False)
        self._next = None
        self._calm_since = None
        self._unknown_since = None

    def _wanted(self, gpu, cpu, lanes):
        slow = [k for k, v in (lanes or {}).items() if v is not None and v > LANE_SLOW_MS]
        if (gpu is not None and gpu > GPU_CRIT) or (cpu is not None and cpu > CPU_CRIT):
            return "critical", f"{'GPU %.0f C' % gpu if gpu is not None and gpu > GPU_CRIT else 'CPU %.0f C' % cpu} is above the critical limit"
        if (gpu is not None and gpu >= GPU_HOT) or (cpu is not None and cpu >= CPU_HOT):
            return "hot", f"{'GPU %.0f C' % gpu if gpu is not None and gpu >= GPU_HOT else 'CPU %.0f C' % cpu} is in the hot range"
        if slow:
            return "hot", f"lane {slow[0]} took {lanes[slow[0]]:.0f} ms (limit {LANE_SLOW_MS:.0f} ms)"
        if gpu is None and cpu is None:
            return "normal", "no temperature reading: only lane times are being watched"
        return "normal", "temperatures and lane times are within limits"

    def update(self, now, gpu_c=None, cpu_c=None, lane_ms=None, user_moving=True):
        """Call as often as you like; it decides every 2 s and otherwise returns the current settings."""
        if self._next is not None and now < self._next:
            return self.settings
        self._next = now + EVERY_S
        unknown = gpu_c is None and cpu_c is None
        if unknown:
            self._unknown_since = self._unknown_since if self._unknown_since is not None else now
            if now - self._unknown_since >= UNKNOWN_WARN_S:
                self.log.warning_every("temps", 30.0, "governor.no_temperature", seconds=round(now - self._unknown_since, 1), tier=self.tier,
                                       hint="no GPU or CPU temperature is readable (nvidia-smi missing? CPU sensors are not available on Windows): "
                                            "the governor cannot protect the hardware, only lane times are being watched")
        else:
            self._unknown_since = None
        want, reason = self._wanted(gpu_c, cpu_c, lane_ms)
        prev = self.tier
        if RANK[want] > RANK[self.tier]:
            self.tier, self._calm_since = want, None                      # heat is acted on at once
        elif RANK[want] < RANK[self.tier] and not unknown:
            hot_gpu = gpu_c is not None and gpu_c > (GPU_HOT if self.tier == "hot" else GPU_CRIT) - MARGIN_C
            hot_cpu = cpu_c is not None and cpu_c > (CPU_HOT if self.tier == "hot" else CPU_CRIT) - MARGIN_C
            slow = any(v is not None and v > LANE_SLOW_MS * 0.9 for v in (lane_ms or {}).values())
            if hot_gpu or hot_cpu or slow:
                self._calm_since = None
                reason = "still too close to the limit to relax"
            else:
                self._calm_since = self._calm_since if self._calm_since is not None else now
                if now - self._calm_since >= CALM_S:
                    self.tier, self._calm_since = want, None
                else:
                    reason = f"cooler, waiting {CALM_S - (now - self._calm_since):.0f} s more before relaxing"
        elif want == self.tier:
            self._calm_since = None
        self.settings = dict(_settings(self.tier), reason=reason, temps_unknown=unknown,
                             background_ok=(not user_moving) and gpu_c is not None and gpu_c < BACKGROUND_GPU_MAX and self.tier == "normal")
        if self.tier != prev:
            self.log.info("governor.tier", tier=self.tier, previous=prev, reason=reason, gpu_c=gpu_c, cpu_c=cpu_c, lanes=lane_ms, settings=self.settings)
        return self.settings

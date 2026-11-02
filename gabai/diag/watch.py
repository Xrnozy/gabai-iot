"""PC-side watcher for the Pi link and the sensor streams. Call update() once per loop pass (it acts every `period_s`) and
note_imu() with every batch of IMU samples. It logs connection changes at once and a statistics line every period, and turns the
numbers into checks (gabai.diag.checks) so a failing one is named in the log with its hint.
"""
import numpy as np

from gabai.diag import checks as C
from gabai.diag import log as dlog
from gabai.diag.pi_status import active_flags

G = 9.80665


class LinkWatcher:
    def __init__(self, period_s=5.0):
        self.period = period_s
        self.log = dlog.get("link")
        self.imu_log = dlog.get("imu")
        self._connected = None
        self._t_last = None
        self._prev = {}
        self._imu_n, self._imu_t_prev, self._imu_gap_max = 0, None, 0.0
        self._still_norms = []
        self._flags = None

    def note_imu(self, samples):
        """samples: tuples (t, gx, gy, gz, ax, ay, az) on the Pi clock, gyro in deg/s, accel in g (as received)."""
        for s in samples:
            t = s[0]
            if self._imu_t_prev is not None:
                self._imu_gap_max = max(self._imu_gap_max, t - self._imu_t_prev)
            self._imu_t_prev = t
            self._imu_n += 1
            if (s[1] ** 2 + s[2] ** 2 + s[3] ** 2) ** 0.5 < 2.0:        # gyro quiet: the accelerometer should read exactly 1 g
                self._still_norms.append((s[4] ** 2 + s[5] ** 2 + s[6] ** 2) ** 0.5 * G)
        if len(self._still_norms) > 4000:
            self._still_norms = self._still_norms[-4000:]

    def update(self, now, st, health):
        con = bool(st.get("connected"))
        if self._connected is not None and con != self._connected:
            if con:
                self.log.info("link.up", peer=st.get("peer"), connections=st.get("connections"))
            else:
                self.log.warning("link.lost", last_error=st.get("last_error"), frames=st.get("frames"),
                                 hint="see link.disconnected just before this line for the reason")
        elif self._connected is None and con:
            self.log.info("link.up", peer=st.get("peer"), connections=st.get("connections"))
        self._connected = con
        if self._t_last is None:
            self._t_last = now
            self._prev = dict(frames=st.get("frames", 0), bytes=st.get("bytes", 0), overwritten=st.get("overwritten", 0),
                              decode_errors=st.get("decode_errors", 0), imu_overflow=st.get("imu_overflow", 0))
            return
        if now - self._t_last < self.period:
            return
        w = now - self._t_last
        d = {k: st.get(k, 0) - self._prev.get(k, 0) for k in self._prev}
        rtt = health.get("rtt_s")
        imu_hz = self._imu_n / w if self._imu_n else None
        imu_gap_ms = round(self._imu_gap_max * 1000, 1) if self._imu_n else None
        flags = active_flags(health.get("throttled"))
        self.log.info("link.stats", window_s=round(w, 1), connected=con, recv_fps=round(d["frames"] / w, 2), mbit_s=round(d["bytes"] * 8 / w / 1e6, 2),
                      overwritten_in_window=d["overwritten"], decode_errors_in_window=d["decode_errors"], imu_overflow_in_window=d["imu_overflow"],
                      frame_age_s=st.get("frame_age_s"), inter_arrival_max_ms=st.get("inter_arrival_max_ms"), rejected=st.get("rejected"),
                      connections=st.get("connections"), imu_hz=None if imu_hz is None else round(imu_hz, 1), imu_max_gap_ms=imu_gap_ms,
                      imu_queue=st.get("imu_queue"), mag_queue=st.get("mag_queue"), pi_temp_c=health.get("temp_c"), pi_flags=flags,
                      pi_mem_mb=health.get("mem_available_mb"), pi_load1=health.get("load1"), pi_fps=health.get("fps"),
                      rtt_ms=None if rtt is None else round(rtt * 1000, 1), clock_offset_s=health.get("clock_offset_s"),
                      cam_fps=[health.get("cam_fps_left"), health.get("cam_fps_right")], pair_dt_ms=health.get("pair_dt_ms"),
                      grab_fail=[health.get("grab_fail_left"), health.get("grab_fail_right")], i2c_errors=(st.get("imu") or {}).get("i2c_errors"),
                      msg_counts=st.get("msg_counts"))
        checks = [C.check_frame_age(st.get("frame_age_s")), C.check_pi_power(health.get("throttled")),
                  C.check_imu_rate(imu_hz, imu_gap_ms)]
        if len(self._still_norms) >= 200:
            checks.append(C.check_gravity(float(np.mean(self._still_norms))))
        if rtt is not None and rtt * 1000 > 300:
            self.log.warning_every("rtt", 20.0, "link.rtt_high", rtt_ms=round(rtt * 1000, 1), hint="Wi-Fi latency spikes delay every alert by about this much")
        for c in checks:
            if c.status == "fail":
                self.log.warning_every(f"chk:{c.name}", 20.0, "check.fail", check=c.name, value=c.value, limit=c.limit, detail=c.detail,
                                       hint=c.hint, component=c.component)
            elif c.status == "suspect":
                self.log.info_every(f"chk:{c.name}", 30.0, "check.suspect", check=c.name, value=c.value, limit=c.limit, detail=c.detail,
                                    hint=c.hint, component=c.component)
        self._t_last = now
        self._prev = dict(frames=st.get("frames", 0), bytes=st.get("bytes", 0), overwritten=st.get("overwritten", 0),
                          decode_errors=st.get("decode_errors", 0), imu_overflow=st.get("imu_overflow", 0))
        self._imu_n, self._imu_gap_max = 0, 0.0

"""BMI160 gyro/accelerometer on the Pi (I2C bus 1, address 0x68), polled in a thread.

Wiring and the check script: docs/phase-S-manual-test.md (Test 0). Only the gyro matters for flagging fast turns.
The decode helpers are pure so they can be tested without hardware.
"""
import math
import struct
import threading
import time
from collections import deque
from itertools import islice

ADDR = 0x68
GYRO_RANGE_DPS = 500          # register 0x43 = 0x02
ACCEL_RANGE_G = 2             # register 0x41 = 0x03


def decode_gyro(raw, range_dps=GYRO_RANGE_DPS):
    return tuple(v * range_dps / 32768.0 for v in struct.unpack("<3h", raw))


def decode_accel(raw, range_g=ACCEL_RANGE_G):
    return tuple(v * range_g / 32768.0 for v in struct.unpack("<3h", raw))


def peak_speed(samples, now, window):
    """Largest angular speed magnitude (deg/s) among samples newer than `now - window`."""
    best = 0.0
    for t, (x, y, z) in samples:
        if now - t <= window:
            best = max(best, math.sqrt(x * x + y * y + z * z))
    return best


def new_samples_since(buf, last_t, limit=120):
    """Samples (t, gx, gy, gz, ax, ay, az) newer than last_t, newest `limit` at most."""
    out = [s for s in buf if s[0] > last_t]
    return out[-limit:]


class Bmi160:
    def __init__(self, bus=1, rate_hz=200):
        from smbus2 import SMBus
        self.bus = SMBus(bus)
        cid = self.bus.read_byte_data(ADDR, 0x00)
        if cid != 0xD1:
            raise OSError(f"BMI160 chip id 0x{cid:02X}, expected 0xD1 (check wiring/address)")
        self.bus.write_byte_data(ADDR, 0x7E, 0x11)  # accel normal mode
        time.sleep(0.05)
        self.bus.write_byte_data(ADDR, 0x7E, 0x15)  # gyro normal mode
        time.sleep(0.1)
        self.bus.write_byte_data(ADDR, 0x41, 0x03)  # accel +-2 g
        self.bus.write_byte_data(ADDR, 0x40, 0x29)  # accel 200 Hz
        self.bus.write_byte_data(ADDR, 0x43, 0x02)  # gyro +-500 dps
        self.bus.write_byte_data(ADDR, 0x42, 0x29)  # gyro 200 Hz
        self.period = 1.0 / rate_hz
        self.samples = deque(maxlen=400)            # (t, (gx, gy, gz)) bias-corrected, last ~2 s
        self.full = deque(maxlen=6000)              # (t, gx, gy, gz, ax, ay, az): gyro in deg/s, accel in g. 30 s at 200 Hz (about 1 MB): also the resend window after a lost link
        self.n, self._next, self._lock = 0, 0, threading.Lock()      # samples counted so far (sample k has seq k), first seq not yet handed to drain_seq
        self._sent_t = 0.0
        self.bias = (0.0, 0.0, 0.0)
        self.accel = (0.0, 0.0, 1.0)
        self.errors = 0
        self._stop = False

    def _read(self):
        raw = bytes(self.bus.read_i2c_block_data(ADDR, 0x0C, 12))  # gyro xyz then accel xyz
        return decode_gyro(raw[:6]), decode_accel(raw[6:])

    def calibrate(self, seconds=1.5):
        """Average the gyro while the rig is still. Returns the spread so the caller can warn if it moved."""
        xs = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            xs.append(self._read()[0])
            time.sleep(self.period)
        n = len(xs)
        self.bias = tuple(sum(v[i] for v in xs) / n for i in range(3))
        spread = max(math.sqrt(sum((v[i] - self.bias[i]) ** 2 for i in range(3))) for v in xs)
        return self.bias, spread

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while not self._stop:
            try:
                g, a = self._read()
                t = time.monotonic()
                gc = tuple(g[i] - self.bias[i] for i in range(3))
                self.samples.append((t, gc))
                with self._lock:
                    self.full.append((t, *gc, *a))
                    self.n += 1
                self.accel = a
            except OSError:
                self.errors += 1
            time.sleep(self.period)

    def drain(self):
        """Samples not yet sent, as lists, with the Pi's monotonic timestamps (same clock as the camera frames)."""
        new = new_samples_since(list(self.full), self._sent_t)
        if new:
            self._sent_t = new[-1][0]
        return [[round(new[i][0], 5)] + [round(v, 3) for v in new[i][1:]] for i in range(len(new))]

    def drain_seq(self, limit=400):
        """(seq of the first row, rows): the OLDEST samples not yet handed out, at most `limit`. Nothing is skipped (drain() keeps the newest 120 and drops the rest). A sample
        that has already left the ring is gone: the seq then jumps and the receiver counts it as missing."""
        with self._lock:
            first = self.n - len(self.full)
            start = max(self._next, first)
            rows = list(islice(self.full, start - first, start - first + limit))
            self._next = start + len(rows)
        return start, [[round(r[0], 5)] + [round(v, 3) for v in r[1:]] for r in rows]

    def rewind(self, seq):
        """The receiver still needs samples from `seq` on (its ACK after a new connection): hand them out again."""
        with self._lock:
            self._next = min(self._next, max(int(seq), 0))

    def snapshot(self, window=0.3):
        now = time.monotonic()
        s = list(self.samples)
        last = s[-1][1] if s else (0.0, 0.0, 0.0)
        return {"gyro_dps": [round(v, 2) for v in last], "peak_dps": round(peak_speed(s, now, window), 1),
                "accel_g": [round(v, 3) for v in self.accel], "i2c_errors": self.errors}

    def stop(self):
        self._stop = True

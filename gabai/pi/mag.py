"""HMC5883L magnetometer on the Pi (software I2C bus 3, address 0x1E), polled in a thread.

The compass is wired to header pins 35/37, so it needs this line in /boot/firmware/config.txt and a reboot:
    dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=26,i2c_gpio_scl=19
"""
import struct
import threading
import time
from collections import deque

ADDR = 0x1E


def decode_mag(raw):
    """6 bytes from register 0x03 -> (x, y, z) raw counts. The chip sends X, Z, Y (big-endian). -4096 = overflow."""
    x, z, y = struct.unpack(">3h", raw)
    if -4096 in (x, y, z):
        return None
    return (x, y, z)


class Hmc5883l:
    def __init__(self, bus=3, rate_hz=50):
        from smbus2 import SMBus
        self.bus = SMBus(bus)
        ident = bytes(self.bus.read_i2c_block_data(ADDR, 0x0A, 3))
        if ident != b"H43":
            raise OSError(f"HMC5883L id {ident!r}, expected b'H43' (a QMC5883L clone answers at 0x0D instead)")
        self.bus.write_byte_data(ADDR, 0x00, 0x78)  # 8 samples averaged, 75 Hz output
        self.bus.write_byte_data(ADDR, 0x01, 0x20)  # gain 1.3 Ga (0.92 mG/count)
        self.bus.write_byte_data(ADDR, 0x02, 0x00)  # continuous measurement
        time.sleep(0.1)
        self.period = 1.0 / rate_hz
        self.buf = deque(maxlen=400)                # (t, x, y, z)
        self.overflows = 0
        self.errors = 0
        self._sent_t = 0.0
        self._stop = False

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        last = None
        while not self._stop:
            try:
                s = bytes(self.bus.read_i2c_block_data(ADDR, 0x03, 6))
                v = decode_mag(s)
                if v is None:
                    self.overflows += 1
                elif s != last:  # the chip updates at 75 Hz; skip repeats of the same reading
                    self.buf.append((time.monotonic(), *v))
                last = s
            except OSError:
                self.errors += 1
            time.sleep(self.period)

    def drain(self, limit=60):
        """Readings not sent yet, as [t, x, y, z] on the Pi's monotonic clock."""
        new = [s for s in list(self.buf) if s[0] > self._sent_t][-limit:]
        if new:
            self._sent_t = new[-1][0]
        return [[round(s[0], 5), s[1], s[2], s[3]] for s in new]

    def stop(self):
        self._stop = True

"""Runs on the Pi. Confirms both sensors answer on the wiring from the diagram.

BMI160   : I2C bus 1 (header pins 3/5), address 0x68 (SA0 to GND), chip id register 0x00 = 0xD1.
HMC5883L : second bus (header pins 37 SDA = GPIO26, 35 SCL = GPIO19), address 0x1E, id regs 0x0A..0x0C = "H43".
Pins 35/37 are not the Pi's hardware I2C, so the second bus needs this line in /boot/firmware/config.txt
(older images: /boot/config.txt), then a reboot:
    dtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=26,i2c_gpio_scl=19
Needs: sudo apt install python3-smbus2 i2c-tools   (and I2C enabled in raspi-config)
"""
import argparse
import struct
import time

from smbus2 import SMBus


def check_bmi160(bus_no):
    with SMBus(bus_no) as b:
        cid = b.read_byte_data(0x68, 0x00)
        ok = cid == 0xD1
        print(f"BMI160  bus {bus_no} 0x68 chip id=0x{cid:02X} -> {'OK' if ok else 'UNEXPECTED (want 0xD1)'}")
        if not ok:
            return False
        b.write_byte_data(0x68, 0x7E, 0x11)  # accel normal mode
        time.sleep(0.05)
        b.write_byte_data(0x68, 0x7E, 0x15)  # gyro normal mode
        time.sleep(0.1)
        raw = bytes(b.read_i2c_block_data(0x68, 0x0C, 12))  # gyro xyz then accel xyz, int16 LE
        gx, gy, gz, ax, ay, az = struct.unpack("<6h", raw)
        g = (ax * ax + ay * ay + az * az) ** 0.5 / 16384.0  # default range +-2 g
        print(f"        accel magnitude at rest = {g:.2f} g (want ~1.0), gyro raw = {gx},{gy},{gz} (want near 0)")
        return 0.8 < g < 1.2


def check_hmc5883l(bus_no):
    with SMBus(bus_no) as b:
        ident = bytes(b.read_i2c_block_data(0x1E, 0x0A, 3))
        ok = ident == b"H43"
        print(f"HMC5883L bus {bus_no} 0x1E id={ident!r} -> {'OK' if ok else 'UNEXPECTED (want H43)'}")
        if not ok:
            return False
        b.write_byte_data(0x1E, 0x00, 0x70)  # 8 avg, 15 Hz
        b.write_byte_data(0x1E, 0x02, 0x00)  # continuous mode
        time.sleep(0.2)
        raw = bytes(b.read_i2c_block_data(0x1E, 0x03, 6))  # X, Z, Y big endian
        x, z, y = struct.unpack(">3h", raw)
        print(f"        field raw x,y,z = {x},{y},{z} (not all zero; rotate the board and they should change)")
        return any((x, y, z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu-bus", type=int, default=1)
    ap.add_argument("--mag-bus", type=int, default=3)
    a = ap.parse_args()
    results = {}
    for name, fn, bus in (("BMI160", check_bmi160, a.imu_bus), ("HMC5883L", check_hmc5883l, a.mag_bus)):
        try:
            results[name] = fn(bus)
        except OSError as e:
            print(f"{name}: no response on bus {bus} ({e}). Check wiring, enable I2C, and the overlay line.")
            results[name] = False
    print("RESULT:", results)


if __name__ == "__main__":
    main()

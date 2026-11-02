"""Runs on the Pi. Pulses an LED on a GPIO pin at irregular intervals for the sync test.

Wiring: GPIO pin -> 330 ohm resistor -> LED anode; LED cathode -> GND. Default BCM pin 17 (header pin 11).
Point the LED so both cameras see it (put it in the ROI passed to pi_capture.py).
Needs: sudo apt install python3-gpiozero   (or use any strobe/flashing light instead, no script needed)
"""
import argparse
import random
import time

from gpiozero import LED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", type=int, default=17)
    ap.add_argument("--pulse-ms", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=90)
    a = ap.parse_args()
    led = LED(a.pin)
    end = time.monotonic() + a.duration
    n = 0
    while time.monotonic() < end:
        time.sleep(0.4 + random.random() * 0.4)
        led.on()
        time.sleep(a.pulse_ms / 1000)
        led.off()
        n += 1
    print("pulses:", n)


if __name__ == "__main__":
    main()

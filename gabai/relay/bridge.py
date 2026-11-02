"""Feeds the relay from the existing Pi link (gabai.pc.receiver.LiveReceiver) and runs the periodic state tick."""
import threading
import time


class PiBridge:
    def __init__(self, relay, frames, receiver, poll_s=0.25):
        self.relay, self.frames, self.receiver, self.poll_s = relay, frames, receiver, poll_s
        self._stop = threading.Event()
        self._last_seq = None

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="relay-bridge").start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            self.step()
            time.sleep(self.poll_s)

    def step(self):
        pair = self.receiver.latest()
        if pair is not None and pair.seq != self._last_seq:
            self._last_seq = pair.seq
            self.frames.put(pair.seq, pair.left, pair.right)
        self.relay.update_pi(self.receiver.stats())
        self.frames.persist()


class Ticker:
    """Offline detection and escalation must run even when nothing is calling the API."""

    def __init__(self, relay, every_s=2.0):
        self.relay, self.every_s = relay, every_s
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True, name="relay-tick").start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.every_s):
            try:
                self.relay.tick()
            except Exception:
                pass

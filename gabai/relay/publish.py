"""Send results to the relay over HTTP from another process (the mapper), without ever slowing it down.

The newest result replaces any waiting one; a relay that is down only means results are dropped (one note is printed).
"""
import json
import os
import threading
import urllib.request

from .core import default_db_path


def pi_token():
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(default_db_path())), "relay_tokens.json"), encoding="utf-8") as f:
            return json.load(f)["pi"]
    except (OSError, ValueError, KeyError):
        return ""


class RelayPublisher:
    def __init__(self, url="http://127.0.0.1:8090", token=None):
        self.url, self.token = url.rstrip("/"), token if token is not None else pi_token()
        self._latest, self._cv, self._warned = None, threading.Condition(), False
        threading.Thread(target=self._run, daemon=True, name="relay-publisher").start()

    def publish(self, result):
        with self._cv:
            self._latest = result
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                while self._latest is None:
                    self._cv.wait()
                result, self._latest = self._latest, None
            try:
                req = urllib.request.Request(self.url + "/api/pi/nav", data=json.dumps(result).encode(), method="POST",
                                             headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
                urllib.request.urlopen(req, timeout=2).read()
                self._warned = False
            except Exception as e:
                if not self._warned:
                    print(f"relay publisher: cannot reach {self.url} ({type(e).__name__}); dropping results until it answers", flush=True)
                    self._warned = True

"""Map tiles served from the PC, cached on disk.

Phones ask the relay (`/tiles/z/x/y.png`), never the internet. A tile that is not cached yet is fetched once from OpenStreetMap
with a proper User-Agent, saved, and served from disk from then on, so the map keeps working when the hotspot has no internet.
Usage policy: https://operations.osmfoundation.org/policies/tiles/ (no bulk downloads; keep `prefetch` areas small).

    py -m gabai.relay.tiles --lat 14.5995 --lon 120.9842 --km 0.6 --zoom 15 17
"""
import argparse
import math
import os
import time
import urllib.request

UPSTREAM = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "GabAi-Relay/1.0 (local caregiver map cache, low volume)"
MAX_ZOOM = 19


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=6) as r:
        return r.read()


class Tiles:
    def __init__(self, cache_dir, fetch=_fetch):
        self.dir, self._fetch = cache_dir, fetch

    def get(self, z, x, y):
        """PNG bytes, or None when the tile is neither cached nor reachable. Coordinates are validated: they become a file path."""
        if not (0 <= z <= MAX_ZOOM and 0 <= x < 2 ** z and 0 <= y < 2 ** z):
            return None
        path = os.path.join(self.dir, str(z), str(x), f"{y}.png")
        try:
            with open(path, "rb") as f:
                return f.read()
        except OSError:
            pass
        try:
            data = self._fetch(UPSTREAM.format(z=z, x=x, y=y))
        except Exception:
            return None
        if not data.startswith(b"\x89PNG"):                       # an HTML "blocked" page must never be cached as a tile
            return None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return data

    def prefetch(self, lat, lon, km, zooms, pause_s=0.5):
        """Fill the cache around a point, politely (one request per pause_s). Returns (cached, failed)."""
        dlat = km / 111.0
        dlon = km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
        ok = bad = 0
        for z in zooms:
            x0, y1 = tile_xy(lat - dlat, lon - dlon, z)
            x1, y0 = tile_xy(lat + dlat, lon + dlon, z)
            for x in range(x0, x1 + 1):
                for y in range(y0, y1 + 1):
                    cached = os.path.exists(os.path.join(self.dir, str(z), str(x), f"{y}.png"))
                    if self.get(z, x, y):
                        ok += 1
                        if not cached:
                            time.sleep(pause_s)
                    else:
                        bad += 1
        return ok, bad


def main():
    from .core import default_db_path
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--km", type=float, default=0.6, help="half-width of the square to cache")
    ap.add_argument("--zoom", type=int, nargs=2, default=[15, 17], metavar=("MIN", "MAX"))
    a = ap.parse_args()
    t = Tiles(os.path.join(os.path.dirname(os.path.abspath(default_db_path())), "tiles"))
    print("cached %d tiles, %d failed" % t.prefetch(a.lat, a.lon, a.km, range(a.zoom[0], a.zoom[1] + 1)))


if __name__ == "__main__":
    main()

"""Extract only MH_01_easy.zip (ASL format) from ETH's 12.7 GB machine_hall.zip using HTTP range requests.

Reads the zip directory remotely, then downloads just that member's compressed bytes. The download is
resumable (a .part file) and backs off on HTTP 429 rate limiting.
"""
import io
import os
import sys
import time
import urllib.request
import zipfile
import zlib

URL = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/7b2419c1-62b5-4714-b7f8-485e5fe3e5fe/content"
WANT = "MH_01_easy"
BLOCK = 64 * 1024 * 1024
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "datasets")


def fetch(start, end, tries=20):
    for i in range(tries):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            wait = min(10 * (i + 1), 90)
            print(f"  retry in {wait}s ({e})", flush=True)
            time.sleep(wait)
    raise SystemExit("giving up after repeated errors (rate limit?); rerun to resume")


class RangeFile(io.RawIOBase):
    def __init__(self, size):
        self.size, self.pos = size, 0
        self._start, self._buf = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        if not (self._start <= self.pos and self.pos + n <= self._start + len(self._buf)):
            want = max(n, 1024 * 1024)
            self._start = self.pos
            self._buf = fetch(self.pos, min(self.pos + want, self.size) - 1)
        off = self.pos - self._start
        data = self._buf[off:off + n]
        self.pos += len(data)
        return data


def main():
    size = None
    for i in range(8):
        try:
            head = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(head, timeout=60) as r:
                size = int(r.headers["Content-Range"].split("/")[1])
            break
        except Exception as e:
            print(f"  size probe failed ({e}); waiting {15 * (i + 1)}s", flush=True)
            time.sleep(15 * (i + 1))
    if size is None:
        raise SystemExit("server keeps refusing (rate limit); try again later")
    print("remote size", size)
    zf = zipfile.ZipFile(RangeFile(size))
    members = [m for m in zf.infolist() if m.filename.endswith(f"{WANT}/{WANT}.zip")]  # not the .bag
    for m in members:
        print(f"{m.filename}  {m.file_size/1e6:.0f} MB  compressed {m.compress_size/1e6:.0f} MB  method {m.compress_type}")
    if len(members) != 1 or members[0].compress_size > 3e9:
        raise SystemExit("unexpected member list or size over 3 GB; not downloading")
    m = members[0]
    dest = os.path.join(OUT, m.filename)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    hdr = fetch(m.header_offset, m.header_offset + 29)  # local header: 30 bytes + name + extra, then data
    nlen, elen = int.from_bytes(hdr[26:28], "little"), int.from_bytes(hdr[28:30], "little")
    start = m.header_offset + 30 + nlen + elen
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    with open(part, "ab") as f:
        while have < m.compress_size:
            end = min(start + have + BLOCK, start + m.compress_size) - 1
            chunk = fetch(start + have, end)
            f.write(chunk)
            f.flush()
            have += len(chunk)
            print(f"{m.filename}: {have/1e6:.0f}/{m.compress_size/1e6:.0f} MB", flush=True)
    d = zlib.decompressobj(-15) if m.compress_type == 8 else None
    with open(part, "rb") as src, open(dest, "wb") as dst:
        while blk := src.read(8 * 1024 * 1024):
            dst.write(d.decompress(blk) if d else blk)
        if d:
            dst.write(d.flush())
    os.remove(part)
    print("wrote", dest, os.path.getsize(dest), "bytes (expected", m.file_size, ")")


if __name__ == "__main__":
    sys.exit(main())

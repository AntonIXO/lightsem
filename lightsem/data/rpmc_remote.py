"""Read the 42.5 GB RPMC-L2 HDF5 WITHOUT downloading it.

The dataset ships as four ~10 GiB byte-split parts that concatenate into one
RPMC_L2.h5. This box has 1.8 GB free, so a full download is impossible.

Instead: HuggingFace serves HTTP range requests (verified: 206 + content-range),
so we expose the four parts as ONE seekable file-like object and let h5py's
driver pull only the byte ranges it actually needs. HDF5's B-tree index means
listing keys and reading a single dataset touches megabytes, not gigabytes.

A block cache keeps this usable: HDF5 issues many small reads, and without
caching each becomes an HTTPS round trip.
"""
from __future__ import annotations

import io
import os
import threading

import requests

BASE = ("https://huggingface.co/datasets/RS2002/RPMC-L2/resolve/main/"
        "data/RPMC_L2_parts/")
PARTS = [
    ("RPMC_L2_part_aa", 10737418240),
    ("RPMC_L2_part_ab", 10737418240),
    ("RPMC_L2_part_ac", 10737418240),
    ("RPMC_L2_part_ad", 10348374404),
]
TOTAL = sum(sz for _, sz in PARTS)
BLOCK = 1 << 22  # 4 MiB


class RemoteSplitFile(io.RawIOBase):
    """Seekable read-only view over byte-split remote parts."""

    def __init__(self, cache_dir: str | None = None, verbose: bool = False):
        self._pos = 0
        self._lock = threading.Lock()
        self._mem: dict[int, bytes] = {}
        self._cache_dir = cache_dir
        self._verbose = verbose
        self.bytes_fetched = 0
        self.requests_made = 0
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._s = requests.Session()
        # Precompute part offsets.
        self._offsets = []
        off = 0
        for name, sz in PARTS:
            self._offsets.append((off, off + sz, name))
            off += sz

    # --- io plumbing -----------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, pos, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = pos
        elif whence == io.SEEK_CUR:
            self._pos += pos
        else:
            self._pos = TOTAL + pos
        return self._pos

    # --- range fetch -----------------------------------------------------
    def _fetch_block(self, idx: int) -> bytes:
        if idx in self._mem:
            return self._mem[idx]
        path = None
        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"b{idx:07d}.bin")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    data = f.read()
                self._mem[idx] = data
                return data

        start = idx * BLOCK
        end = min(start + BLOCK, TOTAL) - 1
        chunks = []
        # A block may straddle two parts.
        for p_start, p_end, name in self._offsets:
            if end < p_start or start > p_end - 1:
                continue
            a = max(start, p_start) - p_start
            b = min(end, p_end - 1) - p_start
            r = self._s.get(BASE + name,
                            headers={"Range": f"bytes={a}-{b}"}, timeout=60)
            r.raise_for_status()
            chunks.append(r.content)
            self.requests_made += 1
            self.bytes_fetched += len(r.content)
        data = b"".join(chunks)
        self._mem[idx] = data
        if path:
            with open(path, "wb") as f:
                f.write(data)
        if self._verbose:
            print(f"    [fetch] block {idx} ({len(data)/1e6:.1f} MB) "
                  f"total={self.bytes_fetched/1e6:.1f} MB")
        return data

    def readinto(self, b):
        n = len(b)
        if n == 0 or self._pos >= TOTAL:
            return 0
        n = min(n, TOTAL - self._pos)
        out = bytearray()
        pos = self._pos
        remaining = n
        with self._lock:
            while remaining > 0:
                idx = pos // BLOCK
                off = pos % BLOCK
                blk = self._fetch_block(idx)
                take = min(remaining, len(blk) - off)
                if take <= 0:
                    break
                out += blk[off:off + take]
                pos += take
                remaining -= take
        b[:len(out)] = out
        self._pos = pos
        return len(out)


def open_h5(cache_dir="data/rpmc_cache", verbose=False):
    """Return (h5py.File, RemoteSplitFile) reading the remote dataset."""
    import h5py
    raw = RemoteSplitFile(cache_dir=cache_dir, verbose=verbose)
    buf = io.BufferedReader(raw, buffer_size=BLOCK)
    return h5py.File(buf, "r"), raw

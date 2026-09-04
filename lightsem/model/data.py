"""RPMC-L2 dataset for the LightSem macro layer.

FRAME RATES (derived, not assumed -- see docs):
  light histograms : 30 fps  (video frames, 1920x1080 -> hist sums to 2073600)
  openl3           : 10 Hz   (hop 0.1 s)  <- the TRUE information rate of x
  librosa features : 43.07 Hz (hop 512 @ 22050)

extract_rpmc.py upsampled openl3 (10 Hz) onto the 30 fps light grid, so 2 of
every 3 x-frames are interpolation artefacts. We undo that here: decimate to
10 Hz and average y over each group of 3 frames. Averaging y is not lossy for
our purpose -- it IS the macro layer. Beat-rate flashes are the micro layer's
job and are produced deterministically from the beatgrid, never predicted.

SPLIT IS BY SONG. Frames from one show must never straddle train/val, or the
model can memorise that venue's palette and we would report leakage as skill.
"""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass

import numpy as np

FPS_LIGHT = 30.0
HZ_AUDIO = 10.0
DECIM = int(round(FPS_LIGHT / HZ_AUDIO))  # 3
TARGET_NAMES = ["hue_cos", "hue_sin", "sat", "val", "val_p90", "hue_conc"]


@dataclass
class Song:
    key: str
    x: np.ndarray  # (T, 512) float32, 10 Hz
    y: np.ndarray  # (T, 6)   float32, 10 Hz


def _decimate(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = (len(x) // DECIM) * DECIM
    if n == 0:
        return x[:0], y[:0]
    # x: take every 3rd (these are the real openl3 frames, not interpolants)
    xd = x[:n].reshape(n // DECIM, DECIM, -1)[:, 0, :]
    # y: average the 3 -> low-pass to macro band
    yd = y[:n].reshape(n // DECIM, DECIM, -1).mean(1)
    return xd, yd


def load_songs(root: str = "data/cache/rpmc_train", min_frames: int = 160) -> list[Song]:
    songs = []
    for p in sorted(glob.glob(os.path.join(root, "*.npz"))):
        d = np.load(p)
        x, y = d["x"].astype(np.float32), d["y"].astype(np.float32)
        if len(x) != len(y):
            continue
        x, y = _decimate(x, y)
        if len(x) < min_frames or not (np.isfinite(x).all() and np.isfinite(y).all()):
            continue
        songs.append(Song(key=os.path.basename(p)[:-4], x=x, y=y))
    return songs


def split_songs(songs: list[Song], val_frac: float = 0.15, test_frac: float = 0.15
                ) -> tuple[list[Song], list[Song], list[Song]]:
    """Deterministic hash split -- stable if shards are added later."""
    def bucket(s: Song) -> float:
        h = hashlib.sha256(s.key.encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    tr, va, te = [], [], []
    for s in songs:
        b = bucket(s)
        (te if b < test_frac else va if b < test_frac + val_frac else tr).append(s)
    return tr, va, te


class WindowIndex:
    """Sampleable (song, end_frame) pairs. Causal: a window is x[e-W:e].

    Targets are taken at e-1 (now) and e-1+h for each anticipation horizon,
    so a window is only valid if all horizons stay inside the same song.
    """

    def __init__(self, songs: list[Song], window: int, horizons: list[int]):
        self.songs = songs
        self.window = window
        self.horizons = horizons
        hmax = max(horizons)
        pairs = []
        for i, s in enumerate(songs):
            last = len(s.x) - hmax  # e-1+hmax <= len-1
            for e in range(window, last + 1):
                pairs.append((i, e))
        self.pairs = np.array(pairs, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.pairs)


def stack_arrays(songs: list[Song]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate songs -> (X, Y, song_id) for fast GPU-resident indexing."""
    X = np.concatenate([s.x for s in songs])
    Y = np.concatenate([s.y for s in songs])
    sid = np.concatenate([np.full(len(s.x), i, np.int64) for i, s in enumerate(songs)])
    return X, Y, sid


def song_offsets(songs: list[Song]) -> np.ndarray:
    off = np.zeros(len(songs) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(s.x) for s in songs])
    return off

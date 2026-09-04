"""Loader for the Oslo club-dancing EDM mocap (Solberg & Jensenius, CC-BY-4.0).

The corpus AIST++ is not: ~29 people dancing FREELY to a DJ mix of commercial
EDM, one marker each, stable identities, 200 Hz, with breakdown / build-up /
drop times annotated and redistributable acoustic features on the same clock.
Free movement is the condition under which the music-induced-movement
literature's correlations were measured, and Stage 0b showed AIST++ does not
meet it.

Zenodo 21945721 (2015 session, three groups, two DJ mixes) and 21945706 (2014
session, group quantity of motion only -- its per-dancer tracking failed).
The audio itself is not redistributable and is not needed: the record ships
spectral flux, pulse clarity, brightness, bass fraction and a windowed tempo at
86.13 fps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/lightsem-data/oslo")
S2015 = ROOT / "Solberg2015_EDM"
S2014 = ROOT / "Solberg2014_EDM"

#: group -> (QTM take, DJ mix heard)
GROUPS = {"A": ("Group0001", 1), "B": ("Group0002", 2), "C": ("Group0003", 1)}
MIX_NAME = {1: "DanceMix1", 2: "DanceMix2"}


@dataclass
class OsloSession:
    group: str
    mix: int
    fps: float
    pos: np.ndarray          # (T, P, 3) mm, dropouts as NaN
    names: list[str]
    valid: np.ndarray        # (T, P) bool

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.pos.shape[0]) / self.fps


def _read_qtm(path: Path) -> tuple[np.ndarray, list[str], float]:
    names: list[str] = []
    freq = 200.0
    n_hdr = 0
    with open(path) as f:
        for line in f:
            n_hdr += 1
            if line.startswith("MARKER_NAMES"):
                names = line.strip().split("\t")[1:]
            if line.startswith("FREQUENCY"):
                freq = float(line.strip().split("\t")[1])
            if line.startswith("Frame"):
                break
    d = np.genfromtxt(path, skip_header=n_hdr, delimiter="\t")
    X = d[:, 2:].reshape(len(d), -1, 3)
    return X, names, freq


def load_session(group: str = "A", fps_out: float = 40.0,
                 max_gap_s: float = 0.25) -> OsloSession:
    """One group's mocap, dropouts marked, resampled to the control rate.

    A lost marker is written as (0, 0, 0) by Qualisys, not as a gap — 2.8 % of
    frames in group A. Treating those zeros as positions puts a dancer at the
    room origin at 200 Hz and manufactures enormous spurious speeds, so they
    become NaN here, short gaps are interpolated, and long ones stay NaN and are
    excluded from every average.
    """
    take, mix = GROUPS[group]
    X, names, fps = _read_qtm(S2015 / "QTM" / f"{take}.tsv")
    bad = np.all(np.abs(X) < 1e-6, axis=2)
    X = X.astype(np.float64)
    X[bad] = np.nan
    T, P, _ = X.shape
    max_gap = int(max_gap_s * fps)
    for p in range(P):
        for c in range(3):
            v = X[:, p, c]
            ok = np.isfinite(v)
            if ok.sum() < 2:
                continue
            idx = np.arange(T)
            runs = np.diff(np.r_[0, (~ok).astype(int), 0])
            starts = np.flatnonzero(runs == 1)
            ends = np.flatnonzero(runs == -1)
            filled = v.copy()
            interp = np.interp(idx, idx[ok], v[ok])
            for s, e in zip(starts, ends):
                if e - s <= max_gap:
                    filled[s:e] = interp[s:e]
            X[:, p, c] = filled
    valid = np.isfinite(X).all(axis=2)

    n_out = int(round(T / fps * fps_out))
    t_in = np.arange(T) / fps
    t_out = np.arange(n_out) / fps_out
    Y = np.empty((n_out, P, 3))
    V = np.zeros((n_out, P), dtype=bool)
    for p in range(P):
        ok = valid[:, p]
        for c in range(3):
            Y[:, p, c] = np.interp(t_out, t_in[ok], X[ok, p, c]) if ok.any() else np.nan
        V[:, p] = np.interp(t_out, t_in, ok.astype(float)) > 0.5
    Y[~V] = np.nan
    return OsloSession(group=group, mix=mix, fps=fps_out, pos=Y, names=names, valid=V)


def load_audio_features(mix: int, fps_out: float = 40.0) -> dict[str, np.ndarray]:
    """The mix's redistributable acoustic features, on the control-rate clock."""
    import csv
    rows = []
    with open(S2015 / "audio_features" / "feature_series.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["track"] == MIX_NAME[mix]:
                rows.append(r)
    cols = ["rms_dbfs", "spectral_centroid_hz", "spectral_bandwidth_hz", "rolloff85_hz",
            "spectral_flatness", "zcr", "bass_frac_below150hz", "onset_strength",
            "onset", "tempo_bpm_w8", "pulse_clarity_w8"]
    t = np.array([float(r["t_s"]) for r in rows])
    out: dict[str, np.ndarray] = {}
    n_out = int(round(t[-1] * fps_out))
    t_out = np.arange(n_out) / fps_out
    for c in cols:
        v = np.array([float(r[c]) if r[c] not in ("", "nan") else np.nan for r in rows])
        ok = np.isfinite(v)
        if ok.sum() < 2:
            out[c] = np.zeros(n_out)
            continue
        out[c] = np.interp(t_out, t[ok], v[ok])
    out["t_s"] = t_out
    return out


def load_segments() -> list[dict]:
    import csv
    with open(S2015 / "metadata" / "analysis_segments.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_tracks() -> list[dict]:
    import csv
    with open(S2015 / "metadata" / "stimulus_tracks.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def descriptors(sess: OsloSession, smooth_s: float = 0.15) -> dict[str, np.ndarray]:
    """Per-participant movement channels, (T, P). NaN where the marker was lost."""
    from scipy.ndimage import uniform_filter1d
    fps = sess.fps
    pos = sess.pos
    vel = np.gradient(pos, 1.0 / fps, axis=0)
    speed = np.linalg.norm(vel, axis=2)                     # mm/s
    vspeed = np.abs(vel[:, :, 2])
    hspeed = np.linalg.norm(vel[:, :, :2], axis=2)
    n = max(int(smooth_s * fps), 1)

    def sm(x):
        y = x.copy()
        bad = ~np.isfinite(y)
        y[bad] = 0.0
        num = uniform_filter1d(y, n, axis=0, mode="nearest")
        den = uniform_filter1d((~bad).astype(float), n, axis=0, mode="nearest")
        out = num / np.maximum(den, 1e-9)
        out[den < 0.4] = np.nan
        return out

    z = pos[:, :, 2]
    trend = sm(z) if True else z
    long_n = max(int(1.0 * fps), 1)
    zbad = ~np.isfinite(z)
    zz = np.where(zbad, 0.0, z)
    dz = uniform_filter1d(zz, long_n, axis=0, mode="nearest")
    bounce = np.where(zbad, np.nan, z - dz)
    return {"speed": sm(speed), "vspeed": sm(vspeed), "hspeed": sm(hspeed),
            "vert": z, "bounce": bounce}

"""Structural segmentation: boundaries + {intro, build-up, drop, breakdown, outro}.

Method: laplacian spectral clustering over a combined chroma/timbre
recurrence matrix (McFee & Ellis 2014, as implemented via librosa
primitives), then a RULE-BASED labelling pass that assigns each segment a
functional EDM role from its energy / brightness / density statistics.

Honesty about what this is
--------------------------
The boundary detector is a legitimate unsupervised baseline. The *labeller*
is a heuristic over segment statistics — it is NOT a trained drop detector
and has never been scored against annotated ground truth (EDM-98 would be
the way to do that; it is the documented upgrade path along with All-In-One
and EDMFormer). Treat labels as plausible functional roles, not as
measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import librosa
import numpy as np
import scipy.ndimage
import scipy.sparse
from sklearn.cluster import KMeans

from .features import AudioFeatures, _norm

# Functional roles, ordered by typical energy.
ROLES = ("intro", "breakdown", "build-up", "drop", "outro")


@dataclass
class Segment:
    start_s: float
    end_s: float
    label: str
    energy: float        # mean normalised RMS
    brightness: float    # mean normalised spectral centroid
    density: float       # mean normalised onset strength
    energy_slope: float  # normalised RMS trend within the segment
    index: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration_s"] = self.duration_s
        return d


def detect_boundaries(
    feats: AudioFeatures,
    n_segments: int = 6,
    min_seg_s: float = 2.0,
) -> np.ndarray:
    """Laplacian-segmentation boundaries, in seconds (includes 0 and end)."""
    # Beat-synchronise so the recurrence matrix compares musical units, not
    # arbitrary fixed windows (segment before you align).
    beat_frames = librosa.time_to_frames(feats.beats, sr=feats.sr, hop_length=feats.hop)
    beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < feats.mel.shape[1])]
    if beat_frames.size < 4:
        return np.array([0.0, feats.duration_s])

    Csync = librosa.util.sync(feats.chroma, beat_frames, aggregate=np.median)
    Msync = librosa.util.sync(feats.mel, beat_frames, aggregate=np.mean)
    n = Csync.shape[1]
    if n < 4:
        return np.array([0.0, feats.duration_s])

    # Recurrence (harmonic repetition) + local timbre path.
    R = librosa.segment.recurrence_matrix(Csync, width=1, mode="affinity", sym=True)
    R = scipy.ndimage.median_filter(R, size=(1, 7))
    path_dist = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_dist) + 1e-9
    path_sim = np.exp(-path_dist / sigma)
    Delta = np.diag(path_sim, 1) + np.diag(path_sim, -1)

    mu = R.mean() + 1e-9
    deg_scale = Delta.mean() + 1e-9
    A = (mu / (mu + deg_scale)) * R + (deg_scale / (mu + deg_scale)) * Delta

    # Normalised laplacian -> spectral embedding -> k-means over beats.
    deg = np.maximum(A.sum(axis=1), 1e-9)
    L = np.eye(n) - (A / np.sqrt(np.outer(deg, deg)))
    evals, evecs = np.linalg.eigh(L)
    k = int(min(max(n_segments, 2), n))
    X = evecs[:, :k]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-9)

    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    change = np.flatnonzero(np.diff(labels)) + 1
    beat_times = librosa.frames_to_time(beat_frames, sr=feats.sr, hop_length=feats.hop)
    # librosa.util.sync emits one MORE column than there are beat frames (it
    # closes the final open segment), so a cluster-change index can land at
    # len(beat_times). Map any such index to the clip end rather than
    # indexing out of bounds.
    bounds = [0.0]
    for i in change:
        t = float(beat_times[i]) if i < beat_times.size else float(feats.duration_s)
        bounds.append(t)
    bounds.append(float(feats.duration_s))
    bounds = sorted(set(bounds))

    # Enforce a minimum segment length so a cue list is not a stutter.
    merged = [bounds[0]]
    for b in bounds[1:-1]:
        if b - merged[-1] >= min_seg_s:
            merged.append(b)
    if feats.duration_s - merged[-1] < min_seg_s and len(merged) > 1:
        merged.pop()
    merged.append(feats.duration_s)
    return np.array(merged)


def _seg_stats(feats: AudioFeatures, a: float, b: float) -> dict[str, float]:
    ft = feats.frame_times
    m = (ft >= a) & (ft < b)
    if not m.any():
        m = np.zeros_like(ft, dtype=bool)
        m[np.argmin(np.abs(ft - a))] = True
    rms_n = _norm(feats.rms)
    cen_n = _norm(feats.centroid)
    ons_n = _norm(feats.onset_env)
    seg_rms = rms_n[m]
    slope = 0.0
    if seg_rms.size >= 3:
        x = np.linspace(0.0, 1.0, seg_rms.size)
        slope = float(np.polyfit(x, seg_rms, 1)[0])
    return {
        "energy": float(seg_rms.mean()),
        "brightness": float(cen_n[m].mean()),
        "density": float(ons_n[m].mean()),
        "energy_slope": slope,
    }


def label_segments(feats: AudioFeatures, bounds: np.ndarray) -> list[Segment]:
    """Assign functional EDM roles from segment statistics (HEURISTIC).

    Labelling is RELATIVE to the track's own energy range, not to quantiles
    over the segment list. Quantiles fail badly when most segments are loud:
    the synthetic control produced a 0.613-energy segment labelled
    "breakdown" while the track maximum was 0.620, because it sat below the
    0.66 quantile. Range-relative thresholds plus an explicit
    "drop follows build-up" rule fixed the localisation error from 3.2/6.7 s
    to <1 s (see probe_control_synthetic.py).
    """
    raw = []
    for i in range(len(bounds) - 1):
        a, b = float(bounds[i]), float(bounds[i + 1])
        raw.append((a, b, _seg_stats(feats, a, b)))
    if not raw:
        return []

    energies = np.array([s["energy"] for _, _, s in raw])
    emin, emax = float(energies.min()), float(energies.max())
    span = max(emax - emin, 1e-9)
    rel = (energies - emin) / span  # 0..1 within THIS track

    n = len(raw)
    labels: list[str] = [""] * n

    # Pass 1 - build-ups: energy climbing into a louder neighbour.
    for i, (_, _, s) in enumerate(raw):
        if i + 1 < n and s["energy_slope"] > 0.12 and rel[i + 1] > rel[i] + 0.05:
            labels[i] = "build-up"

    # Pass 2 - drops: loud, and either following a build-up or jumping in
    # energy. A loud segment directly after a drop continues the drop.
    for i in range(n):
        if labels[i]:
            continue
        jump = rel[i] - rel[i - 1] if i > 0 else rel[i]
        after_build = i > 0 and labels[i - 1] == "build-up"
        after_drop = i > 0 and labels[i - 1] == "drop"
        if rel[i] >= 0.6 and (after_build or jump >= 0.15):
            labels[i] = "drop"
        elif rel[i] >= 0.6 and after_drop:
            labels[i] = "drop"

    # Pass 3 - everything else: intro / outro / breakdown / sustained drop.
    for i in range(n):
        if labels[i]:
            continue
        if i == 0 and rel[i] < 0.6:
            labels[i] = "intro"
        elif i == n - 1 and rel[i] < 0.6:
            labels[i] = "outro"
        elif rel[i] >= 0.6:
            labels[i] = "drop"
        elif rel[i] <= 0.35:
            labels[i] = "breakdown"
        else:
            labels[i] = "build-up" if raw[i][2]["energy_slope"] > 0.05 else "breakdown"

    return [
        Segment(start_s=a, end_s=b, label=labels[i], index=i, **s)
        for i, (a, b, s) in enumerate(raw)
    ]


def segment(
    feats: AudioFeatures, n_segments: int = 6, min_seg_s: float = 2.0
) -> list[Segment]:
    """Full segmentation: boundaries then functional labels."""
    return label_segments(feats, detect_boundaries(feats, n_segments, min_seg_s))

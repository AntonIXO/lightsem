"""Per-track audio-motion sync offsets for AIST++ — **REFUTED, do not use**.

Kept as reproducible provenance for a hypothesis that looked strong and was
wrong. The tempting story: pairing an AIST++ clip with its track from sample 0
leaves the dance period-locked but phase-random (footfall beat alignment
+0.019 above the circular-shift null, CI [-0.006, +0.044], while each clip's
footfalls are tightly periodic at R = 0.55, median Rayleigh z = 5.2). Clips
sharing a music id agreed on a phase across different choreographies and
dancers (concentration R = 0.59), the per-track estimate was split-half stable
(r = 0.83), and applying it lifted HELD-OUT footfall alignment to +0.128,
CI [+0.103, +0.153] — better than the audio beat-trigger baseline's +0.101.

Every one of those numbers is real and the conclusion was still wrong. A clock
offset shifts *every* channel of a clip by the same amount, so it must improve
channels it was not fitted to. Fitted on footfalls and tested on vertical
bounce minima of held-out clips, it did the opposite:

    vertical bounce, raw:        +0.0382 above null, CI [+0.0230, +0.0541]
    vertical bounce, corrected:  +0.0081 above null, CI [-0.0068, +0.0228]

So there is no missing clock offset. What the data actually show is a
channel-specific phase: the vertical bounce is aligned to the beat with no
correction at all (the Toiviainen 2010 / Burger 2013 prediction), while
footfall phase is a property of the choreography, which in street dance is
frequently and deliberately off the beat.

The pipeline therefore applies NO correction. ``load_offsets`` returns nothing
unless a file is explicitly passed, and callers must not reintroduce it.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from ..motion.io import CACHE_ROOT

OFFSET_PATH = CACHE_ROOT / "track_sync_offsets.json"
MIN_CLIPS = 6
MIN_R = 0.30


def _circmean(phases: np.ndarray, weights: np.ndarray | None = None):
    a = 2 * np.pi * np.asarray(phases, dtype=np.float64)
    w = np.ones(a.size) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        return 0.0, 0.0
    C = float((w * np.cos(a)).sum() / w.sum())
    S = float((w * np.sin(a)).sum() / w.sum())
    return (np.arctan2(S, C) / (2 * np.pi)) % 1.0, float(np.hypot(C, S))


def fit_track_offsets(index: list[dict], min_clips: int = MIN_CLIPS,
                      min_r: float = MIN_R) -> dict[str, dict]:
    """Estimate one offset per music id from per-clip beat phases."""
    by_music = collections.defaultdict(list)
    for r in index:
        by_music[r["music_id"]].append(r)
    out: dict[str, dict] = {}
    for mid, rows in by_music.items():
        good = [r for r in rows if r.get("beat_phase_R", 0.0) > min_r]
        if len(good) < min_clips:
            out[mid] = {"offset_s": 0.0, "concentration": 0.0, "n_clips": len(good),
                        "fitted": False}
            continue
        beat = 60.0 / float(good[0]["tempo"])
        ph = [(r["beat_offset_s"] / beat) % 1.0 for r in good]
        w = [r["beat_phase_R"] for r in good]
        mu, R = _circmean(np.array(ph), np.array(w))
        signed = mu if mu <= 0.5 else mu - 1.0          # nearest beat, signed
        out[mid] = {"offset_s": float(signed * beat), "concentration": float(R),
                    "n_clips": len(good), "beat_s": float(beat), "fitted": True}
    return out


def save_offsets(offsets: dict, path: str | Path = OFFSET_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(offsets, indent=1, sort_keys=True))
    return path


def load_offsets(path: str | Path | None = None) -> dict:
    """Load fitted offsets. Returns {} by default — the correction is refuted."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def aligned_beats(beats: np.ndarray, offset_s: float) -> np.ndarray:
    """A track's beat grid expressed in the MOTION clock."""
    return np.asarray(beats, dtype=np.float64) + float(offset_s)

"""Corpus-level robust scalers for descriptor channels.

Per-clip normalisation would be a lie the moment this runs live: you cannot
divide by a percentile of a song you have not heard yet. So the mapping from
body to light is calibrated ONCE over the corpus (median and inter-quartile
range per channel) and those constants ship with the conductor.

The squash is ``0.5 + 0.5 * tanh((x - median) / (1.5 * IQR))``: bounded,
smooth, and it degrades gracefully on out-of-range input instead of clipping a
festival's louder-than-training energy to a flat 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCALER_PATH = Path(__file__).with_name("scalers.json")


def fit_scalers(channel_samples: dict[str, np.ndarray]) -> dict[str, dict]:
    out = {}
    for k, v in channel_samples.items():
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            out[k] = {"median": 0.0, "iqr": 1.0}
            continue
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        out[k] = {"median": float(med), "iqr": float(max(q3 - q1, 1e-6)),
                  "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
    return out


def save_scalers(scalers: dict, path: str | Path = SCALER_PATH) -> Path:
    path = Path(path)
    path.write_text(json.dumps(scalers, indent=1, sort_keys=True))
    return path


def load_scalers(path: str | Path = SCALER_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `bash run.sh scripts/data/build_targets.py scalers` first")
    return json.loads(path.read_text())


class Squash:
    """Corpus-calibrated 0..1 squash for one channel set."""

    def __init__(self, scalers: dict):
        self.s = scalers

    def __call__(self, name: str, x: np.ndarray, gain: float = 1.5) -> np.ndarray:
        p = self.s.get(name)
        if p is None:
            x = np.asarray(x, dtype=np.float64)
            return 0.5 + 0.5 * np.tanh(x)
        return 0.5 + 0.5 * np.tanh((np.asarray(x, dtype=np.float64) - p["median"])
                                   / (gain * p["iqr"]))

    def signed(self, name: str, x: np.ndarray, gain: float = 1.5) -> np.ndarray:
        """-1..1 version for channels whose sign is meaningful (laterality)."""
        p = self.s.get(name)
        scale = gain * p["iqr"] if p else 1.0
        return np.tanh(np.asarray(x, dtype=np.float64) / scale)

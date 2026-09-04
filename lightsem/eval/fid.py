"""Frechet distance and diversity over light features, with an honest floor.

FID on a few dozen clips is a biased estimate: the covariance of a 24-d
feature set from 40 samples is badly conditioned, and the bias is of the same
order as the effects we care about. Two consequences are baked in here:

* covariance is shrunk toward its diagonal (Ledoit-Wolf style, fixed lambda),
* :func:`split_half_floor` reports what the SAME distribution scores against
  itself at the same sample size — the number every FID must be read against.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg


def _stats(X: np.ndarray, shrink: float = 0.15):
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    C = np.cov(X, rowvar=False)
    C = np.atleast_2d(C)
    C = (1 - shrink) * C + shrink * np.diag(np.diag(C) + 1e-8)
    return mu, C


def frechet(X: np.ndarray, Y: np.ndarray, shrink: float = 0.15) -> float:
    """Frechet distance between two feature sets (the FID formula)."""
    if len(X) < 3 or len(Y) < 3:
        return float("nan")
    mu1, C1 = _stats(X, shrink)
    mu2, C2 = _stats(Y, shrink)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(C1 @ C2)      # scipy >= 1.18 dropped disp=
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        return float("nan")
    return float(diff @ diff + np.trace(C1) + np.trace(C2) - 2 * np.trace(covmean))


class FeatureScaler:
    """Standardise features on the REAL set so all distances are comparable."""

    def __init__(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-9] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sd


def diversity(X: np.ndarray, n_pairs: int = 2000,
              rng: np.random.Generator | None = None) -> float:
    """Mean pairwise Euclidean distance — mode collapse shows up here, not in FID."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 2:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    m = i != j
    if not m.any():
        return float("nan")
    return float(np.linalg.norm(X[i[m]] - X[j[m]], axis=1).mean())


def split_half_floor(X: np.ndarray, n_rep: int = 20, shrink: float = 0.15,
                     rng: np.random.Generator | None = None) -> dict:
    """FID of a set against itself, at the same sample size. The floor."""
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 8:
        return {"mean": float("nan"), "sd": float("nan"), "n_rep": 0}
    vals = []
    for _ in range(n_rep):
        idx = rng.permutation(n)
        a, b = idx[: n // 2], idx[n // 2:]
        vals.append(frechet(X[a], X[b], shrink))
    v = np.asarray(vals, dtype=float)
    return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)), "n_rep": n_rep}

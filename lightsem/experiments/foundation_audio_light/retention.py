"""Representation-retention diagnostics for adapted music features."""

from __future__ import annotations

from typing import Any

import numpy as np

from .layer_selection import effective_rank


def compare_features(reference: Any, candidate: Any) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cur.shape:
        raise ValueError("reference and candidate feature arrays must have the same shape")
    if ref.ndim != 2 or not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("feature arrays must be finite T,D")
    ref_center = ref - ref.mean(0, keepdims=True)
    cur_center = cur - cur.mean(0, keepdims=True)
    ref_norm = np.linalg.norm(ref_center)
    cur_norm = np.linalg.norm(cur_center)
    cosine = float(np.sum(ref_center * cur_center) / max(ref_norm * cur_norm, 1e-12))
    return {
        "finite": True,
        "frames": int(ref.shape[0]),
        "dimension": int(ref.shape[1]),
        "mse": float(np.mean((ref - cur) ** 2)),
        "centered_cosine": cosine,
        "reference_effective_rank": effective_rank(ref),
        "candidate_effective_rank": effective_rank(cur),
        "reference_constant_fraction": float(np.mean(ref.std(0) <= 1e-8)),
        "candidate_constant_fraction": float(np.mean(cur.std(0) <= 1e-8)),
        "scope": "representation retention diagnostic; not a lighting-quality verdict",
    }


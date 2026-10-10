"""Screening diagnostics and explicit pilot gates for StereoAudioV1."""
from __future__ import annotations

from typing import Mapping

import numpy as np


def representation_gate(metrics: Mapping[str, float | bool | None]) -> dict:
    """Apply the predeclared directional gate without inventing missing scores."""
    required = ("timing_delta", "spatial_relative_error", "relation_lag_delta",
                "timing_regression", "silent_fpr_increase", "lr_swap_pass",
                "finite", "noncollapse")
    missing = [key for key in required if key not in metrics or metrics[key] is None]
    if missing:
        return {"pass": False, "reason": "missing_metrics", "missing": missing}
    timing = float(metrics["timing_delta"])
    spatial = float(metrics["spatial_relative_error"])
    relation = float(metrics["relation_lag_delta"])
    regress = float(metrics["timing_regression"])
    fpr = float(metrics["silent_fpr_increase"])
    pass_value = bool(
        (timing >= .02 or float(metrics.get("timing_auroc_delta", -np.inf)) >= .02)
        and spatial <= -.10 and relation <= -.015 and regress <= .02 and fpr <= .05
        and metrics["lr_swap_pass"] and metrics["finite"] and metrics["noncollapse"]
    )
    return {"pass": pass_value, "reason": "thresholds_and_invariants" if pass_value else "threshold_failure",
            "thresholds": {"timing_delta_min": .02, "spatial_relative_error_max": -.10,
                           "relation_lag_delta_max": -.015, "timing_regression_max": .02,
                           "silent_fpr_increase_max": .05}}


def native_gate(metrics: Mapping[str, float | bool | None]) -> dict:
    required = ("raw_stability_pass", "beat_lag_improvement", "phrase_no_regression",
                "frozen_body_pass", "limb_response_pass", "audio_swap_pass",
                "dmx_retention", "duplicate_increase", "inactivity_increase")
    missing = [key for key in required if key not in metrics or metrics[key] is None]
    if missing:
        return {"pass": False, "reason": "missing_metrics", "missing": missing}
    passed = bool(
        metrics["raw_stability_pass"] and float(metrics["beat_lag_improvement"]) >= .10
        and metrics["phrase_no_regression"] and metrics["frozen_body_pass"]
        and metrics["limb_response_pass"] and metrics["audio_swap_pass"]
        and float(metrics["dmx_retention"]) >= .95
        and float(metrics["duplicate_increase"]) <= .05
        and float(metrics["inactivity_increase"]) <= .05
    )
    return {"pass": passed, "reason": "native_diagnostic_thresholds" if passed else "threshold_failure",
            "thresholds": {"beat_lag_improvement_min": .10, "dmx_retention_min": .95,
                           "duplicate_increase_max": .05, "inactivity_increase_max": .05}}


def feature_diagnostics(features: np.ndarray, events: np.ndarray, spatial: np.ndarray,
                        relation: np.ndarray) -> dict:
    """Report finite/rank/activity/duplicate diagnostics before model scoring."""
    values = (features, events, spatial, relation)
    finite = bool(all(np.isfinite(value).all() for value in values))
    rank = int(np.linalg.matrix_rank(features)) if finite else 0
    duplicate_fraction = float(np.mean(np.all(np.diff(features, axis=0) == 0, axis=1))) if len(features) > 1 else 0.
    return {
        "finite": finite, "rank": rank, "feature_dim": int(features.shape[-1]),
        "noncollapse": bool(rank > 1 and float(np.std(features)) > 1e-8),
        "exact_adjacent_duplicate_fraction": duplicate_fraction,
        "event_count": int(np.sum(events[:, -1] > np.quantile(events[:, -1], .9))) if len(events) else 0,
        "spatial_range": (np.max(spatial, axis=0) - np.min(spatial, axis=0)).astype(float).tolist() if len(spatial) else [],
        "relation_range": (np.max(relation, axis=0) - np.min(relation, axis=0)).astype(float).tolist() if len(relation) else [],
    }


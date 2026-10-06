"""Frozen cross-validation rhythm heads as an explicit Base92 intervention.

Policy is fixed before song inference: divide each of five heads' task scores
by its saved positive threshold, average the five, and peak-pick at one. These
are threshold-relative scores, never calibrated probabilities. The ensemble
is a new diagnostic policy, not a separately validated deployment head.

Only real consecutive detected beats define phase. No beats, extrapolated
tempo, gap repair, downbeat re-anchoring, or independent clock are invented.
The five Base92 timing fields preserve online.py's formulas inside that
coverage, including detection-index modulo four and midpoint-interpolated IBI.
Other 87 acoustic channels remain exactly unchanged. Missing interior beats
therefore produce a longer real interval; that failure is not concealed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import find_peaks

from lightsem.model.light_data import AUDIO_SCALARS, N_MELS


TASKS = ("onsets", "beats", "downbeats")
TIMING_CHANNELS = ("beat_phase_sin", "beat_phase_cos", "beat_accent", "bar_phase", "beat_period_s")
POLICY_ID = "guitarset-five-fold-threshold-relative-mean-v1"
RHYTHM_SOURCE = Path("/mnt/lightsem-data/experiments/lightsem/audio_understanding_20261004/source_rhythm_v2/audio_study/rhythm.py")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing_indices() -> np.ndarray:
    """Derive the feature positions by name and verify the frozen Base92 schema."""
    if N_MELS != 64 or len(AUDIO_SCALARS) != 28 or len(set(AUDIO_SCALARS)) != 28:
        raise ValueError("Expected the unique 64-mel plus 28-scalar Base92 schema")
    indices = np.asarray([N_MELS + AUDIO_SCALARS.index(name) for name in TIMING_CHANNELS])
    if not np.array_equal(indices, np.arange(87, 92)):
        raise ValueError("Base92 timing schema changed; the first 87 acoustic channels must be preserved")
    return indices


def policy_metadata() -> dict:
    """Serializable predeclared policy and the actual implementation identities."""
    return {
        "policy_id": POLICY_ID,
        "head_budget": 5,
        "folds": [0, 1, 2, 3, 4],
        "tasks": list(TASKS),
        "aggregation": "mean(raw_score[fold,task] / saved_positive_threshold[fold,task])",
        "event_threshold": 1.0,
        "event_spacing_s": {"onsets": .025, "beats": .15, "downbeats": .15},
        "score_semantics": "Unbounded threshold-relative score; not probability or calibrated confidence",
        "bar_phase": "(index of previous REAL detected beat modulo 4 + phase) / 4; first detection has index zero",
        "downbeat_use": "Returned diagnostic events only; no downbeat re-anchoring of Base92",
        "coverage": "first real beat <= t < last real beat; no beat extrapolation or insertion",
        "missing_interior_beat": "One longer real interval, without imputed beat or tempo repair",
        "beat_period_s": "Linear interpolation of real IBI values at their interval midpoints, as online.py, inside beat coverage only",
        "unsupported": "TRAIN normalizer mean in raw fields; exactly zero in returned normalized timing fields",
        "native_resampling": "Patch at 40 Hz, linearly interpolate raw Base92 to 30 Hz, then reapply event coverage and normalize; never analytically synthesize a fresh 30 Hz clock",
        "normalization_precision": "Float64 TRAIN mean/std arithmetic then float32 output, matching run_body_transfer_demo.py; baseline and all variants must use the same helper",
        "timing_channels": list(TIMING_CHANNELS),
        "timing_indices": timing_indices().tolist(),
        "unchanged_channels": "Base92 columns 0 through 86, copied without editing",
        "augmentation": False,
        "new_fitting_or_threshold_tuning": False,
        "scope": "New diagnostic ensemble of existing GuitarSet cross-validation heads; not a validated deployment head or full latent-to-dance adapter",
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "base92_reference": {"path": str(Path(__file__).with_name("online.py")), "sha256": sha256(Path(__file__).with_name("online.py"))},
        "rhythm_reference": {"path": str(RHYTHM_SOURCE), "sha256": sha256(RHYTHM_SOURCE)},
    }


def _finite(value, name, ndim=None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (ndim is not None and result.ndim != ndim) or not np.isfinite(result).all():
        raise ValueError(f"Expected finite {name}" + (f" with {ndim} dimensions" if ndim is not None else ""))
    return result


def _times(value, name="timestamps") -> np.ndarray:
    result = _finite(value, name, 1)
    if len(result) > 1 and not np.all(np.diff(result) > 0):
        raise ValueError(f"{name} must be strictly increasing; duplicate timestamps are invalid")
    return result


@dataclass(frozen=True)
class RidgeHead:
    method: str
    fold: int
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray
    thresholds: np.ndarray
    provenance: dict


def threshold_lookup(fit_record: Mapping) -> np.ndarray:
    """Read report thresholds in the saved model's declared task order."""
    thresholds = fit_record.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(TASKS):
        raise ValueError("fit_records thresholds must name exactly onsets, beats, downbeats")
    values = _finite([thresholds[name] for name in TASKS], "saved task thresholds", 1)
    if np.any(values <= 0):
        raise ValueError("Threshold-relative ensemble requires strictly positive saved thresholds")
    return values


def load_fold(fit_record: Mapping) -> RidgeHead:
    """Authenticate a fit_records row and its exact NPZ schema; never fit a head."""
    path = Path(fit_record["checkpoint"])
    expected = fit_record["sha256"]
    if sha256(path) != expected:
        raise ValueError("Ridge checkpoint hash mismatch: " + str(path))
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"mean", "scale", "weights", "intercept", "thresholds"}:
            raise ValueError("Unexpected rhythm ridge NPZ schema")
        arrays = {key: _finite(saved[key], "ridge " + key).copy() for key in saved.files}
    mean, scale, weights = (arrays[k] for k in ("mean", "scale", "weights"))
    if (mean.ndim != 1 or not len(mean) or scale.shape != mean.shape
            or weights.shape != (len(mean), len(TASKS))
            or arrays["intercept"].shape != (len(TASKS),)
            or arrays["thresholds"].shape != (len(TASKS),) or np.any(scale <= 0)):
        raise ValueError("Invalid ridge dimensions or nonpositive scaler")
    if not np.array_equal(arrays["thresholds"], threshold_lookup(fit_record)):
        raise ValueError("NPZ thresholds differ from the authenticated fit record")
    fold = fit_record.get("outer_fold")
    if type(fold) is not int or fold not in range(5):
        raise ValueError("Expected outer fold 0 through 4")
    if fit_record.get("inner_validation_fold") != (fold + 1) % 5 or fit_record.get("alpha") != 1e-3:
        raise ValueError("Changed GuitarSet fold/regularization contract")
    for value in arrays.values():
        value.setflags(write=False)
    return RidgeHead(str(fit_record["method"]), fold, **arrays,
                     provenance={"path": str(path.resolve()), "sha256": expected,
                                 "outer_fold": fold, "inner_validation_fold": (fold + 1) % 5,
                                 "alpha": 1e-3, "thresholds": arrays["thresholds"].tolist()})


def load_ensemble(report_path: str | Path, method: str, *, expected_sha256: str) -> tuple[RidgeHead, ...]:
    """Load exactly all five authenticated heads for one representation."""
    path = Path(report_path)
    if sha256(path) != expected_sha256:
        raise ValueError("Rhythm fit report hash mismatch")
    report = json.loads(path.read_text())
    rows = [row for row in report["fit_records"] if row.get("method") == method]
    if len(rows) != 5 or {row.get("outer_fold") for row in rows} != set(range(5)):
        raise ValueError("Exactly one saved head per fold 0 through 4 is required")
    heads = tuple(load_fold(row) for row in sorted(rows, key=lambda row: row["outer_fold"]))
    if len({len(head.mean) for head in heads}) != 1:
        raise ValueError("All five heads must share the feature dimension")
    for head in heads:
        head.provenance.update({"fit_report": str(path.resolve()), "fit_report_sha256": expected_sha256})
    return heads


def pick_events(scores, times, *, task: str, threshold: float = 1.0) -> np.ndarray:
    """Original scipy peak policy, with explicit uniform-40-Hz validation."""
    if task not in TASKS or not np.isfinite(threshold):
        raise ValueError("Unknown rhythm task or nonfinite threshold")
    times = _times(times)
    scores = _finite(scores, "event scores", 1)
    if scores.shape != times.shape:
        raise ValueError("Score/timestamp length mismatch")
    if len(times) > 1 and not np.allclose(np.diff(times), 1 / 40, rtol=0, atol=1e-7):
        raise ValueError("Rhythm scores must use a complete uniform 40 Hz timeline")
    spacing_s = .025 if task == "onsets" else .15
    spacing = max(1, int(round(spacing_s / np.median(np.diff(times))))) if len(times) > 1 else 1
    return times[find_peaks(scores, height=threshold, distance=spacing)[0]].copy()


def score_ensemble(features, times, heads: tuple[RidgeHead, ...]) -> dict:
    """Score only supplied features. Returns every head, threshold and event."""
    if len(heads) != 5 or tuple(head.fold for head in heads) != tuple(range(5)):
        raise ValueError("Use exactly five heads ordered by folds 0 through 4")
    if len({head.method for head in heads}) != 1:
        raise ValueError("Cannot mix representations inside an ensemble")
    features, times = _finite(features, "rhythm features", 2), _times(times)
    if features.shape != (len(times), len(heads[0].mean)):
        raise ValueError("Feature dimension/timestamp contract differs from trained heads")
    thresholds = np.stack([head.thresholds for head in heads])
    if not np.isfinite(thresholds).all() or np.any(thresholds <= 0):
        raise ValueError("All saved thresholds must be finite and strictly positive")
    raw = np.stack([((features - h.mean) / h.scale) @ h.weights + h.intercept for h in heads])
    relative = _finite(raw / thresholds[:, None, :], "threshold-relative scores", 3)
    ensemble = relative.mean(axis=0)
    events = {task: pick_events(ensemble[:, i], times, task=task) for i, task in enumerate(TASKS)}
    return {"times": times.copy(), "per_head_scores": raw, "thresholds": thresholds,
            "per_head_relative_scores": relative, "ensemble_scores": ensemble,
            "events": events, "metadata": {**policy_metadata(), "method": heads[0].method,
                                             "heads": [dict(head.provenance) for head in heads]}}


def apply_timing_bridge(base92, times, beats, normalizers: Mapping) -> dict:
    """Replace five raw timing fields and return matched normalized input.

    Use ``normalized92`` for inference: unsupported timing fields are assigned
    exact zero after normalization, avoiding float32 rounding of raw TRAIN means.
    The returned raw ``base92`` retains the caller's floating dtype and the first
    87 channels bit-for-bit. A final detected beat has no next real beat, so its
    frame and all later frames are unsupported. Long intervals stay long.
    """
    original = np.asarray(base92)
    if original.dtype.kind != "f" or original.ndim != 2 or original.shape[1] != 92 or not np.isfinite(original).all():
        raise ValueError("Expected finite floating Base92 with shape (T,92)")
    times, beats = _times(times), _times(beats, "detected beats")
    if len(times) != len(original):
        raise ValueError("Base92/timestamp length mismatch")
    mean, std = (_finite(normalizers[key], "TRAIN " + key, 1) for key in ("mean", "std"))
    if mean.shape != (92,) or std.shape != (92,) or np.any(std <= 0):
        raise ValueError("Expected 92 finite TRAIN means and positive standard deviations")
    indices = timing_indices()
    result = original.copy()
    result[:, indices] = mean[indices]
    coverage = np.zeros(len(times), dtype=bool)
    interval_index = np.full(len(times), -1, dtype=np.int64)
    if len(beats) >= 2:
        coverage = (times >= beats[0]) & (times < beats[-1])
        interval_index[coverage] = np.searchsorted(beats, times[coverage], side="right") - 1
        index = interval_index[coverage]
        intervals = np.diff(beats)
        phase = (times[coverage] - beats[index]) / intervals[index]
        period = np.interp(times[coverage], beats[:-1] + intervals / 2, intervals)
        values = np.column_stack((np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase),
                                  (1 - phase) ** 2.5, (index % 4 + phase) / 4, period))
        result[np.ix_(coverage, indices)] = values
    normalized = ((result.astype(np.float64) - mean) / std).astype(np.float32)
    normalized[np.ix_(~coverage, indices)] = 0.
    if not np.isfinite(result).all() or not np.isfinite(normalized).all():
        raise ValueError("Timing bridge produced nonfinite model input")
    if not np.array_equal(result[:, :87], original[:, :87]):
        raise AssertionError("Timing bridge changed acoustic channels")
    return {"base92": result, "normalized92": normalized, "coverage": coverage,
            "interval_index": interval_index, "detected_beats": beats.copy(),
            "real_intervals_s": np.diff(beats), "metadata": {
                **policy_metadata(), "covered_frames": int(coverage.sum()),
                "total_frames": len(times), "detected_beat_count": len(beats),
                "no_beats_inserted": True, "acoustic_channels_exact": True}}


def prepare_native_input(base92_40, times40, times30, normalizers: Mapping, *, beats=None) -> dict:
    """Identical 40-to-30 Hz preprocessing for baseline and interventions.

    ``beats=None`` means the original Base92 baseline. An empty array means the
    explicit no-detections intervention; it never falls back to the baseline.
    This helper preserves the accepted demonstration's float64 normalizer
    arithmetic, not the older training evaluator's float32 normalizer rounding.
    All arms must use the same helper. The consumer's support mask is reapplied
    after interpolation to prevent unsupported-frame leakage at either edge.
    """
    times40, times30 = _times(times40, "40 Hz timestamps"), _times(times30, "30 Hz timestamps")
    if (len(times40) < 2 or not np.allclose(np.diff(times40), 1 / 40, rtol=0, atol=1e-7)
            or (len(times30) > 1 and not np.allclose(np.diff(times30), 1 / 30, rtol=0, atol=1e-7))):
        raise ValueError("Native input requires uniform 40 Hz source and 30 Hz target timestamps")
    if len(times30) and (times30[0] < times40[0] or times30[-1] > times40[-1]):
        raise ValueError("Native interpolation cannot extrapolate outside source timestamps")
    original = np.asarray(base92_40)
    if (original.dtype.kind != "f" or original.shape != (len(times40), 92)
            or not np.isfinite(original).all()):
        raise ValueError("Expected finite floating Base92 matching the 40 Hz timestamps")
    mean, std = (_finite(normalizers[key], "TRAIN " + key, 1) for key in ("mean", "std"))
    if mean.shape != (92,) or std.shape != (92,) or np.any(std <= 0):
        raise ValueError("Expected 92 finite TRAIN means and positive standard deviations")
    intervention = beats is not None
    if intervention:
        patched = apply_timing_bridge(original, times40, beats, normalizers)
        source = patched["base92"]
        events = patched["detected_beats"]
        coverage = ((times30 >= events[0]) & (times30 < events[-1])) if len(events) >= 2 else np.zeros(len(times30), dtype=bool)
    else:
        source = original
        events, coverage = None, None
    raw30 = np.stack([np.interp(times30, times40, source[:, j]) for j in range(92)], axis=1).astype(np.float32)
    if intervention:
        raw30[np.ix_(~coverage, timing_indices())] = mean[timing_indices()]
    normalized = ((raw30 - mean) / std).astype(np.float32)
    if intervention:
        normalized[np.ix_(~coverage, timing_indices())] = 0.
    if not np.isfinite(raw30).all() or not np.isfinite(normalized).all():
        raise ValueError("Native interpolation produced nonfinite input")
    return {"base92": raw30, "normalized92": normalized, "times": times30.copy(),
            "coverage": coverage, "detected_beats": events,
            "metadata": {**policy_metadata(), "intervention": intervention,
                         "baseline": "original Base92" if not intervention else None,
                         "coverage_reapplied_after_resampling": intervention}}

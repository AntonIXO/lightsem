"""Frozen, recording-local metrics for the Stage 11 campaign.

Scene inputs are six-channel, 10 Hz *whole active recordings*. Value channels
must already have their recording-level normalization applied; no evaluator
fits a scaler to generated samples. ``fit_scales`` must receive training data
only. All results are JSON-safe: undefined measurements are ``None``, never a
fabricated zero. Canonical-song and generation-seed aggregation belongs to the
campaign driver; bootstrap inputs have already averaged generation seeds.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import periodogram
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score


VERSION = "stage11_metrics.v1"
LAGS = (1, 5, 10, 20, 40)
SCENE_CHANNELS = ("hue_cos", "hue_sin", "saturation", "value", "value_p90", "hue_concentration")
PRIMARY_MOTION_CHANNELS = ("com_speed", "vert_bounce_amp")


def _matrix(values, name, *, channels=None, min_frames=1):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or len(result) < min_frames or result.shape[1] < 1:
        raise ValueError(f"{name} must be a nonempty (frames, channels) matrix")
    if channels is not None and result.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _ratio(numerator, denominator, epsilon=1e-12):
    return float(numerator / denominator) if denominator > epsilon else None


def _mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def normalize_recording_scene(recording_y, active_mask=None, *, value_indices=(3, 4), floor=1e-3):
    """Normalize value/p90 once per recording, returning data and its transform.

    Fit using active frames only, before any crop. Reuse the returned matrix for
    every crop of this recording. This target representation transform is
    distinct from ``fit_scales``' cross-recording, training-only evaluation
    calibration. An active mask does not trim the returned matrix.
    """
    y = _matrix(recording_y, "recording_y", channels=6)
    floor = _positive(floor, "floor")
    indices = np.asarray(value_indices, dtype=int)
    if len(indices) != len(set(indices.tolist())) or np.any((indices < 0) | (indices >= 6)):
        raise ValueError("value_indices must be unique valid scene channel indices")
    mask = np.ones(len(y), dtype=bool) if active_mask is None else np.asarray(active_mask)
    if mask.shape != (len(y),) or mask.dtype.kind != "b" or mask.sum() < 2:
        raise ValueError("active_mask must select at least two frames")
    means = y[mask][:, indices].mean(axis=0)
    stds = np.maximum(y[mask][:, indices].std(axis=0), floor)
    result = y.copy()
    result[:, indices] = (result[:, indices] - means) / stds
    return result, {"value_indices": indices.tolist(), "mean": means.tolist(),
                    "std": stds.tolist(), "active_frames": int(mask.sum()), "floor": floor}


def _novelty(y, channel_std, fps):
    """Return continuous novelty and the shared domain with complete windows."""
    width = max(1, int(round(fps)))
    scores = np.zeros(len(y), dtype=np.float64)
    valid = np.zeros(len(y), dtype=bool)
    if len(y) <= 2 * width:
        return scores, valid
    smoothed = uniform_filter1d(y, size=3, axis=0, mode="nearest")
    integral = np.vstack([np.zeros((1, y.shape[1])), np.cumsum(smoothed, axis=0)])
    centres = np.arange(width, len(y) - width)
    pre = (integral[centres] - integral[centres - width]) / width
    post = (integral[centres + width] - integral[centres]) / width
    delta = (post - pre) / np.asarray(channel_std)[None, :]
    # Cumulative-sum roundoff must never turn a constant scene into an event.
    delta[:, np.ptp(y, axis=0) == 0.0] = 0.0
    scores[centres] = np.linalg.norm(delta, axis=1)
    valid[centres] = True
    return scores, valid


def fit_scales(training_recordings, *, fps=10.0, lags=LAGS, floor=1e-3, novelty_quantile=0.90):
    """Fit JSON-serializable IQR/novelty calibration from training recordings.

    Differences and pre/post windows never cross recording boundaries. Frame
    pooling is used only to fit fixed calibration statistics, not to weight
    held-out song outcomes.
    """
    fps = _positive(fps, "fps")
    floor = _positive(floor, "floor")
    if not 0 < novelty_quantile < 1:
        raise ValueError("novelty_quantile must lie strictly between zero and one")
    lags = tuple(int(lag) for lag in lags)
    if not lags or len(set(lags)) != len(lags) or any(lag < 1 for lag in lags):
        raise ValueError("lags must be distinct positive integers")
    recordings = [_matrix(y, "training_recording", channels=6) for y in training_recordings]
    if not recordings:
        raise ValueError("at least one training recording is required")
    pooled = np.concatenate(recordings, axis=0)

    def iqr(values):
        return np.maximum(np.subtract(*np.percentile(values, [75, 25], axis=0)), floor)

    lag_iqr = []
    for lag in lags:
        differences = [y[lag:] - y[:-lag] for y in recordings if len(y) > lag]
        if not differences:
            raise ValueError(f"no training recording is long enough for lag {lag}")
        lag_iqr.append(iqr(np.concatenate(differences, axis=0)).tolist())
    channel_std = np.maximum(pooled.std(axis=0), floor)
    valid_novelty = []
    for y in recordings:
        scores, valid = _novelty(y, channel_std, fps)
        if valid.any():
            valid_novelty.append(scores[valid])
    if not valid_novelty:
        raise ValueError("no training recording supports complete novelty windows")
    return {
        "version": VERSION, "fps": fps, "channel_count": 6, "lags": list(lags),
        "floor": floor, "lag_iqr": lag_iqr, "marginal_iqr": iqr(pooled).tolist(),
        "novelty_std": channel_std.tolist(),
        "novelty_threshold": float(np.quantile(np.concatenate(valid_novelty), novelty_quantile)),
        "novelty_quantile": float(novelty_quantile), "novelty_window_s": 1.0,
        "training_recordings": len(recordings), "training_frames": len(pooled),
        "training_novelty_frames": int(sum(len(x) for x in valid_novelty)),
    }


def _validate_scales(scales, fps):
    if scales.get("version") != VERSION or scales.get("channel_count") != 6:
        raise ValueError("incompatible Stage 11 calibration version/channel count")
    if not math.isclose(float(scales["fps"]), fps):
        raise ValueError("evaluation fps must equal calibration fps")
    for key, shape in (("marginal_iqr", (6,)), ("novelty_std", (6,)),
                       ("lag_iqr", (len(scales["lags"]), 6))):
        values = np.asarray(scales[key], dtype=float)
        if values.shape != shape or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"invalid frozen {key}")
    threshold = float(scales["novelty_threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("invalid frozen novelty_threshold")


def scene_novelty(y, scales, *, fps=10.0, return_valid_mask=False):
    """Continuous generated/reference novelty using the same frozen scaling.

    Edge frames without complete one-second pre/post windows have score zero
    and are excluded through the valid mask in every AP calculation.
    """
    fps = _positive(fps, "fps")
    _validate_scales(scales, fps)
    y = _matrix(y, "y", channels=6)
    scores, valid = _novelty(y, scales["novelty_std"], fps)
    return (scores, valid) if return_valid_mask else scores


def _reference_peaks(scores, valid, threshold, refractory_frames):
    if len(scores) < 3:
        return np.array([], dtype=int)
    eligible = valid[1:-1] & (scores[1:-1] > threshold)
    eligible &= (scores[1:-1] > scores[:-2]) & (scores[1:-1] > scores[2:])
    candidates = np.flatnonzero(eligible) + 1
    # Strongest-first nonmaximum suppression; equal strengths prefer earlier.
    ranked = sorted(candidates.tolist(), key=lambda index: (-scores[index], index))
    accepted = []
    for index in ranked:
        if all(abs(index - previous) >= refractory_frames for previous in accepted):
            accepted.append(index)
    return np.asarray(sorted(accepted), dtype=int)


def event_average_precision(reference_binary, scores, *, fps=40.0, tolerance_s=0.2, valid_mask=None):
    """Tie-correct AP for one recording/window with symmetric event tolerance.

    Call independently for separate windows or songs. Pooling them before
    dilation leaks event labels across their artificial boundary. Missing or
    one-class references return ``ap=None`` with a reason and explicit counts.
    """
    fps = _positive(fps, "fps")
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and nonnegative")
    labels = np.asarray(reference_binary)
    predictions = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
        raise ValueError("reference_binary and scores must be matching nonempty vectors")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("reference_binary must contain only finite binary values")
    if not np.isfinite(predictions).all():
        raise ValueError("scores contains nonfinite values")
    valid = np.ones(len(labels), dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    if valid.shape != labels.shape or valid.dtype.kind != "b":
        raise ValueError("valid_mask must be a matching boolean vector")
    radius = int(math.floor(tolerance_s * fps + 1e-9))
    # Excluded centres cannot seed positives in the eligible domain.
    events = labels.astype(bool) & valid
    expanded = maximum_filter1d(events.astype(np.uint8), size=2 * radius + 1,
                                mode="constant", cval=0).astype(bool)
    n_valid = int(valid.sum())
    n_positive = int(expanded[valid].sum())
    n_events = int(events.sum())
    reason = ("no_valid_frames" if n_valid == 0 else "no_reference_events" if n_events == 0
              else "one_class_reference" if n_positive == n_valid else None)
    ap = None if reason else float(average_precision_score(expanded[valid], predictions[valid]))
    return {"ap": ap, "event_count": n_events, "valid_frames": n_valid,
            "positive_frames": n_positive, "negative_frames": n_valid - n_positive,
            "prevalence": float(n_positive / n_valid) if n_valid else None,
            "reason": reason, "tolerance_frames": radius}


def beat_only_scores(n_frames, beat_times, *, fps=10.0, sigma_s=0.1):
    """A continuous Gaussian nearest-beat baseline on recording-local times."""
    fps = _positive(fps, "fps")
    sigma_s = _positive(sigma_s, "sigma_s")
    if n_frames < 1:
        raise ValueError("n_frames must be positive")
    beats = np.asarray(beat_times, dtype=np.float64)
    if beats.ndim != 1 or not np.isfinite(beats).all():
        raise ValueError("beat_times must be a finite vector of local seconds")
    beats = np.sort(beats)
    if len(beats) == 0:
        return np.zeros(n_frames)
    times = np.arange(n_frames) / fps
    insertion = np.searchsorted(beats, times)
    left = beats[np.clip(insertion - 1, 0, len(beats) - 1)]
    right = beats[np.clip(insertion, 0, len(beats) - 1)]
    distance = np.minimum(np.abs(times - left), np.abs(times - right))
    return np.exp(-0.5 * (distance / sigma_s) ** 2)


def _hf_energy(y, std, fps):
    frequencies, power = periodogram(y / np.asarray(std)[None, :], fs=fps, axis=0,
                                     detrend="constant", scaling="spectrum", window="boxcar")
    band = (frequencies >= 2.0) & (frequencies <= 5.0)
    return float(power[band].sum(axis=0).mean())


def scene_metrics(reference_y, generated_y, beat_times, scales, *, fps=10.0,
                  circular_shifts=200, seed=1100, include_arrays=False):
    """Evaluate one reference/generated recording pair with frozen calibration.

    Circular nulls roll continuous scores within the common trimmed evaluation
    domain, preserving the score multiset and reference-positive count. Their
    offsets are at least ten seconds from either end of that domain. Recordings
    too short for this null retain ``None`` null summaries and an explicit reason.
    """
    fps = _positive(fps, "fps")
    _validate_scales(scales, fps)
    reference = _matrix(reference_y, "reference_y", channels=6, min_frames=2)
    generated = _matrix(generated_y, "generated_y", channels=6, min_frames=2)
    if reference.shape != generated.shape:
        raise ValueError("reference and generated recordings must be aligned and equal length")
    if not isinstance(circular_shifts, (int, np.integer)) or circular_shifts < 0:
        raise ValueError("circular_shifts must be a nonnegative integer")
    if len(reference) <= max(scales["lags"]):
        raise ValueError("recording is too short to evaluate every frozen lag")
    terms, real_variation, generated_variation = [], [], []
    for lag, lag_scale in zip(scales["lags"], scales["lag_iqr"]):
        real_delta = reference[lag:] - reference[:-lag]
        generated_delta = generated[lag:] - generated[:-lag]
        for channel, scale in enumerate(lag_scale):
            terms.append(wasserstein_distance(real_delta[:, channel], generated_delta[:, channel]) / scale)
            real_variation.append(np.abs(real_delta[:, channel]).mean() / scale)
            generated_variation.append(np.abs(generated_delta[:, channel]).mean() / scale)
    marginal = [wasserstein_distance(reference[:, ch], generated[:, ch]) / scales["marginal_iqr"][ch]
                for ch in range(6)]
    reference_hf = _hf_energy(reference, scales["novelty_std"], fps)
    generated_hf = _hf_energy(generated, scales["novelty_std"], fps)
    real_scores, valid = _novelty(reference, scales["novelty_std"], fps)
    generated_scores, _ = _novelty(generated, scales["novelty_std"], fps)
    events = _reference_peaks(real_scores, valid, scales["novelty_threshold"],
                              max(1, int(round(2.0 * fps))))
    event_binary = np.zeros(len(reference), dtype=bool)
    event_binary[events] = True
    novelty = event_average_precision(event_binary, generated_scores, fps=fps, valid_mask=valid)
    novelty["eligible"] = novelty["event_count"] >= 3 and novelty["ap"] is not None
    novelty["eligibility_reason"] = (None if novelty["eligible"] else novelty["reason"] or "fewer_than_three_reference_events")
    beat = beat_only_scores(len(reference), beat_times, fps=fps)
    novelty["beat_only_ap"] = event_average_precision(event_binary, beat, fps=fps, valid_mask=valid)["ap"]
    nulls, offsets = [], []
    minimum = int(math.ceil(10.0 * fps))
    maximum = int(valid.sum()) - minimum
    null_reason = None
    if circular_shifts == 0:
        null_reason = "disabled"
    elif novelty["ap"] is None:
        null_reason = novelty["reason"]
    elif maximum < minimum:
        null_reason = "recording_too_short_for_ten_second_offsets"
    else:
        offsets = np.random.default_rng(seed).integers(minimum, maximum + 1, size=circular_shifts).tolist()
        radius = novelty["tolerance_frames"]
        positive = maximum_filter1d(event_binary.astype(np.uint8), size=2 * radius + 1,
                                    mode="constant", cval=0).astype(bool)[valid]
        interior = generated_scores[valid]
        nulls = [float(average_precision_score(positive, np.roll(interior, offset))) for offset in offsets]
    novelty.update({"circular_shift_aps": nulls, "circular_shift_offsets_frames": offsets,
                    "circular_shift_mean_ap": _mean_or_none(nulls),
                    "circular_shift_max_ap": float(max(nulls)) if nulls else None,
                    "circular_shift_reason": null_reason})
    result = {
        "version": VERSION, "frames": len(reference), "fps": fps,
        "temporal_error": float(np.mean(terms)), "temporal_terms": [float(x) for x in terms],
        "temporal_term_count": len(terms), "marginal_error": float(np.mean(marginal)),
        "marginal_terms": [float(x) for x in marginal],
        "variation_ratio": _ratio(np.mean(generated_variation), np.mean(real_variation)),
        "reference_variation": float(np.mean(real_variation)),
        "generated_variation": float(np.mean(generated_variation)),
        "hf_energy_ratio_2to5": _ratio(generated_hf, reference_hf),
        "reference_hf_energy_2to5": reference_hf, "generated_hf_energy_2to5": generated_hf,
        "novelty_ap": novelty["ap"], "novelty": novelty,
    }
    if include_arrays:
        result["arrays"] = {"reference_novelty": real_scores.tolist(), "generated_novelty": generated_scores.tolist(),
                            "reference_event_frames": events.tolist(), "valid_mask": valid.tolist()}
    return result


def _ema_recording(values, fps, tau):
    alpha = -math.expm1(-1.0 / (fps * tau))
    result = values.copy()
    for frame in range(1, len(values)):
        result[frame] = result[frame - 1] + alpha * (values[frame] - result[frame - 1])
    return result


def motion_r2(reference, prediction, channel_names=None, *, fps=40.0, tau=2.0):
    """Within-recording R² after independent causal EMA of target/prediction.

    Mapping inputs infer channel names. Matrix inputs require explicit names;
    no column-order convention can silently assign the primary endpoint. A
    constant reference channel has undefined R² and cannot pass a gate.
    """
    fps, tau = _positive(fps, "fps"), _positive(tau, "tau")
    if isinstance(reference, Mapping):
        if not isinstance(prediction, Mapping):
            raise ValueError("prediction must be a mapping when reference is")
        channel_names = list(channel_names or reference.keys())
        reference = np.column_stack([reference[name] for name in channel_names])
        prediction = np.column_stack([prediction[name] for name in channel_names])
    if channel_names is None:
        raise ValueError("matrix inputs require channel_names")
    names = list(channel_names)
    if len(set(names)) != len(names):
        raise ValueError("channel_names must be unique")
    ref = _matrix(reference, "reference", min_frames=2)
    pred = _matrix(prediction, "prediction", min_frames=2)
    if ref.shape != pred.shape or len(names) != ref.shape[1]:
        raise ValueError("reference, prediction and channel names must align")
    ref, pred = _ema_recording(ref, fps, tau), _ema_recording(pred, fps, tau)
    denominator = np.square(ref - ref.mean(axis=0)).sum(axis=0)
    numerator = np.square(ref - pred).sum(axis=0)
    per_channel = {name: float(1 - numerator[ch] / denominator[ch]) if denominator[ch] > 1e-12 else None
                   for ch, name in enumerate(names)}
    primary = [per_channel.get(name) for name in PRIMARY_MOTION_CHANNELS]
    primary_valid = all(value is not None for value in primary)
    return {"version": VERSION, "frames": len(ref), "fps": fps, "tau_s": tau,
            "r2_by_channel": per_channel, "primary_r2": float(np.mean(primary)) if primary_valid else None,
            "primary_channels": list(PRIMARY_MOTION_CHANNELS),
            "primary_reason": None if primary_valid else "missing_or_constant_primary_channel"}


def paired_hierarchical_bootstrap(baseline, candidate, *, higher_is_better=True,
                                  n_bootstrap=4000, seed=1100, confidence=0.95):
    """Paired, crossed seed/song bootstrap of already aggregated song outcomes.

    Arrays must be complete (training_seed, canonical_song) matrices with the
    same identities in the same order. Independently resample each axis, then
    take their Cartesian product. Shared song and seed effects consequently
    remain shared; seed×song cells are never claimed to be independent songs.
    Improvement is positive for both higher- and lower-is-better metrics.
    """
    base = _matrix(baseline, "baseline")
    cand = _matrix(candidate, "candidate")
    if base.shape != cand.shape:
        raise ValueError("paired matrices must have matching seed and song axes")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    delta = cand - base if higher_is_better else base - cand
    n_seeds, n_songs = delta.shape
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        seed_indices = rng.integers(0, n_seeds, n_seeds)
        song_indices = rng.integers(0, n_songs, n_songs)
        samples[draw] = delta[np.ix_(seed_indices, song_indices)].mean()
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return {"delta": float(delta.mean()), "ci_low": float(low), "ci_high": float(high),
            "confidence": confidence, "n_training_seeds": n_seeds, "n_songs": n_songs,
            "n_paired_cells": int(delta.size), "n_bootstrap": n_bootstrap, "bootstrap_seed": seed,
            "per_seed_delta": delta.mean(axis=1).tolist(), "per_song_delta": delta.mean(axis=0).tolist(),
            "baseline_mean": float(base.mean()), "candidate_mean": float(cand.mean()),
            "higher_is_better": bool(higher_is_better),
            "replicated_training": n_seeds >= 2}


def _finite_metric(value):
    return value is not None and isinstance(value, (float, int, np.floating, np.integer)) and math.isfinite(value)


def _bootstrap_integrity(result, min_training_seeds):
    return (result.get("n_training_seeds", 0) >= min_training_seeds and result.get("n_songs", 0) > 0
            and all(_finite_metric(result.get(key)) for key in ("delta", "ci_low", "ci_high")))


def evaluate_motion_gate(bootstrap_result, *, min_training_seeds=5):
    checks = {"complete_paired_results": _bootstrap_integrity(bootstrap_result, min_training_seeds),
              "higher_is_better": bootstrap_result.get("higher_is_better") is True,
              "magnitude_at_least_0.02": _finite_metric(bootstrap_result.get("delta")) and bootstrap_result["delta"] >= 0.02,
              "interval_above_zero": _finite_metric(bootstrap_result.get("ci_low")) and bootstrap_result["ci_low"] > 0}
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_temporal_gate(baseline_metrics, candidate_metrics, bootstrap_result, *,
                           eligible_fraction, min_training_seeds=5):
    """Prospective magnitude/validity gates on canonical-song aggregate metrics.

    Marginal deterioration allowance is max(5% of baseline error, 0.02) to
    avoid an undefined relative tolerance near zero. Undefined reference
    variation/HF energy stays missing and fails eligibility instead of passing.
    """
    base = baseline_metrics.get("temporal_error")
    cand = candidate_metrics.get("temporal_error")
    bm, cm = baseline_metrics.get("marginal_error"), candidate_metrics.get("marginal_error")
    variation, hf = candidate_metrics.get("variation_ratio"), candidate_metrics.get("hf_energy_ratio_2to5")
    checks = {
        "complete_paired_results": _bootstrap_integrity(bootstrap_result, min_training_seeds),
        "paired_estimand_matches_metrics": (
            bootstrap_result.get("higher_is_better") is False
            and _finite_metric(base) and _finite_metric(cand)
            and _finite_metric(bootstrap_result.get("baseline_mean"))
            and _finite_metric(bootstrap_result.get("candidate_mean"))
            and math.isclose(base, bootstrap_result["baseline_mean"], rel_tol=1e-8, abs_tol=1e-10)
            and math.isclose(cand, bootstrap_result["candidate_mean"], rel_tol=1e-8, abs_tol=1e-10)),
        "magnitude_15_percent_and_0.02": (_finite_metric(base) and _finite_metric(cand)
                                          and base - cand >= max(0.02, 0.15 * base)),
        "interval_above_zero": _finite_metric(bootstrap_result.get("ci_low")) and bootstrap_result["ci_low"] > 0,
        "marginal_fidelity": _finite_metric(bm) and _finite_metric(cm) and cm - bm <= max(0.02, 0.05 * bm),
        "variation_at_least_70_percent": _finite_metric(variation) and variation >= 0.7,
        "hf_energy_at_most_double": _finite_metric(hf) and hf <= 2.0,
        "eligible_at_least_80_percent": _finite_metric(eligible_fraction) and 0.8 <= eligible_fraction <= 1.0,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "required_temporal_reduction": max(0.02, 0.15 * base) if _finite_metric(base) else None}


def evaluate_conditioning_gate(control_bootstraps, *, eligible_fraction, min_training_seeds=5,
                               required_controls=("beat_only", "donor_audio", "shuffled_training", "circular_shift")):
    """Require matched audio to clear every predeclared null on paired songs.

    Each bootstrap is matched-AP minus one control's AP. This prevents choosing
    a weak control after inspecting results. The driver must supply the same
    eligible canonical-song intersection to all four comparisons.
    """
    comparisons = {}
    for name in required_controls:
        result = control_bootstraps.get(name, {})
        comparisons[name] = (_bootstrap_integrity(result, min_training_seeds)
                             and result.get("higher_is_better") is True
                             and result["delta"] >= 0.02 and result["ci_low"] > 0)
    eligible = _finite_metric(eligible_fraction) and 0.8 <= eligible_fraction <= 1.0
    return {"passed": bool(required_controls) and eligible and all(comparisons.values()),
            "checks": {"eligible_at_least_80_percent": eligible, **comparisons}}

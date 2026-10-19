"""Music--motion alignment and conditioning diagnostics.

This module is deliberately a measurement layer.  It does not smooth a saved
native body, create a fallback trajectory, or turn a high beat score into an
artistic claim.  The primary output is a vector of synchrony, structure and
counterfactual measurements with explicit nulls.

The implementation follows the conventions already used by :mod:`lightsem`
(``lightsem.eval.bas``): Beat Alignment is reported next to a metronome upper
reference and a circular-shift null.  Motion statistics are computed from
root-relative, *raw* positions before mapping or DMX emission.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from .bas import bas_report, circular_phase


EPS = 1e-12


def _finite_1d(value: Any, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _body(value: Any) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim == 2 and value.shape[-1] == 3:
        value = value[:, None, :]
    if value.ndim != 3 or value.shape[-1] != 3 or value.shape[0] < 4:
        raise ValueError("body must have shape (T,J,3), with T >= 4")
    if not np.isfinite(value).all():
        raise ValueError("body contains non-finite values")
    return value


def _zscore(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    sd = float(value.std())
    return (value - float(value.mean())) / max(sd, EPS)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size != a.size:
        return None
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if float(np.std(a)) < EPS or float(np.std(b)) < EPS:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size != a.size:
        return None
    if float(np.std(a)) < EPS or float(np.std(b)) < EPS:
        return None
    result = spearmanr(a, b)
    value = float(result.statistic)
    return value if np.isfinite(value) else None


def _nanmean_or_none(value: Any) -> float | None:
    arr = np.asarray(value, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def motion_profile(body: Any, fps: float = 30.0, *, exclude_root: bool = True) -> dict[str, Any]:
    """Return raw root-relative motion features.

    ``energy`` is the mean joint speed.  It is an intentionally simple
    kinematic signal: if a native generator is frozen, this signal is frozen;
    if it only shakes the root, root-relative energy does not award that as a
    dance gesture.
    """
    arr = _body(body)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    relative = arr - arr[:, :1, :]
    velocity = np.diff(relative, axis=0) * float(fps)
    joints = slice(1, None) if exclude_root and arr.shape[1] > 1 else slice(None)
    joint_speed = np.linalg.norm(velocity[:, joints], axis=-1)
    energy = joint_speed.mean(axis=1)
    frame_time = np.arange(1, arr.shape[0], dtype=np.float64) / float(fps)
    jerk = np.diff(relative, n=3, axis=0) * float(fps ** 3)
    jerk_mag = np.linalg.norm(jerk[:, joints], axis=-1).reshape(-1)
    return {
        "body": arr,
        "root_relative": relative,
        "velocity": velocity,
        "joint_speed": joint_speed,
        "energy": energy,
        "time": frame_time,
        "fps": float(fps),
        "finite": True,
        "active_fraction": float(np.mean(energy > 1e-5)),
        "inactive_fraction": float(np.mean(energy <= 1e-5)),
        "speed_p95": float(np.percentile(energy, 95)),
        "jerk_p95": float(np.percentile(jerk_mag, 95)) if jerk_mag.size else None,
        "root_range": float(np.ptp(arr[:, 0, :], axis=0).max()),
        "scope": "raw native body; root-relative kinematics; no filtering or mapper",
    }


def motion_events(
    body: Any,
    fps: float = 30.0,
    *,
    prominence_fraction: float = 0.15,
    height_quantile: float = 0.60,
    refractory_s: float = 0.15,
) -> dict[str, Any]:
    """Detect kinematic events from local maxima of raw joint speed.

    The detector is a descriptive event extractor, not a learned critic.  The
    threshold is expressed relative to the clip and is always returned so an
    audit cannot hide a rate change behind an unnamed constant.
    """
    profile = motion_profile(body, fps)
    energy = np.asarray(profile["energy"], dtype=np.float64)
    spread = float(np.percentile(energy, 95) - np.percentile(energy, 5))
    prominence = max(spread * float(prominence_fraction), EPS)
    height = float(np.quantile(energy, np.clip(height_quantile, 0.0, 1.0)))
    distance = max(1, int(round(float(refractory_s) * fps)))
    peaks, props = find_peaks(energy, distance=distance, prominence=prominence, height=height)
    return {
        "times": profile["time"][peaks],
        "indices": peaks.astype(int),
        "strength": np.asarray(props.get("peak_heights", np.zeros(len(peaks))), dtype=float),
        "count": int(len(peaks)),
        "threshold": {"prominence": prominence, "height": height,
                       "height_quantile": float(height_quantile),
                       "refractory_s": float(refractory_s)},
        "energy": energy,
        "time": profile["time"],
        "profile": profile,
    }


def latent_metrics(latent: Any, *, constant_tolerance: float = 1e-8) -> dict[str, Any]:
    """Measure native foundation latent geometry without calling it a verdict.

    The latent can be ``(T,D)`` or ``(T,slots,D)``.  It is flattened per time
    step so a slot permutation cannot manufacture rank.  Effective rank is
    spectral entropy, while ``constant_channel_fraction`` reports a separate
    output-inactivity mechanism.  A healthy rank does not prove useful audio
    conditioning; pair it with :func:`audio_predictive_gain` and interventions.
    """
    arr = np.asarray(latent, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[0] < 2:
        raise ValueError("latent must have shape (T,D) or (T,...) with T >= 2")
    if not np.isfinite(arr).all():
        return {"finite": False, "shape": list(arr.shape),
                "nonfinite_count": int((~np.isfinite(arr)).sum()),
                "scope": "native latent only; no representation verdict from nonfinite data"}
    matrix = arr.reshape(arr.shape[0], -1)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular * singular
    total = float(energy.sum())
    if total > EPS:
        p = energy[energy > EPS] / total
        effective_rank = float(np.exp(-np.sum(p * np.log(p))))
        rank_fraction = float(effective_rank / max(1, min(centered.shape)))
    else:
        effective_rank, rank_fraction = 0.0, 0.0
    channel_sd = matrix.std(axis=0)
    pair = []
    for i in range(min(matrix.shape[0], 256)):
        for j in range(i + 1, min(matrix.shape[0], 256)):
            ai, bj = matrix[i], matrix[j]
            den = np.linalg.norm(ai) * np.linalg.norm(bj)
            if den > EPS:
                pair.append(float(np.dot(ai, bj) / den))
    return {
        "finite": True,
        "shape": list(arr.shape),
        "effective_rank": effective_rank,
        "rank_fraction": rank_fraction,
        "constant_channel_fraction": float(np.mean(channel_sd <= constant_tolerance)),
        "channel_std_p50": float(np.percentile(channel_sd, 50)),
        "channel_std_p95": float(np.percentile(channel_sd, 95)),
        "temporal_delta_rms": float(np.sqrt(np.mean(np.diff(matrix, axis=0) ** 2))),
        "sampled_pairwise_cosine_median": float(np.median(pair)) if pair else None,
        "scope": "saved native foundation latent; effective rank is a geometry diagnostic, not audio-conditioning proof",
    }


def _aligned_series(
    audio_time: np.ndarray,
    audio_signal: np.ndarray,
    motion_time: np.ndarray,
    motion_signal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate both signals to the coarsest stable common clock."""
    audio_time = _finite_1d(audio_time, "audio_time")
    audio_signal = _finite_1d(audio_signal, "audio_signal")
    motion_time = _finite_1d(motion_time, "motion_time")
    motion_signal = _finite_1d(motion_signal, "motion_signal")
    if len(audio_time) != len(audio_signal) or len(motion_time) != len(motion_signal):
        raise ValueError("time and signal lengths differ")
    if np.any(np.diff(audio_time) <= 0) or np.any(np.diff(motion_time) <= 0):
        raise ValueError("time axes must be strictly increasing")
    lo, hi = max(audio_time[0], motion_time[0]), min(audio_time[-1], motion_time[-1])
    if hi <= lo:
        raise ValueError("audio and motion clocks do not overlap")
    # The audio clock is normally denser; using it preserves onset timing while
    # avoiding an invented high-frequency grid.
    grid = audio_time[(audio_time >= lo) & (audio_time <= hi)]
    if grid.size < 4:
        grid = np.linspace(lo, hi, max(4, min(len(audio_time), len(motion_time))))
    return grid, np.interp(grid, audio_time, audio_signal), np.interp(grid, motion_time, motion_signal)


def lag_scan(
    audio_time: Any,
    audio_signal: Any,
    motion_time: Any,
    motion_signal: Any,
    *,
    max_lag_s: float = 0.75,
    n_null: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure synchrony and a circular-shift null on continuous signals."""
    grid, audio, motion = _aligned_series(audio_time, audio_signal, motion_time, motion_signal)
    step = float(np.median(np.diff(grid)))
    max_lag = max(1, int(round(max_lag_s / max(step, EPS))))
    audio_z, motion_z = _zscore(audio), _zscore(motion)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, m = audio_z[:lag], motion_z[-lag:]
        elif lag > 0:
            a, m = audio_z[lag:], motion_z[:-lag]
        else:
            a, m = audio_z, motion_z
        value = _corr(a, m)
        if value is not None:
            rows.append((value, -abs(lag), lag))
    best = max(rows) if rows else None
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(int(n_null)):
        shift = int(rng.integers(len(motion_z)))
        shifted = np.roll(motion_z, shift)
        vals = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                vals.append(_corr(audio_z[:lag], shifted[-lag:]))
            elif lag > 0:
                vals.append(_corr(audio_z[lag:], shifted[:-lag]))
            else:
                vals.append(_corr(audio_z, shifted))
        vals = [x for x in vals if x is not None]
        if vals:
            null.append(max(vals))
    null_arr = np.asarray(null, dtype=float)
    best_value = float(best[0]) if best else None
    best_lag_s = float(best[2] * step) if best else None
    null_mean = float(null_arr.mean()) if null_arr.size else None
    null_sd = float(null_arr.std()) if null_arr.size else None
    return {
        "zero_lag_correlation": _corr(audio_z, motion_z),
        "best_correlation": best_value,
        "best_lag_s": best_lag_s,
        "lag_resolution_s": step,
        "max_lag_s": float(max_lag_s),
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_z": ((best_value - null_mean) / max(null_sd, EPS)
                    if best_value is not None and null_mean is not None and null_sd is not None else None),
        "null_count": int(null_arr.size),
        "grid_frames": int(grid.size),
        "scope": "continuous onset/kinetic-energy cross-correlation; circular shifts preserve motion autocorrelation",
    }


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray,
                       x_test: np.ndarray, ridge: float) -> np.ndarray:
    """Small deterministic ridge helper used by the predictive-gain probe."""
    mu, sd = x_train.mean(axis=0), x_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    a = (x_train - mu) / sd
    b = (x_test - mu) / sd
    reg = np.eye(a.shape[1] + 1, dtype=float) * float(ridge)
    reg[0, 0] = 0.0  # intercept is the first explicit column
    design = np.column_stack([np.ones(len(a)), a])
    test = np.column_stack([np.ones(len(b)), b])
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y_train)
    return test @ coef


def audio_predictive_gain(
    audio_time: Any,
    audio_signal: Any,
    motion_time: Any,
    motion_signal: Any,
    *,
    history_s: float = 1.0,
    horizon_s: float = 0.5,
    target_fps: float = 10.0,
    folds: int = 5,
    ridge: float = 1e-2,
    n_shuffle: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate audio's incremental predictive information about future motion.

    This is a blocked cross-validated *proxy*, not an unbiased conditional
    mutual-information estimator.  The history-only ridge predicts future
    kinetic energy from its own past; the augmented ridge receives the same
    motion history plus causal audio history.  A circular audio shift is a
    negative control.  Positive gain over both the history baseline and the
    shift null is evidence that the representation carries useful temporal
    information, not proof of choreography.
    """
    if history_s <= 0 or horizon_s <= 0 or target_fps <= 0 or folds < 2:
        raise ValueError("history_s, horizon_s, target_fps and folds must be valid")
    _, audio, motion = _aligned_series(audio_time, audio_signal, motion_time, motion_signal)
    # Re-interpolate to a declared modest clock; this avoids giving a linear
    # probe an accidental advantage from a denser audio frame rate.
    lo = max(float(np.asarray(audio_time)[0]), float(np.asarray(motion_time)[0]))
    hi = min(float(np.asarray(audio_time)[-1]), float(np.asarray(motion_time)[-1]))
    step = 1.0 / float(target_fps)
    grid = np.arange(lo, hi + 0.5 * step, step)
    at = _finite_1d(audio_time, "audio_time")
    mt = _finite_1d(motion_time, "motion_time")
    av = np.interp(grid, at, _finite_1d(audio_signal, "audio_signal"))
    mv = np.interp(grid, mt, _finite_1d(motion_signal, "motion_signal"))
    h = max(1, int(round(history_s * target_fps)))
    f = max(1, int(round(horizon_s * target_fps)))
    rows, targets = [], []
    # Fixed lag taps are transparent and avoid an adaptive neural critic.
    motion_lags = np.unique(np.linspace(1, h, min(h, 8)).round().astype(int))
    audio_lags = np.unique(np.linspace(0, h, min(h + 1, 8)).round().astype(int))
    for t in range(h, len(grid) - f):
        rows.append(np.r_[mv[t - motion_lags], av[t - audio_lags]])
        targets.append(mv[t + f])
    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    if len(y) < max(20, folds * 4):
        return {"status": "unqualified_insufficient_frames", "frames": int(len(y))}
    # Split into contiguous blocks so neighbouring windows do not leak across
    # train/test folds.  The final fold can be shorter by at most one sample.
    fold_ids = np.minimum((np.arange(len(y)) * folds) // len(y), folds - 1)
    def evaluate(audio_values: np.ndarray) -> tuple[float, float]:
        hist = x[:, :len(motion_lags)]
        aug = np.column_stack([hist, audio_values])
        base_losses, aug_losses = [], []
        for fold in range(folds):
            train, test = fold_ids != fold, fold_ids == fold
            if test.sum() < 2 or train.sum() < 2:
                continue
            base_pred = _ridge_fit_predict(hist[train], y[train], hist[test], ridge)
            aug_pred = _ridge_fit_predict(aug[train], y[train], aug[test], ridge)
            base_losses.append(float(np.mean((base_pred - y[test]) ** 2)))
            aug_losses.append(float(np.mean((aug_pred - y[test]) ** 2)))
        return float(np.mean(base_losses)), float(np.mean(aug_losses))
    base_mse, audio_mse = evaluate(x[:, len(motion_lags):])
    rng = np.random.default_rng(seed)
    shuffled_gains = []
    raw_audio = x[:, len(motion_lags):]
    for _ in range(int(n_shuffle)):
        shifted = np.roll(raw_audio, int(rng.integers(len(raw_audio))), axis=0)
        b, s = evaluate(shifted)
        shuffled_gains.append((b - s) / max(b, EPS))
    gain = (base_mse - audio_mse) / max(base_mse, EPS)
    null = np.asarray(shuffled_gains, dtype=float)
    return {
        "status": "measured",
        "history_mse": base_mse,
        "audio_augmented_mse": audio_mse,
        "incremental_gain": float(gain),
        "shuffle_gain_mean": float(null.mean()) if null.size else None,
        "shuffle_gain_sd": float(null.std()) if null.size else None,
        "shuffle_gain_z": float((gain - null.mean()) / max(null.std(), EPS)) if null.size else None,
        "history_s": float(history_s), "horizon_s": float(horizon_s),
        "target_fps": float(target_fps), "folds": int(folds),
        "frames": int(len(y)),
        "scope": "blocked ridge probe; incremental audio predictive gain proxy, not strict CMI or choreography proof",
    }


def feature_family_predictive_gain(
    audio_time: Any,
    audio_features: Any,
    motion_time: Any,
    motion_signal: Any,
    *,
    history_s: float = 1.0,
    horizon_s: float = 0.5,
    target_fps: float = 10.0,
    pca_components: int = 16,
    folds: int = 5,
    ridge: float = 1e-2,
    n_shuffle: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare a simple or foundation feature family above motion history.

    ``audio_features`` is ``(T,D)`` and may be Base92, a MuQ hidden layer, or
    any explicitly named frozen representation.  Each fold fits a
    train-only standardized PCA before the ridge probe, preventing a high-
    dimensional layer from winning through scale or test leakage.  A circular
    feature-time shift is the null.  This is the advanced-layer counterpart to
    :func:`audio_predictive_gain`, still a predictive proxy rather than CMI.
    """
    at = _finite_1d(audio_time, "audio_time")
    mt = _finite_1d(motion_time, "motion_time")
    features = np.asarray(audio_features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    if features.ndim != 2 or len(features) != len(at) or not np.isfinite(features).all():
        raise ValueError("audio_features must be finite with shape (len(audio_time), D)")
    motion = _finite_1d(motion_signal, "motion_signal")
    if len(mt) != len(motion):
        raise ValueError("motion_time and motion_signal lengths differ")
    if history_s <= 0 or horizon_s <= 0 or target_fps <= 0 or folds < 2:
        raise ValueError("history_s, horizon_s, target_fps and folds must be valid")
    lo, hi = max(at[0], mt[0]), min(at[-1], mt[-1])
    step = 1.0 / float(target_fps)
    grid = np.arange(lo, hi + 0.5 * step, step)
    av = np.column_stack([np.interp(grid, at, features[:, d]) for d in range(features.shape[1])])
    mv = np.interp(grid, mt, motion)
    h = max(1, int(round(history_s * target_fps)))
    f = max(1, int(round(horizon_s * target_fps)))
    motion_lags = np.unique(np.linspace(1, h, min(h, 8)).round().astype(int))
    feature_lags = np.unique(np.linspace(0, h, min(h + 1, 4)).round().astype(int))
    rows_motion, rows_audio, targets = [], [], []
    for t in range(h, len(grid) - f):
        rows_motion.append(mv[t - motion_lags])
        rows_audio.append(np.concatenate([av[t - lag] for lag in feature_lags]))
        targets.append(mv[t + f])
    hm = np.asarray(rows_motion, dtype=float)
    ha = np.asarray(rows_audio, dtype=float)
    y = np.asarray(targets, dtype=float)
    if len(y) < max(20, folds * 4):
        return {"status": "unqualified_insufficient_frames", "frames": int(len(y)),
                "feature_dim": int(features.shape[1])}
    fold_ids = np.minimum((np.arange(len(y)) * folds) // len(y), folds - 1)

    def project(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = train.mean(0), train.std(0)
        sd[sd < 1e-8] = 1.0
        a, b = (train - mu) / sd, (test - mu) / sd
        _, _, vt = np.linalg.svd(a, full_matrices=False)
        k = min(int(pca_components), vt.shape[0], max(1, len(train) - 1))
        basis = vt[:k].T
        return a @ basis, b @ basis

    def evaluate(feature_rows: np.ndarray) -> tuple[float, float]:
        base_losses, aug_losses = [], []
        for fold in range(folds):
            train, test = fold_ids != fold, fold_ids == fold
            if test.sum() < 2 or train.sum() < 2:
                continue
            tr_a, te_a = project(feature_rows[train], feature_rows[test])
            base_pred = _ridge_fit_predict(hm[train], y[train], hm[test], ridge)
            aug = np.column_stack([hm[train], tr_a])
            aug_test = np.column_stack([hm[test], te_a])
            aug_pred = _ridge_fit_predict(aug, y[train], aug_test, ridge)
            base_losses.append(float(np.mean((base_pred - y[test]) ** 2)))
            aug_losses.append(float(np.mean((aug_pred - y[test]) ** 2)))
        return float(np.mean(base_losses)), float(np.mean(aug_losses))

    base_mse, feature_mse = evaluate(ha)
    rng = np.random.default_rng(seed)
    null_gains = []
    for _ in range(int(n_shuffle)):
        shifted = np.roll(ha, int(rng.integers(len(ha))), axis=0)
        b, s = evaluate(shifted)
        null_gains.append((b - s) / max(b, EPS))
    null = np.asarray(null_gains, dtype=float)
    gain = (base_mse - feature_mse) / max(base_mse, EPS)
    return {
        "status": "measured",
        "feature_dim": int(features.shape[1]),
        "pca_components": int(min(pca_components, features.shape[1], max(1, len(y) - 1))),
        "history_mse": base_mse,
        "feature_augmented_mse": feature_mse,
        "incremental_gain": float(gain),
        "shuffle_gain_mean": float(null.mean()) if null.size else None,
        "shuffle_gain_sd": float(null.std()) if null.size else None,
        "shuffle_gain_z": float((gain - null.mean()) / max(null.std(), EPS)) if null.size else None,
        "history_s": float(history_s), "horizon_s": float(horizon_s),
        "target_fps": float(target_fps), "folds": int(folds), "frames": int(len(y)),
        "scope": "blocked ridge + train-only PCA; feature-family incremental audio predictive proxy",
    }


def _sample_at(times: np.ndarray, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.interp(points, times, values, left=np.nan, right=np.nan)


def beat_bin_alignment(
    audio_time: Any,
    audio_onset: Any,
    motion_time: Any,
    motion_energy: Any,
    beats: Any,
    *,
    n_null: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare audio accents and motion energy in the same beat bins."""
    at, ao = _finite_1d(audio_time, "audio_time"), _finite_1d(audio_onset, "audio_onset")
    mt, me = _finite_1d(motion_time, "motion_time"), _finite_1d(motion_energy, "motion_energy")
    bt = _finite_1d(beats, "beats")
    if len(at) != len(ao) or len(mt) != len(me):
        raise ValueError("feature time and value lengths differ")
    if len(bt) < 4 or np.any(np.diff(bt) <= 0):
        return {"status": "unqualified_insufficient_beats", "count": int(len(bt))}
    # A bin is the interval preceding each beat.  This avoids evaluating the
    # same onset sample at the beat and makes the metric sensitive to accents
    # between beats as well as to the beat event itself.
    left, right = bt[:-1], bt[1:]
    a, m = [], []
    for lo, hi in zip(left, right):
        am = (at >= lo) & (at < hi)
        mm = (mt >= lo) & (mt < hi)
        if am.any() and mm.any():
            a.append(float(np.mean(ao[am])))
            m.append(float(np.mean(me[mm])))
    a, m = np.asarray(a), np.asarray(m)
    observed = _rank_corr(a, m)
    rng = np.random.default_rng(seed)
    null = []
    if len(m) >= 4:
        for _ in range(int(n_null)):
            null.append(_rank_corr(a, np.roll(m, int(rng.integers(len(m))))))
    null = np.asarray([x for x in null if x is not None], dtype=float)
    nm, ns = (float(null.mean()), float(null.std())) if null.size else (None, None)
    return {
        "status": "measured" if observed is not None else "unqualified_constant_signal",
        "audio_motion_rank_correlation": observed,
        "null_mean": nm,
        "null_sd": ns,
        "null_z": ((observed - nm) / max(ns, EPS)
                    if observed is not None and nm is not None and ns is not None else None),
        "count": int(len(a)),
        "scope": "beat-bin Spearman correlation; circular beat-bin permutation null",
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > EPS else None


def bar_structure_alignment(
    audio_time: Any,
    audio_onset: Any,
    motion_time: Any,
    motion_energy: Any,
    beats: Any,
    *,
    beats_per_bar: int = 4,
) -> dict[str, Any]:
    """Measure within-bar accent/energy shape, not only tempo locking."""
    at, ao = _finite_1d(audio_time, "audio_time"), _finite_1d(audio_onset, "audio_onset")
    mt, me = _finite_1d(motion_time, "motion_time"), _finite_1d(motion_energy, "motion_energy")
    bt = _finite_1d(beats, "beats")
    if len(at) != len(ao) or len(mt) != len(me) or beats_per_bar < 2:
        raise ValueError("invalid bar-alignment inputs")
    if len(bt) < beats_per_bar + 2 or np.any(np.diff(bt) <= 0):
        return {"status": "unqualified_insufficient_beats", "bar_count": 0}
    profiles = []
    for start in range(0, len(bt) - beats_per_bar, beats_per_bar):
        bins_a, bins_m = [], []
        for lo, hi in zip(bt[start:start + beats_per_bar], bt[start + 1:start + beats_per_bar + 1]):
            aa, mm = (at >= lo) & (at < hi), (mt >= lo) & (mt < hi)
            if not aa.any() or not mm.any():
                break
            bins_a.append(float(np.mean(ao[aa])))
            bins_m.append(float(np.mean(me[mm])))
        if len(bins_a) == beats_per_bar:
            av, mv = np.asarray(bins_a), np.asarray(bins_m)
            # Normalise inside each bar so the score sees accent placement,
            # while absolute loudness remains a separate diagnostic.
            av, mv = av / max(av.sum(), EPS), mv / max(mv.sum(), EPS)
            profiles.append((_cosine(av, mv), _rank_corr(av, mv)))
    cos = np.asarray([x[0] for x in profiles if x[0] is not None], dtype=float)
    rank = np.asarray([x[1] for x in profiles if x[1] is not None], dtype=float)
    return {
        "status": "measured" if cos.size else "unqualified_constant_signal",
        "cosine_mean": float(cos.mean()) if cos.size else None,
        "cosine_median": float(np.median(cos)) if cos.size else None,
        "rank_correlation_mean": float(rank.mean()) if rank.size else None,
        "bar_count": int(len(profiles)),
        "beats_per_bar": int(beats_per_bar),
        "scope": "within-bar normalized accent/kinetic-energy profiles; no phrase labels assumed",
    }


def phrase_structure_alignment(
    audio_time: Any,
    audio_onset: Any,
    motion_time: Any,
    motion_energy: Any,
    beats: Any,
    *,
    beats_per_bar: int = 4,
    n_null: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare beat-synchronous bar recurrence and phrase novelty.

    Each bar is represented by its normalized within-bar accent/energy profile
    plus a standardized bar-total feature.  The recurrence correlation checks
    whether repeated/changed bars occur in the same places; novelty correlation
    checks boundaries between phrases.  A circular bar shift is the null.  A
    repeated one-bar metronome can score beat alignment but has no matching
    novelty pattern and normally fails this layer.
    """
    at, ao = _finite_1d(audio_time, "audio_time"), _finite_1d(audio_onset, "audio_onset")
    mt, me = _finite_1d(motion_time, "motion_time"), _finite_1d(motion_energy, "motion_energy")
    bt = _finite_1d(beats, "beats")
    if len(at) != len(ao) or len(mt) != len(me) or beats_per_bar < 2:
        raise ValueError("invalid phrase-alignment inputs")
    if len(bt) < 2 * beats_per_bar + 1 or np.any(np.diff(bt) <= 0):
        return {"status": "unqualified_insufficient_beats", "bar_count": 0}
    audio_bars, motion_bars = [], []
    for start in range(0, len(bt) - beats_per_bar, beats_per_bar):
        aa, mm = [], []
        for lo, hi in zip(bt[start:start + beats_per_bar], bt[start + 1:start + beats_per_bar + 1]):
            am, mmask = (at >= lo) & (at < hi), (mt >= lo) & (mt < hi)
            if not am.any() or not mmask.any():
                break
            aa.append(float(np.mean(ao[am])))
            mm.append(float(np.mean(me[mmask])))
        if len(aa) == beats_per_bar:
            audio_bars.append(aa)
            motion_bars.append(mm)
    a = np.asarray(audio_bars, dtype=float)
    m = np.asarray(motion_bars, dtype=float)
    if len(a) < 4:
        return {"status": "unqualified_insufficient_bars", "bar_count": int(len(a))}
    def profile(x):
        shape = x / np.maximum(x.sum(1, keepdims=True), EPS)
        total = np.log1p(x.sum(1))
        total = (total - total.mean()) / max(float(total.std()), EPS)
        return np.column_stack([shape, total])
    ap, mp = profile(a), profile(m)
    def upper_dist(x):
        x = (x - x.mean(0)) / np.maximum(x.std(0), EPS)
        d = x[:, None, :] - x[None, :, :]
        return np.sqrt(np.sum(d * d, axis=-1))[np.triu_indices(len(x), 1)]
    ad, md = upper_dist(ap), upper_dist(mp)
    recurrence = _corr(ad, md)
    an = np.linalg.norm(np.diff(ap, axis=0), axis=1)
    mn = np.linalg.norm(np.diff(mp, axis=0), axis=1)
    novelty = _rank_corr(an, mn)
    rng = np.random.default_rng(seed)
    null_rec, null_nov = [], []
    for _ in range(int(n_null)):
        shift = int(rng.integers(len(mp)))
        rolled = np.roll(mp, shift, axis=0)
        rd = upper_dist(rolled)
        rnov = np.linalg.norm(np.diff(rolled, axis=0), axis=1)
        value = _corr(ad, rd)
        nvalue = _rank_corr(an, rnov)
        if value is not None:
            null_rec.append(value)
        if nvalue is not None:
            null_nov.append(nvalue)
    nr, nn = np.asarray(null_rec), np.asarray(null_nov)
    rec_z = ((recurrence - nr.mean()) / max(nr.std(), EPS)
             if recurrence is not None and nr.size else None)
    nov_z = ((novelty - nn.mean()) / max(nn.std(), EPS)
             if novelty is not None and nn.size else None)
    return {
        "status": "measured" if recurrence is not None or novelty is not None else "unqualified_constant_signal",
        "bar_count": int(len(a)),
        "beats_per_bar": int(beats_per_bar),
        "recurrence_correlation": recurrence,
        "novelty_correlation": novelty,
        "recurrence_null_mean": float(nr.mean()) if nr.size else None,
        "recurrence_null_sd": float(nr.std()) if nr.size else None,
        "recurrence_null_z": rec_z,
        "novelty_null_mean": float(nn.mean()) if nn.size else None,
        "novelty_null_sd": float(nn.std()) if nn.size else None,
        "novelty_null_z": nov_z,
        "audio_novelty_mean": float(an.mean()) if an.size else None,
        "motion_novelty_mean": float(mn.mean()) if mn.size else None,
        "scope": "beat-synchronous bar recurrence and phrase novelty; circular bar shift null",
    }


def _actor_descriptor(body: Any, audio: Mapping[str, Any], fps: float) -> np.ndarray:
    """Compact distribution descriptor for actor calibration, not imitation."""
    profile = motion_profile(body, fps)
    report = synchrony_report(body, audio, fps=fps, n_null=30, seed=0)
    events = report["events"]
    energy = profile["energy"]
    joint = profile["joint_speed"].mean(0)
    joint = joint / max(float(joint.sum()), EPS)
    values = [
        float(events.get("count", 0) / max(profile["time"][-1], EPS)),
        float(profile["active_fraction"]),
        float(profile["speed_p95"]),
        float(profile["jerk_p95"] or 0.0),
        float(report["circular_phase"].get("R") or 0.0),
        float(report["beat_bins"].get("audio_motion_rank_correlation") or 0.0),
        float(report["bar_structure"].get("cosine_mean") or 0.0),
        float(report["phrase_structure"].get("recurrence_correlation") or 0.0),
        float(report["phrase_structure"].get("novelty_correlation") or 0.0),
        *joint.tolist(),
    ]
    return np.asarray(values, dtype=float)


def actor_calibration(
    generated: Any,
    references: Sequence[Any],
    audio: Mapping[str, Any],
    *,
    fps: float = 30.0,
) -> dict[str, Any]:
    """Compare generated descriptors to an actor reference distribution.

    The result is a percentile/calibration diagnostic.  It never computes a
    framewise target loss and does not claim the generated choreography should
    copy any one actor.
    """
    if len(references) < 2:
        return {"status": "unqualified_need_two_actor_references", "reference_count": len(references)}
    ref = np.asarray([_actor_descriptor(body, audio, fps) for body in references], dtype=float)
    gen = _actor_descriptor(generated, audio, fps)
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd < 1e-8] = 1.0
    rz = (ref - mu) / sd
    gz = (gen - mu) / sd
    pair = []
    for i in range(len(rz)):
        d = np.delete(rz, i, axis=0) - rz[i]
        if len(d):
            pair.append(float(np.min(np.linalg.norm(d, axis=1))))
    gen_dist = np.linalg.norm(rz - gz, axis=1)
    nearest = float(gen_dist.min()) if gen_dist.size else None
    percentile = (float(np.mean(np.asarray(pair) <= nearest) * 100)
                  if nearest is not None and pair else None)
    return {
        "status": "measured",
        "reference_count": int(len(ref)),
        "descriptor_dim": int(ref.shape[1]),
        "generated_nearest_reference_distance": nearest,
        "real_leave_one_out_nearest_median": float(np.median(pair)) if pair else None,
        "generated_distance_percentile_vs_real": percentile,
        "scope": "actor distribution calibration; no framewise imitation or choreography proof",
    }


def synchrony_report(
    body: Any,
    audio: Mapping[str, Any],
    *,
    fps: float = 30.0,
    n_null: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute beat, continuous, bar and phase diagnostics for one pair."""
    profile = motion_profile(body, fps)
    events = motion_events(body, fps)
    time = _finite_1d(audio["time"], "audio.time")
    onset = _finite_1d(audio["onset"], "audio.onset")
    beats = _finite_1d(audio["beats"], "audio.beats")
    duration = float(max(time[-1], profile["time"][-1]))
    # ``bas_report`` needs at least one null draw to avoid an undefined
    # nanmean path.  Conditioning matrices may request zero null draws for a
    # quick cross-song score; one deterministic draw keeps the report finite.
    if len(events["times"]):
        beat = bas_report(events["times"], beats, duration, n_null=max(1, int(n_null)),
                          rng=np.random.default_rng(seed))
    else:
        beat = {"bas": None, "metronome": None, "null_mean": None,
                "null_sd": None, "ratio_metronome": None, "z": None,
                "n_events": 0, "sigma": 0.1}
    phase = circular_phase(events["times"], beats)
    lag = lag_scan(time, onset, profile["time"], profile["energy"],
                   n_null=n_null, seed=seed)
    bins = beat_bin_alignment(time, onset, profile["time"], profile["energy"],
                              beats, n_null=n_null, seed=seed)
    bars = bar_structure_alignment(time, onset, profile["time"], profile["energy"], beats,
                                   beats_per_bar=int(audio.get("beats_per_bar", 4)))
    phrases = phrase_structure_alignment(time, onset, profile["time"], profile["energy"], beats,
                                         beats_per_bar=int(audio.get("beats_per_bar", 4)),
                                         n_null=n_null, seed=seed)
    return {
        "motion": {k: v for k, v in profile.items() if k not in {"body", "root_relative", "velocity", "joint_speed", "energy", "time"}},
        "events": {k: v for k, v in events.items() if k not in {"energy", "time", "profile"}},
        "beat_alignment": beat,
        "circular_phase": phase,
        "continuous": lag,
        "beat_bins": bins,
        "bar_structure": bars,
        "phrase_structure": phrases,
        "audio": {"duration_s": duration, "beat_count": int(len(beats)),
                  "beats_per_bar": int(audio.get("beats_per_bar", 4))},
    }


def conditioning_matrix(
    bodies: Mapping[str, Any],
    audios: Mapping[str, Mapping[str, Any]],
    *,
    fps: float = 30.0,
    n_null: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Score every generated body against every song's audio.

    The diagonal-minus-off-diagonal margin is a direct anti-reuse test.  A
    model that emits one attractive trajectory for every song can have a high
    BAS against every beat grid, but it cannot earn a diagonal advantage.
    """
    labels = [str(k) for k in bodies]
    if set(labels) != {str(k) for k in audios} or len(labels) < 2:
        return {"status": "unqualified_missing_two_song_pairs", "labels": labels}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    evidence_matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    details: dict[str, Any] = {}
    for i, body_label in enumerate(labels):
        details[body_label] = {}
        for j, audio_label in enumerate(labels):
            report = synchrony_report(bodies[body_label], audios[audio_label],
                                      fps=fps, n_null=n_null, seed=seed)
            value = report["beat_alignment"].get("bas")
            matrix[i, j] = float(value) if value is not None else np.nan
            components = [float(value)] if value is not None else []
            bin_value = report["beat_bins"].get("audio_motion_rank_correlation")
            bar_value = report["bar_structure"].get("cosine_mean")
            if bin_value is not None:
                components.append((float(bin_value) + 1.0) / 2.0)
            if bar_value is not None:
                components.append(float(bar_value))
            if components:
                evidence_matrix[i, j] = float(np.mean(components))
            details[body_label][audio_label] = {
                "bas": value,
                "beat_bin_rank_correlation": report["beat_bins"].get("audio_motion_rank_correlation"),
                "bar_cosine": report["bar_structure"].get("cosine_mean"),
                "best_lag_s": report["continuous"].get("best_lag_s"),
            }
    diagonal = np.diag(matrix)
    off = matrix[~np.eye(len(labels), dtype=bool)]
    pair_rows = []
    for i, label in enumerate(labels):
        wrong = np.delete(evidence_matrix[i], i)
        own_evidence = evidence_matrix[i, i]
        wrong_bas = _nanmean_or_none(np.delete(matrix[i], i))
        wrong_evidence = _nanmean_or_none(wrong)
        pair_rows.append({"body": label, "own_audio_bas": float(diagonal[i]) if np.isfinite(diagonal[i]) else None,
                          "wrong_audio_bas_mean": wrong_bas,
                          "own_audio_evidence": float(own_evidence) if np.isfinite(own_evidence) else None,
                          "wrong_audio_evidence_mean": wrong_evidence,
                          "margin": (float(own_evidence - wrong_evidence)
                                     if np.isfinite(own_evidence) and wrong_evidence is not None else None)})
    evidence_diag = np.diag(evidence_matrix)
    evidence_off = evidence_matrix[~np.eye(len(labels), dtype=bool)]
    diag_mean, off_mean = _nanmean_or_none(evidence_diag), _nanmean_or_none(evidence_off)
    return {
        "status": "measured" if np.isfinite(matrix).all() else "unqualified_nonfinite_score",
        "labels": labels,
        "bas_matrix": matrix,
        "evidence_matrix": evidence_matrix,
        "diagonal_mean": diag_mean,
        "off_diagonal_mean": off_mean,
        "diagonal_margin": (float(diag_mean - off_mean)
                            if diag_mean is not None and off_mean is not None else None),
        "pair_rows": pair_rows,
        "details": details,
        "scope": "same evaluator, same clock, every body against every song; diagonal is own-song conditioning evidence",
    }


def classify(
    synchrony: Mapping[str, Any],
    conditioning: Mapping[str, Any] | None = None,
    *,
    min_bas_null_z: float = 2.0,
    min_conditioning_margin: float = 0.03,
    min_bar_cosine: float = 0.60,
) -> dict[str, Any]:
    """Apply conservative, preregisterable labels to measured evidence.

    This is intentionally a gate report rather than a weighted artistic score.
    Thresholds are arguments and are echoed in the output; changing them is a
    new protocol.  ``diagnostic_only`` remains the highest result without a
    calibrated actor reference and raw-body health receipt.
    """
    beat = synchrony.get("beat_alignment", {})
    bins = synchrony.get("beat_bins", {})
    bars = synchrony.get("bar_structure", {})
    motion = synchrony.get("motion", {})
    reasons = []
    if not motion.get("finite", False):
        return {"status": "failed_safety", "reasons": ["nonfinite_motion"]}
    if motion.get("active_fraction", 0.0) <= 0.01:
        return {"status": "collapsed_motion", "reasons": ["inactive_root_relative_body"]}
    beat_ok = (beat.get("z") is not None and beat.get("z", -np.inf) >= min_bas_null_z)
    bin_ok = (bins.get("null_z") is not None and bins.get("null_z", -np.inf) >= min_bas_null_z)
    bar_ok = (bars.get("cosine_mean") is not None and bars.get("cosine_mean", -np.inf) >= min_bar_cosine)
    if not beat_ok:
        reasons.append("beat_alignment_not_above_shift_null")
    if not bin_ok:
        reasons.append("accent_energy_not_above_shift_null")
    if not bar_ok:
        reasons.append("within_bar_structure_missing")
    cond_ok = False
    if conditioning is not None:
        margin = conditioning.get("diagonal_margin")
        cond_ok = margin is not None and margin >= min_conditioning_margin
        if not cond_ok:
            reasons.append("no_own_song_diagonal_advantage")
    else:
        reasons.append("conditioning_matrix_missing")
    if cond_ok and beat_ok and bin_ok and bar_ok:
        status = "diagnostic_music_aligned"
    elif beat_ok and not bin_ok and not bar_ok:
        status = "beat_locked_only"
    elif conditioning is not None and not cond_ok:
        status = "unconditioned_or_shared_prior"
    else:
        status = "unqualified_musicality"
    return {
        "status": status,
        "reasons": reasons,
        "thresholds": {"min_bas_null_z": float(min_bas_null_z),
                        "min_conditioning_margin": float(min_conditioning_margin),
                        "min_bar_cosine": float(min_bar_cosine)},
        "interpretation": "engineering evidence only; artistic acceptance still requires calibrated actor reference and human review",
    }


__all__ = [
    "motion_profile", "motion_events", "latent_metrics", "lag_scan",
    "audio_predictive_gain", "feature_family_predictive_gain", "beat_bin_alignment",
    "bar_structure_alignment", "phrase_structure_alignment", "actor_calibration",
    "synchrony_report", "conditioning_matrix", "classify",
]

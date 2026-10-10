"""Versioned stereo and spatial audio descriptors for the dance pilot.

This module deliberately treats a mastered stereo mix as a *stereo image*.
It does not claim physical azimuth, distance, or elevation.  MuQ/Base92 stay
on their existing mono contracts; these descriptors are an additional,
hashable input branch for experiments that explicitly opt in.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_FPS = 40.0
DEFAULT_SPATIAL_FPS = 10.0
DEFAULT_RELATION_FPS = 2.0
EPS = 1e-8

BANDS = (
    (50.0, 100.0),
    (100.0, 400.0),
    (400.0, 1600.0),
    (1600.0, 6400.0),
    (6400.0, 12800.0),
)

FEATURE_NAMES = (
    "left_log_rms", "right_log_rms", "mid_log_rms", "side_log_rms",
    "ild_db", "width_fraction", "pan_balance", "coherence",
    "phase_sin", "phase_cos", "left_centroid", "right_centroid",
    "left_rolloff", "right_rolloff",
    *[f"flux_left_band_{i}" for i in range(len(BANDS))],
    *[f"flux_right_band_{i}" for i in range(len(BANDS))],
    *[f"energy_delta_band_{i}" for i in range(len(BANDS))],
)

EVENT_NAMES = (
    "onset_left", "onset_right", "onset_mid", "onset_side",
    "flux_left", "flux_right", "transient_ratio", "event_strength",
)

SPATIAL_NAMES = (
    "ild_db", "width_fraction", "pan_balance", "coherence",
    "phase_sin", "phase_cos", "left_centroid", "right_centroid",
    "left_rolloff", "right_rolloff", "energy_delta_high_band",
)

RELATION_NAMES = (
    "coaccent", "lag_s", "lag_confidence", "subdivision_2",
    "subdivision_3", "quiet_fraction",
)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _as_stereo(waveform: np.ndarray) -> tuple[np.ndarray, bool]:
    value = np.asarray(waveform, dtype=np.float64)
    mono = False
    if value.ndim == 1:
        value = np.stack((value, value))
        mono = True
    elif value.ndim == 2 and value.shape[0] == 2:
        pass
    elif value.ndim == 2 and value.shape[0] == 1:
        value = np.concatenate((value, value), axis=0)
        mono = True
    elif value.ndim == 2 and value.shape[1] == 2:
        value = value.T
    else:
        raise ValueError("Expected waveform [2,N], [N,2], or mono [N]")
    if value.shape[1] < 8 or not np.isfinite(value).all():
        raise ValueError("Waveform must be finite and contain at least 8 samples")
    return np.ascontiguousarray(value), mono


def _frame_spectrum(waveform: np.ndarray, sample_rate: int, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample_rate <= 0 or fps <= 0:
        raise ValueError("sample_rate and fps must be positive")
    hop = max(1, int(round(sample_rate / fps)))
    n_fft = max(256, 2 * hop)
    frames = max(1, int(np.ceil(max(1, waveform.shape[1] - n_fft) / hop)) + 1)
    total = (frames - 1) * hop + n_fft
    pad = total - waveform.shape[1]
    if pad > 0:
        mode = "reflect" if waveform.shape[1] > 1 else "edge"
        waveform = np.pad(waveform, ((0, 0), (0, pad)), mode=mode)
    starts = np.arange(frames, dtype=np.int64) * hop
    indices = starts[:, None] + np.arange(n_fft, dtype=np.int64)[None, :]
    window = np.hanning(n_fft).astype(np.float64)
    framed = waveform[:, indices] * window[None, None, :]
    spectrum = np.fft.rfft(framed, axis=-1)
    times = (starts + n_fft / 2.0) / sample_rate
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    return spectrum, times, freqs


def _safe_log(value: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(value, EPS))


def _mean_pool(value: np.ndarray, times: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Pool a frame sequence into a fixed-rate causal-free analysis stream."""
    if len(value) == 0:
        raise ValueError("Cannot pool an empty sequence")
    duration = float(times[-1])
    count = max(1, int(np.floor(duration * fps)) + 1)
    edges = np.arange(count + 1, dtype=np.float64) / fps
    out, out_times = [], []
    for i in range(count):
        keep = (times >= edges[i]) & (times < edges[i + 1])
        if not keep.any():
            nearest = int(np.argmin(np.abs(times - (edges[i] + edges[i + 1]) / 2)))
            out.append(value[nearest])
        else:
            out.append(np.mean(value[keep], axis=0))
        out_times.append((edges[i] + edges[i + 1]) / 2)
    return np.asarray(out), np.asarray(out_times)


def resample_stream(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    """Interpolate a descriptor stream onto a declared downstream clock."""
    values = np.asarray(values, dtype=np.float64)
    source_time = np.asarray(source_time, dtype=np.float64)
    target_time = np.asarray(target_time, dtype=np.float64)
    if values.ndim != 2 or len(source_time) != len(values) or len(source_time) == 0:
        raise ValueError("Expected values[T,C] and matching nonempty source timestamps")
    if np.any(np.diff(source_time) <= 0) or not np.isfinite(values).all() or not np.isfinite(target_time).all():
        raise ValueError("Invalid descriptor timestamps or values")
    return np.stack([np.interp(target_time, source_time, values[:, i])
                     for i in range(values.shape[1])], axis=1).astype(np.float32)


def _event_relation(events: np.ndarray, fps: float, out_fps: float) -> np.ndarray:
    hop = max(1, int(round(fps / out_fps)))
    window = max(hop * 8, int(round(4.0 * fps)))
    max_lag = max(1, int(round(.5 * fps)))
    rows = []
    for end in range(0, len(events), hop):
        lo = max(0, end - window + 1)
        segment = events[lo:end + 1]
        if len(segment) < 4:
            rows.append(np.zeros(len(RELATION_NAMES), dtype=np.float64))
            continue
        left, right = segment[:, 0], segment[:, 1]
        left -= left.mean(); right -= right.mean()
        denom = np.linalg.norm(left) * np.linalg.norm(right)
        coaccent = float(np.dot(left, right) / denom) if denom > EPS else 0.0
        correlations, lags = [], []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                a, b = left[:lag], right[-lag:]
            elif lag > 0:
                a, b = left[lag:], right[:-lag]
            else:
                a, b = left, right
            d = np.linalg.norm(a) * np.linalg.norm(b)
            correlations.append(float(np.dot(a, b) / d) if d > EPS else 0.0)
            lags.append(lag)
        best = int(np.argmax(correlations))
        confidence = float(max(0.0, correlations[best] - np.median(correlations)))
        # Energy at the two common subdivisions is an interpretable relation
        # descriptor, not a tempo or a generated motion clock.
        env = np.maximum(segment.mean(1), 0.0)
        subdivision_2 = float(np.mean(env[::2]) - np.mean(env[1::2])) if len(env) > 2 else 0.0
        subdivision_3 = float(np.mean(env[::3]) - np.mean(env[1::3])) if len(env) > 3 else 0.0
        quiet = float(np.mean(env <= np.quantile(env, .25)))
        rows.append(np.array((coaccent, lags[best] / fps, confidence,
                              subdivision_2, subdivision_3, quiet), dtype=np.float64))
    return np.asarray(rows)


@dataclass
class StereoAudioV1:
    """Finite, timestamped stereo descriptors and derived semantic streams."""

    time_40: np.ndarray
    features_40: np.ndarray
    events_40: np.ndarray
    time_10: np.ndarray
    spatial_10: np.ndarray
    time_2: np.ndarray
    relation_2: np.ndarray
    metadata: dict

    def validate(self) -> "StereoAudioV1":
        arrays = (self.time_40, self.features_40, self.events_40,
                  self.time_10, self.spatial_10, self.time_2, self.relation_2)
        if any(not np.isfinite(np.asarray(x)).all() for x in arrays):
            raise ValueError("StereoAudioV1 contains non-finite values")
        if self.features_40.ndim != 2 or self.features_40.shape[1] != len(FEATURE_NAMES):
            raise ValueError("Invalid 40Hz stereo feature shape")
        if self.events_40.shape != (len(self.time_40), len(EVENT_NAMES)):
            raise ValueError("Invalid event stream shape")
        if self.spatial_10.shape != (len(self.time_10), len(SPATIAL_NAMES)):
            raise ValueError("Invalid spatial stream shape")
        if self.relation_2.shape != (len(self.time_2), len(RELATION_NAMES)):
            raise ValueError("Invalid relation stream shape")
        for time in (self.time_40, self.time_10, self.time_2):
            if len(time) == 0 or np.any(np.diff(time) <= 0):
                raise ValueError("Stereo timestamps must be strictly increasing")
        if self.metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported StereoAudioV1 schema")
        return self

    @property
    def feature_dim(self) -> int:
        return int(self.features_40.shape[1])

    @property
    def event_dim(self) -> int:
        return int(self.events_40.shape[1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(f"Preserving existing stereo artifact: {path}")
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, time_40=self.time_40.astype(np.float64),
                                features_40=self.features_40.astype(np.float32),
                                events_40=self.events_40.astype(np.float32),
                                time_10=self.time_10.astype(np.float64),
                                spatial_10=self.spatial_10.astype(np.float32),
                                time_2=self.time_2.astype(np.float64),
                                relation_2=self.relation_2.astype(np.float32))
        meta_path = path.with_suffix(".json")
        meta = {**self.metadata, "feature_names": list(FEATURE_NAMES),
                "event_names": list(EVENT_NAMES), "spatial_names": list(SPATIAL_NAMES),
                "relation_names": list(RELATION_NAMES), "npz_sha256": sha256(path)}
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        path.with_name(path.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(path)}) + "\n")
        meta_path.with_name(meta_path.name + ".sha256.json").write_text(json.dumps({"sha256": sha256(meta_path)}) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "StereoAudioV1":
        path = Path(path)
        meta_path = path.with_suffix(".json")
        sidecar = path.with_name(path.name + ".sha256.json")
        if sidecar.exists() and json.loads(sidecar.read_text())["sha256"] != sha256(path):
            raise ValueError("Stereo NPZ hash mismatch")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing stereo metadata: {meta_path}")
        meta_sidecar = meta_path.with_name(meta_path.name + ".sha256.json")
        if meta_sidecar.exists() and json.loads(meta_sidecar.read_text())["sha256"] != sha256(meta_path):
            raise ValueError("Stereo metadata hash mismatch")
        meta = json.loads(meta_path.read_text())
        if meta.get("npz_sha256") != sha256(path):
            raise ValueError("Stereo metadata does not bind NPZ")
        with np.load(path, allow_pickle=False) as arrays:
            result = cls(*(arrays[key].copy() for key in
                           ("time_40", "features_40", "events_40", "time_10",
                            "spatial_10", "time_2", "relation_2")), meta)
        return result.validate()


def extract_stereo_v1(waveform: np.ndarray, sample_rate: int, *, fps: float = DEFAULT_FPS,
                      spatial_fps: float = DEFAULT_SPATIAL_FPS,
                      relation_fps: float = DEFAULT_RELATION_FPS) -> StereoAudioV1:
    """Extract deterministic L/R, mid/side, event, spatial and relation streams."""
    wave, mono_input = _as_stereo(waveform)
    spectrum, times, freqs = _frame_spectrum(wave, int(sample_rate), float(fps))
    power = np.abs(spectrum) ** 2
    left, right = spectrum[0], spectrum[1]
    left_power, right_power = power[0], power[1]
    mid = .5 * (left + right)
    side = .5 * (left - right)
    mid_power, side_power = np.abs(mid) ** 2, np.abs(side) ** 2
    left_energy = left_power.mean(1)
    right_energy = right_power.mean(1)
    mid_energy = mid_power.mean(1)
    side_energy = side_power.mean(1)
    total_energy = left_energy + right_energy
    active = total_energy > EPS
    cross = left * np.conj(right)
    weights = left_power + right_power
    coherence = np.abs(np.sum(cross * weights, axis=1)) / np.maximum(
        np.sqrt(np.sum(left_power * weights, axis=1) * np.sum(right_power * weights, axis=1)), EPS)
    phase = np.angle(np.sum(cross * weights, axis=1))
    centroid_l = np.sum(left_power * freqs[None, :], axis=1) / np.maximum(left_power.sum(1), EPS)
    centroid_r = np.sum(right_power * freqs[None, :], axis=1) / np.maximum(right_power.sum(1), EPS)
    cdf_l = np.cumsum(left_power, axis=1) / np.maximum(left_power.sum(1, keepdims=True), EPS)
    cdf_r = np.cumsum(right_power, axis=1) / np.maximum(right_power.sum(1, keepdims=True), EPS)
    roll_l = freqs[np.argmax(cdf_l >= .95, axis=1)]
    roll_r = freqs[np.argmax(cdf_r >= .95, axis=1)]
    ild = 10.0 * np.log10(np.maximum(right_energy, EPS) / np.maximum(left_energy, EPS))
    width = side_energy / np.maximum(mid_energy + side_energy, EPS)
    pan = (right_energy - left_energy) / np.maximum(total_energy, EPS)

    flux_l = np.maximum(np.diff(_safe_log(left_power + 1e-6), axis=0, prepend=(_safe_log(left_power + 1e-6)[:1])), 0.0)
    flux_r = np.maximum(np.diff(_safe_log(right_power + 1e-6), axis=0, prepend=(_safe_log(right_power + 1e-6)[:1])), 0.0)
    feature_rows = [
        _safe_log(np.sqrt(left_energy)), _safe_log(np.sqrt(right_energy)),
        _safe_log(np.sqrt(mid_energy)), _safe_log(np.sqrt(side_energy)),
        ild, width, pan, np.clip(coherence, 0., 1.), np.sin(phase), np.cos(phase),
        centroid_l / max(float(sample_rate), 1.), centroid_r / max(float(sample_rate), 1.),
        roll_l / max(float(sample_rate), 1.), roll_r / max(float(sample_rate), 1.),
    ]
    for lo, hi in BANDS:
        keep = (freqs >= lo) & (freqs < hi)
        feature_rows.append(flux_l[:, keep].mean(1) if keep.any() else np.zeros(len(times)))
    for lo, hi in BANDS:
        keep = (freqs >= lo) & (freqs < hi)
        feature_rows.append(flux_r[:, keep].mean(1) if keep.any() else np.zeros(len(times)))
    for lo, hi in BANDS:
        keep = (freqs >= lo) & (freqs < hi)
        if keep.any():
            le, re = left_power[:, keep].mean(1), right_power[:, keep].mean(1)
            feature_rows.append((re - le) / np.maximum(re + le, EPS))
        else:
            feature_rows.append(np.zeros(len(times)))
    features = np.stack(feature_rows, axis=1)
    onset_left = np.maximum(np.diff(_safe_log(left_energy), prepend=_safe_log(left_energy)[:1]), 0.)
    onset_right = np.maximum(np.diff(_safe_log(right_energy), prepend=_safe_log(right_energy)[:1]), 0.)
    onset_mid = np.maximum(np.diff(_safe_log(mid_energy), prepend=_safe_log(mid_energy)[:1]), 0.)
    onset_side = np.maximum(np.diff(_safe_log(side_energy), prepend=_safe_log(side_energy)[:1]), 0.)
    event_strength = np.sqrt(np.maximum(onset_left * onset_right, 0.))
    transient = np.maximum(flux_l.mean(1) + flux_r.mean(1), 0.)
    event_rows = [onset_left, onset_right, onset_mid, onset_side,
                  flux_l.mean(1), flux_r.mean(1),
                  transient / np.maximum(transient + total_energy, EPS), event_strength]
    events = np.stack(event_rows, axis=1)
    spatial_source = np.stack((ild, width, pan, np.clip(coherence, 0., 1.),
                               np.sin(phase), np.cos(phase), centroid_l / max(float(sample_rate), 1.),
                               centroid_r / max(float(sample_rate), 1.), roll_l / max(float(sample_rate), 1.),
                               roll_r / max(float(sample_rate), 1.), feature_rows[-1]), axis=1)
    spatial, time_10 = _mean_pool(spatial_source, times, float(spatial_fps))
    relation = _event_relation(events[:, [0, 1, 7]], float(fps), float(relation_fps))
    time_2 = np.arange(len(relation), dtype=np.float64) / float(relation_fps) + .5 / float(relation_fps)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "fps_40": float(fps), "fps_spatial": float(spatial_fps), "fps_relation": float(relation_fps),
        "sample_rate_hz": int(sample_rate), "mono_input": bool(mono_input),
        "channel_contract": "original L/R retained; no independent channel normalization",
        "spatial_semantics": "stereo image descriptors; not physical azimuth/elevation/distance",
        "active_fraction": float(active.mean()), "feature_dim": len(FEATURE_NAMES),
        "event_dim": len(EVENT_NAMES), "spatial_dim": len(SPATIAL_NAMES),
        "relation_dim": len(RELATION_NAMES),
    }
    return StereoAudioV1(times, features.astype(np.float32), events.astype(np.float32),
                         time_10, spatial.astype(np.float32), time_2,
                         relation.astype(np.float32), metadata).validate()


def metamorphic_check(waveform: np.ndarray, sample_rate: int) -> dict:
    """Check L/R swap, known pan, width monotonicity, and finite output."""
    wave, _ = _as_stereo(waveform)
    original = extract_stereo_v1(wave, sample_rate)
    swapped = extract_stereo_v1(wave[::-1], sample_rate)
    pairs = {"ild_db": -1., "pan_balance": -1., "width_fraction": 1.,
             "coherence": 1., "phase_sin": -1., "phase_cos": 1.}
    errors = {}
    for name, sign in pairs.items():
        i = FEATURE_NAMES.index(name)
        errors[name] = float(np.max(np.abs(original.features_40[:, i] - sign * swapped.features_40[:, i])))
    return {"schema_version": SCHEMA_VERSION, "finite": True,
            "lr_swap_max_abs_error": errors,
            "passes_lr_swap": bool(max(errors.values(), default=0.) < 1e-5),
            "active_fraction": original.metadata["active_fraction"],
            "spatial_semantics": original.metadata["spatial_semantics"]}

"""Frozen CPU palette instruments, separate from categorical-TV diagnostics.

Math definition only; no cohort qualification, real-data result, human color
preference, rhythm accuracy, or quantization-floor claim is made by this module.
The caller must register source/config/calibration provenance before scientific
scoring and supply independent common reference observer tracks. No extractor,
mask predictor, token vocabulary, threshold, or model is fit here.

Joint HS uses targets.py's 12 circular hue centers h/12 and 6 saturation centers
s/5. Embed each of the 72 joint bins at (S*cos(2*pi*H), S*sin(2*pi*H)). Project on
36 fixed directions j*pi/36, j=0..35. For each projection compute exact weighted
1D W1 on the fixed bin support, divide by diameter 2, then average directions.
This sliced-W1 proxy is in [0,1], circular in hue, and retains joint HS coupling.
Hue at S=0 has deliberately zero geometric identity. No value/brightness enters.

Temporal components, each compared by unweighted empirical 1D W1:
1. log1p(observed maximal same-token residence seconds), including censored runs;
2. HS prototype distance at observed token-changing cue boundaries, or [0] with
   an explicit no-change flag when valid support exists but no change is observed;
3. ONE per-record mean adjacent-HS distance / .1 s, excluding cue-crossing steps;
4. per-pair HS distance at EXACTLY 8 s (80 steps), excluding any invalid support.
Every component is divided by its TRAIN-only, canonical-song-balanced IQR and
all FOUR scaled errors are averaged equally. Any undefined component makes the
aggregate undefined. Piecewise-held outputs may have zero inter-cue drift for
all arms; this component is always exposed, not hidden or silently removed.

Marginal error is HS sliced-W1 between duration-weighted mean valid HS tracks,
divided by the TRAIN IQR of per-canonical-song marginal distances to the TRAIN
mean. TRAIN record means are weighted equally within songs; songs equally.
TRAIN component quantiles give equal mass to songs, then recordings, then each
record's empirical samples (drift has one sample). Quantiles are inverse weighted
empirical CDFs: smallest x with CDF(x)>=q. IQR = Q(.75)-Q(.25); IQR<=0, missing
samples, or nonfinite scales FAIL, without an epsilon floor or DEV refitting.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np

from .targets import HS_DIM, HUE_BINS, SATURATION_BINS, SCHEMA_VERSION as TARGET_SCHEMA_VERSION

SCHEMA_VERSION = 1
MATH_ID = "joint_hs_sliced_w1_36_directions_temporal4_10Hz_8s_v1"
SAMPLE_DT_S = .1
LONG_LAG_STEPS = 80
COMPONENTS = ("log1p_residence_s", "conditional_mark_change", "mean_intercue_drift_per_s",
              "eight_second_change")
SCOPE = "authored_video_joint_HS_distribution_proxy_not_aesthetic_or_rhythm_accuracy"

_hue = np.repeat(np.arange(HUE_BINS) / HUE_BINS, SATURATION_BINS)
_sat = np.tile(np.arange(SATURATION_BINS) / (SATURATION_BINS - 1), HUE_BINS)
_embedding = np.column_stack((_sat * np.cos(2*np.pi*_hue), _sat * np.sin(2*np.pi*_hue)))
_angles = np.arange(36) * np.pi / 36
_support = _embedding @ np.stack((np.cos(_angles), np.sin(_angles)))
_ORDERS = np.argsort(_support, axis=0, kind="stable").T
_GAPS = np.diff(np.take_along_axis(_support.T, _ORDERS, axis=1), axis=1)


def _json_hash(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _distribution_rows(histograms, name: str) -> np.ndarray:
    rows = np.asarray(histograms, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != HS_DIM:
        raise ValueError(f"{name} must have shape (n, {HS_DIM})")
    if (not np.isfinite(rows).all() or np.any(rows < 0)
            or not np.allclose(rows.sum(axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError(f"{name} must contain normalized finite nonnegative joint-HS distributions")
    # Remove only validated float32 summation round-off, not arbitrary scaling.
    return rows / rows.sum(axis=1, keepdims=True)


def _distances(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    result = np.zeros(p.shape[0], dtype=float)
    delta = p - q
    for order, gaps in zip(_ORDERS, _GAPS):
        result += np.abs(np.cumsum(delta[:, order], axis=1)[:, :-1]) @ gaps
    return result / (36. * 2.)


def joint_hs_distance(p, q) -> float:
    """36-direction sliced-W1 / 2 on normalized 72-bin joint-HS histograms.

    For sorted fixed projected positions x_j, W1 = sum_j |sum_(i<=j)(p_i-q_i)|
    * (x_(j+1)-x_j). Equal-position bins contribute zero width. Exact same-support
    weighted transport is computed, never transport between hue/saturation
    marginals separately. Zero histograms are invalid, not achromatic colors.
    """
    p, q = np.asarray(p), np.asarray(q)
    if p.shape != (HS_DIM,) or q.shape != (HS_DIM,):
        raise ValueError(f"p and q must each have shape ({HS_DIM},)")
    return float(_distances(_distribution_rows(p[None], "p"), _distribution_rows(q[None], "q"))[0])


def palette_prototypes_hash(prototypes) -> str:
    """Content fingerprint of a supplied TRAIN-fitted normalized K x 72 vocabulary.

    Validates content, NOT the claim that it was fit only on TRAIN. The external
    frozen vocabulary manifest must establish that provenance independently.
    """
    rows = _distribution_rows(prototypes, "prototypes")
    if not len(rows):
        raise ValueError("prototypes must contain at least one palette")
    return _json_hash(dict(schema_version=TARGET_SCHEMA_VERSION, prototypes=rows.tolist()))


@dataclass(frozen=True)
class PaletteTrack:
    """One recording/draw track, with explicit common observer support.

    Samples at i/10 represent [i/10,(i+1)/10); duration is N/10, including the
    final sample's .1 s hold. Cues begin at 0, coincide with samples, and each has
    one generated/reference token ID. Between cues tokens hold, though observed
    HS need NOT equal their quantized prototype. Invalid rows are zero; validity
    is an EXTERNAL common mask, never a model-dependent abstention mechanism.
    """
    canonical_song: str
    recording: str
    times_s: np.ndarray
    hs_histograms: np.ndarray
    valid: np.ndarray
    cue_times_s: np.ndarray
    token_ids: np.ndarray
    prototypes_sha256: str
    schema_version: int = SCHEMA_VERSION
    target_schema_version: int = TARGET_SCHEMA_VERSION


@dataclass(frozen=True)
class PaletteSummary:
    component_samples: dict[str, np.ndarray]
    marginal_hs: np.ndarray | None
    observed_valid_time_fraction: float
    valid_duration_s: float
    duration_s: float
    no_observed_change: bool | None
    observed_change_count: int
    invalid_change_count: int
    censored_residence_segments: int
    censored_residence_edges: int
    intercue_pair_count: int
    eight_second_pair_count: int
    residence_scope: str = "observed_runs; invalid/clip_edge_censored_lengths_are_lower_bounds"


@dataclass(frozen=True)
class PaletteCalibration:
    schema_version: int
    math_id: str
    partition: str
    prototypes_sha256: str
    component_iqrs: tuple[float, ...]
    marginal_iqr: float
    training_mean_hs: tuple[float, ...]
    training_song_ids: tuple[str, ...]
    training_record_ids: tuple[tuple[str, str], ...]
    training_content_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class PaletteScore:
    temporal_raw: dict[str, float | None]
    temporal_scaled: dict[str, float | None]
    temporal_mean_scaled: float | None
    marginal_raw: float | None
    marginal_scaled: float | None
    undefined_components: tuple[str, ...]
    generated: PaletteSummary
    reference: PaletteSummary
    calibration_sha256: str
    prototypes_sha256: str
    valid_scope: str = SCOPE


def _validate_track(track: PaletteTrack, prototypes: np.ndarray, prototype_hash: str):
    if (type(track.schema_version) is not int or track.schema_version != SCHEMA_VERSION
            or type(track.target_schema_version) is not int
            or track.target_schema_version != TARGET_SCHEMA_VERSION):
        raise ValueError("unknown palette or joint-HS target schema")
    if any(not isinstance(value, str) or not value for value in (track.canonical_song, track.recording)):
        raise ValueError("canonical_song and recording must be explicit nonempty IDs")
    if track.prototypes_sha256 != prototype_hash:
        raise ValueError("track prototype hash differs from the shared frozen vocabulary")
    times = np.asarray(track.times_s, dtype=float)
    if (times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all()
            or not np.allclose(times, np.arange(len(times)) * SAMPLE_DT_S, atol=1e-10, rtol=0)):
        raise ValueError("timestamps must be gap-free regular 10 Hz starting at 0")
    valid = np.asarray(track.valid)
    if valid.dtype != np.bool_ or valid.shape != times.shape:
        raise ValueError("valid must be the externally supplied boolean per-sample mask")
    hist = np.asarray(track.hs_histograms, dtype=float)
    if hist.shape != (len(times), HS_DIM) or not np.isfinite(hist).all() or np.any(hist < 0):
        raise ValueError("HS track must have finite nonnegative shape (N, 72)")
    if np.any(hist[~valid] != 0):
        raise ValueError("invalid HS rows must be explicit zero placeholders")
    normalized = np.zeros_like(hist)
    normalized[valid] = _distribution_rows(hist[valid], "valid HS rows")
    cues = np.asarray(track.cue_times_s, dtype=float)
    if (cues.ndim != 1 or not len(cues) or not np.isfinite(cues).all() or cues[0] != 0
            or np.any(np.diff(cues) <= 0) or cues[-1] > times[-1] + 1e-10):
        raise ValueError("cues must start at 0 and strictly increase within sampled support")
    indices = np.rint(cues / SAMPLE_DT_S).astype(int)
    if (np.any(indices < 0) or np.any(indices >= len(times)) or np.any(np.diff(indices) <= 0)
            or not np.allclose(cues, indices * SAMPLE_DT_S, atol=1e-10, rtol=0)):
        raise ValueError("cue times must land on the same 10 Hz grid")
    tokens = np.asarray(track.token_ids)
    if (tokens.shape != cues.shape or not np.issubdtype(tokens.dtype, np.integer)
            or np.any(tokens < 0) or np.any(tokens >= len(prototypes))):
        raise ValueError("one integer vocabulary token is required per cue")
    return times, normalized, valid, indices, tokens.astype(np.int64)


def summarize_palette_track(track: PaletteTrack, prototypes) -> PaletteSummary:
    """Compute frozen raw summaries; no comparison or scale fitting.

    Adjacent/8 s pairs require ALL samples on their closed index interval valid.
    A cue-crossing step is i-1 -> i for a supplied cue at sample i>0. Mark-change
    prototype distances use differing consecutive cue IDs, admitted only when
    samples immediately before/at that cue are valid. This does NOT demand that
    every frame since the previous cue was valid: only the observed boundary is
    being measured. Invalid changes are counted separately. With any valid time
    but no observed changes the defined conditional sample is [0] plus a flag.
    All-invalid support is undefined, including the conditional sample.

    Residence is split at every invalid span and token change; clip/invalid edges
    are flagged as censored. No residence, adjacent pair, or lag pair crosses a
    gap. A too-short track has undefined 8 s statistics. No minimum valid-fraction
    threshold is invented; eligibility is for the external prespecified cohort.
    """
    prototype_hash = palette_prototypes_hash(prototypes)
    prototypes = _distribution_rows(prototypes, "prototypes")
    times, hist, valid, cues, tokens = _validate_track(track, prototypes, prototype_hash)
    n = len(times)
    active_tokens = tokens[np.searchsorted(cues, np.arange(n), side="right") - 1]
    residences, censored_segments, censored_edges = [], 0, 0
    index = 0
    while index < n:
        if not valid[index]:
            index += 1
            continue
        start = index
        index += 1
        while index < n and valid[index] and active_tokens[index] == active_tokens[start]:
            index += 1
        edges = int(start == 0 or not valid[start-1]) + int(index == n or not valid[index])
        censored_edges += edges
        censored_segments += int(edges > 0)
        residences.append((index-start) * SAMPLE_DT_S)
    changed = tokens[1:] != tokens[:-1]
    boundary_valid = valid[cues[1:]-1] & valid[cues[1:]]
    observed = changed & boundary_valid
    mark_distances = _distances(prototypes[tokens[:-1][observed]], prototypes[tokens[1:][observed]])
    no_change = not bool(len(mark_distances)) if valid.any() else None
    if no_change:
        mark_distances = np.array([0.])
    intercue = valid[:-1] & valid[1:]
    intercue[cues[1:]-1] = False
    intercue_distances = _distances(hist[:-1][intercue], hist[1:][intercue]) / SAMPLE_DT_S
    drift = np.array([intercue_distances.mean()]) if len(intercue_distances) else np.empty(0)
    starts = np.arange(max(0, n-LONG_LAG_STEPS))
    invalid_prefix = np.r_[0, np.cumsum(~valid)]
    keep_lag = invalid_prefix[starts+LONG_LAG_STEPS+1] == invalid_prefix[starts]
    starts = starts[keep_lag]
    lag = _distances(hist[starts], hist[starts+LONG_LAG_STEPS])
    samples = dict(zip(COMPONENTS, (np.log1p(residences), mark_distances, drift, lag)))
    return PaletteSummary(samples, hist[valid].mean(axis=0) if valid.any() else None,
                          float(valid.mean()), float(valid.sum()*SAMPLE_DT_S), n*SAMPLE_DT_S,
                          no_change, int(observed.sum()), int((changed & ~boundary_valid).sum()),
                          censored_segments, censored_edges, int(intercue.sum()), len(starts))


def _empirical_w1(a: np.ndarray, b: np.ndarray) -> float | None:
    if not len(a) or not len(b):
        return None
    a, b = np.sort(a), np.sort(b)
    points = np.sort(np.r_[a, b])
    delta = np.diff(points)
    cdf_a = np.searchsorted(a, points[:-1], side="right") / len(a)
    cdf_b = np.searchsorted(b, points[:-1], side="right") / len(b)
    return float(np.sum(np.abs(cdf_a-cdf_b)*delta))


def _weighted_iqr(values: np.ndarray, weights: Sequence[Fraction]) -> float:
    # Exact rational hierarchy weights prevent a .25 CDF boundary from shifting
    # when an identical recording is duplicated or a sample group is reordered.
    masses = defaultdict(Fraction)
    for value, weight in zip(values, weights):
        masses[float(value)] += weight
    total = sum(masses.values(), Fraction())
    cumulative, quantiles = Fraction(), []
    for value in sorted(masses):
        cumulative += masses[value]
        while len(quantiles) < 2 and cumulative >= total * (Fraction(1, 4), Fraction(3, 4))[len(quantiles)]:
            quantiles.append(value)
    return float(quantiles[1]-quantiles[0])


def _calibration_payload(calibration: PaletteCalibration) -> dict:
    payload = asdict(calibration)
    payload.pop("content_sha256")
    return payload


def _track_hash(track: PaletteTrack) -> str:
    # Hash supplied content, not just derived summaries, to preserve provenance.
    scalar = dict(canonical_song=track.canonical_song, recording=track.recording,
                  prototypes_sha256=track.prototypes_sha256, schema_version=track.schema_version,
                  target_schema_version=track.target_schema_version)
    digester = hashlib.sha256(json.dumps(scalar, sort_keys=True).encode())
    for name in ("times_s", "hs_histograms", "valid", "cue_times_s", "token_ids"):
        dtype = "?" if name == "valid" else "<i8" if name == "token_ids" else "<f8"
        array = np.ascontiguousarray(getattr(track, name), dtype=dtype)
        digester.update(name.encode())
        digester.update(str(array.shape).encode())
        digester.update(array.tobytes())
    return digester.hexdigest()


def fit_palette_scales(
    training_references: Sequence[PaletteTrack], prototypes, *, partition: str,
) -> PaletteCalibration:
    """Fit ONLY on explicit TRAIN references; return content-hashed calibration.

    Exactly one record per canonical_song/recording ID is allowed (no replicated
    draws/seeds masquerading as records). Each record must define all components.
    Missing/degenerate IQRs raise: do not discard that component, change quantile
    rules, add an epsilon, or refit on DEV. Hashes provide content integrity, not
    proof of split provenance; the external frozen manifest remains required.
    """
    if partition != "TRAIN":
        raise ValueError("scale fitting requires explicit partition='TRAIN'; no DEV/TEST fitting")
    records = tuple(training_references)
    if not records:
        raise ValueError("TRAIN records cannot be empty")
    prototype_hash = palette_prototypes_hash(prototypes)
    summaries, by_song = {}, defaultdict(list)
    hashes = {}
    for record in records:
        key = (record.canonical_song, record.recording)
        if key in summaries:
            raise ValueError("duplicate canonical-song/recording TRAIN record")
        summary = summarize_palette_track(record, prototypes)
        if any(not len(summary.component_samples[name]) for name in COMPONENTS):
            raise ValueError(f"undefined TRAIN component for {key}; qualification blocked")
        summaries[key] = summary
        by_song[record.canonical_song].append(key)
        hashes[key] = _track_hash(record)
    song_ids = tuple(sorted(by_song))
    record_ids = tuple(sorted(summaries))
    iqrs = []
    for name in COMPONENTS:
        values, weights = [], []
        for song in song_ids:
            for key in sorted(by_song[song]):
                samples = summaries[key].component_samples[name]
                unique, counts = np.unique(samples, return_counts=True)
                values.extend(unique)
                denominator = len(song_ids)*len(by_song[song])*len(samples)
                weights.extend(Fraction(int(count), denominator) for count in counts)
        scale = _weighted_iqr(np.asarray(values), weights)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"degenerate TRAIN IQR for {name}; qualification blocked")
        iqrs.append(scale)
    song_marginals = np.stack([np.mean([summaries[key].marginal_hs for key in sorted(by_song[song])], axis=0)
                               for song in song_ids])
    train_mean = song_marginals.mean(axis=0)
    distances = _distances(song_marginals, np.broadcast_to(train_mean, song_marginals.shape))
    marginal_iqr = _weighted_iqr(distances, [Fraction(1)]*len(distances))
    if not np.isfinite(marginal_iqr) or marginal_iqr <= 0:
        raise ValueError("degenerate TRAIN marginal IQR; qualification blocked")
    training_hash = _json_hash([[*key, hashes[key]] for key in record_ids])
    payload = dict(schema_version=SCHEMA_VERSION, math_id=MATH_ID, partition="TRAIN",
                   prototypes_sha256=prototype_hash, component_iqrs=tuple(iqrs),
                   marginal_iqr=marginal_iqr, training_mean_hs=tuple(map(float, train_mean)),
                   training_song_ids=song_ids, training_record_ids=record_ids,
                   training_content_sha256=training_hash)
    return PaletteCalibration(**payload, content_sha256=_json_hash(payload))


def _validate_calibration(calibration: PaletteCalibration, prototype_hash: str):
    if (type(calibration.schema_version) is not int or calibration.schema_version != SCHEMA_VERSION
            or calibration.math_id != MATH_ID or calibration.partition != "TRAIN"):
        raise ValueError("unknown calibration schema/math or non-TRAIN scales")
    if calibration.prototypes_sha256 != prototype_hash:
        raise ValueError("calibration uses a different prototype vocabulary")
    if _json_hash(_calibration_payload(calibration)) != calibration.content_sha256:
        raise ValueError("calibration content hash mismatch; scales or provenance were altered")
    scales = np.asarray(calibration.component_iqrs, dtype=float)
    if (scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0)
            or not np.isfinite(calibration.marginal_iqr) or calibration.marginal_iqr <= 0):
        raise ValueError("missing/degenerate scales; qualification blocked")
    _distribution_rows(np.asarray(calibration.training_mean_hs)[None], "TRAIN mean")


def compare_palette_tracks(
    generated: PaletteTrack, reference: PaletteTrack, prototypes, calibration: PaletteCalibration,
) -> PaletteScore:
    """Compare one generated draw with the common reference; never teacher-force.

    Canonical ID, recording ID, exact timestamps, cues, validity, and prototypes
    MUST match. Generated and reference cue token IDs may differ; their observer
    HS tracks need not equal prototype holds. Quantized-reference self/floor,
    independent-held, constant, donor, and diversity controls belong in separate
    calls/reports, not subtractions concealed in this score. Use metrics.py's
    paired canonical-song aggregation across recordings/draws/seeds afterward.

    Raw and TRAIN-scaled components are all returned. If ANY component is
    undefined, temporal_mean_scaled=None rather than the mean of surviving
    components. Marginal error is independently undefined on all-invalid input.
    This function does not decide the external 80% cohort or scientific gates.
    """
    prototype_hash = palette_prototypes_hash(prototypes)
    _validate_calibration(calibration, prototype_hash)
    a = summarize_palette_track(generated, prototypes)
    b = summarize_palette_track(reference, prototypes)
    if (generated.canonical_song != reference.canonical_song or generated.recording != reference.recording):
        raise ValueError("generated/reference canonical-song and recording IDs must match")
    for name in ("times_s", "cue_times_s", "valid"):
        if not np.array_equal(getattr(generated, name), getattr(reference, name)):
            raise ValueError(f"generated/reference must use identical {name}")
    raw = {name: _empirical_w1(a.component_samples[name], b.component_samples[name]) for name in COMPONENTS}
    scaled = {name: (raw[name] / scale if raw[name] is not None else None)
              for name, scale in zip(COMPONENTS, calibration.component_iqrs)}
    undefined = tuple(name for name in COMPONENTS if raw[name] is None)
    aggregate = float(np.mean(list(scaled.values()))) if not undefined else None
    marginal = (joint_hs_distance(a.marginal_hs, b.marginal_hs)
                if a.marginal_hs is not None and b.marginal_hs is not None else None)
    return PaletteScore(raw, scaled, aggregate, marginal,
                        marginal/calibration.marginal_iqr if marginal is not None else None,
                        undefined, a, b, calibration.content_sha256, prototype_hash)

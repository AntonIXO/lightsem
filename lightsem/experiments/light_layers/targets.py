"""CPU-only chromatic targets for the new lighting-layer campaign.

This module reads no datasets and fits no parameters. Inputs are RGB, never BGR.
The new 96-vector is NOT the historical six-vector: the latter can only be
attached as an already-computed auxiliary reference. No brightness statistic
enters the chromatic histogram, novelty distance, or event threshold.

Histogram schema: 12 circular hue centers ``k/12`` by 6 saturation centers
``k/5``, flattened hue-major. Bilinear deposition avoids a discontinuity at
red (hue 0/1). Four quadrants follow in row-major image order: top-left,
top-right, bottom-left, bottom-right. Color statistics use only retained pixels
with S >= .12 and V >= .08. Value summaries instead use ALL retained pixels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = 1
HUE_BINS = 12
SATURATION_BINS = 6
HS_DIM = HUE_BINS * SATURATION_BINS
QUADRANT_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
QUADRANT_FEATURE_NAMES = (
    "hue_cos", "hue_sin", "mean_saturation", "hue_concentration",
    "chromatic_fraction", "retained_mask_fraction",
)
RICH_DIM = HS_DIM + len(QUADRANT_NAMES) * len(QUADRANT_FEATURE_NAMES)
MIN_SATURATION = .12
MIN_VALUE = .08
NOVELTY_WINDOW_S = .3
MIN_EVENT_DISTANCE_S = .5
MAX_SAMPLE_GAP_S = .2


def rich_feature_names() -> tuple[str, ...]:
    """Stable, unique names in exact 96-vector order."""
    return tuple(f"hs_h{h:02d}_s{s:02d}" for h in range(HUE_BINS)
                 for s in range(SATURATION_BINS)) + tuple(
        f"{quadrant}_{feature}" for quadrant in QUADRANT_NAMES
        for feature in QUADRANT_FEATURE_NAMES)


@dataclass(frozen=True)
class ValueSummary:
    """HSV V, not luma; independent accent inputs over all retained pixels.

    Empty-mask summaries are zero placeholders with ``valid=False``. Black
    retained pixels ARE valid value observations even though color is invalid.
    """
    valid: bool
    mean: float
    std: float
    p90: float
    maximum: float


@dataclass(frozen=True)
class FrameColorTargets:
    rich: np.ndarray
    hs_histogram: np.ndarray
    quadrants: np.ndarray
    color_valid: bool
    quadrant_color_valid: np.ndarray
    retained_pixels: int
    valid_pixels: int
    retained_fraction: float  # retained / full image
    valid_fraction: float  # chromatic-valid / full image
    chromatic_fraction: float  # chromatic-valid / retained
    value: ValueSummary
    legacy_v6: np.ndarray | None


@dataclass(frozen=True)
class ChromaticNovelty:
    """Finite scores plus explicit eligibility; ineligible scores are zero.

    ``times_s`` are the original input sample times, not an invented frame grid.
    Windows use zero-order hold only between actual samples, without endpoint
    extrapolation. This is a sampling convention, not native temporal precision.
    Mask activity is the changed-mask-pixel fraction at each sample boundary.
    If it was not supplied, zero placeholders have ``mask_activity_known=False``.
    Activity is diagnostic, NOT an automatic veto or proof of robustness.
    """
    times_s: np.ndarray
    scores: np.ndarray
    eligible: np.ndarray
    window_eligible: np.ndarray
    color_eligible: np.ndarray
    mask_eligible: np.ndarray
    cut_eligible: np.ndarray
    gap_eligible: np.ndarray
    pre_histograms: np.ndarray
    post_histograms: np.ndarray
    retained_fraction: np.ndarray
    chromatic_fraction: np.ndarray
    window_min_retained_fraction: np.ndarray
    window_min_chromatic_fraction: np.ndarray
    window_retained_fraction_range: np.ndarray
    mask_activity: np.ndarray
    mask_activity_known: bool
    window_max_mask_activity: np.ndarray
    window_s: float
    max_gap_s: float


@dataclass(frozen=True)
class ChromaticEvents:
    times_s: np.ndarray
    scores: np.ndarray
    eligible: np.ndarray
    events: np.ndarray
    event_indices: np.ndarray
    event_times_s: np.ndarray
    event_scores: np.ndarray
    threshold: float
    min_distance_s: float


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum, minimum = rgb.max(axis=-1), rgb.min(axis=-1)
    delta = maximum - minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    hue = np.zeros_like(delta)
    channel = rgb.argmax(axis=-1)
    nonzero = delta > 0
    for index, numerator, offset in (
        (0, rgb[..., 1] - rgb[..., 2], 0.),
        (1, rgb[..., 2] - rgb[..., 0], 2.),
        (2, rgb[..., 0] - rgb[..., 1], 4.),
    ):
        use = nonzero & (channel == index)
        hue[use] = numerator[use] / delta[use] + offset
    return np.mod(hue / 6., 1.), saturation, maximum


def _hs_histogram(hue: np.ndarray, saturation: np.ndarray) -> np.ndarray:
    histogram = np.zeros((HUE_BINS, SATURATION_BINS), dtype=np.float64)
    if not hue.size:
        return histogram.ravel().astype(np.float32)
    hposition = np.mod(hue, 1.) * HUE_BINS
    hfloor = np.floor(hposition).astype(np.int64)
    hweight = hposition - hfloor
    sposition = np.clip(saturation, 0., 1.) * (SATURATION_BINS - 1)
    sfloor = np.floor(sposition).astype(np.int64)
    sweight = sposition - sfloor
    for hi, hw in ((hfloor % HUE_BINS, 1. - hweight),
                   ((hfloor + 1) % HUE_BINS, hweight)):
        for si, sw in ((sfloor, 1. - sweight),
                       (np.minimum(sfloor + 1, SATURATION_BINS - 1), sweight)):
            np.add.at(histogram, (hi, si), hw * sw)
    return (histogram / hue.size).ravel().astype(np.float32)


def extract_color_targets(
    rgb: np.ndarray, retained_mask: np.ndarray, *, legacy_v6: np.ndarray | None = None,
) -> FrameColorTargets:
    """Extract one RGB frame; a True mask pixel is retained background.

    Float RGB must be finite and in [0, 1]; uint8 RGB is divided by 255.
    Other integer types and non-boolean masks are rejected rather than guessed.
    Empty masks are allowed. A color-invalid region has zero color components,
    explicit validity False, and still reports its retained-mask coverage.
    Odd image dimensions place the extra row/column in bottom/right quadrants.
    ``legacy_v6`` is optional finite shape-(6,) data copied without modification;
    this function never reconstructs or normalizes old training targets.
    """
    image = np.asarray(rgb)
    mask = np.asarray(retained_mask)
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("rgb must have nonempty shape (height, width, 3)")
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.
    elif np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float64)
        if not np.isfinite(image).all() or np.any((image < 0) | (image > 1)):
            raise ValueError("float rgb must be finite and in [0, 1]")
    else:
        raise ValueError("rgb must be uint8 or floating point")
    if mask.dtype != np.bool_ or mask.shape != image.shape[:2]:
        raise ValueError("retained_mask must be boolean with the frame's spatial shape")
    auxiliary = None
    if legacy_v6 is not None:
        auxiliary = np.asarray(legacy_v6, dtype=np.float64)
        if auxiliary.shape != (6,) or not np.isfinite(auxiliary).all():
            raise ValueError("legacy_v6 must be a finite shape-(6,) auxiliary reference")
        auxiliary = auxiliary.copy()
    hue, saturation, value = _rgb_to_hsv(image)
    valid = mask & (saturation >= MIN_SATURATION) & (value >= MIN_VALUE)
    retained_pixels, valid_pixels = int(mask.sum()), int(valid.sum())
    histogram = _hs_histogram(hue[valid], saturation[valid])
    height, width = mask.shape
    quadrants = np.zeros((4, 6), dtype=np.float32)
    quadrant_valid = np.zeros(4, dtype=bool)
    regions = ((slice(0, height // 2), slice(0, width // 2)),
               (slice(0, height // 2), slice(width // 2, width)),
               (slice(height // 2, height), slice(0, width // 2)),
               (slice(height // 2, height), slice(width // 2, width)))
    for index, region in enumerate(regions):
        local_mask, local_valid = mask[region], valid[region]
        retained, count = int(local_mask.sum()), int(local_valid.sum())
        quadrants[index, 4] = count / retained if retained else 0.
        quadrants[index, 5] = retained / local_mask.size if local_mask.size else 0.
        if count:
            angles = hue[region][local_valid] * (2 * np.pi)
            cosine, sine = float(np.cos(angles).mean()), float(np.sin(angles).mean())
            quadrants[index, :4] = (
                cosine, sine, float(saturation[region][local_valid].mean()),
                min(1., float(np.hypot(cosine, sine))))
            quadrant_valid[index] = True
    values = value[mask]
    summary = (ValueSummary(True, float(values.mean()), float(values.std()),
                            float(np.quantile(values, .9)), float(values.max()))
               if retained_pixels else ValueSummary(False, 0., 0., 0., 0.))
    return FrameColorTargets(
        rich=np.concatenate((histogram, quadrants.ravel())),
        hs_histogram=histogram, quadrants=quadrants, color_valid=valid_pixels > 0,
        quadrant_color_valid=quadrant_valid, retained_pixels=retained_pixels,
        valid_pixels=valid_pixels, retained_fraction=retained_pixels / mask.size,
        valid_fraction=valid_pixels / mask.size,
        chromatic_fraction=valid_pixels / retained_pixels if retained_pixels else 0.,
        value=summary, legacy_v6=auxiliary)


def retained_mask_activity(retained_masks: np.ndarray) -> np.ndarray:
    """Changed-pixel fraction between adjacent masks, even at equal coverage.

    Returns length T with first entry zero (no preceding observation). This
    exposes mask-induced color novelty on a frozen multicolor background.
    """
    masks = np.asarray(retained_masks)
    if masks.dtype != np.bool_ or masks.ndim != 3 or min(masks.shape[1:]) < 1:
        raise ValueError("retained_masks must be boolean (time, height, width)")
    result = np.zeros(len(masks), dtype=np.float64)
    if len(masks) > 1:
        result[1:] = np.mean(masks[1:] != masks[:-1], axis=(1, 2))
    return result


def _times(times_s: np.ndarray) -> np.ndarray:
    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError("times_s must be a finite one-dimensional array")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times_s must be strictly increasing, without duplicate samples")
    return times


def _fraction(values: np.ndarray, n: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (n,) or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must be finite shape-(time,) fractions in [0, 1]")
    return array


def _boolean(values: np.ndarray, n: int, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.bool_ or array.shape != (n,):
        raise ValueError(f"{name} must be a boolean shape-(time,) array")
    return array


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Square-root Jensen-Shannon distance, base 2, range [0, 1].

    Inputs must be nonempty normalized distributions. All-zero invalid color
    histograms are not distributions and must be gated before this function.
    """
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if p.ndim != 1 or not p.size or p.shape != q.shape:
        raise ValueError("distributions must have the same nonempty one-dimensional shape")
    for distribution in (p, q):
        if (not np.isfinite(distribution).all() or np.any(distribution < 0)
                or not np.isclose(distribution.sum(), 1., atol=1e-6, rtol=0)):
            raise ValueError("distributions must be finite, nonnegative, and normalized")
    # Remove float32 round-off only, never normalize unvalidated inputs.
    p, q = p / p.sum(), q / q.sum()
    mixture = .5 * (p + q)
    positive_p, positive_q = p > 0, q > 0
    divergence = .5 * (np.sum(p[positive_p] * np.log2(p[positive_p] / mixture[positive_p]))
                       + np.sum(q[positive_q] * np.log2(q[positive_q] / mixture[positive_q])))
    return float(np.sqrt(np.clip(divergence, 0., 1.)))


def chromatic_novelty(
    times_s: np.ndarray, hs_histograms: np.ndarray, color_valid: np.ndarray, *,
    retained_fraction: np.ndarray, chromatic_fraction: np.ndarray,
    cuts: np.ndarray | None = None, mask_activity: np.ndarray | None = None,
    window_s: float = NOVELTY_WINDOW_S, max_gap_s: float = MAX_SAMPLE_GAP_S,
    min_retained_fraction: float = 0., min_chromatic_fraction: float = 0.,
) -> ChromaticNovelty:
    """Compare adjacent pre/post .3 s windows using actual timestamps.

    A sample is held to the next input timestamp. At candidate time t, compare
    [t-window_s, t) with [t, t+window_s). Both windows must have full observed
    support; no extrapolation, interpolation to 40 Hz, or frame-rate assumption
    is made. A gap > max_gap_s, ANY invalid color observation, a mask-coverage
    failure, or a cut strictly inside their combined support makes t ineligible.
    ``cuts[i]`` marks a cut BEFORE sample i; cuts[0] has no predecessor.

    Each valid histogram must sum to one; invalid rows must be all zero. All
    coverage inputs are mandatory. Minimum coverage defaults require positive
    coverage, without inventing a data-fitted acceptance threshold. Optional
    stricter coverage thresholds must be fixed by the caller. A changing mask
    can still produce a high score: the returned activity and coverage are
    nuisance diagnostics, not evidence that this failure mode was solved.
    """
    times = _times(times_s)
    n = len(times)
    histograms = np.asarray(hs_histograms, dtype=np.float64)
    valid = _boolean(color_valid, n, "color_valid")
    retained = _fraction(retained_fraction, n, "retained_fraction")
    chromatic = _fraction(chromatic_fraction, n, "chromatic_fraction")
    cut = np.zeros(n, dtype=bool) if cuts is None else _boolean(cuts, n, "cuts")
    activity = (np.zeros(n, dtype=np.float64) if mask_activity is None else
                _fraction(mask_activity, n, "mask_activity"))
    if histograms.shape != (n, HS_DIM) or not np.isfinite(histograms).all() or np.any(histograms < 0):
        raise ValueError(f"hs_histograms must be finite nonnegative shape-(time, {HS_DIM})")
    if not np.allclose(histograms[valid].sum(axis=1), 1., atol=1e-6, rtol=0):
        raise ValueError("color-valid histograms must be normalized to one")
    if np.any(histograms[~valid] != 0):
        raise ValueError("color-invalid histograms must be zero placeholders")
    if np.any(valid & ((retained <= 0) | (chromatic <= 0))) or np.any((~valid) & (chromatic != 0)):
        raise ValueError("color_valid and chromatic/retained coverage disagree")
    if not np.isfinite(window_s) or window_s <= 0 or not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("window_s and max_gap_s must be finite and positive")
    for name, value in (("min_retained_fraction", min_retained_fraction),
                        ("min_chromatic_fraction", min_chromatic_fraction)):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    scores = np.zeros(n, dtype=np.float64)
    window_ok, color_ok, mask_ok, cut_ok, gap_ok = [np.zeros(n, dtype=bool) for _ in range(5)]
    pre, post = np.zeros((n, HS_DIM)), np.zeros((n, HS_DIM))
    min_retained, min_chromatic, coverage_range, max_activity = [np.zeros(n) for _ in range(4)]
    sample_mask_ok = ((retained > 0) & (chromatic > 0)
                      & (retained >= min_retained_fraction)
                      & (chromatic >= min_chromatic_fraction))
    # Tolerance only handles floating-point boundary arithmetic, not missing data.
    epsilon = max(1e-10, abs(float(window_s)) * 1e-8)
    for i, center in enumerate(times):
        left, right = center - window_s, center + window_s
        if left < times[0] - epsilon or right > times[-1] + epsilon:
            continue
        left, right = max(left, times[0]), min(right, times[-1])
        first = max(0, int(np.searchsorted(times, left, side="right")) - 1)
        stop = min(n - 1, int(np.searchsorted(times, right, side="left")))
        indices = np.arange(first, stop)
        starts, ends = times[indices], times[indices + 1]
        pre_weights = np.maximum(0., np.minimum(ends, center) - np.maximum(starts, left))
        post_weights = np.maximum(0., np.minimum(ends, right) - np.maximum(starts, center))
        used = pre_weights + post_weights > epsilon
        if not used.any():
            continue
        window_ok[i] = (abs(pre_weights.sum() - window_s) <= epsilon
                        and abs(post_weights.sum() - window_s) <= epsilon)
        active = indices[used]
        color_ok[i] = bool(valid[active].all())
        mask_ok[i] = bool(sample_mask_ok[active].all())
        gap_ok[i] = bool(np.all(ends[used] - starts[used] <= max_gap_s + epsilon))
        # Only boundaries interior to the support can connect two observations.
        boundaries = np.arange(first + 1, stop + 1)
        boundaries = boundaries[(times[boundaries] > left + epsilon)
                                & (times[boundaries] < right - epsilon)]
        cut_ok[i] = not bool(cut[boundaries].any())
        min_retained[i] = retained[active].min()
        min_chromatic[i] = chromatic[active].min()
        coverage_range[i] = np.ptp(retained[active])
        max_activity[i] = float(activity[boundaries].max()) if len(boundaries) else 0.
        if window_ok[i] and color_ok[i] and mask_ok[i] and cut_ok[i] and gap_ok[i]:
            pre[i] = (pre_weights @ histograms[indices]) / pre_weights.sum()
            post[i] = (post_weights @ histograms[indices]) / post_weights.sum()
            scores[i] = jensen_shannon_distance(pre[i], post[i])
    eligible = window_ok & color_ok & mask_ok & cut_ok & gap_ok
    return ChromaticNovelty(
        times_s=times.copy(), scores=scores, eligible=eligible,
        window_eligible=window_ok, color_eligible=color_ok, mask_eligible=mask_ok,
        cut_eligible=cut_ok, gap_eligible=gap_ok, pre_histograms=pre,
        post_histograms=post, retained_fraction=retained.copy(),
        chromatic_fraction=chromatic.copy(), window_min_retained_fraction=min_retained,
        window_min_chromatic_fraction=min_chromatic, window_retained_fraction_range=coverage_range,
        mask_activity=activity.copy(), mask_activity_known=mask_activity is not None,
        window_max_mask_activity=max_activity, window_s=float(window_s), max_gap_s=float(max_gap_s))


def chromatic_event_targets(
    novelty: ChromaticNovelty, *, threshold: float,
    min_distance_s: float = MIN_EVENT_DISTANCE_S,
) -> ChromaticEvents:
    """Local peaks strictly above an EXTERNALLY FROZEN threshold, then time NMS.

    No quantile, normalization, or threshold fitting occurs here. The caller
    must freeze threshold selection on TRAIN only and retain its provenance.
    Tied plateaus produce one candidate at their middle input sample (earlier
    middle for an even count). Time NMS keeps the strongest peak; equal-strength
    ties keep the earlier one. Output event times are exact input timestamps.
    At threshold zero a constant zero score still emits no events.
    """
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite externally frozen value in [0, 1]")
    if not np.isfinite(min_distance_s) or min_distance_s < 0:
        raise ValueError("min_distance_s must be finite and nonnegative")
    times = _times(novelty.times_s)
    n = len(times)
    scores = _fraction(novelty.scores, n, "scores")
    eligible = _boolean(novelty.eligible, n, "eligible")
    effective = np.where(eligible, scores, -np.inf)
    candidates: list[int] = []
    i = 0
    while i < n:
        if not eligible[i] or scores[i] <= threshold:
            i += 1
            continue
        end = i + 1
        while end < n and eligible[end] and abs(scores[end] - scores[i]) <= 1e-12:
            end += 1
        before = effective[i - 1] if i else -np.inf
        after = effective[end] if end < n else -np.inf
        if scores[i] > before and scores[i] > after:
            candidates.append((i + end - 1) // 2)
        i = end
    kept: list[int] = []
    for index in sorted(candidates, key=lambda k: (-scores[k], times[k])):
        if all(abs(times[index] - times[other]) >= min_distance_s - 1e-10 for other in kept):
            kept.append(index)
    indices = np.array(sorted(kept), dtype=np.int64)
    events = np.zeros(n, dtype=bool)
    events[indices] = True
    return ChromaticEvents(
        times_s=times.copy(), scores=scores.copy(), eligible=eligible.copy(), events=events,
        event_indices=indices, event_times_s=times[indices].copy(), event_scores=scores[indices].copy(),
        threshold=float(threshold), min_distance_s=float(min_distance_s))

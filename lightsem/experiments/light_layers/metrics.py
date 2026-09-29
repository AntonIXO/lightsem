"""CPU-only common instruments. No data access, fitting, or event extraction.

Visual-event tolerance is fixed at +/-0.2 s. These are authored-video proxies,
NOT the independent-annotation 70 ms rhythm instrument. Callers supply frozen
reference events and explicit candidate times/confidences, after their declared
local-maximum extraction and >=0.2 s suppression. This module neither changes
those definitions nor runs NMS, threshold fitting, masking, or segmentation.

Palette metrics below are normalized CATEGORICAL proxies on a shared vocabulary
and cue inventory. They are NOT the preregistered TRAIN-scaled joint-HS marginal
or residence/distance/drift/long-lag Wasserstein composite. That composite needs
separately frozen definitions/scales. Neither novelty scores nor these proxies
establish color quality, human preference, or mask accuracy.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

VISUAL_TOLERANCE_S = 0.2
BOOTSTRAP_REPLICATES = 10_000
FIXED_SHIFT_COUNT = 20
# Arithmetic slack only, not additional label precision or a fitted tolerance.
_BOUNDARY_EPS_S = 1e-12


def _vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


def _positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _intervals(intervals, duration: float) -> np.ndarray:
    if intervals is None:
        return np.array([[0., duration]])
    array = np.asarray(intervals, dtype=float)
    if array.size == 0 and array.shape in ((0,), (0, 2)):
        return np.empty((0, 2))
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("eligible_intervals_s must have finite shape (n, 2)")
    if (np.any(array[:, 0] < 0) or np.any(array[:, 1] > duration)
            or np.any(array[:, 0] >= array[:, 1])
            or np.any(array[1:, 0] < array[:-1, 1])):
        raise ValueError("eligible intervals must be sorted, nonoverlapping, positive, within song")
    # Adjacent intervals represent one continuous region, not an artificial gap.
    merged = []
    for start, stop in array:
        if merged and start == merged[-1][1]:
            merged[-1][1] = stop
        else:
            merged.append([start, stop])
    return np.asarray(merged, dtype=float).reshape(-1, 2)


def _regions(times: np.ndarray, intervals: np.ndarray, name: str) -> np.ndarray:
    regions = np.full(len(times), -1, dtype=int)
    for region, (start, stop) in enumerate(intervals):
        regions[(times >= start) & (times < stop)] = region
    if np.any(regions < 0):
        raise ValueError(f"{name} contains events outside eligible half-open intervals; filter explicitly")
    return regions


@dataclass(frozen=True)
class VisualEventScore:
    channel: str
    average_precision: float | None
    reference_count: int
    candidate_count: int
    confidence_thresholds: np.ndarray
    true_positives: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    deployment_threshold: float
    emitted_count: int
    matched_count: int
    false_cues_per_minute: float | None
    reference_events_per_minute: float | None
    emitted_events_per_minute: float | None
    eligible_duration_s: float
    duration_s: float
    coverage: float
    tolerance_s: float = VISUAL_TOLERANCE_S
    scope: str = "authored_video_visual_event_proxy_not_acoustic_beat_accuracy"


def visual_event_ap(
    reference_times_s, candidate_times_s, confidences, *, duration_s: float,
    deployment_threshold: float, channel: str, eligible_intervals_s=None,
) -> VisualEventScore:
    """Rank explicit candidates; return AP and a separately fixed operating point.

    At every distinct confidence c (descending), select scores >= c. TP(c) is
    maximum-cardinality one-to-one bipartite matching within inclusive +/-0.2 s
    (1e-12 s arithmetic slack), in the SAME eligible continuous region. An
    augmenting-path update may rematch previous candidates; greedy nearest
    matching can undercount and is not used. Equal confidence values enter as
    ONE group. Thus input ordering within a tie cannot inflate AP.

    P_g = TP_g / candidates_g; R_g = TP_g / reference_count.
    Noninterpolated AP = sum_g (R_g - R_(g-1)) * P_g. A positive song with no
    predictions has AP=0. A no-reference song has AP=None and empty recall, even
    when predictions exist; its false-cue rate and coverage remain defined.
    Precision is empty only when there are no candidates. Ties are exact float
    equality, not a data-dependent quantization. Confidences need only be finite.

    Deployment uses the EXPLICIT externally frozen score >= deployment_threshold,
    not every AP candidate or a target-count-matched threshold. False cues/min =
    60 * (emitted - matched) / eligible_duration_s. With zero eligible duration,
    rates and AP are None and coverage is 0, never fabricated success.

    Eligible intervals are half-open [start, stop), with [0,duration_s) default.
    All supplied events must be eligible: out-of-region events raise instead of
    silently disappearing. Reference duplicates raise; duplicate predictions are
    allowed and compete for one target (no implicit deduplication). Local maxima
    and >=0.2 s prediction NMS are an upstream contract, not performed here.
    """
    duration = _positive(duration_s, "duration_s")
    if channel not in ("accent", "color"):
        raise ValueError("channel must explicitly be 'accent' or 'color'")
    if not np.isfinite(deployment_threshold):
        raise ValueError("deployment_threshold must be finite and externally frozen")
    reference = _vector(reference_times_s, "reference_times_s")
    candidates = _vector(candidate_times_s, "candidate_times_s")
    scores = _vector(confidences, "confidences")
    if len(scores) != len(candidates):
        raise ValueError("candidate_times_s and confidences must have equal length")
    if len(np.unique(reference)) != len(reference):
        raise ValueError("duplicate reference events are not allowed")
    intervals = _intervals(eligible_intervals_s, duration)
    reference = np.sort(reference)
    reference_regions = _regions(reference, intervals, "reference_times_s")
    candidate_regions = _regions(candidates, intervals, "candidate_times_s")
    eligible_duration = float(np.sum(intervals[:, 1] - intervals[:, 0]))
    # The deterministic tie order only selects a matching, never AP rank order.
    order = np.lexsort((np.arange(len(scores)), candidates, -scores))
    adjacency = []
    for index in range(len(candidates)):
        distances = np.abs(reference - candidates[index])
        adjacent = np.flatnonzero((distances <= VISUAL_TOLERANCE_S + _BOUNDARY_EPS_S)
                                  & (reference_regions == candidate_regions[index]))
        adjacent = adjacent[np.lexsort((reference[adjacent], distances[adjacent]))]
        adjacency.append(adjacent)
    owner = np.full(len(reference), -1, dtype=int)
    candidate_match = np.full(len(candidates), -1, dtype=int)

    def augment(start: int) -> bool:
        # Iterative alternating-path search avoids recursion limits on long songs.
        queue = deque([start])
        seen_candidates = {start}
        predecessor = {}
        while queue:
            candidate = queue.popleft()
            for target in adjacency[candidate]:
                target = int(target)
                if target in predecessor:
                    continue
                predecessor[target] = candidate
                previous = int(owner[target])
                if previous == -1:
                    while target != -1:
                        candidate = predecessor[target]
                        old_target = int(candidate_match[candidate])
                        owner[target] = candidate
                        candidate_match[candidate] = target
                        target = old_target
                    return True
                if previous not in seen_candidates:
                    seen_candidates.add(previous)
                    queue.append(previous)
        return False

    thresholds, counts, precisions = [], [], []
    true_count, index = 0, 0
    while index < len(order):
        threshold = float(scores[order[index]])
        stop = index
        while stop < len(order) and scores[order[stop]] == threshold:
            true_count += int(augment(int(order[stop])))
            stop += 1
        thresholds.append(threshold)
        counts.append(true_count)
        precisions.append(true_count / stop)
        index = stop
    thresholds = np.asarray(thresholds, dtype=float)
    counts = np.asarray(counts, dtype=int)
    precision = np.asarray(precisions, dtype=float)
    recall = counts / len(reference) if len(reference) else np.empty(0)
    ap = (float(np.sum(np.diff(np.r_[0., recall]) * precision))
          if len(reference) else None)
    emitted = int(np.count_nonzero(scores >= deployment_threshold))
    selected_groups = np.flatnonzero(thresholds >= deployment_threshold)
    matched = int(counts[selected_groups[-1]]) if len(selected_groups) else 0
    rate = lambda count: 60. * count / eligible_duration if eligible_duration else None
    return VisualEventScore(
        channel, ap, len(reference), len(candidates), thresholds, counts, precision, recall,
        float(deployment_threshold), emitted, matched, rate(emitted - matched),
        rate(len(reference)), rate(emitted), eligible_duration, duration, eligible_duration / duration)


@dataclass(frozen=True)
class MetricObservation:
    """One complete recording/coach/draw/seed metric, NOT one frame/window.

    Use one declared coach ID for a recording-level target. Multiple windows
    must be combined into that complete recording's instrument before this API.
    ``None`` explicitly denotes an undefined metric, such as no-reference AP.
    """
    song: str
    recording: str
    coach: str
    seed: int
    draw: int
    value: float | None


@dataclass(frozen=True)
class SongAggregation:
    values: dict[str, float | None]
    macro_mean: float | None
    eligible_songs: tuple[str, ...]
    undefined_songs: tuple[str, ...]
    defined_observations: int
    undefined_observations: int
    defined_observations_by_song: dict[str, int]
    total_observations_by_song: dict[str, int]
    song_coverage: float


def _declared_ids(values: Sequence, name: str) -> tuple:
    result = tuple(values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be nonempty and unique")
    return result


def _observation_map(rows: Iterable[MetricObservation]) -> dict:
    result = {}
    for row in rows:
        if not all(isinstance(value, str) and value for value in (row.song, row.recording, row.coach)):
            raise ValueError("song, recording, and coach IDs must be nonempty strings")
        key = (row.song, row.seed, row.recording, row.coach, row.draw)
        if key in result:
            raise ValueError(f"duplicate metric observation: {key}")
        if row.value is not None and not np.isfinite(row.value):
            raise ValueError("metric values must be finite or explicit None, not NaN")
        result[key] = row.value
    return result


def _mean_defined(values) -> float | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def macro_song_mean(
    observations: Iterable[MetricObservation], *, expected_songs: Sequence[str],
    expected_seeds: Sequence[int], expected_draws: Sequence[int],
) -> SongAggregation:
    """Equal-weight hierarchy: draw -> coach -> recording -> seed -> SONG.

    Every declared song/seed/draw must be represented, including undefined AP.
    Each song's recording/coach inventory must agree across seeds and draws.
    Missing cells and duplicate cells raise. Within a recording/coach, undefined
    status must also agree across seeds/draws: model failure is not eligibility.
    A entirely missing recording cannot be detected without an external manifest;
    freeze/audit that inventory upstream. Paired inference checks arm inventories.

    None children are excluded ONLY from this metric's means; counts and wholly
    undefined song IDs are returned. Thus AP may be conditional on positive
    recordings/songs, explicitly reported. Separately aggregate false-cue rates
    and eligible-duration coverage for ALL songs, not just AP-eligible songs.
    Do not pool events/frames or let songs with more recordings/draws dominate.
    Seeds are averaged, never selected and never counted as independent songs.
    """
    songs = _declared_ids(expected_songs, "expected_songs")
    seeds = _declared_ids(expected_seeds, "expected_seeds")
    draws = _declared_ids(expected_draws, "expected_draws")
    cells = _observation_map(observations)
    if {key[0] for key in cells} != set(songs):
        raise ValueError("observed song IDs differ from the explicit expected inventory")
    inventory = defaultdict(set)
    for song, seed, recording, coach, draw in cells:
        if seed not in seeds or draw not in draws:
            raise ValueError("undeclared training seed or generation draw")
        inventory[song].add((recording, coach))
    for song in songs:
        for recording, coach in inventory[song]:
            keys = [(song, seed, recording, coach, draw) for seed in seeds for draw in draws]
            if any(key not in cells for key in keys):
                raise ValueError("incomplete recording/coach/seed/draw cells")
            if len({cells[key] is None for key in keys}) != 1:
                raise ValueError("undefined eligibility differs across seeds/draws")
    values = {}
    defined_counts, total_counts = {}, {}
    for song in songs:
        seed_means = []
        recordings = sorted({item[0] for item in inventory[song]})
        for seed in seeds:
            recording_means = []
            for recording in recordings:
                coaches = sorted(coach for rec, coach in inventory[song] if rec == recording)
                coach_means = [_mean_defined(cells[(song, seed, recording, coach, draw)]
                                            for draw in draws) for coach in coaches]
                recording_means.append(_mean_defined(coach_means))
            seed_means.append(_mean_defined(recording_means))
        values[song] = _mean_defined(seed_means)
        song_cells = [value for key, value in cells.items() if key[0] == song]
        defined_counts[song] = sum(value is not None for value in song_cells)
        total_counts[song] = len(song_cells)
    eligible = tuple(song for song in songs if values[song] is not None)
    undefined = tuple(song for song in songs if values[song] is None)
    defined_count = sum(defined_counts.values())
    return SongAggregation(values, _mean_defined(values.values()), eligible, undefined,
                           defined_count, len(cells) - defined_count, defined_counts,
                           total_counts, len(eligible) / len(songs))


@dataclass(frozen=True)
class PairedBootstrap:
    difference: float
    interval: tuple[float, float]
    p_value_two_sided: float
    song_differences: dict[str, float]
    excluded_undefined_songs: tuple[str, ...]
    song_coverage: float
    confidence: float
    random_seed: int
    replicates: int = BOOTSTRAP_REPLICATES
    unit: str = "canonical_song_after_paired_seed_and_within_song_averaging"


def paired_song_bootstrap(
    candidate: Iterable[MetricObservation], comparator: Iterable[MetricObservation], *,
    expected_songs: Sequence[str], expected_seeds: Sequence[int], expected_draws: Sequence[int],
    random_seed: int = 0, confidence: float = .95,
) -> PairedBootstrap:
    """Paired 10,000-replicate percentile bootstrap; unit = canonical song.

    First validate identical leaf inventories and undefined patterns, then apply
    macro_song_mean to each arm. For n defined songs, d_i = candidate_i -
    comparator_i. Each replicate samples n song indices with replacement and
    averages d_i. CI is numpy's linear empirical quantiles at alpha/2 and
    1-alpha/2. Require n>=2; still no power/normality guarantee is asserted.
    Frames, coaches, recordings, draws, and training seeds are NOT resampled.

    Two-sided null-centered bootstrap p = (1 + count(|d*_mean - d_mean| >=
    |d_mean|)) / (10000+1). This is an approximate bootstrap test, not an exact
    randomization test. CI is UNADJUSTED; holm_adjust corrects p-values only and
    does not turn these intervals into multiplicity-corrected intervals.
    """
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    candidate, comparator = tuple(candidate), tuple(comparator)
    a_cells, b_cells = _observation_map(candidate), _observation_map(comparator)
    if a_cells.keys() != b_cells.keys():
        raise ValueError("paired arms must have identical song/recording/coach/seed/draw cells")
    if any((a_cells[key] is None) != (b_cells[key] is None) for key in a_cells):
        raise ValueError("paired arms must share metric eligibility; do not drop failures")
    arguments = dict(expected_songs=expected_songs, expected_seeds=expected_seeds,
                     expected_draws=expected_draws)
    a, b = macro_song_mean(candidate, **arguments), macro_song_mean(comparator, **arguments)
    # Stable canonical order makes row/inventory ordering irrelevant to the RNG.
    songs = sorted(a.eligible_songs)
    if len(songs) < 2:
        raise ValueError("paired bootstrap needs at least two defined canonical songs")
    differences = np.array([a.values[song] - b.values[song] for song in songs])
    rng = np.random.default_rng(random_seed)
    means = np.empty(BOOTSTRAP_REPLICATES)
    # Bound memory without changing draw order or the sampling unit.
    for start in range(0, BOOTSTRAP_REPLICATES, 256):
        stop = min(start + 256, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(songs), size=(stop - start, len(songs)))
        means[start:stop] = differences[indices].mean(axis=1)
    estimate = float(differences.mean())
    alpha = 1. - confidence
    low, high = np.quantile(means, [alpha / 2., 1. - alpha / 2.], method="linear")
    p_value = (1. + np.count_nonzero(np.abs(means - estimate) >= abs(estimate))) / (len(means) + 1.)
    return PairedBootstrap(estimate, (float(low), float(high)), float(p_value),
                           dict(zip(songs, map(float, differences))), a.undefined_songs,
                           a.song_coverage, float(confidence), random_seed)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values for the COMPLETE predeclared family.

    Sorted p_(i), i=0..m-1: adjusted_(i) = min(1, max_(j<=i) (m-j)*p_(j)).
    Exact p ties use name order and receive the same adjusted value. Reject
    missing/NaN values rather than shrinking a family after seeing results.
    Empty family returns {}; family completeness is the caller's manifest duty.
    This helper does NOT compute simultaneous or corrected confidence intervals.
    """
    if any(not np.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("p-values must be finite in [0, 1]; no missing family members")
    adjusted, running = {}, 0.
    for index, name in enumerate(sorted(p_values, key=lambda key: (p_values[key], key))):
        running = max(running, (len(p_values) - index) * p_values[name])
        adjusted[name] = min(1., float(running))
    return {name: adjusted[name] for name in p_values}


def fixed_shift_metadata(
    canonical_song: str, duration_s: float, offsets_s, *, minimum_circular_distance_s: float,
) -> dict:
    """Validate/record 20 externally frozen shifts; do not search for bad scores.

    Each within-song transform is t' = (t + offset_s) mod duration_s. Record raw
    and modulo offsets plus min(offset_mod, duration-offset_mod). Require 20
    unique nontrivial circular offsets, each at least the caller's PREDECLARED
    minimum distance (>0.2 s). This geometric check does not prove a periodic
    song is decorrelated. No cross-song wrapping, data-dependent redraw, or event
    extraction occurs here. Recheck shifted events against fixed eligibility
    explicitly upstream; never roll invalid regions into valid labels silently.
    """
    if not isinstance(canonical_song, str) or not canonical_song:
        raise ValueError("canonical_song must be a nonempty ID")
    duration = _positive(duration_s, "duration_s")
    minimum = _positive(minimum_circular_distance_s, "minimum_circular_distance_s")
    if minimum <= VISUAL_TOLERANCE_S:
        raise ValueError("minimum circular distance must exceed visual-label tolerance")
    offsets = _vector(offsets_s, "offsets_s")
    if len(offsets) != FIXED_SHIFT_COUNT:
        raise ValueError("exactly 20 externally fixed offsets are required")
    modulo = np.mod(offsets, duration)
    distance = np.minimum(modulo, duration - modulo)
    if np.any(distance < minimum):
        raise ValueError("trivial or insufficient circular shift")
    if np.any(np.diff(np.sort(modulo)) <= _BOUNDARY_EPS_S):
        raise ValueError("duplicate offsets modulo song duration")
    return dict(canonical_song=canonical_song, duration_s=duration, offsets_s=offsets.tolist(),
                offsets_modulo_s=modulo.tolist(), circular_distances_s=distance.tolist(),
                minimum_circular_distance_s=minimum, shift_count=FIXED_SHIFT_COUNT,
                transform="(event_time_s + offset_s) % duration_s",
                scope="within_one_canonical_song_only; fixed_before_evaluation")


@dataclass(frozen=True)
class PaletteSummary:
    """Duration-weighted occupancy, cue-step transitions, and separate diversity."""
    occupancy: np.ndarray
    cue_transition_joint: np.ndarray | None
    changed_transition_joint: np.ndarray | None
    used_states: int
    occupied_vocabulary_fraction: float
    entropy_normalized: float
    effective_states: float
    change_count: int
    changes_per_minute: float
    keep_fraction: float | None
    observed_residence_s: np.ndarray


@dataclass(frozen=True)
class PaletteComparison:
    categorical_time_occupancy_tv: float
    categorical_cue_transition_joint_tv: float | None
    categorical_changed_transition_joint_tv: float | None
    generated: PaletteSummary
    reference: PaletteSummary
    scope: str = "fixed_vocabulary_categorical_histogram_proxy_not_preregistered_HS_or_temporal_error"


def _palette_summary(states: np.ndarray, times: np.ndarray, duration: float, k: int) -> PaletteSummary:
    lengths = np.diff(np.r_[times, duration])
    occupancy = np.bincount(states, weights=lengths, minlength=k) / duration
    positive = occupancy > 0
    entropy = float(-np.sum(occupancy[positive] * np.log(occupancy[positive])))
    changed = states[1:] != states[:-1]
    pair_ids = states[:-1] * k + states[1:]

    def histogram(ids):
        return (np.bincount(ids, minlength=k*k).reshape(k, k) / len(ids)
                if len(ids) else None)

    transitions = histogram(pair_ids)
    changed_transitions = histogram(pair_ids[changed])
    run_starts = np.r_[0., times[1:][changed]]
    return PaletteSummary(
        occupancy, transitions, changed_transitions, int(positive.sum()), float(positive.sum() / k),
        entropy / np.log(k) if k > 1 else 0., float(np.exp(entropy)), int(changed.sum()),
        60. * changed.sum() / duration, float(1. - changed.mean()) if len(changed) else None,
        np.diff(np.r_[run_starts, duration]))


def palette_metrics(
    generated_states, reference_states, cue_times_s, *, duration_s: float, vocabulary_size: int,
) -> PaletteComparison:
    """Compare GENERATED (not teacher-forced) palette tokens on ONE cue inventory.

    Integer states in [0,K) are held on [cue_i,cue_(i+1)); first cue is exactly
    0, final state holds to duration_s. Times are strictly increasing and < end.
    A state at every cue is mandatory, even a keep. Both arms must use the same
    externally frozen vocabulary. One draw per call; aggregate paired draws and
    seeds later. No RGB/intensity, segmentation, label extraction, fitting, or
    cue retiming is done here. Never fit vocabulary on held-out sequences.

    Occupancy p_k = sum_i duration_i * 1[state_i=k] / song_duration.
    Cue-step joint q_ab = count(adjacent cue pair (a,b))/(number_of_cues-1),
    INCLUDING diagonal keep pairs. Changed-only joint excludes keeps, with its
    own count denominator. All histograms sum to one when defined. Error is total
    variation = .5*sum(abs(p-q)), bounded [0,1]; no arbitrary integer-state
    geometry is assumed. Transition error is None if either histogram is empty,
    not zero or a fabricated maximal penalty. Joint transition TV is NOT a
    row-normalized conditional-probability error or the temporal primary metric.

    Diversity is reported separately: used states, used/K, entropy/log(K), and
    exp(entropy), using duration occupancy (K=1 normalized entropy=0). Residence
    lengths are observed within-clip runs INCLUDING censored boundary runs,
    not uncensored lifetimes. A constant palette has effective_states=1 and
    no changed-transition histogram. Report reference-relative diversity; do not
    call constancy success from an undefined conditional error. High diversity
    alone is not success either: independent held/flicker can have perfect
    occupancy and wrong joint transitions. These summaries ignore exact phase.
    """
    duration = _positive(duration_s, "duration_s")
    if (isinstance(vocabulary_size, (bool, np.bool_))
            or not isinstance(vocabulary_size, (int, np.integer)) or vocabulary_size < 1):
        raise ValueError("vocabulary_size must be a positive integer")
    times = _vector(cue_times_s, "cue_times_s")
    if (not len(times) or times[0] != 0 or np.any(np.diff(times) <= 0)
            or times[-1] >= duration):
        raise ValueError("cue times must start at 0, strictly increase, and end before duration")
    validated = []
    for name, tokens in (("generated_states", generated_states), ("reference_states", reference_states)):
        states = np.asarray(tokens)
        if (states.shape != times.shape or not np.issubdtype(states.dtype, np.integer)
                or np.any(states < 0) or np.any(states >= vocabulary_size)):
            raise ValueError(f"{name} must be integer vocabulary tokens, one per shared cue")
        validated.append(states.astype(np.int64))
    generated, reference = [_palette_summary(states, times, duration, vocabulary_size)
                            for states in validated]

    def tv(p, q):
        return float(.5 * np.abs(p - q).sum()) if p is not None and q is not None else None

    return PaletteComparison(tv(generated.occupancy, reference.occupancy),
                             tv(generated.cue_transition_joint, reference.cue_transition_joint),
                             tv(generated.changed_transition_joint, reference.changed_transition_joint),
                             generated, reference)

"""Bounded CPU instruments for the dynamic-dance experiment.

These are engineering and timing measurements, not semantic labels or artistic
scores.  A constant body, a busy body, an RMS-driven body, and music-appropriate
choreography are different hypotheses.  Missing evidence is reported as missing,
never as a zero or an automatic pass. No function generates dance or light.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
import numpy as np

from lightsem.conductor.body_transfer_v2.skeleton import PARENTS

REGIONS = {
    'left_arm': (16, 18, 20), 'right_arm': (17, 19, 21),
    'left_leg': (1, 4, 7, 10), 'right_leg': (2, 5, 8, 11),
    'torso': (3, 6, 9, 12, 15),
}
FEET = (7, 8, 10, 11)
# Frozen before outcome inspection. These are engineering tolerances, not
# validated thresholds of human musical perception.
GATE_PARAMETERS = {
    'version': 'dynamic-dance-engineering-v1',
    'bone_drift_allowance_m': .002, 'bone_drift_relative_allowance': .10,
    'foot_slide_allowance_m_s': .02, 'foot_slide_relative_allowance': .10,
    'jerk_relative_allowance': .20, 'jerk_absolute_allowance_m_s3': .5,
    'acceleration_saturation_allowance_fraction': .02,
    'audio_response_floor_m': 1e-5, 'repeat_noise_multiplier': 5.,
    'musical_metric_minimum_lift': .02,
    'emitted_response_minimum_ticks': 1.,
    'velocity_retention_correlation_allowance': .05,
    'speed_saturation_threshold_fraction': .98,
    'hold_speed_threshold_m_s': .05, 'support_height_threshold_m': .05,
    'support_vertical_speed_threshold_m_s': .20,
    'scope': 'Fixed engineering gates; no hold quota; no artistic qualification.',
}


def _times(times, n, minimum=4):
    t = np.asarray(times, float)
    if t.shape != (n,) or n < minimum or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f'Expected >= {minimum} finite, strictly increasing timestamps matching frames')
    return t


def _stats(values):
    x = np.asarray(values, float)
    if not x.size:
        return dict.fromkeys(('min', 'median', 'p95', 'max', 'mean', 'std'))
    if not np.isfinite(x).all():
        raise ValueError('Statistics require finite values')
    result = dict(zip(('min', 'median', 'p95', 'max'), map(float, np.quantile(x, [0, .5, .95, 1]))))
    result.update(mean=float(x.mean()), std=float(x.std()))
    return result


def _corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape or not a.size:
        return None
    a, b = a - a.mean(), b - b.mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / scale) if scale > 1e-20 else None


def _derivative(x, times):
    # Forward interval differences retain alternating-frame jitter, which a
    # centered derivative erases. Final interval value repeats only for plotting.
    dt = np.diff(times).reshape((-1,) + (1,) * (np.asarray(x).ndim - 1))
    derivative = np.diff(x, axis=0) / dt
    return np.concatenate((derivative, derivative[-1:]), axis=0)


def _runs(mask, times):
    """Durations on [t0,t_last]; final frame never fabricates extra support."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    widths = np.diff(times, append=times[-1])
    durations = [float(widths[a:b].sum()) for a, b in zip(starts, ends)]
    return {'count': len(durations), 'durations_s': durations,
            'longest_s': max(durations, default=0.), 'total_s': sum(durations)}


def _body_and_times(body, times=None):
    """Accept arrays or the public DanceResult contract without importing it."""
    if times is None:
        if isinstance(body, Mapping):
            times, body = body.get('times', body.get('time')), body.get('world', body.get('body'))
        else:
            times, body = getattr(body, 'times', getattr(body, 'time', None)), getattr(body, 'world', getattr(body, 'body', None))
    b = np.asarray(body, float)
    if b.ndim != 3 or b.shape[1:] != (22, 3):
        raise ValueError('Expected body [T,22,3]')
    return b, _times(times, len(b))


def dynamics_series(body, times=None):
    """Frame-aligned SI-unit traces; root-relative regions exclude root travel."""
    b, t = _body_and_times(body, times)
    if not np.isfinite(b).all():
        raise ValueError('Body must be finite')
    v = _derivative(b, t)
    relative = b - b[:, :1]
    rv = _derivative(relative, t)
    return {'time': t, 'root_speed_m_s': np.linalg.norm(v[:, 0], axis=-1),
            'world_speed_m_s': np.linalg.norm(v, axis=-1).mean(1),
            'relative_speed_m_s': np.linalg.norm(rv, axis=-1).mean(1),
            'regional_speed_m_s': {name: np.linalg.norm(rv[:, joints], axis=-1).mean(1)
                                   for name, joints in REGIONS.items()},
            'world_velocity_m_s': v, 'relative_velocity_m_s': rv}


def body_dynamics(body, times=None, *, contacts=None, reference_body=None, floor_y=None):
    """Body activity and physical diagnostics, with separate root/relative motion.

    When contacts are absent, support is a geometric proxy based on foot height
    and vertical speed only. Horizontal speed is intentionally not a support
    criterion: excluding sliding feet would make the sliding metric self-pass.
    Share ``floor_y`` and ``reference_body`` across compared conditions. A floor
    estimated from the current body is explicitly identified in the report.
    """
    if contacts is None and not isinstance(body, np.ndarray):
        contacts = body.get('contacts') if isinstance(body, Mapping) else getattr(body, 'contacts', None)
    b, t = _body_and_times(body, times)
    result = {'finite': bool(np.isfinite(b).all()), 'frames': len(b),
              'duration_s': float(t[-1] - t[0]),
              'scope': 'Kinematic and geometric proxies; motion amount is not choreography.'}
    if not result['finite']:
        return {**result, 'status': 'invalid_nonfinite_body', 'nonfinite_values': int((~np.isfinite(b)).sum())}
    reference = b if reference_body is None else np.asarray(reference_body, float)
    if reference.ndim != 3 or reference.shape[1:] != (22, 3) or not np.isfinite(reference).all():
        raise ValueError('Reference body must be finite [T,22,3]')
    s = dynamics_series(b, t)
    velocity, relative_velocity = s['world_velocity_m_s'], s['relative_velocity_m_s']
    acceleration = _derivative(velocity, t)
    jerk = _derivative(acceleration, t)
    relative_acceleration = _derivative(relative_velocity, t)
    relative_jerk = _derivative(relative_acceleration, t)
    parents = np.asarray(PARENTS[1:])
    reference_lengths = np.median(np.linalg.norm(reference[:, 1:] - reference[:, parents], axis=-1), axis=0)
    lengths = np.linalg.norm(b[:, 1:] - b[:, parents], axis=-1)
    drift = np.abs(lengths - reference_lengths)
    estimated_floor = floor_y is None
    floor = float(np.quantile(reference[:, FEET, 1], .02)) if estimated_floor else float(floor_y)
    if not np.isfinite(floor):
        raise ValueError('Floor must be finite')
    foot_velocity = velocity[:, FEET]
    horizontal_speed = np.linalg.norm(foot_velocity[..., [0, 2]], axis=-1)
    geometric = ((b[:, FEET, 1] <= floor + GATE_PARAMETERS['support_height_threshold_m']) &
                 (np.abs(foot_velocity[..., 1]) <= GATE_PARAMETERS['support_vertical_speed_threshold_m_s']))
    supplied = None
    if contacts is not None:
        c = np.asarray(contacts)
        if c.shape != geometric.shape or not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
            raise ValueError('Contacts must be finite [T,4] values in [0,1], feet 7,8,10,11')
        supplied = c >= .5
    # Geometric support remains primary even when a model predicts no contacts.
    held = s['world_speed_m_s'] < GATE_PARAMETERS['hold_speed_threshold_m_s']
    supported_hold = held & geometric.any(1)
    result.update(status='measured', root_speed_m_s=_stats(s['root_speed_m_s']),
        world_joint_speed_m_s=_stats(np.linalg.norm(velocity, axis=-1)),
        world_mean_speed_m_s=_stats(s['world_speed_m_s']),
        relative_joint_speed_m_s=_stats(np.linalg.norm(relative_velocity, axis=-1)),
        acceleration_m_s2=_stats(np.sqrt(np.mean(np.sum(acceleration ** 2, axis=-1), axis=1))),
        jerk_m_s3=_stats(np.sqrt(np.mean(np.sum(jerk ** 2, axis=-1), axis=1))),
        relative_acceleration_m_s2=_stats(np.sqrt(np.mean(np.sum(relative_acceleration ** 2, axis=-1), axis=1))),
        relative_jerk_m_s3=_stats(np.sqrt(np.mean(np.sum(relative_jerk ** 2, axis=-1), axis=1))),
        regional={name: {'speed_m_s': _stats(s['regional_speed_m_s'][name]),
                        'acceleration_m_s2': _stats(np.linalg.norm(relative_acceleration[:, joints], axis=-1)),
                        'jerk_m_s3': _stats(np.linalg.norm(relative_jerk[:, joints], axis=-1)),
                        'held': _runs(s['regional_speed_m_s'][name] < .05, t)}
                  for name, joints in REGIONS.items()},
        derivative_method='Forward interval differences; acceleration/jerk summaries use per-frame joint RMS.',
        geometry={'bone_drift_m': _stats(drift.max(axis=1)), 'all_bone_drift_m': _stats(drift),
                  'bone_drift_gate_summary': 'p95 over frames of maximum bone error per frame', 'reference_bone_lengths_m': reference_lengths.tolist(),
                  'reference': 'own median bones' if reference_body is None else 'supplied body median bones',
                  'floor_y_m': floor, 'floor_estimated': estimated_floor,
                  'floor_penetration_m': _stats(np.maximum(floor - b[:, FEET, 1], 0.))},
        contacts={'kind': 'height/vertical-speed geometric support proxy, not ground truth',
                  'foot_joint_order': list(FEET), 'geometric_support_fraction': float(geometric.mean()),
                  'frames_without_geometric_support_fraction': float((~geometric.any(1)).mean()),
                  'geometric_supported_slide_m_s': _stats(horizontal_speed[geometric]),
                  'predicted_contact_fraction': float(supplied.mean()) if supplied is not None else None,
                  'predicted_contact_slide_m_s': _stats(horizontal_speed[supplied]) if supplied is not None else None,
                  'predicted_vs_geometric_agreement': float((supplied == geometric).mean()) if supplied is not None else None},
        holds={'threshold_m_s': .05, 'world': _runs(held, t), 'world_supported': _runs(supported_hold, t),
               'scope': 'Descriptive; no target proportion or minimum hold quota.'},
        exactly_constant_body=bool(np.all(b == b[:1])))
    return result


def gesture_events(speed, times, *, prominence_m_s=.08, minimum_separation_s=.2):
    """Fixed speed-peak proxy, not movement/phrase annotations."""
    from scipy.signal import find_peaks
    x = np.asarray(speed, float)
    t = _times(times, len(x))
    if x.ndim != 1 or not np.isfinite(x).all() or prominence_m_s <= 0 or minimum_separation_s <= 0:
        raise ValueError('Finite speed vector and positive fixed event thresholds required')
    candidates = find_peaks(x, prominence=prominence_m_s)[0]
    selected = []
    # Descending prominence selection; impose separation in seconds, not frames.
    for index in sorted(candidates, key=lambda i: (-x[i], t[i])):
        if all(abs(t[index] - t[j]) >= minimum_separation_s for j in selected):
            selected.append(int(index))
    return t[sorted(selected)]


def _event_score(gestures, events, tolerance):
    # Earliest feasible one-to-one pairing maximizes matched count on a line.
    i = j = 0
    errors = []
    while i < len(gestures) and j < len(events):
        delta = gestures[i] - events[j]
        if abs(delta) <= tolerance:
            errors.append(abs(float(delta))); i += 1; j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    n = len(errors)
    precision = n / len(gestures) if len(gestures) else 0.
    recall = n / len(events) if len(events) else None
    f1 = 2 * n / (len(gestures) + len(events)) if len(events) else None
    return {'gesture_count': len(gestures), 'musical_event_count': len(events), 'matched_count': n,
            'precision': precision, 'recall': recall, 'f1': f1,
            'absolute_timing_error_s': _stats(errors)}


def event_alignment(gesture_times, event_times, *, start_s, end_s, tolerance_s=.12,
                    shift_fractions=(.113, .197, .283, .379, .463, .571, .659, .743, .857)):
    """One-to-one event precision/recall against a fixed circular-shift null.

    Events must be selected from audio before observing candidate movement.
    Empty audio references are unmeasurable; a quiet body with reference events
    has recall/F1 zero. Busy jitter cannot earn recall without paying precision.
    """
    if not np.isfinite([start_s, end_s, tolerance_s]).all() or end_s <= start_s or tolerance_s <= 0:
        raise ValueError('Finite positive window and tolerance required')
    arrays = [np.asarray(v, float) for v in (gesture_times, event_times)]
    if any(x.ndim != 1 or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Event times must be finite vectors')
    g, e = [np.unique(x[(x >= start_s) & (x < end_s)]) for x in arrays]
    fractions = np.asarray(shift_fractions, float)
    if fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all() or np.any((fractions <= 0) | (fractions >= 1)):
        raise ValueError('Fixed nonzero circular-shift fractions must lie in (0,1)')
    duration = end_s - start_s
    score = _event_score(g, e, tolerance_s)
    null = [_event_score(g, np.sort(start_s + ((e - start_s + f * duration) % duration)), tolerance_s)['f1']
            for f in fractions]
    finite_null = [v for v in null if v is not None]
    q95 = float(np.quantile(finite_null, .95)) if finite_null else None
    return {**score, 'tolerance_s': tolerance_s, 'null_shift_fractions': fractions.tolist(),
            'null_f1': null, 'null_f1_p95': q95,
            'f1_minus_null_p95': score['f1'] - q95 if q95 is not None else None,
            'scope': 'Alignment with preselected audio events; not artistic or semantic correctness.'}


def phrase_contrasts(series, times, intervals):
    """Describe predeclared audio intervals without inventing desired directions.

    Each interval has ``start_s``, ``end_s``, optional ``label`` and ``evidence``.
    Higher contrast is not automatically better and is never a semantic gate.
    """
    x = np.asarray(series, float)
    t = _times(times, len(x))
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite one-dimensional trace')
    rows = []
    for interval in intervals:
        a, b = float(interval['start_s']), float(interval['end_s'])
        if not np.isfinite([a, b]).all() or b <= a:
            raise ValueError('Invalid phrase interval')
        selected = (t >= a) & (t < b)
        rows.append({'start_s': a, 'end_s': b, 'label': interval.get('label'),
                     'evidence': interval.get('evidence'), 'observations': int(selected.sum()),
                     'values': _stats(x[selected])})
    medians = [row['values']['median'] for row in rows if row['values']['median'] is not None]
    return {'intervals': rows, 'between_interval_median_std': float(np.std(medians)) if medians else None,
            'scope': 'Audio-selected interval descriptions; no gold semantic direction or hold quota.'}


def selective_response(original, perturbed, *, assigned_regions, repeat_noise_m=0.):
    """Root-relative source-removal/shift response; no musical benefit inference."""
    a, b = np.asarray(original, float), np.asarray(perturbed, float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (22, 3) or not np.isfinite([a, b]).all():
        raise ValueError('Two finite bodies on the same clock are required')
    selected = set(assigned_regions)
    if not selected or not selected <= REGIONS.keys() or not np.isfinite(repeat_noise_m) or repeat_noise_m < 0:
        raise ValueError('Declare known assigned regions and nonnegative repeated-run noise')
    difference = (a - a[:, :1]) - (b - b[:, :1])
    by_region = {name: float(np.sqrt(np.mean(difference[:, joints] ** 2))) for name, joints in REGIONS.items()}
    intended = float(np.mean([by_region[n] for n in selected]))
    rest = [by_region[n] for n in REGIONS if n not in selected]
    unintended = float(np.mean(rest)) if rest else None
    floor = max(GATE_PARAMETERS['audio_response_floor_m'], 5 * repeat_noise_m)
    return {'regional_rms_difference_m': by_region, 'assigned_mean_rms_m': intended,
            'unassigned_mean_rms_m': unintended,
            'assigned_minus_unassigned_m': intended - unintended if unintended is not None else None,
            'assigned_response_above_noise': intended > floor, 'numerical_floor_m': floor,
            'scope': 'Source sensitivity/selectivity only. Use loudness-matched source interventions to test content.'}


def _interpolate(values, times, targets):
    flat = np.asarray(values).reshape(len(times), -1)
    return np.stack([np.interp(targets, times, column) for column in flat.T], axis=1)


def tempo_aligned_comparison(body_a, times_a, beats_a, body_b, times_b, beats_b, *, samples_per_beat=8, max_beats=256):
    """Compare body shape on beat ordinal, making same-tempo reuse visible."""
    a, ta = _body_and_times(body_a, times_a)
    b, tb = _body_and_times(body_b, times_b)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Bodies must be finite')
    beats = []
    for values, t in ((beats_a, ta), (beats_b, tb)):
        v = np.asarray(values, float)
        if v.ndim != 1 or not np.isfinite(v).all() or np.any(np.diff(v) <= 0):
            raise ValueError('Beat timestamps must be finite and increasing')
        beats.append(v[(v >= t[0]) & (v <= t[-1])])
    count = min(len(beats[0]), len(beats[1]), max_beats)
    if count < 2:
        return {'status': 'insufficient_beats', 'shared_beats': count, 'rms_difference_m': None}
    if samples_per_beat < 1 or max_beats < 2:
        raise ValueError('Positive sampling budget required')
    grid = np.arange((count - 1) * samples_per_beat + 1) / samples_per_beat
    qa, qb = [np.interp(grid, np.arange(count), v[:count]) for v in beats]
    aa = _interpolate(a - a[:, :1], ta, qa)
    bb = _interpolate(b - b[:, :1], tb, qb)
    return {'status': 'measured', 'shared_beats': count, 'samples': len(grid),
            'rms_difference_m': float(np.sqrt(np.mean((aa - bb) ** 2))),
            'centered_shape_correlation': _corr(aa, bb),
            'exactly_identical_aligned_shapes': bool(np.array_equal(aa, bb)),
            'scope': 'Ordinal beat alignment; different section lengths/beat errors can confound comparison.'}


def recurrence_alignment(body, times, music_features, music_times, *, max_frames=128,
                         minimum_pair_separation_s=2., shift_fractions=(.113, .283, .463, .659, .857)):
    """Body/audio recurrence Spearman with deterministic shifted-audio null.

    This compares self-distance geometry, not controller labels. Still only a
    proxy: acoustic repetition does not prescribe repeated choreography.
    Runtime/memory are bounded by max_frames, not song duration squared.
    """
    from scipy.stats import spearmanr
    b, t = _body_and_times(body, times)
    f = np.asarray(music_features, float)
    ft = _times(music_times, len(f), minimum=2)
    if f.ndim != 2 or not f.shape[1] or not np.isfinite(f).all() or not np.isfinite(b).all() or max_frames < 4:
        raise ValueError('Finite body/features and >=4 sample budget required')
    start, end = max(t[0], ft[0]), min(t[-1], ft[-1])
    if end <= start:
        raise ValueError('Body and music timestamps must overlap')
    q = np.linspace(start, end, min(max_frames, len(t), len(ft)))
    poses = _interpolate(b - b[:, :1], t, q)
    features = _interpolate(f, ft, q)
    ia, ib = np.triu_indices(len(q), 1)
    selected = q[ib] - q[ia] >= minimum_pair_separation_s
    ia, ib = ia[selected], ib[selected]
    def distances(x):
        # One pooled scale preserves feature geometry without amplifying nearly
        # constant dimensions. Spearman is invariant to this common scale.
        return np.linalg.norm(x[ia] - x[ib], axis=1)
    bd = distances(poses)
    def score(x):
        md = distances(x)
        if len(bd) < 3 or np.ptp(bd) <= 1e-12 or np.ptp(md) <= 1e-12:
            return None
        return float(spearmanr(bd, md).statistic)
    actual = score(features)
    shifts = sorted(set(max(1, min(len(q) - 1, round(float(frac) * len(q)))) for frac in shift_fractions))
    null = [score(np.roll(features, shift, axis=0)) for shift in shifts]
    valid = [v for v in null if v is not None]
    p95 = float(np.quantile(valid, .95)) if valid else None
    return {'spearman': actual, 'shifted_null': null, 'shift_frames': shifts,
            'null_p95': p95, 'lift_over_null_p95': actual - p95 if actual is not None and p95 is not None else None,
            'sampled_frames': len(q), 'compared_pairs': len(ia),
            'scope': 'Acoustic/body recurrence correspondence; not a gold motif or semantic score.'}


def motif_recurrence(body, times, interval_pairs, *, samples_per_interval=32):
    """Shape/velocity recurrence on explicitly supplied audio recurrence pairs."""
    b, t = _body_and_times(body, times)
    if not np.isfinite(b).all() or samples_per_interval < 4:
        raise ValueError('Finite body and >=4 motif samples required')
    relative = b - b[:, :1]
    rows = []
    for pair in interval_pairs:
        first, second = pair
        for start, end in (first, second):
            if not np.isfinite([start, end]).all() or not t[0] <= start < end <= t[-1]:
                raise ValueError('Motif intervals must lie inside body timeline')
        a, c = [_interpolate(relative, t, np.linspace(start, end, samples_per_interval)) for start, end in pair]
        rows.append({'intervals_s': [list(first), list(second)],
                     'shape_rms_m': float(np.sqrt(np.mean((a - c) ** 2))),
                     'shape_correlation': _corr(a, c),
                     'phase_velocity_correlation': _corr(np.diff(a, axis=0), np.diff(c, axis=0)),
                     'both_stationary': bool(np.ptp(a, axis=0).max() < 1e-8 and np.ptp(c, axis=0).max() < 1e-8)})
    return {'pairs': rows, 'scope': 'Duration-normalized shape comparison; reused/constant poses are disclosed, not praised.'}


def duplicate_streams(values, names=None, *, near_rms_tol=0., max_near_pairs=4096):
    """Full exact hashing; bounded deterministic near-pair RMS inspection.

    ``values`` is time x streams x any trailing coordinates. Near pairs are
    pairwise observations, never transitive clusters. No temporal subsampling
    hides short differences in otherwise identical streams.
    """
    x = np.asarray(values)
    if x.ndim < 2 or len(x) < 1 or not np.isfinite(x).all() or near_rms_tol < 0:
        raise ValueError('Finite time x streams array and nonnegative tolerance required')
    count = x.shape[1]
    names = list(map(str, range(count))) if names is None else list(names)
    if len(names) != count or len(set(names)) != count or max_near_pairs < 0:
        raise ValueError('Unique stream names and nonnegative pair budget required')
    groups = {}
    for i, name in enumerate(names):
        key = hashlib.sha256(np.ascontiguousarray(x[:, i] + 0).tobytes()).hexdigest()
        groups.setdefault(key, []).append(name)
    exact = [v for v in groups.values() if len(v) > 1]
    total = count * (count - 1) // 2
    near = []
    # Triangular pair indexing without allocating N squared pair arrays.
    ordinals = np.unique(np.linspace(0, total - 1, min(total, max_near_pairs)).astype(np.int64)) if total else []
    starts = np.arange(count, dtype=np.int64) * (2 * count - np.arange(count) - 1) // 2
    for ordinal in ordinals:
        i = int(np.searchsorted(starts, ordinal, side='right') - 1)
        j = int(i + 1 + ordinal - starts[i])
        delta = x[:, i].astype(float) - x[:, j].astype(float)
        rms = float(np.sqrt(np.mean(delta ** 2)))
        if rms <= near_rms_tol:
            near.append({'names': [names[i], names[j]], 'rms': rms, 'exact': bool(np.array_equal(x[:, i], x[:, j]))})
    return {'total_streams': count, 'unique_exact_streams': len(groups), 'exact_groups': exact,
            'near_pairs': near, 'near_rms_tolerance': float(near_rms_tol),
            'near_pairs_examined': len(ordinals), 'total_pairs': total, 'near_pairs_truncated': len(ordinals) < total}


def emission_diagnostics(requested_angles, emitted_angles, times, *, maximum_speed_deg_s=90.,
                         maximum_acceleration_deg_s2=360., tick_deg=540. / 65535.):
    """Compare requested motor angles with actual decoded DMX motor angles.

    Use only movers and a consistent, unwrapped pan branch. Mechanical limits
    apply to the 2D pan/tilt vector norm, matching the actual convex fitter.
    ``tick_deg`` is the largest relevant actual 16-bit axis tick.
    """
    a, b = np.asarray(requested_angles, float), np.asarray(emitted_angles, float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError('Matching [T,movers,2] requested/emitted angles required')
    t = _times(times, len(a))
    if not np.isfinite([maximum_speed_deg_s, maximum_acceleration_deg_s2, tick_deg]).all() or min(maximum_speed_deg_s, maximum_acceleration_deg_s2, tick_deg) <= 0:
        raise ValueError('Positive finite limits and actual DMX tick required')
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {'finite': False, 'status': 'invalid_nonfinite_angles'}
    dt = np.diff(t)[:, None, None]
    av, bv = np.diff(a, axis=0) / dt, np.diff(b, axis=0) / dt
    vt = (t[1:] + t[:-1]) / 2
    ba = np.diff(bv, axis=0) / np.diff(vt)[:, None, None]
    aa = np.diff(av, axis=0) / np.diff(vt)[:, None, None]
    fraction = GATE_PARAMETERS['speed_saturation_threshold_fraction']
    def proportion(mask):
        return float(mask.mean()) if mask.size else None
    source_energy = float(np.mean(av ** 2)) if av.size else 0.
    return {'finite': True, 'status': 'measured', 'moving_outputs': a.shape[1], 'tick_deg': float(tick_deg),
            'requested_speed_deg_s': _stats(np.linalg.norm(av, axis=-1)),
            'emitted_speed_deg_s': _stats(np.linalg.norm(bv, axis=-1)),
            'requested_acceleration_deg_s2': _stats(np.linalg.norm(aa, axis=-1)),
            'emitted_acceleration_deg_s2': _stats(np.linalg.norm(ba, axis=-1)),
            'emitted_speed_axis_max_deg_s': float(np.abs(bv).max()) if bv.size else None,
            'emitted_acceleration_axis_max_deg_s2': float(np.abs(ba).max()) if ba.size else None,
            'speed_saturation_fraction': proportion(np.linalg.norm(bv, axis=-1) >= fraction * maximum_speed_deg_s),
            'acceleration_saturation_fraction': proportion(np.linalg.norm(ba, axis=-1) >= fraction * maximum_acceleration_deg_s2),
            'velocity_correlation': _corr(av, bv),
            'velocity_rms_retention_ratio': float(np.sqrt(np.mean(bv ** 2) / source_energy)) if source_energy > 1e-20 else None,
            'angle_error_deg': _stats(np.linalg.norm(a - b, axis=-1)),
            'fraction_emitted_below_1_deg_s': proportion(np.linalg.norm(bv, axis=-1) < 1.),
            'exactly_constant_emitted': bool(np.all(b == b[:1])),
            'scope': 'Requested versus decoded emitted motion only; colors and brightness do not count.'}


def emitted_response(original, perturbed, *, tick_deg, repeat_noise_deg=0.):
    a, b = np.asarray(original, float), np.asarray(perturbed, float)
    if a.shape != b.shape or not a.size or not np.isfinite([a, b]).all():
        raise ValueError('Finite matching decoded motion arrays required')
    if not np.isfinite([tick_deg, repeat_noise_deg]).all() or tick_deg <= 0 or repeat_noise_deg < 0:
        raise ValueError('Positive tick and nonnegative repeat noise required')
    rms = float(np.sqrt(np.mean((a - b) ** 2)))
    floor = max(float(tick_deg), 5 * repeat_noise_deg)
    return {'rms_difference_deg': rms, 'minimum_effect_deg': floor,
            'above_quantization_and_noise': bool(rms >= floor),
            'scope': 'Emitted sensitivity only; does not establish beneficial musical adaptation.'}


def comparison_gates(candidate, baseline, *, candidate_emission=None, baseline_emission=None, evidence=None):
    """Conservative predeclared engineering gates; root owns scientific verdict.

    Evidence accepts ``audio_response_m``, ``audio_repeat_noise_m``,
    ``musical_score``, ``musical_control_p95``, ``musical_reference_kind``
    (``independent`` required), and ``emitted_response`` from emitted_response.
    Missing evidence is ``unqualified``, not fabricated zero or a pass.
    """
    evidence = {} if evidence is None else evidence
    p = GATE_PARAMETERS
    gates = {}
    def record(name, passed, **details):
        gates[name] = {'status': 'unqualified' if passed is None else ('pass' if passed else 'fail'), **details}
    record('finite_body', bool(candidate.get('finite')))
    def nonregression(name, path, absolute, relative):
        def get(report):
            for key in path:
                if not isinstance(report, Mapping) or key not in report:
                    return None
                report = report[key]
            return report
        c, b = get(candidate), get(baseline)
        limit = b + max(absolute, relative * b) if b is not None else None
        record(name, c <= limit if c is not None and limit is not None else None,
               candidate=c, baseline=b, maximum=limit)
    nonregression('bone_nonregression', ('geometry', 'bone_drift_m', 'p95'), p['bone_drift_allowance_m'], p['bone_drift_relative_allowance'])
    nonregression('supported_slide_nonregression', ('contacts', 'geometric_supported_slide_m_s', 'p95'), p['foot_slide_allowance_m_s'], p['foot_slide_relative_allowance'])
    # Jerk allowance is additive: baseline * 1.2 + .5, as frozen in protocol.
    c, b = candidate.get('jerk_m_s3', {}).get('p95'), baseline.get('jerk_m_s3', {}).get('p95')
    limit = b * (1 + p['jerk_relative_allowance']) + p['jerk_absolute_allowance_m_s3'] if b is not None else None
    record('jerk_nonregression', c <= limit if c is not None and limit is not None else None, candidate=c, baseline=b, maximum=limit)
    noise = evidence.get('audio_repeat_noise_m')
    audio = evidence.get('audio_response_m')
    floor = max(p['audio_response_floor_m'], p['repeat_noise_multiplier'] * noise) if noise is not None else None
    record('audio_sensitivity', audio > floor if audio is not None and floor is not None else None, observed_m=audio, minimum_m=floor)
    independent = evidence.get('musical_reference_kind') == 'independent'
    score, control = evidence.get('musical_score'), evidence.get('musical_control_p95')
    lift = score - control if score is not None and control is not None else None
    record('musical_control_lift', lift > p['musical_metric_minimum_lift'] if independent and lift is not None else None,
           lift=lift, reference_kind=evidence.get('musical_reference_kind'),
           reason=None if independent else 'No independent reference; controller labels cannot qualify their own output.')
    ce, be = candidate_emission or {}, baseline_emission or {}
    record('finite_emission', bool(ce['finite']) if 'finite' in ce else None)
    c, b = ce.get('acceleration_saturation_fraction'), be.get('acceleration_saturation_fraction')
    record('emitted_saturation_nonregression', c <= b + p['acceleration_saturation_allowance_fraction'] if c is not None and b is not None else None,
           candidate=c, baseline=b)
    c, b = ce.get('velocity_correlation'), be.get('velocity_correlation')
    record('emitted_retention_nonregression', c >= b - p['velocity_retention_correlation_allowance'] if c is not None and b is not None else None,
           candidate=c, baseline=b)
    response = evidence.get('emitted_response', {})
    record('emitted_effect', response.get('above_quantization_and_noise'), **response)
    states = [v['status'] for v in gates.values()]
    status = 'failed_engineering' if 'fail' in states else ('unqualified_missing_evidence' if 'unqualified' in states else 'engineering_pass_only')
    return {'status': status, 'gates': gates, 'parameters': dict(p),
            'artistic_verdict': 'pending_user_review', 'scientific_verdict': 'not_assigned_by_diagnostics'}


def release_contrast(series, times, release_intervals, reference_intervals, *,
                     denominator_floor=.01, shift_fractions=(.113, .197, .283, .379, .463, .571, .659, .743, .857)):
    """D2/D5 preselected release-versus-context effort contrast with shifted null.

    Intervals are (start,end) seconds, selected from audio before dance output.
    Both sets shift together, preserving durations and relative placement. A
    positive contrast means lower effort at declared release intervals. This is
    an explicit hypothesis, never a claim that every musical release must slow.
    Body units: m/s with floor .01; emitted units: deg/s with an explicit floor
    chosen before outcomes. The controller's own release labels cannot serve as
    an independent reference in comparison_gates.
    """
    x = np.asarray(series, float)
    t = _times(times, len(x))
    if x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0) or not np.isfinite(denominator_floor) or denominator_floor <= 0:
        raise ValueError('Nonnegative finite effort trace and positive denominator floor required')
    # Samples describe [time,time+dt) support; an end-exclusive review crop
    # still covers its final interval. This is not an extra body observation.
    support_end = t[-1] + float(np.median(np.diff(t)))
    duration = support_end - t[0]
    weights = np.diff(t, append=support_end)
    def parse(intervals):
        values = np.asarray(intervals, float)
        if not values.size:
            return np.empty((0, 2))
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError('Intervals must be finite [N,2]')
        if np.any(values[:, 1] <= values[:, 0]) or np.any(values[:, 0] < t[0]) or np.any(values[:, 1] > support_end + 1e-8):
            raise ValueError('Intervals must be positive and within trace window')
        return values
    release, reference = parse(release_intervals), parse(reference_intervals)
    def mask(intervals, shift):
        q = t[0] + ((t - t[0] - shift) % duration)
        result = np.zeros(len(t), bool)
        for start, end in intervals:
            result |= (q >= start) & (q < end)
        return result
    if np.any(mask(release, 0.) & mask(reference, 0.)):
        raise ValueError('Release and reference intervals must not overlap')
    def score(shift):
        low, context = mask(release, shift), mask(reference, shift)
        if not weights[low].sum() or not weights[context].sum():
            return None, None, None
        lv = float(np.average(x[low], weights=weights[low]))
        cv = float(np.average(x[context], weights=weights[context]))
        return (cv - lv) / (cv + lv + denominator_floor), lv, cv
    fractions = np.asarray(shift_fractions, float)
    if fractions.ndim != 1 or not len(fractions) or not np.isfinite(fractions).all() or np.any((fractions <= 0) | (fractions >= 1)):
        raise ValueError('Fixed nonzero circular-shift fractions required')
    actual, low, context = score(0.)
    null = [score(f * duration)[0] for f in fractions]
    valid = [v for v in null if v is not None]
    p95 = float(np.quantile(valid, .95)) if valid else None
    return {'contrast': actual, 'release_mean': low, 'reference_mean': context,
            'null_contrasts': null, 'null_p95': p95, 'shift_fractions': fractions.tolist(),
            'lift_over_null_p95': actual - p95 if actual is not None and p95 is not None else None,
            'denominator_floor': denominator_floor,
            'scope': 'Predeclared effort hypothesis; inferred audio release labels are not semantic gold.'}


def predictive_gain(target, prediction, baseline_prediction, *, mask=None):
    """D6 valid-sample development MSE gain over one explicitly named comparator.

    Positive = lower prediction error. Call separately for RMS-only and D2.
    Targets are motion-derived proxies, not semantic or artistic annotations.
    """
    y, p, b = [np.asarray(value, float) for value in (target, prediction, baseline_prediction)]
    if y.shape != p.shape or y.shape != b.shape or not y.ndim or not y.size:
        raise ValueError('Matching nonempty target/model/comparator prediction arrays required')
    valid = np.ones(y.shape, bool)
    if mask is not None:
        supplied = np.asarray(mask)
        if supplied.dtype != bool:
            raise ValueError('Valid mask must be boolean')
        if supplied.shape == y.shape[:1]:
            supplied = supplied.reshape((-1,) + (1,) * (y.ndim - 1))
        try:
            valid = np.broadcast_to(supplied, y.shape)
        except ValueError as exc:
            raise ValueError('Valid mask cannot broadcast to targets') from exc
    if not all(np.isfinite(value[valid]).all() for value in (y, p, b)):
        raise ValueError('Valid prediction/target observations must be finite')
    if not valid.any():
        return {'relative_mse_gain': None, 'prediction_mse': None, 'baseline_mse': None, 'valid_values': 0}
    mse, base = float(np.mean((p[valid] - y[valid]) ** 2)), float(np.mean((b[valid] - y[valid]) ** 2))
    return {'relative_mse_gain': (base - mse) / base if base > 1e-20 else None,
            'prediction_mse': mse, 'baseline_mse': base, 'valid_values': int(valid.sum()),
            'scope': 'Development predictive gain against motion proxies; no artistic or held-out-test claim.'}


def conditioning_gain(correct_loss, donor_loss, shifted_loss):
    """D7 smaller of matched correct-vs-donor/shift relative flow-loss gains.

    Arrays must refer to the same examples/noise/masks/checkpoint. This function
    validates shape/numerics; callers must authenticate those experimental facts.
    """
    arrays = [np.asarray(value, float) for value in (correct_loss, donor_loss, shifted_loss)]
    if any(x.shape != arrays[0].shape or not x.size or not np.isfinite(x).all() or np.any(x < 0) for x in arrays):
        raise ValueError('Matching finite nonnegative per-example losses required')
    c, d, s = [float(x.mean()) for x in arrays]
    donor = (d - c) / d if d > 1e-20 else None
    shift = (s - c) / s if s > 1e-20 else None
    return {'correct_loss': c, 'donor_loss': d, 'shifted_loss': s,
            'relative_donor_gain': donor, 'relative_shift_gain': shift,
            'minimum_relative_gain': min(donor, shift) if donor is not None and shift is not None else None,
            'matched_examples': int(arrays[0].size),
            'scope': 'Matched loss conditioning proxy; separately inspect generated body and emitted response.'}

"""Supplemental CPU audit of musical effects in requested and emitted motion.

Added after D1--D5 numerical outcomes. These are descriptive measurements, NEVER
new preregistered primaries, replacement verdicts, or independent semantic proof.
The module reads authenticated artifacts; only the separate CLI writes a new
exclusive report. It does not generate bodies, fit mappings, or emit hardware.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from . import campaign, contracts, diagnostics, evaluation
from .contracts import MusicTimeline
from lightsem.conductor import body_transfer

MUSICAL_PROTOCOL = {
    'event_match_tolerance_s': .12,
    'event_minimum_separation_s': .2,
    'event_null_shift_fractions': [.113, .197, .283, .379, .463, .571, .659, .743, .857],
    'event_speed_prominence_deg_s': 1.,
    'interval_s': [60, 180],
    'recurrence_minimum_pair_separation_s': 2.,
    'recurrence_null_shift_fractions': [.113, .283, .463, .659, .857],
    'recurrence_samples_max': 128,
    'source_selectivity_reference': 'loudness-matched sum-all-stems reconstruction versus same-source removal',
    'source_selectivity_roles': 'Static primary fixture role; declared D3 source-region assignment, blends and missing groups disclosed',
    'stages': ['body', 'requested motion', 'actual emitted decoded DMX'],
}
AUDIT_PARAMETERS = {
    **MUSICAL_PROTOCOL,
    'version': 'supplemental-musical-transfer-v1',
    'scope': 'Post-D1-D5 descriptive audit; original primaries/gates unchanged; no artistic verdict.',
    'speed_clock': 'Forward interval vector speed at interval start; final value repeats for frame-aligned display only.',
    'speed_pooling': 'Arithmetic mean of per-mover Euclidean pan/tilt speed; every mover also reported separately.',
    'recurrence_normalization': 'Divide pan/tilt degrees by each fixture mechanical pan/tilt range; no data-fitted scaling.',
    'recurrence_distance': 'Euclidean feature self-distance; Spearman versus original-song MuQ self-distance.',
    'source_selectivity_pooling': 'Fixture RMS within primary region, then equal observed regions; fixture-equal alternative also reported.',
    'quantization': 'Actual decoded angles reconstructed to 16-bit integer ticks; one-tick screening, repeat noise not assumed zero.',
    'body_event_prominence_m_s': .08,
    'release_denominator_floors': {'body_m_s': .01, 'motor_deg_s': 1.},
}


def _time(values, count, minimum=4):
    t = np.asarray(values, float)
    if t.shape != (count,) or len(t) < minimum or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Finite strictly increasing matching timestamps required')
    return t


def _interpolate(values, times, targets):
    x = np.asarray(values, float).reshape(len(times), -1)
    return np.stack([np.interp(targets, times, column) for column in x.T], axis=-1)


def feature_recurrence(values, times, music_features, music_times):
    """Generic recurrence, with the exact body-audit sampling/null recipe.

    Values are actual features, never angles reshaped into a fake skeleton.
    Constant/no-feature representations are unmeasurable rather than perfect.
    """
    from scipy.stats import spearmanr
    x, audio = np.asarray(values, float), np.asarray(music_features, float)
    if x.ndim < 2 or audio.ndim != 2:
        raise ValueError('Time by feature arrays required')
    t, at = _time(times, len(x)), _time(music_times, len(audio), minimum=2)
    if not np.isfinite(x).all() or not np.isfinite(audio).all():
        raise ValueError('Nonfinite recurrence inputs')
    if not x[0].size or not audio.shape[1]:
        return {'spearman': None, 'shifted_null': [], 'null_p95': None,
                'lift_over_null_p95': None, 'status': 'no_features'}
    first, last = max(t[0], at[0]), min(t[-1], at[-1])
    if last <= first:
        return {'spearman': None, 'shifted_null': [], 'null_p95': None,
                'lift_over_null_p95': None, 'status': 'no_clock_overlap'}
    q = np.linspace(first, last, min(MUSICAL_PROTOCOL['recurrence_samples_max'], len(t), len(at)))
    selected, acoustic = _interpolate(x, t, q), _interpolate(audio, at, q)
    i, j = np.triu_indices(len(q), 1)
    far = q[j] - q[i] >= MUSICAL_PROTOCOL['recurrence_minimum_pair_separation_s']
    i, j = i[far], j[far]
    def distances(features):
        # Bounded row batches also support many output channels on CPU.
        result = np.empty(len(i))
        for start in range(0, len(i), 512):
            a, b = i[start:start+512], j[start:start+512]
            result[start:start+512] = np.linalg.norm(features[a] - features[b], axis=1)
        return result
    physical = distances(selected)
    def score(features):
        measured = distances(features)
        if len(physical) < 3 or np.ptp(physical) <= 1e-12 or np.ptp(measured) <= 1e-12:
            return None
        return float(spearmanr(physical, measured).statistic)
    correct = score(acoustic)
    shifts = sorted(set(max(1, min(len(q)-1, round(f*len(q))))
                        for f in MUSICAL_PROTOCOL['recurrence_null_shift_fractions']))
    null = [score(np.roll(acoustic, amount, axis=0)) for amount in shifts]
    valid = [v for v in null if v is not None]
    p95 = float(np.quantile(valid, .95)) if valid else None
    return {'status': 'measured' if correct is not None else 'undefined_constant_or_insufficient_pairs',
            'spearman': correct, 'shifted_null': null, 'shift_frames': shifts, 'null_p95': p95,
            'lift_over_null_p95': correct - p95 if correct is not None and p95 is not None else None,
            'sampled_frames': len(q), 'compared_pairs': len(i),
            'scope': 'Acoustic/motion recurrence correspondence; not annotated motifs or a new primary score.'}


def fixture_role_map(patch):
    """Static primary-role routing; disclose every blended or unassigned fixture."""
    role_region = {joint: region for region, joints in evaluation.SOURCE_JOINTS.items() for joint in joints}
    rows, moving_index = [], 0
    for index, fixture in enumerate(patch):
        kind = fixture['kind']
        if kind not in ('beam', 'laser', 'bulb'):
            raise ValueError('Unknown fixture capability')
        role = fixture.get('role_index')
        role = int(role) if isinstance(role, (int, np.integer)) and not isinstance(role, bool) else None
        secondary = fixture.get('secondary_role')
        secondary = int(secondary) if isinstance(secondary, (int, np.integer)) and not isinstance(secondary, bool) else None
        blend = float(fixture.get('role_blend', 0.))
        if not np.isfinite(blend) or not 0 <= blend <= 1:
            raise ValueError('Invalid static fixture blend')
        mover = kind != 'bulb'
        rows.append({'patch_index': index, 'moving_index': moving_index if mover else None,
                     'name': fixture['name'], 'kind': kind,
                     'primary_role': role if mover else None,
                     'primary_region': role_region.get(role) if mover else None,
                     'secondary_role': secondary if mover and blend > 0 else None,
                     'secondary_region': role_region.get(secondary) if mover and blend > 0 else None,
                     'role_blend': blend if mover else None, 'blended': bool(mover and blend > 0),
                     'unassigned_reason': 'bulb brightness field has no motor role' if not mover else
                         'missing/unknown primary skeletal role' if role not in role_region else None,
                     'routing_caveat': 'Pelvis role also reads torso joint9 orientation' if mover and role == 0 else None})
        moving_index += int(mover)
    return rows


def _ranges(patch):
    values = np.asarray([[f['pan_range_deg'], f['tilt_range_deg']] for f in patch if f['kind'] != 'bulb'], float)
    values = values.reshape(-1, 2)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Positive finite mechanical ranges required')
    return values


def motor_speed(angles, times):
    a = np.asarray(angles, float)
    t = _time(times, len(a))
    if a.ndim != 3 or a.shape[-1] != 2 or not np.isfinite(a).all():
        raise ValueError('Finite [T,movers,2] motor angles required')
    v = np.linalg.norm(np.diff(a, axis=0) / np.diff(t)[:, None, None], axis=-1)
    return np.concatenate((v, v[-1:]), axis=0)


def motor_event_alignment(angles, times, events, *, start_s=60., end_s=180.):
    speed = motor_speed(angles, times)
    if events is None or not speed.shape[1]:
        return {'pooled': None, 'per_fixture': [], 'status': 'missing_audio_events_or_movers'}
    def metric(trace):
        # The reused detector is unit-agnostic; explicitly replace its m/s default.
        peaks = diagnostics.gesture_events(trace, times,
            prominence_m_s=MUSICAL_PROTOCOL['event_speed_prominence_deg_s'],
            minimum_separation_s=MUSICAL_PROTOCOL['event_minimum_separation_s'])
        return {**diagnostics.event_alignment(peaks, events, start_s=start_s, end_s=end_s,
                    tolerance_s=MUSICAL_PROTOCOL['event_match_tolerance_s'],
                    shift_fractions=MUSICAL_PROTOCOL['event_null_shift_fractions']),
                'gesture_times_s': peaks.tolist()}
    return {'status': 'measured', 'pooled': metric(speed.mean(axis=1)),
            'per_fixture': [metric(speed[:, index]) for index in range(speed.shape[1])],
            'prominence_deg_s': MUSICAL_PROTOCOL['event_speed_prominence_deg_s'],
            'scope': 'Fixed motor-speed peak proxy; event precision penalizes busy jitter.'}


def regional_motor_selectivity(reference, removed, patch, assigned_regions, *, decoded):
    """Primary-region response, separate from global movement and quantization.

    Return all-primary and unblended-only views. Unknown-role movers and bulbs
    never become complement evidence. Primary roles are routing, not exclusive
    causal attributions: mapper01 also depends on root/parent bone geometry.
    """
    a, b = np.asarray(reference, float), np.asarray(removed, float)
    ranges = _ranges(patch)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != ranges.shape or not len(a) or not np.isfinite([a, b]).all():
        raise ValueError('Matching finite [T,movers,2] reference/removal angles required')
    assigned = tuple(assigned_regions)
    if not assigned or not set(assigned) <= evaluation.SOURCE_JOINTS.keys():
        raise ValueError('Known declared D3 assigned regions required')
    delta = b - a
    tick_delta = None
    if decoded:
        # Recover actual integer commands; no floating error changes one-tick tests.
        original_ticks = np.rint((.5 + a/ranges) * 65535).astype(np.int64)
        removed_ticks = np.rint((.5 + b/ranges) * 65535).astype(np.int64)
        tick_delta = removed_ticks - original_ticks
        delta = tick_delta * (ranges / 65535.)
    rms = np.sqrt(np.mean(delta ** 2, axis=(0, 2)))
    normalized = np.sqrt(np.mean((delta / ranges) ** 2, axis=(0, 2)))
    tick_rms = np.sqrt(np.mean(tick_delta.astype(float) ** 2, axis=(0, 2))) if decoded else None
    roles = fixture_role_map(patch)
    per_fixture = []
    for role in roles:
        index = role['moving_index']
        if index is None:
            continue
        per_fixture.append({**role, 'rms_difference_deg': float(rms[index]),
            'range_normalized_rms_difference': float(normalized[index]),
            'rms_difference_ticks': float(tick_rms[index]) if decoded else None,
            'above_one_tick_rms': bool(tick_rms[index] >= 1.) if decoded else None,
            'max_abs_command_tick_difference': int(np.abs(tick_delta[:, index]).max()) if decoded else None,
            'fraction_frames_any_changed_command': float(np.any(tick_delta[:, index] != 0, axis=-1).mean()) if decoded else None})
    def summarize(unblended):
        eligible = [r for r in per_fixture if r['primary_region'] is not None and (not unblended or not r['blended'])]
        grouped = {name: [r for r in eligible if r['primary_region'] == name] for name in evaluation.SOURCE_JOINTS}
        means = {name: float(np.mean([r['rms_difference_deg'] for r in rows])) if rows else None for name, rows in grouped.items()}
        missing_assigned = [name for name in assigned if not grouped[name]]
        missing_complement = [name for name in grouped if name not in assigned and not grouped[name]]
        selected = [means[name] for name in assigned if means[name] is not None]
        other = [value for name, value in means.items() if name not in assigned and value is not None]
        left = float(np.mean(selected)) if selected and not missing_assigned else None
        right = float(np.mean(other)) if other else None
        assigned_fixtures = [r for r in eligible if r['primary_region'] in assigned]
        complement_fixtures = [r for r in eligible if r['primary_region'] not in assigned]
        fixture_a = float(np.mean([r['rms_difference_deg'] for r in assigned_fixtures])) if assigned_fixtures and not missing_assigned else None
        fixture_b = float(np.mean([r['rms_difference_deg'] for r in complement_fixtures])) if complement_fixtures else None
        return {'regional_mean_rms_deg': means, 'assigned_mean_rms_deg': left,
            'complement_mean_rms_deg': right, 'assigned_minus_complement_deg': left-right if left is not None and right is not None else None,
            'fixture_equal_assigned_minus_complement_deg': fixture_a-fixture_b if fixture_a is not None and fixture_b is not None else None,
            'missing_assigned_regions': missing_assigned, 'missing_complement_regions': missing_complement,
            'complete_region_coverage': not missing_assigned and not missing_complement,
            'assigned_fixture_names': [r['name'] for r in assigned_fixtures],
            'complement_fixture_names': [r['name'] for r in complement_fixtures],
            'assigned_above_one_tick_count': sum(r['above_one_tick_rms'] is True for r in assigned_fixtures) if decoded else None,
            'complement_above_one_tick_count': sum(r['above_one_tick_rms'] is True for r in complement_fixtures) if decoded else None}
    return {'assigned_regions': list(assigned), 'per_fixture': per_fixture,
            'all_primary_roles': summarize(False), 'unblended_only': summarize(True),
            'blended_fixture_names': [r['name'] for r in per_fixture if r['blended']],
            'unassigned_fixture_names': [r['name'] for r in roles if r['primary_region'] is None],
            'decoded_integer_command_differences': bool(decoded),
            'scope': 'Descriptive static-role selectivity; blends/root/bone parents prevent exclusive causal attribution.',
            'noise_scope': 'One-tick screening only; no repeated-run noise estimate or new pass/fail gate.'}


def _subtract(first, second):
    return float(first-second) if first is not None and second is not None else None


def _stage_releases(body_trace, body_time, requested_speed, emitted_speed, time, release):
    def measured(values, clock, floor):
        if not release or not release.get('release_intervals') or not release.get('reference_intervals'):
            return None
        intervals = release['release_intervals'] + release['reference_intervals']
        end = clock[-1] + float(np.median(np.diff(clock)))
        if any(a < clock[0] or b > end+1e-8 for a, b in intervals):
            return None
        return diagnostics.release_contrast(values, clock, release['release_intervals'], release['reference_intervals'], denominator_floor=floor)
    return {'body': measured(body_trace, body_time, .01),
            'requested': measured(requested_speed, time, 1.) if requested_speed is not None else None,
            'emitted': measured(emitted_speed, time, 1.) if emitted_speed is not None else None}


def measure_stages(pair, music, events, release):
    """Measure the same original-song reference at every processing stage."""
    mask, body_mask = pair['window'], pair['body_window']
    t = pair['arrays']['time'][mask]
    bt = pair['body'].time[body_mask]
    body = pair['body'].world[body_mask]
    requested = pair['requested'][mask]
    emitted = pair['controls']['angles_deg'][mask][:, pair['moving']]
    series = diagnostics.dynamics_series(body, bt)
    body_events = None
    if events is not None:
        peaks = diagnostics.gesture_events(series['relative_speed_m_s'], bt,
                    prominence_m_s=AUDIT_PARAMETERS['body_event_prominence_m_s'],
                    minimum_separation_s=MUSICAL_PROTOCOL['event_minimum_separation_s'])
        body_events = diagnostics.event_alignment(peaks, events, start_s=60., end_s=180.,
                    tolerance_s=MUSICAL_PROTOCOL['event_match_tolerance_s'],
                    shift_fractions=MUSICAL_PROTOCOL['event_null_shift_fractions'])
    ranges = _ranges(pair['patch'])
    req_events = motor_event_alignment(requested, t, events)
    em_events = motor_event_alignment(emitted, t, events)
    recurrences = {
        'body': diagnostics.recurrence_alignment(body, bt, music.muq, music.muq_time),
        'requested': feature_recurrence(requested/ranges, t, music.muq, music.muq_time),
        'emitted': feature_recurrence(emitted/ranges, t, music.muq, music.muq_time)}
    req_speed, em_speed = motor_speed(requested, t), motor_speed(emitted, t)
    releases = _stage_releases(series['relative_speed_m_s'], bt,
        req_speed.mean(1) if req_speed.shape[1] else None,
        em_speed.mean(1) if em_speed.shape[1] else None, t, release)
    for row in (req_events, em_events):
        for metric, fixture in zip(row['per_fixture'], [p for p in pair['patch'] if p['kind'] != 'bulb']):
            metric['fixture_name'] = fixture['name']
    event_metrics = {'body': body_events, 'requested': req_events, 'emitted': em_events}
    return {'event_alignment': event_metrics, 'acoustic_recurrence': recurrences,
            'release_contrast': releases,
            'trace_attenuation': {
                'event_f1_emitted_minus_requested': _subtract(evaluation._nested(em_events, 'pooled', 'f1'), evaluation._nested(req_events, 'pooled', 'f1')),
                'event_null_lift_emitted_minus_requested': _subtract(evaluation._nested(em_events, 'pooled', 'f1_minus_null_p95'), evaluation._nested(req_events, 'pooled', 'f1_minus_null_p95')),
                'recurrence_spearman_emitted_minus_requested': _subtract(recurrences['emitted']['spearman'], recurrences['requested']['spearman']),
                'recurrence_null_lift_emitted_minus_requested': _subtract(recurrences['emitted']['lift_over_null_p95'], recurrences['requested']['lift_over_null_p95']),
                'release_contrast_emitted_minus_requested': _subtract(evaluation._nested(releases['emitted'], 'contrast'), evaluation._nested(releases['requested'], 'contrast'))},
            'motor_activity': {'requested_mean_speed_deg_s': float(req_speed.mean()) if req_speed.size else None,
                               'emitted_mean_speed_deg_s': float(em_speed.mean()) if em_speed.size else None,
                               'exactly_constant_emitted': bool(np.all(emitted == emitted[:1]))},
            'requested_pan_branch_jumps_full_prefix': pair['requested_branch_jumps'],
            'scope': 'Requested angles retain compiler atan2 branch; no phase/lag optimization or new motion generation.'}


def _condition_delta(candidate, comparator):
    result = {}
    for metric, field in (('event_alignment', 'f1'), ('acoustic_recurrence', 'spearman'), ('release_contrast', 'contrast')):
        result[metric] = {}
        for stage in ('body', 'requested', 'emitted'):
            path = (metric, stage, 'pooled', field) if metric == 'event_alignment' and stage != 'body' else (metric, stage, field)
            result[metric][stage] = _subtract(evaluation._nested(candidate, *path), evaluation._nested(comparator, *path))
    result['scope'] = 'Candidate minus this comparator; separate from agreement with automatic labels and shifted-null lift.'
    return result


def _protocol(root, pins):
    path = root / 'records/SUPPLEMENTAL_DELIVERY_AUDITS_PROTOCOL.json'
    if not path.exists():
        raise ValueError('Supplemental parameters must be frozen before execution')
    value = campaign.read(path)
    actual = value.get('musical_transfer', {})
    if any(actual.get(key) != expected for key, expected in MUSICAL_PROTOCOL.items()):
        raise ValueError('Supplemental musical-transfer parameters differ from frozen recipe')
    if (value.get('status') != 'frozen_before_execution' or value.get('original_plan_unchanged') is not True
            or value.get('original_primaries_and_gates_unchanged') is not True):
        raise ValueError('Supplemental audit cannot alter original plan/primaries/gates')
    evaluation._pin(pins, path)
    return value


def _code_pins(pins):
    for module in (campaign, contracts, diagnostics, evaluation, body_transfer):
        evaluation._pin(pins, Path(module.__file__))
    evaluation._pin(pins, Path(__file__))
    evaluation._pin(pins, Path(__file__).parents[3]/'scripts/motion/audit_dynamic_dance_transfer.py')


def _comparison_inputs(candidate, comparator):
    """Matched cross-method inputs; checkpoints intentionally differ by method."""
    a, b = candidate['body'].metadata, comparator['body'].metadata
    if a.get('arm') == 'D6':
        declared = b.get('campaign_inputs', {})
        values = {'source_body_sha256': (a.get('source_body_sha256'), declared.get('source_body_sha256')),
                  'timeline_sha256': (a.get('timeline_sha256'), declared.get('music_npz_sha256'))}
    else:
        fields = ('seed', 'draw_seed', 'feature_schema_sha256', 'timeline_sha256', 'base92_sha256')
        values = {key: (a.get(key), b.get(key)) for key in fields}
    checks = {key: left == right if left is not None and right is not None else None
              for key, (left, right) in values.items()}
    return {'checks': checks, 'status': 'pass' if all(value is True for value in checks.values()) else 'unqualified',
            'scope': 'Matched registered cross-method inputs; a different checkpoint/mechanism is intentional.'}


def audit_pair(root, method, song):
    """Read one logical method and every available relevant comparison; no writes."""
    root = Path(root)
    if method not in campaign.METHODS:
        raise ValueError('Expected logical D1..D8 method')
    pins = {}
    protocol = _protocol(root, pins)
    _code_pins(pins)
    plan = campaign.read(root/'PLAN.json')
    if plan.get('review_start_s') != 60. or plan.get('review_duration_s') != 120.:
        raise ValueError('Supplemental audit is fixed to song interval60-180s')
    track = evaluation._track(plan, song)
    evaluation._pin(pins, root/'PLAN.json')
    artifact_id = campaign.resolve_variant(root, method)
    pins.update(campaign.variant_binding_pins(root))
    identifiers = {'baseline': 'baseline', 'candidate': artifact_id}
    expected = set(evaluation.EXPECTED_CONTROLS[method])
    expected |= {p.name.split('__', 1)[1] for p in (root/'bodies').glob(artifact_id+'__*') if p.is_dir()}
    identifiers.update({name: artifact_id+'__'+name for name in sorted(expected)})
    for comparator in {'D6': ('D2',), 'D7': ('continued',), 'D8': ('D7', 'D8_flat')}.get(method, ()):
        identifiers['comparator_'+comparator] = campaign.resolve_variant(root, comparator)
    pairs, missing = {}, {}
    for label, identifier in identifiers.items():
        pair, absent = evaluation._load_pair(root, identifier, song, plan, pins)
        if pair is None:
            missing[label] = absent
        else:
            pairs[label] = pair
    condition_aliases = {}
    if 'shuffled_plan' not in pairs and 'shifted_plan' in pairs:
        pairs['shuffled_plan'] = pairs.pop('shifted_plan')
        missing.pop('shuffled_plan', None)
        condition_aliases['shuffled_plan'] = 'shifted_plan artifact; explicit same-audio plan corruption'
    base = pairs.get('baseline')
    candidate = pairs.get('candidate')
    result = {'schema_version': 1, 'method': method, 'artifact_id': artifact_id, 'song': song,
              'status': 'supplemental_descriptive_only', 'window_s': [60, 180],
              'parameters': AUDIT_PARAMETERS, 'inputs': pins, 'missing_conditions': missing, 'condition_aliases': condition_aliases,
              'development_scope': protocol.get('development_scope'),
              'original_primaries_and_gates_changed': False, 'scientific_verdict': 'not_assigned',
              'independent_semantic_evidence': False, 'human_artistic_acceptance': 'pending_user_review'}
    if base is None or candidate is None:
        result.update(status='missing_required_outputs', conditions={}, control_superiority={}, source_selectivity={})
        return result
    music_path = evaluation._music_path(root, track)
    music = MusicTimeline.load(music_path)
    for path in (music_path, music_path.with_suffix('.json'), music_path.with_suffix('.json.sha256.json')):
        evaluation._pin(pins, path)
    event_path = root/'records/audio_intervals'/(song+'.json')
    events = None
    if event_path.exists():
        event_record = campaign.read(event_path)
        if event_record.get('music_sha256') != campaign.sha(music_path):
            raise ValueError('Audio-only event reference differs from original song')
        events = event_record.get('attack_times')
        evaluation._pin(pins, event_path)
    release_path = event_path.with_name(song+'.release.json')
    release = evaluation._auth_json(release_path) if release_path.exists() else None
    if release:
        expected_release = {'music_sha256': campaign.sha(music_path),
            'music_metadata_sha256': campaign.sha(music_path.with_suffix('.json')), 'plan_sha256': campaign.sha(root/'PLAN.json')}
        if any(release.get('inputs', {}).get(key) != value for key, value in expected_release.items()):
            raise ValueError('Frozen release reference changed')
        evaluation._pin(pins, release_path); evaluation._pin(pins, str(release_path)+'.sha256.json')
    conditions, provenance = {}, {}
    for label, pair in pairs.items():
        evaluation._same_clock_and_patch(base, pair)
        fixed_p2 = bool(np.array_equal(base['controls']['rgb'], pair['controls']['rgb']))
        conditions[label] = {'artifact_id': pair['id'], **measure_stages(pair, music, events, release)}
        provenance[label] = {'P2_RGB_exact_to_original_baseline': fixed_p2,
            'fixed_patch_and_clock': True,
            'same_pipeline': evaluation._provenance(candidate, pair) if label not in ('baseline', 'candidate') and not label.startswith('comparator_') else None,
            'generation_input_match': _comparison_inputs(candidate, pair) if label.startswith('comparator_') else None}
    result.update(conditions=conditions, condition_provenance=provenance,
        original_song_reference={'events': str(event_path) if events is not None else None, 'music': str(music_path),
            'scope': 'All controls scored against the original song, so wrong-audio controls test fit to that song.'},
        control_superiority={label: _condition_delta(conditions['candidate'], values)
                             for label, values in conditions.items() if label != 'candidate'},
        fixture_roles=fixture_role_map(candidate['patch']),
        P2_all_conditions_exact=all(row['P2_RGB_exact_to_original_baseline'] for row in provenance.values()))
    if not result['P2_all_conditions_exact']:
        result['status'] = 'invalid_fixed_P2_invariant'
    cf_path = root/'records/COUNTERFACTUAL_PROTOCOL_v2.json'
    cf_protocol = campaign.read(cf_path) if cf_path.exists() else None
    evaluation._pin(pins, cf_path)
    result['source_selectivity'] = {}
    if method == 'D3':
        for name, assigned in evaluation.SOURCE_REGIONS.items():
            removed = pairs.get(name)
            rows = {}
            for label, reference in (('original_reference', candidate), ('reconstruction_reference', pairs.get('reconstruction'))):
                if removed is None or reference is None:
                    rows[label] = None
                    continue
                mask = candidate['window']
                requested = regional_motor_selectivity(reference['requested'][mask], removed['requested'][mask], candidate['patch'], assigned, decoded=False)
                emitted = regional_motor_selectivity(reference['controls']['angles_deg'][mask][:, candidate['moving']],
                    removed['controls']['angles_deg'][mask][:, candidate['moving']], candidate['patch'], assigned, decoded=True)
                evidence = (evaluation._reconstruction_reference(reference, removed, name, music, cf_protocol, pins)
                            if label == 'reconstruction_reference' else evaluation._sensitivity(reference, removed, name))
                rows[label] = {'requested': requested, 'emitted': emitted,
                    'body_selectivity_m': evidence['body']['assigned_minus_unassigned_m'],
                    'fixed_input_provenance': evidence['fixed_input_provenance'],
                    'removal_reference_provenance': evidence.get('removal_reference_provenance'),
                    'P2_RGB_exact': evidence['P2_RGB_exact'],
                    'selectivity_emitted_minus_requested_deg': _subtract(
                        emitted['all_primary_roles']['assigned_minus_complement_deg'], requested['all_primary_roles']['assigned_minus_complement_deg'])}
            result['source_selectivity'][name] = rows
    # Pin and reference, but never overwrite or promote existing primary decisions.
    comparison_path = root/'timelines'/artifact_id/song/'comparison.json'
    if comparison_path.exists():
        comparison = campaign.read(comparison_path)
        evaluation._pin(pins, comparison_path)
        result['original_primary_reference'] = {'path': str(comparison_path), 'primary': comparison.get('primary'),
            'method_postgate': comparison.get('method_postgate'),
            'scope': 'Historical original decision copied for context; never recomputed or replaced by this audit.'}
    return result

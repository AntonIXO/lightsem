"""CPU-only comparison of authenticated dynamic-dance experiment artifacts.

This module never fits a model, opens TEST data, or selects a checkpoint/seed.
Derived reports are revisioned separately from immutable mapping reports.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from lightsem.conductor.body_transfer import decode
from .campaign import METHODS, array_sha, read, sha, write_new
from .contracts import DanceResult, MusicTimeline
from .diagnostics import (FEET, REGIONS, body_dynamics, comparison_gates,
    dynamics_series, emission_diagnostics, emitted_response, event_alignment,
    gesture_events, recurrence_alignment, release_contrast, selective_response,
    tempo_aligned_comparison)

PRIMARY = {
    'D1': ('dd-event-alignment-lift', .02),
    'D2': ('dd-phrase-release-lift', .02),
    'D3': ('dd-source-regional-selectivity', .005),
    'D4': ('dd-motif-recurrence-lift', .02),
    'D5': ('dd-emitted-release-lift', .02),
    'D6': ('dd-planner-predictive-gain', .05),
    'D7': ('dd-native-conditioning-gain', .05),
    'D8': ('dd-hierarchical-recurrence-lift', .02),
}
WAVEFORM_CONTROLS = ('gain', 'shift', 'mute_drums', 'mute_vocals', 'donor', 'reconstruction')
EXPECTED_CONTROLS = {
    method: WAVEFORM_CONTROLS + (('rms_only', 'shifted_schedule') if method in ('D1', 'D2', 'D3', 'D4')
                                else ('shuffled_plan',) if method == 'D8' else ())
    for method in METHODS
}
# Assignment is a declared D3 intervention hypothesis, not a biological law.
SOURCE_REGIONS = {'mute_drums': ('support',), 'mute_vocals': ('head', 'left_arm')}
SOURCE_JOINTS = {'support': (0, 1, 2, 4, 5, 7, 8, 10, 11),
                 'torso': (3, 6, 9), 'head': (12, 15),
                 'left_arm': (13, 16, 18, 20), 'right_arm': (14, 17, 19, 21)}
RELEASE_FLOORS = {'body_m_s': .01, 'motor_deg_s': 1.}
BODY_CLOCK_ROUNDOFF_S = 1e-8


def _body_review_mask(times, start, end):
    """Keep the intended native frames at boundaries despite float64 roundoff."""
    times = np.asarray(times)
    return (times >= start - BODY_CLOCK_ROUNDOFF_S) & (times < end - BODY_CLOCK_ROUNDOFF_S)


def _track(plan, song):
    rows = [r for r in plan['tracks'] if r['id'] == song]
    if len(rows) != 1:
        raise ValueError('Unknown/ambiguous campaign song: ' + song)
    return rows[0]


def _music_path(root, track):
    return Path(root) / 'features/timelines' / (track['id'] + '.npz')


def _auth_json(path):
    path = Path(path)
    seal = Path(str(path) + '.sha256.json')
    if not seal.exists() or read(seal).get('sha256') != sha(path):
        raise ValueError('Missing or changed authenticated evaluation JSON: ' + str(path))
    return read(path)


def _pin(pins, path):
    path = Path(path)
    if path.exists():
        pins[str(path)] = sha(path)


def freeze_release_intervals(root, song):
    """Freeze an audio-only release hypothesis before inspecting candidate bodies.

    Uses exactly audio.preselect_intervals: an eligible texture-thinning or
    functional-boundary 8 s window and the preceding non-overlapping 8 s context.
    No qualifying window means no release reference, not a fabricated interval.
    """
    from .audio import preselect_intervals
    root = Path(root)
    plan = read(root / 'PLAN.json')
    track = _track(plan, song)
    path = root / 'records/audio_intervals' / (song + '.release.json')
    music_path = _music_path(root, track)
    inputs = {'music_sha256': sha(music_path), 'music_metadata_sha256': sha(music_path.with_suffix('.json')),
              'plan_sha256': sha(root / 'PLAN.json'), 'selection': 'audio.preselect_intervals/v1'}
    if path.exists():
        existing = _auth_json(path)
        if existing['inputs'] != inputs:
            raise ValueError('Frozen release reference input changed')
        return existing
    music = MusicTimeline.load(music_path)
    start = float(plan['review_start_s'])
    end = start + float(plan['review_duration_s'])
    selected = preselect_intervals(music, start, end)
    release = [r for r in selected if r.get('role') == 'release_hypothesis']
    report = {'inputs': inputs, 'song': song, 'selected_audio_intervals': selected,
              'release_intervals': [[r['start'], r['end']] for r in release],
              'reference_intervals': [[r['reference_start'], r['reference_end']] for r in release],
              'human_annotated': False, 'independent_semantic_reference': False,
              'denominator_floors': RELEASE_FLOORS,
              'scope': 'Frozen automatic audio hypothesis, not independent semantic or artistic truth.'}
    write_new(path, report)
    write_new(str(path) + '.sha256.json', {'sha256': sha(path)})
    return report


def _load_pair(root, identifier, song, plan, pins):
    body_path = root / 'bodies' / identifier / song / 'body.npz'
    folder = root / 'timelines' / identifier / song
    required = [body_path, folder / 'timeline.npz', folder / 'patch.json', folder / 'report.json']
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        return None, missing
    body = DanceResult.load(body_path)
    report = read(folder / 'report.json')
    for key, file in (('timeline_sha256', 'timeline.npz'), ('patch_sha256', 'patch.json')):
        if report.get(key) != sha(folder / file):
            raise ValueError('Mapped artifact changed: ' + str(folder / file))
    if report.get('body_sha256') != sha(body_path):
        raise ValueError('Mapped body does not match authenticated source: ' + identifier)
    for path in required + [body_path.with_suffix('.json'), body_path.with_suffix('.json.sha256.json')]:
        _pin(pins, path)
    with np.load(folder / 'timeline.npz', allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in ('time', 'body', 'universes', 'targets')}
    patch = read(folder / 'patch.json')
    controls = decode(arrays['universes'], patch)
    n = len(arrays['time'])
    if arrays['body'].shape != (n, 22, 3) or arrays['targets'].shape != (n, len(patch), 3):
        raise ValueError('Mapped timeline shape mismatch: ' + identifier)
    if not all(np.isfinite(v).all() for v in arrays.values()) or np.any(np.diff(arrays['time']) <= 0):
        raise ValueError('Invalid mapped timeline: ' + identifier)
    moving = np.array([p['kind'] != 'bulb' for p in patch])
    positions = np.asarray([p['position'] for p in patch], float)[moving]
    vectors = arrays['targets'][:, moving] - positions
    length = np.linalg.norm(vectors, axis=-1)
    if np.any(length <= 1e-12):
        raise ValueError('Requested target coincides with moving fixture')
    requested = np.stack((np.rad2deg(np.arctan2(vectors[..., 0], -vectors[..., 1])),
                          np.rad2deg(np.arcsin(np.clip(vectors[..., 2] / length, -1, 1)))), axis=-1)
    ticks = [max(p['pan_range_deg'], p['tilt_range_deg']) / 65535. for p in patch if p['kind'] != 'bulb']
    tick = max(ticks) if ticks else None
    start = float(plan['review_start_s'])
    end = start + float(plan['review_duration_s'])
    emitted_window = (arrays['time'] >= start) & (arrays['time'] < end)
    body_window = _body_review_mask(body.time, start, end)
    if min(emitted_window.sum(), body_window.sum()) < 4:
        raise ValueError('Insufficient review-window observations: ' + identifier)
    return {'id': identifier, 'body': body, 'arrays': arrays, 'patch': patch, 'controls': controls,
            'requested': requested, 'requested_branch_jumps': int(np.sum(np.abs(np.diff(requested[..., 0], axis=0)) > 180.)), 'moving': moving, 'tick': tick, 'report': report,
            'window': emitted_window, 'body_window': body_window}, []


def _same_clock_and_patch(a, b):
    if a['patch'] != b['patch'] or not np.array_equal(a['arrays']['time'], b['arrays']['time']):
        raise ValueError('Controlled comparison changed fixture patch or emitted timestamps')
    at, bt = np.asarray(a['body'].time), np.asarray(b['body'].time)
    exact = np.array_equal(at, bt)
    difference = float(np.max(np.abs(at - bt))) if at.shape == bt.shape and at.size else None
    if not exact:
        nominal = np.arange(len(at)) / 30.
        if (at.shape != bt.shape or at.ndim != 1 or not len(at)
                or not np.isfinite(at).all() or not np.isfinite(bt).all()
                or not np.allclose(at, nominal, rtol=0., atol=BODY_CLOCK_ROUNDOFF_S)
                or not np.allclose(bt, nominal, rtol=0., atol=BODY_CLOCK_ROUNDOFF_S)
                or difference > BODY_CLOCK_ROUNDOFF_S):
            raise ValueError('Controlled comparison changed body timestamps')
    return {'native_body_clocks_exact': bool(exact), 'maximum_native_clock_difference_s': difference,
            'admitted_native_clock_roundoff_s': BODY_CLOCK_ROUNDOFF_S,
            'emitted_clock_exact': True,
            'policy': 'Only nominal native30Hz roundoff is admitted; saved timestamps are retained. Emitted clocks remain exact.'}


def _speed_trace(angles, times):
    speed = np.linalg.norm(np.diff(angles, axis=0) / np.diff(times)[:, None, None], axis=-1)
    values = speed.mean(axis=1) if speed.shape[1] else np.zeros(len(times) - 1)
    return np.r_[values, values[-1]]


def _scores(pair, baseline, music, selection, release, plan):
    body, mask = pair['body'], pair['body_window']
    original = baseline['body']
    floor = float(np.quantile(original.world[:, FEET, 1], .02))
    physical = body_dynamics(body.world[mask], body.time[mask], contacts=body.contacts[mask],
                             reference_body=original.world, floor_y=floor)
    series = dynamics_series(body.world[mask], body.time[mask])
    t = pair['arrays']['time'][pair['window']]
    requested = pair['requested'][pair['window']]
    emitted = pair['controls']['angles_deg'][pair['window']][:, pair['moving']]
    transfer = plan.get('transfer', {})
    emission = emission_diagnostics(requested, emitted, t,
        maximum_speed_deg_s=float(transfer.get('maximum_speed_deg_s', 90.)),
        maximum_acceleration_deg_s2=float(transfer.get('maximum_acceleration_deg_s2', 360.)),
        tick_deg=pair['tick'] or 540. / 65535.)
    attack_times = selection.get('attack_times') if selection else None
    start = float(plan['review_start_s'])
    end = start + float(plan['review_duration_s'])
    event = event_alignment(gesture_events(series['relative_speed_m_s'], body.time[mask]), attack_times,
                            start_s=start, end_s=end) if attack_times is not None else None
    recurrence = recurrence_alignment(body.world[mask], body.time[mask], music.muq, music.muq_time)
    def contrast(values, clock, unit):
        if not release or not release.get('release_intervals') or not release.get('reference_intervals'):
            return None
        intervals = release['release_intervals'] + release['reference_intervals']
        if any(a < clock[0] or b > clock[-1] + float(np.median(np.diff(clock))) + 1e-8 for a, b in intervals):
            return None
        return release_contrast(values, clock, release['release_intervals'], release['reference_intervals'],
                                denominator_floor=RELEASE_FLOORS[unit])
    return {'body': physical, 'emission': emission, 'event_alignment': event,
            'body_release': contrast(series['relative_speed_m_s'], body.time[mask], 'body_m_s'),
            'requested_release': contrast(_speed_trace(requested, t), t, 'motor_deg_s'),
            'emitted_release': contrast(_speed_trace(emitted, t), t, 'motor_deg_s'),
            'recurrence': recurrence, 'requested_pan_branch_jumps_full_prefix': pair['requested_branch_jumps'],
            'requested_branch_policy': 'Exact compiler atan2 branch; discontinuities are retained because mechanics receives them.'}


def _nested(report, *keys):
    for key in keys:
        if not isinstance(report, dict) or key not in report:
            return None
        report = report[key]
    return report


def _provenance(a, b):
    """Audit fixed model/noise inputs; never infer changed audio from body hashes."""
    am, bm = a['body'].metadata, b['body'].metadata
    learned = 'checkpoint_sha256' in am
    if learned:
        fields = ('checkpoint_sha256', 'seed', 'draw_seed', 'feature_schema_sha256')
        checks = {field: am.get(field) == bm.get(field) if field in am and field in bm else None for field in fields}
        if am.get('arm') == 'D6':
            checks['same_source_body'] = (am.get('source_body_sha256') == bm.get('source_body_sha256')
                if am.get('source_body_sha256') and bm.get('source_body_sha256') else None)
        if am.get('arm') in ('D7', 'D8'):
            checks['base92_recomputed'] = (am.get('base92_sha256') != bm.get('base92_sha256')
                if am.get('base92_sha256') and bm.get('base92_sha256') and bm.get('plan_override') is None else None)
        checks['changed_audio_timeline'] = (am['timeline_sha256'] != bm['timeline_sha256']
            if am.get('timeline_sha256') and bm.get('timeline_sha256') else None)
        if bm.get('plan_override') is not None:
            checks.pop('changed_audio_timeline', None)
            checks.pop('base92_recomputed', None)
            checks['same_audio_for_plan_intervention'] = (am.get('timeline_sha256') == bm.get('timeline_sha256')
                if am.get('timeline_sha256') and bm.get('timeline_sha256') else None)
            checks['same_base92_for_plan_intervention'] = (am.get('base92_sha256') == bm.get('base92_sha256')
                if am.get('base92_sha256') and bm.get('base92_sha256') else None)
    else:
        ac, bc = am.get('campaign_inputs', {}), bm.get('campaign_inputs', {})
        checks = {'same_source_body': ac.get('source_body_sha256') == bc.get('source_body_sha256')
                  if ac.get('source_body_sha256') and bc.get('source_body_sha256') else None,
                  'same_method': ac.get('method') == bc.get('method') if ac.get('method') and bc.get('method') else None,
                  'changed_audio_timeline': ac.get('music_npz_sha256') != bc.get('music_npz_sha256')
                  if ac.get('music_npz_sha256') and bc.get('music_npz_sha256') else None}
    return {'checks': checks, 'status': 'pass' if checks and all(v is True for v in checks.values()) else 'unqualified',
            'scope': 'Artifact identities of fixed inputs; musical usefulness measured separately.'}


def _sensitivity(candidate, other, name):
    _same_clock_and_patch(candidate, other)
    assigned = tuple(REGIONS)
    mask = candidate['body_window']
    body = selective_response(candidate['body'].world[mask], other['body'].world[mask], assigned_regions=assigned)
    if name in SOURCE_REGIONS:
        a, b = candidate['body'].world[mask], other['body'].world[mask]
        delta = (a - a[:, :1]) - (b - b[:, :1])
        regions = {key: float(np.sqrt(np.mean(delta[:, joints] ** 2))) for key, joints in SOURCE_JOINTS.items()}
        assigned = SOURCE_REGIONS[name]
        intended = float(np.mean([regions[r] for r in assigned]))
        unintended = float(np.mean([regions[r] for r in SOURCE_JOINTS if r not in assigned]))
        body.update(regional_rms_difference_m=regions, assigned_mean_rms_m=intended,
                    unassigned_mean_rms_m=unintended, assigned_minus_unassigned_m=intended - unintended,
                    assigned_regions=list(assigned), assigned_response_above_noise=intended > 1e-5,
                    scope='Exact registered D3 source-to-region hypothesis; locality is not a biological law.')
    wm = candidate['window']
    a = candidate['controls']['angles_deg'][wm][:, candidate['moving']]
    b = other['controls']['angles_deg'][wm][:, other['moving']]
    emitted = emitted_response(a, b, tick_deg=candidate['tick']) if a.size else None
    return {'body': body, 'emitted': emitted, 'fixed_input_provenance': _provenance(candidate, other),
            'noise_note': 'Quantization-floor screening only; repeated-run noise is unavailable unless separately supplied.',
            'P2_RGB_exact': bool(np.array_equal(candidate['controls']['rgb'], other['controls']['rgb'])),
            'exact_body_duplicate': bool(np.array_equal(candidate['body'].world, other['body'].world)),
            'exact_emitted_motion_duplicate': bool(np.array_equal(a, b)),
            'exact_full_visual_payload_duplicate': bool(np.array_equal(candidate['body'].world, other['body'].world)
                and np.array_equal(candidate['arrays']['universes'], other['arrays']['universes']))}


def _method_postgate(pair):
    """Expose an already computed method gate without changing its decision."""
    metadata = pair['body'].metadata
    passed = metadata.get('postgate_passed')
    if passed is not None and not isinstance(passed, bool):
        raise ValueError('Declared method postgate must be boolean')
    failures = metadata.get('infeasibility', [])
    return {'passed': passed, 'status': 'not_reported' if passed is None else 'pass' if passed else 'fail',
            'infeasibility': failures,
            'unmet_anchor_count': metadata.get('infeasible_or_unmet_anchors'),
            'recorded_qualification': metadata.get('qualification'),
            'source': 'Authenticated DanceResult.metadata; original method decision, not a new threshold.',
            'recorded_transfer_before': metadata.get('transfer_before'),
            'recorded_transfer_after': metadata.get('transfer_after')}


def _counterfactual_music(pair, pins):
    metadata = pair['body'].metadata
    path = metadata.get('timeline_path', metadata.get('music_path'))
    if not path or not Path(path).is_file():
        return None, 'counterfactual MusicTimeline path unavailable'
    expected = metadata.get('timeline_sha256', metadata.get('campaign_inputs', {}).get('music_npz_sha256'))
    if expected is None or sha(path) != expected:
        return None, 'counterfactual MusicTimeline is not bound to body provenance'
    timeline = MusicTimeline.load(path)
    for p in (Path(path), Path(path).with_suffix('.json'), Path(path).with_suffix('.json.sha256.json')):
        _pin(pins, p)
    return timeline, None


def _reconstruction_reference(reconstruction, removed, name, original_music, protocol, pins):
    """Measure reconstruction-to-removal effect with authenticated waveform roles.

    Original-to-removal response remains in its existing report location. This
    additional reference separates the separator residual from named-source
    removal; original body, weights, noise and P2 remain fixed within the pair.
    """
    result = _sensitivity(reconstruction, removed, name)
    reconstruction_music, reconstruction_error = _counterfactual_music(reconstruction, pins)
    removed_music, removed_error = _counterfactual_music(removed, pins)
    checks = {'fixed_pipeline': result['fixed_input_provenance']['status'] == 'pass',
              'fixed_P2_RGB': result['P2_RGB_exact'],
              'registered_reconstruction_reference': bool(protocol and
                  protocol.get('counterfactuals', {}).get('reconstruction', {}).get('kind') == 'source_reconstruction' and
                  protocol.get('counterfactuals', {}).get(name, {}).get('kind') == 'source_removal')}
    reasons = [e for e in (reconstruction_error, removed_error) if e]
    if reconstruction_music is not None and removed_music is not None:
        rm, mm, original = reconstruction_music.metadata, removed_music.metadata, original_music.metadata
        ri, mi = rm.get('intervention') or {}, mm.get('intervention') or {}
        source = name.removeprefix('mute_')
        checks.update(
            reconstruction_role=ri.get('kind') == 'source_reconstruction',
            named_removal_role=mi.get('kind') == 'source_removal' and mi.get('source') == source,
            both_loudness_matched=ri.get('loudness_match') is True and mi.get('loudness_match') is True,
            same_original_waveform=bool(original.get('input_sha256')) and
                rm.get('input_sha256') == mm.get('input_sha256') == original['input_sha256'],
            same_pinned_separator=bool(original.get('separator')) and
                rm.get('separator') == mm.get('separator') == original['separator'],
            same_loudness_target=ri.get('loudness_before_lufs') is not None and
                ri.get('loudness_before_lufs') == mi.get('loudness_before_lufs'),
            measured_matching_method=bool(ri.get('match_method')) and ri.get('match_method') == mi.get('match_method'))
    else:
        checks['authenticated_waveform_roles'] = False
    result['reference'] = 'loudness-matched sum-all-stems reconstruction'
    result['removal_reference_provenance'] = {
        'status': 'pass' if all(v is True for v in checks.values()) else 'unqualified',
        'checks': checks, 'missing_evidence': reasons,
        'scope': 'Registered reconstruction reference; separation is an estimate, not instrument-presence ground truth.'}
    return result


def _donor_alignment(candidate, donor, original_music, pins):
    metadata = donor['body'].metadata
    path = metadata.get('timeline_path', metadata.get('music_path'))
    if not path or not Path(path).exists():
        return {'status': 'missing_counterfactual_audio_timeline_path',
                'scope': 'Cannot establish matched BPM or tempo alignment from changed body hashes.'}
    expected = metadata.get('timeline_sha256', metadata.get('campaign_inputs', {}).get('music_npz_sha256'))
    if expected is None or sha(path) != expected:
        raise ValueError('Donor audio path does not match generated-body provenance')
    audio = MusicTimeline.load(path)
    _pin(pins, path); _pin(pins, Path(path).with_suffix('.json'))
    rows = []
    for pair, music in ((candidate, original_music), (donor, audio)):
        mask = pair['body_window']
        t = pair['body'].time[mask]
        beats = music.beats[(music.beats >= t[0]) & (music.beats <= t[-1])]
        rows.append((pair['body'].world[mask], t, beats))
    bpms = [float(60 / np.median(np.diff(row[2]))) if len(row[2]) > 1 else None for row in rows]
    similar = abs(bpms[0] - bpms[1]) / max(bpms) <= .05 if all(x is not None for x in bpms) else None
    return {'status': 'measured', 'beat_alignment': tempo_aligned_comparison(*rows[0], *rows[1]),
            'estimated_bpm': bpms, 'similar_bpm_within_5pct': similar,
            'donor_music_path': str(path),
            'scope': 'Recomputed original/donor beats; matched-tempo sensitivity is not artistic benefit.'}


def _learning_support(root, arm, seed, report, pins):
    """Authenticate complete frozen DEV support and the trained checkpoint."""
    protocol_path = root / 'learning/protocol.json'
    run = root / 'learning/runs' / arm / ('seed_' + str(seed))
    checkpoint = run / 'final.pt'
    train_report = run / 'report.json'
    missing = [str(p) for p in (protocol_path, checkpoint, train_report) if not p.exists()]
    if missing:
        return {'status': 'unqualified', 'reasons': ['missing protocol/checkpoint/training report: ' + p for p in missing]}
    protocol, training = _auth_json(protocol_path), _auth_json(train_report)
    for p in (protocol_path, train_report, checkpoint):
        _pin(pins, p); _pin(pins, str(p) + '.sha256.json')
    reasons = []
    protocol_hash, checkpoint_hash = sha(protocol_path), sha(checkpoint)
    identity, trained = report.get('identity', {}), training.get('identity', {})
    if identity.get('protocol_sha256') != protocol_hash or identity.get('checkpoint_sha256') != checkpoint_hash:
        reasons.append('evaluation identity differs from frozen protocol/checkpoint')
    if trained.get('arm') != arm or trained.get('seed') != seed or trained.get('protocol_sha256') != protocol_hash:
        reasons.append('training identity differs from arm/seed/protocol')
    if training.get('steps') != protocol.get('training', {}).get('steps'):
        reasons.append('training exposure is incomplete or changed')
    if training.get('artifacts', {}).get(str(checkpoint)) != checkpoint_hash:
        reasons.append('final checkpoint differs from training receipt')
    if protocol.get('test_access') is not False or training.get('test_access') is not False:
        reasons.append('development-only data policy not authenticated')
    expected = protocol.get('evaluation', {}).get('windows', [])
    actual = report.get('rows', [])
    if not expected or len(actual) != len(expected):
        reasons.append('evaluation window coverage incomplete')
    else:
        for expected_row, actual_row in zip(expected, actual):
            if any(actual_row.get(key) != value for key, value in expected_row.items()):
                reasons.append('evaluation windows/noise/donors differ from frozen order and identities')
                break
            conditions = actual_row.get('conditions', {})
            required = ['correct', 'rms_control'] if arm == 'planner' else ['correct', 'donor']
            if arm != 'planner' and expected_row.get('shift_start') is not None:
                required.append('shift')
            field = 'local_speed_rmse' if arm == 'planner' else 'flow_loss'
            if any(conditions.get(condition, {}).get(field) is None for condition in required):
                reasons.append('a required paired primary condition is missing')
                break
    return {'status': 'pass' if not reasons else 'unqualified', 'reasons': reasons,
            'expected_windows': len(expected), 'observed_windows': len(actual),
            'checkpoint_sha256': checkpoint_hash, 'protocol_sha256': protocol_hash,
            'steps': training.get('steps')}


def _learning(root, method, pins):
    arm = 'planner' if method == 'D6' else method
    rows = []
    for seed in (0, 1):
        path = root / 'learning/evaluation' / arm / ('seed_' + str(seed) + '.json')
        if not path.exists():
            rows.append({'seed': seed, 'status': 'missing', 'path': str(path)})
            continue
        report = _auth_json(path)
        _pin(pins, path); _pin(pins, str(path) + '.sha256.json')
        if report.get('test_access') is not False:
            raise ValueError('Learning report does not authenticate development-only scope')
        support = _learning_support(root, arm, seed, report, pins)
        per_comparator = {}
        comparators = ('rms_control', 'donor', 'shift', 'shuffled_plan') if method == 'D6' else ('donor', 'shift')
        field = 'local_speed_rmse' if method == 'D6' else 'flow_loss'
        for comparator in comparators:
            grouped = {}
            for row in report.get('rows', []):
                conditions = row.get('conditions', {})
                correct = conditions.get('correct', {}).get(field)
                other = conditions.get(comparator, {}).get(field)
                if correct is None or other is None:
                    continue
                if not np.isfinite([correct, other]).all() or min(correct, other) < 0:
                    raise ValueError('Invalid learning loss values')
                # Convert RMSE back to MSE for D6 registered predictive gain.
                if method == 'D6':
                    correct, other = correct ** 2, other ** 2
                grouped.setdefault(row.get('canonical_id', row.get('key')), []).append((correct, other))
            song_values = [np.mean(values, axis=0) for values in grouped.values()]
            if song_values:
                correct, other = np.mean(song_values, axis=0)
                gain = float((other - correct) / other) if other > 1e-20 else None
            else:
                correct = other = gain = None
            per_comparator[comparator] = {'relative_gain': gain, 'canonical_songs': len(song_values),
                                          'correct_loss': float(correct) if correct is not None else None,
                                          'comparison_loss': float(other) if other is not None else None}
        primary = per_comparator.get('rms_control', {}).get('relative_gain') if method == 'D6' else None
        if method in ('D7', 'D8'):
            values = [per_comparator[c]['relative_gain'] for c in ('donor', 'shift')]
            primary = min(values) if all(v is not None for v in values) else None
        rows.append({'seed': seed, 'status': 'measured' if primary is not None and support['status'] == 'pass' else 'unqualified_primary_evidence',
                     'primary': primary if support['status'] == 'pass' else None, 'observed_proxy': primary, 'support': support, 'comparisons': per_comparator, 'field': field, 'source': str(path),
                     'scope': report.get('claim_scope'), 'original_summary': report.get('summary')})
    return {'seeds': rows, 'display_seed': 0, 'display_primary': rows[0].get('primary'),
            'both_seeds_available': all(r.get('primary') is not None for r in rows),
            'scope': 'Canonical-song-equal exposed-development metrics; model selection forbidden.'}


def _primary(method, scores, controls, sensitivities, learning, comparators):
    metric, threshold = PRIMARY[method]
    value = None
    missing = []
    details = {}
    if method == 'D1':
        value = _nested(scores, 'event_alignment', 'f1_minus_null_p95')
    elif method == 'D2':
        value = _nested(scores, 'body_release', 'lift_over_null_p95')
    elif method == 'D4':
        value = _nested(scores, 'recurrence', 'lift_over_null_p95')
    elif method == 'D5':
        value = _nested(scores, 'emitted_release', 'lift_over_null_p95')
    elif method == 'D3':
        values = []
        for name in SOURCE_REGIONS:
            original_row = sensitivities.get(name)
            row = original_row.get('reconstruction_reference') if original_row else None
            if (row is None or original_row.get('fixed_input_provenance', {}).get('status') != 'pass'
                    or original_row.get('P2_RGB_exact') is not True
                    or row['fixed_input_provenance']['status'] != 'pass'
                    or row.get('removal_reference_provenance', {}).get('status') != 'pass'):
                missing.append(name + ': authenticated reconstruction-to-removal comparison')
            else:
                effect = row['body']['assigned_minus_unassigned_m']
                if effect is not None:
                    values.append(effect)
        value = min(values) if len(values) == len(SOURCE_REGIONS) else None
        details['source_assignments'] = {k: list(v) for k, v in SOURCE_REGIONS.items()}
        details['source_removal_reference'] = 'loudness-matched sum-all-stems reconstruction'
        details['original_reference_selectivity_m'] = {name: _nested(sensitivities.get(name), 'body', 'assigned_minus_unassigned_m') for name in SOURCE_REGIONS}
    elif method in ('D6', 'D7'):
        value = learning.get('display_primary') if learning else None
        if not learning or not learning['both_seeds_available']:
            missing.append('complete development evaluations for both seeds')
    elif method == 'D8':
        candidate = _nested(scores, 'recurrence', 'spearman')
        d7 = _nested(comparators.get('D7'), 'recurrence', 'spearman')
        shuffled = _nested(controls.get('shuffled_plan'), 'recurrence', 'spearman')
        controls_valid = (_nested(sensitivities.get('shuffled_plan'), 'fixed_input_provenance', 'status') == 'pass'
            and _nested(comparators.get('D7'), 'generation_input_match', 'status') == 'pass')
        if candidate is not None and d7 is not None and shuffled is not None and controls_valid:
            value = min(candidate - d7, candidate - shuffled)
        else:
            missing.append('D7 and same-audio shuffled-plan recurrence with authenticated fixed inputs')
        if not learning or not learning.get('both_seeds_available'):
            missing.append('complete authenticated development evaluations for both seeds')
        details['against_D7'] = candidate - d7 if candidate is not None and d7 is not None else None
        details['against_shuffled_plan'] = candidate - shuffled if candidate is not None and shuffled is not None else None
    if value is None:
        missing.append('defined primary metric and reference')
    # Separate diagnostic threshold from stronger controls and qualification.
    comparator_deltas = {}
    path = {'D1': ('event_alignment', 'f1'), 'D2': ('body_release', 'contrast'),
            'D4': ('recurrence', 'spearman'), 'D5': ('emitted_release', 'contrast')}.get(method)
    if path:
        candidate = _nested(scores, *path)
        for name, row in controls.items():
            other = _nested(row, *path)
            comparator_deltas[name] = candidate - other if candidate is not None and other is not None else None
    baseline_value = _nested(comparators.get('baseline'), *path) if path else None
    observed = _nested(scores, *path) if path else None
    return {'diagnostic_id': metric, 'value': value, 'minimum_increment': threshold,
            'effect_above_baseline': observed - baseline_value if observed is not None and baseline_value is not None else None,
            'magnitude_gate': 'unqualified' if value is None else ('pass' if value >= threshold else 'fail'),
            'missing_evidence': sorted(set(missing)), 'control_deltas': comparator_deltas, **details,
            'qualification': 'diagnostic_only_pending_user_review',
            'independent_semantic_reference': False,
            'scope': 'Registered proxy magnitude; neither controller labels nor this gate establish artistic success.'}


def _write_derived(path, report):
    """Keep historical derived reports; update only the explicit latest view."""
    path = Path(path)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n'
    key = hashlib.sha256(payload.encode()).hexdigest()
    history = path.parent / 'comparison_history' / (path.stem + '_' + key[:16] + '.json')
    history.parent.mkdir(parents=True, exist_ok=True)
    if not history.exists():
        with history.open('x') as stream:
            stream.write(payload)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(payload)
    os.replace(temporary, path)
    return report


def score_pair(root, method, song):
    """Score available outputs and controls; absent evidence remains explicit."""
    root = Path(root)
    if method not in METHODS:
        raise ValueError('Expected D1..D8 method')
    plan = read(root / 'PLAN.json')
    track = _track(plan, song)
    pins = {}
    from .campaign import resolve_variant, variant_binding_pins
    artifact_id = resolve_variant(root, method)
    pins.update(variant_binding_pins(root))
    for path in (root / 'PLAN.json', Path(__file__), Path(__file__).with_name('diagnostics.py')):
        _pin(pins, path)
    base, missing_base = _load_pair(root, 'baseline', song, plan, pins)
    candidate, missing_candidate = _load_pair(root, artifact_id, song, plan, pins)
    output = root / 'timelines' / artifact_id / song / 'comparison.json'
    if base is None or candidate is None:
        return _write_derived(root / 'reports/missing_comparisons' / (artifact_id + '_' + song + '.json'),
            {'method': method, 'artifact_id': artifact_id, 'song': song, 'status': 'unqualified_missing_outputs',
            'missing': missing_base + missing_candidate, 'inputs': pins, 'artistic_acceptance': 'pending_user_review'})
    clock_precision = _same_clock_and_patch(candidate, base)
    music_path = _music_path(root, track)
    music = MusicTimeline.load(music_path)
    _pin(pins, music_path); _pin(pins, music_path.with_suffix('.json'))
    selection_path = root / 'records/audio_intervals' / (song + '.json')
    release_path = selection_path.with_name(song + '.release.json')
    selection = read(selection_path) if selection_path.exists() else None
    release = _auth_json(release_path) if release_path.exists() else None
    for path in (selection_path, release_path, Path(str(release_path) + '.sha256.json')):
        _pin(pins, path)
    if selection and selection.get('music_sha256') != sha(music_path):
        raise ValueError('Frozen event observations use different audio features')
    if release and (release['inputs']['music_sha256'] != sha(music_path)
            or release['inputs']['music_metadata_sha256'] != sha(music_path.with_suffix('.json'))
            or release['inputs']['plan_sha256'] != sha(root / 'PLAN.json')):
        raise ValueError('Frozen release intervals use different audio features')
    baseline_scores = _scores(base, base, music, selection, release, plan)
    candidate_scores = _scores(candidate, base, music, selection, release, plan)
    controls, sensitivities, missing, control_pairs = {}, {}, {}, {}
    expected = set(EXPECTED_CONTROLS[method])
    available = root / 'bodies'
    expected |= {p.name.split('__', 1)[1] for p in available.glob(artifact_id + '__*') if p.is_dir()}
    for name in sorted(expected):
        other, absent = _load_pair(root, artifact_id + '__' + name, song, plan, pins)
        if other is None:
            missing[name] = absent
            continue
        _same_clock_and_patch(candidate, other)
        control_pairs[name] = other
        controls[name] = _scores(other, base, music, selection, release, plan)
        sensitivities[name] = _sensitivity(candidate, other, name)
        if name == 'donor':
            sensitivities[name]['tempo_aligned_donor'] = _donor_alignment(candidate, other, music, pins)
    cf_protocol_path = root / 'records/COUNTERFACTUAL_PROTOCOL_v2.json'
    cf_protocol = read(cf_protocol_path) if cf_protocol_path.exists() else None
    _pin(pins, cf_protocol_path)
    reconstruction = control_pairs.get('reconstruction')
    if reconstruction is not None:
        for name in SOURCE_REGIONS:
            if name in control_pairs:
                sensitivities[name]['reconstruction_reference'] = _reconstruction_reference(
                    reconstruction, control_pairs[name], name, music, cf_protocol, pins)
    comparators = {'baseline': baseline_scores}
    if method in ('D6', 'D7', 'D8'):
        for identifier in ({'D6': ('D2',), 'D7': ('continued',), 'D8': ('D7', 'D8_flat')}[method]):
            other, absent = _load_pair(root, identifier, song, plan, pins)
            if other is None:
                missing['comparator_' + identifier] = absent
            else:
                _same_clock_and_patch(candidate, other)
                comparators[identifier] = _scores(other, base, music, selection, release, plan)
                am, bm = candidate['body'].metadata, other['body'].metadata
                fields = ('seed', 'draw_seed', 'feature_schema_sha256', 'timeline_sha256', 'base92_sha256')
                checks = {key: am.get(key) == bm.get(key) if am.get(key) is not None and bm.get(key) is not None else None for key in fields}
                comparators[identifier]['generation_input_match'] = {'checks': checks, 'status': 'pass' if all(v is True for v in checks.values()) else 'unqualified'}
    # The prescribed corrupted-plan output is often named shifted_plan.
    if 'shuffled_plan' not in controls and 'shifted_plan' in controls:
        controls['shuffled_plan'] = controls['shifted_plan']
        sensitivities['shuffled_plan'] = sensitivities['shifted_plan']
        missing.pop('shuffled_plan', None)
    learning = _learning(root, method, pins) if method in ('D6', 'D7', 'D8') else None
    if learning is not None:
        support = learning['seeds'][0].get('support', {})
        declared = candidate['body'].metadata.get('checkpoint_sha256')
        learning['display_body_checkpoint_matches'] = (declared == support.get('checkpoint_sha256')
            if declared and support.get('checkpoint_sha256') else None)
        if learning['display_body_checkpoint_matches'] is not True:
            learning['display_primary'] = None
            learning['body_binding_reason'] = 'Displayed body checkpoint is not authenticated to development evaluation.'
    gates = comparison_gates(candidate_scores['body'], baseline_scores['body'],
        candidate_emission=candidate_scores['emission'], baseline_emission=baseline_scores['emission'],
        evidence={'musical_reference_kind': 'automatic_audio_proxy'})
    physical_names = ('finite_body', 'bone_nonregression', 'supported_slide_nonregression',
                      'jerk_nonregression', 'finite_emission', 'emitted_saturation_nonregression',
                      'emitted_retention_nonregression')
    physical_states = [gates['gates'][name]['status'] for name in physical_names]
    physical_status = ('fail' if 'fail' in physical_states else
                       'unqualified' if 'unqualified' in physical_states else 'pass')
    counts = {kind: sum(p['kind'] == kind for p in candidate['patch']) for kind in ('beam', 'laser', 'bulb')}
    exact_p2 = bool(np.array_equal(candidate['controls']['rgb'], base['controls']['rgb']))
    baseline_difference = _sensitivity(candidate, base, 'baseline')
    method_postgate = _method_postgate(candidate)
    body_hash = array_sha(candidate['body'].world)
    angles = candidate['controls']['angles_deg'][:, candidate['moving']]
    emitted_hash = array_sha(angles)
    visual_hash = hashlib.sha256((body_hash + array_sha(candidate['arrays']['universes'])).encode()).hexdigest()
    report = {'schema_version': 1, 'method': method, 'artifact_id': artifact_id, 'song': song,
        'clock_precision': clock_precision,
        'status': 'diagnostic_pending_user_review', 'inputs': pins,
        'primary': _primary(method, candidate_scores, controls, sensitivities, learning, comparators),
        'baseline': baseline_scores, 'candidate': candidate_scores, 'controls': controls,
        'control_sensitivity': sensitivities, 'comparators': comparators, 'learning': learning,
        'missing_controls': missing, 'baseline_difference': baseline_difference,
        'engineering_gates': gates, 'physical_engineering_status': physical_status,
        'failure_diagnostic': physical_status == 'fail' or method_postgate['passed'] is False,
        'method_postgate': method_postgate,
        'method_infeasibility': method_postgate['infeasibility'],
        'control_method_postgates': {name: _method_postgate(pair) for name, pair in control_pairs.items()},
        'counterfactual_protocol': {'path': str(cf_protocol_path), 'sha256': sha(cf_protocol_path)} if cf_protocol else None,
        'physical_failure_reasons': [name for name in physical_names if gates['gates'][name]['status'] == 'fail'],
        'installation': {'counts': counts, 'expected_counts': plan['counts'], 'P2_RGB_exact': exact_p2,
                         'fixed_patch_and_clock': True, 'actual_emitted_DMX': True,
                         'separate_color_diagnostics': candidate['report'].get('diagnostics', {}).get('color'),
                         'separate_brightness_diagnostics': candidate['report'].get('diagnostics', {}).get('brightness'),
                         'mapping_hard_checks': candidate['report'].get('hard_checks')},
        'identities': {'body': body_hash, 'emitted_motion': emitted_hash, 'full_visual_payload': visual_hash},
        'artistic_acceptance': 'pending_user_review',
        'semantic_qualification': 'unqualified_without_independent_reference',
        'scope': 'Actual authenticated output diagnostics; thresholds, hashes and activity do not prove choreography.'}
    if report['failure_diagnostic']:
        report['status'] = 'failure_diagnostic'
    if not exact_p2:
        report['status'] = 'failed_fixed_P2_invariant'
    expected_counts = {'beam': plan['counts']['beams'], 'laser': plan['counts']['lasers'], 'bulb': plan['counts']['bulbs']}
    if counts != expected_counts:
        report['status'] = 'failed_fixed_fixture_count'
    return _write_derived(output, report)


def summarize(root):
    """Aggregate latest comparisons and exact duplicates without fitting anything."""
    root = Path(root)
    plan = read(root / 'PLAN.json')
    reports, missing, pins, stale, checked = [], [], {}, [], {}
    from .campaign import resolve_variant, variant_binding_pins
    artifact_ids = {method: resolve_variant(root, method) for method in METHODS}
    pins.update(variant_binding_pins(root))
    for method in METHODS:
        for track in plan['tracks']:
            path = root / 'timelines' / artifact_ids[method] / track['id'] / 'comparison.json'
            if path.exists():
                report = read(path)
                invalid = []
                if report.get('method') != method or report.get('artifact_id', method) != artifact_ids[method]:
                    invalid.append('logical/artifact variant identity mismatch')
                for source, expected in report.get('inputs', {}).items():
                    if source not in checked:
                        checked[source] = sha(source) if Path(source).is_file() else None
                    if checked[source] != expected:
                        invalid.append(source)
                if invalid:
                    stale.append({'method': method, 'artifact_id': artifact_ids[method], 'song': track['id'], 'changed_or_missing_inputs': invalid})
                    missing.append({'method': method, 'artifact_id': artifact_ids[method], 'song': track['id'], 'reason': 'stale_comparison'})
                else:
                    reports.append(report); _pin(pins, path)
            else:
                missing.append({'method': method, 'artifact_id': artifact_ids[method], 'song': track['id']})
    duplicates = {}
    for kind in ('body', 'emitted_motion', 'full_visual_payload'):
        groups = {}
        for report in reports:
            value = report.get('identities', {}).get(kind)
            if value:
                # Different soundtracks are never grouped as the same demo.
                groups.setdefault((report['song'], value), []).append(report['method'])
        duplicates[kind] = [{'song': song, 'representative': methods[0], 'diagnostic_duplicates': methods[1:]}
                             for (song, _), methods in groups.items() if len(methods) > 1]
    cross_song = []
    tracks = plan['tracks']
    if len(tracks) == 2:
        music = [MusicTimeline.load(_music_path(root, track)) if _music_path(root, track).exists() else None for track in tracks]
        if all(m is not None for m in music):
            for method in METHODS:
                paths = [root / 'bodies' / artifact_ids[method] / t['id'] / 'body.npz' for t in tracks]
                if not all(p.exists() for p in paths):
                    continue
                bodies = [DanceResult.load(p) for p in paths]
                for p in paths:
                    _pin(pins, p)
                rows = []
                for body, audio in zip(bodies, music):
                    start, end = plan['review_start_s'], plan['review_start_s'] + plan['review_duration_s']
                    mask = (body.time >= start) & (body.time < end)
                    beats = audio.beats[(audio.beats >= start) & (audio.beats < end)]
                    rows.append((body.world[mask], body.time[mask], beats))
                metric = tempo_aligned_comparison(*rows[0], *rows[1])
                bpms = [float(60 / np.median(np.diff(r[2]))) if len(r[2]) > 1 else None for r in rows]
                similar = abs(bpms[0] - bpms[1]) / max(bpms) <= .05 if all(x is not None for x in bpms) else None
                cross_song.append({'method': method, 'artifact_id': artifact_ids[method], 'songs': [t['id'] for t in tracks], 'beat_alignment': metric,
                                   'estimated_bpm': bpms, 'similar_bpm_within_5pct': similar,
                                   'scope': 'Beat-aligned showcase comparison; separate matched-BPM donor intervention still required.'})
    summary = {'schema_version': 1, 'status': 'diagnostic_pending_user_review', 'inputs': pins,
               'comparisons': [{'method': r['method'], 'artifact_id': r.get('artifact_id', r['method']), 'song': r['song'], 'status': r['status'],
                                'primary': r.get('primary'), 'engineering_status': r.get('physical_engineering_status'),
                                'method_postgate': r.get('method_postgate'), 'failure_diagnostic': r.get('failure_diagnostic'),
                                'path': str(root / 'timelines' / r.get('artifact_id', r['method']) / r['song'] / 'comparison.json')}
                               for r in reports],
               'missing_comparisons': missing, 'stale_comparisons': stale, 'duplicates': duplicates, 'cross_song': cross_song,
               'artistic_acceptance': 'pending_user_review',
               'scope': 'No automatic winner; exact visual duplicates share one representative per song.'}
    return _write_derived(root / 'reports/comparison_summary.json', summary)

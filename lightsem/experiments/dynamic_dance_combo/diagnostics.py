"""Read-only full-song diagnostics for the frozen combined dance campaign.

No metric generates motion, changes a checkpoint, or selects a demo. Reports
are append-only and separate body, requested motion, and decoded emitted DMX.
Automatic acoustic references are proxies, never artistic acceptance.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from lightsem.conductor.body_transfer import decode
from lightsem.experiments.dynamic_dance import diagnostics as d
from lightsem.experiments.dynamic_dance import transfer_audit as ta
from lightsem.experiments.dynamic_dance.campaign import array_sha, read, sha, write_new
from lightsem.experiments.dynamic_dance.contracts import DanceResult, MusicTimeline


PHYSICAL_GATES = ('finite_body', 'bone_nonregression', 'supported_slide_nonregression',
                  'jerk_nonregression', 'finite_emission', 'emitted_saturation_nonregression',
                  'emitted_retention_nonregression')
COMPARATORS = {'C3': 'D7_reference', 'C4': 'C3'}
CONTROLS = {'C3': ('C3_shuffled', 'G78', 'G78_shift'),
            'C4': ('C4_loudness', 'C4_shifted')}
WAVEFORM_CONTROLS = ('gain', 'shift', 'mute_drums', 'mute_vocals', 'donor', 'reconstruction')
PARAMETERS = {
    'version': 'full-song-combination-diagnostics-v1',
    'native_fps': 30., 'lighting_fps': 40., 'native_clock_roundoff_s': 1e-8,
    'minimum_primary_increment': .02, 'recurrence_samples_max': 128,
    'recurrence_minimum_pair_separation_s': 2.,
    'recurrence_null_shift_fractions': [.113, .283, .463, .659, .857],
    'release_denominator_floors': {'body_m_s': .01, 'motor_deg_s': 1.},
    'event_tolerance_s': .12, 'body_event_prominence_m_s': .08,
    'motor_event_prominence_deg_s': 1., 'event_minimum_separation_s': .2,
    'scope': 'Complete own-song exposed showcase; fixed proxies, no artistic verdict.',
}


def _pin(pins, path, *, required=True):
    path = Path(path)
    if required or path.exists():
        pins[str(path)] = sha(path)


def _authenticated(path):
    path = Path(path)
    if read(str(path) + '.sha256.json')['sha256'] != sha(path):
        raise ValueError('Changed authenticated report: ' + str(path))
    return read(path)


def _write(path, value):
    """Identical repeat reads are allowed; changed evidence needs a new root."""
    path = Path(path)
    json.dumps(value, allow_nan=False)
    if path.exists():
        if _authenticated(path) != value:
            raise FileExistsError('Refuse to overwrite prior diagnostic: ' + str(path))
        return value
    write_new(path, value)
    write_new(str(path) + '.sha256.json', {'sha256': sha(path)})
    return value


def _track(plan, track):
    name = track['id'] if isinstance(track, dict) else track
    matching = [row for row in plan['tracks'] if row['id'] == name]
    if len(matching) != 1:
        raise ValueError('Unknown or ambiguous song: ' + str(name))
    if isinstance(track, dict) and track != matching[0]:
        raise ValueError('Track differs from frozen PLAN')
    return matching[0]


def _music(track, pins):
    path = Path(track['music_timeline'])
    for p in (path, path.with_suffix('.json'), path.with_suffix('.json.sha256.json')):
        _pin(pins, p)
    music = MusicTimeline.load(path)
    if abs(float(music.metadata['duration_s']) - float(track['duration_s'])) > 1e-6:
        raise ValueError('MusicTimeline does not cover the declared complete song')
    if 'audio_sha256' in track and music.metadata.get('input_sha256') != track['audio_sha256']:
        raise ValueError('MusicTimeline waveform identity differs from PLAN')
    return music


def freeze_release_intervals(root, track):
    """Freeze the historical audio-only recipe on [0, complete song duration).

    Must run before any bodies exist; loading an already sealed selection is
    allowed afterward. No qualifying release remains explicitly unmeasurable.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from lightsem.experiments.dynamic_dance.audio import preselect_intervals
    from lightsem.experiments.dynamic_dance.retiming import musical_evidence
    root = Path(root)
    plan = read(root / 'PLAN.json')
    track = _track(plan, track)
    pins = {}
    _pin(pins, root / 'PLAN.json')
    music = _music(track, pins)
    path = root / 'records/audio_intervals' / (track['id'] + '.release.json')
    if path.exists():
        prior = _authenticated(path)
        if prior['inputs'] != pins:
            raise ValueError('Frozen audio reference inputs changed')
        return prior
    if any((root / 'bodies').glob('*/*/body.npz')):
        raise RuntimeError('Audio references must be frozen before viewing candidate bodies')
    duration = float(track['duration_s'])
    selected = preselect_intervals(music, 0., duration)
    releases = [row for row in selected if row.get('role') == 'release_hypothesis']
    time = np.arange(int(np.ceil(duration * 30))) / 30.
    evidence = musical_evidence(music, time)
    attack = gaussian_filter1d(evidence['attack'], .04 * 30.)
    peaks = find_peaks(attack, distance=15,
                      prominence=max(float(np.std(attack)) * .2, 1e-8))[0]
    result = {'song': track['id'], 'inputs': pins, 'interval_s': [0., duration],
        'selection': 'audio.preselect_intervals/v1 over full song; D1 audio attack peak recipe',
        'selected_audio_intervals': selected,
        'release_intervals': [[r['start'], r['end']] for r in releases],
        'reference_intervals': [[r['reference_start'], r['reference_end']] for r in releases],
        'attack_times': time[peaks].tolist(),
        'section_boundaries': [float(s['start']) for s in music.sections],
        'human_annotated': False, 'independent_semantic_reference': False,
        'scope': 'Pre-output automatic acoustic hypothesis; no semantic gold labels.'}
    return _write(path, result)


def _clock(time, count, fps, label):
    time = np.asarray(time)
    if time.shape != (count,) or count < 4 or not np.isfinite(time).all():
        raise ValueError('Invalid full-song ' + label + ' clock')
    if not np.allclose(time, np.arange(count) / fps, rtol=0, atol=1e-8):
        raise ValueError('Not the full zero-based ' + label + ' clock')


def _load_pair(root, identifier, track, plan, pins):
    song = track['id']
    body_path = root / 'bodies' / identifier / song / 'body.npz'
    folder = root / 'timelines' / identifier / song
    required = (body_path, folder / 'timeline.npz', folder / 'patch.json', folder / 'report.json')
    absent = [str(p) for p in required if not p.exists()]
    if absent:
        return None, absent
    body = DanceResult.load(body_path)
    report = (_authenticated(folder / 'report.json') if (folder / 'report.json.sha256.json').exists()
              else read(folder / 'report.json'))
    bindings = {'body_sha256': body_path, 'timeline_sha256': folder / 'timeline.npz',
                'patch_sha256': folder / 'patch.json'}
    for key, path in bindings.items():
        if report.get(key, report.get('inputs', {}).get(key)) != sha(path):
            raise ValueError('Mapped artifact binding changed: ' + str(path))
    declared_plan = report.get('inputs', {}).get('plan_sha256', report.get('inputs', {}).get(str(root / 'PLAN.json')))
    if declared_plan is not None and declared_plan != sha(root / 'PLAN.json'):
        raise ValueError('Mapping used a different PLAN')
    for path, digest in report.get('inputs', {}).items():
        if isinstance(path, str) and Path(path).is_absolute() and isinstance(digest, str):
            if sha(path) != digest:
                raise ValueError('Mapped input changed: ' + path)
            _pin(pins, path)
    for path in required + (body_path.with_suffix('.json'), body_path.with_suffix('.json.sha256.json')):
        _pin(pins, path)
    for path in (body_path.parent / 'plan.npz', body_path.parent / 'plan.json',
                 body_path.parent / 'plan.json.sha256.json', folder / 'report.json.sha256.json'):
        _pin(pins, path, required=False)
    with np.load(folder / 'timeline.npz', allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    for key in ('time', 'body', 'targets', 'universes', 'dimmer'):
        if key not in arrays:
            raise ValueError('Required timeline array missing: ' + key)
    if any(value.dtype.kind not in 'fibu' or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Timeline contains invalid/nonfinite values')
    duration = float(track['duration_s'])
    nb = int(track.get('body_frames', np.ceil(duration * 30.)))
    nl = int(track.get('lighting_frames', np.ceil(duration * 40.)))
    if nb != int(np.ceil(duration * 30.)) or nl != int(np.ceil(duration * 40.)):
        raise ValueError('PLAN frame counts do not cover the complete track')
    _clock(body.time, nb, 30., 'body')
    _clock(arrays['time'], nl, 40., 'lighting')
    patch = read(folder / 'patch.json')
    n = len(patch)
    if arrays['body'].shape != (nl, 22, 3) or arrays['targets'].shape != (nl, n, 3) or arrays['dimmer'].shape != (nl, n):
        raise ValueError('Mapped full-song array shape mismatch')
    moving = np.asarray([p['kind'] != 'bulb' for p in patch])
    if not moving.any():
        raise ValueError('Comparison requires moving outputs')
    controls = decode(arrays['universes'], patch)
    vectors = arrays['targets'][:, moving] - np.asarray([p['position'] for p in patch])[moving]
    length = np.linalg.norm(vectors, axis=-1)
    if np.any(length <= 1e-12):
        raise ValueError('Requested target coincides with fixture position')
    requested = np.stack((np.rad2deg(np.arctan2(vectors[..., 0], -vectors[..., 1])),
        np.rad2deg(np.arcsin(np.clip(vectors[..., 2] / length, -1, 1)))), axis=-1)
    ranges = np.asarray([[p['pan_range_deg'], p['tilt_range_deg']] for p in patch if p['kind'] != 'bulb'])
    color_path = root / 'colors' / song / 'REPORT.json'
    color_report = None
    if color_path.exists():
        if report.get('inputs', {}).get(str(color_path)) != sha(color_path):
            raise ValueError('Unbound external color evidence')
        color_report = read(color_path)
        _pin(pins, color_path)
    return {'id': identifier, 'body': body, 'report': report, 'arrays': arrays, 'patch': patch,
            'color_report': color_report,
            'controls': controls, 'moving': moving, 'requested': requested,
            'ranges': ranges, 'tick': float(ranges.max() / 65535.)}, []


def _same_clock_and_patch(candidate, reference):
    if candidate['patch'] != reference['patch'] or not np.array_equal(candidate['arrays']['time'], reference['arrays']['time']):
        raise ValueError('Comparison changed patch or emitted timestamps')
    a, b = candidate['body'].time, reference['body'].time
    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-8):
        raise ValueError('Comparison changed native body timestamps')


def _derivatives(values, times):
    rows = {}
    x, t = np.asarray(values, float), np.asarray(times, float)
    for name in ('speed_deg_s', 'acceleration_deg_s2', 'jerk_deg_s3'):
        x = np.diff(x, axis=0) / np.diff(t)[:, None, None]
        t = (t[1:] + t[:-1]) / 2
        rows[name] = d._stats(np.linalg.norm(x, axis=-1))
    return rows


def _color_and_brightness(pair):
    controls, arrays, patch = pair['controls'], pair['arrays'], pair['patch']
    rgb = np.rint(controls['rgb'] * 255).astype(np.uint8)
    dimmer = controls['dimmer']
    changed = np.any(np.diff(rgb.astype(int), axis=0) != 0, axis=(1, 2))
    groups = {}
    for side, left in (('left', True), ('right', False)):
        ids = [i for i, p in enumerate(patch) if (p['position'][0] < 0) == left]
        if ids:
            groups[side] = {'fixture_count': len(ids),
                'exact_within_group': bool(all(np.array_equal(rgb[:, ids[0]], rgb[:, i]) for i in ids)),
                'quantized_rgb_values': len(np.unique(rgb[:, ids[0]], axis=0))}
    color = {'rgb_changing_ticks': int(changed.sum()), 'spatial_groups': groups,
             'mapper_report': pair['report'].get('diagnostics', {}).get('color'),
             'pinned_color_evidence': (pair.get('color_report') or {}).get('diagnostics'),
             'scope': 'RGB-changing ticks include transitions; they are not palette-event counts.'}
    # Only explicit saved IDs can establish palette changes/occupancy.
    palette = arrays.get('palette_id', arrays.get('palette_ids40'))
    if palette is not None:
        palette = np.asarray(palette)
        if palette.ndim not in (1, 2) or len(palette) != len(rgb):
            raise ValueError('Invalid saved palette ID timeline')
        active = arrays.get('color_active', arrays.get('color_active40'))
        if active is None:
            active = np.ones(len(palette), bool)
            color['palette_activity_scope'] = 'No active mask saved; all declared IDs counted, including possible initialization.'
        else:
            active = np.asarray(active)
            if active.shape != (len(palette),) or not np.all((active == 0) | (active == 1)):
                raise ValueError('Invalid palette activity mask')
            active = active.astype(bool)
            color['palette_activity_scope'] = 'Visible active color ticks only; pre-first-cue initialization IDs excluded.'
        flattened = palette.reshape(len(palette), -1)
        ids, counts = np.unique(flattened[active], axis=0, return_counts=True)
        # Count only adjacent visible changes; do not bridge inactive gaps.
        changing = np.any(np.diff(flattened, axis=0) != 0, axis=1) & active[1:] & active[:-1]
        color.update(palette_changes=int(changing.sum()), palette_active_frames=int(active.sum()),
            palette_occupancy=[{'ids': key.tolist(), 'frames': int(count), 'fraction': float(count / len(rgb))}
                               for key, count in zip(ids, counts)], palette_id_source='saved timeline')
    else:
        color.update(palette_changes=None, palette_occupancy=None,
                     palette_id_source='not saved; inspect pinned mapping color evidence')
    names = [p['name'] for p in patch]
    brightness = {'decoded_dimmer': d._stats(dimmer),
        'inactive_fraction': float((dimmer <= 1 / 255.).mean()),
        'full_dimmer_fraction': float((dimmer >= 254 / 255.).mean()),
        'intent_to_decoded_max_abs': float(np.max(np.abs(arrays['dimmer'] - dimmer))),
        'by_kind': {kind: d._stats(dimmer[:, [p['kind'] == kind for p in patch]])
                    for kind in ('beam', 'laser', 'bulb')},
        'first_30s_by_kind': {kind: d._stats(dimmer[arrays['time'] < 30.][:, [p['kind'] == kind for p in patch]])
                            for kind in ('beam', 'laser', 'bulb')},
        'mapper_report': pair['report'].get('diagnostics', {}).get('brightness', pair['report'].get('brightness')),
        'scope': 'Actual normalized emitted dimmer commands, not photometric light output.'}
    if 'brightness_envelope' in arrays:
        brightness['envelope'] = d._stats(arrays['brightness_envelope'])
    return {'color': color, 'brightness': brightness,
            'duplicates': {'emitted_motion': d.duplicate_streams(controls['angles_deg'][:, pair['moving']],
                                [p['name'] for p in patch if p['kind'] != 'bulb']),
                           'RGB': d.duplicate_streams(rgb, names), 'dimmer': d.duplicate_streams(dimmer, names)},
            'intentional_fixed_controls': ['Bulbs have no pan/tilt channels; omitted from motor statistics.',
                                          'P2 uses two shared spatial RGB groups; identical RGB is intentional.']}


def _scores(pair, reference, music, release, plan, duration):
    body, t = pair['body'], pair['arrays']['time']
    emitted = pair['controls']['angles_deg'][:, pair['moving']]
    requested = pair['requested']
    series = d.dynamics_series(body.world, body.time)
    floor = float(np.quantile(reference['body'].world[:, d.FEET, 1], .02))
    physical = d.body_dynamics(body.world, body.time, contacts=body.contacts,
                              reference_body=reference['body'].world, floor_y=floor)
    limits = plan.get('transfer', {})
    emission = d.emission_diagnostics(requested, emitted, t, tick_deg=pair['tick'],
        maximum_speed_deg_s=float(limits.get('maximum_speed_deg_s', 90.)),
        maximum_acceleration_deg_s2=float(limits.get('maximum_acceleration_deg_s2', 360.)))
    rec = {'body': d.recurrence_alignment(body.world, body.time, music.muq, music.muq_time),
           'requested': ta.feature_recurrence(requested / pair['ranges'], t, music.muq, music.muq_time),
           'emitted': ta.feature_recurrence(emitted / pair['ranges'], t, music.muq, music.muq_time)}
    events = release.get('attack_times') if release else None
    alignment = {'body': None, 'requested': None, 'emitted': None}
    if events is not None:
        peaks = d.gesture_events(series['relative_speed_m_s'], body.time)
        alignment['body'] = d.event_alignment(peaks, events, start_s=0., end_s=duration)
        alignment['requested'] = ta.motor_event_alignment(requested, t, events, start_s=0., end_s=duration)
        alignment['emitted'] = ta.motor_event_alignment(emitted, t, events, start_s=0., end_s=duration)
    releases = ta._stage_releases(series['relative_speed_m_s'], body.time,
        ta.motor_speed(requested, t).mean(1), ta.motor_speed(emitted, t).mean(1), t, release)
    root_velocity = series['world_velocity_m_s'][:, :1]
    root_acceleration = d._derivative(root_velocity, body.time)
    root_jerk = d._derivative(root_acceleration, body.time)
    physical['root_acceleration_m_s2'] = d._stats(np.linalg.norm(root_acceleration, axis=-1))
    physical['root_jerk_m_s3'] = d._stats(np.linalg.norm(root_jerk, axis=-1))
    return {'body': physical, 'emission': emission, 'recurrence': rec, 'release': releases,
        'event_alignment': alignment, 'requested': _derivatives(requested, t),
        'emitted': _derivatives(emitted, t), 'channels': _color_and_brightness(pair),
        'requested_pan_branch_jumps': int((np.abs(np.diff(requested[..., 0], axis=0)) > 180).sum()),
        'constrained_motion': {'status': 'observed_after_DMX_quantization',
            'separate_prequantization_trajectory_saved': False,
            'scope': 'Constrained endpoint is actual decoded DMX; no invented intermediate angle trace.'},
        'frames': {'body': len(body.time), 'lighting': len(t)}, 'interval_s': [0., duration]}


def _difference(candidate, reference):
    a, b = candidate['body'].world, reference['body'].world
    ca, cb = candidate['controls'], reference['controls']
    angular = ca['angles_deg'][:, candidate['moving']] - cb['angles_deg'][:, reference['moving']]
    return {'body_rms_m': float(np.sqrt(np.mean((a - b) ** 2))),
        'root_relative_joint_rms_m': float(np.sqrt(np.mean(((a-a[:, :1]) - (b-b[:, :1])) ** 2))),
        'requested_motor_rms_deg': float(np.sqrt(np.mean((candidate['requested'] - reference['requested']) ** 2))),
        'regional_root_relative_rms_m': {name: float(np.sqrt(np.mean(((a-a[:, :1])[:, ids] -
                    (b-b[:, :1])[:, ids]) ** 2))) for name, ids in d.REGIONS.items()},
        'emitted_motor_rms_deg': float(np.sqrt(np.mean(angular ** 2))),
        'above_one_tick_rms': bool(np.sqrt(np.mean(angular ** 2)) >= candidate['tick']),
        'repeat_noise_estimated': False, 'exact_body_duplicate': bool(np.array_equal(a, b)),
        'exact_emitted_motion_duplicate': bool(np.array_equal(ca['angles_deg'], cb['angles_deg'])),
        'exact_full_visual_payload_duplicate': bool(np.array_equal(a, b) and
            np.array_equal(candidate['arrays']['universes'], reference['arrays']['universes'])),
        'RGB_exact': bool(np.array_equal(ca['rgb'], cb['rgb'])),
        'brightness_rms': float(np.sqrt(np.mean((ca['dimmer'] - cb['dimmer']) ** 2))),
        'scope': 'Input/output difference only; differing hashes and movement do not establish choreography.'}


def _generation_metadata(metadata):
    """Find one explicit inherited generation receipt, never guess from hashes."""
    rows = []
    def collect(value):
        if isinstance(value, dict):
            if 'initial_noise_sha256' in value and 'checkpoints' in value:
                rows.append(value)
            else:
                for child in value.values():
                    collect(child)
    collect(metadata)
    return rows[0] if len(rows) == 1 else None


def _source_response(candidate, other, reference=None):
    """Report measured source interventions and their actual fixed-input proof."""
    original = candidate if reference is None else reference
    a = _generation_metadata(original['body'].metadata)
    b = _generation_metadata(other['body'].metadata)
    keys = ('checkpoints', 'initial_noise_sha256', 'draw_seed', 'display_seed', 'residual_cap', 'integration')
    fixed = {key: a.get(key) == b.get(key) if a and b and key in a and key in b else None for key in keys}
    different_audio = (a.get('base92_analyzed_waveform_sha256') != b.get('base92_analyzed_waveform_sha256')
                      if a and b and a.get('base92_analyzed_waveform_sha256') and b.get('base92_analyzed_waveform_sha256') else None)
    result = _difference(original, other)
    result['fixed_generation_inputs'] = {'checks': fixed,
        'status': 'pass' if all(value is True for value in fixed.values()) else 'unqualified',
        'analyzed_waveform_changed': different_audio,
        'reference_id': original['id'], 'intervention_id': other['id']}
    paired = {}
    for name, meta in (('reference', a), ('intervention', b)):
        base_hash = meta.get('base92_analyzed_waveform_sha256') if meta else None
        extra_hash = meta.get('analyzed_waveform_sha256') if meta else None
        paired[name] = (base_hash == extra_hash if base_hash and extra_hash else None)
    result['fixed_generation_inputs']['both_audio_paths_share_waveform'] = paired
    if not all(value is True for value in paired.values()):
        result['fixed_generation_inputs']['status'] = 'unqualified'
    ea, eb = original['arrays'].get('brightness_envelope'), other['arrays'].get('brightness_envelope')
    result['global_brightness_envelope_held_fixed'] = bool(np.array_equal(ea, eb)) if ea is not None and eb is not None else None
    result['brightness_scope'] = 'Global original-song loudness envelope and RGB held fixed for body-path isolation; body accents may still change.'
    return result


def _delta(a, b):
    return float(a - b) if a is not None and b is not None else None


def compare(root, method, track):
    """Write an authenticated full-song C3 or C4 report; never repair artifacts."""
    if method not in COMPARATORS:
        raise ValueError('Expected C3 or C4')
    root = Path(root)
    plan = read(root / 'PLAN.json')
    track = _track(plan, track)
    pins = {}
    for path in (root / 'PLAN.json', Path(__file__), Path(d.__file__), Path(ta.__file__)):
        _pin(pins, path)
    music = _music(track, pins)
    song, duration = track['id'], float(track['duration_s'])
    reference, absent_reference = _load_pair(root, COMPARATORS[method], track, plan, pins)
    candidate, absent_candidate = _load_pair(root, method, track, plan, pins)
    if absent_reference or absent_candidate:
        raise FileNotFoundError('Missing required comparison outputs: ' + repr(absent_reference + absent_candidate))
    _same_clock_and_patch(candidate, reference)
    # Every movie displays D7 on the left. C4's scientific comparator remains
    # C3; authenticate the distinct displayed payload without rescoring it.
    display_reference = reference
    if COMPARATORS[method] != 'D7_reference':
        display_reference, absent_display = _load_pair(root, 'D7_reference', track, plan, pins)
        if absent_display:
            raise FileNotFoundError('Missing required display reference outputs: ' + repr(absent_display))
        _same_clock_and_patch(candidate, display_reference)
    release_path = root / 'records/audio_intervals' / (song + '.release.json')
    release = _authenticated(release_path) if release_path.exists() else None
    if release is not None:
        for path, digest in release['inputs'].items():
            if sha(path) != digest:
                raise ValueError('Audio interval input changed: ' + path)
        if release['inputs'].get(str(root / 'PLAN.json')) != sha(root / 'PLAN.json'):
            raise ValueError('Audio intervals belong to another PLAN')
        _pin(pins, release_path)
        _pin(pins, str(release_path) + '.sha256.json')
    baseline = _scores(reference, reference, music, release, plan, duration)
    scores = _scores(candidate, reference, music, release, plan, duration)
    controls, differences, missing, control_pairs = {}, {}, {}, {}
    expected_controls = CONTROLS[method] + tuple(method + '__' + name for name in WAVEFORM_CONTROLS)
    for name in expected_controls:
        other, absent = _load_pair(root, name, track, plan, pins)
        if absent:
            missing[name] = absent
            continue
        _same_clock_and_patch(candidate, other)
        controls[name] = _scores(other, reference, music, release, plan, duration)
        differences[name] = _difference(candidate, other)
        control_pairs[name] = other
    source_response = {}
    reconstruction = control_pairs.get(method + '__reconstruction')
    for name in WAVEFORM_CONTROLS:
        other = control_pairs.get(method + '__' + name)
        if other is not None:
            row = _source_response(candidate, other)
            if name.startswith('mute_'):
                row['reconstruction_reference'] = _source_response(candidate, other, reconstruction) if reconstruction is not None else None
                row['removal_scope'] = 'Loudness-matched separated-source reconstruction is the causal reference; mixture comparison is descriptive.'
            source_response[name] = row
    metric, field = ('recurrence', 'spearman') if method == 'C3' else ('release', 'contrast')
    def value(row, stage):
        return (row[metric].get(stage) or {}).get(field)
    deltas = {stage: _delta(value(scores, stage), value(baseline, stage))
              for stage in ('body', 'requested', 'emitted')}
    primary = {'diagnostic_id': 'dd-combo-emitted-recurrence-lift' if method == 'C3' else 'dd-combo-emitted-release-lift',
        'value': deltas['emitted'], 'minimum_increment': .02,
        'baseline_id': COMPARATORS[method], 'stage_deltas': deltas,
        'magnitude_gate': 'unqualified' if deltas['emitted'] is None else ('pass' if deltas['emitted'] >= .02 else 'fail'),
        'control_deltas': {name: _delta(value(scores, 'emitted'), value(row, 'emitted')) for name, row in controls.items()},
        'independent_semantic_reference': False,
        'qualification': 'proxy_only_pending_both_songs_controls_and_user_review'}
    gates = d.comparison_gates(scores['body'], baseline['body'], candidate_emission=scores['emission'],
        baseline_emission=baseline['emission'], evidence={'musical_reference_kind': 'automatic_audio_proxy'})
    failures = [name for name in PHYSICAL_GATES if gates['gates'][name]['status'] != 'pass']
    emission = scores['emission']
    limits = plan.get('transfer', {})
    for key, maximum in (('emitted_speed_deg_s', limits.get('maximum_speed_deg_s', 90.)),
                         ('emitted_acceleration_deg_s2', limits.get('maximum_acceleration_deg_s2', 360.))):
        if emission[key]['max'] > maximum + 1e-6:
            failures.append(key + '_hard_limit')
    exact_p2 = np.array_equal(candidate['controls']['rgb'], reference['controls']['rgb'])
    if not exact_p2:
        failures.append('fixed_P2_RGB')
    counts = {kind: sum(p['kind'] == kind for p in candidate['patch']) for kind in ('beam', 'laser', 'bulb')}
    expected = {kind: plan['counts'][plural] for kind, plural in (('beam', 'beams'), ('laser', 'lasers'), ('bulb', 'bulbs'))}
    if counts != expected:
        failures.append('fixed_fixture_counts')
    brightness_error = scores['channels']['brightness']['intent_to_decoded_max_abs']
    if brightness_error > .5 / 255. + 1e-8:
        failures.append('brightness_emission_quantization')
    preflight = {}
    for path in sorted((root / 'preflight').glob('*.json')):
        if path.name.endswith('.sha256.json'):
            continue
        if song in path.name:
            _pin(pins, path)
            preflight[str(path)] = read(path)
    report = {'schema_version': 1, 'method': method, 'song': song, 'comparator': COMPARATORS[method],
        'display_reference_id': 'D7_reference', 'scientific_comparator': COMPARATORS[method],
        'status': 'failure_diagnostic' if failures or primary['magnitude_gate'] != 'pass' else 'diagnostic_pending_user_review',
        'failure_diagnostic': bool(failures or primary['magnitude_gate'] != 'pass'),
        'physical_engineering_status': 'fail' if failures else 'pass',
        'physical_failure_reasons': failures, 'primary': primary, 'parameters': PARAMETERS,
        'candidate': scores, 'baseline': baseline, 'controls': controls, 'missing_controls': missing,
        'baseline_difference': _difference(candidate, reference), 'control_differences': differences,
        'source_response': source_response,
        'engineering_gates': gates, 'inputs': pins, 'source_preflight': preflight,
        'installation': {'counts': counts, 'P2_RGB_exact': bool(exact_p2), 'fixed_patch_and_clock': True,
                         'actual_emitted_DMX': True},
        'identities': {'body': array_sha(candidate['body'].world),
                       'emitted_motion': array_sha(candidate['controls']['angles_deg'][:, candidate['moving']]),
                       'universes': array_sha(candidate['arrays']['universes'])},
        'artistic_acceptance': 'pending_user_review', 'baseline_promotion': False,
        'scope': 'Full-track exposed-song proxies; physical pass, activity and hash differences do not establish musical choreography.'}
    return _write(root / 'reports/comparisons' / (method + '_' + song + '.json'), report)


def audit(root, track):
    """Compare the two requested compositions without selecting between them."""
    return {method: compare(root, method, track) for method in COMPARATORS}

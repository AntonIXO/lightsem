"""Ancillary CPU-only timing audit of already frozen body-transfer artifacts.

Run ``freeze`` before ``run``. This script never generates or edits trajectories,
beats, render inputs, thresholds, or the main campaign's scientific verdicts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

# This is an independent CPU audit, including all numerical library workers.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _thread_setting in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_setting] = '1'
import numpy as np

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v2')
REPO = Path('/root/lightsem')
NATIVE = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/data/shpongle/input.json')
SHP_BUS = Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908/previews/shpongle_20260908/audio_bus.npz')
SHP_RECEIPT = SHP_BUS.with_name('demo.json')
CURE_ROOT = REPO / 'runs/cross_song_20260930_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def verified(path, expected):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError('Input identity mismatch: ' + str(path))
    return actual


def _beat_provenance(plan):
    tracks = {t['id']: t for t in plan['tracks']}
    native = read(NATIVE)
    verified(NATIVE, read(str(NATIVE) + '.sha256.json')['sha256'])
    shp_audio = tracks['shpongle']['audio']
    verified(shp_audio, native['source']['source_audio_sha256'])
    verified(SHP_BUS, native['source']['source_bus_sha256'])
    receipt = read(SHP_RECEIPT)
    verified(SHP_BUS, receipt['artifacts'][str(SHP_BUS)]['sha256'])
    verified(shp_audio, receipt['source_audio_sha256'])
    prepared = read(CURE_ROOT / 'PREPARATION.json')
    cure = next(r for r in prepared['records'] if r['id'] == '02')
    verified(tracks['cure_for_me']['audio'], cure['source_sha256'])
    verified(CURE_ROOT / '02/audio_bus.npz', cure['bus_sha256'])
    verified(CURE_ROOT / '02/rhythm_clock.npz', cure['clock_sha256'])
    verified(cure['prefix_audio'], cure['prefix_audio_sha256'])
    paths = [NATIVE, Path(str(NATIVE) + '.sha256.json'), SHP_BUS, SHP_RECEIPT,
             CURE_ROOT / 'PREPARATION.json', CURE_ROOT / '02/audio_bus.npz',
             CURE_ROOT / '02/rhythm_clock.npz', Path(cure['prefix_audio'])]
    specs = {
        'shpongle': {'path': str(SHP_BUS), 'beat_key': 'beats', 'time_key': 'times',
                    'source_audio': shp_audio, 'method': 'Authenticated cached constant-grid audio detector; not annotated beats.'},
        'cure_for_me': {'path': str(CURE_ROOT / '02/rhythm_clock.npz'), 'beat_key': 'beat_times',
                       'time_key': 'time', 'source_audio': tracks['cure_for_me']['audio'],
                       'method': 'Authenticated cached local pulse tracker; not annotated beats.'}}
    for spec in specs.values():
        with np.load(spec['path'], allow_pickle=False) as z:
            beats, times = z[spec['beat_key']], z[spec['time_key']]
            if (beats.ndim != 1 or len(beats) < 3 or not np.isfinite(beats).all()
                    or np.any(np.diff(beats) <= 0) or np.any(np.diff(times) <= 0)
                    or times[0] > 60 or times[-1] < 179.9):
                raise ValueError('Beat cache lacks finite monotone absolute-time coverage')
        spec['sha256'] = sha(spec['path'])
    return specs, paths


def freeze(root):
    selftest()
    plan = read(root / 'PLAN.json')
    beat_specs, beat_paths = _beat_provenance(plan)
    files = [Path(__file__).resolve(), root / 'PLAN.json', root / 'source_manifest.json'] + beat_paths
    timelines = []
    for track in plan['tracks']:
        for key in ('body', 'body_provenance', 'audio', 'baseline_timeline', 'baseline_patch'):
            p = Path(track[key])
            if str(p) in plan['files']:
                verified(p, plan['files'][str(p)])
            files.append(p)
        timelines.append({'algorithm': 'V1', 'song': track['id'], 'timeline': track['baseline_timeline'],
                          'patch': track['baseline_patch']})
    for algorithm in plan['algorithms']:
        for track in plan['tracks']:
            directory = root / 'timelines' / algorithm['id'] / track['id']
            timelines.append({'algorithm': algorithm['id'], 'song': track['id'],
                              'timeline': str(directory / 'timeline.npz'), 'patch': str(directory / 'patch.json')})
    for row in timelines:
        files.extend([Path(row['timeline']), Path(row['patch'])])
    decoder = Path(plan['source_root']) / 'lightsem/conductor/body_transfer.py'
    verified(decoder, plan['files'][str(decoder)])
    files.append(decoder)
    protocol = {
        'id': 'body-transfer-20261004-ancillary-timing-v1',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'source_script': str(Path(__file__).resolve()), 'source_script_sha256': sha(__file__),
        'scope': 'Ancillary descriptive analysis of already observed frozen outputs; no new hypothesis verdict, dancer requalification or broad music-conditioning claim.',
        'synthetic_metric_controls_passed_before_freeze': True,
        'files': {str(p): sha(p) for p in sorted(set(files))},
        'main_verdicts': {str(root / p): sha(root / p) for p in
            ['COMPARISON.json', 'records/VIBESCIENCE_VERDICTS.json']},
        'timelines': timelines, 'beat_caches': beat_specs,
        'window_s': [60., 180.], 'fps': 40., 'unaligned_sample_hz': 4.,
        'beat_samples_per_interval': 8, 'recurrence_max_frames': 240,
        'alignment': 'Start at first authenticated detected beat >=60s in each song. Match chronological beat indices, with 8 subdivisions and linear interpolation within each interval. Stop at smaller complete interval count before179.975s. No fitted offset, tempo doubling or similarity-maximizing warp.',
        'normalization': 'Subtract each entity coordinate temporal mean; divide each entity trajectory by its vector RMS. Exclude entities with RMS<=1e-10 and report them. This removes fixed placement and scalar gains while preserving directional articulation.',
        'velocity': 'Backward derivative at40Hz before resampling; physical per-second velocity retained. Beat alignment changes query times only, not derivative units.',
        'cross_song': 'Compare matching anatomical indices/fixture order at body positions/velocities, mapper field, requested ray directions/velocities and emitted ray directions/velocities; dimmer andRGB separate.',
        'recurrence': 'Within-song Pearson agreement of temporal Euclidean self-distance matrices after the same entity normalization, <=240 timestamp samples. Report source-to-field, source-to-request, source-to-emitted, request-to-emitted, positions and velocities.',
        'timing_nulls': {'circular_shift_fraction': .25, 'wrap_guard_samples': 2,
            'description': 'Roll reference source timing by one quarter; exclude wrapped prefix and 2 boundary samples at both usable ends. Also reverse source order (negate velocity), and constant output. Keep native chronological alignment and report all controls; no search.'},
        'brightness_bounds': {'moving': [.08, .8], 'bulbs': [.025, .9], 'tolerance': 1/255,
            'scope': 'V2 design bounds; V1 compared descriptively under these reference bounds, not claimed to use them. Report full-scale0/1 separately. Activity is derived from the source field; no added controller.'},
        'limitations': ['Only two requested tracks; no independent similar-BPM music pair.',
            'Shpongle constant-grid detector and Cure local tracker differ; beat alignment is detector-relative and not downbeat/phrase alignment.',
            'Equal-length normalized phase or tempo resampling is not a third independent song.',
            'Matching chronological beat numbers does not imply corresponding musical phrases; report each song actual time support.',
            'Cross-song comparisons sample at 4Hz or 8 points per detected beat; recurrence uses <=240 timestamps and does not measure all high-frequency detail.',
            'Low cross-song similarity and distinct hashes alone do not establish choreography or useful audio conditioning.',
            'Metric controls and analyses are ancillary after campaign outputs were observed; original thresholds and verdicts unchanged.']}
    target = root / 'records/ANCILLARY_TIMING_PROTOCOL.json'
    save(target, protocol)
    print(json.dumps({'protocol': str(target), 'sha256': sha(target), 'files': len(protocol['files'])}), flush=True)


def normalize(x):
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x[..., None]
    centered = x - x.mean(0)
    rms = np.sqrt(np.mean(np.sum(centered ** 2, axis=-1), axis=0))
    active = rms > 1e-10
    return centered / np.maximum(rms, 1e-10)[None, :, None], active, rms


def corr(x, y):
    x, y = np.ravel(x), np.ravel(y)
    x, y = x - x.mean(), y - y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / norm) if norm > 1e-12 else None


def similarity(x, y):
    if x.shape != y.shape:
        raise ValueError('Cross-song entity correspondence requires equal shape')
    a, active_a, rms_a = normalize(x)
    b, active_b, rms_b = normalize(y)
    active = active_a & active_b
    scores = [corr(a[:, i], b[:, i]) if active[i] else None for i in range(len(active))]
    return {'centered_normalized_cosine': corr(a[:, active], b[:, active]) if active.any() else None,
            'per_entity_cosine': scores, 'active_entities_both': int(active.sum()), 'entities': len(active),
            'first_native_RMS_mean': float(rms_a.mean()), 'second_native_RMS_mean': float(rms_b.mean()),
            'inactive_not_rewarded': not active.any()}


def distances(x):
    normalized, active, _ = normalize(x)
    if not active.any():
        return None
    f = normalized[:, active].reshape(len(x), -1)
    energy = np.sum(f ** 2, axis=1)
    d = np.sqrt(np.maximum(energy[:, None] + energy[None] - 2 * f @ f.T, 0))
    return d[np.triu_indices(len(x), 1)]


def recurrence_score(a, b):
    aa, bb = distances(a), distances(b)
    return corr(aa, bb) if aa is not None and bb is not None else None


def timing_controls(source, output, velocity=False):
    sample = np.linspace(0, len(source) - 1, min(240, len(source))).round().astype(int)
    a, b = source[sample], output[sample]
    shift, guard = len(sample) // 4, 2
    valid = slice(shift + guard, len(sample) - guard)
    shifted = np.roll(a, shift, axis=0)
    reversed_ = a[::-1] * (-1 if velocity else 1)
    constant = np.repeat(b[:1], len(b), axis=0)
    return {'aligned_recurrence': recurrence_score(a, b),
            'shifted_source_recurrence_wrap_excluded': recurrence_score(shifted[valid], b[valid]),
            'aligned_on_same_shift_control_support': recurrence_score(a[valid], b[valid]),
            'reversed_source_recurrence': recurrence_score(reversed_, b),
            'constant_output_recurrence': recurrence_score(a, constant),
            'sampled_frames': len(sample), 'shift_samples': shift,
            'shift_control_usable_frames': len(a[valid])}


def interpolate(times, values, queries):
    if queries[0] < times[0] or queries[-1] > times[-1]:
        raise ValueError('No extrapolation allowed')
    flat = values.reshape(len(values), -1)
    return np.stack([np.interp(queries, times, flat[:, i]) for i in range(flat.shape[1])], axis=1).reshape((len(queries),) + values.shape[1:])


def derivative(x, fps):
    return np.diff(x, axis=0, prepend=x[:1]) * fps


def bounds_report(x, lower, upper, tolerance=1/255):
    return {'minimum': float(x.min()), 'median': float(np.median(x)), 'maximum': float(x.max()),
            'fraction_near_design_lower': float(np.mean(np.abs(x - lower) <= tolerance)),
            'fraction_near_design_upper': float(np.mean(np.abs(x - upper) <= tolerance)),
            'per_output_fraction_near_design_upper': np.mean(np.abs(x - upper) <= tolerance, axis=0).tolist(),
            'fraction_outside_design_with_quantization_tolerance': float(np.mean((x < lower - tolerance) | (x > upper + tolerance))),
            'fraction_near_fullscale_zero': float(np.mean(x <= tolerance)),
            'fraction_near_fullscale_one': float(np.mean(x >= 1 - tolerance)),
            'reference_design_bounds': [lower, upper]}


def load_stages(row, fps, decode):
    patch = read(row['patch'])
    moving = np.array([p['kind'] != 'bulb' for p in patch])
    with np.load(row['timeline'], allow_pickle=False) as data:
        time, body, targets = data['time'].copy(), data['body'].copy(), data['targets'].copy()
        decoded = decode(data['universes'], patch)
        field = data['field_points'].copy() if 'field_points' in data.files else None
    if not np.allclose(time, np.arange(len(time)) / fps, atol=1e-10):
        raise ValueError('Timeline is not the absolute40Hz source clock')
    body -= body[:, :1] * np.array([1., 0., 1.])
    vectors = targets[:, moving] - np.array([p['position'] for p in patch])[moving]
    requested = vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    stages = {'body_position': body, 'body_velocity': derivative(body, fps),
              'requested_direction': requested, 'requested_velocity': derivative(requested, fps),
              'emitted_direction': decoded['directions'][:, moving],
              'emitted_velocity': derivative(decoded['directions'][:, moving], fps),
              'dimmer': decoded['dimmer'], 'RGB': decoded['rgb']}
    if field is not None:
        stages.update({'field_position': field, 'field_velocity': derivative(field, fps)})
    return time, stages, moving, patch


def run(root):
    protocol_path = root / 'records/ANCILLARY_TIMING_PROTOCOL.json'
    protocol = read(protocol_path)
    verified(__file__, protocol['source_script_sha256'])
    for p, h in {**protocol['files'], **protocol['main_verdicts']}.items():
        verified(p, h)
    plan = read(root / 'PLAN.json')
    sys.path.insert(0, plan['source_root'])
    from lightsem.conductor.body_transfer import decode
    if Path(sys.modules['lightsem.conductor.body_transfer'].__file__).resolve() != Path(plan['source_root']) / 'lightsem/conductor/body_transfer.py':
        raise RuntimeError('Frozen decoder was not imported')
    beats, beat_report = {}, {}
    for song, spec in protocol['beat_caches'].items():
        with np.load(spec['path'], allow_pickle=False) as z:
            b = z[spec['beat_key']].copy()
        beats[song] = b[(b >= 60.) & (b <= 179.975)]
        beat_report[song] = {'cache': spec, 'whole_cache_beat_count': len(b),
            'window_beat_count': len(beats[song]), 'window_first_beat_s': float(beats[song][0]),
            'window_last_beat_s': float(beats[song][-1]),
            'window_local_BPM_p05_median_p95': np.quantile(60 / np.diff(beats[song]), [.05, .5, .95]).tolist()}
    count = min(len(b) - 1 for b in beats.values())
    phase = np.arange(count * 8) / 8
    queries = {'unaligned_seconds': {song: np.arange(60., 180., .25) for song in beats},
               'piecewise_detected_beat_index': {song: np.interp(phase, np.arange(count + 1), b[:count + 1]) for song, b in beats.items()}}
    report = {'protocol': str(protocol_path), 'protocol_sha256': sha(protocol_path),
              'source_script_sha256': sha(__file__), 'started_utc': datetime.now(timezone.utc).isoformat(),
              'source_identities': protocol['files'], 'scope': protocol['scope'],
              'beat_provenance': beat_report, 'aligned_complete_intervals': count,
              'alignment_queries': {mode: {song: {'samples': len(q), 'start_s': float(q[0]), 'end_s': float(q[-1])} for song, q in songs.items()} for mode, songs in queries.items()},
              'algorithms': {}, 'limitations': protocol['limitations']}
    by_algorithm = {}
    for row in protocol['timelines']:
        by_algorithm.setdefault(row['algorithm'], {})[row['song']] = row
    reference_body = {}
    for algorithm, songs in by_algorithm.items():
        result, sampled = {'within_song': {}, 'cross_song': {}}, {}
        for song, row in songs.items():
            times, stages, moving, patch = load_stages(row, 40., decode)
            if song not in reference_body:
                reference_body[song] = stages['body_position'].copy()
            elif not np.array_equal(stages['body_position'], reference_body[song]):
                raise RuntimeError('Algorithm reused or modified its song body')
            window = (times >= 60) & (times < 180)
            body = stages['body_position'][window]
            entry = {'timeline_sha256': sha(row['timeline']), 'patch_sha256': sha(row['patch']),
                     'fixture_names': [p['name'] for p in patch],
                     'brightness': {kind: bounds_report(stages['dimmer'][window][:, mask], *bounds)
                        for kind, mask, bounds in [('moving', moving, [.08, .8]), ('bulbs', ~moving, [.025, .9])]},
                     'recurrence_and_timing_nulls': {}}
            for label, source, target in [('body_to_field_position', 'body_position', 'field_position'),
                    ('body_to_requested_direction', 'body_position', 'requested_direction'),
                    ('body_to_emitted_direction', 'body_position', 'emitted_direction'),
                    ('requested_to_emitted_direction', 'requested_direction', 'emitted_direction'),
                    ('body_velocity_to_field_velocity', 'body_velocity', 'field_velocity'),
                    ('body_velocity_to_requested_velocity', 'body_velocity', 'requested_velocity'),
                    ('body_velocity_to_emitted_velocity', 'body_velocity', 'emitted_velocity'),
                    ('requested_velocity_to_emitted_velocity', 'requested_velocity', 'emitted_velocity')]:
                entry['recurrence_and_timing_nulls'][label] = (
                    timing_controls(stages[source][window], stages[target][window], 'velocity' in source)
                    if target in stages else None)
            result['within_song'][song] = entry
            sampled[song] = {mode: {name: interpolate(times, values, q[song]) for name, values in stages.items()} for mode, q in queries.items()}
        for mode in queries:
            a, b = sampled['shpongle'][mode], sampled['cure_for_me'][mode]
            result['cross_song'][mode] = {name: similarity(a[name], b[name]) for name in a if name in b}
        report['algorithms'][algorithm] = result
        print('ANCILLARY_ALGORITHM', algorithm, flush=True)
    # Controls must score inactivity as undefined, never perfect similarity.
    constant_scores = [cell['constant_output_recurrence'] for a in report['algorithms'].values()
        for song in a['within_song'].values() for cell in song['recurrence_and_timing_nulls'].values() if cell is not None]
    report['constant_controls_all_undefined'] = all(v is None for v in constant_scores)
    if not report['constant_controls_all_undefined']:
        raise RuntimeError('Constant-output control produced a recurrence score')
    for p, h in {**protocol['files'], **protocol['main_verdicts']}.items():
        verified(p, h)
    report['inputs_unchanged_after_analysis'] = True
    report['main_verdicts_unchanged'] = True
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    output = root / 'records/ANCILLARY_TIMING_REPORT.json'
    save(output, report)
    print(json.dumps({'report': str(output), 'sha256': sha(output)}), flush=True)


def selftest():
    t = np.arange(960) / 8
    # Deliberately nonperiodic source trajectory to avoid a shift equaling its motif.
    x = np.stack((np.sin(.15 * t + .003 * t ** 2), np.cos(.41 * t)), axis=-1)[:, None]
    translated = 4 * x + np.array([[[19., -8.]]])
    assert similarity(x, translated)['centered_normalized_cosine'] > .999999
    controls = timing_controls(x, translated)
    assert controls['aligned_recurrence'] > .999999
    assert controls['shifted_source_recurrence_wrap_excluded'] < .9
    assert controls['reversed_source_recurrence'] < .9
    assert controls['constant_output_recurrence'] is None
    assert similarity(x, np.ones_like(x))['centered_normalized_cosine'] is None
    level = np.array([[.8, .08], [.8, .08]])
    br = bounds_report(level, .08, .8)
    assert br['fraction_near_design_upper'] == .5
    assert br['fraction_near_fullscale_one'] == 0
    print('ANCILLARY_SYNTHETIC_CONTROLS_PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('selftest', 'freeze', 'run'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.stage == 'selftest':
        selftest()
    else:
        {'freeze': freeze, 'run': run}[args.stage](args.root)

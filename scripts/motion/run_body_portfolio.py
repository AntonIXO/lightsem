"""Freeze and evaluate fifteen body-to-light algorithms without GPU work.

Run freeze from the workspace after review. All later stages must execute from
the copied source tree: cd ROOT/source; PYTHONPATH=ROOT/source python -m
scripts.motion.run_body_portfolio STAGE --root ROOT. Completed artifacts are
hash-verified and reused; partial or failed attempts are preserved, never
silently regenerated. This script never generates a new dancer or palette.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

# Enforce the campaign's small CPU footprint before NumPy/BLAS imports.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                  'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_variable] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np

from lightsem.conductor.body_transfer import (TransferConfig, build_outputs,
    sample_body, raw_transfer, compile_intent, decode)
from lightsem.conductor.body_transfer_v2 import CATALOG, make_outputs, map_body
from lightsem.conductor.body_transfer_v2.diagnostics import measure
from lightsem.conductor.body_transfer_v2.skeleton import PARENTS


WORKSPACE = Path('/root/lightsem')
DEFAULT_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v2')
REGIONS = {
    'head': {'pivot': 12, 'joints': [15], 'axis': [0., 0., 1.]},
    'torso': {'pivot': 0, 'joints': [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], 'axis': [0., 1., 0.]},
    'left_arm': {'pivot': 16, 'joints': [18, 20], 'axis': [0., 0., 1.]},
    'right_arm': {'pivot': 17, 'joints': [19, 21], 'axis': [0., 0., 1.]},
    'left_leg': {'pivot': 1, 'joints': [4, 7, 10], 'axis': [1., 0., 0.]},
    'right_leg': {'pivot': 2, 'joints': [5, 8, 11], 'axis': [1., 0., 0.]},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    """Exclusive writes and strict JSON preserve failures and prevent reruns."""
    serialized = json.dumps(value, indent=2, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(serialized)
        stream.write('\n')


def guard(root, plan):
    if shutil.disk_usage(root).free < plan.get('minimum_free_bytes', plan.get('min_free_bytes', 20 * 2 ** 30)):
        raise RuntimeError('Campaign requires at least 20 GiB free; existing outputs preserved')
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if size > plan.get('max_bytes', 2 * 2 ** 30):
        raise RuntimeError('Campaign exceeded 2 GiB; existing outputs preserved')


def verify_files(plan):
    if not plan.get('files'):
        raise ValueError('Missing frozen source/data manifest')
    for filename, expected in plan['files'].items():
        if sha(filename) != expected:
            raise RuntimeError('Frozen input changed: ' + filename)


def require_frozen_imports(plan):
    source = Path(plan['source_root']).resolve()
    if not Path(__file__).resolve().is_relative_to(source):
        raise RuntimeError('Execute this stage from ROOT/source with PYTHONPATH=ROOT/source')
    for name, module in list(sys.modules.items()):
        if name == 'lightsem' or name.startswith(('lightsem.', 'scripts.')):
            filename = getattr(module, '__file__', None)
            if filename and not Path(filename).resolve().is_relative_to(source):
                raise RuntimeError('Non-frozen implementation imported: ' + name + ': ' + filename)


def freeze(root, workspace=WORKSPACE):
    """Pin copied executable sources, canonical P2 policy, and both saved dances."""
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'PLAN.json').exists() or (root / 'source').exists() or (root / 'source_manifest.json').exists():
        raise RuntimeError('Freeze already started/completed; preserve it and use a new revision')
    limits = {'max_bytes': 2 * 2 ** 30, 'minimum_free_bytes': 20 * 2 ** 30,
              'min_free_bytes': 20 * 2 ** 30}
    guard(root, limits)
    old = read(workspace / 'experiments/2026-09-30-body-transfer/records/DARK_STAGE_PLAN_V1.json')
    tracks, data_files = [], set()
    for existing in old['tracks']:
        folder = workspace / 'runs/body_transfer_20260930_v1' / existing['id']
        track = {'id': existing['id'], 'title': existing['title'], 'audio': existing['audio'],
            'audio_reference': str(folder / 'audio_reference.m4a'),
            'body': str(folder / 'body.npz'), 'baseline_timeline': str(folder / 'timeline.npz'),
            'baseline_patch': str(folder / 'patch.json'),
            'body_provenance': str(folder / 'BODY_PROVENANCE.json')}
        tracks.append(track)
        data_files.update(track[k] for k in ('audio', 'audio_reference', 'body',
                                            'baseline_timeline', 'baseline_patch', 'body_provenance'))
    if {t['id'] for t in tracks} != {'shpongle', 'cure_for_me'}:
        raise RuntimeError('The frozen campaign requires both requested songs')
    data_files.add(str(root / 'records/VIBESCIENCE_REGISTRATION.json'))
    data_pins = {str(p): sha(p) for p in sorted(data_files)}
    with np.load(tracks[0]['body'], allow_pickle=False) as a, np.load(tracks[1]['body'], allow_pickle=False) as b:
        if np.array_equal(a['world'], b['world']):
            raise RuntimeError('Both songs must have their own saved source body')
    sources = set()
    for folder in ('lightsem', 'scripts'):
        sources.update(p for p in (workspace / folder).rglob('*')
                       if p.is_file() and p.suffix in ('.py', '.html') and '__pycache__' not in p.parts)
    for pattern in ('test_body_transfer*.py', 'test_body_portfolio*.py'):
        sources.update((workspace / 'tests').glob(pattern))
    sources.add(workspace / 'configs/light_layers/baseline.json')
    for filename in ('README.md', 'records/RESEARCH_TRACE.json', 'records/PERPLEXITY_RESEARCH.md'):
        sources.add(workspace / 'experiments/2026-10-04-body-transfer' / filename)
    source_root = root / 'source'
    source_root.mkdir()
    source_pins, origins = {}, {}
    for origin in sorted(sources):
        dest = source_root / origin.relative_to(workspace)
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(origin)
        shutil.copyfile(origin, dest)
        if sha(dest) != expected or sha(origin) != expected:
            raise RuntimeError('Source changed during freeze: ' + str(origin))
        source_pins[str(dest)] = expected
        origins[str(dest)] = str(origin)
    manifest_path = root / 'source_manifest.json'
    save(manifest_path, {'source_root': str(source_root), 'files': source_pins,
        'frozen_paths': sorted(source_pins), 'origins': origins,
        'scope': 'Hashes bind the actual copied modules used at execution, not mutable workspace files.'})
    plan = {'id': root.name, 'output': str(root), 'source_root': str(source_root),
        'source_manifest': str(manifest_path), 'tracks': tracks,
        'algorithms': [{'id': key, 'name': value[0],
                        'papers': ['https://arxiv.org/abs/' + p for p in value[1]]}
                       for key, value in sorted(CATALOG.items())],
        'counts': {'beams': 8, 'lasers': 5, 'bulbs': 50},
        'prefix_s': 180, 'review_start_s': 60, 'review_duration_s': 120,
        'fps': 40., 'body_fps': 30., 'transfer': asdict(TransferConfig()),
        'color_baseline': str(source_root / 'configs/light_layers/baseline.json'),
        'preflight': {'frames': 101, 'angle_degrees': 12., 'regions': REGIONS,
                      'scope': 'Explicit synthetic body inputs are diagnostic interventions only; never demo motion.'},
        'gate': {'minimum_crossing_fraction_drop': .10, 'source_coverage_at_least_v1': True,
                 'scope': 'Engineering selection gate; neither choreography nor user acceptance.'},
        **limits, 'files': {**source_pins, **data_pins, str(manifest_path): sha(manifest_path)},
        'scope': 'Fifteen body transfer candidates on both frozen per-song dances; P2 RGB replayed exactly; no GPU or new dance generation.'}
    save(root / 'PLAN.json', plan)
    verify_files(plan)
    guard(root, plan)
    print('FROZEN', root, len(plan['files']), flush=True)
    return plan


def selected_tracks(plan, track=None):
    result = [t for t in plan['tracks'] if track is None or t['id'] == track]
    if not result:
        raise ValueError('Unknown track: ' + str(track))
    return result


def pairs(plan, algorithm=None, track=None):
    selected = [a for a in plan['algorithms'] if algorithm is None or a['id'] == algorithm]
    if not selected:
        raise ValueError('Unknown algorithm: ' + str(algorithm))
    for method in selected:
        for song in selected_tracks(plan, track):
            yield method, song


def input_pins(root, plan, track):
    return {'plan_sha256': sha(root / 'PLAN.json'),
            'source_manifest_sha256': sha(plan['source_manifest']),
            'body_sha256': sha(track['body']),
            'body_provenance_sha256': sha(track['body_provenance']),
            'P2_timeline_sha256': sha(track['baseline_timeline']),
            'P2_patch_sha256': sha(track['baseline_patch']),
            'audio_sha256': sha(track['audio'])}


def load_source(plan, track):
    with np.load(track['body'], allow_pickle=False) as data:
        world = data['world'].copy()
    required = round(plan['prefix_s'] * plan['body_fps'])
    if world.shape != (required, 22, 3) or not np.isfinite(world).all():
        raise ValueError('Saved body must contain the exact finite frozen 180-second prefix')
    times, body = sample_body(world, plan['body_fps'], plan['fps'])
    with np.load(track['baseline_timeline'], allow_pickle=False) as data:
        dmx = data['universes'].copy()
        saved_body = data['body'].copy()
        saved_time = data['time'].copy()
    if not np.array_equal(body, saved_body) or not np.array_equal(times, saved_time):
        raise RuntimeError('Body resampling differs from the accepted saved baseline')
    patch = read(track['baseline_patch'])
    if len(dmx) != len(body):
        raise ValueError('Saved P2 length differs from frozen source body')
    rgb = []
    for left_side in (True, False):
        group = [p for p in patch if p['kind'] != 'bulb' and (p['position'][0] < 0) == left_side]
        if not group:
            raise ValueError('Baseline patch must expose both P2 spatial groups')
        first = group[0]
        slot = first['address'] - 1 + 5
        color = dmx[:, first['universe'], slot:slot + 3].copy()
        for fixture in patch:
            if (fixture['position'][0] < 0) != left_side:
                continue
            color_offset = 1 if fixture['kind'] == 'bulb' else 5
            start = fixture['address'] - 1 + color_offset
            if not np.array_equal(color, dmx[:, fixture['universe'], start:start + 3]):
                raise RuntimeError('Baseline does not use identical P2 RGB inside its declared spatial group')
        rgb.append(color)
    return times, body, rgb[0], rgb[1], dmx, patch


def hard_checks(dmx, patch, intent, fps, *, enforce_motion_limits=True):
    if not all(np.isfinite(intent[key]).all() for key in ('targets', 'dimmer')):
        raise RuntimeError('Nonfinite mapper intent')
    controls = decode(dmx, patch)
    if not all(np.isfinite(value).all() for value in controls.values()):
        raise RuntimeError('Nonfinite emitted controls')
    moving = np.array([p['kind'] != 'bulb' for p in patch])
    angles = controls['angles_deg'][:, moving]
    speed = np.linalg.norm(np.diff(angles, axis=0) * fps, axis=-1)
    acceleration = np.linalg.norm(np.diff(angles, n=2, axis=0) * fps ** 2, axis=-1)
    vmax = float(speed.max()) if speed.size else 0.
    amax = float(acceleration.max()) if acceleration.size else 0.
    if enforce_motion_limits and (vmax > 90. + 1e-6 or amax > 360. + 1e-6):
        raise RuntimeError('Emitted mechanical limits failed')
    return {'finite': True, 'maximum_speed_deg_s': vmax, 'maximum_acceleration_deg_s2': amax, 'mechanical_limits_enforced': enforce_motion_limits}


def baseline(root, plan, track):
    path = root / 'baseline' / track['id'] / 'report.json'
    pins = input_pins(root, plan, track)
    if path.exists():
        previous = read(path)
        if previous.get('inputs') != pins:
            raise RuntimeError('Existing baseline report changed inputs')
        return previous
    times, body, _, _, dmx, patch = load_source(plan, track)
    intent = raw_transfer(body, patch, TransferConfig(**plan['transfer']))
    numeric = measure(body, intent, dmx, patch, plan['fps'], plan['review_start_s'], plan['review_duration_s'])
    result = {'algorithm': 'v1', 'track': track['id'], 'status': 'diagnostic', 'inputs': pins,
              'diagnostics': numeric, 'hard_checks': hard_checks(dmx, patch, intent, plan['fps']),
              'frames': len(times), 'scope': 'Saved rejected-overlap baseline measured without changing original files.'}
    save(path, result)
    return result


def intervention(pose, region, frames=101, degrees=12.):
    """Explicit raised/held/returned body rotation; never a demo motion source."""
    if frames != 101:
        raise ValueError('Frozen diagnostic design uses 101 frames')
    spec = REGIONS[region]
    envelope = np.zeros(frames)
    u = np.linspace(0., 1., 21)
    smooth = u * u * (3 - 2 * u)
    envelope[20:41] = smooth
    envelope[40:61] = 1.
    envelope[60:81] = smooth[::-1]
    angle = np.deg2rad(degrees) * envelope
    axis = np.asarray(spec['axis'])
    result = np.broadcast_to(pose, (frames, 22, 3)).copy()
    center = pose[spec['pivot']]
    vector = pose[spec['joints']] - center
    rotated = (vector[None] * np.cos(angle)[:, None, None]
               + np.cross(axis, vector)[None] * np.sin(angle)[:, None, None]
               + axis[None, None] * (vector @ axis)[None, :, None] * (1 - np.cos(angle))[:, None, None])
    result[:, spec['joints']] = center + rotated
    # Exact source holds avoid trig(0) round-off masquerading as movement.
    result[envelope == 0] = pose
    return result


def angular_ticks(dmx, patch):
    values = []
    for fixture in patch:
        if fixture['kind'] == 'bulb':
            continue
        a = fixture['address'] - 1
        raw = dmx[:, fixture['universe'], a:a + 4].astype(np.int64)
        values.append(raw[:, [0, 2]] * 256 + raw[:, [1, 3]])
    return np.stack(values, 1) if values else np.zeros((len(dmx), 0, 2), dtype=np.int64)


def _compile(body, outputs, algorithm, left, right, config):
    intent = raw_transfer(body, outputs, config) if algorithm == 'v1' else map_body(body, outputs, algorithm, config.fps)
    dmx, patch, intent = compile_intent(intent, outputs, left, right, config)
    hard = hard_checks(dmx, patch, intent, config.fps)
    return dmx, patch, intent, hard


def preflight(root, plan, track, algorithm):
    path = root / 'preflight' / algorithm / (track['id'] + '.json')
    pins = input_pins(root, plan, track)
    if path.exists():
        previous = read(path)
        if previous.get('inputs') != pins or previous.get('status') == 'failed':
            raise RuntimeError('Existing failed/changed preflight preserved: ' + str(path))
        return previous
    result = {'algorithm': algorithm, 'track': track['id'], 'inputs': pins, 'status': 'started',
              'regions': {}, 'scope': 'Explicit source counterfactuals; motion response is evaluated after DMX compilation.'}
    try:
        _, source, left, right, _, _ = load_source(plan, track)
        frames = plan['preflight']['frames']
        middle = len(source) // 2
        pose = source[middle]
        body = np.repeat(pose[None], frames, axis=0)
        colors = [np.repeat(c[middle:middle + 1], frames, axis=0) for c in (left, right)]
        outputs = (build_outputs if algorithm == 'v1' else make_outputs)(**plan['counts'])
        config = TransferConfig(**plan['transfer'])
        dmx, patch, intent, hard = _compile(body, outputs, algorithm, *colors, config)
        exact = bool(np.all(dmx == dmx[:1]))
        result.update({'frozen_body_DMX_exact': exact, 'frozen_hard_checks': hard,
                       'frozen_DMX_sha256': array_sha(dmx), 'source_pose_index': middle,
                       'source_pose_sha256': array_sha(pose), 'fixed_RGB_sha256': [array_sha(c) for c in colors]})
        if not exact:
            raise RuntimeError('Frozen body produces autonomous emitted motion/intensity')
        original_ticks = angular_ticks(dmx, patch)
        original_levels = decode(dmx, patch)['dimmer']
        for region in REGIONS:
            changed = intervention(pose, region, frames, plan['preflight']['angle_degrees'])
            perturbed, ppatch, raw, limits = _compile(changed, outputs, algorithm, *colors, config)
            tick_delta = np.abs(angular_ticks(perturbed, ppatch) - original_ticks)
            levels = np.abs(decode(perturbed, ppatch)['dimmer'] - original_levels)
            response = np.any(tick_delta >= 1, axis=(0, 2))
            intensity = np.any(levels >= .5 / 255., axis=0)
            result['regions'][region] = {'angular_response': bool(response.any()),
                'moving_output_response_count': int(response.sum()),
                'moving_output_response_names': [p['name'] for p, yes in zip([p for p in patch if p['kind'] != 'bulb'], response) if yes],
                'maximum_angle_change_ticks': int(tick_delta.max()) if tick_delta.size else 0,
                'intensity_response_count': int(intensity.sum()), 'maximum_intensity_change': float(levels.max()),
                'body_input_sha256': array_sha(changed), 'emitted_DMX_sha256': array_sha(perturbed),
                'hard_checks': limits,
                'frozen_prefix_interior_max_tick_difference': int(tick_delta[:10].max()) if tick_delta.size else 0,
                'frozen_suffix_interior_max_tick_difference': int(tick_delta[-10:].max()) if tick_delta.size else 0,
                'settling_scope': 'First/last 10 frames of 20-frame frozen boundaries; offline fitting may anticipate/settle.'}
        coverage = sum(r['angular_response'] for r in result['regions'].values()) / len(REGIONS)
        result.update({'emitted_source_region_coverage': coverage,
                       'status': 'passed' if coverage == 1 else 'diagnostic',
                       'response_threshold': 'At least one emitted 16-bit angular tick; brightness is counted separately.'})
        save(path, result)
        return result
    except Exception as error:
        result.update({'status': 'failed', 'error': type(error).__name__ + ': ' + str(error)})
        if not path.exists():
            save(path, result)
        raise


def verify_complete(folder, pins):
    report_path = folder / 'report.json'
    if not report_path.exists():
        raise RuntimeError('Incomplete output preserved; use a new revision: ' + str(folder))
    report = read(report_path)
    if report.get('inputs') != pins or report.get('status') not in ('engineering_valid', 'diagnostic'):
        raise RuntimeError('Existing output failed or used different inputs: ' + str(folder))
    for name, key in (('timeline.npz', 'timeline_sha256'), ('patch.json', 'patch_sha256')):
        if sha(folder / name) != report.get(key):
            raise RuntimeError('Completed output hash changed: ' + str(folder / name))
    if 'preflight' in report:
        for name, key in (('path', 'sha256'), ('baseline_path', 'baseline_sha256')):
            if sha(report['preflight'][name]) != report['preflight'][key]:
                raise RuntimeError('Completed output preflight changed: ' + report['preflight'][name])
    return report


def generate(root, plan, algorithm, track):
    identifier = algorithm['id']
    folder = root / 'timelines' / identifier / track['id']
    pins = input_pins(root, plan, track)
    if folder.exists():
        return verify_complete(folder, pins)
    controls_path = root / 'preflight' / identifier / (track['id'] + '.json')
    baseline_path = root / 'preflight/v1' / (track['id'] + '.json')
    controls, old_controls = read(controls_path), read(baseline_path)
    for item in (controls, old_controls):
        if item['inputs'] != pins or item['status'] not in ('passed', 'diagnostic') or not item['frozen_body_DMX_exact']:
            raise RuntimeError('Successful frozen-source controls required before generation')
    folder.mkdir(parents=True, exist_ok=False)
    try:
        times, body, left, right, _, _ = load_source(plan, track)
        dmx, patch, intent, hard = _compile(body, make_outputs(**plan['counts']), identifier,
                                           left, right, TransferConfig(**plan['transfer']))
        diagnostics = measure(body, intent, dmx, patch, plan['fps'], plan['review_start_s'], plan['review_duration_s'])
        if not diagnostics['finite']:
            raise RuntimeError('Numeric diagnostics found nonfinite output')
        decoded = decode(dmx, patch)
        for i, fixture in enumerate(patch):
            expected = left if fixture['position'][0] < 0 else right
            if not np.array_equal(np.rint(decoded['rgb'][:, i] * 255).astype(np.uint8), expected):
                raise RuntimeError('Compiled output changed pinned P2 RGB')
        reasons = []
        if controls['emitted_source_region_coverage'] < 1:
            reasons.append('One or more anatomical source interventions had no angular response')
        if any(diagnostics['motion']['constant_by_output']):
            reasons.append('At least one moving output is constant during the review window')
        if diagnostics['duplicates']['desired_target_pair_count']:
            reasons.append('Exact desired target duplicates exist')
        structure_trace = {key:intent[key] for key in ('field_absolute_points', 'shared_stage_points',
            'shared_role_mask', 'shared_similarity_scale') if key in intent}
        np.savez_compressed(folder / 'timeline.npz', universes=dmx, body=body, time=times,
            targets=intent['targets'], source_points=intent['source_points'],
            field_points=intent['field_points'], dimmer=intent['dimmer'], **structure_trace)
        save(folder / 'patch.json', patch)
        report = {'algorithm': identifier, 'name': algorithm['name'], 'track': track['id'],
            'status': 'diagnostic' if reasons else 'engineering_valid', 'diagnostic_reasons': reasons,
            'inputs': pins, 'source_sha256': pins['source_manifest_sha256'],
            'body_sha256': pins['body_sha256'], 'P2_sha256': pins['P2_timeline_sha256'],
            'timeline_sha256': sha(folder / 'timeline.npz'), 'patch_sha256': sha(folder / 'patch.json'),
            'emitted_DMX_sha256': array_sha(dmx), 'emitted_motion_sha256': array_sha(angular_ticks(dmx, patch)),
            'emitted_brightness_sha256': array_sha(np.rint(decoded['dimmer'] * 255).astype(np.uint8)),
            'hard_invariants_passed': True,
            'hard_checks': hard, 'diagnostics': diagnostics, 'method': intent['method'],
            'preflight': {'path': str(controls_path), 'sha256': sha(controls_path),
                'baseline_path': str(baseline_path), 'baseline_sha256': sha(baseline_path),
                'frozen_body_DMX_exact': controls['frozen_body_DMX_exact'],
                'emitted_source_region_coverage': controls['emitted_source_region_coverage'],
                'v1_emitted_source_region_coverage': old_controls['emitted_source_region_coverage']},
            'frames': len(body), 'P2_RGB_exact': True,
            'scope': 'Computed before rendering. No threshold/seed changes or outcome-based rerun; engineering checks do not establish choreography.'}
        save(folder / 'report.json', report)
        return report
    except Exception as error:
        if not (folder / 'failure.json').exists():
            save(folder / 'failure.json', {'inputs': pins, 'algorithm': identifier, 'track': track['id'],
                 'status': 'failed', 'error': type(error).__name__ + ': ' + str(error)})
        raise


def summarize(root, plan):
    rows, duplicate_groups = [], {'DMX': {}, 'motion': {}}
    for algorithm, track in pairs(plan):
        report = verify_complete(root / 'timelines' / algorithm['id'] / track['id'], input_pins(root, plan, track))
        old = read(root / 'baseline' / track['id'] / 'report.json')
        current_crossing = report['diagnostics']['spatial']['crossing_fraction']
        baseline_crossing = old['diagnostics']['spatial']['crossing_fraction']
        drop = baseline_crossing - current_crossing
        coverage = report['preflight']['emitted_source_region_coverage']
        old_coverage = report['preflight']['v1_emitted_source_region_coverage']
        gate = drop >= plan['gate']['minimum_crossing_fraction_drop'] and coverage >= old_coverage
        row = {'algorithm': algorithm['id'], 'name': algorithm['name'], 'track': track['id'],
            'status': report['status'], 'baseline_crossing_fraction': baseline_crossing,
            'crossing_fraction': current_crossing, 'crossing_fraction_drop': drop,
            'source_region_coverage': coverage, 'v1_source_region_coverage': old_coverage,
            'engineering_selection_gate': bool(gate), 'diagnostic_reasons': report['diagnostic_reasons'],
            'limits': report['hard_checks'], 'report': str(root / 'timelines' / algorithm['id'] / track['id'] / 'report.json')}
        rows.append(row)
        for kind, key in (('DMX', 'emitted_DMX_sha256'), ('motion', 'emitted_motion_sha256')):
            duplicate_groups[kind].setdefault(report[key], []).append(algorithm['id'] + '/' + track['id'])
    duplicates = {kind: [group for group in groups.values() if len(group) > 1]
                  for kind, groups in duplicate_groups.items()}
    both = [a['id'] for a in plan['algorithms'] if all(row['engineering_selection_gate']
            for row in rows if row['algorithm'] == a['id'])]
    result = {'plan_sha256': sha(root / 'PLAN.json'), 'rows': rows, 'duplicates': duplicates,
        'both_song_gate_passes': both,
        'scope': 'Absolute crossing drop >=0.10 and source coverage >=V1; numerical selection only, no artistic claim.',
        'limitations': ['Geometric crossings are not volumetric overlap or choreography.',
                       'Distinct streams and nonzero activity do not prove independent gestures.',
                       'P2 color weaknesses remain visible; no events were injected.',
                       'This is a fixed two-song mapper comparison, not music-conditioning model qualification.',
                       'Temporal recurrence/extrema are proxies, not annotated phrase or motif judgments.']}
    path = root / 'COMPARISON.json'
    if path.exists():
        if read(path) != result:
            raise RuntimeError('Existing comparison differs; preserve it and use a new revision')
        return result
    save(path, result)
    lines = ['# Body transfer portfolio comparison', '', result['scope'], '',
             '| Algorithm | Song | Crossing | Drop from V1 | Source coverage | Gate | Status |',
             '|---|---|---:|---:|---:|---|---|']
    for row in rows:
        lines.append(f"| {row['algorithm']} {row['name']} | {row['track']} | {row['crossing_fraction']:.4f} | "
                     f"{row['crossing_fraction_drop']:.4f} | {row['source_region_coverage']:.3f} | "
                     f"{'pass' if row['engineering_selection_gate'] else 'fail'} | {row['status']} |")
    lines += ['', 'Methods passing the numerical gate on both songs: ' + (', '.join(both) or 'none') + '.', '',
              'Exact duplicate groups: ' + json.dumps(duplicates), '',
              *('- ' + limitation for limitation in result['limitations'])]
    with (root / 'COMPARISON.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'baseline', 'preflight', 'generate', 'summarize'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--algorithm')
    parser.add_argument('--track', choices=('shpongle', 'cure_for_me'))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == 'freeze':
        if args.algorithm or args.track:
            raise ValueError('Freeze always pins the complete two-song fifteen-method campaign')
        freeze(root)
        return
    plan = read(root / 'PLAN.json')
    require_frozen_imports(plan)
    verify_files(plan)
    guard(root, plan)
    if args.stage == 'summarize':
        if args.algorithm or args.track:
            raise ValueError('Summary compares all thirty frozen candidates')
        summarize(root, plan)
    elif args.stage == 'baseline':
        for track in selected_tracks(plan, args.track):
            baseline(root, plan, track)
            print('BASELINE', track['id'], flush=True)
    else:
        if args.stage == 'preflight':
            for track in selected_tracks(plan, args.track):
                preflight(root, plan, track, 'v1')
                print('PREFLIGHT', 'v1', track['id'], flush=True)
        for algorithm, track in pairs(plan, args.algorithm, args.track):
            verify_files(plan)
            guard(root, plan)
            if args.stage == 'preflight':
                result = preflight(root, plan, track, algorithm['id'])
            else:
                result = generate(root, plan, algorithm, track)
            require_frozen_imports(plan)
            print(args.stage.upper(), algorithm['id'], track['id'], result['status'], flush=True)
    require_frozen_imports(plan)
    verify_files(plan)
    guard(root, plan)


if __name__ == '__main__':
    main()

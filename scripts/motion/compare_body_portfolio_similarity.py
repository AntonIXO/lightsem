"""Ancillary similarity audit of completed, immutable body-transfer outputs.

Freeze the small descriptive protocol before run. This script only writes its
own protocol/report, uses the frozen DMX decoder, and never generates motion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import numpy as np

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v2')
PROTOCOL_NAME = 'ANCILLARY_SIMILARITY_PROTOCOL.json'
REPORT_NAME = 'ANCILLARY_SIMILARITY_REPORT.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(encoded)


def verify(path, expected):
    if sha(path) != expected:
        raise RuntimeError('Frozen input identity mismatch: ' + str(path))


def normalize(values):
    """One temporal mean per XYZ coordinate, one total XYZ norm per fixture."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('Expected finite (time, fixtures, XYZ) waveforms')
    # Subtract first for exact constant handling, then subtract temporal mean.
    centered = values - values[:1]
    centered -= centered.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered**2, axis=(0, 2)))
    active = norms > 1e-12
    normalized = centered / np.maximum(norms, 1e-12)[None, :, None]
    return normalized, active, norms


def angular_velocity(directions, fps):
    """Shortest sphere rotation per frame, world XYZ rad/s, no observable roll.

    omega = atan2(||u_prev cross u||, u_prev dot u) / dt * axis.
    Exact repeats have zero velocity. Antipodal directions are undefined and
    rejected instead of silently choosing a motion axis.
    """
    previous = np.concatenate([directions[:1], directions[:-1]], axis=0)
    cross = np.cross(previous, directions)
    cross_norm = np.linalg.norm(cross, axis=-1)
    dot = np.clip(np.sum(previous * directions, axis=-1), -1.0, 1.0)
    if np.any((cross_norm <= 1e-12) & (dot < 0)):
        raise ValueError('Antipodal direction step has undefined angular velocity axis')
    angle = np.arctan2(cross_norm, dot)
    scale = np.divide(angle * fps, cross_norm, out=np.zeros_like(angle), where=cross_norm > 1e-12)
    return cross * scale[..., None]


def as_nullable(matrix, defined):
    return [[float(value) if valid else None for value, valid in zip(row, mask)]
            for row, mask in zip(matrix, defined)]


def ranked_pairs(matrix, defined, names, count=12, absolute=False):
    pairs = [{'first': names[i], 'second': names[j], 'signed_cosine': float(matrix[i, j]),
              'absolute_cosine': float(abs(matrix[i, j]))}
             for i in range(len(names)) for j in range(i + 1, len(names)) if defined[i, j]]
    pairs.sort(key=lambda p: (-p['absolute_cosine'] if absolute else -p['signed_cosine'],
                             p['first'], p['second']))
    return pairs[:count]


def within_method(values, names):
    normalized, active, norms = normalize(values)
    matrix = np.einsum('tix,tjx->ij', normalized, normalized)
    matrix = np.clip(matrix, -1., 1.)
    defined = active[:, None] & active[None, :]
    upper = matrix[np.triu(defined, 1)]
    return {
        'fixture_order': names,
        'signed_cosine_matrix': as_nullable(matrix, defined),
        'active_fixtures': int(active.sum()),
        'undefined_fixture_names': [name for name, keep in zip(names, active) if not keep],
        'total_centered_temporal_xyz_norm_per_fixture': norms.tolist(),
        'defined_pair_count': int(len(upper)),
        'signed_cosine_pair_quantiles_p05_p50_p95': np.quantile(upper, [.05, .5, .95]).tolist() if len(upper) else None,
        'top_signed_similar_pairs': ranked_pairs(matrix, defined, names),
        'top_absolute_similar_pairs': ranked_pairs(matrix, defined, names, absolute=True),
    }, (normalized, active)


def between_methods(waveforms, method_names, fixture_names):
    count = len(method_names)
    matrix = np.zeros((count, count))
    defined = np.zeros((count, count), dtype=bool)
    coverage = np.zeros((count, count), dtype=int)
    per_fixture = {}
    for i, first in enumerate(method_names):
        a, active_a = waveforms[first]
        for j in range(i, count):
            second = method_names[j]
            b, active_b = waveforms[second]
            active = active_a & active_b
            scores = np.clip(np.einsum('tix,tix->i', a, b), -1., 1.)
            coverage[i, j] = coverage[j, i] = int(active.sum())
            if active.any():
                matrix[i, j] = matrix[j, i] = float(scores[active].mean())
                defined[i, j] = defined[j, i] = True
            if i != j:
                per_fixture[first + '__' + second] = [float(s) if keep else None for s, keep in zip(scores, active)]
    return {
        'method_order': method_names,
        'fixture_order': fixture_names,
        'aggregation': 'Mean signed cosine over matching fixtures defined in both methods; equal fixture weight.',
        'signed_cosine_matrix': as_nullable(matrix, defined),
        'matching_defined_fixture_count_matrix': coverage.tolist(),
        'per_matching_fixture_signed_cosine': per_fixture,
        'top_signed_similar_method_pairs': ranked_pairs(matrix, defined, method_names),
        'top_absolute_similar_method_pairs': ranked_pairs(matrix, defined, method_names, absolute=True),
    }


def exact_pairs(values, names):
    return [[names[i], names[j]] for i in range(len(names)) for j in range(i + 1, len(names))
            if np.array_equal(values[:, i], values[:, j])]


def selftest():
    t = np.array([-2., -1., 0., 1., 3., 4.])
    source = np.column_stack([t, t**2, t**3])[:, None]
    shifted_scaled = 3.5 * source + np.array([11., -18., 4.])
    normalized, active, _ = normalize(np.concatenate([source, shifted_scaled, -source, np.ones_like(source)], axis=1))
    assert active.tolist() == [True, True, True, False]
    np.testing.assert_allclose(normalized[:, 0], normalized[:, 1], atol=1e-12)
    result, _ = within_method(np.concatenate([source, shifted_scaled, -source, np.ones_like(source)], axis=1), ['a', 'b', 'c', 'constant'])
    np.testing.assert_allclose(result['signed_cosine_matrix'][0][:3], [1., 1., -1.], atol=1e-12)
    assert result['signed_cosine_matrix'][0][3] is None
    angle = np.arange(8) * .01
    directions = np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)])[:, None]
    omega = angular_velocity(directions, 40.)
    np.testing.assert_allclose(omega[1:, 0], np.tile([0., 0., .4], (7, 1)), atol=1e-12)
    fixed = np.repeat(directions[:1], 20, axis=0)
    np.testing.assert_array_equal(angular_velocity(fixed, 40.), np.zeros_like(fixed))
    assert not normalize(fixed)[1].any()


def freeze(root):
    selftest()
    plan = read(root / 'PLAN.json')
    timelines = []
    paths = {Path(__file__).resolve(), root / 'PLAN.json', root / 'source_manifest.json'}
    for track in plan['tracks']:
        timelines.append({'method': 'V1', 'song': track['id'], 'timeline': track['baseline_timeline'],
                          'patch': track['baseline_patch'], 'existing_report': str(root / 'baseline' / track['id'] / 'report.json')})
        for key in ('body', 'body_provenance', 'audio', 'baseline_timeline', 'baseline_patch'):
            path = Path(track[key])
            if str(path) in plan['files']:
                verify(path, plan['files'][str(path)])
            paths.add(path)
        for algorithm in plan['algorithms']:
            folder = root / 'timelines' / algorithm['id'] / track['id']
            timelines.append({'method': algorithm['id'], 'song': track['id'],
                              'timeline': str(folder / 'timeline.npz'), 'patch': str(folder / 'patch.json'),
                              'existing_report': str(folder / 'report.json')})
    for row in timelines:
        paths.update(Path(row[key]) for key in ('timeline', 'patch', 'existing_report'))
        if row['method'] != 'V1':
            receipt = read(row['existing_report'])
            verify(row['timeline'], receipt['timeline_sha256'])
            verify(row['patch'], receipt['patch_sha256'])
    decoder = Path(plan['source_root']) / 'lightsem/conductor/body_transfer.py'
    verify(decoder, plan['files'][str(decoder)])
    paths.add(decoder)
    preserved = {str(root / name): sha(root / name) for name in (
        'COMPARISON.json', 'COMPARISON.md', 'records/VIBESCIENCE_VERDICTS.json')}
    protocol = {
        'id': 'body-transfer-20261004-ancillary-similarity-v1',
        'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Descriptive comparison of completed outputs; no new generation, pass/fail threshold, or scientific verdict.',
        'source_script': str(Path(__file__).resolve()), 'source_script_sha256': sha(__file__),
        'files': {str(path): sha(path) for path in sorted(paths)},
        'preserved_existing_reports_and_verdicts': preserved,
        'timelines': timelines,
        'methods_between': [a['id'] for a in plan['algorithms']],
        'methods_within': ['V1'] + [a['id'] for a in plan['algorithms']],
        'method_names': {'V1': 'Saved body-transfer V1', **{a['id']: a['name'] for a in plan['algorithms']}},
        'songs': [t['id'] for t in plan['tracks']],
        'window_s': [60., 180.], 'fps': 40., 'expected_frames': 4800, 'expected_moving_fixtures': 13,
        'direction_signal': 'World XYZ unit directions decoded from saved emitted DMX using frozen decoder.',
        'velocity_signal': 'World XYZ minimal angular velocity rad/s = atan2(norm(u_prev cross u), dot(u_prev,u))*40*unit(u_prev cross u); computed on entire prefix before selecting window. Repeated directions yield zero; antipodal steps rejected. Beam roll is unobservable.',
        'normalization': 'For each fixture separately subtract its temporal mean XYZ vector, then divide all its centered XYZ samples by one total temporal XYZ L2 norm. No per-axis normalization, rotation fitting, lag search, speed warping or fixture permutation.',
        'undefined': 'A centered norm <=1e-12 is numerical zero and excluded; undefined similarities are JSON null, never rewarded. This is a numerical validity rule, not a collapse or acceptance threshold.',
        'within_method': '13x13 signed cosine matrix over same-time normalized waveforms; all unique pairs; top12 signed and absolute pairs; quantiles and raw centered norms.',
        'between_methods': '15x15 matrix of mean same-fixture signed cosine; equal weight among fixtures defined in both methods, coverage reported. Also retain all105 pairs by13 fixture scores and top12 signed/absolute method pairs. V1 participates only in within-method comparisons.',
        'exact_duplicates': 'Separately report exact array-equal requested target pairs, decoded direction pairs, decoded angle pairs and angular-velocity pairs, within window, without tolerance or offset normalization.',
        'synthetic_controls': 'Before freezing: affine offset/positive gain invariance, negative inversion=-1, constants undefined, known angular rotation=0.4rad/s, repeated direction angular velocity=0.',
        'interpretation_limits': [
            'High motion correlation can legitimately reflect related source limbs or shared gesture timing; it alone does not establish collapse.',
            'Removing offsets and positive scalar amplitude reveals waveform similarity but discards spatial layout and motion magnitude; physical norms are retained.',
            'Signed and absolute similarities are different: opposing gestures may be coordinated and should not be called identical.',
            'Different methods are inspired adaptations; numeric nonidentity does not establish distinct visible improvement.',
            'World-coordinate orientation remains part of comparison; no fitted rotation aligns beams at different stage locations.',
            'Moving fixture motion only: color, brightness, bulb patterns, occlusion, camera projection and human judgment are outside this metric.',
            'The protocol is ancillary after the campaign outputs existed; existing evidence and verdicts remain unchanged.',
            'No claim of independent dance sources or new choreography follows from these matrices.',
        ],
    }
    path = root / 'records' / PROTOCOL_NAME
    save_new(path, protocol)
    print(json.dumps({'protocol': str(path), 'sha256': sha(path), 'frozen_files': len(paths)}), flush=True)


def run(root):
    protocol_path = root / 'records' / PROTOCOL_NAME
    report_path = root / 'records' / REPORT_NAME
    if report_path.exists():
        raise FileExistsError('Existing ancillary report is immutable: ' + str(report_path))
    protocol = read(protocol_path)
    for path, expected in {**protocol['files'], **protocol['preserved_existing_reports_and_verdicts']}.items():
        verify(path, expected)
    verify(__file__, protocol['source_script_sha256'])
    plan = read(root / 'PLAN.json')
    sys.path.insert(0, plan['source_root'])
    from lightsem.conductor.body_transfer import decode
    if Path(sys.modules['lightsem.conductor.body_transfer'].__file__).resolve() != Path(plan['source_root']) / 'lightsem/conductor/body_transfer.py':
        raise RuntimeError('Expected the frozen decoder, not workspace implementation')
    report = {
        'protocol': str(protocol_path), 'protocol_sha256': sha(protocol_path),
        'source_script_sha256': sha(__file__), 'started_utc': datetime.now(timezone.utc).isoformat(),
        'scope': protocol['scope'], 'source_input_sha256': protocol['files'],
        'method_names': protocol['method_names'], 'normalization': protocol['normalization'],
        'interpretation_limits': protocol['interpretation_limits'], 'songs': {},
    }
    for song in protocol['songs']:
        records, signal_cache = {}, {'decoded_unit_direction': {}, 'angular_velocity_xyz_rad_s': {}}
        fixture_order = None
        for row in [r for r in protocol['timelines'] if r['song'] == song]:
            patch = read(row['patch'])
            moving = np.array([p['kind'] != 'bulb' for p in patch])
            names = [p['name'] for p in patch if p['kind'] != 'bulb']
            if len(names) != protocol['expected_moving_fixtures']:
                raise ValueError('Expected13 moving fixtures')
            if fixture_order is None:
                fixture_order = names
            elif names != fixture_order:
                raise ValueError('Fixture identity/order changed; no permutation is fitted')
            with np.load(row['timeline'], allow_pickle=False) as data:
                time = data['time']
                if not np.allclose(time, np.arange(len(time)) / protocol['fps'], rtol=0, atol=1e-10):
                    raise ValueError('Expected absolute 40Hz timeline')
                selected = (time >= 60.) & (time < 180.)
                if int(selected.sum()) != protocol['expected_frames']:
                    raise ValueError('Missing full60–180s evaluation window')
                decoded = decode(data['universes'], patch)
                directions_all = decoded['directions'][:, moving]
                if not np.allclose(np.linalg.norm(directions_all, axis=-1), 1., atol=1e-10):
                    raise ValueError('Decoded directions are not unit vectors')
                omega = angular_velocity(directions_all, protocol['fps'])[selected]
                directions = directions_all[selected]
                angles = decoded['angles_deg'][selected][:, moving]
                desired = data['targets'][selected][:, moving]
                first_time, last_time = float(time[selected][0]), float(time[selected][-1])
            signals = {'decoded_unit_direction': directions, 'angular_velocity_xyz_rad_s': omega}
            record = {'method': row['method'], 'source_timeline_sha256': protocol['files'][row['timeline']],
                      'window_first_s': first_time, 'window_last_s': last_time, 'frames': len(directions),
                      'fixture_order': names, 'signals': {},
                      'exact_duplicates_without_centering': {
                          'desired_target_pairs': exact_pairs(desired, names),
                          'decoded_unit_direction_pairs': exact_pairs(directions, names),
                          'decoded_pan_tilt_angle_pairs': exact_pairs(angles, names),
                          'angular_velocity_xyz_pairs': exact_pairs(omega, names)}}
            for name, signal in signals.items():
                summary, normalized = within_method(signal, names)
                record['signals'][name] = summary
                if row['method'] != 'V1':
                    signal_cache[name][row['method']] = normalized
            records[row['method']] = record
        between = {name: between_methods(waveforms, protocol['methods_between'], fixture_order)
                   for name, waveforms in signal_cache.items()}
        report['songs'][song] = {'within_method': records, 'between_methods': between}
        print(json.dumps({'song_complete': song,
            'top_direction_pairs': between['decoded_unit_direction']['top_signed_similar_method_pairs'][:3],
            'top_angular_velocity_pairs': between['angular_velocity_xyz_rad_s']['top_signed_similar_method_pairs'][:3]}), flush=True)
    for path, expected in {**protocol['files'], **protocol['preserved_existing_reports_and_verdicts']}.items():
        verify(path, expected)
    report['source_inputs_and_existing_reports_unchanged_after_analysis'] = True
    report['completed_utc'] = datetime.now(timezone.utc).isoformat()
    save_new(report_path, report)
    print(json.dumps({'report': str(report_path), 'sha256': sha(report_path)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('selftest', 'freeze', 'run'))
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.stage == 'selftest':
        selftest()
        print('Synthetic similarity controls passed.')
    elif args.stage == 'freeze':
        freeze(args.root)
    else:
        run(args.root)

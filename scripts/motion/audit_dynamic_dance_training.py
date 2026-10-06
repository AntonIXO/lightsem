"""Independent, CPU-only audit of frozen dynamic-dance training evidence.

This program reads completed artifacts. It never trains, regenerates outputs,
selects a checkpoint, or computes a Vibescience verdict. Its only write is an
exclusively created JSON output. Missing evidence is unqualified, not zero.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


ARMS = ('planner', 'continued', 'D7', 'D8', 'D8_flat')
FLOW_ARMS = ARMS[1:]
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20261005
RECOVERY_EXACT_FIELDS = ('valid_frames', 'gradient', 'lr', 'total', 'native', 'velocity',
                         'physical_position', 'physical_velocity', 'physical_planted_foot_velocity',
                         'physical_root_displacement', 'physical_floor_penetration')
REGIONS = ((0, 1, 2, 4, 5, 7, 8, 10, 11), (3, 6, 9, 12), (15,),
           (13, 16, 18, 20), (14, 17, 19, 21))


class EvidenceError(ValueError):
    """Missing or contradictory evidence; preserve all input artifacts."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def array_hash(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    require(not value.dtype.hasobject, 'Object arrays are not evidence')
    result = hashlib.sha256()
    result.update(json.dumps({'dtype': value.dtype.str, 'shape': value.shape}, sort_keys=True).encode())
    result.update(value.tobytes())
    return result.hexdigest()


def state_hash(value):
    """Stable nested checkpoint/RNG digest, not a pickle-container digest."""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return array_hash(value)
    if isinstance(value, dict):
        value = {str(k): state_hash(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [state_hash(v) for v in value]
    elif isinstance(value, np.generic):
        value = value.item()
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class Reader:
    def __init__(self):
        self.pins = {}

    def sha(self, path, expected=None):
        path = Path(path)
        require(path.is_file(), 'Missing file: ' + str(path))
        # Hash once per unchanged inode/size/mtime, including large input files.
        stat = path.stat()
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        prior = self.pins.get(str(path))
        if prior is None or prior['stamp'] != stamp:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            self.pins[str(path)] = {'sha256': h.hexdigest(), 'stamp': stamp}
        actual = self.pins[str(path)]['sha256']
        require(expected is None or actual == expected, 'SHA256 mismatch: ' + str(path))
        return actual

    def auth(self, path):
        sidecar = Path(str(path) + '.sha256.json')
        self.sha(sidecar)
        expected = json.loads(sidecar.read_text())['sha256']
        self.sha(path, expected)
        return json.loads(Path(path).read_text())

    def files(self, values):
        for path, expected in values.items():
            self.sha(path, expected)

    def checkpoint(self, path):
        self.sha(path)
        # These are hash-authenticated local campaign checkpoints, not uploads.
        return torch.load(path, map_location='cpu', weights_only=False)

    def npz(self, path):
        self.sha(path)
        with np.load(path, allow_pickle=False) as source:
            return {key: source[key].copy() for key in source.files}

    def artifact(self, item):
        self.sha(item['path'], item['sha256'])
        self.sha(item['metadata_path'], item['metadata_sha256'])
        receipt = self.auth(item['metadata_path'])
        require(receipt['kind'] == 'DynamicDanceEvaluation', 'Unknown evaluation artifact kind')
        require(receipt['npz_sha256'] == item['sha256'], 'Artifact receipt/container mismatch')
        arrays = self.npz(item['path'])
        hashes = {key: array_hash(value) for key, value in arrays.items()}
        require(hashes == item['arrays'] == receipt['identity']['arrays'],
                'Evaluation array digest mismatch: ' + item['path'])
        return arrays, receipt['identity']['metadata']


def _finite_loss(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0


def _interval(samples):
    valid = np.asarray(samples, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    undefined = len(samples) - len(valid)
    interval = np.quantile(valid, [.025, .975]).tolist() if len(valid) else None
    return {'draws': len(samples), 'defined_draws': len(valid), 'undefined_draws': undefined,
            'ci95': interval if not undefined else None,
            'conditional_defined_draws_ci95': interval if undefined else None}


def paired_estimate(rows, windows, comparator, field='flow_loss', square=False,
                    draws=BOOTSTRAP_DRAWS):
    """Equal canonical songs, equal recordings within song, explicit support.

    The primary is (mean_song(comparison)-mean_song(correct))/mean_song(comparison).
    The producer's distinct mean-of-recording-relative-gains is retained separately.
    """
    expected = {w['key']: w for w in windows}
    observed = {r['key']: r for r in rows}
    absent_shift = [w['key'] for w in windows if comparator == 'shift' and w['shift_start'] is None]
    eligible = [w for w in windows if w['key'] not in absent_shift]
    missing, groups, producer = [], defaultdict(list), defaultdict(list)
    if len(observed) != len(rows) or len(expected) != len(windows):
        missing.append('duplicate recording identities')
    if set(expected) != set(observed):
        missing.append('window coverage differs: expected=' + repr(sorted(expected)) + ', observed=' + repr(sorted(observed)))
    for window in eligible:
        row = observed.get(window['key'], {})
        if any(row.get(k) != v for k, v in window.items()):
            missing.append(window['key'] + ': frozen window identity differs')
            continue
        conditions = row.get('conditions', {})
        a = conditions.get('correct', {}).get(field)
        b = conditions.get(comparator, {}).get(field)
        if not (_finite_loss(a) and _finite_loss(b)):
            missing.append(window['key'] + ': missing/nonfinite paired ' + field)
            continue
        groups[window['canonical_id']].append((a * a, b * b) if square else (a, b))
        producer[window['canonical_id']].append((b-a)/max(abs(b), 1e-12))
    support = {'expected_recordings': len(eligible), 'observed_recordings': sum(map(len, groups.values())),
               'absent_shift_windows': absent_shift,
               'expected_canonical_songs': len({w['canonical_id'] for w in eligible}),
               'observed_canonical_songs': len(groups), 'missing': missing}
    result = {'status': 'unqualified', 'support': support,
              'estimand': 'ratio_of_canonical_mean_MSE' if square else 'ratio_of_canonical_mean_native_flow_loss',
              'before_comparison_loss': None, 'after_correct_loss': None, 'relative_gain': None,
              'bootstrap': None, 'canonical_values': {}, 'producer_distinct_estimand': None}
    if missing or not groups:
        return result
    keys = sorted(groups)
    values = np.asarray([np.mean(groups[key], axis=0) for key in keys], dtype=np.float64)
    a, b = values.mean(0)
    ix = np.random.default_rng(BOOTSTRAP_SEED).integers(len(keys), size=(draws, len(keys)))
    samples = values[ix].mean(1)
    ratios = np.full(draws, np.nan)
    np.divide(samples[:, 1] - samples[:, 0], samples[:, 1], out=ratios, where=samples[:, 1] > 1e-20)
    # Producer uses RMSE relative gain for planner, NOT the squared-MSE estimand.
    means = np.asarray([np.mean(producer[key]) for key in keys])
    producer_ix = np.random.default_rng(BOOTSTRAP_SEED).integers(len(keys), size=(2000, len(keys)))
    result.update(status='measured' if b > 1e-20 else 'unqualified_zero_denominator',
                  before_comparison_loss=float(b), after_correct_loss=float(a),
                  relative_gain=float((b-a)/b) if b > 1e-20 else None,
                  canonical_values={k: {'correct': float(v[0]), 'comparison': float(v[1])} for k, v in zip(keys, values)},
                  bootstrap={'seed': BOOTSTRAP_SEED, 'unit': 'canonical song', 'descriptive': True,
                             **_interval(ratios)},
                  producer_distinct_estimand={
                      'name': 'canonical_mean_of_recording_relative_' + ('RMSE' if square else 'native_flow_loss') + '_gains',
                      'relative_gain': float(means.mean()),
                      'ci95': np.quantile(means[producer_ix].mean(1), [.025, .975]).tolist(),
                      'draws': 2000, 'seed': BOOTSTRAP_SEED, 'denominator_floor': 1e-12,
                      'ci_not_transferable_to_primary_estimand': True})
    return result


def minimum_conditioning_gain(comparisons, windows, draws=BOOTSTRAP_DRAWS):
    """Joint song bootstrap of min(donor gain, shift gain), retaining absent shifts."""
    needed = [comparisons.get(key, {}) for key in ('donor', 'shift')]
    result = {'relative_gain': None, 'bootstrap': None,
              'estimand': 'minimum_of_donor_and_shift_ratio_of_canonical_mean_native_flow_losses'}
    if any(row.get('status') != 'measured' for row in needed):
        return {**result, 'status': 'unqualified_missing_or_undefined_comparator'}
    songs = sorted({window['canonical_id'] for window in windows})
    ix = np.random.default_rng(BOOTSTRAP_SEED).integers(len(songs), size=(draws, len(songs)))
    samples = []
    for comparison in needed:
        values = np.asarray([[comparison['canonical_values'].get(k, {}).get('correct', np.nan),
                              comparison['canonical_values'].get(k, {}).get('comparison', np.nan)] for k in songs])
        selected = values[ix]
        count = np.isfinite(selected[:, :, 0]).sum(1)
        sums = np.nansum(selected, axis=1)
        ratio = np.full(draws, np.nan)
        np.divide(sums[:, 1]-sums[:, 0], sums[:, 1], out=ratio,
                  where=(count > 0) & (sums[:, 1] > count * 1e-20))
        samples.append(ratio)
    result.update(status='measured', relative_gain=min(r['relative_gain'] for r in needed),
                  bootstrap={'seed': BOOTSTRAP_SEED, 'unit': 'joint canonical-song draw; each comparator uses its prespecified support',
                             'descriptive': True, **_interval(np.minimum(*samples))})
    return result


def activity(value, scale=None):
    value = np.asarray(value)
    require(value.ndim == 2 and len(value) > 0, 'Expected time by channels')
    if not np.isfinite(value).all():
        return {'finite': False, 'metrics_unavailable_reason': 'nonfinite output retained'}
    fixed = np.all(value == value[:1], axis=0)
    centered = value.astype(np.float64)-value.astype(np.float64).mean(0)
    centered[:, fixed] = 0
    variance = np.square(centered).mean(0)
    if scale is not None:
        scale = np.asarray(scale)
        require(scale.shape == (value.shape[1],) and np.all(np.isfinite(scale)) and np.all(scale > 0), 'Invalid rank scale')
        centered /= scale
    eigen = np.maximum(np.linalg.eigvalsh(centered.T @ centered / len(value)), 0)
    p = eigen[eigen > 0] / eigen.sum() if eigen.sum() else np.empty(0)
    return {'finite': True, 'frames': len(value), 'temporal_observation_available': len(value) > 1,
            'exact_constant': bool(fixed.all()), 'exact_constant_channel_count': int(fixed.sum()),
            'mean_channel_std': float(np.sqrt(variance).mean()),
            'effective_rank': float(np.exp(-np.sum(p*np.log(p)))) if len(p) else 0.,
            'effective_rank_scale': 'TRAIN channel std' if scale is not None else 'raw channels',
            'interpretation': 'descriptive temporal output rank, not a representation-collapse or choreography verdict'}


def response(value, reference):
    require(value.shape == reference.shape, 'Response shape mismatch')
    if not np.isfinite(value).all() or not np.isfinite(reference).all():
        return {'finite': False, 'rms_difference': None, 'exact_equal': False}
    delta = value.astype(np.float64)-reference.astype(np.float64)
    return {'finite': True, 'rms_difference': float(np.sqrt(np.square(delta).mean())),
            'max_abs_difference': float(np.max(np.abs(delta))), 'exact_equal': bool(np.array_equal(value, reference))}


def body_diagnostics(native, world, scale):
    require(native.ndim == 2 and native.shape[1] == 139 and world.shape == (len(native), 22, 3), 'Invalid body dimensions')
    result = {'native': activity(native, scale), 'world': activity(world.reshape(len(world), -1)),
              'hold_threshold_m_s': .05, 'hold_threshold_scope': 'exploratory physical inactivity only'}
    if not result['native']['finite'] or not result['world']['finite']:
        return result
    speed = np.linalg.norm(np.diff(world.astype(np.float64), axis=0)*30., axis=-1)
    relative = world.astype(np.float64)-world.astype(np.float64)[:, :1]
    relative_speed = np.linalg.norm(np.diff(relative, axis=0)*30., axis=-1)[:, 1:]
    result.update(root_relative=activity(relative.reshape(len(world), -1)),
                  contact_probability_range=[float(native[:, 135:].min()), float(native[:, 135:].max())],
                  contact_coverage_at_half=float(np.mean(native[:, 135:] >= .5)),
                  contact_outside_unit_interval=float(np.mean((native[:, 135:] < 0) | (native[:, 135:] > 1))),
                  world_hold_occupancy=float(np.mean(speed.mean(-1) <= .05)) if speed.size else None,
                  root_relative_hold_occupancy=float(np.mean(relative_speed.mean(-1) <= .05)) if relative_speed.size else None,
                  world_speed_p05_p50_p95_m_s=np.quantile(speed, [.05, .5, .95]).tolist() if speed.size else None,
                  root_speed_p05_p50_p95_m_s=np.quantile(speed[:, 0], [.05, .5, .95]).tolist() if speed.size else None,
                  regional_mean_speed_m_s=[float(speed[:, group].mean()) for group in REGIONS] if speed.size else None)
    return result


def summarize_diagnostics(rows):
    """Separate descriptive activity, physical inactivity and input response."""
    components = defaultdict(list)
    input_changed = output_static_under_changed_input = 0
    duplicates = defaultdict(list)
    for row in rows:
        for condition, evidence in row['conditions'].items():
            for component in ('native', 'world', 'root_relative', 'local_plan', 'global_plan'):
                if component in evidence:
                    components[component].append(evidence[component])
            input_response = evidence.get('input_response', {})
            if input_response.get('exact_equal') is False:
                input_changed += 1
                output_response = evidence.get('native_response', evidence.get('local_response', {}))
                output_static_under_changed_input += output_response.get('exact_equal') is True
            for kind in ('native', 'world'):
                if kind+'_sha256' in evidence:
                    duplicates[(kind, evidence[kind+'_sha256'])].append(row['key']+'/'+condition)
    result = {}
    for component, values in components.items():
        ranks = [value['effective_rank'] for value in values if value.get('finite')]
        result[component] = {'observed_conditions': len(values),
                             'nonfinite_conditions': sum(not value.get('finite') for value in values),
                             'exact_constant_conditions': sum(value.get('exact_constant', False) for value in values),
                             'effective_rank_min_median_max': np.quantile(ranks, [0, .5, 1]).tolist() if ranks else None}
    holds = [e['world_hold_occupancy'] for row in rows for e in row['conditions'].values()
             if e.get('world_hold_occupancy') is not None]
    return {'component_variation': result, 'input_changed_conditions': input_changed,
            'exact_output_unchanged_despite_changed_input': output_static_under_changed_input,
            'world_hold_occupancy_min_median_max': np.quantile(holds, [0, .5, 1]).tolist() if holds else None,
            'exact_output_duplicate_groups': [{'component': kind, 'sha256': digest, 'conditions': labels}
                                              for (kind, digest), labels in sorted(duplicates.items()) if len(labels) > 1],
            'interpretation': 'No arbitrary effective-rank threshold; output inactivity, duplicate streams and input insensitivity are separate observations. Hidden representation collapse was not measured.'}


def _same(a, b, message):
    require(canonical_json(a) == canonical_json(b), message)


def context(root, reader):
    protocol_path = root/'learning/protocol.json'
    protocol = reader.auth(protocol_path)
    require(protocol['seeds'] == [0, 1] and protocol['arms'] == list(ARMS), 'Unapproved arms/seeds')
    for name, expected in (('steps', 30000), ('batch', 32), ('planner_batch', 4)):
        require(protocol['training'][name] == expected, 'Unapproved training ' + name)
    require(protocol['test_access'] is False, 'Protocol does not bind development-only access')
    require(protocol['display_seed'] == 0 and protocol['generation_seed'] == 7001, 'Display/draw seed changed')
    reader.files(protocol['inputs']); reader.files(protocol['sources'])
    manifest = reader.auth(root/'learning/cache/manifest.json')
    norms = reader.auth(root/'learning/cache/normalizers.json')
    reader.sha(root/'learning/cache/normalizers.json', manifest['normalizers_sha256'])
    require(manifest['test_access'] is False, 'Cache TEST access is not excluded')
    require(len({e['key'] for e in manifest['entries']}) == len(manifest['entries']), 'Duplicate cache keys')
    require(all(e['partition'] in ('train', 'val') for e in manifest['entries']), 'Unexpected cache partition')
    train_ids = {e['canonical_id'] for e in manifest['entries'] if e['partition'] == 'train'}
    val_ids = {e['canonical_id'] for e in manifest['entries'] if e['partition'] == 'val'}
    require(not train_ids.intersection(val_ids), 'Canonical TRAIN/VAL leakage')
    for entry in manifest['entries']:
        reader.sha(entry['cache_path'], entry['cache_sha256']); reader.files(entry['inputs'])
        reader.sha(entry['path'], entry['sha256']); reader.sha(entry['audio_path'], entry['audio_sha256'])
    entries = {e['key']: e for e in manifest['entries']}
    require({w['key'] for w in protocol['evaluation']['windows']} == {e['key'] for e in manifest['entries'] if e['partition'] == 'val'},
            'Frozen evaluation does not cover complete VAL recordings')
    for w in protocol['evaluation']['windows']:
        require(w['canonical_id'] == entries[w['key']]['canonical_id'], 'Evaluation canonical identity mismatch')
        require(entries[w['donor_key']]['partition'] == 'val' and entries[w['donor_key']]['canonical_id'] != w['canonical_id'], 'Invalid donor identity')
        require(w['draw_seed'] == 7001, 'Changed evaluation draw seed')
    incumbent_paths = [Path(p) for p in protocol['inputs'] if p.endswith('.pt')]
    require(len(incumbent_paths) == 1, 'Ambiguous incumbent checkpoint')
    incumbent = reader.checkpoint(incumbent_paths[0])
    def named(name):
        matches = [p for p in protocol['inputs'] if Path(p).name == name]
        require(len(matches) == 1, 'Ambiguous frozen ' + name)
        return reader.auth(matches[0])
    return {'protocol': protocol, 'protocol_sha256': reader.sha(protocol_path), 'manifest': manifest,
            'norms': norms, 'base_norms': named('audio_normalizers.json'), 'motion_norms': named('motion_normalizers.json'),
            'entries': entries, 'incumbent_hashes': {k: array_hash(v) for k, v in incumbent['model'].items()}}


def audit_run(root, arm, seed, ctx, reader):
    folder = root/f'learning/runs/{arm}/seed_{seed}'
    report = reader.auth(folder/'report.json')
    identity = {'arm': arm, 'seed': seed, 'protocol_sha256': ctx['protocol_sha256']}
    plan_index = None
    if arm in ('D8', 'D8_flat'):
        branch = 'global' if arm == 'D8' else 'local'
        path = root/f'learning/plans/seed_{seed}/{branch}/index.json'
        plan_index = reader.auth(path)
        identity.update(planner_sha256=reader.sha(root/f'learning/runs/planner/seed_{seed}/final.pt'),
                        plan_index_sha256=reader.sha(path))
        _same(plan_index['identity'], {'checkpoint_sha256': identity['planner_sha256'],
              'cache_sha256': reader.sha(root/'learning/cache/manifest.json'), 'branch': branch}, 'Plan index identity mismatch')
        require(set(plan_index['entries']) == set(ctx['entries']), 'Plan cache coverage mismatch')
        for key, item in plan_index['entries'].items():
            reader.sha(item['path'], item['sha256'])
            arrays = reader.npz(item['path'])
            n = (ctx['entries'][key]['n_frames']+14)//15
            require(arrays['plan'].shape == (n, 34) and np.isfinite(arrays['plan']).all(), 'Invalid exported plan')
            require(np.array_equal(arrays['time'], np.arange(n)*.5), 'Invalid exported plan clock')
    _same(report['identity'], identity, 'Training report identity mismatch')
    require(report['steps'] == 30000 and report['test_access'] is False, 'Incomplete or non-DEV training')
    expected = [folder/name for name in ('initial.pt', 'final.pt', 'training.jsonl')]
    require(all(str(p) in report['artifacts'] for p in expected), 'Training artifacts are not bound')
    reader.files(report['artifacts'])
    initial, final = reader.checkpoint(expected[0]), reader.checkpoint(expected[1])
    for state, step in ((initial, 0), (final, 30000)):
        _same(state['identity'], identity, 'Checkpoint identity mismatch')
        require(state['step'] == step, 'Checkpoint step mismatch')
        _same(state['model_config'], report['model_config'], 'Checkpoint/report model configuration mismatch')
        for field, norm in (('normalizers', 'motion_norms'), ('audio_normalizers', 'base_norms'), ('extra_normalizers', 'norms')):
            _same(state[field], ctx[norm], 'Checkpoint normalizers differ: ' + field)
        _same(state['feature_names'], ctx['manifest']['feature_names'], 'Feature order differs')
        require(all(torch.isfinite(v).all().item() for v in state['model'].values()), 'Nonfinite model tensor')
    require(final['scheduler']['last_epoch'] == 30000, 'Scheduler exposure mismatch')
    optimizer_steps = [int(v['step']) for v in final['optimizer']['state'].values() if 'step' in v]
    require(optimizer_steps and all(step == 30000 for step in optimizer_steps), 'Optimizer exposure mismatch')
    logs = [json.loads(line) for line in expected[2].read_text().splitlines() if line.strip()]
    steps = [row['step'] for row in logs]
    require(steps and steps[0] == 1 and steps[-1] == 30000 and all(a < b for a, b in zip(steps, steps[1:])), 'Training log step sequence mismatch')
    require(all(row['arm'] == arm and row['seed'] == seed for row in logs), 'Training log identity mismatch')
    require(set(range(100, 30001, 100)).issubset(steps), 'Training log lacks scheduled observations')
    required_losses = (['total'] + [branch+'_'+kind for branch in ('local', 'global')
                                  for kind in ('speed', 'support', 'hold', 'duration')]) if arm == 'planner' else [
        'total', 'native', 'velocity', 'physical_position', 'physical_velocity',
        'physical_planted_foot_velocity', 'physical_root_displacement', 'physical_floor_penetration']
    for row in logs:
        require(all(_finite_loss(row.get(name)) for name in ('training_seconds', 'gradient', 'lr', *required_losses)),
                'Training log lacks finite nonnegative learning observations')
        require(type(row.get('valid_frames')) is int and row['valid_frames'] > 0, 'Invalid logged valid-frame exposure')
    require(all(a['training_seconds'] <= b['training_seconds'] for a, b in zip(logs, logs[1:])), 'Training elapsed time moves backward')
    require(report['training_seconds'] == final['training_seconds'] == logs[-1]['training_seconds'], 'Training elapsed time receipts differ')
    require(state_hash(initial['rng']['numpy_batch']) == state_hash(np.random.default_rng(10000+seed).bit_generator.state),
            'Initial sampler state differs from registered seed')
    initial_base = {k: array_hash(initial['model'][k]) for k in ctx['incumbent_hashes']} if arm != 'planner' else None
    if arm != 'planner':
        require(initial_base == ctx['incumbent_hashes'], 'Initial base differs from pinned incumbent')
        for prefix in ('audio_projection.', 'plan_projection.', 'adapters.'):
            require(any(key.startswith(prefix) for key in initial['model']), 'Missing initial conditioning weights: ' + prefix)
        require(all(torch.count_nonzero(value).item() == 0 for key, value in initial['model'].items() if key.startswith('adapters.')),
                'Initial adapter outputs are not zero')
        dynamic = report['model_config']['dynamic']
        require(dynamic['arm'] == arm and dynamic['extra_dim'] == ctx['protocol']['extra_dim'], 'Dynamic configuration mismatch')
        _same(report['model_config']['physical_weights'], ctx['protocol']['physical_weights'], 'Physical safeguards differ')
    else:
        _same(report['model_config'], ctx['protocol']['planner'], 'Planner configuration differs')
    cursors = {}
    for label, state in (('initial', initial), ('final_consumed', final)):
        cursors[label] = {'sampler': state_hash(state['rng']['numpy_batch']),
                          'draw': array_hash(state['draw_rng']), 'dropout': array_hash(state['drop_rng'])}
    return {'status': 'verified', 'arm': arm, 'seed': seed, 'steps': 30000,
            'checkpoint_sha256': reader.sha(folder/'final.pt'), 'report_sha256': reader.sha(folder/'report.json'),
            'model_config': report['model_config'], 'initial_base_sha256': state_hash(initial_base) if initial_base else None,
            'initial_all_weights_sha256': state_hash(initial['model']),
            'initial_base_equals_pinned_incumbent': initial_base is not None, 'rng_cursors': cursors,
            'logged_observations': len(logs), 'training_seconds': report['training_seconds'],
            'exposure_limit': 'steps and paired consumed RNG cursors verified; per-batch hashes were not recorded'}, plan_index


def paired_exposure(runs, seed):
    selected = [runs.get((arm, seed)) for arm in FLOW_ARMS]
    missing = [arm for arm, row in zip(FLOW_ARMS, selected) if row is None or row.get('status') != 'verified']
    if missing:
        return {'seed': seed, 'status': 'unqualified', 'missing_arms': missing}
    base_equal = len({r['initial_base_sha256'] for r in selected}) == 1
    all_weights_equal = len({r['initial_all_weights_sha256'] for r in selected}) == 1
    equal = {stage: {kind: len({r['rng_cursors'][stage][kind] for r in selected}) == 1
                     for kind in ('sampler', 'draw', 'dropout')} for stage in ('initial', 'final_consumed')}
    configs = [{**r['model_config'], 'dynamic': {k: v for k, v in r['model_config']['dynamic'].items() if k != 'arm'}} for r in selected]
    config_equal = all(canonical_json(c) == canonical_json(configs[0]) for c in configs)
    good = base_equal and all_weights_equal and config_equal and all(all(v.values()) for v in equal.values())
    return {'seed': seed, 'status': 'verified' if good else 'unqualified', 'initial_base_equal': base_equal,
            'config_equal_except_arm': config_equal, 'rng_cursors_equal': equal,
            'all_initial_weights_equal': all_weights_equal,
            'evidence': 'same frozen sampler, seed, data/config, update count and consumed sampler/draw/dropout cursors',
            'limitation': 'No per-update batch hashes exist; cursor equality is not a retrospective batch transcript.'}


def fit_rms_control(ctx, reader):
    names = ctx['norms']['feature_names']
    columns = [i for i, name in enumerate(names) if 'mixture' in name.lower() and 'rms' in name.lower()]
    require(len(columns) == 1, 'Ambiguous mixture RMS')
    column = columns[0]
    records = [e for e in ctx['manifest']['entries'] if e['partition'] == 'train']
    counts = Counter(e['canonical_id'] for e in records)
    xx, xy = np.zeros((2, 2)), np.zeros((2, 5))
    for entry in records:
        arrays = reader.npz(entry['cache_path'])
        valid = arrays['target_valid']
        rms, target = arrays['pooled'][:, column][valid], arrays['target_log_speed'][valid]
        if not len(rms):
            continue
        x = np.stack((np.ones_like(rms), np.log1p(np.maximum(rms, 0))), -1).astype(np.float64)
        weight = 1/(len(counts)*counts[entry['canonical_id']]*len(x))
        xx += weight*x.T@x
        xy += weight*x.T@target
    return column, np.linalg.solve(xx + np.diag([0., .001]), xy)


def planner_audio(entry, ctx, reader):
    arrays = reader.npz(entry['cache_path'])
    mean, std = np.asarray(ctx['norms']['mean'], np.float32), np.asarray(ctx['norms']['std'], np.float32)
    return ((arrays['pooled']-mean)/std).astype(np.float32), arrays


def flow_audio(entry, start, frames, ctx, reader, plan_index=None, shifted_plan=False):
    base = reader.npz(entry['audio_path'])['x'][start:start+frames]
    extra = reader.npz(entry['cache_path'])['extra'][start:start+frames].astype(np.float32)
    plan = np.zeros((frames, 34), np.float32)
    if plan_index:
        arrays = reader.npz(plan_index['entries'][entry['key']]['path'])
        plan = np.stack([np.interp((np.arange(frames)+start)/30., arrays['time'], arrays['plan'][:, i]) for i in range(34)], -1)
    packed = np.concatenate(((base-np.asarray(ctx['base_norms']['mean']))/np.asarray(ctx['base_norms']['std']),
                              (extra-np.asarray(ctx['norms']['mean']))/np.asarray(ctx['norms']['std']), plan), -1).astype(np.float32)
    if shifted_plan:
        packed[:, -34:] = np.roll(packed[:, -34:], frames//2, axis=0)
    return packed


def _close(actual, expected, label, atol=1e-7, rtol=2e-5):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False), label + ' differs')


def _plan_from_raw(raw):
    # CPU torch preserves the producer's float32 nonlinearities; no model runs.
    value = torch.from_numpy(raw)
    return torch.cat((torch.nn.functional.softplus(value[:, :5]), value[:, 5:14].sigmoid(),
                      value[:, 14:].reshape(-1, 5, 4).softmax(-1).flatten(1)), -1).numpy()


def audit_evaluation(root, arm, seed, run, ctx, reader, plan_index=None):
    path = root/f'learning/evaluation/{arm}/seed_{seed}.json'
    report = reader.auth(path)
    identity = {'checkpoint_sha256': run['checkpoint_sha256'], 'protocol_sha256': ctx['protocol_sha256']}
    _same(report['identity'], identity, 'Evaluation identity mismatch')
    require(report['test_access'] is False, 'Evaluation is not DEV-only')
    reader.files(report['artifacts'])
    reader.sha(report['artifact_index']['path'], report['artifact_index']['sha256'])
    index = reader.auth(report['artifact_index']['path'])
    _same(index['identity'], identity, 'Artifact index identity mismatch')
    _same(index['artifacts'], report['artifacts'], 'Artifact coverage index mismatch')
    if plan_index:
        planner_report = reader.auth(root/f'learning/evaluation/planner/seed_{seed}.json')
        planner_hash = reader.sha(root/f'learning/runs/planner/seed_{seed}/final.pt')
        require(planner_report['identity'] == {'checkpoint_sha256': planner_hash, 'protocol_sha256': ctx['protocol_sha256']},
                'Plan reference evaluation identity differs')
        branch = 'global' if arm == 'D8' else 'local'
        for key, entry in ctx['entries'].items():
            if entry['partition'] != 'val':
                continue
            raw_path = root/f'learning/evaluation/planner/seed_{seed}_artifacts/{key}_full_plans.npz'
            reader.sha(raw_path, planner_report['artifacts'][str(raw_path)])
            raw = reader.npz(raw_path)
            exported = reader.npz(plan_index['entries'][key]['path'])
            _close(exported['plan'], _plan_from_raw(raw[branch]), 'Exported ' + branch + ' plan versus saved final-planner output')
    windows = ctx['protocol']['evaluation']['windows']
    require(len(report['rows']) == len(windows), 'Evaluation window coverage incomplete')
    for row, window in zip(report['rows'], windows):
        require(all(row.get(k) == v for k, v in window.items()), 'Evaluation frozen window identity/order mismatch')
    full, checked_paths, details, bindings = {}, set(), [], {}
    if arm == 'planner':
        column, coefficients = fit_rms_control(ctx, reader)
        require(report['rms_baseline']['column'] == column and report['rms_baseline']['ridge'] == .001, 'RMS control configuration changed')
        _close(report['rms_baseline']['coefficients'], coefficients, 'TRAIN-only RMS coefficients', atol=1e-12, rtol=1e-10)
        # All complete-song predictions are saved, including donor recordings.
        for key, entry in ctx['entries'].items():
            if entry['partition'] != 'val':
                continue
            artifact_path = Path(report['artifact_index']['path']).parent/(key+'_full_plans.npz')
            receipt_path = artifact_path.with_suffix('.json')
            receipt = reader.auth(receipt_path)
            item = {'path': str(artifact_path), 'sha256': report['artifacts'][str(artifact_path)],
                    'metadata_path': str(receipt_path), 'metadata_sha256': report['artifacts'][str(receipt_path)],
                    'arrays': receipt['identity']['arrays']}
            arrays, metadata = reader.artifact(item)
            checked_paths.update((str(artifact_path), str(receipt_path)))
            audio, cached = planner_audio(entry, ctx, reader)
            _same({k: metadata[k] for k in identity}, identity, 'Full plan checkpoint identity mismatch')
            require(metadata['key'] == key and metadata['normalized_input_sha256'] == array_hash(audio), 'Full planner input binding differs')
            require(np.array_equal(arrays['time'], np.arange(len(audio))*.5), 'Full planner clock mismatch')
            for branch in ('local', 'global'):
                require(arrays[branch].shape == (len(audio), 34) and np.isfinite(arrays[branch]).all(), 'Invalid full plan')
            for name, value in cached.items():
                if name.startswith('target_'):
                    require(name in arrays and np.array_equal(value, arrays[name]), 'Saved full plan target differs: ' + name)
            full[key] = arrays
    for row, window in zip(report['rows'], windows):
        key, frames = window['key'], window['frames']
        conditions = row['conditions']
        required = {'correct', 'donor'}
        if window['shift_start'] is not None:
            required.add('shift')
        if arm == 'planner':
            required.update(('shuffled_plan', 'rms_control'))
        elif arm == 'D8':
            required.add('shifted_plan')
        require(set(conditions) == required, 'Missing/unexpected conditions: ' + key)
        correct_arrays = None
        correct_input = None
        same_noise = set()
        condition_details = {}
        for condition in ('correct', 'donor', 'shift', 'shuffled_plan', 'shifted_plan', 'rms_control'):
            if condition not in conditions:
                continue
            item = conditions[condition]
            arrays, meta = reader.artifact(item['artifact'])
            checked_paths.update((item['artifact']['path'], item['artifact']['metadata_path']))
            _same({k: meta[k] for k in identity}, identity, 'Per-condition checkpoint identity mismatch')
            require(meta['key'] == key, 'Per-condition recording binding differs')
            expected_condition = 'correct' if condition == 'rms_control' else condition
            require(meta['condition'] == expected_condition, 'Per-condition label differs')
            donor = window['donor_key'] if condition == 'donor' else key
            position = window['donor_start'] if condition == 'donor' else window['shift_start'] if condition == 'shift' else window['start']
            if arm == 'planner':
                audio, cached = planner_audio(ctx['entries'][donor], ctx, reader)
                n, start, audio_start = (frames+14)//15, window['start']//15, position//15
                conditioned = audio[audio_start:audio_start+n]
                require(meta['normalized_input_sha256'] == array_hash(conditioned) and meta['full_input_sha256'] == array_hash(audio), 'Planner input hash binding differs')
                valid = full[key]['target_valid'][start:start+n]
                target = full[key]['target_log_speed'][start:start+n]
                require(valid.any(), 'No valid proxy target observations')
                require(np.array_equal(arrays['time'], np.arange(n)*.5), 'Planner condition clock differs')
                require(np.array_equal(arrays['target_valid'], valid) and np.array_equal(arrays['target_log_speed'], target), 'Planner condition target differs')
                if condition == 'rms_control':
                    require(item['array_name'] == 'rms_log_speed_prediction', 'RMS array binding differs')
                    actual = arrays['rms_log_speed_prediction']
                    _close(item['local_speed_rmse'], np.sqrt(np.square(actual[valid]-target[valid]).mean()), 'RMS score')
                    continue
                if condition == 'shuffled_plan':
                    permutation = np.random.default_rng(7001).permutation(n)
                    require(np.array_equal(arrays['permutation'], permutation), 'Plan permutation changed')
                    require(np.array_equal(arrays['local_raw'], full[key]['local'][start:start+n][permutation]), 'Shuffled plan differs')
                    local_plan = _plan_from_raw(arrays['local_raw'])
                    _close(item['local_speed_rmse'], np.sqrt(np.square(local_plan[valid, :5]-target[valid]).mean()), 'Shuffled planner score')
                    condition_details[condition] = {'local_plan': activity(local_plan),
                                                  'local_response': response(local_plan, correct_arrays['local_plan'])}
                    continue
                for name, values in full[key].items():
                    if name.startswith('target_'):
                        require(name in arrays and np.array_equal(arrays[name], values[start:start+n]),
                                'Planner condition target differs: ' + name)
                require(meta['audio_key'] == donor and meta['audio_start_bin'] == audio_start, 'Planner audio location differs')
                detail = {}
                for branch in ('local', 'global'):
                    raw = full[donor][branch][audio_start:audio_start+n]
                    require(np.array_equal(arrays[branch+'_raw'], raw), 'Condition raw plan not from saved full song')
                    plan = _plan_from_raw(raw)
                    _close(arrays[branch+'_plan'], plan, 'Transformed plan')
                    require(_finite_loss(item.get(branch+'_speed_rmse')) and _finite_loss(item.get(branch+'_hold_brier')),
                            'Missing/nonfinite planner speed or hold score')
                    _close(item[branch+'_speed_rmse'], np.sqrt(np.square(plan[valid, :5]-target[valid]).mean()), 'Planner speed RMSE')
                    _close(item[branch+'_hold_brier'], np.square(plan[valid, 9:14]-arrays['target_hold'][valid]).mean(), 'Planner hold Brier score')
                    detail[branch+'_plan'] = activity(arrays[branch+'_plan'])
                    if condition != 'correct':
                        detail[branch+'_response'] = response(arrays[branch+'_plan'], correct_arrays[branch+'_plan'])
                x = np.stack((np.ones(n), np.log1p(np.maximum(cached['pooled'][audio_start:audio_start+n, column], 0))), -1)
                expected_rms = np.maximum(x@coefficients, 0)
                _close(arrays['rms_log_speed_prediction'], expected_rms, 'TRAIN-only RMS prediction', atol=1e-10, rtol=1e-10)
                _close(item['rms_speed_rmse'], np.sqrt(np.square(expected_rms[valid]-target[valid]).mean()), 'RMS condition score')
            else:
                require(meta['audio_key'] == donor and meta['audio_start_frame'] == position and meta['target_start_frame'] == window['start'], 'Flow audio/target location differs')
                target_entry = ctx['entries'][key]
                require(meta['target_motion_path'] == target_entry['path'] and meta['target_motion_sha256'] == target_entry['sha256'], 'Target motion provenance differs')
                conditioned = flow_audio(ctx['entries'][donor], position, frames, ctx, reader, plan_index, condition == 'shifted_plan')
                require(meta['normalized_input_sha256'] == item['normalized_input_sha256'] == array_hash(conditioned), 'Flow normalized input binding differs')
                require(meta['normalized_input_shape'] == list(conditioned.shape), 'Flow input shape binding differs')
                noise = meta['shared_initial_noise_sha256']
                require(isinstance(noise, str) and len(noise) == 64 and int(noise, 16) >= 0, 'Invalid initial noise hash')
                require(noise == item['shared_initial_noise_sha256'] and meta['draw_seed'] == window['draw_seed'] and meta['shared_initial_noise_shape'] == [frames, 139], 'Flow noise metadata differs')
                same_noise.add(noise)
                require(np.array_equal(arrays['time'], np.arange(frames)/30.), 'Flow output clock differs')
                require(len(arrays['native']) == frames, 'Flow frame count differs')
                detail = body_diagnostics(arrays['native'], arrays['world'], ctx['motion_norms']['std'])
                detail.update(native_sha256=array_hash(arrays['native']), world_sha256=array_hash(arrays['world']))
                require(detail['native']['finite'] and detail['world']['finite'], 'Nonfinite saved generation')
                for name in ('flow_loss', 'temporal_flow_loss', 'physical_total_loss'):
                    if condition != 'shifted_plan':
                        require(_finite_loss(item.get(name)), 'Missing/nonfinite evaluation loss: ' + name)
                if condition != 'correct':
                    detail['native_response'] = response(arrays['native'], correct_arrays['native'])
                    detail['world_response'] = response(arrays['world'], correct_arrays['world'])
                    _close(item['native_difference_from_correct_rms'], detail['native_response']['rms_difference'], 'Native response RMS')
                else:
                    _close(item['native_difference_from_correct_rms'], 0, 'Correct response RMS')
            if condition == 'correct':
                correct_arrays, correct_input = arrays, conditioned
            detail['input_response'] = response(conditioned, correct_input)
            recorded = item.get('normalized_input_response_from_correct')
            require(recorded is not None, 'Missing input response diagnostic')
            _close(recorded['rms_difference'], detail['input_response']['rms_difference'], 'Input response RMS')
            require(recorded['exact_equal'] == detail['input_response']['exact_equal'], 'Input equality diagnostic differs')
            condition_details[condition] = detail
        if arm != 'planner':
            require(len(same_noise) == 1, 'Noise changed across counterfactual conditions')
            bindings[key] = {'noise_sha256': next(iter(same_noise)),
                             'correct_input_sha256': conditions['correct']['normalized_input_sha256'],
                             'correct_audio_sha256': array_hash(correct_input[:, :-34]),
                             'window': window}
        details.append({'key': key, 'canonical_id': window['canonical_id'], 'conditions': condition_details})
    require(checked_paths == set(report['artifacts']), 'Uninspected or unindexed saved evaluation artifacts')
    comparators = ('rms_control', 'donor', 'shift', 'shuffled_plan') if arm == 'planner' else ('donor', 'shift')
    estimates = {name: paired_estimate(report['rows'], windows, name,
                  'local_speed_rmse' if arm == 'planner' else 'flow_loss', square=arm == 'planner') for name in comparators}
    primary = estimates['rms_control'] if arm == 'planner' else minimum_conditioning_gain(estimates, windows)
    return {'status': 'verified', 'arm': arm, 'seed': seed, 'report_sha256': reader.sha(path),
            'comparisons': estimates, 'primary': primary, 'diagnostics': details,
            'diagnostic_summary': summarize_diagnostics(details),
            'input_bindings': bindings, 'producer_summary': report.get('summary'),
            'producer_primary_flow_loss': report.get('primary_flow_loss'),
            'scope': 'exposed DEV; no artistic, representation-collapse or emitted-light qualification',
            'verification_limits': ['Outputs and input hashes verified; no checkpoint forward evaluation or flow-loss recomputation.',
                                    'CUDA initial noise tensors were not saved; equal authenticated hashes/shape/seed bind paired noise, not CPU byte replay.',
                                    'Saved native/world arrays authenticated independently; decoder correctness is not inferred from hashes.']}, report['rows']


def matched_comparison(evaluations, raw_rows, runs, seed, before, after, windows):
    result = {'seed': seed, 'before_arm': before, 'after_arm': after, 'status': 'unqualified'}
    ids = ((before, seed), (after, seed))
    if any(evaluations.get(key, {}).get('status') != 'verified' or runs.get(key, {}).get('status') != 'verified' for key in ids):
        return {**result, 'reason': 'Missing complete verified paired training/evaluation'}
    left, right = [evaluations[key]['input_bindings'] for key in ids]
    require(set(left) == set(right), 'Matched arm window support differs')
    for key in left:
        for field in ('noise_sha256', 'correct_audio_sha256', 'window'):
            _same(left[key][field], right[key][field], 'Matched arms differ in ' + field)
    cursors_match = runs[ids[0]]['rng_cursors'] == runs[ids[1]]['rng_cursors']
    base_match = (runs[ids[0]]['initial_base_sha256'] == runs[ids[1]]['initial_base_sha256']
                  and runs[ids[0]]['initial_all_weights_sha256'] == runs[ids[1]]['initial_all_weights_sha256'])
    if not cursors_match or not base_match:
        return {**result, 'reason': 'Paired initial base or consumed RNG cursors differ'}
    by_key = [{row['key']: row for row in raw_rows[key]} for key in ids]
    rows = [{**w, 'conditions': {'correct': by_key[1][w['key']]['conditions']['correct'],
                                 'matched_control': by_key[0][w['key']]['conditions']['correct']}} for w in windows]
    return {**result, 'status': 'measured', 'correct_audio_and_noise_bindings_equal': True,
            'correct_condition_loss_gain': paired_estimate(rows, windows, 'matched_control'),
            'conditioning_gain_before': evaluations[ids[0]]['primary'],
            'conditioning_gain_after': evaluations[ids[1]]['primary'],
            'scope': 'paired exposed-DEV before/after; no improvement verdict'}


def recovery_history(root, ctx, reader):
    """Disclose an authorized exact-state resume and inspect its replayed log tail."""
    path = root/'records/D7_SEED0_RECOVERY_AUTHORIZATION.json'
    if not path.exists():
        return {'status': 'no_recovery_authorization_present'}
    reader.sha(path)
    authorization = json.loads(path.read_text())
    require(authorization['training_protocol_sha256'] == ctx['protocol_sha256'], 'Recovery protocol differs')
    require(authorization['status'] == 'parent_authorized_result_neutral_resume', 'Unrecognized recovery authorization')
    require(authorization['no_final_outcome'] is True, 'Recovery was not registered before final outcome')
    reader.files(authorization['archive_files'])
    reader.sha(authorization['halted_receipt'], authorization['halted_receipt_sha256'])
    reader.sha(authorization['resume_checkpoint'], authorization['resume_checkpoint_sha256'])
    checkpoint = reader.checkpoint(authorization['resume_checkpoint'])
    identity = {'arm': 'D7', 'seed': 0, 'protocol_sha256': ctx['protocol_sha256']}
    _same(checkpoint['identity'], identity, 'Recovery checkpoint identity differs')
    require(checkpoint['step'] == authorization['resume_step'] == 21000, 'Recovery checkpoint step differs')
    require(checkpoint['scheduler']['last_epoch'] == 21000, 'Recovery scheduler differs')
    require(all(int(state['step']) == 21000 for state in checkpoint['optimizer']['state'].values() if 'step' in state), 'Recovery optimizer differs')
    archive = Path(authorization['preserved_archive'])
    original = root/'learning/runs/D7/seed_0'
    require(Path(authorization['original_run']) == original, 'Recovery destination differs')
    reader.sha(original/'initial.pt', reader.sha(archive/'initial.pt'))
    recovery_root = root/'recovery_artifacts/D7_seed0_resume_v1'
    receipt_path = recovery_root/'RECOVERY.json'
    reader.sha(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    require(receipt['authorization_sha256'] == reader.sha(path), 'Recovery receipt authorization differs')
    require(Path(receipt['original_run']) == original and Path(receipt['preserved_archive']) == archive,
            'Recovery receipt run/archive binding differs')
    require(receipt['resume_step'] == 21000 and receipt['final_step'] == 30000, 'Recovery receipt steps differ')
    copied = receipt['copied_artifacts']
    require(all(Path(p).resolve().is_relative_to(recovery_root.resolve()) for p in copied),
            'Recovered artifact lies outside recovery directory')
    reader.files(copied)
    reader.files(receipt['standard_artifacts'])
    for name in ('initial.pt', 'final.pt', 'training.jsonl'):
        require(str(original/name) in receipt['standard_artifacts'], 'Recovery receipt omits standard artifact: ' + name)
    recovered_paths = {}
    for name in ('initial.pt', 'final.pt', 'training.jsonl', 'report.json'):
        candidates = [Path(p) for p in copied if Path(p).name == name]
        require(len(candidates) == 1, 'Recovery receipt has missing/ambiguous copied artifact: ' + name)
        recovered_paths[name] = candidates[0]
        reader.sha(candidates[0], reader.sha(original/name))
    recovered_dirs = {p.parent for p in recovered_paths.values()}
    require(len(recovered_dirs) == 1, 'Recovery receipt artifacts do not share one completed run directory')
    recovered = next(iter(recovered_dirs))
    reader.sha(root/'records/resume_d7_seed0_v1.py')
    before = [json.loads(line) for line in (archive/'training.jsonl').read_text().splitlines() if line.strip()]
    after = [json.loads(line) for line in (original/'training.jsonl').read_text().splitlines() if line.strip()]
    require(before[-1]['step'] == authorization['last_logged_step'] == 21800, 'Archived last observed update differs')
    before = {row['step']: row for row in before if 21000 < row['step'] <= 21800}
    after = {row['step']: row for row in after if row['step'] in before}
    require(set(before) == set(after) == set(range(21100, 21801, 100)), 'Interrupted replay evidence is incomplete')
    observations = []
    timing_rows = []
    for step in sorted(before):
        require(set(before[step]) == set(after[step]), 'Replayed scientific log fields differ')
        require(all(_finite_loss(row.get(field)) for row in (before[step], after[step])
                    for field in (*RECOVERY_EXACT_FIELDS, 'training_seconds')), 'Missing/nonfinite recovery log observation')
        scientific_before = {k: before[step][k] for k in RECOVERY_EXACT_FIELDS}
        scientific_after = {k: after[step][k] for k in RECOVERY_EXACT_FIELDS}
        identity_equal = all(before[step][k] == after[step][k] for k in ('arm', 'seed', 'step'))
        observations.append({'step': step, 'exact_optimization_data_fields_equal': scientific_before == scientific_after,
                             'identity_fields_equal': identity_equal, 'before': scientific_before, 'after': scientific_after})
        old_time, new_time = before[step]['training_seconds'], after[step]['training_seconds']
        timing_rows.append({'step': step, 'old_training_seconds': old_time,
                            'resumed_training_seconds': new_time, 'difference_seconds': new_time-old_time})
    clarification_path = root/'records/D7_RECOVERY_TIMING_CLARIFICATION.json'
    clarification_sha256 = None
    if clarification_path.exists():
        clarification_sha256 = reader.sha(clarification_path)
        clarification = json.loads(clarification_path.read_text())
        _same(clarification['exact_fields'], list(RECOVERY_EXACT_FIELDS), 'Recovery timing clarification fields differ')
        require(clarification['elapsed_wall_time_is_not_claimed_identical'] is True, 'Recovery clarification claims identical wall timing')
        reader.sha(clarification['prior_record'], clarification['prior_record_sha256'])
        _same(clarification['timing_rows'], timing_rows, 'Recovery timing clarification differs from observed logs')
    return {'status': 'verified_archives_and_resume_evidence', 'authorization_sha256': reader.sha(path),
            'recovery_receipt_sha256': reader.sha(receipt_path), 'recovered_run_directory': str(recovered),
            'cause': authorization['cause'], 'resume_step': 21000, 'last_interrupted_logged_step': 21800,
            'compute_accounting': authorization['compute_accounting'],
            'all_replayed_optimization_data_fields_exact': all(row['exact_optimization_data_fields_equal'] for row in observations),
            'all_replayed_identity_fields_equal': all(row['identity_fields_equal'] for row in observations),
            'exact_fields': list(RECOVERY_EXACT_FIELDS),
            'replayed_observations': observations,
            'elapsed_wall_time_is_not_claimed_identical': True, 'timing_rows': timing_rows,
            'timing_clarification_sha256': clarification_sha256,
            'scope': 'Same seed and frozen recipe resumed; scientific log equality is measured, not assumed. Wall timing is excluded from deterministic comparison.'}


def audit(root):
    root, reader = Path(root).resolve(), Reader()
    result = {'schema': 1, 'root': str(root), 'status': 'unqualified', 'errors': [],
              'interpretation': 'Evidence audit and descriptive estimates only; no Vibescience or deployment verdict.',
              'seed_scope': 'Two fixed training seeds; no broad seed robustness claim.',
              'bootstrap_scope': '5,000 descriptive canonical-song draws, seed 20261005, actual ratio estimand; separate from producer 2,000-draw mean-relative-gain intervals.'}
    try:
        ctx = context(root, reader)
    except (EvidenceError, OSError, KeyError, ValueError, RuntimeError) as error:
        result['errors'].append({'component': 'frozen_context', 'reason': str(error)})
        result['pins'] = {p: v['sha256'] for p, v in reader.pins.items()}
        return result
    runs, evaluations, raw_rows = {}, {}, {}
    for arm in ARMS:
        for seed in (0, 1):
            key = (arm, seed)
            try:
                runs[key], plans = audit_run(root, arm, seed, ctx, reader)
            except (EvidenceError, OSError, KeyError, ValueError, RuntimeError) as error:
                runs[key] = {'arm': arm, 'seed': seed, 'status': 'unqualified', 'reason': str(error)}
                result['errors'].append({'component': f'training/{arm}/{seed}', 'reason': str(error)})
                continue
            try:
                evaluations[key], raw_rows[key] = audit_evaluation(root, arm, seed, runs[key], ctx, reader, plans)
            except (EvidenceError, OSError, KeyError, ValueError, RuntimeError) as error:
                evaluations[key] = {'arm': arm, 'seed': seed, 'status': 'unqualified', 'reason': str(error)}
                result['errors'].append({'component': f'evaluation/{arm}/{seed}', 'reason': str(error)})
    exposure = [paired_exposure(runs, seed) for seed in (0, 1)]
    matches = []
    for seed in (0, 1):
        for before, after in (('continued', 'D7'), ('D8_flat', 'D8'), ('D7', 'D8')):
            try:
                matches.append(matched_comparison(evaluations, raw_rows, runs, seed, before, after, ctx['protocol']['evaluation']['windows']))
            except (EvidenceError, OSError, KeyError, ValueError, RuntimeError) as error:
                result['errors'].append({'component': f'matched/{before}/{after}/{seed}', 'reason': str(error)})
    try:
        result['recovery_history'] = recovery_history(root, ctx, reader)
    except (EvidenceError, OSError, KeyError, ValueError, RuntimeError) as error:
        result['errors'].append({'component': 'recovery_history', 'reason': str(error)})
    result.update(training=list(runs.values()), evaluation=list(evaluations.values()), paired_exposure=exposure,
                  matched_comparisons=matches, pins={p: v['sha256'] for p, v in reader.pins.items()},
                  protocol_sha256=ctx['protocol_sha256'])
    primary_defined = all(row.get('primary', {}).get('status') == 'measured' for row in evaluations.values())
    matched_defined = all(row.get('correct_condition_loss_gain', {}).get('status') == 'measured' for row in matches)
    result['primary_estimands_all_defined'] = primary_defined
    result['matched_estimands_all_defined'] = matched_defined
    complete = (not result['errors'] and primary_defined and matched_defined
                and all(r['status'] == 'verified' for r in exposure) and all(r['status'] == 'measured' for r in matches))
    result['status'] = 'verified_complete_evidence' if complete else 'unqualified'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; never overwrite existing evidence')
    parser.add_argument('--allow-incomplete', action='store_true', help='Return success for a progress audit, retaining unqualified status')
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'Output exists; choose a new audit filename')
    torch.set_num_threads(2)
    report = audit(args.root)
    # No mkdir/sidecar writes: explicit sole output must have an existing parent.
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(canonical_json({'status': report['status'], 'output': str(args.output), 'errors': len(report['errors'])}))
    return 0 if report['status'] == 'verified_complete_evidence' or args.allow_incomplete else 1


if __name__ == '__main__':
    sys.exit(main())

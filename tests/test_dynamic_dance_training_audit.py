"""Synthetic evidence only: never open campaign results or use a GPU."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    'independent_dynamic_training_audit',
    Path(__file__).resolve().parents[1]/'scripts/motion/audit_dynamic_dance_training.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auth(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    Path(str(path)+'.sha256.json').write_text(json.dumps({'sha256': sha(path)}))


def npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return str(path)


def window(key, canonical, shift=15):
    return {'key': key, 'canonical_id': canonical, 'frames': 15, 'start': 0,
            'donor_key': 'b' if key == 'a' else 'a', 'donor_start': 0,
            'shift_start': shift, 'draw_seed': 7001, 'similar_bpm': True,
            'donor_log_bpm_distance': 0.}


def rows_for(values):
    windows = [window(str(i), song) for i, (song, a, b) in enumerate(values)]
    rows = [{**w, 'conditions': {'correct': {'flow_loss': a}, 'donor': {'flow_loss': b}}}
            for w, (song, a, b) in zip(windows, values)]
    return rows, windows


def test_ratio_and_producer_estimands_are_distinct_and_canonical_equal():
    rows, windows = rows_for([('song1', 1., 2.), ('song1', 1., 2.), ('song2', 90., 100.)])
    result = audit.paired_estimate(rows, windows, 'donor')
    assert result['relative_gain'] == pytest.approx(11/102)
    assert result['after_correct_loss'] == 45.5
    assert result['before_comparison_loss'] == 51.
    assert result['producer_distinct_estimand']['relative_gain'] == pytest.approx(.3)
    assert result['bootstrap']['draws'] == 5000
    assert result['producer_distinct_estimand']['draws'] == 2000
    assert result['bootstrap']['ci95'] != result['producer_distinct_estimand']['ci95'] or result['relative_gain'] != .3
    assert result == audit.paired_estimate(rows, windows, 'donor')


def test_d6_squares_rmse_before_canonical_averaging():
    w = window('a', 'canonical')
    rows = [{**w, 'conditions': {'correct': {'local_speed_rmse': .5}, 'rms_control': {'local_speed_rmse': 1.}}}]
    result = audit.paired_estimate(rows, [w], 'rms_control', 'local_speed_rmse', square=True)
    assert result['relative_gain'] == .75
    assert result['producer_distinct_estimand']['relative_gain'] == .5
    assert result['estimand'] == 'ratio_of_canonical_mean_MSE'


def test_missing_required_recording_cannot_become_favorable_subset():
    rows, windows = rows_for([('s1', .01, 1.), ('s2', 3., 1.)])
    del rows[1]['conditions']['donor']
    result = audit.paired_estimate(rows, windows, 'donor')
    assert result['status'] == 'unqualified'
    assert result['relative_gain'] is None
    assert result['bootstrap'] is None
    assert result['canonical_values'] == {}
    assert result['support']['observed_recordings'] == 1


def test_absent_shift_is_explicit_and_joint_undefined_draws_are_not_hidden():
    rows, windows = rows_for([('s1', .2, 1.), ('s2', .8, 1.)])
    windows[1]['shift_start'] = None
    rows[1]['shift_start'] = None
    rows[0]['conditions']['shift'] = {'flow_loss': .5}
    donor = audit.paired_estimate(rows, windows, 'donor')
    shift = audit.paired_estimate(rows, windows, 'shift')
    assert shift['support']['absent_shift_windows'] == ['1']
    assert shift['support']['expected_canonical_songs'] == 1
    primary = audit.minimum_conditioning_gain({'donor': donor, 'shift': shift}, windows)
    assert primary['relative_gain'] == pytest.approx(.5)
    assert primary['bootstrap']['undefined_draws'] > 0
    assert primary['bootstrap']['ci95'] is None
    assert primary['bootstrap']['conditional_defined_draws_ci95'] is not None


def test_zero_denominator_is_not_favorable_or_silently_dropped():
    rows, windows = rows_for([('s1', 1., 0.)])
    result = audit.paired_estimate(rows, windows, 'donor')
    assert result['relative_gain'] is None
    assert result['status'] == 'unqualified_zero_denominator'
    assert result['bootstrap']['undefined_draws'] == 5000
    assert result['bootstrap']['defined_draws'] == 0


def test_inactivity_is_distinct_from_native_variation_and_root_translation():
    native = np.zeros((4, 139), np.float32)
    native[:, 2] = np.arange(4)
    world = np.zeros((4, 22, 3), np.float32)
    world[:, :, 0] = np.arange(4)[:, None]
    result = audit.body_diagnostics(native, world, np.ones(139))
    assert not result['native']['exact_constant']
    assert not result['world']['exact_constant']
    assert result['root_relative']['exact_constant']
    assert result['root_relative_hold_occupancy'] == 1.
    assert result['world_hold_occupancy'] == 0.
    assert 'representation-collapse' in result['native']['interpretation']


def artifact(path, arrays, metadata):
    path = Path(path)
    npz(path, **arrays)
    hashes = {k: audit.array_hash(v) for k, v in arrays.items()}
    receipt = {'kind': 'DynamicDanceEvaluation', 'npz_sha256': sha(path),
               'identity': {'arrays': hashes, 'metadata': metadata}}
    auth(path.with_suffix('.json'), receipt)
    return {'path': str(path), 'sha256': sha(path), 'metadata_path': str(path.with_suffix('.json')),
            'metadata_sha256': sha(path.with_suffix('.json')), 'arrays': hashes}


def test_array_hash_shape_dtype_and_container_binding(tmp_path):
    x = np.arange(4, dtype=np.float32)
    assert audit.array_hash(x) != audit.array_hash(x.reshape(2, 2))
    assert audit.array_hash(x) != audit.array_hash(x.astype(np.float64))
    item = artifact(tmp_path/'evidence.npz', {'native': x}, {'label': 'synthetic'})
    arrays, metadata = audit.Reader().artifact(item)
    assert np.array_equal(arrays['native'], x)
    assert metadata == {'label': 'synthetic'}
    np.savez(item['path'], native=x+1)
    with pytest.raises(audit.EvidenceError, match='SHA256 mismatch'):
        audit.Reader().artifact(item)


def synthetic_context(root):
    names = ['mixture_rms']
    norms = {'mean': [0.], 'std': [1.], 'feature_names': names}
    base = {'mean': [0.]*92, 'std': [1.]*92}
    motion = {'mean': [0.]*139, 'std': [1.]*139}
    paths = {}
    for name, value in (('learning/cache/normalizers.json', norms),
                        ('data/audio_normalizers.json', base), ('data/motion_normalizers.json', motion)):
        path = root/name
        auth(path, value)
        paths[str(path)] = sha(path)
    incumbent = root/'data/incumbent.pt'
    torch.save({'model': {'base.weight': torch.ones(2)}}, incumbent)
    paths[str(incumbent)] = sha(incumbent)
    entries = []
    for i, key in enumerate(('train', 'a', 'b')):
        features = np.zeros((30, 139), np.float32)
        native = npz(root/f'data/{key}_native.npz', features=features)
        audio = npz(root/f'data/{key}_audio.npz', x=np.full((30, 92), i, np.float32))
        cache = npz(root/f'learning/cache/{key}.npz', extra=np.full((30, 1), i, np.float16),
                    pooled=np.array([[1.+i], [2.+i]], np.float32),
                    target_log_speed=np.repeat(np.array([[.2], [.3]], np.float32), 5, axis=1),
                    target_valid=np.ones(2, bool), target_support=np.ones((2, 4), np.float32),
                    target_hold=np.zeros((2, 5), np.float32), target_duration=np.ones((2, 5), np.int64),
                    target_duration_valid=np.ones((2, 5), bool))
        entries.append({'key': key, 'canonical_id': key, 'partition': 'train' if key == 'train' else 'val',
                        'path': native, 'sha256': sha(native), 'audio_path': audio, 'audio_sha256': sha(audio),
                        'cache_path': cache, 'cache_sha256': sha(cache), 'inputs': {native: sha(native), audio: sha(audio)},
                        'n_frames': 30})
    manifest = {'entries': entries, 'feature_names': names, 'test_access': False,
                'normalizers_sha256': sha(root/'learning/cache/normalizers.json')}
    auth(root/'learning/cache/manifest.json', manifest)
    paths[str(root/'learning/cache/manifest.json')] = sha(root/'learning/cache/manifest.json')
    source = root/'sources/frozen.py'
    source.parent.mkdir(parents=True)
    source.write_text('# Synthetic frozen source\n')
    protocol = {'seeds': [0, 1], 'arms': list(audit.ARMS), 'test_access': False,
                'display_seed': 0, 'generation_seed': 7001, 'inputs': paths,
                'sources': {str(source): sha(source)}, 'extra_dim': 1, 'planner': {'input_dim': 1},
                'physical_weights': {'position': .5},
                'training': {'steps': 30000, 'batch': 32, 'planner_batch': 4},
                'evaluation': {'windows': [window('a', 'a'), window('b', 'b', None)]}}
    auth(root/'learning/protocol.json', protocol)
    return audit.context(root, audit.Reader())


def synthetic_run(root, ctx, arm, seed):
    run = root/f'learning/runs/{arm}/seed_{seed}'
    run.mkdir(parents=True)
    identity = {'arm': arm, 'seed': seed, 'protocol_sha256': ctx['protocol_sha256']}
    plans = None
    if arm in ('D8', 'D8_flat'):
        branch = 'global' if arm == 'D8' else 'local'
        entries = {}
        for key in ctx['entries']:
            raw = np.full((2, 34), -2., np.float32)
            raw[1] += .25
            if branch == 'global':
                raw += .1
            path = npz(root/f'learning/plans/seed_{seed}/{branch}/{key}.npz',
                       time=np.arange(2)*.5, plan=audit._plan_from_raw(raw))
            entries[key] = {'path': path, 'sha256': sha(path)}
        plan_path = root/f'learning/plans/seed_{seed}/{branch}/index.json'
        plans = {'identity': {'checkpoint_sha256': sha(root/f'learning/runs/planner/seed_{seed}/final.pt'),
                             'cache_sha256': sha(root/'learning/cache/manifest.json'), 'branch': branch}, 'entries': entries}
        auth(plan_path, plans)
        identity.update(planner_sha256=plans['identity']['checkpoint_sha256'], plan_index_sha256=sha(plan_path))
    config = {'input_dim': 1} if arm == 'planner' else {
        'dynamic': {'arm': arm, 'extra_dim': 1}, 'physical_weights': {'position': .5}}
    state = {'identity': identity, 'model_config': config, 'model': {'base.weight': torch.ones(2)},
             'normalizers': ctx['motion_norms'], 'audio_normalizers': ctx['base_norms'], 'extra_normalizers': ctx['norms'],
             'feature_names': ctx['manifest']['feature_names'], 'training_seconds': 1.,
             'rng': {'numpy_batch': np.random.default_rng(10000+seed).bit_generator.state}, 'draw_rng': torch.tensor([seed], dtype=torch.uint8),
             'drop_rng': torch.tensor([seed+1], dtype=torch.uint8)}
    if arm != 'planner':
        state['model'].update({'audio_projection.weight': torch.ones((2, 1)),
                               'plan_projection.weight': torch.ones((2, 34)),
                               'adapters.0.weight': torch.zeros((2, 2))})
    for step, name in ((0, 'initial.pt'), (30000, 'final.pt')):
        torch.save({**state, 'step': step, 'scheduler': {'last_epoch': step},
                    'optimizer': {'state': {0: {'step': torch.tensor(step)}}}}, run/name)
    losses = (['total'] + [branch+'_'+kind for branch in ('local', 'global')
                          for kind in ('speed', 'support', 'hold', 'duration')]) if arm == 'planner' else [
        'total', 'native', 'velocity', 'physical_position', 'physical_velocity',
        'physical_planted_foot_velocity', 'physical_root_displacement', 'physical_floor_penetration']
    logs = [{'arm': arm, 'seed': seed, 'step': step, 'training_seconds': 1., 'gradient': .1,
             'valid_frames': 15, 'lr': .0003, **{name: .1 for name in losses}}
            for step in [1]+list(range(100, 30001, 100))]
    (run/'training.jsonl').write_text('\n'.join(json.dumps(row) for row in logs))
    report = {'identity': identity, 'steps': 30000, 'test_access': False, 'model_config': config,
              'training_seconds': 1., 'artifacts': {str(run/name): sha(run/name) for name in ('initial.pt', 'final.pt', 'training.jsonl')}}
    auth(run/'report.json', report)
    return plans


def synthetic_evaluation(root, ctx, arm, seed, plan_index):
    out = root/f'learning/evaluation/{arm}/seed_{seed}.json'
    folder = out.parent/f'seed_{seed}_artifacts'
    identity = {'checkpoint_sha256': sha(root/f'learning/runs/{arm}/seed_{seed}/final.pt'),
                'protocol_sha256': ctx['protocol_sha256']}
    artifacts = {}
    def retain(name, arrays, metadata):
        item = artifact(folder/(name+'.npz'), arrays, {**identity, **metadata})
        artifacts[item['path']] = item['sha256']
        artifacts[item['metadata_path']] = item['metadata_sha256']
        return item
    full = {}
    reader = audit.Reader()
    if arm == 'planner':
        column, coefficients = audit.fit_rms_control(ctx, reader)
        for key in ('a', 'b'):
            audio, cached = audit.planner_audio(ctx['entries'][key], ctx, reader)
            raw = np.full((2, 34), -2., np.float32)
            raw[1] += .25
            full[key] = {'time': np.arange(2)*.5, 'local': raw, 'global': raw+.1,
                         **{k: v for k, v in cached.items() if k.startswith('target_')}}
            retain(key+'_full_plans', full[key], {'key': key, 'normalized_input_sha256': audit.array_hash(audio)})
    rows = []
    for window_ in ctx['protocol']['evaluation']['windows']:
        key, frames = window_['key'], window_['frames']
        row = {**window_, 'conditions': {}}
        conditions = ['correct', 'donor'] + (['shift'] if window_['shift_start'] is not None else [])
        if arm == 'D8':
            conditions.append('shifted_plan')
        correct_input = correct_native = None
        for condition in conditions:
            donor = window_['donor_key'] if condition == 'donor' else key
            position = window_['donor_start'] if condition == 'donor' else window_['shift_start'] if condition == 'shift' else window_['start']
            metadata = {'key': key, 'condition': condition, 'audio_key': donor}
            if arm == 'planner':
                audio, cached = audit.planner_audio(ctx['entries'][donor], ctx, reader)
                input_ = audio[position//15:position//15+1]
                valid, target = full[key]['target_valid'][:1], full[key]['target_log_speed'][:1]
                arrays = {'time': np.arange(1)*.5,
                          **{k: v[:1] for k, v in full[key].items() if k.startswith('target_')}}
                scores = {}
                for branch in ('local', 'global'):
                    raw = full[donor][branch][position//15:position//15+1]
                    plan = audit._plan_from_raw(raw)
                    arrays[branch+'_raw'] = raw
                    arrays[branch+'_plan'] = plan
                    scores[branch+'_speed_rmse'] = float(np.sqrt(np.square(plan[:, :5]-target).mean()))
                    scores[branch+'_hold_brier'] = float(np.square(plan[:, 9:14]-arrays['target_hold']).mean())
                x = np.stack((np.ones(1), np.log1p(np.maximum(cached['pooled'][position//15:position//15+1, column], 0))), -1)
                arrays['rms_log_speed_prediction'] = np.maximum(x@coefficients, 0)
                scores['rms_speed_rmse'] = float(np.sqrt(np.square(arrays['rms_log_speed_prediction']-target).mean()))
                metadata.update(normalized_input_sha256=audit.array_hash(input_), full_input_sha256=audit.array_hash(audio), audio_start_bin=position//15)
            else:
                input_ = audit.flow_audio(ctx['entries'][donor], position, frames, ctx, reader, plan_index, condition == 'shifted_plan')
                native = np.zeros((frames, 139), np.float32)
                native[:, 0] = {'correct': 0., 'donor': .01, 'shift': .02, 'shifted_plan': .03}[condition]
                world = np.zeros((frames, 22, 3), np.float32)
                arrays = {'native': native, 'world': world, 'time': np.arange(frames)/30.}
                if correct_native is None:
                    correct_native = native
                scores = {'flow_loss': .6 if condition == 'correct' else 1., 'temporal_flow_loss': 1., 'physical_total_loss': 1.,
                          'native_difference_from_correct_rms': audit.response(native, correct_native)['rms_difference'],
                          'normalized_input_sha256': audit.array_hash(input_), 'shared_initial_noise_sha256': '1'*64}
                entry = ctx['entries'][key]
                metadata.update(audio_start_frame=position, target_start_frame=window_['start'],
                                target_motion_path=entry['path'], target_motion_sha256=entry['sha256'],
                                normalized_input_sha256=audit.array_hash(input_), normalized_input_shape=list(input_.shape),
                                shared_initial_noise_sha256='1'*64, shared_initial_noise_shape=[frames, 139], draw_seed=7001)
            if correct_input is None:
                correct_input = input_
            scores['normalized_input_response_from_correct'] = audit.response(input_, correct_input)
            scores['artifact'] = retain(key+'_'+condition, arrays, metadata)
            row['conditions'][condition] = scores
        if arm == 'planner':
            audio, _ = audit.planner_audio(ctx['entries'][key], ctx, reader)
            shuffled = full[key]['local'][:1]
            arr = {'time': np.arange(1)*.5, 'target_log_speed': full[key]['target_log_speed'][:1],
                   'target_valid': full[key]['target_valid'][:1], 'local_raw': shuffled, 'permutation': np.array([0])}
            item = retain(key+'_shuffled_plan', arr, {'key': key, 'condition': 'shuffled_plan',
                          'normalized_input_sha256': audit.array_hash(audio[:1]), 'full_input_sha256': audit.array_hash(audio)})
            row['conditions']['shuffled_plan'] = {'local_speed_rmse': row['conditions']['correct']['local_speed_rmse'], 'artifact': item}
            row['conditions']['rms_control'] = {'local_speed_rmse': row['conditions']['correct']['rms_speed_rmse'],
                                               'artifact': row['conditions']['correct']['artifact'], 'array_name': 'rms_log_speed_prediction'}
        rows.append(row)
    auth(folder/'index.json', {'identity': identity, 'artifacts': artifacts})
    result = {'identity': identity, 'rows': rows, 'test_access': False, 'artifacts': artifacts,
              'artifact_index': {'path': str(folder/'index.json'), 'sha256': sha(folder/'index.json')}, 'summary': {}}
    if arm == 'planner':
        result['rms_baseline'] = {'column': column, 'coefficients': coefficients.tolist(), 'ridge': .001}
    auth(out, result)


@pytest.fixture
def campaign(tmp_path):
    torch.set_num_threads(2)
    ctx = synthetic_context(tmp_path)
    for arm in audit.ARMS:
        for seed in (0, 1):
            plans = synthetic_run(tmp_path, ctx, arm, seed)
            synthetic_evaluation(tmp_path, ctx, arm, seed, plans)
    return tmp_path


def test_complete_synthetic_campaign_verifies_every_arm_and_matched_control(campaign):
    result = audit.audit(campaign)
    assert result['errors'] == []
    assert result['status'] == 'verified_complete_evidence'
    assert len(result['training']) == 10
    assert len(result['evaluation']) == 10
    assert len(result['matched_comparisons']) == 6
    assert all(row['status'] == 'verified' for row in result['paired_exposure'])
    matched = {(r['before_arm'], r['after_arm']) for r in result['matched_comparisons']}
    assert ('continued', 'D7') in matched and ('D8_flat', 'D8') in matched
    json.dumps(result, allow_nan=False)


def test_missing_run_is_unqualified_and_cli_output_is_exclusive(campaign, tmp_path):
    (campaign/'learning/runs/D8/seed_1/report.json').unlink()
    result = audit.audit(campaign)
    assert result['status'] == 'unqualified'
    assert any(e['component'] == 'training/D8/1' for e in result['errors'])
    output = tmp_path/'audit.json'
    assert audit.main(['--root', str(campaign), '--output', str(output)]) == 1
    preserved = output.read_bytes()
    with pytest.raises(audit.EvidenceError, match='Output exists'):
        audit.main(['--root', str(campaign), '--output', str(output), '--allow-incomplete'])
    assert output.read_bytes() == preserved
    progress = tmp_path/'progress.json'
    assert audit.main(['--root', str(campaign), '--output', str(progress), '--allow-incomplete']) == 0
    assert json.loads(progress.read_text())['status'] == 'unqualified'


def test_authenticated_but_changed_final_sampler_breaks_pairing(campaign):
    run = campaign/'learning/runs/D7/seed_0'
    checkpoint = torch.load(run/'final.pt', map_location='cpu', weights_only=False)
    checkpoint['rng']['numpy_batch']['seed'] = 9876
    torch.save(checkpoint, run/'final.pt')
    report = json.loads((run/'report.json').read_text())
    report['artifacts'][str(run/'final.pt')] = sha(run/'final.pt')
    auth(run/'report.json', report)
    reader = audit.Reader()
    ctx = audit.context(campaign, reader)
    runs = {(arm, 0): audit.audit_run(campaign, arm, 0, ctx, reader)[0] for arm in audit.FLOW_ARMS}
    result = audit.paired_exposure(runs, 0)
    assert result['status'] == 'unqualified'
    assert not result['rng_cursors_equal']['final_consumed']['sampler']


def test_frozen_source_change_fails_before_any_estimate(campaign):
    (campaign/'sources/frozen.py').write_text('# Changed source\n')
    result = audit.audit(campaign)
    assert result['status'] == 'unqualified'
    assert result['errors'][0]['component'] == 'frozen_context'
    assert 'training' not in result


def test_input_hash_must_reconstruct_from_frozen_audio_not_just_match_receipts(campaign):
    path = campaign/'learning/evaluation/D7/seed_0.json'
    report = json.loads(path.read_text())
    item = report['rows'][0]['conditions']['donor']
    metadata_path = Path(item['artifact']['metadata_path'])
    receipt = json.loads(metadata_path.read_text())
    item['normalized_input_sha256'] = receipt['identity']['metadata']['normalized_input_sha256'] = 'f'*64
    auth(metadata_path, receipt)
    item['artifact']['metadata_sha256'] = report['artifacts'][str(metadata_path)] = sha(metadata_path)
    index_path = Path(report['artifact_index']['path'])
    index = json.loads(index_path.read_text())
    index['artifacts'] = report['artifacts']
    auth(index_path, index)
    report['artifact_index']['sha256'] = sha(index_path)
    auth(path, report)
    result = audit.audit(campaign)
    assert result['status'] == 'unqualified'
    assert any('normalized input binding differs' in e['reason'] for e in result['errors'])


def test_authenticated_wrong_training_step_is_rejected(campaign):
    path = campaign/'learning/runs/continued/seed_1/report.json'
    report = json.loads(path.read_text())
    report['steps'] = 29999
    auth(path, report)
    result = audit.audit(campaign)
    assert result['status'] == 'unqualified'
    assert any(e['component'] == 'training/continued/1' for e in result['errors'])


@pytest.mark.parametrize('change,match', [
    ('missing_loss', 'learning observations'),
    ('nonfinite_gradient', 'learning observations'),
    ('wrong_sampler_seed', 'registered seed'),
    ('nonzero_adapter', 'adapter outputs'),
])
def test_authenticated_training_evidence_still_requires_actual_fields(campaign, change, match):
    run = campaign/'learning/runs/D7/seed_0'
    if change in ('missing_loss', 'nonfinite_gradient'):
        logs = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
        if change == 'missing_loss':
            del logs[0]['native']
        else:
            logs[0]['gradient'] = float('nan')
        (run/'training.jsonl').write_text('\n'.join(json.dumps(row) for row in logs))
    else:
        state = torch.load(run/'initial.pt', map_location='cpu', weights_only=False)
        if change == 'wrong_sampler_seed':
            state['rng']['numpy_batch'] = np.random.default_rng(10001).bit_generator.state
        else:
            state['model']['adapters.0.weight'].fill_(1)
        torch.save(state, run/'initial.pt')
    report = json.loads((run/'report.json').read_text())
    report['artifacts'] = {name: sha(name) for name in report['artifacts']}
    auth(run/'report.json', report)
    reader = audit.Reader()
    ctx = audit.context(campaign, reader)
    with pytest.raises(audit.EvidenceError, match=match):
        audit.audit_run(campaign, 'D7', 0, ctx, reader)


@pytest.mark.parametrize('target,remove', [
    ('target_support', False), ('target_hold', False), ('target_duration', False),
    ('target_duration_valid', False), ('target_support', True), ('target_hold', True),
    ('target_duration', True), ('target_duration_valid', True),
])
def test_authenticated_planner_window_requires_all_original_targets(campaign, target, remove):
    path = campaign/'learning/evaluation/planner/seed_0.json'
    report = json.loads(path.read_text())
    condition = report['rows'][0]['conditions']['donor']
    item = condition['artifact']
    with np.load(item['path'], allow_pickle=False) as values:
        arrays = {key: values[key].copy() for key in values.files}
    if remove:
        del arrays[target]
    elif arrays[target].dtype == bool:
        arrays[target] = ~arrays[target]
    else:
        arrays[target] += 1
    receipt = json.loads(Path(item['metadata_path']).read_text())
    replacement = artifact(item['path'], arrays, receipt['identity']['metadata'])
    condition['artifact'] = replacement
    for p, field in ((replacement['path'], 'sha256'), (replacement['metadata_path'], 'metadata_sha256')):
        report['artifacts'][p] = replacement[field]
    index_path = Path(report['artifact_index']['path'])
    index = json.loads(index_path.read_text())
    index['artifacts'] = report['artifacts']
    auth(index_path, index)
    report['artifact_index']['sha256'] = sha(index_path)
    auth(path, report)
    reader = audit.Reader()
    ctx = audit.context(campaign, reader)
    run, plans = audit.audit_run(campaign, 'planner', 0, ctx, reader)
    with pytest.raises(audit.EvidenceError, match='Planner condition target differs: '+target):
        audit.audit_evaluation(campaign, 'planner', 0, run, ctx, reader, plans)


@pytest.mark.parametrize('branch', ['local', 'global'])
@pytest.mark.parametrize('change', ['wrong', 'missing', 'nonfinite'])
def test_authenticated_planner_hold_brier_is_required_and_recomputed(campaign, branch, change):
    path = campaign/'learning/evaluation/planner/seed_0.json'
    report = json.loads(path.read_text())
    condition = report['rows'][0]['conditions']['donor']
    field = branch+'_hold_brier'
    if change == 'missing':
        del condition[field]
    elif change == 'nonfinite':
        condition[field] = float('nan')
    else:
        condition[field] += .1
    auth(path, report)
    reader = audit.Reader()
    ctx = audit.context(campaign, reader)
    run, plans = audit.audit_run(campaign, 'planner', 0, ctx, reader)
    match = 'Planner hold Brier score differs' if change == 'wrong' else 'Missing/nonfinite planner speed or hold score'
    with pytest.raises(audit.EvidenceError, match=match):
        audit.audit_evaluation(campaign, 'planner', 0, run, ctx, reader, plans)


def test_undefined_primary_cannot_qualify_complete_evidence(campaign, monkeypatch):
    original = audit.audit_evaluation
    def undefined(*args, **kwargs):
        result, rows = original(*args, **kwargs)
        result['primary'] = {'status': 'unqualified_zero_denominator', 'relative_gain': None}
        return result, rows
    monkeypatch.setattr(audit, 'audit_evaluation', undefined)
    result = audit.audit(campaign)
    assert result['status'] == 'unqualified'
    assert not result['primary_estimands_all_defined']


def recovery_fixture(campaign, layout='completed_run'):
    import shutil
    run = campaign/'learning/runs/D7/seed_0'
    archive = campaign/'interrupted_runs/D7_seed0_attempt1'
    archive.mkdir(parents=True)
    shutil.copy2(run/'initial.pt', archive/'initial.pt')
    checkpoint = torch.load(run/'final.pt', map_location='cpu', weights_only=False)
    checkpoint['step'] = checkpoint['scheduler']['last_epoch'] = 21000
    checkpoint['optimizer']['state'][0]['step'] = torch.tensor(21000)
    torch.save(checkpoint, archive/'latest.pt')
    logs = [json.loads(line) for line in (run/'training.jsonl').read_text().splitlines()]
    interrupted = [row for row in logs if row['step'] <= 21800]
    (archive/'training.jsonl').write_text('\n'.join(json.dumps(row) for row in interrupted))
    (archive/'status.json').write_text(json.dumps(interrupted[-1]))
    recovery_root = campaign/'recovery_artifacts/D7_seed0_resume_v1'
    recovered = recovery_root/layout if layout else recovery_root
    recovered.parent.mkdir(parents=True)
    shutil.copytree(run, recovered)
    records = campaign/'records'
    records.mkdir()
    (records/'halted.json').write_text('{"exit": 143}')
    (records/'resume_d7_seed0_v1.py').write_text('# Synthetic wrapper\n')
    authorization = {'training_protocol_sha256': sha(campaign/'learning/protocol.json'),
                     'status': 'parent_authorized_result_neutral_resume', 'no_final_outcome': True,
                     'archive_files': {str(p): sha(p) for p in archive.iterdir()},
                     'halted_receipt': str(records/'halted.json'), 'halted_receipt_sha256': sha(records/'halted.json'),
                     'resume_checkpoint': str(archive/'latest.pt'), 'resume_checkpoint_sha256': sha(archive/'latest.pt'),
                     'preserved_archive': str(archive), 'original_run': str(run), 'resume_step': 21000,
                     'last_logged_step': 21800, 'cause': 'synthetic SIGTERM', 'compute_accounting': 'retain original debit'}
    (records/'D7_SEED0_RECOVERY_AUTHORIZATION.json').write_text(json.dumps(authorization))
    def write_receipt():
        receipt = {'authorization_sha256': sha(records/'D7_SEED0_RECOVERY_AUTHORIZATION.json'),
                   'original_run': str(run), 'preserved_archive': str(archive), 'resume_step': 21000, 'final_step': 30000,
                   'copied_artifacts': {str(p): sha(p) for p in recovered.iterdir() if p.name != 'RECOVERY.json'},
                   'standard_artifacts': {str(run/name): sha(run/name) for name in ('initial.pt', 'final.pt', 'training.jsonl')}}
        (recovery_root/'RECOVERY.json').write_text(json.dumps(receipt))
    write_receipt()
    return run, archive, recovered, records, logs, authorization, write_receipt


@pytest.mark.parametrize('layout', ['', 'completed_run'])
def test_recovery_reports_exact_observed_log_replay_and_preserves_archive(campaign, layout):
    import shutil
    run, archive, recovered, records, logs, authorization, write_receipt = recovery_fixture(campaign, layout)
    reader = audit.Reader()
    ctx = audit.context(campaign, reader)
    report = audit.recovery_history(campaign, ctx, reader)
    assert report['all_replayed_optimization_data_fields_exact']
    assert report['all_replayed_identity_fields_equal']
    assert report['recovered_run_directory'] == str(recovered)
    assert len(report['exact_fields']) == 11
    assert [r['step'] for r in report['replayed_observations']] == list(range(21100, 21801, 100))
    # Scientific inequality is reported truthfully; it is not silently averaged.
    logs[211]['gradient'] += .5
    (run/'training.jsonl').write_text('\n'.join(json.dumps(row) for row in logs))
    shutil.copy2(run/'training.jsonl', recovered/'training.jsonl')
    write_receipt()
    report = audit.recovery_history(campaign, ctx, audit.Reader())
    assert not report['all_replayed_optimization_data_fields_exact']
    assert sha(archive/'training.jsonl') == authorization['archive_files'][str(archive/'training.jsonl')]


def test_recovery_wall_timing_is_reported_separately_and_startup_record_is_allowed(campaign):
    import shutil
    run, archive, recovered, records, logs, authorization, write_receipt = recovery_fixture(campaign)
    timing = []
    for row in logs:
        if 21000 < row['step'] <= 21800:
            old_time = row['training_seconds']
            row['training_seconds'] += 1.25
            timing.append({'step': row['step'], 'old_training_seconds': old_time,
                           'resumed_training_seconds': row['training_seconds'], 'difference_seconds': 1.25})
    startup = {**logs[210], 'step': 21001}
    logs.append(startup)
    logs.sort(key=lambda row: row['step'])
    (run/'training.jsonl').write_text('\n'.join(json.dumps(row) for row in logs))
    shutil.copy2(run/'training.jsonl', recovered/'training.jsonl')
    write_receipt()
    prior = records/'D7_RESUME_TAIL_VERIFICATION.json'
    prior.write_text('{"earlier_wording": "preserved"}')
    clarification = {'exact_fields': list(audit.RECOVERY_EXACT_FIELDS),
                     'elapsed_wall_time_is_not_claimed_identical': True, 'timing_rows': timing,
                     'prior_record': str(prior), 'prior_record_sha256': sha(prior)}
    clarification_path = records/'D7_RECOVERY_TIMING_CLARIFICATION.json'
    clarification_path.write_text(json.dumps(clarification))
    reader = audit.Reader()
    ctx = audit.context(campaign, reader)
    result = audit.recovery_history(campaign, ctx, reader)
    assert result['all_replayed_optimization_data_fields_exact']
    assert result['all_replayed_identity_fields_equal']
    assert result['elapsed_wall_time_is_not_claimed_identical']
    assert result['timing_clarification_sha256'] == sha(clarification_path)
    assert result['timing_rows'] == timing
    assert sha(archive/'training.jsonl') == authorization['archive_files'][str(archive/'training.jsonl')]


@pytest.mark.parametrize('change,match', [
    ('authorization', 'authorization differs'), ('missing_copy', 'missing/ambiguous copied artifact'),
    ('changed_copy', 'SHA256 mismatch'),
])
def test_recovery_receipt_path_resolution_keeps_hash_and_identity_checks(campaign, change, match):
    run, archive, recovered, records, logs, authorization, write_receipt = recovery_fixture(campaign)
    receipt_path = recovered.parent/'RECOVERY.json'
    receipt = json.loads(receipt_path.read_text())
    if change == 'authorization':
        receipt['authorization_sha256'] = '0'*64
    elif change == 'missing_copy':
        del receipt['copied_artifacts'][str(recovered/'report.json')]
    else:
        (recovered/'training.jsonl').write_text('changed synthetic evidence')
    receipt_path.write_text(json.dumps(receipt))
    reader = audit.Reader()
    ctx = audit.context(campaign, reader)
    with pytest.raises(audit.EvidenceError, match=match):
        audit.recovery_history(campaign, ctx, reader)

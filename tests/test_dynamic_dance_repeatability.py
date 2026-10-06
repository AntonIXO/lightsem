"""CPU synthetic checks for measured repeats, immutable provenance and mapping."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.motion.audit_dynamic_dance_repeatability import (
    array_difference, windowed_difference, regional_body_difference,
    infer_receipt_match, command_match, mapping_comparison,
    verify_frozen_source, load_frozen_modules, write_new, sha, _guard_job, _numeric,
)


def test_array_equality_is_not_container_hash_or_dtype_equality(tmp_path):
    value = np.arange(12).reshape(4, 3)
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez(a, data=value); np.savez_compressed(b, data=value)
    assert sha(a) != sha(b)
    with np.load(a) as first, np.load(b) as second:
        score = array_difference(first['data'], second['data'])
    assert score['exact'] and score['byte_exact'] and score['rms'] == 0.
    typed = array_difference(value, value.astype(float))
    assert typed['exact'] and not typed['byte_exact']
    assert array_difference(None, None)['status'] == 'not_applicable'
    assert array_difference(None, value)['rms'] is None
    assert array_difference(value, value[:-1])['status'] == 'shape_mismatch'
    assert array_difference(np.array([np.nan]), np.zeros(1))['rms'] is None


def test_repeat_windows_do_not_hide_prefix_noise():
    t = np.arange(180.)
    first = np.zeros((180, 2)); second = first.copy()
    second[2, 0] = .5
    score = windowed_difference(first, second, t)
    assert score['full_prefix']['maximum_absolute'] == .5
    assert score['review_60_180']['exact']
    assert score['review_60_180']['rms'] == 0.


def test_root_translation_and_regional_repeat_noise_are_separate():
    rng = np.random.default_rng(4)
    body = rng.normal(size=(10, 22, 3))
    regions = {'left_arm': (16, 18, 20), 'right_arm': (17, 19, 21)}
    translated = regional_body_difference(body, body + [2., 0., 0.], regions)
    assert translated['root_xyz_m']['rms'] > 1.
    assert translated['root_relative_world_m']['maximum_absolute'] < 1e-14
    varied = body.copy(); varied[:, 20, 1] += .03
    report = regional_body_difference(body, varied, regions)
    assert report['regions']['left_arm']['rms'] > 0
    assert report['regions']['right_arm']['rms'] == 0


def receipt(arm='D7'):
    value = {'arm': arm, 'seed': 0, 'draw_seed': 7001, 'seconds': 180., 'plan_override': None,
        'checkpoint': '/frozen/checkpoint.pt', 'checkpoint_sha256': 'checkpoint',
        'timeline_path': '/frozen/music.npz', 'timeline_sha256': 'music', 'timeline_metadata_sha256': 'meta',
        'feature_schema_sha256': 'schema', 'source_key': 'song', 'audio_input_path': '/song.wav',
        'audio_input_sha256': 'audio', 'analyzed_waveform_sha256': 'waveform', 'audio_intervention': None,
        'newly_generated_motion': arm != 'D6', 'output_sha256': 'original'}
    if arm == 'D6':
        value.update(source_body='/accepted/body.npz', source_body_sha256='body', planner_branch='local',
                     regions=['left_arm'], effort_to_rate='frozen conversion', source_speed_estimator='incoming interval')
    else:
        value.update(base92_sha256='base92', source_clock='one full initial noise')
    if arm == 'D8':
        value['planner_sha256'] = 'planner'
    return value


@pytest.mark.parametrize('arm', ['D6', 'D7', 'D8'])
def test_only_output_hash_can_differ_in_infer_receipts(arm):
    original = receipt(arm); repeat = deepcopy(original)
    repeat['output_sha256'] = 'different zip or arrays'
    assert infer_receipt_match(original, repeat, arm)['status'] == 'matched'
    repeat['draw_seed'] = 7002
    bad = infer_receipt_match(original, repeat, arm)
    assert bad['status'] == 'invalid_same_input_provenance'
    assert 'draw_seed' in bad['changed_fields']
    assert 'draw_seed' in bad['invalid_fixed_fields']
    repeat = deepcopy(original); repeat['unregistered_new_option'] = 'changed'
    assert infer_receipt_match(original, repeat, arm)['status'] != 'matched'


def job(output):
    return {'id': 'job', 'command': ['python', 'train.py', 'infer', '--arm', 'D7', '--seed', '0', '--output', str(output)],
        'cwd': '/frozen/training_v1', 'python': '/python', 'env': {'MODE': 'fixed'},
        'source_files': {'/frozen/train.py': 'sha'}, 'inputs': {'/frozen/audio': 'input'},
        'imports': {'train': '/frozen/train.py'}, 'gpu': True, 'budget_group': 'evaluation', 'timeout_seconds': 1200}


def test_command_repeat_allows_only_output_and_bookkeeping_changes():
    a, b = job('/main/body.npz'), job('/repeat/body.npz')
    b['id'] = 'repeat-job'; b['inputs']['/protocol'] = 'protocol'
    assert command_match(a, b, '/main/body.npz', '/repeat/body.npz')['status'] == 'matched'
    b['env']['MODE'] = 'different'
    assert 'env' in command_match(a, b, '/main/body.npz', '/repeat/body.npz')['changed_fields']
    b = job('/repeat/body.npz'); b['command'][b['command'].index('--seed')+1] = '1'
    assert 'command_except_output' in command_match(a, b, '/main/body.npz', '/repeat/body.npz')['changed_fields']


def test_actual_mapper_recompile_and_color_changes_are_separate():
    from lightsem.conductor.body_transfer_v2.common import REFERENCE
    from lightsem.conductor.body_transfer_v2 import make_outputs, map_body
    from lightsem.conductor.body_transfer import compile_intent, decode
    time = np.arange(81)/40.
    body = np.repeat(REFERENCE[None], len(time), axis=0)
    body[:, 20, 1] += .03*np.sin(time*2)
    outputs = make_outputs(beams=2, lasers=0, bulbs=2)
    colors = np.full((len(time), 3), 100, np.uint8)
    def generate(rgb):
        intent = map_body(body, outputs, '01')
        dmx, patch, _ = compile_intent(intent, outputs, rgb, rgb)
        return {'time': time, 'body': body, 'targets': intent['targets'], 'universes': dmx}, patch
    original, patch = generate(colors)
    repeated, repeated_patch = generate(colors)
    assert patch == repeated_patch
    metrics, delta = mapping_comparison(original, repeated, patch, decode)
    assert metrics['universes_uint8']['full_prefix']['exact']
    assert metrics['requested_angles_deg']['full_prefix']['rms'] == 0.
    assert not np.any(delta['emitted_ticks_delta'])
    recolored, _ = generate(np.full_like(colors, 200))
    changed, _ = mapping_comparison(original, recolored, patch, decode)
    assert changed['emitted_angles_deg']['full_prefix']['exact']
    assert changed['emitted_dimmer']['full_prefix']['exact']
    assert not changed['emitted_RGB']['full_prefix']['exact']
    assert not changed['universes_uint8']['full_prefix']['exact']


def frozen_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    write_new(root/'PLAN.json', {})
    source = root/'sources/verification_v1'
    names = ['lightsem/__init__.py', 'lightsem/experiments/__init__.py',
        'lightsem/experiments/dynamic_dance/__init__.py', 'lightsem/conductor/__init__.py',
        *['lightsem/experiments/dynamic_dance/'+name+'.py' for name in ('campaign', 'contracts', 'mapping', 'diagnostics', 'evaluation')],
        'lightsem/conductor/body_transfer.py']
    files = {}
    for name in names:
        path = source/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('SOURCE_ID = "synthetic_verification_v1"\n')
        files[str(path)] = sha(path)
    path = root/'records/verification_v1_manifest.json'
    write_new(path, {'source_root': str(source), 'plan_sha256': sha(root/'PLAN.json'), 'files': files})
    return source, path


def test_frozen_source_hash_and_clean_import_admission(tmp_path):
    source, path = frozen_fixture(tmp_path)
    pins = {}
    actual, manifest = verify_frozen_source(tmp_path, sha(path), pins)
    assert actual == source
    assert str(source/'lightsem/experiments/dynamic_dance/mapping.py') in pins
    with pytest.raises(RuntimeError, match='clean process'):
        load_frozen_modules(source, manifest, {})  # pytest already imported scripts/lightsem
    changed = source/'lightsem/experiments/dynamic_dance/mapping.py'
    changed.write_text('CHANGED = True\n')
    with pytest.raises(ValueError, match='input changed'):
        verify_frozen_source(tmp_path, sha(path), {})


def test_fresh_process_resolves_all_project_imports_to_frozen_tree(tmp_path):
    source, manifest = frozen_fixture(tmp_path)
    script = Path(__file__).resolve().parents[1]/'scripts/motion/audit_dynamic_dance_repeatability.py'
    code = ('import runpy; from pathlib import Path; '
        f'm=runpy.run_path({str(script)!r},run_name="repeat_test"); '
        f'r=Path({str(tmp_path)!r}); p={{}}; '
        f's,v=m["verify_frozen_source"](r,{sha(manifest)!r},p); '
        'mods=m["load_frozen_modules"](s,v,p); '
        'assert all(Path(x.__file__).is_relative_to(s) for x in mods.values()); print("frozen-imports-ok")')
    done = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}, check=True)
    assert 'frozen-imports-ok' in done.stdout


def guard_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    Path(str(path)+'.sha256').write_text(sha(path)+'\n')


def test_guard_completion_binds_actual_runtime_environment_and_body(tmp_path):
    body = tmp_path/'body.npz'; np.savez(body, values=np.ones(3))
    j = job(body); j.update(cwd=str(tmp_path), source_files={}, inputs={}, imports={})
    queue = {'id': 'q', 'guard_sha256': 'guard', 'jobs': [j]}
    qpath = tmp_path/'queue.json'; qpath.write_text(json.dumps(queue))
    identity = {key: j[key] for key in ('command', 'cwd', 'python', 'source_files', 'inputs', 'imports', 'gpu', 'budget_group', 'timeout_seconds', 'env')}
    identity.update(job_id=j['id'], queue_id='q', queue_sha256=sha(qpath), guard_sha256='guard',
        job_sha256=hashlib.sha256(json.dumps(j, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())
    folder = tmp_path/'records/dynamic_jobs/job'
    guard_write(folder/'started.json', {'identity': identity})
    guard_write(folder/'manifest.json', {str(body): {'sha256': sha(body), 'bytes': body.stat().st_size}})
    (folder/'stdout.log').write_text('completed synthetic inference\n')
    done = {'status': 'completed', 'returncode': 0, 'identity': identity,
        'manifest_sha256': sha(folder/'manifest.json'), 'started_sha256': sha(folder/'started.json'), 'log_sha256': sha(folder/'stdout.log')}
    guard_write(folder/'completed.json', done)
    assert _guard_job(tmp_path, queue, qpath, j, body, {})['job_id'] == 'job'
    altered = deepcopy(identity); altered['env'] = {'MODE': 'actually different'}
    guard_write(folder/'started.json', {'identity': altered})
    done['identity'] = altered; done['started_sha256'] = sha(folder/'started.json')
    guard_write(folder/'completed.json', done)
    with pytest.raises(ValueError, match='environment'):
        _guard_job(tmp_path, queue, qpath, j, body, {})


def test_numeric_evidence_creates_only_new_directories_and_refuses_overwrite(tmp_path):
    path = tmp_path/'new/a/b/c/numeric.npz'
    result = _numeric(path, {'delta': np.array([0., .125])}, {'minimum_free_bytes': 0})
    assert result['sha256'] == sha(path)
    with np.load(path) as data:
        np.testing.assert_array_equal(data['delta'], [0., .125])
    with pytest.raises(FileExistsError):
        _numeric(path, {'delta': np.zeros(2)}, {'minimum_free_bytes': 0})


def inference_correction_fixture(root, monkeypatch):
    import scripts.motion.audit_dynamic_dance_repeatability as audit
    write_new(root/'PLAN.json', {})
    old_source, new_source = root/'sources/training_v1', root/'sources/inference_v2'
    old_text = 'def infer(x):\n    return {"old": x}\n'
    new_text = ('def save_inference_plan_telemetry():\n    return {"telemetry": "path"}\n\n'
                'def infer(x):\n    return {"old": x, **save_inference_plan_telemetry()}\n')
    contents = {'scripts/motion/train_dynamic_dance.py': ('CLI = True\n', 'CLI = True\n'),
                'lightsem/experiments/dynamic_dance/training.py': (old_text, new_text),
                'tests/test_dynamic_dance_learning.py': ('OLD_TEST = True\n', 'NEW_TEST = True\n')}
    maps = [{}, {}]
    for relative, texts in contents.items():
        for index, (source, text) in enumerate(zip((old_source, new_source), texts)):
            path = source/relative
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
            maps[index][str(path)] = sha(path)
    base_path = root/'records/training_v1_manifest.json'
    write_new(base_path, {'source_root': str(old_source), 'plan_sha256': sha(root/'PLAN.json'), 'files': maps[0]})
    original_queue = root/'records/queues/infer-learned-v1.json'
    write_new(original_queue, {'jobs': [], 'scope': 'synthetic preserved unexecuted queue'})
    correction_path = root/'records/INFERENCE_TELEMETRY_CORRECTION_PROTOCOL.json'
    write_new(correction_path, {'status': 'frozen_before_any_showcase_inference', 'new_inference_source': 'inference_v2',
        'original_training_and_evaluation_source': 'training_v1', 'no_model_or_outcome_selection': True,
        'original_inference_queue': str(original_queue), 'original_inference_queue_sha256': sha(original_queue)})
    new_manifest = root/'records/inference_v2_manifest.json'
    write_new(new_manifest, {'source_root': str(new_source), 'plan_sha256': sha(root/'PLAN.json'), 'files': maps[1],
        'based_on_manifest': str(base_path), 'based_on_manifest_sha256': sha(base_path),
        'correction_protocol_sha256': sha(correction_path)})
    relative = 'lightsem/experiments/dynamic_dance/training.py'
    proof_path = root/'records/INFERENCE_TELEMETRY_DIFF_VERIFICATION.json'
    write_new(proof_path, {'body_metadata_and_inference_receipt_provenance_unchanged': True,
        'entire_module_AST_identical_after_removing_new_IO_helper_and_metadata_unpack': True,
        'model_inputs_weights_RNG_math_and_call_order_unchanged': True,
        'training_and_evaluation_unchanged': True, 'original_inference_queue_executed': False,
        'removed_calls': 1, 'removed_helpers': 1, 'only_changed_existing_paths': list(audit.TELEMETRY_CHANGED_PATHS),
        'original_source': str(old_source/relative), 'new_source': str(new_source/relative),
        'original_sha256': sha(old_source/relative), 'new_sha256': sha(new_source/relative)})
    # Production exposes no override option. Tests replace only fixed fixture digests.
    monkeypatch.setattr(audit, 'INFERENCE_CORRECTION_PINS', {path.name: sha(path) for path in (proof_path, new_manifest, correction_path)})
    script, module = new_source/'scripts/motion/train_dynamic_dance.py', new_source/relative
    job = {'cwd': str(new_source), 'python': '/python', 'command': ['/python', str(script), 'infer'],
           'source_files': maps[1], 'imports': {'lightsem.experiments.dynamic_dance.training': str(module)},
           'inputs': {str(root/'records'/name): digest for name, digest in audit.INFERENCE_CORRECTION_PINS.items()}}
    return job, proof_path, new_manifest


def test_inference_job_must_bind_only_approved_v2_correction(tmp_path, monkeypatch):
    from scripts.motion.audit_dynamic_dance_repeatability import training_job_binding, authenticate_inference_correction
    value, proof_path, manifest_path = inference_correction_fixture(tmp_path, monkeypatch)
    pins = {}
    proof = authenticate_inference_correction(tmp_path, pins)
    assert proof['independent_AST_check']['entire_module_AST_restored']
    assert proof['cloned_files'] == 3
    assert str(proof_path) in pins and str(manifest_path) in pins
    assert str(Path(str(proof_path)+'.sha256.json')) in pins
    assert training_job_binding(tmp_path, value, {})['script'] == value['command'][1]
    value['imports']['lightsem.experiments.dynamic_dance.training'] = str(tmp_path/'sources/training_v1/lightsem/experiments/dynamic_dance/training.py')
    with pytest.raises(ValueError, match='exact approved inference_v2'):
        training_job_binding(tmp_path, value, {})


def test_resealed_proof_drift_cannot_change_fixed_correction_allowlist(tmp_path, monkeypatch):
    from scripts.motion.audit_dynamic_dance_repeatability import authenticate_inference_correction
    _, proof_path, _ = inference_correction_fixture(tmp_path, monkeypatch)
    proof = json.loads(proof_path.read_text())
    proof['training_and_evaluation_unchanged'] = False
    proof_path.write_text(json.dumps(proof))
    Path(str(proof_path)+'.sha256.json').write_text(json.dumps({'sha256': sha(proof_path)}))
    with pytest.raises(ValueError, match='Authenticated input changed'):
        authenticate_inference_correction(tmp_path, {})


def test_changed_cloned_dependency_is_rejected(tmp_path, monkeypatch):
    from scripts.motion.audit_dynamic_dance_repeatability import authenticate_inference_correction
    job, _, _ = inference_correction_fixture(tmp_path, monkeypatch)
    Path(job['command'][1]).write_text('CLI = "unapproved"\n')
    with pytest.raises(ValueError, match='Authenticated input changed'):
        authenticate_inference_correction(tmp_path, {})


def test_independent_ast_check_rejects_math_change_and_extra_telemetry_call():
    from scripts.motion.audit_dynamic_dance_repeatability import verify_telemetry_ast
    old = 'def infer(x):\n    return {"old": x}\n'
    corrected = ('def save_inference_plan_telemetry():\n    return {}\n'
                 'def infer(x):\n    return {"old": x, **save_inference_plan_telemetry()}\n')
    assert verify_telemetry_ast(old, corrected)['entire_module_AST_restored']
    with pytest.raises(ValueError, match='computational module AST'):
        verify_telemetry_ast(old, corrected.replace('"old": x,', '"old": x+1,'))
    with pytest.raises(ValueError, match='computational module AST'):
        verify_telemetry_ast(old, corrected.replace('**save_inference_plan_telemetry()', '**save_inference_plan_telemetry(), **save_inference_plan_telemetry()'))


def test_v2_execution_must_pin_whole_tree_and_approval_records(tmp_path, monkeypatch):
    from scripts.motion.audit_dynamic_dance_repeatability import training_job_binding
    value, proof_path, _ = inference_correction_fixture(tmp_path, monkeypatch)
    missing_source = deepcopy(value)
    del missing_source['source_files'][str(tmp_path/'sources/inference_v2/tests/test_dynamic_dance_learning.py')]
    with pytest.raises(ValueError, match='complete approved'):
        training_job_binding(tmp_path, missing_source, {})
    del value['inputs'][str(proof_path)]
    with pytest.raises(ValueError, match='did not freeze'):
        training_job_binding(tmp_path, value, {})


def repeat_execution_fixture(root, monkeypatch):
    import scripts.motion.audit_dynamic_dance_repeatability as audit
    write_new(root/'PLAN.json', {})
    write_new(root/'records/REPEATABILITY_PROTOCOL_v1.json', {'scope': 'synthetic exact six repeats'})
    originals, repeats = [], []
    queues = {}
    for arm in audit.LEARNED:
        for song in audit.SONGS:
            v2 = (arm, song) == ('D6', 'shpongle')
            source = root/'sources'/('inference_v2' if v2 else 'inference_v3')
            original_dir, repeat_dir = root/'bodies'/arm/song, root/'repeatability'/arm/song
            original_queue = root/'records/queues'/('infer-learned-v2.json' if v2 else 'infer-remainder-v3.json')
            repeat_queue = root/'records/queues'/('repeat-learned-v2-shpongle.json' if v2 else 'repeat-learned-v3.json')
            for path in (original_queue, repeat_queue):
                if path not in queues:
                    write_new(path, {'scope': 'synthetic queue'}); queues[path] = sha(path)
            original_command = ['/python', str(source/'scripts/motion/train_dynamic_dance.py'), 'infer',
                                '--arm', arm, '--seed', '0', '--output', str(original_dir/'body.npz')]
            repeat_command = list(original_command); repeat_command[-1] = str(repeat_dir/'body.npz')
            entry = {'method': arm, 'song': song, 'source': str(source), 'job_id': 'original-'+arm+'-'+song,
                     'queue': str(original_queue), 'queue_sha256': queues[original_queue],
                     'canonical_command': original_command, 'standard_output_directory': str(original_dir), 'recovery_receipt': None}
            originals.append(entry)
            repeats.append({'method': arm, 'song': song, 'source': str(source), 'original_job_id': entry['job_id'],
                'original_queue': entry['queue'], 'original_queue_sha256': entry['queue_sha256'],
                'original_canonical_command': original_command, 'original_output_directory': str(original_dir),
                'original_recovery_receipt': None, 'repeat_queue': str(repeat_queue), 'repeat_queue_sha256': queues[repeat_queue],
                'repeat_job_id': 'repeat-'+arm+'-'+song, 'repeat_canonical_command': repeat_command,
                'repeat_output_directory': str(repeat_dir)})
    path = root/'records/REPEAT_EXECUTION_PLAN_v3.json'
    plan = {'schema_version': 1, 'plan_sha256': sha(root/'PLAN.json'),
            'generation_execution_plan': str(root/'records/GENERATION_EXECUTION_PLAN_v3.json'),
            'generation_execution_plan_sha256': audit.GENERATION_PLAN_SHA256,
            'repeatability_protocol_sha256': sha(root/'records/REPEATABILITY_PROTOCOL_v1.json'), 'entries': repeats}
    write_new(path, plan)
    monkeypatch.setattr(audit, 'REPEAT_EXECUTION_PLAN_SHA256', sha(path))
    return {'entries': originals}, plan, path


def test_repeat_plan_keeps_one_v2_original_and_five_v3_cases(tmp_path, monkeypatch):
    from scripts.motion.audit_dynamic_dance_repeatability import authenticate_repeat_plan
    original, plan, path = repeat_execution_fixture(tmp_path, monkeypatch)
    report = authenticate_repeat_plan(tmp_path, sha(path), {}, generation=original)
    assert sum(row['source'].endswith('inference_v2') for row in report['entries']) == 1
    assert sum(row['source'].endswith('inference_v3') for row in report['entries']) == 5
    assert report['entries'][0]['original_canonical_command'][1].endswith('inference_v2/scripts/motion/train_dynamic_dance.py')


@pytest.mark.parametrize('change', ['source', 'seed', 'duplicate', 'original_job'])
def test_repeat_plan_rejects_source_arguments_and_cohort_substitution(tmp_path, monkeypatch, change):
    import scripts.motion.audit_dynamic_dance_repeatability as audit
    original, plan, path = repeat_execution_fixture(tmp_path, monkeypatch)
    if change == 'source':
        plan['entries'][0]['source'] = str(tmp_path/'workspace')
    elif change == 'seed':
        command = plan['entries'][0]['repeat_canonical_command']; command[command.index('--seed')+1] = '1'
    elif change == 'duplicate':
        plan['entries'][1] = deepcopy(plan['entries'][0])
    else:
        plan['entries'][0]['original_job_id'] = 'different-original'
    path.write_text(json.dumps(plan))
    Path(str(path)+'.sha256.json').write_text(json.dumps({'sha256': sha(path)}))
    # Exercise structural binding in addition to the fixed digest rejection tested elsewhere.
    monkeypatch.setattr(audit, 'REPEAT_EXECUTION_PLAN_SHA256', sha(path))
    with pytest.raises(ValueError):
        audit.authenticate_repeat_plan(tmp_path, sha(path), {}, generation=original)


def test_repeat_plan_proof_drift_fails_even_with_new_seal(tmp_path, monkeypatch):
    import scripts.motion.audit_dynamic_dance_repeatability as audit
    original, plan, path = repeat_execution_fixture(tmp_path, monkeypatch)
    approved = sha(path)
    plan['entries'][0]['repeat_job_id'] = 'unapproved-repeat'
    path.write_text(json.dumps(plan))
    Path(str(path)+'.sha256.json').write_text(json.dumps({'sha256': sha(path)}))
    with pytest.raises(ValueError, match='Authenticated input changed'):
        audit.authenticate_repeat_plan(tmp_path, approved, {}, generation=original)

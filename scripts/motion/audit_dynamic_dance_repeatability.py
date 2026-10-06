"""Measure saved-body mapper and same-input learned repeat variation on CPU.

Original main outputs remain displayed. This auditor never generates a dancer,
changes a mapper, chooses a repeat, or assigns a musical/scientific verdict.
The CLI launches a clean worker importing the complete verification_v1 tree.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1')
METHODS = tuple('D'+str(i) for i in range(1, 9))
SONGS = ('shpongle', 'cure_for_me')
LEARNED = ('D6', 'D7', 'D8')
AUDITOR_VERSION = 'repeatability-v3-explicit-per-case-execution-and-recovery'
INFERENCE_CORRECTION_PINS = {
    'INFERENCE_TELEMETRY_DIFF_VERIFICATION.json': '0cb3c82730e555e071077febd7ffdf8c4f2aafe4a226ec19ef98b6586873864d',
    'inference_v2_manifest.json': '5511736e15de9e68b467a1d610e8a76fc5d35403690936efa7dec5f5712f2483',
    'INFERENCE_TELEMETRY_CORRECTION_PROTOCOL.json': '035122ac5eb91a25f3e4958f2684331a0079a9ddbe73cfb37b3d0bce3bba1c71',
}
TELEMETRY_CHANGED_PATHS = ('lightsem/experiments/dynamic_dance/training.py', 'tests/test_dynamic_dance_learning.py')
NUMERICAL_CORRECTION_PINS = {
    'inference_v3_manifest.json': 'd9b2b74c1789cc25a7394b72296b6e1280783c92a67971e04b28ce57e46a62db',
    'RETIMING_NUMERICAL_CORRECTION_PROTOCOL_v3.json': 'ed78c725e20e040df0bcaea72e7d81b2b77d37c9f3ed0f886cb0de3254bd958f',
    'RETIMING_NUMERICAL_DIFF_VERIFICATION_v3.json': '85f3b17e68f045ff734cea855f2b722e7ecc48e74ce19fc327e86830906a3ca7',
}
GENERATION_PLAN_SHA256 = 'e0dc13cd9441a4d11458e8b2bf8f239f3f5f5dd557ce72d4533483660858184c'
REPEAT_EXECUTION_PLAN_SHA256 = 'fd9dd120dc93dfc322af7bbcec3d59ead9750662b45e050566cf62d334b65235'
RECOVERY_WRAPPER_SHA256 = '4b49a5b5f90b8d7962742bde065c56f2830c6f8f288149f47e4bd1db93340840'
RECOVERY_AUTH_SHA256 = '69a6cd7a0faaeb12d75a7bb31a9e0680c047cb12a7424933018f85915c133802'
RECOVERY_PRIMARY_SHA256 = '923ee8e4b68c9413fd8793ba0a7a6d8eb8c1a5600954a14259eee5c9a8a3b7da'
RECOVERY_FILES = ('body.npz', 'body.json', 'body.json.sha256.json', 'inference_receipt.json',
                  'inference_receipt.json.sha256.json', 'learned_plan.npz', 'plan.npz', 'plan.json', 'plan.json.sha256.json')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    seal = Path(str(path)+'.sha256.json')
    if path.exists() or seal.exists():
        raise FileExistsError('Preserve existing repeatability evidence: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    with seal.open('x') as stream:
        json.dump({'sha256': sha(path)}, stream)
        stream.write('\n')
    return sha(path)


def _pin(path, pins, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError('Authenticated input changed: '+str(path))
    prior = pins.get(str(path))
    if prior is not None and prior != actual:
        raise ValueError('Input changed during audit: '+str(path))
    pins[str(path)] = actual
    return actual


def authenticated_json(path, pins, *, guard=False):
    path = Path(path)
    seal = Path(str(path)+('.sha256' if guard else '.sha256.json'))
    expected = seal.read_text().strip() if guard else read(seal)['sha256']
    _pin(path, pins, expected); _pin(seal, pins)
    return read(path)


def array_difference(original, repeat):
    """Numerical and byte equality are separate; NPZ container hashes are not arrays."""
    if original is None and repeat is None:
        return {'status': 'not_applicable', 'exact': None, 'byte_exact': None, 'rms': None, 'maximum_absolute': None}
    if original is None or repeat is None:
        return {'status': 'missing_from_one', 'exact': None, 'byte_exact': None, 'rms': None, 'maximum_absolute': None}
    a, b = np.asarray(original), np.asarray(repeat)
    detail = {'original_shape': list(a.shape), 'repeat_shape': list(b.shape),
              'original_dtype': str(a.dtype), 'repeat_dtype': str(b.dtype)}
    if a.shape != b.shape:
        return {**detail, 'status': 'shape_mismatch', 'exact': False, 'byte_exact': False, 'rms': None, 'maximum_absolute': None}
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {**detail, 'status': 'nonfinite', 'exact': None, 'byte_exact': None, 'rms': None, 'maximum_absolute': None}
    if not a.size:
        return {**detail, 'status': 'no_observations', 'exact': True, 'byte_exact': a.dtype == b.dtype, 'rms': None, 'maximum_absolute': None}
    delta = b.astype(np.float64)-a.astype(np.float64)
    exact = bool(np.array_equal(a, b))
    return {**detail, 'status': 'measured', 'exact': exact,
            'byte_exact': bool(a.dtype == b.dtype and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()),
            'rms': float(np.sqrt(np.mean(delta**2))), 'maximum_absolute': float(np.abs(delta).max()),
            'fraction_values_different': float(np.mean(a != b))}


def windowed_difference(a, b, time):
    t = np.asarray(time)
    full = array_difference(a, b)
    if a is None or b is None:
        return {'full_prefix': full, 'review_60_180': dict(full)}
    if len(a) != len(t) or len(b) != len(t):
        return {'full_prefix': full, 'review_60_180': {'status': 'clock_mismatch', 'rms': None, 'maximum_absolute': None}}
    mask = (t >= 60.) & (t < 180.)
    return {'full_prefix': full, 'review_60_180': array_difference(np.asarray(a)[mask], np.asarray(b)[mask])}


def regional_body_difference(a, b, regions):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (22, 3):
        raise ValueError('Same-clock22-joint bodies required')
    relative_a, relative_b = a-a[:, :1], b-b[:, :1]
    return {'root_xyz_m': array_difference(a[:, 0], b[:, 0]),
            'root_relative_world_m': array_difference(relative_a, relative_b),
            'regions': {name: array_difference(relative_a[:, joints], relative_b[:, joints]) for name, joints in regions.items()},
            'scope': 'Observed one-repeat root-relative position differences in metres; not a statistical upper bound.'}


def infer_receipt_match(original, repeat, arm):
    """Only the output content hash may differ in frozen infer receipts."""
    a = {key: value for key, value in original.items() if key != 'output_sha256'}
    b = {key: value for key, value in repeat.items() if key != 'output_sha256'}
    changed = sorted(key for key in set(a)|set(b) if key not in a or key not in b or a[key] != b[key])
    required = ('arm', 'seed', 'draw_seed', 'seconds', 'plan_override', 'checkpoint', 'checkpoint_sha256',
                'timeline_path', 'timeline_sha256', 'timeline_metadata_sha256', 'feature_schema_sha256',
                'source_key', 'audio_input_path', 'audio_input_sha256', 'analyzed_waveform_sha256', 'audio_intervention')
    required += ('source_body', 'source_body_sha256', 'planner_branch', 'regions', 'effort_to_rate', 'source_speed_estimator') if arm == 'D6' else ('base92_sha256', 'source_clock')
    if arm == 'D8':
        required += ('planner_sha256',)
    missing = sorted(key for key in required if key not in a or key not in b)
    constants = {'arm': arm, 'seed': 0, 'draw_seed': 7001, 'seconds': 180., 'plan_override': None, 'audio_intervention': None}
    invalid = sorted(key for key, value in constants.items() if a.get(key) != value or b.get(key) != value)
    if a.get('newly_generated_motion') is not (arm != 'D6') or b.get('newly_generated_motion') is not (arm != 'D6'):
        invalid.append('newly_generated_motion')
    return {'status': 'matched' if not changed and not missing and not invalid else 'invalid_same_input_provenance',
            'changed_fields': changed, 'missing_fields': missing, 'invalid_fixed_fields': invalid,
            'scope': 'Whole authenticated inference receipt, excluding only output_sha256; no arbitrary metadata exclusions.'}


def command_match(original_job, repeat_job, original_output, repeat_output):
    def normalized(job, expected):
        command = list(job['command'])
        if command.count('--output') != 1:
            raise ValueError('Exactly one explicit inference output required')
        index = command.index('--output')+1
        if command[index] != str(expected):
            raise ValueError('Infer CLI does not point to declared output')
        command[index] = '<declared-output>'
        return command
    changed = []
    if normalized(original_job, original_output) != normalized(repeat_job, repeat_output):
        changed.append('command_except_output')
    for key in ('cwd', 'python', 'env', 'source_files', 'imports', 'gpu', 'budget_group', 'timeout_seconds'):
        if original_job.get(key) != repeat_job.get(key):
            changed.append(key)
    if any(repeat_job.get('inputs', {}).get(path) != digest for path, digest in original_job.get('inputs', {}).items()):
        changed.append('original_input_hashes')
    return {'status': 'matched' if not changed else 'invalid_repeat_command', 'changed_fields': changed}


def requested_angles(targets, patch):
    movers = [i for i, fixture in enumerate(patch) if fixture['kind'] != 'bulb']
    positions = np.asarray([patch[i]['position'] for i in movers], float).reshape(-1, 3)
    vectors = np.asarray(targets)[:, movers]-positions
    length = np.linalg.norm(vectors, axis=-1)
    if np.any(length <= 0):
        raise ValueError('Target at moving-fixture origin')
    return np.stack((np.degrees(np.arctan2(vectors[..., 0], -vectors[..., 1])),
                     np.degrees(np.arcsin(np.clip(vectors[..., 2]/length, -1, 1)))), axis=-1)


def mapping_comparison(original, recomputed, patch, decoder):
    if not np.array_equal(original['time'], recomputed['time']):
        raise ValueError('Recompiled mapping changed timestamp clock')
    time = original['time']
    a, b = decoder(original['universes'], patch), decoder(recomputed['universes'], patch)
    moving = np.array([p['kind'] != 'bulb' for p in patch])
    ar, br = requested_angles(original['targets'], patch), requested_angles(recomputed['targets'], patch)
    ranges = np.asarray([[p['pan_range_deg'], p['tilt_range_deg']] for p in patch if p['kind'] != 'bulb']).reshape(-1, 2)
    at = np.rint((.5+a['angles_deg'][:, moving]/ranges)*65535).astype(np.int64)
    bt = np.rint((.5+b['angles_deg'][:, moving]/ranges)*65535).astype(np.int64)
    result = {name: windowed_difference(x, y, time) for name, x, y in (
        ('body40_m', original['body'], recomputed['body']), ('requested_targets_m', original['targets'], recomputed['targets']),
        ('requested_angles_deg', ar, br), ('emitted_angles_deg', a['angles_deg'][:, moving], b['angles_deg'][:, moving]),
        ('emitted_command_ticks', at, bt), ('emitted_RGB', a['rgb'], b['rgb']),
        ('emitted_dimmer', a['dimmer'], b['dimmer']), ('universes_uint8', original['universes'], recomputed['universes']))}
    result['scope'] = 'Exact compiler atan2 branch; decoded command motion, color and brightness are separate.'
    return result, {'time': time, 'requested_angles_delta_deg': br-ar,
                    'emitted_angles_delta_deg': b['angles_deg'][:, moving]-a['angles_deg'][:, moving],
                    'emitted_ticks_delta': bt-at, 'RGB_delta': b['rgb']-a['rgb'], 'dimmer_delta': b['dimmer']-a['dimmer']}


def verify_frozen_source(root, expected_manifest, pins):
    path = root/'records/verification_v1_manifest.json'
    _pin(path, pins, expected_manifest)
    manifest = read(path)
    source = (root/'sources/verification_v1').resolve()
    if Path(manifest['source_root']).resolve() != source or manifest['plan_sha256'] != sha(root/'PLAN.json'):
        raise ValueError('verification_v1 belongs to another source root or PLAN')
    for filename, expected in manifest['files'].items():
        if not Path(filename).resolve().is_relative_to(source):
            raise ValueError('Frozen mapper source escaped verification_v1')
        _pin(filename, pins, expected)
    return source, manifest


def verify_imported_sources(source, manifest, pins):
    checked = {}
    for name, module in tuple(sys.modules.items()):
        if name not in ('lightsem', 'scripts') and not name.startswith(('lightsem.', 'scripts.')):
            continue
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(source) or str(path) not in manifest['files']:
            raise ValueError('Mixed/unfrozen mapper import: '+name+' '+str(path))
        _pin(path, pins, manifest['files'][str(path)])
        checked[name] = str(path)
    return checked


def load_frozen_modules(source, manifest, pins):
    foreign = [name for name in sys.modules if name in ('lightsem', 'scripts') or name.startswith(('lightsem.', 'scripts.'))]
    if foreign:
        raise RuntimeError('Frozen mapper requires a clean process before any project imports')
    sys.path.insert(0, str(source))
    modules = {name: importlib.import_module('lightsem.experiments.dynamic_dance.'+name)
               for name in ('campaign', 'contracts', 'mapping', 'diagnostics', 'evaluation')}
    modules['decoder'] = importlib.import_module('lightsem.conductor.body_transfer')
    verify_imported_sources(source, manifest, pins)
    return modules


def validate_protocol(root, expected, pins):
    path = root/'records/REPEATABILITY_PROTOCOL_v1.json'
    _pin(path, pins, expected)
    protocol = read(path)
    learned, mapper = protocol['learned_repeats'], protocol['mapper_repeats']
    if (protocol.get('status') != 'frozen_before_learned_showcase_generation' or tuple(learned['arms']) != LEARNED
            or tuple(learned['songs']) != SONGS or learned['count'] != 6 or learned['draw_seed'] != 7001
            or learned['training_seed'] != 0 or learned['seconds'] != 180
            or tuple(mapper['methods']) != METHODS or tuple(mapper['songs']) != SONGS):
        raise ValueError('Repeatability protocol differs from frozen16+6 cohort')
    return protocol


def nominal_clock_metrics(time):
    """Existing native-clock admission tolerance; preserve every stored value."""
    t = np.asarray(time, float)
    if t.shape != (5400,) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Expected complete finite native30Hz clock')
    nominal = np.arange(5400)/30.
    error = t-nominal
    maximum = float(np.abs(error).max())
    if maximum > 1e-8:
        raise ValueError('Native timestamps differ meaningfully from nominal30Hz clock')
    return {'nominal_fps': 30., 'existing_admission_tolerance_s': 1e-8,
            'exactly_nominal': bool(np.array_equal(t, nominal)),
            'maximum_absolute_nominal_error_s': maximum, 'rms_nominal_error_s': float(np.sqrt(np.mean(error**2))),
            'saved_first_s': float(t[0]), 'saved_last_s': float(t[-1]), 'timestamps_rewritten': False}


def _body(path, modules, pins):
    body = modules['contracts'].DanceResult.load(path)
    for p in (path, path.with_suffix('.json'), path.with_suffix('.json.sha256.json')):
        _pin(p, pins)
    if body.world.shape != (5400, 22, 3):
        raise ValueError('Repeat audit requires complete180-second native30Hz body')
    nominal_clock_metrics(body.time)
    return body


def _guard_job(root, queue, queue_path, job, body_path, pins):
    _pin(queue_path, pins)
    folder = root/'records/dynamic_jobs'/job['id']
    done = authenticated_json(folder/'completed.json', pins, guard=True)
    started = authenticated_json(folder/'started.json', pins, guard=True)
    manifest = authenticated_json(folder/'manifest.json', pins, guard=True)
    if done.get('status') != 'completed' or done.get('returncode') != 0 or done['identity'] != started['identity']:
        raise ValueError('Inference guard did not complete authenticated job')
    identity = done['identity']
    object_sha = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if identity.get('job_sha256') != object_sha or identity.get('queue_sha256') != sha(queue_path):
        raise ValueError('Guard job/queue identity differs')
    fields = ('command', 'cwd', 'python', 'source_files', 'inputs', 'imports', 'gpu', 'budget_group', 'timeout_seconds')
    if (any(identity.get(key) != job.get(key) for key in fields)
            or identity.get('env', {}) != job.get('env', {})
            or identity.get('job_id') != job['id'] or identity.get('queue_id') != queue['id']
            or identity.get('guard_sha256') != queue['guard_sha256']
            or identity.get('outputs') != job.get('outputs')):
        raise ValueError('Guard execution differs from frozen command/source/environment')
    for field, name in (('manifest_sha256', 'manifest.json'), ('started_sha256', 'started.json'), ('log_sha256', 'stdout.log')):
        _pin(folder/name, pins, done[field])
    if str(body_path) not in manifest or manifest[str(body_path)]['sha256'] != sha(body_path):
        raise ValueError('Guard output manifest does not bind body')
    for path, info in manifest.items():
        _pin(path, pins, info['sha256'])
    for key in ('inputs', 'source_files'):
        for path, expected in job.get(key, {}).items():
            _pin(path, pins, expected)
    return {'job_id': job['id'], 'queue': str(queue_path), 'completed': str(folder/'completed.json')}


def verify_telemetry_ast(original_text, corrected_text):
    """Independently reproduce the narrowly registered computational identity."""
    before, after = ast.parse(original_text), ast.parse(corrected_text)
    name = 'save_inference_plan_telemetry'
    helpers = [node for node in after.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(helpers) != 1:
        raise ValueError('Telemetry correction must introduce exactly the registered helper')
    after.body = [node for node in after.body if node not in helpers]
    class RemoveMetadataUnpack(ast.NodeTransformer):
        removed = 0
        def visit_Dict(self, node):
            pairs = []
            for key, value in zip(node.keys, node.values):
                if key is None and isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == name:
                    self.removed += 1
                else:
                    pairs.append((key, value))
            node.keys = [key for key, _ in pairs]
            node.values = [value for _, value in pairs]
            return self.generic_visit(node)
    remover = RemoveMetadataUnpack()
    after = remover.visit(after)
    if remover.removed != 1 or ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
        raise ValueError('Telemetry correction changed original computational module AST')
    return {'entire_module_AST_restored': True, 'removed_helpers': 1, 'removed_metadata_unpacks': 1}


def authenticate_inference_correction(root, pins):
    """Accept only the specific root-approved correction, never arbitrary sources."""
    root = Path(root).resolve()
    paths = {name: root/'records'/name for name in INFERENCE_CORRECTION_PINS}
    for name, path in paths.items():
        _pin(path, pins, INFERENCE_CORRECTION_PINS[name])
    proof = authenticated_json(paths['INFERENCE_TELEMETRY_DIFF_VERIFICATION.json'], pins)
    manifest = authenticated_json(paths['inference_v2_manifest.json'], pins)
    correction = read(paths['INFERENCE_TELEMETRY_CORRECTION_PROTOCOL.json'])
    old_source, new_source = root/'sources/training_v1', root/'sources/inference_v2'
    base_path = root/'records/training_v1_manifest.json'
    original_module = old_source/TELEMETRY_CHANGED_PATHS[0]
    corrected_module = new_source/TELEMETRY_CHANGED_PATHS[0]
    original_queue = root/'records/queues/infer-learned-v1.json'
    if (correction.get('status') != 'frozen_before_any_showcase_inference'
            or correction.get('new_inference_source') != 'inference_v2'
            or correction.get('original_training_and_evaluation_source') != 'training_v1'
            or correction.get('no_model_or_outcome_selection') is not True
            or correction.get('original_inference_queue') != str(original_queue)):
        raise ValueError('Telemetry correction does not declare the fixed source/selection policy')
    _pin(original_queue, pins, correction['original_inference_queue_sha256'])
    true_fields = ('body_metadata_and_inference_receipt_provenance_unchanged',
                   'entire_module_AST_identical_after_removing_new_IO_helper_and_metadata_unpack',
                   'model_inputs_weights_RNG_math_and_call_order_unchanged', 'training_and_evaluation_unchanged')
    if (any(proof.get(key) is not True for key in true_fields)
            or proof.get('original_inference_queue_executed') is not False
            or proof.get('removed_calls') != 1 or proof.get('removed_helpers') != 1
            or sorted(proof.get('only_changed_existing_paths', [])) != sorted(TELEMETRY_CHANGED_PATHS)
            or proof.get('original_source') != str(original_module) or proof.get('new_source') != str(corrected_module)):
        raise ValueError('Telemetry AST proof does not authenticate the registered limited correction')
    if (manifest.get('source_root') != str(new_source)
            or manifest.get('based_on_manifest') != str(base_path)
            or manifest.get('correction_protocol_sha256') != INFERENCE_CORRECTION_PINS['INFERENCE_TELEMETRY_CORRECTION_PROTOCOL.json']
            or manifest.get('plan_sha256') != sha(root/'PLAN.json')):
        raise ValueError('Corrected inference manifest has another source/PLAN/addendum identity')
    _pin(base_path, pins, manifest['based_on_manifest_sha256'])
    base = read(base_path)
    if base.get('source_root') != str(old_source) or base.get('plan_sha256') != sha(root/'PLAN.json'):
        raise ValueError('Original training manifest identity changed')
    base_files, new_files = {}, {}
    for collection, source, target in ((base['files'], old_source, base_files), (manifest['files'], new_source, new_files)):
        for filename, expected in collection.items():
            path = Path(filename)
            if not path.resolve().is_relative_to(source):
                raise ValueError('Inference source manifest escaped its declared tree')
            target[path.relative_to(source).as_posix()] = expected
            _pin(path, pins, expected)
    if set(new_files) != set(base_files):
        raise ValueError('This approved inference correction must clone the entire original source-file set')
    changed = sorted(name for name in base_files if base_files[name] != new_files[name])
    if changed != sorted(TELEMETRY_CHANGED_PATHS):
        raise ValueError('Inference correction changed an unapproved original source file')
    _pin(original_module, pins, proof['original_sha256'])
    _pin(corrected_module, pins, proof['new_sha256'])
    restored = verify_telemetry_ast(original_module.read_text(), corrected_module.read_text())
    return {'source_root': str(new_source), 'manifest': str(paths['inference_v2_manifest.json']),
            'manifest_sha256': INFERENCE_CORRECTION_PINS['inference_v2_manifest.json'],
            'proof_sha256': INFERENCE_CORRECTION_PINS['INFERENCE_TELEMETRY_DIFF_VERIFICATION.json'],
            'addendum_sha256': INFERENCE_CORRECTION_PINS['INFERENCE_TELEMETRY_CORRECTION_PROTOCOL.json'],
            'cloned_files': len(base_files), 'changed_paths': changed, 'independent_AST_check': restored,
            'scope': 'Explicit telemetry-only correction; original numerical computation and repeat settings unchanged.'}


def verify_regional_roundoff_ast(original_text, corrected_text):
    before, after = ast.parse(original_text), ast.parse(corrected_text)
    helpers = {'_regional_mean_roundoff_bound', '_project_regional_rates'}
    found = [node.name for node in after.body if isinstance(node, ast.FunctionDef) and node.name in helpers]
    imports = [node for node in after.body if isinstance(node, ast.Import) and len(node.names) == 1
               and node.names[0].name in ('json', 'warnings') and node.names[0].asname is None]
    if set(found) != helpers or len(found) != 2 or sorted(node.names[0].name for node in imports) != ['json', 'warnings']:
        raise ValueError('Numerical correction must have exactly the registered helpers/imports')
    old_projector = [node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == '_project_rates']
    new_projector = [node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == '_project_rates']
    if len(old_projector) != 1 or len(new_projector) != 1 or ast.dump(old_projector[0], include_attributes=False) != ast.dump(new_projector[0], include_attributes=False):
        raise ValueError('Original physical _project_rates changed')
    after.body = [node for node in after.body if node not in imports and not (isinstance(node, ast.FunctionDef) and node.name in helpers)]
    class RestoreCall(ast.NodeTransformer):
        restored = 0
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_project_regional_rates':
                if len(node.args) != 6 or node.keywords or [getattr(arg, 'id', None) for arg in node.args[2:]] != ['common_time', 'start', 'end', 'fps']:
                    raise ValueError('Unexpected numerical correction call signature')
                self.restored += 1
                node.func.id = '_project_rates'; node.args = node.args[:2]
            return self.generic_visit(node)
    restore = RestoreCall(); after = restore.visit(after)
    if restore.restored != 1 or ast.dump(before, include_attributes=False) != ast.dump(after, include_attributes=False):
        raise ValueError('Numerical correction changed the original full module AST')
    return {'whole_module_AST_restored': True, '_project_rates_unchanged': True,
            'restored_calls': 1, 'removed_helpers': 2, 'removed_imports': 2}


def authenticate_numerical_correction(root, pins):
    root = Path(root).resolve()
    telemetry = authenticate_inference_correction(root, pins)
    records = {}
    for name, expected in NUMERICAL_CORRECTION_PINS.items():
        path = root/'records'/name; _pin(path, pins, expected)
        records[name] = authenticated_json(path, pins)
    manifest = records['inference_v3_manifest.json']
    proof = records['RETIMING_NUMERICAL_DIFF_VERIFICATION_v3.json']
    protocol = records['RETIMING_NUMERICAL_CORRECTION_PROTOCOL_v3.json']
    old_source, new_source = root/'sources/inference_v2', root/'sources/inference_v3'
    relative = 'lightsem/experiments/dynamic_dance/retiming.py'
    if (protocol.get('old_source') != 'inference_v2' or protocol.get('new_source') != 'inference_v3'
            or protocol.get('status') != 'frozen_before_corrected_inference'):
        raise ValueError('Numerical correction source/status is unapproved')
    for filename, digest in protocol.get('records', {}).items():
        _pin(filename, pins, digest)
    if (manifest.get('source_root') != str(new_source) or manifest.get('plan_sha256') != sha(root/'PLAN.json')
            or manifest.get('based_on_manifest') != str(root/'records/inference_v2_manifest.json')
            or manifest.get('based_on_manifest_sha256') != INFERENCE_CORRECTION_PINS['inference_v2_manifest.json']):
        raise ValueError('Numerical source manifest has unapproved ancestry')
    true_fields = ('successful_projector_path_returns_original_directly', 'unchanged_project_rates_AST',
                   'whole_module_AST_restored_by_removing_two_helpers_two_imports_and_restoring_single_call',
                   'workspace_D4_changes_not_imported')
    if (any(proof.get(key) is not True for key in true_fields)
            or proof.get('only_changed_existing_source') != relative
            or proof.get('new_helpers') != ['_regional_mean_roundoff_bound', '_project_regional_rates']
            or proof.get('old_source') != str(old_source/relative) or proof.get('new_source') != str(new_source/relative)):
        raise ValueError('Numerical AST proof does not bind the registered correction')
    base = read(root/'records/inference_v2_manifest.json')
    old_files = {str(Path(path).relative_to(old_source)): digest for path, digest in base['files'].items()}
    new_files = {}
    for filename, digest in manifest['files'].items():
        path = Path(filename)
        if not path.resolve().is_relative_to(new_source):
            raise ValueError('Numerical source escaped inference_v3')
        new_files[str(path.relative_to(new_source))] = digest; _pin(path, pins, digest)
    if set(old_files)-set(new_files) or set(new_files)-set(old_files) != {'tests/test_dynamic_dance_clock_roundoff.py'}:
        raise ValueError('Numerical correction added/deleted unapproved source paths')
    if [name for name in old_files if old_files[name] != new_files[name]] != [relative]:
        raise ValueError('Numerical correction changed unapproved prior code')
    _pin(old_source/relative, pins, proof['old_sha256']); _pin(new_source/relative, pins, proof['new_sha256'])
    restored = verify_regional_roundoff_ast((old_source/relative).read_text(), (new_source/relative).read_text())
    return {'source_root': str(new_source), 'manifest_sha256': NUMERICAL_CORRECTION_PINS['inference_v3_manifest.json'],
            'independent_AST_check': restored, 'telemetry_correction': telemetry,
            'scope': 'Only the pinned rejected-roundoff exception; successful projection and all physical limits unchanged.'}


def training_job_binding(root, job, pins, *, source_name='inference_v2'):
    if source_name not in ('inference_v2', 'inference_v3'):
        raise ValueError('Unapproved inference source')
    correction = (authenticate_inference_correction(root, pins) if source_name == 'inference_v2'
                  else authenticate_numerical_correction(root, pins))
    source = (root/'sources'/source_name).resolve()
    script, module = source/'scripts/motion/train_dynamic_dance.py', source/'lightsem/experiments/dynamic_dance/training.py'
    command = job.get('command', [])
    if (Path(job['cwd']).resolve() != source or len(command) < 3
            or command[:3] != [job['python'], str(script), 'infer']
            or job.get('imports', {}).get('lightsem.experiments.dynamic_dance.training') != str(module)):
        raise ValueError('Repeat must execute the exact approved '+source_name+' infer CLI and module')
    manifest = read(root/'records'/(source_name+'_manifest.json'))
    if job.get('source_files') != manifest['files']:
        raise ValueError('Inference job did not pin the complete approved '+source_name+' source tree')
    required = {**INFERENCE_CORRECTION_PINS, **(NUMERICAL_CORRECTION_PINS if source_name == 'inference_v3' else {})}
    for name, expected in required.items():
        if job.get('inputs', {}).get(str((root/'records'/name).resolve())) != expected:
            raise ValueError('Inference job did not freeze the approved correction: '+name)
    return {'source_root': str(source), 'script': str(script), 'training_module': str(module), 'explicit_correction': correction}


def verify_recovery_copies(receipt, standard_directory, copied_directory, guard_manifest, pins):
    """Bind standard files to guard-owned copies by filename, never hash multiset."""
    standard, copied = Path(standard_directory).resolve(), Path(copied_directory).resolve()
    if receipt.get('standard_output_directory') != str(standard) or receipt.get('copied_output_directory') != str(copied):
        raise ValueError('Recovery copy directory identity changed')
    original_files, copied_files = receipt.get('original_artifacts', {}), receipt.get('copied_artifacts', {})
    expected_original = {str(standard/name) for name in RECOVERY_FILES}
    expected_copied = {str(copied/name) for name in RECOVERY_FILES}
    if set(original_files) != expected_original or set(copied_files) != expected_copied:
        raise ValueError('Recovery must retain the exact nine original/copy files')
    if ({str(p.resolve()) for p in standard.iterdir() if p.is_file()} != expected_original
            or {str(p.resolve()) for p in copied.iterdir() if p.is_file()} != expected_copied):
        raise ValueError('Unexpected/missing recovery directory files')
    receipt_path = copied.parent/'RECOVERY.json'
    expected_guard = expected_copied | {str(receipt_path), str(receipt_path)+'.sha256.json'}
    if set(guard_manifest) != expected_guard:
        raise ValueError('Recovery guard manifest must bind exact copied files plus receipt/seal')
    for name in RECOVERY_FILES:
        source, copy = standard/name, copied/name
        digest = original_files[str(source)]
        if copied_files[str(copy)] != digest or guard_manifest[str(copy)]['sha256'] != digest:
            raise ValueError('Recovery original/copy relative filename hashes differ: '+name)
        _pin(source, pins, digest); _pin(copy, pins, digest)
    for path in (receipt_path, Path(str(receipt_path)+'.sha256.json')):
        _pin(path, pins, guard_manifest[str(path)]['sha256'])
    return {'files_verified': len(RECOVERY_FILES), 'standard_output_directory': str(standard),
            'copied_output_directory': str(copied), 'guard_body_path': str(copied/'body.npz')}


def authenticate_generation_plan(root, pins):
    root = Path(root).resolve()
    authenticate_numerical_correction(root, pins)
    path = root/'records/GENERATION_EXECUTION_PLAN_v3.json'
    _pin(path, pins, GENERATION_PLAN_SHA256)
    plan = authenticated_json(path, pins)
    if plan.get('schema_version') != 1 or plan.get('plan_sha256') != sha(root/'PLAN.json'):
        raise ValueError('Generation execution plan identity changed')
    entries = plan.get('entries', [])
    if len(entries) != 48 or len({(e['method'], e['song']) for e in entries}) != 48:
        raise ValueError('Generation execution plan must contain48 unique logical outputs')
    for filename, digest in plan.get('correction_files', {}).items():
        _pin(filename, pins, digest)
    recipe_path = root/'records/queues/infer-learned-v2.json'
    if plan.get('original_recipe_queue') != str(recipe_path):
        raise ValueError('Original48-generation recipe queue changed')
    _pin(recipe_path, pins, plan['original_recipe_queue_sha256'])
    recipe = read(recipe_path)
    recipe_jobs = {job['id']: job for job in recipe['jobs']}
    if len(recipe_jobs) != 48 or {e['original_recipe_job_id'] for e in entries} != set(recipe_jobs):
        raise ValueError('Execution plan does not preserve exact48 original generation recipes')
    for entry in entries:
        source_name = 'inference_v2' if (entry['method'], entry['song']) == ('D6', 'shpongle') else 'inference_v3'
        if entry['source'] != str(root/'sources'/source_name):
            raise ValueError('Execution plan substituted an unapproved per-case source')
        expected_output = root/'bodies'/entry['method']/entry['song']
        if entry['standard_output_directory'] != str(expected_output):
            raise ValueError('Generation execution plan changed consumer output identity')
        command = entry['canonical_command']
        original = list(recipe_jobs[entry['original_recipe_job_id']]['command'])
        original[1] = str(root/'sources'/source_name/'scripts/motion/train_dynamic_dance.py')
        if command != original:
            raise ValueError('Execution recovery changed original inference arguments')
        recovery = (entry['method'], entry['song']) == ('D6', 'cure_for_me')
        if recovery:
            if entry['job_id'] != 'recover-d6-cure-infer-v3' or entry['recovery_receipt'] != str(root/'recovery_artifacts/D6_cure_infer_v3/RECOVERY.json'):
                raise ValueError('Unapproved Cure recovery identity')
        elif entry.get('recovery_receipt') is not None:
            raise ValueError('Only the explicitly authorized Cure recovery may normalize a wrapper')
    return plan


def authenticate_repeat_plan(root, expected_sha256, pins, generation=None):
    if expected_sha256 != REPEAT_EXECUTION_PLAN_SHA256:
        raise ValueError('Unapproved repeat-execution-plan pin')
    generation = authenticate_generation_plan(root, pins) if generation is None else generation
    path = root/'records/REPEAT_EXECUTION_PLAN_v3.json'
    _pin(path, pins, expected_sha256)
    plan = authenticated_json(path, pins)
    if (plan.get('schema_version') != 1 or plan.get('plan_sha256') != sha(root/'PLAN.json')
            or plan.get('generation_execution_plan') != str(root/'records/GENERATION_EXECUTION_PLAN_v3.json')
            or plan.get('generation_execution_plan_sha256') != GENERATION_PLAN_SHA256
            or plan.get('repeatability_protocol_sha256') != sha(root/'records/REPEATABILITY_PROTOCOL_v1.json')):
        raise ValueError('Repeat execution plan ancestry/settings changed')
    entries = plan.get('entries', [])
    required = {(arm, song) for arm in LEARNED for song in SONGS}
    if len(entries) != 6 or {(entry['method'], entry['song']) for entry in entries} != required:
        raise ValueError('Exactly six original-source learned repeats required')
    originals = {(entry['method'], entry['song']): entry for entry in generation['entries']}
    same_fields = {'original_job_id': 'job_id', 'original_queue': 'queue', 'original_queue_sha256': 'queue_sha256',
                   'original_canonical_command': 'canonical_command', 'original_output_directory': 'standard_output_directory',
                   'original_recovery_receipt': 'recovery_receipt', 'source': 'source'}
    for entry in entries:
        original = originals[entry['method'], entry['song']]
        if any(entry.get(key) != original.get(value) for key, value in same_fields.items()):
            raise ValueError('Repeat plan changed its original execution/source binding')
        expected_output = root/'repeatability'/entry['method']/entry['song']
        if entry['repeat_output_directory'] != str(expected_output):
            raise ValueError('Repeat output path differs from fixed cohort')
        command = list(entry['original_canonical_command'])
        if command.count('--output') != 1:
            raise ValueError('Expected one committed original output argument')
        command[command.index('--output')+1] = str(expected_output/'body.npz')
        if command != entry['repeat_canonical_command']:
            raise ValueError('Repeat command changes more than output destination')
        queue_name = 'repeat-learned-v2-shpongle.json' if (entry['method'], entry['song']) == ('D6', 'shpongle') else 'repeat-learned-v3.json'
        if entry['repeat_queue'] != str(root/'records/queues'/queue_name):
            raise ValueError('Unapproved per-source repeat queue')
        _pin(entry['repeat_queue'], pins, entry['repeat_queue_sha256'])
    return plan


def execution_job(root, entry, pins):
    qpath = Path(entry['queue'])
    _pin(qpath, pins, entry['queue_sha256'])
    queue = read(qpath)
    jobs = [job for job in queue['jobs'] if job['id'] == entry['job_id']]
    if len(jobs) != 1:
        raise ValueError('Execution plan job identity is missing/ambiguous')
    job = jobs[0]
    if job.get('outputs') != entry['guard_outputs'] or Path(job['cwd']).resolve() != Path(entry['source']).resolve():
        raise ValueError('Execution guard outputs or working source differs from plan')
    return queue, qpath, job


def authenticate_recovery_execution(root, entry, queue, qpath, job, pins):
    """The sole approved wrapper-to-canonical normalization, after proof/copy checks."""
    if (entry['method'], entry['song'], entry['job_id']) != ('D6', 'cure_for_me', 'recover-d6-cure-infer-v3'):
        raise ValueError('Unapproved recovery normalization')
    wrapper = root/'records/recover_d6_cure_v3.py'
    authorization_path = root/'records/D6_CURE_RECOVERY_AUTH_v3.json'
    _pin(wrapper, pins, RECOVERY_WRAPPER_SHA256); _pin(authorization_path, pins, RECOVERY_AUTH_SHA256)
    authorization = authenticated_json(authorization_path, pins)
    standard = root/'bodies/D6/cure_for_me'
    recovery = root/'recovery_artifacts/D6_cure_infer_v3'
    copied = recovery/'completed_body'
    if (authorization.get('failed_job') != 'infer-d6-cure_for_me-v2'
            or authorization.get('prior_body_or_plan_output') is not False
            or authorization.get('standard_output') != str(standard/'body.npz')
            or authorization.get('recovery_directory') != str(recovery)
            or entry['guard_outputs'] != [str(recovery)]):
        raise ValueError('Recovery authorization changed source/output/history')
    for filename, digest in authorization['protected_files'].items():
        _pin(filename, pins, digest)
    command = job['command']
    if len(command) < 4 or command[1] != str(wrapper):
        raise ValueError('Cure job does not execute the approved wrapper')
    canonical = [command[0], *command[2:]]
    if canonical != entry['canonical_command'] or canonical != authorization['canonical_command']:
        raise ValueError('Cure wrapper arguments differ from pinned canonical infer command')
    expected_sources = {**read(root/'records/inference_v3_manifest.json')['files'], str(wrapper): RECOVERY_WRAPPER_SHA256}
    if job['source_files'] != expected_sources:
        raise ValueError('Recovery source pins differ from v3 plus exact approved wrapper')
    if job.get('inputs', {}).get(str(authorization_path)) != RECOVERY_AUTH_SHA256:
        raise ValueError('Recovery job did not pin its authorization at execution')
    guard = _guard_job(root, queue, qpath, job, copied/'body.npz', pins)
    receipt_path = recovery/'RECOVERY.json'
    if entry.get('recovery_receipt') != str(receipt_path):
        raise ValueError('Recovery receipt path differs from committed execution')
    receipt = authenticated_json(receipt_path, pins)
    if receipt.get('authorization') != str(authorization_path) or receipt.get('authorization_sha256') != RECOVERY_AUTH_SHA256:
        raise ValueError('Recovery receipt does not bind approved authorization')
    guard_manifest = read(root/'records/dynamic_jobs'/job['id']/'manifest.json')
    copies = verify_recovery_copies(receipt, standard, copied, guard_manifest, pins)
    primary_path = root/'records/D6_CURE_RECOVERY_PRIMARY_VERIFICATION_v3.json'
    _pin(primary_path, pins, RECOVERY_PRIMARY_SHA256)
    primary = read(primary_path)
    if (primary.get('guard_receipt_sha256') != sha(root/'records/dynamic_jobs'/job['id']/'completed.json')
            or primary.get('recovery_receipt_sha256') != sha(receipt_path)
            or primary.get('all_original_and_copy_hashes_match') is not True
            or primary.get('original_Shpongle_and_failed_records_unchanged') is not True):
        raise ValueError('Independent recovery verification does not bind actual guard/copy evidence')
    normalized = {**job, 'command': canonical,
                  'source_files': {path: digest for path, digest in job['source_files'].items() if path != str(wrapper)}}
    training_job_binding(root, normalized, pins, source_name='inference_v3')
    return normalized, {'guard': guard, 'copies': copies, 'primary_verification': str(primary_path),
                         'normalization': 'Removed only pinned wrapper argv element and its source pin; inputs/environment unchanged.'}


def _repeat_provenance(root, arm, song, original_path, repeated_path, a, b, pins, repeat_plan):
    ra = authenticated_json(original_path.parent/'inference_receipt.json', pins)
    rb = authenticated_json(repeated_path.parent/'inference_receipt.json', pins)
    _pin(original_path, pins, ra['output_sha256']); _pin(repeated_path, pins, rb['output_sha256'])
    check = infer_receipt_match(ra, rb, arm)
    if check['status'] != 'matched':
        raise ValueError('Invalid same-input receipt: '+json.dumps(check, sort_keys=True))
    for receipt, body in ((ra, a), (rb, b)):
        for key, value in receipt.items():
            if key != 'output_sha256' and (key not in body.metadata or body.metadata[key] != value):
                raise ValueError('Inference receipt/body metadata disagree: '+key)
        for path_field, hash_field in (('checkpoint', 'checkpoint_sha256'), ('timeline_path', 'timeline_sha256'),
                                      ('audio_input_path', 'audio_input_sha256'), ('source_body', 'source_body_sha256')):
            if path_field in receipt:
                _pin(receipt[path_field], pins, receipt[hash_field])
        _pin(Path(receipt['timeline_path']).with_suffix('.json'), pins, receipt['timeline_metadata_sha256'])
    entries = [entry for entry in repeat_plan['entries'] if (entry['method'], entry['song']) == (arm, song)]
    if len(entries) != 1:
        raise ValueError('Missing/ambiguous committed repeat identity')
    entry = entries[0]
    if original_path != Path(entry['original_output_directory'])/'body.npz' or repeated_path != Path(entry['repeat_output_directory'])/'body.npz':
        raise ValueError('Repeat body paths differ from committed execution')
    generation = read(root/'records/GENERATION_EXECUTION_PLAN_v3.json')
    original_entry = next(row for row in generation['entries'] if (row['method'], row['song']) == (arm, song))
    oq, oqpath, original_job = execution_job(root, original_entry, pins)
    repeat_entry = {'queue': entry['repeat_queue'], 'queue_sha256': entry['repeat_queue_sha256'],
                   'job_id': entry['repeat_job_id'], 'guard_outputs': [entry['repeat_output_directory']],
                   'source': entry['source']}
    rq, rqpath, repeat_job = execution_job(root, repeat_entry, pins)
    source_name = Path(entry['source']).name
    recovery = None
    if original_entry.get('recovery_receipt') is not None:
        canonical_original, recovery = authenticate_recovery_execution(root, original_entry, oq, oqpath, original_job, pins)
        original_guard = recovery['guard']
    else:
        if original_job['command'] != entry['original_canonical_command']:
            raise ValueError('Original guard command differs from execution plan')
        training_job_binding(root, original_job, pins, source_name=source_name)
        original_guard = _guard_job(root, oq, oqpath, original_job, original_path, pins)
        canonical_original = original_job
    if repeat_job['command'] != entry['repeat_canonical_command']:
        raise ValueError('Repeat guard command differs from execution plan')
    training_job_binding(root, repeat_job, pins, source_name=source_name)
    cli = command_match(canonical_original, repeat_job, original_path, repeated_path)
    if cli['status'] != 'matched':
        raise ValueError('Repeat changed canonical inference invocation: '+json.dumps(cli))
    repeat_guard = _guard_job(root, rq, rqpath, repeat_job, repeated_path, pins)
    jobs = [original_guard, repeat_guard]
    # The exact CLI also binds the Base92 path, which infer receipts store only by hash.
    argv = canonical_original['command']
    if arm in ('D7', 'D8'):
        if '--base92' not in argv:
            raise ValueError('Flow repetition lacks explicit Base92 input')
        _pin(argv[argv.index('--base92')+1], pins, ra['base92_sha256'])
    if arm == 'D8':
        _pin(root/'learning/runs/planner/seed_0/final.pt', pins, ra['planner_sha256'])
    return {'receipt': check, 'command': cli, 'jobs': jobs, 'execution': entry, 'verified_recovery': recovery}


def _numeric(path, arrays, plan):
    required = sum(np.asarray(value).nbytes for value in arrays.values())
    existing = path.parent
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < plan.get('minimum_free_bytes', 20*2**30)+required:
        raise RuntimeError('Numeric repeat evidence would violate disk reserve')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(str(path))
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return {'path': str(path), 'sha256': sha(path), 'arrays': {name: list(np.asarray(value).shape) for name, value in arrays.items()}}


def _compile(body, track, plan, modules):
    time, world = modules['mapping'].lighting_body(body.world)
    left, right = modules['mapping'].p2_colors(track)
    dmx, patch, intent, hard = modules['mapping'].compile_body(world, left, right, plan)
    return {'time': time, 'body': world, 'universes': dmx, 'targets': intent['targets']}, patch, hard


def _saved_mapping(root, artifact_id, song, body_path, pins):
    folder = root/'timelines'/artifact_id/song
    report = read(folder/'report.json')
    _pin(folder/'report.json', pins)
    _pin(body_path, pins, report['body_sha256'])
    _pin(folder/'timeline.npz', pins, report['timeline_sha256'])
    _pin(folder/'patch.json', pins, report['patch_sha256'])
    with np.load(folder/'timeline.npz', allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in ('time', 'body', 'targets', 'universes')}
    return arrays, read(folder/'patch.json')


def run_audit(root, output, modules, pins, source, manifest, protocol, repeat_plan):
    """Worker body: supplied modules have already passed frozen-source admission."""
    plan = read(root/'PLAN.json'); _pin(root/'PLAN.json', pins)
    for path, expected in plan['files'].items():
        _pin(path, pins, expected)
    pins.update(modules['campaign'].variant_binding_pins(root))
    tracks = {track['id']: track for track in plan['tracks']}
    if set(tracks) != set(SONGS):
        raise ValueError('Expected both committed showcase songs')
    results, originals = [], {}
    for method in METHODS:
        artifact_id = modules['campaign'].resolve_variant(root, method)
        for song in SONGS:
            item = {'kind': 'same_body_mapper_repeat', 'method': method, 'artifact_id': artifact_id, 'song': song}
            folder = output/'mapper'/method/song
            try:
                path = root/'bodies'/artifact_id/song/'body.npz'
                body = _body(path, modules, pins)
                saved, patch = _saved_mapping(root, artifact_id, song, path, pins)
                for p in (tracks[song]['baseline_timeline'], tracks[song]['baseline_patch']):
                    _pin(p, pins)
                rebuilt, rebuilt_patch, hard = _compile(body, tracks[song], plan, modules)
                if patch != rebuilt_patch:
                    raise ValueError('Fixed recompile changed fixture patch')
                imported = verify_imported_sources(source, manifest, pins)
                metrics, deltas = mapping_comparison(saved, rebuilt, patch, modules['decoder'].decode)
                arrays = {**{'recompiled_'+key: value for key, value in rebuilt.items()}, **deltas}
                evidence = _numeric(folder/'numeric.npz', arrays, plan)
                item.update(status='measured', native_clock=nominal_clock_metrics(body.time), measurements=metrics, numeric_evidence=evidence, hard_checks=hard,
                            actual_imports=imported)
                if method in LEARNED:
                    originals[(method, song)] = (body, saved, rebuilt, patch, path)
            except Exception as error:
                item.update(status='unresolved', failure_type=type(error).__name__, failure=str(error))
            write_new(folder/'report.json', item); results.append(item)
            print(json.dumps({'kind': item['kind'], 'method': method, 'song': song, 'status': item['status']}), flush=True)
    repeated = []
    for arm in LEARNED:
        for song in SONGS:
            item = {'kind': 'same_input_learned_repeat', 'method': arm, 'song': song}
            folder = output/'learned'/arm/song
            try:
                if (arm, song) not in originals:
                    raise ValueError('Original mapping unresolved; no substituted reference')
                a, saved, rebuilt_original, patch, original_path = originals[(arm, song)]
                repeated_path = root/'repeatability'/arm/song/'body.npz'
                b = _body(repeated_path, modules, pins)
                if not np.array_equal(a.time, b.time):
                    raise ValueError('Original and learned repeat timestamps differ')
                provenance = _repeat_provenance(root, arm, song, original_path, repeated_path, a, b, pins, repeat_plan)
                if (arm == 'D6' and (a.native is not None or b.native is not None)) or (arm != 'D6' and (a.native is None or b.native is None)):
                    raise ValueError('Native representation differs from declared arm contract')
                body_metrics = {name: windowed_difference(getattr(a, name), getattr(b, name), a.time)
                                for name in ('world', 'native', 'contacts', 'source_time')}
                region_full = regional_body_difference(a.world, b.world, modules['diagnostics'].REGIONS)
                mask = (a.time >= 60.) & (a.time < 180.)
                region_review = regional_body_difference(a.world[mask], b.world[mask], modules['diagnostics'].REGIONS)
                d3_regions = modules['evaluation'].SOURCE_JOINTS
                region_d3 = regional_body_difference(a.world[mask], b.world[mask], d3_regions)
                plans = [modules['contracts'].DancePlan.load(path.parent/'plan.npz') for path in (original_path, repeated_path)]
                if not (np.array_equal(plans[0].time, plans[1].time) and np.array_equal(plans[0].time, a.time)):
                    raise ValueError('Original/repeated dance plans do not share the body clock')
                for path in (original_path.parent/'plan.npz', repeated_path.parent/'plan.npz'):
                    for p in (path, path.with_suffix('.json'), path.with_suffix('.json.sha256.json')):
                        _pin(p, pins)
                plan_metrics = {name: windowed_difference(getattr(plans[0], name), getattr(plans[1], name), plans[0].time)
                                for name in ('time', 'source_time', 'effort')}
                rebuilt_repeat, repeated_patch, hard = _compile(b, tracks[song], plan, modules)
                if repeated_patch != patch:
                    raise ValueError('Learned repeat changed fixed fixture patch')
                verify_imported_sources(source, manifest, pins)
                same_mapper, deltas = mapping_comparison(rebuilt_original, rebuilt_repeat, patch, modules['decoder'].decode)
                versus_saved, _ = mapping_comparison(saved, rebuilt_repeat, patch, modules['decoder'].decode)
                arrays = {**{'repeat_'+name: value for name, value in rebuilt_repeat.items()}, **deltas,
                          'native_time': a.time, 'world_delta_m': b.world.astype(float)-a.world.astype(float),
                          'root_relative_world_delta_m': (b.world.astype(float)-b.world[:, :1].astype(float))-(a.world.astype(float)-a.world[:, :1].astype(float))}
                if a.native is not None:
                    arrays['native_delta'] = b.native.astype(float)-a.native.astype(float)
                evidence = _numeric(folder/'numeric.npz', arrays, plan)
                item.update(status='measured', native_clock={'original': nominal_clock_metrics(a.time), 'repeat': nominal_clock_metrics(b.time), 'exact_repeat_clock': True}, provenance=provenance, body=body_metrics, plans=plan_metrics,
                    regional_repeat_floor={'full_prefix': region_full, 'review_60_180': region_review, 'D3_regions_review': region_d3},
                    through_identical_recompiled_mapper=same_mapper, versus_original_saved_mapping=versus_saved,
                    hard_checks=hard, numeric_evidence=evidence)
            except Exception as error:
                item.update(status='unresolved', failure_type=type(error).__name__, failure=str(error))
            write_new(folder/'report.json', item); repeated.append(item)
            print(json.dumps({'kind': item['kind'], 'method': arm, 'song': song, 'status': item['status']}), flush=True)
    # Recheck every frozen input after computation; partial evidence is preserved on drift.
    for path, expected in list(pins.items()):
        _pin(path, pins, expected)
    report = {'schema_version': 1, 'auditor_version': AUDITOR_VERSION,
        'inference_source_correction': authenticate_numerical_correction(root, pins),
        'repeat_execution_plan': repeat_plan, 'status': 'measured' if all(row['status']=='measured' for row in results+repeated) else 'incomplete_or_unresolved',
        'expected_mapper_cases': 16, 'expected_learned_cases': 6, 'mapper_repeats': results, 'learned_repeats': repeated,
        'inputs': pins, 'protocol': protocol,
        'scope': 'One committed repeat; observed pair variation is not a statistical upper bound, new scientific arm, or semantic qualification.',
        'original_outputs_remain_displayed': True, 'scientific_verdict': 'not_assigned', 'thresholds_added_or_relaxed': False}
    write_new(output/'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verification-manifest-sha256', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--repeat-execution-plan-sha256', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output_dir.resolve()
    if output == root or any(output.is_relative_to(root/name) for name in ('bodies', 'timelines', 'sources', 'repeatability')):
        parser.error('Output must be a new directory outside original inputs and sources')
    if not args.worker:
        if output.exists():
            raise FileExistsError('Preserve existing audit directory: '+str(output))
        pins = {}
        validate_protocol(root, args.protocol_sha256, pins)
        authenticate_repeat_plan(root, args.repeat_execution_plan_sha256, pins)
        source, _ = verify_frozen_source(root, args.verification_manifest_sha256, pins)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--root', str(root), '--output-dir', str(output),
                   '--verification-manifest-sha256', args.verification_manifest_sha256, '--protocol-sha256', args.protocol_sha256,
                   '--repeat-execution-plan-sha256', args.repeat_execution_plan_sha256]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1')
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        subprocess.run(command, cwd=source, env=env, check=True)
        return
    if output.exists():
        raise FileExistsError('Preserve existing worker output: '+str(output))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    output.mkdir(parents=True, exist_ok=False)
    pins = {}
    try:
        _pin(Path(__file__), pins)
        protocol = validate_protocol(root, args.protocol_sha256, pins)
        correction = authenticate_numerical_correction(root, pins)
        repeat_plan = authenticate_repeat_plan(root, args.repeat_execution_plan_sha256, pins)
        source, manifest = verify_frozen_source(root, args.verification_manifest_sha256, pins)
        modules = load_frozen_modules(source, manifest, pins)
        write_new(output/'ADMISSION.json', {'inputs': pins, 'auditor_version': AUDITOR_VERSION, 'inference_source_correction': correction, 'verification_source': str(source), 'CPU_only': True,
            'auditor': str(Path(__file__).resolve()), 'python': sys.executable,
            'numpy_version': np.__version__, 'python_version': sys.version,
            'dependency_versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'clarabel')},
            'command': sys.argv})
        report = run_audit(root, output, modules, pins, source, manifest, protocol, repeat_plan)
        if report['status'] != 'measured':
            raise RuntimeError('Repeatability audit preserved unresolved cases; see report.json')
    except Exception as error:
        write_new(output/'failure.json', {'status': 'failed', 'type': type(error).__name__, 'message': str(error), 'inputs': pins})
        raise


if __name__ == '__main__':
    main()

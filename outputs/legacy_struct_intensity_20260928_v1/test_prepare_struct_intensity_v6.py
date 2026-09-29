#!/usr/bin/env python3
"""Synthetic readiness tests. Written but NOT executed during source preparation.

Future execution reads only this preparer source. All arrays are generated here;
no torch import, checkpoint load, real plan, cache, manifest, model, or main call
is needed. The tests do not establish scientific or preparation authorization.
"""
from __future__ import annotations

import ast
import builtins
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
# Initialize the assertion module before setUp installs all filesystem guards.
# numpy.testing may read installed-package direct_url.json on first import.
import numpy.testing as npt


DRIVER_PATH = Path(__file__).with_name('prepare_struct_intensity_v6.py')
EXPECTED_TEST_COUNT = 112  # Source-only count; not a native test result.
REAL_IMPORT = builtins.__import__


def synthetic_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'lightsem'}:
        raise AssertionError('forbidden numerical/model import: ' + name)
    return REAL_IMPORT(name, *args, **kwargs)


class ShapeOnlyTensor:
    """Metadata only; never contains weights or supports numerical conversion."""
    def __init__(self, shape, dtype='torch.float32', device='cpu'):
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def __array__(self, *args, **kwargs):
        raise AssertionError('checkpoint state values must not be inspected')


class PreparationV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is FUTURE test execution, not an action taken when writing this file.
        cls.source = DRIVER_PATH.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.source, filename=str(DRIVER_PATH))
        spec = importlib.util.spec_from_file_location('synthetic_prepare_v6', DRIVER_PATH)
        cls.driver = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, 'dont_write_bytecode', True), \
             mock.patch('builtins.__import__', side_effect=synthetic_import):
            spec.loader.exec_module(cls.driver)

    def setUp(self):
        self.d = self.driver
        self.plan = self.make_plan()
        # Pure-helper tests must fail if they reach any payload or operational API.
        for name in ('checked_path', 'read_ref', 'local_ref', 'load_checkpoint_cpu',
                     'read_npz_ref', 'inventory', 'guard', 'bounded_receipt_write',
                     'prepare_authenticated', 'CappedLog', 'durable_progress', 'main'):
            patcher = mock.patch.object(self.d, name,
                side_effect=AssertionError('forbidden operational helper: ' + name))
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('load', 'fromfile', 'memmap'):
            patcher = mock.patch.object(np, name,
                side_effect=AssertionError('forbidden numerical file read: ' + name))
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ('open', 'read_bytes', 'read_text'):
            patcher = mock.patch.object(Path, name,
                side_effect=AssertionError('forbidden filesystem read: Path.' + name))
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch('builtins.open', side_effect=AssertionError('forbidden file open'))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch('builtins.__import__', side_effect=synthetic_import)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_plan(self):
        d = self.d
        mids = ['synthetic_recording_%02d' % i for i in range(31)]
        rows = []
        for i in range(40):
            rows.append({
                'clip': 'synthetic_clip_%02d' % i,
                'recording_id': mids[i % 31],
                'T_metadata': 9000 if i < 39 else 371393 - 39 * 9000,
                'start_s_metadata': 0.0,
                'original_shuffled_donor_recording_id': mids[(i + 1) % 31],
                'canonical_id': 'synthetic_canonical_%02d' % (i % 31),
                'original_shuffled_donor_canonical_id': 'synthetic_canonical_%02d' % ((i + 1) % 31),
                'current_partition': 'train', 'current_Stage11_TEST_aliases': [],
            })
        slots = []
        for mid in mids:
            for kind, root in (('audio', d.AUDIO_ROOT), ('structure', d.ROLE_ROOT)):
                slots.append({'path': str(root / (mid + '.npz')), 'kind': kind, 'recording_id': mid})
        cfg = {'d_model': 8, 'n_blocks': 2, 'conv_kernel': 3,
               'n_a': 4, 'n_b': 3, 'n_c': 17,
               'expected_dim_none': 92, 'expected_dim_struct_shuf': 100}
        return {
            'document_revision': 6,
            'cohorts': {'clip_index': rows},
            'qualified_gate_scope': {'original_G_struct_contrast_completion_claim_permitted': False},
            'preparation_v6': {
                'scope': 'PREPARATION_ONLY_NO_MODEL_OR_SCORE',
                'packet_root': str(d.PACKET), 'packet_cap_bytes': d.PACKET_CAP,
                'global_cap_bytes': d.GLOBAL_CAP, 'comparison_reserve_bytes': d.COMPARISON_RESERVE,
                'original_root': str(d.ORIGINAL), 'workspace_root': str(d.WORKSPACE),
                'minimum_synthetic_tests': 24, 'expected_synthetic_tests': EXPECTED_TEST_COUNT,
                'native_synthetic_test_command': 'synthetic command; never executed',
                'input_slots': slots,
                'identity_projection_ref': self.fake_ref('identity.json'),
                'source_refs': [self.fake_ref('dependency.py')],
                'checkpoint_refs': {arm: self.fake_ref(arm + '.pt') for arm in d.ARMS},
            },
            'generation_and_mapper': {'model_configuration': cfg},
            'lineage': {'calibration_reports': {
                arm: {'event_thresholds': {'thresholds': [0.2, 0.4, 0.6]},
                      'smoothing': {'smooth_tau': 0.25}}
                for arm in d.ARMS}},
        }

    @staticmethod
    def fake_ref(name):
        return {'path': '/synthetic-not-a-real-input/' + name,
                'size_bytes': 1, 'sha256': 'a' * 64}

    def make_checkpoint(self, arm='none'):
        cfg = dict(self.plan['generation_and_mapper']['model_configuration'])
        cfg.pop('expected_dim_none')
        cfg.pop('expected_dim_struct_shuf')
        cfg.update(in_dim=92 if arm == 'none' else 100, ff_mult=4, dropout=0.1)
        calibration = self.plan['lineage']['calibration_reports'][arm]
        return {
            'cfg': cfg,
            'model': {key: ShapeOnlyTensor(shape)
                      for key, shape in self.d.expected_state_shapes(cfg).items()},
            'mu': np.zeros(cfg['in_dim'], dtype=np.float32),
            'sd': np.ones(cfg['in_dim'], dtype=np.float32),
            'block_names': {'a': list(self.d.A_NAMES), 'c': list(self.d.C_NAMES)},
            'event_thr': copy.deepcopy(calibration['event_thresholds']['thresholds']),
            'smooth_tau': calibration['smoothing']['smooth_tau'],
        }

    def make_bus(self, n=200):
        bus = {'times': np.arange(n, dtype=np.float64) / 40.0,
               'mel': np.zeros((64, n), dtype=np.float32),
               'beats': np.array([0.0, 0.5], dtype=np.float64),
               'downbeats': np.array([0.0], dtype=np.float64),
               'meta': json.dumps({'fps': 40.0, 'duration_s': n / 40.0,
                                   'tempo': 120.0, 'flags': {'oracle_beat': False}})}
        bus.update({'ch__' + name: np.zeros(n, dtype=np.float32)
                    for name in self.d.AUDIO_SCALARS})
        return bus

    def make_structure(self, n=200, segments=None):
        if segments is None:
            split = (n // 2) / 40.0
            segments = [(0.0, split, 'breakdown'), (split, n / 40.0, 'drop')]
        t = np.arange(n, dtype=np.float64) / 40.0
        roles = np.zeros(n, dtype=np.int8)
        table = []
        for a, b, label in segments:
            roles[(t >= a) & (t < b)] = self.d.ROLES.index(label)
            table.append({'start_s': a, 'end_s': b, 'role': label})
        return {'t': t, 'role': roles, 'energy': np.zeros(n, dtype=np.float32),
                'to_rise_s': np.zeros(n, dtype=np.float32),
                'to_drop_s': np.zeros(n, dtype=np.float32),
                'meta': json.dumps({'roles': list(self.d.ROLES),
                                    'duration_s': n / 40.0, 'segments': table})}

    def qualify(self, n=200, start=0.0, bus=None, target=None, donor=None):
        bus = self.make_bus(n) if bus is None else bus
        target = self.make_structure(n) if target is None else target
        donor = self.make_structure(n) if donor is None else donor
        row = {'clip': 'synthetic_clip', 'recording_id': 'synthetic_recipient',
               'T_metadata': n, 'start_s_metadata': start,
               'original_shuffled_donor_recording_id': 'synthetic_donor'}
        return self.d.qualify_clip(row, bus,
            self.d.validate_structure(target, np), self.d.validate_structure(donor, np), np)

    def make_authorization(self):
        plan_ref = self.fake_ref('plan.json')
        driver_ref = self.fake_ref('driver.py')
        test_refs = [self.fake_ref('test.py')]
        argv = ['/synthetic/driver.py', 'prepare', '--plan', '/synthetic/plan.json',
                '--authorization', '/synthetic/auth.json']
        prep = self.plan['preparation_v6']
        inputs = []
        for slot in prep['input_slots']:
            entry = dict(slot)
            entry.update(size_bytes=1, sha256='b' * 64, current_Stage11_partition='train',
                         known_current_TEST_aliases=[], identity_projection_ref=prep['identity_projection_ref'],
                         cross_corpus_identity='UNRESOLVED_NOT_INDEPENDENT',
                         approval='ROOT_APPROVED_HISTORICAL_DESCRIPTIVE_ONLY')
            inputs.append(entry)
        auth = {
            'scope': 'PREPARATION_ONLY_NO_MODEL_OR_SCORE', 'authorized_by': 'root',
            'plan_ref': plan_ref, 'driver_ref': driver_ref, 'test_refs': test_refs, 'argv': argv,
            'allow_cpu_checkpoint_deserialization': True,
            'accept_historical_descriptive_unknown_cross_corpus_aliases': True,
            'native_registration_receipt_verified': True,
            'native_registration_receipt_ref': self.fake_ref('registration.json'),
            'other_reserved_bytes': 0, 'remaining_project_GPU_seconds': 0,
            'original_inventory_sha256': 'c' * 64,
            'preparation_log': {'path': str(self.d.SOURCE_DIR / 'preparation_v6.log'),
                                'max_bytes': 65536},
            'preparation_external_limits_bound': {
                'wall_seconds': 900, 'host_RAM_bytes': 4294967296, 'GPU_access': False},
            'source_refs': prep['source_refs'], 'checkpoint_refs': prep['checkpoint_refs'],
            'input_refs': inputs,
            'native_synthetic_test_result': {
                'exit_code': 0, 'failures': 0, 'errors': 0,
                'skipped': 0, 'expected_failures': 0, 'unexpected_successes': 0,
                'tests_run': EXPECTED_TEST_COUNT, 'tests_expected': EXPECTED_TEST_COUNT,
                'plan_ref': plan_ref, 'driver_ref': driver_ref, 'test_refs': test_refs,
                'real_model_forwards': 0, 'sealed_TEST_reads': 0, 'checkpoint_deserializations': 0,
                'command': 'synthetic command; never executed', 'log_ref': self.fake_ref('log.txt'),
            },
        }
        for key in ('allow_model_construction', 'allow_model_forward', 'allow_scoring',
                    'allow_GPU', 'allow_TEST_access', 'allow_raw_manifest_access'):
            auth[key] = False
        return auth, (self.plan, plan_ref, driver_ref, test_refs, argv)

    def resources(self, **changes):
        values = dict(original_bytes=0, workspace_bytes=0, comparison_bytes=0,
                      external_packet_bytes=0, packet_workspace_bytes=0, other_reserved_bytes=0,
                      free_mnt_bytes=self.d.FREE_FLOOR + self.d.PACKET_CAP,
                      free_root_bytes=self.d.FREE_FLOOR + self.d.PACKET_CAP)
        values.update(changes)
        return self.d.resource_values(**values)

    def function(self, name):
        return next(node for node in self.tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    @staticmethod
    def call_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = PreparationV6Tests.call_name(node.value)
            return prefix + '.' + node.attr
        return ''

    def test_static_cli_has_prepare_only(self):
        calls = [n for n in ast.walk(self.function('main')) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                 and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'stage']
        self.assertEqual(len(calls), 1)
        choices = next(k.value for k in calls[0].keywords if k.arg == 'choices')
        self.assertEqual(ast.literal_eval(choices), ['prepare'])

    def test_static_no_model_eval_smoke_dispatch(self):
        forbidden = {'forward', 'predict', 'inference', 'infer', 'evaluate', 'eval', 'smoke',
                     'train', 'fit', 'sample', 'generate', 'score', 'LightTrunk', 'Conformer'}
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.assertNotIn(node.name, forbidden)
            if isinstance(node, ast.Call):
                self.assertNotIn(self.call_name(node.func).split('.')[-1], forbidden)
            if isinstance(node, ast.Import):
                self.assertTrue(all(a.name.split('.')[0] != 'lightsem' for a in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual((node.module or '').split('.')[0], 'lightsem')

    def test_static_no_source_execution_or_subprocess(self):
        forbidden = {'exec', 'eval', 'compile', '__import__', 'os.system', 'os.popen',
                     'subprocess.run', 'subprocess.Popen', 'runpy.run_path', 'runpy.run_module'}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                self.assertNotIn(self.call_name(node.func), forbidden)

    def test_static_torch_load_is_single_restricted_cpu_call(self):
        calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                 and self.call_name(n.func) == 'torch.load']
        self.assertEqual(len(calls), 1)
        keywords = {k.arg: k.value for k in calls[0].keywords}
        self.assertIs(ast.literal_eval(keywords['weights_only']), True)
        self.assertEqual(ast.literal_eval(keywords['map_location']), 'cpu')
        self.assertNotIn(None, keywords)

    def test_static_torch_import_is_local_to_checkpoint_loader(self):
        imports = [n for n in ast.walk(self.tree) if isinstance(n, ast.Import)
                   and any(a.name == 'torch' for a in n.names)]
        self.assertEqual(len(imports), 1)
        self.assertIn(imports[0], list(ast.walk(self.function('load_checkpoint_cpu'))))
        top_imports = [n for n in self.tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertTrue(all('torch' not in ast.unparse(n) for n in top_imports))

    def test_static_checkpoint_loader_has_no_fallback(self):
        loader = self.function('load_checkpoint_cpu')
        self.assertFalse(any(isinstance(n, ast.Try) for n in ast.walk(loader)))
        contexts = [n for n in ast.walk(loader) if isinstance(n, ast.With)]
        self.assertTrue(any(self.call_name(item.context_expr.func) == 'torch.serialization.safe_globals'
                            for node in contexts for item in node.items
                            if isinstance(item.context_expr, ast.Call)))

    def test_static_authorization_precedes_payload_access(self):
        main = self.function('main')
        calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
        auth_line = next(n.lineno for n in calls
                         if self.call_name(n.func) == 'validate_authorization')
        for name in ('prepare_authenticated', 'read_ref', 'guard', 'CappedLog'):
            relevant = [n for n in calls if self.call_name(n.func) == name]
            self.assertTrue(relevant)
            self.assertTrue(all(n.lineno > auth_line for n in relevant))
        dispatches = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                      and self.call_name(n.func) == 'prepare_authenticated']
        self.assertEqual(len(dispatches), 1)
        self.assertIn(dispatches[0], calls)
        enclosing = [node for node in ast.walk(main) if isinstance(node, ast.With)
                     and dispatches[0] in list(ast.walk(node))]
        contexts = [item for node in enclosing for item in node.items
                    if isinstance(item.context_expr, ast.Call)]
        capped = [item for item in contexts
                  if self.call_name(item.context_expr.func) == 'CappedLog']
        self.assertEqual(len(capped), 1)
        self.assertIsInstance(capped[0].optional_vars, ast.Name)
        sink_name = capped[0].optional_vars.id
        for redirect in ('contextlib.redirect_stdout', 'contextlib.redirect_stderr'):
            matches = [item.context_expr for item in contexts
                       if self.call_name(item.context_expr.func) == redirect]
            self.assertEqual(len(matches), 1)
            self.assertEqual(len(matches[0].args), 1)
            self.assertIsInstance(matches[0].args[0], ast.Name)
            self.assertEqual(matches[0].args[0].id, sink_name)
        payload = self.function('prepare_authenticated')
        payload_calls = [self.call_name(n.func) for n in ast.walk(payload) if isinstance(n, ast.Call)]
        self.assertIn('load_checkpoint_cpu', payload_calls)
        self.assertIn('read_npz_ref', payload_calls)
        publication = [n for n in ast.walk(payload) if isinstance(n, ast.Call)
                       and self.call_name(n.func) == 'bounded_receipt_write']
        self.assertEqual(len(publication), 1)
        status_prints = [n for n in ast.walk(payload) if isinstance(n, ast.Call)
                         and self.call_name(n.func) == 'print']
        self.assertTrue(status_prints)
        self.assertTrue(all(n.lineno < publication[0].lineno for n in status_prints))
        self.assertTrue(any(isinstance(n, ast.Import) and any(a.name == 'numpy' for a in n.names)
                            for n in ast.walk(payload)))
        self.assertFalse(any(isinstance(n, ast.Import) and any(a.name == 'numpy' for a in n.names)
                             for n in ast.walk(main)))

    def test_static_npz_loading_disables_pickle(self):
        calls = [n for n in ast.walk(self.function('read_npz_ref')) if isinstance(n, ast.Call)
                 and self.call_name(n.func) == 'np.load']
        self.assertEqual(len(calls), 1)
        keywords = {k.arg: k.value for k in calls[0].keywords}
        self.assertIs(ast.literal_eval(keywords['allow_pickle']), False)

    def test_json_rejects_duplicate_keys(self):
        with self.assertRaises(self.d.PreparationError):
            self.d.parse_json('{"a": 1, "a": 2}')

    def test_json_rejects_nonfinite_constants(self):
        for value in ('NaN', 'Infinity', '-Infinity'):
            with self.subTest(value=value), self.assertRaises(self.d.PreparationError):
                self.d.parse_json('{"a": ' + value + '}')

    def test_scalar_metadata_rejects_vector_and_nonobject(self):
        for value in (np.array(['{}']), '[]', 'null', 2):
            with self.subTest(value=str(value)), self.assertRaises(self.d.PreparationError):
                self.d.scalar_meta(value)

    def test_valid_synthetic_plan(self):
        self.assertIsNone(self.d.validate_plan(self.plan))

    def test_plan_rejects_wrong_scope_and_revision(self):
        for mutation in ('scope', 'revision'):
            plan = copy.deepcopy(self.plan)
            if mutation == 'scope':
                plan['preparation_v6']['scope'] = 'RUN_MODEL'
            else:
                plan['document_revision'] = 2
            with self.subTest(mutation=mutation), self.assertRaises(self.d.PreparationError):
                self.d.validate_plan(plan)

    def test_plan_rejects_duplicate_clip_and_wrong_frame_total(self):
        for mutation in ('duplicate', 'frames'):
            plan = copy.deepcopy(self.plan)
            rows = plan['cohorts']['clip_index']
            if mutation == 'duplicate':
                rows[1]['clip'] = rows[0]['clip']
            else:
                rows[0]['T_metadata'] += 1
            with self.subTest(mutation=mutation), self.assertRaises(self.d.PreparationError):
                self.d.validate_plan(plan)

    def test_plan_rejects_known_test_alias(self):
        self.plan['cohorts']['clip_index'][0]['current_Stage11_TEST_aliases'] = ['synthetic_test']
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_plan(self.plan)

    def test_plan_rejects_nontrain_partition(self):
        self.plan['cohorts']['clip_index'][0]['current_partition'] = 'test'
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_plan(self.plan)

    def test_plan_rejects_canonical_self_donor(self):
        row = self.plan['cohorts']['clip_index'][0]
        row['original_shuffled_donor_canonical_id'] = row['canonical_id']
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_plan(self.plan)

    def test_plan_rejects_donor_outside_union(self):
        self.plan['cohorts']['clip_index'][0]['original_shuffled_donor_recording_id'] = 'unapproved'
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_plan(self.plan)

    def test_plan_rejects_duplicate_or_replaced_input_slot(self):
        for mutation in ('duplicate', 'replace'):
            plan = copy.deepcopy(self.plan)
            slots = plan['preparation_v6']['input_slots']
            if mutation == 'duplicate':
                slots[1] = dict(slots[0])
            else:
                slots[0]['path'] = '/synthetic/unapproved.npz'
            with self.subTest(mutation=mutation), self.assertRaises(self.d.PreparationError):
                self.d.validate_plan(plan)

    def test_plan_rejects_full_gate_claim(self):
        self.plan['qualified_gate_scope']['original_G_struct_contrast_completion_claim_permitted'] = True
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_plan(self.plan)

    def test_expected_state_shapes_represent_all_heads(self):
        blob = self.make_checkpoint()
        shapes = self.d.expected_state_shapes(blob['cfg'])
        self.assertEqual(shapes['inp.1.weight'], (8, 92))
        self.assertEqual(shapes['blocks.1.attn.qkv.weight'], (24, 8))
        self.assertEqual(shapes['blocks.0.conv.dw.weight'], (8, 1, 3))
        self.assertEqual(shapes['head_a.weight'], (4, 8))
        self.assertEqual(shapes['head_b.weight'], (3, 8))
        self.assertEqual(shapes['head_c.net.0.weight'], (320, 43))
        self.assertEqual(shapes['head_c.net.4.weight'], (17, 320))

    def test_checkpoint_metadata_all_arms_without_values(self):
        for arm in self.d.ARMS:
            with self.subTest(arm=arm):
                blob = self.make_checkpoint(arm)
                result = self.d.validate_checkpoint_metadata(blob, arm, self.plan, np)
                self.assertEqual(result['arm'], arm)
                self.assertFalse(result['model_constructed'])
                self.assertTrue(result['state_values_not_scored_or_used_in_forward'])
                self.assertEqual(result['state_tensor_count'], len(blob['model']))

    def test_checkpoint_rejects_unknown_field(self):
        blob = self.make_checkpoint()
        blob['optimizer'] = {}
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_checkpoint_metadata(blob, 'none', self.plan, np)

    def test_checkpoint_rejects_changed_config(self):
        blob = self.make_checkpoint()
        blob['cfg']['in_dim'] = 100
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_checkpoint_metadata(blob, 'none', self.plan, np)

    def test_checkpoint_rejects_block_reordering(self):
        blob = self.make_checkpoint()
        blob['block_names']['a'].reverse()
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_checkpoint_metadata(blob, 'none', self.plan, np)

    def test_checkpoint_rejects_normalizer_shape_dtype_nonfinite(self):
        for name, value in (('mu', np.zeros(91, dtype=np.float32)),
                            ('mu', np.zeros(92, dtype=np.float64)),
                            ('mu', np.full(92, np.nan, dtype=np.float32)),
                            ('sd', np.full(92, np.inf, dtype=np.float32)),
                            ('sd', np.zeros(92, dtype=np.float32)),
                            ('sd', -np.ones(92, dtype=np.float32))):
            blob = self.make_checkpoint()
            blob[name] = value
            with self.subTest(name=name, dtype=str(value.dtype), shape=value.shape), \
                 self.assertRaises(self.d.PreparationError):
                self.d.validate_checkpoint_metadata(blob, 'none', self.plan, np)

    def test_checkpoint_rejects_calibration_drift(self):
        for key, value in (('event_thr', [0.1, 0.2, 0.3]), ('smooth_tau', 2.0)):
            blob = self.make_checkpoint()
            blob[key] = value
            with self.subTest(key=key), self.assertRaises(self.d.PreparationError):
                self.d.validate_checkpoint_metadata(blob, 'none', self.plan, np)

    def test_checkpoint_rejects_state_key_shape_dtype_device(self):
        for mutation in ('missing', 'extra', 'shape', 'dtype', 'device'):
            blob = self.make_checkpoint()
            state = blob['model']
            key = next(iter(state))
            if mutation == 'missing':
                del state[key]
            elif mutation == 'extra':
                state['unexpected.weight'] = ShapeOnlyTensor((1,))
            else:
                setattr(state[key], mutation, {'shape': (999,), 'dtype': 'torch.float64',
                                               'device': 'cuda:0'}[mutation])
            with self.subTest(mutation=mutation), self.assertRaises(self.d.PreparationError):
                self.d.validate_checkpoint_metadata(blob, 'none', self.plan, np)

    def test_bus_valid_clock_and_shape(self):
        result = self.d.validate_bus(self.make_bus(), np)
        self.assertEqual(result['frames'], 200)
        self.assertEqual(result['fps'], 40)
        self.assertIs(result['oracle_beat'], False)

    def test_bus_rejects_empty_clock(self):
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_bus(self.make_bus(0), np)

    def test_bus_rejects_wrong_fps(self):
        bus = self.make_bus()
        meta = json.loads(bus['meta'])
        meta['fps'] = 30.0
        bus['meta'] = json.dumps(meta)
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_bus(bus, np)

    def test_bus_rejects_shifted_nonmonotone_and_nonfinite_clocks(self):
        for mutation in ('shift', 'repeat', 'nonfinite', 'integer'):
            bus = self.make_bus()
            if mutation == 'shift':
                bus['times'] += 0.025
            elif mutation == 'repeat':
                bus['times'][1] = bus['times'][0]
            elif mutation == 'nonfinite':
                bus['times'][1] = np.nan
            else:
                bus['times'] = np.arange(200, dtype=np.int32)
            with self.subTest(mutation=mutation), self.assertRaises(self.d.PreparationError):
                self.d.validate_bus(bus, np)

    def test_bus_accepts_float32_clock_roundoff(self):
        bus = self.make_bus()
        bus['times'] = bus['times'].astype(np.float32)
        self.assertEqual(self.d.validate_bus(bus, np)['frames'], 200)

    def test_bus_rejects_mel_length_or_axis_mismatch(self):
        for shape in ((64, 199), (200, 64)):
            bus = self.make_bus()
            bus['mel'] = np.zeros(shape, dtype=np.float32)
            with self.subTest(shape=shape), self.assertRaises(self.d.PreparationError):
                self.d.validate_bus(bus, np)

    def test_bus_rejects_scalar_length_and_nonfinite(self):
        for value in (np.zeros(199, dtype=np.float32), np.full(200, np.inf, dtype=np.float32)):
            bus = self.make_bus()
            bus['ch__rms'] = value
            with self.subTest(shape=value.shape), self.assertRaises(self.d.PreparationError):
                self.d.validate_bus(bus, np)

    def test_bus_rejects_unsorted_or_nonfinite_beats(self):
        for value in (np.array([0.5, 0.0]), np.array([0.5, 0.5]), np.array([np.nan])):
            bus = self.make_bus()
            bus['beats'] = value
            with self.subTest(value=value.tolist()), self.assertRaises(self.d.PreparationError):
                self.d.validate_bus(bus, np)

    def test_structure_valid_segments(self):
        info = self.d.validate_structure(self.make_structure(), np)
        self.assertEqual(info['frames'], 200)
        self.assertTrue(info['valid'].all())
        np.testing.assert_array_equal(info['section_ids'][:100], np.zeros(100, dtype=np.uint16))
        np.testing.assert_array_equal(info['section_ids'][100:], np.ones(100, dtype=np.uint16))

    def test_structure_accepts_nan_features_and_counts_them(self):
        data = self.make_structure()
        for key in ('energy', 'to_rise_s', 'to_drop_s'):
            data[key][:3] = np.nan
        info = self.d.validate_structure(data, np)
        self.assertEqual(info['missing_counts'], {'energy': 3, 'to_rise_s': 3, 'to_drop_s': 3})
        self.assertTrue(info['valid'].all())

    def test_structure_rejects_infinite_feature(self):
        for key in ('energy', 'to_rise_s', 'to_drop_s'):
            data = self.make_structure()
            data[key][0] = np.inf
            with self.subTest(key=key), self.assertRaises(self.d.PreparationError):
                self.d.validate_structure(data, np)

    def test_structure_rejects_role_out_of_range_or_wrong_dtype(self):
        for role in (np.full(200, -1, dtype=np.int16), np.full(200, 5, dtype=np.int16),
                     np.ones(200, dtype=np.float32)):
            data = self.make_structure()
            data['role'] = role
            with self.subTest(dtype=str(role.dtype)), self.assertRaises(self.d.PreparationError):
                self.d.validate_structure(data, np)

    def test_structure_rejects_role_order_and_duration_drift(self):
        for mutation in ('order', 'duration'):
            data = self.make_structure()
            meta = json.loads(data['meta'])
            if mutation == 'order':
                meta['roles'].reverse()
            else:
                meta['duration_s'] += 0.1
            data['meta'] = json.dumps(meta)
            with self.subTest(mutation=mutation), self.assertRaises(self.d.PreparationError):
                self.d.validate_structure(data, np)

    def test_structure_rejects_overlap_unknown_and_disagreement(self):
        for mutation in ('overlap', 'unknown', 'disagreement'):
            data = self.make_structure()
            meta = json.loads(data['meta'])
            if mutation == 'overlap':
                meta['segments'][1]['start_s'] = 2.0
            elif mutation == 'unknown':
                meta['segments'][1]['role'] = 'chorus'
            else:
                data['role'][100] = 1
            data['meta'] = json.dumps(meta)
            with self.subTest(mutation=mutation), self.assertRaises(self.d.PreparationError):
                self.d.validate_structure(data, np)

    def test_structure_gap_is_invalid_not_intro(self):
        data = self.make_structure(200, [(0.0, 2.0, 'breakdown'), (3.0, 5.0, 'drop')])
        info = self.d.validate_structure(data, np)
        self.assertEqual(int(info['valid'].sum()), 160)
        self.assertFalse(info['valid'][80:120].any())
        self.assertTrue(np.all(info['section_ids'][80:120] == 65535))

    def test_clip_qualifies_without_computing_scores(self):
        result = self.qualify()
        self.assertTrue(result['eligible'])
        self.assertEqual(result['status'], 'INPUT_QUALIFIED')
        self.assertEqual(result['retained_role_frames']['breakdown'], 100)
        self.assertEqual(result['retained_role_frames']['drop'], 100)
        self.assertFalse(result['ratio_or_prediction_computed'])
        self.assertEqual(result['reasons'], [])

    def test_clip_rejects_bus_shorter_than_exact_slice(self):
        with self.assertRaises(self.d.PreparationError):
            self.qualify(bus=self.make_bus(199))

    def test_clip_uses_recipient_start_index_not_donor_start(self):
        target = self.make_structure(240, [(0.0, 1.0, 'intro'),
                     (1.0, 3.5, 'breakdown'), (3.5, 6.0, 'drop')])
        result = self.qualify(start=1.0, bus=self.make_bus(240), target=target,
                              donor=self.make_structure(160))
        self.assertTrue(result['eligible'])
        self.assertEqual(result['i0'], 40)
        self.assertEqual(result['retained_role_frames']['drop'], 100)
        self.assertEqual(result['donor_conditioning_observed_frames'], 160)
        self.assertEqual(result['donor_conditioning_zero_tail_frames'], 40)
        self.assertEqual(result['donor_valid_role_frames_within_conditioning'], 160)

    def test_clip_accepts_half_frame_rounding(self):
        result = self.qualify(start=0.0125)
        self.assertEqual(result['i0'], 0)
        self.assertEqual(abs(result['roundoff_s']), 0.0125)
        self.assertTrue(result['eligible'])

    def test_clip_rounds_up_and_checks_resulting_slice(self):
        with self.assertRaises(self.d.PreparationError):
            self.qualify(start=0.013)
        result = self.qualify(start=0.013, bus=self.make_bus(201),
                              target=self.make_structure(201))
        self.assertEqual(result['i0'], 1)
        self.assertLessEqual(abs(result['roundoff_s']), 0.0125 + 1e-9)

    def test_clip_exact_95_percent_coverage_is_eligible(self):
        target = self.make_structure(200, [(0.0, 2.375, 'breakdown'), (2.625, 5.0, 'drop')])
        result = self.qualify(target=target)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['target_valid_role_frames'], 190)
        self.assertEqual(result['target_invalid_role_frames'], 10)

    def test_clip_one_frame_below_95_percent_is_ineligible(self):
        target = self.make_structure(200, [(0.0, 2.35, 'breakdown'), (2.625, 5.0, 'drop')])
        result = self.qualify(target=target)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['target_valid_role_frames'], 189)
        self.assertIn('TARGET_VALID_COVERAGE_LT95_PERCENT', result['reasons'])

    def test_clip_exact_80_frame_intersections_are_retained(self):
        result = self.qualify(n=160)
        self.assertTrue(result['eligible'])
        self.assertEqual([s['intersection_frames'] for s in result['sections']], [80, 80])
        self.assertTrue(all(s['retained'] for s in result['sections']))

    def test_clip_79_frame_intersection_is_not_retained(self):
        target = self.make_structure(200, [(0.0, 1.975, 'breakdown'), (1.975, 5.0, 'drop')])
        result = self.qualify(target=target)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['sections'][0]['intersection_frames'], 79)
        self.assertEqual(result['sections'][0]['reason'], 'INTERSECTION_LT80_FRAMES')
        self.assertEqual(result['retained_role_frames']['breakdown'], 0)

    def test_clip_does_not_merge_separate_short_sections(self):
        target = self.make_structure(200, [(0.0, 1.0, 'breakdown'),
                         (1.0, 2.0, 'breakdown'), (2.0, 5.0, 'drop')])
        result = self.qualify(target=target)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['retained_role_frames']['breakdown'], 0)
        self.assertIn('BREAKDOWN_RETAINED_LT80_FRAMES', result['reasons'])
        self.assertEqual(len(result['sections']), 3)

    def test_clip_source_section_length_cannot_replace_intersection_length(self):
        target = self.make_structure(240, [(0.0, 3.0, 'breakdown'), (3.0, 6.0, 'drop')])
        result = self.qualify(n=160, start=2.0, bus=self.make_bus(240), target=target)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['sections'][0]['intersection_frames'], 40)
        self.assertFalse(result['sections'][0]['retained'])

    def test_clip_target_tail_remains_invalid_without_truncation(self):
        result = self.qualify(target=self.make_structure(180))
        self.assertEqual(result['T'], 200)
        self.assertEqual(result['target_conditioning_zero_tail_frames'], 20)
        self.assertEqual(result['target_invalid_role_frames'], 20)
        self.assertFalse(result['eligible'])
        self.assertIn('TARGET_VALID_COVERAGE_LT95_PERCENT', result['reasons'])

    def test_clip_donor_roles_do_not_supply_target_support(self):
        target = self.make_structure(200, [(0.0, 5.0, 'intro')])
        result = self.qualify(target=target)
        self.assertFalse(result['eligible'])
        self.assertIn('DROP_RETAINED_LT80_FRAMES', result['reasons'])
        self.assertIn('BREAKDOWN_RETAINED_LT80_FRAMES', result['reasons'])

    def test_resource_zero_usage_reserves_whole_packet_and_comparison(self):
        result = self.resources()
        self.assertEqual(result['packet_remaining_bytes'], self.d.PACKET_CAP)
        self.assertEqual(result['combined_committed_bytes'], self.d.PACKET_CAP + self.d.COMPARISON_RESERVE)

    def test_resource_accepts_exact_global_cap_and_rejects_one_more_byte(self):
        original = self.d.GLOBAL_CAP - self.d.PACKET_CAP - self.d.COMPARISON_RESERVE
        self.assertEqual(self.resources(original_bytes=original)['combined_committed_bytes'], self.d.GLOBAL_CAP)
        with self.assertRaises(self.d.PreparationError):
            self.resources(original_bytes=original + 1)

    def test_resource_does_not_double_count_used_packet_or_comparison(self):
        result = self.resources(workspace_bytes=self.d.COMPARISON_RESERVE + 100,
                                comparison_bytes=self.d.COMPARISON_RESERVE,
                                packet_workspace_bytes=100, external_packet_bytes=200)
        self.assertEqual(result['packet_used_bytes'], 300)
        self.assertEqual(result['combined_committed_bytes'], self.d.COMPARISON_RESERVE + self.d.PACKET_CAP)

    def test_resource_rejects_negative_boolean_and_float_counts(self):
        for value in (-1, True, 1.0):
            with self.subTest(value=value), self.assertRaises(self.d.PreparationError):
                self.resources(other_reserved_bytes=value)

    def test_resource_rejects_workspace_overlap(self):
        with self.assertRaises(self.d.PreparationError):
            self.resources(workspace_bytes=100, comparison_bytes=60, packet_workspace_bytes=41)

    def test_resource_rejects_packet_and_comparison_overages(self):
        for change in ({'external_packet_bytes': self.d.PACKET_CAP + 1},
                       {'workspace_bytes': self.d.COMPARISON_RESERVE + 1,
                        'comparison_bytes': self.d.COMPARISON_RESERVE + 1}):
            with self.subTest(change=change), self.assertRaises(self.d.PreparationError):
                self.resources(**change)

    def test_resource_free_floor_includes_all_remaining_packet_writes(self):
        for key in ('free_mnt_bytes', 'free_root_bytes'):
            with self.subTest(key=key), self.assertRaises(self.d.PreparationError):
                self.resources(**{key: self.d.FREE_FLOOR + self.d.PACKET_CAP - 1})
        result = self.resources(external_packet_bytes=self.d.PACKET_CAP,
                                free_mnt_bytes=self.d.FREE_FLOOR, free_root_bytes=self.d.FREE_FLOOR)
        self.assertEqual(result['packet_remaining_bytes'], 0)

    def test_resource_other_reservations_count_against_global_cap(self):
        original = self.d.GLOBAL_CAP - self.d.PACKET_CAP - self.d.COMPARISON_RESERVE
        with self.assertRaises(self.d.PreparationError):
            self.resources(original_bytes=original, other_reserved_bytes=1)

    def test_authorization_valid_synthetic_bindings_only(self):
        auth, args = self.make_authorization()
        self.assertIsNone(self.d.validate_authorization(auth, *args))

    def test_authorization_rejects_missing_root_or_wrong_scope(self):
        for key, value in (('authorized_by', 'child'), ('scope', 'EVALUATE')):
            auth, args = self.make_authorization()
            auth[key] = value
            with self.subTest(key=key), self.assertRaises(self.d.PreparationError):
                self.d.validate_authorization(auth, *args)

    def test_authorization_rejects_any_forbidden_permission(self):
        for key in ('allow_model_construction', 'allow_model_forward', 'allow_scoring',
                    'allow_GPU', 'allow_TEST_access', 'allow_raw_manifest_access'):
            auth, args = self.make_authorization()
            auth[key] = True
            with self.subTest(key=key), self.assertRaises(self.d.PreparationError):
                self.d.validate_authorization(auth, *args)

    def test_authorization_rejects_command_and_hash_drift(self):
        for mutation in ('argv', 'plan_ref', 'driver_ref', 'test_refs'):
            auth, args = self.make_authorization()
            if mutation == 'argv':
                auth['argv'] = auth['argv'] + ['--unauthorized']
            elif mutation == 'test_refs':
                auth[mutation] = [self.fake_ref('different.py')]
            else:
                auth[mutation] = self.fake_ref('different.json')
            with self.subTest(mutation=mutation), self.assertRaises(self.d.PreparationError):
                self.d.validate_authorization(auth, *args)

    def test_authorization_rejects_failed_incomplete_and_scope_exceeding_tests(self):
        mutations = (('exit_code', 1), ('failures', 1), ('errors', 1),
                     ('tests_run', 23), ('tests_expected', 25), ('real_model_forwards', 1),
                     ('sealed_TEST_reads', 1), ('checkpoint_deserializations', 1))
        for key, value in mutations:
            auth, args = self.make_authorization()
            auth['native_synthetic_test_result'][key] = value
            with self.subTest(key=key), self.assertRaises(self.d.PreparationError):
                self.d.validate_authorization(auth, *args)

    def test_authorization_rejects_receipt_source_binding_drift(self):
        auth, args = self.make_authorization()
        auth['native_synthetic_test_result']['driver_ref'] = self.fake_ref('other.py')
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_authorization(auth, *args)

    def test_authorization_rejects_unverified_native_registration(self):
        auth, args = self.make_authorization()
        auth['native_registration_receipt_verified'] = False
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_authorization(auth, *args)

    def test_authorization_rejects_changed_external_limits(self):
        auth, args = self.make_authorization()
        auth['preparation_external_limits_bound']['GPU_access'] = True
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_authorization(auth, *args)

    def test_authorization_rejects_input_alias_identity_or_approval_drift(self):
        changes = (('current_Stage11_partition', 'test'),
                   ('known_current_TEST_aliases', ['synthetic_test']),
                   ('cross_corpus_identity', 'INDEPENDENT'),
                   ('approval', 'CHILD_APPROVED'), ('identity_projection_ref', {}))
        for key, value in changes:
            auth, args = self.make_authorization()
            auth['input_refs'][0][key] = value
            with self.subTest(key=key), self.assertRaises(self.d.PreparationError):
                self.d.validate_authorization(auth, *args)

    def test_authorization_rejects_missing_input_and_changed_checkpoint_map(self):
        for mutation in ('input', 'checkpoint'):
            auth, args = self.make_authorization()
            if mutation == 'input':
                auth['input_refs'].pop()
            else:
                auth['checkpoint_refs'] = dict(auth['checkpoint_refs'], none=self.fake_ref('other.pt'))
            with self.subTest(mutation=mutation), self.assertRaises(self.d.PreparationError):
                self.d.validate_authorization(auth, *args)


    def test_bus_accepts_actual_duration_within_one_frame(self):
        bus = self.make_bus()
        meta = json.loads(bus['meta'])
        meta['duration_s'] -= 0.02
        bus['meta'] = json.dumps(meta)
        self.assertEqual(self.d.validate_bus(bus, np)['frames'], 200)

    def test_bus_rejects_duration_more_than_one_frame_from_grid(self):
        for drift in (-0.03, 0.03):
            bus = self.make_bus()
            meta = json.loads(bus['meta'])
            meta['duration_s'] += drift
            bus['meta'] = json.dumps(meta)
            with self.subTest(drift=drift), self.assertRaises(self.d.PreparationError):
                self.d.validate_bus(bus, np)

    def test_bus_rejects_finite_features_that_overflow_float32(self):
        for key in ('mel', 'ch__rms'):
            bus = self.make_bus()
            bus[key] = np.full(bus[key].shape, 1e100, dtype=np.float64)
            with self.subTest(key=key), self.assertRaises(self.d.PreparationError):
                self.d.validate_bus(bus, np)

    def test_clip_rejects_invalid_start_before_rounding(self):
        for start in (float('nan'), float('inf'), -float('inf'), -0.025, True, '0.0', None):
            with self.subTest(start=start), self.assertRaises(self.d.PreparationError):
                self.qualify(start=start)

    def test_clip_rejects_invalid_metadata_frame_count(self):
        bus = self.make_bus()
        info = self.d.validate_structure(self.make_structure(), np)
        for count in (0, -1, True, 200.0):
            row = {'clip': 'synthetic_clip', 'recording_id': 'synthetic_recipient',
                   'T_metadata': count, 'start_s_metadata': 0.0,
                   'original_shuffled_donor_recording_id': 'synthetic_donor'}
            with self.subTest(count=count), self.assertRaises(self.d.PreparationError):
                self.d.qualify_clip(row, bus, info, info, np)

    def test_structure_rejects_energy_float32_overflow(self):
        data = self.make_structure()
        data['energy'] = np.full(200, 1e100, dtype=np.float64)
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_structure(data, np)

    def test_structure_large_finite_distances_are_bounded_before_range_check(self):
        data = self.make_structure()
        data['to_rise_s'] = np.full(200, 1e100, dtype=np.float64)
        data['to_drop_s'] = np.full(200, -1e100, dtype=np.float64)
        info = self.d.validate_structure(data, np)
        self.assertTrue(info['valid'].all())

    def test_authorization_allows_synthetic_identity_ref_without_size(self):
        ref = self.plan['preparation_v6']['identity_projection_ref']
        del ref['size_bytes']
        auth, args = self.make_authorization()
        self.assertEqual(len(ref['sha256']), 64)
        self.assertIsNone(self.d.validate_authorization(auth, *args))
        changed = dict(ref, sha256='d' * 64)
        auth['input_refs'][0]['identity_projection_ref'] = changed
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_authorization(auth, *args)

    def test_static_control_reads_are_bounded_even_without_declared_size(self):
        reader = self.function('read_ref')
        calls = [n for n in ast.walk(reader) if isinstance(n, ast.Call)]
        sizes = [n for n in calls if self.call_name(n.func) == 'ref.get'
                 and n.args and isinstance(n.args[0], ast.Constant)
                 and n.args[0].value == 'size_bytes']
        self.assertEqual(len(sizes), 1)
        reads = [n for n in calls if self.call_name(n.func) == 'stream.read']
        self.assertEqual(len(reads), 1)
        self.assertEqual(ast.unparse(reads[0].args[0]), 'max_bytes + 1')
        self.assertTrue(any(self.call_name(n.func) == 'digest' for n in calls))


    @staticmethod
    def synthetic_log():
        return (b'test_one (synthetic.Tests.test_one) ... ok\n'
                b'test_two (synthetic.Tests.test_two) ... ok\n\n'
                b'----------------------------------------------------------------------\n'
                b'Ran 2 tests in 0.001s\n\nOK\n')

    def test_native_log_accepts_complete_verbose_success(self):
        self.assertIsNone(self.d.validate_native_test_log(self.synthetic_log(), 2))

    def test_native_log_rejects_missing_success_line(self):
        data = self.synthetic_log().replace(b'test_two (synthetic.Tests.test_two) ... ok\n', b'')
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_native_test_log(data, 2)

    def test_native_log_rejects_duplicate_summary(self):
        data = self.synthetic_log().replace(b'Ran 2 tests in 0.001s',
                     b'Ran 2 tests in 0.001s\nRan 2 tests in 0.001s')
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_native_test_log(data, 2)

    def test_native_log_rejects_wrong_summary_count(self):
        data = self.synthetic_log().replace(b'Ran 2 tests', b'Ran 3 tests')
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_native_test_log(data, 2)

    def test_native_log_rejects_missing_ok_and_trailing_output(self):
        for data in (self.synthetic_log().replace(b'\nOK\n', b'\n'),
                     self.synthetic_log() + b'unexpected trailing output\n'):
            with self.subTest(data=data), self.assertRaises(self.d.PreparationError):
                self.d.validate_native_test_log(data, 2)

    def test_native_log_rejects_nonpass_status_even_with_enough_ok_lines(self):
        for status in ('skipped', 'expected failure', 'unexpected success', 'FAIL', 'ERROR'):
            extra = ('test_extra (synthetic.Tests.test_extra) ... ' + status + '\n').encode()
            with self.subTest(status=status), self.assertRaises(self.d.PreparationError):
                self.d.validate_native_test_log(extra + self.synthetic_log(), 2)

    def test_authorization_rejects_skips_and_expected_failure_statuses(self):
        for key in ('skipped', 'expected_failures', 'unexpected_successes'):
            auth, args = self.make_authorization()
            auth['native_synthetic_test_result'][key] = 1
            with self.subTest(key=key), self.assertRaises(self.d.PreparationError):
                self.d.validate_authorization(auth, *args)

    def test_authorization_requires_exact_planned_test_count(self):
        auth, args = self.make_authorization()
        receipt = auth['native_synthetic_test_result']
        receipt['tests_run'] = receipt['tests_expected'] = EXPECTED_TEST_COUNT - 1
        with self.assertRaises(self.d.PreparationError):
            self.d.validate_authorization(auth, *args)

    def test_static_native_log_gate_precedes_payload_access(self):
        calls = [n for n in ast.walk(self.function('main')) if isinstance(n, ast.Call)]
        gate = next(n.lineno for n in calls if self.call_name(n.func) == 'validate_native_test_log')
        dispatches = [n for n in calls if self.call_name(n.func) == 'prepare_authenticated']
        self.assertEqual(len(dispatches), 1)
        self.assertGreater(dispatches[0].lineno, gate)

    def test_declared_test_count_matches_discoverable_methods(self):
        methods = [name for name in dir(type(self)) if name.startswith('test_')]
        self.assertEqual(len(methods), EXPECTED_TEST_COUNT)


    def test_bus_and_structure_reject_adjacent_jitter_within_absolute_tolerance(self):
        for kind in ('bus', 'structure'):
            data = self.make_bus() if kind == 'bus' else self.make_structure()
            key = 'times' if kind == 'bus' else 't'
            data[key][::2] += 0.00009
            data[key][1::2] -= 0.00009
            validator = self.d.validate_bus if kind == 'bus' else self.d.validate_structure
            with self.subTest(kind=kind), self.assertRaises(self.d.PreparationError):
                validator(data, np)

    def test_bus_rejects_null_or_list_flags_but_accepts_absent_flags(self):
        for flags in (None, []):
            bus = self.make_bus()
            meta = json.loads(bus['meta'])
            meta['flags'] = flags
            bus['meta'] = json.dumps(meta)
            with self.subTest(flags=flags), self.assertRaises(self.d.PreparationError):
                self.d.validate_bus(bus, np)
        bus = self.make_bus()
        meta = json.loads(bus['meta'])
        del meta['flags']
        bus['meta'] = json.dumps(meta)
        self.assertEqual(self.d.validate_bus(bus, np)['oracle_beat'], 'UNKNOWN')


    def producer_structure(self, tail_samples=0, spans=None, stft_last_index=430):
        # Hypothetical sample count -> centered-STFT last center, waveform end,
        # rounded40Hz length and role masks. No librosa import or audio read.
        sample_count = stft_last_index * 512 + tail_samples
        center = (sample_count // 512) * 512 / 22050.0
        endpoint = sample_count / 22050.0
        n = int(round(center * 40.0))
        if spans is None:
            spans = [(0.0, center / 2, 'breakdown'), (center / 2, endpoint, 'drop')]
        data = self.make_structure(n, spans)
        meta = json.loads(data['meta'])
        meta['duration_s'] = center
        data['meta'] = json.dumps(meta)
        return data, center, endpoint

    def test_producer_zero_final_tail_is_valid(self):
        data, center, endpoint = self.producer_structure(0)
        info = self.d.validate_structure(data, np)
        self.assertEqual(endpoint, center)
        self.assertEqual(info['cached_duration_last_STFT_center_s'], center)
        self.assertEqual(info['final_endpoint_minus_STFT_center_s'], 0.0)
        self.assertEqual(info['frames'], int(round(center * 40)))

    def test_producer_near_one_hop_final_tail_is_valid_without_extension(self):
        data, center, endpoint = self.producer_structure(511)
        before = {key: value.copy() for key, value in data.items() if isinstance(value, np.ndarray)}
        info = self.d.validate_structure(data, np)
        self.assertGreater(endpoint - center, 1e-4)
        self.assertLess(endpoint - center, 512 / 22050.0)
        self.assertAlmostEqual(info['final_endpoint_minus_STFT_center_s'], 511 / 22050.0)
        self.assertEqual(info['cached_duration_last_STFT_center_s'], center)
        self.assertEqual(info['frames'], len(data['t']))
        self.assertTrue(info['cache_arrays_not_extended'])
        for key in before:
            np.testing.assert_array_equal(before[key], data[key])

    def test_producer_final_tail_above_hop_is_invalid(self):
        data, center, _ = self.producer_structure(0)
        meta = json.loads(data['meta'])
        meta['segments'][-1]['end_s'] = center + 512 / 22050.0 + 1e-6
        data['meta'] = json.dumps(meta)
        with self.assertRaisesRegex(self.d.PreparationError, 'final segment exceeds'):
            self.d.validate_structure(data, np)

    def test_producer_intermediate_endpoint_gets_no_hop_exception(self):
        data, center, _ = self.producer_structure(511)
        meta = json.loads(data['meta'])
        mid = center + (512 / 22050.0) / 2
        meta['segments'][0]['end_s'] = mid
        meta['segments'][1]['start_s'] = mid
        data['meta'] = json.dumps(meta)
        with self.assertRaisesRegex(self.d.PreparationError, 'intermediate segment exceeds'):
            self.d.validate_structure(data, np)

    def test_producer_final_start_gets_no_hop_exception(self):
        data, center, endpoint = self.producer_structure(511)
        meta = json.loads(data['meta'])
        meta['segments'] = [{'start_s': center + 0.001, 'end_s': endpoint, 'role': 'drop'}]
        data['meta'] = json.dumps(meta)
        with self.assertRaisesRegex(self.d.PreparationError, 'out-of-duration'):
            self.d.validate_structure(data, np)

    def test_producer_tail_does_not_permit_overlap(self):
        data, center, _ = self.producer_structure(511)
        meta = json.loads(data['meta'])
        meta['segments'][1]['start_s'] = center / 2 - 0.025
        data['meta'] = json.dumps(meta)
        with self.assertRaisesRegex(self.d.PreparationError, 'overlapping'):
            self.d.validate_structure(data, np)

    def test_producer_tail_does_not_fill_role_gap(self):
        center = 430 * 512 / 22050.0
        endpoint = center + 511 / 22050.0
        spans = [(0.0, center / 2 - 1.0, 'breakdown'),
                 (center / 2 + 1.0, endpoint, 'drop')]
        data, _, _ = self.producer_structure(511, spans)
        info = self.d.validate_structure(data, np)
        gap = (data['t'] >= center / 2 - 1.0) & (data['t'] < center / 2 + 1.0)
        self.assertTrue(gap.any())
        self.assertFalse(info['valid'][gap].any())
        self.assertTrue(np.all(info['section_ids'][gap] == 65535))
        self.assertEqual(info['frames'], len(data['t']))

    def test_authorization_rejects_nonempty_unpublished_native_command(self):
        auth, args = self.make_authorization()
        auth['native_synthetic_test_result']['command'] = 'different nonempty native command'
        with self.assertRaisesRegex(self.d.PreparationError, 'exact published command'):
            self.d.validate_authorization(auth, *args)


    def test_producer_sample_count_fixtures_cover_both_40hz_rounding_directions(self):
        signs = set()
        for index in (400, 430):
            for residual in (0, 511):
                with self.subTest(index=index, residual=residual):
                    data, center, endpoint = self.producer_structure(residual, stft_last_index=index)
                    info = self.d.validate_structure(data, np)
                    expected_samples = index * 512 + residual
                    self.assertEqual(endpoint, expected_samples / 22050.0)
                    self.assertEqual(center, (expected_samples // 512) * 512 / 22050.0)
                    self.assertEqual(info['frames'], round(center * 40))
                    signs.add(1 if info['frames'] / 40.0 > center else -1)
                    self.assertEqual(info['cached_duration_last_STFT_center_s'], center)
        self.assertEqual(signs, {-1, 1})

    def test_numpy_assertion_works_inside_active_filesystem_guards(self):
        # These deliberate calls must still hit the unchanged active guards.
        with self.assertRaisesRegex(AssertionError, 'forbidden filesystem read: Path.read_text'):
            Path('/synthetic-not-a-real-input/no-file.txt').read_text()
        with self.assertRaisesRegex(AssertionError, 'forbidden numerical file read: load'):
            np.load('/synthetic-not-a-real-input/no-array.npz')
        self.assertIs(np.testing, npt)
        npt.assert_array_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))
        np.testing.assert_array_equal(np.array([4, 5]), np.array([4, 5]))

    def qualify_with_buffer_capture(self, **kwargs):
        buffers = {}
        original_full = np.full
        def capture(shape, fill_value, *args, **options):
            array = original_full(shape, fill_value, *args, **options)
            if fill_value == 255 and array.dtype == np.uint8:
                buffers['roles'] = array
            elif fill_value == 65535 and array.dtype == np.uint16:
                buffers['sections'] = array
            return array
        with mock.patch.object(np, 'full', side_effect=capture):
            result = self.qualify(**kwargs)
        return result, buffers

    def test_representative_structure_fixture_uses_producer_int8(self):
        self.assertEqual(self.make_structure()['role'].dtype, np.int8)

    def test_producer_int8_all_valid_buffer_copy_and_no_input_mutation(self):
        target = self.make_structure(200)
        donor = self.make_structure(200)
        before_target = {k: v.copy() for k, v in target.items() if isinstance(v, np.ndarray)}
        before_donor = {k: v.copy() for k, v in donor.items() if isinstance(v, np.ndarray)}
        result, buffers = self.qualify_with_buffer_capture(n=200, target=target, donor=donor)
        self.assertTrue(result['eligible'])
        self.assertEqual(target['role'].dtype, np.int8)
        self.assertEqual(buffers['roles'].dtype, np.uint8)
        self.assertEqual(buffers['sections'].dtype, np.uint16)
        npt.assert_array_equal(buffers['roles'], target['role'])
        self.assertFalse(np.any(buffers['roles'] == 255))
        self.assertFalse(np.any(buffers['sections'] == 65535))
        for key in before_target:
            npt.assert_array_equal(target[key], before_target[key])
        for key in before_donor:
            npt.assert_array_equal(donor[key], before_donor[key])

    def test_producer_int8_gap_and_tail_keep_uint8_sentinels_without_mutation(self):
        target = self.make_structure(200, [(0.0, 2.0, 'breakdown'), (3.0, 5.0, 'drop')])
        donor = self.make_structure(200)
        original_roles = target['role'].copy()
        result, buffers = self.qualify_with_buffer_capture(n=240, bus=self.make_bus(240),
                                                         target=target, donor=donor)
        expected_roles = np.full(240, 255, dtype=np.uint8)
        expected_roles[:80] = 1
        expected_roles[120:200] = 3
        expected_sections = np.full(240, 65535, dtype=np.uint16)
        expected_sections[:80] = 0
        expected_sections[120:200] = 1
        npt.assert_array_equal(buffers['roles'], expected_roles)
        npt.assert_array_equal(buffers['sections'], expected_sections)
        npt.assert_array_equal(target['role'], original_roles)
        self.assertEqual(target['role'].dtype, np.int8)
        self.assertEqual(result['target_valid_role_frames'], 160)
        self.assertEqual(result['target_invalid_role_frames'], 80)
        self.assertEqual(result['target_conditioning_zero_tail_frames'], 40)
        self.assertFalse(result['eligible'])
        self.assertIn('TARGET_VALID_COVERAGE_LT95_PERCENT', result['reasons'])

    def test_unsigned_role_compatibility_keeps_gap_tail_and_source_unchanged(self):
        target = self.make_structure(200, [(0.0, 2.0, 'breakdown'), (3.0, 5.0, 'drop')])
        donor = self.make_structure(200)
        target['role'] = target['role'].astype(np.uint8)
        donor['role'] = donor['role'].astype(np.uint8)
        before = target['role'].copy()
        result, buffers = self.qualify_with_buffer_capture(n=240, bus=self.make_bus(240),
                                                         target=target, donor=donor)
        npt.assert_array_equal(target['role'], before)
        self.assertEqual(target['role'].dtype, np.uint8)
        self.assertEqual(buffers['roles'].dtype, np.uint8)
        self.assertTrue(np.all(buffers['roles'][80:120] == 255))
        self.assertTrue(np.all(buffers['roles'][200:] == 255))
        self.assertTrue(np.all(buffers['sections'][80:120] == 65535))
        self.assertEqual(result['target_invalid_role_frames'], 80)

    def test_producer_int8_invalid_label_is_not_cast_into_validity(self):
        target = self.make_structure(200)
        target['role'][0] = 5
        before = target['role'].copy()
        with self.assertRaisesRegex(self.d.PreparationError, 'invalid role values'):
            self.qualify_with_buffer_capture(n=200, target=target)
        npt.assert_array_equal(target['role'], before)

    def test_static_progress_is_after_verified_metadata_and_recorded_clip_outcome(self):
        payload = self.function('prepare_authenticated')
        calls = [n for n in ast.walk(payload) if isinstance(n, ast.Call)]
        validation = next(n.lineno for n in calls if self.call_name(n.func) == 'validate_checkpoint_metadata')
        append = next(n.lineno for n in calls if self.call_name(n.func) == 'rows.append')
        progress = {n.args[0].value: n.lineno for n in calls
                    if self.call_name(n.func) == 'durable_progress' and n.args
                    and isinstance(n.args[0], ast.Constant)}
        self.assertEqual(set(progress), {'CHECKPOINT_METADATA_VERIFIED', 'CLIP_QUALIFICATION_RECORDED'})
        self.assertGreater(progress['CHECKPOINT_METADATA_VERIFIED'], validation)
        self.assertGreater(progress['CLIP_QUALIFICATION_RECORDED'], append)
        self.assertTrue(any(isinstance(n, ast.Constant) and n.value == 'progress_only_not_readiness'
                            for n in ast.walk(self.function('durable_progress'))))

    def test_static_progress_is_bounded_capped_and_fsynced(self):
        function = self.function('durable_progress')
        calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
        names = [self.call_name(n.func) for n in calls]
        self.assertIn('sink.write', names)
        self.assertIn('sink.flush_durable', names)
        self.assertTrue(any(isinstance(n, ast.Constant) and n.value == 2048 for n in ast.walk(function)))
        capped = next(n for n in self.tree.body if isinstance(n, ast.ClassDef) and n.name == 'CappedLog')
        sync = next(n for n in capped.body if isinstance(n, ast.FunctionDef) and n.name == 'flush_durable')
        sync_calls = [n for n in ast.walk(sync) if isinstance(n, ast.Call)]
        flush_line = next(n.lineno for n in sync_calls if self.call_name(n.func) == 'self.flush')
        fsync_line = next(n.lineno for n in sync_calls if self.call_name(n.func) == 'os.fsync')
        self.assertLess(flush_line, fsync_line)

if __name__ == '__main__':
    unittest.main()

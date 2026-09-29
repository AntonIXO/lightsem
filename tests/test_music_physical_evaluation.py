"""CPU-only engineering. No checkpoint, VAL/TEST arrays, or GPU calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

PATH = Path(__file__).resolve().parents[1]/'scripts/motion/eval_music_physical_revision.py'
spec = importlib.util.spec_from_file_location('physical_evaluation', PATH)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
MOD = ev.frozen_modules()
torch.set_num_threads(2)


def normalizers():
    mean = np.zeros(139, np.float32)
    mean[2] = 1.
    mean[3:135] = np.tile([1., 0., 0., 0., 1., 0.], 22)
    mean[135:] = .5
    return {'mean': mean.tolist(), 'std': np.ones(139).tolist(), 'delta_scale': np.ones(139).tolist(),
            'summary': {'mean': [0.]*18, 'std': [1.]*18}}


def tiny_model():
    torch.manual_seed(4)
    config = MOD['music_model'].MusicConfig(width=8, dilations=(1, 2), frames=60)
    model = MOD['music_revision_model'].MusicRevisionModel(config, normalizers(), arm='masked_native')
    with torch.no_grad():
        model.output.weight.normal_(0, .015)
        model.output.bias.normal_(0, .01)
    return model.eval()


def entry(key, song, frames, partition='val'):
    return {'key': key, 'canonical_id': song, 'n_frames': frames, 'partition': partition}


def metric_row(song, key, start, value):
    return {'canonical_id': song, 'key': key, 'start': start, 'metrics': {'v': value}}


def gate_summary(sliding=.1):
    metrics = {}
    for key in ['kinetic', 'travel', 'contact_coverage']:
        metrics['source_'+key] = metrics['prediction_'+key] = 1.
    for key in ['sliding', 'penetration', 'high_frequency']:
        metrics['source_'+key] = 0.
        metrics['prediction_'+key] = sliding if key == 'sliding' else .01
    return {'metrics': metrics, 'per_song': {str(i): {'correct_es': .9, 'donor_es': 1., 'shift_es': 1.} for i in range(4)}}


class PersistenceTests(unittest.TestCase):
    def test_json_replay_and_drift(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'x.json'
            ev.save_json(path, {'x': 2})
            content = path.read_bytes()
            ev.save_json(path, {'x': 2})
            self.assertEqual(content, path.read_bytes())
            self.assertEqual(ev.load_json(path), {'x': 2})
            with self.assertRaises(ValueError): ev.save_json(path, {'x': 3})
            envelope = json.loads(content)
            envelope['payload']['x'] = 4
            path.write_text(json.dumps(envelope))
            with self.assertRaises(ValueError): ev.load_json(path)

    def test_npz_restart_and_finite(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'x.npz'
            a = {'x': np.ones((2, 3), np.float32)}
            ev.save_npz(path, a)
            ev.save_npz(path, a)
            with self.assertRaises(ValueError): ev.save_npz(path, {'x': a['x']*2})
            with self.assertRaises(ValueError): ev.save_npz(path, {'x': a['x'].astype(np.float64)})
            with self.assertRaises(FloatingPointError): ev.save_npz(path, {'x': a['x']*np.nan})

    def test_orphan_npz_replay_and_row_hash(self):
        with tempfile.TemporaryDirectory() as d:
            path, arr = Path(d)/'row.json', Path(d)/'row.npz'
            ev.save_npz(arr, {'v': np.zeros(2)})
            row = {'identity': {'cell': 1}, 'spec': {'window': 3}, 'arrays_path': str(arr),
                   'arrays_sha256': ev.digest(arr), 'finite_all_draws': True}
            ev.save_json(path, row)
            self.assertEqual(ev.verify_row(path, row['identity'], row['spec']), row)
            with self.assertRaises(ValueError): ev.verify_row(path, {'cell': 2}, row['spec'])
            arr.write_bytes(b'corruption')
            with self.assertRaises(ValueError): ev.verify_row(path, row['identity'], row['spec'])

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): ev.save_json(Path(d)/'x', {'v': float('nan')})


class ProtocolTests(unittest.TestCase):
    def test_short_controls_are_metadata_only_supported_and_stable(self):
        entries = [entry('a', 'songA', 600), entry('b', 'songA', 800), entry('c', 'songC', 1500), entry('t', 'train', 2000, 'train')]
        result = ev.short_windows(entries)
        self.assertEqual(result, ev.short_windows(list(reversed(entries))))
        self.assertEqual(len(result), 2)
        for w in result:
            self.assertEqual(w['donor_key'], 'c')
            self.assertIsNone(w['shift_start'])
            self.assertIn('shift', w['unsupported'])
            self.assertEqual(w['start'], 0)
        ev.validate_windows(result, entries)

    def test_short_no_valid_donor_not_fabricated(self):
        result = ev.short_windows([entry('a', 'same', 600), entry('b', 'same', 500)])
        self.assertTrue(all(w['donor_key'] is None for w in result))
        self.assertIn('donor', result[0]['unsupported'])

    def test_bad_shift_and_same_song_donor_rejected(self):
        entries = [entry('a', 's1', 2000), entry('b', 's2', 2000)]
        w = {'key': 'a', 'canonical_id': 's1', 'start': 0, 'frames': 960,
             'donor_key': 'b', 'donor_start': 0, 'shift_start': 1000, 'draw_seeds': list(range(6))}
        ev.validate_windows([w], entries, primary=True)
        with self.assertRaises(ValueError): ev.validate_windows([{**w, 'shift_start': 100}], entries, primary=True)
        with self.assertRaises(ValueError): ev.validate_windows([{**w, 'donor_key': 'a'}], entries, primary=True)
        with self.assertRaises(ValueError): ev.validate_windows([w, w], entries, primary=True)

    def test_control_exact_slices_no_padding(self):
        values = {'a': (np.arange(2000*92, dtype=np.float32).reshape(2000, 92), np.zeros((2000, 139), np.float32)),
                  'b': (np.ones((1000, 92), np.float32), np.zeros((1000, 139), np.float32))}
        w = {'key': 'a', 'start': 10, 'frames': 600, 'donor_key': 'b', 'donor_start': 20, 'shift_start': 900}
        controls, target = ev.control_arrays(w, values.__getitem__)
        np.testing.assert_array_equal(controls['correct'], values['a'][0][10:610])
        np.testing.assert_array_equal(controls['shift'], values['a'][0][900:1500])
        self.assertEqual(target.shape, (600, 139))
        with self.assertRaises(ValueError): ev.control_arrays({**w, 'donor_start': 900}, values.__getitem__)


class StatisticsTests(unittest.TestCase):
    def test_equal_song_record_window_not_flat_mean(self):
        rows = [metric_row('a', 'a1', 0, 0), metric_row('a', 'a1', 1, 2),
                metric_row('a', 'a2', 0, 9), metric_row('b', 'b1', 0, 20)]
        reduced = ev.aggregate(rows)
        self.assertEqual(reduced['per_song']['a']['v'], 5)
        self.assertEqual(reduced['metrics']['v'], 12.5)
        self.assertNotEqual(reduced['metrics']['v'], np.mean([0, 2, 9, 20]))
        with self.assertRaises(ValueError): ev.aggregate(rows+[rows[0]])

    def test_bootstrap_fixed_seed_and_paired_song(self):
        songs = {str(i): {'correct_es': .8+i/100, 'donor_es': 1+i/100} for i in range(6)}
        a = ev.comparison(songs, 'donor')
        self.assertEqual(a, ev.comparison(dict(reversed(list(songs.items()))), 'donor'))
        self.assertEqual(a['bootstrap_draws'], 10000)
        self.assertEqual(a['bootstrap_seed'], 20260910)
        self.assertGreater(a['relative_ci95'][0], 0)

    def test_declared_gate_boundaries(self):
        physical, baseline = gate_summary(.2), gate_summary(.25)
        passed = ev.numerical_gates(physical, baseline, 5e-5)
        self.assertTrue(all(passed['gates'].values()))
        self.assertFalse(ev.numerical_gates(physical, baseline, 5.00001e-5)['gates']['continuity'])
        self.assertFalse(ev.numerical_gates(gate_summary(.200001), baseline, 0)['gates']['sliding_absolute'])
        self.assertFalse(ev.numerical_gates(physical, gate_summary(.249), 0)['gates']['sliding_reduction_vs_matched_native'])
        for name, metric, low, high in [('energy', 'kinetic', .5, 2.), ('travel', 'travel', .25, 2.), ('contacts', 'contact_coverage', .5, 1.5)]:
            for boundary in (low, high):
                q = gate_summary()
                q['metrics']['prediction_'+metric] = boundary
                self.assertTrue(ev.numerical_gates(q, baseline, 0)['gates'][name])
            q['metrics']['prediction_'+metric] = high+.001
            self.assertFalse(ev.numerical_gates(q, baseline, 0)['gates'][name])

    def test_conditioning_both_controls_and_ci_strict(self):
        q, baseline = gate_summary(), gate_summary(.2)
        for values in q['per_song'].values(): values['shift_es'] = .9
        result = ev.numerical_gates(q, baseline, 0)
        self.assertFalse(result['gates']['conditioning_magnitude'])
        self.assertFalse(result['gates']['conditioning_ci'])

    def test_quality_ratio_of_weighted_means(self):
        q = gate_summary()
        q['metrics']['prediction_kinetic'], q['metrics']['source_kinetic'] = 4., 2.
        self.assertEqual(ev.quality(q)['kinetic_ratio'], 2.)
        q['metrics']['source_sliding'] = 1.
        self.assertEqual(ev.quality(q)['sliding_excess'], 0.)


class SyntheticModelTests(unittest.TestCase):
    def test_frozen_import_identity(self):
        for module in MOD.values():
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(ev.EXPERIMENT/'source'))

    def test_six_draw_32euler_same_noise_conditions_and_restart(self):
        model, norms = tiny_model(), normalizers()
        w = {'frames': 61, 'draw_seeds': [10, 20, 30, 40, 50, 60]}
        target = np.tile(np.asarray(norms['mean'], np.float32), (61, 1))
        audio = np.random.default_rng(42).normal(size=(61, 92)).astype(np.float32)
        controls = {c: audio.copy() for c in ['correct', 'donor', 'shift']}
        calls = []
        original = model.generate
        def checked(*args, **kwargs):
            calls.append((kwargs['steps'], kwargs['initial_noise'].clone()))
            return original(*args, **kwargs)
        with patch.object(model, 'generate', side_effect=checked):
            metrics, arrays, row = ev.evaluate_window(model, w, controls, target, norms, MOD, True)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(step == 32 for step, noise in calls))
        self.assertTrue(all(torch.equal(calls[0][1], noise) for _, noise in calls))
        self.assertEqual(calls[0][1].shape, (6, 61, 139))
        self.assertEqual(arrays['correct'].shape, (6, 4, 18))
        self.assertEqual(row['summary_frames_scored'], 60)
        self.assertEqual(row['physical_frames_scored'], 61)
        self.assertEqual(row['prediction_content_sha256']['correct'], row['prediction_content_sha256']['donor'])
        self.assertEqual(metrics['correct_es'], metrics['shift_es'])
        metrics2, arrays2, row2 = ev.evaluate_window(model, w, controls, target, norms, MOD, True)
        self.assertEqual(row, row2)
        self.assertEqual(metrics, metrics2)
        for key in arrays: np.testing.assert_array_equal(arrays[key], arrays2[key])

    def test_nonfinite_generated_sample_never_dropped(self):
        model, norms = tiny_model(), normalizers()
        prediction = torch.zeros((6, 30, 139))
        prediction[4, 9, 0] = float('nan')
        target = np.tile(np.asarray(norms['mean'], np.float32), (30, 1))
        with patch.object(model, 'generate', return_value=prediction):
            with self.assertRaises(FloatingPointError):
                ev.evaluate_window(model, {'frames': 30, 'draw_seeds': list(range(6))},
                                   {'correct': np.zeros((30, 92), np.float32)}, target, norms, MOD)

    def test_full_chunk_and_single_root_integration(self):
        model = tiny_model()
        audio = np.random.default_rng(5).normal(size=(1007, 92)).astype(np.float32)
        target = np.tile(np.asarray(normalizers()['mean'], np.float32), (1007, 1))
        target[:, 0] = 3.
        arrays, row = ev.evaluate_full(model, audio, target, 7001, MOD, True)
        self.assertLessEqual(row['chunk_max_abs'], 5e-5)
        self.assertEqual(arrays['features'].shape, (1007, 139))
        np.testing.assert_array_equal(arrays['root'], arrays['joints'][:, 0])
        np.testing.assert_allclose(arrays['reference_root'][-1, 0], 100.6, atol=2e-5)
        # World-space slice preserves displacement; it is NOT reset to zero at a clip boundary.
        self.assertGreater(arrays['reference_root'][960, 0], 90.)
        with torch.inference_mode():
            expected = MOD['geometry'].root_from_features(torch.from_numpy(arrays['features'])).numpy()
        np.testing.assert_array_equal(arrays['root'], expected)

    def test_generated_contact_labels_not_reference(self):
        features = torch.tensor(np.tile(np.asarray(normalizers()['mean'], np.float32), (1, 60, 1)))
        features[:, :, 0] = 1.
        features[:, :, 135:] = 1.
        on, _ = MOD['music_evaluation'].physical(features)
        features[:, :, 135:] = 0.
        off, _ = MOD['music_evaluation'].physical(features)
        self.assertGreater(on['sliding'], .99)
        self.assertEqual(off['sliding'], 0.)
        self.assertEqual(on['contact_coverage'], 1.)
        self.assertEqual(off['contact_coverage'], 0.)


class OrchestrationTests(unittest.TestCase):
    def test_checkpoint_step_identity_and_normalizers_rejected_before_model_load(self):
        norms = normalizers()
        identity = {'arm': 'physical', 'seed': 0, 'protocol_sha256': 'protocol', 'source_sha256': 'source'}
        p = {'source_sha256': 'source', 'design': {'weights': {}}}
        state = {'identity': identity, 'step': 10000, 'normalizers': norms, 'audio_normalizers': {}}
        with patch.object(ev, 'digest', return_value='protocol'), patch.object(torch, 'load', return_value=state):
            with self.assertRaisesRegex(ValueError, 'Intermediate'):
                ev.load_model('physical', 'final', p, {}, MOD, 'cpu')
            state['step'] = 30000
            state['identity'] = {**identity, 'arm': 'masked_native'}
            with self.assertRaisesRegex(ValueError, 'identity'):
                ev.load_model('physical', 'final', p, {}, MOD, 'cpu')
            state['identity'] = identity
            with patch.object(ev, 'read_json', return_value={'wrong': 1}):
                with self.assertRaisesRegex(ValueError, 'normalizers'):
                    ev.load_model('physical', 'final', p, {}, MOD, 'cpu')

    def test_budget_resume_complete_report_without_selection(self):
        from contextlib import ExitStack
        model, norms = tiny_model(), normalizers()
        windows = [{'key': 'a', 'canonical_id': 'songA', 'start': start, 'frames': 30,
                    'donor_key': 'b', 'donor_start': 0, 'shift_start': 30-start,
                    'draw_seeds': [1+start+i for i in range(6)]} for start in (0, 30)]
        q = {'main_windows': windows}
        audio = np.random.default_rng(7).normal(size=(60, 92)).astype(np.float32)
        target = np.tile(np.asarray(norms['mean'], np.float32), (60, 1))
        original_read = ev.read_json
        manifest_path = ev.NATIVE/'data/audio_manifest.json'
        def reads(path):
            return {'entries': []} if Path(path) == manifest_path else original_read(path)
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            out = Path(d)
            stack.enter_context(patch.object(ev, 'completed_training', return_value={}))
            stack.enter_context(patch.object(ev, 'frozen_modules', return_value=MOD))
            stack.enter_context(patch.object(ev, 'load_model', return_value=(model, norms, {})))
            stack.enter_context(patch.object(ev, 'execution_identity', return_value={'synthetic': True}))
            stack.enter_context(patch.object(ev, 'read_json', side_effect=reads))
            stack.enter_context(patch.object(ev, 'load_val', return_value=(audio, target)))
            folder = out/'masked_native/final/primary'
            ev.stage_inference('primary', 'masked_native', 'final', 'cpu', 1, out, {}, q)
            self.assertTrue((folder/'000.json').exists())
            self.assertFalse((folder/'001.json').exists())
            self.assertFalse((folder/'report.json').exists())
            first_hash = ev.digest(folder/'000.json')
            ev.stage_inference('primary', 'masked_native', 'final', 'cpu', 1, out, {}, q)
            self.assertEqual(ev.digest(folder/'000.json'), first_hash)
            report_hash = ev.digest(folder/'report.json')
            with patch.object(model, 'generate', side_effect=AssertionError('Completed row must not regenerate')):
                ev.stage_inference('primary', 'masked_native', 'final', 'cpu', 1, out, {}, q)
            self.assertEqual(ev.digest(folder/'report.json'), report_hash)
            result = ev.load_json(folder/'report.json')
            self.assertEqual(result['expected_items'], 2)
            self.assertEqual([r['start'] for r in result['rows']], [0, 30])
            self.assertTrue(result['complete'])

    def test_nonfinite_stage_retained_and_blocks_restart(self):
        from contextlib import ExitStack
        model, norms = tiny_model(), normalizers()
        w = {'key': 'a', 'canonical_id': 's', 'start': 0, 'frames': 30,
             'donor_key': None, 'shift_start': None, 'draw_seeds': list(range(6))}
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            out = Path(d)
            original_read = ev.read_json
            def reads(path):
                return {'entries': []} if Path(path) == ev.NATIVE/'data/audio_manifest.json' else original_read(path)
            stack.enter_context(patch.object(ev, 'completed_training', return_value={}))
            stack.enter_context(patch.object(ev, 'frozen_modules', return_value=MOD))
            stack.enter_context(patch.object(ev, 'load_model', return_value=(model, norms, {})))
            stack.enter_context(patch.object(ev, 'execution_identity', return_value={'synthetic': True}))
            stack.enter_context(patch.object(ev, 'read_json', side_effect=reads))
            stack.enter_context(patch.object(ev, 'load_val', return_value=(np.zeros((30, 92), np.float32), np.zeros((30, 139), np.float32))))
            stack.enter_context(patch.object(ev, 'evaluate_window', side_effect=FloatingPointError('synthetic nonfinite draw')))
            with self.assertRaises(FloatingPointError):
                ev.stage_inference('short', 'physical', 'final', 'cpu', 1, out, {}, {'short_windows': [w]})
            failure = out/'physical/final/short/failure.json'
            self.assertTrue(failure.exists())
            self.assertEqual(ev.load_json(failure)['samples_discarded'], 0)
            with self.assertRaisesRegex(ValueError, 'Prior nonfinite failure'):
                ev.stage_inference('short', 'physical', 'final', 'cpu', 1, out, {}, {'short_windows': [w]})


if __name__ == '__main__':
    unittest.main()

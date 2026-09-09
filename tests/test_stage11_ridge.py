"""CPU-only ridge tests on temporary fixtures; never fit the project dataset."""

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from lightsem.experiments import stage11_ridge as ridge
from lightsem.experiments.stage11_common import authenticated_json, digest, json_digest, read_authenticated


class RidgeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(threadpool_limits(limits=2))
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))

    def fixture(self):
        norm = {'x_mean': [.5] * 92, 'x_std': [2.] * 92,
                'y_center': [1., 2., 3., 4.] + [0.] * 17,
                'y_scale': [2., 3., 4., 5.] + [1.] * 17,
                'a_names': list(ridge.A_NAMES), 'fit_population': 'old_training_only'}
        authenticated_json(self.root/'normalizers.json', {'motion': norm})
        entries = []
        for index, (song, recording, old, partition) in enumerate([
            ('old-song', 'recording-a', True, 'train'),
            ('old-song', 'recording-a', True, 'train'),
            ('old-song', 'recording-b', True, 'train'),
            ('new-song', 'recording-c', False, 'train'),
            ('val-song-a', 'recording-v-a', False, 'val'),
            ('val-song-a', 'recording-v-a', False, 'val'),
            ('val-song-b', 'recording-v-b', False, 'val'),
        ]):
            rng = np.random.default_rng(index)
            features = rng.normal(size=(401, 92))
            x = (.5 + 2 * features).astype(np.float32)
            y = np.zeros((401, 21), np.float32)
            y[:, :4] = (features[:, :4] * .3 + .1) * norm['y_scale'][:4] + norm['y_center'][:4]
            path = self.root/f'coach{index}.npz'
            np.savez(path, x=x, y=y)
            entries.append({'key': f'coach{index}', 'canonical_id': song, 'recording_id': recording,
                            'partition': partition, 'is_old': old, 'path': str(path),
                            'n_frames': len(x), 'sha256': digest(path)})
        # A nonexistent test file makes any accidental hashing/open fail.
        entries.append({'key': 'forbidden-test', 'canonical_id': 'test-song', 'recording_id': 'test-recording',
                        'partition': 'test', 'is_old': False, 'path': str(self.root/'DO_NOT_OPEN_TEST.npz'),
                        'n_frames': 401, 'sha256': 'forbidden'})
        data = {'motion': entries, 'normalizers_sha256': digest(self.root/'normalizers.json')}
        authenticated_json(self.root/'training_data.json', data)
        protocol = {'frozen': True, 'training_seeds': list(range(5)),
                    'data_sha256': digest(self.root/'training_data.json'),
                    'normalizers_sha256': digest(self.root/'normalizers.json')}
        authenticated_json(self.root/'protocol.json', protocol)
        source = self.root/'source'
        source.mkdir()
        (source/'baseline_stub.py').write_text('# frozen source fixture\n')
        files = {'baseline_stub.py': digest(source/'baseline_stub.py')}
        authenticated_json(self.root/'frozen_sources.json', {
            'snapshot': str(source), 'files': files, 'source_sha256': json_digest(files),
        })
        return data, norm

    def test_hierarchy_balances_songs_recordings_and_coaches(self):
        entries = [{'key': 'a1', 'canonical_id': 'a', 'recording_id': 'a-rec1'},
                   {'key': 'a2', 'canonical_id': 'a', 'recording_id': 'a-rec1'},
                   {'key': 'a3', 'canonical_id': 'a', 'recording_id': 'a-rec2'},
                   {'key': 'b1', 'canonical_id': 'b', 'recording_id': 'b-rec1'}]
        weights, summary = ridge.hierarchy_weights(entries)
        self.assertEqual(weights, {'a1': .25, 'a2': .25, 'a3': .5, 'b1': 1.})
        self.assertEqual(summary['canonical_songs'], 2)
        self.assertEqual(sum(weights.values()), 2)

    def test_intercept_is_not_regularized(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(150, 92)) + 2
        y = np.tile([2., -3., 5., 7.], (len(x), 1))
        stats = ridge.empty_statistics()
        ridge.accumulate(stats, x, y, 1.0)
        coefficients, alpha = ridge.solve_ridge(stats, 1)
        self.assertEqual(alpha, .001)
        np.testing.assert_allclose(coefficients[:92], 0, atol=1e-9)
        np.testing.assert_allclose(coefficients[92], y[0], atol=1e-8)

    def test_regularization_strength_is_invariant_to_duplicate_song_population(self):
        rng = np.random.default_rng(19)
        x, y = rng.normal(size=(130, 92)), rng.normal(size=(130, 4))
        one, two = ridge.empty_statistics(), ridge.empty_statistics()
        ridge.accumulate(one, x, y, 1.)
        ridge.accumulate(two, x, y, 1.)
        ridge.accumulate(two, x, y, 1.)
        a, alpha_a = ridge.solve_ridge(one, 1)
        b, alpha_b = ridge.solve_ridge(two, 2)
        self.assertEqual(alpha_b, 2 * alpha_a)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)

    def test_sample_count_cannot_overweight_a_coach(self):
        rng = np.random.default_rng(29)
        x, y = rng.normal(size=(120, 92)), rng.normal(size=(120, 4))
        a, b = ridge.empty_statistics(), ridge.empty_statistics()
        ridge.accumulate(a, x, y, 1.)
        ridge.accumulate(b, np.repeat(x, 3, axis=0), np.repeat(y, 3, axis=0), 1.)
        np.testing.assert_allclose(a['xtx'], b['xtx'], atol=1e-12)
        np.testing.assert_allclose(a['xty'], b['xty'], atol=1e-12)

    def test_full_fixture_freezes_before_first_read_and_never_opens_test(self):
        data, _ = self.fixture()
        opened = []
        actual_load = np.load
        actual_metric = ridge.motion_r2
        metric_frames = []

        def load(path, *args, **kwargs):
            path = Path(path)
            self.assertNotIn('DO_NOT_OPEN_TEST', path.name)
            recipe = read_authenticated(self.root/'results/cpu_ridge_protocol.json')
            self.assertEqual(recipe['normalized_regularization'], .001)
            self.assertFalse(recipe['test_tensors_permitted'])
            opened.append(str(path))
            return actual_load(path, *args, **kwargs)

        def metric(reference, prediction, names, **kwargs):
            metric_frames.append((len(reference), kwargs['fps'], kwargs['tau']))
            return actual_metric(reference, prediction, names, **kwargs)

        with patch.object(ridge.np, 'load', side_effect=load), patch.object(ridge, 'motion_r2', side_effect=metric), patch('builtins.print'):
            result = ridge.run_ridge(self.root)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['fit']['old']['canonical_songs'], 1)
        self.assertEqual(result['fit']['expanded']['canonical_songs'], 2)
        self.assertEqual(result['fit']['old']['effective_alpha'], .001)
        self.assertEqual(result['fit']['expanded']['effective_alpha'], .002)
        self.assertEqual(result['fit']['old']['sampled_frames'], 3 * 11)
        self.assertEqual(result['fit']['expanded']['sampled_frames'], 4 * 11)
        self.assertEqual(metric_frames, [(401, 40., 2.)] * 6)
        self.assertEqual(len(opened), 7)
        self.assertEqual(result['n_validation_coaches'], 3)
        self.assertEqual(result['n_validation_recordings'], 2)
        self.assertEqual(result['n_validation_canonical_songs'], 2)
        self.assertEqual(result['test_tensors_read'], 0)
        self.assertFalse(result['can_override_neural_gates'])
        self.assertFalse(result['test_unlock_allowed'])
        for entry in data['motion']:
            self.assertEqual(entry['path'] in result['inputs'], entry['partition'] != 'test')
        self.assertFalse((self.root/'test_unlock.json').exists())
        with actual_load(self.root/'results/cpu_ridge_coefficients.npz', allow_pickle=False) as coefficients:
            self.assertEqual(coefficients['old'].shape, (93, 4))
            self.assertEqual(coefficients['expanded'].shape, (93, 4))

    def test_completed_output_is_reused_without_refit_and_input_drift_is_rejected(self):
        data, _ = self.fixture()
        with patch('builtins.print'):
            first = ridge.run_ridge(self.root)
        with patch.object(ridge.np, 'load', side_effect=AssertionError('must not refit')):
            second = ridge.run_ridge(self.root)
        self.assertEqual(first, second)
        Path(data['motion'][0]['path']).write_bytes(b'tampered input')
        with self.assertRaisesRegex(RuntimeError, 'Artifact drift'):
            ridge.run_ridge(self.root)

    def test_frozen_recipe_cannot_be_changed_for_another_hyperparameter(self):
        self.fixture()
        with patch('builtins.print'):
            ridge.run_ridge(self.root)
        with patch.object(ridge, 'NORMALIZED_RIDGE', .01):
            with self.assertRaisesRegex(RuntimeError, 'incompatible CPU ridge protocol'):
                ridge.run_ridge(self.root)

    def test_loader_refuses_test_before_hash_or_open(self):
        with patch.object(ridge, 'digest', side_effect=AssertionError('must not hash test')):
            with self.assertRaisesRegex(RuntimeError, 'never open a test tensor'):
                ridge._load_allowed_tensor({'partition': 'test', 'path': 'forbidden'}, {})

    def test_metadata_overlap_and_non_old_calibration_are_rejected(self):
        data, norm = self.fixture()
        bad = copy.deepcopy(data)
        bad['motion'][-1]['canonical_id'] = bad['motion'][0]['canonical_id']
        with self.assertRaisesRegex(ValueError, 'split overlap'):
            ridge._populations(bad)
        norm['fit_population'] = 'all_songs'
        with self.assertRaisesRegex(ValueError, 'old-training normalization'):
            ridge._normalization({'motion': norm})

    def test_validation_summary_weights_recordings_before_songs(self):
        def row(song, recording, old, expanded):
            metrics = lambda value: {'primary_r2': value, 'r2_by_channel': dict.fromkeys(ridge.A_NAMES, value)}
            return {'canonical_id': song, 'recording_id': recording, 'old': metrics(old), 'expanded': metrics(expanded)}
        rows = [row('a', 'a1', 0, .2), row('a', 'a1', 0, .2), row('a', 'a2', 1, 1.2), row('b', 'b1', 1, 1.4)]
        recordings, songs, total = ridge.summarize_validation(rows)
        self.assertEqual(len(recordings), 3)
        self.assertEqual(len(songs), 2)
        self.assertAlmostEqual(total['old']['primary_r2'], .75)
        self.assertAlmostEqual(total['expanded']['primary_r2'], 1.05)
        self.assertAlmostEqual(total['delta']['primary_r2'], .3)


if __name__ == '__main__':
    unittest.main()

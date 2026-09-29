"""CPU-only contract tests for dance-supervised no-clock feature preparation."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from lightsem.experiments.dance_no_clock import dance_no_clock_data as data


class NoClockDataTests(unittest.TestCase):
    def test_exact_feature_and_target_selection(self):
        tree = ast.parse(Path('/root/lightsem/lightsem/model/light_data.py').read_text())
        names = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'AUDIO_SCALARS' for target in node.targets))
        self.assertEqual(data.SCALARS, names[:22])
        self.assertEqual(len(data.FEATURE_NAMES), 86)
        self.assertEqual(len(data.TARGET_NAMES), 20)
        self.assertNotIn('metric_level', data.TARGET_NAMES)
        self.assertFalse(any('beat' in name or 'pulse' in name or 'bar_phase' in name for name in data.FEATURE_NAMES))

    def test_filter_grid_constant_and_alias_suppression(self):
        x = np.full((401, 2), [2., -3.], np.float32)
        y = data.downsample_10hz(x)
        self.assertEqual(y.shape, (101, 2))
        np.testing.assert_allclose(y, np.broadcast_to([2., -3.], y.shape), atol=1e-6)
        t = np.arange(800)/40
        waves = np.stack([np.sin(2*np.pi*t), np.sin(2*np.pi*8*t)], 1)
        result = data.downsample_10hz(waves)[20:-20]
        self.assertGreater(float(np.sqrt(np.mean(result[:, 0]**2))), .70)
        self.assertLess(float(np.sqrt(np.mean(result[:, 1]**2))), 1e-3)

    def test_inference_never_calls_legacy_clock_extractors(self):
        import librosa
        from lightsem.audio import online, beatgrid
        t = np.arange(44100*3)/44100
        y = (.2*np.sin(2*np.pi*220*t)+.03*np.sin(2*np.pi*3700*t)).astype(np.float32)
        forbidden = AssertionError('Clock/legacy bus extraction must never be invoked')
        with (patch.object(online, 'extract_bus', side_effect=forbidden) as bus,
              patch.object(online, 'fit_constant_grid', side_effect=forbidden) as online_grid,
              patch.object(beatgrid, 'fit_constant_grid', side_effect=forbidden) as grid,
              patch.object(online, 'trailing_pulse_clarity', side_effect=forbidden) as pulse,
              patch.object(librosa.beat, 'beat_track', side_effect=forbidden) as beats):
            high = data.extract_nonclock_audio(y=y, output_fps=40.)
            low = data.extract_nonclock_audio(y=y, output_fps=10.)
        for mock in (bus, online_grid, grid, pulse, beats):
            mock.assert_not_called()
        self.assertEqual(high['x'].shape, (120, 86))
        self.assertEqual(low['x'].shape, (30, 86))
        np.testing.assert_array_equal(low['x'], data.downsample_10hz(high['x']))
        np.testing.assert_array_equal(low['times'], np.arange(30)/10)

    def test_equal_song_recording_coach_weights(self):
        entries = [{'key': key, 'canonical_id': song, 'recording_id': rec, 'partition': 'train'}
                   for key, song, rec in [('a1c1', 'a', 'a1'), ('a1c2', 'a', 'a1'), ('a2c1', 'a', 'a2'), ('b1c1', 'b', 'b1')]]
        weights = data.hierarchical_weights(entries)
        self.assertEqual(weights, {'a1c1': .125, 'a1c2': .125, 'a2c1': .25, 'b1c1': .5})
        with self.assertRaises(ValueError):
            data.hierarchical_weights([{**entries[0], 'partition': 'val'}])

    def test_test_partition_rejected_before_any_source_read(self):
        with patch.object(data, 'digest', side_effect=AssertionError('TEST tensor was accessed')):
            with self.assertRaisesRegex(ValueError, 'TRAIN/VAL'):
                data._prepare_entry(({'partition': 'test', 'path': '/forbidden_test.npz'}, Path('/unused')))

    def test_weighted_iqr_has_declared_inverse_cdf_and_floor(self):
        values = np.array([[0., 4.], [1., 4.], [10., 4.]])
        result = data.weighted_iqr(values, [.1, .8, .1])
        self.assertEqual(result, [.001, .001])


if __name__ == '__main__':
    unittest.main()

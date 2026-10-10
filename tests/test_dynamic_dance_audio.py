"""Meaningful CPU checks for timestamp, provenance and waveform intervention contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from lightsem.experiments.dynamic_dance.contracts import DancePlan, DanceResult, MusicTimeline
from lightsem.experiments.dynamic_dance.audio import (
    extract_base92, extract_muq_stream, integrated_loudness, model_features, physical_observations,
    _stitch_muq_chunks,
    mixture_observations, preselect_intervals, section_recurrence, source_attack_events,
    waveform_counterfactual,
)


def timeline(duration=10.):
    t = np.arange(round(duration * 30)) / 30
    u = np.arange(round(duration * 25)) / 25
    muq = np.broadcast_to((1 + u[:, None] / duration), (len(u), 1024)).astype(np.float32).copy()
    return MusicTimeline(t, muq, u, np.zeros((len(t), 4, 4), np.float32),
                         np.zeros((len(t), 4), np.float32), np.arange(1., duration),
                         np.arange(1., duration, 4),
                         [{'start': 0., 'end': duration, 'label': 'verse', 'confidence': None}],
                         np.ones((1, 1), np.float32), {'duration_s': duration})


class ContractTests(unittest.TestCase):
    def test_roundtrip_and_tamper_detection(self):
        music = timeline()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'music.npz'
            music.save(path)
            loaded = MusicTimeline.load(path)
            np.testing.assert_array_equal(music.muq_time, loaded.muq_time)
            self.assertIsNone(loaded.sections[0]['confidence'])
            with self.assertRaises(FileExistsError):
                music.save(path)
            meta = path.with_suffix('.json')
            meta.write_text(meta.read_text().replace('verse', 'chorus'))
            with self.assertRaisesRegex(ValueError, 'metadata changed'):
                MusicTimeline.load(path)

    def test_reject_nonfinite_and_ambiguous_timestamps(self):
        music = timeline()
        music.muq_time[1] = music.muq_time[0]
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            music.validate()
        music = timeline()
        music.sources[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            music.validate()

    def test_body_contracts_allow_bool_contacts_and_disclosed_recomposition(self):
        t = np.arange(4) / 30
        result = DanceResult(t, np.zeros((4, 22, 3)), np.zeros((4, 4), bool), source_time=np.array([0., 1., .5, 2.]))
        # D4 fragment reordering is explicitly allowed; retiming-specific monotonic
        # constraints belong to the selected method, not a misleading global check.
        result.validate()
        plan = DancePlan(t, np.arange(4) / 30, np.ones((4, 3)), metadata={'rate_bound': 1.5})
        with tempfile.TemporaryDirectory() as d:
            result.save(Path(d) / 'body.npz')
            plan.save(Path(d) / 'plan.npz')
            np.testing.assert_array_equal(DanceResult.load(Path(d) / 'body.npz').source_time, result.source_time)
            self.assertEqual(DancePlan.load(Path(d) / 'plan.npz').metadata['rate_bound'], 1.5)

    def test_model_feature_times_not_frame_indices(self):
        music = timeline()
        target = np.array([.1, 1.25, 3.33])
        x, names = model_features(music, target)
        self.assertEqual(x.shape, (3, 1061))
        np.testing.assert_allclose(x[:, 0], 1 + target / 10, atol=1e-6)
        np.testing.assert_allclose(x[:, names.index('beat_previous_s')], [8., .25, .33], atol=1e-6)
        np.testing.assert_allclose(x[:, names.index('section_verse')], 1)

    def test_declared_pure_retiming_rejects_backwards_excess_speed_and_endpoints(self):
        t = np.arange(5) / 30
        metadata = {'method': 'D1', 'source_order_preserved': True}
        bad_clocks = [np.array([0, .05, .01, .1, t[-1]]),
                      np.array([0, .07, .09, .11, t[-1]]),
                      t + .01, t * .9]
        for clock in bad_clocks:
            with self.subTest(clock=clock.tolist()):
                with self.assertRaisesRegex(ValueError, 'Pure retiming'):
                    DancePlan(t, clock, np.ones((5, 1)), metadata=metadata).validate()
                with self.assertRaisesRegex(ValueError, 'Pure retiming'):
                    DanceResult(t, np.zeros((5, 22, 3)), np.zeros((5, 4)), source_time=clock, metadata=metadata).validate()
        regional = np.repeat(t[:, None], 5, axis=1)
        DancePlan(t, regional, np.ones((5, 5)), metadata=metadata).validate()
        regional[2, 4] = .12
        with self.assertRaisesRegex(ValueError, 'Pure retiming'):
            DancePlan(t, regional, np.ones((5, 5)), metadata=metadata).validate()

    def test_duration_bounds_catch_wrong_song_and_incomplete_muq(self):
        for field in ('beats', 'downbeats', 'muq_time', 'time'):
            music = timeline()
            value = getattr(music, field).copy()
            value[-1] = 100.
            setattr(music, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'duration'):
                music.validate()
        music = timeline()
        music.sections[0]['end'] = 100
        with self.assertRaisesRegex(ValueError, 'duration'):
            music.validate()
        music = timeline()
        music.muq_time = music.muq_time[:100]
        music.muq = music.muq[:100]
        with self.assertRaisesRegex(ValueError, 'cover'):
            music.validate()


class PhysicalTests(unittest.TestCase):
    def test_silence_is_zero_and_gain_is_not_semantic_change(self):
        sr = 22050
        t = np.arange(2 * 30) / 30
        quiet = physical_observations(np.zeros(sr * 2), sr, t)
        np.testing.assert_array_equal(quiet, 0)
        sample = np.arange(sr * 2) / sr
        wave = .1 * np.sin(2 * np.pi * 440 * sample)
        a = mixture_observations(wave, sr, t)
        b = mixture_observations(wave * .5, sr, t)
        np.testing.assert_allclose(b[:, [0, 1, 3]], a[:, [0, 1, 3]] * .5, atol=1e-7)
        np.testing.assert_allclose(b[:, 2], a[:, 2], atol=1e-4)

    def test_real_audio_impulse_produces_localized_attack(self):
        sr = 22050
        wave = np.zeros(sr * 3)
        wave[sr:sr + 500] = .5
        time_grid = np.arange(90) / 30
        obs = physical_observations(wave, sr, time_grid)
        peak_time = time_grid[np.argmax(obs[:, 1])]
        self.assertLess(abs(peak_time - 1), .06)
        self.assertEqual(float(obs[:20, 1].sum()), 0.)

    def test_waveform_interventions_change_waveform_before_models(self):
        mixture = np.arange(40, dtype=np.float32).reshape(2, 20)
        shifted, metadata = waveform_counterfactual(mixture, 10, {'kind': 'time_shift', 'seconds': .3})
        np.testing.assert_array_equal(shifted[:, :3], 0)
        np.testing.assert_array_equal(shifted[:, 3:], mixture[:, :-3])
        self.assertIn('AllInOne', metadata['recompute_required'])
        stems = np.stack([mixture * x for x in (.1, .2, .3, .4)])
        removed, _ = waveform_counterfactual(mixture, 10, {'kind': 'source_removal', 'source': 'bass', 'loudness_match': False}, stems)
        np.testing.assert_allclose(removed, stems[[0, 2, 3]].sum(0))
        reconstructed, _ = waveform_counterfactual(mixture, 10, {'kind': 'source_reconstruction', 'loudness_match': False}, stems)
        np.testing.assert_allclose(reconstructed, stems.sum(0))
        gain, _ = waveform_counterfactual(mixture, 10, {'kind': 'gain', 'db': -6})
        np.testing.assert_allclose(gain, mixture * 10 ** (-6 / 20))

    def test_loudness_matched_removal_preserves_lufs(self):
        sr = 8000
        t = np.arange(sr * 2) / sr
        sources = np.stack([np.stack([.05 * np.sin(2 * np.pi * f * t)] * 2) for f in (100, 200, 500, 900)]).astype(np.float32)
        mixture = sources.sum(0)
        changed, meta = waveform_counterfactual(mixture, sr, {'kind': 'source_removal', 'source': 'vocals'}, sources)
        self.assertLess(abs(integrated_loudness(mixture, sr) - integrated_loudness(changed, sr)), .15)
        self.assertFalse(np.array_equal(mixture, changed))
        self.assertIn('BS.1770', meta['match_method'])

    def test_no_attacks_fabricated_and_audio_windows_are_deterministic(self):
        music = timeline(190)
        self.assertEqual(source_attack_events(music), [])
        first, second = preselect_intervals(music), preselect_intervals(music)
        self.assertEqual(first, second)
        self.assertTrue(all(60 <= r['start'] < r['end'] <= 180 for r in first))

    def test_base92_waveform_binding_and_explicit_resampling(self):
        import hashlib
        from types import SimpleNamespace
        from lightsem.model.light_data import AUDIO_SCALARS
        wave = np.arange(2 * 44100 * 8, dtype=np.float32).reshape(2, -1) * 1e-8
        t40 = np.arange(320) / 40
        t30 = np.arange(240) / 30
        digest = hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()
        bus = SimpleNamespace(mel=np.ones((64, 320)), times=t40,
                              channels={name: np.ones(320) for name in AUDIO_SCALARS},
                              flags={'oracle_beat': True})
        with tempfile.TemporaryDirectory() as d, patch('lightsem.audio.online.extract_bus', return_value=bus) as extract, patch('lightsem.experiments.dynamic_dance.audio.guard_disk'):
            entry = {'key': 'shifted', 'wav_path': 'original.wav', 'wav_sha256': 'original_identity'}
            receipt = extract_base92(wave, t30, entry, d, digest, {'kind': 'time_shift', 'seconds': 1})
            np.testing.assert_array_equal(extract.call_args.kwargs['y'], wave.mean(axis=0))
            self.assertEqual(receipt['analyzed_waveform_sha256'], digest)
            with np.load(receipt['path']) as arrays:
                self.assertEqual(arrays['x'].shape, (320, 92))
                self.assertEqual(arrays['x30'].shape, (240, 92))
                np.testing.assert_array_equal(arrays['time30'], t30)
            with self.assertRaisesRegex(ValueError, 'another analyzed waveform'):
                extract_base92(wave, t30, entry, d, 'different_hash', {'kind': 'time_shift', 'seconds': 1})

    def test_counterfactual_pipeline_binds_processed_provenance_in_both_inputs(self):
        """Actual waveform enrichment must reach both serialized model inputs.

        Frozen neural models are replaced by CPU fixtures; waveform intervention,
        loudness measurement, physical observations, Base92 serialization and the
        downstream training binding validator execute their production paths.
        """
        from contextlib import ExitStack
        from types import SimpleNamespace
        import soundfile as sf
        from lightsem.model.light_data import AUDIO_SCALARS
        from lightsem.experiments.dynamic_dance.audio import AudioAnalyzer
        from lightsem.experiments.dynamic_dance.contracts import sha256
        from lightsem.experiments.dynamic_dance.training import validate_audio_binding

        sr = 44100
        sample_time = np.arange(sr * 8) / sr
        stems = np.stack([np.stack([.02 * np.sin(2 * np.pi * f * sample_time)] * 2)
                          for f in (110, 220, 440, 880)]).astype(np.float32)
        original = stems.sum(0)
        requested = {'kind': 'source_removal', 'source': 'vocals', 'loudness_match': True}
        t40 = np.arange(320) / 40
        bus = SimpleNamespace(mel=np.ones((64, 320)), times=t40,
                              channels={name: np.ones(320) for name in AUDIO_SCALARS},
                              flags={'oracle_beat': True})
        muq_time = np.arange(200) / 25
        structure = {'sections': [{'start': 0., 'end': 8., 'label': 'verse', 'confidence': None}]}
        captured_waves = []
        def separator(wave):
            captured_waves.append(wave.copy())
            return stems if len(captured_waves) == 1 else np.stack([wave * .25] * 4)

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            audio_path = root / 'source.wav'
            sf.write(audio_path, original.T, sr, subtype='FLOAT')
            entry = {'key': 'removed_vocals', 'partition': 'counterfactual',
                     'wav_path': str(audio_path), 'wav_sha256': sha256(audio_path),
                     'timeline_path': str(root / 'features/timelines/removed_vocals.npz')}
            stack.enter_context(patch('lightsem.experiments.dynamic_dance.audio.guard_disk'))
            base_extract = stack.enter_context(patch('lightsem.audio.online.extract_bus', return_value=bus))
            stack.enter_context(patch.object(AudioAnalyzer, 'separate', side_effect=separator))
            stack.enter_context(patch.object(AudioAnalyzer, 'muq', return_value=SimpleNamespace(metadata={'fixture': True})))
            stack.enter_context(patch('lightsem.experiments.dynamic_dance.audio.extract_muq_stream',
                                      return_value=(np.ones((200, 1024), np.float32), muq_time)))
            stack.enter_context(patch.object(AudioAnalyzer, 'beats',
                                      return_value=(np.arange(1., 8.), np.array([1., 5.]), {'fixture': True})))
            structure_extract = stack.enter_context(patch('lightsem.experiments.dynamic_dance.structure.extract_from_stems', return_value=structure))
            receipt = AudioAnalyzer(device='cpu', structure_device='cpu').extract(entry, root, requested)
            music = MusicTimeline.load(entry['timeline_path'])
            paired = validate_audio_binding(music, receipt['base92']['path'])
            processed = music.metadata['intervention']
            self.assertEqual(paired['intervention'], processed)
            self.assertEqual(music.metadata['base92']['intervention'], processed)
            self.assertIn('loudness_before_lufs', processed)
            self.assertIn('removal_before_match_lufs', processed)
            self.assertIn('recompute_required', processed)
            self.assertEqual(receipt['intervention'], requested)
            self.assertNotEqual(processed, requested)
            self.assertEqual(len(captured_waves), 2)
            self.assertFalse(np.array_equal(captured_waves[0], captured_waves[1]))
            np.testing.assert_array_equal(base_extract.call_args.kwargs['y'], captured_waves[1].mean(0))
            self.assertEqual(structure_extract.call_args.kwargs['stem_provenance']['counterfactual'], processed)
            self.assertEqual(paired['analyzed_waveform_sha256'], music.metadata['analyzed_waveform_sha256'])

    def test_recurrence_respects_music_content(self):
        z = np.zeros((4, 1024))
        z[:2, 0] = 1
        z[2:, 1] = 1
        sections = [{'start': 0, 'end': 2}, {'start': 2, 'end': 4}]
        np.testing.assert_allclose(section_recurrence(z, np.arange(4), sections), np.eye(2))

    def test_muq_last_real_chunk_monotonic_no_duplicate_timestamps(self):
        class FakeExtractor:
            def __init__(self):
                self.seen = []
            def extract_batch(self, waves):
                self.seen.append([wave.copy() for wave in waves])
                return [(np.zeros((200, 1024), np.float32), np.arange(200) / 25)
                        for _ in waves]
        extractor = FakeExtractor()
        wave = np.arange(17 * 24000 + 100, dtype=np.float32)
        z, t = extract_muq_stream(extractor, wave)
        self.assertEqual(len(extractor.seen), 1)
        self.assertEqual(len(extractor.seen[0]), 3)
        np.testing.assert_array_equal(extractor.seen[0][-1], wave[-8 * 24000:])
        self.assertTrue(np.all(np.diff(t) > 0))
        self.assertLess(wave.size / 24000 - t[-1], .041)
        self.assertEqual(z.shape[1], 1024)

    def test_muq_overlap_stitch_averages_real_overlap_without_gaps(self):
        t = np.arange(200) / 25
        first = np.zeros((200, 1024), np.float32)
        second = np.ones((200, 1024), np.float32)
        value, time = _stitch_muq_chunks([(first, t), (second, t + 4.)])
        self.assertTrue(np.all(np.diff(time) > 0))
        overlap = (time >= 4.) & (time < 8.)
        np.testing.assert_allclose(value[overlap], .5, atol=1e-6)

    def test_muq_stitch_keeps_fractional_final_duration_endpoint(self):
        t = np.arange(200) / 25
        value, time = _stitch_muq_chunks([(np.zeros((200, 1024), np.float32), t)], target_end=8.)
        self.assertAlmostEqual(float(time[-1]), 8.)
        self.assertEqual(value.shape[0], 201)


if __name__ == '__main__':
    unittest.main()

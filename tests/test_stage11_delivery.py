"""CPU checks for delivery's frozen checkpoint and per-arm provenance boundary."""

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lightsem.experiments.stage11 import stage11_delivery as delivery
from lightsem.experiments.stage11.stage11_common import authenticated_json, artifact_manifest, digest, json_digest, read_authenticated
from lightsem.model.light_trunk import LightTrunk, TrunkConfig


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.protocol = {'frozen': True, 'training_seeds': [0, 1, 2, 3, 4], 'extension_seeds': [5],
                         'arms': {'D_full': {'family': 'motion', 'steps': 10}}}
        self.binding = dict.fromkeys(('protocol_sha256', 'data_sha256', 'normalizers_sha256', 'source_sha256'), 'fixed-binding')
        self.normalizers = {'motion': {'x_mean': [0.] * 92, 'x_std': [1.] * 92,
                                      'a_names': ['a' + str(i) for i in range(4)], 'c_names': ['c' + str(i) for i in range(17)],
                                      'y_center': [i * .05 for i in range(21)], 'y_scale': [.1 + i * .03 for i in range(21)]},
                            'scene_metrics': {'training_only': True}}
        self.model = LightTrunk(TrunkConfig(d_model=16, n_blocks=1, n_a=4, n_c=17))
        self.identity = {**self.binding, 'arm': 'D_full', 'seed': 0, 'config': self.protocol['arms']['D_full']}
        self.blob = {'identity': self.identity, 'step': 10, 'family': 'motion',
                     'normalizers': self.normalizers['motion'], 'metric_scales': self.normalizers['scene_metrics'],
                     'cfg': self.model.cfg.__dict__, 'model': self.model.state_dict()}
        self.run = self.root / 'runs' / 'D_full' / 'seed_0'
        self.run.mkdir(parents=True)
        self.write_training()
        self.stack.enter_context(patch.object(delivery, 'load_run_model', side_effect=lambda *args:
                                              (self.model, torch.load(self.run / 'best.pt', map_location='cpu', weights_only=False))))
        reference = self.root / 'reference.pt'
        reference.write_bytes(b'pinned reference postprocessing')
        self.assets = {'reference_checkpoint': str(reference), 'reference_sha256': digest(reference),
                       'event_thr': {'footfall': .6, 'jump': .7, 'hit': .8}, 'smooth_tau': {'c0': .3}}
        authenticated_json(self.root / 'render_assets.json', self.assets)
        self.protocol['render_assets_sha256'] = digest(self.root / 'render_assets.json')

    def write_training(self, blob=None):
        torch.save(self.blob if blob is None else blob, self.run / 'best.pt')
        report = {'status': 'completed', 'identity': self.identity, 'steps': 10, 'best_step': 10,
                  'nonfinite': False, 'oom': False, 'artifacts': artifact_manifest([self.run / 'best.pt'])}
        authenticated_json(self.run / 'report.json', report)
        return report

    def load_verified(self, **kwargs):
        return delivery._load_verified_render_model(self.root, 'D_full', 0, self.protocol,
                                                    self.normalizers, self.binding, **kwargs)

    def test_verified_checkpoint_records_exact_model_and_training_provenance(self):
        _, _, provenance = self.load_verified()
        self.assertEqual(provenance['arm'], 'D_full')
        self.assertEqual(provenance['training_seed'], 0)
        self.assertEqual(provenance['checkpoint_sha256'], digest(self.run / 'best.pt'))
        self.assertEqual(provenance['training_report_sha256'], digest(self.run / 'report.json'))
        self.assertEqual(provenance['normalizers_sha256'], json_digest(self.normalizers['motion']))

    def test_replaced_checkpoint_is_rejected_before_loading_weights(self):
        (self.run / 'best.pt').write_bytes(b'replaced')
        with patch.object(delivery, 'load_run_model') as load:
            with self.assertRaisesRegex(RuntimeError, 'Artifact drift'):
                self.load_verified()
            load.assert_not_called()

    def test_checkpoint_semantics_must_match_report_and_frozen_normalization(self):
        changes = [
            ('family', 'scene', 'family differs'),
            ('step', 9, 'selected step'),
            ('normalizers', {'wrong': True}, 'normalization differs'),
            ('metric_scales', {'wrong': True}, 'metric calibration differs'),
            ('cfg', {'wrong': True}, 'architecture differs'),
            ('identity', {'wrong': True}, 'identity'),
        ]
        for key, value, error in changes:
            with self.subTest(key=key):
                blob = copy.deepcopy(self.blob)
                blob[key] = value
                self.write_training(blob)
                with self.assertRaisesRegex(RuntimeError, error):
                    self.load_verified()

    def test_test_model_requires_exact_selected_family_seed_and_checkpoint(self):
        unlock = {'eligible_families': ['motion'], 'arms': {'D_full': {
            'training_seeds': [0], 'checkpoints': {'0': digest(self.run / 'best.pt')}}}}
        self.load_verified(unlock=unlock)
        unlock['arms']['D_full']['checkpoints']['0'] = 'wrong'
        with self.assertRaisesRegex(RuntimeError, 'Unreviewed checkpoint'):
            self.load_verified(unlock=unlock)
        unlock['arms']['D_full']['training_seeds'] = [1]
        with self.assertRaisesRegex(RuntimeError, 'family/arm/seed'):
            self.load_verified(unlock=unlock)

    def test_asset_hash_is_bound_to_protocol_even_when_sidecar_is_valid(self):
        context = (self.protocol, {}, self.normalizers, self.binding)
        changed = {**self.assets, 'event_thr': {'footfall': .01}}
        authenticated_json(self.root / 'render_assets.json', changed)
        with patch('lightsem.experiments.stage11.stage11_evaluation._context', return_value=context):
            with self.assertRaisesRegex(RuntimeError, 'postprocessing binding changed'):
                delivery._render_context(self.root, 'val')

    def test_strict_test_unlock_runs_before_manifest_or_array_access(self):
        context = (self.protocol, {}, self.normalizers, self.binding)
        with patch('lightsem.experiments.stage11.stage11_evaluation._context', return_value=context), \
             patch('lightsem.experiments.stage11.stage11_evaluation._test_unlock', side_effect=RuntimeError('strict unlock denied')) as unlock, \
             patch.object(delivery, 'verify_training_inputs') as verify_inputs, \
             patch('lightsem.experiments.stage11.stage11_data.load_manifest') as manifest:
            with self.assertRaisesRegex(RuntimeError, 'strict unlock denied'):
                delivery.render_campaign(self.root, 'test')
            unlock.assert_called_once_with(self.root, self.protocol, self.binding)
            verify_inputs.assert_not_called()
            manifest.assert_not_called()

    def test_motion_adapter_inverts_checkpoint_scale_and_preserves_pinned_postprocessing(self):
        adapter = delivery.motion_adapter(self.model, self.blob, self.assets, 'cpu')
        x = np.array([-.5, 0, 1], dtype=np.float32)
        names = self.normalizers['motion']['a_names'] + self.normalizers['motion']['c_names']
        for i, name in enumerate(names):
            expected = x * self.normalizers['motion']['y_scale'][i] + self.normalizers['motion']['y_center'][i]
            np.testing.assert_array_equal(adapter._unscale(name, x), expected)
        self.assertEqual(adapter.event_thr, self.assets['event_thr'])
        self.assertEqual(adapter.smooth_tau, self.assets['smooth_tau'])

    def test_motion_only_delivery_preserves_102_seed_and_hashes_source_candidates(self):
        # Exercise the entire delivery orchestration with CPU renderer/model
        # doubles. Every operation that could allocate CUDA or EGL is replaced.
        from lightsem.experiments.stage11 import stage11_render
        visual = self.root / 'visual.npz'
        np.savez(visual, x=np.zeros((400, 52), np.float32))
        audio = self.root / 'audio.wav'
        audio.write_bytes(b'audio')
        manifest = {'recordings': [{'partition': 'test', 'eligible_core': True, 'canonical_id': f'song{i}',
                                   'recording_id': f'video{i}', 'audio_path': str(audio), 'wav_path': str(audio),
                                   'visual_path': str(visual), 'active_start_s': 0.0} for i in range(20)]}
        bus = SimpleNamespace(times=np.arange(1600) / 40, beats=np.arange(0, 40, .5),
                              channels={'bar_phase': np.mod(np.arange(1600) / 80, 1)})
        context = (self.protocol, {}, self.normalizers, self.binding, self.assets, {'eligible_families': ['motion']}, ['motion'])

        class Model:
            def to(self, device): return self
            def eval(self): return self
            def cpu(self): return self

        def model_stub(root, arm, seed, *args, **kwargs):
            return Model(), {'family': 'motion'}, {'arm': arm, 'family': 'motion', 'training_seed': seed,
                                                  'checkpoint_sha256': 'a' * 64}

        def audio_stub(source, target, offset, duration):
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'excerpt audio')
            return target

        def video_stub(frames, target, **kwargs):
            target = Path(target)
            target.write_bytes(b'fake encoded video')
            return target

        def pairs_stub(root, protocol, candidates, **kwargs):
            for candidate in candidates:
                for name, provenance in candidate['arm_provenance'].items():
                    self.assertEqual(provenance['generation_seed'], 102 if name.endswith('_g102') else 101)
            folder = Path(root) / 'results' / 'renders'
            folder.mkdir()
            (folder / 'pairs.json').write_text('{"pairs":[]}')
            (folder / 'ratings.jsonl').write_text('')
            return {'n_primary': 40, 'n_repeat': 4, 'n_sham': 4}

        fake_conductor = SimpleNamespace(render=lambda desc, bus: SimpleNamespace(frames=np.full((1600, 512), desc, dtype=np.uint8)))
        with patch.object(delivery, '_render_context', return_value=context), \
             patch.object(delivery, 'verify_training_inputs'), \
             patch.object(delivery, '_load_verified_render_model', side_effect=model_stub), \
             patch.object(delivery, 'motion_adapter', return_value=SimpleNamespace(descriptors=lambda bus, seed: seed)), \
             patch.object(delivery.gc, 'collect'), \
             patch.object(delivery.torch.cuda, 'empty_cache'), \
             patch('lightsem.experiments.stage11.stage11_data.load_manifest', return_value=manifest), \
             patch('lightsem.motion.io.load_bus', return_value=bus), \
             patch('lightsem.motion.io.slice_bus', return_value=bus), \
             patch('lightsem.conductor.hybrid.ConductorHybrid', return_value=fake_conductor), \
             patch.object(stage11_render, 'extract_audio_excerpt', side_effect=audio_stub), \
             patch.object(stage11_render, 'render_frames', side_effect=video_stub), \
             patch.object(stage11_render, 'prepare_pairs', side_effect=pairs_stub), \
             patch('builtins.print'):
            report = delivery.render_campaign(self.root, 'test')
        candidate_path = self.root / 'results' / 'render_candidates.json'
        candidates = read_authenticated(candidate_path)['candidates']
        self.assertEqual(len(candidates), 20)
        self.assertEqual(report['n_primary'] + report['n_repeat'] + report['n_sham'], 48)
        self.assertIn(str(candidate_path), report['artifacts'])
        self.assertIn(str(candidate_path) + '.sha256.json', report['artifacts'])
        for candidate in candidates:
            self.assertNotIn('generation_seed', candidate)
            for path in candidate['arms'].values():
                self.assertEqual(report['artifacts'][path]['sha256'], digest(path))
        self.assertTrue(all(not path.endswith('ratings.jsonl') for path in report['artifacts']))


if __name__ == '__main__':
    unittest.main()

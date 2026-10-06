"""CPU engineering counterexamples; these do not establish musical quality."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lightsem.experiments.dynamic_dance import training
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline
from lightsem.experiments.dynamic_dance.learning import (
    DynamicFlow, DynamicFlowConfig, EffortPlanner, PlannerConfig,
    drop_named_sources, kinematic_plan_targets, plan_condition, planner_losses,
    pool_half_seconds,
)
from lightsem.experiments.motion_study.music_model import MusicConfig, MusicModel
from lightsem.experiments.motion_study.geometry import joints_from_features
from lightsem.experiments.training_runtime import BatchStream


def setUpModule():
    torch.set_num_threads(2)


def held(frames, batch=1):
    value = torch.zeros(batch, frames, 139)
    value[..., 2] = 1.1
    value[..., 3:135] = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(22)
    value[..., 135:] = 1
    return value


def normalizers():
    return {'mean': torch.linspace(-.1, .1, 139).tolist(),
            'std': torch.linspace(.7, 1.3, 139).tolist(),
            'delta_scale': torch.linspace(.2, 1.2, 139).tolist()}


def incumbent():
    torch.manual_seed(1729)
    model = MusicModel(MusicConfig(width=8, frames=24, dilations=(1, 2), steps=3), normalizers())
    # A nonzero incumbent prevents vacuous zero-output parity/sensitivity tests.
    with torch.no_grad():
        model.output.weight.normal_(std=.2)
        model.output.bias.normal_(std=.02)
    return model


def flow(arm='D7', active=False):
    base = incumbent()
    result = DynamicFlow(base.config, normalizers(),
                         DynamicFlowConfig(extra_dim=12, arm=arm, adapter_width=8))
    result.load_incumbent(base.state_dict())
    if active:
        with torch.no_grad():
            for adapter in result.adapters:
                adapter.weight.normal_(std=.2)
                adapter.bias.normal_(std=.03)
    return result


def fixture(model, frames=37, batch=2):
    generator = torch.Generator().manual_seed(812)
    audio = torch.randn(batch, frames, model.conditioning_dim, generator=generator)
    state = torch.randn(batch, frames, 139, generator=generator)
    time = torch.linspace(.2, .7, batch)
    return audio, state, time


class KinematicPlanTests(unittest.TestCase):
    def test_constant_root_velocity_is_physical_speed_in_every_region(self):
        native = held(31)[0]
        native[:, 0] = .3
        native[:, 1] = -.4
        actual = kinematic_plan_targets(native)
        torch.testing.assert_close(actual['log_speed'], torch.full((3, 5), torch.log1p(torch.tensor(.5))),
                                   atol=1e-6, rtol=1e-6)
        self.assertEqual(actual['valid'].tolist(), [True, True, True])
        self.assertEqual(actual['hold'].sum().item(), 0)
        self.assertEqual(actual['duration_valid'].sum().item(), 0)
        torch.testing.assert_close(actual['support'], torch.ones(3, 4))

    def test_d6_observed_speed_does_not_cancel_alternating_jitter(self):
        from lightsem.experiments.motion_study.geometry import joints_from_features
        native = held(31)[0]
        native[:, 0] = torch.where(torch.arange(31) % 2 == 0, .3, -.3)
        world = joints_from_features(native).numpy()
        centered = np.linalg.norm(np.gradient(world.astype(np.float64), axis=0)*30, axis=-1)
        np.testing.assert_allclose(centered[1:-1], 0., atol=1e-12)
        observed = training.observed_regional_speed(world)
        np.testing.assert_allclose(observed, .3, atol=3e-6, rtol=0)
        pooled, _ = pool_half_seconds(torch.from_numpy(observed.astype(np.float32))[None])
        torch.testing.assert_close(torch.log1p(pooled[0]), kinematic_plan_targets(native)['log_speed'],
                                   atol=2e-7, rtol=1e-6)

    def test_d6_observed_speed_explicit_endpoint_alignment(self):
        world = np.zeros((4, 22, 3), np.float64)
        world[:, :, 0] = np.array([0., .01, .03, .03])[:, None]
        expected = np.repeat(np.array([.3, .3, .6, 0.])[:, None], 5, axis=1)
        np.testing.assert_allclose(training.observed_regional_speed(world), expected, atol=1e-14)
        np.testing.assert_array_equal(training.observed_regional_speed(world[:1]), np.zeros((1, 5)))

    def test_partial_bin_uses_observed_frames_and_masked_pool_ignores_padding(self):
        value = torch.arange(17, dtype=torch.float32)[None, :, None]
        pooled, valid = pool_half_seconds(value)
        torch.testing.assert_close(pooled[0, :, 0], torch.tensor([7., 15.5]))
        self.assertEqual(valid.tolist(), [[True, True]])
        padded = torch.cat((value, torch.full((1, 13, 1), float('nan'))), 1)
        mask = torch.arange(30)[None] < 17
        padded_pool, padded_valid = pool_half_seconds(padded, mask)
        torch.testing.assert_close(padded_pool, pooled)
        self.assertTrue(torch.equal(padded_valid, valid))
        native = held(17)[0]
        native[15:, 135:] = 0
        targets = kinematic_plan_targets(native)
        torch.testing.assert_close(targets['support'], torch.tensor([[1.] * 4, [0.] * 4]))

    def test_articulation_changes_its_region_without_inventing_root_motion(self):
        native = held(31)[0]
        angle = torch.linspace(0., .4, 31)
        zeros = torch.zeros_like(angle)
        # Local shoulder rotation: the left elbow/wrist move, root/support do not.
        native[:, 3+16*6:3+17*6] = torch.stack(
            (angle.cos(), -angle.sin(), zeros, angle.sin(), angle.cos(), zeros), -1)
        targets = kinematic_plan_targets(native)
        self.assertGreater(targets['log_speed'][:, 3].min().item(), .02)
        torch.testing.assert_close(targets['log_speed'][:, [0, 1, 2, 4]], torch.zeros(3, 4),
                                   atol=1e-7, rtol=0)

    def test_duration_remaining_and_final_run_right_censoring(self):
        native = held(77)[0]
        # Outgoing velocity at 29 drives frame 30, giving a clean 1s transition.
        native[29:, 0] = .3
        targets = kinematic_plan_targets(native)
        self.assertEqual(targets['duration'][:2, 0].tolist(), [1, 0])
        self.assertTrue(targets['duration_valid'][:2].all())
        self.assertFalse(targets['duration_valid'][2:].any())
        self.assertTrue(targets['hold'][:2].bool().all())
        self.assertFalse(targets['hold'][2:].bool().any())
        self.assertEqual(targets['valid'].shape, (6,))

    def test_stationary_one_frame_has_no_observed_speed_or_duration_label(self):
        targets = kinematic_plan_targets(held(1)[0])
        self.assertFalse(targets['valid'].any())
        self.assertFalse(targets['duration_valid'].any())
        self.assertTrue(torch.isfinite(targets['log_speed']).all())
        for invalid in (torch.zeros(0, 139), torch.zeros(3, 138), torch.full((3, 139), float('nan'))):
            with self.assertRaises(ValueError):
                kinematic_plan_targets(invalid)


class PlannerTests(unittest.TestCase):
    def planner(self):
        torch.manual_seed(244)
        return EffortPlanner(PlannerConfig(input_dim=6, width=8, dilations=(1, 2), heads=2))

    def test_remote_audio_changes_global_plan_but_not_outside_local_receptive_field(self):
        model = self.planner().eval()
        audio = torch.randn(1, 64, 6, generator=torch.Generator().manual_seed(43))
        valid = torch.ones(1, 64, dtype=torch.bool)
        changed = audio.clone()
        changed[:, 50:, 1] += 8
        before, after = model(audio, valid), model(changed, valid)
        torch.testing.assert_close(before['local'][:, :20], after['local'][:, :20], atol=0, rtol=0)
        self.assertGreater((before['global'][:, :20] - after['global'][:, :20]).abs().max().item(), 1e-4)
        self.assertGreater((before['local'][:, 50:] - after['local'][:, 50:]).abs().max().item(), 1e-4)

    def test_planner_padding_content_and_all_padded_rows_are_finite(self):
        model = self.planner()
        audio = torch.randn(2, 17, 6, generator=torch.Generator().manual_seed(44))
        valid = torch.arange(17)[None] < torch.tensor([9, 0])[:, None]
        expected = model(audio, valid)
        audio[~valid] = float('nan')
        audio.requires_grad_()
        actual = model(audio, valid)
        for branch in actual:
            torch.testing.assert_close(actual[branch], expected[branch], atol=0, rtol=0)
            self.assertTrue(torch.isfinite(actual[branch]).all())
            self.assertEqual(actual[branch][~valid].abs().sum().item(), 0)
        sum(value.square().sum() for value in actual.values()).backward()
        self.assertTrue(torch.isfinite(audio.grad).all())
        self.assertEqual(audio.grad[~valid].abs().sum().item(), 0)

    def test_censored_duration_has_no_training_gradient(self):
        native = held(77)[0]
        native[29:, 0] = .3
        targets = {name: value[None] for name, value in kinematic_plan_targets(native).items()}
        raw = torch.randn(1, 6, 34, generator=torch.Generator().manual_seed(61), requires_grad=True)
        loss = planner_losses({'local': raw}, targets, targets['valid'])
        loss['total'].backward()
        gradient = raw.grad[..., 14:].reshape(1, 6, 5, 4)
        self.assertEqual(gradient[~targets['duration_valid']].abs().sum().item(), 0)
        self.assertGreater(gradient[targets['duration_valid']].abs().sum().item(), 0)

    def test_planner_loss_padding_values_and_gradients_are_ignored(self):
        targets = {name: value[None] for name, value in kinematic_plan_targets(held(46)[0]).items()}
        valid = torch.tensor([[True, True, False, False]])
        raw = torch.randn(1, 4, 34, generator=torch.Generator().manual_seed(6), requires_grad=True)
        expected = planner_losses({'local': raw}, targets, valid)
        expected['total'].backward()
        expected_grad = raw.grad.clone()
        changed_targets = copy.deepcopy(targets)
        for key in ('log_speed', 'support', 'hold'):
            changed_targets[key][~valid] = float('nan')
        altered = raw.detach().clone()
        altered[~valid] = float('nan')
        altered.requires_grad_()
        actual = planner_losses({'local': altered}, changed_targets, valid)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        actual['total'].backward()
        self.assertTrue(torch.isfinite(altered.grad).all())
        torch.testing.assert_close(altered.grad, expected_grad)
        self.assertEqual(altered.grad[~valid].abs().sum().item(), 0)


class DynamicFlowTests(unittest.TestCase):
    def test_every_warmstarted_arm_matches_nonzero_incumbent_with_shared_noise(self):
        base = incumbent()
        for arm in ('continued', 'D7', 'D8', 'D8_flat'):
            with self.subTest(arm=arm):
                candidate = flow(arm)
                audio, state, time = fixture(candidate)
                expected = base.vector_field(audio[..., :92], state, time)
                self.assertGreater(expected.abs().sum().item(), 0)
                torch.testing.assert_close(candidate.vector_field(audio, state, time), expected, atol=0, rtol=0)
                torch.testing.assert_close(candidate.generate(audio, initial_noise=state, chunked=False),
                                           base.generate(audio[..., :92], initial_noise=state, chunked=False),
                                           atol=0, rtol=0)

    def test_nonzero_adapters_respond_to_audio_and_correct_plan_arm_only(self):
        for arm in ('continued', 'D7', 'D8', 'D8_flat'):
            with self.subTest(arm=arm):
                model = flow(arm, active=True)
                audio, state, time = fixture(model)
                baseline = model.vector_field(audio, state, time)
                donor = audio.clone()
                donor[..., 92:104] = donor[..., 92:104].flip(1) + 1
                audio_delta = (model.vector_field(donor, state, time)[..., :135] - baseline[..., :135]).abs().max().item()
                alternate_plan = audio.clone()
                alternate_plan[..., -34:] += 2
                plan_delta = (model.vector_field(alternate_plan, state, time)[..., :135] - baseline[..., :135]).abs().max().item()
                if arm == 'continued':
                    self.assertEqual(audio_delta, 0)
                else:
                    self.assertGreater(audio_delta, 1e-4)
                if arm in ('D8', 'D8_flat'):
                    self.assertGreater(plan_delta, 1e-4)
                else:
                    self.assertEqual(plan_delta, 0)

    def test_active_adapters_preserve_exact_halo_with_precomputed_full_song_plan(self):
        for arm in ('D7', 'D8', 'D8_flat'):
            with self.subTest(arm=arm):
                model = flow(arm, active=True)
                audio, state, time = fixture(model, frames=103)
                planner = EffortPlanner(PlannerConfig(input_dim=12, width=8, dilations=(1,), heads=2))
                branch = 'global' if arm == 'D8' else 'local'
                # Full recording plan is calculated once, before decoder chunking.
                plans = planner(audio[..., 92:104], torch.ones(audio.shape[:2], dtype=torch.bool))
                audio[..., -34:] = plan_condition(plans[branch]).detach()
                expected = model.vector_field(audio, state, time)
                for core in (1, 17, 29):
                    actual = model.vector_field_chunked(audio, state, time, core_frames=core)
                    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=2e-6)
                self.assertEqual(model.halo, incumbent().halo)

    def test_flow_padding_content_does_not_change_losses_or_parameter_gradients(self):
        model = flow('D8', active=True)
        audio, noise, time = fixture(model, frames=11)
        target = held(11, batch=2)
        target[..., 0] = .15
        valid = torch.arange(11)[None] < torch.tensor([4, 11])[:, None]
        expected = model.losses_from_draws(audio, target, valid, noise, time)
        expected['total'].backward()
        gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        model.zero_grad(set_to_none=True)
        for value in (audio, target, noise):
            value[~valid] = float('nan')
        audio.requires_grad_()
        noise.requires_grad_()
        target.requires_grad_()
        actual = model.losses_from_draws(audio, target, valid, noise, time)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        actual['total'].backward()
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(audio.grad).all())
        self.assertTrue(torch.isfinite(noise.grad).all())
        self.assertEqual(audio.grad[~valid].abs().sum().item(), 0)
        self.assertEqual(noise.grad[~valid].abs().sum().item(), 0)
        self.assertIsNone(target.grad)

    def test_fully_masked_flow_has_zero_losses_and_finite_zero_gradients(self):
        model = flow('D8', active=True)
        audio, noise, time = fixture(model, frames=1)
        target = held(1, batch=2)
        valid = torch.zeros(2, 1, dtype=torch.bool)
        result = model.losses_from_draws(audio, target, valid, noise, time)
        self.assertTrue(all(value.item() == 0 for value in result.values()))
        result['total'].backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() == 0
                            for parameter in model.parameters() if parameter.grad is not None))

    def test_incumbent_loading_rejects_missing_base_or_unknown_keys(self):
        state = incumbent().state_dict()
        missing = dict(state)
        missing.pop('input.weight')
        with self.assertRaises(ValueError):
            flow().load_incumbent(missing)
        with self.assertRaises(ValueError):
            flow().load_incumbent({**state, 'unregistered': torch.zeros(1)})


class NamedSourceDropoutTests(unittest.TestCase):
    def test_explicit_source_mask_keeps_base_and_plan_and_has_no_hidden_rng(self):
        audio = torch.arange(2*7*138, dtype=torch.float32).reshape(2, 7, 138)
        original = audio.clone()
        source_indices = ((92, 93), (94, 95), (96, 97), (98, 99))
        keep = torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 1.]])
        random_state = torch.random.get_rng_state().clone()
        actual = drop_named_sources(audio, source_indices, keep)
        torch.testing.assert_close(audio, original, atol=0, rtol=0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), random_state))
        torch.testing.assert_close(actual[..., :92], audio[..., :92], atol=0, rtol=0)
        torch.testing.assert_close(actual[..., 100:], audio[..., 100:], atol=0, rtol=0)
        for batch in range(2):
            for source, indices in enumerate(source_indices):
                torch.testing.assert_close(actual[batch, :, list(indices)],
                                           audio[batch, :, list(indices)] * keep[batch, source], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            drop_named_sources(audio, source_indices, torch.ones(4))


class TrainingDataTests(unittest.TestCase):
    """Small real NPZ/JSON fixtures, with no dataset, accelerator or model downloads."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='lightsem-dynamic-learning-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.native_root = self.root/'native'
        self.campaign = self.root/'campaign'
        (self.native_root/'data').mkdir(parents=True)
        self.campaign.mkdir()
        self.entries, index = [], []
        # Unequal recording counts and durations expose frame-weighted norm leaks.
        definitions = [('a', 'same_song', 'train', 1, 2.),
                       ('b', 'same_song', 'train', 17, 6.),
                       ('c', 'other_song', 'train', 46, 10.),
                       ('d', 'development', 'val', 23, 100.)]
        for key, canonical, partition, length, level in definitions:
            native_path = self.native_root/f'{key}_native.npz'
            audio_path = self.native_root/f'{key}_base.npz'
            native = held(length)[0].numpy()
            native[:, 0] = level*.01
            np.savez(native_path, features=native)
            np.savez(audio_path, x=np.full((length, 92), level, np.float32))
            waveform_sha = training.json_digest({'synthetic_waveform_fixture': key})
            timeline = MusicTimeline(
                time=np.arange(length)/30., muq=np.full((1, 1024), level, np.float32),
                muq_time=np.array([0.]), sources=np.full((length, 4, 4), level, np.float32),
                mixture=np.full((length, 4), level, np.float32), beats=np.array([]),
                downbeats=np.array([]), sections=[{'start': 0., 'end': length/30., 'label': 'verse'}],
                recurrence=np.ones((1, 1), np.float32),
                metadata={'fixture': key, 'input_sha256': waveform_sha, 'intervention': None})
            timeline_path = self.root/f'{key}_timeline.npz'
            timeline.save(timeline_path)
            entry = {'key': key, 'canonical_id': canonical, 'partition': partition, 'n_frames': length,
                     'path': str(native_path), 'sha256': training.digest(native_path),
                     'audio_path': str(audio_path), 'audio_sha256': training.digest(audio_path),
                     'wav_sha256': waveform_sha}
            self.entries.append(entry)
            index.append({**entry, 'timeline_path': str(timeline_path),
                          'timeline_sha256': training.digest(timeline_path),
                          'timeline_metadata_sha256': training.digest(timeline_path.with_suffix('.json'))})
        # No files exist for TEST. Even a digest or array-open attempt must fail.
        self.entries.append({'key': 'closed', 'canonical_id': 'closed_song', 'partition': 'test',
                             'path': str(self.root/'DO_NOT_OPEN_TEST.npz')})
        training.authenticated_json(self.native_root/'data/audio_manifest.json', {'entries': self.entries})
        self.index_path = self.campaign/'index.json'
        self.index_path.write_text(json.dumps({'entries': index + [self.entries[-1]]}))
        for mock in (patch.object(training, 'NATIVE_ROOT', self.native_root),
                     patch.object(training, 'require_space', return_value=100.)):
            mock.start()
            self.addCleanup(mock.stop)
        self.manifest = training.prepare_cache(self.campaign, self.index_path)
        self.norms = training.read_authenticated(self.campaign/'learning/cache/normalizers.json')
        self.base_norms = {'mean': [1.]*92, 'std': [2.]*92}

    def resident(self, manifest=None, plan_paths=None):
        return training.FlowResident(self.manifest if manifest is None else manifest,
                                     self.base_norms, self.norms, frames=30, device='cpu',
                                     plan_paths=plan_paths)

    def test_implementation_hashes_cover_real_current_launcher_and_model_sources(self):
        hashes = training.implementation_hashes()
        self.assertTrue(any(path.endswith('/scripts/motion/train_dynamic_dance.py') for path in hashes))
        self.assertTrue(any(path.endswith('/learning.py') for path in hashes))
        self.assertTrue(any(path.endswith('/training_runtime.py') for path in hashes))
        for path, digest in hashes.items():
            self.assertTrue(Path(path).is_file(), path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(digest, training.digest(path))

    def test_prepare_cache_preserves_short_recordings_and_train_only_equal_song_norms(self):
        self.assertEqual(self.manifest['coverage']['train'],
                         {'recordings': 3, 'canonical_songs': 2, 'frames': 64})
        self.assertEqual(self.manifest['coverage']['val']['frames'], 23)
        self.assertFalse(self.manifest['test_access'])
        self.assertEqual({row['key'] for row in self.manifest['entries']}, {'a', 'b', 'c', 'd'})
        # .25*a + .25*b + .5*c = 7; development's level100 must not leak.
        self.assertAlmostEqual(self.norms['mean'][0], 7.)
        self.assertAlmostEqual(self.norms['std'][0], np.sqrt(11.))
        for entry in self.manifest['entries']:
            with np.load(entry['cache_path'], allow_pickle=False) as cache:
                self.assertEqual(len(cache['extra']), entry['n_frames'])
                self.assertEqual(len(cache['pooled']), (entry['n_frames']+14)//15)
                self.assertEqual(cache['extra'].dtype, np.float16)
                self.assertTrue(np.isfinite(cache['extra']).all())
                if entry['n_frames'] == 1:
                    self.assertFalse(cache['target_valid'].any())

    def test_cache_repeat_is_authenticated_and_corruption_cannot_be_reused(self):
        self.assertEqual(training.prepare_cache(self.campaign, self.index_path), self.manifest)
        path = Path(self.manifest['entries'][0]['cache_path'])
        with path.open('ab') as stream:
            stream.write(b'altered')
        with self.assertRaisesRegex(RuntimeError, 'Frozen input changed'):
            training.prepare_cache(self.campaign, self.index_path)
        with self.assertRaisesRegex(ValueError, 'Planner cache changed'):
            training.PlannerData(self.manifest, self.norms)
        with self.assertRaisesRegex(RuntimeError, 'Frozen input changed'):
            self.resident()

    def test_training_rejects_authenticated_unpaired_or_intervened_audio(self):
        index = json.loads(self.index_path.read_text())
        row = index['entries'][0]
        metadata_path = Path(row['timeline_path']).with_suffix('.json')
        original = json.loads(metadata_path.read_text())
        for case, edits in [('different_waveform', {'input_sha256': 'e'*64}),
                            ('intervened', {'intervention': {'kind': 'gain', 'gain': .5}})]:
            with self.subTest(case=case):
                changed = copy.deepcopy(original)
                changed['metadata'].update(edits)
                training.authenticated_json(metadata_path, changed)
                row['timeline_metadata_sha256'] = training.digest(metadata_path)
                index_path = self.root/f'{case}_index.json'
                index_path.write_text(json.dumps(index))
                with self.assertRaisesRegex(ValueError, 'not the original paired recording'):
                    training.prepare_cache(self.root/f'{case}_cache', index_path)

    def test_samplers_do_not_open_val_or_test_when_training(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['entries'].append(self.entries[-1])
        for row in manifest['entries']:
            if row['partition'] != 'train':
                for field in ('path', 'audio_path', 'cache_path'):
                    row[field] = str(self.root/'MUST_NOT_BE_READ.npz')
        self.assertEqual(self.resident(manifest).coverage['recordings'], 3)
        self.assertEqual(len(training.PlannerData(manifest, self.norms).records), 3)

    def test_flow_retains_all_short_frames_once_and_never_crops_across_recordings(self):
        data = self.resident()
        self.assertEqual(data.coverage, {'recordings': 3, 'canonical_songs': 2,
                                         'short_recordings': 2, 'real_frames': 64, 'padding_frames': 42})
        self.assertEqual(data.lengths.tolist(), [1, 17, 46])
        for entry, start, length in zip(data.entries, data.starts, data.lengths):
            with np.load(entry['path'], allow_pickle=False) as source:
                torch.testing.assert_close(data.y[start:start+length], torch.from_numpy(source['features']))
            with np.load(entry['audio_path'], allow_pickle=False) as source:
                torch.testing.assert_close(data.x[start:start+length], torch.from_numpy((source['x']-1.)/2.))
            self.assertTrue(data.b[start:start+length].all())
            if length < data.frames:
                self.assertFalse(data.b[start+length:start+data.frames].any())
                self.assertEqual(data.y[start+length:start+data.frames].abs().sum().item(), 0)
                self.assertEqual(data.x[start+length:start+data.frames].abs().sum().item(), 0)
        indices = data.draw_indices(256, 30, np.random.default_rng(47))
        self.assertEqual(set(data.b[indices].sum(1).tolist()), {1, 17, 30})
        self.assertTrue(np.all(np.diff(indices, axis=1) == 1))
        for row in indices:
            self.assertTrue(any(start <= row[0] <= row[-1] < start+max(length, 30)
                                for start, length in zip(data.starts, data.lengths)))
        with self.assertRaises(ValueError):
            data.draw_indices(1, 29, np.random.default_rng(47))

    def test_canonical_uniform_sampling_and_replay_ignore_manifest_order(self):
        first = self.resident()
        reversed_manifest = {**self.manifest, 'entries': list(reversed(self.manifest['entries']))}
        second = self.resident(reversed_manifest)
        rng_a, rng_b = np.random.default_rng(901), np.random.default_rng(901)
        a, b = first.draw_indices(4000, 30, rng_a), second.draw_indices(4000, 30, rng_b)
        np.testing.assert_array_equal(a, b)
        recording = np.searchsorted(first.starts, a[:, 0], side='right')-1
        frequencies = np.bincount(recording, minlength=3)/len(recording)
        np.testing.assert_allclose(frequencies, [.25, .25, .5], atol=.025, rtol=0)
        state = copy.deepcopy(rng_a.bit_generator.state)
        expected = first.draw_indices(17, 30, rng_a)
        rng_a.bit_generator.state = state
        np.testing.assert_array_equal(first.draw_indices(17, 30, rng_a), expected)

    def test_depth_zero_batchstream_matches_paired_batches_and_rng(self):
        data = self.resident()
        direct_rng, stream_rng = np.random.default_rng(99), np.random.default_rng(99)
        with BatchStream(data, 5, 30, stream_rng, depth=0) as stream:
            for _ in range(3):
                expected = data.batch(5, 30, direct_rng)
                actual = stream.next()
                for a, b in zip(expected, actual):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                self.assertEqual(direct_rng.bit_generator.state, stream_rng.bit_generator.state)
                self.assertEqual(actual[0].shape, (5, 30, 92+1061+34))
                self.assertEqual(actual[2].dtype, torch.bool)
                self.assertEqual(actual[1][~actual[2]].abs().sum().item(), 0)
        with self.assertRaises(RuntimeError):
            stream.next()

    def test_planner_complete_recordings_and_labels_are_masked_without_repetition(self):
        data = training.PlannerData(self.manifest, self.norms)
        self.assertEqual(data.frames, 4)
        audio, valid, targets = data.batch(256, np.random.default_rng(87), 'cpu')
        self.assertEqual(set(valid.sum(1).tolist()), {1, 2, 4})
        self.assertEqual(audio[~valid].abs().sum().item(), 0)
        self.assertFalse(targets['valid'][~valid].any())
        self.assertFalse(targets['duration_valid'][~valid].any())
        self.assertFalse(targets['valid'][valid.sum(1) == 1].any())
        for row in range(len(audio)):
            length = int(valid[row].sum())
            record = next(record for record in data.records if len(record['audio']) == length)
            torch.testing.assert_close(audio[row, :length], torch.from_numpy(record['audio']), atol=0, rtol=0)
            for key in targets:
                torch.testing.assert_close(targets[key][row, :length], torch.from_numpy(record['targets'][key]),
                                           atol=0, rtol=0)
        # A model can consume the full padded recording; no invalid labels score.
        model = EffortPlanner(PlannerConfig(1061, width=8, dilations=(1,), heads=2))
        loss = planner_losses(model(audio[:5], valid[:5]), {key: value[:5] for key, value in targets.items()}, valid[:5])
        self.assertTrue(all(torch.isfinite(value).all() for value in loss.values()))
        loss['total'].backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                            if parameter.grad is not None))

    def test_precomputed_plan_interpolation_and_authentication(self):
        plans = {}
        for entry in self.manifest['entries']:
            if entry['partition'] != 'train':
                continue
            path = self.root/f'{entry["key"]}_plan.npz'
            np.savez(path, time=np.array([0., 1., 2.]),
                     plan=np.repeat(np.array([[0.], [1.], [2.]], np.float32), 34, axis=1))
            plans[entry['key']] = {'path': str(path), 'sha256': training.digest(path)}
        data = self.resident(plan_paths=plans)
        for start, length in zip(data.starts, data.lengths):
            expected = np.repeat((np.arange(length)/30.)[:, None], 34, axis=1)
            np.testing.assert_allclose(data.host_aux[start:start+length, -34:], expected, atol=5e-4, rtol=0)
        with Path(plans['a']['path']).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'Frozen input changed'):
            self.resident(plan_paths=plans)


class InferenceBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='lightsem-dynamic-binding-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root/'base92.npz'
        np.savez(self.path, x=np.zeros((2, 92), np.float32), time=np.arange(2)/30.)
        self.intervention = {'kind': 'gain', 'gain': .5, 'source_audio_sha256': 'b'*64}
        self.receipt = {'kind': 'Base92', 'npz_sha256': training.digest(self.path),
                        'analyzed_waveform_sha256': 'a'*64, 'intervention': self.intervention}
        training.authenticated_json(self.path.with_suffix('.json'), self.receipt)
        self.timeline = MusicTimeline(
            time=np.arange(2)/30., muq=np.zeros((1, 1024), np.float32), muq_time=np.array([0.]),
            sources=np.zeros((2, 4, 4), np.float32), mixture=np.zeros((2, 4), np.float32),
            beats=np.array([]), downbeats=np.array([]), sections=[], recurrence=np.zeros((0, 0)),
            metadata={'analyzed_waveform_sha256': 'a'*64, 'intervention': copy.deepcopy(self.intervention)})

    def test_inference_music_provenance_retains_resolvable_timeline_and_source_identity(self):
        timeline_path = self.root/'gain_song.npz'
        self.timeline.metadata.update(key='original_song_gain_half',
                                      input_path='/source/original_song.wav', input_sha256='e'*64)
        self.timeline.save(timeline_path)
        result = training.timeline_provenance(self.timeline, timeline_path)
        self.assertEqual(result['timeline_path'], str(timeline_path.resolve()))
        self.assertEqual(result['timeline_sha256'], training.digest(timeline_path))
        self.assertEqual(result['timeline_metadata_sha256'], training.digest(timeline_path.with_suffix('.json')))
        self.assertEqual(result['source_key'], 'original_song_gain_half')
        self.assertEqual(result['audio_input_path'], '/source/original_song.wav')
        self.assertEqual(result['audio_input_sha256'], 'e'*64)
        self.assertEqual(result['analyzed_waveform_sha256'], 'a'*64)
        self.assertEqual(result['audio_intervention'], self.intervention)
        self.timeline.metadata['intervention']['gain'] = .9
        self.assertEqual(result['audio_intervention']['gain'], .5)
        self.assertEqual(MusicTimeline.load(result['timeline_path']).metadata['intervention']['gain'], .5)

    def test_authenticated_same_waveform_and_intervention_are_required(self):
        self.assertEqual(training.validate_audio_binding(self.timeline, self.path), self.receipt)
        self.timeline.metadata['analyzed_waveform_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError, 'different or unknown waveforms'):
            training.validate_audio_binding(self.timeline, self.path)
        self.timeline.metadata['analyzed_waveform_sha256'] = 'a'*64
        self.timeline.metadata['intervention']['gain'] = 1.
        with self.assertRaisesRegex(ValueError, 'intervention provenance differs'):
            training.validate_audio_binding(self.timeline, self.path)
        self.timeline.metadata['intervention'] = None
        with self.assertRaisesRegex(ValueError, 'intervention provenance differs'):
            training.validate_audio_binding(self.timeline, self.path)

    def test_optional_timeline_feature_hash_binds_the_exact_paired_arrays(self):
        self.timeline.metadata['base92_sha256'] = training.digest(self.path)
        self.assertEqual(training.validate_audio_binding(self.timeline, self.path), self.receipt)
        self.timeline.metadata['base92_sha256'] = 'd'*64
        with self.assertRaisesRegex(ValueError, 'paired-feature receipt'):
            training.validate_audio_binding(self.timeline, self.path)
        self.timeline.metadata.pop('base92_sha256')
        self.timeline.metadata['base92'] = {'npz_sha256': training.digest(self.path)}
        self.assertEqual(training.validate_audio_binding(self.timeline, self.path), self.receipt)
        self.timeline.metadata['base92']['npz_sha256'] = 'd'*64
        with self.assertRaisesRegex(ValueError, 'paired-feature receipt'):
            training.validate_audio_binding(self.timeline, self.path)

    def test_wrong_kind_or_changed_arrays_cannot_pass_an_authenticated_receipt(self):
        training.authenticated_json(self.path.with_suffix('.json'), {**self.receipt, 'kind': 'MusicTimeline'})
        with self.assertRaisesRegex(ValueError, 'Unauthenticated or changed Base92'):
            training.validate_audio_binding(self.timeline, self.path)
        training.authenticated_json(self.path.with_suffix('.json'), self.receipt)
        np.savez(self.path, x=np.ones((2, 92), np.float32), time=np.arange(2)/30.)
        with self.assertRaisesRegex(ValueError, 'Unauthenticated or changed Base92'):
            training.validate_audio_binding(self.timeline, self.path)

    def test_changed_receipt_and_unknown_waveforms_are_rejected(self):
        metadata_path = self.path.with_suffix('.json')
        metadata_path.write_text(json.dumps({**self.receipt, 'intervention': None}))
        with self.assertRaisesRegex(RuntimeError, 'Artifact digest mismatch'):
            training.validate_audio_binding(self.timeline, self.path)
        training.authenticated_json(metadata_path, self.receipt)
        for invalid in (None, '', 'short'):
            self.timeline.metadata['analyzed_waveform_sha256'] = invalid
            with self.assertRaisesRegex(ValueError, 'different or unknown waveforms'):
                training.validate_audio_binding(self.timeline, self.path)


class EvaluationProtocolTests(unittest.TestCase):
    def manifest(self):
        entries = []
        for key, canonical, frames, bpm in [('a', 'same', 1, 120.), ('b', 'second', 517, 119.),
                                            ('c', 'third', 1801, 120.5), ('d', 'same', 2100, 120.),
                                            ('e', 'fourth', 2100, 126.)]:
            entries.append({'key': key, 'canonical_id': canonical, 'n_frames': frames,
                            'tempo_bpm': bpm, 'partition': 'val'})
        entries.extend([{'key': 'closed_train', 'partition': 'train'}, {'key': 'closed_test', 'partition': 'test'}])
        return {'entries': entries}

    def test_windows_keep_short_recordings_and_choose_distinct_nearest_bpm_donors(self):
        manifest = self.manifest()
        windows = training.evaluation_windows(manifest)
        lookup = {row['key']: row for row in windows}
        self.assertEqual(set(lookup), {'a', 'b', 'c', 'd', 'e'})
        self.assertEqual(lookup['a']['frames'], 1)
        self.assertEqual(lookup['b']['frames'], 517)
        self.assertEqual(lookup['a']['start'], 0)
        self.assertEqual(lookup['b']['start'], 0)
        self.assertIsNone(lookup['a']['shift_start'])
        self.assertIsNone(lookup['b']['shift_start'])
        self.assertEqual(lookup['a']['donor_key'], 'c')  # Exact-BPM d is same canonical song.
        self.assertEqual(lookup['c']['donor_key'], 'd')
        entries = {row['key']: row for row in manifest['entries']}
        for row in windows:
            entry, donor = entries[row['key']], entries[row['donor_key']]
            self.assertNotEqual(entry['canonical_id'], donor['canonical_id'])
            self.assertGreaterEqual(donor['n_frames'], row['frames'])
            self.assertEqual(row['start'] % 15, 0)
            self.assertEqual(row['donor_start'] % 15, 0)
            self.assertLessEqual(row['start']+row['frames'], entry['n_frames'])
            self.assertLessEqual(row['donor_start']+row['frames'], donor['n_frames'])
            self.assertTrue(row['similar_bpm'])
        self.assertEqual(windows, training.evaluation_windows({'entries': list(reversed(manifest['entries']))}))

    def test_within_song_shifts_are_contiguous_and_never_wrap_or_repeat(self):
        entries = {row['key']: row for row in self.manifest()['entries']}
        windows = training.evaluation_windows(self.manifest())
        for row in windows:
            shift = row['shift_start']
            if shift is None:
                continue
            self.assertGreaterEqual(abs(shift-row['start']), 150)
            self.assertGreaterEqual(shift, 0)
            self.assertLessEqual(shift+row['frames'], entries[row['key']]['n_frames'])
            self.assertIn(shift, (0, entries[row['key']]['n_frames']-row['frames']))
        self.assertEqual(next(row['shift_start'] for row in windows if row['key'] == 'c'), 841)

    def test_missing_canonical_distinct_donor_cannot_silently_reuse_source(self):
        entries = self.manifest()['entries']
        only_same = {'entries': [entry for entry in entries if entry['key'] in ('a', 'd')]}
        with self.assertRaisesRegex(ValueError, 'No canonical-distinct contiguous VAL donor'):
            training.evaluation_windows(only_same)

    def test_completed_hierarchical_training_is_idempotent_with_full_planner_identity(self):
        with tempfile.TemporaryDirectory(prefix='lightsem-dynamic-complete-test-') as directory:
            root = Path(directory)
            protocol = {'inputs': {}, 'sources': {}, 'arms': ['D8', 'D8_flat'], 'seeds': [0],
                        'training': {'steps': 1000}}
            training.authenticated_json(root/'learning/protocol.json', protocol)
            checkpoint = root/'learning/runs/planner/seed_0/final.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'Synthetic identity fixture; never deserialized')
            for branch in ('global', 'local'):
                training.authenticated_json(root/f'learning/plans/seed_0/{branch}/index.json', {'branch': branch})
            for arm, branch in (('D8', 'global'), ('D8_flat', 'local')):
                with self.subTest(arm=arm):
                    identity = training._run_identity(root, arm, 0)
                    self.assertEqual(identity['planner_sha256'], training.digest(checkpoint))
                    self.assertEqual(identity['plan_index_sha256'],
                                     training.digest(root/f'learning/plans/seed_0/{branch}/index.json'))
                    report = {'identity': identity, 'artifacts': {}, 'steps': 1000}
                    training.authenticated_json(root/f'learning/runs/{arm}/seed_0/report.json', report)
                    # Completed reuse must return before runtime/model/GPU initialization.
                    with patch.object(training, 'require_space', return_value=100.), \
                         patch.object(training, 'TrainingRuntime', side_effect=AssertionError('GPU path reached')):
                        self.assertEqual(training.train(root, arm, 0), report)
                        self.assertEqual(training.train(root, arm, 0, resume=True), report)
            checkpoint.write_bytes(b'Changed planner checkpoint')
            with patch.object(training, 'require_space', return_value=100.):
                with self.assertRaisesRegex(ValueError, 'Completed run identity changed'):
                    training.train(root, 'D8_flat', 0)


class EvaluationActivityTests(unittest.TestCase):
    def test_constants_jitter_and_independent_variation_remain_distinct(self):
        constant = np.full((8, 3), .7, np.float32)
        inactive = training.temporal_activity(constant)
        self.assertTrue(inactive['exact_constant'])
        self.assertEqual(inactive['exact_constant_channels'], [0, 1, 2])
        self.assertEqual(inactive['mean_channel_std'], 0.)
        self.assertEqual(inactive['effective_rank'], 0.)
        alternating = np.tile([-1., 1.], 4)
        jitter = np.stack((alternating, alternating*2, alternating*3), -1)
        repeated = training.temporal_activity(jitter)
        self.assertFalse(repeated['exact_constant'])
        self.assertGreater(repeated['mean_channel_std'], 0)
        self.assertAlmostEqual(repeated['effective_rank'], 1., places=6)
        # A distinct temporal coordinate adds rank; duplicating outputs does not.
        distinct = np.stack((alternating, np.tile([-2., -2., 2., 2.], 2), np.full(8, 5.)), -1)
        varied = training.temporal_activity(distinct, scale=np.array([1., 2., 1.]))
        self.assertEqual(varied['exact_constant_channels'], [2])
        np.testing.assert_allclose(varied['channel_variance'], [1., 4., 0.], atol=0, rtol=0)
        self.assertAlmostEqual(varied['effective_rank'], 2., places=6)
        duplicate = training.temporal_activity(np.tile(jitter, (1, 5)))
        self.assertAlmostEqual(duplicate['effective_rank'], repeated['effective_rank'], places=6)

    def test_changed_input_does_not_establish_changed_or_active_output(self):
        source = np.zeros((8, 4), np.float32)
        donor = np.ones_like(source)
        input_response = training.response_activity(donor, source)
        output_response = training.response_activity(source, source.copy())
        self.assertEqual(input_response['rms_difference'], 1.)
        self.assertEqual(input_response['max_abs_difference'], 1.)
        self.assertFalse(input_response['exact_equal'])
        self.assertEqual(output_response['rms_difference'], 0.)
        self.assertTrue(output_response['exact_equal'])
        # Different constant outputs are sensitive to input but remain inactive.
        self.assertTrue(training.temporal_activity(source)['exact_constant'])
        self.assertTrue(training.temporal_activity(donor)['exact_constant'])

    def test_root_travel_is_separate_from_articulation_and_native_temporal_variance(self):
        native = held(31)[0].numpy()
        fixed_world = joints_from_features(torch.from_numpy(native)).numpy().astype(np.float64)
        stationary = training.body_activity(native, fixed_world)
        self.assertTrue(stationary['finite'])
        self.assertTrue(stationary['world_exact_constant'])
        self.assertTrue(stationary['root_relative_exact_constant'])
        self.assertEqual(stationary['world_hold_occupancy'], 1.)
        self.assertEqual(stationary['root_relative_hold_occupancy'], 1.)
        # Binary-exact displacement keeps relative geometry exactly unchanged.
        native[:, 0] = .9375
        travel = fixed_world.copy()
        travel[..., 0] += (np.arange(31)/32.)[:, None]
        moving = training.body_activity(native, travel)
        self.assertTrue(moving['native']['exact_constant'])
        self.assertFalse(moving['world_exact_constant'])
        self.assertTrue(moving['root_relative_exact_constant'])
        self.assertEqual(moving['interval_count'], 30)
        self.assertEqual(moving['root_speed_quantiles_m_s']['p50'], .9375)
        self.assertEqual(moving['world_joint_speed_quantiles_m_s']['p50'], .9375)
        self.assertEqual(moving['root_relative_joint_speed_quantiles_m_s']['max'], 0.)
        self.assertEqual(moving['world_hold_occupancy'], 0.)
        self.assertEqual(moving['root_relative_hold_occupancy'], 1.)

    def test_alternating_articulation_is_active_but_low_rank_with_stationary_root(self):
        native = held(31)[0]
        angle = torch.tensor([-.2, .2]*16)[:31]
        zero = torch.zeros_like(angle)
        native[:, 3+16*6:3+17*6] = torch.stack(
            (angle.cos(), -angle.sin(), zero, angle.sin(), angle.cos(), zero), -1)
        world = joints_from_features(native)
        activity = training.body_activity(native.numpy(), world.numpy(), native_scale=np.ones(139))
        self.assertTrue(activity['finite'])
        self.assertFalse(activity['world_exact_constant'])
        self.assertFalse(activity['root_relative_exact_constant'])
        self.assertAlmostEqual(activity['native']['effective_rank'], 1., places=6)
        self.assertEqual(activity['root_speed_quantiles_m_s']['max'], 0.)
        self.assertGreater(activity['root_relative_joint_speed_quantiles_m_s']['max'], 1.)
        self.assertEqual(activity['world_hold_occupancy'], 0.)
        self.assertEqual(activity['root_relative_hold_occupancy'], 0.)

    def test_one_frame_and_nonfinite_body_do_not_become_inactivity_verdicts(self):
        native = held(1)[0].numpy()
        world = joints_from_features(torch.from_numpy(native)).numpy()
        single = training.body_activity(native, world)
        self.assertEqual(single['interval_count'], 0)
        self.assertFalse(single['native']['temporal_observation_available'])
        self.assertIsNone(single['world_joint_speed_quantiles_m_s'])
        self.assertIsNone(single['world_hold_occupancy'])
        self.assertIsNone(single['root_relative_hold_occupancy'])
        native[0, 0] = float('nan')
        failure = training.body_activity(native, world)
        self.assertFalse(failure['finite'])
        self.assertFalse(failure['native_finite'])
        self.assertTrue(failure['world_finite'])
        self.assertNotIn('world_hold_occupancy', failure)
        self.assertIn('failure artifact', failure['metrics_unavailable_reason'])
        for fps in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                training.body_activity(held(1)[0].numpy(), world, fps=fps)

    def test_activity_rejects_invalid_shapes_values_and_scales(self):
        for invalid in (np.zeros((0, 2)), np.zeros((3, 0)), np.ones(3), np.full((3, 2), np.nan)):
            with self.assertRaises(ValueError):
                training.temporal_activity(invalid)
        for scale in ([1.], [1., 0.], [1., float('nan')]):
            with self.assertRaises(ValueError):
                training.temporal_activity(np.ones((3, 2)), scale=scale)
        with self.assertRaises(ValueError):
            training.response_activity(np.zeros((3, 2)), np.zeros((3, 1)))
        with self.assertRaises(ValueError):
            training.response_activity(np.zeros((3, 2)), np.full((3, 2), np.nan))


class EvaluationArtifactTests(unittest.TestCase):
    def test_array_hash_includes_shape_dtype_and_values_but_not_layout(self):
        source = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2]
        self.assertFalse(source.flags.c_contiguous)
        self.assertEqual(training.array_sha256(source), training.array_sha256(source.copy()))
        self.assertEqual(training.array_sha256(source), training.array_sha256(torch.from_numpy(source)))
        self.assertNotEqual(training.array_sha256(source), training.array_sha256(source.reshape(-1)))
        self.assertNotEqual(training.array_sha256(source), training.array_sha256(source.astype(np.float64)))
        changed = source.copy()
        changed[0, 0] += 1
        self.assertNotEqual(training.array_sha256(source), training.array_sha256(changed))
        with self.assertRaises(ValueError):
            training.array_sha256(np.array([{'key': 'value'}], dtype=object))

    def test_artifact_retains_actual_body_inputs_and_noise_and_exact_retry(self):
        native = held(4)[0].requires_grad_()
        arrays = {'native': native, 'world': joints_from_features(native),
                  'time': np.arange(4)/30., 'normalized_input': np.arange(16, dtype=np.float32).reshape(4, 4),
                  'initial_noise': torch.zeros(4, 139)}
        metadata = {'arm': 'D7', 'seed': 1, 'condition': 'donor', 'song_key': 'synthetic_fixture'}
        with tempfile.TemporaryDirectory(prefix='lightsem-evaluation-artifact-test-') as directory:
            path = Path(directory)/'condition.npz'
            result = training.save_evaluation_artifact(path, arrays, metadata)
            self.assertEqual(result['sha256'], training.digest(path))
            self.assertEqual(result['metadata_sha256'], training.digest(path.with_suffix('.json')))
            receipt = training.read_authenticated(path.with_suffix('.json'))
            self.assertEqual(receipt['kind'], 'DynamicDanceEvaluation')
            self.assertEqual(receipt['identity']['metadata'], metadata)
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(arrays))
                for key, value in arrays.items():
                    expected = value.detach().numpy() if isinstance(value, torch.Tensor) else value
                    np.testing.assert_array_equal(saved[key], expected)
                    self.assertEqual(result['arrays'][key], training.array_sha256(saved[key]))
            self.assertEqual(training.save_evaluation_artifact(path, arrays, metadata), result)
            self.assertEqual(training.digest(path), result['sha256'])
            changed = {**arrays, 'normalized_input': arrays['normalized_input']+1}
            with self.assertRaisesRegex(ValueError, 'Refusing to replace changed evaluation artifact'):
                training.save_evaluation_artifact(path, changed, metadata)
            with self.assertRaisesRegex(ValueError, 'Refusing to replace changed evaluation artifact'):
                training.save_evaluation_artifact(path, arrays, {**metadata, 'condition': 'correct'})
            self.assertEqual(training.digest(path), result['sha256'])

    def test_tampering_or_partial_artifacts_are_rejected_and_preserved(self):
        arrays = {'raw': np.arange(6, dtype=np.float32).reshape(2, 3)}
        with tempfile.TemporaryDirectory(prefix='lightsem-evaluation-tamper-test-') as directory:
            root = Path(directory)
            damaged = root/'damaged.npz'
            training.save_evaluation_artifact(damaged, arrays, {'case': 'tamper'})
            with damaged.open('ab') as stream:
                stream.write(b'tampered arrays')
            damaged_hash = training.digest(damaged)
            with self.assertRaisesRegex(RuntimeError, 'Frozen input changed'):
                training.save_evaluation_artifact(damaged, arrays, {'case': 'tamper'})
            self.assertEqual(training.digest(damaged), damaged_hash)
            changed_metadata = root/'metadata.npz'
            training.save_evaluation_artifact(changed_metadata, arrays, {'case': 'metadata'})
            with changed_metadata.with_suffix('.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(RuntimeError, 'Artifact digest mismatch'):
                training.save_evaluation_artifact(changed_metadata, arrays, {'case': 'metadata'})
            partial = root/'partial.npz'
            partial.write_bytes(b'partial evidence')
            with self.assertRaisesRegex(ValueError, 'Partial evaluation artifact preserved'):
                training.save_evaluation_artifact(partial, arrays, {})
            self.assertEqual(partial.read_bytes(), b'partial evidence')

    def test_nonfinite_arrays_are_retained_as_failure_evidence(self):
        native = held(3)[0].numpy()
        native[1, 7] = float('nan')
        native[2, 8] = float('inf')
        with tempfile.TemporaryDirectory(prefix='lightsem-evaluation-failure-test-') as directory:
            path = Path(directory)/'failure.npz'
            result = training.save_evaluation_artifact(path, {'native': native}, {'finite': False})
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['native'], native)
                self.assertEqual(training.array_sha256(saved['native']), result['arrays']['native'])
            self.assertEqual(training.save_evaluation_artifact(path, {'native': native.copy()}, {'finite': False}), result)


class InferencePlanTelemetryTests(unittest.TestCase):
    def test_exact_full_and_coarse_plan_values_are_saved_without_state_changes(self):
        from lightsem.experiments.dynamic_dance.contracts import DancePlan
        coarse = plan_condition(torch.arange(3*34, dtype=torch.float32).reshape(3, 34)/100).numpy()
        # A shifted diagnostic must expose the rolled values actually used.
        coarse = np.roll(coarse, 1, axis=0)
        coarse_time = np.arange(len(coarse))*.5
        time = np.arange(17)/30.
        plan = np.stack([np.interp(time, coarse_time, coarse[:, i]) for i in range(34)], -1).astype(np.float32)
        original_plan, original_coarse = plan.copy(), coarse.copy()
        source = {'arm': 'D8', 'plan_override': 'shift', 'timeline_path': '/frozen/song.npz',
                  'analyzed_waveform_sha256': 'a'*64, 'checkpoint_sha256': 'b'*64}
        original_source = copy.deepcopy(source)
        torch_rng, numpy_rng = torch.get_rng_state().clone(), np.random.get_state()
        with tempfile.TemporaryDirectory(prefix='lightsem-plan-telemetry-') as directory:
            path = Path(directory)/'plan_telemetry.npz'
            added = training.save_inference_plan_telemetry(path, time, plan, arm='D8',
                        source_evidence=source, coarse_time=coarse_time, coarse_plan=coarse)
            self.assertTrue(added['planner_active'])
            self.assertEqual(added['plan_telemetry']['sha256'], training.digest(path))
            metadata = training.read_authenticated(path.with_suffix('.json'))
            self.assertEqual(metadata['source_evidence'], source)
            self.assertEqual(metadata['channel_slices']['duration'], [14, 34])
            self.assertIn('post-intervention', metadata['coarse_values_role'])
            with np.load(path, allow_pickle=False) as saved:
                for prefix, values, times in [('', plan, time), ('coarse_', coarse, coarse_time)]:
                    expected = {'time': times, 'plan': values, 'log_speed': values[:, :5],
                                'support': values[:, 5:9], 'hold': values[:, 9:14],
                                'duration': values[:, 14:].reshape(len(values), 5, 4)}
                    for key, value in expected.items():
                        name = prefix+key
                        np.testing.assert_array_equal(saved[name], value)
                        self.assertEqual(saved[name].dtype, value.dtype)
                        self.assertEqual(metadata['arrays'][name]['sha256'], training.array_sha256(value))
            artifact = DancePlan(time=time, source_time=time.copy(), effort=plan[:, :5], metadata={**source, **added})
            artifact.save(Path(directory)/'plan.npz')
            readback = DancePlan.load(Path(directory)/'plan.npz')
            np.testing.assert_array_equal(readback.effort, original_plan[:, :5])
            self.assertEqual(readback.metadata['plan_telemetry']['path'], str(path.resolve()))
            with self.assertRaises(FileExistsError):
                training.save_inference_plan_telemetry(path, time, plan, arm='D8',
                    source_evidence=source, coarse_time=coarse_time, coarse_plan=coarse)
        np.testing.assert_array_equal(plan, original_plan)
        np.testing.assert_array_equal(coarse, original_coarse)
        self.assertEqual(source, original_source)
        self.assertTrue(torch.equal(torch_rng, torch.get_rng_state()))
        current_numpy = np.random.get_state()
        self.assertEqual(numpy_rng[0], current_numpy[0])
        np.testing.assert_array_equal(numpy_rng[1], current_numpy[1])
        self.assertEqual(numpy_rng[2:], current_numpy[2:])

    def test_no_planner_slots_are_explicit_and_never_fabricated_decisions(self):
        with tempfile.TemporaryDirectory(prefix='lightsem-no-planner-telemetry-') as directory:
            for arm in ('D7', 'continued'):
                path = Path(directory)/(arm+'.npz')
                plan = np.zeros((3, 34), np.float32)
                added = training.save_inference_plan_telemetry(path, np.arange(3)/30., plan,
                                                               arm=arm, source_evidence={'arm': arm})
                self.assertFalse(added['planner_active'])
                self.assertIn('not conditioning', added['effort_role'])
                metadata = training.read_authenticated(path.with_suffix('.json'))
                self.assertIn('no planning decisions', metadata['values_role'])
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['plan'], plan)
                    self.assertFalse(any(key.startswith('coarse_') for key in saved.files))
            with self.assertRaisesRegex(ValueError, 'unused zero conditioning slots'):
                training.save_inference_plan_telemetry(Path(directory)/'invented.npz', np.arange(3)/30.,
                    np.ones((3, 34), np.float32), arm='D7', source_evidence={})

    def test_metadata_telemetry_call_follows_generation_and_never_updates_provenance(self):
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(training.infer))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'save_inference_plan_telemetry']
        self.assertEqual(len(calls), 1)
        call = calls[0]
        metadata = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict) and call in node.values)
        self.assertIsNone(metadata.keys[metadata.values.index(call)])
        generation = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Attribute) and node.func.attr == 'generate')
        self.assertGreater(call.lineno, generation.lineno)
        source = next(keyword.value for keyword in call.keywords if keyword.arg == 'source_evidence')
        self.assertIsInstance(source, ast.Name)
        self.assertEqual(source.id, 'provenance')
        # The call is contained only by copied metadata, not provenance.update.
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'update':
                self.assertNotIn(call, list(ast.walk(node)))


if __name__ == '__main__':
    unittest.main()

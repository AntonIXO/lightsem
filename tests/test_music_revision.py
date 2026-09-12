"""Synthetic CPU engineering checks only; no fitting or dataset/checkpoint access."""
import copy
import inspect
import unittest

import numpy as np
import torch

from lightsem.experiments.motion_study.music_model import MusicConfig, MusicModel
from lightsem.experiments.motion_study.music_revision_model import (
    MusicRevisionModel, PhysicalWeights, masked_native_losses,
)
from lightsem.experiments.motion_study.music_revision_data import MusicRevisionResident
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


def model(arm='masked_native', weights=PhysicalWeights()):
    torch.manual_seed(17)
    result = MusicRevisionModel(MusicConfig(width=8, frames=8, dilations=(1, 2), steps=2),
                                normalizers(), arm=arm, weights=weights)
    # Avoid making equivalence tests trivial through the native zero-output init.
    with torch.no_grad():
        result.output.weight.normal_(std=.02)
        result.output.bias.normal_(std=.01)
    return result


def fixture(m, lengths=(3, 8), frames=8):
    generator = torch.Generator(device='cpu').manual_seed(43)
    audio = torch.randn(len(lengths), frames, 92, generator=generator)
    target = held(frames, len(lengths))
    target[..., 0] = .2
    target[..., 3:135] += .02*torch.randn(len(lengths), frames, 132, generator=generator)
    valid = torch.arange(frames)[None] < torch.tensor(lengths)[:, None]
    noise = m.noise(len(lengths), frames, generator=generator, device='cpu')
    time = torch.linspace(.2, .7, len(lengths))
    return audio, target, valid, noise, time


def records():
    result = []
    for key, canonical, length in [('a', 'song1', 1), ('b', 'song1', 3), ('c', 'song2', 12)]:
        result.append({'key': key, 'canonical_id': canonical, 'partition': 'train',
                       'audio': np.arange(length*92, dtype=np.float32).reshape(length, 92),
                       'features': held(length)[0].numpy()})
    # Missing arrays are intentional: non-TRAIN data must not be inspected.
    result.append({'key': 'closed', 'canonical_id': 'test', 'partition': 'test'})
    return result


def resident(values=None):
    return MusicRevisionResident.from_recordings(records() if values is None else values,
        {'mean': np.ones(92), 'std': np.full(92, 2.)}, frames=8, device='cpu')


class RevisionModelTests(unittest.TestCase):
    def test_masked_native_all_valid_original_parity_and_gradients(self):
        revision = model()
        original = MusicModel(revision.config, normalizers())
        original.load_state_dict(revision.state_dict(), strict=True)
        args = fixture(revision, (8, 8))
        actual = revision.losses_from_draws(*args)
        expected = original.losses_from_draws(*args)
        self.assertEqual(actual.keys(), expected.keys())
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], atol=2e-7, rtol=2e-7)
        actual['total'].backward()
        expected['total'].backward()
        for a, b in zip(revision.parameters(), original.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-6)

    def test_zero_physical_weights_exact_native_parity(self):
        baseline, physical = model(), model('physical')
        args = fixture(baseline)
        before, after = baseline.losses_from_draws(*args), physical.losses_from_draws(*args)
        for name in before:
            self.assertTrue(torch.equal(before[name], after[name]), name)
        self.assertEqual(len(after), 8)
        before['total'].backward()
        after['total'].backward()
        for a, b in zip(baseline.parameters(), physical.parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad))

    def test_fixed_weights_are_explicit_and_terms_separate(self):
        weights = PhysicalWeights(.001, .002, .003, .004, .005)  # Test fixture, not a recipe.
        m = model('physical', weights)
        losses = m.losses_from_draws(*fixture(m))
        expected = losses['native']+.05*losses['velocity']
        for name, value in vars(weights).items():
            expected = expected+value*losses['physical_'+name]
        self.assertTrue(torch.equal(expected, losses['total']))
        self.assertEqual(m.config_dict()['physical_weights'], vars(weights))
        losses['total'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in m.parameters()), 0)
        for invalid in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                PhysicalWeights(position=invalid)
        with self.assertRaises(ValueError):
            model(weights=PhysicalWeights(position=1))
        with self.assertRaises(ValueError):
            MusicRevisionModel(MusicConfig(arm='regression'), normalizers(), arm='physical')

    def test_padding_content_invariance_nonzero_network_and_all_losses(self):
        weights = PhysicalWeights(.001, .001, .001, .001, .001)
        for arm in ('masked_native', 'physical'):
            m = model(arm, weights if arm == 'physical' else PhysicalWeights())
            args = fixture(m)
            original = m.losses_from_draws(*args)
            original['total'].backward()
            gradients = [p.grad.clone() for p in m.parameters()]
            m.zero_grad(set_to_none=True)
            audio, target, valid, noise, time = [v.clone() for v in args]
            for value in (audio, target, noise):
                value[~valid] = float('nan')
            audio.requires_grad_()
            noise.requires_grad_()
            actual = m.losses_from_draws(audio, target, valid, noise, time)
            for name in original:
                self.assertTrue(torch.equal(actual[name], original[name]), name)
            actual['total'].backward()
            for p, gradient in zip(m.parameters(), gradients):
                self.assertTrue(torch.equal(p.grad, gradient))
            self.assertEqual(audio.grad[~valid].abs().sum().item(), 0)
            self.assertEqual(noise.grad[~valid].abs().sum().item(), 0)

    def test_native_loss_extension_and_boundary_derivative_exclusion(self):
        prediction = torch.randn(1, 3, 139, generator=torch.Generator().manual_seed(9), requires_grad=True)
        desired = held(3)
        valid = torch.ones(1, 3, dtype=torch.bool)
        base = masked_native_losses(prediction, desired, torch.ones(139), valid)
        gradient, = torch.autograd.grad(base['total'], prediction)
        pad = torch.full((1, 2, 139), float('nan'))
        extended = torch.cat((pad, prediction.detach(), pad), 1).requires_grad_()
        mask = torch.tensor([[False, False, True, True, True, False, False]])
        actual = masked_native_losses(extended, torch.cat((pad, desired, pad), 1), torch.ones(139), mask)
        for name in base:
            torch.testing.assert_close(actual[name], base[name])
        actual['total'].backward()
        torch.testing.assert_close(extended.grad[:, 2:5], gradient)
        self.assertEqual(extended.grad[~mask].abs().sum().item(), 0)
        # Large error change across an invalid gap is never a derivative target.
        p = torch.zeros(1, 5, 139)
        p[:, 3:] = 100
        losses = masked_native_losses(p, torch.zeros_like(p), torch.ones(139),
                                      torch.tensor([[True, True, False, True, True]]))
        self.assertEqual(losses['velocity'].item(), 0)

    def test_empty_and_one_frame_masks_are_finite(self):
        m = model('physical', PhysicalWeights(.001, .001, .001, .001, .001))
        args = fixture(m, (1,), frames=1)
        losses = m.losses_from_draws(*args)
        self.assertEqual(losses['velocity'].item(), 0)
        self.assertEqual(losses['physical_velocity'].item(), 0)
        self.assertEqual(losses['physical_planted_foot_velocity'].item(), 0)
        audio, target, valid, noise, time = args
        losses = m.losses_from_draws(audio, target, torch.zeros_like(valid), noise, time)
        self.assertTrue(all(v.item() == 0 for v in losses.values()))
        losses['total'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() and p.grad.abs().sum() == 0 for p in m.parameters()))

    def test_clean_endpoint_at_time_one_and_no_reference_gradient(self):
        m = model('physical')
        audio, target, valid, noise, time = fixture(m, (8, 8))
        target.requires_grad_()
        result = m.losses_from_draws(audio, target, valid, noise, torch.ones_like(time))
        for name in ('position', 'velocity', 'root_displacement'):
            self.assertLess(result['physical_'+name].item(), 1e-8)
        sum(result.values()).backward()
        self.assertIsNone(target.grad)

    def test_generation_inherits_audio_only_api_and_matches(self):
        a, b = model(), model('physical', PhysicalWeights(position=.001))
        self.assertIs(MusicRevisionModel.generate, MusicModel.generate)
        self.assertIs(MusicRevisionModel.vector_field, MusicModel.vector_field)
        self.assertIs(MusicRevisionModel.noise, MusicModel.noise)
        self.assertIs(MusicRevisionModel.draws, MusicModel.draws)
        self.assertFalse({'target', 'reference', 'valid', 'contacts'} & set(inspect.signature(a.generate).parameters))
        audio = fixture(a)[0]
        first, second = a.generate(audio, seed=23), b.generate(audio, seed=23)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_compile_forward_only_parity_and_eager_safety(self):
        eager, compiled = model('physical', PhysicalWeights(position=.001)), model('physical', PhysicalWeights(position=.001))
        graphs = []
        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward
        fn = compiled.loss_function(compile_vector_field=True, backend=backend)
        args = fixture(eager)
        expected, actual = eager.losses_from_draws(*args), fn(*args)
        self.assertEqual(len(graphs), 1)  # fullgraph=True also rejects forward breaks.
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        actual['total'].backward()
        expected['total'].backward()
        for a, b in zip(eager.parameters(), compiled.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
        bad = list(args)
        bad[-1] = torch.full_like(bad[-1], 1.1)
        with self.assertRaises(ValueError):
            fn(*bad)
        with torch.no_grad():
            compiled.std[0] = 0
        with self.assertRaises(ValueError):
            fn(*args)
        self.assertEqual(len(graphs), 1)

    def test_aot_eager_compiled_forward_backward_fp32(self):
        m = model('physical', PhysicalWeights(.001, .001, .001, .001, .001))
        fn = m.loss_function(compile_vector_field=True, backend='aot_eager')
        # This CPU's oneDNN does not support BF16 convolution backward. The
        # independent physical-loss suite checks BF16 -> FP32 FK/gradients.
        result = fn(*fixture(m))
        self.assertTrue(all(value.dtype == torch.float32 for value in result.values()))
        result['total'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_input_validations_are_not_optional(self):
        m = model()
        args = list(fixture(m))
        args[2] = args[2].float()
        with self.assertRaises(ValueError):
            m.losses_from_draws(*args)
        args = list(fixture(m))
        args[0][0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            m.losses_from_draws(*args)


class RevisionDataTests(unittest.TestCase):
    def test_every_short_train_record_available_once_without_repetition(self):
        data = resident()
        self.assertEqual(data.coverage['recordings'], 3)
        self.assertEqual(data.coverage['canonical_songs'], 2)
        self.assertEqual(data.coverage['short_recordings'], 2)
        self.assertEqual(data.lengths.tolist(), [1, 3, 12])
        for i, source in enumerate(records()[:3]):
            start, length = int(data.starts[i]), int(data.lengths[i])
            torch.testing.assert_close(data.y[start:start+length], torch.from_numpy(source['features']))
            torch.testing.assert_close(data.x[start:start+length], torch.from_numpy((source['audio']-1)/2))
            self.assertTrue(data.b[start:start+length].all())
            if length < data.frames:
                self.assertEqual(data.x[start+length:start+8].abs().sum().item(), 0)
                self.assertEqual(data.y[start+length:start+8].abs().sum().item(), 0)
                self.assertFalse(data.b[start+length:start+8].any())
        indices = data.draw_indices(256, 8, np.random.default_rng(7))
        self.assertEqual(set(data.b[indices].sum(1).tolist()), {1, 3, 8})
        for row in indices:
            self.assertEqual(np.diff(row).tolist(), [1]*7)
            self.assertTrue(any(start <= row[0] <= row[-1] < start+max(length, 8)
                                for start, length in zip(data.starts, data.lengths)))

    def test_canonical_equal_probabilities_and_order_independent_replay(self):
        a, b = resident(), resident(list(reversed(records())))
        probs = {row['key']: row['sampling_probability'] for row in a.coverage['rows']}
        self.assertEqual(probs, {'a': .25, 'b': .25, 'c': .5})
        self.assertEqual(a.coverage['expected_valid_frames_per_crop'], 5.)
        rng_a, rng_b = np.random.default_rng(19), np.random.default_rng(19)
        for _ in range(3):
            self.assertTrue(np.array_equal(a.draw_indices(16, 8, rng_a), b.draw_indices(16, 8, rng_b)))
        state = copy.deepcopy(rng_a.bit_generator.state)
        first = a.draw_indices(12, 8, rng_a)
        rng_a.bit_generator.state = state
        self.assertTrue(np.array_equal(first, a.draw_indices(12, 8, rng_a)))

    def test_batchstream_cpu_depth_zero_and_paired_exposure(self):
        data = resident()
        baseline, physical = model(), model('physical')
        rng_a, rng_b = np.random.default_rng(8), np.random.default_rng(8)
        with BatchStream(data, 4, 8, rng_a, depth=0) as first, BatchStream(data, 4, 8, rng_b, depth=0) as second:
            for step in range(3):
                a, b = first.next(), second.next()
                self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))
                self.assertEqual(a[2].dtype, torch.bool)
                # In separate runs use identical saved global Torch RNG state.
                torch.manual_seed(90+step)
                noise_a, time_a = baseline.draws(a[0])
                torch.manual_seed(90+step)
                noise_b, time_b = physical.draws(b[0])
                self.assertTrue(torch.equal(noise_a, noise_b))
                self.assertTrue(torch.equal(time_a, time_b))
                la = baseline.losses_from_draws(*a, noise_a, time_a)
                lb = physical.losses_from_draws(*b, noise_b, time_b)
                self.assertTrue(torch.equal(la['total'], lb['total']))

    def test_no_empty_or_misaligned_record_silently_dropped(self):
        values = records()
        values[0]['features'] = np.zeros((0, 139), np.float32)
        with self.assertRaises(ValueError):
            resident(values)
        values = records()
        values[0]['audio'] = np.zeros((2, 92), np.float32)
        with self.assertRaises(ValueError):
            resident(values)
        with self.assertRaises(ValueError):
            resident().draw_indices(1, 9, np.random.default_rng(1))


if __name__ == '__main__':
    unittest.main()

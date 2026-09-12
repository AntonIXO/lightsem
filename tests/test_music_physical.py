"""CPU-only counterexamples for opt-in physical losses; no model training."""
import unittest

import torch

from lightsem.experiments.motion_study.geometry import FPS, joints_from_features
from lightsem.experiments.motion_study.music_physical import (
    clean_estimate, motion_kinematics, physical_losses,
)


def held(frames=8, batch=1):
    value = torch.zeros(batch, frames, 139)
    value[..., 2] = 1.1  # Entire canonical skeleton is above floor.
    value[..., 3:135] = torch.tensor([1., 0, 0, 0, 1, 0]).repeat(22)
    value[..., 135:] = 1
    return value


def losses(generated, reference, valid=None, mean=None, std=None):
    if valid is None:
        valid = torch.ones(generated.shape[:2], dtype=torch.bool)
    return physical_losses(generated, reference, torch.zeros(139) if mean is None else mean,
                           torch.ones(139) if std is None else std, valid)


class PhysicalLossTests(unittest.TestCase):
    def test_clean_estimate_algebra(self):
        generator = torch.Generator().manual_seed(42)
        noise = torch.randn(3, 5, 139, generator=generator)
        target = torch.randn(3, 5, 139, generator=generator)
        time = torch.tensor([0., .5, 1.])
        state = (1-time[:, None, None])*noise + time[:, None, None]*target
        actual = clean_estimate(state, target-noise, time)
        torch.testing.assert_close(actual, target, atol=3e-7, rtol=1e-5)
        for t in (0., .5, 1.):
            torch.testing.assert_close(clean_estimate((1-t)*noise+t*target, target-noise, t), target)
        torch.testing.assert_close(clean_estimate(state, target-noise, time[:, None, None]), target)

    def test_identity_and_fk_convention(self):
        reference = held()
        for value in losses(reference, reference).values():
            self.assertEqual(value.item(), 0)
        valid = torch.ones(1, 8, dtype=torch.bool)
        reference[..., 0] = .2
        actual = motion_kinematics(reference, valid)
        torch.testing.assert_close(actual['positions'], joints_from_features(reference))
        torch.testing.assert_close(actual['velocities'], torch.diff(joints_from_features(reference), dim=1)*FPS)

    def test_planted_sliding_and_hidden_generated_contacts(self):
        reference = held()
        generated = reference.clone()
        generated[..., 0] = .3
        before = losses(generated, reference)
        self.assertAlmostEqual(before['planted_foot_velocity'].item(), .3**2, places=6)
        generated[..., 135:] = 0  # Cannot evade the reference-selected training loss.
        after = losses(generated, reference)
        for name in before:
            torch.testing.assert_close(before[name], after[name])
        reference[..., 135:] = 0
        self.assertEqual(losses(generated, reference)['planted_foot_velocity'].item(), 0)

    def test_reference_contact_requires_both_endpoints(self):
        reference = held(3)
        reference[:, 1, 135:] = 0
        generated = held(3)
        generated[..., 0] = 1
        self.assertEqual(losses(generated, reference)['planted_foot_velocity'].item(), 0)

    def test_root_drift_accumulates_without_shrinking_reference_travel(self):
        reference = held(960)
        generated = reference.clone()
        generated[..., 0] += .1
        result = losses(generated, reference)
        elapsed = torch.arange(960) / FPS
        self.assertAlmostEqual(result['root_displacement'].item(), ((.1*elapsed)**2).mean().item(), places=5)
        root = motion_kinematics(generated, torch.ones(1, 960, dtype=torch.bool))['root']
        self.assertAlmostEqual(root[0, -1, 0].item(), 3.1966667, places=5)
        reference[..., 0] = 1.2
        self.assertEqual(losses(reference, reference)['root_displacement'].item(), 0)

    def test_rotation_row_scaling_is_geometry_equivalent(self):
        reference = held()
        reference[..., 3:135] += .05 * torch.randn(1, 8, 132, generator=torch.Generator().manual_seed(7))
        reference[..., 135:] = 0  # Moving reference is not a planted-foot fixture.
        generated = reference.clone()
        generated[..., 3:135] *= 2
        self.assertGreater((generated-reference).square().mean().item(), 0)
        before, after = losses(reference, reference), losses(generated, reference)
        for name in before:
            torch.testing.assert_close(before[name], after[name], atol=1e-10, rtol=1e-5)

    def test_floor_penetration_is_separate(self):
        reference = held()
        generated = reference.clone()
        generated[..., 2] = .1
        result = losses(generated, reference)
        self.assertGreater(result['floor_penetration'].item(), 0)
        self.assertEqual(result['root_displacement'].item(), 0)
        self.assertEqual(result['velocity'].item(), 0)

    def test_finite_gradients_and_fp32_under_autocast(self):
        reference = held()
        generated = reference.clone()
        generated[..., 0] = .25
        generated[..., 2] = .2
        generated = generated.to(torch.bfloat16).requires_grad_()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = losses(generated, reference)
            kinematics = motion_kinematics(generated, torch.ones(1, 8, dtype=torch.bool))
        self.assertTrue(all(value.dtype == torch.float32 for value in result.values()))
        self.assertEqual(kinematics['positions'].dtype, torch.float32)
        for value in result.values():
            gradient, = torch.autograd.grad(value, generated, retain_graph=True)
            self.assertTrue(torch.isfinite(gradient).all())
        gradient, = torch.autograd.grad(sum(result.values()), generated)
        self.assertGreater(gradient[..., :135].abs().sum().item(), 0)
        self.assertEqual(gradient[..., 135:].abs().sum().item(), 0)

    def test_zero_rotation_fallback_gradients(self):
        reference = held()
        generated = reference.clone()
        generated[..., 3:135] = 0
        generated.requires_grad_()
        sum(losses(generated, reference).values()).backward()
        self.assertTrue(torch.isfinite(generated.grad).all())

    def test_padding_invariance_and_no_boundary_gradient(self):
        reference = held(4)
        generated = reference.clone()
        generated[..., 0] = .2
        generated[..., 2] = .3
        generated.requires_grad_()
        base = losses(generated, reference)
        base_grad, = torch.autograd.grad(sum(base.values()), generated)
        # NaNs in padding must not affect values or gradients, including prefix padding.
        pad = torch.full((1, 3, 139), float('nan'))
        padded = torch.cat((pad, generated.detach(), pad), 1).requires_grad_()
        padded_ref = torch.cat((pad, reference, pad), 1)
        valid = torch.tensor([[False]*3 + [True]*4 + [False]*3])
        actual = losses(padded, padded_ref, valid)
        for name in base:
            torch.testing.assert_close(base[name], actual[name])
        sum(actual.values()).backward()
        torch.testing.assert_close(padded.grad[:, 3:7], base_grad)
        self.assertEqual(padded.grad[~valid].abs().sum().item(), 0)
        self.assertEqual(padded.grad[0, 6, :2].abs().sum().item(), 0)

    def test_one_frame_and_all_padding(self):
        reference = held(1, batch=2)
        generated = reference.clone()
        generated[0, :, 2] -= .2
        valid = torch.tensor([[True], [False]])
        result = losses(generated, reference, valid)
        self.assertAlmostEqual(result['position'].item(), .04, places=6)
        self.assertEqual(result['velocity'].item(), 0)
        self.assertEqual(result['planted_foot_velocity'].item(), 0)
        generated.requires_grad_()
        empty = losses(generated, reference, torch.zeros_like(valid))
        self.assertTrue(all(v.item() == 0 for v in empty.values()))
        sum(empty.values()).backward()
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertEqual(generated.grad.abs().sum().item(), 0)

    def test_mixed_short_recordings_pool_only_valid_frames(self):
        reference = held(8, batch=3)
        generated = reference.clone()
        generated[..., 2] += .1
        valid = torch.arange(8)[None] < torch.tensor([1, 3, 8])[:, None]
        generated[~valid] = float('nan')
        reference[~valid] = float('nan')
        result = losses(generated, reference, valid)
        self.assertAlmostEqual(result['position'].item(), .01, places=6)
        self.assertEqual(result['velocity'].item(), 0)
        self.assertEqual(result['planted_foot_velocity'].item(), 0)
        self.assertEqual(motion_kinematics(generated, valid)['edges'].sum().item(), 9)

    def test_internal_gap_resets_origin_and_excludes_edges(self):
        generated = held(5)
        generated[..., 0] = .3
        valid = torch.tensor([[True, True, False, True, True]])
        generated[:, 2] = float('nan')
        actual = motion_kinematics(generated, valid)
        torch.testing.assert_close(actual['root'][0, :, 0], torch.tensor([0., .01, 0., 0., .01]))
        self.assertEqual(actual['edges'].tolist(), [[True, False, False, True]])
        self.assertEqual(actual['velocities'][0, 1:3].abs().sum().item(), 0)

    def test_nontrivial_normalizers_and_clean_chain_gradient(self):
        reference = held()
        mean = torch.linspace(-.2, .3, 139)
        std = torch.linspace(.5, 2., 139)
        normalized = (reference-mean)/std
        for v in losses(normalized, reference, mean=mean, std=std).values():
            self.assertLess(v.item(), 1e-10)
        state = normalized.clone()
        velocity = torch.zeros_like(state, requires_grad=True)
        state[..., 0] += .5
        estimate = clean_estimate(state, velocity, .5)
        sum(losses(estimate, reference, mean=mean, std=std).values()).backward()
        self.assertTrue(torch.isfinite(velocity.grad).all())
        self.assertGreater(velocity.grad.abs().sum().item(), 0)

    def test_invalid_inputs(self):
        reference = held()
        with self.assertRaises(ValueError):
            clean_estimate(reference, reference, 1.1)
        with self.assertRaises(ValueError):
            losses(reference, reference, torch.ones(1, 8))
        with self.assertRaises(ValueError):
            losses(reference, reference, std=torch.zeros(139))


if __name__ == '__main__':
    unittest.main()

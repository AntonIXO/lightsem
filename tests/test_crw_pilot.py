"""CPU contract tests for the StereoCRW reproduction probe."""
from __future__ import annotations

import unittest

import torch

from lightsem.experiments.audio_stereo.crw_pilot import (
    StereoCRW,
    StereoCRWConfig,
    correspondence_distribution,
    cycle_loss,
    extract_patches,
    native_conditioner_features,
    shifted_cycle_targets,
    stereo_crw_loss,
    training_step,
)


class StereoCRWContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = StereoCRWConfig()

    def test_canonical_patch_geometry_and_modern_stft(self) -> None:
        wave = torch.randn(2, self.config.required_samples)
        patches = extract_patches(wave, self.config)
        self.assertEqual(tuple(patches.shape), (1, 2, 49, 1024))
        model = StereoCRW(self.config)
        images = model.encoder.stft_images(patches[:, 0].reshape(-1, 1024))
        self.assertEqual(images.shape[1], 2)  # explicit real/imag channels
        self.assertTrue(torch.isfinite(images).all())

    def test_offset_sign_matches_original_convention(self) -> None:
        rows_pos, labels_pos = shifted_cycle_targets(5, 1, device=torch.device("cpu"))
        rows_neg, labels_neg = shifted_cycle_targets(5, -1, device=torch.device("cpu"))
        self.assertEqual(rows_pos.tolist(), [1, 2, 3])
        self.assertEqual(rows_neg.tolist(), [1, 2, 3])
        self.assertEqual(labels_pos.tolist(), [[0, 1, 2]])
        self.assertEqual(labels_neg.tolist(), [[2, 3, 4]])

    def test_cycle_loss_and_native_features_are_finite(self) -> None:
        left = torch.randn(2, 7, 16, requires_grad=True)
        right = torch.randn(2, 7, 16, requires_grad=True)
        loss = cycle_loss(left, right, shift_offset=torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(left.grad).all())
        affinity, _ = correspondence_distribution(left.detach(), right.detach())
        clip_features = native_conditioner_features(affinity)
        patch_features = native_conditioner_features(affinity, per_patch=True)
        self.assertEqual(tuple(clip_features.shape), (2, 64))
        self.assertEqual(tuple(patch_features.shape), (2, 7, 64))
        self.assertTrue(torch.isfinite(clip_features).all())
        self.assertTrue(torch.isfinite(patch_features).all())

    def test_mono_and_lr_swap_have_structured_correspondence(self) -> None:
        n = self.config.required_samples
        left = torch.randn(1, n)
        right = torch.randn(1, n)
        model = StereoCRW(self.config).eval()
        with torch.no_grad():
            normal = model(torch.stack((left, right), dim=1))
            swapped = model(torch.stack((right, left), dim=1))
            mono = model(torch.stack((left, left), dim=1))
        self.assertTrue(torch.isfinite(normal["loss"]))
        self.assertTrue(torch.allclose(normal["affinity_l2r"], swapped["affinity_r2l"], atol=1e-5))
        # A mono input shares embeddings exactly; row-wise softmax makes the
        # two normalized affinity matrices differ by row normalization even
        # though their pre-softmax similarity matrix is symmetric.
        self.assertTrue(torch.allclose(mono["left_embeddings"], mono["right_embeddings"], atol=1e-5))
        self.assertGreater(float((normal["native_features"] - mono["native_features"]).abs().mean()), 1e-5)

    def test_cpu_training_step_updates_shared_encoder(self) -> None:
        model = StereoCRW(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        wave = torch.randn(1, 2, self.config.required_samples)
        before = model.encoder.backbone.conv1.weight.detach().clone()
        result = training_step(model, optimizer, wave)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in result.values()))
        self.assertFalse(torch.equal(before, model.encoder.backbone.conv1.weight.detach()))


if __name__ == "__main__":
    unittest.main()

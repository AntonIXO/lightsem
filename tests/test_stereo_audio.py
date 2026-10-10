"""Contract tests for the StereoAudioV1 pilot branch."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from lightsem.audio.stereo_v1 import StereoAudioV1, extract_stereo_v1, metamorphic_check
from lightsem.experiments.audio_stereo.model import (
    StereoConditioner, StereoConditionerConfig, StereoNativeAdapter, conditioner_loss,
)


class StereoFeatureTests(unittest.TestCase):
    def wave(self, seconds=2.0, sr=16000):
        t = np.arange(round(seconds * sr)) / sr
        left = .25 * np.sin(2 * np.pi * 220 * t) + .05 * np.sin(2 * np.pi * 880 * t)
        right = .12 * np.sin(2 * np.pi * 220 * t + .2) + .05 * np.sin(2 * np.pi * 660 * t)
        return np.stack((left, right)), sr

    def test_shapes_finite_and_swap_equivariance(self):
        wave, sr = self.wave()
        value = extract_stereo_v1(wave, sr)
        self.assertEqual(value.features_40.shape[1], 29)
        self.assertEqual(value.events_40.shape[1], 8)
        self.assertEqual(value.spatial_10.shape[1], 11)
        self.assertEqual(value.relation_2.shape[1], 6)
        self.assertTrue(np.isfinite(value.features_40).all())
        check = metamorphic_check(wave, sr)
        self.assertTrue(check["passes_lr_swap"])

    def test_roundtrip_binds_npz_and_metadata(self):
        wave, sr = self.wave()
        value = extract_stereo_v1(wave, sr)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            value.save(path)
            loaded = StereoAudioV1.load(path)
            np.testing.assert_array_equal(value.features_40, loaded.features_40)
            with self.assertRaises(FileExistsError):
                value.save(path)


class StereoModelTests(unittest.TestCase):
    def test_factorized_streams_and_zero_adapter(self):
        config = StereoConditionerConfig(arm="factorized")
        model = StereoConditioner(config)
        x = torch.randn(2, 80, config.input_dim)
        outputs = model(x)
        self.assertEqual(tuple(outputs["event"].shape), (2, 80, config.event_dim))
        self.assertEqual(tuple(outputs["spatial"].shape), (2, 20, config.spatial_dim))
        self.assertEqual(tuple(outputs["relation"].shape), (2, 4, config.relation_dim))
        adapter = StereoNativeAdapter(config.fused_dim, 16, relation_dim=config.relation_dim)
        controls = adapter(outputs)
        self.assertEqual(tuple(controls["flow_bias"].shape), (2, 80, 16))
        self.assertEqual(tuple(controls["planner_context"].shape), (2, 4, 80))
        self.assertEqual(float(controls["flow_bias"].abs().max().detach()), 0.)
        self.assertEqual(float(controls["planner_context"].abs().max().detach()), 0.)

    def test_descriptor_loss_backpropagates(self):
        config = StereoConditionerConfig(arm="fused")
        model = StereoConditioner(config)
        x = torch.randn(2, 80, config.input_dim)
        out = model(x)
        events = torch.randn(2, 80, config.event_dim)
        spatial = torch.randn(2, 20, config.spatial_dim)
        relation = torch.randn(2, 4, config.relation_dim)
        loss, parts = conditioner_loss(out, events, spatial, relation)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.fuse.weight.grad).all())
        self.assertEqual(set(parts), {"event", "spatial", "relation", "total"})


if __name__ == "__main__":
    unittest.main()

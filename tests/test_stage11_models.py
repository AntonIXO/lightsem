"""CPU checks for Stage 11's scientific model and exact sampling contracts."""

import io
import json
import unittest
from dataclasses import replace

import torch
from torch.nn import functional as F

from lightsem.experiments.stage11_models import SceneFlowConfig, SceneFlowModel


def small_config(flow_kind="temporal", **kwargs):
    return SceneFlowConfig(
        flow_kind=flow_kind, in_dim=7, scene_dim=3, d_model=16,
        n_blocks=1, n_heads=4, ff_mult=2, conv_kernel=3,
        attn_window=8, dropout=0.1, flow_dilations=(1, 2), **kwargs,
    )


class SceneFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(710)

    def test_default_parameter_match_and_shared_seed_initialization(self):
        models = {}
        for kind, count in (("temporal", 3727494), ("pointwise", 3730374)):
            torch.manual_seed(123)
            models[kind] = SceneFlowModel(kind)
            self.assertEqual(models[kind].n_params(), count)
        temporal = models["temporal"].state_dict()
        for key, value in models["pointwise"].state_dict().items():
            if not key.startswith("denoiser."):
                self.assertTrue(torch.equal(value, temporal[key]), key)
        self.assertLess(abs(3727494 / 3730374 - 1), 0.0008)
        self.assertEqual(models["temporal"].cfg.encoder_radius, 138)
        self.assertEqual(models["temporal"].cfg.velocity_radius, 126)
        self.assertEqual(models["pointwise"].cfg.velocity_radius, 0)

    def test_cfm_loss_matches_injected_noise_and_sequence_times(self):
        model = SceneFlowModel(config=small_config()).eval()
        x, y, noise = torch.randn(2, 23, 7), torch.randn(2, 23, 3), torch.randn(2, 23, 3)
        t = torch.tensor([0.2, 0.8]).view(2, 1, 1)
        z = (1 - t) * noise + t * y
        expected = F.mse_loss(model.velocity(x, z, t), y - noise)
        torch.testing.assert_close(model.loss(x, y, noise=noise, t=t), expected)
        with self.assertRaisesRegex(ValueError, "one flow time"):
            model.loss(x, y, noise=noise, t=t.expand(2, 23, 1))

    def test_explicit_generators_ignore_global_noise_rng(self):
        model = SceneFlowModel(config=small_config()).eval()
        x, y = torch.randn(2, 19, 7), torch.randn(2, 19, 3)
        first = model.loss(x, y, generator=torch.Generator().manual_seed(9))
        torch.randn(800)
        second = model.loss(x, y, generator=torch.Generator().manual_seed(9))
        self.assertTrue(torch.equal(first, second))
        sample = model.sample(x, generator=torch.Generator().manual_seed(12))
        self.assertTrue(torch.equal(sample, model.sample(x, generator=torch.Generator().manual_seed(12))))
        self.assertFalse(torch.equal(sample, model.sample(x, generator=torch.Generator().manual_seed(13))))

    def test_separate_flow_time_and_noise_streams(self):
        model = SceneFlowModel(config=small_config()).eval()
        x, y = torch.randn(2, 17, 7), torch.randn(2, 17, 3)
        loss = model.loss(
            x, y, noise_generator=torch.Generator().manual_seed(5),
            time_generator=torch.Generator().manual_seed(11),
        )
        noise = torch.randn(y.shape, generator=torch.Generator().manual_seed(5))
        t = torch.rand((2, 1, 1), generator=torch.Generator().manual_seed(11))
        self.assertTrue(torch.equal(loss, model.loss(x, y, noise=noise, t=t)))

    def test_gradient_temporal_reach_and_pointwise_control(self):
        for kind in ("temporal", "pointwise"):
            model = SceneFlowModel(config=small_config(kind)).eval()
            h = torch.randn(1, 31, 16)
            z = torch.randn(1, 31, 3, requires_grad=True)
            model.velocity_from_h(h, z, 0.4)[0, 15].sum().backward()
            self.assertGreater(z.grad[0, 15].abs().sum().item(), 0)
            neighbour = z.grad[0, 14].abs().sum().item()
            if kind == "pointwise":
                self.assertEqual(neighbour, 0)
            else:
                self.assertGreater(neighbour, 0)
            self.assertEqual(z.grad[0, :9].abs().sum().item(), 0)

    def test_all_parameters_receive_finite_training_gradients(self):
        for kind in ("temporal", "pointwise"):
            model = SceneFlowModel(config=small_config(kind)).train()
            loss = model.loss(
                torch.randn(2, 29, 7), torch.randn(2, 29, 3),
                generator=torch.Generator().manual_seed(3),
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_chunked_encoder_and_global_euler_match_full_song(self):
        for kind in ("temporal", "pointwise"):
            model = SceneFlowModel(config=small_config(kind)).eval()
            x, noise = torch.randn(2, 113, 7), torch.randn(2, 113, 3)
            full_h = model.encode_audio(x)
            for chunk_size in (13, 30, 150):
                torch.testing.assert_close(
                    model.encode_audio_chunked(x, chunk_size), full_h,
                    atol=1e-5, rtol=0,
                )
                for steps in (1, 8):
                    full = model.sample(x, steps=steps, noise=noise)
                    chunked = model.sample_chunked(x, steps=steps, chunk_size=chunk_size, noise=noise)
                    torch.testing.assert_close(chunked, full, atol=1e-5, rtol=0)
                    self.assertTrue(torch.isfinite(chunked).all())

    def test_global_noise_is_reused_across_chunk_sizes_and_not_mutated(self):
        model = SceneFlowModel(config=small_config()).train()
        x = torch.zeros(1, 79, 7)
        noise = torch.randn(1, 79, 3)
        original = noise.clone()
        full = model.sample(x, noise=noise)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(noise, original))
        for chunk_size in (7, 21, 80):
            chunked = model.sample_chunked(x, chunk_size=chunk_size, noise=noise)
            torch.testing.assert_close(chunked, full, atol=1e-5, rtol=0)
        seeded_full = model.sample(x, generator=torch.Generator().manual_seed(4))
        seeded_chunked = model.sample_chunked(x, chunk_size=9, generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(seeded_chunked, seeded_full, atol=1e-5, rtol=0)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(noise, original))

    def test_default_temporal_halos_match_full_song_for_eight_steps(self):
        model = SceneFlowModel("temporal").eval()
        x, noise = torch.randn(1, 701, 52), torch.randn(1, 701, 6)
        full = model.sample(x, steps=8, noise=noise)
        chunked = model.sample_chunked(x, steps=8, chunk_size=157, noise=noise)
        torch.testing.assert_close(chunked, full, atol=1e-5, rtol=0)

    def test_checkpoint_config_state_and_generator_roundtrip(self):
        model = SceneFlowModel(config=small_config()).eval()
        x, y = torch.randn(2, 11, 7), torch.randn(2, 11, 3)
        generator = torch.Generator().manual_seed(87)
        model.loss(x, y, generator=generator)
        payload = {"config": json.loads(json.dumps(model.config_dict())),
                   "model": model.state_dict(), "generator": generator.get_state()}
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        restored = SceneFlowModel.from_config(loaded["config"]).eval()
        restored.load_state_dict(loaded["model"])
        restored_generator = torch.Generator()
        restored_generator.set_state(loaded["generator"])
        self.assertTrue(torch.equal(
            model.loss(x, y, generator=generator),
            restored.loss(x, y, generator=restored_generator),
        ))
        torch.testing.assert_close(model.sample(x, noise=y), restored.sample(x, noise=y), atol=0, rtol=0)

    def test_reject_invalid_shapes_and_steps(self):
        model = SceneFlowModel(config=small_config()).eval()
        x = torch.randn(2, 12, 7)
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                model.sample(x, steps=steps)
        with self.assertRaises(ValueError):
            model.encode_audio(x[:, :0])
        with self.assertRaises(ValueError):
            model.sample(x, noise=torch.randn(2, 13, 3))
        with self.assertRaises(ValueError):
            model.velocity(x, torch.randn(2, 11, 3), 0.5)
        with self.assertRaises(ValueError):
            model.velocity(x, torch.randn(2, 12, 3), torch.randn(2, 12))
        with self.assertRaises(ValueError):
            replace(small_config(), flow_kind="unknown")


if __name__ == "__main__":
    unittest.main()

"""Synthetic-only unit tests; this module never opens datasets or results.

CPU:  bash run_torch.sh -m unittest discover -s tests -p test_light_layers_model.py -v
CUDA: LIGHT_LAYERS_TEST_CUDA=1 bash run_torch.sh -m unittest discover -s tests \
          -p test_light_layers_model.py -k tiny_cuda -v
"""

from dataclasses import replace
import inspect
import os
import unittest

import torch

from lightsem.model.light_layers import (
    IndependentLightLayers, LightLayersConfig, SharedLightLayers,
    build_light_layers, count_trainable_parameters, match_independent_width,
)


def small_config(**changes):
    # Width 128 also permits a <=5% match with this deliberately shallow model.
    base = LightLayersConfig(in_dim=10, d_model=128, n_blocks=1, ff_mult=2,
                             conv_kernel=3, attn_window=8, dropout=0.0,
                             palette_size=8, visual_aux_dim=6, pose_aux_dim=4,
                             max_frames40=64)
    return replace(base, **changes)


def gradient_magnitude(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


class LightLayersModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.rng_state = torch.get_rng_state()
        torch.manual_seed(20260922)

    def tearDown(self):
        torch.set_rng_state(self.rng_state)

    def test_default_configuration(self):
        cfg = LightLayersConfig()
        self.assertEqual((cfg.in_dim, cfg.d_model, cfg.n_blocks, cfg.n_heads), (94, 128, 3, 4))
        self.assertEqual((cfg.conv_kernel, cfg.attn_window, cfg.palette_size), (15, 160, 32))
        self.assertEqual(cfg.null_palette_id, 32)
        self.assertEqual(cfg.context_radius10, 258)
        self.assertTrue(cfg.trunk_config().offline)

    def test_invalid_configuration(self):
        bad = [dict(in_dim=0), dict(in_dim=True), dict(d_model=126), dict(d_model=0),
               dict(n_blocks=0), dict(n_heads=3), dict(n_heads=0), dict(ff_mult=0),
               dict(conv_kernel=2), dict(attn_window=1), dict(attn_window=7),
               dict(dropout=-0.1), dict(dropout=1), dict(dropout=float("nan")),
               dict(dropout=float("inf")), dict(dropout=True), dict(palette_size=1),
               dict(visual_aux_dim=4), dict(visual_aux_dim=True), dict(pose_aux_dim=6),
               dict(pose_aux_dim=False), dict(max_frames40=0), dict(max_frames40=6)]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LightLayersConfig(**kwargs)

    def test_shapes_and_auxiliary_outputs_are_training_only(self):
        for visual in (0, 6, 96):
            for pose in (0, 4):
                for cls in (SharedLightLayers, IndependentLightLayers):
                    with self.subTest(visual=visual, pose=pose, model=cls.__name__):
                        model = cls(small_config(visual_aux_dim=visual, pose_aux_dim=pose))
                        x = torch.randn(2, 24, 10)
                        out = model(x)
                        self.assertEqual(out["accent_logit"].shape, (2, 24))
                        self.assertEqual(out["chromatic_change_logit"].shape, (2, 6))
                        self.assertEqual(out["context10"].shape, (2, 6, model.cfg.d_model))
                        self.assertEqual(model.palette_logits(out["context10"]).shape, (2, 6, 8))
                        for name, dim in (("visual_aux", visual), ("pose_aux", pose)):
                            if dim:
                                self.assertEqual(out[name].shape, (2, 6, dim))
                            else:
                                self.assertNotIn(name, out)
                        inference = model.eval()(x)
                        self.assertEqual(set(inference),
                                         {"accent_logit", "chromatic_change_logit", "context10"})

    def test_invalid_audio_and_explicit_length_limit(self):
        model = SharedLightLayers(small_config())
        invalid = [torch.randn(2, 23, 10), torch.randn(2, 68, 10),
                   torch.randn(2, 24, 11), torch.randn(0, 24, 10),
                   torch.randn(2, 0, 10), torch.randn(24, 10),
                   torch.ones(2, 24, 10, dtype=torch.long), None]
        for x in invalid:
            with self.subTest(shape=getattr(x, "shape", None)), self.assertRaises(ValueError):
                model(x)
        self.assertEqual(model(torch.randn(1, 4, 10))["accent_logit"].shape, (1, 4))
        self.assertEqual(model(torch.randn(1, 64, 10))["context10"].shape[1], 16)

    def test_no_target_modality_inference_input(self):
        for cls in (SharedLightLayers, IndependentLightLayers):
            signature = inspect.signature(cls.forward)
            self.assertEqual(tuple(signature.parameters), ("self", "audio40"))
            model = cls(small_config()).eval()
            x = torch.randn(1, 8, 10)
            for target_name in ("pose", "video", "targets", "palette_targets"):
                with self.subTest(target_name=target_name), self.assertRaises(TypeError):
                    model(x, **{target_name: torch.randn(1, 2, 4)})

    def test_aligned_pooling_and_raw_acoustic_skip(self):
        model = SharedLightLayers(small_config()).eval()
        x = torch.randn(2, 24, 10)
        captured = {}
        def capture_pooled(module, args):
            captured["pooled"] = args[0].detach().clone()
        def capture_accent(module, args):
            captured["accent"] = args[0].detach().clone()
        h1 = model.encoders["shared"].inp.register_forward_pre_hook(capture_pooled)
        h2 = model.accent_head[0].register_forward_pre_hook(capture_accent)
        try:
            out = model(x)
        finally:
            h1.remove()
            h2.remove()
        torch.testing.assert_close(captured["pooled"], x.reshape(2, 6, 4, 10).mean(dim=2))
        torch.testing.assert_close(captured["accent"][..., -10:], x)
        torch.testing.assert_close(captured["accent"][..., :-10],
                                   out["context10"].repeat_interleave(4, dim=1))
        # A zero-mean within-bin change cannot enter the slow context, but it
        # remains visible to the high-rate accent head. No dataset is involved.
        zero = torch.zeros(1, 8, 10)
        perturbed = zero.clone()
        perturbed[:, 0, 0] = 1
        perturbed[:, 1, 0] = -1
        a, b = model(zero), model(perturbed)
        torch.testing.assert_close(a["context10"], b["context10"], rtol=0, atol=0)
        torch.testing.assert_close(a["chromatic_change_logit"], b["chromatic_change_logit"],
                                   rtol=0, atol=0)
        self.assertFalse(torch.allclose(a["accent_logit"][:, :2], b["accent_logit"][:, :2]))
        torch.testing.assert_close(a["accent_logit"][:, 2:], b["accent_logit"][:, 2:],
                                   rtol=0, atol=0)

    def test_all_heads_and_all_encoders_receive_gradients(self):
        for cls in (SharedLightLayers, IndependentLightLayers):
            with self.subTest(model=cls.__name__):
                model = cls(small_config())
                audio = torch.randn(2, 24, 10, requires_grad=True)
                out = model(audio)
                previous = torch.arange(12).reshape(2, 6).clamp_max(8)
                palette = model.palette_logits(out["context10"], previous)
                loss = sum(out[key].square().mean() for key in
                           ("accent_logit", "chromatic_change_logit", "visual_aux", "pose_aux"))
                (loss + palette.square().mean()).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                for name, module in list(model.encoders.items()) + [
                        ("accent", model.accent_head), ("chromatic", model.chromatic_head),
                        ("palette", model.palette_head), ("state", model.palette_state_embedding),
                        ("visual_aux", model.visual_aux_head), ("pose_aux", model.pose_aux_head)]:
                    self.assertGreater(gradient_magnitude(module), 0, name)
                self.assertGreater(float(audio.grad.abs().sum()), 0)

    def test_independent_head_routing_and_color_auxiliary_attachment(self):
        model = IndependentLightLayers(small_config())
        for output_name, branch in (("accent_logit", "accent"),
                                    ("chromatic_change_logit", "chromatic"),
                                    ("palette", "palette"), ("visual_aux", "chromatic"),
                                    ("pose_aux", "chromatic")):
            with self.subTest(output=output_name):
                model.zero_grad(set_to_none=True)
                out = model(torch.randn(1, 16, 10))
                value = (model.palette_logits(out["context10"])
                         if output_name == "palette" else out[output_name])
                value.square().mean().backward()
                for name, encoder in model.encoders.items():
                    if name == branch:
                        self.assertGreater(gradient_magnitude(encoder), 0)
                    else:
                        self.assertTrue(all(p.grad is None for p in encoder.parameters()), name)

    def test_palette_state_changes_predictions_without_hidden_state(self):
        for cls in (SharedLightLayers, IndependentLightLayers):
            model = cls(small_config()).eval()
            context = model(torch.randn(2, 24, 10))["context10"]
            zeros = torch.zeros(2, 6, dtype=torch.long)
            ones = torch.ones_like(zeros)
            null = torch.full_like(zeros, model.cfg.null_palette_id)
            a = model.palette_logits(context, zeros)
            b = model.palette_logits(context, ones)
            self.assertFalse(torch.allclose(a, b))
            torch.testing.assert_close(a, model.palette_logits(context, zeros), rtol=0, atol=0)
            null_logits = model.palette_logits(context, null)
            torch.testing.assert_close(null_logits, model.palette_logits(context), rtol=0, atol=0)
            drop = torch.zeros_like(zeros, dtype=torch.bool)
            drop[:, ::2] = True
            dropped = model.palette_logits(context, zeros, state_drop_mask=drop)
            torch.testing.assert_close(dropped[drop], null_logits[drop], rtol=0, atol=0)
            torch.testing.assert_close(dropped[~drop], a[~drop], rtol=0, atol=0)
            self.assertTrue(bool((zeros == 0).all()))

    def test_palette_state_input_validation(self):
        model = SharedLightLayers(small_config())
        context = torch.randn(2, 6, model.cfg.d_model)
        bad_ids = [torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 6),
                   torch.full((2, 6), -1), torch.full((2, 6), 9), [0]]
        for previous in bad_ids:
            with self.subTest(previous=str(previous)), self.assertRaises(ValueError):
                model.palette_logits(context, previous)
        for drop in (torch.zeros(2, 6), torch.zeros(2, 5, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                model.palette_logits(context, state_drop_mask=drop)
        for context in (torch.randn(2, 6, 3), torch.randn(0, 6, model.cfg.d_model),
                        torch.randn(2, model.cfg.d_model), None):
            with self.assertRaises(ValueError):
                model.palette_logits(context)

    def test_outputs_are_raw_logits_not_forced_events(self):
        model = SharedLightLayers(small_config()).eval()
        with torch.no_grad():
            model.accent_head[-1].weight.zero_()
            model.accent_head[-1].bias.fill_(-2)
            model.chromatic_head.weight.zero_()
            model.chromatic_head.bias.fill_(-3)
            model.palette_head[-1].weight.zero_()
            model.palette_head[-1].bias.fill_(-4)
        out = model(torch.randn(1, 16, 10))
        self.assertTrue(bool((out["accent_logit"] == -2).all()))
        self.assertTrue(bool((out["chromatic_change_logit"] == -3).all()))
        self.assertTrue(bool((model.palette_logits(out["context10"]) == -4).all()))

    def test_default_parameter_match_and_identical_temporal_context(self):
        for visual, pose in ((0, 0), (6, 0), (96, 4)):
            cfg = LightLayersConfig(visual_aux_dim=visual, pose_aux_dim=pose)
            shared = SharedLightLayers(cfg)
            independent = IndependentLightLayers(cfg)
            report = independent.parameter_match
            self.assertEqual(report.shared_params, count_trainable_parameters(shared))
            self.assertEqual(report.independent_params, count_trainable_parameters(independent))
            self.assertLessEqual(report.relative_difference, 0.05)
            self.assertLess(independent.cfg.d_model, shared.cfg.d_model)
            self.assertEqual(replace(independent.cfg, d_model=cfg.d_model), cfg)
            self.assertEqual(independent.cfg.context_radius10, cfg.context_radius10)
            encoder_ids = [set(id(p) for p in encoder.parameters())
                           for encoder in independent.encoders.values()]
            self.assertEqual(len(encoder_ids), 3)
            for i in range(3):
                for j in range(i):
                    self.assertTrue(encoder_ids[i].isdisjoint(encoder_ids[j]))
            if (visual, pose) == (0, 0):
                self.assertEqual(report.independent_d_model, 76)
                self.assertEqual(report.shared_params, 1_229_406)
                self.assertEqual(report.independent_params, 1_277_018)

    def test_parameter_search_is_deterministic_count_only_and_rng_neutral(self):
        cfg = LightLayersConfig()
        before = torch.get_rng_state().clone()
        report = match_independent_width(cfg, (76, 72, 76))
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(report.candidate_widths, (72, 76))
        self.assertEqual(report.independent_d_model, 76)
        self.assertEqual(report, match_independent_width(cfg, (72, 76)))
        for widths in ((), (128,), (3,), (0,), (True,), (4,)):
            with self.subTest(widths=widths), self.assertRaises(ValueError):
                match_independent_width(cfg, widths)
        with self.assertRaises(ValueError):
            match_independent_width(LightLayersConfig(d_model=4))

    def test_factory(self):
        cfg = small_config()
        self.assertIsInstance(build_light_layers(cfg), SharedLightLayers)
        self.assertIsInstance(build_light_layers(cfg, architecture="independent"), IndependentLightLayers)
        with self.assertRaises(ValueError):
            build_light_layers(cfg, architecture="unknown")
        with self.assertRaises(ValueError):
            build_light_layers(cfg, allowed_widths=(76,))


class LightLayersCudaSmokeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LIGHT_LAYERS_TEST_CUDA") == "1",
                         "opt-in tiny synthetic CUDA forward/backward")
    def test_tiny_cuda_forward_backward(self):
        self.assertTrue(torch.cuda.is_available(), "CUDA smoke was requested but CUDA is unavailable")
        # One tiny batch; no data loader, optimizer, training run, or performance claim.
        model = IndependentLightLayers(small_config()).cuda().train()
        audio = torch.randn(1, 8, 10, device="cuda", requires_grad=True)
        out = model(audio)
        previous = torch.tensor([[8, 2]], device="cuda")
        palette = model.palette_logits(out["context10"], previous)
        loss = palette.square().mean() + sum(out[key].square().mean() for key in
                                             ("accent_logit", "chromatic_change_logit",
                                              "visual_aux", "pose_aux"))
        loss.backward()
        torch.cuda.synchronize()
        self.assertEqual(out["accent_logit"].shape, (1, 8))
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(audio.grad).all()))
        for name, encoder in model.encoders.items():
            self.assertGreater(gradient_magnitude(encoder), 0, name)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)


if __name__ == "__main__":
    unittest.main()

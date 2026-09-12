"""CPU analytical/source contracts only. These do NOT prove GPU pixel parity."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from lightsem.dmx.fixtures import Fixture, Rig, club_rig
from lightsem.sim.renderer import RenderConfig, StageRenderer
from lightsem.sim.shaders import FRAGMENT_SHADER


def smoothstep(lo, hi, x):
    t = min(1.0, max(0.0, (x - lo) / (hi - lo)))
    return t * t * (3.0 - 2.0 * t)


def cpu_cone(angle, full_beam=6.0, profile="legacy", softness=0.6, kind=0):
    """Double-precision reference for GLSL lightAt's cone factor, not radiance."""
    cos_a = math.cos(math.radians(angle))
    edge = math.cos(math.radians(full_beam / 2))
    legacy_softness = 0.16 if kind == 1 else 0.018
    cone = smoothstep(edge - legacy_softness, edge + legacy_softness * 0.4, cos_a)
    if profile == "angular" and kind == 0:
        outer = math.cos(math.radians(full_beam / 2 + softness))
        inner = math.cos(math.radians(max(full_beam / 2 - softness, 0)))
        cone = smoothstep(outer, inner, cos_a)
    if kind == 2:
        cone = smoothstep(-0.15, 0.55, cos_a)
    return cone


def fake_renderer(cfg=None):
    obj = StageRenderer.__new__(StageRenderer)
    obj.cfg = cfg or RenderConfig(width=2, height=2)
    obj.rig = club_rig()
    names = ("u_res u_time u_cam_pos u_cam_target u_haze u_fixed_exposure "
             "u_exposure_scale u_camera_scale u_beam_edge_profile u_beam_edge_softness_deg u_beam_flux_reference_deg "
             "u_n_lights u_lpos u_ldir u_lcol u_lcos u_lint u_lkind").split()
    obj.prog = {name: Mock(value=None) for name in names}
    obj.ctx = Mock()
    obj.vao = Mock()
    obj.fbo = SimpleNamespace(read=lambda **kw: bytes(obj.cfg.width * obj.cfg.height * 3))
    return obj


class ConeProfileTests(unittest.TestCase):
    def test_angular_full_center_inner_edge_and_cutoff(self):
        for angle in (0, 1, 2.4):
            self.assertAlmostEqual(cpu_cone(angle, profile="angular"), 1.0)
        for angle in (3.6, 4, 90, 180):
            self.assertAlmostEqual(cpu_cone(angle, profile="angular"), 0.0)
        values = [cpu_cone(a, profile="angular") for a in np.linspace(0, 5, 501)]
        self.assertTrue(all(0 <= value <= 1 for value in values))
        self.assertTrue(np.all(np.diff(values) <= 0))

    def test_angular_half_strength_is_near_nominal_not_legacy_floor(self):
        outer = math.cos(math.radians(3.6))
        inner = math.cos(math.radians(2.4))
        half_angle = math.degrees(math.acos((outer + inner) / 2))
        self.assertAlmostEqual(half_angle, 3.059, delta=.001)
        self.assertAlmostEqual(cpu_cone(half_angle, profile="angular"), .5)
        self.assertGreater(cpu_cone(3, profile="angular"), .5)

    def test_angular_softness_larger_than_half_angle_still_has_full_axis(self):
        self.assertEqual(cpu_cone(0, full_beam=.5, profile="angular"), 1.)
        self.assertEqual(cpu_cone(.85, full_beam=.5, profile="angular"), 0.)

    def test_legacy_absolute_half_strength_width_has_constant_floor(self):
        half_angle = math.degrees(math.acos(math.cos(math.radians(3)) - .3 * .018))
        self.assertAlmostEqual(2 * half_angle, 13.34, delta=.01)
        self.assertAlmostEqual(cpu_cone(half_angle), .5)
        floor_full_width = 2 * math.degrees(math.acos(1 - .3 * .018))
        self.assertGreater(floor_full_width, 11.9)
        self.assertLess(cpu_cone(0), 1.)  # Legacy narrow cone never reaches full strength.
        for full in (.001, .01, .1, 6):
            width = 2 * math.degrees(math.acos(math.cos(math.radians(full / 2)) - .3 * .018))
            self.assertGreaterEqual(width, floor_full_width)

    def test_non_heads_are_exactly_unchanged(self):
        for kind in (1, 2, 3):
            for angle in np.linspace(0, 180, 361):
                self.assertEqual(cpu_cone(angle, kind=kind), cpu_cone(angle, profile="angular", kind=kind))

    def test_shader_equations_match_analytical_contract(self):
        # Source checks bind the CPU reference to the intended GLSL equations;
        # only a real GL render can check compilation and numerical execution.
        for expression in (
            'float softness = (u_lkind[i] == 1) ? 0.16 : 0.018;',
            'float cone = smoothstep(edge - softness, edge + softness * 0.4, cosA);',
            'if (u_beam_edge_profile == 1 && u_lkind[i] == 0)',
            'float nominal_half = acos(clamp(edge, -1.0, 1.0));',
            'float softness_rad = radians(u_beam_edge_softness_deg);',
            'float outer_cos = cos(nominal_half + softness_rad);',
            'float inner_cos = cos(max(nominal_half - softness_rad, 0.0));',
            'cone = smoothstep(outer_cos, inner_cos, clamp(cosA, -1.0, 1.0));',
            'fwd + uv.x * right * u_camera_scale + uv.y * up * u_camera_scale',
        ):
            self.assertIn(expression, FRAGMENT_SHADER)


class ConfigAndUniformTests(unittest.TestCase):
    def test_defaults_remain_legacy(self):
        cfg = RenderConfig()
        self.assertEqual((cfg.camera_scale, cfg.beam_edge_profile, cfg.beam_edge_softness_deg), (.85, "legacy", .6))
        self.assertEqual((cfg.cam_pos, cfg.cam_target, cfg.cam_orbit), ((0., 2.6, 11.), (0., 3.2, -3.), .07))
        self.assertEqual((cfg.haze, cfg.intensity_gain, cfg.exposure_profile), (.115, 2.4, "legacy"))
        self.assertEqual((cfg.fixed_exposure_scale, cfg.fixed_role_gains), (.12, (1., 1., 1., 1.)))

    def test_invalid_configs(self):
        cases = [dict(beam_edge_profile="sharp"), dict(width=0), dict(height=1.5),
                 dict(fps=0), dict(haze=-1), dict(intensity_gain=-1),
                 dict(cam_pos=(0, 0, 0), cam_target=(0, 1, 0)),
                 dict(cam_target=(0, 0)), dict(exposure_profile="auto"),
                 dict(fixed_exposure_scale=0), dict(fixed_role_gains=(1, -1, 1, 1))]
        for value in (0, -1, math.inf, -math.inf, math.nan):
            cases.append(dict(camera_scale=value))
            cases.append(dict(beam_edge_softness_deg=value))
        cases.extend([dict(beam_edge_softness_deg=90), dict(beam_edge_softness_deg=180)])
        for name in ("fps", "haze", "intensity_gain", "cam_orbit", "fixed_exposure_scale"):
            for value in (math.inf, math.nan):
                cases.append({name: value})
        cases.append(dict(cam_pos=(math.nan, 0, 1)))
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                RenderConfig(**case)

    def test_invalid_angular_fixture_rejected_before_gpu_creation(self):
        for low, high in ((0, 32), (32, 6), (6, 181), (math.nan, 32), (6, math.inf)):
            fixture = Fixture("test", "moving_head", 1, beam_min_deg=low, beam_max_deg=high)
            with patch("lightsem.sim.renderer.moderngl.create_context") as create:
                with self.assertRaisesRegex(ValueError, "beam range"):
                    StageRenderer(Rig([fixture]), RenderConfig(beam_edge_profile="angular"))
                create.assert_not_called()

    def test_explicit_controls_uploaded_and_dmx_unmodified(self):
        cfg = RenderConfig(width=2, height=2, camera_scale=.4, beam_edge_profile="angular",
                           beam_edge_softness_deg=.8, cam_pos=(1., 3., 10.), cam_target=(0., 2., -2.))
        renderer = fake_renderer(cfg)
        universe = renderer.rig.universe()
        original = universe.copy()
        renderer.render_frame(universe, 0.)
        self.assertEqual(renderer.prog["u_camera_scale"].value, .4)
        self.assertEqual(renderer.prog["u_beam_edge_profile"].value, 1)
        self.assertEqual(renderer.prog["u_beam_edge_softness_deg"].value, .8)
        self.assertEqual(renderer.prog["u_cam_pos"].value, cfg.cam_pos)
        self.assertEqual(renderer.prog["u_cam_target"].value, cfg.cam_target)
        np.testing.assert_array_equal(universe, original)

    def test_presentation_controls_do_not_change_light_arrays(self):
        legacy = fake_renderer()
        angular = fake_renderer(RenderConfig(width=2, height=2, camera_scale=.4, beam_edge_profile="angular"))
        universe = np.random.default_rng(42).integers(0, 256, 512, dtype=np.uint8)
        legacy.render_frame(universe, 2.)
        angular.render_frame(universe, 2.)
        self.assertEqual(legacy.prog["u_beam_edge_profile"].value, 0)
        for name in ("u_lpos", "u_ldir", "u_lcol", "u_lcos", "u_lint", "u_lkind"):
            self.assertEqual(legacy.prog[name].write.call_args, angular.prog[name].write.call_args)

    def test_every_missing_uniform_fails_loudly(self):
        for name in fake_renderer().prog:
            renderer = fake_renderer()
            del renderer.prog[name]
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, name):
                renderer.render_frame(renderer.rig.universe(), 0.)
            renderer.vao.render.assert_not_called()


if __name__ == "__main__":
    unittest.main()

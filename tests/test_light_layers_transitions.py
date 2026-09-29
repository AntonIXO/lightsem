"""Cue timing, interrupted fades and protected DMX contract checks."""
import unittest
import numpy as np

from lightsem.conductor.rig import canonical_rig
from lightsem.model.light_layers.overlay import apply_overlay
from lightsem.model.light_layers.transitions import fade_rgb_targets
from scripts.light_layers.prepare_baseline_review import replay_choices


class ColorTransitions(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(81, dtype=float) / 40
        self.x = np.tile([1., 0., 0.], (len(self.t), 1))
        self.x[20:] = [0., 0., 1.]

    def test_no_anticipation_midpoint_and_completion(self):
        before = self.x.copy()
        out = fade_rgb_targets(self.x, self.t, duration_s=.5)
        np.testing.assert_array_equal(out[:21], np.tile([1., 0., 0.], (21, 1)))
        np.testing.assert_allclose(out[30], [.5, 0., .5], atol=1e-14)
        np.testing.assert_array_equal(out[40:], self.x[40:])
        np.testing.assert_array_equal(self.x, before)
        self.assertTrue(np.all(np.diff(out[20:41, 0]) <= 0))

    def test_retarget_starts_at_current_color(self):
        self.x[30:] = [0., 1., 0.]
        out = fade_rgb_targets(self.x, self.t, duration_s=.5)
        np.testing.assert_allclose(out[30], [.5, 0., .5])
        np.testing.assert_allclose(out[40], [.25, .5, .25])
        np.testing.assert_array_equal(out[50:], self.x[50:])

    def test_identity_modes_and_no_black_initialization(self):
        for kwargs in ({"duration_s": 0}, {"active": np.zeros(len(self.t), bool)}):
            out = fade_rgb_targets(self.x, self.t, **kwargs)
            np.testing.assert_array_equal(out, self.x)
            self.assertFalse(np.shares_memory(out, self.x))
        constant = np.tile([.2, .4, .8], (len(self.t), 1))
        np.testing.assert_array_equal(fade_rgb_targets(constant, self.t), constant)

    def test_activation_keeps_past_and_blends_first_learned_color(self):
        active = self.t >= .5
        out = fade_rgb_targets(self.x, self.t, duration_s=.5, active=active)
        np.testing.assert_array_equal(out[:20], self.x[:20])
        np.testing.assert_array_equal(out[20], [1., 0., 0.])
        active[25:30] = False
        out = fade_rgb_targets(self.x, self.t, active=active)
        np.testing.assert_array_equal(out[25:30], self.x[25:30])

    def test_future_changes_cannot_change_prefix(self):
        later = self.x.copy()
        later[35:] = [.1, .9, .4]
        np.testing.assert_array_equal(fade_rgb_targets(later, self.t)[:35],
                                      fade_rgb_targets(self.x, self.t)[:35])

    def test_fade_overlay_leaves_motion_dimmers_and_other_slots_exact(self):
        rig = canonical_rig()
        baseline = np.random.default_rng(2).integers(0, 256, (len(self.t), 512), dtype=np.uint8)
        track = fade_rgb_targets(self.x, self.t)
        result = apply_overlay(baseline, rig, self.t, rgb={"mh1": track})
        f = next(f for f in rig.fixtures if f.name == "mh1")
        mask = np.zeros(512, bool)
        mask[f.address - 1 + np.array([6, 7, 8])] = True
        np.testing.assert_array_equal(result[:, ~mask], baseline[:, ~mask])
        self.assertTrue(np.all(result[:, mask] >= 0))

    def test_invalid_contracts_and_empty(self):
        for duration in (-1, float("nan"), float("inf"), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                fade_rgb_targets(self.x, self.t, duration_s=duration)
        for active in (np.ones(len(self.t)), np.ones(len(self.t) - 1, bool)):
            with self.assertRaises(ValueError):
                fade_rgb_targets(self.x, self.t, active=active)
        with self.assertRaises(ValueError):
            fade_rgb_targets(self.x, self.t + .1)
        with self.assertRaises(ValueError):
            fade_rgb_targets(self.x * 2, self.t)
        out = fade_rgb_targets(np.empty((0, 3)), np.empty(0))
        self.assertEqual(out.shape, (0, 3))

    def test_palette_state_replay_uses_generated_history_or_null(self):
        choices = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.uint8)
        indices = np.array([0, 2, 4])
        own, previous = replay_choices(choices, indices, 6, history=True)
        null, null_previous = replay_choices(choices, indices, 6, history=False)
        np.testing.assert_array_equal(own, [0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(previous, [2, 0, 0, 1, 1, 0])
        np.testing.assert_array_equal(null, [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(null_previous, [2]*6)

    def test_palette_replay_rejects_incomplete_or_invalid_actions(self):
        good = np.zeros((2, 3), np.uint8)
        for table, indices in ((good, [1, 3]), (good, [0, 0]), (good, [0, 7]),
                               (good, [0]), (good + 2, [0, 3])):
            with self.assertRaises(ValueError):
                replay_choices(table, np.array(indices), 6, history=True)


if __name__ == "__main__":
    unittest.main()

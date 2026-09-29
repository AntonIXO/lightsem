import json
import unittest

import numpy as np
from scipy.stats import wasserstein_distance

from lightsem.experiments.stage11.stage11_metrics import (
    beat_only_scores, evaluate_conditioning_gate, evaluate_motion_gate,
    evaluate_temporal_gate, event_average_precision, fit_scales, motion_r2,
    normalize_recording_scene, paired_hierarchical_bootstrap, scene_metrics, scene_novelty,
)


def signal(frames=1200):
    time = np.arange(frames) / 10
    return np.column_stack([np.sin(time * (0.37 + channel * 0.13)) for channel in range(6)])


class Stage11MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = signal()
        cls.scales = fit_scales([cls.reference, signal(800) * 1.2])

    def test_constant_novelty_is_exactly_zero_with_frozen_scaling(self):
        for value in (0.0, 1.23456789, 1e8):
            novelty = scene_novelty(np.full((1000, 6), value), self.scales)
            np.testing.assert_array_equal(novelty, np.zeros(1000))

    def test_fixed_scales_do_not_hide_reduced_variation(self):
        novelty = scene_novelty(self.reference, self.scales)
        small = scene_novelty(self.reference * 0.1, self.scales)
        np.testing.assert_allclose(small, novelty * 0.1, atol=1e-12)

    def test_lag_wasserstein_uses_frozen_training_scale(self):
        generated = self.reference * 0.4
        result = scene_metrics(self.reference, generated, [], self.scales, circular_shifts=0)
        lag = self.scales["lags"][0]
        expected = wasserstein_distance(np.diff(self.reference[:, 0], n=lag),
                                        np.diff(generated[:, 0], n=lag)) / self.scales["lag_iqr"][0][0]
        self.assertAlmostEqual(result["temporal_terms"][0], expected)
        self.assertEqual(result["temporal_term_count"], 30)
        self.assertAlmostEqual(result["variation_ratio"], 0.4)
        self.assertAlmostEqual(result["hf_energy_ratio_2to5"], 0.16)

    def test_identical_trajectory_has_zero_errors(self):
        result = scene_metrics(self.reference, self.reference, [], self.scales, circular_shifts=0)
        self.assertEqual(result["temporal_error"], 0)
        self.assertEqual(result["marginal_error"], 0)
        self.assertAlmostEqual(result["variation_ratio"], 1)

    def test_tied_predictions_have_exact_prevalence_ap(self):
        events = np.zeros(100, dtype=bool)
        events[50] = True
        result = event_average_precision(events, np.ones(100), fps=10)
        self.assertEqual(result["positive_frames"], 5)
        self.assertAlmostEqual(result["ap"], 0.05)
        self.assertEqual(result["ap"], result["prevalence"])

    def test_event_tolerance_is_symmetric_and_boundary_local(self):
        left, right = np.zeros(10, dtype=bool), np.zeros(10, dtype=bool)
        left[-1] = True
        a = event_average_precision(left, np.zeros(10), fps=10)
        b = event_average_precision(right, np.zeros(10), fps=10)
        self.assertEqual(a["positive_frames"], 3)
        self.assertEqual(b["positive_frames"], 0)
        self.assertIsNone(b["ap"])
        self.assertEqual(b["reason"], "no_reference_events")
        labels = np.zeros(21, dtype=bool)
        labels[10] = True
        perfect = np.zeros(21)
        perfect[8:13] = 1
        self.assertEqual(event_average_precision(labels, perfect, fps=10)["ap"], 1)

    def test_one_class_and_empty_references_are_not_successful_zeroes(self):
        result = event_average_precision(np.ones(10), np.ones(10))
        self.assertIsNone(result["ap"])
        self.assertEqual(result["reason"], "one_class_reference")
        with self.assertRaises(ValueError):
            event_average_precision([], [])
        with self.assertRaises(ValueError):
            fit_scales([])

    def test_local_alignment_beats_known_shift(self):
        events = np.zeros(1000, dtype=bool)
        events[[100, 300, 530, 760]] = True
        scores = np.zeros(1000)
        for index in np.flatnonzero(events):
            scores[index - 2:index + 3] = 1
        aligned = event_average_precision(events, scores, fps=10)
        shifted = event_average_precision(events, np.roll(scores, 100), fps=10)
        self.assertEqual(aligned["ap"], 1)
        self.assertLess(shifted["ap"], 0.1)

    def test_continuous_nulls_preserve_score_count_and_are_reproducible(self):
        y = np.zeros((1200, 6))
        # Narrow ramps avoid flat-top novelty plateaus: strict peaks only.
        for index, value in zip([130, 360, 610, 850, 1060], [1, -2, 2, -1, 3]):
            y[index:] += value
            y[index - 1] += value * 0.2
        calibration = fit_scales([signal(1200)])
        calibration["novelty_threshold"] = 0.01
        a = scene_metrics(y, y, [], calibration, circular_shifts=20, include_arrays=True)
        b = scene_metrics(y, y, [], calibration, circular_shifts=20)
        self.assertTrue(a["novelty"]["eligible"])
        self.assertEqual(a["novelty"]["circular_shift_aps"], b["novelty"]["circular_shift_aps"])
        self.assertEqual(len(a["novelty"]["circular_shift_aps"]), 20)
        self.assertGreater(a["novelty_ap"], a["novelty"]["circular_shift_mean_ap"])
        scores = np.array(a["arrays"]["generated_novelty"])[a["arrays"]["valid_mask"]]
        for offset in a["novelty"]["circular_shift_offsets_frames"]:
            self.assertGreaterEqual(offset, 100)
            self.assertLessEqual(offset, len(scores) - 100)
            np.testing.assert_array_equal(np.sort(scores), np.sort(np.roll(scores, offset)))

    def test_constant_reference_has_explicit_missing_metrics_and_no_events(self):
        y = np.ones((1000, 6))
        result = scene_metrics(y, y, [], self.scales)
        self.assertIsNone(result["variation_ratio"])
        self.assertIsNone(result["hf_energy_ratio_2to5"])
        self.assertIsNone(result["novelty_ap"])
        self.assertFalse(result["novelty"]["eligible"])
        json.dumps(result, allow_nan=False)

    def test_recording_normalization_reuses_identical_overlap_targets(self):
        y = signal(300)
        active = np.arange(300) >= 20
        normalized, transform = normalize_recording_scene(y, active)
        np.testing.assert_allclose(normalized[active, 3:5].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(normalized[active, 3:5].std(axis=0), 1)
        np.testing.assert_array_equal(normalized[50:150][50:], normalized[100:200][:50])
        np.testing.assert_array_equal(normalized[:, :3], y[:, :3])
        self.assertEqual(transform["active_frames"], 280)

    def test_training_differences_do_not_cross_recording_boundaries(self):
        scales = fit_scales([np.zeros((100, 6)), np.ones((100, 6)) * 100])
        np.testing.assert_array_equal(scales["lag_iqr"], np.full((5, 6), 0.001))
        self.assertEqual(scales["novelty_threshold"], 0)

    def test_motion_ema_is_independent_and_constant_r2_stays_missing(self):
        x = np.column_stack([np.sin(np.arange(1000) / 40), np.cos(np.arange(1000) / 50)])
        names = ["com_speed", "vert_bounce_amp"]
        self.assertEqual(motion_r2(x, x, names)["primary_r2"], 1)
        self.assertIsNone(motion_r2(np.ones_like(x), np.ones_like(x), names)["primary_r2"])
        # Correct constant predictions on a subsequent recording must start in
        # that recording's own state, irrespective of earlier recording values.
        second = motion_r2(x + 100, x + 100, names)
        self.assertEqual(second["primary_r2"], 1)
        with self.assertRaises(ValueError):
            motion_r2(x, x)

    def test_bootstrap_resamples_both_axes_and_keeps_counts_honest(self):
        baseline = np.zeros((5, 40))
        candidate = np.arange(5)[:, None] + np.arange(40)[None, :] / 40
        result = paired_hierarchical_bootstrap(baseline, candidate, n_bootstrap=2000)
        self.assertEqual(result["n_training_seeds"], 5)
        self.assertEqual(result["n_songs"], 40)
        self.assertEqual(result["n_paired_cells"], 200)
        np.testing.assert_allclose(result["per_seed_delta"], candidate.mean(axis=1))
        # Seed-level offsets dominate and cannot be averaged away as 200 IID rows.
        self.assertGreater(result["ci_high"] - result["ci_low"], 1.5)
        inverse = paired_hierarchical_bootstrap(candidate, baseline, higher_is_better=False, n_bootstrap=2000)
        self.assertEqual(result["delta"], inverse["delta"])
        self.assertEqual(result["ci_low"], inverse["ci_low"])
        with self.assertRaises(ValueError):
            paired_hierarchical_bootstrap(np.empty((0, 40)), np.empty((0, 40)))
        with self.assertRaises(ValueError):
            paired_hierarchical_bootstrap([[np.nan]], [[1]])

    def test_gates_require_magnitude_replication_and_complete_controls(self):
        bootstrap = paired_hierarchical_bootstrap(np.zeros((5, 40)), np.ones((5, 40)) * 0.03, n_bootstrap=10)
        self.assertTrue(evaluate_motion_gate(bootstrap)["passed"])
        self.assertFalse(evaluate_motion_gate({})["passed"])
        baseline = {"temporal_error": 1, "marginal_error": 0.01}
        candidate = {"temporal_error": 0.8, "marginal_error": 0.025,
                     "variation_ratio": 0.8, "hf_energy_ratio_2to5": 1.2}
        temporal_bootstrap = paired_hierarchical_bootstrap(np.ones((5, 40)), np.ones((5, 40)) * 0.8,
                                                           higher_is_better=False, n_bootstrap=10)
        self.assertTrue(evaluate_temporal_gate(baseline, candidate, temporal_bootstrap, eligible_fraction=0.9)["passed"])
        self.assertFalse(evaluate_temporal_gate(baseline, candidate, bootstrap, eligible_fraction=0.9)["passed"])
        candidate["hf_energy_ratio_2to5"] = None
        self.assertFalse(evaluate_temporal_gate(baseline, candidate, temporal_bootstrap, eligible_fraction=0.9)["passed"])
        controls = {key: bootstrap for key in ("beat_only", "donor_audio", "shuffled_training", "circular_shift")}
        self.assertTrue(evaluate_conditioning_gate(controls, eligible_fraction=1)["passed"])
        del controls["donor_audio"]
        self.assertFalse(evaluate_conditioning_gate(controls, eligible_fraction=1)["passed"])

    def test_nonfinite_values_are_rejected_and_calibration_is_json_safe(self):
        json.dumps(self.scales, allow_nan=False)
        bad = self.reference.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            scene_metrics(self.reference, bad, [], self.scales)
        with self.assertRaises(ValueError):
            scene_novelty(self.reference, self.scales, fps=40)
        np.testing.assert_array_equal(beat_only_scores(10, []), np.zeros(10))


if __name__ == "__main__":
    unittest.main()

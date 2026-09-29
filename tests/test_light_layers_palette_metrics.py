"""Synthetic-only primary palette instruments; no data, training, torch, or GPU."""
from dataclasses import replace
from fractions import Fraction
import unittest

import numpy as np

from lightsem.model.light_layers.palette_metrics import (
    COMPONENTS, HS_DIM, MATH_ID, PaletteTrack, compare_palette_tracks,
    fit_palette_scales, joint_hs_distance, palette_prototypes_hash,
    summarize_palette_track, _empirical_w1, _weighted_iqr,
)


def point(hue, saturation=5):
    value = np.zeros(HS_DIM)
    value[hue*6+saturation] = 1.
    return value


PROTOTYPES = np.stack([point(0), point(4), point(8), point(0, 1)])
PROTOTYPE_HASH = palette_prototypes_hash(PROTOTYPES)


def track(histograms, *, tokens=None, cue_indices=None, valid=None, song="synthetic", recording="r"):
    histograms = np.asarray(histograms, dtype=float).copy()
    n = len(histograms)
    if valid is None:
        valid = np.ones(n, dtype=bool)
    histograms[~valid] = 0.
    if cue_indices is None:
        cue_indices = [0]
    if tokens is None:
        tokens = [0]*len(cue_indices)
    return PaletteTrack(song, recording, np.arange(n)*.1, histograms, np.asarray(valid, dtype=bool),
                        np.asarray(cue_indices)*.1, np.asarray(tokens, dtype=int), PROTOTYPE_HASH)


def held_track(tokens, cues, n, **kwargs):
    states = np.asarray(tokens)[np.searchsorted(cues, np.arange(n), side="right")-1]
    return track(PROTOTYPES[states], tokens=tokens, cue_indices=cues, **kwargs)


def synthetic_train():
    records = []
    for i, (n, step, base, amplitude) in enumerate(((120, 10, .0, .10), (140, 20, .2, .15),
                                                  (170, 40, .45, .20), (200, 100, .7, .25))):
        mix = base + amplitude*np.arange(n)/(n-1)
        histograms = (1-mix[:, None])*PROTOTYPES[0] + mix[:, None]*PROTOTYPES[1]
        cues = np.arange(0, n, step)
        choices = [0, 3] if i == 0 else [0, 1] if i == 1 else [1, 2] if i == 2 else [0, 2]
        tokens = np.array(choices)[np.arange(len(cues)) % 2]
        records.append(track(histograms, tokens=tokens, cue_indices=cues, song=f"train-{i}"))
    return records


class JointHSDistanceTests(unittest.TestCase):
    def test_identity_opposite_point_known_geometry_and_symmetry(self):
        red, opposite = point(0), point(6)
        self.assertEqual(joint_hs_distance(red, red), 0.)
        expected = np.mean(np.abs(np.cos(np.arange(36)*np.pi/36)))
        self.assertAlmostEqual(joint_hs_distance(red, opposite), expected, places=14)
        self.assertAlmostEqual(joint_hs_distance(red, point(4)), joint_hs_distance(point(4), red))
        self.assertGreater(joint_hs_distance(red, opposite), .6)
        self.assertLessEqual(joint_hs_distance(red, opposite), 1.)

    def test_circular_wrap_and_achromatic_origin(self):
        # Equally close sides of hue wrap, using circular joint-bin interpolation.
        near_left = .99*point(0) + .01*point(11)
        near_right = .99*point(0) + .01*point(1)
        self.assertLess(joint_hs_distance(near_left, near_right), .004)
        self.assertAlmostEqual(joint_hs_distance(point(0), point(11)),
                               joint_hs_distance(point(0), point(1)), places=14)
        self.assertEqual(joint_hs_distance(point(0, 0), point(7, 0)), 0.)

    def test_joint_correlation_not_separate_marginal_geometry(self):
        a = .5*point(0, 1) + .5*point(6, 5)
        b = .5*point(0, 5) + .5*point(6, 1)
        np.testing.assert_array_equal(a.reshape(12, 6).sum(0), b.reshape(12, 6).sum(0))
        np.testing.assert_array_equal(a.reshape(12, 6).sum(1), b.reshape(12, 6).sum(1))
        self.assertGreater(joint_hs_distance(a, b), .2)

    def test_normalization_and_schema_width_not_guessed(self):
        for bad in (np.zeros(HS_DIM), point(0)*2, np.ones(6), np.full(HS_DIM, np.nan)):
            with self.assertRaises(ValueError):
                joint_hs_distance(bad, point(0))
        with self.assertRaises(ValueError):
            palette_prototypes_hash(np.zeros((0, HS_DIM)))
        self.assertEqual(palette_prototypes_hash(PROTOTYPES.copy()), PROTOTYPE_HASH)
        self.assertNotEqual(palette_prototypes_hash(PROTOTYPES[::-1]), PROTOTYPE_HASH)

    def test_empirical_w1_exact_samples_and_undefined(self):
        self.assertEqual(_empirical_w1(np.array([0., 1.]), np.array([1.])), .5)
        self.assertEqual(_empirical_w1(np.array([0., 2.]), np.array([1., 1.])), 1.)
        self.assertEqual(_empirical_w1(np.ones(5), np.ones(1)), 0.)
        self.assertIsNone(_empirical_w1(np.empty(0), np.ones(1)))


class TemporalSummaryTests(unittest.TestCase):
    def test_piecewise_held_drift_zero_residence_and_mark_prototypes(self):
        example = held_track([0, 0, 1, 1, 2], [0, 20, 40, 80, 100], 120)
        summary = summarize_palette_track(example, PROTOTYPES)
        np.testing.assert_allclose(summary.component_samples[COMPONENTS[0]], np.log1p([4., 6., 2.]))
        expected_marks = [joint_hs_distance(PROTOTYPES[0], PROTOTYPES[1]),
                          joint_hs_distance(PROTOTYPES[1], PROTOTYPES[2])]
        np.testing.assert_allclose(summary.component_samples[COMPONENTS[1]], expected_marks)
        np.testing.assert_array_equal(summary.component_samples[COMPONENTS[2]], [0.])
        self.assertEqual(summary.observed_change_count, 2)
        self.assertFalse(summary.no_observed_change)
        self.assertEqual(summary.intercue_pair_count, 119-4)
        self.assertEqual(summary.eight_second_pair_count, 40)
        self.assertEqual(summary.censored_residence_segments, 2)
        self.assertEqual(summary.censored_residence_edges, 2)

    def test_raw_observer_trajectory_is_not_replaced_by_token_prototypes(self):
        mix = np.linspace(0., 1., 101)
        hist = (1-mix[:, None])*PROTOTYPES[0] + mix[:, None]*PROTOTYPES[1]
        example = track(hist)
        summary = summarize_palette_track(example, PROTOTYPES)
        whole_distance = joint_hs_distance(PROTOTYPES[0], PROTOTYPES[1])
        self.assertAlmostEqual(summary.component_samples[COMPONENTS[2]][0], whole_distance/10.)
        np.testing.assert_allclose(summary.component_samples[COMPONENTS[3]], .8*whole_distance)
        np.testing.assert_allclose(summary.marginal_hs, .5*(PROTOTYPES[0]+PROTOTYPES[1]))
        np.testing.assert_array_equal(summary.component_samples[COMPONENTS[1]], [0.])
        self.assertTrue(summary.no_observed_change)

    def test_invalid_span_splits_residence_and_blocks_entire_lag_path(self):
        valid = np.ones(120, dtype=bool)
        valid[40:60] = False
        example = track(np.tile(PROTOTYPES[0], (120, 1)), valid=valid)
        summary = summarize_palette_track(example, PROTOTYPES)
        np.testing.assert_allclose(summary.component_samples[COMPONENTS[0]], np.log1p([4., 6.]))
        self.assertEqual(summary.censored_residence_segments, 2)
        self.assertEqual(summary.censored_residence_edges, 4)
        self.assertEqual(summary.eight_second_pair_count, 0)  # endpoints alone would be wrong
        self.assertEqual(summary.intercue_pair_count, 98)
        self.assertAlmostEqual(summary.observed_valid_time_fraction, 100/120)
        self.assertEqual(summary.valid_duration_s, 10.)
        self.assertIn("lower_bounds", summary.residence_scope)

    def test_exact_8second_lag_and_short_track_undefined(self):
        short = summarize_palette_track(track(np.tile(PROTOTYPES[0], (80, 1))), PROTOTYPES)
        exact = summarize_palette_track(track(np.tile(PROTOTYPES[0], (81, 1))), PROTOTYPES)
        self.assertEqual(short.eight_second_pair_count, 0)
        self.assertEqual(exact.eight_second_pair_count, 1)
        np.testing.assert_array_equal(exact.component_samples[COMPONENTS[3]], [0.])
        valid = np.ones(81, dtype=bool)
        valid[40] = False
        blocked = summarize_palette_track(track(np.tile(PROTOTYPES[0], (81, 1)), valid=valid), PROTOTYPES)
        self.assertEqual(blocked.eight_second_pair_count, 0)

    def test_invalid_change_count_and_no_change_flag(self):
        valid = np.ones(120, dtype=bool)
        valid[59] = False
        summary = summarize_palette_track(held_track([0, 1], [0, 60], 120, valid=valid), PROTOTYPES)
        self.assertEqual(summary.invalid_change_count, 1)
        self.assertEqual(summary.observed_change_count, 0)
        self.assertTrue(summary.no_observed_change)
        np.testing.assert_array_equal(summary.component_samples[COMPONENTS[1]], [0.])
        # An earlier invalid span does not erase a later directly observed boundary.
        valid[:] = True
        valid[20] = False
        direct = summarize_palette_track(held_track([0, 1], [0, 60], 120, valid=valid), PROTOTYPES)
        self.assertEqual(direct.observed_change_count, 1)

    def test_all_invalid_is_undefined_not_constant_color_success(self):
        example = track(np.tile(PROTOTYPES[0], (120, 1)), valid=np.zeros(120, dtype=bool))
        summary = summarize_palette_track(example, PROTOTYPES)
        self.assertTrue(all(not len(samples) for samples in summary.component_samples.values()))
        self.assertIsNone(summary.marginal_hs)
        self.assertIsNone(summary.no_observed_change)
        self.assertEqual(summary.observed_valid_time_fraction, 0.)

    def test_unknown_schema_bad_gaps_ids_and_tokens_rejected(self):
        example = held_track([0, 1], [0, 60], 120)
        bad_times = example.times_s.copy()
        bad_times[50:] += .1
        invalid_hist = example.hs_histograms.copy()
        invalid_hist[0] = 0
        for bad in (replace(example, schema_version=99), replace(example, target_schema_version=99),
                    replace(example, canonical_song=""), replace(example, recording=""),
                    replace(example, prototypes_sha256="wrong"), replace(example, times_s=bad_times),
                    replace(example, cue_times_s=np.array([0., 6.05])),
                    replace(example, token_ids=np.array([0, 8])),
                    replace(example, token_ids=np.array([0., 1.])),
                    replace(example, hs_histograms=invalid_hist),
                    replace(example, valid=np.zeros(120, dtype=bool))):
            with self.subTest(bad=bad.schema_version), self.assertRaises(ValueError):
                summarize_palette_track(bad, PROTOTYPES)


class CalibrationAndComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic_train()
        cls.calibration = fit_palette_scales(cls.train, PROTOTYPES, partition="TRAIN")

    def test_calibration_train_only_hash_content_and_order(self):
        calibration = self.calibration
        self.assertEqual(calibration.partition, "TRAIN")
        self.assertEqual(calibration.math_id, MATH_ID)
        self.assertEqual(calibration.prototypes_sha256, PROTOTYPE_HASH)
        self.assertEqual(len(calibration.content_sha256), 64)
        self.assertEqual(len(calibration.training_content_sha256), 64)
        self.assertTrue(all(scale > 0 for scale in calibration.component_iqrs))
        self.assertGreater(calibration.marginal_iqr, 0.)
        reversed_input = fit_palette_scales(self.train[::-1], PROTOTYPES, partition="TRAIN")
        self.assertEqual(calibration, reversed_input)
        for partition in ("DEV", "TEST", "train", ""):
            with self.assertRaises(ValueError):
                fit_palette_scales(self.train, PROTOTYPES, partition=partition)

    def test_recordings_do_not_dominate_canonical_scale_weights(self):
        # Duplicating A's one record many times under unique recording IDs must
        # not increase song A's mass relative to other canonical songs.
        duplicated = [replace(self.train[0], recording=f"r{i}") for i in range(7)] + self.train[1:]
        calibration = fit_palette_scales(duplicated, PROTOTYPES, partition="TRAIN")
        np.testing.assert_allclose(calibration.component_iqrs, self.calibration.component_iqrs, atol=1e-14)
        self.assertAlmostEqual(calibration.marginal_iqr, self.calibration.marginal_iqr, places=14)
        np.testing.assert_allclose(calibration.training_mean_hs, self.calibration.training_mean_hs, atol=1e-14)
        self.assertNotEqual(calibration.training_content_sha256, self.calibration.training_content_sha256)

    def test_exact_weighted_inverse_cdf_iqr_and_sample_duplication(self):
        # Two songs: one contributes 100 zero samples, the other [1,2]. Equal song
        # weighting yields Q25=0,Q75=1, not the frame-pooled collapsed IQR=0.
        values = np.r_[np.zeros(100), [1., 2.]]
        weights = [Fraction(1, 200)]*100 + [Fraction(1, 4)]*2
        self.assertEqual(_weighted_iqr(values, weights), 1.)
        doubled = np.repeat(values, 2)
        doubled_weights = [weight/2 for weight in weights for _ in range(2)]
        self.assertEqual(_weighted_iqr(doubled, doubled_weights), 1.)

    def test_identity_self_floor_zero_all_components(self):
        reference = self.train[0]
        score = compare_palette_tracks(reference, reference, PROTOTYPES, self.calibration)
        self.assertEqual(score.temporal_mean_scaled, 0.)
        self.assertEqual(score.marginal_raw, 0.)
        self.assertEqual(score.marginal_scaled, 0.)
        self.assertEqual(score.undefined_components, ())
        self.assertTrue(all(value == 0 for value in score.temporal_raw.values()))
        self.assertIn("not_aesthetic_or_rhythm", score.valid_scope)

    def test_constant_independent_held_and_common_cue_controls(self):
        cues = np.arange(0, 120, 10)
        reference_tokens = [0]*6 + [1]*6
        reference = held_track(reference_tokens, cues, 120)
        independent = held_track([0, 1]*6, cues, 120)
        constant = held_track([0]*12, cues, 120)
        independent_score = compare_palette_tracks(independent, reference, PROTOTYPES, self.calibration)
        constant_score = compare_palette_tracks(constant, reference, PROTOTYPES, self.calibration)
        self.assertAlmostEqual(independent_score.marginal_raw, 0., places=14)
        self.assertGreater(independent_score.temporal_raw[COMPONENTS[0]], 0.)
        self.assertGreater(independent_score.temporal_mean_scaled, 0.)
        self.assertGreater(constant_score.marginal_raw, 0.)
        self.assertGreater(constant_score.temporal_mean_scaled, 0.)
        self.assertTrue(constant_score.generated.no_observed_change)
        # Explicitly visible uninformative component, not a hidden aggregate rescue.
        self.assertEqual(independent_score.temporal_raw[COMPONENTS[2]], 0.)
        self.assertEqual(constant_score.temporal_raw[COMPONENTS[2]], 0.)
        expected = np.mean(list(constant_score.temporal_scaled.values()))
        self.assertEqual(constant_score.temporal_mean_scaled, expected)

    def test_quantization_floor_is_separate_caller_control(self):
        raw_reference = self.train[1]
        cues = np.rint(raw_reference.cue_times_s/.1).astype(int)
        quantized = held_track(raw_reference.token_ids, cues, len(raw_reference.times_s),
                               song=raw_reference.canonical_song, recording=raw_reference.recording)
        floor = compare_palette_tracks(quantized, raw_reference, PROTOTYPES, self.calibration)
        self.assertGreater(floor.marginal_raw, 0.)
        self.assertEqual(floor.temporal_raw[COMPONENTS[0]], 0.)
        self.assertEqual(floor.temporal_raw[COMPONENTS[1]], 0.)
        self.assertGreater(floor.temporal_raw[COMPONENTS[2]], 0.)

    def test_undefined_component_never_averages_survivors(self):
        short = held_track([0, 1], [0, 40], 80)
        score = compare_palette_tracks(short, short, PROTOTYPES, self.calibration)
        self.assertEqual(score.undefined_components, (COMPONENTS[3],))
        self.assertIsNone(score.temporal_mean_scaled)
        self.assertEqual(score.temporal_scaled[COMPONENTS[0]], 0.)
        self.assertEqual(score.marginal_scaled, 0.)
        invalid = track(np.tile(PROTOTYPES[0], (100, 1)), valid=np.zeros(100, dtype=bool))
        absent = compare_palette_tracks(invalid, invalid, PROTOTYPES, self.calibration)
        self.assertEqual(absent.undefined_components, COMPONENTS)
        self.assertIsNone(absent.temporal_mean_scaled)
        self.assertIsNone(absent.marginal_scaled)

    def test_mismatched_cues_validity_ids_and_vocabulary_fail(self):
        reference = self.train[0]
        changed_cues = reference.cue_times_s.copy()
        changed_cues[-1] += .1
        validity = reference.valid.copy()
        validity[0] = False
        hist = reference.hs_histograms.copy()
        hist[0] = 0
        for generated in (replace(reference, cue_times_s=changed_cues),
                          replace(reference, valid=validity, hs_histograms=hist),
                          replace(reference, canonical_song="other"),
                          replace(reference, recording="other")):
            with self.assertRaises(ValueError):
                compare_palette_tracks(generated, reference, PROTOTYPES, self.calibration)
        with self.assertRaises(ValueError):
            compare_palette_tracks(reference, reference, PROTOTYPES[::-1], self.calibration)

    def test_scale_tamper_schema_and_degenerate_fit_are_blocked(self):
        reference = self.train[0]
        altered = replace(self.calibration, marginal_iqr=self.calibration.marginal_iqr*2)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            compare_palette_tracks(reference, reference, PROTOTYPES, altered)
        for altered in (replace(self.calibration, schema_version=99),
                        replace(self.calibration, partition="DEV"),
                        replace(self.calibration, content_sha256="bad"),
                        replace(self.calibration, training_content_sha256="bad")):
            with self.assertRaises(ValueError):
                compare_palette_tracks(reference, reference, PROTOTYPES, altered)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            fit_palette_scales(self.train+self.train[:1], PROTOTYPES, partition="TRAIN")
        constant = held_track([0], [0], 100)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            fit_palette_scales([constant], PROTOTYPES, partition="TRAIN")
        short = held_track([0], [0], 80)
        with self.assertRaisesRegex(ValueError, "undefined"):
            fit_palette_scales([short], PROTOTYPES, partition="TRAIN")


if __name__ == "__main__":
    unittest.main()

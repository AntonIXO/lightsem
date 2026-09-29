"""Synthetic-only metric controls. No datasets, labels, vault, GPU, or torch."""
import itertools
import unittest
from dataclasses import replace

import numpy as np

from lightsem.model.light_layers.metrics import (
    BOOTSTRAP_REPLICATES, VISUAL_TOLERANCE_S, MetricObservation,
    fixed_shift_metadata, holm_adjust, macro_song_mean, paired_song_bootstrap,
    palette_metrics, visual_event_ap,
)


def events(reference, candidates, scores, **kwargs):
    arguments = dict(duration_s=60., deployment_threshold=.5, channel="color")
    arguments.update(kwargs)
    return visual_event_ap(reference, candidates, scores, **arguments)


def row(song, value, *, recording="recording", coach="coach", seed=0, draw=0):
    return MetricObservation(song, recording, coach, seed, draw, value)


def brute_matches(reference, candidates, tolerance=VISUAL_TOLERANCE_S):
    """Exhaustive oracle for tiny graphs, independent of augmenting-path code."""
    if not len(candidates):
        return 0
    best = brute_matches(reference, candidates[1:], tolerance)
    for index, target in enumerate(reference):
        if abs(candidates[0] - target) <= tolerance + 1e-12:
            rest = list(reference[:index]) + list(reference[index+1:])
            best = max(best, 1 + brute_matches(rest, candidates[1:], tolerance))
    return best


class VisualEventTests(unittest.TestCase):
    def test_positive_and_negative_and_channel_scope(self):
        for channel in ("color", "accent"):
            positive = events([1., 4., 8.], [1., 4., 8.], [.9, .8, .7], channel=channel)
            self.assertEqual(positive.average_precision, 1.)
            self.assertEqual(positive.matched_count, 3)
            self.assertEqual(positive.false_cues_per_minute, 0.)
            self.assertEqual(positive.tolerance_s, .2)
            self.assertIn("not_acoustic_beat_accuracy", positive.scope)
            negative = events([1., 4., 8.], [2., 5., 9.], [.9, .8, .7], channel=channel)
            self.assertEqual(negative.average_precision, 0.)
            self.assertEqual(negative.false_cues_per_minute, 3.)

    def test_confidence_ranking_and_operating_threshold_are_separate(self):
        good = events([1.], [1., 3.], [.9, .3])
        bad = events([1.], [1., 3.], [.3, .9])
        self.assertEqual(good.average_precision, 1.)
        self.assertEqual(bad.average_precision, .5)
        self.assertEqual(good.emitted_count, 1)
        self.assertEqual(good.matched_count, 1)
        self.assertEqual(bad.emitted_count, 1)
        self.assertEqual(bad.matched_count, 0)
        self.assertEqual(bad.false_cues_per_minute, 1.)
        abstain = events([1.], [1., 3.], [.9, .3], deployment_threshold=1.)
        self.assertEqual(abstain.average_precision, 1.)
        self.assertEqual(abstain.emitted_count, 0)
        self.assertEqual(abstain.false_cues_per_minute, 0.)

    def test_tied_groups_never_choose_lucky_input_order(self):
        for permutation in itertools.permutations([1., 3., 5.]):
            result = events([1., 3.], permutation, [.5, .5, .5])
            self.assertAlmostEqual(result.average_precision, 2/3)
            np.testing.assert_array_equal(result.confidence_thresholds, [.5])
            np.testing.assert_array_equal(result.true_positives, [2])
            self.assertEqual(result.emitted_count, 3)  # >= threshold, not >
        grouped = events([1., 3., 7.], [1., 5., 3.], [.9, .8, .8])
        self.assertAlmostEqual(grouped.average_precision, 5/9)
        np.testing.assert_allclose(grouped.precision, [1., 2/3])
        np.testing.assert_allclose(grouped.recall, [1/3, 2/3])

    def test_one_to_one_duplicate_predictions_and_target_validation(self):
        result = events([1.], [1., 1., 1.], [.7, .7, .7])
        self.assertEqual(result.matched_count, 1)
        self.assertAlmostEqual(result.average_precision, 1/3)
        self.assertEqual(result.false_cues_per_minute, 2.)
        one_prediction = events([1., 1.1], [1.05], [.8])
        self.assertEqual(one_prediction.average_precision, .5)
        with self.assertRaisesRegex(ValueError, "duplicate reference"):
            events([1., 1.], [1.], [.8])

    def test_augmenting_path_not_nearest_greedy(self):
        result = events([1., 1.3], [1.15, .85], [.9, .8])
        self.assertEqual(result.matched_count, 2)
        self.assertEqual(result.average_precision, 1.)
        # Within-tie graph matching must also remain order invariant.
        for candidate in ([1.15, .85], [.85, 1.15]):
            self.assertEqual(events([1., 1.3], candidate, [.8, .8]).average_precision, 1.)

    def test_matching_against_exhaustive_small_graphs(self):
        rng = np.random.default_rng(31)
        for _ in range(100):
            references = sorted(rng.choice(np.arange(0, 1, .1), size=4, replace=False))
            candidates = rng.choice(np.arange(0, 1, .05), size=5, replace=True).tolist()
            scores = rng.choice([.2, .5, .9], size=5)
            result = events(references, candidates, scores)
            for threshold, actual in zip(result.confidence_thresholds, result.true_positives):
                selected = [time for time, score in zip(candidates, scores) if score >= threshold]
                self.assertEqual(actual, brute_matches(references, selected))

    def test_fixed_visual_tolerance_inclusive_both_sides(self):
        for prediction in (.8, 1.2):
            self.assertEqual(events([1.], [prediction], [.9]).average_precision, 1.)
        for prediction in (.8 - 1e-8, 1.2 + 1e-8):
            self.assertEqual(events([1.], [prediction], [.9]).average_precision, 0.)
        # A .19 s error is a VISUAL match. It cannot demonstrate 70 ms accuracy.
        self.assertEqual(events([1.], [1.19], [.9]).average_precision, 1.)
        with self.assertRaises(TypeError):
            events([1.], [1.], [.9], tolerance_s=.07)

    def test_empty_no_event_and_no_coverage_are_distinct(self):
        positive_empty = events([1.], [], [])
        self.assertEqual(positive_empty.average_precision, 0.)
        no_events = events([], [2., 8.], [.9, .6])
        self.assertIsNone(no_events.average_precision)
        self.assertEqual(no_events.false_cues_per_minute, 2.)
        self.assertEqual(no_events.coverage, 1.)
        self.assertEqual(no_events.recall.size, 0)
        empty = events([], [], [])
        self.assertIsNone(empty.average_precision)
        self.assertEqual(empty.false_cues_per_minute, 0.)
        self.assertEqual(empty.reference_events_per_minute, 0.)
        invalid = events([], [], [], eligible_intervals_s=[])
        self.assertIsNone(invalid.average_precision)
        self.assertIsNone(invalid.false_cues_per_minute)
        self.assertEqual(invalid.coverage, 0.)

    def test_eligibility_boundary_and_false_cue_denominator(self):
        result = events([], [0., 20.], [.8, .8], eligible_intervals_s=[[0., 10.], [20., 40.]])
        self.assertEqual(result.eligible_duration_s, 30.)
        self.assertEqual(result.coverage, .5)
        self.assertEqual(result.false_cues_per_minute, 4.)
        for time in (-.01, 10., 19.9, 40., 60.):
            with self.assertRaisesRegex(ValueError, "outside eligible"):
                events([], [time], [.8], eligible_intervals_s=[[0., 10.], [20., 40.]])
        with self.assertRaisesRegex(ValueError, "outside eligible"):
            events([1.], [], [], eligible_intervals_s=[])
        # A gap cannot be bridged even if absolute event error is < .2 s.
        gap = events([1.], [1.1], [.9], eligible_intervals_s=[[0., 1.05], [1.08, 2.]])
        self.assertEqual(gap.average_precision, 0.)
        contiguous = events([1.], [1.1], [.9], eligible_intervals_s=[[0., 1.05], [1.05, 2.]])
        self.assertEqual(contiguous.average_precision, 1.)

    def test_fixed_shift_negative_control(self):
        reference = np.array([1., 8., 17.])
        metadata = fixed_shift_metadata("synthetic-song", 60., np.arange(1., 21.),
                                        minimum_circular_distance_s=.5)
        shifted = (reference + metadata["offsets_modulo_s"][0]) % metadata["duration_s"]
        self.assertEqual(events(reference, reference, [.9]*3).average_precision, 1.)
        self.assertEqual(events(reference, shifted, [.9]*3).average_precision, 0.)

    def test_invalid_inputs_fail_loudly(self):
        cases = [dict(duration_s=0), dict(duration_s=np.inf), dict(channel="beat"),
                 dict(deployment_threshold=np.nan), dict(eligible_intervals_s=[[0, 61]]),
                 dict(eligible_intervals_s=[[3, 4], [1, 2]]),
                 dict(eligible_intervals_s=[[0, 4], [3, 5]]),
                 dict(eligible_intervals_s=[[2, 2]])]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                events([], [], [], **overrides)
        for reference, candidate, score in [([np.nan], [], []), ([], [1], [np.inf]),
                                            ([], [1], []), ([[1]], [], [])]:
            with self.assertRaises(ValueError):
                events(reference, candidate, score)


class SongAggregationTests(unittest.TestCase):
    def test_macro_song_not_recording_coach_or_draw_pooling(self):
        observations = []
        for seed in (0, 1, 2):
            for draw in (0, 1):
                # Song A: mean(recording R=mean(coaches=0,1), recording S=1)=.75.
                observations.extend([row("A", 0., recording="R", coach="c1", seed=seed, draw=draw),
                                     row("A", 1., recording="R", coach="c2", seed=seed, draw=draw),
                                     row("A", 1., recording="S", seed=seed, draw=draw),
                                     row("B", 0., seed=seed, draw=draw)])
        result = macro_song_mean(observations, expected_songs=["A", "B"],
                                 expected_seeds=[0, 1, 2], expected_draws=[0, 1])
        self.assertEqual(result.values, {"A": .75, "B": 0.})
        self.assertEqual(result.macro_mean, .375)
        self.assertEqual(result.defined_observations, 24)
        self.assertEqual(result.song_coverage, 1.)

    def test_no_event_ap_is_counted_and_false_cues_stay_in_macro(self):
        positive, quiet = events([1.], [1.], [.9]), events([], [3., 8.], [.9, .9])
        arguments = dict(expected_songs=["positive", "quiet"], expected_seeds=[0], expected_draws=[0])
        ap = macro_song_mean([row("positive", positive.average_precision),
                              row("quiet", quiet.average_precision)], **arguments)
        false_cues = macro_song_mean([row("positive", positive.false_cues_per_minute),
                                      row("quiet", quiet.false_cues_per_minute)], **arguments)
        coverage = macro_song_mean([row("positive", positive.coverage), row("quiet", quiet.coverage)],
                                   **arguments)
        self.assertEqual(ap.macro_mean, 1.)
        self.assertEqual(ap.undefined_songs, ("quiet",))
        self.assertEqual(ap.undefined_observations, 1)
        self.assertEqual(ap.song_coverage, .5)
        self.assertEqual(false_cues.macro_mean, 1.)
        self.assertEqual(false_cues.song_coverage, 1.)
        self.assertEqual(coverage.macro_mean, 1.)

    def test_mixed_positive_and_no_event_recordings_explicit_counts(self):
        result = macro_song_mean([row("A", .4, recording="p"), row("A", None, recording="q")],
                                 expected_songs=["A"], expected_seeds=[0], expected_draws=[0])
        self.assertEqual(result.macro_mean, .4)
        self.assertEqual(result.defined_observations_by_song, {"A": 1})
        self.assertEqual(result.total_observations_by_song, {"A": 2})
        self.assertEqual(result.undefined_observations, 1)
        all_none = macro_song_mean([row("A", None)], expected_songs=["A"],
                                   expected_seeds=[0], expected_draws=[0])
        self.assertIsNone(all_none.macro_mean)
        self.assertEqual(all_none.song_coverage, 0.)

    def test_missing_duplicate_nan_or_seed_selected_cells_rejected(self):
        defaults = dict(expected_songs=["A"], expected_seeds=[0], expected_draws=[0])
        for observations in ([row("A", 1.), row("A", 2.)], [row("A", np.nan)], []):
            with self.assertRaises(ValueError):
                macro_song_mean(observations, **defaults)
        with self.assertRaises(ValueError):
            macro_song_mean([row("A", 1.)], **(defaults | {"expected_seeds": [0, 1, 2]}))
        with self.assertRaises(ValueError):
            macro_song_mean([row("A", 1.)], **(defaults | {"expected_draws": [0, 1]}))
        with self.assertRaises(ValueError):
            macro_song_mean([row("A", 1.)], **(defaults | {"expected_songs": ["A", "B"]}))
        with self.assertRaises(ValueError):
            macro_song_mean([row("A", 1.), row("A", None, seed=1)],
                            **(defaults | {"expected_seeds": [0, 1]}))
        for key in defaults:
            with self.assertRaises(ValueError):
                macro_song_mean([row("A", 1.)], **(defaults | {key: []}))

    def test_paired_bootstrap_exact_unit_and_rng(self):
        a = [row("A", .8), row("B", .3), row("C", .9), row("quiet", None)]
        b = [row("A", .4), row("B", .2), row("C", .7), row("quiet", None)]
        result = paired_song_bootstrap(a, b, expected_songs=["A", "B", "C", "quiet"],
                                       expected_seeds=[0], expected_draws=[0], random_seed=19)
        differences = np.array([.8-.4, .3-.2, .9-.7])
        rng = np.random.default_rng(19)
        means = differences[rng.integers(0, 3, size=(10_000, 3))].mean(axis=1)
        np.testing.assert_allclose(result.interval, np.quantile(means, [.025, .975]))
        self.assertEqual(result.replicates, 10_000)
        self.assertEqual(result.difference, differences.mean())
        self.assertEqual(result.excluded_undefined_songs, ("quiet",))
        self.assertEqual(result.song_coverage, .75)
        expected_p = (1 + np.count_nonzero(np.abs(means-differences.mean()) >= abs(differences.mean())))/10_001
        self.assertEqual(result.p_value_two_sided, expected_p)

    def test_seed_draw_duplication_does_not_create_independent_units(self):
        a = [row("A", .8), row("B", .4)]
        b = [row("A", .3), row("B", .2)]
        basic = dict(expected_songs=["A", "B"], expected_seeds=[0], expected_draws=[0])
        original = paired_song_bootstrap(a, b, **basic)
        expanded_a = [replace(item, seed=seed, draw=draw) for item in a
                      for seed in (0, 1, 2) for draw in (0, 1, 2, 3)]
        expanded_b = [replace(item, seed=seed, draw=draw) for item in b
                      for seed in (0, 1, 2) for draw in (0, 1, 2, 3)]
        expanded = paired_song_bootstrap(expanded_a, expanded_b, expected_songs=["A", "B"],
                                         expected_seeds=[0, 1, 2], expected_draws=[0, 1, 2, 3])
        np.testing.assert_allclose(original.interval, expanded.interval, atol=1e-15)
        self.assertAlmostEqual(original.difference, expanded.difference)
        self.assertEqual(len(expanded.song_differences), 2)
        identity = paired_song_bootstrap(a, a, **basic)
        self.assertEqual(identity.difference, 0.)
        self.assertEqual(identity.interval, (0., 0.))
        self.assertEqual(identity.p_value_two_sided, 1.)
        reversed_rows = paired_song_bootstrap(a[::-1], b[::-1], **basic)
        self.assertEqual(original, reversed_rows)

    def test_unpaired_bootstrap_cannot_drop_failure(self):
        a = [row("A", 1.), row("B", .5)]
        b = [row("A", .5), row("B", None)]
        arguments = dict(expected_songs=["A", "B"], expected_seeds=[0], expected_draws=[0])
        with self.assertRaisesRegex(ValueError, "eligibility"):
            paired_song_bootstrap(a, b, **arguments)
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_song_bootstrap(a, b[:1], **arguments)
        with self.assertRaisesRegex(ValueError, "at least two"):
            paired_song_bootstrap(a[:1], a[:1], expected_songs=["A"],
                                  expected_seeds=[0], expected_draws=[0])
        with self.assertRaises(ValueError):
            paired_song_bootstrap(a, a, **arguments, confidence=1.)


class StatisticalHelpersTests(unittest.TestCase):
    def test_holm_known_example_ties_and_empty(self):
        result = holm_adjust({"b": .04, "a": .01, "d": .8, "c": .03})
        self.assertEqual(result, {"b": .09, "a": .04, "d": .8, "c": .09})
        self.assertEqual(holm_adjust({"b": .02, "a": .02, "c": .7}),
                         {"b": .06, "a": .06, "c": .7})
        self.assertEqual(holm_adjust({}), {})
        self.assertEqual(holm_adjust({"x": 0., "y": 1.}), {"x": 0., "y": 1.})
        for invalid in (-.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                holm_adjust({"a": invalid})

    def test_fixed_shifts_metadata_exact_signed_wrap_and_validation(self):
        offsets = np.arange(1., 21.)
        offsets[0] = -1.
        result = fixed_shift_metadata("song-A", 60., offsets, minimum_circular_distance_s=.5)
        self.assertEqual(result["shift_count"], 20)
        self.assertEqual(result["duration_s"], 60.)
        self.assertEqual(result["offsets_s"][0], -1.)
        self.assertEqual(result["offsets_modulo_s"][0], 59.)
        self.assertEqual(result["circular_distances_s"][0], 1.)
        self.assertIn("within_one_canonical_song_only", result["scope"])
        for bad in (np.arange(19.), np.arange(20.), np.r_[1., 61., np.arange(2., 20.)],
                    np.r_[np.nan, np.arange(1., 20.)]):
            with self.assertRaises(ValueError):
                fixed_shift_metadata("A", 60., bad, minimum_circular_distance_s=.5)
        with self.assertRaises(ValueError):
            fixed_shift_metadata("A", 60., offsets, minimum_circular_distance_s=.2)
        with self.assertRaises(ValueError):
            fixed_shift_metadata("", 60., offsets, minimum_circular_distance_s=.5)


class PaletteTests(unittest.TestCase):
    def test_reference_self_floor_and_normalized_histograms(self):
        reference = np.array([0, 0, 0, 1, 1, 2])
        result = palette_metrics(reference, reference, np.arange(6.), duration_s=6., vocabulary_size=4)
        self.assertEqual(result.categorical_time_occupancy_tv, 0.)
        self.assertEqual(result.categorical_cue_transition_joint_tv, 0.)
        self.assertEqual(result.categorical_changed_transition_joint_tv, 0.)
        self.assertIn("not_preregistered", result.scope)
        for histogram in (result.generated.occupancy, result.generated.cue_transition_joint,
                          result.generated.changed_transition_joint):
            self.assertAlmostEqual(histogram.sum(), 1.)
        self.assertEqual(result.generated.occupied_vocabulary_fraction, .75)
        np.testing.assert_allclose(result.generated.observed_residence_s, [3., 2., 1.])

    def test_occupancy_is_duration_weighted_not_cue_count(self):
        result = palette_metrics([0, 1], [1, 0], [0., 1.], duration_s=10., vocabulary_size=2)
        np.testing.assert_allclose(result.generated.occupancy, [.1, .9])
        self.assertAlmostEqual(result.categorical_time_occupancy_tv, .8)
        self.assertEqual(result.categorical_cue_transition_joint_tv, 1.)
        self.assertEqual(result.categorical_changed_transition_joint_tv, 1.)
        self.assertEqual(result.generated.changes_per_minute, 6.)
        self.assertEqual(result.generated.keep_fraction, 0.)

    def test_constant_palette_does_not_pass_diversity_or_changed_metric(self):
        result = palette_metrics([0]*8, [0, 0, 1, 1, 2, 2, 3, 3], np.arange(8.),
                                 duration_s=8., vocabulary_size=4)
        self.assertEqual(result.generated.effective_states, 1.)
        self.assertEqual(result.generated.entropy_normalized, 0.)
        self.assertEqual(result.generated.occupied_vocabulary_fraction, .25)
        self.assertEqual(result.reference.occupied_vocabulary_fraction, 1.)
        self.assertLess(result.generated.effective_states / result.reference.effective_states, .7)
        self.assertLess(result.generated.used_states / result.reference.used_states, .7)
        self.assertEqual(result.generated.change_count, 0)
        self.assertIsNone(result.categorical_changed_transition_joint_tv)
        self.assertGreater(result.categorical_time_occupancy_tv, 0.)
        self.assertGreater(result.categorical_cue_transition_joint_tv, 0.)

    def test_independent_held_same_marginal_wrong_persistence(self):
        # A fixed synthetic independent cue-held draw, not a retimed cue stream.
        reference = np.repeat([0, 1], 4)
        independent_held = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        result = palette_metrics(independent_held, reference, np.arange(8.),
                                 duration_s=8., vocabulary_size=2)
        self.assertEqual(result.categorical_time_occupancy_tv, 0.)
        self.assertEqual(result.generated.effective_states, result.reference.effective_states)
        self.assertAlmostEqual(result.categorical_cue_transition_joint_tv, 6/7)
        self.assertGreater(result.generated.change_count, result.reference.change_count)
        self.assertLess(result.generated.keep_fraction, result.reference.keep_fraction)
        # Color-event success is not inferred from palette occupancy/diversity.
        event_result = events([4.], np.arange(1., 8.), [.9]*7, duration_s=8.)
        self.assertAlmostEqual(event_result.average_precision, 1/7)

    def test_distributions_are_not_exact_phase_accuracy(self):
        reference = [0, 0, 1, 1, 0]
        reordered = [0, 1, 1, 0, 0]
        result = palette_metrics(reordered, reference, np.arange(5.), duration_s=5., vocabulary_size=2)
        self.assertEqual(result.categorical_time_occupancy_tv, 0.)
        self.assertEqual(result.categorical_cue_transition_joint_tv, 0.)
        self.assertEqual(result.categorical_changed_transition_joint_tv, 0.)
        self.assertEqual(events([2., 4.], [1., 3.], [.9, .9]).average_precision, 0.)

    def test_single_cue_empty_transition_not_fake_zero_error(self):
        result = palette_metrics([0], [1], [0.], duration_s=6., vocabulary_size=2)
        self.assertEqual(result.categorical_time_occupancy_tv, 1.)
        self.assertIsNone(result.categorical_cue_transition_joint_tv)
        self.assertIsNone(result.categorical_changed_transition_joint_tv)
        self.assertIsNone(result.generated.keep_fraction)
        np.testing.assert_array_equal(result.generated.observed_residence_s, [6.])
        constant_self = palette_metrics([0], [0], [0.], duration_s=1., vocabulary_size=1)
        self.assertEqual(constant_self.categorical_time_occupancy_tv, 0.)
        self.assertEqual(constant_self.generated.entropy_normalized, 0.)
        self.assertIsNone(constant_self.categorical_changed_transition_joint_tv)

    def test_category_renaming_invariance_and_bounded_errors(self):
        rng = np.random.default_rng(19)
        for _ in range(20):
            a, b = rng.integers(0, 4, size=(2, 20))
            permutation = rng.permutation(4)
            before = palette_metrics(a, b, np.arange(20.), duration_s=20., vocabulary_size=4)
            after = palette_metrics(permutation[a], permutation[b], np.arange(20.),
                                    duration_s=20., vocabulary_size=4)
            for name in ("categorical_time_occupancy_tv", "categorical_cue_transition_joint_tv",
                         "categorical_changed_transition_joint_tv"):
                value = getattr(before, name)
                self.assertGreaterEqual(value, 0.)
                self.assertLessEqual(value, 1.+1e-15)
                self.assertAlmostEqual(value, getattr(after, name))

    def test_invalid_palette_and_cue_inputs_raise_not_retune(self):
        invalid = [([], [], [], 3., 2), ([0., 1.], [0, 1], [0, 1], 3., 2),
                   ([0, 2], [0, 1], [0, 1], 3., 2), ([0, -1], [0, 1], [0, 1], 3., 2),
                   ([0], [0, 1], [0, 1], 3., 2), ([0, 1], [0, 1], [1, 2], 3., 2),
                   ([0, 1], [0, 1], [0, 0], 3., 2), ([0, 1], [0, 1], [0, 3], 3., 2),
                   ([0, 1], [0, 1], [0, np.nan], 3., 2), ([0], [0], [0], 0., 2),
                   ([0], [0], [0], 1., 0), ([0], [0], [0], 1., True)]
        for generated, reference, times, duration, vocabulary in invalid:
            with self.subTest(generated=generated, times=times), self.assertRaises(ValueError):
                palette_metrics(generated, reference, times, duration_s=duration, vocabulary_size=vocabulary)


if __name__ == "__main__":
    unittest.main()

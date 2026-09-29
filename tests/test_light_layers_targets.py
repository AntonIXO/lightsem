"""Synthetic-only tests for new CPU visual targets; no dataset or torch access."""
import colorsys
from dataclasses import replace
import unittest

import numpy as np

from lightsem.experiments.light_layers.targets import (
    HS_DIM, HUE_BINS, MIN_SATURATION, MIN_VALUE, QUADRANT_NAMES, RICH_DIM,
    SATURATION_BINS, chromatic_event_targets, chromatic_novelty,
    extract_color_targets, jensen_shannon_distance, retained_mask_activity,
    rich_feature_names,
)


def frame(color, shape=(8, 8)):
    return np.broadcast_to(np.asarray(color, dtype=float), (*shape, 3)).copy()


def target(color, shape=(8, 8)):
    return extract_color_targets(frame(color, shape), np.ones(shape, dtype=bool))


def novelty_from_targets(times, frames, **kwargs):
    return chromatic_novelty(
        times, np.stack([item.hs_histogram for item in frames]),
        np.array([item.color_valid for item in frames]),
        retained_fraction=np.array([item.retained_fraction for item in frames]),
        chromatic_fraction=np.array([item.chromatic_fraction for item in frames]),
        **kwargs)


class FrameTargetTests(unittest.TestCase):
    def test_exact_schema_and_names(self):
        self.assertEqual((HUE_BINS, SATURATION_BINS, HS_DIM, RICH_DIM), (12, 6, 72, 96))
        names = rich_feature_names()
        self.assertEqual(len(names), RICH_DIM)
        self.assertEqual(len(set(names)), RICH_DIM)
        self.assertEqual(names[:2], ("hs_h00_s00", "hs_h00_s01"))
        self.assertEqual(names[71], "hs_h11_s05")
        self.assertEqual(names[72], "top_left_hue_cos")
        self.assertEqual(names[-1], "bottom_right_retained_mask_fraction")
        result = target([1, 0, 0])
        self.assertEqual(result.rich.shape, (96,))
        self.assertEqual(result.rich.dtype, np.float32)
        np.testing.assert_array_equal(result.rich[:72], result.hs_histogram)
        np.testing.assert_array_equal(result.rich[72:], result.quadrants.ravel())

    def test_rgb_order_not_opencv_bgr(self):
        red = target([1, 0, 0])
        blue = target([0, 0, 1])
        self.assertEqual(np.argmax(red.hs_histogram), 0 * 6 + 5)
        self.assertEqual(np.argmax(blue.hs_histogram), 8 * 6 + 5)
        self.assertEqual(red.hs_histogram.sum(), 1.)
        self.assertEqual(blue.hs_histogram.sum(), 1.)
        np.testing.assert_allclose(red.quadrants[:, 0], 1.)
        np.testing.assert_allclose(blue.quadrants[:, :2],
                                   np.tile([-.5, -np.sqrt(3)/2], (4, 1)), atol=1e-7)

    def test_uint8_matches_float_without_implicit_scaling(self):
        pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        keep = np.ones((8, 8), dtype=bool)
        integer = extract_color_targets(pixels, keep)
        floating = extract_color_targets(pixels.astype(float)/255, keep)
        np.testing.assert_array_equal(integer.rich, floating.rich)
        self.assertEqual(integer.value, floating.value)
        with self.assertRaises(ValueError):
            extract_color_targets(pixels.astype(float), keep)

    def test_circular_hue_wrap_is_continuous(self):
        below = target(colorsys.hsv_to_rgb(1-1e-6, .9, .8))
        above = target(colorsys.hsv_to_rgb(1e-6, .9, .8))
        self.assertLess(jensen_shannon_distance(below.hs_histogram, above.hs_histogram), .005)
        self.assertGreater(below.quadrants[0, 0], .999)
        self.assertLess(below.quadrants[0, 1], 0.)
        self.assertGreater(above.quadrants[0, 1], 0.)
        np.testing.assert_allclose(target(colorsys.hsv_to_rgb(0, .9, .8)).rich,
                                   target(colorsys.hsv_to_rgb(1, .9, .8)).rich, atol=1e-7)

    def test_histogram_is_joint_not_two_marginals(self):
        a = np.empty((8, 8, 3))
        b = np.empty_like(a)
        a[:, :4] = colorsys.hsv_to_rgb(0, .4, 1.)
        a[:, 4:] = colorsys.hsv_to_rgb(.5, 1., 1.)
        b[:, :4] = colorsys.hsv_to_rgb(0, 1., 1.)
        b[:, 4:] = colorsys.hsv_to_rgb(.5, .4, 1.)
        keep = np.ones((8, 8), dtype=bool)
        ha = extract_color_targets(a, keep).hs_histogram.reshape(12, 6)
        hb = extract_color_targets(b, keep).hs_histogram.reshape(12, 6)
        np.testing.assert_allclose(ha.sum(0), hb.sum(0))
        np.testing.assert_allclose(ha.sum(1), hb.sum(1))
        self.assertGreater(jensen_shannon_distance(ha.ravel(), hb.ravel()), .9)

    def test_dark_and_achromatic_are_not_red(self):
        for color in ([0, 0, 0], [.7, .7, .7], [.07, 0, 0], [.9, .85, .85]):
            with self.subTest(color=color):
                result = target(color)
                self.assertFalse(result.color_valid)
                self.assertFalse(result.quadrant_color_valid.any())
                np.testing.assert_array_equal(result.hs_histogram, np.zeros(72))
                np.testing.assert_array_equal(result.quadrants[:, :5], np.zeros((4, 5)))
                np.testing.assert_array_equal(result.quadrants[:, 5], np.ones(4))
                self.assertTrue(np.isfinite(result.rich).all())
                self.assertTrue(result.value.valid)
                self.assertAlmostEqual(result.value.mean, max(color))
        self.assertTrue(target([MIN_VALUE, 0, 0]).color_valid)
        self.assertTrue(target(colorsys.hsv_to_rgb(.3, MIN_SATURATION + 1e-5, .5)).color_valid)
        self.assertFalse(target(colorsys.hsv_to_rgb(.3, MIN_SATURATION - 1e-5, .5)).color_valid)

    def test_empty_mask_explicit_finite_zero(self):
        result = extract_color_targets(frame([1, 0, 0]), np.zeros((8, 8), dtype=bool))
        self.assertFalse(result.color_valid)
        self.assertFalse(result.value.valid)
        self.assertEqual((result.valid_pixels, result.retained_pixels), (0, 0))
        self.assertEqual((result.valid_fraction, result.retained_fraction, result.chromatic_fraction), (0, 0, 0))
        np.testing.assert_array_equal(result.rich, np.zeros(96))
        self.assertEqual((result.value.mean, result.value.std, result.value.p90, result.value.maximum), (0, 0, 0, 0))

    def test_normalized_over_valid_pixels_and_independent_value(self):
        pixels = frame([0, 0, 0], (4, 4))
        pixels[0] = [1., 0, 0]
        pixels[1] = [.5, .5, .5]
        keep = np.ones((4, 4), dtype=bool)
        keep[-1] = False
        result = extract_color_targets(pixels, keep)
        self.assertEqual((result.retained_pixels, result.valid_pixels), (12, 4))
        self.assertEqual(result.retained_fraction, .75)
        self.assertEqual(result.valid_fraction, .25)
        self.assertAlmostEqual(result.chromatic_fraction, 1/3)
        self.assertEqual(result.hs_histogram[5], 1.)
        self.assertAlmostEqual(result.value.mean, .5)
        self.assertEqual(result.value.p90, 1.)
        self.assertEqual(result.value.maximum, 1.)

    def test_quadrant_order_and_odd_image_coverage(self):
        pixels = frame([1, 1, 0], (5, 7))
        pixels[:2, :3] = [1, 0, 0]
        pixels[:2, 3:] = [0, 1, 0]
        pixels[2:, :3] = [0, 0, 1]
        result = extract_color_targets(pixels, np.ones((5, 7), dtype=bool))
        self.assertEqual(QUADRANT_NAMES, ("top_left", "top_right", "bottom_left", "bottom_right"))
        for row, color in zip(result.quadrants, ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0])):
            np.testing.assert_allclose(row, target(color).quadrants[0], atol=1e-7)
        np.testing.assert_allclose(result.hs_histogram[[5, 4*6+5, 8*6+5, 2*6+5]],
                                   np.array([6, 8, 9, 12])/35, atol=1e-7)

    def test_empty_quadrants_small_images_do_not_leak_nans(self):
        result = target([1, 0, 0], (1, 1))
        np.testing.assert_array_equal(result.quadrant_color_valid, [False, False, False, True])
        np.testing.assert_array_equal(result.quadrants[:3], np.zeros((3, 6)))
        self.assertTrue(np.isfinite(result.rich).all())

    def test_hue_concentration_is_resultant_not_legacy_entropy(self):
        pixels = frame([1, 0, 0])
        pixels[:, ::2] = [0, 1, 1]
        result = extract_color_targets(pixels, np.ones((8, 8), dtype=bool))
        np.testing.assert_allclose(result.quadrants[:, :2], 0., atol=1e-7)
        np.testing.assert_allclose(result.quadrants[:, 3], 0., atol=1e-7)
        self.assertTrue(result.quadrant_color_valid.all())

    def test_uniform_value_scaling_changes_value_not_color(self):
        rng = np.random.default_rng(14)
        pixels = rng.uniform(.2, .95, (8, 8, 3))
        mask = np.ones((8, 8), dtype=bool)
        bright = extract_color_targets(pixels, mask)
        dark = extract_color_targets(pixels * .5, mask)
        np.testing.assert_allclose(bright.rich, dark.rich, atol=1e-7)
        self.assertAlmostEqual(dark.value.mean, bright.value.mean * .5)
        self.assertAlmostEqual(dark.value.p90, bright.value.p90 * .5)
        self.assertAlmostEqual(dark.value.std, bright.value.std * .5)

    def test_legacy_is_optional_copied_auxiliary_only(self):
        self.assertIsNone(target([1, 0, 0]).legacy_v6)
        reference = np.array([.1, -.2, .3, 10., -9., .7])
        result = extract_color_targets(frame([1, 0, 0]), np.ones((8, 8), bool), legacy_v6=reference)
        np.testing.assert_array_equal(result.legacy_v6, reference)
        reference[:] = 0
        self.assertEqual(result.legacy_v6[3], 10.)
        np.testing.assert_array_equal(result.rich, target([1, 0, 0]).rich)
        for bad in ([1, 2], [0, 0, 0, 0, 0, np.nan]):
            with self.assertRaises(ValueError):
                extract_color_targets(frame([1, 0, 0]), np.ones((8, 8), bool), legacy_v6=bad)

    def test_invalid_frame_and_mask_inputs_rejected(self):
        for pixels in (np.zeros((8, 8)), np.zeros((0, 8, 3)), np.zeros((8, 8, 4)),
                       np.zeros((8, 8, 3), dtype=np.int16), frame([np.nan, 0, 0]),
                       frame([np.inf, 0, 0]), frame([1.01, 0, 0]), frame([-.1, 0, 0])):
            with self.subTest(shape=pixels.shape, dtype=pixels.dtype):
                with self.assertRaises(ValueError):
                    extract_color_targets(pixels, np.ones((8, 8), bool))
        for mask in (np.ones((8, 8), np.uint8), np.ones((8, 7), bool)):
            with self.assertRaises(ValueError):
                extract_color_targets(frame([1, 0, 0]), mask)


class ChromaticSequenceTests(unittest.TestCase):
    def setUp(self):
        self.times = 7.037 + np.arange(41)/10
        self.red = target([1, 0, 0])
        self.green = target([0, 1, 0])
        self.blue = target([0, 0, 1])

    def test_js_distance_normalized_and_symmetric(self):
        red, blue = self.red.hs_histogram, self.blue.hs_histogram
        self.assertEqual(jensen_shannon_distance(red, red), 0.)
        self.assertEqual(jensen_shannon_distance(red, blue), 1.)
        mix = .25*red + .75*blue
        self.assertEqual(jensen_shannon_distance(red, mix), jensen_shannon_distance(mix, red))
        for bad in (np.zeros(72), red * 2, np.full(72, np.nan), -red):
            with self.assertRaises(ValueError):
                jensen_shannon_distance(red, bad)

    def test_monochromatic_hue_flip_peaks_at_actual_input_time(self):
        novelty = novelty_from_targets(self.times, [self.red]*20 + [self.blue]*21)
        events = chromatic_event_targets(novelty, threshold=.5)
        self.assertAlmostEqual(novelty.scores[20], 1.)
        np.testing.assert_array_equal(events.event_indices, [20])
        np.testing.assert_array_equal(events.event_times_s, self.times[[20]])
        np.testing.assert_array_equal(events.times_s, self.times)
        self.assertFalse(novelty.mask_activity_known)
        self.assertFalse(novelty.window_eligible[:3].any())
        self.assertFalse(novelty.window_eligible[-3:].any())

    def test_fixed_luma_hue_and_saturation_changes_trigger(self):
        # Rec.709 luma equality, despite differing hue and V.
        red_rgb = np.array([.9, 0, 0])
        green_rgb = np.array([0, .9*.2126/.7152, 0])
        self.assertAlmostEqual(red_rgb @ [.2126, .7152, .0722], green_rgb @ [.2126, .7152, .0722])
        hue = novelty_from_targets(self.times, [target(red_rgb)]*20 + [target(green_rgb)]*21)
        self.assertTrue(chromatic_event_targets(hue, threshold=.5).events[20])
        # Same hue, lower saturation, and the same luma.
        saturated = np.array(colorsys.hsv_to_rgb(.3, 1., .8))
        muted = np.array(colorsys.hsv_to_rgb(.3, .3, 1.))
        muted *= (saturated @ [.2126, .7152, .0722]) / (muted @ [.2126, .7152, .0722])
        self.assertAlmostEqual(saturated @ [.2126, .7152, .0722], muted @ [.2126, .7152, .0722])
        saturation = novelty_from_targets(self.times, [target(saturated)]*20 + [target(muted)]*21)
        self.assertTrue(chromatic_event_targets(saturation, threshold=.5).events[20])

    def test_brightness_only_negative_and_zero_threshold_negative(self):
        stable = novelty_from_targets(self.times, [self.red]*41)
        self.assertFalse(chromatic_event_targets(stable, threshold=0.).events.any())
        brightness = [target(np.array([.8, .25, .1]) * (.25 if i % 4 else 1.)) for i in range(41)]
        novelty = novelty_from_targets(self.times, brightness)
        self.assertLess(novelty.scores.max(), 1e-6)
        self.assertFalse(chromatic_event_targets(novelty, threshold=.1).events.any())

    def test_invalid_color_boundaries_never_become_hue_events(self):
        for invalid in (target([0, 0, 0]), target([1, 1, 1]), target([.02, 0, 0])):
            for frames in ([invalid]*20 + [self.blue]*21, [self.red]*20 + [invalid]*21):
                novelty = novelty_from_targets(self.times, frames)
                self.assertFalse(novelty.eligible[20])
                self.assertFalse(novelty.color_eligible[20])
                self.assertFalse(chromatic_event_targets(novelty, threshold=.1).events.any())
                self.assertTrue(np.isfinite(novelty.scores).all())

    def test_empty_masks_all_invalid_no_events(self):
        empty = extract_color_targets(frame([1, 0, 0]), np.zeros((8, 8), bool))
        novelty = novelty_from_targets(self.times, [empty]*41)
        self.assertFalse(novelty.eligible.any())
        self.assertFalse(chromatic_event_targets(novelty, threshold=0.).events.any())
        np.testing.assert_array_equal(novelty.scores, np.zeros(41))

    def test_cuts_exclude_crossing_windows_without_suppressing_later_change(self):
        cuts = np.zeros(41, bool)
        cuts[15] = True
        novelty = novelty_from_targets(self.times, [self.red]*15 + [self.blue]*15 + [self.green]*11, cuts=cuts)
        self.assertFalse(novelty.cut_eligible[15])
        self.assertFalse(novelty.eligible[15])
        self.assertTrue(novelty.cut_eligible[18])  # cut exactly at left edge is outside support
        np.testing.assert_array_equal(chromatic_event_targets(novelty, threshold=.5).event_indices, [30])

    def test_mask_coverage_gate_is_explicit_and_configurable(self):
        keep = np.zeros((8, 8), bool)
        keep[0] = True
        red = extract_color_targets(frame([1, 0, 0]), keep)
        blue = extract_color_targets(frame([0, 0, 1]), keep)
        permissive = novelty_from_targets(self.times, [red]*20 + [blue]*21)
        strict = novelty_from_targets(self.times, [red]*20 + [blue]*21, min_retained_fraction=.2)
        self.assertTrue(permissive.eligible[20])
        self.assertFalse(strict.mask_eligible[20])
        self.assertAlmostEqual(strict.window_min_retained_fraction[20], .125)
        self.assertFalse(chromatic_event_targets(strict, threshold=.5).events.any())

    def test_moving_mask_frozen_background_exposes_nuisance_at_equal_coverage(self):
        background = frame([1, 0, 0])
        background[:, 4:] = [0, 0, 1]
        masks = np.zeros((41, 8, 8), bool)
        masks[:20, :, :4] = True
        masks[20:, :, 4:] = True
        frames = [extract_color_targets(background, keep) for keep in masks]
        activity = retained_mask_activity(masks)
        novelty = novelty_from_targets(self.times, frames, mask_activity=activity)
        # This is a deliberately demonstrated failure of the raw color score,
        # not a control silently marked as passing by removing its event.
        self.assertTrue(chromatic_event_targets(novelty, threshold=.5).events[20])
        self.assertTrue(novelty.mask_activity_known)
        self.assertEqual(novelty.window_max_mask_activity[20], 1.)
        self.assertEqual(novelty.window_retained_fraction_range[20], 0.)
        np.testing.assert_array_equal(novelty.retained_fraction, np.full(41, .5))
        self.assertEqual(activity[0], 0.)
        stationary_masks = np.broadcast_to(masks[0], masks.shape)
        np.testing.assert_array_equal(retained_mask_activity(stationary_masks), np.zeros(41))

    def test_irregular_times_use_duration_weights_not_sample_counts(self):
        times = np.array([0., .05, .17, .30, .42, .58, .61, .76, .90, 1.03, 1.15])
        frames = [self.red]*4 + [self.blue, self.red] + [self.green]*5
        novelty = novelty_from_targets(times, frames)
        expected_pre = self.red.hs_histogram * (14/30) + self.blue.hs_histogram * (16/30)
        self.assertTrue(novelty.eligible[6])
        np.testing.assert_allclose(novelty.pre_histograms[6], expected_pre, atol=1e-7)
        np.testing.assert_allclose(novelty.post_histograms[6], self.green.hs_histogram, atol=1e-7)
        np.testing.assert_array_equal(novelty.times_s, times)
        events = chromatic_event_targets(novelty, threshold=.8)
        self.assertTrue(all(t in times for t in events.event_times_s))

    def test_decimated_40_to_10_hz_keeps_observation_times_not_native_precision(self):
        source_times = 2.037 + np.arange(161)/40
        source_targets = [self.red]*81 + [self.blue]*80
        # The real transition is at index 81. A 10 Hz observation first sees
        # it at index 84; targets must not invent its missing native timestamp.
        source_event = source_times[81]
        observed_times = source_times[::4]
        observed_targets = source_targets[::4]
        novelty = novelty_from_targets(observed_times, observed_targets)
        events = chromatic_event_targets(novelty, threshold=.5)
        np.testing.assert_array_equal(novelty.times_s, observed_times)
        np.testing.assert_array_equal(events.event_times_s, [source_times[84]])
        self.assertAlmostEqual(events.event_times_s[0] - source_event, .075)

    def test_large_timestamp_gap_is_not_interpolated_into_evidence(self):
        times = np.r_[np.arange(10)/10, 1.5 + np.arange(15)/10]
        frames = [self.red]*10 + [self.blue]*15
        novelty = novelty_from_targets(times, frames)
        self.assertFalse(novelty.gap_eligible[10])
        self.assertFalse(novelty.eligible[10])
        self.assertFalse(chromatic_event_targets(novelty, threshold=.5).events.any())

    def test_timestamps_reject_duplicate_decreasing_nan_and_bad_rank(self):
        for times in ([0., 0., .1], [0., .2, .1], [0., np.nan, .2], [0., np.inf, .2], [[0., .1, .2]]):
            with self.assertRaises(ValueError):
                novelty_from_targets(times, [self.red]*3)

    def test_single_or_empty_sequence_is_explicit_and_finite(self):
        one = novelty_from_targets([.123], [self.red])
        self.assertFalse(one.eligible.any())
        empty = chromatic_novelty(np.array([]), np.empty((0, 72)), np.empty(0, bool),
                                  retained_fraction=np.empty(0), chromatic_fraction=np.empty(0))
        self.assertEqual(empty.scores.shape, (0,))
        self.assertEqual(chromatic_event_targets(empty, threshold=.1).event_indices.size, 0)

    def test_event_threshold_required_fixed_and_strict(self):
        novelty = novelty_from_targets(self.times, [self.red]*20 + [self.blue]*21)
        with self.assertRaises(TypeError):
            chromatic_event_targets(novelty)
        for threshold in (np.nan, np.inf, -.01, 1.01):
            with self.assertRaises(ValueError):
                chromatic_event_targets(novelty, threshold=threshold)
        low = chromatic_event_targets(novelty, threshold=.5)
        high = chromatic_event_targets(novelty, threshold=1.)
        self.assertEqual(low.threshold, .5)
        np.testing.assert_array_equal(low.scores, high.scores)
        self.assertEqual(low.event_indices.size, 1)
        self.assertEqual(high.event_indices.size, 0)

    def test_event_distance_uses_seconds_and_plateaus_emit_once(self):
        novelty = novelty_from_targets(self.times, [self.red]*41)
        scores = np.zeros(41)
        scores[7:9] = .7
        scores[11] = .8  # stronger and less than .5 s from first candidate
        scores[16] = .9  # exactly .5 s from previous candidate, allowed
        scores[25:31] = .6  # one event, not repeated events along plateau
        edited = replace(novelty, scores=scores, eligible=np.ones(41, bool))
        events = chromatic_event_targets(edited, threshold=.5)
        np.testing.assert_array_equal(events.event_indices, [11, 16, 27])
        self.assertTrue(np.all(np.diff(events.event_times_s) >= .5 - 1e-9))
        np.testing.assert_array_equal(events.event_scores, scores[events.event_indices])
        np.testing.assert_array_equal(events.events.nonzero()[0], events.event_indices)

    def test_histogram_and_coverage_contract_rejects_invalid_inputs(self):
        hist = np.tile(self.red.hs_histogram, (3, 1))
        valid = np.ones(3, bool)
        base = dict(times_s=np.arange(3)/10, hs_histograms=hist, color_valid=valid,
                    retained_fraction=np.ones(3), chromatic_fraction=np.ones(3))
        for change in (
            {"hs_histograms": hist * 2}, {"hs_histograms": np.full((3, 72), np.nan)},
            {"hs_histograms": np.zeros((3, 71))}, {"hs_histograms": -hist},
            {"color_valid": [1, 1, 1]}, {"color_valid": np.zeros(3, bool)},
            {"retained_fraction": [1, 0, 1]}, {"retained_fraction": [1, 1.1, 1]},
            {"chromatic_fraction": [1, np.nan, 1]}, {"cuts": [0, 0, 0]},
            {"mask_activity": [0, 2, 0]}, {"window_s": 0}, {"max_gap_s": np.inf},
            {"min_retained_fraction": -.1}, {"min_chromatic_fraction": 1.1},
        ):
            with self.subTest(change=list(change)):
                with self.assertRaises(ValueError):
                    chromatic_novelty(**(base | change))
        for bad in (np.zeros((2, 8, 8), np.uint8), np.zeros((8, 8), bool), np.zeros((2, 0, 8), bool)):
            with self.assertRaises(ValueError):
                retained_mask_activity(bad)


if __name__ == "__main__":
    unittest.main()

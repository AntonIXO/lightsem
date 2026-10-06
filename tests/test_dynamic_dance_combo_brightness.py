import unittest

import numpy as np

from lightsem.conductor.body_transfer_v2 import make_outputs, map_body
from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.conductor.body_transfer import compile_intent, TransferConfig
from lightsem.experiments.dynamic_dance_combo.brightness import (
    apply_brightness, historical_envelope, historical_six_head_dimmers,
)


def curve(seconds=8., quiet_until=3.):
    times = np.arange(1, round(seconds*10)+1)/10
    levels = np.where(times < quiet_until, -60., -15.)
    return {"time": times, "momentary": levels.copy(), "short_term": levels.copy()}


class HistoricalBrightnessTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(320)/40
        self.body = np.repeat(REFERENCE[None], len(self.times), axis=0)

    def test_quiet_intro_is_quiet_even_with_fast_body(self):
        self.body[:, 20, 0] += np.arange(len(self.times))*.1
        outputs = make_outputs()
        intent = map_body(self.body, outputs, "01")
        result, trace = apply_brightness(self.body, self.times, outputs, intent, curve())
        moving = [i for i, output in enumerate(outputs) if output["kind"] != "bulb"]
        self.assertLessEqual(result["dimmer"][:100, moving].max(), .12)
        self.assertTrue((trace["activity"][:100] == 0).all())
        self.assertGreater(result["dimmer"][-40:, moving].mean(), .7)

    def test_only_dimmer_changes_and_input_not_mutated(self):
        outputs = make_outputs()
        intent = map_body(self.body, outputs, "01")
        old = {key: value.copy() for key, value in intent.items() if isinstance(value, np.ndarray)}
        result, _ = apply_brightness(self.body, self.times, outputs, intent, curve())
        for key, value in old.items():
            np.testing.assert_array_equal(intent[key], value)
            if key != "dimmer":
                np.testing.assert_array_equal(result[key], value)
        self.assertFalse(np.array_equal(result["dimmer"], old["dimmer"]))

    def test_whole_track_anchor_is_used_before_cropping(self):
        full = historical_envelope(curve(), self.times)
        prefix = historical_envelope(curve(), self.times[:120])
        np.testing.assert_array_equal(prefix["level"], full["level"][:120])
        self.assertEqual(prefix["evidence"]["anchor_lufs"], -15.)
        with self.assertRaisesRegex(ValueError, "zero-based"):
            historical_envelope(curve(), self.times[40:])

    def test_silent_input_is_constant_and_never_an_onset(self):
        silent = curve()
        silent["momentary"][:] = -120.
        silent["short_term"][:] = -120.
        result = historical_envelope(silent, self.times)
        np.testing.assert_array_equal(result["level"], np.full(len(self.times), .12))
        self.assertEqual(result["evidence"]["anchor_lufs"], -23.)

    def test_constant_gain_does_not_change_relative_loudness_control(self):
        original = curve()
        changed = {key: value.copy() for key, value in original.items()}
        changed["momentary"] += 6.
        changed["short_term"] += 6.
        np.testing.assert_array_equal(historical_envelope(original, self.times)["level"],
                                      historical_envelope(changed, self.times)["level"])

    def test_extra_outputs_preserve_each_existing_brightness_stream(self):
        self.body[:, 20, 0] += np.linspace(0, .7, len(self.times))
        small = make_outputs()
        large = make_outputs(beams=23, lasers=11, bulbs=83)
        first, _ = apply_brightness(self.body, self.times, small,
                                    map_body(self.body, small, "01"), curve())
        second, _ = apply_brightness(self.body, self.times, large,
                                     map_body(self.body, large, "01"), curve())
        indices = {fixture["name"]: i for i, fixture in enumerate(large)}
        for index, fixture in enumerate(small):
            np.testing.assert_array_equal(first["dimmer"][:, index],
                                          second["dimmer"][:, indices[fixture["name"]]])

    def test_bulb_spatial_ratios_preserved(self):
        outputs = make_outputs(beams=0, lasers=0, bulbs=17)
        intent = map_body(self.body, outputs, "01")
        result, trace = apply_brightness(self.body, self.times, outputs, intent, curve())
        np.testing.assert_allclose(result["dimmer"], intent["dimmer"]*trace["level"][:, None]/.9,
                                   rtol=2e-16, atol=1e-16)

    def test_static_body_and_constant_audio_do_not_animate(self):
        constant = curve(quiet_until=0)
        outputs = make_outputs()
        result, _ = apply_brightness(self.body, self.times, outputs,
                                     map_body(self.body, outputs, "01"), constant)
        np.testing.assert_array_equal(result["dimmer"], np.repeat(result["dimmer"][:1], len(self.times), axis=0))

    def test_audio_brightness_does_not_change_emitted_motion_or_rgb(self):
        outputs = make_outputs(beams=2, lasers=1, bulbs=2)
        intent = map_body(self.body, outputs, "01")
        changed, _ = apply_brightness(self.body, self.times, outputs, intent, curve())
        left = np.repeat(np.array([[40, 90, 250]], np.uint8), len(self.times), axis=0)
        right = left[:, ::-1].copy()
        before, patch, _ = compile_intent(intent, outputs, left, right, TransferConfig())
        after, other_patch, _ = compile_intent(changed, outputs, left, right, TransferConfig())
        self.assertEqual(patch, other_patch)
        allowed = np.zeros(before.shape[1:], bool)
        for fixture in patch:
            offset = 0 if fixture["kind"] == "bulb" else 4
            allowed[fixture["universe"], fixture["address"]-1+offset] = True
        np.testing.assert_array_equal(before[:, ~allowed], after[:, ~allowed])
        self.assertTrue(np.any(before[:, allowed] != after[:, allowed]))
        # A frozen body may now brighten with audio; angles remain constant.
        for fixture in patch:
            if fixture["kind"] != "bulb":
                angle_bytes = after[:, fixture["universe"], fixture["address"]-1:fixture["address"]+3]
                np.testing.assert_array_equal(angle_bytes, np.repeat(angle_bytes[:1], len(self.times), axis=0))

    def test_old_six_head_quiet_formula(self):
        result = historical_six_head_dimmers(self.body, self.times, curve())
        root = self.body[0, 0]
        regions = [self.body[0, 20]-root,
                   .65*(self.body[0, 10]-root)+.35*(self.body[0, 4]-root),
                   .65*(self.body[0, 12]-root)+.35*(self.body[0, 15]-root)+[-.12, 0, 0],
                   .65*(self.body[0, 12]-root)+.35*(self.body[0, 15]-root)+[.12, 0, 0],
                   .65*(self.body[0, 11]-root)+.35*(self.body[0, 5]-root),
                   self.body[0, 21]-root]
        expected = [.12*(.94+.06*side*np.tanh(region[0]))
                    for side, region in zip([-1, -1, -1, 1, 1, 1], regions)]
        np.testing.assert_array_equal(result[0], expected)

    def test_invalid_or_incomplete_audio_and_geometry_rejected(self):
        for bad in (curve(seconds=2), {**curve(), "short_term": np.full(80, np.nan)}):
            with self.assertRaises(ValueError):
                historical_envelope(bad, self.times)
        with self.assertRaises(ValueError):
            historical_six_head_dimmers(self.body[:-1], self.times, curve())
        with self.assertRaises(ValueError):
            historical_envelope(curve(), self.times+1e-3)


if __name__ == "__main__":
    unittest.main()

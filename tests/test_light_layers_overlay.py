"""Synthetic byte-contract tests only; no datasets, models, media, or rendering."""
from dataclasses import asdict
import unittest

import numpy as np

from lightsem.dmx.decode import decode_stream
from lightsem.dmx.fixtures import Fixture, Rig, club_rig
from lightsem.model.light_layers.overlay import (
    AccentConfig, AccentEvent, OverlayConfig, allowed_slot_mask,
    apply_accent_events, apply_overlay, sample_accent_events,
)


class OverlayContracts(unittest.TestCase):
    def setUp(self):
        self.rig = club_rig()
        self.times = np.arange(121, dtype=float) / 40
        self.baseline = np.random.default_rng(7001).integers(
            0, 256, (len(self.times), 512), dtype=np.uint8)
        self.original = self.baseline.copy()
        self.patch_before = asdict(self.rig)
        self.pulse_config = AccentConfig(
            max_strength=.75, min_width_s=.1, max_width_s=.8, refractory_s=.3)

    def tearDown(self):
        np.testing.assert_array_equal(self.baseline, self.original)
        self.assertEqual(asdict(self.rig), self.patch_before)

    def slot(self, name, offsets):
        fixture = next(f for f in self.rig.fixtures if f.name == name)
        return fixture.address - 1 + np.asarray(offsets)

    def assert_only(self, out, slots):
        mask = np.zeros(512, dtype=bool)
        mask[np.asarray(slots, dtype=int).reshape(-1)] = True
        np.testing.assert_array_equal(out[:, ~mask], self.baseline[:, ~mask])
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape, self.baseline.shape)
        self.assertFalse(np.shares_memory(out, self.baseline))

    def test_default_and_empty_maps_are_byte_exact_independent_copies(self):
        for kwargs in ({}, {"rgb": {}, "dimmer_gain": {}, "dimmer_envelope": {}},
                       {"config": OverlayConfig(0, 0)}):
            with self.subTest(kwargs=kwargs):
                out = apply_overlay(self.baseline, self.rig, self.times, **kwargs)
                self.assert_only(out, [])
                out[0, 0] ^= 255
                np.testing.assert_array_equal(self.baseline, self.original)

    def test_allowed_mask_independently_matches_personalities(self):
        expected = np.zeros(512, dtype=bool)
        for fixture in self.rig.fixtures:
            if fixture.kind == "moving_head":
                offsets = [4, 6, 7, 8]
            elif fixture.kind == "wash":
                offsets = [0, 2, 3, 4]
            elif fixture.kind == "projector":
                offsets = [2, 3, 4]
            else:
                offsets = []
            for offset in offsets:
                expected[fixture.address - 1 + offset] = True
        np.testing.assert_array_equal(allowed_slot_mask(self.rig), expected)

    def test_rgb_only_exact_channels_no_dimmer_compensation(self):
        track = np.tile([.0, .5, 1.], (len(self.times), 1))
        rgb = {"mh1": track, "wash1": track, "proj1": track}
        out = apply_overlay(self.baseline, self.rig, self.times, rgb=rgb)
        slots = np.concatenate([self.slot("mh1", [6, 7, 8]),
                                self.slot("wash1", [2, 3, 4]),
                                self.slot("proj1", [2, 3, 4])])
        self.assert_only(out, slots)
        for name, offsets in (("mh1", [6, 7, 8]), ("wash1", [2, 3, 4]),
                              ("proj1", [2, 3, 4])):
            np.testing.assert_array_equal(out[:, self.slot(name, offsets)],
                                          np.tile([0, 128, 255], (len(self.times), 1)))
        np.testing.assert_array_equal(track, np.tile([0, .5, 1], (len(self.times), 1)))

    def test_achromatic_black_white_and_gray_are_valid(self):
        for value, byte in ((0., 0), (1., 255), (.5, 128)):
            with self.subTest(value=value):
                out = apply_overlay(self.baseline, self.rig, self.times,
                                    rgb={"mh1": np.full((len(self.times), 3), value)})
                self.assert_only(out, self.slot("mh1", [6, 7, 8]))
                self.assertTrue(np.all(out[:, self.slot("mh1", [6, 7, 8])] == byte))

    def test_dimmer_gain_only_and_identity(self):
        gain = np.linspace(0, 3, len(self.times))
        slot = int(self.slot("mh1", 4))
        out = apply_overlay(self.baseline, self.rig, self.times,
                            dimmer_gain={"mh1": gain})
        self.assert_only(out, [slot])
        expected = np.floor(np.clip(self.baseline[:, slot].astype(float) * gain,
                                    0, 255) + .5).astype(np.uint8)
        np.testing.assert_array_equal(out[:, slot], expected)
        identity = apply_overlay(self.baseline, self.rig, self.times,
                                 dimmer_gain={"mh1": np.ones(len(self.times))})
        self.assert_only(identity, [])

    def test_dimmer_envelope_only_is_absolute_not_rgb_rescaling(self):
        track = np.linspace(0, 1, len(self.times))
        out = apply_overlay(self.baseline, self.rig, self.times,
                            dimmer_envelope={"wash1": track})
        slot = int(self.slot("wash1", 0))
        self.assert_only(out, [slot])
        np.testing.assert_array_equal(out[:, slot], np.floor(track * 255 + .5).astype(np.uint8))

    def test_all_angles_and_every_other_byte_survive_combined_overlay(self):
        rgb, gain, slots = {}, {}, []
        rng = np.random.default_rng(881)
        for fixture in self.rig.fixtures:
            if fixture.kind in ("moving_head", "wash", "projector"):
                rgb[fixture.name] = rng.random((len(self.times), 3))
                offsets = [6, 7, 8] if fixture.kind == "moving_head" else [2, 3, 4]
                slots.extend(self.slot(fixture.name, offsets))
            if fixture.kind in ("moving_head", "wash"):
                gain[fixture.name] = rng.random(len(self.times)) * 2
                slots.append(int(self.slot(fixture.name, 4 if fixture.kind == "moving_head" else 0)))
        out = apply_overlay(self.baseline, self.rig, self.times, rgb=rgb, dimmer_gain=gain)
        self.assert_only(out, slots)
        before = decode_stream(self.rig, self.baseline)
        after = decode_stream(self.rig, out)
        for field in ("pan", "tilt", "strobe", "zoom"):
            np.testing.assert_array_equal(after[field], before[field])

    def test_integer_caps_apply_to_requested_fields_only(self):
        out = apply_overlay(self.baseline, self.rig, self.times,
                            rgb={"proj1": np.ones((len(self.times), 3))},
                            dimmer_envelope={"mh1": np.ones(len(self.times))},
                            config=OverlayConfig(rgb_ceiling_byte=140, dimmer_ceiling_byte=90))
        rgb_slots = self.slot("proj1", [2, 3, 4])
        dimmer_slot = int(self.slot("mh1", 4))
        self.assert_only(out, [*rgb_slots, dimmer_slot])
        self.assertTrue(np.all(out[:, rgb_slots] == 140))
        self.assertTrue(np.all(out[:, dimmer_slot] == 90))

    def test_huge_finite_gain_saturates_without_overflow_or_wrap(self):
        slot = int(self.slot("wash1", 0))
        with np.errstate(over="raise", invalid="raise"):
            out = apply_overlay(self.baseline, self.rig, self.times,
                                dimmer_gain={"wash1": np.full(len(self.times), 1e308)},
                                config=OverlayConfig(dimmer_ceiling_byte=201))
        self.assert_only(out, [slot])
        np.testing.assert_array_equal(out[:, slot], np.where(self.baseline[:, slot] > 0, 201, 0))

    def test_caps_refuse_noninteger_out_of_range_and_boolean_values(self):
        for value in (-1, 256, 2.5, True, np.nan, np.inf, "90"):
            for field in ("rgb_ceiling_byte", "dimmer_ceiling_byte"):
                with self.subTest(value=value, field=field), self.assertRaises(ValueError):
                    OverlayConfig(**{field: value})

    def test_invalid_rgb_values_and_shapes_are_refused(self):
        bad = [np.ones(3), np.ones((len(self.times)-1, 3)),
               np.ones((len(self.times), 4)), np.ones((1, 3)),
               np.full((len(self.times), 3), -1e-9),
               np.full((len(self.times), 3), 1.00001),
               np.full((len(self.times), 3), np.nan),
               np.full((len(self.times), 3), np.inf),
               np.full((len(self.times), 3), 1j),
               np.full((len(self.times), 3), "0.5")]
        for track in bad:
            with self.subTest(shape=track.shape, dtype=track.dtype), self.assertRaises(ValueError):
                apply_overlay(self.baseline, self.rig, self.times, rgb={"mh1": track})

    def test_invalid_dimmer_values_shapes_conflicts_and_mapping_are_refused(self):
        for field in ("dimmer_gain", "dimmer_envelope"):
            for track in (1., np.ones((len(self.times), 1)), np.ones(len(self.times)-1),
                          np.full(len(self.times), -.1), np.full(len(self.times), np.inf),
                          np.full(len(self.times), np.nan)):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    apply_overlay(self.baseline, self.rig, self.times, **{field: {"mh1": track}})
        with self.assertRaises(ValueError):
            apply_overlay(self.baseline, self.rig, self.times,
                          dimmer_envelope={"mh1": np.full(len(self.times), 1.01)})
        with self.assertRaises(ValueError):
            apply_overlay(self.baseline, self.rig, self.times,
                          dimmer_gain={"mh1": np.ones(len(self.times))},
                          dimmer_envelope={"mh1": np.ones(len(self.times))})
        with self.assertRaises(ValueError):
            apply_overlay(self.baseline, self.rig, self.times, rgb=np.ones((len(self.times), 3)))

    def test_protected_kinds_and_unknown_names_cannot_be_requested(self):
        for name in ("blinder1", "strobe1", "not-in-rig"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                apply_overlay(self.baseline, self.rig, self.times,
                              rgb={name: np.ones((len(self.times), 3))})
        for name in ("proj1", "blinder1", "strobe1", "not-in-rig"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                apply_overlay(self.baseline, self.rig, self.times,
                              dimmer_gain={name: np.ones(len(self.times))})

    def test_full_timeline_finite_uniform_and_zero_origin_required(self):
        bad_times = [self.times[1:], self.times + .025, self.times[:, None],
                     np.arange(len(self.times)), np.zeros(len(self.times))]
        for index, value in ((30, np.nan), (30, np.inf), (30, .5), (30, .751)):
            times = self.times.copy()
            times[index] = value
            bad_times.append(times)
        for times in bad_times:
            with self.subTest(shape=times.shape), self.assertRaises(ValueError):
                apply_overlay(self.baseline, self.rig, times)

    def test_baseline_shape_and_dtype_are_not_coerced(self):
        for baseline in (self.baseline.astype(float), self.baseline[:, :511],
                         self.baseline[:, :, None], self.baseline.tolist()):
            with self.assertRaises(ValueError):
                apply_overlay(baseline, self.rig, self.times)

    def test_patch_validation_including_overlap_and_complete_footprint(self):
        invalid = [Rig([Fixture("a", "laser", 1)]),
                   Rig([Fixture("a", "wash", 0)]),
                   Rig([Fixture("a", "wash", 507)]),
                   Rig([Fixture("a", "wash", 1.5)]),
                   Rig([Fixture("a", "wash", True)]),
                   Rig([Fixture("a", "wash", 1), Fixture("b", "strobe", 7)]),
                   Rig([Fixture("a", "wash", 1), Fixture("a", "wash", 20)]),
                   Rig([Fixture("", "wash", 1)])]
        for rig in invalid:
            with self.subTest(rig=rig), self.assertRaises(ValueError):
                apply_overlay(self.baseline, rig, self.times)
        edge = Rig([Fixture("edge", "wash", 506)])
        out = apply_overlay(self.baseline, edge, self.times,
                            rgb={"edge": np.zeros((len(self.times), 3))})
        self.assert_only(out, [507, 508, 509])

    def test_empty_full_record_and_single_frame_are_valid(self):
        empty = np.empty((0, 512), dtype=np.uint8)
        out = apply_overlay(empty, self.rig, np.empty(0, dtype=float),
                            rgb={"mh1": np.empty((0, 3))})
        np.testing.assert_array_equal(out, empty)
        out = apply_overlay(self.baseline[:1], self.rig, np.array([0.]),
                            dimmer_gain={"mh1": np.ones(1)})
        np.testing.assert_array_equal(out, self.baseline[:1])

    def test_readonly_noncontiguous_baseline_supported_without_mutation(self):
        baseline = self.baseline[::2]
        baseline.setflags(write=False)
        times = self.times[::2]
        track = np.full((len(times), 3), .5)
        track.setflags(write=False)
        out = apply_overlay(baseline, self.rig, times, rgb={"mh1": track})
        self.assertFalse(np.shares_memory(out, baseline))
        np.testing.assert_array_equal(baseline, self.original[::2])

    def test_empty_and_zero_accents_abstain_even_with_low_caps(self):
        for events in (None, {}, {"mh1": []}, {"mh1": [AccentEvent(1., 0., .2)]}):
            with self.subTest(events=events):
                out = apply_accent_events(self.baseline, self.rig, self.times, events,
                                          accent_config=self.pulse_config,
                                          config=OverlayConfig(0, 0))
                self.assert_only(out, [])
        np.testing.assert_array_equal(sample_accent_events(self.times, [],
                                      config=self.pulse_config), np.zeros(len(self.times)))

    def test_single_accent_has_bounded_smooth_support_and_no_extra_pulses(self):
        event = AccentEvent(1., .5, .4)
        pulse = sample_accent_events(self.times, [event], config=self.pulse_config)
        self.assertAlmostEqual(pulse[40], .5)
        self.assertAlmostEqual(pulse[36], .25)
        self.assertAlmostEqual(pulse[44], .25)
        self.assertTrue(np.all(pulse[(self.times < .8) | (self.times > 1.2)] == 0))
        self.assertTrue(np.all(np.diff(pulse[32:41]) >= -1e-12))
        self.assertTrue(np.all(np.diff(pulse[40:49]) <= 1e-12))
        self.assertLessEqual(pulse.max(), self.pulse_config.max_strength)
        out = apply_accent_events(self.baseline, self.rig, self.times, {"mh1": [event]},
                                  accent_config=self.pulse_config, config=OverlayConfig(0, 90))
        slot = int(self.slot("mh1", 4))
        self.assert_only(out, [slot])
        np.testing.assert_array_equal(out[pulse == 0], self.baseline[pulse == 0])
        self.assertTrue(np.all(out[pulse > 0, slot] <= 90))

    def test_overlapping_accents_use_max_not_unbounded_sum(self):
        events = [AccentEvent(1., .75, .8), AccentEvent(1.3, .75, .8)]
        pulse = sample_accent_events(self.times, events, config=self.pulse_config)
        singles = [sample_accent_events(self.times, [event], config=self.pulse_config)
                   for event in events]
        np.testing.assert_array_equal(pulse, np.maximum(*singles))
        self.assertLessEqual(pulse.max(), .75)

    def test_invalid_accent_decisions_are_rejected_not_silently_selected(self):
        invalid = [[AccentEvent(1., .8, .2)], [AccentEvent(1., -.1, .2)],
                   [AccentEvent(1., .5, .01)], [AccentEvent(1., .5, 1.)],
                   [AccentEvent(-.1, .5, .2)], [AccentEvent(4., .5, .2)],
                   [AccentEvent(np.nan, .5, .2)],
                   [AccentEvent(1., .5, .2), AccentEvent(1.1, .4, .2)],
                   [AccentEvent(1.5, .5, .2), AccentEvent(1., .4, .2)],
                   [{"time": 1., "strength": .5, "width_s": .2}]]
        for events in invalid:
            with self.subTest(events=events), self.assertRaises(ValueError):
                sample_accent_events(self.times, events, config=self.pulse_config)
        for name in ("strobe1", "blinder1", "proj1", "unknown"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                apply_accent_events(self.baseline, self.rig, self.times,
                                    {name: [AccentEvent(1., .5, .2)]},
                                    accent_config=self.pulse_config)

    def test_accent_config_requires_explicit_finite_consistent_bounds(self):
        for values in ((1.1, .1, .5, .3), (.5, 0., .5, .3), (.5, .6, .5, .3),
                       (.5, .1, .5, 0.), (.5, .1, np.inf, .3)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AccentConfig(*values)


if __name__ == "__main__":
    unittest.main()

import numpy as np
import unittest

from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.dance_no_clock_render import REQUIRED, controls_to_dmx


def test_constant_controls_produce_exactly_constant_light():
    for level in (-2., 0., 2.):
        frames, _, report = controls_to_dmx(np.full((101, len(REQUIRED)), level), REQUIRED, 10.1)
        assert np.all(frames == frames[0])
        assert not report['independent_temporal_generators']


def test_dynamic_controls_drive_light_and_steering_without_strobe():
    z = np.zeros((100, len(REQUIRED)))
    z[30:60, REQUIRED.index('lateral_pos')] = 3.
    z[40:70, REQUIRED.index('qom')] = 3.
    frames, trace, report = controls_to_dmx(z, REQUIRED, 10.)
    baseline, _, _ = controls_to_dmx(np.zeros_like(z), REQUIRED, 10.)
    assert not np.array_equal(frames[:, :4], baseline[:, :4])
    assert np.ptp(trace['dimmer'][:, 0]) > .1
    for f in canonical_rig().fixtures:
        offset = 5 if f.kind == 'moving_head' else 1
        assert np.all(frames[:, f.address-1+offset] == 0)
    for h in report['heads']:
        assert h['pan_speed_deg_s'] <= 540.
        assert h['tilt_speed_deg_s'] <= 270.


def test_body_gestures_restore_angular_range_and_side_contrast():
    z = np.zeros((100, len(REQUIRED)))
    z[:, REQUIRED.index('lateral_pos')] = np.linspace(-2., 2., 100)
    z[:, REQUIRED.index('vert_bounce')] = np.linspace(-1., 1., 100)
    _, trace, report = controls_to_dmx(z, REQUIRED, 10.)
    assert all(h['pan_range_deg'][1]-h['pan_range_deg'][0] > 300 for h in report['heads'])
    assert all(h['tilt_range_deg'][1]-h['tilt_range_deg'][0] > 45 for h in report['heads'])
    heads = trace['dimmer'][:, :6]
    assert (heads.max(1)/heads.min(1)).max() > 6.


def test_accents_and_colour_come_from_predicted_motion():
    z = np.zeros((100, len(REQUIRED)))
    z[:, REQUIRED.index('foot_speed')] = -1.
    z[20:25, REQUIRED.index('foot_speed')] = 2.
    z[:, REQUIRED.index('hand_speed')] = -1.
    z[60:65, REQUIRED.index('hand_speed')] = 2.
    z[50:, REQUIRED.index('section_energy')] = 2.
    _, trace, _ = controls_to_dmx(z, REQUIRED, 10.)
    assert np.all(trace['foot_accent'][:60] == 0)
    assert trace['foot_accent'][85] > .9 and trace['hand_accent'][85] == 0
    assert trace['hand_accent'][245] > .9 and trace['foot_accent'][245] == 0
    assert trace['hue'][-1]-trace['hue'][0] > .25


def test_repeated_calls_do_not_depend_on_random_state_or_prior_calls():
    z = np.random.default_rng(11).normal(size=(100, len(REQUIRED)))
    first, _, _ = controls_to_dmx(z, REQUIRED, 10.)
    controls_to_dmx(-z, REQUIRED, 10.)
    np.random.seed(300)
    second, _, _ = controls_to_dmx(z, REQUIRED, 10.)
    assert np.array_equal(first, second)


def test_missing_nonfinite_or_clock_channels_are_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        controls_to_dmx(np.zeros((100, 2)), ['qom', 'com_speed'], 10.)
    z = np.zeros((100, len(REQUIRED)))
    z[0, 0] = np.nan
    with unittest.TestCase().assertRaises(ValueError):
        controls_to_dmx(z, REQUIRED, 10.)
    with unittest.TestCase().assertRaises(ValueError):
        controls_to_dmx(np.zeros((100, len(REQUIRED)+1)), [*REQUIRED, 'beat_period_s'], 10.)


def test_long_constant_tail_settles_without_restarting_a_pattern():
    z = np.zeros((200, len(REQUIRED)))
    z[:5, REQUIRED.index('lateral_pos')] = -10.
    z[5:, REQUIRED.index('lateral_pos')] = 10.
    frames, _, _ = controls_to_dmx(z, REQUIRED, 20.)
    assert np.all(frames[-400:] == frames[-1])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for key, value in globals().items()
                              if key.startswith('test_') and callable(value))

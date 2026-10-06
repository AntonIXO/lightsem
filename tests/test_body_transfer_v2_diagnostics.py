"""CPU counterexamples: distinguish activity, duplicated intent and lost timing."""
import json
import numpy as np
import pytest

from lightsem.conductor.body_transfer import build_outputs
from lightsem.conductor.body_transfer_v2.diagnostics import measure, _pair_indices


def body(frames):
    result = np.zeros((frames, 22, 3))
    result[..., 1] = 1.
    result[:, :, 0] = np.linspace(-.6, .6, 22)
    return result


def pack(outputs, angles, levels, rgb=None):
    """Independent test encoder; avoids solver and any algorithm under test."""
    frames = len(levels)
    if rgb is None:
        rgb = np.full((frames, len(outputs), 3), 160, np.uint8)
    values, patch, universe, address, head = [], [], 0, 0, 0
    for i, output in enumerate(outputs):
        width = 4 if output['kind'] == 'bulb' else 8
        if address + width > 512:
            universe, address = universe + 1, 0
        v = np.zeros((frames, width), np.uint8)
        offset = 0
        if width == 8:
            ticks = np.rint((.5 + angles[:, head] / [540., 270.]) * 65535).astype(np.int64)
            v[:, :4] = np.stack((ticks[:, 0] >> 8, ticks[:, 0] & 255,
                                  ticks[:, 1] >> 8, ticks[:, 1] & 255), axis=1)
            offset, head = 4, head + 1
        v[:, offset] = np.rint(levels[:, i] * 255).astype(np.uint8)
        v[:, offset + 1:offset + 4] = rgb[:, i]
        patch.append({**output, 'universe': universe, 'address': address + 1, 'n_channels': width})
        address += width
        values.append(v)
    dmx = np.zeros((frames, universe + 1, 512), np.uint8)
    for p, v in zip(patch, values):
        dmx[:, p['universe'], p['address'] - 1:p['address'] - 1 + p['n_channels']] = v
    return dmx, patch


def targets_from_angles(angles, outputs):
    positions = np.array([o['position'] for o in outputs])
    pan, tilt = np.moveaxis(np.deg2rad(angles), -1, 0)
    direction = np.stack((np.sin(pan) * np.cos(tilt), -np.cos(pan) * np.cos(tilt), np.sin(tilt)), -1)
    return positions + 5 * direction


def case(desired, actual=None, levels=None):
    t, n, _ = desired.shape
    outputs = build_outputs(n, 0, 0)
    levels = np.full((t, n), .6) if levels is None else levels
    dmx, patch = pack(outputs, desired if actual is None else actual, levels)
    intent = {'targets': targets_from_angles(desired, outputs), 'dimmer': levels,
              'field_points': body(t)}
    return body(t), intent, dmx, patch


def test_identical_targets_can_have_distinct_angles_and_static_spread_is_not_motion():
    t = 20
    outputs = build_outputs(2, 0, 0)
    targets = np.broadcast_to([0., 0., -3.], (t, 2, 3)).copy()
    vectors = targets - np.array([p['position'] for p in outputs])
    angles = np.stack((np.rad2deg(np.arctan2(vectors[..., 0], -vectors[..., 1])),
                       np.rad2deg(np.arcsin(vectors[..., 2] / np.linalg.norm(vectors, axis=-1)))), -1)
    levels = np.ones((t, 2)) * .6
    dmx, patch = pack(outputs, angles, levels)
    report = measure(body(t), {'targets': targets, 'dimmer': levels}, dmx, patch)
    assert report['duplicates']['desired_targets']['unique_streams'] == 1
    assert report['duplicates']['emitted_angles']['unique_streams'] == 2
    assert report['motion']['constant_by_output'] == [True, True]
    assert report['dimensions']['actual_hits']['participation_ratio'] == 0
    # Static widely separated targets still have zero temporal information.
    desired = np.tile([[[-30., -20.], [30., -20.]]], (t, 1, 1))
    spread = measure(*case(desired))
    assert spread['spatial']['horizontal_span_m']['median'] > 1
    assert spread['dimensions']['desired_targets']['constant']
    assert spread['recurrence']['intent_vs_emitted'] is None
    json.dumps(report, allow_nan=False)


def test_delayed_waveform_loses_timing_even_with_equal_range_and_distinct_hashes():
    fps, t, delay = 40., 800, 8
    time = np.arange(t) / fps
    source = 12 * np.sin(2 * np.pi * .8 * time)
    desired = np.stack((source, np.full(t, -20.)), axis=-1)[:, None]
    actual = desired.copy()
    actual[:, 0, 0] = 12 * np.sin(2 * np.pi * .8 * (time - delay / fps))
    report = measure(*case(desired, actual), fps=fps, start_s=0, duration_s=20)
    temporal = report['angular_transfer']['temporal']
    assert abs(temporal['best_velocity_lag_s'] - .2) <= temporal['velocity_lag_resolution_s']
    assert temporal['velocity_correlation_at_best_lag'] > .99
    assert temporal['velocity_correlation_zero_lag'] < .65
    assert .17 < temporal['extrema_absolute_timing_error_s']['median'] < .23
    assert temporal['extrema_recall'] > .9


def test_attenuation_is_separate_from_motion_range_and_brightness():
    fps, t = 40., 800
    source = 10 * np.sin(2 * np.pi * 1.5 * np.arange(t) / fps)
    desired = np.stack((source, np.full(t, -20.)), axis=-1)[:, None]
    actual = desired.copy()
    actual[:, 0, 0] *= .4
    report = measure(*case(desired, actual), start_s=0)
    assert report['angular_transfer']['band_emitted_to_requested_RMS_ratio']['1-2Hz'] == pytest.approx(.4, abs=.002)
    assert report['brightness']['moving']['range_by_output'] == [0.]
    assert report['motion']['speed_deg_s']['max'] > 0


def test_darkness_does_not_hide_fixed_brightness_geometric_clutter():
    desired = np.tile([[[45., -20.], [-45., -20.]]], (30, 1, 1))
    lit = measure(*case(desired))
    dark = measure(*case(desired, levels=np.zeros((30, 2))))
    assert lit['spatial']['fixed_brightness_ray_crossing_fraction'] == dark['spatial']['fixed_brightness_ray_crossing_fraction']
    assert dark['spatial']['dimmer_weighted_ray_crossing_fraction'] is None
    assert dark['brightness']['moving']['fraction_at_zero'] == 1
    json.dumps(dark, allow_nan=False)


def test_bulb_only_single_output_and_short_windows_have_defined_json():
    t = 2
    outputs = build_outputs(0, 0, 3)
    levels = np.array([[0, .5, 1], [.1, .6, .9]])
    rgb = np.full((t, 3, 3), 40, np.uint8)
    rgb[1, :, 0] = 80
    dmx, patch = pack(outputs, np.empty((t, 0, 2)), levels, rgb)
    intent = {'targets': np.tile([p['position'] for p in outputs], (t, 1, 1)), 'dimmer': levels}
    report = measure(body(t), intent, dmx, patch)
    assert report['counts']['moving'] == 0
    assert report['window']['used_full_short_control']
    assert report['motion']['speed_deg_s']['max'] is None
    assert report['spatial']['fixed_brightness_ray_crossing_fraction'] is None
    assert report['color']['palette_cue_count'] is None
    assert report['color']['groups']['right']['streams'][0]['RGB_changing_ticks_including_fades'] == 1
    assert report['color']['groups']['right']['streams'][0]['distinct_RGB_colors'] == 2
    json.dumps(report, allow_nan=False)
    single = measure(*case(np.tile([[[0., -20.]]], (t, 1, 1))))
    assert single['spatial']['total_pairs'] == 0
    assert single['motion']['acceleration_deg_s2']['max'] is None
    json.dumps(single, allow_nan=False)


def test_nonfinite_intent_fails_and_large_pair_sampling_is_bounded():
    b, intent, dmx, patch = case(np.tile([[[0., -20.]]], (4, 1, 1)))
    intent['targets'][0, 0, 0] = np.nan
    report = measure(b, intent, dmx, patch)
    assert report['finite'] is False and report['status'] == 'invalid_nonfinite_input'
    json.dumps(report, allow_nan=False)
    a, b, total = _pair_indices(10000)
    assert total == 49_995_000 and len(a) == 4096
    assert np.all((a >= 0) & (a < b) & (b < 10000))
    assert len(set(zip(a, b))) == 4096


def test_hold_duration_is_lost_when_output_keeps_moving():
    t = 160
    desired = np.tile([[[0., -20.]]], (t, 1, 1))
    actual = desired.copy()
    actual[:, 0, 0] = np.linspace(-10, 10, t)
    report = measure(*case(desired, actual))
    temporal = report['angular_transfer']['temporal']
    assert temporal['requested_hold_fraction'] == 1
    assert temporal['emitted_hold_fraction'] == 0
    assert temporal['hold_iou'] == 0
    assert temporal['best_velocity_lag_s'] is None

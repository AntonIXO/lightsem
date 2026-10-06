"""Counterexamples for source replacement, limb loss and output fanout."""
import numpy as np
import pytest
from lightsem.conductor.body_transfer import build_outputs, raw_transfer, constrained_transfer, decode


def body(frames=40):
    from lightsem.experiments.motion_study.geometry import OFFSETS, PARENTS
    pose = np.zeros((22, 3))
    pose[0, 1] = 1.
    for joint in range(1, 22):
        pose[joint] = pose[PARENTS[joint]] + OFFSETS[joint]
    return np.repeat(pose[None], frames, axis=0)


def test_frozen_body_has_no_autonomous_motion_or_bulb_chase():
    output = build_outputs()
    world = body()
    color = np.tile(np.array([40, 160, 220], np.uint8), (len(world), 1))
    dmx, patch, trace = constrained_transfer(world, output, color, color)
    assert np.all(trace['targets'] == trace['targets'][:1])
    assert np.all(dmx == dmx[:1])
    emitted = decode(dmx, patch)
    assert np.all(emitted['dimmer'] == emitted['dimmer'][:1])


def test_wrist_gesture_changes_assigned_outputs_without_moving_other_limbs():
    outputs = build_outputs()
    stationary = body(81)
    moved = stationary.copy()
    pulse = np.r_[np.zeros(20), np.linspace(0, .3, 21), np.linspace(.3, 0, 20), np.zeros(20)]
    moved[:, 20, 1] += pulse
    before, after = raw_transfer(stationary, outputs), raw_transfer(moved, outputs)
    for index, item in enumerate(outputs):
        if item['kind'] == 'bulb':
            continue
        if item['joint_weights'][20] > 0:
            # A hand may already point at the wall: its height, not depth,
            # carries the vertical gesture across the floor/wall chart.
            assert np.linalg.norm(np.ptp(after['targets'][:, index], axis=0)) > .01
            changed = np.flatnonzero(np.any(after['targets'][:, index] != before['targets'][:, index], axis=1))
            assert changed[0] == np.flatnonzero(pulse)[0]
        else:
            np.testing.assert_array_equal(before['targets'][:, index], after['targets'][:, index])
    assert not np.array_equal(before['dimmer'], after['dimmer'])


def test_eight_outputs_retain_more_than_four_distinct_body_attachments():
    outputs = build_outputs(beams=8, lasers=0, bulbs=0)
    weights = np.asarray([v['joint_weights'] for v in outputs])
    assert np.linalg.matrix_rank(weights) == 8
    assert len({tuple(w) for w in weights}) == 8


def test_dmx_spill_and_coordinate_inverse_use_emitted_bytes():
    outputs = build_outputs(beams=1, lasers=0, bulbs=130)
    world = body(5)
    color = np.tile(np.array([255, 100, 20], np.uint8), (len(world), 1))
    dmx, patch, raw = constrained_transfer(world, outputs, color, color)
    assert dmx.shape == (len(world), 2, 512)
    assert all(p['address'] + p['n_channels'] - 1 <= 512 for p in patch)
    emitted = decode(dmx, patch)
    desired = raw['targets'][:, 0] - np.asarray(outputs[0]['position'])
    desired /= np.linalg.norm(desired, axis=1)[:, None]
    np.testing.assert_allclose(emitted['directions'][:, 0], desired, atol=1e-4)
    np.testing.assert_array_equal(np.rint(emitted['rgb'][:, -1] * 255).astype(np.uint8), color)
    changed = dmx.copy()
    last = patch[-1]
    changed[:, last['universe'], last['address'] - 1] = 0
    assert np.all(decode(changed, patch)['dimmer'][:, -1] == 0)


def test_invalid_or_missing_body_is_not_replaced_by_fallback():
    with pytest.raises(ValueError):
        raw_transfer(np.full((40, 22, 3), np.nan), build_outputs())
    with pytest.raises(ValueError):
        build_outputs(beams=0, lasers=0, bulbs=0)

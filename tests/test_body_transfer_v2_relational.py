"""Counterexamples for relational transfer collapse and invented motion."""
import json

import numpy as np
import pytest

from lightsem.conductor.body_transfer_v2.common import REFERENCE, prepare_features
from lightsem.conductor.body_transfer_v2.relational import FUNCTIONS


def body(frames=17):
    return np.broadcast_to(REFERENCE, (frames, 22, 3)).copy()


@pytest.mark.parametrize('key', FUNCTIONS)
def test_frozen_noncanonical_pose_is_exactly_stationary_and_deterministic(key):
    world = body()
    world[:, 18, 2] += .15
    world[:, 20, 1] -= .3
    world[:, 21, 2] += .25
    f = prepare_features(world)
    a, b = FUNCTIONS[key](f), FUNCTIONS[key](f)
    a.validate(len(world))
    json.dumps(a.meta, allow_nan=False)
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.points, np.broadcast_to(a.points[:1], a.points.shape))
    if a.activity is not None:
        np.testing.assert_array_equal(a.activity, np.broadcast_to(a.activity[:1], a.activity.shape))


@pytest.mark.parametrize('key', FUNCTIONS)
def test_reference_is_zero_and_degenerate_pose_stays_finite(key):
    rest = FUNCTIONS[key](prepare_features(body()))
    np.testing.assert_allclose(rest.points, 0., atol=2e-14)
    collapsed = np.zeros((5, 22, 3))
    FUNCTIONS[key](prepare_features(collapsed)).validate(len(collapsed))


@pytest.mark.parametrize('key', FUNCTIONS)
def test_wrist_change_reaches_its_role_at_the_source_event(key):
    world = body()
    world[5:12, 20, 1] += .25
    result = FUNCTIONS[key](prepare_features(world))
    assert np.linalg.norm(result.points[7, 20]) > .01
    np.testing.assert_allclose(result.points[:5], 0., atol=2e-14)
    np.testing.assert_allclose(result.points[12:], 0., atol=2e-14)
    assert 20 in result.attribution[20]


@pytest.mark.parametrize('key', ['06', '07', '09'])
def test_fixed_wrist_retains_changed_elbow_information(key):
    """The legacy tip-only output loses this articulation entirely."""
    world = body()
    world[5:12, 18, 2] += .25
    result = FUNCTIONS[key](prepare_features(world))
    assert np.linalg.norm(result.points[7, 20]) > .001
    assert 18 in result.attribution[20]


def test_proximity_tracks_partner_and_support_does_not_invent_release():
    world = body(30)
    world[:, 20] = REFERENCE[9] + [.2, 0., 0.]
    world[:, 21] = world[:, 20] + [.6, 0., 0.]
    a = FUNCTIONS['08'](prepare_features(world))
    world[10:20, 21] = world[10:20, 20] + [.12, 0., 0.]
    b = FUNCTIONS['08'](prepare_features(world))
    assert np.linalg.norm(b.points[15, 20] - a.points[15, 20]) > .01
    np.testing.assert_allclose(b.points[20:], a.points[20:], atol=1e-14)
    assert 21 in b.attribution[20]


def test_distance_constellation_is_sensitive_to_reflection_and_world_turn():
    world = body()
    world[:, 20, 2] += .3
    f = prepare_features(world)
    reflected = world.copy()
    reflected[:, :, 2] *= -1
    r = FUNCTIONS['06'](prepare_features(reflected))
    a = FUNCTIONS['06'](f)
    assert np.max(np.abs(a.points - r.points)) > .05


def test_poisson_is_nontrivial_and_retains_uniform_body_lift():
    world = body()
    world[:, 18, 2] += .2
    f = prepare_features(world)
    a = FUNCTIONS['09'](f)
    assert not np.allclose(a.points, f.displacement)
    assert np.linalg.norm(a.points[0, 21]) > 1e-4
    lifted = body()
    lifted[:, :, 1] += .4
    b = FUNCTIONS['09'](prepare_features(lifted))
    np.testing.assert_allclose(b.points[:, :, 1], .1, atol=1e-14)
    np.testing.assert_allclose(b.points[:, :, [0, 2]], 0., atol=1e-14)


def bone_preserving_clap():
    """Two-link arm IK supplies a clap without stretching bones or moving torso."""
    amount = np.r_[np.linspace(0., 1., 61), np.ones(15), np.linspace(1., 0., 61)]
    world = body(len(amount))
    center = (REFERENCE[16] + REFERENCE[17]) / 2. + [0., -.20, .42]
    for shoulder, elbow, wrist, sign in ((16, 18, 20, 1), (17, 19, 21, -1)):
        base = REFERENCE[shoulder]
        upper = np.linalg.norm(REFERENCE[elbow] - base)
        lower = np.linalg.norm(REFERENCE[wrist] - REFERENCE[elbow])
        destination = center + [sign * .03, 0., 0.]
        tip = (1. - amount[:, None]) * REFERENCE[wrist] + amount[:, None] * destination
        delta = tip - base
        distance = np.linalg.norm(delta, axis=1)
        assert np.all(distance <= upper + lower + 1e-12)
        assert np.all(distance >= abs(upper - lower))
        axis = delta / distance[:, None]
        along = (upper ** 2 - lower ** 2 + distance ** 2) / (2. * distance)
        bend = np.array([0., -1., 0.]) - axis * (-axis[:, 1, None])
        bend /= np.linalg.norm(bend, axis=1, keepdims=True)
        height = np.sqrt(np.maximum(upper ** 2 - along ** 2, 0.))
        world[:, elbow] = base + along[:, None] * axis + height[:, None] * bend
        world[:, wrist] = tip
        np.testing.assert_allclose(np.linalg.norm(world[:, elbow] - base, axis=1), upper, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(tip - world[:, elbow], axis=1), lower, atol=1e-12)
    return world


def test_proximity_v3_bone_preserving_clap_converges_and_releases():
    world = bone_preserving_clap()
    result = FUNCTIONS['08'](prepare_features(world)).validate(len(world))
    assert result.shared_groups == ((9, 15, 20, 21),)
    assert result.absolute_points.shape == world.shape
    np.testing.assert_allclose(result.points,
                               result.absolute_points - result.absolute_reference, atol=1e-14)
    separation = np.linalg.norm(result.absolute_points[:, 20] - result.absolute_points[:, 21], axis=1)
    assert separation[0] > 1.
    assert separation[60] < .00001
    assert np.all(np.diff(separation[:61]) <= 1e-10)
    np.testing.assert_allclose(result.absolute_points[-1], result.absolute_points[0], atol=1e-12)
    np.testing.assert_array_equal(result.absolute_points[60:76],
                                  np.broadcast_to(result.absolute_points[60], (16, 22, 3)))
    for joint in (9, 15, 20, 21):
        assert result.attribution[joint] == [0, 9, 15, 20, 21]


def test_proximity_v3_isolated_pair_contracts_without_overshoot():
    from lightsem.conductor.body_transfer_v2.relational import _smooth_proximity
    world = body(101)
    source_distance = np.linspace(.6, .001, len(world))
    # Put this pair away from the torso/head to isolate the analytic two-node case.
    world[:, 20] = np.column_stack((source_distance / 2., np.full(len(world), 3.), np.zeros(len(world))))
    world[:, 21] = np.column_stack((-source_distance / 2., np.full(len(world), 3.), np.zeros(len(world))))
    result = FUNCTIONS['08'](prepare_features(world))
    c = _smooth_proximity(source_distance, .10, .45)
    w = c / (2. * np.maximum(1. - c, 1e-5))
    signed_separation = result.absolute_points[:, 20, 0] - result.absolute_points[:, 21, 0]
    np.testing.assert_allclose(signed_separation, source_distance / (1. + 2. * w), atol=1e-14)
    assert np.all(signed_separation >= 0.)
    assert np.all(signed_separation <= source_distance)
    np.testing.assert_allclose((result.absolute_points[:, 20] + result.absolute_points[:, 21]) / 2.,
                               np.broadcast_to([0., 3., 0.], (len(world), 3)), atol=2e-11)


def test_proximity_v3_frozen_contact_absolute_geometry_is_exact_and_repeatable():
    pose = bone_preserving_clap()[60]
    world = np.broadcast_to(pose, (89, 22, 3)).copy()
    first = FUNCTIONS['08'](prepare_features(world))
    second = FUNCTIONS['08'](prepare_features(world))
    np.testing.assert_array_equal(first.absolute_points, second.absolute_points)
    np.testing.assert_array_equal(first.absolute_reference, second.absolute_reference)
    np.testing.assert_array_equal(first.absolute_points,
                                  np.broadcast_to(first.absolute_points[:1], world.shape))

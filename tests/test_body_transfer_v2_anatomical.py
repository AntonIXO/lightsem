"""Source-dependence and anatomical-information checks for the five fields."""
import json

import numpy as np
import pytest

from lightsem.conductor.body_transfer_v2.anatomical import FUNCTIONS
from lightsem.conductor.body_transfer_v2.common import REFERENCE, prepare_features
from lightsem.conductor.body_transfer_v2.skeleton import PARENTS


def body(frames=7):
    return np.broadcast_to(REFERENCE, (frames, 22, 3)).copy()


@pytest.mark.parametrize('identifier', FUNCTIONS)
def test_fixed_neutral_is_zero_and_metadata_serializes(identifier):
    field = FUNCTIONS[identifier](prepare_features(body())).validate(7)
    np.testing.assert_allclose(field.points, 0, atol=1e-14)
    json.dumps(field.meta, allow_nan=False)


@pytest.mark.parametrize('identifier', FUNCTIONS)
def test_frozen_non_neutral_body_is_exactly_constant_and_deterministic(identifier):
    pose = body()
    pose[:, 18] += [.12, -.16, .10]
    pose[:, 20] += [.24, .32, -.12]
    pose[:, 15] += [.08, .04, -.03]
    first = FUNCTIONS[identifier](prepare_features(pose)).validate(len(pose))
    again = FUNCTIONS[identifier](prepare_features(pose)).validate(len(pose))
    np.testing.assert_array_equal(first.points, again.points)
    np.testing.assert_array_equal(first.points, np.broadcast_to(first.points[:1], first.points.shape))
    assert np.linalg.norm(first.points) > .01


@pytest.mark.parametrize('identifier', FUNCTIONS)
@pytest.mark.parametrize('joint,delta', [(20, [.12, .20, .07]),
                                       (15, [.12, .02, .08]),
                                       (9, [.10, .03, -.06])])
def test_limb_head_and_torso_counterfactuals_reach_their_samples(identifier, joint, delta):
    original = body()
    edited = original.copy()
    edited[:, joint] += delta
    a = FUNCTIONS[identifier](prepare_features(original)).points
    b = FUNCTIONS[identifier](prepare_features(edited)).points
    assert np.linalg.norm(b[:, joint] - a[:, joint]) > .01


@pytest.mark.parametrize('identifier', FUNCTIONS)
def test_left_wrist_change_does_not_drive_right_wrist(identifier):
    original = body()
    edited = original.copy()
    edited[:, 20] += [.1, .25, -.15]
    a = FUNCTIONS[identifier](prepare_features(original)).points
    b = FUNCTIONS[identifier](prepare_features(edited)).points
    np.testing.assert_array_equal(a[:, 21], b[:, 21])


@pytest.mark.parametrize('identifier', FUNCTIONS)
def test_horizontal_world_translation_is_removed_and_height_survives(identifier):
    original = body()
    translated = original + np.array([11., 0., -17.])
    raised = original + np.array([0., .20, 0.])
    a = FUNCTIONS[identifier](prepare_features(original)).points
    b = FUNCTIONS[identifier](prepare_features(translated)).points
    c = FUNCTIONS[identifier](prepare_features(raised)).points
    np.testing.assert_allclose(a, b, atol=2e-13)
    assert np.linalg.norm(c - a) > .01


@pytest.mark.parametrize('identifier', FUNCTIONS)
def test_collapsed_and_straight_limbs_remain_finite(identifier):
    collapsed = np.zeros((7, 22, 3))
    field = FUNCTIONS[identifier](prepare_features(collapsed)).validate(7)
    np.testing.assert_array_equal(field.points, np.broadcast_to(field.points[:1], field.points.shape))
    straight = body()
    straight[:, [16, 18, 20]] = np.array([[.15, 1.5, 0.], [.45, 1.5, 0.], [.75, 1.5, 0.]])
    FUNCTIONS[identifier](prepare_features(straight)).validate(7)


def test_support_change_moves_upper_body_relative_to_support():
    original = body()
    raised_foot = original.copy()
    raised_foot[:, 10, 1] += .2
    a = FUNCTIONS['05'](prepare_features(original)).points
    b = FUNCTIONS['05'](prepare_features(raised_foot)).points
    assert np.linalg.norm(b[:, 20] - a[:, 20]) > .01
    assert 10 in FUNCTIONS['05'](prepare_features(raised_foot)).attribution[20]


def test_bend_plane_detects_elbow_change_with_fixed_wrist():
    original = body()
    flexed = original.copy()
    flexed[:, 18] += [0., .18, .12]
    a = FUNCTIONS['02'](prepare_features(original)).points
    b = FUNCTIONS['02'](prepare_features(flexed)).points
    np.testing.assert_array_equal(original[:, 20], flexed[:, 20])
    assert np.linalg.norm(b[:, 20] - a[:, 20]) > .1


@pytest.mark.parametrize('identifier', FUNCTIONS)
def test_every_observed_perturbation_is_declared_in_attribution(identifier):
    original = body()
    reference = FUNCTIONS[identifier](prepare_features(original))
    for source in range(22):
        edited = original.copy()
        edited[:, source] += [.011, .019, -.013]
        changed = FUNCTIONS[identifier](prepare_features(edited)).points
        responding = np.flatnonzero(np.linalg.norm(changed - reference.points, axis=(0, 2)) > 1e-10)
        for sample in responding:
            assert source in reference.attribution[sample], (identifier, source, sample)


def test_sculpture_absolute_fk_preserves_fixed_expanded_segment_lengths():
    source = body()
    source[:, 18, 1] += np.linspace(0., .3, len(source))
    source[:, 20, 2] += np.linspace(0., -.2, len(source))
    source[:, 9, 0] += np.linspace(0., .1, len(source))
    features = prepare_features(source)
    field = FUNCTIONS['03'](features).validate(len(source))
    parents = np.asarray(PARENTS[1:])
    actual = np.linalg.norm(field.absolute_points[:, 1:] - field.absolute_points[:, parents], axis=-1)
    gains = np.full(22, 1.4)
    gains[[18, 19]] = 1.8
    gains[[20, 21]] = 2.8
    gains[[7, 8]] = 1.9
    gains[[10, 11]] = 1.2
    expected = np.linalg.norm(features.reference_bones[1:], axis=-1) * gains[1:]
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-14)
    np.testing.assert_array_equal(field.points, field.absolute_points - field.absolute_reference)


def test_sculpture_neutral_is_an_expanded_connected_geometry_not_zero_points():
    field = FUNCTIONS['03'](prepare_features(body())).validate(7)
    assert field.shared_groups == (tuple(range(22)),)
    np.testing.assert_allclose(field.absolute_points,
                               np.broadcast_to(field.absolute_reference, field.absolute_points.shape), atol=1e-14)
    np.testing.assert_allclose(field.points, 0, atol=1e-14)
    parents = np.asarray(PARENTS[1:])
    edges = field.absolute_reference[1:] - field.absolute_reference[parents]
    source_edges = REFERENCE[1:] - REFERENCE[parents]
    assert np.all(np.linalg.norm(edges, axis=-1) > np.linalg.norm(source_edges, axis=-1))
    np.testing.assert_allclose(edges / np.linalg.norm(edges, axis=-1, keepdims=True),
                               source_edges / np.linalg.norm(source_edges, axis=-1, keepdims=True), atol=1e-14)
    assert np.ptp(field.absolute_reference[:, 0]) > np.ptp(REFERENCE[:, 0])
    assert np.ptp(field.absolute_reference[:, 1]) > np.ptp(REFERENCE[:, 1])


def test_sculpture_shared_absolute_group_is_exactly_constant_for_a_frozen_pose():
    frozen = body()
    frozen[:, 20] += [.1, .2, -.15]
    frozen[:, 15] += [-.08, .03, .02]
    field = FUNCTIONS['03'](prepare_features(frozen)).validate(len(frozen))
    for group in field.shared_groups:
        values = field.absolute_points[:, group]
        np.testing.assert_array_equal(values, np.broadcast_to(values[:1], values.shape))
    assert np.linalg.norm(field.absolute_points[0] - field.absolute_reference) > .1

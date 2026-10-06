"""Check spatial meaning, beyond source sensitivity and distinct DMX hashes.

The shoulder clap is a bone-preserving counterexample to the original V2
readout. Emitted beam proximity is measured on finite stage ray segments;
their wall/floor endpoints need not coincide when an encounter is in air.
"""
from copy import deepcopy

import numpy as np
import pytest

from lightsem.conductor.body_transfer import compile_intent, decode
from lightsem.conductor.body_transfer_v2 import make_outputs, map_body
from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.conductor.body_transfer_v2.skeleton import PARENTS
from scripts.rendering.render_body_transfer import ray_hits


def segment_distance(origin_a, origin_b, end_a, end_b):
    """Exact closest distance on two closed segments, including endpoint cases."""
    u, v, w = end_a - origin_a, end_b - origin_b, origin_a - origin_b
    a = np.sum(u * u, axis=-1)
    b = np.sum(u * v, axis=-1)
    c = np.sum(v * v, axis=-1)
    d, e = np.sum(u * w, axis=-1), np.sum(v * w, axis=-1)
    determinant = a * c - b * b
    s = (b * e - c * d) / np.maximum(determinant, 1e-20)
    t = (a * e - b * d) / np.maximum(determinant, 1e-20)
    interior = ((s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
                & (determinant > 1e-12))
    candidates = [np.where(interior,
        np.linalg.norm(w + s[..., None] * u - t[..., None] * v, axis=-1), np.inf)]
    for s_endpoint in (0., 1.):
        parameter = np.clip((e + b * s_endpoint) / c, 0., 1.)
        candidates.append(np.linalg.norm(w + s_endpoint * u - parameter[..., None] * v, axis=-1))
    for t_endpoint in (0., 1.):
        parameter = np.clip((b * t_endpoint - d) / a, 0., 1.)
        candidates.append(np.linalg.norm(w + parameter[..., None] * u - t_endpoint * v, axis=-1))
    return np.min(candidates, axis=0)


@pytest.fixture(scope='module')
def clap():
    angles = np.r_[np.zeros(40), np.linspace(0., 1.8, 161), np.full(80, 1.8),
                   np.linspace(1.8, 0., 161), np.zeros(40)]
    body = np.repeat(REFERENCE[None], len(angles), axis=0)
    axis = np.array([0., 1., 0.])
    for pivot, joints, sign in ((16, [18, 20], 1.), (17, [19, 21], -1.)):
        vectors = body[:, joints] - body[:, pivot:pivot + 1]
        angle = sign * angles
        cosine, sine = np.cos(angle)[:, None, None], np.sin(angle)[:, None, None]
        body[:, joints] = (body[:, pivot:pivot + 1] + vectors * cosine
                           + np.cross(axis, vectors) * sine
                           + axis * np.sum(vectors * axis, axis=-1, keepdims=True) * (1 - cosine))
    return body


def _compile(body, outputs, identifier):
    intent = map_body(body, outputs, identifier)
    color = np.broadcast_to(np.array([80, 160, 220], np.uint8), (len(body), 3)).copy()
    dmx, patch, _ = compile_intent(intent, outputs, color, color)
    return intent, dmx, patch, decode(dmx, patch)


@pytest.fixture(scope='module')
def compiled_clap(clap):
    return _compile(clap, make_outputs(beams=2, lasers=0, bulbs=0), '08')


def test_segment_distance_checks_finite_rays_not_infinite_lines_or_hits():
    # Crossing inside the segments, despite separated endpoints.
    assert segment_distance(np.array([0., 0., 0.]), np.array([1., 0., 0.]),
                            np.array([1., 1., 0.]), np.array([0., 1., 0.])) == 0.
    # Infinite lines cross beyond one endpoint: finite rays remain separated.
    assert segment_distance(np.array([0., 0., 0.]), np.array([2., 1., 0.]),
                            np.array([1., 0., 0.]), np.array([2., -1., 0.])) == 1.
    # Parallel segments exercise the singular interior solution.
    assert segment_distance(np.array([0., 0., 0.]), np.array([0., 1., 0.]),
                            np.array([1., 0., 0.]), np.array([1., 1., 0.])) == 1.


def test_shoulder_clap_preserves_bones_and_closes_without_crossing(clap):
    parents = np.asarray(PARENTS[1:])
    lengths = np.linalg.norm(clap[:, 1:] - clap[:, parents], axis=-1)
    np.testing.assert_allclose(lengths, np.broadcast_to(lengths[:1], lengths.shape), atol=2e-15)
    separation = np.linalg.norm(clap[:, 20] - clap[:, 21], axis=-1)
    assert separation[20] > 1.4
    assert separation[240] < .10
    assert np.all(clap[:, 20, 0] > clap[:, 21, 0])
    assert np.all(np.diff(separation[40:201]) < 0.)


def test_proximity_approach_survives_internal_and_desired_geometry(compiled_clap):
    intent, _, _, _ = compiled_clap
    internal = np.linalg.norm(intent['field_absolute_points'][:, 20]
                              - intent['field_absolute_points'][:, 21], axis=-1)
    desired = np.linalg.norm(intent['targets'][:, 0] - intent['targets'][:, 1], axis=-1)
    assert np.all(np.diff(internal[40:201]) < 0.)
    assert np.all(np.diff(desired[40:201]) < 0.)
    assert internal[240] < 2e-6
    assert desired[240] < 4e-6
    np.testing.assert_allclose(desired, 2 * intent['shared_similarity_scale'][:, 0] * internal,
                               atol=2e-14)


def test_proximity_approach_and_contact_survive_emitted_finite_rays(compiled_clap):
    intent, _, patch, controls = compiled_clap
    positions = np.asarray([item['position'] for item in patch])
    hits = ray_hits(positions, controls['directions'])
    closest = segment_distance(positions[0], positions[1], hits[:, 0], hits[:, 1])
    desired = np.linalg.norm(intent['targets'][:, 0] - intent['targets'][:, 1], axis=-1)
    assert closest[20] > 1.
    assert np.max(closest[225:265]) < .003
    assert np.corrcoef(desired[40:201], closest[40:201])[0, 1] > .90
    # A millimetre tolerance admits DMX rounding, not metre-scale inversion.
    assert np.max(np.diff(closest[40:201])) < .001
    np.testing.assert_allclose(closest[460], closest[20], atol=.001)


def test_proximity_release_restores_geometry_and_bulb_centers(compiled_clap):
    intent, _, _, _ = compiled_clap
    for key in ('targets', 'field_absolute_points', 'bulb_field_centers'):
        np.testing.assert_array_equal(intent[key][460], intent[key][20])
    centers = intent['bulb_field_centers']
    assert abs(centers[20, 20] - centers[20, 21]) > .2
    assert abs(centers[240, 20] - centers[240, 21]) < 1e-6
    # Pure wrist-role targets read the same shared encounter geometry.
    np.testing.assert_array_equal(intent['targets'][:, 0], intent['shared_stage_points'][:, 20])
    np.testing.assert_array_equal(intent['targets'][:, 1], intent['shared_stage_points'][:, 21])


@pytest.mark.parametrize('identifier', ['03', '08'])
@pytest.mark.parametrize('frame', [20, 120, 240])
def test_frozen_source_has_constant_targets_bulbs_and_dmx(identifier, frame, clap):
    frozen = np.repeat(clap[frame:frame + 1], 13, axis=0)
    intent, dmx, _, _ = _compile(frozen, make_outputs(), identifier)
    for key in ('targets', 'dimmer', 'bulb_field_centers', 'shared_stage_points'):
        np.testing.assert_array_equal(intent[key], np.broadcast_to(intent[key][:1], intent[key].shape))
    np.testing.assert_array_equal(dmx, np.broadcast_to(dmx[:1], dmx.shape))


@pytest.mark.parametrize('identifier', ['03', '08'])
def test_appended_and_permuted_fixtures_preserve_existing_semantics(identifier, clap):
    body = clap[170:211]
    outputs = make_outputs()
    original = map_body(body, outputs, identifier)
    expanded = make_outputs(beams=13, lasers=11, bulbs=75)[::-1]
    new = map_body(body, expanded, identifier)
    by_name = {item['name']: i for i, item in enumerate(expanded)}
    indices = [by_name[item['name']] for item in outputs]
    for key in ('targets', 'dimmer'):
        np.testing.assert_array_equal(original[key], new[key][:, indices])
    for key in ('field_absolute_points', 'shared_stage_points', 'bulb_field_centers'):
        np.testing.assert_array_equal(original[key], new[key])
    # Also inspect actual quantized commands for the two encounter fixtures.
    baseline = _compile(body, outputs[:2], identifier)[3]
    larger = _compile(body, [expanded[by_name['beam_002']], expanded[by_name['beam_003']],
                             expanded[by_name['beam_001']]], identifier)[3]
    for key in ('angles_deg', 'directions', 'dimmer', 'rgb'):
        np.testing.assert_array_equal(baseline[key], larger[key][:, [2, 0]])


@pytest.mark.parametrize('vertical_lift', [0., 3.])
def test_fk_absolute_pairwise_shape_is_one_similarity_before_mechanics(vertical_lift, clap):
    body = clap[[20, 80, 160, 240, 360, 460]].copy()
    body[:, :, 1] += vertical_lift
    # Explicit pure-role outputs inspect all 22 samples, including roles not
    # present in the default 13 moving fixtures; no blended sample is relabeled.
    prototype = make_outputs(beams=1, lasers=0, bulbs=0)[0]
    outputs = []
    for role in range(22):
        item = deepcopy(prototype)
        item.update(name=f'role_{role:02d}', role_index=role, secondary_role=role,
                    role_blend=0., spatial_slot=role)
        outputs.append(item)
    intent = map_body(body, outputs, '03')
    absolute = intent['field_absolute_points']
    targets = intent['targets']
    np.testing.assert_array_equal(targets, intent['shared_stage_points'])
    a, b = np.triu_indices(22, 1)
    source_edges = absolute[:, a] - absolute[:, b]
    target_edges = targets[:, a] - targets[:, b]
    scale = 2 * intent['shared_similarity_scale'][:, 0]
    # Vector equality establishes a single rotation as well as equal ratios.
    np.testing.assert_allclose(target_edges,
        source_edges * np.array([-1., 1., -1.]) * scale[:, None, None], atol=3e-14)
    np.testing.assert_allclose(np.linalg.norm(target_edges, axis=-1),
        np.linalg.norm(source_edges, axis=-1) * scale[:, None], atol=3e-14)
    assert np.all(abs(targets - [0., 2.4, -3.5]) <= np.array([7.5, 2.2, 2.3]) + 1e-12)
    if vertical_lift:
        assert np.all(intent['shared_similarity_scale'] < 1.)
    # Neutral hands, feet and head retain vertical structure instead of one row.
    assert np.ptp(targets[0, [20, 21, 7, 8, 15], 1]) > .5

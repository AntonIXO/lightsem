"""CPU integration counterexamples from body geometry through emitted DMX.

Synthetic articulation is test input, never a fallback animation source. These
checks establish engineering invariants, not choreography or music quality.
"""
from copy import deepcopy
from itertools import combinations
import json

import numpy as np
import pytest

from lightsem.conductor.body_transfer import (
    build_outputs, compile_intent, decode, raw_transfer,
)
from lightsem.conductor.body_transfer_v2 import CATALOG, algorithms, make_outputs, map_body
from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.conductor.body_transfer_v2.skeleton import PARENTS


IDS = tuple(f'{i:02d}' for i in range(1, 16))
FRAMES = 49


def _body(frames=FRAMES):
    return np.broadcast_to(REFERENCE, (frames, 22, 3)).copy()


def _pulse():
    # One reach, a hold, and a return. No periodic input is needed to test time.
    return np.r_[np.zeros(7), np.linspace(0, 1, 13), np.ones(9),
                 np.linspace(1, 0, 13), np.zeros(7)]


def _rotate_subtree(body, pivot, angles, axis):
    """Preserve bone lengths while rotating all descendants around a joint."""
    descendants = []
    for joint in range(22):
        parent = PARENTS[joint]
        while parent >= 0:
            if parent == pivot:
                descendants.append(joint)
                break
            parent = PARENTS[parent]
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    vectors = body[:, descendants] - body[:, pivot:pivot + 1]
    cosine = np.cos(angles)[:, None, None]
    sine = np.sin(angles)[:, None, None]
    rotated = (vectors * cosine + np.cross(axis, vectors) * sine
               + np.sum(vectors * axis, axis=-1, keepdims=True) * axis * (1 - cosine))
    body[:, descendants] = body[:, pivot:pivot + 1] + rotated
    return body


@pytest.fixture(scope='module')
def articulated():
    body = _body()
    pulse = _pulse()
    _rotate_subtree(body, 16, .70 * pulse, [0, 0, 1])
    _rotate_subtree(body, 17, -.38 * pulse ** 2, [0, 1, 0])
    _rotate_subtree(body, 1, .25 * pulse, [1, 0, 0])
    _rotate_subtree(body, 12, .55 * pulse, [0, 0, 1])
    return body


def _compile(intent, outputs):
    color = np.broadcast_to(np.array([40, 160, 220], dtype=np.uint8),
                            (len(intent['targets']), 3)).copy()
    dmx, patch, trace = compile_intent(intent, outputs, color, color)
    return dmx, patch, decode(dmx, patch), trace


@pytest.fixture(scope='module')
def intents(articulated):
    outputs = make_outputs()
    return {identifier: map_body(articulated, outputs, identifier) for identifier in IDS}


def test_catalog_has_exactly_fifteen_callable_algorithms():
    assert tuple(sorted(CATALOG)) == IDS
    assert tuple(sorted(algorithms())) == IDS
    assert all(callable(function) for function in algorithms().values())


def test_default_rig_preserves_baseline_physical_positions():
    before, after = build_outputs(), make_outputs()
    assert len(after) == 63
    for old, new in zip(before, after):
        assert (old['name'], old['kind']) == (new['name'], new['kind'])
        np.testing.assert_array_equal(old['position'], new['position'])
        if old['kind'] != 'bulb':
            assert old['pan_range_deg'] == new['pan_range_deg']
            assert old['tilt_range_deg'] == new['tilt_range_deg']


def test_expanding_counts_preserves_fixture_identity_geometry_and_roles():
    small = {item['name']: item for item in make_outputs(beams=2, lasers=1, bulbs=3)}
    baseline = {item['name']: item for item in make_outputs()}
    large = {item['name']: item for item in make_outputs(beams=21, lasers=18, bulbs=137)}
    for name, fixture in small.items():
        assert fixture == baseline[name]
    for name, fixture in baseline.items():
        assert fixture == large[name]


def test_appended_fixtures_do_not_reuse_existing_physical_positions():
    outputs = make_outputs(beams=35, lasers=35, bulbs=130)
    positions = [tuple(item['position']) for item in outputs]
    assert len(set(positions)) == len(positions)


@pytest.mark.parametrize('identifier', IDS)
def test_fixture_permutation_only_permutes_intent(identifier, articulated, intents):
    outputs = make_outputs()
    permutation = np.r_[np.arange(1, len(outputs), 2), np.arange(0, len(outputs), 2)]
    reordered = map_body(articulated, [outputs[i] for i in permutation], identifier)
    original = intents[identifier]
    for key in ('targets', 'dimmer', 'source_points'):
        np.testing.assert_array_equal(reordered[key], original[key][:, permutation])
    assert reordered['attribution'] == [original['attribution'][i] for i in permutation]
    np.testing.assert_array_equal(reordered['field_points'], original['field_points'])


@pytest.mark.parametrize('identifier', IDS)
def test_extra_fixtures_preserve_existing_source_timing(identifier, articulated, intents):
    old_outputs = make_outputs()
    expanded = make_outputs(beams=12, lasers=9, bulbs=60)
    new = map_body(articulated, expanded, identifier)
    by_name = {item['name']: i for i, item in enumerate(expanded)}
    indices = [by_name[item['name']] for item in old_outputs]
    for key in ('targets', 'dimmer', 'source_points'):
        np.testing.assert_array_equal(intents[identifier][key], new[key][:, indices])


@pytest.mark.parametrize('identifier', IDS)
def test_permutation_and_expansion_preserve_decoded_existing_fixtures(identifier, articulated):
    body = articulated[8:25]
    outputs = make_outputs(beams=3, lasers=1, bulbs=2)
    _, _, original, _ = _compile(map_body(body, outputs, identifier), outputs)
    expanded = make_outputs(beams=5, lasers=3, bulbs=5)[::-1]
    _, _, added, _ = _compile(map_body(body, expanded, identifier), expanded)
    by_name = {item['name']: i for i, item in enumerate(expanded)}
    indices = [by_name[item['name']] for item in outputs]
    # Patch offsets legitimately change. Decoding must recover the same motion,
    # color and levels for every preserved physical fixture.
    for key in ('angles_deg', 'directions', 'dimmer', 'rgb'):
        np.testing.assert_array_equal(original[key], added[key][:, indices])


@pytest.mark.parametrize('identifier', IDS)
def test_frozen_non_neutral_body_produces_constant_emitted_dmx(identifier, articulated):
    frozen = np.repeat(articulated[23:24], 11, axis=0)
    outputs = make_outputs()
    intent = map_body(frozen, outputs, identifier)
    for key in ('targets', 'dimmer', 'field_points'):
        np.testing.assert_array_equal(intent[key], np.broadcast_to(intent[key][:1], intent[key].shape))
    dmx, _, controls, _ = _compile(intent, outputs)
    np.testing.assert_array_equal(dmx, np.broadcast_to(dmx[:1], dmx.shape))
    for key in ('directions', 'dimmer'):
        np.testing.assert_array_equal(controls[key], np.broadcast_to(controls[key][:1], controls[key].shape))


@pytest.mark.parametrize('identifier', IDS)
def test_repeat_calls_are_exact_and_inputs_are_not_mutated(identifier, articulated, intents):
    outputs = make_outputs()
    original_body, original_outputs = articulated.copy(), deepcopy(outputs)
    repeated = map_body(articulated, outputs, identifier)
    for key in ('targets', 'dimmer', 'field_points', 'field_activity'):
        np.testing.assert_array_equal(repeated[key], intents[identifier][key])
    np.testing.assert_array_equal(articulated, original_body)
    assert outputs == original_outputs
    json.dumps(repeated['method'], allow_nan=False)


def test_fifteen_algorithms_are_not_exact_duplicate_raw_intents(intents):
    # Different hashes alone cannot establish different choreography. This
    # narrow invariant merely prevents exact duplicates entering the gallery.
    for a, b in combinations(IDS, 2):
        assert not np.array_equal(intents[a]['targets'], intents[b]['targets']), (a, b)


@pytest.mark.parametrize('identifier', IDS)
@pytest.mark.parametrize('pivot,axis,source_joint', [(16, [0, 0, 1], 20), (12, [0, 0, 1], 15)])
def test_limb_and_head_gestures_survive_common_mechanics(identifier, pivot, axis, source_joint):
    outputs = make_outputs()
    original, changed = _body(), _body()
    _rotate_subtree(changed, pivot, .65 * _pulse(), axis)
    before = map_body(original, outputs, identifier)
    after = map_body(changed, outputs, identifier)
    # Both perturbations preserve every skeleton edge length.
    parent = np.asarray(PARENTS[1:])
    np.testing.assert_allclose(np.linalg.norm(changed[:, 1:] - changed[:, parent], axis=-1),
                               np.linalg.norm(original[:, 1:] - original[:, parent], axis=-1), atol=1e-14)
    _, _, emitted_before, _ = _compile(before, outputs)
    _, _, emitted_after, _ = _compile(after, outputs)
    assigned = [i for i, item in enumerate(outputs)
                if item['kind'] != 'bulb' and item['role_index'] == source_joint]
    assert assigned
    assert any(source_joint in after['attribution'][i] for i in assigned)
    # Direction changes, rather than dimmer or RGB changes, demonstrate motion.
    delta = np.linalg.norm(emitted_after['directions'][:, assigned]
                           - emitted_before['directions'][:, assigned], axis=-1)
    assert np.max(delta) > 1e-4, (identifier, source_joint)
    angles = emitted_after['angles_deg'][:, :13]
    assert np.max(np.linalg.norm(np.diff(angles, axis=0) * 40, axis=-1)) <= 90 + 1e-8
    assert np.max(np.linalg.norm(np.diff(angles, n=2, axis=0) * 40**2, axis=-1)) <= 360 + 1e-8


def test_elbow_nullspace_of_legacy_tip_mapping_is_visible_after_dmx():
    original, changed = _body(), _body()
    changed[:, 18, 2] += .20 * _pulse()
    # Deliberately non-rigid counterexample isolates endpoint-pooling loss.
    legacy_outputs = build_outputs(beams=1, lasers=0, bulbs=0)
    np.testing.assert_array_equal(raw_transfer(original, legacy_outputs)['targets'],
                                  raw_transfer(changed, legacy_outputs)['targets'])
    outputs = make_outputs(beams=1, lasers=0, bulbs=0)
    a = _compile(map_body(original, outputs, '02'), outputs)[2]['directions']
    b = _compile(map_body(changed, outputs, '02'), outputs)[2]['directions']
    assert np.max(np.linalg.norm(a - b, axis=-1)) > .01


def test_support_frame_uses_physical_foot_velocity_but_has_no_running_state():
    pose = _body(3)
    previous = pose.copy()
    previous[0, 10, 0] -= .03
    outputs = make_outputs()
    stable = map_body(pose, outputs, '05')
    approached = map_body(previous, outputs, '05')
    # The same current pose can have different support if the foot just moved.
    assert np.max(np.abs(stable['field_points'][1] - approached['field_points'][1])) > .001
    # One more identical sample removes the derivative, with no filter tail.
    np.testing.assert_array_equal(stable['field_points'][2], approached['field_points'][2])
    # Other anatomical pose fields do not acquire the support history rule.
    for identifier in ('01', '02', '03', '04'):
        a, b = map_body(pose, outputs, identifier), map_body(previous, outputs, identifier)
        np.testing.assert_array_equal(a['field_points'][1:], b['field_points'][1:])


@pytest.mark.parametrize('identifier', IDS)
def test_only_bulbs_spill_into_multiple_universes_and_decode_last_fixture(identifier):
    outputs = make_outputs(beams=0, lasers=0, bulbs=130)
    intent = map_body(_body(3), outputs, identifier)
    dmx, patch, controls, _ = _compile(intent, outputs)
    assert dmx.shape == (3, 2, 512)
    assert all(item['address'] + item['n_channels'] - 1 <= 512 for item in patch)
    np.testing.assert_array_equal(controls['directions'], 0)
    last = patch[-1]
    assert last['universe'] == 1
    changed = dmx.copy()
    changed[:, last['universe'], last['address'] - 1] = 0
    assert np.all(decode(changed, patch)['dimmer'][:, -1] == 0)
    assert np.all(controls['dimmer'][:, -1] > 0)


@pytest.mark.parametrize('invalid', [np.full((3, 22, 3), np.nan), np.zeros((1, 22, 3)),
                                   np.zeros((3, 21, 3)), np.full((3, 22, 3), np.inf)])
def test_invalid_body_is_rejected_without_fallback(invalid):
    with pytest.raises(ValueError):
        map_body(invalid, make_outputs(), '01')


@pytest.mark.parametrize('fps', [0, -1, np.nan, np.inf])
def test_invalid_fps_is_rejected(fps):
    with pytest.raises(ValueError):
        map_body(_body(3), make_outputs(), '01', fps=fps)


@pytest.mark.parametrize('count', [-1, 10001, True, 1.5])
def test_invalid_count_is_rejected(count):
    with pytest.raises(ValueError):
        make_outputs(beams=count)


def test_unknown_algorithm_and_invalid_installations_are_rejected():
    with pytest.raises((KeyError, ValueError), match='99'):
        map_body(_body(3), make_outputs(), '99')
    with pytest.raises(ValueError):
        make_outputs(beams=0, lasers=0, bulbs=0)
    with pytest.raises(ValueError):
        map_body(_body(3), [], '01')
    outputs = make_outputs()
    with pytest.raises(ValueError):
        map_body(_body(3), outputs + [deepcopy(outputs[0])], '01')
    outputs[0]['kind'] = 'unsupported'
    with pytest.raises(ValueError):
        map_body(_body(3), outputs, '01')


@pytest.mark.parametrize('key,value', [('targets', np.nan), ('dimmer', -0.01), ('dimmer', 1.01)])
def test_compiler_rejects_invalid_intent(key, value):
    outputs = make_outputs(beams=1, lasers=0, bulbs=0)
    intent = map_body(_body(3), outputs, '01')
    intent[key][0, 0] = value
    with pytest.raises(ValueError):
        _compile(intent, outputs)


@pytest.mark.parametrize('key,value', [('kind', 'unsupported'), ('pan_range_deg', 0.),
                                     ('tilt_range_deg', -1.), ('position', [np.nan, 6., -1.])])
def test_compiler_rejects_invalid_fixture_geometry_and_capability(key, value):
    outputs = make_outputs(beams=1, lasers=0, bulbs=0)
    intent = map_body(_body(3), outputs, '01')
    outputs[0][key] = value
    with pytest.raises(ValueError):
        _compile(intent, outputs)

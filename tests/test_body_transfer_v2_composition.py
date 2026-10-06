"""Source-dependence and numerical contracts for composition algorithms.

Run CPU-only with OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2.
These checks concern fields; the parent campaign checks emitted fixture DMX.
"""
import json

import numpy as np
import pytest

from lightsem.conductor.body_transfer_v2.common import REFERENCE, prepare_features
from lightsem.conductor.body_transfer_v2.composition import (
    FUNCTIONS,
    _canonical_eigenbasis,
    _harmonic_operator,
    _transport_operator,
    _virtual_laplacian,
)


def articulated_body(frames=96):
    """Piecewise gestures, holds, and opposite limb changes without RNG."""
    body = np.broadcast_to(REFERENCE, (frames, 22, 3)).copy()
    progress = np.linspace(0, 1, frames)
    reach = np.clip((progress - 0.12) / 0.23, 0, 1)
    retract = np.clip((progress - 0.68) / 0.23, 0, 1)
    body[:, [18, 20], 1] += (0.45 * (reach - retract))[:, None]
    body[:, 20, 2] += 0.30 * (reach - retract)
    body[:, [19, 21], 2] -= (0.20 * np.clip(progress / 0.5, 0, 1))[:, None]
    body[:, [4, 7, 10], 0] += (0.15 * np.clip((progress - 0.50) / 0.20, 0, 1))[:, None]
    body[:, [12, 15], 0] -= (0.07 * reach)[:, None]
    return body


@pytest.mark.parametrize("identifier", sorted(FUNCTIONS))
def test_frozen_noncanonical_body_stays_constant(identifier):
    pose = articulated_body()[42]
    features = prepare_features(np.repeat(pose[None], 128, axis=0))
    result = FUNCTIONS[identifier](features).validate(128)
    assert np.max(np.abs(result.points - result.points[:1])) < 1e-12
    if result.activity is not None:
        assert np.array_equal(result.activity, np.zeros((128, 22)))
    json.dumps(result.meta, allow_nan=False)


@pytest.mark.parametrize("identifier", sorted(FUNCTIONS))
def test_canonical_body_has_zero_deformation(identifier):
    features = prepare_features(np.repeat(REFERENCE[None], 8, axis=0))
    result = FUNCTIONS[identifier](features).validate(8)
    assert np.max(np.abs(result.points)) < 1e-12


@pytest.mark.parametrize("identifier", sorted(FUNCTIONS))
def test_changed_left_forearm_affects_attributed_outputs(identifier):
    body = np.repeat(REFERENCE[None], 24, axis=0)
    changed = body.copy()
    changed[:, [18, 20], 1] += 0.22
    changed[:, 20, 2] += 0.13
    original = FUNCTIONS[identifier](prepare_features(body)).validate(24)
    result = FUNCTIONS[identifier](prepare_features(changed)).validate(24)
    difference = np.max(np.abs(result.points - original.points), axis=(0, 2))
    assert np.max(difference) > 1e-3
    for sample in np.flatnonzero(difference > 1e-10):
        assert {18, 20}.intersection(result.attribution[sample])


def test_six_algorithms_are_not_duplicate_fields():
    features = prepare_features(articulated_body())
    fields = {identifier: function(features).validate(96).points
              for identifier, function in FUNCTIONS.items()}
    for identifier, field in fields.items():
        assert np.max(np.ptp(field, axis=0)) > 0.02, identifier
        for other, other_field in fields.items():
            if other > identifier:
                assert np.sqrt(np.mean((field - other_field)**2)) > 0.005, (identifier, other)


def test_field_pose_does_not_depend_on_sequence_length_or_neighbors():
    body = articulated_body(96)
    original_features = prepare_features(body)
    # All six position fields are instantaneous pose transforms. Laban's
    # optional activity intentionally uses body derivatives and is excluded.
    scrambled = body[np.array([75, 13, 51, 42, 5])]
    for function in FUNCTIONS.values():
        original = function(original_features).points
        reordered = function(prepare_features(scrambled)).points
        np.testing.assert_allclose(reordered, original[[75, 13, 51, 42, 5]], atol=1e-12)


def test_transport_balances_body_coverage_and_fixture_capacity():
    operator, plan = _transport_operator()
    np.testing.assert_allclose(plan.sum(0), 1 / 22, atol=1e-10, rtol=0)
    np.testing.assert_allclose(plan.sum(1), 1 / 22, atol=1e-10, rtol=0)
    np.testing.assert_allclose(operator.sum(1), 1, atol=1e-12, rtol=0)
    assert np.all(operator > 0)
    assert np.linalg.matrix_rank(operator, tol=1e-6) >= 12


def test_harmonic_anchors_are_exact_and_interior_is_spatially_harmonic():
    operator, _ = _harmonic_operator()
    anchors = [20, 21, 7, 8, 9, 15]
    interior = [j for j in range(22) if j not in anchors]
    np.testing.assert_allclose((_virtual_laplacian() @ operator)[interior], 0, atol=1e-12)
    assert operator[20, 20] == 1.45
    assert operator[20, 18] == -0.45
    assert operator[9, 9] == 1


def test_eigenbasis_handles_repeated_eigenvalues_deterministically():
    # Complete graph has an eigenvalue repeated 5 times.
    laplacian = 6 * np.eye(6) - np.ones((6, 6))
    values, basis = _canonical_eigenbasis(laplacian)
    np.testing.assert_allclose(basis.T @ basis, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(laplacian @ basis, basis * values, atol=1e-12)
    np.testing.assert_array_equal(basis, _canonical_eigenbasis(laplacian)[1])


@pytest.mark.parametrize("identifier", sorted(FUNCTIONS))
def test_full_trajectory_is_finite_and_chunk_independent(identifier):
    body = articulated_body(2400)
    result = FUNCTIONS[identifier](prepare_features(body)).validate(len(body))
    # Checks the bearing solver's 128-frame chunk boundary and ensures no
    # algorithm introduces a prefix-length-dependent position field.
    tail = FUNCTIONS[identifier](prepare_features(body[129:133])).points
    np.testing.assert_allclose(result.points[129:133], tail, atol=1e-12)
    json.dumps(result.meta, allow_nan=False)


def test_degenerate_collocated_body_is_finite():
    body = np.zeros((8, 22, 3))
    for function in FUNCTIONS.values():
        result = function(prepare_features(body)).validate(len(body))
        assert np.isfinite(result.points).all()

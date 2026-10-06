"""Synthetic coverage tests for the supplemental whole-body hip perturbation."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from lightsem.conductor.body_transfer import TransferConfig
from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.conductor.body_transfer_v2.skeleton import PARENTS
from scripts.motion.audit_dynamic_dance_hip_timing import audit_pose, yaw_pulse, TARGETS, _bind_prior_inputs
from scripts.motion.audit_dynamic_dance_counts import COUNT_CASES
from scripts.motion.run_body_portfolio import intervention, REGIONS


def test_existing_interventions_do_not_change_hip_positions():
    for region in REGIONS:
        body = intervention(REFERENCE, region, frames=101, degrees=12.)
        np.testing.assert_array_equal(body[:, [1, 2]], np.broadcast_to(REFERENCE[[1, 2]], (101, 2, 3)))


def test_rigid_yaw_moves_hips_with_exact_existing_clock_root_height_and_return():
    body, envelope = yaw_pulse(REFERENCE)
    assert body.shape == (181, 22, 3)
    assert np.flatnonzero(envelope)[0] == 21
    assert np.flatnonzero(envelope)[-1] == 79
    expected = np.zeros(181); u = np.linspace(0, 1, 21); ramp = u*u*(3-2*u)
    expected[20:41] = ramp; expected[40:61] = 1; expected[60:81] = ramp[::-1]
    np.testing.assert_array_equal(envelope, expected)
    np.testing.assert_array_equal(body[:, 0], np.broadcast_to(REFERENCE[0], (181, 3)))
    np.testing.assert_array_equal(body[..., 1], np.broadcast_to(REFERENCE[:, 1], (181, 22)))
    np.testing.assert_array_equal(body[envelope == 0], np.broadcast_to(REFERENCE, (int((envelope == 0).sum()), 22, 3)))
    lengths = np.linalg.norm(body[:, 1:]-body[:, np.asarray(PARENTS)[1:]], axis=-1)
    np.testing.assert_allclose(lengths, np.broadcast_to(lengths[:1], lengths.shape), atol=1e-14, rtol=0)
    assert np.all(np.linalg.norm(body[50, [1, 2]]-REFERENCE[[1, 2]], axis=-1) > .005)


def test_actual_117_rig_hip_roles_receive_source_requested_and_emitted_response():
    plan = {'counts': COUNT_CASES['63'], 'transfer': asdict(TransferConfig())}
    color = np.array([12, 24, 48], np.uint8)
    report = audit_pose(REFERENCE, [color, color], plan)
    assert report['synthetic_only'] and report['frozen_only_no_autonomous_motion']
    assert report['fixture_identities_exact']
    assert report['source_onset_frame'] == 21 and report['source_return_frame'] == 80
    assert report['vertical_coordinates_exact'] and report['root_exact'] and report['unperturbed_frames_exact']
    assert len(report['all_fixture_timings']) == 117
    for name, joint in TARGETS.items():
        target = report['target_fixtures'][name]
        assert target['identity']['role_index'] == joint and target['identity']['role_blend'] == 0
        assert target['direct_hip_position_m']['status'] == 'observed'
        for key in ('source_point_m', 'requested_angles_deg', 'constrained_emitted_angles_deg'):
            assert target['stages'][key]['status'] == 'observed'
        assert target['coverage_status'] == 'observed_and_settled'
    assert len(report['broader_body_response']['constrained_emitted_angles_deg']) > 2
    json.dumps(report, allow_nan=False)


def test_degenerate_unexcited_hips_remain_unresolved_without_motion_fallback():
    pose = REFERENCE.copy(); pose[1] = pose[0]; pose[2] = pose[0]
    plan = {'counts': COUNT_CASES['63'], 'transfer': asdict(TransferConfig())}
    color = np.array([12, 24, 48], np.uint8)
    report = audit_pose(pose, [color, color], plan)
    for name in TARGETS:
        target = report['target_fixtures'][name]
        assert target['coverage_status'] == 'unqualified_unresolved_response'
        assert 'hip_position' in target['unresolved_stages']
        assert target['direct_hip_position_m']['first_response_s'] is None
        assert target['stages']['constrained_emitted_angles_deg']['lag_from_source_onset_s'] is None


@pytest.mark.parametrize('pose', [np.zeros((21, 3)), np.full((22, 3), np.nan)])
def test_invalid_source_pose_is_rejected(pose):
    with pytest.raises(ValueError):
        yaw_pulse(pose)


def test_changed_compiler_clock_cannot_use_the_declared_40_hz_timing():
    plan = {'counts': COUNT_CASES['63'], 'transfer': asdict(TransferConfig(fps=30.))}
    with pytest.raises(ValueError, match='compiler clock'):
        audit_pose(REFERENCE, [np.zeros(3, np.uint8)]*2, plan)


def test_addendum_rejects_changed_plan_and_protected_baseline_inputs():
    expected = {'PLAN.json': 'plan-sha', 'P2.npz': 'color-sha', 'body.npz': 'body-sha'}
    _bind_prior_inputs({'inputs': expected}, expected)
    for key in expected:
        changed = {**expected, key: 'changed-sha'}
        with pytest.raises(ValueError, match='differs from prior count audit'):
            _bind_prior_inputs({'inputs': changed}, expected)
    with pytest.raises(ValueError, match='differs from prior count audit'):
        _bind_prior_inputs({'inputs': {'body.npz': 'body-sha'}}, expected)

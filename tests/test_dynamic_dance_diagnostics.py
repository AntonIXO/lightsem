"""Frozen diagnostics must distinguish inactivity, jitter and conditioning."""
import json
import numpy as np
import pytest

from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.experiments.dynamic_dance.diagnostics import (
    GATE_PARAMETERS, body_dynamics, comparison_gates, duplicate_streams,
    dynamics_series, emission_diagnostics, emitted_response, event_alignment,
    gesture_events, motif_recurrence, phrase_contrasts, recurrence_alignment,
    selective_response, tempo_aligned_comparison, release_contrast, predictive_gain, conditioning_gain,
)


def body_fixture(seconds=10., fps=30):
    time = np.arange(round(seconds * fps) + 1) / fps
    return np.repeat(REFERENCE[None].astype(float), len(time), axis=0), time


def arm_motion(body, time):
    # An analytic synthetic intervention is test data, not a demo controller.
    body = body.copy()
    angle = .3 * np.sin(2 * np.pi * time / 3)
    vector = body[:, [18, 20]] - body[:, 16:17]
    cosine, sine = np.cos(angle)[:, None], np.sin(angle)[:, None]
    x, y = vector[..., 0].copy(), vector[..., 1].copy()
    vector[..., 0], vector[..., 1] = x * cosine - y * sine, x * sine + y * cosine
    body[:, [18, 20]] = body[:, 16:17] + vector
    return body


def test_static_body_is_not_collapse_or_choreographic_success():
    body, time = body_fixture()
    result = body_dynamics(body, time, floor_y=0.)
    assert result['exactly_constant_body']
    assert result['world_mean_speed_m_s']['p95'] < 1e-12
    assert result['holds']['world']['longest_s'] == pytest.approx(10.)
    assert 'collapse' not in result
    json.dumps(result, allow_nan=False)


def test_root_travel_is_separate_from_relative_articulation_and_slide_not_hidden():
    body, time = body_fixture()
    body[:, :, 0] += .2 * time[:, None]
    result = body_dynamics(body, time, floor_y=0.)
    assert result['root_speed_m_s']['median'] == pytest.approx(.2)
    assert result['relative_joint_speed_m_s']['p95'] < 1e-12
    assert result['contacts']['geometric_supported_slide_m_s']['median'] == pytest.approx(.2)
    assert result['geometry']['bone_drift_m']['max'] < 1e-12
    assert result['holds']['world']['longest_s'] == 0.


def test_jitter_does_not_pass_as_better_motion():
    baseline, time = body_fixture()
    candidate = arm_motion(baseline, time)
    jitter = candidate.copy()
    jitter[:, 20, 0] += .01 * np.where(np.arange(len(time)) % 2, 1., -1.)
    clean, noisy = body_dynamics(candidate, time), body_dynamics(jitter, time, reference_body=baseline)
    assert noisy['jerk_m_s3']['p95'] > clean['jerk_m_s3']['p95'] * 100
    gates = comparison_gates(noisy, clean)
    assert gates['gates']['jerk_nonregression']['status'] == 'fail'
    assert gates['gates']['bone_nonregression']['status'] == 'fail'
    assert gates['artistic_verdict'] == 'pending_user_review'


def test_missing_contacts_and_empty_references_stay_unqualified():
    body, time = body_fixture()
    report = body_dynamics(body, time, floor_y=-100.)
    assert report['contacts']['geometric_supported_slide_m_s']['p95'] is None
    assert comparison_gates(report, report)['gates']['supported_slide_nonregression']['status'] == 'unqualified'
    assert event_alignment([], [], start_s=0, end_s=10)['f1'] is None
    assert event_alignment([], [2., 3.], start_s=0, end_s=10)['f1'] == 0.


def test_accepts_dance_result_contract_and_reports_nonfinite_body():
    body, time = body_fixture()
    result = DanceResult(time, body, np.zeros((len(time), 4)))
    assert body_dynamics(result)['finite']
    body[4, 3, 1] = np.nan
    failed = body_dynamics(result)
    assert not failed['finite']
    assert failed['nonfinite_values'] == 1
    json.dumps(failed, allow_nan=False)


def test_event_precision_penalizes_jitter_and_uses_fixed_shift_null():
    events = np.array([1.1, 2.7, 4.3, 8.2, 11.6, 15.2, 18.3])
    correct = event_alignment(events + .03, events, start_s=0, end_s=20)
    busy = event_alignment(np.arange(.05, 20, .1), events, start_s=0, end_s=20)
    assert correct['f1'] == 1.
    assert correct['f1_minus_null_p95'] > .5
    assert busy['recall'] == 1.
    assert busy['precision'] < .05
    assert busy['f1'] < .1
    # Matching is one-to-one even when multiple gestures share one beat.
    many = event_alignment([1., 1.01, 1.02], [1.], start_s=0, end_s=2)
    assert many['matched_count'] == 1


def test_peak_proxy_and_phrase_trace_do_not_reward_rms_only_controller():
    time = np.arange(0., 20., .02)
    content_events = np.array([2.1, 5.8, 9.3, 14.6, 17.2])
    rms_events = np.array([1., 4., 8., 12., 18.5])
    def pulses(events):
        return np.exp(-((time[:, None] - events[None]) / .10) ** 2).sum(1)
    content = gesture_events(pulses(content_events), time)
    rms_only = gesture_events(pulses(rms_events), time)
    correct = event_alignment(content, content_events, start_s=0, end_s=20)
    wrong = event_alignment(rms_only, content_events, start_s=0, end_s=20)
    assert correct['f1'] > .99 and wrong['f1'] == 0.
    # Contrasts are descriptive, with no assertion that larger is preferable.
    phrase = phrase_contrasts(pulses(rms_events), time, [dict(start_s=0., end_s=10., label='A')])
    assert 'semantic_score' not in phrase


def test_regional_intervention_excludes_root_shift_and_measures_selectivity():
    body, time = body_fixture()
    changed = arm_motion(body, time)
    report = selective_response(body, changed, assigned_regions=['left_arm'])
    assert report['assigned_response_above_noise']
    assert report['unassigned_mean_rms_m'] == 0.
    translated = body + np.array([100., 20., -30.])
    report = selective_response(body, translated, assigned_regions=['left_arm'])
    assert not report['assigned_response_above_noise']


def test_tempo_alignment_reveals_reused_motion():
    body, time = body_fixture(seconds=12.)
    body = arm_motion(body, time)
    slower_time = time * 1.2
    report = tempo_aligned_comparison(body, time, np.arange(13.), body, slower_time, np.arange(13.) * 1.2)
    assert report['rms_difference_m'] < 1e-12
    assert report['centered_shape_correlation'] == pytest.approx(1.)
    assert tempo_aligned_comparison(body, time, [], body, time, [])['status'] == 'insufficient_beats'


def test_recurrence_matches_independent_audio_geometry_over_shifts():
    body, time = body_fixture(seconds=20.)
    rng = np.random.default_rng(451)
    knot_t = np.linspace(0., 20., 17)
    latent = np.interp(time, knot_t, rng.normal(size=len(knot_t)))
    body[:, 20, 0] += latent
    report = recurrence_alignment(body, time, latent[:, None], time)
    assert report['spearman'] > .999
    assert report['lift_over_null_p95'] > .4
    assert report['sampled_frames'] <= 128
    stationary, _ = body_fixture(seconds=20.)
    assert recurrence_alignment(stationary, time, latent[:, None], time)['spearman'] is None
    json.dumps(report, allow_nan=False)


def test_motif_metric_discloses_constant_pose_recurrence():
    body, time = body_fixture()
    report = motif_recurrence(body, time, [((0., 3.), (6., 9.))])
    assert report['pairs'][0]['both_stationary']
    assert report['pairs'][0]['shape_rms_m'] == 0.
    assert report['pairs'][0]['phase_velocity_correlation'] is None


def test_exact_and_near_duplicate_detection_inspects_full_stream():
    x = np.zeros((200, 4, 2))
    x[:, 1] = -0.
    x[:, 2] = .0001
    x[199, 3, 0] = 1.
    report = duplicate_streams(x, ['a', 'b', 'c', 'd'], near_rms_tol=.001)
    assert report['exact_groups'] == [['a', 'b']]
    assert report['unique_exact_streams'] == 3
    assert len(report['near_pairs']) == 3
    assert not any('d' in pair['names'] for pair in report['near_pairs'])


def test_emitted_motion_cannot_be_inferred_from_requested_motion():
    time = np.arange(200) / 40.
    requested = np.zeros((len(time), 2, 2))
    requested[:, :, 0] = time[:, None] * 10
    blocked = emission_diagnostics(requested, np.zeros_like(requested), time)
    assert blocked['requested_speed_deg_s']['median'] == pytest.approx(10.)
    assert blocked['emitted_speed_deg_s']['max'] == 0.
    assert blocked['velocity_rms_retention_ratio'] == 0.
    assert blocked['exactly_constant_emitted']
    saturated = emission_diagnostics(requested, requested, time, maximum_speed_deg_s=10.)
    assert saturated['speed_saturation_fraction'] == 1.  # actual fitter bounds 2D norm
    tick = 540. / 65535.
    assert not emitted_response(requested, requested + tick / 2, tick_deg=tick)['above_quantization_and_noise']
    assert emitted_response(requested, requested + tick * 2, tick_deg=tick)['above_quantization_and_noise']


def test_controller_own_labels_cannot_qualify_musical_result():
    body, time = body_fixture()
    report = body_dynamics(body, time)
    gates = comparison_gates(report, report, evidence={
        'musical_score': 1., 'musical_control_p95': .1, 'musical_reference_kind': 'controller_own_labels',
        'audio_response_m': .1, 'audio_repeat_noise_m': 0.,
    })
    assert gates['gates']['audio_sensitivity']['status'] == 'pass'
    assert gates['gates']['musical_control_lift']['status'] == 'unqualified'
    assert gates['gates']['emitted_effect']['status'] == 'unqualified'
    assert gates['status'] == 'unqualified_missing_evidence'
    assert GATE_PARAMETERS['musical_metric_minimum_lift'] == .02
    json.dumps(gates, allow_nan=False)


def test_fixed_mapper_static_and_limb_response_reaches_actual_emitted_bytes():
    from lightsem.conductor.body_transfer import compile_intent, decode
    from lightsem.conductor.body_transfer_v2 import make_outputs, map_body
    body, time = body_fixture(seconds=2., fps=40)
    outputs = make_outputs(beams=2, lasers=0, bulbs=3)
    color = np.full((len(body), 3), 100, np.uint8)
    def emit(world, rig):
        intent = map_body(world, rig, '01')
        dmx, patch, _ = compile_intent(intent, rig, color, color)
        return dmx, decode(dmx, patch), patch
    dmx, original, patch = emit(body, outputs)
    np.testing.assert_array_equal(dmx, np.broadcast_to(dmx[:1], dmx.shape))
    _, changed, _ = emit(arm_motion(body, time), outputs)
    assert np.max(np.abs(changed['angles_deg'] - original['angles_deg'])) > .01
    # Existing fixture identities preserve timing when extra outputs are added.
    expanded = make_outputs(beams=4, lasers=2, bulbs=7)
    _, larger, larger_patch = emit(arm_motion(body, time), expanded)
    by_name = {p['name']: i for i, p in enumerate(larger_patch)}
    indices = [by_name[p['name']] for p in patch]
    np.testing.assert_array_equal(changed['angles_deg'], larger['angles_deg'][:, indices])


@pytest.mark.parametrize('kind', ['time', 'contacts', 'shape'])
def test_invalid_inputs_do_not_silently_pass(kind):
    body, time = body_fixture()
    with pytest.raises(ValueError):
        if kind == 'time':
            body_dynamics(body, time * 0)
        elif kind == 'contacts':
            body_dynamics(body, time, contacts=np.zeros((len(time), 2)))
        else:
            body_dynamics(body[:, :21], time)


def test_release_contrast_beats_duration_matched_null_without_hold_quota():
    time = np.arange(0., 30.01, .02)
    release = [(3., 5.), (11., 13.), (24., 26.)]
    context = [(1., 3.), (9., 11.), (22., 24.)]
    trace = np.ones(len(time))
    for start, end in release:
        trace[(time >= start) & (time < end)] = .2
    report = release_contrast(trace, time, release, context)
    assert report['contrast'] > .65
    assert report['lift_over_null_p95'] > .3
    constant = release_contrast(np.ones(len(time)), time, release, context)
    assert constant['contrast'] == 0.
    assert release_contrast(trace, time, [], context)['contrast'] is None
    with pytest.raises(ValueError):
        release_contrast(trace, time, release, release)


def test_predictive_gain_honors_short_record_masks_and_zero_baseline():
    target = np.ones((4, 2))
    prediction = target + .1
    comparator = target + 1.
    target[-1] = np.nan
    mask = np.array([True, True, True, False])
    report = predictive_gain(target, prediction, comparator, mask=mask)
    assert report['relative_mse_gain'] == pytest.approx(.99)
    assert report['valid_values'] == 6
    assert predictive_gain(np.zeros(4), np.ones(4), np.zeros(4))['relative_mse_gain'] is None


def test_conditioning_gain_requires_both_donor_and_temporal_sensitivity():
    report = conditioning_gain([.5, .5], [1., 1.], [.5, .5])
    assert report['relative_donor_gain'] == .5
    assert report['minimum_relative_gain'] == 0.
    assert conditioning_gain([0.], [0.], [0.])['minimum_relative_gain'] is None


def test_emitted_saturation_uses_actual_vector_norm_not_axes():
    t = np.arange(100) / 40.
    angles = np.stack((3*t, 4*t), -1)[:, None]
    result = emission_diagnostics(angles, angles, t, maximum_speed_deg_s=5.)
    assert result['emitted_speed_axis_max_deg_s'] == pytest.approx(4.)
    assert result['emitted_speed_deg_s']['median'] == pytest.approx(5.)
    assert result['speed_saturation_fraction'] == 1.


def test_release_contrast_accepts_end_exclusive_sample_support():
    t = np.arange(0., 20., .1)
    trace = np.where(t >= 12., .2, 1.)
    result = release_contrast(trace, t, [(12., 20.)], [(4., 12.)])
    assert result['contrast'] > .6

"""Synthetic musical effects, not activity hashes, must survive transfer."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.conductor.body_transfer_v2.installation import make_outputs
from lightsem.experiments.dynamic_dance.campaign import read, sha, write_new
from lightsem.experiments.dynamic_dance.contracts import DanceResult, MusicTimeline
from lightsem.experiments.dynamic_dance.diagnostics import recurrence_alignment
from lightsem.experiments.dynamic_dance.transfer_audit import (
    MUSICAL_PROTOCOL, AUDIT_PARAMETERS, audit_pair, feature_recurrence,
    fixture_role_map, motor_event_alignment, regional_motor_selectivity,
)


def pulses(time, events):
    return 2. + 7. * np.exp(-((time[:, None]-np.asarray(events)[None])/.10)**2).sum(1)


def moving_angles(time, events):
    speed = pulses(time, events)
    pan = np.r_[0., np.cumsum(speed[:-1]*np.diff(time))]
    return np.stack((pan, np.zeros_like(pan)), -1)[:, None]


def test_event_alignment_distinguishes_activity_from_correct_timing():
    time = np.arange(0., 20., .025)
    events = np.array([1.3, 3.7, 7.4, 11.8, 15.3, 18.1])
    requested = moving_angles(time, events)
    shifted = moving_angles(time, events+.4)
    constant_speed = np.stack((time*5., time*0.), -1)[:, None]
    correct = motor_event_alignment(requested, time, events, start_s=0., end_s=20.)['pooled']
    wrong = motor_event_alignment(shifted, time, events, start_s=0., end_s=20.)['pooled']
    busy_without_accents = motor_event_alignment(constant_speed, time, events, start_s=0., end_s=20.)['pooled']
    assert correct['f1'] == 1.
    assert correct['f1_minus_null_p95'] > .5
    assert wrong['f1'] == 0.
    assert busy_without_accents['gesture_count'] == 0
    assert busy_without_accents['f1'] == 0.
    assert np.max(constant_speed) > 90.  # large activity still fails intended timing


def test_busy_jitter_pays_event_precision_cost():
    time = np.arange(0., 20., .025)
    events = np.array([1.3, 3.7, 7.4, 11.8, 15.3, 18.1])
    busy = moving_angles(time, np.arange(.3, 19.8, .225))
    score = motor_event_alignment(busy, time, events, start_s=0., end_s=20.)['pooled']
    assert score['recall'] > .8
    assert score['precision'] < .12
    assert score['f1'] < .22


def test_generic_recurrence_matches_body_recipe_without_fake_skeleton():
    time = np.arange(601)/30
    rng = np.random.default_rng(64)
    knots = np.linspace(0., 20., 17)
    latent = np.interp(time, knots, rng.normal(size=len(knots)))
    body = np.repeat(REFERENCE[None], len(time), axis=0)
    body[:, 20, 0] += latent
    relative = body-body[:, :1]
    original = recurrence_alignment(body, time, latent[:, None], time)
    generic = feature_recurrence(relative, time, latent[:, None], time)
    assert generic['spearman'] == pytest.approx(original['spearman'])
    np.testing.assert_allclose(generic['shifted_null'], original['shifted_null'])
    assert generic['sampled_frames'] == 128
    assert generic['spearman'] > .999
    shuffled = feature_recurrence(np.roll(latent, 157)[:, None], time, latent[:, None], time)
    assert shuffled['spearman'] < generic['spearman']-.5
    constant = feature_recurrence(np.zeros((len(time), 4)), time, latent[:, None], time)
    assert constant['spearman'] is None and constant['lift_over_null_p95'] is None


def test_role_map_matches_frozen_d3_groups_and_discloses_pelvis_dependency():
    patch = make_outputs()
    rows = fixture_role_map(patch)
    support = [r['name'] for r in rows if r['primary_region'] == 'support']
    vocals = [r['name'] for r in rows if r['primary_region'] in ('head', 'left_arm')]
    assert support == ['beam_003', 'beam_004', 'laser_001', 'laser_002', 'laser_005']
    assert vocals == ['beam_001', 'beam_005', 'beam_007', 'laser_003']
    assert next(r for r in rows if r['name'] == 'laser_005')['routing_caveat']
    assert len([r for r in rows if r['moving_index'] is None]) == 50
    assert all(r['primary_region'] is None for r in rows if r['kind'] == 'bulb')


def tick_angles(patch, frames=100):
    ranges = np.array([[p['pan_range_deg'], p['tilt_range_deg']] for p in patch if p['kind'] != 'bulb'])
    return np.broadcast_to((32768/65535-.5)*ranges, (frames, len(ranges), 2)).copy(), ranges


def test_source_selectivity_distinguishes_assigned_global_and_complement_response():
    patch = make_outputs()
    original, ranges = tick_angles(patch)
    roles = fixture_role_map(patch)
    assigned_indices = [r['moving_index'] for r in roles if r['primary_region'] in ('head', 'left_arm')]
    assigned = original.copy()
    assigned[:, assigned_indices] += ranges[assigned_indices]/65535*100
    result = regional_motor_selectivity(original, assigned, patch, ['head', 'left_arm'], decoded=True)
    assert result['all_primary_roles']['assigned_minus_complement_deg'] > 0
    assert result['all_primary_roles']['complement_above_one_tick_count'] == 0
    assert result['all_primary_roles']['assigned_above_one_tick_count'] == 4
    global_change = original+ranges/65535*100
    shared = regional_motor_selectivity(original, global_change, patch, ['head', 'left_arm'], decoded=True)
    assert shared['all_primary_roles']['assigned_minus_complement_deg'] == pytest.approx(0., abs=1e-12)
    complement = global_change.copy()
    complement[:, assigned_indices] = original[:, assigned_indices]
    inverse = regional_motor_selectivity(original, complement, patch, ['head', 'left_arm'], decoded=True)
    assert inverse['all_primary_roles']['assigned_minus_complement_deg'] < 0


def test_subquantization_motion_and_missing_groups_do_not_qualify():
    patch = make_outputs(beams=1, lasers=0, bulbs=2)
    original, ranges = tick_angles(patch)
    tiny = regional_motor_selectivity(original, original+.2*ranges/65535, patch, ['head', 'left_arm'], decoded=True)
    assert tiny['per_fixture'][0]['max_abs_command_tick_difference'] == 0
    assert tiny['per_fixture'][0]['above_one_tick_rms'] is False
    assert tiny['all_primary_roles']['assigned_minus_complement_deg'] is None
    assert tiny['all_primary_roles']['missing_assigned_regions'] == ['head']
    bulbs = make_outputs(beams=0, lasers=0, bulbs=2)
    blank = regional_motor_selectivity(np.zeros((10, 0, 2)), np.zeros((10, 0, 2)), bulbs, ['support'], decoded=True)
    assert blank['all_primary_roles']['assigned_minus_complement_deg'] is None


def test_blends_unknown_roles_and_permutation_are_explicit():
    patch = make_outputs(beams=23, lasers=11, bulbs=0)
    patch.append({**patch[0], 'name': 'unknown', 'role_index': 99})
    original, ranges = tick_angles(patch)
    removed = original+ranges/65535*10
    removed[:, -1] += 100.  # unknown role must not enter complement
    first = regional_motor_selectivity(original, removed, patch, ['support'], decoded=True)
    assert first['all_primary_roles']['assigned_minus_complement_deg'] == pytest.approx(0., abs=1e-12)
    assert first['blended_fixture_names']
    assert first['unassigned_fixture_names'] == ['unknown']
    permutation = np.arange(len(patch))[::-1]
    second = regional_motor_selectivity(original[:, permutation], removed[:, permutation], [patch[i] for i in permutation], ['support'], decoded=True)
    assert second['all_primary_roles']['assigned_minus_complement_deg'] == pytest.approx(first['all_primary_roles']['assigned_minus_complement_deg'])
    assert second['unblended_only']['assigned_minus_complement_deg'] == pytest.approx(first['unblended_only']['assigned_minus_complement_deg'])


def synthetic_campaign(root):
    plan = {'tracks': [{'id': 'synthetic'}], 'review_start_s': 60., 'review_duration_s': 120.,
            'prefix_s': 180., 'counts': {'beams': 1, 'lasers': 0, 'bulbs': 1},
            'transfer': {'maximum_speed_deg_s': 90., 'maximum_acceleration_deg_s2': 360.}}
    write_new(root/'PLAN.json', plan)
    write_new(root/'records/SUPPLEMENTAL_DELIVERY_AUDITS_PROTOCOL.json', {
        'musical_transfer': MUSICAL_PROTOCOL, 'status': 'frozen_before_execution',
        'original_plan_unchanged': True, 'original_primaries_and_gates_unchanged': True,
        'development_scope': 'Synthetic post-outcome supplemental audit fixture'})
    t = np.arange(5400)/30
    audio = np.zeros((len(t), 1024), np.float32)
    audio[:, 0] = np.sin(t/7)+t/180
    timeline = MusicTimeline(t, audio, t.copy(), np.ones((len(t), 4, 4)), np.ones((len(t), 4)),
        np.arange(180.), np.arange(0., 180., 4.), [{'start': 0., 'end': 180., 'label': 'synthetic'}],
        np.ones((1, 1)), {'duration_s': 180.})
    path = root/'features/timelines/synthetic.npz'
    timeline.save(path)
    events = [65.3, 78.7, 90.4, 113.8, 135.3, 161.1]
    write_new(root/'records/audio_intervals/synthetic.json', {'music_sha256': sha(path), 'attack_times': events})
    return events


def save_synthetic_pair(root, method, events, *, loses_timing=False, color=100):
    t = np.arange(5400)/30
    world = np.repeat(REFERENCE[None], len(t), axis=0)
    bp = root/'bodies'/method/'synthetic/body.npz'
    DanceResult(t, world, np.zeros((len(t), 4)), metadata={}).save(bp)
    time = np.arange(7200)/40
    requested = moving_angles(time, events)*.08
    # Larger, finite isolated accents satisfy1deg/s prominence while conserving
    # moderate full-prefix angles; this is synthetic metric data, not a mapper.
    speed = 1.+pulses(time, events)
    pan = np.cumsum((speed-speed.mean())/40)
    requested[:, 0, 0] = pan
    emitted = requested.copy()
    if loses_timing:
        emitted[:, 0, 0] = np.linspace(-30., 30., len(time))
    patch = make_outputs(beams=1, lasers=0, bulbs=1)
    for p, address, width in zip(patch, [1, 9], [8, 4]):
        p.update(universe=0, address=address, n_channels=width)
    ticks = np.rint((.5+emitted[:, 0]/[540., 270.])*65535).astype(int)
    dmx = np.zeros((len(time), 1, 512), np.uint8)
    dmx[:, 0, 0] = ticks[:, 0] >> 8; dmx[:, 0, 1] = ticks[:, 0] & 255
    dmx[:, 0, 2] = ticks[:, 1] >> 8; dmx[:, 0, 3] = ticks[:, 1] & 255
    dmx[:, 0, 4] = 150; dmx[:, 0, 5:8] = color
    dmx[:, 0, 8] = 150; dmx[:, 0, 9:12] = color
    targets = np.zeros((len(time), 2, 3))
    angle = np.deg2rad(requested[:, 0, 0])
    targets[:, 0] = np.asarray(patch[0]['position']) + np.stack((3*np.sin(angle), -3*np.cos(angle), angle*0), -1)
    targets[:, 1] = patch[1]['position']
    folder = root/'timelines'/method/'synthetic'
    folder.mkdir(parents=True)
    np.savez_compressed(folder/'timeline.npz', time=time, body=np.repeat(REFERENCE[None], len(time), axis=0), targets=targets, universes=dmx)
    write_new(folder/'patch.json', patch)
    write_new(folder/'report.json', {'body_sha256': sha(bp), 'timeline_sha256': sha(folder/'timeline.npz'),
        'patch_sha256': sha(folder/'patch.json'), 'diagnostics': {}})
    return folder


def test_audit_reads_actual_dmx_and_writes_nothing_to_originals(tmp_path):
    events = synthetic_campaign(tmp_path)
    save_synthetic_pair(tmp_path, 'baseline', events)
    folder = save_synthetic_pair(tmp_path, 'D1', events, loses_timing=True)
    write_new(folder/'comparison.json', {'primary': {'diagnostic_id': 'old_primary', 'value': .123}})
    files = {str(p): sha(p) for p in tmp_path.rglob('*') if p.is_file()}
    report = audit_pair(tmp_path, 'D1', 'synthetic')
    assert report['status'] == 'supplemental_descriptive_only'
    candidate = report['conditions']['candidate']
    assert candidate['event_alignment']['requested']['pooled']['f1'] > .8
    assert candidate['event_alignment']['emitted']['pooled']['f1'] == 0
    assert candidate['motor_activity']['emitted_mean_speed_deg_s'] > 0
    assert report['control_superiority']['baseline']['event_alignment']['emitted'] < -.8
    assert report['scientific_verdict'] == 'not_assigned'
    assert report['original_primary_reference']['primary']['value'] == .123
    assert report['missing_conditions']
    assert {str(p): sha(p) for p in tmp_path.rglob('*') if p.is_file()} == files
    json.dumps(report, allow_nan=False)


def test_color_difference_cannot_change_emitted_motion_metrics(tmp_path):
    events = synthetic_campaign(tmp_path)
    save_synthetic_pair(tmp_path, 'baseline', events)
    save_synthetic_pair(tmp_path, 'D1', events, color=220)
    report = audit_pair(tmp_path, 'D1', 'synthetic')
    assert report['status'] == 'invalid_fixed_P2_invariant'
    assert not report['P2_all_conditions_exact']
    assert report['control_superiority']['baseline']['event_alignment']['emitted'] == 0
    assert report['control_superiority']['baseline']['acoustic_recurrence']['emitted'] == 0


def test_frozen_parameter_drift_is_rejected_before_reading_candidates(tmp_path):
    synthetic_campaign(tmp_path)
    path = tmp_path/'records/SUPPLEMENTAL_DELIVERY_AUDITS_PROTOCOL.json'
    changed = read(path)
    changed['musical_transfer']['event_speed_prominence_deg_s'] = .5
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='differ from frozen'):
        audit_pair(tmp_path, 'D1', 'synthetic')


def test_cli_writes_exclusive_new_audit_and_refuses_original_paths(tmp_path):
    from scripts.motion.audit_dynamic_dance_transfer import main
    events = synthetic_campaign(tmp_path)
    save_synthetic_pair(tmp_path, 'baseline', events)
    save_synthetic_pair(tmp_path, 'D1', events)
    output = tmp_path/'reports/supplemental/D1.json'
    args = ['--root', str(tmp_path), '--method', 'D1', '--song', 'synthetic', '--output', str(output)]
    main(args)
    assert output.exists()
    assert read(str(output)+'.sha256.json')['sha256'] == sha(output)
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(SystemExit):
        main([*args[:-1], str(tmp_path/'timelines/D1/synthetic/new.json')])
    assert AUDIT_PARAMETERS['event_speed_prominence_deg_s'] == 1.


def test_cross_method_provenance_allows_distinct_checkpoints_with_same_inputs():
    from types import SimpleNamespace
    from lightsem.experiments.dynamic_dance.transfer_audit import _comparison_inputs
    inputs = {'arm': 'D8', 'seed': 0, 'draw_seed': 7001, 'feature_schema_sha256': 'schema',
              'timeline_sha256': 'same_music', 'base92_sha256': 'same_base92', 'checkpoint_sha256': 'D8_weights'}
    candidate = {'body': SimpleNamespace(metadata=inputs)}
    other = {'body': SimpleNamespace(metadata={**inputs, 'arm': 'D7', 'checkpoint_sha256': 'D7_weights'})}
    assert _comparison_inputs(candidate, other)['status'] == 'pass'
    other['body'].metadata['draw_seed'] = 9999
    assert _comparison_inputs(candidate, other)['status'] == 'unqualified'


def test_learned_planner_vs_d2_binds_accepted_source_and_audio():
    from types import SimpleNamespace
    from lightsem.experiments.dynamic_dance.transfer_audit import _comparison_inputs
    candidate = {'body': SimpleNamespace(metadata={'arm': 'D6', 'source_body_sha256': 'body', 'timeline_sha256': 'music'})}
    other = {'body': SimpleNamespace(metadata={'campaign_inputs': {'source_body_sha256': 'body', 'music_npz_sha256': 'music'}})}
    assert _comparison_inputs(candidate, other)['status'] == 'pass'
    other['body'].metadata['campaign_inputs']['source_body_sha256'] = 'wrong_body'
    assert _comparison_inputs(candidate, other)['status'] == 'unqualified'

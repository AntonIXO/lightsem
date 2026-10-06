"""Synthetic artifact integration only: no campaign output or TEST data access."""
import json
from pathlib import Path

import numpy as np
import pytest

from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.experiments.dynamic_dance.campaign import read, sha, write_new
from lightsem.experiments.dynamic_dance.contracts import DanceResult, MusicTimeline
from lightsem.experiments.dynamic_dance.evaluation import (
    freeze_release_intervals, score_pair, summarize, _learning, _primary, EXPECTED_CONTROLS,
)


def fixture_campaign(root, songs=('synthetic_song',)):
    root = Path(root)
    root.mkdir(exist_ok=True)
    plan = {'tracks': [{'id': s} for s in songs], 'review_start_s': 0., 'review_duration_s': 20.,
            'prefix_s': 20., 'counts': {'beams': 1, 'lasers': 0, 'bulbs': 1},
            'transfer': {'maximum_speed_deg_s': 90., 'maximum_acceleration_deg_s2': 360.}}
    write_new(root / 'PLAN.json', plan)
    for song in songs:
        t = np.arange(601) / 30
        muq = np.zeros((len(t), 1024), np.float32)
        muq[:, 0] = np.sin(t / 2) + t / 20
        sources = np.ones((len(t), 4, 4), np.float32) * .1
        sources[:, :, 0] += .03 * np.sin(t[:, None])
        mixture = np.ones((len(t), 4), np.float32) * .1
        timeline = MusicTimeline(t, muq, t.copy(), sources, mixture, np.arange(21.),
            np.arange(0., 21., 4.), [{'start': 0., 'end': 14., 'label': 'A'}, {'start': 14., 'end': 20., 'label': 'B'}],
            np.eye(2), {'scope': 'synthetic test fixture', 'duration_s': 20.,
                       'input_sha256': 'a' * 64, 'separator': {'checkpoint_sha256': 'b' * 64}})
        path = root / 'features/timelines' / (song + '.npz')
        timeline.save(path)
        write_new(root / 'records/audio_intervals' / (song + '.json'),
                  {'music_sha256': sha(path), 'attack_times': [1.3, 3.7, 7.4, 11.8, 15.3, 18.1], 'section_boundaries': [14.]})
    return plan


def save_pair(root, identifier, song='synthetic_song', *, body_delta=0., color=100, metadata_extra=None):
    root = Path(root)
    t = np.arange(601) / 30
    world = np.repeat(REFERENCE[None].astype(float), len(t), axis=0)
    world[:, 20, 1] += .15 * np.sin(t / 2) + body_delta
    metadata = {'campaign_inputs': {'source_body_sha256': 'same', 'method': identifier.split('__')[0],
                                    'music_npz_sha256': identifier}}
    metadata.update(metadata_extra or {})
    body = DanceResult(t, world, np.zeros((len(t), 4)), metadata=metadata)
    bp = root / 'bodies' / identifier / song / 'body.npz'
    body.save(bp)
    time = np.arange(801) / 40
    body40 = np.stack([np.interp(time, t, world[:, j, c]) for j in range(22) for c in range(3)], -1).reshape(-1, 22, 3)
    patch = [{'name': 'beam_001', 'kind': 'beam', 'position': [0., 6., 0.], 'pan_range_deg': 540.,
              'tilt_range_deg': 270., 'universe': 0, 'address': 1, 'n_channels': 8},
             {'name': 'bulb_001', 'kind': 'bulb', 'position': [-1., 5., 0.],
              'universe': 0, 'address': 9, 'n_channels': 4}]
    pan = 8 * np.sin(time / 2)
    ticks = np.rint((.5 + pan / 540) * 65535).astype(int)
    dmx = np.zeros((len(time), 1, 512), np.uint8)
    dmx[:, 0, 0] = ticks >> 8
    dmx[:, 0, 1] = ticks & 255
    dmx[:, 0, 2] = 128
    dmx[:, 0, 4] = 150
    dmx[:, 0, 5:8] = color
    dmx[:, 0, 8] = 150
    dmx[:, 0, 9:12] = color
    targets = np.zeros((len(time), 2, 3))
    targets[:, 0, 0] = 3 * np.sin(np.deg2rad(pan))
    targets[:, 0, 1] = 6 - 3 * np.cos(np.deg2rad(pan))
    targets[:, 1] = [-1., 5., 0.]
    folder = root / 'timelines' / identifier / song
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'timeline.npz', time=time, body=body40, targets=targets, universes=dmx)
    write_new(folder / 'patch.json', patch)
    write_new(folder / 'report.json', {'body_sha256': sha(bp), 'timeline_sha256': sha(folder / 'timeline.npz'),
        'patch_sha256': sha(folder / 'patch.json'), 'diagnostics': {'color': {'measured': True}, 'brightness': {'measured': True}},
        'hard_checks': {'source': 'synthetic fixture'}})


def test_freeze_release_reads_only_audio_and_is_immutable(tmp_path):
    fixture_campaign(tmp_path)
    report = freeze_release_intervals(tmp_path, 'synthetic_song')
    assert report['release_intervals'] == [[10., 18.]]
    assert report['reference_intervals'] == [[2., 10.]]
    assert not report['independent_semantic_reference']
    assert not (tmp_path / 'bodies').exists()
    assert freeze_release_intervals(tmp_path, 'synthetic_song') == report


def test_score_actual_artifacts_and_missing_controls_are_not_zero(tmp_path):
    fixture_campaign(tmp_path)
    freeze_release_intervals(tmp_path, 'synthetic_song')
    save_pair(tmp_path, 'baseline')
    save_pair(tmp_path, 'D1')
    before = sha(tmp_path / 'timelines/D1/synthetic_song/report.json')
    report = score_pair(tmp_path, 'D1', 'synthetic_song')
    assert report['status'] == 'diagnostic_pending_user_review'
    assert 'rms_only' in report['missing_controls']
    assert 'shifted_schedule' in report['missing_controls']
    assert report['baseline_difference']['exact_body_duplicate']
    assert report['baseline_difference']['exact_full_visual_payload_duplicate']
    assert report['installation']['P2_RGB_exact']
    assert report['installation']['actual_emitted_DMX']
    assert report['engineering_gates']['gates']['musical_control_lift']['status'] == 'unqualified'
    assert report['primary']['qualification'] == 'diagnostic_only_pending_user_review'
    assert sha(tmp_path / 'timelines/D1/synthetic_song/report.json') == before
    json.dumps(report, allow_nan=False)


def test_new_controls_create_report_revision_and_keep_prior_evidence(tmp_path):
    fixture_campaign(tmp_path)
    save_pair(tmp_path, 'baseline')
    save_pair(tmp_path, 'D2')
    first = score_pair(tmp_path, 'D2', 'synthetic_song')
    assert first['primary']['value'] is None  # no frozen release reference
    save_pair(tmp_path, 'D2__rms_only')
    second = score_pair(tmp_path, 'D2', 'synthetic_song')
    assert 'rms_only' in second['controls'] and 'rms_only' not in second['missing_controls']
    history = list((tmp_path / 'timelines/D2/synthetic_song/comparison_history').glob('*.json'))
    assert len(history) == 2
    assert any('rms_only' in read(p)['missing_controls'] for p in history)


def test_changed_mapped_bytes_fail_instead_of_using_stale_report(tmp_path):
    fixture_campaign(tmp_path)
    save_pair(tmp_path, 'baseline')
    save_pair(tmp_path, 'D1')
    path = tmp_path / 'timelines/D1/synthetic_song/timeline.npz'
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Mapped artifact changed'):
        score_pair(tmp_path, 'D1', 'synthetic_song')


def test_color_changes_are_not_motion_improvements(tmp_path):
    fixture_campaign(tmp_path)
    save_pair(tmp_path, 'baseline')
    save_pair(tmp_path, 'D5', color=200)
    report = score_pair(tmp_path, 'D5', 'synthetic_song')
    assert report['status'] == 'failed_fixed_P2_invariant'
    assert report['baseline_difference']['exact_emitted_motion_duplicate']
    assert not report['baseline_difference']['exact_full_visual_payload_duplicate']
    assert report['primary']['value'] is None


def test_summary_groups_duplicate_payloads_and_exposes_missing_methods(tmp_path):
    fixture_campaign(tmp_path, songs=('a', 'b'))
    for song in ('a', 'b'):
        save_pair(tmp_path, 'baseline', song)
        for method in ('D1', 'D2'):
            save_pair(tmp_path, method, song)
            score_pair(tmp_path, method, song)
    report = summarize(tmp_path)
    assert len(report['comparisons']) == 4
    assert len(report['missing_comparisons']) == 12
    assert len(report['duplicates']['full_visual_payload']) == 2
    assert report['duplicates']['full_visual_payload'][0]['representative'] == 'D1'
    assert len(report['cross_song']) == 2
    assert report['cross_song'][0]['beat_alignment']['rms_difference_m'] < 1e-12
    assert report['cross_song'][0]['similar_bpm_within_5pct']


def save_learning(root, arm, seed, rows):
    path = root / 'learning/evaluation' / arm / f'seed_{seed}.json'
    protocol_path = root / 'learning/protocol.json'
    if not protocol_path.exists():
        write_new(protocol_path, {'evaluation': {'windows': [{k: v for k, v in row.items() if k != 'conditions'} for row in rows]},
                                  'training': {'steps': 30000}, 'test_access': False})
        write_new(str(protocol_path) + '.sha256.json', {'sha256': sha(protocol_path)})
    run = root / 'learning/runs' / arm / f'seed_{seed}'
    run.mkdir(parents=True)
    checkpoint = run / 'final.pt'
    checkpoint.write_bytes(b'synthetic artifact, never loaded as model')
    training = {'identity': {'arm': arm, 'seed': seed, 'protocol_sha256': sha(protocol_path)},
                'steps': 30000, 'test_access': False, 'artifacts': {str(checkpoint): sha(checkpoint)}}
    write_new(run / 'report.json', training)
    write_new(str(run / 'report.json') + '.sha256.json', {'sha256': sha(run / 'report.json')})
    write_new(path, {'rows': rows, 'test_access': False, 'claim_scope': 'synthetic DEV',
                     'identity': {'protocol_sha256': sha(protocol_path), 'checkpoint_sha256': sha(checkpoint)}})
    write_new(str(path) + '.sha256.json', {'sha256': sha(path)})


def test_learning_uses_mse_gain_and_does_not_relabel_summary_loss(tmp_path):
    rows = [{'canonical_id': 'song1', 'conditions': {
        'correct': {'local_speed_rmse': .5}, 'rms_control': {'local_speed_rmse': 1.}}}]
    save_learning(tmp_path, 'planner', 0, rows)
    result = _learning(tmp_path, 'D6', {})
    assert result['display_primary'] == .75  # MSE gain, not 50% RMSE gain
    assert not result['both_seeds_available']
    flow = [{'canonical_id': 'song1', 'conditions': {
        'correct': {'summary_rmse': .5}, 'donor': {'summary_rmse': 1.}, 'shift': {'summary_rmse': 1.}}}]
    save_learning(tmp_path, 'D7', 0, flow)
    assert _learning(tmp_path, 'D7', {})['display_primary'] is None


def test_d7_requires_true_flow_loss_for_both_donor_and_shift(tmp_path):
    rows = [{'canonical_id': 's', 'conditions': {
        'correct': {'flow_loss': .6}, 'donor': {'flow_loss': 1.}, 'shift': {'flow_loss': .8}}}]
    for seed in (0, 1):
        save_learning(tmp_path, 'D7', seed, rows)
    result = _learning(tmp_path, 'D7', {})
    assert result['display_primary'] == pytest.approx(.25)
    assert result['both_seeds_available']


def test_d8_requires_both_d7_and_corrupted_plan_comparator():
    scores = {'recurrence': {'spearman': .7}}
    partial = _primary('D8', scores, {}, {}, None, {'D7': {'recurrence': {'spearman': .4}}})
    assert partial['value'] is None
    full = _primary('D8', scores, {'shuffled_plan': {'recurrence': {'spearman': .69}}},
                    {'shuffled_plan': {'fixed_input_provenance': {'status': 'pass'}}}, None,
                    {'D7': {'recurrence': {'spearman': .4}, 'generation_input_match': {'status': 'pass'}}})
    assert full['value'] == pytest.approx(.01)
    assert full['magnitude_gate'] == 'fail'


def test_missing_score_does_not_create_future_mapping_directory(tmp_path):
    fixture_campaign(tmp_path)
    report = score_pair(tmp_path, 'D1', 'synthetic_song')
    assert report['status'] == 'unqualified_missing_outputs'
    assert not (tmp_path / 'timelines/D1/synthetic_song').exists()
    assert (tmp_path / 'reports/missing_comparisons/D1_synthetic_song.json').exists()


def test_changed_release_reference_is_rejected(tmp_path):
    fixture_campaign(tmp_path)
    freeze_release_intervals(tmp_path, 'synthetic_song')
    path = tmp_path / 'records/audio_intervals/synthetic_song.release.json'
    report = read(path)
    report['release_intervals'] = [[2., 4.]]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='authenticated'):
        freeze_release_intervals(tmp_path, 'synthetic_song')


def test_summary_excludes_stale_changed_and_missing_inputs(tmp_path):
    fixture_campaign(tmp_path)
    save_pair(tmp_path, 'baseline')
    save_pair(tmp_path, 'D1')
    score_pair(tmp_path, 'D1', 'synthetic_song')
    path = tmp_path / 'timelines/D1/synthetic_song/timeline.npz'
    with path.open('ab') as stream:
        stream.write(b'changed')
    result = summarize(tmp_path)
    assert not result['comparisons']
    assert len(result['stale_comparisons']) == 1
    path.unlink()
    score_pair(tmp_path, 'D1', 'synthetic_song')
    assert not summarize(tmp_path)['comparisons']


def test_partial_development_rows_do_not_become_complete_evidence(tmp_path):
    rows = [{'canonical_id': 's', 'conditions': {
        'correct': {'flow_loss': .6}, 'donor': {'flow_loss': 1.}, 'shift': {'flow_loss': .8}}}]
    save_learning(tmp_path, 'D7', 0, rows)
    protocol = tmp_path / 'learning/protocol.json'
    data = read(protocol)
    data['evaluation']['windows'].append({'canonical_id': 'missing_song'})
    protocol.write_text(json.dumps(data))
    Path(str(protocol) + '.sha256.json').write_text(json.dumps({'sha256': sha(protocol)}))
    report = _learning(tmp_path, 'D7', {})
    assert report['display_primary'] is None
    assert not report['both_seeds_available']
    assert 'evaluation window coverage incomplete' in report['seeds'][0]['support']['reasons']


def test_method_postgate_failure_keeps_general_physical_pass_separate(tmp_path):
    fixture_campaign(tmp_path)
    save_pair(tmp_path, 'baseline')
    reason = 'actual emitted effort target loss worsened beyond allowance'
    save_pair(tmp_path, 'D5', metadata_extra={'postgate_passed': False, 'infeasibility': [reason],
              'transfer_before': {'emitted_effort_loss': .1}, 'transfer_after': {'emitted_effort_loss': .2}})
    report = score_pair(tmp_path, 'D5', 'synthetic_song')
    assert report['physical_engineering_status'] == 'pass'
    assert report['method_postgate']['status'] == 'fail'
    assert report['method_postgate']['passed'] is False
    assert report['method_infeasibility'] == [reason]
    assert report['failure_diagnostic']
    assert report['status'] == 'failure_diagnostic'
    assert report['method_postgate']['recorded_transfer_after']['emitted_effort_loss'] == .2
    assert report['candidate']['body'] == report['baseline']['body']


def test_all_methods_require_full_waveform_counterfactual_set():
    expected = {'gain', 'shift', 'mute_drums', 'mute_vocals', 'donor', 'reconstruction'}
    assert all(expected <= set(names) for names in EXPECTED_CONTROLS.values())
    assert {'rms_only', 'shifted_schedule'} <= set(EXPECTED_CONTROLS['D3'])


def test_d3_primary_uses_reconstruction_effect_not_original_residual():
    sensitivity = {}
    for name in ('mute_drums', 'mute_vocals'):
        sensitivity[name] = {'fixed_input_provenance': {'status': 'pass'}, 'P2_RGB_exact': True,
            'body': {'assigned_minus_unassigned_m': .1},
            'reconstruction_reference': {'fixed_input_provenance': {'status': 'pass'},
                'removal_reference_provenance': {'status': 'pass'},
                'body': {'assigned_minus_unassigned_m': .002}}}
    report = _primary('D3', {}, {}, sensitivity, None, {})
    assert report['value'] == .002
    assert report['minimum_increment'] == .005
    assert report['magnitude_gate'] == 'fail'
    assert report['original_reference_selectivity_m']['mute_drums'] == .1
    sensitivity['mute_drums']['reconstruction_reference']['removal_reference_provenance']['status'] = 'unqualified'
    assert _primary('D3', {}, {}, sensitivity, None, {})['value'] is None
    del sensitivity['mute_drums']['reconstruction_reference']
    assert _primary('D3', {}, {}, sensitivity, None, {})['value'] is None


def test_reconstruction_reference_saved_alongside_original_response(tmp_path):
    from copy import deepcopy
    fixture_campaign(tmp_path)
    save_pair(tmp_path, 'baseline')
    save_pair(tmp_path, 'D3')
    original = MusicTimeline.load(tmp_path / 'features/timelines/synthetic_song.npz')
    control_specs = {'reconstruction': {'kind': 'source_reconstruction'},
                     'mute_vocals': {'kind': 'source_removal', 'source': 'vocals'}}
    write_new(tmp_path / 'records/COUNTERFACTUAL_PROTOCOL_v2.json', {'counterfactuals': control_specs})
    for index, (name, intervention) in enumerate(control_specs.items(), start=1):
        music = deepcopy(original)
        music.muq[:, 0] += index * .01
        music.metadata['intervention'] = {**intervention, 'loudness_match': True,
            'loudness_before_lufs': -12., 'match_method': 'synthetic BS.1770 measurement'}
        path = tmp_path / 'features/timelines' / (name + '.npz')
        music.save(path)
        save_pair(tmp_path, 'D3__' + name, body_delta=.05 if name == 'reconstruction' else .06,
                  metadata_extra={'music_path': str(path), 'campaign_inputs': {'source_body_sha256': 'same',
                     'method': 'D3', 'music_npz_sha256': sha(path)}})
    report = score_pair(tmp_path, 'D3', 'synthetic_song')
    original_response = report['control_sensitivity']['mute_vocals']
    matched = original_response['reconstruction_reference']
    assert matched['removal_reference_provenance']['status'] == 'pass'
    assert matched['fixed_input_provenance']['status'] == 'pass'
    assert matched['body']['assigned_minus_unassigned_m'] < original_response['body']['assigned_minus_unassigned_m']
    assert original_response['body']['assigned_minus_unassigned_m'] > 0
    assert report['primary']['value'] is None  # no drums-removal evidence


def bind_floor_revision(root):
    receipt = root / 'records/D4_FLOOR_REPAIR.json'
    write_new(receipt, {'scope': 'synthetic implementation correction',
                        'change': 'stitch horizontal XZ only; unchanged selection and metrics'})
    path = root / 'records/VARIANT_BINDINGS.json'
    write_new(path, {'schema_version': 1, 'plan_sha256': sha(root / 'PLAN.json'),
        'selection_policy': 'implementation_correction_only', 'variants': {
            'D4': {'artifact_id': 'D4_floor_v2', 'reason': 'synthetic floor stitching correctness repair',
                   'references': {str(receipt): sha(receipt)}}}})
    seal = Path(str(path) + '.sha256.json')
    write_new(seal, {'sha256': sha(path)})
    return path, seal, receipt


def test_explicit_variant_reads_own_controls_and_preserves_original_reports(tmp_path):
    fixture_campaign(tmp_path)
    save_pair(tmp_path, 'baseline')
    save_pair(tmp_path, 'D4')
    save_pair(tmp_path, 'D4__rms_only', body_delta=.8)
    original = score_pair(tmp_path, 'D4', 'synthetic_song')
    assert original['artifact_id'] == 'D4'
    original_path = tmp_path / 'timelines/D4/synthetic_song/comparison.json'
    original_hash = sha(original_path)
    save_pair(tmp_path, 'D4_floor_v2', body_delta=.04)
    binding_files = bind_floor_revision(tmp_path)
    missing_control = score_pair(tmp_path, 'D4', 'synthetic_song')
    assert missing_control['method'] == 'D4'
    assert missing_control['artifact_id'] == 'D4_floor_v2'
    assert 'rms_only' not in missing_control['controls']
    assert any('D4_floor_v2__rms_only' in p for p in missing_control['missing_controls']['rms_only'])
    save_pair(tmp_path, 'D4_floor_v2__rms_only', body_delta=.05)
    repaired = score_pair(tmp_path, 'D4', 'synthetic_song')
    assert 'rms_only' in repaired['controls']
    assert repaired['primary']['diagnostic_id'] == original['primary']['diagnostic_id']
    assert repaired['primary']['minimum_increment'] == original['primary']['minimum_increment']
    assert sha(original_path) == original_hash
    assert (tmp_path / 'timelines/D4_floor_v2/synthetic_song/comparison.json').exists()
    assert all(repaired['inputs'][str(path)] == sha(path) for path in binding_files)


def test_summary_resolves_repaired_paths_and_body_comparison_with_logical_name(tmp_path):
    fixture_campaign(tmp_path, songs=('a', 'b'))
    old = {}
    for song in ('a', 'b'):
        save_pair(tmp_path, 'baseline', song)
        save_pair(tmp_path, 'D4', song)
        score_pair(tmp_path, 'D4', song)
        path = tmp_path / 'timelines/D4' / song / 'comparison.json'
        old[str(path)] = sha(path)
        save_pair(tmp_path, 'D4_floor_v2', song, body_delta=.1 if song == 'a' else .2)
    binding_files = bind_floor_revision(tmp_path)
    for song in ('a', 'b'):
        score_pair(tmp_path, 'D4', song)
    summary = summarize(tmp_path)
    assert len(summary['comparisons']) == 2
    assert all(r['method'] == 'D4' and r['artifact_id'] == 'D4_floor_v2' for r in summary['comparisons'])
    assert all('/D4_floor_v2/' in r['path'] for r in summary['comparisons'])
    cross = next(r for r in summary['cross_song'] if r['method'] == 'D4')
    assert cross['artifact_id'] == 'D4_floor_v2'
    assert cross['beat_alignment']['rms_difference_m'] > .005
    assert all(sha(path) == digest for path, digest in old.items())
    assert all(summary['inputs'][str(path)] == sha(path) for path in binding_files)


def test_variant_repair_receipt_drift_is_rejected_before_scoring(tmp_path):
    fixture_campaign(tmp_path)
    save_pair(tmp_path, 'baseline')
    save_pair(tmp_path, 'D4_floor_v2')
    _, _, receipt = bind_floor_revision(tmp_path)
    receipt.write_text('{}')
    with pytest.raises(ValueError, match='Changed variant correction evidence'):
        score_pair(tmp_path, 'D4', 'synthetic_song')
    assert not (tmp_path / 'timelines/D4_floor_v2/synthetic_song/comparison.json').exists()

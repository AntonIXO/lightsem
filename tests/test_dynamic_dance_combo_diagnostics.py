"""Synthetic authenticated artifacts: no campaign data, models, or GPU work."""
import json
from pathlib import Path

import numpy as np
import pytest

from lightsem.conductor.body_transfer_v2.common import REFERENCE
from lightsem.experiments.dynamic_dance.campaign import read, sha, write_new
from lightsem.experiments.dynamic_dance.contracts import DanceResult, MusicTimeline
from lightsem.experiments.dynamic_dance_combo import diagnostics as cd


def campaign(root, duration=20.03):
    root = Path(root)
    music_path = root / 'music/song.npz'
    track = {'id': 'song', 'duration_s': duration, 'body_frames': int(np.ceil(duration * 30)),
             'lighting_frames': int(np.ceil(duration * 40)), 'music_timeline': str(music_path),
             'audio_sha256': 'a' * 64}
    plan = {'tracks': [track], 'counts': {'beams': 1, 'lasers': 0, 'bulbs': 1},
            'transfer': {'maximum_speed_deg_s': 90., 'maximum_acceleration_deg_s2': 360.}}
    write_new(root / 'PLAN.json', plan)
    time = np.arange(track['body_frames']) / 30.
    muq = np.zeros((len(time), 1024), np.float32)
    muq[:, 0] = np.sin(time / 2) + time / duration
    sources = np.full((len(time), 4, 4), .1, np.float32)
    sources[:, :, 0] += .03 * np.sin(time[:, None])
    sources[:, 0, 1] += .2 * np.maximum(np.sin(time * 2), 0)
    mixture = np.full((len(time), 4), .1, np.float32)
    sections = [{'start': 0., 'end': duration * .7, 'label': 'A'},
                {'start': duration * .7, 'end': duration, 'label': 'B'}]
    MusicTimeline(time, muq, time.copy(), sources, mixture,
                  np.arange(0., duration, 1.), np.arange(0., duration, 4.), sections,
                  np.eye(2), {'duration_s': duration, 'input_sha256': 'a' * 64,
                             'recipe': {'observation_hz': 30, 'muq_native_hz': 30}}).save(music_path)
    return track


def save_pair(root, identifier, track, *, scale=1., dimmer=150, color=100,
              frame_count=None, generation=None):
    root = Path(root)
    time = np.arange(track['body_frames']) / 30.
    world = np.repeat(REFERENCE[None].astype(float), len(time), axis=0)
    # Synthetic diagnostic stimulus, never a demo-motion fallback.
    world[:, 20, 1] += scale * .05 * np.sin(time / 2)
    metadata = {'test': True}
    if generation is not None:
        metadata['generation'] = {**generation, 'analyzed_waveform_sha256': generation['base92_analyzed_waveform_sha256']}
    path = root / 'bodies' / identifier / track['id'] / 'body.npz'
    DanceResult(time, world, np.zeros((len(time), 4)), metadata=metadata).save(path)
    count = track['lighting_frames'] if frame_count is None else frame_count
    ticktime = np.arange(count) / 40.
    body40 = np.stack([np.interp(ticktime, time, world[:, j, k]) for j in range(22) for k in range(3)], -1).reshape(-1, 22, 3)
    patch = [{'name': 'beam_001', 'kind': 'beam', 'position': [-1., 6., 0.],
              'pan_range_deg': 540., 'tilt_range_deg': 270., 'universe': 0, 'address': 1, 'n_channels': 8},
             {'name': 'bulb_001', 'kind': 'bulb', 'position': [1., 5., 0.],
              'universe': 0, 'address': 9, 'n_channels': 4}]
    pan = scale * 8. * np.sin(ticktime / 2)
    ticks = np.rint((.5 + pan / 540) * 65535).astype(int)
    dmx = np.zeros((count, 1, 512), np.uint8)
    dmx[:, 0, 0], dmx[:, 0, 1] = ticks >> 8, ticks & 255
    dmx[:, 0, 2] = 128
    dmx[:, 0, 4] = dimmer
    dmx[:, 0, 5:8] = color
    dmx[:, 0, 8] = dimmer
    dmx[:, 0, 9:12] = color
    targets = np.zeros((count, 2, 3))
    targets[:, 0, 0] = -1 + 3 * np.sin(np.deg2rad(pan))
    targets[:, 0, 1] = 6 - 3 * np.cos(np.deg2rad(pan))
    targets[:, 1] = [1., 4., 0.]
    folder = root / 'timelines' / identifier / track['id']
    folder.mkdir(parents=True)
    np.savez_compressed(folder / 'timeline.npz', time=ticktime, body=body40,
        targets=targets, universes=dmx, dimmer=np.full((count, 2), dimmer / 255.),
        brightness_envelope=np.full(count, .12), palette_id=np.zeros(count, np.int32))
    write_new(folder / 'patch.json', patch)
    write_new(folder / 'report.json', {'body_sha256': sha(path),
        'timeline_sha256': sha(folder / 'timeline.npz'), 'patch_sha256': sha(folder / 'patch.json'),
        'inputs': {'plan_sha256': sha(root / 'PLAN.json')}})


def test_audio_references_frozen_before_any_bodies_and_authenticated(tmp_path):
    track = campaign(tmp_path)
    frozen = cd.freeze_release_intervals(tmp_path, track)
    assert frozen['interval_s'] == [0., track['duration_s']]
    assert frozen['release_intervals'] and frozen['attack_times']
    assert not frozen['independent_semantic_reference']
    save_pair(tmp_path, 'D7_reference', track)
    assert cd.freeze_release_intervals(tmp_path, 'song') == frozen
    path = tmp_path / 'records/audio_intervals/song.release.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='Changed authenticated'):
        cd.freeze_release_intervals(tmp_path, track)


def test_first_audio_selection_after_body_is_forbidden(tmp_path):
    track = campaign(tmp_path)
    save_pair(tmp_path, 'C3', track)
    with pytest.raises(RuntimeError, match='before viewing'):
        cd.freeze_release_intervals(tmp_path, track)


def test_identical_motion_is_zero_primary_not_a_success(tmp_path):
    track = campaign(tmp_path)
    cd.freeze_release_intervals(tmp_path, track)
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track)
    report = cd.compare(tmp_path, 'C3', track)
    assert report['primary']['value'] == 0.
    assert report['primary']['magnitude_gate'] == 'fail'
    assert report['status'] == 'failure_diagnostic'
    assert report['baseline_difference']['exact_full_visual_payload_duplicate']
    assert report['missing_controls']['C3_shuffled']
    assert report['missing_controls']['C3__donor']
    assert report['candidate']['recurrence']['emitted']['sampled_frames'] == 128
    assert report['candidate']['frames'] == {'body': 601, 'lighting': 802}
    assert report['candidate']['channels']['color']['palette_changes'] == 0
    assert report['candidate']['channels']['color']['palette_occupancy'][0]['fraction'] == 1.
    assert report['baseline_promotion'] is False
    assert cd.compare(tmp_path, 'C3', track) == report
    json.dumps(report, allow_nan=False)


def test_full_song_not_historical_180_second_crop(tmp_path):
    track = campaign(tmp_path, 181.03)
    selection = cd.freeze_release_intervals(tmp_path, track)
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track, scale=.8)
    report = cd.compare(tmp_path, 'C3', track)
    assert report['candidate']['frames'] == {'body': 5431, 'lighting': 7242}
    assert report['candidate']['interval_s'] == [0., 181.03]
    assert report['candidate']['body']['duration_s'] == 181.
    assert report['candidate']['recurrence']['body']['sampled_frames'] == 128
    assert selection['attack_times'][0] < 60.
    assert report['candidate']['event_alignment']['emitted']['pooled']['musical_event_count'] == len(selection['attack_times'])


def test_c4_release_is_actual_emitted_delta_against_c3(tmp_path):
    track = campaign(tmp_path)
    cd.freeze_release_intervals(tmp_path, track)
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track)
    save_pair(tmp_path, 'C4', track, scale=.5)
    report = cd.compare(tmp_path, 'C4', track)
    expected = (report['candidate']['release']['emitted']['contrast'] -
                report['baseline']['release']['emitted']['contrast'])
    assert report['primary']['value'] == expected
    assert report['primary']['baseline_id'] == 'C3'
    assert report['primary']['stage_deltas']['requested'] is not None
    assert 'C4_loudness' in report['missing_controls']
    assert 'C4_shifted' in report['missing_controls']


def test_c4_pins_displayed_d7_without_changing_scientific_comparator(tmp_path):
    track = campaign(tmp_path)
    cd.freeze_release_intervals(tmp_path, track)
    save_pair(tmp_path, 'D7_reference', track, scale=.25)
    save_pair(tmp_path, 'C3', track)
    save_pair(tmp_path, 'C4', track)
    report = cd.compare(tmp_path, 'C4', track)
    assert report['display_reference_id'] == 'D7_reference'
    assert report['scientific_comparator'] == report['comparator'] == 'C3'
    # These are the actual PLAN and displayed-pair input keys required by
    # render_dynamic_combo.load_comparison, obtained from real saved artifacts.
    required = [tmp_path / 'PLAN.json']
    for method in ('D7_reference', 'C4'):
        required.extend(tmp_path / 'timelines' / method / 'song' / name
                        for name in ('timeline.npz', 'patch.json', 'report.json'))
    assert all(report['inputs'][str(path)] == sha(path) for path in required)
    d7_body = tmp_path / 'bodies/D7_reference/song/body.npz'
    assert report['inputs'][str(d7_body)] == sha(d7_body)
    # C4 equals C3 and differs from displayed D7, so these numerical claims
    # would become false if the display reference replaced the comparator.
    assert report['primary']['value'] == 0.
    assert report['primary']['baseline_id'] == 'C3'
    assert report['baseline_difference']['exact_body_duplicate']
    assert report['baseline_difference']['exact_emitted_motion_duplicate']
    for name in cd.PHYSICAL_GATES:
        gate = report['engineering_gates']['gates'][name]
        if 'candidate' in gate and 'baseline' in gate:
            assert gate['candidate'] == gate['baseline']


def test_color_and_brightness_change_does_not_count_as_motion(tmp_path):
    track = campaign(tmp_path)
    cd.freeze_release_intervals(tmp_path, track)
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track, dimmer=70, color=140)
    report = cd.compare(tmp_path, 'C3', track)
    difference = report['baseline_difference']
    assert difference['exact_body_duplicate'] and difference['exact_emitted_motion_duplicate']
    assert difference['emitted_motor_rms_deg'] == 0.
    assert difference['brightness_rms'] > 0.
    assert not difference['RGB_exact']
    assert 'fixed_P2_RGB' in report['physical_failure_reasons']


def test_partial_track_is_rejected_even_if_mapping_hashes_match(tmp_path):
    track = campaign(tmp_path)
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track, frame_count=700)
    with pytest.raises(ValueError, match='full-song lighting clock'):
        cd.compare(tmp_path, 'C3', track)


def test_changed_mapping_receipt_is_rejected(tmp_path):
    track = campaign(tmp_path)
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track)
    folder = tmp_path / 'timelines/C3/song'
    with (folder / 'timeline.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Mapped artifact binding changed'):
        cd.compare(tmp_path, 'C3', track)


def test_new_evidence_does_not_overwrite_old_report(tmp_path):
    track = campaign(tmp_path)
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track)
    cd.compare(tmp_path, 'C3', track)
    save_pair(tmp_path, 'C3_shuffled', track)
    with pytest.raises(FileExistsError, match='Refuse to overwrite'):
        cd.compare(tmp_path, 'C3', track)


def test_source_response_checks_fixed_inputs_and_reconstruction(tmp_path):
    track = campaign(tmp_path)
    receipt = {'checkpoints': {'D7': 'd7', 'D8': 'd8'}, 'initial_noise_sha256': 'same',
               'draw_seed': 7001, 'display_seed': 0, 'residual_cap': .25,
               'integration': 'same', 'base92_analyzed_waveform_sha256': 'mixture'}
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track, generation=receipt)
    save_pair(tmp_path, 'C3__reconstruction', track, generation={**receipt, 'base92_analyzed_waveform_sha256': 'sum'})
    save_pair(tmp_path, 'C3__mute_drums', track, scale=.6, generation={**receipt, 'base92_analyzed_waveform_sha256': 'mute'})
    report = cd.compare(tmp_path, 'C3', track)
    row = report['source_response']['mute_drums']
    assert row['fixed_generation_inputs']['status'] == 'pass'
    assert row['fixed_generation_inputs']['analyzed_waveform_changed']
    assert row['global_brightness_envelope_held_fixed']
    assert row['root_relative_joint_rms_m'] > 0.
    assert row['requested_motor_rms_deg'] > 0.
    assert row['emitted_motor_rms_deg'] > 0.
    assert row['reconstruction_reference']['fixed_generation_inputs']['reference_id'] == 'C3__reconstruction'


def test_changed_seed_is_not_qualified_by_output_difference(tmp_path):
    track = campaign(tmp_path)
    receipt = {'checkpoints': {'D7': 'd7'}, 'initial_noise_sha256': 'one',
               'draw_seed': 7001, 'display_seed': 0, 'residual_cap': .25,
               'integration': 'same', 'base92_analyzed_waveform_sha256': 'mixture'}
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track, generation=receipt)
    save_pair(tmp_path, 'C3__shift', track, scale=.6, generation={**receipt, 'draw_seed': 7, 'initial_noise_sha256': 'two'})
    row = cd.compare(tmp_path, 'C3', track)['source_response']['shift']
    assert row['emitted_motor_rms_deg'] > 0.
    assert row['fixed_generation_inputs']['status'] == 'unqualified'
    assert row['fixed_generation_inputs']['checks']['draw_seed'] is False


def test_missing_release_reference_is_unmeasurable_not_zero(tmp_path):
    track = campaign(tmp_path)
    save_pair(tmp_path, 'D7_reference', track)
    save_pair(tmp_path, 'C3', track)
    save_pair(tmp_path, 'C4', track)
    result = cd.compare(tmp_path, 'C4', track)
    assert result['primary']['value'] is None
    assert result['primary']['magnitude_gate'] == 'unqualified'
    assert result['candidate']['release']['emitted'] is None

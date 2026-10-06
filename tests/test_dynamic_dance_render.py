"""CPU-only authentication, composition and media-command tests for dance review."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from lightsem.conductor.body_transfer import build_outputs
from scripts.rendering import render_dynamic_dance as render


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def campaign(tmp_path):
    n = 80
    body = np.zeros((n, 22, 3))
    body[:, :, 1] = np.linspace(.1, 1.8, 22)
    body[:, :, 0] = np.linspace(-.4, .4, 22)
    body[:, 20, 0] += np.arange(n) / 200
    patch = build_outputs()
    address = 1
    for fixture in patch:
        fixture.update(address=address, universe=0, n_channels=4 if fixture['kind'] == 'bulb' else 8)
        address += fixture['n_channels']
    dmx = np.zeros((n, 1, 512), dtype=np.uint8)
    for fixture in patch:
        start = fixture['address'] - 1
        if fixture['kind'] != 'bulb':
            dmx[:, 0, start:start + 4] = [128, 0, 128, 0]
            start += 4
        dmx[:, 0, start:start + 4] = [165, 50, 130, 220]
    audio = tmp_path / 'source.wav'
    audio.write_bytes(b'fixture audio identity only')
    reference = tmp_path / 'audio_reference.m4a'
    reference.write_bytes(b'fixture reference identity only')
    algorithm = {'id': 'D1', 'name': 'Event retiming'}
    track = {'id': 'shpongle', 'title': 'Shpongle', 'audio': str(audio), 'audio_reference': str(reference)}
    plan = {'prefix_s': 2, 'review_start_s': 0, 'review_duration_s': 1,
            'algorithms': [algorithm], 'tracks': [track],
            'files': {str(audio): render.sha(audio), str(reference): render.sha(reference)}}
    write_json(tmp_path / 'PLAN.json', plan)
    for method in ('baseline', 'D1'):
        out = tmp_path / 'timelines' / method / track['id']
        out.mkdir(parents=True)
        np.savez_compressed(out / 'timeline.npz', body=body, universes=dmx, time=np.arange(n) / 40,
                            source_time=np.arange(n) / 40, playback_rate=np.ones(n), intervention_time=np.array([.5]))
        write_json(out / 'patch.json', patch)
        write_json(out / 'report.json', {'status': 'diagnostic', 'timeline_sha256': render.sha(out / 'timeline.npz'),
                                         'patch_sha256': render.sha(out / 'patch.json')})
    music_path = tmp_path / 'features' / 'timelines' / 'shpongle.npz'
    music_path.parent.mkdir(parents=True)
    sources = np.ones((60, 4, 4)) * .2
    np.savez_compressed(music_path, time=np.arange(60) / 30, sources=sources, beats=np.array([0., .5, 1., 1.5]),
                        downbeats=np.array([0.]))
    write_json(music_path.with_suffix('.json'), {'sections': [{'start': 0., 'end': 2., 'label': 'estimated section'}]})
    return tmp_path, plan, algorithm, track


def test_comparison_authenticates_both_bodies_emitted_dmx_and_music(campaign):
    root, plan, algorithm, track = campaign
    baseline, candidate, music, pins = render.load_comparison(*campaign)
    assert candidate.body.shape == (80, 22, 3)
    assert candidate.indices.tolist()[:4] == [0, 1, 3, 4]
    assert len(candidate.patch) == 63
    assert pins['candidate']['timeline_sha256'] == pins['baseline']['timeline_sha256']
    assert music['sources'].shape == (60, 4, 4)
    out = root / 'timelines' / 'baseline' / track['id'] / 'timeline.npz'
    out.write_bytes(b'changed source')
    with pytest.raises(RuntimeError, match='pre-render numeric report'):
        render.load_comparison(*campaign)


def test_rig_count_and_patch_change_are_not_silently_accepted(campaign):
    root, plan, algorithm, track = campaign
    folder = root / 'timelines' / algorithm['id'] / track['id']
    patch = render.read(folder / 'patch.json')
    patch.pop()
    write_json(folder / 'patch.json', patch)
    report = render.read(folder / 'report.json')
    report['patch_sha256'] = render.sha(folder / 'patch.json')
    write_json(folder / 'report.json', report)
    with pytest.raises(ValueError, match='8 beams'):
        render.load_comparison(*campaign)


def test_measured_trace_derivatives_do_not_depend_on_audio_or_color(campaign):
    baseline, candidate, _, _ = render.load_comparison(*campaign)
    values = render.trace_values(candidate)
    np.testing.assert_allclose(values['upper'], .2 / len(render.UPPER), atol=1e-12)
    np.testing.assert_allclose(values['lower'], 0, atol=1e-12)
    np.testing.assert_allclose(values['emitted'], 0, atol=1e-12)
    controls = {key: np.zeros_like(value) if key in ('rgb', 'dimmer') else value
                for key, value in candidate.controls.items()}
    changed = render.trace_values(replace(candidate, controls=controls))
    for key in values:
        np.testing.assert_array_equal(values[key], changed[key])


def test_all_off_stage_preserves_visible_actual_source_skeletons(campaign):
    root, plan, algorithm, track = campaign
    baseline, candidate, music, _ = render.load_comparison(*campaign)
    strip = render.TimelineStrip(baseline, candidate, music, plan)
    dark = np.zeros((540, 960, 3), dtype=np.uint8)
    first = render.compose(dark, dark, baseline, candidate, 0, algorithm, track, strip)
    frozen = render.compose(dark, dark, baseline, candidate, 0, algorithm, track, strip)
    assert first.shape == (1080, 1920, 3)
    np.testing.assert_array_equal(first, frozen)
    body = candidate.body.copy()
    body[0, 20] += [.8, .2, 0]
    moved = render.compose(dark, dark, baseline, replace(candidate, body=body), 0, algorithm, track, strip)
    assert np.any(first[90:310, 960:] != moved[90:310, 960:])
    np.testing.assert_array_equal(first[90:310, :960], moved[90:310, :960])
    np.testing.assert_array_equal(first[400:800], moved[400:800])
    # The stage itself remains black; captions cannot count as illumination.
    assert not np.any(first[400:800, :958])
    assert not np.any(first[400:800, 962:])


def test_schematic_smoke_has_actual_and_separate_optical_controls(campaign):
    receipt = render.smoke_pair(*campaign, 'schematic')
    assert all(v['changed_pixels'] > 0 for v in receipt['fixture_class_visibility'].values())
    assert receipt['optical_controls_excluded_from_production']
    assert len(receipt['samples']) == 4
    assert any('all_lights_off_with_body' in path for path in receipt['files'])
    assert render.smoke_pair(*campaign, 'schematic') == receipt


def test_render_requires_matching_parent_visual_inspection(campaign):
    root, plan, algorithm, track = campaign
    receipt = render.smoke_pair(*campaign, 'schematic')
    with pytest.raises(RuntimeError, match='Parent must record'):
        render.render_pair(*campaign, 'schematic', 'libx264')
    out = root / 'smoke' / algorithm['id'] / track['id'] / 'schematic'
    write_json(out / 'INSPECTED.json', {'inspected': True, 'smoke_sha256': 'wrong'})
    with pytest.raises(RuntimeError, match='Parent must record'):
        render.render_pair(*campaign, 'schematic', 'libx264')


def test_source_clock_does_not_change_audio_playback_rate(campaign):
    root, plan, algorithm, track = campaign
    command = render.base.command_for(root / 'movie.mp4', track, plan, 'libx264')
    assert command[command.index('-r') + 1] == '30'
    assert command[command.index('-frames:v') + 1] == '30'
    assert 'libx264' in command
    assert not any('atempo' in token or 'nvenc' in token for token in command)
    assert command[command.index('-af') + 1] == 'atrim=duration=1.0,asetpts=PTS-STARTPTS'


def test_visual_duplicate_identity_excludes_zip_metadata(campaign):
    root, plan, algorithm, track = campaign
    expected = render.visual_fingerprint(root, {'id': 'baseline'}, track)
    assert render.visual_fingerprint(root, algorithm, track) == expected
    folder = root / 'timelines' / algorithm['id'] / track['id']
    with np.load(folder / 'timeline.npz') as data:
        payload = {key: data[key] for key in data.files}
    payload['playback_rate'] = np.ones(80) * .3
    np.savez(folder / 'timeline.npz', **payload)
    assert render.visual_fingerprint(root, algorithm, track) == expected
    payload['universes'] = payload['universes'].copy()
    payload['universes'][10, 0, 4] += 1
    np.savez(folder / 'timeline.npz', **payload)
    assert render.visual_fingerprint(root, algorithm, track) != expected


def comparison_fixture(campaign, **changes):
    root, plan, algorithm, track = campaign
    inputs = {str(root / 'PLAN.json'): render.sha(root / 'PLAN.json')}
    for method in ('D1', 'baseline'):
        for name in ('timeline.npz', 'patch.json', 'report.json'):
            path = root / 'timelines' / method / track['id'] / name
            inputs[str(path)] = render.sha(path)
    comparison = {'method': 'D1', 'song': track['id'], 'inputs': inputs,
                  'status': 'failure_diagnostic', 'physical_engineering_status': 'fail',
                  'physical_failure_reasons': ['jerk_nonregression'], 'missing_controls': {'donor': ['missing artifact']},
                  'primary': {'diagnostic_id': 'dd-event-alignment-lift', 'value': .04,
                              'minimum_increment': .02, 'magnitude_gate': 'pass', 'missing_evidence': []}, **changes}
    path = root / 'timelines' / 'D1' / track['id'] / 'comparison.json'
    write_json(path, comparison)
    return path, comparison


def test_comparison_claims_are_pinned_and_bound_to_actual_baseline(campaign):
    path, comparison = comparison_fixture(campaign)
    baseline, candidate, _, pins = render.load_comparison(*campaign)
    assert pins['candidate']['comparison_sha256'] == render.sha(path)
    summary = render.evidence_summary(candidate)
    assert summary['banner'] == 'FAILED ENGINEERING | jerk_nonregression'
    assert summary['primary']['magnitude_gate'] == 'pass'
    assert summary['missing_controls'] == ['donor']
    comparison['physical_failure_reasons'].append('bone_nonregression')
    write_json(path, comparison)
    _, _, _, changed_pins = render.load_comparison(*campaign)
    assert changed_pins != pins
    baseline_input = str(campaign[0] / 'timelines' / 'baseline' / campaign[3]['id'] / 'timeline.npz')
    comparison['inputs'][baseline_input] = 'different baseline'
    write_json(path, comparison)
    with pytest.raises(ValueError, match='displayed baseline/plan'):
        render.load_comparison(*campaign)


def test_proxy_failure_is_not_reported_as_physical_success(campaign):
    _, comparison = comparison_fixture(campaign, status='diagnostic_pending_user_review',
        physical_engineering_status='pass', physical_failure_reasons=[])
    comparison['primary'].update(value=-.03, magnitude_gate='fail')
    summary = render.evidence_summary(comparison)
    assert summary['banner'].startswith('PROXY MAGNITUDE FAILED')
    assert summary['physical_engineering_status'] == 'pass'


def test_alternating_frame_jitter_remains_visible_in_body_and_emitted_trace(campaign):
    _, candidate, _, _ = render.load_comparison(*campaign)
    body = candidate.body.copy()
    body[:, 20, 0] = .1 * (np.arange(len(body)) % 2)
    controls = {key: value.copy() for key, value in candidate.controls.items()}
    controls['angles_deg'][:, 0, 0] = np.arange(len(body)) % 2
    values = render.trace_values(replace(candidate, body=body, controls=controls))
    assert np.min(values['upper']) > .30
    assert np.min(values['emitted']) > 3.


def test_changed_body_does_not_make_identical_emitted_dmx_distinct(campaign):
    root, plan, algorithm, track = campaign
    lighting = render.lighting_fingerprint(root, algorithm, track)
    visual = render.visual_fingerprint(root, algorithm, track)
    path = root / 'timelines' / 'D1' / track['id'] / 'timeline.npz'
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload['body'] = payload['body'].copy()
    payload['body'][:, 20, 0] += .05
    np.savez(path, **payload)
    assert render.lighting_fingerprint(root, algorithm, track) == lighting
    assert render.visual_fingerprint(root, algorithm, track) != visual


def test_gallery_preserves_comparison_failure_and_full_evidence(campaign):
    path, comparison = comparison_fixture(campaign)
    manifest = render.review(campaign[0], campaign[1], 'dark')
    row = manifest['variants'][0]
    assert row['comparison_status'] == 'failure_diagnostic'
    assert row['physical_engineering_status'] == 'fail'
    assert row['failure_notes'] == ['jerk_nonregression']
    assert row['comparison'] == comparison
    assert row['comparison_sha256'] == render.sha(path)
    assert row['primary']['value'] == .04


def test_d4_fragment_boundary_is_never_differentiated_as_playback_speed(campaign):
    root, plan, algorithm, track = campaign
    folder = root / 'timelines' / 'D4' / track['id']
    shutil.copytree(root / 'timelines' / 'D1' / track['id'], folder)
    with np.load(folder / 'timeline.npz') as data:
        payload = {key: data[key] for key in data.files if key != 'playback_rate'}
    payload['source_time'] = payload['source_time'].copy()
    payload['source_time'][40:] -= .5
    def rewrite():
        np.savez(folder / 'timeline.npz', **payload)
        report = render.read(folder / 'report.json')
        report['timeline_sha256'] = render.sha(folder / 'timeline.npz')
        write_json(folder / 'report.json', report)
    rewrite()
    with pytest.raises(ValueError, match='fragment-local'):
        render.load_pair(root, plan, {'id': 'D4'}, track)
    payload['playback_rate'] = np.ones(80)
    rewrite()
    pair = render.load_pair(root, plan, {'id': 'D4'}, track)
    np.testing.assert_array_equal(pair.rate, np.ones(80))
    assert np.any(np.diff(pair.source_time) < 0)


def render_binding_fixture(campaign):
    root, plan, algorithm, track = campaign
    legacy_path = root / 'legacy_PLAN.json'
    write_json(legacy_path, plan)
    plan['files'].pop(track['audio_reference'])
    write_json(root / 'PLAN.json', plan)
    binding = {'schema_version': 1, 'plan_sha256': render.sha(root / 'PLAN.json'),
               'review_start_s': plan['review_start_s'], 'review_duration_s': plan['review_duration_s'],
               'tracks': {track['id']: {'audio': track['audio'], 'audio_sha256': render.sha(track['audio']),
                   'audio_reference': track['audio_reference'],
                   'audio_reference_sha256': render.sha(track['audio_reference']),
                   'verification': {'basis': 'legacy_plan_pin', 'evidence_path': str(legacy_path),
                                    'evidence_sha256': render.sha(legacy_path)}}}}
    path = root / 'RENDER_BINDINGS.json'
    write_json(path, binding)
    return path, binding


def test_unpinned_reference_cannot_fall_back_to_a_trusted_looking_path(campaign):
    root, plan, algorithm, track = campaign
    plan['files'].pop(track['audio_reference'])
    write_json(root / 'PLAN.json', plan)
    with pytest.raises(RuntimeError, match='authenticated RENDER_BINDINGS'):
        render.load_comparison(*campaign)


def test_render_binding_pins_reference_without_rewriting_scientific_plan(campaign):
    path, binding = render_binding_fixture(campaign)
    root, plan, algorithm, track = campaign
    plan_sha = render.sha(root / 'PLAN.json')
    _, _, _, pins = render.load_comparison(*campaign)
    assert pins['render_bindings_sha256'] == render.sha(path)
    assert pins['audio_reference_sha256'] == render.sha(track['audio_reference'])
    assert pins['audio_reference_binding_basis'] == 'legacy_plan_pin'
    assert render.sha(root / 'PLAN.json') == plan_sha
    assert track['audio_reference'] not in plan['files']


@pytest.mark.parametrize('field', ['plan_sha256', 'audio_reference_sha256', 'review_start_s', 'evidence_sha256'])
def test_stale_or_mismatched_render_binding_is_rejected(campaign, field):
    path, binding = render_binding_fixture(campaign)
    track = campaign[3]
    if field == 'audio_reference_sha256':
        binding['tracks'][track['id']][field] = 'stale'
    elif field == 'evidence_sha256':
        binding['tracks'][track['id']]['verification'][field] = 'stale'
    else:
        binding[field] = 'stale'
    write_json(path, binding)
    with pytest.raises(ValueError, match='binding|evidence'):
        render.load_comparison(*campaign)


def test_legacy_evidence_must_pin_the_exact_source_crop(campaign):
    path, binding = render_binding_fixture(campaign)
    track = campaign[3]
    proof = binding['tracks'][track['id']]['verification']
    legacy = render.read(proof['evidence_path'])
    legacy['review_start_s'] += 10
    write_json(Path(proof['evidence_path']), legacy)
    proof['evidence_sha256'] = render.sha(proof['evidence_path'])
    write_json(path, binding)
    with pytest.raises(ValueError, match='different source crop'):
        render.load_comparison(*campaign)


def test_media_audit_accepts_separate_verified_render_binding(campaign, monkeypatch):
    render_binding_fixture(campaign)
    root, plan, algorithm, track = campaign
    _, _, _, pins = render.load_comparison(*campaign)
    folder = root / 'media' / algorithm['id'] / track['id']
    folder.mkdir(parents=True)
    movie = folder / 'schematic.mp4'
    movie.write_bytes(b'media fixture; decoder is stubbed only in this authentication unit test')
    write_json(folder / 'schematic_render.json', {'inputs': pins, 'sha256': render.sha(movie),
                                                'codec': 'libx264', 'frames': 30})
    pcm = b'matching source-crop PCM supplied by decoder test fixture'
    monkeypatch.setattr(render.base, 'decoded_audio', lambda _: pcm)
    monkeypatch.setattr(render.base, 'audit_h264', lambda *_: {
        'pcm_sha256': hashlib.sha256(pcm).hexdigest(), 'decoded_frames': 30,
        'pts_uniform': True, 'full_software_AV_decode': True})
    receipt = render.audit_pair(root, plan, algorithm, track, 'schematic')
    assert receipt['audio_matches_source_crop'] is True
    assert receipt['inputs']['render_bindings_sha256'] == render.sha(root / 'RENDER_BINDINGS.json')


def test_large_campaign_is_admitted_without_unrequested_legacy_size_cap(tmp_path, monkeypatch):
    # Sparse test fixture has a 3 GiB logical size without allocating its data.
    with (tmp_path / 'feature_cache').open('wb') as stream:
        stream.truncate(3 * 2 ** 30)
    monkeypatch.setattr(render.campaign.shutil, 'disk_usage', lambda _: SimpleNamespace(free=21 * 2 ** 30))
    monkeypatch.setattr(render.base, 'guard', lambda *_: pytest.fail('Legacy 2 GiB cap must not run'))
    render.resource_guard(tmp_path, {'minimum_free_bytes': 20 * 2 ** 30})
    with pytest.raises(RuntimeError, match='Explicit campaign storage'):
        render.resource_guard(tmp_path, {'max_bytes': 2 * 2 ** 30})


def test_campaign_free_space_reserve_cannot_be_lowered_by_renderer(tmp_path, monkeypatch):
    monkeypatch.setattr(render.campaign.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 * 2 ** 30 - 1))
    with pytest.raises(RuntimeError, match='20 GiB free-space reserve'):
        render.resource_guard(tmp_path, {'minimum_free_bytes': 1})
    monkeypatch.setattr(render.campaign.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 * 2 ** 30))
    render.resource_guard(tmp_path, {})
    with pytest.raises(RuntimeError, match='Explicit campaign free-space'):
        render.resource_guard(tmp_path, {'minimum_free_bytes': 21 * 2 ** 30})


def variant_fixture(campaign):
    root, plan, _, track = campaign
    algorithm = {'id': 'D4', 'name': 'Recurring motifs'}
    plan['algorithms'] = [algorithm]
    write_json(root / 'PLAN.json', plan)
    original = root / 'timelines' / 'D4' / track['id']
    repaired = root / 'timelines' / 'D4_floor_v2' / track['id']
    for path in (original, repaired):
        shutil.copytree(root / 'timelines' / 'D1' / track['id'], path)
    with np.load(repaired / 'timeline.npz') as data:
        payload = {key: data[key] for key in data.files}
    payload['body'] = payload['body'].copy()
    payload['body'][:, :, 1] -= .1
    np.savez(repaired / 'timeline.npz', **payload)
    report = render.read(repaired / 'report.json')
    report['timeline_sha256'] = render.sha(repaired / 'timeline.npz')
    write_json(repaired / 'report.json', report)
    proof = root / 'records' / 'repairs' / 'D4_floor.json'
    write_json(proof, {'purpose': 'isolated renderer integration test fixture', 'reason': 'XZ alignment preserves floor'})
    bindings = {'schema_version': 1, 'plan_sha256': render.sha(root / 'PLAN.json'),
                'selection_policy': 'implementation_correction_only', 'variants': {
                    'D4': {'artifact_id': 'D4_floor_v2', 'reason': 'XZ alignment preserves floor',
                           'references': {str(proof): render.sha(proof)}}}}
    binding_path = root / 'records' / 'VARIANT_BINDINGS.json'
    write_json(binding_path, bindings)
    write_json(binding_path.with_name(binding_path.name + '.sha256.json'), {'sha256': render.sha(binding_path)})
    inputs = {str(root / 'PLAN.json'): render.sha(root / 'PLAN.json'), **render.campaign.variant_binding_pins(root)}
    for folder in (repaired, root / 'timelines' / 'baseline' / track['id']):
        for filename in ('timeline.npz', 'patch.json', 'report.json'):
            inputs[str(folder / filename)] = render.sha(folder / filename)
    write_json(repaired / 'comparison.json', {'method': 'D4', 'song': track['id'], 'artifact_id': 'D4_floor_v2',
        'status': 'diagnostic_pending_user_review', 'physical_engineering_status': 'pass', 'inputs': inputs,
        'primary': {'magnitude_gate': 'unqualified', 'missing_evidence': ['independent semantic reference']}})
    return root, plan, algorithm, track


def test_correctness_revision_keeps_logical_d4_and_original_artifacts(campaign):
    root, plan, algorithm, track = revised = variant_fixture(campaign)
    original_path = root / 'timelines' / 'D4' / track['id'] / 'timeline.npz'
    original_sha = render.sha(original_path)
    baseline, candidate, music, pins = render.load_comparison(*revised)
    assert candidate.algorithm_id == 'D4'
    assert candidate.artifact_revision == 'D4_floor_v2'
    assert candidate.repair_reason == 'XZ alignment preserves floor'
    assert pins['candidate']['artifact_revision'] == 'D4_floor_v2'
    assert pins['variant_binding_files'] == render.campaign.variant_binding_pins(root)
    np.testing.assert_allclose(candidate.body[:, :, 1], baseline.body[:, :, 1] - .1)
    assert render.visual_fingerprint(root, algorithm, track) == render.visual_fingerprint(root, {'id': 'D4_floor_v2'}, track)
    manifest = render.review(root, plan, 'schematic')
    assert len(manifest['variants']) == 1
    row = manifest['variants'][0]
    assert row['algorithm_id'] == 'D4'
    assert row['artifact_revision'] == 'D4_floor_v2'
    assert row['repair_reason'] == 'XZ alignment preserves floor'
    assert row['revision_provenance']['original_artifact_id'] == 'D4'
    smoke = render.smoke_pair(*revised, 'schematic')
    assert smoke['artifact_revision'] == 'D4_floor_v2'
    assert (root / 'smoke' / 'D4' / track['id'] / 'schematic' / 'SMOKE.json').exists()
    assert not (root / 'smoke' / 'D4_floor_v2').exists()
    assert render.sha(original_path) == original_sha


def test_repaired_comparison_must_pin_the_variant_selection(campaign):
    root, plan, algorithm, track = revised = variant_fixture(campaign)
    comparison_path = root / 'timelines' / 'D4_floor_v2' / track['id'] / 'comparison.json'
    comparison = render.read(comparison_path)
    comparison['inputs'].pop(str(root / 'records' / 'VARIANT_BINDINGS.json'))
    write_json(comparison_path, comparison)
    with pytest.raises(ValueError, match='baseline/plan'):
        render.load_comparison(*revised)


def generated_pair_fixture(campaign, method='D7'):
    root, plan, _, track = campaign
    algorithm = {'id': method, 'name': 'Generated dance'}
    folder = root / 'timelines' / method / track['id']
    shutil.copytree(root / 'timelines' / 'D1' / track['id'], folder)
    body_path = root / 'bodies' / method / track['id'] / 'body.npz'
    body_path.parent.mkdir(parents=True)
    # Array contents are a CPU fixture; this test validates provenance, not a model.
    np.savez(body_path, world=np.zeros((60, 22, 3)), time=np.arange(60) / 30.)
    sidecar = body_path.with_suffix('.json')
    metadata = {'newly_generated_motion': True, 'checkpoint_sha256': 'frozen-checkpoint-fixture',
                'arm': method, 'seed': 0, 'draw_seed': 7001, 'source_clock': 'fresh native30Hz flow'}
    write_json(sidecar, {'schema_version': 1, 'kind': 'DanceResult',
                         'npz_sha256': render.sha(body_path), 'metadata': metadata})
    write_json(sidecar.with_name(sidecar.name + '.sha256.json'), {'sha256': render.sha(sidecar)})
    report = render.read(folder / 'report.json')
    report['body_sha256'] = render.sha(body_path)
    report['inputs'] = {'body_metadata_sha256': render.sha(sidecar)}
    write_json(folder / 'report.json', report)
    return root, plan, algorithm, track


@pytest.mark.parametrize('method', ['D7', 'D8'])
def test_fresh_generated_body_is_distinctly_labeled_from_retiming(campaign, method):
    root, plan, algorithm, track = generated_pair_fixture(campaign, method)
    pair = render.load_pair(root, plan, algorithm, track)
    assert pair.motion_origin['kind'] == 'newly_generated'
    assert pair.motion_origin['label'] == 'newly generated native30Hz body'
    assert pair.motion_origin['metadata']['draw_seed'] == 7001
    assert pair.pins['body_source_sha256'] == render.sha(root / 'bodies' / method / track['id'] / 'body.npz')
    info = render.motion_provenance(root, 'D6', 'D6', track, {})
    assert info['kind'] == 'learned_retiming'
    assert 'retiming' in info['label']


def test_generation_label_cannot_come_from_unverified_metadata(campaign):
    root, plan, algorithm, track = generated_pair_fixture(campaign)
    sidecar = root / 'bodies' / 'D7' / track['id'] / 'body.json'
    details = render.read(sidecar)
    details['metadata']['newly_generated_motion'] = False
    write_json(sidecar, details)
    with pytest.raises(ValueError, match='metadata differs'):
        render.load_pair(root, plan, algorithm, track)
    write_json(sidecar.with_name(sidecar.name + '.sha256.json'), {'sha256': render.sha(sidecar)})
    folder = root / 'timelines' / 'D7' / track['id']
    report = render.read(folder / 'report.json')
    report['inputs']['body_metadata_sha256'] = render.sha(sidecar)
    write_json(folder / 'report.json', report)
    with pytest.raises(ValueError, match='requires authenticated generation provenance'):
        render.load_pair(root, plan, algorithm, track)


def test_d4_marks_recorded_fragment_cuts_without_changing_saved_timeline(campaign):
    root, plan, algorithm, track = generated_pair_fixture(campaign, 'D4')
    body_metadata = root / 'bodies' / 'D4' / track['id'] / 'body.json'
    details = render.read(body_metadata)
    details['metadata'].update(newly_generated_motion=False, fragment_provenance=[
        {'target_seconds': [0., .3], 'source_seconds': [0., .3]},
        {'target_seconds': [.3, 1.], 'source_seconds': [1., 1.7]}])
    write_json(body_metadata, details)
    write_json(body_metadata.with_name(body_metadata.name + '.sha256.json'), {'sha256': render.sha(body_metadata)})
    folder = root / 'timelines' / 'D4' / track['id']
    report = render.read(folder / 'report.json')
    report['inputs']['body_metadata_sha256'] = render.sha(body_metadata)
    write_json(folder / 'report.json', report)
    before = render.sha(folder / 'timeline.npz')
    pair = render.load_pair(root, plan, algorithm, track)
    assert pair.motion_origin['kind'] == 'recomposed_fragments'
    assert .3 in pair.interventions
    assert render.sha(folder / 'timeline.npz') == before


def test_guarded_render_rejects_an_unbound_root(monkeypatch, tmp_path):
    from scripts.motion import dynamic_dance_guard
    monkeypatch.setattr(dynamic_dance_guard, 'authenticate_guarded_job', lambda _: {
        'identity': {'command': ['python', str(render.__file__), '--root', '/different/root', '--guarded']}})
    with pytest.raises(ValueError, match='campaign root'):
        render.require_guard(tmp_path)


def test_review_pending_rows_preserve_failure_without_fake_media(campaign):
    root, plan, algorithm, track = campaign
    folder = root / 'timelines' / algorithm['id'] / track['id']
    report = render.read(folder / 'report.json')
    report['failure'] = 'source-conditioned output failed numeric gate'
    write_json(folder / 'report.json', report)
    manifest = render.review(root, plan, 'dark')
    row = manifest['variants'][0]
    assert row['status'] == 'planned'
    assert 'path' not in row
    assert row['failure'] == report['failure']
    assert (root / 'review' / 'index.html').exists()

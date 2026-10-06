"""Campaign provenance, emitted counterfactuals and no-overwrite behavior."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from lightsem.conductor.body_transfer import TransferConfig, build_outputs, constrained_transfer, sample_body
from lightsem.conductor.body_transfer_v2.common import REFERENCE
from scripts.motion import run_body_portfolio as campaign


@pytest.fixture
def fixture_campaign(tmp_path):
    root = tmp_path / 'campaign'
    root.mkdir()
    old = tmp_path / 'saved'
    old.mkdir()
    world = np.broadcast_to(REFERENCE, (90, 22, 3)).copy()
    times, body = sample_body(world)
    left = np.broadcast_to(np.array([12, 34, 56], np.uint8), (len(body), 3)).copy()
    right = np.broadcast_to(np.array([78, 90, 123], np.uint8), (len(body), 3)).copy()
    dmx, patch, _ = constrained_transfer(body, build_outputs(), left, right)
    np.savez_compressed(old / 'body.npz', world=world)
    np.savez_compressed(old / 'timeline.npz', body=body, time=times, universes=dmx)
    campaign.save(old / 'patch.json', patch)
    campaign.save(old / 'BODY_PROVENANCE.json', {'source': 'test fixture, never campaign music'})
    (old / 'audio.wav').write_bytes(b'test audio identity')
    (old / 'audio_reference.m4a').write_bytes(b'test crop identity')
    campaign.save(root / 'source_manifest.json', {'files': {}})
    track = {'id': 'shpongle', 'title': 'Test source', 'audio': str(old / 'audio.wav'),
        'audio_reference': str(old / 'audio_reference.m4a'), 'body': str(old / 'body.npz'),
        'baseline_timeline': str(old / 'timeline.npz'), 'baseline_patch': str(old / 'patch.json'),
        'body_provenance': str(old / 'BODY_PROVENANCE.json')}
    plan = {'source_manifest': str(root / 'source_manifest.json'), 'tracks': [track],
        'prefix_s': 3, 'body_fps': 30., 'fps': 40., 'counts': {'beams': 8, 'lasers': 5, 'bulbs': 50},
        'transfer': asdict(TransferConfig()), 'review_start_s': 60, 'review_duration_s': 120,
        'preflight': {'frames': 101, 'angle_degrees': 12.},
        'algorithms': [{'id': '01', 'name': 'Bone directions', 'papers': []}]}
    campaign.save(root / 'PLAN.json', plan)
    return root, plan, track, left, right


def test_p2_read_from_actual_baseline_channels(fixture_campaign):
    _, plan, track, left, right = fixture_campaign
    _, body, actual_left, actual_right, _, _ = campaign.load_source(plan, track)
    assert body.shape == (120, 22, 3)
    np.testing.assert_array_equal(actual_left, left)
    np.testing.assert_array_equal(actual_right, right)


def test_song_body_swap_is_detected(fixture_campaign):
    _, plan, track, _, _ = fixture_campaign
    with np.load(track['body']) as data:
        world = data['world'].copy()
    world[:, 15, 0] += .1
    np.savez_compressed(track['body'], world=world)
    with pytest.raises(RuntimeError, match='resampling differs'):
        campaign.load_source(plan, track)


@pytest.mark.parametrize('region', campaign.REGIONS)
def test_counterfactual_has_only_declared_anatomy_and_exact_holds(region):
    changed = campaign.intervention(REFERENCE, region)
    np.testing.assert_array_equal(changed[:21], np.broadcast_to(REFERENCE, changed[:21].shape))
    np.testing.assert_array_equal(changed[80:], np.broadcast_to(REFERENCE, changed[80:].shape))
    np.testing.assert_array_equal(changed[40:61], np.broadcast_to(changed[40], changed[40:61].shape))
    indices = campaign.REGIONS[region]['joints']
    other = [j for j in range(22) if j not in indices]
    np.testing.assert_array_equal(changed[:, other], np.broadcast_to(REFERENCE[other], changed[:, other].shape))
    assert np.linalg.norm(changed[50] - REFERENCE) > .001
    pivot = REFERENCE[campaign.REGIONS[region]['pivot']]
    np.testing.assert_allclose(np.linalg.norm(changed[:, indices] - pivot, axis=-1),
                              np.broadcast_to(np.linalg.norm(REFERENCE[indices] - pivot, axis=-1), (101, len(indices))),
                              atol=1e-14)
    parent = np.asarray(campaign.PARENTS[1:])
    lengths = np.linalg.norm(changed[:, 1:] - changed[:, parent], axis=-1)
    expected = np.linalg.norm(REFERENCE[1:] - REFERENCE[parent], axis=-1)
    np.testing.assert_allclose(lengths, np.broadcast_to(expected, lengths.shape), atol=1e-14)


def test_preflight_measures_emitted_head_information_missing_from_v1(fixture_campaign):
    root, plan, track, _, _ = fixture_campaign
    baseline = campaign.preflight(root, plan, track, 'v1')
    current = campaign.preflight(root, plan, track, '01')
    assert baseline['frozen_body_DMX_exact'] and current['frozen_body_DMX_exact']
    assert not baseline['regions']['head']['angular_response']
    assert current['regions']['head']['angular_response']
    assert current['emitted_source_region_coverage'] > baseline['emitted_source_region_coverage']
    assert current['regions']['head']['maximum_angle_change_ticks'] >= 1
    assert 'intensity_response_count' in current['regions']['head']


def test_freeze_failure_is_saved_and_blocks_retry(fixture_campaign, monkeypatch):
    root, plan, track, _, _ = fixture_campaign
    real = campaign._compile
    def wrong(*args, **kwargs):
        dmx, patch, intent, hard = real(*args, **kwargs)
        dmx[1, 0, 0] ^= 1
        return dmx, patch, intent, hard
    monkeypatch.setattr(campaign, '_compile', wrong)
    with pytest.raises(RuntimeError, match='Frozen body'):
        campaign.preflight(root, plan, track, '01')
    report = campaign.read(root / 'preflight/01/shpongle.json')
    assert report['status'] == 'failed'
    with pytest.raises(RuntimeError, match='failed/changed preflight'):
        campaign.preflight(root, plan, track, '01')


def test_generate_preserves_p2_and_skips_only_verified_complete_outputs(fixture_campaign, monkeypatch):
    root, plan, track, _, _ = fixture_campaign
    campaign.preflight(root, plan, track, 'v1')
    campaign.preflight(root, plan, track, '01')
    algorithm = plan['algorithms'][0]
    result = campaign.generate(root, plan, algorithm, track)
    assert result['status'] == 'diagnostic'  # Deliberately stationary test dance.
    assert result['P2_RGB_exact'] and result['hard_invariants_passed']
    folder = root / 'timelines/01/shpongle'
    with np.load(folder / 'timeline.npz') as data:
        assert set(data.files) == {'universes', 'body', 'time', 'targets', 'source_points', 'field_points', 'dimmer'}
    monkeypatch.setattr(campaign, '_compile', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Unexpected rerun')))
    assert campaign.generate(root, plan, algorithm, track) == result
    with (folder / 'timeline.npz').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(RuntimeError, match='hash changed'):
        campaign.generate(root, plan, algorithm, track)


def test_partial_output_is_preserved_and_not_overwritten(fixture_campaign):
    root, plan, track, _, _ = fixture_campaign
    folder = root / 'timelines/01/shpongle'
    folder.mkdir(parents=True)
    (folder / 'partial').write_text('evidence')
    with pytest.raises(RuntimeError, match='Incomplete output preserved'):
        campaign.generate(root, plan, plan['algorithms'][0], track)
    assert (folder / 'partial').read_text() == 'evidence'


def test_source_verification_and_freeze_no_overwrite(tmp_path):
    source = tmp_path / 'frozen.py'
    source.write_text('constant = 1\n')
    plan = {'files': {str(source): campaign.sha(source)}}
    campaign.verify_files(plan)
    source.write_text('constant = 2\n')
    with pytest.raises(RuntimeError, match='Frozen input changed'):
        campaign.verify_files(plan)
    root = tmp_path / 'old'
    root.mkdir()
    (root / 'PLAN.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Freeze already'):
        campaign.freeze(root, tmp_path)


def test_execution_requires_actual_copied_imports(tmp_path):
    with pytest.raises(RuntimeError, match='Execute this stage'):
        campaign.require_frozen_imports({'source_root': str(tmp_path)})


def test_json_rejects_nonfinite_without_creating_artifact(tmp_path):
    output = tmp_path / 'report.json'
    with pytest.raises(ValueError):
        campaign.save(output, {'bad': float('nan')})
    assert not output.exists()

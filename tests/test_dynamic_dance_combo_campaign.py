"""Small CPU integration fixtures, with all learned generation mocked."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lightsem.conductor.body_transfer import TransferConfig, decode
from lightsem.conductor.body_transfer_v2 import make_outputs
from lightsem.experiments.dynamic_dance.contracts import DanceResult, DancePlan, MusicTimeline
from lightsem.experiments.dynamic_dance_combo import campaign as c
from lightsem.experiments.dynamic_dance_combo import generation as g
from lightsem.experiments.motion_study.geometry import joints_from_features


def frozen_pose(frames):
    native = torch.zeros(frames, 139)
    native[:, 2] = 1.
    native[:, 3:135] = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(22)
    native[:, 135:] = 1.
    return native.numpy(), joints_from_features(native).numpy()


@pytest.fixture
def campaign_fixture(tmp_path, monkeypatch):
    tracks = []
    for song, frames in [('long', 36), ('short', 24)]:
        duration = frames/30
        tm = np.arange(frames)/30
        timeline = MusicTimeline(time=tm, muq=np.zeros((frames, 1024), np.float32), muq_time=tm,
            sources=np.ones((frames, 4, 4), np.float32)*.01, mixture=np.ones((frames, 4), np.float32)*.01,
            beats=np.array([.2, .6]), downbeats=np.array([.2]),
            sections=[{'start': 0., 'end': duration, 'label': 'unknown', 'confidence': .8}], recurrence=np.ones((1, 1)),
            metadata={'duration_s': duration, 'recipe': {'observation_hz': 30, 'muq_native_hz': 30}})
        path = tmp_path/'input'/song/'music.npz'; timeline.save(path)
        tracks.append({'id': song, 'duration_s': duration, 'body_frames': frames, 'lighting_frames': int(duration*40),
                       'music_timeline': str(path), 'base92': str(path.parent/'base.npz'),
                       'waveform_controls': {'gain': {'music_timeline': str(path), 'base92': str(path.parent/'gain_base.npz')}},
                       'loudness_receipt': str(path.parent/'loudness.json'), 'audio': str(path.parent/'audio.wav')})
    c.save(tmp_path/'PLAN.json', {'tracks': tracks, 'counts': {'beams': 8, 'lasers': 5, 'bulbs': 50}, 'transfer': asdict(TransferConfig())})
    calls = []

    class NoiseOnly:
        def noise(self, batch, frames, generator, device):
            # Preserve the real noise API's channel-major draw structure. A
            # shorter independently drawn tensor would differ by channel.
            return torch.randn(batch, 139, frames, generator=generator, device=device).transpose(1, 2)

    monkeypatch.setattr('lightsem.experiments.dynamic_dance.training.load_learned', lambda *args, **kwargs: (NoiseOnly(), {}, Path('unused.pt')))

    def mock_generate(oldroot, timeline, base92, seconds, device, drawseed, *, initial_noise, plan_intervention, output_common_reference):
        frames = int(round(seconds*30)); calls.append({'frames': frames, 'noise': initial_noise.clone(), 'intervention': plan_intervention,
                                                     'base92': base92, 'reference': output_common_reference})
        native, world = frozen_pose(frames)
        time = np.arange(frames)/30
        coarse_count = (frames+14)//15
        local_plan = np.zeros((coarse_count, 34), np.float32)
        local_plan[:, :5] = .1
        local_plan[:, 5:9] = .5
        local_plan[:, 9:14] = .1
        local_plan[:, 14:] = .25
        global_plan = np.zeros((frames, 34), np.float32)
        global_plan[:, :5] = .1
        result = {'time': time, 'combined': {'native': native, 'world': world, 'contacts': native[:, 135:]},
                  'telemetry': {'initial_noise': initial_noise[0].numpy(), 'global_plan': global_plan,
                                'coarse_local_plan': local_plan, 'coarse_time': np.arange(coarse_count)*.5},
                  'metadata': {'test_fixture': True, 'draw_seed': drawseed}}
        if output_common_reference: result['reference'] = result['combined']
        return result

    monkeypatch.setattr(g, 'generate_full', mock_generate)
    return tmp_path, tracks, calls


def test_campaign_uses_shared_maximum_noise_time_prefix(campaign_fixture):
    root, tracks, calls = campaign_fixture
    c.generate(root, 'long')
    c.generate(root, 'short')
    c.generate(root, 'short', 'gain')
    assert torch.equal(calls[0]['noise'][:, :24], calls[1]['noise'])
    assert torch.equal(calls[1]['noise'], calls[2]['noise'])
    assert calls[2]['base92'] == tracks[1]['waveform_controls']['gain']['base92']
    assert calls[0]['reference'] and calls[1]['reference'] and not calls[2]['reference']
    body = DanceResult.load(c.body_path(root, 'G78', 'short'))
    assert not body.metadata['product_baseline']
    assert body.metadata['condition'] == 'correct'
    plan = DancePlan.load(c.body_path(root, 'G78', 'short').parent/'plan.npz')
    assert plan.effort.shape == (24, 5)


def test_plan_shift_and_repeat_are_generation_only_controls(campaign_fixture):
    root, _, calls = campaign_fixture
    assert c.generate(root, 'short', 'plan_shift') == 'G78_shift'
    assert c.generate(root, 'short', 'repeat') == 'G78__repeat'
    assert calls[0]['intervention'] == 'shift_global_half'
    assert calls[1]['intervention'] is None
    assert torch.equal(calls[0]['noise'], calls[1]['noise'])
    assert not calls[0]['reference'] and not calls[1]['reference']


def test_real_postprocess_preserves_same_c3_as_c4_source_and_source_clock_scope(campaign_fixture):
    root, _, _ = campaign_fixture
    c.generate(root, 'short')
    methods = c.postprocess(root, 'short')
    assert methods == ['C3', 'C4', 'C3_shuffled', 'C4_loudness', 'C4_shifted']
    c3path = c.body_path(root, 'C3', 'short')
    c3 = DanceResult.load(c3path)
    c4 = DanceResult.load(c.body_path(root, 'C4', 'short'))
    assert c4.metadata['source_body'] == str(c3path)
    assert c4.metadata['source_body_sha256'] == c.sha(c3path)
    assert 'monotone clocks into C3' in c4.metadata['source_clock_scope']
    assert not c3.metadata['source_order_preserved']
    assert c4.metadata['source_order_preserved']
    assert c3.metadata['d6_enabled'] is False and c4.metadata['d6_enabled'] is True
    np.testing.assert_array_equal(c3.world, c4.world)
    np.testing.assert_allclose(c4.source_time[0], 0, atol=1e-10)
    np.testing.assert_allclose(c4.source_time[-1], c4.time[-1], atol=1e-10)
    rates = np.diff(c4.source_time, axis=0)/np.diff(c4.time)[:, None]
    assert np.all(rates >= -1e-8) and np.all(rates <= 1.5+1e-8)


def test_actual_mapper_restores_audio_dimmer_without_motion_or_color_change(campaign_fixture, monkeypatch):
    root, tracks, _ = campaign_fixture
    c.generate(root, 'short'); c.postprocess(root, 'short')
    track = tracks[1]; count = track['lighting_frames']
    color_folder = root/'colors'/'short'; color_folder.mkdir(parents=True)
    left = np.tile(np.array([18, 45, 77], np.uint8), (count, 1))
    right = np.tile(np.array([42, 19, 111], np.uint8), (count, 1))
    c.arrays(color_folder/'P2.npz', left_rgb=left, right_rgb=right, palette_ids40=np.zeros(count, np.int16))
    c.save(color_folder/'REPORT.json', {'test_fixture': True})
    loud = {'time': np.arange(count)/40, 'momentary': np.linspace(-40., -10., count), 'short_term': np.linspace(-40., -10., count)}
    monkeypatch.setattr(c, 'load_verified_loudness', lambda *args: (loud, {'test_fixture': True}))
    report = c.map_one(root, 'short', 'C3')
    assert report['status'] == 'engineering_valid'
    assert report['P2_RGB_exact']
    folder = root/'timelines'/'C3'/'short'
    with np.load(folder/'timeline.npz') as z:
        decoded = decode(z['universes'], c.read(folder/'patch.json'))
        assert np.all(decoded['angles_deg'] == decoded['angles_deg'][:1])
        assert np.ptp(decoded['dimmer'][:, 0]) > .1
        np.testing.assert_array_equal(np.rint(decoded['rgb'][:, 0]*255).astype(np.uint8), left)
        assert z['body'].shape == (count, 22, 3)
        assert z['brightness_envelope'][0] == .12


def test_postprocess_rejects_generation_only_controls_early(campaign_fixture):
    root, _, _ = campaign_fixture
    # These controls intentionally remain at G78 for direct planning/noise
    # diagnostics. Reject misuse explicitly, rather than reaching path/KeyError.
    for condition in ('plan_shift', 'repeat'):
        with pytest.raises(ValueError, match='generation.only|postprocess|Unsupported'):
            c.postprocess(root, 'short', condition)


def test_arbitrary_count_keeps_shared_fixed_body_streams_with_restored_brightness():
    _, body = frozen_pose(24)
    time = np.arange(24)/40
    loud = {'time': time, 'momentary': np.linspace(-40., -10., len(time)), 'short_term': np.linspace(-40., -10., len(time))}
    left = np.tile(np.array([11, 45, 99], np.uint8), (len(time), 1))
    right = np.tile(np.array([34, 9, 111], np.uint8), (len(time), 1))
    cases = []
    for counts in [(1, 0, 0), (8, 5, 50), (17, 20, 80)]:
        outputs = make_outputs(*counts)
        dmx, patch, intent, trace, hard = c.compile_world(body, time, outputs, left, right, loud, asdict(TransferConfig()))
        decoded = decode(dmx, patch)
        cases.append(({p['name']: i for i, p in enumerate(patch)}, decoded, intent))
        assert hard['maximum_speed_deg_s'] == 0
        assert hard['maximum_acceleration_deg_s2'] == 0
    reference_names, reference, reference_intent = cases[1]
    for names, decoded, intent in (cases[0], cases[2]):
        for name in names.keys() & reference_names.keys():
            i, j = names[name], reference_names[name]
            for key in ('angles_deg', 'directions', 'dimmer', 'rgb'):
                np.testing.assert_array_equal(decoded[key][:, i], reference[key][:, j])
            np.testing.assert_array_equal(intent['targets'][:, i], reference_intent['targets'][:, j])

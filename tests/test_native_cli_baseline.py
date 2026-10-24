"""Baseline wiring, waveform/encoder binding and independent optional modifiers."""
import argparse
import copy
from pathlib import Path

import numpy as np
import pytest

from lightsem.dance.baseline import load_config, validate_music, sha
from lightsem.dance.cli import add_options
from lightsem.dance.pipeline import modify_body
from lightsem.experiments.dynamic_dance.contracts import MusicTimeline, DanceResult
from lightsem.dance.render import modifier_rate_label


def test_render_accepts_underscore_ssl_light_alias(capsys):
    from lightsem.cli import main

    with pytest.raises(SystemExit) as exc:
        main(['render', '--wav', 'song.wav', '--ssl_light', '--help'])
    assert exc.value.code == 0
    assert '--ssl_light' in capsys.readouterr().out


def music(tmp_path, config):
    audio = tmp_path / 'input.wav'; audio.write_bytes(b'waveform identity test')
    required = config['muq']
    meta = {'input_sha256': sha(audio), 'intervention': None,
            'muq': {'representation': 'muq', 'layer': required['layer'],
                    'model_receipt': {'model_id': required['model_id'], 'revision': required['revision'],
                                      'artifacts': {'model.safetensors': {'sha256': required['weights_sha256']}}}}}
    time = np.arange(300) / 30
    timeline = MusicTimeline(time=time, muq=np.ones((250, 1024), np.float16), muq_time=np.arange(250) / 25,
                             sources=np.zeros((300, 4, 4), np.float32), mixture=np.zeros((300, 4), np.float32),
                             beats=np.arange(0, 10, .5), downbeats=np.arange(0, 10, 2.),
                             sections=[{'start': 0., 'end': 10., 'label': 'verse'}], recurrence=np.ones((1, 1)), metadata=meta)
    return audio, timeline


def test_options_are_independent_and_default_to_plain_C():
    parser = argparse.ArgumentParser(); add_options(parser)
    default = parser.parse_args(['--wav', 'song.wav'])
    assert not default.d4max and not default.d6
    only_d6 = parser.parse_args(['--wav', 'song.wav', '--d6'])
    assert only_d6.d6 and not only_d6.d4max
    both = parser.parse_args(['--wav', 'song.wav', '--d4-max', '--d6'])
    assert both.d4max and both.d6
    with pytest.raises(SystemExit):
        parser.parse_args(['--wav', 'song.wav', '--no-muq'])


def test_modifier_rate_label_distinguishes_d4_recomposition_from_d6_pacing():
    assert modifier_rate_label({}) == 'Native clock (×)'
    assert (modifier_rate_label({'d4max_enabled': True, 'd6_enabled': False})
            == 'Recorded D4 local fragment rate (×)')
    assert (modifier_rate_label({'d4max_enabled': True, 'd6_enabled': True})
            == 'Recorded D6 playback rate (×)')


def test_wrong_waveform_and_alternate_encoder_are_rejected(tmp_path):
    config, _ = load_config(); audio, timeline = music(tmp_path, config)
    validate_music(timeline, audio, config)
    other = tmp_path / 'other.wav'; other.write_bytes(b'other soundtrack')
    with pytest.raises(ValueError, match='different audio'):
        validate_music(timeline, other, config)
    timeline.metadata['muq']['representation'] = 'mert'
    with pytest.raises(ValueError, match='pinned frozen MuQ'):
        validate_music(timeline, audio, config)


def test_MuQ_weights_cannot_be_silently_substituted(tmp_path):
    config, _ = load_config(); audio, timeline = music(tmp_path, config)
    timeline.metadata['muq']['model_receipt']['artifacts']['model.safetensors']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='pinned frozen MuQ'):
        validate_music(timeline, audio, config)


@pytest.mark.parametrize('d4max,d6', [(False, False), (True, False), (False, True), (True, True)])
def test_frozen_source_stays_frozen_for_every_modifier_combination(tmp_path, d4max, d6):
    config, _ = load_config(); _, timeline = music(tmp_path, config)
    from lightsem.experiments.motion_study.geometry import OFFSETS, PARENTS
    rest = np.zeros((22, 3)); rest[0, 1] = 1.
    for joint, parent in enumerate(PARENTS[1:], 1):
        rest[joint] = rest[parent] + OFFSETS[joint]
    world = np.broadcast_to(rest, (300, 22, 3)).astype(np.float32).copy()
    source = DanceResult(time=np.arange(300) / 30, world=world, contacts=np.zeros((300, 4)), metadata={'fps': 30, 'muq': timeline.metadata['muq']})
    telemetry = {'coarse_time': (np.arange(20) * 15 + 7) / 30,
                 'coarse_local_plan': np.zeros((20, 34), np.float32)}
    result, plan = modify_body(source, timeline, config, d4max=d4max, d6=d6, telemetry=telemetry)
    np.testing.assert_array_equal(result.world, world)
    assert result.metadata['d4max_enabled'] is d4max and result.metadata['d6_enabled'] is d6
    assert result.metadata['muq']['representation'] == 'muq'
    if not d4max:
        assert not result.metadata.get('fragment_provenance')
    if d6:
        assert np.diff(result.source_time).min() > 0
        assert plan.effort.min() >= .25 - 1e-7


def test_D6_never_enables_D4_implicitly(tmp_path, monkeypatch):
    config, _ = load_config(); _, timeline = music(tmp_path, config)
    def unexpected(*args, **kwargs):
        raise AssertionError('D4 recomposition called when only --d6 was selected')
    import lightsem.experiments.d4max_d8.motifs as motifs
    monkeypatch.setattr(motifs, 'compose', unexpected)
    world = np.zeros((300, 22, 3), np.float32)
    body = DanceResult(time=np.arange(300) / 30, world=world, contacts=np.zeros((300, 4)))
    telemetry = {'coarse_time': (np.arange(20) * 15 + 7) / 30,
                 'coarse_local_plan': np.zeros((20, 34), np.float32)}
    result, _ = modify_body(body, timeline, config, d6=True, telemetry=telemetry)
    assert result.metadata['d6_enabled'] and not result.metadata['d4max_enabled']


def test_source_dependence_probe_checks_all_spatial_axes(monkeypatch):
    """A valid mapper must not fail because its response is Y/Z-only."""
    import lightsem.conductor.body_transfer as transfer
    from lightsem.dance.lighting import _source_dependence_audit

    state = {}

    def compile_world(world, *args, **kwargs):
        state['world'] = np.asarray(world)
        return None, [{'kind': 'bulb'}], None, None, None

    def decode(*args, **kwargs):
        world = state['world']
        # This synthetic source intentionally responds only to joint 3's Y
        # coordinate. The old X-only probe incorrectly rejected it.
        value = world[:, 3, 1]
        return {'directions': np.stack((value, value * 0, value * 0), axis=-1)[:, None],
                'angles_deg': np.zeros((len(world), 1, 2)),
                'dimmer': np.zeros((len(world), 1))}

    monkeypatch.setattr(transfer, 'decode', decode)
    world = np.zeros((8, 22, 3), dtype=np.float64)
    result = _source_dependence_audit(
        world, np.arange(len(world)) / 40., [{'kind': 'bulb'}],
        np.zeros((len(world), 3), dtype=np.uint8),
        np.zeros((len(world), 3), dtype=np.uint8),
        {'time': np.arange(32) / 40., 'momentary': np.zeros(32), 'short_term': np.zeros(32)},
        {}, '01', {'attribution': [[3]]}, compile_world)
    assert result['status'] == 'pass'
    assert result['assigned_joint_responses']['3']['windows'][0]['axis_responses'][1]['responsive']


def test_native_body_grows_planner_context_for_long_songs(monkeypatch):
    """Inference keeps all planner bins when a song exceeds the training pad."""
    import torch
    import lightsem.dance.pipeline as pipeline
    import lightsem.experiments.dynamic_dance.audio as audio_features
    import lightsem.experiments.motion_study.geometry as geometry
    import lightsem.experiments.native_dynamics.evaluation as native_evaluation

    config, _ = load_config()
    frames = 1024 * 15 + 1
    base = np.zeros((frames, 92), np.float32)
    extra = np.zeros((frames, 1061), np.float32)
    captured = {}

    class ModelConfig:
        stereo_dim = 0
        stereo_fusion = 'concat'
        frames = 48

    class Model:
        config = ModelConfig()
        flow = object()

        def generate(self, fine, coarse, valid, indices, noise):
            captured['coarse_shape'] = tuple(coarse.shape)
            captured['valid_count'] = int(valid.sum())
            captured['frame_count'] = int(indices.shape[1])
            plan = {'conditioning': torch.zeros(1, coarse.shape[1], 80)}
            native = torch.zeros(1, indices.shape[1], 139)
            return native, plan

    norms = [
        (torch.zeros(92), torch.ones(92)),
        (torch.zeros(1061), torch.ones(1061)),
    ]
    monkeypatch.setattr(pipeline, 'load_model', lambda *args, **kwargs: (Model(), norms, {}))
    monkeypatch.setattr(audio_features, 'model_features', lambda *args, **kwargs: (extra, []))
    monkeypatch.setattr(native_evaluation, 'cpu_noise', lambda *args, **kwargs: torch.zeros(1, frames, 139))
    monkeypatch.setattr(geometry, 'joints_from_features', lambda value: torch.zeros(value.shape[0], value.shape[1], 22, 3))

    timeline = type('Music', (), {'metadata': {'muq': {}}})()
    body, _, plan, _ = pipeline.native_body(timeline, base, config)

    expected_bins = (frames + 14) // 15
    assert captured == {
        'coarse_shape': (1, expected_bins, 1061),
        'valid_count': expected_bins,
        'frame_count': frames,
    }
    assert plan['planner_context_bins'] == expected_bins
    assert len(body.time) == frames

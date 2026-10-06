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

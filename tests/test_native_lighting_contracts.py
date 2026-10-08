"""Regression tests for native body-to-light contract boundaries."""

import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from lightsem.conductor.body_transfer import decode
from lightsem.dance.lighting import (
    _exact_clock,
    _load_p2_colors,
    _normalise_counts,
    _validate_body_for_lighting,
)


def test_lighting_clock_rejects_shifted_or_irregular_samples():
    with pytest.raises(ValueError, match="zero-based"):
        _exact_clock(np.arange(8, dtype=float) / 40 + .025, 40., "P2 timestamps")
    with pytest.raises(ValueError, match="zero-based"):
        _exact_clock(np.array([0., .025, .05, .076]), 40., "P2 timestamps")


def test_body_contract_rejects_a_song_that_is_more_than_one_native_tick_short():
    body = SimpleNamespace(world=np.zeros((2, 22, 3)), time=np.arange(2) / 30.)
    with pytest.raises(ValueError, match="complete lighting interval"):
        _validate_body_for_lighting(body, .2)


def test_fixture_counts_are_explicit_and_complete():
    assert _normalise_counts({'beams': 1, 'lasers': 0, 'bulbs': 2}) == {
        'beams': 1, 'lasers': 0, 'bulbs': 2}
    with pytest.raises(ValueError, match="exactly"):
        _normalise_counts({'beams': 1, 'bulbs': 2})
    with pytest.raises(ValueError, match="integer"):
        _normalise_counts({'beams': True, 'lasers': 0, 'bulbs': 1})


def test_p2_requires_authenticated_timestamped_stream(tmp_path):
    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'audio')
    colors = tmp_path / 'P2.npz'
    np.savez(colors, left=np.zeros((4, 3), np.uint8), right=np.zeros((4, 3), np.uint8),
             palette_ids40=np.zeros(4, np.int16))
    (tmp_path / 'REPORT.json').write_text(json.dumps({'audio_sha256': hashlib.sha256(b'audio').hexdigest(),
                                                      'P2_sha256': hashlib.sha256(colors.read_bytes()).hexdigest()}))
    with pytest.raises(ValueError, match="timestamps"):
        _load_p2_colors(colors, audio, 4)


def test_decode_rejects_overlapping_or_mis_sized_fixture_patch():
    dmx = np.zeros((2, 1, 512), np.uint8)
    common = {'name': 'beam_001', 'kind': 'beam', 'universe': 0, 'address': 1,
              'n_channels': 8, 'pan_range_deg': 540., 'tilt_range_deg': 270.}
    with pytest.raises(ValueError, match="overlapping"):
        decode(dmx, [common, {**common, 'name': 'beam_002', 'address': 5}])
    with pytest.raises(ValueError, match="width"):
        decode(dmx, [{**common, 'n_channels': 4}])

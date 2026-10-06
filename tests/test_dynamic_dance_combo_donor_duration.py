import numpy as np
import pytest

from scripts.motion.run_dynamic_combo_features import match_duration, waveform_sha


def test_duration_matching_pads_true_silence_and_preserves_original_prefix():
    source = np.arange(10, dtype=np.float32).reshape(2, 5)
    matched, report = match_duration(source, 8)
    assert np.array_equal(matched[:, :5], source)
    assert np.array_equal(matched[:, 5:], np.zeros((2, 3), np.float32))
    assert report['zero_padded_tail_samples'] == 3
    assert report['cropped_tail_samples'] == 0
    assert waveform_sha(matched) != waveform_sha(source)


def test_duration_matching_crops_tail_without_retiming():
    source = np.arange(10, dtype=np.float32).reshape(2, 5)
    matched, report = match_duration(source, 3)
    assert np.array_equal(matched, source[:, :3])
    assert report['cropped_tail_samples'] == 2
    assert report['zero_padded_tail_samples'] == 0


def test_rejects_other_clocks_or_nonfinite_samples():
    with pytest.raises(ValueError):
        match_duration(np.ones((2, 5), np.float64), 8)
    with pytest.raises(ValueError):
        match_duration(np.full((2, 5), np.nan, np.float32), 8)
    with pytest.raises(ValueError):
        match_duration(np.ones((2, 5), np.float32), 8.)

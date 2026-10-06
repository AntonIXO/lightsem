"""P2 identity, whole-song edge handling and RGB-only contract checks."""
from pathlib import Path
import json

import numpy as np
import pytest

from lightsem.experiments.dynamic_dance_combo import colors


def test_full_source_frame_coverage_is_ceil_not_round():
    assert colors.frame_count(277.3066666666667) == 11093
    assert colors.frame_count(201.64) == 8066
    assert colors.frame_count(180.) == 7200
    with pytest.raises(ValueError):
        colors.frame_count(float('nan'))


def test_only_fractional_endpoint_hold_is_allowed():
    source = np.arange(12, dtype=np.uint8).reshape(4, 3)
    result = colors.extend_endpoint(source, 5)
    assert np.array_equal(result[:4], source)
    assert np.array_equal(result[4], source[3])
    result[0] = 0
    assert not np.array_equal(result[0], source[0])
    with pytest.raises(ValueError):
        colors.extend_endpoint(source, 6)
    with pytest.raises(ValueError):
        colors.extend_endpoint(source, 3)


def test_model_padding_preserves_raw_prefix_and_only_completes_last_block():
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(8066, 94)).astype(np.float32)
    padded, padding = colors.pad_feature_block(raw)
    assert padding == 2
    assert np.array_equal(padded[:8066], raw)
    assert np.array_equal(padded[-2:], np.repeat(raw[-1:], 2, axis=0))
    assert np.array_equal(colors.pad_feature_block(padded)[0], padded)
    with pytest.raises(ValueError):
        colors.pad_feature_block(np.zeros((8, 92)))


def test_diagnostics_count_palette_changes_separately_from_rgb_fades():
    left = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                     [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], np.uint8)
    score = {'generated_palette_ids': np.array([1, 2]), 'color_cues': np.array([True, True])}
    report, ids, active = colors.color_diagnostics(left, left, score)
    assert report['color_cue_count'] == 2
    assert report['palette_changes_while_active'] == 1
    assert report['left_rgb_changing_ticks'] == 4
    assert report['palette_occupancy_seconds'] == {'1': .1, '2': .1}
    assert active.all()
    assert ids.tolist() == [1]*4+[2]*4


@pytest.mark.skipif(not colors.P2_SOURCE.exists(), reason='Requires authenticated historical project artifacts')
def test_saved_full_shpongle_colors_match_exactly_and_only_hold_one_tick(tmp_path):
    left, right, report = colors.generate(tmp_path, 'shpongle')
    slots, _ = colors.canonical_slots()
    with np.load(colors.P2_SOURCE, allow_pickle=False) as old:
        assert np.array_equal(left[:-1], old['candidate'][:, slots[0]])
        assert np.array_equal(right[:-1], old['candidate'][:, slots[1]])
    assert len(left) == 11093
    assert np.array_equal(left[-1], left[-2])
    assert report['details']['model_forwards'] == 0
    assert report['details']['terminal_RGB_endpoint_hold_ticks'] == 1
    assert report['diagnostics']['color_cue_count'] == 262
    assert report['diagnostics']['palette_changes_while_active'] == 48
    assert report['new_baseline'] is False
    reread = colors.generate(tmp_path, 'shpongle')
    assert np.array_equal(reread[0], left)
    # Completed output cannot silently be trusted after mutation.
    (tmp_path/'colors/shpongle/P2.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='output changed'):
        colors.generate(tmp_path, 'shpongle')


@pytest.mark.skipif(not (colors.OLD_CURE/'streams.npz').exists(), reason='Requires historical project artifacts')
def test_rgb_only_wrapper_replays_exact_old_cure_P2_without_nonrgb_bytes():
    from lightsem.experiments.visual_next.delivery import color_delivery
    colors.checked_sources()
    with np.load(colors.OLD_CURE/'streams.npz') as old:
        companion, candidate, times = old['companion'], old['candidate'], old['time']
    with np.load(colors.OLD_CURE/'model_outputs.npz') as old:
        scores = {key: old[key] for key in ('generated_palette_ids', 'color_cues')}
    prototypes = json.loads(colors.TRANSFORMS.read_text())['palette_prototypes']
    expected, _ = color_delivery(candidate, companion, times, scores['generated_palette_ids'],
                                 scores['color_cues'], prototypes, layout='two_groups')
    actual, report = colors.two_group_stream(scores, len(candidate), companion, prototypes)
    _, rgb_mask = colors.canonical_slots()
    assert np.array_equal(actual[:, rgb_mask], expected[:, rgb_mask])
    assert np.all(actual[:, ~rgb_mask] == 0)
    assert report['canonical_nonRGB_bytes_all_zero']
    # A longer no-cue input cannot fabricate unknown historical fallback colors.
    no_cues = {'generated_palette_ids': np.zeros(2000, int), 'color_cues': np.zeros(2000, bool)}
    with pytest.raises(ValueError, match='pre-first-cue'):
        colors.two_group_stream(no_cues, 8000, companion, prototypes)


def test_beat_child_requires_the_guarded_parent(tmp_path, monkeypatch):
    # The gate precedes all GPU/model calls once required modules are importable.
    import sys
    import types
    beat = types.ModuleType('beat_this.inference')
    beat.Audio2Frames = None
    post = types.ModuleType('beat_this.model.postprocessor')
    post.Postprocessor = None
    monkeypatch.setitem(sys.modules, 'beat_this.inference', beat)
    monkeypatch.setitem(sys.modules, 'beat_this.model.postprocessor', post)
    monkeypatch.delenv('LIGHTSEM_COMBO_COLOR_LOCK_OWNER', raising=False)
    with pytest.raises(RuntimeError, match='guarded parent'):
        colors._extract_beat_subprocess('unread.wav', tmp_path)

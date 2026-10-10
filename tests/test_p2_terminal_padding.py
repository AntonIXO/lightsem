import numpy as np
import sys
from types import SimpleNamespace


def test_two_group_stream_holds_last_state_for_terminal_fractional_ticks(monkeypatch):
    from lightsem.experiments.dynamic_dance_combo import colors

    mask = np.zeros(512, bool); mask[:6] = True
    monkeypatch.setattr(colors, 'canonical_slots', lambda: ([np.arange(3), np.arange(3, 6)], mask))
    monkeypatch.setitem(sys.modules, 'lightsem.experiments.visual_next.delivery',
                        SimpleNamespace(color_delivery=lambda zero, before, time, ids, cues, prototypes, layout:
                                        (zero, {'ok': True})))
    scores = {'generated_palette_ids': np.zeros(2, np.int64), 'color_cues': np.zeros(1, bool)}
    companion = np.zeros((512, 512), np.uint8)
    stream, _ = colors.two_group_stream(scores, 10, companion, np.zeros((1, 3), np.float32))
    assert stream.shape[0] == 10
    assert np.all(stream[8:] == stream[7])

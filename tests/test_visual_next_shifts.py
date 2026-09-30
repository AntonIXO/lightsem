import numpy as np
import pytest

from lightsem.experiments.visual_next.shifts import stable_shifted_candidates
from lightsem.model.light_layers.evaluation import shift_plan, shifted_candidates


def test_absolute_clock_endpoint_roundoff_stays_in_half_open_segment():
    plan = shift_plan(98.9 + np.arange(640) / 10, [[98.9, 140.]], 10)
    with pytest.raises(ValueError, match='escaped'):
        shifted_candidates([133.2], [.7], plan)
    result = stable_shifted_candidates([133.2], [.7], plan)
    assert result['draws'][3]['times_s'][0] == np.nextafter(140., 98.9)
    assert result['draws'][3]['half_open_endpoint_roundoff_corrections'] == 1
    for row in result['draws']:
        assert 98.9 <= row['times_s'][0] < 140.
        assert np.array_equal(row['confidences'], [.7])


def test_non_boundary_shifts_are_identical_to_historical_instrument():
    plan = shift_plan(np.arange(640) / 10, [[0., 12.], [15., 30.]], 10)
    times, scores = [1.25, 4.75, 20.25], [.2, .7, .9]
    old = shifted_candidates(times, scores, plan)
    new = stable_shifted_candidates(times, scores, plan)
    for a, b in zip(old['draws'], new['draws']):
        for key, value in a.items():
            assert np.array_equal(value, b[key]), key
        assert b['half_open_endpoint_roundoff_corrections'] == 0


def test_empty_and_unqualified_support_does_not_invent_events():
    plan = shift_plan(np.arange(640) / 10, [[0., 1.]], 10)
    result = stable_shifted_candidates([.5], [.8], plan)
    assert result['draws'] is None
    assert not len(result['real_times_s'])

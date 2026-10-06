"""Counterexamples for telemetry-only float64 clock roundoff admission."""
import hashlib

import numpy as np
import pytest

from lightsem.experiments.dynamic_dance_combo.clock_metadata import nonnegative_effort


def clock_with_reverse(ulps=4, region_count=5):
    a = np.float64(.23651252751929422)
    b = a
    for _ in range(ulps): b = np.nextafter(b, -np.inf)
    values = np.tile(np.array([0., a, b, 1.], dtype=np.float64)[:, None], (1, region_count))
    # Only one region has the numerical reversal.
    values[2, np.arange(region_count) != 2] = a
    return values, np.arange(4, dtype=np.float64)/30


def test_observed_four_ulp_reversal_changes_only_one_display_rate():
    source, time = clock_with_reverse()
    before_source, before_time = source.tobytes(), time.tobytes()
    original = np.diff(source, axis=0)*30
    rates, audit = nonnegative_effort(source, time, 30.)
    assert source.tobytes() == before_source and time.tobytes() == before_time
    assert rates[1, 2] == 0
    keep = original >= 0
    assert rates[:-1][keep].tobytes() == original[keep].tobytes()
    assert audit['negative_interval_count'] == 1
    assert audit['negative_display_entry_count'] == 1
    row = audit['entries'][0]
    assert row['negative_step_ulps'] == 4
    assert row['delta_s'] == -1.1102230246251565e-16
    assert row['original_rate'] == -3.3306690738754696e-15
    assert row['bound_s'] == 2.220446049250313e-16
    assert audit['source_time_bytes_sha256'] == hashlib.sha256(before_source).hexdigest()


@pytest.mark.parametrize('fps', [None, 30., 30.000000000001705])
def test_successful_paths_remain_bit_exact_and_upper_rates_not_changed(fps):
    time = np.arange(100, dtype=np.float64)/30
    source = np.column_stack((time, time*.7, time*1.5000000000034106))
    expected = np.diff(source, axis=0)*fps if fps is not None else np.diff(source, axis=0)/np.diff(time)[:, None]
    expected = np.concatenate((expected, expected[-1:]))
    rates, audit = nonnegative_effort(source, time, fps)
    assert rates.tobytes() == expected.tobytes()
    assert rates.max() > 1.5
    assert audit['status'] == 'unchanged' and audit['entries'] == []
    # Preserve the historical float32 plan serialization too.
    assert rates.astype(np.float32).tobytes() == expected.astype(np.float32).tobytes()


def test_mapper_division_mode_preserves_positive_values():
    source, time = clock_with_reverse()
    original = np.diff(source, axis=0)/np.diff(time)[:, None]
    rates, audit = nonnegative_effort(source, time)
    assert rates[:-1][original >= 0].tobytes() == original[original >= 0].tobytes()
    assert audit['entries'][0]['original_rate'] == original[1, 2]


@pytest.mark.parametrize('ulps', [9, 16, 1024])
def test_greater_than_budget_reversal_rejected(ulps):
    source, time = clock_with_reverse(ulps)
    with pytest.raises(ValueError, match='exceeds local float64'):
        nonnegative_effort(source, time, 30.)


def test_eight_ulp_boundary_admitted():
    source, time = clock_with_reverse(8)
    rates, audit = nonnegative_effort(source, time, 30.)
    assert rates[1, 2] == 0
    assert audit['entries'][0]['negative_step_ulps'] == 8


def test_global_track_extent_cannot_expand_early_local_tolerance():
    source = np.array([0., .2, .2-1e-12, 10000.], dtype=np.float64)
    time = np.arange(4, dtype=np.float64)/30
    with pytest.raises(ValueError, match='exceeds local float64'):
        nonnegative_effort(source, time, 30.)


def test_final_negative_interval_has_two_corrected_display_entries():
    a = np.float64(.25)
    source = np.array([0., a, np.nextafter(a, -np.inf)], dtype=np.float64)
    time = np.arange(3, dtype=np.float64)/30
    rates, audit = nonnegative_effort(source, time, 30.)
    assert rates.shape == (3, 1)
    assert np.all(rates[-2:] == 0)
    assert audit['negative_interval_count'] == 1
    assert audit['negative_display_entry_count'] == 2


@pytest.mark.parametrize('kind', ['float32', 'nonfinite', 'negative_timestamp', 'duplicate_time', 'wrong_fps'])
def test_invalid_clocks_rejected(kind):
    source = np.arange(5, dtype=np.float64)/30
    time = source.copy(); fps = 30.
    if kind == 'float32': source = source.astype(np.float32)
    elif kind == 'nonfinite': source[2] = np.nan
    elif kind == 'negative_timestamp': source[0] = -1e-15
    elif kind == 'duplicate_time': time[2] = time[1]
    else: fps = 40.
    with pytest.raises(ValueError): nonnegative_effort(source, time, fps)

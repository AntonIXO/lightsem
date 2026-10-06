"""Regression for D6's rejected, mathematically bounded float64 clock segment."""
import json
import warnings

import numpy as np
import pytest

from lightsem.experiments.dynamic_dance.retiming import (
    _project_rates,
    _project_regional_rates,
    _regional_mean_roundoff_bound,
    bounded_time_map,
    regional_time_maps,
)


def long_capped_clock():
    time = np.arange(5400)/30.
    fps = 1./float(np.median(np.diff(time)))
    desired = np.r_[np.full(2700,.5),np.full(2700,1.5)]
    clock,rates = bounded_time_map(desired,fps=fps)
    return clock,rates,fps


def test_valid_long_clock_rejects_only_due_to_final_timestamp_roundoff():
    clock,rates,fps = long_capped_clock()
    start,end = len(clock)-2,len(clock)-1
    mean = (clock[end]-clock[start])*fps
    assert rates.min() >= 0. and rates.max() <= 1.5
    assert mean > 1.5+1e-12
    # Retain the exact old rejection; do not relax _project_rates itself.
    with pytest.raises(ValueError,match="Required duration"):
        _project_rates(np.asarray([1.5]),mean)
    bound = _regional_mean_roundoff_bound(clock,start,end,fps)
    assert 0. < mean-1.5 < bound
    with pytest.warns(RuntimeWarning) as recorded:
        projected = _project_regional_rates(np.asarray([1.5]),mean,clock,start,end,fps)
    np.testing.assert_array_equal(projected,_project_rates(np.asarray([1.5]),1.5))
    assert projected.min() >= 0. and projected.max() <= 1.5
    message = json.loads(str(recorded[0].message))
    assert message == {"event":"regional_clock_roundoff_retry", "start":start,
                       "end":end,"rejected_mean":mean,"corrected_mean":1.5,
                       "analytic_ulp_bound":bound,"mean_correction":1.5-mean}


def test_long_regional_clock_preserves_existing_endpoints_and_cap():
    clock,rates,fps = long_capped_clock()
    n = len(clock)
    desired = np.repeat(rates[:,None],5,axis=1)
    desired[:,2] = rates[::-1]
    anchors = np.asarray([0,n-2,n-1])
    original = clock.copy()
    with pytest.warns(RuntimeWarning):
        result = regional_time_maps(desired,clock,fps=fps,anchor_indices=anchors)
    np.testing.assert_array_equal(clock,original)
    np.testing.assert_array_equal(result[anchors],np.repeat(clock[anchors,None],5,axis=1))
    assert np.diff(result,axis=0).min() >= 0.
    # Clock subtraction itself has the stated float64 error; projected physical
    # rates above are still clipped at exactly1.5 and timestamps are unchanged.
    bound = _regional_mean_roundoff_bound(clock,n-2,n-1,fps)
    assert np.diff(result,axis=0).max()*fps <= 1.5+bound


@pytest.mark.parametrize("mean",[-5e-13,0.,.01,.75,1.,1.5,1.5+5e-13])
def test_every_successful_projector_call_is_bitwise_unchanged(mean):
    clock,_,fps = long_capped_clock()
    for rates in (np.asarray([.1]),np.linspace(-3.,4.,17),np.linspace(1.,0.,300)):
        expected = _project_rates(rates,mean)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            actual = _project_regional_rates(rates,mean,clock,1,1+len(rates),fps)
        assert not recorded
        np.testing.assert_array_equal(actual,expected)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mean",[-1e-6,1.5+1e-6,1.5001,2.])
def test_real_duration_infeasibility_is_still_rejected(mean):
    clock,_,fps = long_capped_clock()
    start,end = len(clock)-2,len(clock)-1
    assert abs(mean-np.clip(mean,0.,1.5)) > _regional_mean_roundoff_bound(clock,start,end,fps)
    with warnings.catch_warnings(record=True) as recorded:
        with pytest.raises(ValueError,match="Required duration"):
            _project_regional_rates(np.asarray([1.]),mean,clock,start,end,fps)
    assert not recorded


def test_entire_genuinely_overrate_clock_cannot_be_repaired():
    n = 5400
    clock = np.arange(n)/30.*1.500001
    with pytest.raises(ValueError,match="Required duration"):
        regional_time_maps(np.ones((n,5)),clock,fps=30.,anchor_indices=np.asarray([n-2]))


def test_just_beyond_the_derived_roundoff_bound_is_rejected():
    clock,_,fps = long_capped_clock()
    start,end = len(clock)-2,len(clock)-1
    bound = _regional_mean_roundoff_bound(clock,start,end,fps)
    for mean in (-2.*bound,1.5+2.*bound):
        with pytest.raises(ValueError,match="Required duration"):
            _project_regional_rates(np.asarray([1.]),mean,clock,start,end,fps)


def test_unrelated_projection_validation_errors_are_not_retried():
    clock,_,fps = long_capped_clock()
    for bad in (np.asarray([]),np.asarray([np.nan]),np.asarray([np.inf])):
        with warnings.catch_warnings(record=True) as recorded:
            with pytest.raises(ValueError,match="Rates must be finite and nonempty"):
                _project_regional_rates(bad,1.,clock,0,1,fps)
        assert not recorded

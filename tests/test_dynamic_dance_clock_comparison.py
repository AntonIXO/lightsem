"""Native clock roundoff must not reject or move a fixed review boundary."""
from types import SimpleNamespace
import numpy as np
import pytest
from lightsem.experiments.dynamic_dance.evaluation import _same_clock_and_patch, _body_review_mask


def pair(time):
    return {'patch': [], 'arrays': {'time': np.arange(7200)/40.}, 'body': SimpleNamespace(time=time)}


def test_median_inferred_fps_preserves_native_review_frames():
    original=np.arange(5400)/30.
    fps=1/np.median(np.diff(original))
    reconstructed=np.arange(5400)/fps
    assert reconstructed[1800] < 60
    result=_same_clock_and_patch(pair(original),pair(reconstructed))
    assert not result['native_body_clocks_exact']
    assert 0 < result['maximum_native_clock_difference_s'] < 1e-8
    assert np.array_equal(_body_review_mask(original,60,180),_body_review_mask(reconstructed,60,180))
    assert _body_review_mask(reconstructed,60,180).sum()==3600
    assert np.array_equal(reconstructed,np.arange(5400)/fps)


def test_exclusive_end_remains_excluded_when_rounded_below_boundary():
    time=np.arange(5401)/30.
    rounded=time-1e-11
    assert np.array_equal(_body_review_mask(time,60,180),_body_review_mask(rounded,60,180))


@pytest.mark.parametrize('shift',[1e-7,1e-4,1/30])
def test_material_clock_shift_is_not_admitted(shift):
    time=np.arange(5400)/30.
    with pytest.raises(ValueError,match='body timestamps'):
        _same_clock_and_patch(pair(time),pair(time+shift))


def test_emitted_clock_remains_byte_exact():
    time=np.arange(5400)/30.;a=pair(time);b=pair(time.copy())
    b['arrays']['time'][100]+=1e-12
    with pytest.raises(ValueError,match='emitted timestamps'):
        _same_clock_and_patch(a,b)


def test_shape_mismatch_is_not_broadcast():
    time=np.arange(5400)/30.
    with pytest.raises(ValueError,match='body timestamps'):
        _same_clock_and_patch(pair(time),pair(time[:-1]))

import numpy as np
import pytest
from lightsem.experiments.output_diagnostics import (
    motion_activity, palette_activity, representation_activity, cross_input_motion,
)


def test_constant_motion_and_identical_inputs_are_reported():
    angles=np.zeros((80,6,2))
    report=motion_activity(angles,audio_active=np.zeros(80,bool))
    assert report['constant_motion'] and report['median_speed_deg_s']==0
    assert report['active_median_speed_deg_s'] is None
    assert cross_input_motion(angles,angles.copy())['exactly_identical_angles']
    moved=angles.copy()
    moved[:,:,0]=10*np.sin(np.arange(80)[:,None]/10)
    assert not motion_activity(moved)['constant_motion']
    assert cross_input_motion(angles,moved)['angle_rms_difference_deg']>1


def test_inactive_color_is_distinct_from_collapsed_context():
    scores=np.linspace(.01,.49,80)
    color=palette_activity(np.zeros(80,int),np.zeros(80,bool),scores)
    context=representation_activity(np.random.default_rng(4).normal(size=(80,8)))
    assert color['inactive_learned_color_head'] and color['color_probability_std']>0
    assert not context['constant_context'] and context['effective_covariance_rank']>5


def test_repeated_cues_do_not_count_as_palette_changes():
    cues=np.zeros(80,bool);cues[[10,30,50]]=True
    color=palette_activity(np.ones(80,int),cues,np.full(80,.7))
    assert color['cue_count']==3 and color['subsequent_palette_changes']==0
    assert color['single_palette_after_activation'] and color['active_palette_entropy_bits']==0
    collapsed=representation_activity(np.ones((80,8)))
    assert collapsed['constant_context'] and collapsed['effective_covariance_rank']==0


def test_nonfinite_outputs_fail_instead_of_being_filled():
    angles=np.zeros((80,6,2));angles[10,0,0]=np.nan
    with pytest.raises(ValueError):
        motion_activity(angles)

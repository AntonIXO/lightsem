import numpy as np
from scripts.light_layers.extract_visual_next_cuts import positive_intervals,observation_intervals
from lightsem.experiments.visual_next.instruments import edit_veto


def test_native_edit_intervals_include_last_positive_and_single_frame_cuts():
    t=np.arange(10)/30
    positive=np.array([False,True,False,False,True,True,True,False,False,True])
    x=positive_intervals(t,positive)
    assert np.allclose(x,[[1/30,1/30],[4/30,6/30],[9/30,9/30]])
    assert not edit_veto([6/30+.2],x[:2]).any()
    assert positive_intervals(t,np.zeros(10,bool)).shape==(0,2)


def test_observer_available_runs_do_not_bridge_missing_frames():
    t=np.arange(6)/10
    valid=np.array([True,True,False,True,False,False])
    assert np.allclose(observation_intervals(t,valid),[[0,.2],[.3,.4]])

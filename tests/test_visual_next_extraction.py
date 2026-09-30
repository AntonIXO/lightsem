import numpy as np
from lightsem.experiments.visual_next.extraction import AnchorSchedule,cut_reset_indices,FlowControl
from lightsem.experiments.visual_next.features import spatial_color_grid
from lightsem.model.light_layers.targets import extract_color_targets


def test_missing_anchor_does_not_bridge_a_cut_or_shift_regular_schedule():
    t=np.arange(50)/10
    assert cut_reset_indices(t,[[.1,.2],[2,2]])=={3,21}
    schedule=AnchorSchedule({3})
    assert schedule.step(0,False)==(True,False,True)
    assert schedule.step(1,True)==(False,True,False)
    assert schedule.step(2,True)==(False,False,False)
    assert schedule.step(3,False)==(True,False,True)
    assert schedule.step(4,True)==(False,True,False)
    assert schedule.step(20,False)==(False,False,True)
    assert schedule.step(21,True)==(False,False,False)
    assert schedule.step(40,True)==(False,True,False)


def test_optimized_grid_matches_direct_independent_cell_extraction():
    rng=np.random.default_rng(41);rgb=rng.integers(0,256,(25,31,3),dtype=np.uint8);mask=rng.random((25,31))>.2
    actual=spatial_color_grid(rgb,mask).reshape(16,75);expected=[]
    ys=np.linspace(0,25,5,dtype=int);xs=np.linspace(0,31,5,dtype=int)
    for i in range(4):
        for j in range(4):
            region=np.s_[ys[i]:ys[i+1],xs[j]:xs[j+1]];t=extract_color_targets(rgb[region],mask[region])
            expected.append(np.r_[t.hs_histogram,t.hs_histogram@np.tile(np.arange(6)/5,12),t.chromatic_fraction,t.retained_fraction])
    assert np.allclose(actual,expected,atol=1e-7,rtol=0)


def test_flow_static_mask_agreement_and_unknown_first_frame():
    rgb=np.zeros((90,160,3),np.uint8);rgb[20:60,40:80]=150
    mask=np.zeros((90,160),bool);mask[20:60,40:80]=True
    probe=FlowControl()
    assert np.isnan(probe.step(rgb,mask)).all()
    assert probe.step(rgb,mask)[0]>.99
    wrong=~mask
    assert probe.step(rgb,wrong)[0]<.01

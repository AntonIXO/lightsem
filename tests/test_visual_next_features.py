import numpy as np
from lightsem.experiments.visual_next.features import spatial_color_grid,temporal_arms


def test_grid_retains_layout_and_mask_mass_without_luma_feature():
    a=np.zeros((16,16,3),np.uint8);a[:,:8,0]=255;a[:,8:,1]=255
    mask=np.ones((16,16),bool)
    x=spatial_color_grid(a,mask).reshape(16,75)
    swapped=spatial_color_grid(a[:,::-1],mask).reshape(16,75)
    assert not np.array_equal(x,swapped)
    assert np.allclose(x[:,:72].sum(0),swapped[:,:72].sum(0))
    assert np.allclose(x,spatial_color_grid(a//2,mask).reshape(16,75))
    mask[:4,:4]=False
    assert np.all(spatial_color_grid(a,mask).reshape(16,75)[0]==0)


def test_temporal_controls_preserve_current_and_invalidate_missing_history():
    a=np.arange(15,dtype=np.float32)[:,None];valid=np.ones(15,bool)
    arms,support=temporal_arms(a,valid,'fixed_family')
    assert not support[:5].any() and support[5:].all()
    assert np.array_equal(arms['R3'][5],[5,1,5])
    assert np.array_equal(arms['R3-reverse'][5],[5,5,1])
    assert all(np.array_equal(value[5:,0],a[5:,0]) for value in arms.values())
    again,_=temporal_arms(a,valid,'fixed_family')
    assert np.array_equal(arms['R3-shuffle'],again['R3-shuffle'])
    valid[6]=False
    with_gap,s=temporal_arms(a,valid,'fixed_family')
    assert not s[6:12].any() and s[12:].all()
    assert np.array_equal(with_gap['R3-shuffle'][12:],arms['R3-shuffle'][12:])

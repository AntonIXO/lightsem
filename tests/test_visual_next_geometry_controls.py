import numpy as np
from lightsem.model.light_layers.targets import extract_color_targets
from lightsem.experiments.visual_next.instruments import hs_distance
from lightsem.experiments.visual_next.features import spatial_color_grid


def test_moving_mask_can_create_false_color_change_on_a_static_scene():
    rgb=np.zeros((20,20,3),np.uint8);rgb[:,:10,0]=255;rgb[:,10:,2]=255
    left=np.ones((20,20),bool);left[:,:8]=False
    right=np.ones_like(left);right[:,12:]=False
    a=extract_color_targets(rgb,left);b=extract_color_targets(rgb,right)
    assert a.retained_pixels==b.retained_pixels and hs_distance(a.hs_histogram,b.hs_histogram)>.4
    # Fixed mask on duplicated frames is exactly unchanged.
    assert hs_distance(a.hs_histogram,extract_color_targets(rgb.copy(),left).hs_histogram)==0


def test_camera_permutation_preserves_global_color_but_changes_layout():
    rgb=np.zeros((32,32,3),np.uint8);rgb[:,:16,0]=255;rgb[:,16:,1]=255;mask=np.ones((32,32),bool)
    moved=np.roll(rgb,8,axis=1)
    assert hs_distance(extract_color_targets(rgb,mask).hs_histogram,extract_color_targets(moved,mask).hs_histogram)==0
    assert not np.array_equal(spatial_color_grid(rgb,mask),spatial_color_grid(moved,mask))


def test_color_flash_and_edit_are_not_distinguished_by_histogram_alone():
    mask=np.ones((8,8),bool)
    red=np.tile(np.array([255,0,0],np.uint8),(8,8,1));blue=np.tile(np.array([0,0,255],np.uint8),(8,8,1))
    r,b=extract_color_targets(red,mask),extract_color_targets(blue,mask)
    assert hs_distance(r.hs_histogram,b.hs_histogram)==1
    # Whether the same pixel transition is an edit or an authored color cue
    # cannot be inferred from this endpoint alone; it needs separate evidence.
    assert hs_distance(b.hs_histogram,r.hs_histogram)==1

import numpy as np
from lightsem.experiments.visual_next.delivery import color_delivery
from lightsem.experiments.visual_next.palettes import fixture_groups,group_rgb_mask
from lightsem.conductor.rig import canonical_rig


def test_no_cue_does_not_reuse_the_learned_C_colors():
    rng=np.random.default_rng(8);c=rng.integers(0,256,(80,512),dtype=np.uint8);companion=np.zeros_like(c)
    prototypes=np.zeros((32,72));prototypes[:,5]=1
    stream,report=color_delivery(c,companion,np.arange(80)/40,np.zeros(20,int),np.zeros(20,bool),prototypes)
    rgb=group_rgb_mask(fixture_groups(canonical_rig()))
    assert np.all(stream[:,rgb]==0)
    assert np.array_equal(stream[:,~rgb],c[:,~rgb])
    assert report['active_color_seconds']==0


def test_candidate_fade_preserves_nonRGB_and_uses_model_activation():
    c=np.full((80,512),100,np.uint8);companion=np.zeros_like(c)
    prototypes=np.zeros((32,72));prototypes[:,5]=1
    cues=np.zeros(20,bool);cues[5]=True
    stream,report=color_delivery(c,companion,np.arange(80)/40,np.zeros(20,int),cues,prototypes)
    slots=fixture_groups(canonical_rig())[0]['rgb_slots_zero_based']
    assert np.all(stream[20,slots]==0)
    assert np.array_equal(stream[32,slots],[255,0,0])
    assert report['color_cue_count']==1 and report['fade_s']==.3

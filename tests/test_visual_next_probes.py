import numpy as np
from lightsem.experiments.visual_next.probes import controlled_background,LUMA,family_probe_skill


def test_controlled_hue_holds_luma_and_preserves_foreground():
    rng=np.random.default_rng(8);rgb=rng.integers(0,256,(20,24,3),dtype=np.uint8)
    mask=np.ones((20,24),bool);mask[5:12,8:15]=False
    for turn in (0,1/24,2/24,4/24):
        out=controlled_background(rgb,mask,turn)
        assert np.array_equal(out[~mask],rgb[~mask])
        assert np.max(np.abs(out[mask]/255@LUMA-.3))<=.5/255+1e-6
    normal=controlled_background(rgb,mask);mirror=controlled_background(rgb,mask,mirror=True)
    assert np.array_equal(np.sort(normal[mask],axis=0),np.sort(mirror[mask],axis=0))


def test_probe_recovers_linear_information_but_detects_constant_collapse():
    rng=np.random.default_rng(11);x=rng.normal(size=(90,4));y=x@rng.normal(size=(4,2))
    groups=np.repeat(['a','b','c'],30)
    good=family_probe_skill(x,y,groups);bad=family_probe_skill(np.ones_like(x),y,groups)
    assert good['baseline_corrected_skill']>.999
    assert abs(bad['baseline_corrected_skill'])<1e-10

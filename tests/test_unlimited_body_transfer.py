import numpy as np
from lightsem.conductor.body_transfer import compile_intent,TransferConfig,decode
from lightsem.conductor.body_transfer_v2 import make_outputs

def test_unlimited_bypasses_all_temporal_projection(monkeypatch):
    import lightsem.experiments.motion_study.offline_angles as projector
    monkeypatch.setattr(projector,'project_angles',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Temporal projector called')))
    outputs=make_outputs(beams=1,lasers=0,bulbs=0);n=20;pan=np.tile([-35.,35.],10);direction=np.stack([np.sin(np.deg2rad(pan)),-np.cos(np.deg2rad(pan)),np.zeros(n)],-1)
    targets=direction[:,None]*10+np.array(outputs[0]['position']);levels=np.full((n,1),.12);left=np.tile(np.array([20,70,150],np.uint8),(n,1));right=left.copy()
    dm,patch,_=compile_intent({'targets':targets,'dimmer':levels},outputs,left,right,TransferConfig(apply_motion_limits=False))
    value=decode(dm,patch);tick=patch[0]['pan_range_deg']/65535
    np.testing.assert_allclose(value['angles_deg'][:,0,0],pan,atol=tick/2)
    assert np.max(np.abs(np.diff(value['angles_deg'][:,0,0])*40))>90
    np.testing.assert_array_equal(np.rint(value['rgb'][:,0]*255).astype('u1'),left)
    np.testing.assert_array_equal(np.rint(value['dimmer'][:,0]*255),np.rint(levels[:,0]*255))

def test_default_stays_explicitly_limited():
    assert TransferConfig().apply_motion_limits is True

def test_unlimited_frozen_targets_and_output_count_preserve_timing():
    for count in (1,13,67):
        outputs=make_outputs(beams=count,lasers=0,bulbs=0)
        n=12;targets=np.zeros((n,count,3));targets[:]=[0,1,-3]
        # Only one assigned intent changes, at one exact timestamp.
        targets[6:,0,0]+=1
        colors=np.zeros((n,3),dtype='u1')
        dmx,patch,_=compile_intent({'targets':targets,'dimmer':np.ones((n,count))},outputs,colors,colors,TransferConfig(apply_motion_limits=False))
        angles=decode(dmx,patch)['angles_deg']
        changed=np.any(np.diff(angles,axis=0)!=0,axis=-1)
        assert np.array_equal(np.flatnonzero(changed[:,0]),[5])
        assert not changed[:,1:].any()
        assert np.array_equal(angles[:6],np.broadcast_to(angles[0],angles[:6].shape))

def test_unlimited_numeric_admission_does_not_reimpose_hardware_caps():
    from scripts.motion.run_body_portfolio import hard_checks
    import pytest
    outputs=make_outputs(beams=1,lasers=0,bulbs=0);n=12
    targets=np.tile(np.array([[[0.,1.,-3.]]]),(n,1,1));targets[6:,0,0]=5
    result={'targets':targets,'dimmer':np.ones((n,1))};rgb=np.zeros((n,3),dtype='u1')
    dmx,patch,intent=compile_intent(result,outputs,rgb,rgb,TransferConfig(apply_motion_limits=False))
    with pytest.raises(RuntimeError,match='mechanical limits'):hard_checks(dmx,patch,intent,40)
    report=hard_checks(dmx,patch,intent,40,enforce_motion_limits=False)
    assert report['finite'] and not report['mechanical_limits_enforced']
    assert report['maximum_speed_deg_s']>90

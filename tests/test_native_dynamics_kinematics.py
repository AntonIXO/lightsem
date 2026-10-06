import numpy as np
import torch
from lightsem.experiments.native_dynamics.kinematics import observed_dynamics,dynamics_losses,coarse_targets,masked_quantiles,project_native,TORSO_METRES

def test_standing_support_does_not_imply_wholebody_hold():
    t=torch.arange(300)/30;p=torch.zeros(1,300,13,2);p[0,:,5,0]=t*.5;m=torch.ones(1,300,13,dtype=torch.bool)
    d=observed_dynamics(p,m)
    assert d['low'][0,50:250,0].all() # support is still
    assert not d['low'][0,50:250,3].any() # left wrist moves
    assert not d['low'][0,50:250,5].any() # whole-body stop must remain false
    torch.testing.assert_close(d['speed'][0,50:250,3],torch.full((200,),.5/3),atol=1e-4,rtol=0)

def test_invalid_gap_never_teaches_zero_velocity():
    p=torch.zeros(1,300,13,2);m=torch.ones(1,300,13,dtype=torch.bool);m[:,100:120]=False
    d=observed_dynamics(p,m)
    assert not d['speed_valid'][:,96:124].any()
    assert not d['low_valid'][:,100:120].any()

def test_empty_observations_give_finite_zero_loss_and_gradient():
    g=torch.randn(2,180,13,2,requires_grad=True);t=torch.randn_like(g);m=torch.zeros(2,180,13,dtype=torch.bool)
    losses=dynamics_losses(g,t,m);loss=sum(losses.values());assert loss.item()==0;loss.backward();assert torch.isfinite(g.grad).all();assert (g.grad==0).all()

def test_correct_motion_beats_frozen_and_time_shifted_controls():
    time=torch.arange(480)/30
    pose=torch.zeros(1,480,13,2);pose[0,:,5,0]=torch.sin(time*2)*torch.sigmoid((time-6)*2)
    m=torch.ones(1,480,13,dtype=torch.bool)
    exact=dynamics_losses(pose,pose,m);frozen=dynamics_losses(torch.zeros_like(pose),pose,m);shifted=dynamics_losses(torch.roll(pose,120,1),pose,m)
    assert exact['profile']<frozen['profile'];assert exact['profile']<shifted['profile']
    assert exact['speed_distribution']<frozen['speed_distribution']

def test_coarse_targets_are_motion_only_and_keep_region_separation():
    p=np.zeros((300,13,2),np.float32);p[:,5,0]=np.arange(300)/30*.5;m=np.ones((300,13),bool)
    d=coarse_targets(p,m);assert d['plan'].shape==(20,48);assert d['plan_valid'].shape==(20,6,3)
    z=d['plan'].reshape(20,6,8);assert (z[3:17,0,3]==1).all();assert (z[3:17,5,3]==0).all()

def test_masked_quantile_empty_group_never_returns_inf():
    x=torch.randn(2,60,6,requires_grad=True);m=torch.ones_like(x,dtype=torch.bool);m[:,:,0]=False
    q,v=masked_quantiles(x,m);assert torch.isfinite(q).all();assert not v[:,0].any();q.sum().backward();assert torch.isfinite(x.grad).all()

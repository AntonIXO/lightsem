"""Loss-only adversarial signals; never a generation or rendering source."""
import torch
from lightsem.experiments.native_repair.losses import raw_world_losses

def sequence():
    x=torch.zeros(1,180,22,3);x[...,1]=1.;return x,torch.ones(1,180,dtype=torch.bool)

def test_identity_reference_is_unpenalized():
    x,m=sequence();loss=raw_world_losses(x,x,m)
    assert all(float(v)==0 for v in loss.values())

def test_fast_jitter_cannot_hide_behind_observation_smoothing():
    r,m=sequence();t=torch.arange(180)/30
    low=r.clone();high=r.clone();low[0,:,20,0]+=.02*torch.sin(2*torch.pi*1*t);high[0,:,20,0]+=.02*torch.sin(2*torch.pi*8*t)
    a=raw_world_losses(low,r,m);b=raw_world_losses(high,r,m)
    assert b['raw_high_frequency']>a['raw_high_frequency']*4
    assert b['raw_jerk']>a['raw_jerk']*4

def test_root_height_nullspace_has_a_training_signal():
    r,m=sequence();g=r.clone();g[...,1]+=.1;g.requires_grad_();loss=raw_world_losses(g,r,m)
    assert loss['root_height_prior']>3.9
    sum(loss.values()).backward();assert torch.isfinite(g.grad).all();assert g.grad.abs().sum()>0

def test_invalid_padding_cannot_become_motion_target():
    r,m=sequence();m[:,-30:]=False;g=r.clone();a=raw_world_losses(g,r,m);g[:,-30:]=9999.;b=raw_world_losses(g,r,m)
    for k in a:torch.testing.assert_close(a[k],b[k])

def test_jitter_gradient_is_finite():
    r,m=sequence();g=(r+torch.randn_like(r)*.003).requires_grad_();sum(raw_world_losses(g,r,m).values()).backward()
    assert torch.isfinite(g.grad).all() and g.grad.abs().sum()>0

def test_reference_rollout_uses_real_legacy_D7_API():
    from lightsem.experiments.dynamic_dance.learning import DynamicFlow,DynamicFlowConfig
    from lightsem.experiments.motion_study.music_model import MusicConfig
    from lightsem.experiments.native_repair.training import reference_rollout
    torch.manual_seed(3)
    model=DynamicFlow(MusicConfig(arm='flow',width=16,dilations=(1,2)),{'mean':[0.]*139,'std':[1.]*139,'delta_scale':[1.]*139},DynamicFlowConfig(1061,arm='D7')).eval()
    torch.nn.init.normal_(model.output.weight,std=.01)
    condition=torch.randn(1,30,1233);noise=torch.randn(1,30,139)
    audio=torch.cat((condition[...,:1153],torch.zeros(1,30,34)),-1)
    expected=model.generate(audio,steps=32,chunked=False,initial_noise=noise)
    actual=reference_rollout(model,condition,noise)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert not actual.requires_grad

import torch
from lightsem.experiments.native_dynamics.model import Config,NativeDynamicsModel,plan_at_frames,planner_loss
from lightsem.experiments.dynamic_dance.learning import DynamicFlow,DynamicFlowConfig
from lightsem.experiments.motion_study.music_model import MusicConfig

torch.set_num_threads(1)
def models(arm='joint'):
    cfg=MusicConfig(width=16,frames=48,dilations=(1,2),steps=32);norm={'mean':[0.]*139,'std':[1.]*139,'delta_scale':[1.]*139}
    teacher=DynamicFlow(cfg,norm,DynamicFlowConfig(10,arm='D7'))
    torch.nn.init.normal_(teacher.output.weight,std=.01)
    native=NativeDynamicsModel(cfg,norm,Config(arm=arm,extra_dim=10,planner_width=16,planner_heads=4,frames=48))
    native.warmstart(teacher.state_dict());return teacher,native

def test_zero_adapters_preserve_existing_generator_exactly():
    teacher,model=models();fine=torch.randn(2,48,102);coarse=torch.randn(2,8,10);valid=torch.ones(2,8,dtype=torch.bool);idx=torch.arange(48)[None].expand(2,-1)
    packed,plan=model.condition(fine,coarse,valid,idx);old=torch.cat([fine,torch.zeros(2,48,34)],-1);state=torch.randn(2,48,139);t=torch.tensor([.2,.7])
    torch.testing.assert_close(model.flow.vector_field(packed,state,t),teacher.vector_field(old,state,t),rtol=0,atol=0)

def test_zero_temporal_adapter_preserves_native_vector_field_exactly():
    _,model=models(); cfg=model.config
    base_cfg=MusicConfig(width=16,frames=48,dilations=(1,2),steps=32)
    norm={'mean':[0.]*139,'std':[1.]*139,'delta_scale':[1.]*139}
    temporal=NativeDynamicsModel(base_cfg,norm,
        Config(**{**cfg.__dict__,'temporal_adapter':True,'temporal_adapter_scale':.1}))
    loaded=temporal.load_state_dict(model.state_dict(),strict=False)
    assert loaded.unexpected_keys==[]
    assert loaded.missing_keys and all(k.startswith('flow.temporal_adapter.') for k in loaded.missing_keys)
    fine=torch.randn(1,48,102);coarse=torch.randn(1,8,10);valid=torch.ones(1,8,dtype=torch.bool);idx=torch.arange(48)[None]
    cond,_=model.condition(fine,coarse,valid,idx);temporal_cond,_=temporal.condition(fine,coarse,valid,idx)
    state=torch.randn(1,48,139);time=torch.tensor([.4])
    torch.testing.assert_close(model.flow.vector_field(cond,state,time),temporal.flow.vector_field(temporal_cond,state,time),rtol=0,atol=0)

def test_learned_temporal_adapter_has_gradients_and_preserves_chunk_boundaries():
    _,base=models()
    norm={'mean':[0.]*139,'std':[1.]*139,'delta_scale':[1.]*139}
    model=NativeDynamicsModel(base.flow.config,norm,
        Config(**{**base.config.__dict__,'temporal_adapter':True}))
    model.load_state_dict(base.state_dict(),strict=False)
    assert model.flow.halo == base.flow.halo + 2
    for name,parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('flow.temporal_adapter.'))
    with torch.no_grad():
        model.flow.temporal_adapter.weight.normal_(std=.3)
    fine=torch.randn(1,96,102);coarse=torch.randn(1,8,10)
    valid=torch.ones(1,8,dtype=torch.bool);idx=torch.arange(96)[None]
    cond,_=model.condition(fine,coarse,valid,idx)
    state=torch.randn(1,96,139);time=torch.tensor([.5])
    field=model.flow.vector_field(cond,state,time)
    field.diff(n=3,dim=1).square().mean().backward()
    gradient=model.flow.temporal_adapter.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    with torch.no_grad():
        chunked=model.flow.vector_field_chunked(cond,state,time,core_frames=13)
        torch.testing.assert_close(field,chunked,rtol=1e-5,atol=1e-7)
        noise=model.flow.noise(1,96,generator=torch.Generator().manual_seed(7001))
        whole,_=model.generate(fine,coarse,valid,idx,noise,core_frames=960)
        chunk,_=model.generate(fine,coarse,valid,idx,noise,core_frames=13)
        torch.testing.assert_close(whole,chunk,rtol=1e-5,atol=1e-5)

def test_training_rollout_and_inference_share_actual32step_forward():
    teacher,model=models();fine=torch.randn(1,48,102);cond,_=model.condition(fine,torch.randn(1,8,10),torch.ones(1,8,dtype=torch.bool),torch.arange(48)[None]);noise=model.flow.noise(1,48,generator=torch.Generator().manual_seed(7001))
    train,prior=model.rollout(cond,noise,gradient_steps=4,teacher=teacher);infer,_=model.rollout(cond,noise,gradient_steps=0)
    torch.testing.assert_close(train,infer,atol=0,rtol=0)
    (train.square().mean()+prior).backward();assert model.flow.output.weight.grad is not None;assert torch.isfinite(model.flow.output.weight.grad).all()

def test_frame_conditioning_never_reads_padded_tail_plan():
    p=torch.ones(1,3,48);p[:,1]=3;p[:,2]=100
    out=plan_at_frames(p,torch.tensor([[True,True,False]]),torch.tensor([[0,7,22,1000]]))
    torch.testing.assert_close(out[0,:,0],torch.tensor([1.,1.,3.,3.]))

def test_no_motion_targets_are_generator_inputs():
    _,model=models();fine=torch.randn(1,48,102);coarse=torch.randn(1,8,10);valid=torch.ones(1,8,dtype=torch.bool);idx=torch.arange(48)[None]
    a,_=model.condition(fine,coarse,valid,idx);b,_=model.condition(fine,coarse,valid,idx);assert torch.equal(a,b)
    assert model.config_dict()['time_axis']=='original30Hz songtime, no retiming/recomposition/floor'

def test_planner_targets_masked_without_nan_or_false_stop_labels():
    _,model=models();plan=model.planner(torch.randn(2,8,10),torch.ones(2,8,dtype=torch.bool));target=torch.zeros(2,8,48);valid=torch.zeros(2,8,6,3,dtype=torch.bool)
    loss,_=planner_loss(plan,target,valid,torch.zeros(2,8,26),torch.zeros(2,8,dtype=torch.bool));assert loss.item()==0;loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.planner.parameters())


def test_native_pose_gradient_reaches_both_dynamics_and_motif_heads():
    _,model=models()
    for layer in model.flow.native_adapters:torch.nn.init.normal_(layer.weight,std=.02)
    fine=torch.randn(1,48,102);cond,_=model.condition(fine,torch.randn(1,8,10),torch.ones(1,8,dtype=torch.bool),torch.arange(48)[None])
    state=torch.randn(1,48,139);loss=model.flow.vector_field(cond,state,torch.tensor([.5])).square().mean();loss.backward()
    for layer in [model.planner.output,model.planner.recurrence]:
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all() and layer.weight.grad.abs().sum()>0

@torch.inference_mode()
def test_full_generation_matches_training_forward_and_chunk_halo():
    _,model=models();model.eval()
    for layer in model.flow.native_adapters:torch.nn.init.normal_(layer.weight,std=.02)
    fine=torch.randn(1,48,102);coarse=torch.randn(1,8,10);valid=torch.ones(1,8,dtype=torch.bool);idx=torch.arange(48)[None]
    noise=model.flow.noise(1,48,generator=torch.Generator().manual_seed(7001))
    cond,_=model.condition(fine,coarse,valid,idx)
    forward,_=model.rollout(cond,noise,gradient_steps=0)
    whole,_=model.generate(fine,coarse,valid,idx,noise,core_frames=960)
    chunk,_=model.generate(fine,coarse,valid,idx,noise,core_frames=17)
    torch.testing.assert_close(forward,whole,rtol=0,atol=0)
    torch.testing.assert_close(whole,chunk,rtol=1e-5,atol=1e-5)

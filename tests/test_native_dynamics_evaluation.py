import numpy as np
import torch
from lightsem.experiments.native_dynamics.evaluation import metrics,bootstrap_paired,metric_controls

def motion():
    t=np.arange(480)/30
    q=np.zeros((480,13,2),np.float32)
    for j in range(13):q[:,j,0]=(.1+j*.02)*np.sin(t*(1+j*.13)+.3*np.sin(t*.4));q[:,j,1]=j*.03+.1*np.cos(t*.7+j)
    return q,np.ones((480,13),bool)

def test_identity_and_static_cannot_tie():
    torch.set_num_threads(1);q,m=motion();same,_=metrics(q,q,m);static,_=metrics(np.broadcast_to(q[:1],q.shape).copy(),q,m)
    assert same['profile_error']==0 and static['profile_error']>.01
    assert same['distribution_error']==0 and static['constant_pose']

def test_macro_bootstrap_pairs_and_magnitude():
    before=np.linspace(.1,.3,24);after=before*.9
    result=bootstrap_paired(before,after)
    assert result['magnitude_pass'] and result['ci95_delta'][1]<0
    assert not bootstrap_paired(before,before*.99)['magnitude_pass']

def test_temporal_metric_detects_distribution_preserving_shift():
    torch.set_num_threads(1);q,m=motion();controls=metric_controls(q,m)
    assert controls['observed_identity']['profile_error']==0
    for k in ('constant_pose','shifted_observation','oversmoothed_observation','constant_nonzero_speed_trace'):assert controls[k]['profile_error']>0

def test_no_false_global_hold_when_wrist_moves():
    q=np.zeros((480,13,2),np.float32);q[:,5,0]=np.arange(480)/30
    measured,_=metrics(q,q,np.ones((480,13),bool))
    assert measured['false_hold_denominator']>0 and measured['false_hold_count']==0

def test_cpu_noise_matches_original_sampler_exactly():
    from lightsem.experiments.native_dynamics.evaluation import cpu_noise
    from lightsem.experiments.dynamic_dance.learning import DynamicFlow,DynamicFlowConfig
    from lightsem.experiments.motion_study.music_model import MusicConfig
    norm={'mean':[0.]*139,'std':[1.]*139,'delta_scale':[1.]*139}
    model=DynamicFlow(MusicConfig(width=16,frames=48,dilations=(1,2),steps=32),norm,DynamicFlowConfig(10,arm='D7'))
    expected=model.noise(2,96,generator=torch.Generator(device='cpu').manual_seed(7001),device='cpu')
    actual=cpu_noise(model,96,7001,batch=2)
    assert torch.equal(expected,actual) and actual.device.type=='cpu'

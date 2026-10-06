"""Raw generated-motion constraints complement noise-robust observed2D losses.

The detached reference is paired FineDance3D, or the frozen D7 sample for the
same JD audio/noise. The latter is a prior, never labeled JD3D ground truth.
No hardware speed caps and no changes to the generated output at inference.
"""
import torch
import torch.nn.functional as F
from ..motion_study.music_physical import motion_kinematics
from ..native_dynamics.kinematics import masked_mean

WEIGHTS={'raw_high_frequency':1.,'raw_jerk':.5,'root_height_prior':.5,'support_velocity':.10,'floor_prior':1.}

def smooth(x):
    b,t,j,c=x.shape;grid=torch.arange(-4,5,device=x.device,dtype=x.dtype)
    k=torch.exp(-.5*(grid/1.5).square());k=k/k.sum()
    return F.conv1d(x.permute(0,2,3,1).reshape(b*j*c,1,t),k[None,None],padding=4).reshape(b,j,c,t).permute(0,3,1,2)

def window_excess(g,r,mask,floor):
    # 0.5s windows; the raw signal, not the low-pass signal, is scored.
    width=min(15,g.shape[1]);step=5
    gm=g.unfold(1,width,step).square().sum(3).mean(-1).add(1e-16).sqrt()
    rm=r.unfold(1,width,step).square().sum(3).mean(-1).add(1e-16).sqrt()
    good=mask.unfold(1,width,step).all(-1)
    # Reference p90 permits gestures to move in time, while local reference
    # preserves already faster windows. Invalid windows do not set a budget.
    values=torch.where(good[...,None],rm,torch.nan)
    global_budget=torch.nanquantile(values,.9,dim=1,keepdim=True).nan_to_num(0.)
    budget=torch.maximum(rm,global_budget)
    excess=F.relu(torch.log((gm+floor)/(1.1*budget+floor))).square()
    return masked_mean(excess,good[...,None])

def raw_world_losses(g,r,valid):
    r=r.detach();valid=valid.bool();zero=g.sum()*0
    if g.shape[1]<16:return {k:zero for k in WEIGHTS}
    hf_mask=F.conv1d(valid[:,None].float(),g.new_ones((1,1,9)),padding=4)[:,0]==9
    high=window_excess(g-smooth(g),r-smooth(r),hf_mask,.001)
    dg,dr=g,r;edges=valid
    for _ in range(3):dg=torch.diff(dg,dim=1)*30;dr=torch.diff(dr,dim=1)*30;edges=edges[:,:-1]&edges[:,1:]
    jerk=window_excess(dg,dr,edges,10.)
    height=masked_mean(((g[:,:,0,1]-r[:,:,0,1])/.05).square(),valid)
    feet=[7,8,10,11];gv=torch.diff(g[:,:,feet],dim=1)*30;rv=torch.diff(r[:,:,feet],dim=1)*30
    edge=valid[:,:-1]&valid[:,1:]
    thresholds=g.new_tensor([.12,.12,.055,.055]);supported=(r[:,:-1,feet,1]<thresholds)&(r[:,1:,feet,1]<thresholds)&(rv.norm(dim=-1)<.20)&edge[...,None]
    support=masked_mean(((gv-rv)/.20).square().sum(-1),supported)
    # Do not improve a floor prior by translating both body and reference.
    penetration=F.relu(-g[...,1]-.02);reference_penetration=F.relu(-r[...,1]-.02)
    floor=masked_mean((F.relu(penetration-reference_penetration)/.05).square(),valid[...,None])
    return {'raw_high_frequency':high,'raw_jerk':jerk,'root_height_prior':height,'support_velocity':support,'floor_prior':floor}

def raw_native_losses(generated,reference,valid):
    with torch.autocast(device_type=generated.device.type,enabled=False):
        g=motion_kinematics(generated.float(),valid)['positions']
        with torch.no_grad():r=motion_kinematics(reference.float(),valid)['positions']
        return raw_world_losses(g,r,valid)

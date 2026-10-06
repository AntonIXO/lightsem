"""Differentiable observed-time dynamics; masks never become zero-speed targets."""
import numpy as np
import torch
import torch.nn.functional as F
from ..motion_study.geometry import OFFSETS,PARENTS
from ..motion_study.music_physical import motion_kinematics
from .data import SMPL,REGIONS

FPS=30.
GROUPS=tuple(REGIONS)
PROFILE_WINDOWS=(6,30,120)
SPEED_QUANTILES=(.1,.5,.9)
HOLD_THRESHOLD=.10  # projected torso-lengths/s, NOT metric m/s or musical labels
HOLD_TEMPERATURE=.03
PLAN_DIM=48  # 6 regions * (3 log-speed quantiles + low-speed fraction +4 duration bins)

_rest=np.zeros((22,3),float)
for _j,_p in enumerate(PARENTS[1:],1):_rest[_j]=_rest[_p]+OFFSETS[_j]
TORSO_METRES=float(np.linalg.norm(_rest[[16,17],:2].mean(0)-_rest[[1,2],:2].mean(0)))


def project_native(native,valid):
    """Canonical frontal projection in fixed torso units; no fitted dynamic camera."""
    body=motion_kinematics(native,valid)
    points=body['positions'][:,:,SMPL,:2]
    pelvis=body['positions'][:,:,[1,2],:2].mean(-2,keepdim=True)
    return (points-pelvis)/TORSO_METRES,body


def masked_mean(x,mask):
    mask=mask.expand_as(x)
    return torch.where(mask,x,torch.zeros_like(x)).sum()/mask.sum().clamp_min(1)


def box_average(values,mask,width):
    """B,T,R masked averages on original frame positions; no timewarp."""
    b,t,r=values.shape;x=values.transpose(1,2).reshape(b*r,1,t);m=mask.transpose(1,2).reshape(b*r,1,t).float()
    pad=(width//2,(width-1)//2);kernel=x.new_ones((1,1,width))
    total=F.conv1d(F.pad(torch.where(m.bool(),x,0.),pad),kernel)
    count=F.conv1d(F.pad(m,pad),kernel)
    mean=total/count.clamp_min(1)
    good=(count>=.8*width)
    return mean.reshape(b,r,t).transpose(1,2),good.reshape(b,r,t).transpose(1,2)


def smooth_pose(pose,mask):
    b,t,j,_=pose.shape;grid=torch.arange(-4,5,device=pose.device,dtype=pose.dtype);kernel=torch.exp(-.5*(grid/1.5)**2);kernel=kernel/kernel.sum()
    flat=torch.where(mask[...,None],pose,0.).permute(0,2,3,1).reshape(b*j*2,1,t)
    sm=F.conv1d(flat,kernel[None,None],padding=4).reshape(b,j,2,t).permute(0,3,1,2)
    count=F.conv1d(mask.permute(0,2,1).reshape(b*j,1,t).float(),pose.new_ones((1,1,9)),padding=4)
    reliable=(count.reshape(b,j,t).transpose(1,2)==9)
    return torch.where(reliable[...,None],sm,0.),reliable


def observed_dynamics(pose,mask):
    """Same projection/measurement for generated and observed poses.

    Smoothing is a measurement frontend (sigma50ms), never applied to saved
    generated bodies or their DMX. Every derivative needs valid endpoints.
    """
    pose=pose.float();mask=mask.bool();q,valid=smooth_pose(pose,mask)
    velocity=torch.diff(q,dim=1)*FPS;edge=valid[:,:-1]&valid[:,1:]
    velocity=torch.where(edge[...,None],velocity,0.)
    acceleration=torch.diff(velocity,dim=1)*FPS;triple=edge[:,:-1]&edge[:,1:]
    acceleration=torch.where(triple[...,None],acceleration,0.)
    speed=torch.sqrt(velocity.square().sum(-1)+1e-12)
    accel=torch.sqrt(acceleration.square().sum(-1)+1e-12)
    group_speed=[];group_peak=[];group_accel=[];speed_masks=[];accel_masks=[]
    for ids in REGIONS.values():
        group_speed.append(speed[:,:,ids].mean(-1));group_peak.append(speed[:,:,ids].amax(-1));speed_masks.append(edge[:,:,ids].all(-1))
        group_accel.append(accel[:,:,ids].mean(-1));accel_masks.append(triple[:,:,ids].all(-1))
    s=F.pad(torch.stack(group_speed,-1),(0,0,0,1));a=F.pad(torch.stack(group_accel,-1),(0,0,0,2))
    sm=F.pad(torch.stack(speed_masks,-1),(0,0,0,1));am=F.pad(torch.stack(accel_masks,-1),(0,0,0,2))
    profiles=[];pm=[]
    for width in PROFILE_WINDOWS:
        p,m=box_average(s,sm,width);profiles.append(p);pm.append(m)
    profiles=torch.stack(profiles,-1);pm=torch.stack(pm,-1)
    peak=F.pad(torch.stack(group_peak,-1),(0,0,0,1));peak_profile,peak_mask=box_average(peak,sm,6)
    low=(peak_profile<HOLD_THRESHOLD)&peak_mask
    return {'speed':s,'speed_valid':sm,'acceleration':a,'acceleration_valid':am,'profiles':profiles,'profile_valid':pm,'low':low,'low_valid':peak_mask,'peak_profile':peak_profile,'pose_smoothed':q,'pose_valid':valid}


def masked_quantiles(values,mask,quantiles=(.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95)):
    # values B,T,R -> B,R,Q. Sorting itself is differentiable; no infinities
    # reach nonlinear losses when a region has no observed samples.
    x=values.transpose(1,2);m=mask.transpose(1,2);counts=m.sum(-1)
    sorted_values=torch.where(m,x,torch.full_like(x,float('inf'))).sort(-1).values
    qs=x.new_tensor(quantiles);indices=((counts-1).clamp_min(0)[...,None]*qs).long()
    selected=sorted_values.gather(-1,indices)
    good=counts>=4;selected=torch.where(good[...,None],selected,0.)
    return selected,good


def coarse_targets(pose,mask):
    """Offline motion-only 2Hz targets and masked pose context for recurrence.

    Duration labels describe observed low/high-speed bouts; invalid boundaries
    censor durations. They are not audio-derived semantic pause labels.
    """
    with torch.no_grad():
        p=torch.as_tensor(pose,dtype=torch.float32)[None];m=torch.as_tensor(mask,dtype=torch.bool)[None]
        dyn=observed_dynamics(p,m);n=len(pose);bins=(n+14)//15;pad=bins*15-n
        speed=F.pad(dyn['speed'],(0,0,0,pad)).reshape(bins,15,6)
        valid=F.pad(dyn['speed_valid'],(0,0,0,pad)).reshape(bins,15,6)
        qs,qgood=masked_quantiles(speed,valid,SPEED_QUANTILES)
        low=F.pad(dyn['low'],(0,0,0,pad)).reshape(bins,15,6)
        lm=F.pad(dyn['low_valid'],(0,0,0,pad)).reshape(bins,15,6)
        frac=(low.float()*lm).sum(1)/lm.sum(1).clamp_min(1);hold_good=lm.sum(1)>=8
        values=np.zeros((bins,6,8),np.float32);values[:,:,:3]=torch.log1p(qs).numpy();values[:,:,3]=frac.numpy()
        validity=np.zeros((bins,6,3),bool);validity[:,:,0]=qgood.numpy();validity[:,:,1]=hold_good.numpy()
        original_low=dyn['low'][0].numpy();original_valid=dyn['low_valid'][0].numpy()
        durations=np.full((n,6),-1,np.int64)
        for region in range(6):
            state=original_low[:,region];good=original_valid[:,region]
            a=0
            while a<n:
                if not good[a]:a+=1;continue
                b=a+1
                while b<n and good[b] and state[b]==state[a]:b+=1
                # Only transitions observed on both sides establish duration.
                if a>0 and b<n and good[a-1] and good[b]:
                    label=int(np.searchsorted([.5,1.,2.],(b-a)/FPS,side='right'));durations[a:b,region]=label
                a=b
        for b in range(bins):
            idx=min(b*15+7,n-1)
            for region in range(6):
                label=durations[idx,region]
                if label>=0:values[b,region,4+label]=1.;validity[b,region,2]=True
        # Recurrence supervision uses reliable root-relative poses, not music.
        q=F.pad(torch.where(m[...,None],p,0.),(0,0,0,0,0,pad)).reshape(bins,15,13,2)
        vm=F.pad(m,(0,0,0,pad)).reshape(bins,15,13)
        pooled=q.sum(1)/vm.sum(1).clamp_min(1)[...,None]
        recurrence_valid=(vm.sum(1)>=12).all(-1)
        return {'plan':values.reshape(bins,48),'plan_valid':validity,'pose_context':pooled.flatten(1).numpy(),'pose_context_valid':recurrence_valid.numpy(),'time':(np.arange(bins)*15+7)/FPS}


def _huber(a,b):return F.smooth_l1_loss(a,b,reduction='none',beta=.1)


def dynamics_losses(generated,observed,mask):
    g=observed_dynamics(generated,mask);t=observed_dynamics(observed,mask)
    profilemask=g['profile_valid']&t['profile_valid']
    profile=masked_mean(_huber(torch.log1p(g['profiles']),torch.log1p(t['profiles'])),profilemask)
    losses={'profile':profile}
    for name,valid_key in [('speed','speed_valid'),('acceleration','acceleration_valid')]:
        common=g[valid_key]&t[valid_key];gq,good=masked_quantiles(g[name],common);tq,_=masked_quantiles(t[name],common)
        losses[name+'_distribution']=masked_mean(_huber(torch.log1p(gq),torch.log1p(tq)),good[...,None])
    common=g['low_valid']&t['low_valid'];logits=(HOLD_THRESHOLD-g['peak_profile'])/HOLD_TEMPERATURE
    prob=torch.sigmoid(logits)
    # A clipped probability gives zero gradient when fast motion should become
    # quiet. Logit-space BCE stays finite and preserves that learning signal.
    low=F.binary_cross_entropy_with_logits(logits,t['low'].float(),reduction='none')
    losses['low_speed']=masked_mean(low,common)
    duration=generated.sum()*0
    for lag in [15,30,60,120]:
        if generated.shape[1]<=lag:continue
        valid=common[:,:-lag]&common[:,lag:]
        gp=prob[:,:-lag]*prob[:,lag:];tp=t['low'][:,:-lag].float()*t['low'][:,lag:].float()
        duration=duration+masked_mean((gp-tp).square(),valid)/4
    losses['duration']=duration
    # 2Hz body self-similarity within a crop; no deterministic motif copying.
    idx=torch.arange(7,generated.shape[1],15,device=generated.device)
    ga=g['pose_smoothed'][:,idx].flatten(2);ta=t['pose_smoothed'][:,idx].flatten(2)
    good=g['pose_valid'][:,idx].all(-1)&t['pose_valid'][:,idx].all(-1)
    ga=F.normalize(ga,dim=-1,eps=1e-6);ta=F.normalize(ta,dim=-1,eps=1e-6)
    pairs=good[:,:,None]&good[:,None,:]&(torch.abs(idx[:,None]-idx[None,:])>=60)[None]
    losses['recurrence']=masked_mean((ga@ga.transpose(1,2)-ta@ta.transpose(1,2)).square(),pairs)
    return losses


def reprojection_loss(generated,observed,mask):
    # One fixed horizontal reflection per ENTIRE crop handles view/laterality
    # ambiguity. No frame-wise camera, scaling, alignment or time-warp can cheat.
    per=[]
    for sign in [1.,-1.]:
        q=generated*generated.new_tensor([sign,1.]);error=_huber(q,observed)
        v=torch.where(mask[...,None],error,0.).sum((1,2,3))/(mask.sum((1,2))*2).clamp_min(1)
        per.append(v)
    values=torch.stack(per,-1);return values.min(-1).values.mean()

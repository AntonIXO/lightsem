"""Controlled color/layout probes, never the primary LIGHT-quality endpoint."""
from __future__ import annotations
import cv2
import numpy as np
from lightsem.model.light_layers.targets import _rgb_to_hsv,MIN_SATURATION,MIN_VALUE

HUE_TURNS=(0.,1/24,2/24,4/24)
LUMA=np.array([.2126,.7152,.0722])


def controlled_background(rgb,retained,hue_turns=0.,mirror=False):
    """Background hue intervention with fixed encoded-RGB Rec709 luma=.30.

    Original foreground/UI is unchanged. Chromatic background saturation maps
    into[.3,.7], ensuring all hues fit the gamut at fixed luma without clipping;
    achromatic/dark background becomes neutral gray. Geometry/hue variation
    still comes from the real source frame. These are synthetic instruments,
    explicitly not natural-image performance measurements.
    """
    image=np.asarray(rgb);mask=np.asarray(retained)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3 or mask.dtype!=bool or mask.shape!=image.shape[:2]:
        raise ValueError('Expected uint8 RGB and boolean retained mask')
    h,s,v=_rgb_to_hsv(image.astype(np.float64)/255)
    chromatic=(s>=MIN_SATURATION)&(v>=MIN_VALUE)
    saturation=np.where(chromatic,.3+.4*s,0.)
    hsv=np.stack([(h+hue_turns)%1*360,saturation,np.ones_like(h)],axis=-1).astype(np.float32)
    candidate=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB).astype(np.float64)
    candidate*=.3/(candidate@LUMA)[...,None]
    if candidate.min()<-1e-7 or candidate.max()>1+1e-7:raise RuntimeError('Controlled colors exceed fixed gamut')
    result=image.copy();result[mask]=np.floor(np.clip(candidate[mask],0,1)*255+.5).astype(np.uint8)
    if mirror:
        # Symmetric retained support swaps pairs exactly; no foreground pixels
        # are copied into the background and global background colors persist.
        both=mask&mask[:,::-1]
        mirrored=result[:,::-1].copy();result[both]=mirrored[both]
    return result


def explicit_descriptors(grid):
    """Linear circular-color summaries supplied by R1: global2 + spatial32.

    They qualify information retention only. Direct input-summary reconstruction
    is acknowledged as tautological, not an independent quality improvement.
    """
    x=np.asarray(grid,float).reshape(-1,16,75)
    h=np.repeat(np.arange(12)/12,6);s=np.tile(np.arange(6)/5,12)
    weights=np.stack([s*np.cos(2*np.pi*h),s*np.sin(2*np.pi*h)],axis=1)
    colors=x[:,:,:72]@weights;global_color=colors.mean(1)
    layout=(colors-global_color[:,None]).reshape(len(x),32)
    return {'hue':global_color,'layout':layout}


def family_probe_skill(features,targets,families,*,ridge=1e-3):
    """Leave-one-TRAIN-family-out ridge versus the fold's constant mean.

    Each family contributes one mean squared error; rows are not independent
    inference units. This is a calibration instrument, not a significance test.
    """
    x,y=np.asarray(features,float),np.asarray(targets,float);groups=np.asarray(families)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or groups.shape!=(len(x),):raise ValueError('Probe shapes')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or ridge<=0:raise ValueError('Invalid probe values')
    unique=np.unique(groups)
    if len(unique)<3:raise ValueError('At least three source families required')
    rows=[]
    for group in unique:
        train=groups!=group;test=~train
        mean=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-3)
        a=(x[train]-mean)/scale;b=(x[test]-mean)/scale
        ym=y[train].mean(0);target=y[train]-ym
        regularizer=ridge*len(a)
        if a.shape[1]<=len(a):
            beta=np.linalg.solve(a.T@a+regularizer*np.eye(a.shape[1]),a.T@target)
            prediction=b@beta+ym
        else:
            dual=np.linalg.solve(a@a.T+regularizer*np.eye(len(a)),target)
            prediction=(b@a.T)@dual+ym
        error=float(np.mean((prediction-y[test])**2));constant=float(np.mean((ym-y[test])**2))
        rows.append({'family':str(group),'mse':error,'constant_mse':constant,'rows':int(test.sum())})
    risk=float(np.mean([r['mse'] for r in rows]));constant=float(np.mean([r['constant_mse'] for r in rows]))
    return {'mse':risk,'constant_mse':constant,'baseline_corrected_skill':1-risk/constant if constant>0 else None,
            'families':rows,'ridge':ridge,'scope':'TRAIN calibration retention only; no human quality or generalization claim'}

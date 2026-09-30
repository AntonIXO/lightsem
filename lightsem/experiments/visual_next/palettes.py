"""Deterministic candidate colors and fixed fixture partitions for H4.

These are unoptimized palette candidates, never evidence that audio can choose
a spatial assignment. Exact embedding/addresses must be bound before an oracle.
"""
from __future__ import annotations
import colorsys
import numpy as np

RGB_OFFSETS={'moving_head':(6,7,8),'wash':(2,3,4),'projector':(2,3,4)}


def fixture_groups(rig):
    result=[]
    for f in rig.fixtures:
        if f.kind not in RGB_OFFSETS:continue
        right=f.position[0]>=0
        group=('right' if right else 'left')+('_heads' if f.kind=='moving_head' else '_other')
        result.append({'name':f.name,'kind':f.kind,'address':f.address,'x':float(f.position[0]),
                       'o2':'right' if right else 'left','o4':group,
                       'rgb_slots_zero_based':(f.address-1+np.array(RGB_OFFSETS[f.kind])).tolist()})
    slots=[s for f in result for s in f['rgb_slots_zero_based']]
    if len(slots)!=33 or len(set(slots))!=33:raise ValueError('Expected canonical33RGB479protected rig')
    return result


def prototype_representatives(prototype):
    """Three chromatic modes: weighted farthest init and exactly10 Lloyd steps.

    Geometry: saturation-scaled circular hue (s*cos(h),s*sin(h)) plus saturation.
    Occupied chromatic bin centers (s>=.12) only; initial largest mass then farthest
    squared distance times bin mass, lowest index ties. Output modes sorted by
    assigned mass descending then lowest member bin. HSV value=1 is fixed; no
    displayed-luma optimization. Degenerate prototypes raise, never white-fill.
    """
    p=np.asarray(prototype,float)
    if p.shape!=(72,) or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('Expected nonnegative finite72bin prototype')
    ids=np.flatnonzero((p>0)&(np.tile(np.arange(6)/5,12)>=.12))
    if not len(ids):raise ValueError('Prototype has no occupied chromatic bin')
    hue=ids//6/12;sat=ids%6/5;weight=p[ids]
    x=np.stack([sat*np.cos(2*np.pi*hue),sat*np.sin(2*np.pi*hue),sat],axis=1)
    selected=[int(np.argmax(weight))]
    for _ in range(1,min(3,len(ids))):
        distance=((x[:,None]-x[selected][None])**2).sum(2).min(1)*weight
        distance[selected]=-1
        selected.append(int(np.argmax(distance)))
    centers=x[selected].copy()
    for _ in range(10):
        assignment=((x[:,None]-centers[None])**2).sum(2).argmin(1)
        for k in range(len(centers)):
            use=assignment==k
            if use.any():centers[k]=np.average(x[use],axis=0,weights=weight[use])
    assignment=((x[:,None]-centers[None])**2).sum(2).argmin(1)
    modes=[]
    for k,center in enumerate(centers):
        use=assignment==k
        mass=float(weight[use].sum());first=int(ids[use].min()) if use.any() else int(ids[selected[k]])
        h=float(np.arctan2(center[1],center[0])/(2*np.pi)%1)
        s=float(np.clip(center[2],.12,1))
        modes.append({'mass':mass,'first_bin':first,'hue':h,'saturation':s,
                      'rgb':list(colorsys.hsv_to_rgb(h,s,1.))})
    modes.sort(key=lambda m:(-m['mass'],m['first_bin']))
    while len(modes)<3:modes.append(dict(modes[len(modes)%min(3,len(ids))]))
    return modes


def group_rgb_mask(groups):
    mask=np.zeros(512,bool)
    mask[[s for f in groups for s in f['rgb_slots_zero_based']]]=True
    return mask

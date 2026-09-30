"""Reuse one RGB->HSV conversion across unchanged chromatic instruments.

This is a computational optimization. The returned fields use the original
target definitions, including empty/dark-mask validity and hue wraparound.
"""
from __future__ import annotations
import numpy as np
from lightsem.model.light_layers import targets as t


class FrameObservation:
    def __init__(self,rgb):
        image=np.asarray(rgb)
        if image.ndim!=3 or image.shape[-1]!=3 or min(image.shape[:2])<1:raise ValueError('RGB shape')
        if image.dtype==np.uint8:image=image.astype(np.float64)/255
        elif image.dtype.kind=='f':image=image.astype(np.float64)
        else:raise ValueError('RGB dtype')
        if not np.isfinite(image).all() or np.any((image<0)|(image>1)):raise ValueError('RGB values')
        self.h,self.s,self.v=t._rgb_to_hsv(image)
        self.chromatic=(self.s>=t.MIN_SATURATION)&(self.v>=t.MIN_VALUE)

    def mask(self,retained):
        mask=np.asarray(retained)
        if mask.dtype!=bool or mask.shape!=self.h.shape:raise ValueError('Retained mask schema')
        return mask,mask&self.chromatic

    def target(self,retained):
        mask,valid=self.mask(retained);retained_pixels=int(mask.sum());valid_pixels=int(valid.sum())
        histogram=t._hs_histogram(self.h[valid],self.s[valid])
        height,width=mask.shape;quadrants=np.zeros((4,6),np.float32);quadrant_valid=np.zeros(4,bool)
        regions=((slice(0,height//2),slice(0,width//2)),(slice(0,height//2),slice(width//2,width)),
                 (slice(height//2,height),slice(0,width//2)),(slice(height//2,height),slice(width//2,width)))
        for i,region in enumerate(regions):
            m,v=mask[region],valid[region];ret,count=int(m.sum()),int(v.sum())
            quadrants[i,4]=count/ret if ret else 0.;quadrants[i,5]=ret/m.size if m.size else 0.
            if count:
                angle=self.h[region][v]*(2*np.pi);co,si=float(np.cos(angle).mean()),float(np.sin(angle).mean())
                quadrants[i,:4]=(co,si,float(self.s[region][v].mean()),min(1.,float(np.hypot(co,si))))
                quadrant_valid[i]=True
        values=self.v[mask]
        summary=(t.ValueSummary(True,float(values.mean()),float(values.std()),float(np.quantile(values,.9)),float(values.max()))
                 if retained_pixels else t.ValueSummary(False,0.,0.,0.,0.))
        return t.FrameColorTargets(rich=np.concatenate([histogram,quadrants.ravel()]),hs_histogram=histogram,
            quadrants=quadrants,color_valid=valid_pixels>0,quadrant_color_valid=quadrant_valid,
            retained_pixels=retained_pixels,valid_pixels=valid_pixels,retained_fraction=retained_pixels/mask.size,
            valid_fraction=valid_pixels/mask.size,chromatic_fraction=valid_pixels/retained_pixels if retained_pixels else 0.,
            value=summary,legacy_v6=None)

    def grid(self,retained):
        mask,valid=self.mask(retained)
        if min(mask.shape)<4:raise ValueError('Grid requires4x4 support')
        ys=np.linspace(0,mask.shape[0],5,dtype=int);xs=np.linspace(0,mask.shape[1],5,dtype=int);parts=[]
        for i in range(4):
            for j in range(4):
                region=np.s_[ys[i]:ys[i+1],xs[j]:xs[j+1]];m,v=mask[region],valid[region]
                hist=t._hs_histogram(self.h[region][v],self.s[region][v]);count=int(m.sum())
                mean_s=float(hist@np.tile(np.arange(6)/5,12)) if v.any() else 0.
                parts.extend(hist);parts.extend((mean_s,float(v.sum()/count) if count else 0.,float(count/m.size)))
        return np.asarray(parts,np.float32)

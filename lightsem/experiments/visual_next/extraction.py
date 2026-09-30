"""Streaming state for fixed Cutie/hold/lag controls, no corpus selection."""
from __future__ import annotations
from collections import deque
import cv2
import numpy as np


def cut_reset_indices(source_times,edit_intervals):
    """First 10Hz source observation strictly after each last positive cut PTS."""
    times=np.asarray(source_times,float);edits=np.asarray(edit_intervals,float).reshape(-1,2)
    if times.ndim!=1 or np.any(np.diff(times)<=0) or not np.isfinite(times).all():
        raise ValueError('Source timeline must increase')
    if not np.isfinite(edits).all() or np.any(edits[:,1]<edits[:,0]):raise ValueError('Invalid edit intervals')
    return set(int(i) for i in np.searchsorted(times,edits[:,1],side='right') if i<len(times))


class AnchorSchedule:
    """Refresh at fixed2s indices; after missing initial/cut anchor, wait for truth.

    An unavailable regular refresh keeps the existing tracker and is counted.
    A cut always clears memory; no propagation across an unanchored new shot.
    """
    def __init__(self,resets,period=20):
        self.resets=set(resets);self.period=period;self.pending=True
    def step(self,index,sam_available):
        reset=index==0 or index in self.resets
        if reset:self.pending=True
        requested=self.pending or index%self.period==0
        anchor=requested and sam_available
        if anchor:self.pending=False
        return reset,anchor,requested and not sam_available


class TemporalMasks:
    def __init__(self,model,cfg):
        self.model,self.cfg=model,cfg;self.processor=None;self.held=None
    def step(self,rgb,anchor=None,reset=False):
        import torch
        import torch.nn.functional as F
        from cutie.inference.inference_core import InferenceCore
        if reset:self.processor=None;self.held=None
        if self.processor is None:
            if anchor is None:return None,None
            self.processor=InferenceCore(self.model,cfg=self.cfg)
        h,w=rgb.shape[:2];shape=(round(h*480/min(h,w)),round(w*480/min(h,w)))
        image=torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/255
        image=F.interpolate(image[None],size=shape,mode='bilinear',align_corners=False)[0]
        mask=None
        if anchor is not None:
            self.held=anchor.copy()
            mask=torch.from_numpy(anchor.copy()).float().cuda()
            mask=F.interpolate(mask[None,None],size=shape,mode='nearest')[0,0].long()
        with torch.inference_mode():
            probability=self.processor.step(image,mask,objects=[1] if mask is not None else None)
            prediction=self.processor.output_prob_to_mask(probability)
            prediction=F.interpolate(prediction.float()[None,None],size=(h,w),mode='nearest')[0,0]
        return prediction.bool().cpu().numpy(),self.held


class FlowControl:
    """Fixed small Farneback consistency probe, not segmentation truth or RAFT.

    Backward flow current->previous at160x90; nearest mask warp. The adverse
    flow control delays that field by1s. No adaptive thresholds or learned flow.
    """
    def __init__(self):self.previous_gray=None;self.previous_mask=None;self.fields=deque(maxlen=10)
    def step(self,rgb,mask):
        gray=cv2.cvtColor(cv2.resize(rgb,(160,90),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
        current=None if mask is None else cv2.resize(mask.astype(np.uint8),(160,90),interpolation=cv2.INTER_NEAREST)
        result=np.full(2,np.nan)
        if self.previous_gray is not None:
            flow=cv2.calcOpticalFlowFarneback(gray,self.previous_gray,None,.5,3,15,3,5,1.2,0)
            if current is not None and self.previous_mask is not None:
                yy,xx=np.mgrid[:90,:160].astype(np.float32)
                fields=[flow,self.fields[0] if len(self.fields)==10 else None]
                for i,field in enumerate(fields):
                    if field is None:continue
                    mx,my=xx+field[:,:,0],yy+field[:,:,1]
                    supported=(mx>=0)&(mx<=159)&(my>=0)&(my<=89)
                    warped=cv2.remap(self.previous_mask,mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT)
                    if supported.any():result[i]=np.mean(warped[supported]==current[supported])
            self.fields.append(flow)
        self.previous_gray,self.previous_mask=gray,current
        return result

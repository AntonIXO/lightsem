"""Frozen D6 local predictions pace each D4 motif without moving its anchors."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from ..dynamic_dance_combo.campaign import learned_proposals
from ..dynamic_dance.contracts import DancePlan,DanceResult
from ..dynamic_dance.retiming import resample_body,contacts_from_body


def anchored_clock(proposals,anchors,fps=30.,edge_seconds=.4,minimum_rate=0.):
    if not 0<=minimum_rate<1:raise ValueError("Minimum rate must be in [0,1)")
    desired=np.asarray(proposals,np.float64)
    if desired.ndim==2:desired=np.median(desired,axis=1)
    if desired.ndim!=1 or len(desired)<3 or not np.isfinite(desired).all():raise ValueError('Finite whole-body proposals required')
    n=len(desired);anchors=np.unique(np.r_[0,anchors,n-1]).astype(int)
    if anchors[0]!=0 or anchors[-1]!=n-1 or np.any(anchors<0) or np.any(anchors>=n):raise ValueError('Invalid motif anchors')
    proposed=gaussian_filter1d(desired,.15*fps,mode='nearest')
    positions=np.arange(n,dtype=np.float64);rates=np.ones(n);audit=[]
    for a,b in zip(anchors[:-1],anchors[1:]):
        span=b-a
        if span<3:continue
        distance=np.minimum(np.arange(span),np.arange(span)[::-1])/max(edge_seconds*fps,1.)
        u=np.clip(distance,0,1);weight=6*u**5-15*u**4+10*u**3
        raw=proposed[a:b]
        def values(offset):return np.clip(1+weight*(raw-1+offset),minimum_rate,1.5)
        low,high=-4.,4.
        while values(low).mean()>1:low*=2
        while values(high).mean()<1:high*=2
        for _ in range(80):
            mid=(low+high)/2
            if values(mid).mean()<1:low=mid
            else:high=mid
        projected=values((low+high)/2)
        # Work in native-frame coordinates. Nonnegative increments plus an
        # explicit upper endpoint preserve monotonicity even at held intervals.
        local=np.minimum(np.cumsum(projected),float(span))
        positions[a+1:b+1]=a+local;positions[b]=float(b)
        rates[a:b]=projected
        audit.append({'start_frame':int(a),'end_frame':int(b),'sum_rate_error_frames':float(projected.sum()-span),'minimum_rate':float(projected.min()),'maximum_rate':float(projected.max()),'edge_rate_start':float(projected[0]),'edge_rate_end':float(projected[-1])})
    time=np.arange(n)/fps;clock=positions/fps;rates[-1]=rates[-2]
    actual=np.diff(clock)*fps
    if np.any(actual < minimum_rate-1e-9) or np.any(actual>1.5+1e-9):raise ValueError('Anchored time map violated fixed bounds')
    if not np.array_equal(clock[anchors],time[anchors]):raise ValueError('Musical motif anchors moved')
    return clock,np.r_[actual,actual[-1]],{'segments':audit,'anchor_frames':anchors.tolist(),'anchor_max_error_s':0.,'rate_scope':'Single common body clock; five learned regional proposals pooled by median. Rate=1 at motif seams; endpoint integral preserved.','edge_seconds':edge_seconds,'minimum_rate':minimum_rate}


def apply_d6(body,plan,telemetry,music,control=None,*,minimum_rate=0.):
    time=np.arange(len(body.world))/30
    proposals=learned_proposals(body.world,time,telemetry)
    if control=='shift':proposals=np.roll(proposals,len(proposals)//2,axis=0)
    elif control=='rms':
        rms=np.interp(time,music.time,music.mixture[:,0]);proposals=.2+1.3*np.clip(rms/max(np.quantile(rms,.9),1e-8),0,1)
    elif control is not None:raise ValueError('Unknown pacing control')
    anchors=np.array(plan.metadata.get('motif_anchor_frames',[0,len(time)-1]),int)
    clock,rates,audit=anchored_clock(proposals,anchors,minimum_rate=minimum_rate)
    world=body.world.copy() if np.array_equal(clock,time) else resample_body(body.world,clock,30.)
    meta={**body.metadata,'method':'D4_max_D6','d6_enabled':True,'source_order_preserved':True,'source_clock_scope':'Monotone clock into D4-max; all motif boundaries stay at identical song time. Original D7 library jumps are separately retained.','fragment_clock_domain':'D4-max source time','pacing':audit,'control':control,'motion_origin':'D4-max motifs with learned within-motif global pacing'}
    result=DanceResult(time=time,world=world.astype(np.float32),contacts=contacts_from_body(world),source_time=clock,metadata=meta)
    planned=DancePlan(time=time,source_time=clock,effort=rates[:,None].astype(np.float32),metadata=meta)
    result.validate();planned.validate();return result,planned,proposals

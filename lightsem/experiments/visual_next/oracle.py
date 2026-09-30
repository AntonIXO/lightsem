"""Fixed H4 color-only schedule and reference/candidate distance instruments."""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.special import rel_entr
from lightsem.model.light_layers.targets import extract_color_targets,_rgb_to_hsv

GROUP_ORDER=('left_heads','right_heads','left_other','right_other')


@dataclass
class ColorState:
    origin: np.ndarray
    target: np.ndarray
    began: np.ndarray
    duration: float=.3

    @classmethod
    def constant(cls,rgb,time):
        x=np.asarray(rgb,float)
        if x.shape!=(4,3) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('Four group RGB values required')
        return cls(x.copy(),x.copy(),np.full(4,time-.3))

    def at(self,times):
        t=np.asarray(times,float).reshape(-1)
        u=np.clip((t[:,None]-self.began[None])/self.duration,0,1)
        return self.origin[None]+(self.target-self.origin)[None]*(u*u*(3-2*u))[:,:,None]

    def adopt(self,target,time):
        x=np.asarray(target,float)
        if x.shape!=(4,3) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('Invalid RGB target')
        changed=np.any(x!=self.target,axis=1)
        result=ColorState(self.origin.copy(),self.target.copy(),self.began.copy(),self.duration)
        result.origin[changed]=self.at([time])[0,changed]
        result.target[changed]=x[changed];result.began[changed]=time
        return result


def assignments(arm):
    if arm=='O0':return np.repeat(np.arange(4)[:,None],4,axis=1)
    if arm=='O2':return np.asarray([[a,b,a,b] for a,b in itertools.product(range(4),repeat=2)],int)
    if arm=='O4':return np.asarray(list(itertools.product(range(4),repeat=4)),int)
    raise ValueError('Unknown oracle capacity')


def region_histograms_from_grid(grid,geometry_wh):
    """Combine R1 cells by their chromatic pixel mass, then normalize per quadrant.

    The source R1 cache is float16; this is an explicitly quantized automatic
    reference. It is not a manually verified background distribution.
    """
    a=np.asarray(grid,float).reshape(-1,4,4,75);w,h=geometry_wh
    ys=np.linspace(0,h,5,dtype=int);xs=np.linspace(0,w,5,dtype=int)
    area=np.diff(ys)[:,None]*np.diff(xs)[None]
    mass=a[:,:,:,73]*a[:,:,:,74]*area[None]
    result=[];valid=[]
    for r,c in ((0,0),(0,2),(2,0),(2,2)):
        weights=mass[:,r:r+2,c:c+2]
        histogram=(a[:,r:r+2,c:c+2,:72]*weights[:,:,:,None]).sum((1,2))
        total=histogram.sum(1)
        result.append(np.divide(histogram,total[:,None],out=np.zeros_like(histogram),where=total[:,None]>0))
        valid.append(total>0)
    return np.stack(result,axis=1).astype(np.float32),np.stack(valid,axis=1)


def regional_cost(reference,predictions,reference_valid):
    """Mean sqrt(JS base2) on fixed reference support; missing prediction costs1.

    reference Tx4x72, predictions KxTx4x72. No-reference intervals return NaN
    for every candidate and must use the declared persistence/tie rule upstream.
    """
    p,q=np.asarray(reference,float),np.asarray(predictions,float);valid=np.asarray(reference_valid)
    if p.ndim!=3 or p.shape[1:]!=(4,72) or q.ndim!=4 or q.shape[1:]!=p.shape or valid.shape!=p.shape[:2] or valid.dtype!=bool:
        raise ValueError('Regional histogram schema')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p<0) or np.any(q<0):raise ValueError('Invalid histogram mass')
    ps,qs=p.sum(-1),q.sum(-1)
    if np.any(valid&(ps<=0)):raise ValueError('Reference validity conflicts with empty mass')
    if not valid.any():return np.full(len(q),np.nan)
    p=np.divide(p,ps[...,None],out=np.zeros_like(p),where=ps[...,None]>0)
    q=np.divide(q,qs[...,None],out=np.zeros_like(q),where=qs[...,None]>0)
    mid=(p[None]+q)/2
    distance=np.sqrt(np.clip((rel_entr(p[None],mid).sum(-1)+rel_entr(q,mid).sum(-1))/(2*np.log(2)),0,1))
    distance=np.where(qs>0,distance,1.)
    return distance[:,valid].mean(1)


def choose_assignment(cost,candidates,previous,penalty=.01):
    c=np.asarray(cost,float);a=np.asarray(candidates,int);old=np.asarray(previous,int)
    if a.shape!=(len(c),4) or old.shape!=(4,) or np.isinf(c).any():raise ValueError('Assignment schema')
    if np.isnan(c).all():c=np.zeros_like(c)
    elif np.isnan(c).any():raise ValueError('Candidate-dependent missing reference support')
    objective=c+penalty*(a!=old).mean(1)
    # Candidates are fixed lexicographic order; argmin is deterministic for ties.
    i=int(np.argmin(objective));return a[i].copy(),float(objective[i]),i


def palette_quantization_reference(reference,palette_rgb):
    """Unrestricted HS-bin-to-palette quantization comparator, not a score bonus.

    Additive lighting/tonemapping can create intermediate display colors, so
    this descriptive palette limit is not a universal lower bound on renders.
    """
    source=np.asarray(reference,float)
    if source.shape!=(4,72):raise ValueError('Expected one four-region reference')
    colors=np.asarray(palette_rgb,np.uint8).reshape(1,4,3)
    h,s,_=_rgb_to_hsv(colors.astype(float)/255)
    ph=np.repeat(np.arange(12)/12,6);ps=np.tile(np.arange(6)/5,12)
    x=np.stack([ps*np.cos(2*np.pi*ph),ps*np.sin(2*np.pi*ph),ps],axis=1)
    points=np.stack([s.ravel()*np.cos(2*np.pi*h.ravel()),s.ravel()*np.sin(2*np.pi*h.ravel()),s.ravel()],axis=1)
    nearest=((x[:,None]-points[None])**2).sum(-1).argmin(1)
    hist=np.stack([extract_color_targets(colors[:,i:i+1],np.ones((1,1),bool)).hs_histogram for i in range(4)])
    return (source@hist[nearest]).astype(np.float32)


def oracle_gate_report(rows,summary):
    """Declared magnitudes and guards, separate from statistical direction."""
    eligible=[r for r in rows if r['source_frame_coverage']>=.8 and r['scores']['O0']['error'] is not None
              and r['scores']['O4']['error'] is not None]
    if not eligible:return {'qualified':False,'reason':'no eligible source families','families':0}
    mean=lambda arm,key:float(np.mean([r['scores'][arm][key] for r in eligible]))
    base,candidate=mean('O0','error'),mean('O4','error')
    improvement=1-candidate/base if base>0 else None
    controls={a:1-candidate/mean(a,'error') if mean(a,'error')>0 else None for a in ('permuted','donor')}
    ratios=[abs(r['scores']['O4']['mean_luma']-r['scores']['O0']['mean_luma'])/r['scores']['O0']['mean_luma']
            if r['scores']['O0']['mean_luma']>0 else None for r in eligible]
    mismatch=float(np.mean(ratios)) if all(x is not None for x in ratios) else None
    clipping=mean('O4','clipping')-mean('O0','clipping')
    win_fraction=float(np.mean([r['scores']['O4']['error']<r['scores']['O0']['error'] for r in eligible]))
    gates={'source_support':len(eligible)>=30,'relative_error_improvement':improvement is not None and improvement>=.2,
        'paired_interval_and_corrected_p':summary is not None and summary['benefit_ci95'][0]>0 and summary['four_test_bonferroni_p']<.05,
        'families_improving':win_fraction>=.8,'permutation_control':controls['permuted'] is not None and controls['permuted']>=.1,
        'donor_control':controls['donor'] is not None and controls['donor']>=.1,
        'display_luma':mismatch is not None and mismatch<=.05,'clipping_increase':clipping<=.001}
    return {'automated_proxy_gates_pass':all(gates.values()),'gates':gates,'families':len(eligible),
        'relative_improvement':improvement,'controls_relative_improvement':controls,'families_improving':win_fraction,
        'mean_absolute_family_luma_mismatch':mismatch,'clipping_increase_fraction':clipping,
        'independent_manual_reference_gate':'unmeasured; optional in user-authorized exploratory scope',
        'formal_original_hypothesis_confirmed':False}

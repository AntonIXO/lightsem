"""Unconstrained simulation: preserve saved spatial intent and every non-motion byte."""
from pathlib import Path
import json
import numpy as np
from lightsem.conductor.body_transfer import compile_intent,TransferConfig,decode
from scripts.rendering import render_native_dynamics as base

def path(root,method,song):return Path(root)/'unlimited_motion_v1'/method/song/'emission.npz'
def apply(root,method,song,pair):
    p=path(root,method,song);r=base.read(p.with_suffix('.json'))
    assert r['emission_sha256']==base.sha(p)
    assert r['original_pins']==pair.pins
    with np.load(p) as z:pair.controls=decode(z['universes'],pair.patch)
    pair.pins={**pair.pins,'unlimited_emission_sha256':base.sha(p),'unlimited_report_sha256':base.sha(p.with_suffix('.json'))}
    return pair

def generate(root):
    root=Path(root);plan=base.read(root/'PLAN.json');reports=[]
    for method in ('D7_reference','C3','C4'):
      for track in plan['tracks']:
        song=track['id'];pair=base.load_pair(root,plan,method,track)
        with np.load(root/'timelines'/method/song/'timeline.npz') as z:
            targets=z['targets'].copy();old=z['universes'].copy()
        outputs=[{k:v for k,v in p.items() if k not in ('universe','address','n_channels','channels')} for p in pair.patch]
        n=len(old);dummy=np.zeros((n,3),np.uint8)
        requested,patch,_=compile_intent({'targets':targets,'dimmer':pair.controls['dimmer']},outputs,dummy,dummy,TransferConfig(apply_motion_limits=False))
        assert patch==pair.patch
        result=old.copy();mask=np.zeros(old.shape[1:],bool);movers=[];angles=[]
        for j,p in enumerate(patch):
            if p['kind']=='bulb':continue
            movers.append(j);a=p['address']-1;u=p['universe'];mask[u,a:a+4]=True
            result[:,u,a:a+4]=requested[:,u,a:a+4]
            vec=targets[:,j]-np.array(p['position']);angle=np.rad2deg(np.stack([np.arctan2(vec[:,0],-vec[:,1]),np.arcsin(np.clip(vec[:,2]/np.linalg.norm(vec,axis=1),-1,1))],-1));angles.append(angle)
        assert np.array_equal(old[:,~mask],result[:,~mask])
        desired=np.stack(angles,axis=1);actual=decode(result,patch)['angles_deg'][:,movers]
        ranges=np.array([[patch[j]['pan_range_deg'],patch[j]['tilt_range_deg']] for j in movers]);err=np.abs(actual-desired)
        assert np.all(np.abs(desired)<=ranges/2)
        assert np.all(err<=ranges/65535/2+1e-9)
        def metrics(x):
            v=np.diff(x,axis=0)*40;a=np.diff(v,axis=0)*40
            return {'speed_abs_p95_deg_s':float(np.quantile(abs(v),.95)),'speed_abs_max_deg_s':float(abs(v).max()),'acceleration_abs_p95_deg_s2':float(np.quantile(abs(a),.95)),'acceleration_abs_max_deg_s2':float(abs(a).max()),'speed_axes_above_90_fraction':float(np.mean(abs(v)>90)),'acceleration_axes_above_360_fraction':float(np.mean(abs(a)>360))}
        p=path(root,method,song);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:np.savez_compressed(f,universes=result)
        r={'method':method,'song':song,'original_pins':pair.pins,'emission_sha256':base.sha(p),'compiler_sha256':base.sha(Path(__file__).parents[2]/'lightsem/conductor/body_transfer.py'),'mode':'unlimited simulation; direct requested angles, no temporal fitting or speed/acceleration caps','body_unchanged':True,'all_non_motion_bytes_unchanged':True,'range_clipping_count':0,'maximum_angle_error_deg':float(err.max()),'limited':metrics(pair.controls['angles_deg'][:,movers]),'unlimited':metrics(actual),'finite':bool(np.isfinite(actual).all())}
        base.save(p.with_suffix('.json'),r);reports.append(r)
    base.save(root/'reports/UNLIMITED_MOTION_PREFLIGHT.json',{'status':'simulation diagnostic; no model or baseline change','results':reports})
if __name__=='__main__':
    import sys
    generate(sys.argv[1])

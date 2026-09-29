import json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy.signal import welch
import torch
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/actuator_budget_01')
# Execute only copied sources, not whichever defaults happen to be live later.
sys.path.insert(0,str(OUT/'source'))
from lightsem.experiments.motion_study.lighting_map import compose,LightingConfig
from lightsem.experiments.motion_study.surface_revision import mechanics
from lightsem.experiments.motion_study.spatial_observer import spatial_series,summary,gates
from lightsem.experiments.beat_conditioned.beat_conditioned_render import _angles
from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.stage11.stage11_common import verify_sources
verify_sources(OUT)
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
REV=ROOT/'delivery/surface_revision_02'
assert (OUT/'registration.json').exists()
assert not (OUT/'measurements.json').exists(),'Never overwrite measured candidate'
design=json.loads((OUT/'design.json').read_text());torch.set_num_threads(4)
with np.load(ROOT/'delivery/final_diagnostic/inference/shpongle.npz') as z:features=z['flow0']
with np.load(ROOT/'data/shpongle/loudness.npz') as z:loud={k:z[k] for k in z.files}
with np.load(OUT.parent/'timing_revision_02/candidate.npz') as z:old=z['dmx']
with np.load(REV/'articulated_surface_trace.npz') as z:raw=z['raw_targets'][:,:6]
config=json.loads((OUT.parent/'timing_revision_02/design.json').read_text())['config']
baseline,_,_=compose(features,loud,len(features)/30,config=LightingConfig(**config))
assert np.array_equal(old,baseline),'Legacy regression'
dmx,trace,report=compose(features,loud,len(features)/30,config=LightingConfig(**design['config']))
assert np.array_equal(raw,trace['raw_targets'][:,:6]),'Requested motion changed'
rig=canonical_rig();heads=rig.by_kind('moving_head');pos=np.array([f.position for f in heads])
mask=np.ones(512,bool)
for f in heads:mask[f.address-1:f.address+3]=False
parity=bool(np.array_equal(dmx[:,mask],old[:,mask]));assert parity
np.savez_compressed(OUT/'candidate.npz',dmx=dmx)
np.savez_compressed(OUT/'candidate_trace.npz',**trace)
def decode(d):
 arr=[]
 for f in heads:
  a=f.address-1;ticks=np.stack((d[:,a].astype(float)*256+d[:,a+1],d[:,a+2].astype(float)*256+d[:,a+3]),-1)
  arr.append((ticks/65535-.5)*[f.pan_range_deg,f.tilt_range_deg])
 return np.stack(arr,1)
def lagfit(x,y):
 x=x.reshape(len(x),-1);y=y.reshape(len(y),-1);scores=[]
 for lag in range(-80,81):
  a,b=((x[:-lag],y[lag:]) if lag>0 else ((x[-lag:],y[:lag]) if lag<0 else (x,y)))
  a=a-a.mean(0);b=b-b.mean(0);scores.append(float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum())))
 i=int(np.argmax(scores));return {'lag_s':(i-80)/40,'zero_lag_corr':scores[80],'aligned_corr':scores[i]}
def power(a):
 f,p=welch(a.reshape(len(a),-1),fs=40,nperseg=min(len(a),400),axis=0,detrend='linear');p=p.sum(1)
 return {f'{lo}-{hi}':float(p[(f>=lo)&(f<hi)].sum()) for lo,hi in ((.1,.5),(.5,1),(1,2),(2,4))}
pan,tilt=_angles(raw-pos[None]);requested=np.stack((pan,tilt),-1);result={}
for arm,frames in [('baseline',old),('candidate',dmx)]:
 s=spatial_series(frames,rig);a=decode(frames);rows={}
 for label,(lo,hi) in design['intervals'].items():
  sl=slice(round(lo*40),round(hi*40));p=power(a[sl]);q=power(requested[sl]);sp=summary(s,lo,hi)
  rows[label]={'lag':lagfit(requested[sl],a[sl]),'band_amplitude_ratio':{b:float(np.sqrt(p[b]/q[b])) for b in p},'mechanics':mechanics(frames,lo,hi),'spatial':sp,'spatial_gates':gates(sp)}
 result[arm]={'segments':rows,'blocks':[{'start':lo,'gates':gates(summary(s,lo,lo+10))} for lo in range(0,270,10)]}
 print(arm,json.dumps({k:{'lag':v['lag'],'amplitude':v['band_amplitude_ratio'],'mechanics':v['mechanics']} for k,v in rows.items()}),flush=True)
a=result['candidate']['segments']['active'];b=result['baseline']['segments']['active'];q=result['candidate']['segments']['quiet'];f=result['candidate']['segments']['full']
result['gates']={'timing':abs(a['lag']['lag_s'])<=.1,'bandwidth':a['band_amplitude_ratio']['1-2']>=.45 and a['band_amplitude_ratio']['0.5-1']>=.70,'mechanics':f['mechanics']['max']<=90+1e-8 and f['mechanics']['acceleration_max']<=360+1e-8,'quiet':q['mechanics']['median']<=.4*40.872,'photometry':parity,'spatial':a['spatial']['relative_trajectory_4s_median_m']>=.5 and f['spatial']['hit_x_span']['p10']>=4.,'visual':None}
angles=decode(dmx);v=np.diff(angles,axis=0)*40;acc=np.diff(v,axis=0)*40;result['old_cap_exceedance']={'speed_fraction':float((np.linalg.norm(v,axis=-1)>30).mean()),'acceleration_fraction':float((np.linalg.norm(acc,axis=-1)>80).mean())};result['config']=design['config'];result['scope']=design['scope'];result['source_sha256']=json.loads((OUT/'frozen_sources.json').read_text())['source_sha256']
(OUT/'measurements.json').write_text(json.dumps(result,indent=2));print(result['gates'],flush=True)

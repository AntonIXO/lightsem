import json
from pathlib import Path
import numpy as np
from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.beat_conditioned_render import _angles
R=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/timing_revision_02')
with np.load(R/'candidate.npz') as z:d=z['dmx']
with np.load(R/'candidate_trace.npz') as z:raw=z['raw_targets'][:,:6]
h=canonical_rig().by_kind('moving_head');p,t=_angles(raw-np.array([f.position for f in h])[None]);x=np.stack((p,t),-1)
y=[]
for f in h:
 a=f.address-1;ticks=np.stack((d[:,a].astype(float)*256+d[:,a+1],d[:,a+2].astype(float)*256+d[:,a+3]),-1);y.append((ticks/65535-.5)*[f.pan_range_deg,f.tilt_range_deg])
y=np.stack(y,1);out={}
for label,(lo,hi) in {'active':(70,90),'full':(0,277.3),'quiet':(0,20),'contraction':(194,201)}.items():
 u=x[round(lo*40):round(hi*40)].reshape(-1,12);v=y[round(lo*40):round(hi*40)].reshape(-1,12);scores=[]
 for lag in range(-80,81):
  a,b=(u[:-lag],v[lag:]) if lag>0 else ((u[-lag:],v[:lag]) if lag<0 else (u,v))
  a=a-a.mean(0);b=b-b.mean(0);scores.append(float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))))
 best=int(np.argmax(scores));out[label]={'signed_lag_s':(best-80)/40,'correlation':scores[best]}
out['scope']='Supplemental post-result signed[-2,+2]s audit; original nonnegative-lag diagnostic unchanged. Positive means output delayed. Target-to-DMX, not audio.'
(R/'signed_lag.json').write_text(json.dumps(out,indent=2));print(out)

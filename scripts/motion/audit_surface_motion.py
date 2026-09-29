import json
from pathlib import Path
import numpy as np
import torch
from lightsem.experiments.motion_study.geometry import *
from lightsem.experiments.motion_study.lighting_map import ema
from lightsem.experiments.motion_study.delivery import interpolate
from lightsem.experiments.motion_study.spatial_observer import first_hits
from lightsem.experiments.quality_observer import beam_series
from lightsem.conductor.rig import canonical_rig
out=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/surface_revision_01');old=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic')
torch.set_num_threads(4)
with np.load(old/'inference/shpongle.npz') as z:features=z['flow0']
with np.load(old/'lighting/trace.npz') as z:trace={k:z[k] for k in z.files}
with np.load(old/'lighting/hybrid.npz') as z:dmx=z['dmx']
j=interpolate(joints_from_features(torch.from_numpy(features)).numpy(),30,trace['time'])
root=j[:,0];regional=j-root[:,None]
neutral=torch.zeros(1,139);neutral[:,2]=.95;neutral[:,3:135]=torch.tensor([1.,0,0,0,1,0]*22)
nj=joints_from_features(neutral).numpy()[0]
heads=canonical_rig().by_kind('moving_head');v,w=beam_series(dmx,canonical_rig());hits,surf,floor=first_hits(v,np.array([f.position for f in heads]))
requested=trace['targets'][:,:6];rv=requested-np.array([f.position for f in heads]);rv/=np.linalg.norm(rv,axis=-1,keepdims=True)
rhits,rsurf,rfloor=first_hits(rv,np.array([f.position for f in heads]))
report={'canonical_offsets':{'left_wrist':nj[20].tolist(),'right_wrist':nj[21].tolist()},'segments':{}}
for name,(lo,hi) in {'quiet':(0,20),'entry':(45,75),'active':(70,90),'ending':(250,277)}.items():
 sl=slice(lo*40,hi*40)
 row={}
 for label,x in {'world_joints':j,'root':root[:,None],'regional':regional[:,[20,21,10,11,12,15]],'requested_targets':requested,'requested_hits':rhits,'emitted_hits':hits}.items():
  a=x[sl];vel=np.linalg.norm(np.diff(a,axis=0)*40,axis=-1)
  extent=np.linalg.norm(np.quantile(a,.9,axis=0)-np.quantile(a,.1,axis=0),axis=-1)
  row[label]={'velocity_median':float(np.median(vel)),'extent_median':float(np.median(extent))}
 row['requested_floor_span_median']=float(np.median(np.ptp(rfloor[sl,:,0],axis=1)))
 row['emitted_floor_span_median']=float(np.median(np.ptp(floor[sl,:,0],axis=1)))
 row['actuator_hit_error_median_m']=float(np.nanmedian(np.linalg.norm(hits[sl]-rhits[sl],axis=-1)))
 report['segments'][name]=row
np.savez_compressed(out/'native_stage_trace.npz',joints=j,regional=regional,root=root,requested_hits=rhits)
(out/'native_stage_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)

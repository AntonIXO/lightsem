import numpy as np,json
from pathlib import Path
from lightsem.experiments.motion_study.spatial_observer import *
from lightsem.conductor.rig import canonical_rig
out=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/surface_revision_01');rig=canonical_rig();result={}
for name in ['floor_reanchor','articulated_surface']:
 with np.load(out/(name+'_trace.npz')) as z:trace={k:z[k] for k in z.files}
 with np.load(out/(name+'.npz')) as z:d=z['dmx']
 result[name]={}
 for stage,dmx in {'raw_surface':encode_targets(trace['raw_targets'][:,:6],rig),'smoothed_surface':encode_targets(trace['targets'][:,:6],rig),'emitted':d}.items():
  s=spatial_series(dmx,rig)
  result[name][stage]={}
  for label,(lo,hi) in {'quiet':(0,20),'active':(70,90),'ending':(250,277)}.items():
   r=summary(s,lo,hi);result[name][stage][label]=r
print(json.dumps(result),flush=True)
(out/'smoothing_audit.json').write_text(json.dumps(result,indent=2))

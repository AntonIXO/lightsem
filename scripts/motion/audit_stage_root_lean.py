import json
from pathlib import Path
import numpy as np
from lightsem.experiments.motion_study.articulated_stage import stage_targets,NEUTRAL
from lightsem.experiments.motion_study.lighting_map import LightingConfig
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_geometry_02')
cfg=LightingConfig(**json.loads((ROOT/'design.json').read_text())['config']);x=[-5,-3,-1,1,3,5];out={}
for axis in ('yaw','roll','pitch'):
 rows=[]
 for deg in (-30,0,30):
  a=np.deg2rad(deg);c,s=np.cos(a),np.sin(a)
  R=np.array([[c,0,s],[0,1,0],[-s,0,c]]) if axis=='yaw' else (np.array([[c,-s,0],[s,c,0],[0,0,1]]) if axis=='roll' else np.array([[1,0,0],[0,c,-s],[0,s,c]]))
  body=(NEUTRAL@R.T)[None]+np.array([0,.95,0])[None,None]
  target=np.array([stage_targets(body,p,i,np.ones(1),cfg)[0][0] for i,p in enumerate(x)])
  rows.append({'degrees':deg,'target':target.tolist()})
 out[axis]={'rows':rows,'minus_plus_target_max_difference_m':float(np.max(np.abs(np.array(rows[0]['target'])-np.array(rows[2]['target']))))}
(ROOT/'root_lean_diagnosis.json').write_text(json.dumps({'scope':'Unpreregistered isolated rigid-body orientation counterexamples, existing stage02 code; no new candidate.','results':out},indent=2));print({k:v['minus_plus_target_max_difference_m'] for k,v in out.items()})

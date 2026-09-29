import json
from pathlib import Path
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_geometry_01')
with np.load(ROOT/'candidate_trace.npz') as z:t=z['time'];raw=z['raw_targets'][:,:6];delta=z['regional']
rows={}
for label,(lo,hi) in {'full':(0,277.3),'quiet':(0,20),'active':(70,90),'contraction':(194,201)}.items():
 sl=(t>=lo)&(t<hi);r=raw[sl];d=delta[sl]
 rows[label]={'raw_wall_fraction':float((r[:,:,1]>0).mean()),'raw_unfolded_quantiles_by_head':np.quantile(1-r[:,:,2]+r[:,:,1],[.1,.5,.9],axis=0).tolist(),'delta_y_quantiles':np.quantile(d[:,:,1],[.1,.5,.9],axis=0).tolist(),'projected_delta_quantiles':np.quantile(4*d[:,:,1]-3*d[:,:,2],[.1,.5,.9],axis=0).tolist()}
(ROOT/'chart_diagnosis.json').write_text(json.dumps({'scope':'Post-result raw mapper distribution; no new candidate','segments':rows},indent=2));print(json.dumps(rows,indent=2))

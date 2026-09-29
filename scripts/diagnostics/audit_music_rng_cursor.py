import json,hashlib
from pathlib import Path
import numpy as np
import torch
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01')

def canon(x):
 if isinstance(x,torch.Tensor):return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()}
 if isinstance(x,np.ndarray):return {'dtype':str(x.dtype),'shape':list(x.shape),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}
 if isinstance(x,dict):return {k:canon(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [canon(v) for v in x]
 if isinstance(x,np.generic):return x.item()
 return x
rows={}
for arm in ('masked_native','physical'):
 state=torch.load(OUT/'runs'/arm/'latest.pt',map_location='cpu',weights_only=False)
 rows[arm]={'step':state['step'],'rng':canon(state['rng'])};del state
assert rows['masked_native']['step']==rows['physical']['step']
checks={k:rows['masked_native']['rng'][k]==rows['physical']['rng'][k] for k in rows['masked_native']['rng']}
report={'scope':'CPU-only paired training RNG cursors at same completed segment; does not inspect VAL/model outputs. Endpoints and logged counts support exposure matching, not standalone proof of all intermediate tensors.','step':rows['physical']['step'],'matches':checks,'cursors':rows}
p=OUT/f'rng_cursor_{report["step"]}.json';assert not p.exists();p.write_text(json.dumps(report,indent=2));print(checks);assert all(checks.values())

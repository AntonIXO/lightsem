import json, os, h5py, numpy as np
from pathlib import Path
out={}
contract=Path('/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_contract_20261016_v1/CONTRACT.json')
if contract.exists():
 c=json.loads(contract.read_text()); out['canonical']={'path':str(contract),'rows':len(c.get('rows',[])),'keys':list(c.keys())[:10],'bytes':contract.stat().st_size}
for p in [Path('/mnt/lightsem-data/rpmc_l2/RPMC_L2.h5'),Path('/mnt/lightsem-data/open_datasets/roskilde_2023/songs.pkl'),Path('/mnt/lightsem-data/open_datasets/roskilde_2023/data.csv')]:
 out[p.name]={'path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None}
with h5py.File('/mnt/lightsem-data/rpmc_l2/RPMC_L2.h5','r') as f: out['RPMC_L2.h5']['samples']=len(f)
for p in [Path('/mnt/lightsem-data/experiments/lightsem/foundation_cyclic_motion_prior_open_combo_20261021_v2/TRAINING.json'),Path('/mnt/lightsem-data/experiments/lightsem/foundation_cyclic_motion_prior_20261021_v5/TRAINING.json')]:
 if p.exists(): out[p.name]=json.loads(p.read_text())
for p in [Path('/mnt/lightsem-data/experiments/lightsem/audio_pipeline_integration_20261005/features/muq/shpongle.npz'),Path('/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b/demo_features/features/base92/shpongle.npz'),Path('/mnt/lightsem-data/experiments/lightsem/audio_pipeline_integration_20261005/features/muq/cure_for_me.npz'),Path('/mnt/lightsem-data/experiments/lightsem/native_bf16_v7b/demo_features/features/base92/cure_for_me.npz')]:
 with np.load(p) as z: out[p.name]={'path':str(p),'keys':z.files,'shapes':{k:list(z[k].shape) for k in z.files}}
print(json.dumps(out,indent=2,default=str))

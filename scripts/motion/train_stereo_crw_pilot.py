from pathlib import Path
import json, random, hashlib
import numpy as np, torch
from lightsem.experiments.audio_stereo.crw_pilot import StereoCRW,StereoCRWConfig,training_step
from lightsem.experiments.audio_stereo.spain_pilot import equal_power_repan
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/three_spatial_methods_20261011_v1/train_npz')
OUT=Path('/mnt/lightsem-data/experiments/lightsem/three_spatial_methods_20261011_v1/learning/crw_seed0')
if OUT.exists(): raise FileExistsError(OUT)
paths=sorted(ROOT.glob('*.npz')); data=[]
for p in paths:
 with np.load(p) as z:data.append(torch.from_numpy(z['stems'].astype('float32')))
seed=0;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
device='cuda'; cfg=StereoCRWConfig(); model=StereoCRW(cfg).to(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
losses=[]
for step in range(500):
 batch=[]
 for j in range(4):
  stems=data[random.randrange(len(data))]
  n=stems.shape[-1]; st=random.randrange(max(1,n-cfg.required_samples+1)); chunk=stems[:,st:st+cfg.required_samples]
  ang=torch.empty(4).uniform_(-45,45);mix,_=equal_power_repan(chunk[None].to(device),ang[None].to(device)); batch.append(mix[0])
 wave=torch.stack(batch)
 r=training_step(model,opt,wave);losses.append(r['loss'])
 if step%100==0: print(step,r,flush=True)
OUT.mkdir(parents=True); cp=OUT/'final.pt'; torch.save({'state_dict':model.state_dict(),'config':cfg.config_dict(),'seed':seed,'steps':500,'loss_history':losses,'train_manifest':str(ROOT/'MANIFEST.json'),'train_manifest_sha256':hashlib.sha256((ROOT/'MANIFEST.json').read_bytes()).hexdigest()},cp)
rep={'schema_version':1,'status':'complete','method':'StereoCRW','seed':seed,'steps':500,'checkpoint':str(cp),'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'loss_first':losses[0],'loss_last':losses[-1],'finite':bool(np.isfinite(losses).all()),'diagnostic_only':True,'pretrained_fma_weights':False,'reference_commit':'c08488505042fcdc2788d339ba23901929e7b6d4'}
(OUT/'REPORT.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep))

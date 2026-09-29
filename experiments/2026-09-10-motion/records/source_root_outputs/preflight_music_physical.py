import json,time,hashlib,gc,copy
from pathlib import Path
import numpy as np
import torch
from lightsem.experiments.motion_study.music_model import MusicConfig
from lightsem.experiments.motion_study.music_revision_model import MusicRevisionModel,PhysicalWeights
from lightsem.experiments.motion_study.music_revision_data import MusicRevisionResident
from lightsem.experiments.motion_study.geometry import joints_from_features
from lightsem.experiments.training_runtime import TrainingRuntime,BatchStream,backward_checked
from lightsem.experiments.stage11_common import seed_all,gpu_lock,digest
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01')
assert not (OUT/'engineering.json').exists()
plan=json.loads((OUT/'engineering_plan.json').read_text())
manifest=json.loads((ROOT/'data/audio_manifest.json').read_text());audio_norms=json.loads((ROOT/'data/audio_normalizers.json').read_text());norms=json.loads((ROOT/'data/motion_normalizers.json').read_text())
torch.set_num_threads(4)
with gpu_lock(ROOT):
 seed_all(12345);runtime=TrainingRuntime();runtime.apply()
 data=MusicRevisionResident(manifest,audio_norms,frames=960,device='cuda')
 coverage=data.coverage;hold=[]
 with torch.inference_mode():
  for e in data.entries:
   with np.load(e['path']) as z:f=z['features']
   n=len(f)//60
   j=joints_from_features(torch.from_numpy(f[:n*60])).numpy().reshape(n,60,22,3)
   v=np.diff(j,axis=1)*30;k=(v*v).sum(-1).mean((1,2));travel=np.linalg.norm(np.diff(j[:,:,0,[0,2]],axis=1),axis=-1).sum(1)
   hold.append({'key':e['key'],'quiet_2s_windows':int((k<.01).sum()),'stationary_2s_windows':int((travel<.1).sum()),'windows':n})
 coverage['hold_audit']={'definition':'nonoverlapping2s TRAIN windows; quiet kinetic<.01m2/s2, stationary horizontal root travel<.1m; not contact labels','rows':hold,'quiet_windows':sum(r['quiet_2s_windows'] for r in hold),'stationary_windows':sum(r['stationary_2s_windows'] for r in hold)}
 coverage['hold_coverage']='measured source window inventory; exact sampled occurrence recorded via indices, not inferred from contacts'
 (OUT/'coverage.json').write_text(json.dumps(coverage,indent=2));print('COVERAGE',coverage['recordings'],coverage['canonical_songs'],coverage['short_recordings'],coverage['hold_audit']['quiet_windows'],flush=True)
 # Same consumed RNG cursor and gathered tensors for prefetch0 vs2.
 streams=[]
 for depth in (0,2):
  batches=[]
  with BatchStream(data,32,960,np.random.default_rng(54321),depth) as stream:
   for _ in range(4):batches.append(tuple(v.cpu() for v in stream.next()))
  streams.append(batches)
 assert all(torch.equal(a,b) for x,y in zip(*streams) for a,b in zip(x,y))
 del streams
 reports={}
 for arm in plan['arms']:
  seed_all(12345);runtime.apply();torch.compiler.reset()
  model=MusicRevisionModel(MusicConfig(),norms,arm=arm,weights=PhysicalWeights(**plan['weights']) if arm=='physical' else PhysicalWeights()).cuda()
  initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
  optimizer=runtime.make_optimizer(model,3e-4);fn=model.loss_function(compile_vector_field=True);rng=np.random.default_rng(10000)
  torch.cuda.reset_peak_memory_stats();start=time.monotonic();rows=[];hashes=[]
  with BatchStream(data,32,960,rng,2) as stream:
   for step in range(plan['updates_per_arm']):
    x,y,valid=stream.next();noise,t=model.draws(x)
    hashes.append(hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in (x,y,valid,noise,t))).hexdigest())
    optimizer.zero_grad(set_to_none=True)
    with runtime.autocast():losses=fn(x,y,valid,noise,t)
    grad=backward_checked(losses,model);optimizer.step();torch.cuda.synchronize()
    row={'step':step+1,'seconds':time.monotonic()-start,'grad':float(grad),'valid_frames':int(valid.sum()),**{k:float(v.detach()) for k,v in losses.items()}}
    rows.append(row)
    if step in (0,1,7,15,31):print(arm,json.dumps(row),flush=True)
  reports[arm]={'initial_sha256':initial,'draw_hashes':hashes,'rows':rows,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'warm_seconds_per_update':(rows[-1]['seconds']-rows[7]['seconds'])/24,'model_config':model.config_dict()}
  del fn,model,optimizer;gc.collect();torch.cuda.empty_cache()
 assert reports['masked_native']['initial_sha256']==reports['physical']['initial_sha256']
 assert reports['masked_native']['draw_hashes']==reports['physical']['draw_hashes']
 result={'scope':plan['scope'],'plan':plan,'reports':reports,'prefetch_parity':True,'paired_draws':True,'source_sha256':{str(p):digest(p) for p in [Path('lightsem/experiments/motion_study')/n for n in ['music_revision_model.py','music_revision_data.py','music_physical.py','music_model.py']]},'weights_saved':False}
 (OUT/'engineering.json').write_text(json.dumps(result,indent=2));print('PREFLIGHT COMPLETE',flush=True)

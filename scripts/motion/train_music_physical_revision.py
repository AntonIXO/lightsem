"""Fixed30k schedule; optional5k orchestration stops, never interim selection."""
import argparse,json,math,time,sys,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01')
sys.path.insert(0,str(OUT/'source'))
from lightsem.experiments.motion_study.music_model import MusicConfig
from lightsem.experiments.motion_study.music_revision_model import MusicRevisionModel,PhysicalWeights
from lightsem.experiments.motion_study.music_revision_data import MusicRevisionResident
from lightsem.experiments.training_runtime import TrainingRuntime,BatchStream,backward_checked
from lightsem.experiments.stage11.stage11_common import seed_all,gpu_lock,digest,verify_sources,rng_state,restore_rng,atomic_torch,require_space
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
def run(arm,until):
 protocol=json.loads((OUT/'protocol.json').read_text());design=protocol['design'];verify_sources(OUT)
 for name,sha in protocol['inputs'].items():
  if digest(name)!=sha:raise RuntimeError('Frozen input drift '+name)
 assert (OUT/'registration.json').exists()
 if until%5000 or not 0<until<=design['training']['steps']:raise ValueError('Orchestration stops at fixed5k boundaries only')
 require_space(OUT,gb=1.0);torch.set_num_threads(4)
 run=OUT/'runs'/arm;run.mkdir(parents=True,exist_ok=True)
 identity={'arm':arm,'seed':0,'protocol_sha256':digest(OUT/'protocol.json'),'source_sha256':protocol['source_sha256']}
 if (run/'report.json').exists():
  done=json.loads((run/'report.json').read_text());assert done['identity']==identity;print('Already complete',arm);return
 seed_all(0);runtime=TrainingRuntime();runtime.apply()
 manifest=json.loads((ROOT/'data/audio_manifest.json').read_text());audio_norms=json.loads((ROOT/'data/audio_normalizers.json').read_text());norms=json.loads((ROOT/'data/motion_normalizers.json').read_text())
 data=MusicRevisionResident(manifest,audio_norms,frames=960,device='cuda')
 weights=PhysicalWeights(**design['weights']) if arm=='physical' else PhysicalWeights()
 model=MusicRevisionModel(MusicConfig(),norms,arm=arm,weights=weights).cuda()
 optimizer=runtime.make_optimizer(model,design['training']['lr'])
 total=design['training']['steps'];warmup=design['training']['warmup']
 schedule=lambda step:min(1.,(step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,total-warmup))))
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
 rng=np.random.default_rng(10000);start=0;elapsed=0.
 if (run/'latest.pt').exists():
  state=torch.load(run/'latest.pt',map_location='cuda',weights_only=False);assert state['identity']==identity
  model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'],rng)
  start=state['step'];elapsed=state['training_seconds']
  if (run/'training.jsonl').exists():
   lines=(run/'training.jsonl').read_text().splitlines()
   if any(json.loads(line)['step']>start for line in lines):
    shutil.copy2(run/'training.jsonl',run/f'training_before_resume_{time.time_ns()}.jsonl')
    (run/'training.jsonl').write_text(''.join(line+'\n' for line in lines if json.loads(line)['step']<=start))
 if until<=start:print('Segment already reached',arm,start);return
 def save(step,path):
  state={'identity':identity,'step':step,'training_seconds':elapsed,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':rng_state(rng),'model_config':model.config_dict(),'normalizers':norms,'audio_normalizers':audio_norms}
  atomic_torch(path,state);return state
 if start==0:
  if (run/'initial.pt').exists():raise RuntimeError('Ambiguous initial-only restart; inspect before resuming')
  save(0,run/'initial.pt')
  initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
  (run/'initial_identity.json').write_text(json.dumps({'model_sha256':initial_hash,'identity':identity},indent=2))
  other=OUT/'runs'/('physical' if arm=='masked_native' else 'masked_native')/'initial_identity.json'
  if other.exists():assert json.loads(other.read_text())['model_sha256']==initial_hash
 torch.compiler.reset();fn=model.loss_function(compile_vector_field=True);model.train();torch.cuda.reset_peak_memory_stats();wall=time.monotonic();interval=wall
 with BatchStream(data,32,960,rng,2) as stream,(run/'training.jsonl').open('a') as log:
  for step in range(start,until):
   x,y,valid=stream.next();noise,t=model.draws(x);optimizer.zero_grad(set_to_none=True)
   with runtime.autocast():losses=fn(x,y,valid,noise,t)
   gradient=backward_checked(losses,model);optimizer.step();scheduler.step()
   if step==start or (step+1)%100==0 or step+1==until:
    torch.cuda.synchronize();now=time.monotonic();elapsed+=now-interval;interval=now
    row={'arm':arm,'step':step+1,'training_seconds':elapsed,'lr':scheduler.get_last_lr()[0],'grad':float(gradient),'valid_frames':int(valid.sum()),**{k:float(v.detach()) for k,v in losses.items()}}
    log.write(json.dumps(row)+'\n');log.flush();(run/'status.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
   if (step+1)%1000==0:save(step+1,run/'latest.pt')
  state=save(until,run/'latest.pt')
  if until==total:
   atomic_torch(run/'final.pt',state)
   result={'identity':identity,'steps':total,'training_seconds':elapsed,'peak_gpu_gib_this_segment':torch.cuda.max_memory_allocated()/2**30,'model_config':model.config_dict(),'final_sha256':digest(run/'final.pt'),'initial_sha256':digest(run/'initial.pt'),'test_access':False}
   (run/'report.json').write_text(json.dumps(result,indent=2));print('TRAINING COMPLETE',arm,flush=True)
 (run/f'segment_{start}_{until}.json').write_text(json.dumps({'start':start,'end':until,'wall_seconds':time.monotonic()-wall,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'identity':identity},indent=2))
 print('SEGMENT COMPLETE',arm,until,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['masked_native','physical'],required=True);p.add_argument('--until',type=int,default=30000);a=p.parse_args()
 with gpu_lock(ROOT):run(a.arm,a.until)

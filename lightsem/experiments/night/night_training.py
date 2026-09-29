"""Sequential matched training with resumable RNG and immutable final-step selection."""
from pathlib import Path
from collections import defaultdict
import argparse,gc,json,math,time,shutil
import numpy as np
import torch
from .night_common import ROOT,ARMS,SEEDS,base_context
from .night_model import NightModel
from ..beat_conditioned.beat_conditioned_training import Resident
from ..training_runtime import TrainingRuntime, BatchStream, backward_checked, runtime_loss
from ..stage11.stage11_common import (authenticated_json,read_authenticated,atomic_json,atomic_torch,digest,
    json_digest,seed_all,rng_state,restore_rng,artifact_manifest,verify_artifacts,verify_sources,gpu_lock,require_space)

class NightResident(Resident):
    def __init__(self,entries,norms,frames,root=ROOT):
        super().__init__(entries,norms,frames)
        self.root=Path(root);self.family=None;self.host_aux=None
        self.aux_mean=None;self.aux_std=None;self.aux_dim=0

    def set_family(self,family):
        if self.family==family:return
        self.host_aux=None;gc.collect();self.family=family
        if family=='base':self.aux_dim=0;return
        info=read_authenticated(self.root/'features'/family/'training_array.json')
        if info['entry_keys']!=[e['key'] for e in self.entries]:raise ValueError('Auxiliary row order mismatch')
        if digest(info['path'])!=info['sha256']:raise RuntimeError('Auxiliary array drift')
        self.host_aux=np.load(info['path'],mmap_mode='r');self.aux_dim=self.host_aux.shape[1]
        self.aux_mean=torch.tensor(info['mean'],device='cuda',dtype=torch.float32)
        self.aux_std=torch.tensor(info['std'],device='cuda',dtype=torch.float32)

    def draw_indices(self,size,frames,rng):
        # Preserve the legacy 32-crop stream when grouping larger batches.
        if size>32 and size%32==0:
            return np.concatenate([self.draw_indices(32,frames,rng) for _ in range(size//32)])
        selected=[]
        for _ in range(size):
            records=self.groups[int(rng.integers(len(self.groups)))];coaches=records[int(rng.integers(len(records)))]
            selected.append(coaches[int(rng.integers(len(coaches)))])
        selected=np.asarray(selected);off=np.asarray([rng.integers(self.lengths[i]-frames+1) for i in selected])
        ix=self.starts[selected,None]+off[:,None]+np.arange(frames)[None]
        return ix

    def batch(self,size,frames,rng,return_index=False):
        ix=self.draw_indices(size,frames,rng)
        index=torch.as_tensor(ix,device='cuda');x=self.x[index]
        if self.host_aux is not None:
            a=torch.from_numpy(np.asarray(self.host_aux[ix])).to(device='cuda',dtype=torch.float32)
            x=torch.cat((x,(a-self.aux_mean)/self.aux_std),dim=2)
        result=(x,self.y[index],self.b[index])
        return (*result,index) if return_index else result

def preflight(root=ROOT,runtime=None):
    runtime=runtime or TrainingRuntime()
    root=Path(root);m,n=base_context();data=NightResident([e for e in m['entries'] if e['partition']=='train'],n,640,root)
    rows=[];pos=torch.tensor(n['pos_weight'],device='cuda',dtype=torch.float32)
    for arm in ARMS:
        family,architecture=arm.split('_');data.set_family(family);seed_all(0)
        runtime.apply()
        model=NightModel(family,architecture,data.aux_dim).cuda();opt=runtime.make_optimizer(model,3e-4)
        loss_fn=runtime.loss_function(model)
        rng=np.random.default_rng(10000);gen=torch.Generator(device='cuda').manual_seed(30000);times=[]
        torch.cuda.reset_peak_memory_stats()
        with BatchStream(data,32,640,rng,runtime.prefetch) as batches:
            for i in range(32):
                torch.cuda.synchronize();t=time.monotonic();x,y,b=batches.next()
                opt.zero_grad(set_to_none=True);loss=runtime_loss(model,loss_fn,runtime,x,y,b,pos,gen)
                backward_checked(loss,model)
                opt.step();torch.cuda.synchronize();times.append(time.monotonic()-t)
        # Chunk correctness continues to use the strict FP32 reference path.
        seed_all(0)
        model.eval();x,y,b=data.batch(1,1600,rng)
        with torch.inference_mode():
            direct=model.encode_audio(x);chunk=model.encode_audio_chunked(x,640)
            err=float((direct-chunk).abs().max())
            noise=torch.randn((1,1600,17),generator=torch.Generator(device='cpu').manual_seed(101))
            pred,_=model.generate(x,noise,steps=2,chunk_size=640)
        if err>2e-5 or not torch.isfinite(pred).all():raise RuntimeError('Chunk consistency failed')
        rows.append({'arm':arm,'step_seconds_median':float(np.median(times[8:])),
                     'parameters':sum(p.numel() for p in model.parameters()),'chunk_max_abs':err,
                     'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'engineering_updates':32,
                     'saved_scientific_weights':False,'runtime':runtime.as_dict()})
        print(json.dumps(rows[-1]),flush=True)
        del model,opt,loss_fn,x,y,b,pred,direct,chunk,loss;gc.collect();torch.cuda.empty_cache()
    # Equal exposure is set from performance only, before any scientific validation outputs.
    seconds=sum(row['step_seconds_median'] for row in rows)*len(SEEDS)
    steps=max(100,100*int((7.25*3600/seconds)//100))
    report={'rows':rows,'suggested_common_steps':steps,'training_budget_seconds':7.25*3600,
            'formula':'floor_to_100(training_budget / sum(step_seconds for every arm and seed))',
            'gpu':torch.cuda.get_device_name(),'precision':runtime.precision,'runtime':runtime.as_dict(),
            'runtime_source_sha256':digest(Path(__file__).with_name('training_runtime.py')),'source_sha256':digest(__file__),
            'model_source_sha256':digest(Path(__file__).with_name('night_model.py'))}
    authenticated_json(root/'engineering_preflight.json',report);return report

def study_context(root=ROOT):
    root=Path(root);p=read_authenticated(root/'protocol.json');m,n=base_context();f=verify_sources(root)
    if not p['frozen'] or p['source_sha256']!=f['source_sha256']:raise ValueError('Protocol is not frozen')
    for path,sha in p['input_artifacts'].items():
        if digest(path)!=sha:raise RuntimeError('Frozen input changed: '+path)
    return p,m,n,{'protocol_sha256':digest(root/'protocol.json'),'source_sha256':f['source_sha256']}

def run_cell(arm,seed,data,root=ROOT,resume=False,initial_overhead=0.):
    complete_begin=time.monotonic()-initial_overhead
    root=Path(root);p,m,n,identity=study_context(root);cfg=p['training'];run=root/'runs'/arm/f'seed_{seed}'
    if (run/'report.json').exists():
        r=read_authenticated(run/'report.json');verify_artifacts(r['artifacts'])
        if r['identity']['arm']!=arm or r['identity']['seed']!=seed or r['identity']['protocol_sha256']!=identity['protocol_sha256'] or r['steps']!=cfg['steps']:
            raise RuntimeError('Completed cell identity differs')
        return r
    existing=run.exists() and any(run.iterdir())
    if existing and not resume:raise RuntimeError('Explicit recovery required: '+str(run))
    resume_this=bool(existing and resume)
    if resume_this and not ((run/'initial.pt').exists() or (run/'latest.pt').exists()):raise RuntimeError('Unrecognized partial run: '+str(run))
    run.mkdir(parents=True,exist_ok=True);family,architecture=arm.split('_');data.set_family(family)
    seed_all(seed);model=NightModel(family,architecture,data.aux_dim).cuda()
    runtime=TrainingRuntime.from_training(cfg)
    runtime.apply()
    opt=runtime.make_optimizer(model,cfg['lr']);loss_fn=runtime.loss_function(model)
    total=cfg['steps'];warm=cfg['warmup']
    schedule=lambda s:min(1.,(s+1)/warm)*(.1+.9*.5*(1+math.cos(math.pi*max(0,s-warm)/max(1,total-warm))))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,schedule);rng=np.random.default_rng(10000+seed)
    flow_rng=torch.Generator(device='cuda').manual_seed(30000+seed)
    pos=torch.tensor(n['pos_weight'],device='cuda',dtype=torch.float32);start=0;elapsed=0.;prior_complete=0.
    identity={**identity,'arm':arm,'seed':seed,'training':cfg,'model_config':model.config_dict()}
    if resume_this:
        checkpoint=run/'latest.pt' if (run/'latest.pt').exists() else run/'initial.pt'
        z=torch.load(checkpoint,map_location='cuda',weights_only=False)
        if json_digest(z['identity'])!=json_digest(identity):raise RuntimeError('Resume identity differs')
        model.load_state_dict(z['model'])
        if 'optimizer' in z:
            opt.load_state_dict(z['optimizer']);sch.load_state_dict(z['scheduler'])
            restore_rng(z['rng'],rng);flow_rng.set_state(z['flow_rng'].cpu());start=z['step'];elapsed=z['training_seconds']
            prior_complete=z.get('complete_seconds',elapsed)
        log_path=run/'training.jsonl'
        if log_path.exists():
            copy=run/f'training_before_resume_{time.time_ns()}.jsonl';shutil.copy2(log_path,copy)
            lines=[line for line in log_path.read_text().splitlines() if json.loads(line)['step']<=start]
            log_path.write_text(''.join(line+'\n' for line in lines))
            authenticated_json(run/f'resume_{time.time_ns()}.json',{'from_step':start,'checkpoint_sha256':digest(checkpoint),'preserved_log':str(copy),'preserved_log_sha256':digest(copy)})
    else:atomic_torch(run/'initial.pt',{'model':model.state_dict(),'identity':identity,'normalizers':n,'step':0})
    def save(step):
        z={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'rng':rng_state(rng),
           'flow_rng':flow_rng.get_state(),'step':step,'training_seconds':elapsed,
           'complete_seconds':prior_complete+time.monotonic()-complete_begin,'identity':identity,'normalizers':n}
        atomic_torch(run/'latest.pt',z);return z
    model.train();torch.cuda.reset_peak_memory_stats();wall=time.monotonic();interval=wall
    with BatchStream(data,cfg['batch'],cfg['frames'],rng,runtime.prefetch) as batches, (run/'training.jsonl').open('a') as log:
        for step in range(start,total):
            x,y,b=batches.next()
            opt.zero_grad(set_to_none=True)
            losses=runtime_loss(model,loss_fn,runtime,x,y,b,pos,flow_rng,cfg['beat_dropout'])
            grad=backward_checked(losses,model)
            opt.step();sch.step()
            if step==0 or (step+1)%100==0 or step+1==total:
                torch.cuda.synchronize();now=time.monotonic();elapsed+=now-interval;interval=now
                row={'arm':arm,'seed':seed,'step':step+1,**{k:float(v.detach()) for k,v in losses.items()},
                     'grad':float(grad),'lr':sch.get_last_lr()[0],'training_seconds':elapsed}
                log.write(json.dumps(row)+'\n');log.flush();atomic_json(run/'status.json',row);atomic_json(root/'status.json',row)
                print(json.dumps(row),flush=True)
            if (step+1)%cfg.get('checkpoint_interval',1000)==0:
                if (step+1)%100!=0:
                    torch.cuda.synchronize();now=time.monotonic();elapsed+=now-interval;interval=now
                save(step+1)
        final=save(total);atomic_torch(run/'final.pt',final)
    r={'status':'completed','identity':identity,'steps':total,'samples':total*cfg['batch'],
       'training_seconds':elapsed,'wall_seconds':time.monotonic()-wall,'complete_seconds':prior_complete+time.monotonic()-complete_begin,
       'gpu':torch.cuda.get_device_name(),'precision':runtime.precision,'runtime':runtime.as_dict(),
       'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
       'parameters':sum(v.numel() for v in model.parameters()),'selection':'final_scheduled_step_only',
       'artifacts':artifact_manifest([run/'initial.pt',run/'final.pt',run/'latest.pt',run/'training.jsonl'])}
    authenticated_json(run/'report.json',r)
    del model,opt,final,losses,x,y,b;gc.collect();torch.cuda.empty_cache();return r

def campaign(root=ROOT,resume=False):
    root=Path(root);p,m,n,identity=study_context(root)
    loading=time.monotonic()
    data=NightResident([e for e in m['entries'] if e['partition']=='train'],n,p['training']['frames'],p.get('feature_root',root))
    loading=time.monotonic()-loading
    rows=[]
    # Alternate architecture within a feature family; repeat the entire matrix at seed1.
    seeds=p.get('seeds',SEEDS);arms=p.get('arms',ARMS);planned=len(seeds)*len(arms)
    for seed in seeds:
        for arm in arms:
            require_space(root)
            rows.append(run_cell(arm,seed,data,root,resume=resume))
            authenticated_json(root/'training_progress.json',{'completed':len(rows),'planned':planned,
                'rows':[{'arm':r['identity']['arm'],'seed':r['identity']['seed'],'steps':r['steps']} for r in rows]})
    authenticated_json(root/'training_report.json',{'identity':identity,'rows':rows,'loading_seconds':loading,
        'training_seconds':sum(r['training_seconds'] for r in rows),'all_cells_complete':len(rows)==planned})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','campaign']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    with gpu_lock(a.root):
        if a.stage=='preflight':preflight(a.root)
        else:campaign(a.root,a.resume)

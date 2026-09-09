"""One frozen seed-0 pilot. GPU only; resumable RNG, final-step selection."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .beat_conditioned_data import ROOT,event_targets
from .beat_conditioned_model import BeatConditionedModel
from .stage11_common import (read_authenticated,authenticated_json,atomic_json,atomic_torch,
    digest,verify_sources,seed_all,rng_state,restore_rng,artifact_manifest,gpu_lock,require_space)

def context(root,verify_tensors=True):
    root=Path(root);p=read_authenticated(root/'protocol.json');m=read_authenticated(root/'data_manifest.json')
    n=read_authenticated(root/'normalizers.json');f=verify_sources(root)
    identity={'protocol_sha256':digest(root/'protocol.json'),'manifest_sha256':digest(root/'data_manifest.json'),
              'normalizers_sha256':digest(root/'normalizers.json'),'source_sha256':f['source_sha256']}
    assert p['frozen'] and p['manifest_sha256']==identity['manifest_sha256']
    assert p['normalizers_sha256']==identity['normalizers_sha256']==m['normalizers_sha256']
    assert p['source_sha256']==identity['source_sha256']
    parts={};keys=set()
    for e in m['entries']:
        assert e['partition'] in ('train','val') and e['fps']==40. and e['key'] not in keys
        keys.add(e['key']);song=e['canonical_id']
        assert song not in parts or parts[song]==e['partition'];parts[song]=e['partition']
        if verify_tensors and digest(e['path'])!=e['sha256']:raise RuntimeError('Tensor drift')
    return p,m,n,identity

class Resident:
    def __init__(self,entries,norms,frames):
        xs=[];ys=[];bs=[];starts=[];lengths=[];offset=0;groups=defaultdict(lambda:defaultdict(list))
        self.excluded=[e['key'] for e in entries if e['n_frames']<frames]
        self.entries=[e for e in entries if e['n_frames']>=frames]
        xm=np.asarray(norms['x_mean'],np.float32);xscl=np.asarray(norms['x_std'],np.float32)
        ym=np.asarray(norms['y_center'],np.float32);yscl=np.asarray(norms['y_scale'],np.float32)
        for i,e in enumerate(self.entries):
            with np.load(e['path'],allow_pickle=False) as z:
                xs.append((z['x']-xm)/xscl);ys.append((z['y']-ym)/yscl);bs.append(event_targets(z['b']))
            starts.append(offset);lengths.append(e['n_frames']);offset+=e['n_frames']
            groups[e['canonical_id']][e['recording_id']].append(i)
        self.x=torch.from_numpy(np.concatenate(xs)).cuda();del xs
        self.y=torch.from_numpy(np.concatenate(ys)).cuda();del ys
        self.b=torch.from_numpy(np.concatenate(bs)).cuda();del bs
        self.starts=np.asarray(starts);self.lengths=np.asarray(lengths)
        self.groups=[list(v.values()) for k,v in sorted(groups.items())]

    def batch(self,size,frames,rng):
        selected=[]
        for _ in range(size):
            records=self.groups[int(rng.integers(len(self.groups)))];coaches=records[int(rng.integers(len(records)))]
            selected.append(coaches[int(rng.integers(len(coaches)))])
        selected=np.asarray(selected)
        offsets=np.asarray([rng.integers(self.lengths[i]-frames+1) for i in selected])
        index=torch.as_tensor(self.starts[selected,None]+offsets[:,None]+np.arange(frames)[None],device='cuda')
        return self.x[index],self.y[index],self.b[index]

def train(root=ROOT,resume=False):
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    root=Path(root);p,m,n,identity=context(root);cfg=p['training'];run=root/'run'
    if (run/'report.json').exists():raise FileExistsError('Completed training')
    if run.exists() and any(run.iterdir()) and not resume:raise RuntimeError('Explicit resume required')
    require_space(root);run.mkdir(exist_ok=True)
    data=Resident([e for e in m['entries'] if e['partition']=='train'],n,cfg['frames'])
    seed_all(0);model=BeatConditionedModel().cuda();opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=.01)
    total=cfg['steps'];warm=cfg['warmup']
    schedule=lambda s:min(1.,(s+1)/warm)*(.1+.9*.5*(1+math.cos(math.pi*max(0,s-warm)/max(1,total-warm))))
    sch=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
    rng=np.random.default_rng(10000);flow_rng=torch.Generator(device='cuda').manual_seed(30000)
    pos=torch.tensor(n['pos_weight'],device='cuda',dtype=torch.float32);start=0;elapsed=0.
    identity={**identity,'training':cfg,'model_config':model.config_dict(),'seed':0}
    if not resume:
        atomic_torch(run/'initial.pt',{'model':model.state_dict(),'identity':identity,'normalizers':n,'step':0})
    else:
        z=torch.load(run/'latest.pt',map_location='cuda',weights_only=False)
        if z['identity']!=identity:raise RuntimeError('Resume identity differs')
        model.load_state_dict(z['model']);opt.load_state_dict(z['optimizer']);sch.load_state_dict(z['scheduler'])
        restore_rng(z['rng'],rng);flow_rng.set_state(z['flow_rng'].cpu());start=z['step'];elapsed=z['training_seconds']
    def save(step):
        z={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),
           'rng':rng_state(rng),'flow_rng':flow_rng.get_state(),'step':step,'training_seconds':elapsed,
           'identity':identity,'normalizers':n}
        atomic_torch(run/'latest.pt',z);return z
    model.train();torch.cuda.reset_peak_memory_stats()
    if not resume:save(0)
    with (run/'training.jsonl').open('a') as log:
        for step in range(start,total):
            torch.cuda.synchronize();begin=time.monotonic();x,y,b=data.batch(cfg['batch'],cfg['frames'],rng)
            opt.zero_grad(set_to_none=True)
            losses=model.losses(x,y,b,pos,generator=flow_rng,beat_dropout=cfg['beat_dropout'])
            if not all(torch.isfinite(v) for v in losses.values()):raise FloatingPointError('Nonfinite loss')
            losses['total'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
            opt.step();sch.step();torch.cuda.synchronize();elapsed+=time.monotonic()-begin
            if step==0 or (step+1)%100==0:
                row={'step':step+1,**{k:float(v.detach()) for k,v in losses.items()},'grad':float(grad),
                     'training_seconds':elapsed,'lr':sch.get_last_lr()[0]}
                log.write(json.dumps(row)+'\n');log.flush();atomic_json(run/'status.json',row);print(json.dumps(row),flush=True)
            if (step+1)%1000==0:save(step+1)
        final=save(total);atomic_torch(run/'final.pt',final)
    report={'status':'completed','identity':identity,'steps':total,'training_seconds':elapsed,
            'selection':'final_scheduled_step_only','excluded_short':data.excluded,'gpu':torch.cuda.get_device_name(),
            'precision':'fp32','peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'artifacts':artifact_manifest([run/'initial.pt',run/'final.pt',run/'latest.pt',run/'training.jsonl'])}
    authenticated_json(run/'report.json',report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    with gpu_lock(args.root):print(json.dumps(train(args.root,args.resume)),flush=True)

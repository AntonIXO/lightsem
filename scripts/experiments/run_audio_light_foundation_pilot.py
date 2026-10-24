"""One-seed target-led audio-to-light pilot with privileged dance supervision."""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from lightsem.experiments.audio_light_foundation.data import load_rows, IDS, make_fixture_metadata, sha
from lightsem.experiments.audio_light_foundation.model import AudioToLightFoundation,AudioToLightConfig
from lightsem.experiments.audio_light_foundation.losses import losses,teacher_objective,distillation_objective,cross_song_motion_objective
from lightsem.experiments.audio_light_foundation.evaluation import metrics
from lightsem.experiments.foundation_audio_light.evaluation import raw_body_metrics


def windows(rows,starts=(0,256,512,768),length=128,device='cpu'):
    out=[]
    for r in rows:
        for s in starts:
            if s+length<=len(r['audio']): out.append((r,s))
    if not out: raise RuntimeError('no valid windows')
    def stack(k):
        return torch.from_numpy(np.stack([r[k][s:s+length] for r,s in out])).float().to(device)
    return {'audio':stack('audio'),'body':stack('body'),'motion':stack('motion'),'speed':stack('speed'),'acceleration':stack('acceleration'),
            'rgb':stack('rgb'),'brightness':stack('brightness'),'event':stack('event'),'valid':torch.ones((len(out),length),dtype=torch.bool,device=device)},out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--arm',choices=('target_only','target_plus_dance_aux','target_plus_dance_aux_spread'),default='target_plus_dance_aux'); p.add_argument('--steps',type=int,default=80); p.add_argument('--window',type=int,default=128); p.add_argument('--sample-rate',type=int,choices=(10,30),default=10); p.add_argument('--out',required=True); p.add_argument('--device',default='auto'); a=p.parse_args()
    seed=0; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    rows=load_rows(sample_rate_hz=a.sample_rate); train_rows=rows[:6]; val_rows=rows[6:]
    starts=(0,256,512,768) if a.sample_rate==10 else (0,768,1536,2304)
    train,train_meta=windows(train_rows,starts=starts,length=a.window,device=device); val,val_meta=windows(val_rows,starts=starts,length=a.window,device=device)
    fixtures=torch.from_numpy(make_fixture_metadata()).to(device)[None].expand(len(val_meta),-1,-1)
    train_fixtures=torch.from_numpy(make_fixture_metadata()).to(device)[None].expand(len(train_meta),-1,-1)
    model=AudioToLightFoundation(AudioToLightConfig(audio_dim=int(train['audio'].shape[-1]),local_rate_hz=float(a.sample_rate))).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); history=[]; start=time.time(); auxiliary=a.arm in ('target_plus_dance_aux','target_plus_dance_aux_spread'); spread_loss=a.arm=='target_plus_dance_aux_spread'
    model.train()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        ts=torch.arange(a.window,device=device,dtype=torch.float32)[None].expand(len(train_meta),-1)/float(a.sample_rate)
        out=model(train['audio'],train_fixtures,timestamps=ts,valid=train['valid'],duration_seconds=a.window/float(a.sample_rate))
        # Dance is privileged training supervision.  The teacher embedding is
        # detached before distillation so audio inference cannot receive body
        # inputs and the teacher remains constrained by its body reconstruction.
        teacher=model.body_teacher(train['body'])
        out['dance_teacher']=teacher.detach()
        out['dance_body']=model.body_target(teacher)
        teacher_out=model.teacher_forward(train['body'],train_fixtures)
        tl=teacher_objective(teacher_out,train,train['valid']); dl=distillation_objective(out,teacher_out,train['valid'])
        l=losses(out,train,train['valid'],auxiliary,spread_loss=spread_loss)
        l.update(tl); l.update(dl)
        l['cross_song_aim']=cross_song_motion_objective(out,train, [r['id'] for r,s in train_meta], train['valid'])
        l['total']=l['total']+.5*sum(tl.values())+.25*sum(dl.values())+2.0*l['cross_song_aim']
        l['total'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        if step==0 or (step+1)%max(1,a.steps//4)==0: history.append({'step':step+1,**{k:float(v.detach().cpu()) for k,v in l.items()}})
    model.eval(); ts=torch.arange(a.window,device=device,dtype=torch.float32)[None].expand(len(val_meta),-1)/float(a.sample_rate)
    with torch.no_grad():
        out=model(val['audio'],fixtures,timestamps=ts,valid=val['valid'],duration_seconds=a.window/float(a.sample_rate)); teacher=model.body_teacher(val['body']); out['dance_teacher']=teacher.detach(); out['dance_body']=model.body_target(teacher); teacher_out=model.teacher_forward(val['body'],fixtures); val_l=losses(out,val,val['valid'],auxiliary,spread_loss=spread_loss); val_l.update(teacher_objective(teacher_out,val,val['valid'])); val_l.update(distillation_objective(out,teacher_out,val['valid'])); val_l['cross_song_aim']=cross_song_motion_objective(out,val,[r['id'] for r,s in val_meta],val['valid']); diagnostic_aux=losses(out,val,val['valid'],True,spread_loss=spread_loss)
    checks=metrics(model,val,fixtures); root=Path(a.out); root.mkdir(parents=True,exist_ok=False)
    torch.save({'state_dict':model.state_dict(),'config':model.config_dict(),'seed':seed,'arm':a.arm},root/'model.pt')
    provenance=[]
    for r in rows: provenance.append({'id':r['id'],'paths':{k:{'path':v,'sha256':sha(v)} for k,v in r['paths'].items()},'body_raw_metrics':raw_body_metrics(r['raw_world'])})
    result={'protocol_id':'audio-light-foundation-privileged-dance-20261015-v1','status':'diagnostic_only','arm':a.arm,'spread_loss':spread_loss,'seed':seed,'device':str(device),'steps':a.steps,'train_songs':[r['id'] for r in train_rows],'validation_songs':[r['id'] for r in val_rows],'history':history,'validation_losses':{k:float(v.detach().cpu()) for k,v in val_l.items()},'auxiliary_diagnostics':{k:float(v.detach().cpu()) for k,v in diagnostic_aux.items()},'checks':checks,'model_config':model.config_dict(),'provenance':provenance,'inference_contract':'audio+fixture metadata only; body teacher branch absent at inference','promotion':'forbidden'}
    (root/'RESULTS.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); (root/'CONFIG.json').write_text(json.dumps({'args':vars(a),'model':model.config_dict()},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'out':str(root),'status':result['status'],'arm':a.arm,'validation':result['validation_losses'],'checks':checks},indent=2))

if __name__=='__main__': main()

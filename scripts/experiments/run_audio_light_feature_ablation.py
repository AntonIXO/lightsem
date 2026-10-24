"""Controlled feature/clock ablation for the audio-only LightSem pilot.

All arms keep the 1116-wide interface, model width, split, seed, windows and
optimizer budget fixed. Disabled feature groups are zeroed at the input; this
is an intervention, not a claim that zero is a natural missing-value code.
No rendering is performed.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np, torch
from lightsem.experiments.audio_light_foundation.data import load_rows, make_fixture_metadata, sha
from lightsem.experiments.audio_light_foundation.model import AudioToLightFoundation, AudioToLightConfig
from lightsem.experiments.audio_light_foundation.losses import losses, teacher_objective, distillation_objective, cross_song_motion_objective

ARMS={"full":"full","muq_only":"muq_only","base92_only":"base92_only","no_clock":"no_clock"}

def make_windows(rows, starts, length, device):
    pairs=[(r,s) for r in rows for s in starts if s+length<=len(r['audio'])]
    def st(k): return torch.from_numpy(np.stack([r[k][s:s+length] for r,s in pairs])).float().to(device)
    return {k:st(k) for k in ('audio','body','motion','speed','acceleration','rgb','brightness','event')}, pairs

def input_arm(x, arm):
    y=x.clone()
    if arm=='muq_only': y[...,1024:]=0
    elif arm=='base92_only': y[...,:1024]=0
    return y

def times(batch, length, rate, zero=False):
    if zero: return torch.zeros((batch,length),device='cuda' if torch.cuda.is_available() else 'cpu')
    return torch.arange(length,device='cuda' if torch.cuda.is_available() else 'cpu',dtype=torch.float32)[None].expand(batch,-1)/rate

def masked_mean(x): return float(x.detach().cpu())
def metrics(out, target, pairs, rate):
    aim=out['aim'].detach().cpu().numpy(); rgb=out['rgb'].detach().cpu().numpy(); br=out['brightness'].detach().cpu().numpy(); ev=out['event'].detach().cpu().numpy();
    ta=target['motion'].detach().cpu().numpy(); tr=target['rgb'].detach().cpu().numpy(); tb=target['brightness'].detach().cpu().numpy(); te=target['event'].detach().cpu().numpy()
    def p99_deriv(x, order):
        z=x
        for _ in range(order): z=np.diff(z,axis=1,prepend=z[:,:1])*rate
        return float(np.quantile(np.linalg.norm(z,axis=-1),.99))
    # Aim proxy is body-derived and not physical pan/tilt.
    aim_mae=float(np.mean(np.abs(aim-ta))); rgb_mae=float(np.mean(np.abs(rgb-tr))); br_mae=float(np.mean(np.abs(br-tb)))
    ev_bce=float(np.mean(np.maximum(ev,0)-ev*te+np.log1p(np.exp(-np.abs(ev)))))
    # Pair corresponding windows of the two held-out songs after removing the fixture centroid.
    by={}
    for i,(r,s) in enumerate(pairs): by.setdefault(r['id'],[]).append(i)
    cors=[]
    ids=sorted(by)
    if len(ids)>=2:
      for ia,ib in zip(by[ids[0]],by[ids[1]]):
        a=aim[ia]-aim[ia].mean(1,keepdims=True); b=aim[ib]-aim[ib].mean(1,keepdims=True)
        cors.append(float(np.corrcoef(a.reshape(-1),b.reshape(-1))[0,1]))
    return {'aim_mae':aim_mae,'rgb_mae':rgb_mae,'brightness_mae':br_mae,'event_bce':ev_bce,
            'aim_velocity_p99':p99_deriv(aim,1),'aim_acceleration_p99':p99_deriv(aim,2),'aim_jerk_p99':p99_deriv(aim,3),
            'cross_song_aim_corr_mean':float(np.mean(cors)) if cors else None,
            'aim_spread_mean':float(np.mean(np.linalg.norm(aim[:,:,None,:]-aim[:,None,:,:],axis=-1))),
            'finite':bool(np.isfinite(aim).all() and np.isfinite(rgb).all() and np.isfinite(br).all() and np.isfinite(ev).all())}

def run_arm(arm, rows, train, train_pairs, val, val_pairs, steps, rate, device, out_root):
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    model=AudioToLightFoundation(AudioToLightConfig(audio_dim=1116,local_rate_hz=rate)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    tf=torch.from_numpy(make_fixture_metadata()).float().to(device)
    train_fx=tf[None].expand(len(train_pairs),-1,-1); val_fx=tf[None].expand(len(val_pairs),-1,-1)
    train_x=input_arm(train['audio'],arm); val_x=input_arm(val['audio'],arm)
    train_valid=torch.ones(train_x.shape[:2],dtype=torch.bool,device=device); val_valid=torch.ones(val_x.shape[:2],dtype=torch.bool,device=device)
    no_clock=arm=='no_clock'
    ts_train=torch.zeros_like(train_valid,dtype=torch.float32) if no_clock else torch.arange(train_x.shape[1],device=device,dtype=torch.float32)[None].expand(len(train_pairs),-1)/rate
    ts_val=torch.zeros_like(val_valid,dtype=torch.float32) if no_clock else torch.arange(val_x.shape[1],device=device,dtype=torch.float32)[None].expand(len(val_pairs),-1)/rate
    history=[]
    model.train()
    for step in range(steps):
      opt.zero_grad(set_to_none=True)
      out=model(train_x,train_fx,timestamps=ts_train,valid=train_valid,duration_seconds=train_x.shape[1]/rate)
      teacher=model.body_teacher(train['body']); out['dance_teacher']=teacher.detach(); out['dance_body']=model.body_target(teacher)
      teacher_out=model.teacher_forward(train['body'],train_fx)
      tl=teacher_objective(teacher_out,train,train_valid); dl=distillation_objective(out,teacher_out,train_valid)
      l=losses(out,train,train_valid,True,spread_loss=True); l.update(tl); l.update(dl)
      l['cross_song_aim']=cross_song_motion_objective(out,train,[r['id'] for r,s in train_pairs],train_valid)
      l['total']=l['total']+.5*sum(tl.values())+.25*sum(dl.values())+2*l['cross_song_aim']; l['total'].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
      if step==0 or step+1==steps: history.append({'step':step+1,**{k:float(v.detach().cpu()) for k,v in l.items()}})
    model.eval()
    with torch.no_grad():
      out=model(val_x,val_fx,timestamps=ts_val,valid=val_valid,duration_seconds=val_x.shape[1]/rate)
      teacher=model.body_teacher(val['body']); out['dance_teacher']=teacher.detach(); out['dance_body']=model.body_target(teacher)
      vl=losses(out,val,val_valid,True,spread_loss=True)
      # Within-song temporal shuffle control: preserve each sample's marginal features but destroy local alignment.
      shuffled=val_x.clone(); g=torch.Generator(device=device).manual_seed(123)
      for i in range(shuffled.shape[0]): shuffled[i]=shuffled[i,torch.randperm(shuffled.shape[1],generator=g,device=device)]
      sh=model(shuffled,val_fx,timestamps=ts_val,valid=val_valid,duration_seconds=val_x.shape[1]/rate)
    m=metrics(out,val,val_pairs,rate); m['temporal_audio_shuffle_aim_delta']=float((out['aim']-sh['aim']).abs().mean().cpu()); m['temporal_audio_shuffle_rgb_delta']=float((out['rgb']-sh['rgb']).abs().mean().cpu()); m['temporal_audio_shuffle_brightness_delta']=float((out['brightness']-sh['brightness']).abs().mean().cpu())
    root=out_root/arm; root.mkdir(parents=True,exist_ok=False); torch.save({'state_dict':model.state_dict(),'config':model.config_dict(),'arm':arm,'seed':0},root/'model.pt')
    (root/'RESULTS.json').write_text(json.dumps({'arm':arm,'steps':steps,'rate':rate,'history':history,'validation_losses':{k:float(v.detach().cpu()) for k,v in vl.items()},'metrics':m,'promotion':'forbidden'},indent=2,sort_keys=True)+'\n')
    return {'arm':arm,'validation_losses':{k:float(v.detach().cpu()) for k,v in vl.items()},'metrics':m}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--steps',type=int,default=400); p.add_argument('--rate',type=int,choices=(30,),default=30); a=p.parse_args()
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=load_rows(sample_rate_hz=a.rate); train_rows=rows[:6]; val_rows=rows[6:]; starts=(0,768,1536,2304); length=384
 train,tp=make_windows(train_rows,starts,length,device); val,vp=make_windows(val_rows,starts,length,device); root=Path(a.out); root.mkdir(parents=True,exist_ok=False)
 # trivial baselines are computed on held-out target and are reported for every target head.
 rgb_val=np.concatenate([r['rgb'][s:s+length] for r,s in vp]); br_val=np.concatenate([r['brightness'][s:s+length] for r,s in vp])
 base={'rgb_mae_median':float(np.mean(np.abs(rgb_val-np.median(rgb_val,axis=0,keepdims=True)))), 'brightness_mae_median':float(np.mean(np.abs(br_val-np.median(br_val,axis=0,keepdims=True))))}
 results={'protocol':'audio-light-feature-ablation-20261015-v1','status':'diagnostic_only','seed':0,'steps':a.steps,'rate':a.rate,'train_songs':[r['id'] for r in train_rows],'validation_songs':[r['id'] for r in val_rows],'arms':{},'baselines':base,'notes':['zeroed feature groups are controls, not natural missingness','aim target is body-derived proxy, not physical pan/tilt','no rendering or promotion']}
 for arm in ARMS: results['arms'][arm]=run_arm(arm,rows,train,tp,val,vp,a.steps,a.rate,device,root)
 (root/'SUMMARY.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n'); print(json.dumps(results,indent=2))
if __name__=='__main__': main()

from __future__ import annotations
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from lightsem.experiments.audio_light_foundation.model import AudioToLightFoundation, AudioToLightConfig
from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata

CONTRACT=Path('/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_contract_20261016_v1/CONTRACT.json')
MUQ=Path('/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3')
ROOT=Path('/mnt/lightsem-data/cache')

def one(root,s):
 p=sorted(Path(root).glob(f'*={s}.npz'))
 if len(p)!=1: raise FileNotFoundError((root,s,len(p)))
 return p[0]

def _nearest(values,times,target):
 right=np.searchsorted(times,target,side='left').clip(0,len(times)-1); left=np.maximum(right-1,0)
 return values[np.where(np.abs(times[left]-target)<=np.abs(times[right]-target),left,right)]

def load(row):
 sid=row['id']; ap=one(ROOT/'beatsaver_audio',sid); vp=one(ROOT/'beatsaver_visual',sid); lp=one(ROOT/'beatsaver_light',sid); mp=MUQ/f'{sid}.npz'
 with np.load(ap,allow_pickle=False) as z:
  mel=np.asarray(z['mel'],np.float32).T; scalar=('rms','rms_db','flux','flux_lf','flux_lmid','flux_mid','flux_hmid','flux_hf','energy_lf','energy_lmid','energy_mid','energy_hmid','energy_hf','onset','onset_ema','flux_ema','flux_lf_ema','flux_hf_ema','centroid','flatness','rolloff','percussive_ratio','pulse_clarity','beat_phase_sin','beat_phase_cos','beat_accent','bar_phase','beat_period_s'); names=[x for x in z.files if x.startswith('ch__')]; base=np.concatenate([mel,np.stack([z['ch__'+x] for x in scalar],1).astype(np.float32)],1); bt=np.arange(len(base),dtype=np.float32)/40
 with np.load(mp,allow_pickle=False) as z: muq=np.asarray(z['muq'],np.float32); mt=np.asarray(z['muq_time'],np.float32)
 with np.load(vp,allow_pickle=False) as z: ev=np.asarray(z['ev_any'],np.float32); et=np.arange(len(ev),dtype=np.float32)/10
 with np.load(lp,allow_pickle=False) as z: rgb=np.asarray(z['rgb'],np.float32); inten=np.asarray(z['inten'],np.float32)
 n=min(len(ev),len(rgb[0])//4,len(muq),int(np.floor(float(mt[-1])*10))+1,int(np.floor(float(bt[-1])*10))+1)
 t=np.arange(n,dtype=np.float32)/10
 audio=np.concatenate([_nearest(muq,mt,t),_nearest(base,bt,t)],1)
 rb=rgb[:,:n*4].reshape(rgb.shape[0],n,4,3).mean(2).transpose(1,0,2); ib=inten[:,:n*4].reshape(inten.shape[0],n,4).mean(2).transpose(1,0)[...,None]; groups=np.arange(8)%max(1,rb.shape[1])
 return {'id':sid,'wav_sha256':row['wav_sha256'],'audio':audio,'rgb':np.clip(rb[:,groups],0,1),'brightness':np.clip(ib[:,groups],0,1),'event':_nearest(ev,et,t)[:,None]}

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def metric(o,t):
 v=t['valid']; m=lambda x,mask: (x*mask.expand_as(x)).sum()/mask.expand_as(x).sum().clamp_min(1); return {'rgb':m((o['rgb']-t['rgb']).abs(),v[...,None,None]),'brightness':m((o['brightness']-t['brightness']).abs(),v[...,None,None]),'event':m(F.binary_cross_entropy_with_logits(o['event'],t['event'],reduction='none'),v[...,None])}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--steps',type=int,default=2500); ap.add_argument('--seed',type=int,default=0); a=ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); started=time.time()
 contract=json.loads(CONTRACT.read_text()); rows=[load(x) for x in contract['rows']]; groups={};
 for r in rows: groups.setdefault(r['wav_sha256'],[]).append(r)
 gkeys=np.array(sorted(groups)); rng=np.random.default_rng(a.seed); gkeys=gkeys[rng.permutation(len(gkeys))]; cut=max(1,int(.8*len(gkeys))); trgroups=set(gkeys[:cut]); trainrows=[r for r in rows if r['wav_sha256'] in trgroups]; valrows=[r for r in rows if r['wav_sha256'] not in trgroups]
 W=128; starts=lambda r:list(range(0,max(1,len(r['audio'])-W+1),W)); train=[(r,s) for r in trainrows for s in starts(r) if s+W<=len(r['audio'])]; valid=[(r,s) for r in valrows for s in starts(r) if s+W<=len(r['audio'])]
 fx=torch.from_numpy(make_fixture_metadata()).float().to(dev); model=AudioToLightFoundation(AudioToLightConfig(audio_dim=1116,local_rate_hz=10)).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); hist=[]; bs=16
 def batch(ix): return {'audio':torch.from_numpy(np.stack([r['audio'][s:s+W] for r,s in ix])).float().to(dev),'rgb':torch.from_numpy(np.stack([r['rgb'][s:s+W] for r,s in ix])).float().to(dev),'brightness':torch.from_numpy(np.stack([r['brightness'][s:s+W] for r,s in ix])).float().to(dev),'event':torch.from_numpy(np.stack([r['event'][s:s+W] for r,s in ix])).float().to(dev),'valid':torch.ones((len(ix),W),dtype=torch.bool,device=dev)}
 model.train()
 for step in range(a.steps):
  ix=[train[(step*bs+j)%len(train)] for j in range(bs)]; t=batch(ix); ts=torch.arange(W,device=dev).float()[None].expand(bs,-1)/10; o=model(t['audio'],fx[None].expand(bs,-1,-1),timestamps=ts,valid=t['valid'],duration_seconds=W/10); lparts=metric(o,t); l=sum((1.,2.5,1.5)[i]*list(lparts.values())[i] for i in range(3)); opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
  if step==0 or (step+1)%500==0: hist.append({'step':step+1,'total':float(l.detach().cpu()),**{k:float(v.detach().cpu()) for k,v in lparts.items()}})
 model.eval(); audit=[]
 with torch.no_grad():
  for r in valrows[:2]:
   n=min(len(r['audio']),512); t=batch([(r,0)]); # batch is W-sized; use direct full prefix below
   aa=torch.from_numpy(r['audio'][:n][None]).float().to(dev); vv=torch.ones((1,n),dtype=torch.bool,device=dev); ts=torch.arange(n,device=dev).float()[None]/10; o=model(aa,fx[None],timestamps=ts,valid=vv,duration_seconds=n/10)
   angle=o['aim'].detach().cpu().numpy()[0]; rgb=o['rgb'].detach().cpu().numpy()[0]; br=o['brightness'].detach().cpu().numpy()[0]; ev=torch.sigmoid(o['event']).detach().cpu().numpy()[0,:,0]; lat=o['audio_latent'].detach().cpu().numpy()[0]
   rank=int(np.linalg.matrix_rank(lat-lat.mean(0),tol=1e-4)); audit.append({'id':r['id'],'rgb_std':float(rgb.std()),'rgb_below001_fraction':float(np.mean(rgb<.01)),'brightness_std':float(br.std()),'event_mean':float(ev.mean()),'event_entropy':float(-(ev.mean()*np.log(max(ev.mean(),1e-8))+(1-ev.mean())*np.log(max(1-ev.mean(),1e-8)))),'latent_effective_rank':rank,'angle_spread_std':float(angle.std()),'finite':bool(np.isfinite(rgb).all() and np.isfinite(br).all() and np.isfinite(ev).all())})
 # deterministic/audio swap on the first validation prefix
 r0,r1=valrows[0],valrows[1]; n=min(len(r0['audio']),len(r1['audio']),512); with_audio=lambda r: torch.from_numpy(r['audio'][:n][None]).float().to(dev); vv=torch.ones((1,n),dtype=torch.bool,device=dev); ts=torch.arange(n,device=dev).float()[None]/10
 with torch.no_grad():
  o1=model(with_audio(r0),fx[None],timestamps=ts,valid=vv,duration_seconds=n/10); o2=model(with_audio(r0),fx[None],timestamps=ts,valid=vv,duration_seconds=n/10); osw=model(with_audio(r1),fx[None],timestamps=ts,valid=vv,duration_seconds=n/10)
 det=max(float(torch.max(torch.abs(o1[k]-o2[k])).cpu()) for k in ['aim','rgb','brightness','event']); swap={k:float(torch.mean(torch.abs(o1[k]-osw[k])).cpu()) for k in ['aim','rgb','brightness','event']}
 out=Path(a.out); out.mkdir(parents=True,exist_ok=False); ck=out/'model.pt'; torch.save({'state_dict':model.state_dict(),'config':model.config_dict(),'seed':a.seed,'input_contract':'canonical MuQ1024 layer9 + Base92 92 = 1116; no zero padding'},ck); res={'schema_version':1,'protocol':'audio-light-foundation-v2-all300-canonical-20261016-v1','status':'diagnostic_only','rows':len(rows),'unique_audio_groups':len(groups),'train_groups':len(trgroups),'validation_groups':len(gkeys)-len(trgroups),'train_songs':len(trainrows),'validation_songs':len(valrows),'train_windows':len(train),'validation_windows':len(valid),'steps':a.steps,'seed':a.seed,'runtime_seconds':time.time()-started,'input_contract':'canonical MuQ1024 layer9 + Base92 92 = 1116-D; no zero padding; 52-D visual x excluded','muq_root':str(MUQ),'muq_receipt_path':str(MUQ/'COMPLETE_V2.json'),'muq_receipt_sha256':sha(MUQ/'COMPLETE_V2.json'),'history':hist,'validation_audit':audit,'deterministic_max_abs_diff':det,'audio_swap_mean_abs_delta':swap,'verdict':'collapse_only diagnostic; promotion forbidden until independent raw motion and two-song gates pass','promotion':'forbidden','model_checkpoint':str(ck),'model_checkpoint_sha256':sha(ck)}; (out/'RESULTS.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()

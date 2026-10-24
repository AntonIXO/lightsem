from __future__ import annotations
import argparse,json,hashlib,random
from pathlib import Path
import numpy as np, torch
from lightsem.experiments.audio_light_foundation.model import AudioToLightFoundation,AudioToLightConfig
from lightsem.experiments.audio_light_foundation.data import make_fixture_metadata
ROOT=Path('/mnt/lightsem-data/cache'); WAV=Path('/mnt/lightsem-data/beatsaver/wav')
def one(root,s):
 p=sorted((root).glob(f'*={s}.npz')); return p[0] if len(p)==1 else None
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b):h.update(b)
 return h.hexdigest()
def load(s):
 ap=one(ROOT/'beatsaver_audio',s);vp=one(ROOT/'beatsaver_visual',s);lp=one(ROOT/'beatsaver_light',s)
 if not(ap and vp and lp): return None
 with np.load(ap,allow_pickle=False) as z:
  mel=z['mel'].astype('f4').T; names=[x for x in z.files if x.startswith('ch__')]; base=np.stack([z[x] for x in names],1).astype('f4')
 # Base92-compatible: mel64 + 28 scalar channels in AUDIO_SCALARS ordering.
 scalar=('rms','rms_db','flux','flux_lf','flux_lmid','flux_mid','flux_hmid','flux_hf','energy_lf','energy_lmid','energy_mid','energy_hmid','energy_hf','onset','onset_ema','flux_ema','flux_lf_ema','flux_hf_ema','centroid','flatness','rolloff','percussive_ratio','pulse_clarity','beat_phase_sin','beat_phase_cos','beat_accent','bar_phase','beat_period_s')
 idx=[names.index('ch__'+x) for x in scalar]
 x=np.concatenate([mel,base[:,idx]],1)
 with np.load(vp,allow_pickle=False) as z: y=z['x'].astype('f4'); ev=z['ev_any'].astype('f4')[:,None]
 with np.load(lp,allow_pickle=False) as z:
  rgb=z['rgb'].astype('f4'); inten=z['inten'].astype('f4')
 n=min(len(x),len(y),len(ev),rgb.shape[1]//4)
 # 40Hz light -> 10Hz target, canonical 8 virtual slots.
 rgb=rgb[:,:n*4].reshape(rgb.shape[0],n,4,3).mean(2).transpose(1,0,2); inten=inten[:,:n*4].reshape(inten.shape[0],n,4).mean(2).transpose(1,0)[...,None]
 groups=np.arange(8)%max(1,rgb.shape[1]); return {'id':s,'audio':x[:n],'rgb':np.clip(rgb[:n,groups],0,1),'brightness':np.clip(inten[:n,groups],0,1),'event':ev[:n],'motion':np.zeros((n,8,2),dtype='f4')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=2500);p.add_argument('--seed',type=int,default=0);a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 ids=sorted({Path(x).stem.split('=')[-1] for x in (ROOT/'beatsaver_audio').glob('*.npz')}); rows=[load(s) for s in ids]; rows=[r for r in rows if r]; rng=np.random.default_rng(a.seed); order=rng.permutation(len(rows)); cut=max(1,int(.8*len(rows))); tr=[rows[i] for i in order[:cut]];va=[rows[i] for i in order[cut:]]
 # windows indexed on CPU, fixed 128@10Hz; no all-data GPU stacking.
 W=128; starts=lambda r:list(range(0,max(1,len(r['audio'])-W+1),W))
 train=[(r,s) for r in tr for s in starts(r) if s+W<=len(r['audio'])]; valid=[(r,s) for r in va for s in starts(r) if s+W<=len(r['audio'])]
 model=AudioToLightFoundation(AudioToLightConfig(audio_dim=1116,local_rate_hz=10)).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); fx=torch.from_numpy(make_fixture_metadata()).float().to(dev); hist=[]
 def batch(ix):
  A=np.stack([np.pad(r['audio'][s:s+W],((0,0),(1024,0))) for r,s in ix]); return {'audio':torch.from_numpy(A).float().to(dev),'rgb':torch.from_numpy(np.stack([r['rgb'][s:s+W] for r,s in ix])).float().to(dev),'brightness':torch.from_numpy(np.stack([r['brightness'][s:s+W] for r,s in ix])).float().to(dev),'event':torch.from_numpy(np.stack([r['event'][s:s+W] for r,s in ix])).float().to(dev),'valid':torch.ones((len(ix),W),dtype=torch.bool,device=dev)}
 def loss(o,t):
  v=t['valid']; mm=lambda x,m:torch.where(m.expand_as(x),x,torch.zeros_like(x)).sum()/m.expand_as(x).sum().clamp_min(1); rgb=mm((o['rgb']-t['rgb']).abs(),v[...,None,None]); br=mm((o['brightness']-t['brightness']).abs(),v[...,None,None]); ev=mm(torch.nn.functional.binary_cross_entropy_with_logits(o['event'],t['event'],reduction='none'),v[...,None]); return rgb+2.5*br+1.5*ev,{'rgb':rgb,'brightness':br,'event':ev}
 model.train(); bs=16
 for step in range(a.steps):
  ix=[train[(step*bs+j)%len(train)] for j in range(bs)]; t=batch(ix); ts=torch.arange(W,device=dev).float()[None].expand(bs,-1)/10; o=model(t['audio'],fx[None].expand(bs,-1,-1),timestamps=ts,valid=t['valid'],duration_seconds=W/10); l,parts=loss(o,t); opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if step==0 or (step+1)%500==0: hist.append({'step':step+1,'total':float(l.detach().cpu()),**{k:float(v.detach().cpu()) for k,v in parts.items()}})
 model.eval();ix=valid[:min(256,len(valid))];t=batch(ix);ts=torch.arange(W,device=dev).float()[None].expand(len(ix),-1)/10
 with torch.no_grad():o=model(t['audio'],fx[None].expand(len(ix),-1,-1),timestamps=ts,valid=t['valid'],duration_seconds=W/10);_,parts=loss(o,t)
 root=Path(a.out);root.mkdir(parents=True,exist_ok=False);torch.save({'state_dict':model.state_dict(),'config':model.config_dict(),'seed':a.seed,'input_contract':'Base92 padded to 1116; MuQ zeroed; target-only'},root/'model.pt'); res={'protocol':'audio-light-all300-base92-targetonly-20261016-v1','status':'diagnostic_only','rows':len(rows),'train_songs':len(tr),'validation_songs':len(va),'train_windows':len(train),'validation_windows':len(valid),'steps':a.steps,'history':hist,'validation_losses':{k:float(v.detach().cpu()) for k,v in parts.items()},'input_contract':'92-D authenticated BeatSaver audio cache padded with zero MuQ channels; no MuQ/body/dance; authored RGB/brightness/event only','promotion':'forbidden'};(root/'RESULTS.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');print(json.dumps(res,indent=2))
if __name__=='__main__':main()

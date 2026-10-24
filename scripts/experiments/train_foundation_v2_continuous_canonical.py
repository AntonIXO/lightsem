from __future__ import annotations
import json, random, time, hashlib
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
import sys
sys.path.insert(0,'/root/lightsem')
from lightsem.experiments.audio_light_foundation.data import load_rows, make_fixture_metadata
from lightsem.experiments.foundation_audio_light import FoundationAudioLightModel, FoundationConfig
from lightsem.experiments.foundation_audio_light.motion_head import MotionHead
from lightsem.experiments.foundation_audio_light.safety_projection import SafetyConfig
class AnchoredModel(FoundationAudioLightModel):
 def forward(self,audio,body,fixtures,**kwargs):
  out=super().forward(audio,body,fixtures,**kwargs)
  anchor=torch.stack((torch.atan2(fixtures[...,1],fixtures[...,0])/torch.pi*0.90,fixtures[...,1]*0.25),-1)
  out['angle']=(anchor[:,None]+0.10*out['angle']).clamp(-0.99,0.99)
  return out

OUT=Path('/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v9_warm_v7')
SEED=0; STEPS=400; WINDOW=256; CHUNK=32; FPS=10.
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def crop(r):
 n=min(len(r[k]) for k in ['audio','body','motion','speed','acceleration','rgb','brightness','event'])
 return {**r,**{k:r[k][:n] for k in ['audio','body','motion','speed','acceleration','rgb','brightness','event']}}
def pairwise(x):
 d=x[:,:,None,:]-x[:,None,:,:]; return torch.linalg.vector_norm(d,dim=-1)
def loss(out,t):
 mask4=t['valid'][...,None,None]
 mask3=t['valid'][...,None]
 angle_err=F.smooth_l1_loss(out['angle'],t['motion'],reduction='none'); angle=(angle_err*mask4).sum()/mask4.expand_as(angle_err).sum().clamp_min(1)
 pd=pairwise(out['angle']); td=pairwise(t['motion']); spread_err=F.smooth_l1_loss(pd,td,reduction='none'); spread=(spread_err*t['valid'][:, :, None, None]).sum()/(t['valid'][:, :, None, None].expand_as(spread_err).sum().clamp_min(1))
 vel_target=(t['speed']*0.45).expand_as(out['velocity']); vel_err=F.smooth_l1_loss(out['velocity'],vel_target,reduction='none'); vel=(vel_err*mask4).sum()/mask4.expand_as(vel_err).sum().clamp_min(1)
 acc_target=(t['acceleration']*1.5).expand_as(out['acceleration']); acc_err=F.smooth_l1_loss(out['acceleration'],acc_target,reduction='none'); acc=(acc_err*mask4).sum()/mask4.expand_as(acc_err).sum().clamp_min(1)
 rgb_err=(out['rgb']-t['rgb']).abs(); rgb=(rgb_err*mask4).sum()/mask4.expand_as(rgb_err).sum().clamp_min(1)
 br_err=(out['brightness']-t['brightness']).abs(); br=(br_err*mask4).sum()/mask4.expand_as(br_err).sum().clamp_min(1)
 ev_err=F.binary_cross_entropy_with_logits(out['event'],t['event'],reduction='none'); ev=(ev_err*mask3).sum()/mask3.expand_as(ev_err).sum().clamp_min(1)
 total=10*angle+8*spread+0.5*vel+0.25*acc+rgb+br+0.5*ev
 return total,{'angle':angle,'spread':spread,'velocity':vel,'acceleration':acc,'rgb':rgb,'brightness':br,'event':ev}
def make_batch(rows, device, rng):
 chosen=[]
 for _ in range(4):
  r=rows[int(rng.integers(len(rows)))]; s=int(rng.integers(0,len(r['audio'])-WINDOW+1)); chosen.append((r,s))
 def st(k): return torch.from_numpy(np.stack([r[k][s:s+WINDOW] for r,s in chosen])).float().to(device)
 return {'audio':st('audio'),'body':st('body'),'motion':st('motion'),'speed':st('speed'),'acceleration':st('acceleration'),'rgb':st('rgb'),'brightness':st('brightness'),'event':st('event'),'valid':torch.ones((4,WINDOW),dtype=torch.bool,device=device)}
def forward_chunked(model,audio,body,fx):
 state=None; outacc={}; n=audio.shape[1]
 with torch.no_grad():
  for s in range(0,n,CHUNK):
   e=min(n,s+CHUNK); ts=torch.arange(s,e,device=audio.device).float()[None].expand(audio.shape[0],-1)/FPS
   o=model(audio[:,s:e],body[:,s:e],fx,timestamps=ts,valid=torch.ones((audio.shape[0],e-s),dtype=torch.bool,device=audio.device),state=state,duration_seconds=n/FPS,return_state=True); state=o['state']
   for k in ['angle','velocity','acceleration','jerk','rgb','brightness','event']: outacc.setdefault(k,[]).append(o[k].detach().cpu())
 return {k:torch.cat(v,1).numpy() for k,v in outacc.items()}
def main():
 random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); rng=np.random.default_rng(SEED); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 rows=[crop(r) for r in load_rows(sample_rate_hz=10, muq_root='/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3')]; train=[r for r in rows[:6] if len(r['audio'])>=WINDOW]; val=rows[6:]; fx=torch.from_numpy(make_fixture_metadata()).float().to(device)[None].expand(2,-1,-1)
 cfg=FoundationConfig(audio_dim=1116); model=AnchoredModel(cfg).to(device); model.motion=MotionHead(cfg.width,SafetyConfig(dt=0.1,max_angle=1.0,max_velocity=0.18,max_acceleration=0.5,max_jerk=1.5)).to(device); warm=Path('/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v7_bounded_residual/model.pt'); model.load_state_dict(torch.load(warm,map_location=device)['state_dict'])
 opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); hist=[]; started=time.time(); model.train()
 for step in range(STEPS):
  t=make_batch(train,device,rng); ts=torch.arange(WINDOW,device=device).float()[None].expand(4,-1)/FPS; o=model(t['audio'],t['body'],torch.from_numpy(make_fixture_metadata()).float().to(device)[None].expand(4,-1,-1),timestamps=ts,valid=t['valid'],duration_seconds=WINDOW/FPS,return_state=True); l,parts=loss(o,t); opt.zero_grad(set_to_none=True); l.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
  if step==0 or (step+1)%100==0: hist.append({'step':step+1,'total':float(l.detach().cpu()),**{k:float(v.detach().cpu()) for k,v in parts.items()}})
 model.eval(); val_preds=[]
 for r in val:
  va_audio=torch.from_numpy(r['audio'][None]).float().to(device); va_body=torch.from_numpy(r['body'][None]).float().to(device); fx1=torch.from_numpy(make_fixture_metadata()[None]).float().to(device); val_preds.append((r['id'],forward_chunked(model,va_audio,va_body,fx1)))
 pred=val_preds[-1][1]
 OUT.mkdir(parents=True,exist_ok=False); torch.save({'state_dict':model.state_dict(),'config':{**model.config_dict(),'motion_safety':{'dt':0.1,'max_velocity':0.18,'max_acceleration':0.5,'max_jerk':1.5}},'seed':SEED},OUT/'model.pt'); np.savez_compressed(OUT/'PREDICTIONS_10HZ.npz',time=np.arange(pred['angle'].shape[1])/FPS,**{k:v for k,v in pred.items()}); receipt={'schema_version':1,'protocol':'audio-light-foundation-v2-continuous-20261016-v1','status':'diagnostic_only','seed':SEED,'steps':STEPS,'warm_start':str(warm),'window':WINDOW,'chunk':CHUNK,'device':str(device),'train_songs':[r['id'] for r in train],'validation_songs':[r['id'] for r in val],'model_checkpoint':str(OUT/'model.pt'),'model_checkpoint_sha256':sha(OUT/'model.pt'),'predictions_10hz':str(OUT/'PREDICTIONS_10HZ.npz'),'predictions_sha256':sha(OUT/'PREDICTIONS_10HZ.npz'),'history':hist,'runtime_seconds':time.time()-started,'inference_contract':'canonical MuQ1024 layer9 + Base92 1116-D, body-conditioned continuous attention plus fixture-metadata spatial anchor and learned residual; no DMX/video yet','promotion':'forbidden pending collapse/slow-motion/different-side audit'}; (OUT/'TRAINING.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+chr(10)); print(json.dumps(receipt,indent=2))
main()

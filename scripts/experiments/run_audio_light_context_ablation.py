"""Bounded offline bidirectional context comparison, same LightSem pilot recipe."""
from pathlib import Path
import argparse,json,random,numpy as np,torch
from torch import nn
from lightsem.experiments.audio_light_foundation.data import load_rows,make_fixture_metadata
from lightsem.experiments.audio_light_foundation.model import AudioToLightFoundation,AudioToLightConfig
from lightsem.experiments.audio_light_foundation.losses import losses,teacher_objective,distillation_objective,cross_song_motion_objective
from scripts.experiments.run_audio_light_feature_ablation import make_windows,metrics
class BoundedBidirectional(nn.Module):
 def __init__(self,base,halo=16):
  super().__init__(); self.base=base; d=base.config.width; self.halo=halo
  self.attn=nn.MultiheadAttention(d,base.config.heads,batch_first=True); self.norm=nn.LayerNorm(d)
 def forward(self,audio,fixtures,timestamps=None,valid=None,state=None,duration_seconds=None):
  x=self.base.audio(audio,timestamps=timestamps,valid=valid,duration_seconds=duration_seconds); B,T,D=x.shape; chunks=[]
  # Each 32-frame query block sees a 16-frame future and past halo only.
  for s in range(0,T,32):
   e=min(T,s+32); ks=max(0,s-self.halo); ke=min(T,e+self.halo); q=x[:,s:e]; kv=x[:,ks:ke]
   kvalid=valid[:,ks:ke] if valid is not None else None
   z,_=self.attn(q,kv,kv,key_padding_mask=(~kvalid if kvalid is not None else None),need_weights=False)
   chunks.append(self.norm(q+z))
  x=torch.cat(chunks,1); hidden,new_state=self.base.context(x,valid,state,return_state=True)
  seed=self.base.context.latent_seed[None].expand(B,-1,-1); out=self.base._decode(hidden,fixtures,seed); out['audio_latent']=hidden; out['state']=new_state; return out
 def teacher_forward(self,*a,**kw): return self.base.teacher_forward(*a,**kw)
 def body_teacher(self,*a,**kw): return self.base.body_teacher(*a,**kw)
 def body_target(self,*a,**kw): return self.base.body_target(*a,**kw)
 def config_dict(self): return {**self.base.config_dict(),'context_variant':'bounded_bidirectional_halo16'}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=400);p.add_argument('--out',required=True);a=p.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); random.seed(0);np.random.seed(0);torch.manual_seed(0)
 rows=load_rows(sample_rate_hz=30); tr,tp=make_windows(rows[:6],(0,768,1536,2304),384,device); va,vp=make_windows(rows[6:],(0,768,1536,2304),384,device); fx=torch.from_numpy(make_fixture_metadata()).float().to(device); trf=fx[None].expand(len(tp),-1,-1); vaf=fx[None].expand(len(vp),-1,-1); tv=torch.ones(tr['audio'].shape[:2],dtype=torch.bool,device=device); vv=torch.ones(va['audio'].shape[:2],dtype=torch.bool,device=device); ts=torch.arange(384,device=device,dtype=torch.float32)[None].expand(len(tp),-1)/30; vs=torch.arange(384,device=device,dtype=torch.float32)[None].expand(len(vp),-1)/30
 base=AudioToLightFoundation(AudioToLightConfig(audio_dim=1116,local_rate_hz=30)).to(device); model=BoundedBidirectional(base).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); hist=[]; model.train()
 for step in range(a.steps):
  opt.zero_grad(set_to_none=True); o=model(tr['audio'],trf,timestamps=ts,valid=tv,duration_seconds=384/30); teacher=model.body_teacher(tr['body']);o['dance_teacher']=teacher.detach();o['dance_body']=model.body_target(teacher);to=model.teacher_forward(tr['body'],trf);tl=teacher_objective(to,tr,tv);dl=distillation_objective(o,to,tv);l=losses(o,tr,tv,True,spread_loss=True);l.update(tl);l.update(dl);l['cross_song_aim']=cross_song_motion_objective(o,tr,[r['id'] for r,s in tp],tv);l['total']=l['total']+.5*sum(tl.values())+.25*sum(dl.values())+2*l['cross_song_aim'];l['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
  if step==0 or step+1==a.steps:hist.append({'step':step+1,**{k:float(v.detach().cpu()) for k,v in l.items()}})
 model.eval()
 with torch.no_grad():
  o=model(va['audio'],vaf,timestamps=vs,valid=vv,duration_seconds=384/30);teacher=model.body_teacher(va['body']);o['dance_teacher']=teacher.detach();o['dance_body']=model.body_target(teacher);vl=losses(o,va,vv,True,spread_loss=True);m=metrics(o,va,vp,30);sh=va['audio'].clone();g=torch.Generator(device=device).manual_seed(123)
  for i in range(len(sh)):sh[i]=sh[i,torch.randperm(sh.shape[1],generator=g,device=device)]
  so=model(sh,vaf,timestamps=vs,valid=vv,duration_seconds=384/30);m['temporal_audio_shuffle_aim_delta']=float((o['aim']-so['aim']).abs().mean().cpu());m['temporal_audio_shuffle_rgb_delta']=float((o['rgb']-so['rgb']).abs().mean().cpu());m['temporal_audio_shuffle_brightness_delta']=float((o['brightness']-so['brightness']).abs().mean().cpu())
 root=Path(a.out);root.mkdir(parents=True,exist_ok=False);torch.save({'state_dict':model.state_dict(),'config':model.config_dict(),'seed':0},root/'model.pt');res={'protocol':'audio-light-bounded-bidirectional-context-20261015-v1','status':'diagnostic_only','seed':0,'steps':a.steps,'validation_losses':{k:float(v.detach().cpu()) for k,v in vl.items()},'metrics':m,'history':hist,'model_config':model.config_dict(),'future_past_halo_frames':16,'promotion':'forbidden'};(root/'RESULTS.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');print(json.dumps(res,indent=2))
if __name__=='__main__':main()

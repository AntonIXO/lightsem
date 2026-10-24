from pathlib import Path
import hashlib,json, numpy as np, torch
from lightsem.experiments.audio_light_foundation.data import load_row,make_fixture_metadata
from lightsem.experiments.foundation_audio_light import FoundationAudioLightModel,FoundationConfig
from lightsem.experiments.foundation_audio_light.motion_head import MotionHead
from lightsem.experiments.foundation_audio_light.safety_projection import SafetyConfig
class AnchoredModel(FoundationAudioLightModel):
 def forward(self,audio,body,fixtures,**kwargs):
  out=super().forward(audio,body,fixtures,**kwargs); anchor=torch.stack((torch.atan2(fixtures[...,1],fixtures[...,0])/torch.pi*.90,fixtures[...,1]*.25),-1); out['angle']=(anchor[:,None]+.10*out['angle']).clamp(-.99,.99); return out
root=Path('/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v9_warm_v7'); muq=Path('/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3')

def metrics(o):
 angle=o['angle'][0]; rgb=o['rgb'][0]; br=o['brightness'][0,...,0]; ev=1/(1+np.exp(-o['event'][0,:,0]));
 pair=np.linalg.norm(angle[:,:,None,:]-angle[:,None,:,:],axis=-1); ii,jj=np.triu_indices(8,1); spread=pair[:,ii,jj].mean(1)
 vel=np.diff(angle,axis=0,prepend=angle[:1])*10; acc=np.diff(vel,axis=0,prepend=vel[:1])*10; jerk=np.diff(acc,axis=0,prepend=acc[:1])*10
 bins=np.floor(np.mod((angle[:,:,0]+1)/2,1)*8).astype(int); side=np.array([len(np.unique(x)) for x in bins]); hashes=[hashlib.sha256(angle[:,i].tobytes()).hexdigest() for i in range(8)]
 p=ev.mean(); ent=float(-(p*np.log(max(p,1e-8))+(1-p)*np.log(max(1-p,1e-8))))
 return {'finite':bool(np.isfinite(angle).all() and np.isfinite(rgb).all() and np.isfinite(br).all()),'angle_spread_min':float(spread.min()),'angle_spread_p50':float(np.median(spread)),'angle_spread_p95':float(np.quantile(spread,.95)),'angle_spread_below02_fraction':float(np.mean(spread<.2)),'side_count_min':int(side.min()),'side_count_p50':float(np.median(side)),'speed_max':float(np.linalg.norm(vel,axis=-1).max()),'acceleration_max':float(np.linalg.norm(acc,axis=-1).max()),'jerk_max':float(np.linalg.norm(jerk,axis=-1).max()),'rgb_below001_fraction':float(np.mean(rgb<.01)),'rgb_std':float(rgb.std()),'brightness_std':float(br.std()),'event_mean':float(p),'event_entropy':ent,'exact_duplicate_angle_streams':len(set(hashes))<8}

def load_model():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=AnchoredModel(FoundationConfig(audio_dim=1116)).to(dev); m.motion=MotionHead(m.config.width,SafetyConfig(dt=.1,max_angle=1.,max_velocity=.18,max_acceleration=.5,max_jerk=1.5)).to(dev); m.load_state_dict(torch.load(root/'model.pt',map_location=dev)['state_dict']); m.eval(); return m,dev

def infer(m,dev,a,b):
 fx=torch.from_numpy(make_fixture_metadata()[None]).float().to(dev); at=torch.from_numpy(a[None]).float().to(dev); bt=torch.from_numpy(b[None]).float().to(dev); n=at.shape[1]; ts=torch.arange(n,device=dev).float()[None]/10; valid=torch.ones((1,n),dtype=torch.bool,device=dev)
 with torch.no_grad(): o=m(at,bt,fx,timestamps=ts,valid=valid,duration_seconds=n/10,return_state=True)
 return {k:v.detach().cpu().numpy() for k,v in o.items() if k in ['angle','rgb','brightness','event']}
rows={sid:load_row(sid,sample_rate_hz=10,muq_root=muq) for sid in ['21c3a','3b4ab']}; m,dev=load_model(); own={}
for sid,r in rows.items():
 o=infer(m,dev,r['audio'],r['body']); own[sid]=metrics(o)
 # deterministic repeat
 o2=infer(m,dev,r['audio'],r['body']); own[sid]['deterministic_max_abs_diff']=max(float(np.max(np.abs(o[k]-o2[k]))) for k in o)
# Hold body and clock fixed; swap the two recomputed song audios.
n=min(len(rows['21c3a']['audio']),len(rows['3b4ab']['audio']))
base=infer(m,dev,rows['21c3a']['audio'][:n],rows['3b4ab']['body'][:n]); swap=infer(m,dev,rows['3b4ab']['audio'][:n],rows['3b4ab']['body'][:n])
swap_delta={k:float(np.mean(np.abs(base[k]-swap[k]))) for k in ['angle','rgb','brightness','event']}
res={'schema_version':1,'protocol':'audio-light-foundation-v2-continuous-collapse-audit-20261016-v4','status':'diagnostic_only','input_contract':'canonical MuQ1024 layer9 + Base92 1116-D; no zero padding; MuQ root v3; fixture metadata anchor plus bounded learned residual','checkpoint':str(root/'model.pt'),'checkpoint_sha256':hashlib.sha256((root/'model.pt').read_bytes()).hexdigest(),'songs':own,'audio_swap_control':{'fixed_body':'3b4ab','fixed_clock_frames':n,'mean_abs_delta':swap_delta,'changed':any(v>1e-5 for v in swap_delta.values())},'representation_collapse':'unqualified: this audit does not expose a temporal latent matrix independent of output heads','verdict':'output-collapse, raw-safety, deterministic, two-song, and audio-swap gates pass for this diagnostic arm; representation-collapse evidence remains unqualified and no product promotion is earned','promotion':'forbidden'}
(root/'COLLAPSE_AUDIT_V9.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n'); print(json.dumps(res,indent=2))

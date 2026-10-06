"""Frozen inference diagnostics only: exact replay, finer integration, plan disconnection."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
from lightsem.experiments.native_dynamics.evaluation import setup,load_final,cpu_noise
from lightsem.experiments.native_dynamics.showcase import input_arrays
from lightsem.experiments.native_dynamics.dataset import CONTEXT_BINS
from lightsem.experiments.motion_study.geometry import joints_from_features
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.experiments.dynamic_dance_combo.campaign import save,arrays,sha

@torch.inference_mode()
def diagnostic_integrate(model,fine,coarse,valid,indices,noise,steps):
 conditioning,plan=model.condition(fine,coarse,valid,indices)
 base,biases=model.flow.prepare_conditioning(conditioning);state=noise.clone();halo=model.flow.halo
 for i in range(steps):
  time=state.new_full((len(state),),i/steps);parts=[]
  for start in range(0,state.shape[1],960):
   end=min(start+960,state.shape[1]);lo=max(0,start-halo);hi=min(state.shape[1],end+halo)
   velocity=model.flow.prepared_vector_field(base[:,lo:hi],biases[:,:,:,lo:hi],state[:,lo:hi],time)
   parts.append(velocity[:,start-lo:end-lo])
  state=state+torch.cat(parts,1)/steps
 return model.flow.codec.decode(state.float()*model.flow.std+model.flow.mean),plan

@torch.inference_mode()
def run(root,song):
 from scripts.motion.native_dynamics_guard import authenticate_guarded_job
 authenticate_guarded_job(__file__)
 setup();root=Path(root);out=root/'jitter_audit_v1';plan=json.loads((root/'PLAN.json').read_text());t=next(x for x in plan['tracks'] if x['id']==song)
 model,norms,identity=load_final(root,'joint',0);n=t['body_frames'];noise=cpu_noise(model.flow,max(x['body_frames'] for x in plan['tracks']),7001)[:,:n].contiguous().cuda()
 base,extra,coarse,binding=input_arrays(t,'correct');fine=torch.cat(((torch.tensor(base,device='cuda')-norms[0][0])/norms[0][1],(torch.tensor(extra,device='cuda')-norms[1][0])/norms[1][1]),-1)[None]
 c=torch.zeros(1,CONTEXT_BINS,1061,device='cuda');valid=torch.zeros(1,CONTEXT_BINS,device='cuda',dtype=torch.bool);c[0,:len(coarse)]=(torch.tensor(coarse,device='cuda')-norms[1][0])/norms[1][1];valid[0,:len(coarse)]=True;indices=torch.arange(n,device='cuda')[None]
 ref=DanceResult.load(root/'bodies/C4'/song/'body.npz');rows=[]
 for name,steps,disconnect in [('replay32',32,False),('euler64',64,False),('planner_disconnected',32,True)]:
  if disconnect:
   for layer in model.flow.native_adapters:layer.weight.zero_();layer.bias.zero_()
  native,p=diagnostic_integrate(model,fine,c,valid,indices,noise,steps);x=native[0].cpu().numpy();w=joints_from_features(native)[0].cpu().numpy()
  row={'case':name,'finite':bool(np.isfinite(w).all()),'steps':steps,'planner_disconnected':disconnect,'maximum_native_difference_from_saved':float(np.max(abs(x-ref.native))),'checkpoint':identity,'noise_sha256':hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()}
  if name=='replay32':
   if not np.array_equal(x,ref.native):raise ValueError('Exact replay failed; stop attribution probes')
  else:
   dest=out/'probes_v3'/song/(name+'.npz');arrays(dest,time=ref.time,world=w,native=x);row['artifact']=str(dest);row['sha256']=sha(dest)
  rows.append(row)
 save(out/'probes_v3'/song/'REPORT.json',{'cases':rows,'audio_binding':binding,'scope':'FinerEulerintegration and zeroed nativeplanner adapters are diagnostics onsamecheckpoint/noise/audio. No newmodel or demonstration improvement claim.'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--song',required=True);p.add_argument('--guarded',action='store_true');a=p.parse_args();run(a.root,a.song)

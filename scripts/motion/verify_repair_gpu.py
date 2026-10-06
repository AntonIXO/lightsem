"""Independent FP32 GPU parity check of the actual CPU-generated demo source."""
from pathlib import Path
import argparse,hashlib
import numpy as np,torch
from scripts.motion.native_dynamics_guard import authenticate_guarded_job
from lightsem.experiments.native_dynamics.evaluation import setup,load_final,cpu_noise
from lightsem.experiments.native_dynamics.showcase import input_arrays
from lightsem.experiments.native_dynamics.dataset import CONTEXT_BINS
from lightsem.experiments.motion_study.geometry import joints_from_features
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.experiments.dynamic_dance_combo.campaign import read,save,sha

@torch.inference_mode()
def verify(root,seed,song,out):
    setup();root=Path(root);plan=read(root/'PLAN.json');t=next(x for x in plan['tracks'] if x['id']==song);model,norms,identity=load_final(root,'joint',seed)
    n=t['body_frames'];noise=cpu_noise(model.flow,max(x['body_frames'] for x in plan['tracks']),7001)[:,:n].contiguous().cuda();base,extra,coarse,binding=input_arrays(t,'correct')
    fine=torch.cat(((torch.tensor(base,device='cuda')-norms[0][0])/norms[0][1],(torch.tensor(extra,device='cuda')-norms[1][0])/norms[1][1]),-1)[None]
    c=torch.zeros(1,CONTEXT_BINS,1061,device='cuda');valid=torch.zeros(1,CONTEXT_BINS,device='cuda',dtype=torch.bool);c[0,:len(coarse)]=(torch.tensor(coarse,device='cuda')-norms[1][0])/norms[1][1];valid[0,:len(coarse)]=True
    native,_=model.generate(fine,c,valid,torch.arange(n,device='cuda')[None],noise);w=joints_from_features(native)[0].cpu().numpy();x=native[0].cpu().numpy()
    path=root/'bodies'/('C4' if seed==0 else 'C4_seed1')/song/'body.npz';saved=DanceResult.load(path)
    error={'native_max_absolute':float(abs(x-saved.native).max()),'relative_world_max_m':float(abs((w-w[:,:1])-(saved.world-saved.world[:,:1])).max()),'root_y_max_m':float(abs(w[:,0,1]-saved.world[:,0,1]).max()),'world_max_m':float(abs(w-saved.world).max())}
    limits={'native_max_absolute':.001,'relative_world_max_m':.001,'root_y_max_m':.001,'world_max_m':.005};passed=all(error[k]<=v for k,v in limits.items())
    save(Path(out),{'passed':passed,'error':error,'limits':limits,'checkpoint':identity,'CPU_body_sha256':sha(path),'audio':binding,'seed':seed,'song':song,'scope':'Numerical parity only; no bodyreplacement or output smoothing.'})
    if not passed:raise ValueError('CPU/GPU generation parity failed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--song',required=True);p.add_argument('--out',required=True);p.add_argument('--guarded',action='store_true');a=p.parse_args();authenticate_guarded_job(__file__);verify(a.root,a.seed,a.song,a.out)

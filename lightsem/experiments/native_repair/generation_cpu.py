"""Actual full-song native generation on CPU while the sole GPU trains.

Same frozen feature paths/noise/32-step solver; no retiming or output filter.
"""
from pathlib import Path
import hashlib,numpy as np,torch
from ..native_dynamics.evaluation import setup,load_final,cpu_noise
from ..native_dynamics.showcase import input_arrays,CONDITIONS,map_compact_control
from ..native_dynamics.dataset import CONTEXT_BINS
from ..native_dynamics.data import read,write,save_npz,sha
from ..motion_study.geometry import joints_from_features
from ..dynamic_dance.contracts import DanceResult
from ..dynamic_dance_combo.campaign import map_one,body_path

@torch.inference_mode()
def generate(root,seed,song,*,controls=True,condition_subset=None):
    setup();torch.set_num_threads(4);root=Path(root);plan=read(root/'PLAN.json');track=next(t for t in plan['tracks'] if t['id']==song)
    model,norms,identity=load_final(root,'joint',seed,device='cpu');n=track['body_frames'];noise=cpu_noise(model.flow,max(t['body_frames'] for t in plan['tracks']),7001)[:,:n].contiguous()
    conditions=CONDITIONS+('plan_shift',) if seed==0 and controls else ('correct',)
    if condition_subset is not None:
        conditions=tuple(condition_subset)
        if not conditions or any(x not in CONDITIONS+('plan_shift',) for x in conditions):raise ValueError('Unknown intervention')
    folder=root/('generation_controls/joint' if condition_subset is not None else 'generation/joint')/f'seed_{seed}'/song;folder.mkdir(parents=True,exist_ok=False);paths=[]
    for condition in conditions:
        base,extra,coarse,binding=input_arrays(track,condition)
        fine=torch.cat(((torch.tensor(base)-norms[0][0])/norms[0][1],(torch.tensor(extra)-norms[1][0])/norms[1][1]),-1)[None]
        c=torch.zeros(1,CONTEXT_BINS,1061);valid=torch.zeros(1,CONTEXT_BINS,dtype=torch.bool);bins=len(coarse);c[0,:bins]=(torch.tensor(coarse)-norms[1][0])/norms[1][1];valid[0,:bins]=True;indices=torch.arange(n)[None]
        override=None
        if condition=='plan_shift':
            p=model.planner(c,valid);override=p['conditioning'].clone();override[:,:bins]=torch.roll(override[:,:bins],bins//2,1)
        native,p=model.generate(fine,c,valid,indices,noise,plan_override=override);x=native[0].numpy();world=joints_from_features(native)[0].numpy()
        if not np.isfinite(world).all() or not np.isfinite(x).all():raise ValueError('Nonfinite actual generated motion')
        method='C4'+('' if seed==0 else '_seed1')+('' if condition=='correct' else '__'+condition)
        meta={'arm':'joint','training_seed':seed,'draw_seed':7001,'method':method,'song':song,'condition':condition,'checkpoint':identity,'audio':binding,'noise_sha256':hashlib.sha256(noise.numpy().tobytes()).hexdigest(),'fps':30.,'motion_origin':'Actual repaired native flow generation; no output processing','newly_generated_motion':True,'source_body_used':False,'retiming':False,'motif_recomposition':False,'rate_floor':False,'product_baseline':False,'native_time':'Original30Hz; generativeflowtime is separate','planner_connected':True,'planner_override':condition=='plan_shift','precision':'FP32 deterministic CPU, official saved source for rendering; GPU parity audited separately','training_recipe':read(root/'TRAINING_PROTOCOL.json').get('recipe','raw3d_repair_v1')}
        dest=body_path(root,method,song);DanceResult(time=np.arange(n)/30,world=world,native=x,contacts=x[:,-4:],metadata=meta).save(dest);paths.append(dest)
        save_npz(folder/(condition+'_planner.npz'),time=(np.arange(bins)*15+7)/30,packed=p['packed'][0,:bins].numpy(),embedding=p['embedding'][0,:bins].numpy(),conditioning=(p['conditioning'] if override is None else override)[0,:bins].numpy())
        if condition=='correct':map_one(root,song,method)
        else:map_compact_control(root,track,method)
        print(seed,song,condition,'CPU generated and mapped',flush=True)
    write(folder/'COMPLETE.json',{'seed':seed,'song':song,'conditions':conditions,'checkpoint':identity,'inference_device':'cpu','no_posthoc_motion_processing':True,'body_artifacts':{str(p):sha(p) for p in paths}})

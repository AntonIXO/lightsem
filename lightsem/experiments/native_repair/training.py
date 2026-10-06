"""Same D7 start, samples, draws, architecture and30000updates; repairedloss only."""
from pathlib import Path
import torch
from ..native_dynamics import training as old
from ..native_dynamics.kinematics import project_native,reprojection_loss,dynamics_losses
from ..motion_study.music_physical import clean_estimate,physical_losses
from ..motion_study.music_revision_model import masked_native_losses
from .losses import raw_native_losses,WEIGHTS

@torch.no_grad()
def reference_rollout(teacher,condition,noise):
    audio=torch.cat((condition[...,:1153].detach(),condition.new_zeros((*condition.shape[:2],34))),-1)
    state=noise.detach().clone()
    for i in range(32):state=state+teacher.vector_field(audio,state,state.new_full((len(state),),i/32))/32
    return teacher.codec.decode(state.float()*teacher.std+teacher.mean)

def batch_losses(model,teacher,batch,noise,tau,runtime,field=None):
    domain=batch['domain'];w=old.LOSS_WEIGHTS
    with runtime.autocast():
        condition,plan=model.condition(batch['audio'],batch['coarse'],batch['coarse_valid'],batch['indices'])
        if domain=='fd':
            target=batch['native'];valid=batch['valid'];normalized=torch.where(valid[...,None],(target-model.flow.mean)/model.flow.std,0.)
            noise=torch.where(valid[...,None],noise,0.);state=(1-tau[:,None,None])*noise+tau[:,None,None]*normalized
            base,bias=model.flow.prepare_conditioning(condition);fn=model.flow.prepared_vector_field if field is None else field
            prediction=fn(base,bias,state,tau);endpoint=clean_estimate(state,prediction,tau);generated=endpoint*model.flow.std+model.flow.mean
            reference=target
        elif domain=='jd':
            if 'native' in batch:raise ValueError('No invented JustDance3Dlabels')
            generated,prior=model.rollout(condition,noise,field=field,teacher=teacher)
            with torch.autocast(device_type=condition.device.type,enabled=False):
                reference=reference_rollout(teacher,condition.float(),noise.float())
        else:raise ValueError(domain)
    plan_total,plan_parts=old.planner_loss(plan,batch['plan_target'],batch['plan_valid'],batch['pose_context'],batch['pose_context_valid'])
    losses={**plan_parts,'planner':plan_total}
    if domain=='fd':
        base_losses=masked_native_losses(prediction,normalized-noise,model.flow.delta_scale,batch['valid'])
        physical=physical_losses(endpoint,target,model.flow.mean,model.flow.std,batch['valid']);total=base_losses['total']
        for name,weight in old.asdict(model.flow.physical_weights).items():total=total+weight*physical[name]
        losses.update({f'flow_{k}':v for k,v in base_losses.items() if k!='total'},**{f'physical_{k}':v for k,v in physical.items()})
        gp,_=project_native(generated,batch['valid']);tp,_=project_native(target,batch['valid']);mask=batch['valid'][...,None].expand(-1,-1,13)
        dyn=dynamics_losses(gp,tp,mask);total=total+w['fd_dynamics_scale']*sum(w[k]*v for k,v in dyn.items());losses.update(dyn)
    else:
        gp,_=project_native(generated,batch['valid']);mask=batch['pose_mask']&batch['valid'][...,None]
        reprojection=reprojection_loss(gp,batch['pose'],mask);total=w['reprojection']*reprojection+w['base_prior']*prior
        dyn=dynamics_losses(gp,batch['pose'],mask);total=total+sum(w[k]*v for k,v in dyn.items());losses.update(reprojection=reprojection,base_prior=prior,**dyn)
    raw=raw_native_losses(generated,reference,batch['valid'])
    total=total+sum(WEIGHTS[k]*v for k,v in raw.items());losses.update(raw)
    losses['total']=total+w['planner']*plan_total
    return losses

def main():
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','train']);p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--guarded',action='store_true');a=p.parse_args()
    if not a.guarded:raise ValueError('Sequential GPU curator required')
    from scripts.motion.native_dynamics_guard import authenticate_guarded_job
    authenticate_guarded_job(Path(__file__).parents[3]/'scripts/motion/train_native_repair.py')
    old.batch_losses=batch_losses
    if a.stage=='train':result=old.train(a.root,'joint',a.seed)
    else:
        # Original performance-only preflight checks matched continued/joint;
        # only joint repaired is scientific. Pilotweights never retained.
        result=old.preflight(a.root)
    print(json.dumps({k:v for k,v in result.items() if k in ('steps','status','training_seconds','selected_steps')}))
if __name__=='__main__':main()

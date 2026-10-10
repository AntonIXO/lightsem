"""Matched native-dynamics fine-tuning; original time, real paired/weak labels."""
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,math,time,os,copy,importlib.util
import numpy as np
import torch
from .data import ROOT,read,write,sha
from .dataset import Records,to_device,OLD
from .model import Config,NativeDynamicsModel,planner_loss
from .kinematics import project_native,reprojection_loss,dynamics_losses
from ..dynamic_dance.training import load_learned
from ..motion_study.music_model import MusicConfig
from ..motion_study.music_physical import clean_estimate,physical_losses
from ..motion_study.music_revision_model import masked_native_losses
from ..training_runtime import TrainingRuntime,backward_checked
from ..stage11.stage11_common import seed_all,atomic_torch,atomic_json,require_space

ARMS=('continued','dynamics','joint')
LOSS_WEIGHTS={'reprojection':.10,'base_prior':.05,'profile':.40,'speed_distribution':.40,'acceleration_distribution':.10,'low_speed':.05,'duration':.10,'recurrence':.10,'planner':.20,'fd_dynamics_scale':.10}


def load_initial(arm,seed,device='cuda'):
    teacher,state,path=load_learned(OLD,'D7',seed,device='cpu')
    teacher.eval().requires_grad_(False)
    cfg=Config(arm=arm);base=MusicConfig(**{k:state['model_config'][k] for k in ('arm','width','dilations','steps')},frames=cfg.frames)
    model=NativeDynamicsModel(base,state['normalizers'],cfg);model.warmstart(state['model'])
    if asdict(model.flow.physical_weights)!=state['model_config']['physical_weights']:raise ValueError('Pretrained physical objectives changed')
    model=model.to(device);teacher=teacher.to(device)
    norms=[tuple(torch.tensor(state[name][k],device=device,dtype=torch.float32) for k in ['mean','std']) for name in ['audio_normalizers','extra_normalizers']]
    return model,teacher,state,norms,{'checkpoint':str(path),'sha256':sha(path)}


def source_dropout(batch,generator):
    keep=torch.rand((len(batch['audio']),4),device=batch['audio'].device,generator=generator)>=.2
    # Frozen model_features order:1024MuQ then4sources x4observations.
    audio=batch['audio'].clone();coarse=batch['coarse'].clone()
    for i in range(4):
        lo=1024+i*4;audio[...,92+lo:92+lo+4]*=keep[:,i,None,None];coarse[...,lo:lo+4]*=keep[:,i,None,None]
    batch={**batch,'audio':audio,'coarse':coarse};return batch


def batch_losses(model,teacher,batch,noise,tau,runtime,field=None):
    domain=batch['domain'];w=LOSS_WEIGHTS
    with runtime.autocast():
        condition,plan=model.condition(batch['audio'],batch['coarse'],batch['coarse_valid'],batch['indices'])
        if domain=='fd':
            if 'native' not in batch:raise ValueError('FineDance needs actual pairednative targets')
            target=batch['native'];valid=batch['valid'];normalized=torch.where(valid[...,None],(target-model.flow.mean)/model.flow.std,0.)
            noise=torch.where(valid[...,None],noise,0.)
            state=(1-tau[:,None,None])*noise+tau[:,None,None]*normalized
            base,bias=model.flow.prepare_conditioning(condition);fn=model.flow.prepared_vector_field if field is None else field
            prediction=fn(base,bias,state,tau)
            endpoint=clean_estimate(state,prediction,tau)
            generated=endpoint*model.flow.std+model.flow.mean
            prior=generated.sum()*0
        elif domain=='jd':
            if 'native' in batch:raise ValueError('JustDance has no native3Dtarget; fakeconditional flowmatching prohibited')
            generated,prior=model.rollout(condition,noise,field=field,teacher=teacher)
        else:raise ValueError('Unknown supervision domain')
    # All measurements/loss geometry stay FP32 even when NN layers use BF16.
    plan_total,plan_parts=planner_loss(plan,batch['plan_target'],batch['plan_valid'],batch['pose_context'],batch['pose_context_valid'])
    losses={**plan_parts,'planner':plan_total}
    if domain=='fd':
        base_losses=masked_native_losses(prediction,normalized-noise,model.flow.delta_scale,batch['valid'])
        physical=physical_losses(endpoint,target,model.flow.mean,model.flow.std,batch['valid'])
        total=base_losses['total']
        for name,weight in asdict(model.flow.physical_weights).items():total=total+weight*physical[name]
        losses.update({f'flow_{k}':v for k,v in base_losses.items() if k!='total'},**{f'physical_{k}':v for k,v in physical.items()})
        if model.config.arm!='continued':
            gp,_=project_native(generated,batch['valid']);tp,_=project_native(target,batch['valid']);mask=batch['valid'][...,None].expand(-1,-1,13)
            dynamics=dynamics_losses(gp,tp,mask)
            total=total+w['fd_dynamics_scale']*sum(w[k]*v for k,v in dynamics.items());losses.update(dynamics)
    else:
        gp,_=project_native(generated,batch['valid']);tp=batch['pose'];mask=batch['pose_mask']&batch['valid'][...,None]
        reprojection=reprojection_loss(gp,tp,mask);total=w['reprojection']*reprojection+w['base_prior']*prior
        losses.update(reprojection=reprojection,base_prior=prior)
        if model.config.arm!='continued':
            dynamics=dynamics_losses(gp,tp,mask);total=total+sum(w[k]*v for k,v in dynamics.items());losses.update(dynamics)
    losses['total']=total+w['planner']*plan_total
    return losses


def make_optimizer(model):
    planner=[p for n,p in model.named_parameters() if p.requires_grad and (n.startswith('planner.') or n.startswith('flow.native_'))]
    backbone=[p for n,p in model.named_parameters() if p.requires_grad and not (n.startswith('planner.') or n.startswith('flow.native_'))]
    return torch.optim.AdamW([{'params':backbone,'lr':2e-5},{'params':planner,'lr':1e-4}],weight_decay=.01,fused=True)


def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _validated_repair_weights(weights, expected):
    if not isinstance(weights,dict) or set(weights)!=set(expected) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError('Invalid raw-motion repair weights')
    return {k:float(weights[k]) for k in sorted(weights)}


def training_loss_for_protocol(protocol):
    """Select the explicit raw-3D repair recipe without changing old runs."""
    weights=protocol.get('repair_weights')
    if weights is None:return batch_losses
    from ..native_repair import training as repair
    from ..native_repair.losses import WEIGHTS
    validated=_validated_repair_weights(weights,WEIGHTS)
    WEIGHTS.update(validated)
    return repair.batch_losses


def loss_spec_for_protocol(protocol):
    """Return the effective loss recipe and the source files that define it.

    The repair entry point used to replace ``batch_losses`` as a process-global
    side effect.  That made the same protocol select different objectives
    depending on which CLI imported it first.  Keep the recipe in the protocol
    and bind its implementation hashes into every run identity instead.
    """
    weights=protocol.get('repair_weights')
    source_paths=[Path(__file__),Path(__file__).with_name('kinematics.py'),
                  Path(__file__).with_name('model.py'),
                  Path(__file__).parents[1]/'motion_study/music_physical.py',
                  Path(__file__).parents[1]/'motion_study/music_revision_model.py']
    spec={'name':'native_dynamics','base_weights':dict(LOSS_WEIGHTS),
          'repair_weights':None,
          'source_hashes':{str(p.resolve()):sha(p) for p in source_paths}}
    if weights is None:return spec
    from ..native_repair import losses as repair_losses
    from ..native_repair import training as repair_training
    validated=_validated_repair_weights(weights,repair_losses.WEIGHTS)
    repair_paths=[Path(repair_training.__file__).resolve(),Path(repair_losses.__file__).resolve()]
    spec.update(name='native_repair',repair_weights=validated,
                source_hashes={**spec['source_hashes'],**{str(p):sha(p) for p in repair_paths}})
    return spec


def preflight(root=ROOT):
    root=Path(root);require_space(root,gb=20);protocol=read(root/'TRAINING_PROTOCOL.json')
    loss_function=training_loss_for_protocol(protocol)
    scope='all' if protocol.get('no_validation_split') else 'train'; records=Records(root,scope);rows=[]
    # Performance only. No DEV/showcase data or quality/acceptance scores.
    for fd_batch,jd_batch in [(8,4),(16,8)]:
        for arm in ['continued','joint']:
            seed_all(0);torch.set_num_threads(2);runtime=TrainingRuntime(precision='bf16',optimizer='adamw_fused',compile='default',prefetch=0);runtime.apply();torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
            model,teacher,state,norms,initial=load_initial(arm,0);optimizer=make_optimizer(model);field=torch.compile(model.flow.prepared_vector_field,fullgraph=True,dynamic=False)
            rngs={k:np.random.default_rng(10000+off) for k,off in [('fd',0),('jd',1)]};draw=torch.Generator(device='cuda').manual_seed(20000);drop=torch.Generator(device='cuda').manual_seed(30000)
            torch.cuda.reset_peak_memory_stats();durations={'fd':[],'jd':[]};warm=time.monotonic()
            for step in range(18):
                torch.cuda.synchronize();start=time.monotonic()
                domain='jd' if step%3==2 else 'fd';batch=records.sample(domain,jd_batch if domain=='jd' else fd_batch,rngs[domain]);batch=to_device(batch,*norms);batch=source_dropout(batch,drop)
                noise=model.flow.noise(len(batch['audio']),480,generator=draw);tau=torch.rand(len(noise),device='cuda',generator=draw)
                optimizer.zero_grad(set_to_none=True)
                losses=loss_function(model,teacher,batch,noise,tau,runtime,field);backward_checked(losses,model);optimizer.step();torch.cuda.synchronize()
                if step>=9:durations[domain].append(time.monotonic()-start)
            row={'arm':arm,'fd_batch':fd_batch,'jd_batch':jd_batch,'seconds_per_update':(2*np.mean(durations['fd'])+np.mean(durations['jd']))/3,'fd_seconds':float(np.mean(durations['fd'])),'jd_seconds':float(np.mean(durations['jd'])),'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'warm_and_measure_seconds':time.monotonic()-warm,'finite_steps':18,'pilot_weights_retained':False}
            rows.append(row);print('PERFORMANCE',json.dumps(row),flush=True)
            del model,teacher,optimizer,field,losses,batch,noise;torch.compiler.reset();torch.cuda.empty_cache()
    eligible=[]
    for fd_batch,jd_batch in [(8,4),(16,8)]:
        found=[r for r in rows if r['fd_batch']==fd_batch]
        if max(r['peak_gpu_gib'] for r in found)<10.5:
            seconds=max(r['seconds_per_update'] for r in found);throughput=(2*fd_batch+jd_batch)*480/3/seconds
            eligible.append((throughput,fd_batch,jd_batch,seconds))
    if not eligible:raise RuntimeError('No batch meets fixedVRAMpreflight; no training')
    _,fd_batch,jd_batch,seconds=max(eligible)
    # Common exposure is fixed strictly from throughput, before any scientific outcomes.
    steps=min(30000,int((9.0*3600)/(6*seconds*1.25)//3000)*3000)
    if steps<3000:raise RuntimeError('Native32-step training infeasible within budget')
    report={'measurements':rows,'selected_fd_batch':fd_batch,'selected_jd_batch':jd_batch,'selected_steps':steps,'estimated_training_seconds':6*steps*seconds*1.25,'selection':'Largest validframe throughput below10.5GiB; equal6-run exposure, target30000steps, reduce only for10hbudget with25%overhead. No modelqualityseen.','data_manifest_sha256':sha(root/'data/MANIFEST.json'),'supervision_sha256':sha(root/'data/SUPERVISION.json'),'features_sha256':sha(root/'features/COMPLETE.json'),'source_sha256':sha(__file__)}
    write(root/'learning/PREFLIGHT.json',report);return report


def train(root,arm,seed,run_dir=None):
    root=Path(root);protocol=read(root/'TRAINING_PROTOCOL.json');config=protocol['training'];run=Path(run_dir) if run_dir is not None else root/'learning/runs'/arm/f'seed_{seed}'
    if arm not in protocol.get('arms',ARMS) or seed not in protocol.get('seeds',(0,1)):raise ValueError('Unregistered arm/seed')
    loss_function=training_loss_for_protocol(protocol)
    loss_spec=loss_spec_for_protocol(protocol)
    if run.exists():raise FileExistsError('Run exists; preserve and review, no implicitresume')
    require_space(root,gb=20);run.mkdir(parents=True);seed_all(seed);torch.set_num_threads(2)
    runtime=TrainingRuntime(precision='bf16',optimizer='adamw_fused',compile='default',prefetch=0);runtime.apply();torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    scope='all' if protocol.get('no_validation_split') else 'train'
    model,teacher,state,norms,initial=load_initial(arm,seed);records=Records(root,scope);optimizer=make_optimizer(model)
    model.train();field=torch.compile(model.flow.prepared_vector_field,fullgraph=True,dynamic=False)
    rngs={'fd':np.random.default_rng(10000+seed),'jd':np.random.default_rng(11000+seed)}
    draw=torch.Generator(device='cuda').manual_seed(20000+seed);drop=torch.Generator(device='cuda').manual_seed(30000+seed)
    total=config['steps'];schedule=lambda step:min(1.,(step+1)/500)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-500)/max(1,total-500))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule);identity={'protocol_sha256':sha(root/'TRAINING_PROTOCOL.json'),'arm':arm,'seed':seed,'initial':initial,'initial_model_array_sha256':state_hash(model),'inputs':records.inputs,'implementation_sha256':sha(__file__),'loss_spec':loss_spec}
    write(run/'IDENTITY.json',identity)
    initial_path=run/'initial_weights.pt';atomic_torch(initial_path,{'model':model.state_dict(),'config':asdict(model.config),'identity':identity,'loss_spec':loss_spec})
    exposure=hashlib.sha256();seen={'fd':0,'jd':0};valid_frames={'fd':0,'jd':0};started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def checkpoint(step,final=False):
        payload={'model':model.state_dict(),'config':asdict(model.config),'base_config':asdict(model.flow.config),'identity':identity,'loss_spec':loss_spec,'step':step,'normalizers':state['normalizers'],'audio_normalizers':state['audio_normalizers'],'extra_normalizers':state['extra_normalizers'],'feature_names':state['feature_names'],'exposure_sha256':exposure.hexdigest(),'counts':seen.copy(),'valid_frames':valid_frames.copy()}
        if final:atomic_torch(run/'final.pt',payload)
        else:
            payload.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rngs={k:v.bit_generator.state for k,v in rngs.items()},draw_rng=draw.get_state(),drop_rng=drop.get_state())
            atomic_torch(run/'latest.pt',payload)
    try:
        with (run/'training.jsonl').open('x') as log:
            for step in range(total):
                domain='jd' if step%3==2 else 'fd';size=config['jd_batch'] if domain=='jd' else config['fd_batch']
                batch=records.sample(domain,size,rngs[domain]);exposure.update(json.dumps([domain,batch['examples']],separators=(',',':')).encode());seen[domain]+=size;valid_frames[domain]+=int(batch['valid'].sum())
                batch=to_device(batch,*norms);batch=source_dropout(batch,drop);noise=model.flow.noise(size,config['frames'],generator=draw);tau=torch.rand(size,device='cuda',generator=draw)
                optimizer.zero_grad(set_to_none=True);losses=loss_function(model,teacher,batch,noise,tau,runtime,field);grad=backward_checked(losses,model);optimizer.step();scheduler.step()
                if step==0 or (step+1)%100==0 or step+1==total:
                    row={'step':step+1,'arm':arm,'seed':seed,'domain':domain,'elapsed_s':time.monotonic()-started,'grad':float(grad),'exposure_sha256':exposure.hexdigest(),**{k:float(v.detach()) for k,v in losses.items()}}
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();atomic_json(run/'STATUS.json',row);print(json.dumps(row),flush=True)
                if (step+1)%1000==0:require_space(root,gb=20);checkpoint(step+1)
            checkpoint(total,True)
        report={'status':'complete','steps':total,'training_seconds':time.monotonic()-started,'identity':identity,'loss_spec':loss_spec,'model_config':model.config_dict(),'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'exposure_sha256':exposure.hexdigest(),'counts':seen,'valid_frames':valid_frames,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'no_posthoc_motion_processing':True,'JD3Dtargets_used':False,'artifacts':{str(p):sha(p) for p in [run/'final.pt',initial_path,run/'training.jsonl']}}
        write(run/'REPORT.json',report);return report
    except BaseException as e:
        write(run/f'FAILURE_{time.time_ns()}.json',{'type':type(e).__name__,'message':str(e),'last_saved_checkpoint':str(run/'latest.pt'),'elapsed_s':time.monotonic()-started,'artifacts_preserved':True});raise


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','train']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,default=0);p.add_argument('--run-dir',type=Path);p.add_argument('--guarded',action='store_true');p.add_argument('--guard-script',type=Path);a=p.parse_args()
    if not a.guarded:raise ValueError('GPU guard required')
    guard_path=a.guard_script or Path(__file__).parents[3]/'scripts/motion/native_dynamics_guard.py'
    spec=importlib.util.spec_from_file_location('_native_training_guard',guard_path); guard=importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
    guard.authenticate_guarded_job(Path(__file__).parents[3]/'scripts/motion/train_native_dynamics.py')
    result=preflight(a.root) if a.stage=='preflight' else train(a.root,a.arm,a.seed,a.run_dir)
    print(json.dumps({k:v for k,v in result.items() if k in ['status','steps','training_seconds','selected_steps','estimated_training_seconds']}))
if __name__=='__main__':main()

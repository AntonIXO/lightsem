#!/usr/bin/env python3
"""Fixed H3 student fit: TRAIN only, paired initialization, final checkpoint only."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,resource,shutil,subprocess,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from lightsem.experiments.stage11.stage11_common import gpu_lock
from lightsem.experiments.visual_next.features import ARMS
from lightsem.model.light_layers.model import LightLayersConfig,build_light_layers,count_trainable_parameters
from lightsem.model.light_layers.training import (SourceStore,TrainingData,BatchSampler,compute_loss,
    seed_all,rng_digest,load_checkpoint,json_hash)
from lightsem.model.light_layers.visual import sha256,write_json


def state_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        a=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


def main_target_digest(data):
    h=hashlib.sha256()
    for rid,t in sorted(data.targets.items()):
        h.update(rid.encode())
        for key in ('color','color_mask','accent10','accent_mask10','palette','palette_mask'):
            a=np.ascontiguousarray(t[key]);h.update(key.encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


def software_versions():
    # torch.__version__ can be a TorchVersion str subclass, which is not an
    # allowlisted metadata type for weights_only checkpoint loading.
    return {'torch':str(torch.__version__),'numpy':str(np.__version__)}


def prepared_data(freeze,arm):
    manifest=json.loads(Path(freeze['training_manifest']['path']).read_text())
    store=SourceStore(manifest,max_ram_bytes=512*1024**2)
    # Explicitly never open DEV features during student fitting.
    records=[store.visual(row['recording_id']) for row in manifest['records'] if row['partition']=='train']
    if len(records)!=91 or len({r.canonical_id for r in records})!=91:raise RuntimeError('Original91TRAIN families required')
    transforms=copy.deepcopy(freeze['transforms'])
    # Main labels/vocabulary/audio scalers stay frozen. Only visual auxiliary changes.
    transforms['aux']['rich']={'mean':[0.]*96,'std':[1.]*96,'counts':[0]*96}
    for record in records:
        path=Path(freeze['projected_root'])/(record.canonical_id+'.npz')
        with np.load(path,allow_pickle=False) as z:
            features=z[arm].astype(np.float32);valid=z['valid'].copy()
        if features.shape!=(640,96) or valid.dtype!=bool or valid.shape!=(640,):raise RuntimeError('Projected target schema')
        if not np.isfinite(features).all() or np.any(features[~valid]!=0):raise RuntimeError('Invalid target placeholders')
        record.rich=features;record.rich_valid=np.broadcast_to(valid[:,None],features.shape).copy()
        record.validate()
    data=TrainingData(store,records,transforms,'I_VR_P0')
    return store,records,transforms,data


def checkpoint(path,model,metadata):
    if path.exists():raise RuntimeError('Refuse checkpoint replacement')
    temp=path.with_suffix('.partial')
    with temp.open('xb') as f:
        torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    'metadata':metadata,'resume_supported':False},f)
    temp.rename(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True,type=Path)
    ap.add_argument('--arm',required=True,choices=ARMS);ap.add_argument('--seed',required=True,type=int,choices=(0,1,2))
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    torch.set_num_threads(2)
    if [args.arm,args.seed] not in freeze['matrix']:raise RuntimeError('Unregistered matrix member')
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    qualification_ref=freeze['representation_qualification']
    if freeze['files'].get(qualification_ref['path'])!=qualification_ref['sha256']:
        raise RuntimeError('Representation qualification is not bound to the fit')
    qualification=json.loads(Path(qualification_ref['path']).read_text())
    if qualification['status']!='controlled_retention_probes_complete':raise RuntimeError('Incomplete representation qualification')
    root=Path(freeze['output']);root.mkdir(exist_ok=True)
    out=root/f'{args.arm}_seed{args.seed}';out.mkdir(exist_ok=False)
    store,records,transforms,data=prepared_data(freeze,args.arm)
    main_digest=main_target_digest(data)
    if main_digest!=freeze['main_targets_sha256']:raise RuntimeError('Paired main targets differ')
    reference=LightLayersConfig(visual_aux_dim=96,pose_aux_dim=0)
    seed_all(args.seed,cuda=False);model=build_light_layers(reference,architecture='independent')
    initial_digest=state_digest(model)
    if initial_digest!=freeze['initialization_sha256'][str(args.seed)]:raise RuntimeError('Paired initialization drift')
    if count_trainable_parameters(model)!=1284410:raise RuntimeError('Student architecture drift')
    sampler=BatchSampler(records,args.seed);dropout=torch.Generator(device='cpu').manual_seed(args.seed+10000)
    report={'status':'running','arm':args.arm,'seed':args.seed,'freeze_sha256':sha256(args.freeze),
        'initial_state_sha256':initial_digest,'main_targets_sha256':main_digest,'train_families':91,
        'parameters':1284410,'reference_model_config':asdict(reference),'model_config':asdict(model.cfg),
        'architecture':'independent','precision':'bf16','selection':'final6000_only_no_DEV_selection',
        'software':software_versions(),'rng_initial':rng_digest(sampler,dropout),
        'representation_qualification':qualification_ref,
        'original_representation_fit_gate_pass':qualification['original_fit_gate_pass'],
        'execution_scope':freeze['execution_scope'],
        'test_access':False,'DEV_access_during_fit':False,'scope':'Exploratory visual-supervision comparison; automatic observer main targets. Failed representation qualification remains attached.'}
    write_json(out/'RUN.json',report)
    started=time.monotonic();sample_hash=hashlib.sha256();final_losses=None;completed_steps=0
    try:
        with gpu_lock():
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
            others=[x.strip() for x in active.splitlines() if x.strip() and x.strip()!=str(os.getpid())]
            if others:raise RuntimeError('GPU is in use: '+','.join(others))
            if not torch.cuda.is_bf16_supported():raise RuntimeError('No CUDA BF16 support')
            torch.cuda.manual_seed_all(args.seed);model.cuda().train();torch.cuda.reset_peak_memory_stats()
            optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
            with (out/'updates.jsonl').open('x') as log:
                for step in range(1,6001):
                    if time.monotonic()-started>freeze['per_fit_max_seconds']:raise RuntimeError('Per-fit runtime cap')
                    samples=sampler.sample(4);sample_hash.update(json.dumps(samples,separators=(',',':')).encode())
                    batch=data.batch(samples,'cuda');optimizer.zero_grad(set_to_none=True)
                    lr=3e-4*min(1.,step/300)
                    for group in optimizer.param_groups:group['lr']=lr
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        losses,outputs=compute_loss(model,batch,transforms,state_dropout=.5,dropout_generator=dropout)
                    if not all(torch.isfinite(x).item() for x in losses.values()):raise RuntimeError('Nonfinite loss')
                    losses['total'].backward()
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    completed_steps=step
                    final_losses={k:float(v.detach()) for k,v in losses.items()}
                    entry={'step':step,'samples':samples,'loss':final_losses,'lr':lr,'gradient_norm':float(grad),
                           'elapsed_s':time.monotonic()-started}
                    log.write(json.dumps(entry,allow_nan=False)+'\n')
                    if step%300==0:
                        log.flush();print('FIT',args.arm,args.seed,step,round(entry['elapsed_s'],2),flush=True)
                        if torch.cuda.max_memory_reserved()>10*2**30 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:
                            raise RuntimeError('Memory cap')
                        if any(shutil.disk_usage(p).free<2**30 for p in (out,'/mnt')):raise RuntimeError('Disk floor')
                        cache=Path(freeze['study_root'])
                        used=sum(p.stat().st_size for d in cache.iterdir() if d.is_dir() and d.name!='papers'
                                 for p in d.rglob('*') if p.is_file())
                        if used>freeze['max_study_bytes']:raise RuntimeError('Study storage allowance exceeded')
                    del batch,losses,outputs
            if sample_hash.hexdigest()!=freeze['sample_sequence_sha256'][str(args.seed)]:raise RuntimeError('Paired data draws drift')
            metadata={**report,'step':6000,'parameter_match':asdict(model.parameter_match)}
            checkpoint(out/'final.pt',model,metadata)
            peak=torch.cuda.max_memory_reserved()
        for p,d in freeze['files'].items():assert sha256(p)==d,p
        report.update(status='complete',steps=6000,elapsed_s=time.monotonic()-started,
            final_weights_sha256=sha256(out/'final.pt'),final_state_sha256=state_digest(model),
            samples_sha256=sample_hash.hexdigest(),updates_sha256=sha256(out/'updates.jsonl'),
            peak_reserved_bytes=peak,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            rng_final=rng_digest(sampler,dropout,cuda=True),last_training_losses=final_losses)
        write_json(out/'REPORT.json',report)
    except BaseException as error:
        write_json(out/'INTERRUPTED.json',{'status':'incomplete','error':type(error).__name__+': '+str(error),
            'elapsed_s':time.monotonic()-started,'steps_completed':completed_steps,
            'final_evaluation_observed':False,'final_checkpoint_exists':(out/'final.pt').exists()})
        raise


if __name__=='__main__':main()

"""Engineering screen and four-cell noninferiority pilot, separate from frozen Night results.

Run with ``bash run_torch.sh -m lightsem.experiments.iteration_speed STAGE``.
No TEST access, feature recomputation, outcome-based retries, or legacy writes.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .night.night_common import ROOT as REFERENCE, BASE, base_context
from .night.night_model import NightModel
from .night.night_training import NightResident, run_cell, study_context
from .training_runtime import TrainingRuntime, BatchStream, runtime_loss, backward_checked
from .stage11.stage11_common import (authenticated_json, read_authenticated, digest, json_digest,
                            artifact_manifest, verify_artifacts, seed_all, freeze_sources,
                            verify_sources, gpu_lock, require_space)

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910')
ARMS = ('base_once', 'muq_film')
SEEDS = (0, 1)


def engineering_inputs(root):
    """A clean screen has no compiler-repair archive; don't require a fake one."""
    root=Path(root)
    paths=[root/'test_report.json',root/'engineering.json']
    archive=root/'engineering_before_compiler_repair.json'
    if archive.is_file():paths.append(archive)
    return paths


def setup_cache(root):
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(root/'compiler_cache'))
    os.environ.setdefault('TRITON_CACHE_DIR', str(root/'triton_cache'))
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '4')


def benchmark_case(data, arm, runtime, batch, *, legacy=False, warmup=20, repeats=3, steps=50):
    if runtime.compile!='off':
        # Shapes/architectures must not exhaust another case's Dynamo budget.
        torch.compiler.reset()
        from torch._dynamo.utils import counters
        counters.clear()
    family, architecture = arm.split('_');data.set_family(family)
    seed_all(0);runtime.apply()
    model=NightModel(family,architecture,data.aux_dim).cuda().train()
    optimizer=runtime.make_optimizer(model,3e-4);fn=runtime.loss_function(model)
    rng=np.random.default_rng(10000);flow=torch.Generator(device='cuda').manual_seed(30000)
    _,norms=base_context();pos=torch.tensor(norms['pos_weight'],device='cuda',dtype=torch.float32)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();windows=[];cuda_windows=[]
    try:
        with BatchStream(data,batch,640,rng,runtime.prefetch) as batches:
            def update():
                if legacy:torch.cuda.synchronize()
                x,y,b=batches.next();optimizer.zero_grad(set_to_none=True)
                if legacy:
                    losses=model.losses(x,y,b,pos,generator=flow)
                    if not all(torch.isfinite(v) for v in losses.values()):raise FloatingPointError('Nonfinite legacy loss')
                    losses['total'].backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(grad):raise FloatingPointError('Nonfinite legacy gradient')
                else:
                    losses=runtime_loss(model,fn,runtime,x,y,b,pos,flow)
                    backward_checked(losses,model)
                optimizer.step()
                if legacy:torch.cuda.synchronize()
            for _ in range(warmup):update()
            torch.cuda.synchronize();startup=time.monotonic()-start
            for _ in range(repeats):
                torch.cuda.synchronize();begin=time.monotonic()
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                for _ in range(steps):update()
                b.record();torch.cuda.synchronize()
                windows.append((time.monotonic()-begin)/steps)
                cuda_windows.append(a.elapsed_time(b)/1000/steps)
        seconds=float(np.median(windows))
        return {'arm':arm,'runtime':runtime.as_dict(),'batch':batch,'legacy':legacy,
                'step_seconds':seconds,'window_seconds_per_step':windows,'cuda_interval_seconds_per_step':cuda_windows,
                'crops_per_second':batch/seconds,'startup_seconds':startup,
                'estimated_run_seconds':startup+486400/batch*seconds,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'free_device_gib':torch.cuda.mem_get_info()[0]/2**30,
                'compiler_unique_graphs':dict(counters['stats']).get('unique_graphs',0) if runtime.compile!='off' else 0,
                'compiler_graph_breaks':dict(counters['graph_break']) if runtime.compile!='off' else {},
                'updates':warmup+repeats*steps,'scientific':False,'status':'completed'}
    finally:
        del model,optimizer,fn;gc.collect();torch.cuda.empty_cache()


def benchmark(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);setup_cache(root);require_space(root)
    path=root/'engineering.json'
    if path.exists():raise FileExistsError('Engineering results already exist')
    m,n=base_context();data=NightResident([e for e in m['entries'] if e['partition']=='train'],n,640,REFERENCE)
    rows=[]
    def measure(label,runtime,batch=32,legacy=False):
        matches=[]
        for arm in ARMS:
            result=benchmark_case(data,arm,runtime,batch,legacy=legacy);result['label']=label
            rows.append(result);matches.append(result)
            authenticated_json(root/'engineering_progress.json',{'rows':rows})
            print(json.dumps(result),flush=True)
        return matches
    def cost(pair):return sum(r['estimated_run_seconds'] for r in pair)
    baseline=measure('legacy_fp32',TrainingRuntime.legacy(),legacy=True)
    measure('tf32',TrainingRuntime.legacy('tf32'))
    runtime=TrainingRuntime.legacy('bf16');winning=measure('bf16',runtime)
    decisions=[]
    for label,field,value in [('fused','optimizer','adamw_fused'),('prefetch','prefetch',2),('compile','compile','default')]:
        candidate=TrainingRuntime(**{**runtime.as_dict(),field:value})
        try:
            result=measure(label,candidate)
            gain=cost(winning)/cost(result)
            accepted=gain>=1.05
            if accepted:runtime=candidate;winning=result
            decisions.append({'label':label,'gain_including_startup':gain,'accepted':accepted})
        except (RuntimeError,torch.cuda.OutOfMemoryError) as error:
            # Compiler/backend/OOM failures are retained engineering exclusions.
            decisions.append({'label':label,'accepted':False,'technical_error':str(error)})
            authenticated_json(root/(label+'_failure.json'),decisions[-1])
            gc.collect();torch.cuda.empty_cache()
            if label!='compile':raise
    batch=32
    for size in (64,128):
        try:
            result=measure('batch_'+str(size),runtime,size)
            throughput_gain=(sum(r['step_seconds']/batch for r in winning)/sum(r['step_seconds']/size for r in result))
            headroom=all(r['free_device_gib']>=1 for r in result)
            accepted=throughput_gain>=1.05 and headroom
            decisions.append({'label':'batch_'+str(size),'throughput_gain':throughput_gain,'headroom':headroom,'accepted':accepted})
            if accepted:batch=size;winning=result
        except torch.cuda.OutOfMemoryError as error:
            decisions.append({'label':'batch_'+str(size),'accepted':False,'technical_error':str(error)})
            gc.collect();torch.cuda.empty_cache()
    report={'rows':rows,'decisions':decisions,'selected_runtime':runtime.as_dict(),'selected_batch':batch,
            'reference':str(REFERENCE),'quality_results_accessed':False,
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'selection':'>=5% improvement including startup per added runtime option; batch >=5% throughput and >=1GiB free',
            'source_files':{str(Path(__file__)):digest(__file__)}}
    authenticated_json(path,report)
    print(json.dumps({'selected_runtime':runtime.as_dict(),'selected_batch':batch}),flush=True)


def repair_compiler_screen(root):
    """Retain the initial technical incident; rerun only contaminated B128 cases."""
    path=root/'engineering.json';original=read_authenticated(path)
    archive=root/'engineering_before_compiler_repair.json'
    if archive.exists() or (root/'protocol.json').exists():raise RuntimeError('Repair already recorded or study frozen')
    authenticated_json(archive,original)
    m,n=base_context();data=NightResident([e for e in m['entries'] if e['partition']=='train'],n,640,REFERENCE)
    runtime=TrainingRuntime(**original['selected_runtime']);setup_cache(root)
    updated=[]
    for row in original['rows']:
        if row['label']=='batch_128':
            row={**row,'admissible':False,'exclusion_reason':'Dynamo recompile limit exhausted by preceding cases; eager fallback'}
        updated.append(row)
    repaired=[]
    for arm in ARMS:
        row=benchmark_case(data,arm,runtime,128);row['label']='batch_128_repaired';repaired.append(row);updated.append(row)
        if row['compiler_unique_graphs']==0 or row['compiler_graph_breaks']:raise RuntimeError('Compiled screen did not capture clean graphs')
        print(json.dumps(row),flush=True)
    previous=[r for r in original['rows'] if r['label']=='compile']
    gain=sum(r['step_seconds']/32 for r in previous)/sum(r['step_seconds']/128 for r in repaired)
    accepted=gain>=1.05 and all(r['free_device_gib']>=1 for r in repaired)
    report={**original,'rows':updated,'selected_batch':128 if accepted else 32,
            'compiler_repair':{'reason':'Initial B128 measurements ran eagerly after Dynamo guard-limit exhaustion',
                'preserved_report':str(archive),'preserved_sha256':digest(archive),'throughput_gain':gain,'accepted':accepted},
            'source_files':{str(Path(__file__)):digest(__file__)}}
    authenticated_json(path,report)


def freeze(root):
    root=Path(root);engineering=read_authenticated(root/'engineering.json')
    if (root/'protocol.json').exists():raise FileExistsError('Pilot already frozen')
    if not read_authenticated(root/'test_report.json')['passed']:raise RuntimeError('Required tests have not passed')
    reference=read_authenticated(REFERENCE/'protocol.json');m,n=base_context()
    verify_sources(REFERENCE)
    # Keep the historical 15GiB free-space floor plus a bounded pilot budget.
    free=shutil.disk_usage(root).free/2**30
    if free<16.15:raise RuntimeError(f'Pilot needs 1.15GiB artifacts plus 15GiB reserve; only {free:.2f}GiB free')
    pins={}
    for path in [REFERENCE/'protocol.json',REFERENCE/'frozen_sources.json',BASE/'data_manifest.json',BASE/'normalizers.json',
                 BASE/'evaluation/input_manifest.json',REFERENCE/'evaluation/input_manifest.json',REFERENCE/'expression_config.json']+engineering_inputs(root):
        read_authenticated(path);pins[str(path)]=digest(path)
    for e in m['entries']:
        if e['partition']=='train':
            if digest(e['path'])!=e['sha256']:raise RuntimeError('Training tensor drift')
            pins[e['path']]=e['sha256']
    for family in ('base','muq'):
        if family=='base':continue
        path=REFERENCE/'features'/family/'training_array.json';info=read_authenticated(path)
        if digest(info['path'])!=info['sha256']:raise RuntimeError('Auxiliary cache drift')
        pins[str(path)]=digest(path);pins[info['path']]=info['sha256']
    prepared=read_authenticated(REFERENCE/'evaluation/input_manifest.json')
    for key,record in prepared['rows'].items():
        if key.split('/')[0] not in ('base','muq'):continue
        for file in record['files'].values():
            if digest(file['path'])!=file['sha256']:raise RuntimeError('Evaluation input drift')
            pins[file['path']]=file['sha256']
    from .night.night_evaluation import inputs
    for row in inputs():pins[row['path']]=row['sha256']
    initial=[]
    for seed in SEEDS:
        for arm in ARMS:
            run=REFERENCE/'runs'/arm/f'seed_{seed}'
            for filename in ('initial.pt','final.pt','report.json'):
                path=run/filename;pins[str(path)]=digest(path)
            report=read_authenticated(run/'report.json');verify_artifacts(report['artifacts'])
            evaluation=REFERENCE/'evaluation'/arm/f'seed_{seed}'
            for filename in ('inference_report.json','video_report.json','summary.json'):
                path=evaluation/filename;info=read_authenticated(path);pins[str(path)]=digest(path)
                for row in info.get('rows',[]):
                    verify_artifacts(row['artifacts'])
                    for artifact,value in row['artifacts'].items():pins[artifact]=value['sha256']
            family,arch=arm.split('_');seed_all(seed);model=NightModel(family,arch,1024 if family=='muq' else 0)
            blob=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
            if not all(torch.equal(v,blob['model'][k]) for k,v in model.state_dict().items()):raise RuntimeError('Initial model differs')
            initial.append({'arm':arm,'seed':seed,'equal_to_reference':True,'sha256':digest(run/'initial.pt')})
    authenticated_json(root/'initialization_audit.json',{'rows':initial})
    authenticated_json(root/'expression_config.json',read_authenticated(REFERENCE/'expression_config.json'))
    source=freeze_sources(root)
    # Tests for this implementation are included in its immutable identity too.
    for test in ('test_training_runtime.py','test_iteration_speed.py','test_iteration_setup.py','test_night_expression.py','test_beat_conditioned.py'):
        src=Path('/root/lightsem/tests')/test;dest=root/'source/tests'/test
        shutil.copy2(src,dest);dest.chmod(0o444);source['files']['tests/'+test]=digest(src)
    source['source_sha256']=json_digest(source['files']);authenticated_json(root/'frozen_sources.json',source)
    batch=engineering['selected_batch']
    protocol={**reference,'source_sha256':source['source_sha256'],'source_snapshot':source['snapshot'],
              'input_artifacts':pins,'arms':list(ARMS),'seeds':list(SEEDS),'main_model_count':4,
              'feature_root':str(REFERENCE),'reference_root':str(REFERENCE),'compact_artifacts':True,
              'training':{**reference['training'],'steps':486400//batch,'warmup':16000//batch,'batch':batch,
                          'precision':engineering['selected_runtime']['precision'],'runtime':engineering['selected_runtime']},
              'efficiency_gates':{'minimum_training_speedup':2.,'maximum_relative_degradation':.03,
                  'bootstrap_samples':10000,'bootstrap_seed':20260910,'ci':[.025,.975],
                  'all_previously_passing_gates_required':True,'no_new_constant_channels':True},
              'artifact_budget_gib':1.15,'engineering_sha256':digest(root/'engineering.json'),
              'compute_budget':{'policy':'Complete all four fixed cells; no time-based truncation','crops_per_run':486400,'maximum_gpu_concurrency':1},
              'claims':'Computational efficiency and quality retention relative to matched archived FP32 baselines; no improved musical-control claim.',
              'selection':'Final scheduled weights, both seeds; no outcome-based retries. Only two variants eligible for runtime recommendation.',
              'test_access':False}
    authenticated_json(root/'protocol.json',protocol)
    print(json.dumps({'frozen':True,'protocol_sha256':digest(root/'protocol.json'),'steps':protocol['training']['steps']}),flush=True)


def train(root,arm,seed,resume=False):
    setup_cache(root);begin=time.monotonic();p,m,n,_=study_context(root)
    verify_sources(root,check_live=True);require_space(root)
    data=NightResident([e for e in m['entries'] if e['partition']=='train'],n,640,REFERENCE)
    loading=time.monotonic()-begin
    r=run_cell(arm,seed,data,root,resume=resume,initial_overhead=loading)
    complete={'arm':arm,'seed':seed,'loading_seconds':loading,'complete_seconds':r['complete_seconds'],
              'training_report_sha256':digest(root/'runs'/arm/f'seed_{seed}'/'report.json'),
              'training':r,'resumed':resume}
    authenticated_json(root/'runs'/arm/f'seed_{seed}'/'complete_report.json',complete)


def evaluate(root,arm,seed):
    from .night.night_evaluation import infer_cell,render_cell
    verify_sources(root,check_live=True)
    infer_cell(arm,seed,root);render_cell(arm,seed,root)


def relative_comparison(before,after,lower_better,*,seed=20260910,repetitions=10000):
    before=np.asarray(before,float);after=np.asarray(after,float)
    if before.shape!=after.shape or before.ndim!=1 or not len(before) or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('Invalid paired comparison')
    if before.mean()<=1e-12:raise ValueError('Relative noninferiority requires a positive baseline')
    sign=1 if lower_better else -1
    delta=sign*(after.mean()/before.mean()-1)
    ix=np.random.default_rng(seed).integers(len(before),size=(repetitions,len(before)))
    draws=sign*(after[ix].mean(1)/before[ix].mean(1)-1)
    lo,hi=np.quantile(draws,[.025,.975])
    return {'before':float(before.mean()),'after':float(after.mean()),'relative_degradation':float(delta),
            'ci95':[float(lo),float(hi)],'songs':len(before),'passed':bool(delta<=.03 and hi<=.03),
            'status':'pass' if delta<=.03 and hi<=.03 else ('inconclusive' if delta<=.03 else 'fail')}


def summary_gates(rows,videos):
    from .night.night_metrics import aggregate
    gates=aggregate(rows,videos)['gates']
    gates['group_independence']=bool(np.mean([v['independent']['unrelated_velocity_abs_correlation'] is not None and
                                            v['independent']['unrelated_velocity_abs_correlation']<.85 for v in videos])>=.75)
    gates['mirror_geometry']=all(v['mirror_target_error_m']<1e-8 and v['mirror_emitted_ray_error_deg']<.05 for v in videos)
    return gates


def summarize(root):
    from .night.night_evaluation import inputs
    p=read_authenticated(root/'protocol.json');_,norms=base_context();input_rows={r['entry']['key']:r for r in inputs()}
    cells=[]
    for seed in SEEDS:
        for arm in ARMS:
            folder=root/'evaluation'/arm/f'seed_{seed}';ref=REFERENCE/'evaluation'/arm/f'seed_{seed}'
            observed=read_authenticated(folder/'inference_report.json');old=read_authenticated(ref/'inference_report.json')
            videos=read_authenticated(folder/'video_report.json')['rows'];oldvideos=read_authenticated(ref/'video_report.json')['rows']
            bykey={r['key']:r for r in observed['rows']};oldkey={r['key']:r for r in old['rows']}
            if set(bykey)!=set(input_rows) or set(oldkey)!=set(input_rows) or len(videos)!=8:raise RuntimeError('Incomplete cohort')
            before={k:[] for k in ('energy','a_mse','event_ap')};after={k:[] for k in before};no_collapse=True
            for key in sorted(input_rows):
                row=bykey[key];prior=oldkey[key];verify_artifacts(row['artifacts']);verify_artifacts(prior['artifacts'])
                metrics=row['metrics'];oldmetrics=prior['metrics']
                before['energy'].append(oldmetrics['energy_score']['full']);after['energy'].append(metrics['energy_score']['full'])
                with np.load(input_rows[key]['path']) as z:
                    truth=(z['y'].astype(np.float32)-np.asarray(norms['y_center'],np.float32))/np.asarray(norms['y_scale'],np.float32)
                with np.load(ref/key/'full.npz') as z:old_a=z['standardized'][0,:,:4].astype(float)
                before['a_mse'].append(float(np.square(old_a-truth[:,:4]).mean()));after['a_mse'].append(metrics['a_mse'])
                valid=[j for j,v in enumerate(oldmetrics['event_ap']['full']) if v is not None]
                if valid:
                    if any(metrics['event_ap']['full'][j] is None for j in valid):raise RuntimeError('AP support changed')
                    before['event_ap'].append(float(np.mean([oldmetrics['event_ap']['full'][j] for j in valid])))
                    after['event_ap'].append(float(np.mean([metrics['event_ap']['full'][j] for j in valid])))
                oldconstant=np.asarray(oldmetrics['native_dispersion']['exact_constant_channels'],bool)
                newconstant=np.asarray(metrics['native_dispersion']['exact_constant_channels'],bool)
                no_collapse &= not bool((newconstant & ~oldconstant).any())
            comparisons={k:relative_comparison(before[k],after[k],k!='event_ap') for k in before}
            oldgates=summary_gates(old['rows'],oldvideos);newgates=summary_gates(observed['rows'],videos)
            lost=[g for g,v in oldgates.items() if v and not newgates[g]]
            runtime=read_authenticated(root/'runs'/arm/f'seed_{seed}'/'complete_report.json')
            baseline=read_authenticated(REFERENCE/'runs'/arm/f'seed_{seed}'/'report.json')
            # Conservative: candidate includes loading; archived baseline excludes it.
            speedup=baseline['wall_seconds']/runtime['complete_seconds']
            gates={'speed':speedup>=2.,'quality':all(c['passed'] for c in comparisons.values()),
                   'preserved_gates':not lost,'no_new_collapse':no_collapse,'integrity':True}
            cells.append({'arm':arm,'seed':seed,'comparisons':comparisons,'speedup':speedup,
                          'baseline_seconds':baseline['wall_seconds'],'candidate_seconds':runtime['complete_seconds'],
                          'baseline_gates':oldgates,'candidate_gates':newgates,'lost_gates':lost,'gates':gates,
                          'all_gates_pass':all(gates.values()),'training_report_sha256':runtime['training_report_sha256']})
    eligible=[arm for arm in ARMS if all(c['all_gates_pass'] for c in cells if c['arm']==arm)]
    report={'cells':cells,'eligible_variants':eligible,'runtime':p['training']['runtime'],'batch':p['training']['batch'],
            'all_gates_pass':all(c['all_gates_pass'] for c in cells),'protocol_sha256':digest(root/'protocol.json'),
            'quality_inference_precision':'fp32','test_access':False,'not_a_musical_control_claim':True}
    authenticated_json(root/'summary.json',report);print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['benchmark','repair-compiler-screen','freeze','train','evaluate','summarize'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--resume',action='store_true');a=p.parse_args();root=a.root
    if a.stage=='freeze':freeze(root);return
    if a.stage=='summarize':summarize(root);return
    with gpu_lock(root):
        if a.stage=='benchmark':benchmark(root)
        elif a.stage=='repair-compiler-screen':repair_compiler_screen(root)
        elif a.stage=='train':
            if a.arm is None or a.seed is None:raise ValueError('Explicit arm/seed required')
            train(root,a.arm,a.seed,a.resume)
        else:
            if a.arm is None or a.seed is None:raise ValueError('Explicit arm/seed required')
            evaluate(root,a.arm,a.seed)


if __name__=='__main__':main()

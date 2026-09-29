"""Immutable, matched and resumable native music-training cells."""
import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .audio_prepare import ROOT
from .music_data import MusicResident
from .music_model import MusicModel,MusicConfig
from ..training_runtime import TrainingRuntime,BatchStream,backward_checked
from ..stage11.stage11_common import (read_authenticated,authenticated_json,digest,json_digest,freeze_sources,verify_sources,
    seed_all,rng_state,restore_rng,atomic_torch,atomic_json,artifact_manifest,verify_artifacts,require_space,gpu_lock)


def build_windows(manifest):
    entries = sorted([e for e in manifest['entries'] if e['partition']=='val' and e['n_frames']>=960],key=lambda e:e['key'])
    rows = []
    for index,e in enumerate(entries):
        donor = next(entries[(index+j)%len(entries)] for j in range(1,len(entries))
                     if entries[(index+j)%len(entries)]['canonical_id']!=e['canonical_id'])
        for start in range(0,e['n_frames']-960+1,960):
            shift = max((0,e['n_frames']-960),key=lambda s:(abs(s-start),s))
            if abs(shift-start)<150:
                raise ValueError('Insufficient within-track shift distance')
            token = int(hashlib.sha256(f'{e["key"]}:{start}'.encode()).hexdigest()[:12],16)
            rows.append({'key':e['key'],'canonical_id':e['canonical_id'],'start':start,'frames':960,
                'donor_key':donor['key'],'donor_start':token%(donor['n_frames']-960+1),
                'shift_start':shift,'draw_seeds':[int((token+7001+i*104729)%2**31) for i in range(6)]})
    return rows


def freeze():
    out = ROOT/'music'
    if (out/'protocol.json').exists():
        return read_authenticated(out/'protocol.json')
    if not read_authenticated(ROOT/'codec/final_acceptance.json')['audio_training_authorized']:
        raise ValueError('Representation prerequisite failed')
    verify_sources(ROOT/'codec')
    verify_sources(ROOT/'codec/review')
    native = read_authenticated(ROOT/'codec/review/protocol.json')
    if digest(Path(__file__).with_name('native_codec.py'))!=native['codec_sha256']:
        raise ValueError('Accepted native codec changed')
    eng = read_authenticated(out/'engineering.json')
    for name,field in [('music_model.py','model_source_sha256'),('music_data.py','data_source_sha256')]:
        if digest(Path(__file__).with_name(name))!=eng[field]:
            raise ValueError('Implementation changed after preflight')
    registration = read_authenticated(out/'registration.json')
    manifest = read_authenticated(ROOT/'data/audio_manifest.json')
    snapshot = freeze_sources(out)
    test = Path('/root/lightsem/tests/test_motion_study.py')
    dest = out/'source/tests/test_motion_study.py'
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(test,dest)
    dest.chmod(0o444)
    snapshot['files']['tests/test_motion_study.py'] = digest(test)
    snapshot['source_sha256'] = json_digest(snapshot['files'])
    authenticated_json(out/'frozen_sources.json',snapshot)
    inputs = [ROOT/'codec/final_acceptance.json',ROOT/'data/audio_manifest.json',ROOT/'data/audio_normalizers.json',
              ROOT/'data/motion_normalizers.json',out/'engineering.json',out/'registration.json']
    historical = Path('/mnt/lightsem-data/experiments/lightsem')
    inputs += [historical/'beat_conditioned_20260909/run/final.pt',
               historical/'beat_conditioned_20260909/normalizers.json',
               historical/'night_expression_20260909/diagnostics/pose_probes/descriptor256/seed_0/final.pt']
    protocol = {'source_sha256':snapshot['source_sha256'],'inputs':{str(p):digest(p) for p in inputs},
        'hypothesis_id':registration['hypothesis_id'],'arms':['regression','flow'],'seed':0,
        'model':MusicModel(MusicConfig(),read_authenticated(ROOT/'data/motion_normalizers.json')).config_dict(),
        'training':{'steps':30000,'frames':960,'batch':eng['selected_batch'],'lr':3e-4,'warmup':500,
                    'checkpoint_interval':1000,'runtime':TrainingRuntime().as_dict()},
        'evaluation':{'windows':build_windows(manifest),'draws':6,'steps':32,'checkpoint':'final scheduled step',
            'bootstrap_draws':10000,'bootstrap_seed':20260910,'aggregation':'equal canonical song / recording / window',
            'score':'unbiased energy score: E RMS(draw-target) - .5 E RMS(distinct_draw-pair), fixed TRAIN-standardized18x64 motion summary vector',
            'shift':'farthest contiguous32s window within the same recording; >=5s displacement; no circular-wrap discontinuity',
            'donor':'next canonical-distinct VAL recording, deterministic hashed contiguous start',
            'direct_audio_lighting':False,'full_review':'single fixed draw of complete12 review recordings, plus full Shpongle and historical-clip incumbent comparisons'},
        'gates':{'conditioning_relative_improvement_min':.05,'conditioning_ci_lower_gt':0.,
            'chunk_max_abs':5e-5,'generated_energy_ratio':[.5,2.],
            'generated_root_travel_ratio':[.25,2.],
            'generated_contact_coverage_ratio':[.5,1.5],
            'high_frequency_fraction_excess_max':.05,'high_frequency_cutoff_hz':6,
            'floor_penetration_p95_excess_max_m':.05,'contact_sliding_excess_max_m_s':.20,
            'visual_review_required':True,'finite_required':True,
            'quality_scope':'canonical-weighted correct-audio generated samples across all fixed windows; full-track continuity and visual review separately'},
        'selection':'all initial/final arms and reviews complete; primary flow must satisfy every gate; no promotion from conditioning alone',
        'seed1':'only after entire pipeline qualifies; finalist plus matched baseline',
        'test_access':False}
    authenticated_json(out/'protocol.json',protocol)
    return protocol


def context():
    p = read_authenticated(ROOT/'music/protocol.json')
    snapshot = verify_sources(ROOT/'music')
    if p['source_sha256']!=snapshot['source_sha256'] or not Path(__file__).resolve().is_relative_to(Path(snapshot['snapshot']).resolve()):
        raise ValueError('Scientific music execution requires the immutable launcher')
    for path,sha in p['inputs'].items():
        if digest(path)!=sha:
            raise ValueError('Frozen music input changed: '+path)
    return p,read_authenticated(ROOT/'data/audio_manifest.json'),read_authenticated(ROOT/'data/audio_normalizers.json'),read_authenticated(ROOT/'data/motion_normalizers.json')


def train(arm,seed=0,resume=False):
    protocol,manifest,audio_norms,norms = context()
    if arm not in protocol['arms']:
        raise ValueError('Unregistered arm')
    receipt = read_authenticated(ROOT/'music/experiment.json')
    if receipt['protocol_sha256']!=digest(ROOT/'music/protocol.json'):
        raise ValueError('Experiment not started for this protocol')
    if seed!=0:
        lock = read_authenticated(ROOT/'confirmation_lock.json')
        if not lock['all_pipeline_gates_passed']:
            raise ValueError('Second seed is not qualified')
    require_space(ROOT,gb=2)
    run = ROOT/'music/runs'/arm/f'seed_{seed}'
    cfg = protocol['training']
    identity = {'arm':arm,'seed':seed,'protocol_sha256':digest(ROOT/'music/protocol.json'),
                'source_sha256':protocol['source_sha256'],'training':cfg}
    if (run/'report.json').exists():
        result = read_authenticated(run/'report.json')
        verify_artifacts(result['artifacts'])
        if result['identity']!=identity:
            raise ValueError('Completed run identity differs')
        return result
    existing = run.exists() and any(run.iterdir())
    if existing and not resume:
        raise ValueError('Partial run requires --resume')
    run.mkdir(parents=True,exist_ok=True)
    seed_all(seed)
    runtime = TrainingRuntime.from_training(cfg)
    runtime.apply()
    data = MusicResident(manifest,audio_norms,cfg['frames'])
    model = MusicModel(MusicConfig(arm=arm),norms).cuda()
    optimizer = runtime.make_optimizer(model,cfg['lr'])
    total,warmup = cfg['steps'],cfg['warmup']
    schedule = lambda step:min(1.,(step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,total-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    rng = np.random.default_rng(10000+seed)
    start,elapsed = 0,0.
    if existing:
        checkpoint = run/'latest.pt' if (run/'latest.pt').exists() else run/'initial.pt'
        state = torch.load(checkpoint,map_location='cuda',weights_only=False)
        if state['identity']!=identity:
            raise ValueError('Resume identity differs')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'],rng)
        start,elapsed = state['step'],state['training_seconds']
        if (run/'training.jsonl').exists():
            prior = run/f'training_before_resume_{time.time_ns()}.jsonl'
            shutil.copy2(run/'training.jsonl',prior)
            (run/'training.jsonl').write_text(''.join(line+'\n' for line in prior.read_text().splitlines() if json.loads(line)['step']<=start))
    def save(step,path):
        state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'rng':rng_state(rng),'identity':identity,'step':step,'training_seconds':elapsed,
                 'model_config':model.config_dict(),'normalizers':norms,'audio_normalizers':audio_norms}
        atomic_torch(path,state)
        return state
    if not existing:
        save(0,run/'initial.pt')
    fn = runtime.loss_function(model)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    wall = interval = time.monotonic()
    with BatchStream(data,cfg['batch'],cfg['frames'],rng,runtime.prefetch) as batches,(run/'training.jsonl').open('a') as log:
        for step in range(start,total):
            x,y,b = batches.next()
            noise,t = model.draws(x) # Consumed by both arms, matching global RNG cursors.
            optimizer.zero_grad(set_to_none=True)
            with runtime.autocast():
                losses = fn(x,y,b,noise,t)
            gradient = backward_checked(losses,model)
            optimizer.step()
            scheduler.step()
            if step==start or (step+1)%100==0 or step+1==total:
                torch.cuda.synchronize()
                now = time.monotonic()
                elapsed += now-interval
                interval = now
                row = {'arm':arm,'seed':seed,'step':step+1,'training_seconds':elapsed,
                       'lr':scheduler.get_last_lr()[0],'grad':float(gradient),
                       **{k:float(v.detach()) for k,v in losses.items()}}
                log.write(json.dumps(row)+'\n')
                log.flush()
                atomic_json(run/'status.json',row)
                atomic_json(ROOT/'status.json',row)
                print(json.dumps(row),flush=True)
            if (step+1)%cfg['checkpoint_interval']==0:
                save(step+1,run/'latest.pt')
        final = save(total,run/'latest.pt')
        atomic_torch(run/'final.pt',final)
    result = {'identity':identity,'steps':total,'model_config':model.config_dict(),
              'parameters':sum(p.numel() for p in model.parameters()),'training_seconds':elapsed,
              'wall_seconds_this_invocation':time.monotonic()-wall,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
              'artifacts':artifact_manifest([run/'initial.pt',run/'final.pt',run/'latest.pt',run/'training.jsonl'])}
    authenticated_json(run/'report.json',result)
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['freeze','train'])
    p.add_argument('--arm',choices=['regression','flow'],default='flow')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    if args.stage=='freeze':
        freeze()
    else:
        with gpu_lock(ROOT):
            train(args.arm,args.seed,args.resume)

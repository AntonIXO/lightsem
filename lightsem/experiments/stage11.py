"""Stage 11 campaign CLI. Every scientific operation names its frozen protocol."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess
import time

from .stage11_common import (ROOT, WORKSPACE, atomic_json, authenticated_json, read_json,
    read_authenticated, digest, json_digest, gpu_lock, freeze_sources, verify_sources,
    verify_artifacts, require_space)


def initialize_protocol(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'protocol.json'
    if path.exists():return read_json(path)
    born=int(subprocess.check_output(['stat','-c','%W',str(root)],text=True).strip()) or int(time.time())
    motion={'family':'motion','steps':12000,'batch':48,'frames':160,'lr':3e-4,
            'validation_steps':[4000,8000,12000],'resident_gpu':True,'log_every':100,
            'precision':'fp32','scheduler':'onecycle','population':'all'}
    scene={'family':'scene','steps':32000,'batch':32,'frames':600,'lr':3e-4,
           'validation_steps':[8000,16000,32000],'resident_gpu':True,'log_every':100,
           'precision':'fp32','scheduler':'warmup_cosine','population':'all'}
    arms={'D_old':dict(motion,population='old'),'D_full':dict(motion),
          'S_pointwise':dict(scene,flow_kind='pointwise'),'S_temporal':dict(scene,flow_kind='temporal'),
          'S_swapped':dict(scene,flow_kind='temporal',audio_swap='nonbeat_48_recipient_beat_4')}
    p={'schema_version':1,'campaign':'stage11_20260908','created_at':time.time(),
       'campaign_started_at':born,'max_wall_hours':24,'deadline':born+24*3600,
       'minimum_gpu_seconds':36000,'evaluation_reserve_hours':2,
       'training_seeds':[0,1,2,3,4],'extension_seeds':[5],
       'generation_seeds':[101,102,103,104],'arms':arms,
       'source_corpora':['justdance'],'excluded_corpora_reason':{'rpmc':'Missing grouping of 699 shards into 35 source performances'},
       'normalization':{'motion':'common old-training statistics; per-checkpoint pinning',
                        'scene':'value and p90 recording-relative before cropping, remaining scales training-only'},
       'evaluation':{'steps':8,'sensitivity_steps':[4,16],'circular_shifts':200,
                     'circular_null':'mean_ap','donor_count':6,'reference_tolerance_s':.2,
                     'bootstrap_draws':4000,'bootstrap_seed':1100,'required_val_songs':40,'required_test_songs':40,
                     'primary_motion_channels':['com_speed','vert_bounce_amp'],'motion_tau_s':2},
       'gates':{'motion_r2_delta':.02,'temporal_relative_reduction':.15,'temporal_absolute_reduction':.02,
                'marginal_relative_deterioration':.05,'marginal_absolute_floor':.02,
                'minimum_variation_ratio':.7,'maximum_hf_ratio':2.,'minimum_eligible_fraction':.8,
                'conditioning_ap_delta':.02,'ci_confidence':.95,'human_preference':'unmeasured; no default promotion'},
       'render':{'presentation_seed':1100,'n_songs':20,'clip_s':30.,'luma_tolerance':.03,
                 'selection_seed_policy':'canonical hash selects training seed and generation seed; never best-looking seed',
                 'comparisons':[['D_full','D_old'],['S_temporal','S_pointwise']]},
       'resource_policy':{'maximum_gpu_processes':1,'cpu_preparation_workers':4,'cpu_metric_workers':2,
                          'minimum_free_gib':15,'gpu_resident_training_cache':True,
                          'batch_choice':'Measured batch128 slower per training window; retain fastest batch32/48 and original update budgets'},
       'frozen':False}
    atomic_json(path,p)
    return p


def freeze(root):
    from .stage11_training import verify_training_inputs
    from .stage11_delivery import pin_render_assets
    root=Path(root);p=read_json(root/'protocol.json');verify_training_inputs(root)
    if (root/'frozen_sources.json').exists():
        verify_sources(root,check_live=True)
        return read_authenticated(root/'protocol.json')
    pin_render_assets(root)
    p['render_assets_sha256']=digest(root/'render_assets.json')
    p['frozen']=True;p['data_sha256']=digest(root/'training_data.json');p['normalizers_sha256']=digest(root/'normalizers.json')
    p['data_manifest_sha256']=digest(root/'data/manifest.json')
    # Protocol identity is finalized before source snapshot publication.
    authenticated_json(root/'protocol.json',p)
    info=freeze_sources(root)
    atomic_json(root/'frozen_identity.json',{'protocol_sha256':digest(root/'protocol.json'),
        'source_sha256':info['source_sha256'],'data_sha256':p['data_sha256'],
        'normalizers_sha256':p['normalizers_sha256'],'git_head':info['git_head']})
    return info


def verify(root):
    from .stage11_training import verify_training_inputs
    root=Path(root);p=read_authenticated(root/'protocol.json')
    if not p.get('frozen'):raise RuntimeError('Protocol not frozen')
    data=verify_training_inputs(root);src=verify_sources(root)
    result={'status':'verified','protocol_sha256':digest(root/'protocol.json'),
            'source_sha256':src['source_sha256'],'data_sha256':digest(root/'training_data.json'),
            'n_motion_clips':len(data['motion']),'n_scene_recordings':len(data['scene']),
            'free_gib':require_space(root),'verified_at':time.time()}
    authenticated_json(root/'verification.json',result)
    return result


def unlock_test(root):
    """Called explicitly by the primary agent after independent artifact review."""
    root=Path(root);p=read_authenticated(root/'protocol.json')
    agg=read_authenticated(root/'results/aggregate_val.json')
    if agg['status'] not in ('complete','completed'):
        raise RuntimeError('Incomplete validation cannot unlock test')
    families=agg.get('eligible_families',[])
    if not families:raise RuntimeError('No family passed validation gates')
    arms={}
    for arm,config in p['arms'].items():
        if config['family'] not in families:continue
        seeds=list(agg.get('arm_seeds',{}).get(arm,p['training_seeds']))
        checks={}
        for seed in seeds:
            path=root/'runs'/arm/f'seed_{seed}'
            report=read_authenticated(path/'report.json');verify_artifacts(report['artifacts'])
            checks[str(seed)]=digest(path/'best.pt')
        arms[arm]={'config_sha256':json_digest(config),'training_seeds':seeds,'checkpoints':checks}
    result={'schema_version':1,'protocol_sha256':digest(root/'protocol.json'),
            'validation_aggregate_sha256':digest(root/'results/aggregate_val.json'),
            'eligible_families':families,'arms':arms,'reviewed_by':'primary_agent', 'reviewed_at':time.time()}
    if (root/'test_unlock.json').exists():raise RuntimeError('Test unlock already exists')
    authenticated_json(root/'test_unlock.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('init');sub.add_parser('freeze');sub.add_parser('verify');sub.add_parser('unlock-test')
    p=sub.add_parser('prepare');p.add_argument('--workers',type=int,default=4);p.add_argument('--tensors-only',action='store_true')
    p=sub.add_parser('train');p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--engineering-steps',type=int)
    p=sub.add_parser('evaluate');p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--controls',action='store_true');p.add_argument('--steps',type=int,default=8)
    p=sub.add_parser('aggregate');p.add_argument('--split',choices=['val','test'],default='val')
    p=sub.add_parser('render');p.add_argument('--split',choices=['val','test'],default='val')
    p=sub.add_parser('rate');p.add_argument('--rater',default='anton')
    p=sub.add_parser('supervise');p.add_argument('--phase',choices=['core','test','diagnostic_render'],default='core');p.add_argument('--resume-supervisor',action='store_true')
    args=parser.parse_args(argv);root=args.root.resolve()
    if args.command=='init':result=initialize_protocol(root)
    elif args.command=='prepare':
        from .stage11_data import prepare_data
        from .stage11_training import prepare_training_data
        initialize_protocol(root)
        if not args.tensors_only:prepare_data(root,workers=args.workers)
        result=prepare_training_data(root)
        result={'status':'prepared','motion_clips':len(result['motion']),'scene_recordings':len(result['scene'])}
    elif args.command=='freeze':result=freeze(root)
    elif args.command=='verify':result=verify(root)
    elif args.command=='unlock-test':result=unlock_test(root)
    elif args.command=='train':
        from .stage11_training import train_run
        with gpu_lock(root):result=train_run(root,args.arm,args.seed,resume=args.resume,engineering_steps=args.engineering_steps)
    elif args.command=='evaluate':
        from .stage11_evaluation import evaluate_run
        with gpu_lock(root):result=evaluate_run(root,args.arm,args.seed,split=args.split,controls=args.controls,steps=args.steps)
    elif args.command=='aggregate':
        from .stage11_evaluation import aggregate
        result=aggregate(root,split=args.split)
    elif args.command=='render':
        from .stage11_delivery import render_campaign
        with gpu_lock(root):result=render_campaign(root,split=args.split)
    elif args.command=='rate':
        from .stage11_rating import rate
        result={'completed':rate(root,rater=args.rater)}
    else:
        from .stage11_supervisor import supervise
        result=supervise(root,phase=args.phase,resume_supervisor=args.resume_supervisor)
    print(json.dumps(result,allow_nan=False,default=str),flush=True)


if __name__=='__main__':main()

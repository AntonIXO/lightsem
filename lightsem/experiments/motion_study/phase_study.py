"""Frozen, analytically fitted codec revision; independent source and visual gates."""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import ROOT as PARENT
from .phase_registration import ROOT
from .phase_model import PhaseCodec, PhaseConfig, fit_moments
from .geometry import FPS, joints_from_features
from .metrics import compare
from .evaluate import aggregate
from .selection import numerical_gates
from .render import render_comparison
from ..stage11.stage11_common import (authenticated_json, read_authenticated, digest, json_digest,
    freeze_sources, verify_sources, atomic_torch, gpu_lock, require_space)

ARMS = ('frame24','phase64','phase128','phase256')


def freeze(root=ROOT):
    root = Path(root)
    out = root/'codec'
    if (out/'protocol.json').exists():
        return read_authenticated(out/'protocol.json')
    prior = read_authenticated(PARENT/'codec/final_selection.json')
    if prior['eligible_arms']:
        raise ValueError('Revision requires failure of all original arms')
    source = read_authenticated(PARENT/'data/source_gate.json')
    if not source['passed']:
        raise ValueError('Source prerequisite failed')
    read_authenticated(root/'registration.json')
    engineering = read_authenticated(out/'engineering.json')
    if engineering['model_source_sha256'] != digest(Path(__file__).with_name('phase_model.py')):
        raise ValueError('Codec changed after engineering verification')
    snapshot = freeze_sources(out)
    test = Path('/root/lightsem/tests/test_motion_study.py')
    dest = out/'source/tests/test_motion_study.py'
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(test,dest)
    dest.chmod(0o444)
    snapshot['files']['tests/test_motion_study.py'] = digest(test)
    snapshot['source_sha256'] = json_digest(snapshot['files'])
    authenticated_json(out/'frozen_sources.json',snapshot)
    paths = [PARENT/'data/manifest.json',PARENT/'data/normalizers.json',PARENT/'data/split.json',
             PARENT/'data/source_gate.json',PARENT/'codec/final_selection.json',
             PARENT/'codec/vibescience_results.json',root/'registration.json',out/'engineering.json']
    gates = dict(read_authenticated(PARENT/'codec/protocol.json')['gates'])
    gates.update(quiet_window_frames=60,quiet_min_windows=1,stationary_min_windows=1,
                 quiet_reduction='maximum excess across every qualifying two-second VAL window',
                 visual_min_fixed_clips=12)
    protocol = {'source_sha256':snapshot['source_sha256'],'inputs':{str(p):digest(p) for p in paths},
        'parent':str(PARENT),'arms':{arm:{'ranks':list(PhaseConfig(arm).ranks),
                 'latent':PhaseConfig(arm).latent,'stride':PhaseConfig(arm).stride} for arm in ARMS},
        'primary_arm':'phase128','primary_baseline':str(PARENT/'codec/evaluation/temporal128/seed_0/final/report.json'),
        'fit':'one full TRAIN pass for exact equal-canonical/record second moments; float64 eigensolver; no gradient steps',
        'dc_centering':'TRAIN mean; AC offsets fixed zero, covariance is zero-origin second moment',
        'side_channels':'native root3 + contacts4 per frame, explicitly counted; generated in future audio model',
        'initialization_seed':20260910,'evaluation':{'partition':'val','frames':480,'stride':480,
            'full_sequences':True,'quiet_frames':60,'bootstrap_seed':20260910,'bootstrap_draws':10000},
        'gates':gates,'selection':'all arms and external reviews complete, then lowest total latent values/sec',
        'test_access':False,'scientific_claim_limit':'compound codec revision; no interpolation-only or compute-matched causal claim'}
    authenticated_json(out/'protocol.json',protocol)
    return protocol


def context(root):
    root = Path(root)
    p = read_authenticated(root/'codec/protocol.json')
    snapshot = verify_sources(root/'codec')
    if p['source_sha256'] != snapshot['source_sha256']:
        raise ValueError('Code identity differs')
    if not Path(__file__).resolve().is_relative_to(Path(snapshot['snapshot']).resolve()):
        raise ValueError('Scientific phase study requires the frozen launcher')
    for path,sha in p['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Frozen input changed: '+path)
    parent = Path(p['parent'])
    return p,read_authenticated(parent/'data/manifest.json'),read_authenticated(parent/'data/normalizers.json')


def fit(root=ROOT):
    root = Path(root)
    protocol,manifest,norms = context(root)
    if not (root/'codec/experiment.json').exists():
        raise ValueError('Start the preregistered experiment before analytic fitting')
    if read_authenticated(root/'codec/experiment.json')['protocol_sha256'] != digest(root/'codec/protocol.json'):
        raise ValueError('Experiment protocol differs')
    out = root/'codec/fit'
    if (out/'report.json').exists():
        prior = read_authenticated(out/'report.json')
        for name,sha in prior['artifacts'].items():
            if digest(out/name) != sha:
                raise ValueError('Fitted transform changed')
        return prior
    require_space(root,gb=2)
    out.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    fitted = fit_moments(manifest['entries'],norms)
    rng = np.random.default_rng(protocol['initialization_seed'])
    initial = {stride:{'dc_mean':np.zeros(132,dtype=np.float32),
                     'basis':[np.linalg.qr(rng.standard_normal((132,132)))[0].astype(np.float32)
                              for _ in range(stride)]} for stride in (1,4)}
    atomic_torch(out/'initial.pt',{'transforms':initial,'protocol_sha256':digest(root/'codec/protocol.json')})
    atomic_torch(out/'final.pt',{'transforms':fitted,'protocol_sha256':digest(root/'codec/protocol.json')})
    result = {'protocol_sha256':digest(root/'codec/protocol.json'),'seconds':time.monotonic()-start,
        'fit_records':sum(e['partition']=='train' for e in manifest['entries']),
        'artifacts':{name:digest(out/name) for name in ('initial.pt','final.pt')},
        'eigenvalues':{str(k):v['eigenvalues'] for k,v in fitted.items()}}
    authenticated_json(out/'report.json',result)
    return result


def load_model(root,arm,checkpoint='final',device='cuda'):
    protocol,manifest,norms = context(root)
    fit_report = read_authenticated(root/'codec/fit/report.json')
    weights = root/'codec/fit'/f'{checkpoint}.pt'
    if digest(weights) != fit_report['artifacts'][weights.name]:
        raise ValueError('Transform changed')
    state = torch.load(weights,map_location='cpu',weights_only=False)
    if state['protocol_sha256'] != digest(root/'codec/protocol.json'):
        raise ValueError('Transform belongs to another protocol')
    cfg = PhaseConfig(arm)
    model = PhaseCodec(cfg,norms,state['transforms'][cfg.stride]).to(device).eval()
    return model,protocol,manifest,norms


def evaluate(root,arm,checkpoint):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model,protocol,manifest,norms = load_model(root,arm,checkpoint)
    out = root/'codec/evaluation'/arm/checkpoint
    if (out/'report.json').exists():
        prior = read_authenticated(out/'report.json')
        if prior['protocol_sha256'] != digest(root/'codec/protocol.json'):
            raise ValueError('Evaluation identity differs')
        return prior
    out.mkdir(parents=True,exist_ok=True)
    rows,sequences,quiet,stationary = [],[],[],[]
    chunk_max = 0.
    for entry in manifest['entries']:
        if entry['partition'] != 'val':
            continue
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('VAL source changed')
        with np.load(entry['path']) as value:
            reference = value['features']
        with torch.inference_mode():
            inp = torch.from_numpy(reference[None]).cuda()
            direct = model(inp)
            full = model.forward_chunked(inp)
            chunk_max = max(chunk_max,float((direct-full).abs().max()))
            prediction = full.cpu().numpy()[0]
        for start in range(0,len(reference)-480+1,480):
            # Independent short-window execution as well as full execution.
            with torch.inference_mode():
                local = model(torch.from_numpy(reference[None,start:start+480]).cuda()).cpu().numpy()[0]
            rows.append({'key':entry['key'],'canonical_id':entry['canonical_id'],'start':start,
                         'metrics':compare(reference[start:start+480],local,norms['velocity_rms'])})
        m = compare(reference,prediction,norms['velocity_rms'])
        seq = {'key':entry['key'],'canonical_id':entry['canonical_id'],'metrics':m}
        if checkpoint=='final':
            path = out/f'{entry["key"]}.npz'
            if path.exists():
                with np.load(path) as prior:
                    if not np.array_equal(prior['features'],prediction):
                        raise ValueError('Partial evaluation replay differs')
            else:
                np.savez_compressed(path,features=prediction)
            seq.update(path=str(path),sha256=digest(path))
        sequences.append(seq)
        for start in range(0,len(reference)-60+1,60):
            m = compare(reference[start:start+60],prediction[start:start+60],norms['velocity_rms'])
            if m['kinetic_reference'] < .01:
                quiet.append({'key':entry['key'],'start':start,'speed_excess':float(max(0,np.sqrt(m['kinetic_prediction'])-np.sqrt(m['kinetic_reference'])))})
            if m['travel_reference_m'] < .1:
                stationary.append({'key':entry['key'],'start':start,'travel_error':abs(m['travel_prediction_m']-m['travel_reference_m'])})
        print(json.dumps({'arm':arm,'checkpoint':checkpoint,'key':entry['key']}),flush=True)
    result = {'arm':arm,'checkpoint':checkpoint,'protocol_sha256':digest(root/'codec/protocol.json'),
        'checkpoint_sha256':digest(root/'codec/fit'/f'{checkpoint}.pt'),'config':model.config_dict(),
        'rows':rows,'sequences':sequences,'window_summary':aggregate(rows),'full_summary':aggregate(sequences),
        'quiet_windows':quiet,'stationary_windows':stationary,'chunk_max_abs':chunk_max,'test_access':False}
    authenticated_json(out/'report.json',result)
    return result


def gates(report,protocol):
    result = numerical_gates(report,protocol)
    result['quiet_holds'] = bool(len(report['quiet_windows'])>=protocol['gates']['quiet_min_windows'] and
        len(report['stationary_windows'])>=protocol['gates']['stationary_min_windows'] and
        max(r['speed_excess'] for r in report['quiet_windows'])<=protocol['gates']['quiet_speed_excess_max_m_s'] and
        max(r['travel_error'] for r in report['stationary_windows'])<=protocol['gates']['stationary_travel_error_max_m'])
    result['continuity'] = result['continuity'] and report['chunk_max_abs']<=protocol['gates']['chunk_max_abs']
    return result


def render(root,arms):
    _,manifest,_ = context(root)
    if len(arms)>2:
        raise ValueError('At most two candidate panels')
    entries = {e['key']:e for e in manifest['entries']}
    reports = {arm:read_authenticated(root/'codec/evaluation'/arm/'final/report.json') for arm in arms}
    indices = {a:{r['key']:r for r in report['sequences']} for a,report in reports.items()}
    out = root/'delivery'/('_'.join(arms))
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    for clip in manifest['review_clips']:
        entry = entries[clip['key']]
        path = out/f'{clip["category"]}_{clip["key"]}_{clip["start"]}.mp4'
        if path.with_suffix('.json').exists():
            prior = read_authenticated(path.with_suffix('.json'))
            if digest(path)!=prior['sha256']:
                raise ValueError('Review artifact changed')
            rows.append(prior)
            continue
        sl = slice(clip['start'],clip['start']+clip['frames'])
        with np.load(entry['path']) as value:
            models = [('Source motion',value['joints'][sl],value['contacts'][sl])]
        for arm in arms:
            row = indices[arm][entry['key']]
            if digest(row['path'])!=row['sha256']:
                raise ValueError('Predictions changed')
            with np.load(row['path']) as value:
                features = value['features']
            with torch.inference_mode():
                joints = joints_from_features(torch.from_numpy(features)).numpy()
            models.append((arm,joints[sl],features[sl,135:]))
        rows.append(render_comparison(path,models,entry['wav_path'],clip['start']/FPS,
                                      entry['key']+' / '+clip['category']))
        print(str(path),flush=True)
    authenticated_json(out/'report.json',{'rows':rows,'arms':arms,
        'evaluations':{a:digest(root/'codec/evaluation'/a/'final/report.json') for a in arms}})


def select(root):
    protocol,_,_ = context(root)
    rows = []
    for arm in ARMS:
        path = root/'codec/evaluation'/arm/'final/report.json'
        report = read_authenticated(path)
        visual = read_authenticated(root/'codec/reviews'/f'{arm}.json')
        if visual['evaluation_sha256']!=digest(path):
            raise ValueError('Review identity differs')
        result = gates(report,protocol)
        result['visual_review'] = bool(visual['passed'])
        result['source_quality'] = read_authenticated(Path(protocol['parent'])/'data/source_gate.json')['passed']
        cfg = PhaseConfig(arm)
        rows.append({'arm':arm,'gates':result,'eligible':all(result.values()),
                     'latent_values_per_second':cfg.latent*FPS/cfg.stride,
                     'evaluation_sha256':digest(path),'window':report['window_summary']['metrics'],
                     'full':report['full_summary']['metrics']})
    eligible = sorted([r for r in rows if r['eligible']],key=lambda r:r['latent_values_per_second'])
    result = {'rows':rows,'eligible_arms':[r['arm'] for r in eligible],
              'selected':eligible[0]['arm'] if eligible else None,'audio_training_authorized':bool(eligible),
              'protocol_sha256':digest(root/'codec/protocol.json'),'test_access':False}
    authenticated_json(root/'codec/final_selection.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','fit','evaluate','render','select'])
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--arm',choices=ARMS,default='phase128')
    parser.add_argument('--arms',nargs='+',default=['frame24','phase128'])
    parser.add_argument('--checkpoint',choices=['initial','final'],default='final')
    args = parser.parse_args()
    if args.stage=='freeze':
        print(json.dumps(freeze(args.root)))
    elif args.stage=='fit':
        print(json.dumps(fit(args.root)))
    elif args.stage=='select':
        print(json.dumps(select(args.root)))
    else:
        with gpu_lock(args.root):
            if args.stage=='evaluate':
                result = evaluate(args.root,args.arm,args.checkpoint)
                print(json.dumps({'arm':args.arm,'window':result['window_summary']['metrics'],
                                  'full':result['full_summary']['metrics']}))
            else:
                render(args.root,args.arms)

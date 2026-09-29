"""Frozen audio controls, physical motion scores and full-track continuity."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import ROOT as SOURCE
from .audio_prepare import ROOT
from .geometry import joints_from_features,FEET
from .music_model import MusicModel,MusicConfig
from .music_data import motion_summary,energy_score
from .music_training import context
from ..stage11.stage11_common import read_authenticated,authenticated_json,digest,seed_all,gpu_lock


def aggregate(rows):
    recordings = defaultdict(list)
    for row in rows:
        recordings[(row['canonical_id'],row['key'])].append(row['metrics'])
    groups = defaultdict(list)
    for (group,key),values in recordings.items():
        groups[group].append({k:float(np.mean([v[k] for v in values])) for k in values[0]})
    per_song = {g:{k:float(np.mean([v[k] for v in values])) for k in values[0]} for g,values in groups.items()}
    metrics = {k:float(np.mean([v[k] for v in per_song.values()])) for k in next(iter(per_song.values()))}
    return {'metrics':metrics,'per_song':per_song}


def load_model(arm,seed,checkpoint):
    protocol,manifest,audio_norms,norms = context()
    run = ROOT/'music/runs'/arm/f'seed_{seed}'
    path = run/f'{checkpoint}.pt'
    state = torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']['protocol_sha256']!=digest(ROOT/'music/protocol.json') or state['identity']['arm']!=arm:
        raise ValueError('Weights belong to another cell')
    if state['step']!=(0 if checkpoint=='initial' else protocol['training']['steps']):
        raise ValueError('Only initialization and final scheduled state are evaluated')
    if checkpoint=='final':
        report = read_authenticated(run/'report.json')
        if report['identity']!=state['identity']:
            raise ValueError('Completed training identity differs')
    model = MusicModel(MusicConfig(arm=arm),norms).cuda().eval()
    model.load_state_dict(state['model'])
    return model,protocol,manifest,audio_norms,norms,path


def load_arrays(manifest,audio_norms):
    audio,motion = {},{}
    for e in manifest['entries']:
        if e['partition']!='val':
            continue
        if digest(e['audio_path'])!=e['audio_sha256'] or digest(e['path'])!=e['sha256']:
            raise ValueError('VAL input changed')
        with np.load(e['audio_path']) as data:
            x = ((data['x']-np.asarray(audio_norms['mean'],np.float32))/np.asarray(audio_norms['std'],np.float32)).astype(np.float32)
        with np.load(e['path']) as data:
            y = data['features']
        audio[e['key']],motion[e['key']] = x,y
    return audio,motion


@torch.inference_mode()
def physical(features):
    """Generated contacts are evaluated against generated feet, never donor labels."""
    joints = joints_from_features(features)
    velocity = torch.diff(joints,dim=1)*30
    energy = velocity.square().sum(-1).mean((1,2))
    travel = torch.linalg.vector_norm(torch.diff(joints[:,:,0],dim=1)[...,[0,2]],dim=-1).sum(1)
    feet = joints[:,:,FEET]
    penetration = torch.quantile((-feet[...,1]).clamp_min(0).flatten(1),.95,dim=1)
    contacts = features[...,135:]> .5
    mask = contacts[:,1:] & contacts[:,:-1]
    foot_speed = torch.linalg.vector_norm(velocity[:,:,FEET],dim=-1)
    sliding = (foot_speed*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
    relative = joints-joints[:,:,:1]
    spectrum = torch.fft.rfft(relative-relative.mean(1,keepdim=True),dim=1).abs().square().sum((2,3))
    freq = torch.fft.rfftfreq(features.shape[1],1/30,device=features.device)
    high = spectrum[:,freq>6].sum(1)/spectrum[:,freq>0].sum(1).clamp_min(1e-12)
    frozen = (torch.linalg.vector_norm(velocity[:,:,[20,21,10,11]],dim=-1).mean(1)<.01).float().mean(1)
    values = {'kinetic':energy,'travel':travel,'penetration':penetration,'sliding':sliding,
              'contact_coverage':contacts.float().mean((1,2)),'high_frequency':high,'frozen_distal_fraction':frozen}
    return {k:float(v.mean()) for k,v in values.items()},joints


def persist_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path) as old:
            if sorted(old.files)!=sorted(arrays) or any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                raise ValueError('Partial inference replay differs')
        return
    temporary = path.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    temporary.replace(path)


def evaluate(arm,seed=0,checkpoint='final'):
    seed_all(0)
    torch.set_num_threads(4)
    model,p,manifest,audio_norms,norms,weights = load_model(arm,seed,checkpoint)
    output = ROOT/'music/evaluation'/arm/f'seed_{seed}'/checkpoint
    identity = {'arm':arm,'seed':seed,'checkpoint':checkpoint,'checkpoint_sha256':digest(weights),
                'protocol_sha256':digest(ROOT/'music/protocol.json')}
    if (output/'report.json').exists():
        prior = read_authenticated(output/'report.json')
        if prior['identity']!=identity:
            raise ValueError('Evaluation identity differs')
        return prior
    output.mkdir(parents=True,exist_ok=True)
    audio,motion = load_arrays(manifest,audio_norms)
    sm = torch.tensor(norms['summary']['mean'],device='cuda',dtype=torch.float32)
    ss = torch.tensor(norms['summary']['std'],device='cuda',dtype=torch.float32)
    rows = []
    for window in p['evaluation']['windows']:
        key,start,frames = window['key'],window['start'],window['frames']
        path = output/'windows'/f'{key}_{start}.json'
        if path.exists():
            row = read_authenticated(path)
            if row['identity']!=identity or digest(row['summary_path'])!=row['summary_sha256']:
                raise ValueError('Partial evaluation identity differs')
            rows.append(row)
            continue
        with torch.inference_mode():
            target = torch.from_numpy(motion[key][None,start:start+frames]).cuda()
            source_quality,source_joints = physical(target)
            target_summary = (motion_summary(target,source_joints)[0]-sm)/ss
            controls = {'correct':audio[key][start:start+frames],
                        'donor':audio[window['donor_key']][window['donor_start']:window['donor_start']+frames],
                        'shift':audio[key][window['shift_start']:window['shift_start']+frames]}
            draws = p['evaluation']['draws'] if arm=='flow' else 1
            noise = torch.cat([model.noise(1,frames,generator=torch.Generator(device='cuda').manual_seed(s))
                               for s in window['draw_seeds'][:draws]],dim=0)
            arrays = {'reference':target_summary.cpu().numpy()}
            scores,prediction_hashes = {},{}
            for condition,x in controls.items():
                x = torch.from_numpy(x[None]).cuda().expand(draws,-1,-1)
                prediction = model.generate(x,steps=p['evaluation']['steps'],initial_noise=noise,chunked=True)
                if not bool(prediction.isfinite().all()):
                    authenticated_json(output/'failure.json',{'identity':identity,'window':window,'condition':condition,'reason':'nonfinite native generation'})
                    raise FloatingPointError('Nonfinite generation; no sample discarded')
                if condition=='correct':
                    predicted_quality,joints = physical(prediction)
                else:
                    joints = joints_from_features(prediction)
                summary = (motion_summary(prediction,joints)-sm)/ss
                scores[condition] = energy_score(summary,target_summary)
                arrays[condition] = summary.cpu().numpy()
                prediction_hashes[condition] = hashlib.sha256(prediction.cpu().numpy().tobytes()).hexdigest()
        summary_path = path.with_suffix('.npz')
        persist_npz(summary_path,arrays)
        metrics = {**{k+'_es':v for k,v in scores.items()},
                   **{'source_'+k:v for k,v in source_quality.items()},
                   **{'prediction_'+k:v for k,v in predicted_quality.items()}}
        row = {**window,'identity':identity,'metrics':metrics,'effective_draws':draws,
               'summary_path':str(summary_path),'summary_sha256':digest(summary_path),
               'prediction_content_sha256':prediction_hashes}
        authenticated_json(path,row)
        rows.append(row)
        print(json.dumps({'arm':arm,'checkpoint':checkpoint,'key':key,'start':start,**scores}),flush=True)
    result = {'identity':identity,'rows':rows,'summary':aggregate(rows),'test_access':False,
              'direct_audio_lighting_enabled':False,'score_names':norms['summary']['names']}
    authenticated_json(output/'report.json',result)
    return result


def full_sequences(arm,seed=0):
    seed_all(0)
    torch.set_num_threads(4)
    model,p,manifest,audio_norms,norms,weights = load_model(arm,seed,'final')
    audio,motion = load_arrays(manifest,audio_norms)
    review = read_authenticated(SOURCE/'data/manifest.json')['review_clips']
    output = ROOT/'music/full_sequences'/arm/f'seed_{seed}'
    identity = {'arm':arm,'seed':seed,'checkpoint_sha256':digest(weights),'protocol_sha256':digest(ROOT/'music/protocol.json')}
    if (output/'report.json').exists():
        return read_authenticated(output/'report.json')
    rows = []
    for key in dict.fromkeys(c['key'] for c in review):
        path = output/f'{key}.npz'
        receipt = path.with_suffix('.json')
        if receipt.exists():
            row = read_authenticated(receipt)
            if row['identity']!=identity or digest(path)!=row['sha256']:
                raise ValueError('Full sequence replay identity differs')
            rows.append(row)
            continue
        with torch.inference_mode():
            x = torch.from_numpy(audio[key][None]).cuda()
            direct = model.generate(x,seed=7001,chunked=False)
            chunk = model.generate(x,seed=7001,chunked=True)
            error = float((direct-chunk).abs().max())
            if not bool(chunk.isfinite().all()):
                raise FloatingPointError('Nonfinite full-track generation')
            features = chunk[0].cpu().numpy()
        persist_npz(path,{'features':features})
        row = {'key':key,'path':str(path),'sha256':digest(path),'frames':len(features),'draw_seed':7001,
               'chunk_max_abs':error,'identity':identity}
        authenticated_json(receipt,row)
        rows.append(row)
        print(json.dumps({'arm':arm,'key':key,'full_frames':len(features),'chunk_max_abs':error}),flush=True)
    result = {'identity':identity,'rows':rows,'chunk_max_abs':max(r['chunk_max_abs'] for r in rows),
              'test_access':False,'root':'integrate once at consumer; full native feature sequences saved'}
    authenticated_json(output/'report.json',result)
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['evaluate','full'])
    p.add_argument('--arm',choices=['regression','flow'],default='flow')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--checkpoint',choices=['initial','final'],default='final')
    args = p.parse_args()
    with gpu_lock(ROOT):
        if args.stage=='evaluate':
            evaluate(args.arm,args.seed,args.checkpoint)
        else:
            full_sequences(args.arm,args.seed)

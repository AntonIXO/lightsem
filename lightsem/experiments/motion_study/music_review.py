"""Frozen visual harness including the incumbent and its integration control."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import ROOT as SOURCE
from .audio_prepare import ROOT
from .geometry import joints_from_features
from .music_render import render as render_movie
from ..night_common import BASE,ROOT as NIGHT
from ..night_evaluation import generate_groups,normalized_input
from ..beat_conditioned_model import BeatConditionedModel
from ..night_probes import PoseDecoder
from ..stage11_common import read_authenticated,authenticated_json,digest,freeze_sources,verify_sources,gpu_lock,seed_all


def freeze():
    out = ROOT/'music/review'
    if (out/'protocol.json').exists():
        return read_authenticated(out/'protocol.json')
    music = read_authenticated(ROOT/'music/protocol.json')
    verify_sources(ROOT/'music')
    read_authenticated(out/'integration_experiment.json')
    paths = [ROOT/'music/protocol.json',ROOT/'data/audio_manifest.json',SOURCE/'data/manifest.json',
             BASE/'run/final.pt',BASE/'normalizers.json',
             NIGHT/'diagnostics/pose_probes/descriptor256/seed_0/final.pt',out/'integration_experiment.json']
    for path in paths:
        if str(path) in music['inputs'] and digest(path)!=music['inputs'][str(path)]:
            raise ValueError('Historical input differs from music freeze')
    snapshot = freeze_sources(out)
    p = {'source_sha256':snapshot['source_sha256'],'inputs':{str(p):digest(p) for p in paths},
         'clips':read_authenticated(SOURCE/'data/manifest.json')['review_clips'],
         'incumbent':{'checkpoint':str(BASE/'run/final.pt'),'pose_decoder':str(paths[-2]),'seed':101,'steps':[8,32],
                      'cohort':'all full recordings underlying the12 fixed source clips; same audio and full noise for both integrations'},
         'generated':'read immutable full native features from the music experiment; rendering does not alter motion',
         'test_access':False,'new_training':False}
    authenticated_json(out/'protocol.json',p)
    return p


def context():
    out = ROOT/'music/review'
    p = read_authenticated(out/'protocol.json')
    snapshot = verify_sources(out)
    if not Path(__file__).resolve().is_relative_to(Path(snapshot['snapshot']).resolve()):
        raise ValueError('Use the frozen music review launcher')
    for path,sha in p['inputs'].items():
        if digest(path)!=sha:
            raise ValueError('Review input changed')
    return out,p


def pose_metrics(pose):
    velocity = np.diff(pose,axis=0)*40
    centered = pose-pose.mean(0)
    power = np.abs(np.fft.rfft(centered,axis=0))**2
    frequencies = np.fft.rfftfreq(len(pose),1/40)
    return {'high_frequency_fraction':float(power[frequencies>6].sum()/max(power[frequencies>0].sum(),1e-12)),
            'velocity_energy_tl2_s2':float((velocity*velocity).sum(-1).mean())}


def incumbent():
    out,p = context()
    seed_all(0)
    torch.set_num_threads(4)
    entries = {e['key']:e for e in read_authenticated(ROOT/'data/audio_manifest.json')['entries']}
    weights = torch.load(p['incumbent']['checkpoint'],map_location='cpu',weights_only=False)
    model = BeatConditionedModel().cuda().eval()
    model.load_state_dict(weights['model'])
    n = read_authenticated(BASE/'normalizers.json')
    pose_state = torch.load(p['incumbent']['pose_decoder'],map_location='cpu',weights_only=False)
    pose_model = PoseDecoder(pose_state['input_dim'],pose_state['width']).cuda().eval()
    pose_model.load_state_dict(pose_state['model'])
    mean,std = np.asarray(pose_state['pose_normalizers']['mean'],np.float32),np.asarray(pose_state['pose_normalizers']['std'],np.float32)
    rows = []
    for key in dict.fromkeys(c['key'] for c in p['clips']):
        e = entries[key]
        path = out/'incumbent'/f'{key}.npz'
        receipt = path.with_suffix('.json')
        if receipt.exists():
            row = read_authenticated(receipt)
            if digest(path)!=row['sha256'] or row['protocol_sha256']!=digest(out/'protocol.json'):
                raise ValueError('Incumbent replay differs')
            rows.append(row)
            continue
        if digest(e['base92_path'])!=e['base92_sha256']:
            raise ValueError('Raw audio cache changed')
        with np.load(e['base92_path']) as data:
            raw,times = data['x'],data['times']
        x = normalized_input(raw,np.zeros((len(raw),0),np.float32),n,None)
        values,metrics = {'times':times},{}
        for steps in p['incumbent']['steps']:
            descriptors,events = generate_groups(model,x,seeds=(p['incumbent']['seed'],),steps=steps)
            inp = np.concatenate((descriptors[0],events),axis=1)
            pred = (pose_model.predict(torch.from_numpy(inp[None]))[0].numpy()*std+mean).reshape(-1,17,2)
            values['pose'+str(steps)] = pred
            metrics[str(steps)] = pose_metrics(pred)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            with np.load(path) as old:
                if any(not np.array_equal(old[k],v) for k,v in values.items()):
                    raise ValueError('Interrupted baseline generation differs')
        else:
            np.savez_compressed(path,**values)
        row = {'key':key,'canonical_id':e['canonical_id'],'path':str(path),'sha256':digest(path),
               'metrics':metrics,'frames':len(raw),'protocol_sha256':digest(out/'protocol.json')}
        authenticated_json(receipt,row)
        rows.append(row)
        print(json.dumps({'key':key,'incumbent':metrics}),flush=True)
    result = {'rows':rows,'protocol_sha256':digest(out/'protocol.json'),'test_access':False}
    authenticated_json(out/'incumbent/report.json',result)
    return result


def render(seed=0):
    out,p = context()
    entries = {e['key']:e for e in read_authenticated(ROOT/'data/audio_manifest.json')['entries']}
    native = {}
    for arm in ('regression','flow'):
        report = read_authenticated(ROOT/'music/full_sequences'/arm/f'seed_{seed}/report.json')
        if report['identity']['protocol_sha256']!=digest(ROOT/'music/protocol.json'):
            raise ValueError('Native generated input belongs to another study')
        native[arm] = {r['key']:r for r in report['rows']}
    old = {r['key']:r for r in read_authenticated(out/'incumbent/report.json')['rows']}
    rows = []
    for clip in p['clips']:
        key = clip['key']
        e = entries[key]
        path = ROOT/'delivery/music'/f'seed_{seed}'/f'{clip["category"]}_{key}_{clip["start"]}.mp4'
        if path.with_suffix('.json').exists():
            prior = read_authenticated(path.with_suffix('.json'))
            if digest(path)!=prior['sha256']:
                raise ValueError('Review movie changed')
            rows.append(prior)
            continue
        sl = slice(clip['start'],clip['start']+clip['frames'])
        with np.load(e['path']) as source:
            panels = [('Source motion',source['joints'][sl],source['contacts'][sl])]
        for arm in ('regression','flow'):
            row = native[arm][key]
            if digest(row['path'])!=row['sha256']:
                raise ValueError('Generated full sequence changed')
            with np.load(row['path']) as data:
                features = data['features']
            with torch.inference_mode():
                joints = joints_from_features(torch.from_numpy(features)).numpy()
            panels.append((arm,joints[sl],features[sl,135:]))
        if digest(old[key]['path'])!=old[key]['sha256']:
            raise ValueError('Historical prediction changed')
        poses = []
        with np.load(old[key]['path']) as data:
            target = np.arange(clip['start'],clip['start']+clip['frames'])/30
            for steps in (8,32):
                pose = data['pose'+str(steps)].reshape(-1,34)
                poses.append(np.stack([np.interp(target,data['times'],pose[:,i]) for i in range(34)],axis=1).reshape(-1,17,2))
        rows.append(render_movie(path,panels,*poses,e['wav_path'],clip['start']/30,key+' / '+clip['category']))
        print(str(path),flush=True)
    result = {'rows':rows,'seed':seed,'review_protocol_sha256':digest(out/'protocol.json'),
              'music_protocol_sha256':digest(ROOT/'music/protocol.json'),
              'evaluations':{a:digest(ROOT/'music/evaluation'/a/f'seed_{seed}/final/report.json') for a in ('regression','flow')}}
    authenticated_json(out/f'render_report_seed_{seed}.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','incumbent','render'])
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    if args.stage=='freeze':
        freeze()
    else:
        with gpu_lock(ROOT):
            incumbent() if args.stage=='incumbent' else render(args.seed)

"""Fresh visual review when strict source-review equivalence cannot transfer."""
import argparse
from pathlib import Path

import numpy as np
import torch

from . import ROOT as SOURCE
from .audio_prepare import ROOT
from .native_codec import NativeCodec
from .geometry import joints_from_features
from .render import render_comparison
from ..stage11.stage11_common import read_authenticated, authenticated_json, digest, freeze_sources, verify_sources, gpu_lock


def freeze():
    out = ROOT/'codec/review'
    if (out/'protocol.json').exists():
        return read_authenticated(out/'protocol.json')
    audit = read_authenticated(ROOT/'codec/acceptance.json')
    if not all(v for k,v in audit['gates'].items() if k!='source_visual_equivalence'):
        raise ValueError('Fresh visual review cannot override numerical failure')
    snapshot = freeze_sources(out)
    p = {'source_sha256':snapshot['source_sha256'],
         'audit_sha256':digest(ROOT/'codec/acceptance.json'),
         'source_manifest_sha256':digest(SOURCE/'data/manifest.json'),
         'codec_sha256':digest(Path(__file__).with_name('native_codec.py')),
         'rule':'all numerical gates retained; 12 fresh source/native clip reviews replace only the failed review-transfer shortcut',
         'transfer_gate_remains_failed':True,'test_access':False}
    authenticated_json(out/'protocol.json',p)
    return p


def context():
    out = ROOT/'codec/review'
    p = read_authenticated(out/'protocol.json')
    snapshot = verify_sources(out)
    if not Path(__file__).resolve().is_relative_to(Path(snapshot['snapshot']).resolve()):
        raise ValueError('Fresh review requires its frozen launcher')
    if digest(ROOT/'codec/acceptance.json')!=p['audit_sha256'] or digest(SOURCE/'data/manifest.json')!=p['source_manifest_sha256']:
        raise ValueError('Inputs changed')
    return out,p


def render():
    out,p = context()
    manifest = read_authenticated(SOURCE/'data/manifest.json')
    entries = {e['key']:e for e in manifest['entries']}
    model = NativeCodec()
    rows = []
    for clip in manifest['review_clips']:
        e = entries[clip['key']]
        if digest(e['path'])!=e['sha256']:
            raise ValueError('Source changed')
        path = ROOT/'delivery/native139'/f'{clip["category"]}_{clip["key"]}_{clip["start"]}.mp4'
        if path.with_suffix('.json').exists():
            prior = read_authenticated(path.with_suffix('.json'))
            if digest(path)!=prior['sha256']:
                raise ValueError('Review movie changed')
            rows.append(prior)
            continue
        with np.load(e['path']) as source:
            raw = source['features']
            with torch.inference_mode():
                decoded = model(torch.from_numpy(raw[None]))[0]
                joints = joints_from_features(decoded).numpy()
            sl = slice(clip['start'],clip['start']+clip['frames'])
            panels = [('Source motion',source['joints'][sl],source['contacts'][sl]),
                      ('Native139 (uncompressed)',joints[sl],decoded.numpy()[sl,135:])]
        rows.append(render_comparison(path,panels,e['wav_path'],clip['start']/30,clip['key']+' / '+clip['category']))
        print(str(path),flush=True)
    authenticated_json(out/'report.json',{'rows':rows,'protocol_sha256':digest(out/'protocol.json')})


def select():
    out,p = context()
    render_report = read_authenticated(out/'report.json')
    visual = read_authenticated(out/'visual_review.json')
    if visual['render_report_sha256']!=digest(out/'report.json') or len(render_report['rows'])!=12:
        raise ValueError('Incomplete or mismatched review')
    audit = read_authenticated(ROOT/'codec/acceptance.json')
    gates = {k:v for k,v in audit['gates'].items() if k!='source_visual_equivalence'}
    gates['fresh_visual_review'] = visual['passed']
    result = {'gates':gates,'passed':all(gates.values()),'audio_training_authorized':all(gates.values()),
        'audit_sha256':digest(ROOT/'codec/acceptance.json'),'review_protocol_sha256':digest(out/'protocol.json'),
        'visual_review_sha256':digest(out/'visual_review.json'),'transfer_shortcut_passed':False,
        'model':audit['model'],'test_access':False,'interpretation':audit['interpretation']}
    authenticated_json(ROOT/'codec/final_acceptance.json',result)
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage',choices=['freeze','render','select'])
    args = p.parse_args()
    if args.stage=='freeze':
        freeze()
    elif args.stage=='render':
        with gpu_lock(ROOT):
            render()
    else:
        select()

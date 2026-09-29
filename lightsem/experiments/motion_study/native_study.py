"""Engineering acceptance of a full native reference after compression failures."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import ROOT as SOURCE
from .audio_prepare import ROOT
from .phase_registration import ROOT as PHASE
from .native_codec import NativeCodec
from .geometry import joints_from_features
from .metrics import compare
from .evaluate import aggregate
from .phase_study import gates
from ..stage11.stage11_common import read_authenticated, authenticated_json, digest, freeze_sources, verify_sources, gpu_lock


def freeze():
    out = ROOT/'codec'
    if (out/'protocol.json').exists():
        return read_authenticated(out/'protocol.json')
    failed = read_authenticated(PHASE/'codec/final_selection.json')
    if failed['eligible_arms']:
        raise ValueError('Native revision follows complete compression failure')
    read_authenticated(PHASE/'codec/vibescience_results.json')
    snapshot = freeze_sources(out)
    paths = [SOURCE/'data/manifest.json',SOURCE/'data/normalizers.json',SOURCE/'data/source_gate.json',
             SOURCE/'data/source_visual_review.json',PHASE/'codec/final_selection.json',PHASE/'codec/vibescience_results.json']
    p = {'source_sha256':snapshot['source_sha256'],'inputs':{str(p):digest(p) for p in paths},
         'gates':read_authenticated(PHASE/'codec/protocol.json')['gates'],
         'model':NativeCodec().config_dict(),'test_access':False,
         'type':'uncompressed engineering reference; no learned codec hypothesis, optimization or superiority claim',
         'visual_evidence':'reuse the 12 independently reviewed source clips only if native feature equality is exact and decoded joint equality is within5e-7m on every VAL frame',
         'selection':'sole explicit lossless reference; does not turn failed compression studies into passes'}
    authenticated_json(out/'protocol.json',p)
    return p


def audit():
    torch.set_num_threads(4)
    p = read_authenticated(ROOT/'codec/protocol.json')
    snapshot = verify_sources(ROOT/'codec')
    if not Path(__file__).resolve().is_relative_to(Path(snapshot['snapshot']).resolve()):
        raise ValueError('Use frozen native launcher')
    for path,sha in p['inputs'].items():
        if digest(path)!=sha:
            raise ValueError('Frozen native input changed')
    output = ROOT/'codec/acceptance.json'
    if output.exists():
        return read_authenticated(output)
    manifest = read_authenticated(SOURCE/'data/manifest.json')
    norms = read_authenticated(SOURCE/'data/normalizers.json')
    model = NativeCodec().cuda().eval()
    rows,sequences,quiet,stationary = [],[],[],[]
    feature_max,joint_max,chunk_max = 0.,0.,0.
    for entry in manifest['entries']:
        if entry['partition']!='val':
            continue
        if digest(entry['path'])!=entry['sha256']:
            raise ValueError('VAL tensor changed')
        with np.load(entry['path']) as data:
            reference = data['features']
            source_joints = data['joints']
        with torch.inference_mode():
            value = torch.from_numpy(reference[None]).cuda()
            decoded = model(value)
            chunks = model.forward_chunked(value)
            chunk_max = max(chunk_max,float((decoded-chunks).abs().max()))
            prediction = decoded[0].cpu().numpy()
            # CPU FP32 uses the same FK path as the source preparation.
            pred_joints = joints_from_features(torch.from_numpy(prediction)).numpy()
        feature_max = max(feature_max,float(np.max(np.abs(reference-prediction))))
        joint_max = max(joint_max,float(np.max(np.abs(source_joints-pred_joints))))
        sequences.append({'key':entry['key'],'canonical_id':entry['canonical_id'],
                          'metrics':compare(reference,prediction,norms['velocity_rms'])})
        for start in range(0,len(reference)-480+1,480):
            rows.append({'key':entry['key'],'canonical_id':entry['canonical_id'],'start':start,
                'metrics':compare(reference[start:start+480],prediction[start:start+480],norms['velocity_rms'])})
        for start in range(0,len(reference)-60+1,60):
            m = compare(reference[start:start+60],prediction[start:start+60],norms['velocity_rms'])
            if m['kinetic_reference']<.01:
                quiet.append({'key':entry['key'],'start':start,'speed_excess':float(max(0,np.sqrt(m['kinetic_prediction'])-np.sqrt(m['kinetic_reference'])))})
            if m['travel_reference_m']<.1:
                stationary.append({'key':entry['key'],'start':start,'travel_error':abs(m['travel_prediction_m']-m['travel_reference_m'])})
        print(json.dumps({'key':entry['key'],'feature_max':feature_max,'joint_max':joint_max}),flush=True)
    result = {'protocol_sha256':digest(ROOT/'codec/protocol.json'),'model':model.config_dict(),
              'rows':rows,'sequences':sequences,'window_summary':aggregate(rows),'full_summary':aggregate(sequences),
              'quiet_windows':quiet,'stationary_windows':stationary,'chunk_max_abs':chunk_max,
              'feature_max_abs':feature_max,'joint_max_abs':joint_max,'test_access':False}
    numerical = gates(result,p)
    source = read_authenticated(SOURCE/'data/source_gate.json')
    numerical.update(source_quality=source['passed'],integrity=True,
                     source_visual_equivalence=feature_max==0. and joint_max<=5e-7)
    result.update(gates=numerical,passed=all(numerical.values()),audio_training_authorized=all(numerical.values()),
        visual_review={'basis':p['visual_evidence'],'source_review_sha256':digest(SOURCE/'data/source_visual_review.json'),
                       'new_human_review':False},
        interpretation='Fidelity is expected from an uncompressed identity representation. This is engineering acceptance only; learned compression remains failed and audio conditioning remains untested.')
    authenticated_json(output,result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','audit'])
    args = parser.parse_args()
    if args.stage=='freeze':
        freeze()
    else:
        with gpu_lock(ROOT):
            audit()

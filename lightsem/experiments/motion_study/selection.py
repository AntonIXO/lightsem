"""Review packs and independently computed absolute gates; no automatic visual verdict."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import ROOT
from .geometry import FPS, joints_from_features
from .model import CodecConfig
from .render import render_comparison
from .training import load_context
from ..stage11.stage11_common import authenticated_json, read_authenticated, digest, gpu_lock


def numerical_gates(report, protocol):
    window, full = report['window_summary']['metrics'], report['full_summary']['metrics']
    both = (window, full)
    gates = protocol['gates']
    return {
        'joint_fidelity': all(m['joint_error_height'] <= gates['joint_error_height_max'] for m in both),
        'velocity_fidelity': all(m['velocity_nrmse'] <= gates['velocity_nrmse_max'] for m in both),
        'contacts': all(m['contact_f1'] >= gates['contact_f1_min'] for m in both),
        'travel_energy': all(.9 <= m['kinetic_ratio'] <= 1.1 and .9 <= m['travel_ratio'] <= 1.1 for m in both),
        'quiet_holds': all(m['quiet_speed_excess'] <= gates['quiet_speed_excess_max_m_s'] and
                           m['stationary_travel_error'] <= gates['stationary_travel_error_max_m'] for m in both),
        'continuity': all(m['penetration_excess'] <= gates['foot_penetration_excess_max_m'] and
                          m['sliding_excess'] <= gates['foot_sliding_excess_max_m_s'] for m in both)}


def render_reviews(root=ROOT, arms=('frame24', 'temporal64')):
    root = Path(root)
    protocol, manifest, _ = load_context(root)
    entries = {e['key']: e for e in manifest['entries']}
    reports = {arm: read_authenticated(root/'codec/evaluation'/arm/'seed_0/final/report.json') for arm in arms}
    indices = {arm: {r['key']: r for r in report['sequences']} for arm, report in reports.items()}
    if len(arms) > 2:
        raise ValueError('Source plus at most two models per review pack')
    output = root/'delivery'/('codec_'+'_'.join(arms))
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for clip in manifest['review_clips']:
        entry = entries[clip['key']]
        path = output/f'{clip["category"]}_{clip["key"]}_{clip["start"]}.mp4'
        if path.with_suffix('.json').exists():
            report = read_authenticated(path.with_suffix('.json'))
            if digest(path) != report['sha256']:
                raise ValueError('Codec review artifact changed')
            results.append(report)
            continue
        sl = slice(clip['start'], clip['start']+clip['frames'])
        with np.load(entry['path']) as value:
            source = value['joints'][sl]
            contact = value['contacts'][sl]
        models = [('Source motion', source, contact)]
        for arm in arms:
            row = indices[arm][entry['key']]
            if digest(row['path']) != row['sha256']:
                raise ValueError('Evaluation sequence changed')
            with np.load(row['path']) as value:
                features = value['features']
            # Integrate the entire saved trajectory ONCE, then slice the review.
            # Never hide accumulated root error by restarting each review clip.
            with torch.inference_mode():
                joints = joints_from_features(torch.from_numpy(features)).numpy()[sl]
            models.append((arm, joints, features[sl, 135:]))
        report = render_comparison(path, models, entry['wav_path'], clip['start']/FPS,
                                   clip['key']+' / '+clip['category'])
        results.append(report)
        print(json.dumps({'stage': 'codec_review', 'key': clip['key'], 'path': str(path)}), flush=True)
    result = {'rows': results, 'arms': list(arms), 'protocol_sha256': digest(root/'codec/protocol.json'),
              'evaluations': {arm: digest(root/'codec/evaluation'/arm/'seed_0/final/report.json') for arm in arms},
              'source_sha256': digest(__file__)}
    authenticated_json(output/'report.json', result)
    return result


def select(root=ROOT, include_fallback=False):
    root = Path(root)
    protocol, _, _ = load_context(root)
    arms = list(protocol['arms']) + (['temporal128'] if include_fallback else [])
    rows = []
    for arm in arms:
        path = root/'codec/evaluation'/arm/'seed_0/final/report.json'
        report = read_authenticated(path)
        if report['protocol_sha256'] != digest(root/'codec/protocol.json'):
            raise ValueError('Mismatched evaluation protocol')
        gates = numerical_gates(report, protocol)
        visual_path = root/'codec/reviews'/f'{arm}_seed_0.json'
        if visual_path.exists():
            visual = read_authenticated(visual_path)
            if visual['evaluation_sha256'] != digest(path):
                raise ValueError('Visual review refers to different predictions')
            gates['visual_review'] = bool(visual['passed'])
        else:
            gates['visual_review'] = False
        gates['source_quality'] = read_authenticated(root/'data/source_gate.json')['passed']
        cfg = CodecConfig(arm=arm)
        rows.append({'arm': arm, 'gates': gates, 'eligible': all(gates.values()),
                     'latent_values_per_second': cfg.latent*FPS/cfg.stride,
                     'evaluation_sha256': digest(path), 'window': report['window_summary']['metrics'],
                     'full': report['full_summary']['metrics']})
    eligible = sorted([r for r in rows if r['eligible']], key=lambda r: (r['latent_values_per_second'], r['window']['velocity_nrmse']))
    result = {'rows': rows, 'eligible_arms': [r['arm'] for r in eligible],
              'selected': eligible[0]['arm'] if eligible else None,
              'audio_training_authorized': bool(eligible), 'test_access': False,
              'protocol_sha256': digest(root/'codec/protocol.json')}
    authenticated_json(root/'codec'/('final_selection.json' if include_fallback else 'initial_selection.json'), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['render', 'select'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--arms', nargs='+', default=['frame24', 'temporal64'])
    parser.add_argument('--include-fallback', action='store_true')
    args = parser.parse_args()
    if args.stage == 'render':
        with gpu_lock(args.root):
            render_reviews(args.root, tuple(args.arms))
    else:
        print(json.dumps(select(args.root, args.include_fallback)), flush=True)

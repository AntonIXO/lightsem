"""Fixed-window and full-sequence codec evaluation; no test-set access."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import ROOT
from .geometry import FPS, joints_from_features
from .model import CodecConfig, MotionCodec
from .metrics import compare
from .training import load_context
from ..stage11_common import authenticated_json, read_authenticated, digest, gpu_lock

SCALARS = ('joint_error_height', 'velocity_nrmse', 'contact_f1', 'kinetic_reference',
           'kinetic_prediction', 'travel_reference_m', 'travel_prediction_m', 'acceleration_rmse')


def aggregate(rows):
    """Equal canonical song / recording / fixed window, never frame pseudoreplication."""
    recordings = defaultdict(list)
    for row in rows:
        recordings[(row['canonical_id'], row['key'])].append(row['metrics'])
    grouped = defaultdict(list)
    for (group, key), values in recordings.items():
        metrics = {name: float(np.mean([v[name] for v in values])) for name in SCALARS}
        active_energy = [v for v in values if v['kinetic_reference'] >= .01]
        active_travel = [v for v in values if v['travel_reference_m'] >= .1]
        metrics['active_energy_reference'] = float(np.mean([v['kinetic_reference'] for v in active_energy])) if active_energy else 0.
        metrics['active_energy_prediction'] = float(np.mean([v['kinetic_prediction'] for v in active_energy])) if active_energy else 0.
        metrics['active_travel_reference'] = float(np.mean([v['travel_reference_m'] for v in active_travel])) if active_travel else 0.
        metrics['active_travel_prediction'] = float(np.mean([v['travel_prediction_m'] for v in active_travel])) if active_travel else 0.
        quiet = [v for v in values if v['kinetic_reference'] < .01]
        stationary = [v for v in values if v['travel_reference_m'] < .1]
        metrics['quiet_speed_excess'] = max([max(0., np.sqrt(v['kinetic_prediction'])-np.sqrt(v['kinetic_reference'])) for v in quiet], default=0.)
        metrics['stationary_travel_error'] = max([abs(v['travel_prediction_m']-v['travel_reference_m']) for v in stationary], default=0.)
        metrics['penetration_excess'] = float(np.mean([max(0., v['physical_prediction']['penetration_p95_m']-v['physical_reference']['penetration_p95_m']) for v in values]))
        metrics['sliding_excess'] = float(np.mean([max(0., v['physical_prediction']['sliding_on_reference_contacts_m_s']-v['physical_reference']['sliding_on_reference_contacts_m_s']) for v in values]))
        grouped[group].append(metrics)
    per_song = {group: {name: float(np.mean([v[name] for v in values])) for name in values[0]} for group, values in grouped.items()}
    metrics = {name: float(np.mean([values[name] for values in per_song.values()])) for name in next(iter(per_song.values()))}
    metrics['kinetic_ratio'] = metrics['active_energy_prediction']/max(metrics['active_energy_reference'], 1e-8)
    metrics['travel_ratio'] = metrics['active_travel_prediction']/max(metrics['active_travel_reference'], 1e-8)
    return {'metrics': metrics, 'per_song': per_song}


def evaluate(arm, seed=0, checkpoint='final', root=ROOT):
    torch.set_num_threads(4)
    root = Path(root)
    protocol, manifest, norms = load_context(root)
    run = root/'codec/runs'/arm/f'seed_{seed}'
    weights = run/f'{checkpoint}.pt'
    folder = root/'codec/evaluation'/arm/f'seed_{seed}'/checkpoint
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'report.json').exists():
        result = read_authenticated(folder/'report.json')
        if result['checkpoint_sha256'] != digest(weights):
            raise ValueError('Evaluation weights changed')
        return result
    state = torch.load(weights, map_location='cpu', weights_only=False)
    if state['identity']['protocol_sha256'] != digest(root/'codec/protocol.json'):
        raise ValueError('Weights belong to another study')
    if checkpoint == 'final' and state['step'] != protocol['training']['steps']:
        raise ValueError('Only final scheduled checkpoint may qualify')
    model = MotionCodec(CodecConfig(arm=arm), norms).cuda().eval()
    model.load_state_dict(state['model'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    entries = [e for e in manifest['entries'] if e['partition']=='val']
    rows, sequences = [], []
    frames = protocol['evaluation']['frames']
    for entry in entries:
        if digest(entry['path']) != entry['sha256']:
            raise ValueError('Validation source changed')
        with np.load(entry['path']) as value:
            features = value['features']
        starts = list(range(0, len(features)-frames+1, frames))
        for batch_start in range(0, len(starts), 16):
            offsets = starts[batch_start:batch_start+16]
            target = np.stack([features[start:start+frames] for start in offsets])
            with torch.inference_mode():
                prediction = model(torch.from_numpy(target).cuda()).cpu().numpy()
            for start, reference, pred in zip(offsets, target, prediction):
                rows.append({'key': entry['key'], 'canonical_id': entry['canonical_id'], 'start': start,
                             'metrics': compare(reference, pred, norms['velocity_rms'])})
        with torch.inference_mode():
            full = model.forward_chunked(torch.from_numpy(features[None]).cuda()).cpu().numpy()[0]
        full_metrics = compare(features, full, norms['velocity_rms'])
        sequence = {'key': entry['key'], 'canonical_id': entry['canonical_id'], 'metrics': full_metrics}
        if checkpoint == 'final':
            destination = folder/f'{entry["key"]}.npz'
            if destination.exists():
                with np.load(destination) as prior:
                    if not np.array_equal(prior['features'], full):
                        raise ValueError('Partial evaluation replay differs')
            else:
                np.savez_compressed(destination, features=full)
            sequence.update(path=str(destination), sha256=digest(destination))
        sequences.append(sequence)
        print(json.dumps({'stage': 'codec_evaluate', 'arm': arm, 'checkpoint': checkpoint, 'key': entry['key'],
                          'full_joint_error_height': full_metrics['joint_error_height']}), flush=True)
    window_summary, full_summary = aggregate(rows), aggregate(sequences)
    result = {'arm': arm, 'seed': seed, 'checkpoint': checkpoint, 'checkpoint_sha256': digest(weights),
              'protocol_sha256': digest(root/'codec/protocol.json'), 'precision': 'fp32',
              'rows': rows, 'sequences': sequences, 'window_summary': window_summary,
              'full_summary': full_summary, 'selection_state': 'initialization_only' if checkpoint=='initial' else 'awaiting_all_arms_and_visual_review',
              'test_access': False}
    authenticated_json(folder/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--arm', choices=['frame24', 'temporal64', 'temporal128'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint', choices=['initial', 'final'], default='final')
    args = parser.parse_args()
    with gpu_lock(args.root):
        result = evaluate(args.arm, args.seed, args.checkpoint, args.root)
        print(json.dumps({'arm': args.arm, 'window': result['window_summary']['metrics'], 'full': result['full_summary']['metrics']}), flush=True)

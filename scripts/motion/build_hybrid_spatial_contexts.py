#!/usr/bin/env python3
"""Keep SpaIn and oBSRNN in separate coordinates for learned native fusion.

Raw authenticated 64-D streams are concatenated (SpaIn first), never averaged.
Training computes normalizers using paired TRAIN frames; the native planner
and flow each learn a projection of the resulting 128-D condition. No
showcase-specific normalization or feature fitting is permitted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'spain_obsrnn_concat128_v2'
BRANCHES = {'spain': [0, 64], 'obsrnn': [64, 128]}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_manifest(path, split):
    data = json.loads(Path(path).read_text())
    if data.get('test_access') is not False or data.get('status') != 'complete':
        raise ValueError(f'Unauthenticated manifest status/split: {path}')
    if 'aligned' in str(path) or data.get('normalization'):
        raise ValueError('Use raw method features, not per-song alignment')
    rows = data['rows']
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate ids')
    if any(r.get('split', data.get('split')) != split for r in rows):
        raise ValueError(f'Expected {split} rows: {path}')
    return data, {r['id']: r for r in rows}

def load_context(row):
    path = Path(row['context_path'])
    if sha(path) != row['context_sha256']:
        raise ValueError(f'Changed source context: {path}')
    with np.load(path, allow_pickle=False) as z:
        t, x = z['time30'].astype('f8'), z['fused30'].astype('f4')
    if x.shape != (len(t), 64) or len(t) < 2 or not np.isfinite(x).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f'Invalid context clock/values: {path}')
    return t, x

def concatenate_contexts(ts, xs, tb, xb):
    # Preserve the SpaIn native clock; BSRNN has 40 Hz frame-centred
    # observations. Endpoint extension is bounded to 100 ms, recorded below.
    if abs(ts[0]-tb[0]) > .1 or abs(ts[-1]-tb[-1]) > .1:
        raise ValueError('Method clocks cover different source intervals')
    b = np.stack([np.interp(ts, tb, xb[:, i]) for i in range(64)], 1)
    return ts, np.concatenate((xs, b), axis=1).astype('f4')

def build(train_spain, train_bsrnn, show_spain, show_bsrnn, out):
    if out.exists():
        raise FileExistsError(out)
    # Authenticate every input before writing any new context.
    prepared = []
    checkpoint_pairs = []
    for split, sp, bp in [('train', train_spain, train_bsrnn), ('showcase', show_spain, show_bsrnn)]:
        sm, sr = read_manifest(sp, split); bm, br = read_manifest(bp, split)
        if sr.keys() != br.keys():
            raise ValueError('Hybrid source id sets differ')
        checkpoint_pairs.append((sm['checkpoint_sha256'], bm['checkpoint_sha256']))
        entries = []
        for key in sorted(sr):
            a, b = sr[key], br[key]
            sa = a.get('audio_sha256', a.get('source_audio_sha256'))
            sb = b.get('audio_sha256', b.get('source_audio_sha256'))
            if sa is None or sa != sb:
                raise ValueError(f'Audio identity mismatch: {key}')
            t, x = concatenate_contexts(*load_context(a), *load_context(b))
            entries.append((key, t, x, sa, a, b))
        prepared.append((split, sp, bp, entries))
    if checkpoint_pairs[0] != checkpoint_pairs[1]:
        raise ValueError('TRAIN/showcase method checkpoints differ')
    out.mkdir(parents=True)
    for split, sp, bp, entries in prepared:
        folder = out / split; folder.mkdir(); rows = []
        for key, t, x, audio_hash, a, b in entries:
            path = folder / f'{key}.npz'
            np.savez_compressed(path, time30=t, fused30=x)
            rows.append({'id': key, 'split': split, 'context_path': str(path), 'context_sha256': sha(path),
                         'audio_sha256': audio_hash, 'feature_dim': 128, 'group': a.get('group'),
                         'source_hashes': [a['context_sha256'], b['context_sha256']]})
        payload = {'schema_version': 2, 'schema': SCHEMA, 'method': 'spain_obsrnn_learned_fusion',
                   'feature_dim': 128, 'branches': BRANCHES, 'normalization': 'native TRAIN stats only',
                   'status': 'complete', 'split': split, 'test_access': False, 'rows': rows,
                   'source_manifests': {str(p): sha(p) for p in [sp, bp]},
                   'method_checkpoint_sha256': list(checkpoint_pairs[0]), 'max_endpoint_extension_s': .1}
        name = 'CONTEXT_MANIFEST.json' if split == 'train' else 'SHOWCASE_CONTEXT_MANIFEST.json'
        (out/name).write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')

def main():
    p=argparse.ArgumentParser()
    for flag in ('train-spain','train-bsrnn','show-spain','show-bsrnn','out'):
        p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args();build(a.train_spain,a.train_bsrnn,a.show_spain,a.show_bsrnn,a.out)
    print('Built authenticated concat128 TRAIN and showcase contexts:', a.out)
if __name__=='__main__':main()

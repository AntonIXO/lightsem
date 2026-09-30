#!/usr/bin/env python3
"""CPU verification of every frozen fit, log, checkpoint and paired identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from lightsem.model.light_layers.training import load_checkpoint
from lightsem.model.light_layers.visual import sha256, write_json
from scripts.light_layers.train_visual_next import state_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Preserve the previous audit')
    queue = json.loads(args.queue.read_text())
    cells = queue['cells']
    expected = {(arm, seed) for arm in ('R0', 'R1', 'Rrand', 'R2', 'R3', 'R3-shuffle', 'R3-reverse') for seed in (0, 1, 2)}
    expected |= {(arm, 0) for arm in ('cut_only', 'mask_only', 'combined')}
    if len(cells) != 24 or {(r['arm'], r['seed']) for r in cells} != expected:
        raise RuntimeError('Wrong study membership')
    frozen = {}
    for cell in cells:
        path = cell['freeze']
        assert sha256(path) == cell['freeze_sha256'], path
        for source, digest in json.loads(Path(path).read_text())['files'].items():
            if source in frozen and frozen[source] != digest:
                raise RuntimeError('Conflicting frozen input identities')
            frozen[source] = digest
    for source, digest in frozen.items():
        assert sha256(source) == digest, source
    torch.set_num_threads(2)
    verified = []
    for cell in cells:
        directory = Path(cell['output'])
        r = json.loads((directory / 'REPORT.json').read_text())
        arm, seed = cell['arm'], cell['seed']
        assert r['status'] == 'complete' and r['steps'] == 6000
        assert (r['arm'], r['seed']) == (arm, seed)
        assert r['freeze_sha256'] == cell['freeze_sha256']
        assert r['initial_state_sha256'] == queue['initialization_sha256'][str(seed)]
        assert r['main_targets_sha256'] == cell['main_targets_sha256']
        assert r['samples_sha256'] == queue['sample_sequence_sha256'][str(seed)]
        assert not r['test_access'] and not r['DEV_access_during_fit']
        assert r['original_representation_fit_gate_pass'] is False
        assert r['parameters'] == 1284410 and r['train_families'] == 91 and r['precision'] == 'bf16'
        assert r['peak_reserved_bytes'] <= 10*2**30 and r['peak_rss_bytes'] <= 8*2**30
        assert r['elapsed_s'] <= queue['per_fit_max_seconds']
        assert sha256(directory / 'updates.jsonl') == r['updates_sha256']
        assert sha256(directory / 'final.pt') == r['final_weights_sha256']
        samples = hashlib.sha256()
        with (directory / 'updates.jsonl').open() as log:
            for step, line in enumerate(log, 1):
                update = json.loads(line)
                assert update['step'] == step and len(update['samples']) == 4
                assert update['lr'] == 3e-4*min(1., step/300)
                assert math.isfinite(update['gradient_norm'])
                assert all(math.isfinite(value) for value in update['loss'].values())
                samples.update(json.dumps(update['samples'], separators=(',', ':')).encode())
        assert step == 6000 and samples.hexdigest() == r['samples_sha256']
        model, metadata = load_checkpoint(directory / 'final.pt', device='cpu')
        assert (metadata['arm'], metadata['seed'], metadata['step']) == (arm, seed, 6000)
        assert metadata['main_targets_sha256'] == cell['main_targets_sha256']
        assert metadata['original_representation_fit_gate_pass'] is False
        assert all(torch.isfinite(tensor).all() for tensor in model.state_dict().values())
        assert state_digest(model) == r['final_state_sha256']
        verified.append({'arm': arm, 'seed': seed, 'REPORT_sha256': sha256(directory / 'REPORT.json'),
                         'checkpoint_sha256': r['final_weights_sha256'], 'steps': 6000,
                         'updates_sha256': r['updates_sha256'], 'elapsed_s': r['elapsed_s'],
                         'state_digest_verified': True, 'strict_weights_only_load': True})
        del model
        print('FULL_FIT_INDEPENDENTLY_VERIFIED', arm, seed, flush=True)
    for source, digest in frozen.items():
        assert sha256(source) == digest, source
    write_json(args.output, {'status': 'all24_full_fits_independently_verified', 'queue_sha256': sha256(args.queue),
                            'cells': verified, 'frozen_inputs_verified': len(frozen), 'DEV_scored': False,
                            'representation_qualification_pass': False, 'GPU_used': False,
                            'scope': 'Full membership, provenance, paired draws, finite training and strict checkpoint validation; no efficacy claim.'})


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Short, disposable TRAIN-only exercise of the actual H3 training route.

This is engineering evidence, not a study cell or a selectable checkpoint.
Inputs, seed, first 32 sampled batches and optimizer settings are fixed before
execution. No DEV or TEST input and no checkpoint selection are supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import torch

from lightsem.experiments.stage11.stage11_common import gpu_lock
from lightsem.model.light_layers.model import LightLayersConfig, build_light_layers
from lightsem.model.light_layers.training import BatchSampler, compute_loss, seed_all
from lightsem.model.light_layers.visual import sha256, write_json
from scripts.light_layers.train_visual_next import main_target_digest, prepared_data, state_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', required=True, type=Path)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    for path, digest in freeze['files'].items():
        if sha256(path) != digest:
            raise RuntimeError('Frozen input changed: ' + path)
    if freeze['arm'] != 'R0' or freeze['seed'] != 0 or freeze['steps'] != 32:
        raise RuntimeError('Engineering smoke requires R0, seed0, exactly32 updates')
    root = Path(freeze['output'])
    root.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    _, records, transforms, data = prepared_data(freeze, 'R0')
    if main_target_digest(data) != freeze['main_targets_sha256']:
        raise RuntimeError('Original TRAIN targets changed')
    seed_all(0, cuda=False)
    model = build_light_layers(LightLayersConfig(visual_aux_dim=96, pose_aux_dim=0), architecture='independent')
    initial = state_digest(model)
    if initial != freeze['initialization_sha256']['0']:
        raise RuntimeError('Paired initialization changed')
    sampler = BatchSampler(records, 0)
    dropout = torch.Generator(device='cpu').manual_seed(10000)
    draw_hash = hashlib.sha256()
    completed = 0
    start = None
    try:
        with gpu_lock():
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
            others = [p.strip() for p in active.splitlines() if p.strip() and p.strip() != str(os.getpid())]
            if others:
                raise RuntimeError('GPU is in use: ' + ','.join(others))
            start = time.monotonic()
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('CUDA BF16 is unavailable')
            torch.cuda.manual_seed_all(0)
            model.cuda().train()
            torch.cuda.reset_peak_memory_stats()
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            updates = []
            for step in range(1, 33):
                samples = sampler.sample(4)
                draw_hash.update(json.dumps(samples, separators=(',', ':')).encode())
                batch = data.batch(samples, 'cuda')
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = 3e-4 * min(1., step / 300)
                before = time.monotonic()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    losses, outputs = compute_loss(model, batch, transforms, state_dropout=.5, dropout_generator=dropout)
                if not all(torch.isfinite(value).item() for value in losses.values()):
                    raise RuntimeError('Nonfinite training loss')
                losses['total'].backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                auxiliary_gradient = sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                                         if 'visual_aux_head' in name and p.grad is not None)
                optimizer.step()
                torch.cuda.synchronize()
                completed = step
                updates.append({'step': step, 'seconds': time.monotonic() - before,
                                'losses': {key: float(value.detach()) for key, value in losses.items()},
                                'gradient_norm': float(gradient), 'visual_aux_gradient_l1': auxiliary_gradient})
                if time.monotonic() - start > freeze['max_gpu_seconds']:
                    raise RuntimeError('Smoke runtime allowance exceeded')
                if torch.cuda.max_memory_reserved() > 10 * 1024**3 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 8 * 1024**3:
                    raise RuntimeError('Smoke memory allowance exceeded')
                del batch, losses, outputs
            if not any(row['visual_aux_gradient_l1'] > 0 for row in updates):
                raise RuntimeError('Visual auxiliary head received no gradient')
            if draw_hash.hexdigest() != freeze['first32_samples_sha256']:
                raise RuntimeError('Frozen smoke data draws changed')
            final = state_digest(model)
            if final == initial:
                raise RuntimeError('Optimizer did not change the model')
            elapsed = time.monotonic() - start
            peak = torch.cuda.max_memory_reserved()
        for path, digest in freeze['files'].items():
            if sha256(path) != digest:
                raise RuntimeError('Frozen input changed: ' + path)
        write_json(root / 'REPORT.json', {
            'status': 'engineering_student_smoke_complete', 'freeze_sha256': sha256(args.freeze),
            'arm': 'R0', 'seed': 0, 'updates': updates, 'steps': completed,
            'initial_state_sha256': initial, 'final_discarded_state_sha256': final,
            'first32_sample_sequence_sha256': draw_hash.hexdigest(),
            'GPU_seconds': elapsed, 'peak_reserved_bytes': peak,
            'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'checkpoint_saved': False, 'DEV_access': False, 'TEST_access': False,
            'scope': 'Engineering finite-gradient/optimizer/BF16 check on real TRAIN data. Not a study fit or efficacy estimate; full matrix starts again from its fixed untrained states.',
        })
        print('STUDENT_SMOKE_COMPLETE', completed, elapsed, peak, flush=True)
    except BaseException as error:
        write_json(root / 'INTERRUPTED.json', {
            'status': 'engineering_smoke_incomplete', 'steps_completed': completed,
            'error': type(error).__name__ + ': ' + str(error),
            'GPU_seconds': None if start is None else time.monotonic() - start,
            'checkpoint_saved': False, 'DEV_access': False, 'TEST_access': False,
        })
        raise


if __name__ == '__main__':
    main()

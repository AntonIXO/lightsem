"""Engineering preflight, immutable codec protocol and sequential resumable training."""
import argparse
import gc
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import ROOT
from .model import CodecConfig, MotionCodec
from ..training_runtime import TrainingRuntime, BatchStream, backward_checked
from ..stage11.stage11_common import (authenticated_json, read_authenticated, atomic_json, atomic_torch,
    digest, json_digest, seed_all, rng_state, restore_rng, artifact_manifest, verify_artifacts,
    gpu_lock, require_space, freeze_sources, verify_sources)


class MotionResident:
    """Uniform song -> recording -> crop; resident real motion and source joints."""
    def __init__(self, manifest, frames=480):
        self.entries = [e for e in manifest['entries'] if e['partition']=='train' and e['n_frames']>=frames]
        self.host_aux, self.aux_dim = None, 0
        grouped = defaultdict(list)
        arrays, joints, starts, lengths = [], [], [], []
        cursor = 0
        for i, entry in enumerate(self.entries):
            if digest(entry['path']) != entry['sha256']:
                raise ValueError('Prepared training input changed')
            with np.load(entry['path']) as value:
                arrays.append(value['features'])
                joints.append(value['joints'])
            lengths.append(len(arrays[-1]))
            starts.append(cursor)
            cursor += lengths[-1]
            grouped[entry['canonical_id']].append(i)
        self.groups = [grouped[k] for k in sorted(grouped)]
        self.starts, self.lengths = np.asarray(starts), np.asarray(lengths)
        self.x = torch.from_numpy(np.concatenate(arrays)).cuda()
        self.y = torch.from_numpy(np.concatenate(joints)).cuda()
        self.b = torch.zeros((len(self.x), 1), device='cuda')
        if not self.groups:
            raise ValueError('No training crops')

    def draw_indices(self, size, frames, rng):
        selected = []
        for _ in range(size):
            group = self.groups[int(rng.integers(len(self.groups)))]
            selected.append(group[int(rng.integers(len(group)))])
        offsets = np.asarray([rng.integers(self.lengths[i]-frames+1) for i in selected])
        return self.starts[selected, None]+offsets[:, None]+np.arange(frames)[None]

    def batch(self, size, frames, rng):
        index = torch.as_tensor(self.draw_indices(size, frames, rng), device='cuda')
        return self.x[index], self.y[index], self.b[index]


def benchmark(root=ROOT):
    root = Path(root)
    report = root/'codec/engineering.json'
    if report.exists():
        return read_authenticated(report)
    manifest = read_authenticated(root/'data/manifest.json')
    normalizers = read_authenticated(root/'data/normalizers.json')
    data = MotionResident(manifest)
    rows = []
    runtime = TrainingRuntime()
    for arm in ('frame24', 'temporal64'):
        for batch in (16, 32):
            seed_all(0)
            runtime.apply()
            model = MotionCodec(CodecConfig(arm=arm), normalizers).cuda()
            optimizer = runtime.make_optimizer(model, 3e-4)
            fn = runtime.loss_function(model)
            rng = np.random.default_rng(10000)
            times = []
            torch.cuda.reset_peak_memory_stats()
            with BatchStream(data, batch, 480, rng, runtime.prefetch) as batches:
                for step in range(32):
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    x, y, b = batches.next()
                    optimizer.zero_grad(set_to_none=True)
                    with runtime.autocast():
                        losses = fn(x, y, b)
                    backward_checked(losses, model)
                    optimizer.step()
                    torch.cuda.synchronize()
                    times.append(time.monotonic()-begin)
                    if step == 0:
                        print(json.dumps({'stage': 'engineering_compile', 'arm': arm, 'batch': batch,
                                          'first_update_seconds': times[-1]}), flush=True)
            # Strict FP32 seam check; no scientific weights or validation metrics.
            model.eval()
            seed_all(0)
            sample = data.batch(1, 2200, np.random.default_rng(11000))[0]
            with torch.inference_mode():
                full, chunked = model(sample), model.forward_chunked(sample)
            error = float((full-chunked).abs().max())
            if error > 5e-5:
                raise RuntimeError(f'Codec chunk disagreement: {arm}: {error}')
            row = {'arm': arm, 'batch': batch, 'update_seconds_median': float(np.median(times[8:])),
                   'first_update_seconds': times[0], 'engineering_updates': 32,
                   'parameters': sum(p.numel() for p in model.parameters()), 'chunk_max_abs': error,
                   'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                   'runtime': runtime.as_dict(), 'scientific_weights_saved': False}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del model, optimizer, fn, x, y, b, losses, sample, full, chunked
            gc.collect()
            torch.cuda.empty_cache()
    # Choose by combined matched sample throughput, never by validation outcomes.
    scores = {batch: sum(r['update_seconds_median']/batch for r in rows if r['batch']==batch) for batch in (16, 32)}
    chosen = min(scores, key=lambda batch: (scores[batch], batch))
    result = {'rows': rows, 'selected_batch': chosen, 'selection': 'lowest sum(seconds / crops) across both initial arms',
              'source_sha256': digest(__file__), 'model_source_sha256': digest(Path(__file__).with_name('model.py')),
              'runtime_source_sha256': digest(Path(__file__).parents[1]/'training_runtime.py'),
              'manifest_sha256': digest(root/'data/manifest.json'), 'gpu': torch.cuda.get_device_name()}
    authenticated_json(report, result)
    return result


def freeze(root=ROOT):
    root = Path(root)
    output = root/'codec'
    protocol_path = output/'protocol.json'
    if protocol_path.exists():
        return read_authenticated(protocol_path)
    source_gate = read_authenticated(root/'data/source_gate.json')
    if not source_gate['passed']:
        raise RuntimeError('Source quality gate has not passed')
    engineering = read_authenticated(output/'engineering.json')
    if engineering['model_source_sha256'] != digest(Path(__file__).with_name('model.py')):
        raise RuntimeError('Model changed after engineering preflight')
    snapshot = freeze_sources(output)
    test = Path('/root/lightsem/tests/test_motion_study.py')
    destination = output/'source/tests/test_motion_study.py'
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(test, destination)
    destination.chmod(0o444)
    snapshot['files']['tests/test_motion_study.py'] = digest(test)
    snapshot['source_sha256'] = json_digest(snapshot['files'])
    authenticated_json(output/'frozen_sources.json', snapshot)
    inputs = [root/'data/manifest.json', root/'data/normalizers.json', root/'data/split.json',
              root/'data/source_gate.json', root/'data/duplicate_offset_audit.json', root/'codec/engineering.json',
              root/'vibescience_registration.json']
    protocol = {'frozen': True, 'source_sha256': snapshot['source_sha256'],
                'inputs': {str(path): digest(path) for path in inputs},
                'arms': ['frame24', 'temporal64'], 'fallback_arm': 'temporal128', 'seeds': [0],
                'training': {'steps': 20000, 'frames': 480, 'batch': engineering['selected_batch'],
                             'lr': 3e-4, 'warmup': 500, 'checkpoint_interval': 1000,
                             'runtime': TrainingRuntime().as_dict()},
                'evaluation': {'frames': 480, 'stride': 480, 'partition': 'val',
                               'selection': 'all non-overlapping16s windows; final scheduled checkpoint',
                               'bootstrap_groups': 'canonical_song', 'bootstrap_draws': 10000, 'bootstrap_seed': 20260910,
                               'full_sequences': 'all VAL recordings plus12 fixed source review clips'},
                'gates': {'joint_error_height_max': .02, 'velocity_nrmse_max': .15, 'contact_f1_min': .90,
                          'scope': 'fixed windows and full sequences must both pass; no root-drift hiding by resets',
                          'active_kinetic_ratio': [.9, 1.1], 'active_travel_ratio': [.9, 1.1],
                          'active_kinetic_reference_min': .01, 'active_travel_reference_min_m': .1,
                          'quiet_speed_excess_max_m_s': .02, 'stationary_travel_error_max_m': .034,
                          'foot_penetration_excess_max_m': .005, 'foot_sliding_excess_max_m_s': .03,
                          'chunk_max_abs': 5e-5, 'visual_review_required': True},
                'test_access': False, 'selection': 'lowest latent bitrate satisfying every gate',
                'training_seed_policy': 'seed1 only after complete pipeline finalist qualifies; not early exploration'}
    authenticated_json(protocol_path, protocol)
    return protocol


def load_context(root):
    root = Path(root)
    protocol = read_authenticated(root/'codec/protocol.json')
    snapshot = verify_sources(root/'codec')
    if snapshot['source_sha256'] != protocol['source_sha256']:
        raise ValueError('Frozen code identity mismatch')
    if not Path(__file__).resolve().is_relative_to(Path(snapshot['snapshot']).resolve()):
        raise RuntimeError('Scientific training must import the frozen codec/source tree, not live files')
    for path, sha in protocol['inputs'].items():
        if digest(path) != sha:
            raise ValueError('Frozen input changed: '+path)
    return protocol, read_authenticated(root/'data/manifest.json'), read_authenticated(root/'data/normalizers.json')


def train(arm, seed=0, root=ROOT, resume=False):
    root = Path(root)
    protocol, manifest, norms = load_context(root)
    if seed != 0:
        lock = read_authenticated(root/'confirmation_lock.json')
        if not lock['all_pipeline_gates_passed'] or arm not in lock['codec_arms']:
            raise RuntimeError('Second training seed is not yet qualified')
    if arm == 'temporal128':
        selection = read_authenticated(root/'codec/initial_selection.json')
        if selection['eligible_arms']:
            raise RuntimeError('Fallback width is only authorized when both initial codecs fail')
    elif arm not in protocol['arms']:
        raise ValueError('Unregistered arm')
    require_space(root, gb=3)
    run = root/'codec/runs'/arm/f'seed_{seed}'
    identity = {'arm': arm, 'seed': seed, 'protocol_sha256': digest(root/'codec/protocol.json'),
                'source_sha256': protocol['source_sha256'], 'training': protocol['training']}
    if (run/'report.json').exists():
        result = read_authenticated(run/'report.json')
        verify_artifacts(result['artifacts'])
        if result['identity'] != identity:
            raise ValueError('Completed cell has a different identity')
        return result
    existing = run.exists() and any(run.iterdir())
    if existing and not resume:
        raise RuntimeError('Partial run requires explicit --resume')
    run.mkdir(parents=True, exist_ok=True)
    config = protocol['training']
    seed_all(seed)
    runtime = TrainingRuntime.from_training(config)
    runtime.apply()
    data = MotionResident(manifest, config['frames'])
    model = MotionCodec(CodecConfig(arm=arm), norms).cuda()
    optimizer = runtime.make_optimizer(model, config['lr'])
    total, warmup = config['steps'], config['warmup']
    schedule = lambda step: min(1., (step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0, step-warmup)/max(1, total-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(10000+seed)
    start, elapsed = 0, 0.
    if existing:
        checkpoint = run/'latest.pt' if (run/'latest.pt').exists() else run/'initial.pt'
        state = torch.load(checkpoint, map_location='cuda', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Resume identity differs')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'], rng)
        start, elapsed = state['step'], state['training_seconds']
        if (run/'training.jsonl').exists():
            preserved = run/f'training_before_resume_{time.time_ns()}.jsonl'
            shutil.copy2(run/'training.jsonl', preserved)
            lines = [line for line in preserved.read_text().splitlines() if json.loads(line)['step']<=start]
            (run/'training.jsonl').write_text(''.join(line+'\n' for line in lines))
    def save(step, path):
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'rng': rng_state(rng),
                 'identity': identity, 'step': step, 'training_seconds': elapsed,
                 'model_config': model.config_dict(), 'normalizers': norms}
        atomic_torch(path, state)
        return state
    if not existing:
        save(0, run/'initial.pt')
    fn = runtime.loss_function(model)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    wall = interval = time.monotonic()
    with BatchStream(data, config['batch'], config['frames'], rng, runtime.prefetch) as batches, (run/'training.jsonl').open('a') as log:
        for step in range(start, total):
            x, y, b = batches.next()
            optimizer.zero_grad(set_to_none=True)
            with runtime.autocast():
                losses = fn(x, y, b)
            gradient = backward_checked(losses, model)
            optimizer.step()
            scheduler.step()
            if step == start or (step+1) % 100 == 0 or step+1 == total:
                torch.cuda.synchronize()
                now = time.monotonic()
                elapsed += now-interval
                interval = now
                row = {'arm': arm, 'seed': seed, 'step': step+1, 'training_seconds': elapsed,
                       'lr': scheduler.get_last_lr()[0], 'grad': float(gradient),
                       **{key: float(value.detach()) for key, value in losses.items()}}
                log.write(json.dumps(row)+'\n')
                log.flush()
                atomic_json(run/'status.json', row)
                atomic_json(root/'status.json', row)
                print(json.dumps(row), flush=True)
            if (step+1) % config['checkpoint_interval'] == 0:
                save(step+1, run/'latest.pt')
        final = save(total, run/'latest.pt')
        atomic_torch(run/'final.pt', final)
    result = {'identity': identity, 'model_config': model.config_dict(), 'steps': total,
              'training_seconds': elapsed, 'wall_seconds_this_invocation': time.monotonic()-wall,
              'parameters': sum(p.numel() for p in model.parameters()),
              'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30, 'gpu': torch.cuda.get_device_name(),
              'artifacts': artifact_manifest([run/'initial.pt', run/'final.pt', run/'latest.pt', run/'training.jsonl'])}
    authenticated_json(run/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['benchmark', 'freeze', 'train'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--arm', choices=['frame24', 'temporal64', 'temporal128'], default='frame24')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with gpu_lock(args.root):
        if args.stage == 'benchmark':
            benchmark(args.root)
        elif args.stage == 'freeze':
            freeze(args.root)
        else:
            train(args.arm, args.seed, args.root, args.resume)

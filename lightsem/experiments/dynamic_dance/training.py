"""Authenticated, resumable D6--D8 training and audio-only inference.

CLI owns both existing GPU locks. CPU cache preparation never opens TEST motion.
The performance preflight contains TRAIN-only throughput, not quality selection.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .learning import (DEFAULT_PHYSICAL, DynamicFlow, DynamicFlowConfig, EffortPlanner,
                       PLAN_CHANNELS, PlannerConfig, REGION_NAMES, drop_named_sources,
                       kinematic_plan_targets, plan_condition, planner_losses, pool_half_seconds,
                       unpack_plan)
from ..motion_study.music_model import MusicConfig
from ..motion_study.geometry import joints_from_features
from ..training_runtime import TrainingRuntime, BatchStream, backward_checked
from ..stage11.stage11_common import (atomic_json, atomic_torch, authenticated_json, digest,
    gpu_lock, json_digest, read_authenticated, require_space, restore_rng, rng_state, seed_all)

NATIVE_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
DEFAULT_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/dynamic_dance_20261005_v1')
INCUMBENT = NATIVE_ROOT / 'music/runs/flow/seed_0/final.pt'
INCUMBENT_SHA256 = '2cf7be614307a5e56e0e336e11a95d23f6139ee7a09ab4179845e9c6d0e3f213'
FLOW_ARMS = ('continued', 'D7', 'D8', 'D8_flat')


def implementation_hashes():
    here = Path(__file__)
    paths = [here, here.with_name('learning.py'),
             here.parents[3] / 'scripts/motion/train_dynamic_dance.py']
    old = here.parent.parent / 'motion_study'
    paths += [old / name for name in ('music_model.py', 'native_codec.py', 'geometry.py', 'music_physical.py', 'music_revision_model.py')]
    paths += [here.parent.parent / 'training_runtime.py', here.parent.parent / 'stage11/stage11_common.py']
    return {str(path): digest(path) for path in paths}


def verify_files(files):
    for path, expected in files.items():
        if digest(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _load_json(path):
    return json.loads(Path(path).read_text())


def prepare_cache(root=DEFAULT_ROOT, index=None):
    """Authenticate full TRAIN/VAL cache and fit canonical-equal TRAIN norms."""
    from .contracts import MusicTimeline
    from .audio import model_features
    root = Path(root)
    index = Path(index or root/'features/index.json')
    out = root/'learning/cache'
    out.mkdir(parents=True, exist_ok=True)
    identity = {'index_sha256': digest(index), 'implementation': implementation_hashes(),
                'audio_feature_source_sha256': digest(Path(__file__).with_name('audio.py')),
                'contracts_source_sha256': digest(Path(__file__).with_name('contracts.py')),
                'native_manifest_sha256': digest(NATIVE_ROOT/'data/audio_manifest.json')}
    if (out/'manifest.json').exists():
        result = read_authenticated(out/'manifest.json')
        if result['identity'] != identity:
            raise ValueError('Existing cache identity differs; preserve it and use a new root')
        verify_files({entry['cache_path']: entry['cache_sha256'] for entry in result['entries']})
        return result
    manifest = read_authenticated(NATIVE_ROOT/'data/audio_manifest.json')
    expected = {entry['key']: entry for entry in manifest['entries'] if entry['partition'] in ('train', 'val')}
    feature_index = _load_json(index)
    supplied = {entry['key']: entry for entry in feature_index['entries'] if entry.get('partition') in ('train', 'val')}
    if set(supplied) != set(expected):
        raise ValueError('Feature cache must contain all verified TRAIN/VAL recordings, including short ones')
    counts = Counter(entry['canonical_id'] for entry in expected.values() if entry['partition'] == 'train')
    rows, mean, second, feature_names = [], None, None, None
    for key in sorted(expected):
        require_space(root, gb=20)
        entry, cached = expected[key], supplied[key]
        for field in ('canonical_id', 'partition', 'n_frames'):
            if field in cached and cached[field] != entry[field]:
                raise ValueError('Feature index differs from paired manifest: ' + key)
        inputs = {entry['path']: entry['sha256'], entry['audio_path']: entry['audio_sha256'],
                  cached['timeline_path']: cached['timeline_sha256']}
        metadata_path = str(Path(cached['timeline_path']).with_suffix('.json'))
        inputs[metadata_path] = cached['timeline_metadata_sha256']
        verify_files(inputs)
        timeline = MusicTimeline.load(cached['timeline_path'])
        if timeline.metadata.get('input_sha256') != entry['wav_sha256'] or timeline.metadata.get('intervention') is not None:
            raise ValueError('Training timeline is not the original paired recording: ' + key)
        extra, names = model_features(timeline, target_time=np.arange(entry['n_frames'])/30.)
        names = list(names)
        if feature_names is not None and names != feature_names:
            raise ValueError('Changing audio feature schema')
        feature_names = names
        extra = np.asarray(extra, np.float32)
        if extra.shape != (entry['n_frames'], len(names)) or not np.isfinite(extra).all():
            raise ValueError('Invalid aligned model audio features')
        if np.max(np.abs(extra)) > np.finfo(np.float16).max:
            raise ValueError('Raw features exceed cache float16 range')
        with np.load(entry['path'], allow_pickle=False) as arrays:
            native = arrays['features'].copy()
        if native.shape != (entry['n_frames'], 139):
            raise ValueError('Native recording length differs')
        targets = kinematic_plan_targets(native)
        pooled, _ = pool_half_seconds(torch.from_numpy(extra)[None])
        path = out/'records'/f'{key}.npz'
        _atomic_npz(path, extra=extra.astype(np.float16), pooled=pooled[0].numpy(),
                    **{f'target_{name}': value.numpy() for name, value in targets.items()})
        row = {**entry, 'cache_path': str(path), 'cache_sha256': digest(path), 'inputs': inputs,
               'tempo_bpm': float(60/np.median(np.diff(timeline.beats))) if len(timeline.beats) > 1 else None}
        rows.append(row)
        if entry['partition'] == 'train':
            weight = 1/(len(counts)*counts[entry['canonical_id']])
            d = extra.astype(np.float64)
            if mean is None:
                mean, second = np.zeros(extra.shape[-1]), np.zeros(extra.shape[-1])
            mean += weight*d.mean(0)
            second += weight*np.square(d).mean(0)
    norms = {'mean': mean.tolist(), 'std': np.sqrt(np.maximum(second-mean*mean, 1e-6)).tolist(),
             'fit': 'equal canonical song / recording / real TRAIN frame; no VAL fit',
             'feature_names': feature_names, 'raw_cache_dtype': 'float16; pooled targets/features float32'}
    authenticated_json(out/'normalizers.json', norms)
    result = {'identity': identity, 'entries': rows, 'feature_names': feature_names,
              'normalizers_sha256': digest(out/'normalizers.json'), 'test_access': False,
              'coverage': {partition: {'recordings': sum(e['partition'] == partition for e in rows),
                           'canonical_songs': len({e['canonical_id'] for e in rows if e['partition'] == partition}),
                           'frames': sum(e['n_frames'] for e in rows if e['partition'] == partition)}
                           for partition in ('train', 'val')}}
    authenticated_json(out/'manifest.json', result)
    return result


class PlannerData:
    def __init__(self, manifest, norms, partition='train'):
        self.records = []
        grouped = defaultdict(list)
        mean, std = np.asarray(norms['mean'], np.float32), np.asarray(norms['std'], np.float32)
        for entry in sorted((e for e in manifest['entries'] if e['partition'] == partition), key=lambda e: e['key']):
            if digest(entry['cache_path']) != entry['cache_sha256']:
                raise ValueError('Planner cache changed')
            with np.load(entry['cache_path'], allow_pickle=False) as source:
                record = {'audio': ((source['pooled']-mean)/std).astype(np.float32),
                          'targets': {name[7:]: source[name].copy() for name in source.files if name.startswith('target_')},
                          'entry': entry}
            grouped[entry['canonical_id']].append(len(self.records))
            self.records.append(record)
        self.groups = [grouped[key] for key in sorted(grouped)]
        if not self.records:
            raise ValueError('No planner recordings in requested partition')
        # Fixed complete-recording shape avoids compile shape-selection artifacts.
        self.frames = max(len(record['audio']) for record in self.records)

    def batch(self, size, rng, device):
        selected = [int(rng.choice(self.groups[int(rng.integers(len(self.groups)))])) for _ in range(size)]
        records = [self.records[index] for index in selected]
        audio = np.zeros((size, self.frames, records[0]['audio'].shape[-1]), np.float32)
        valid = np.zeros((size, self.frames), bool)
        targets = {name: np.zeros((size, self.frames, *value.shape[1:]), value.dtype)
                   for name, value in records[0]['targets'].items()}
        for index, record in enumerate(records):
            length = len(record['audio'])
            audio[index, :length] = record['audio']
            valid[index, :length] = True
            for name in targets:
                targets[name][index, :length] = record['targets'][name]
        return (torch.from_numpy(audio).to(device), torch.from_numpy(valid).to(device),
                {name: torch.from_numpy(value).to(device) for name, value in targets.items()})


class FlowResident:
    """Canonical-equal crop sampling; every short recording retained and masked."""
    def __init__(self, manifest, base_norms, norms, frames, device, plan_paths=None):
        self.entries = sorted((e for e in manifest['entries'] if e['partition'] == 'train'), key=lambda e: e['key'])
        arrays, targets, masks, aux, starts, lengths = [], [], [], [], [], []
        grouped, cursor = defaultdict(list), 0
        mean, std = np.asarray(base_norms['mean'], np.float32), np.asarray(base_norms['std'], np.float32)
        self.aux_dim = len(norms['mean']) + PLAN_CHANNELS
        for index, entry in enumerate(self.entries):
            verify_files({entry['path']: entry['sha256'], entry['audio_path']: entry['audio_sha256'],
                          entry['cache_path']: entry['cache_sha256']})
            with np.load(entry['path'], allow_pickle=False) as source:
                motion = source['features'].copy()
            with np.load(entry['audio_path'], allow_pickle=False) as source:
                base = ((source['x']-mean)/std).astype(np.float32)
            with np.load(entry['cache_path'], allow_pickle=False) as source:
                extra = source['extra'].copy()
            length, capacity = len(motion), max(len(motion), frames)
            if base.shape != (length, 92) or extra.shape[0] != length:
                raise ValueError('Aligned cache lengths differ')
            x, y, a = np.zeros((capacity, 92), np.float32), np.zeros((capacity, 139), np.float32), np.zeros((capacity, self.aux_dim), np.float16)
            x[:length], y[:length], a[:length, :extra.shape[1]] = base, motion, extra
            if plan_paths is not None:
                plan_entry = plan_paths[entry['key']]
                verify_files({plan_entry['path']: plan_entry['sha256']})
                with np.load(plan_entry['path'], allow_pickle=False) as source:
                    plan = source['plan']
                    plan_time = source['time']
                target_time = np.arange(length)/30.
                a[:length, -PLAN_CHANNELS:] = np.stack([np.interp(target_time, plan_time, plan[:, i]) for i in range(PLAN_CHANNELS)], -1)
            arrays.append(x); targets.append(y); aux.append(a)
            masks.append(np.arange(capacity) < length)
            starts.append(cursor); lengths.append(length)
            cursor += capacity
            grouped[entry['canonical_id']].append(index)
        self.frames, self.device = frames, torch.device(device)
        self.groups = [grouped[key] for key in sorted(grouped)]
        self.starts, self.lengths = np.asarray(starts), np.asarray(lengths)
        self.x = torch.from_numpy(np.concatenate(arrays)).to(device)
        self.y = torch.from_numpy(np.concatenate(targets)).to(device)
        self.b = torch.from_numpy(np.concatenate(masks)).to(device)
        self.host_aux = np.concatenate(aux)
        self.aux_mean = torch.tensor(norms['mean']+[0.]*PLAN_CHANNELS, dtype=torch.float32, device=device)
        self.aux_std = torch.tensor(norms['std']+[1.]*PLAN_CHANNELS, dtype=torch.float32, device=device)
        self.coverage = {'recordings': len(self.entries), 'canonical_songs': len(self.groups),
                         'short_recordings': int((self.lengths < frames).sum()),
                         'real_frames': int(self.lengths.sum()), 'padding_frames': int(cursor-self.lengths.sum())}

    def draw_indices(self, size, frames, rng):
        if frames != self.frames:
            raise ValueError('Crop length changed after data initialization')
        selected = np.asarray([rng.choice(self.groups[int(rng.integers(len(self.groups)))]) for _ in range(size)])
        offsets = np.asarray([rng.integers(max(1, self.lengths[index]-frames+1)) for index in selected])
        return self.starts[selected, None]+offsets[:, None]+np.arange(frames)[None]

    def batch(self, size, frames, rng):
        ix = self.draw_indices(size, frames, rng)
        index = torch.as_tensor(ix, device=self.device)
        extra = torch.from_numpy(self.host_aux[ix].astype(np.float32)).to(self.device)
        return torch.cat((self.x[index], (extra-self.aux_mean)/self.aux_std), -1), self.y[index], self.b[index]


def _context(root):
    root = Path(root)
    manifest = read_authenticated(root/'learning/cache/manifest.json')
    norms = read_authenticated(root/'learning/cache/normalizers.json')
    if digest(root/'learning/cache/normalizers.json') != manifest['normalizers_sha256']:
        raise ValueError('Feature normalization changed')
    return manifest, norms, read_authenticated(NATIVE_ROOT/'data/audio_normalizers.json'), read_authenticated(NATIVE_ROOT/'data/motion_normalizers.json')


def _new_flow(arm, norms, extra_dim, device):
    if digest(INCUMBENT) != INCUMBENT_SHA256:
        raise ValueError('Pinned incumbent checkpoint changed')
    checkpoint = torch.load(INCUMBENT, map_location='cpu', weights_only=False)
    model = DynamicFlow(MusicConfig(), norms, DynamicFlowConfig(extra_dim=extra_dim, arm=arm))
    model.load_incumbent(checkpoint['model'])
    return model.to(device)


def _source_indices(names):
    sources = ('drums', 'bass', 'other', 'vocals')
    indices = [[92+i for i, name in enumerate(names) if source in str(name).lower()] for source in sources]
    if any(not group for group in indices):
        raise ValueError('Feature names must identify all four separated sources')
    return indices


def preflight(root=DEFAULT_ROOT, batches=(16, 32), updates=40):
    """TRAIN-only actual updates measure speed, then discard all pilot weights."""
    root = Path(root)
    out = root/'learning/preflight.json'
    if out.exists():
        return read_authenticated(out)
    require_space(root, gb=20)
    manifest, norms, base_norms, motion_norms = _context(root)
    runtime = TrainingRuntime()
    seed_all(0); runtime.apply(); torch.set_num_threads(4)
    rows = []
    for batch in batches:
        # The heaviest flow configuration determines a shared flow batch.
        torch.compiler.reset()
        model = _new_flow('D8', motion_norms, len(norms['mean']), 'cuda')
        data = FlowResident(manifest, base_norms, norms, 960, 'cuda')
        optimizer = runtime.make_optimizer(model, 3e-4)
        fn = model.loss_function(compile_vector_field=True)
        rng = np.random.default_rng(10000)
        torch.cuda.reset_peak_memory_stats()
        try:
            with BatchStream(data, batch, 960, rng, 2) as stream:
                started = time.monotonic()
                for step in range(updates+4):
                    x, y, valid = stream.next()
                    noise, t = model.draws(x)
                    optimizer.zero_grad(set_to_none=True)
                    with runtime.autocast():
                        losses = fn(x, y, valid, noise, t)
                    backward_checked(losses, model); optimizer.step()
                    if step == 3:
                        torch.cuda.synchronize(); compiled_seconds = time.monotonic()-started; started = time.monotonic()
                torch.cuda.synchronize()
                rows.append({'kind': 'flow', 'batch': batch, 'updates': updates,
                             'seconds_per_update': (time.monotonic()-started)/updates,
                             'compile_and_warmup_seconds': compiled_seconds,
                             'peak_gib': torch.cuda.max_memory_allocated()/2**30})
        finally:
            del optimizer, model, data
            torch.cuda.empty_cache()
    seed_all(0)
    data = PlannerData(manifest, norms)
    model = EffortPlanner(PlannerConfig(len(norms['mean']))).cuda()
    optimizer = runtime.make_optimizer(model, 3e-4)
    rng = np.random.default_rng(10000)
    started = time.monotonic()
    for step in range(updates+4):
        x, valid, targets = data.batch(4, rng, 'cuda')
        optimizer.zero_grad(set_to_none=True)
        with runtime.autocast():
            losses = planner_losses(model(x, valid), targets, valid)
        backward_checked(losses, model); optimizer.step()
        if step == 3:
            torch.cuda.synchronize(); started = time.monotonic()
    torch.cuda.synchronize()
    rows.append({'kind': 'planner', 'batch': 4, 'updates': updates,
                 'seconds_per_update': (time.monotonic()-started)/updates,
                 'full_song_bins': data.frames})
    selected = min((row for row in rows if row['kind'] == 'flow'), key=lambda row: row['seconds_per_update']/row['batch'])
    # 8 flow runs + 2 paired-branch planner runs; reserve 20% for initialization,
    # checkpointing, validation and slowdown. Exposure chosen before outcomes.
    per_step = 8*selected['seconds_per_update'] + 2*rows[-1]['seconds_per_update']
    steps = min(30000, int((12*3600*.8)/per_step)//1000*1000)
    if steps < 1000:
        raise RuntimeError('Insufficient training budget for a meaningful fixed campaign')
    result = {'source': implementation_hashes(), 'cache_sha256': digest(root/'learning/cache/manifest.json'),
              'measurements': rows, 'selected_batch': selected['batch'], 'planner_batch': 4,
              'selected_steps': steps, 'estimated_training_seconds': steps*per_step,
              'selection': 'throughput only; no held-out quality, no model selection', 'pilot_weights_retained': False}
    authenticated_json(out, result)
    return result


def evaluation_windows(manifest):
    """One fixed contiguous window per exposed VAL recording, including shorts."""
    entries = sorted((entry for entry in manifest['entries'] if entry['partition'] == 'val'), key=lambda e: e['key'])
    windows = []
    for entry in entries:
        frames = min(960, entry['n_frames'])
        start = ((entry['n_frames']-frames)//2)//15*15
        candidates = [donor for donor in entries if donor['canonical_id'] != entry['canonical_id'] and donor['n_frames'] >= frames]
        if not candidates:
            raise ValueError('No canonical-distinct contiguous VAL donor')
        def difference(donor):
            a, b = entry.get('tempo_bpm'), donor.get('tempo_bpm')
            return abs(math.log(a/b)) if a and b else float('inf')
        donor = min(candidates, key=lambda value: (difference(value), value['key']))
        shift = max((0, entry['n_frames']-frames), key=lambda value: abs(value-start))
        windows.append({'key': entry['key'], 'canonical_id': entry['canonical_id'], 'start': start, 'frames': frames,
                        'donor_key': donor['key'], 'donor_start': ((donor['n_frames']-frames)//2)//15*15,
                        'donor_log_bpm_distance': difference(donor) if math.isfinite(difference(donor)) else None,
                        'similar_bpm': difference(donor) <= math.log(1.1),
                        'shift_start': shift if abs(shift-start) >= 150 else None,
                        'draw_seed': 7001})
    return windows


def freeze_protocol(root=DEFAULT_ROOT):
    root = Path(root)
    path = root/'learning/protocol.json'
    pre = read_authenticated(root/'learning/preflight.json')
    if pre['source'] != implementation_hashes():
        raise ValueError('Implementation changed since performance preflight')
    manifest, norms, _, _ = _context(root)
    if digest(root/'learning/cache/manifest.json') != pre['cache_sha256']:
        raise ValueError('Cache changed since preflight')
    inputs = {str(INCUMBENT): INCUMBENT_SHA256}
    for p in (root/'learning/cache/manifest.json', root/'learning/cache/normalizers.json',
              root/'learning/preflight.json', NATIVE_ROOT/'data/motion_normalizers.json',
              NATIVE_ROOT/'data/audio_normalizers.json'):
        inputs[str(p)] = digest(p)
    protocol = {'inputs': inputs, 'sources': implementation_hashes(), 'seeds': [0, 1],
                'arms': ['planner', *FLOW_ARMS], 'display_seed': 0, 'generation_seed': 7001,
                'training': {'steps': pre['selected_steps'], 'batch': pre['selected_batch'],
                             'planner_batch': pre['planner_batch'], 'frames': 960,
                             'lr': 3e-4, 'warmup': 500, 'checkpoint_interval': 1000,
                             'runtime': TrainingRuntime().as_dict()},
                'physical_weights': asdict(DEFAULT_PHYSICAL), 'extra_dim': len(norms['mean']),
                'source_indices': _source_indices(manifest['feature_names']),
                'planner': EffortPlanner(PlannerConfig(len(norms['mean']))).config_dict(),
                'planner_train': 'equal canonical/full-recording; local+global objectives equally weighted',
                'D6': 'local planner applied to each song own accepted body, endpoint-bounded retiming',
                'D8': 'same planner global branch frozen before decoder fitting',
                'D8_flat': 'same planner local branch frozen before decoder fitting',
                'evaluation': {'windows': evaluation_windows(manifest),
                    'score': 'primary masked native flow noise-target MSE at fixed t=.1,.5,.9; secondary single-draw TRAIN-standardized18channel physical summary RMSE',
                    'flow_loss_times': [.1, .5, .9], 'minimum_relative_improvement': .05,
                    'paired_conditions': ['correct', 'donor', 'shift'], 'planner_control': 'TRAIN-only logRMS linear model',
                    'donor': 'canonical-distinct nearest estimated BPM, no tempo warping',
                    'shift': 'farthest nonwrapped contiguous window at least5s distant; unavailable short cases reported',
                    'bootstrap_draws': 2000, 'bootstrap_seed': 20261005, 'no_model_selection': True},
                'test_access': False, 'training_budget_seconds': 43200}
    if path.exists():
        if read_authenticated(path) != protocol:
            raise ValueError('Refusing to replace a different frozen learning protocol')
    else:
        authenticated_json(path, protocol)
    return protocol


def _protocol(root):
    result = read_authenticated(Path(root)/'learning/protocol.json')
    verify_files(result['inputs']); verify_files(result['sources'])
    return result


def export_training_plans(root, seed):
    """Frozen planner sees audio only; export once for both matched decoders."""
    root = Path(root)
    checkpoint_path = root/f'learning/runs/planner/seed_{seed}/final.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = EffortPlanner(PlannerConfig(**checkpoint['planner_config'])).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    manifest, norms, _, _ = _context(root)
    for branch in ('local', 'global'):
        destination = root/f'learning/plans/seed_{seed}/{branch}'
        existing = destination/'index.json'
        identity = {'checkpoint_sha256': digest(checkpoint_path), 'cache_sha256': digest(root/'learning/cache/manifest.json'), 'branch': branch}
        if existing.exists():
            index = read_authenticated(existing)
            if index['identity'] != identity:
                raise ValueError('Plan cache identity changed')
            verify_files({row['path']: row['sha256'] for row in index['entries'].values()})
            continue
        rows = {}
        for partition in ('train', 'val'):
            data = PlannerData(manifest, norms, partition)
            for record in data.records:
                x = torch.from_numpy(record['audio'])[None].cuda()
                valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
                with torch.inference_mode():
                    value = plan_condition(model(x, valid)[branch])[0].cpu().numpy()
                path = destination/f'{record["entry"]["key"]}.npz'
                _atomic_npz(path, time=np.arange(len(value))*.5, plan=value)
                rows[record['entry']['key']] = {'path': str(path), 'sha256': digest(path)}
        authenticated_json(existing, {'identity': identity, 'entries': rows})


def _run_identity(root, arm, seed):
    root = Path(root)
    identity = {'arm': arm, 'seed': seed, 'protocol_sha256': digest(root/'learning/protocol.json')}
    if arm in ('D8', 'D8_flat'):
        branch = 'global' if arm == 'D8' else 'local'
        identity.update(planner_sha256=digest(root/f'learning/runs/planner/seed_{seed}/final.pt'),
                        plan_index_sha256=digest(root/f'learning/plans/seed_{seed}/{branch}/index.json'))
    return identity


def train(root=DEFAULT_ROOT, arm='planner', seed=0, resume=False, until=None):
    root = Path(root)
    protocol = _protocol(root)
    if arm not in protocol['arms'] or seed not in protocol['seeds']:
        raise ValueError('Unregistered arm or seed')
    require_space(root, gb=20)
    config = protocol['training']
    total = config['steps']
    until = total if until is None else until
    if until > total or until < 1 or (until != total and until % 1000):
        raise ValueError('Stops only at registered final step or 1000-update checkpoint boundaries')
    run = root/f'learning/runs/{arm}/seed_{seed}'
    identity = _run_identity(root, arm, seed)
    if (run/'report.json').exists():
        result = read_authenticated(run/'report.json')
        if result['identity'] != identity:
            raise ValueError('Completed run identity changed')
        verify_files(result['artifacts'])
        return result
    existed = run.exists() and any(run.iterdir())
    if existed and not resume:
        raise ValueError('Partial run requires --resume; previous artifacts are preserved')
    run.mkdir(parents=True, exist_ok=True)
    seed_all(seed); torch.set_num_threads(4)
    runtime = TrainingRuntime(**config['runtime']); runtime.apply()
    manifest, norms, base_norms, motion_norms = _context(root)
    if arm == 'planner':
        model = EffortPlanner(PlannerConfig(protocol['extra_dim'])).cuda()
        data = PlannerData(manifest, norms)
    else:
        plan_paths = None
        if arm in ('D8', 'D8_flat'):
            branch = 'global' if arm == 'D8' else 'local'
            plan_index = read_authenticated(root/f'learning/plans/seed_{seed}/{branch}/index.json')
            checkpoint = root/f'learning/runs/planner/seed_{seed}/final.pt'
            if plan_index['identity']['checkpoint_sha256'] != digest(checkpoint):
                raise ValueError('Plan cache differs from frozen final planner')
            identity = {**identity, 'planner_sha256': digest(checkpoint),
                        'plan_index_sha256': digest(root/f'learning/plans/seed_{seed}/{branch}/index.json')}
            plan_paths = plan_index['entries']
        model = _new_flow(arm, motion_norms, protocol['extra_dim'], 'cuda')
        data = FlowResident(manifest, base_norms, norms, config['frames'], 'cuda', plan_paths)
    optimizer = runtime.make_optimizer(model, config['lr'])
    warmup = config['warmup']
    schedule = lambda step: min(1., (step+1)/warmup)*(.1+.9*.5*(1+math.cos(math.pi*max(0, step-warmup)/max(1, total-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    rng = np.random.default_rng(10000+seed)
    # Model architecture must never consume paired noise/dropout RNG cursors.
    draw_generator = torch.Generator(device='cuda').manual_seed(20000+seed)
    drop_generator = torch.Generator(device='cuda').manual_seed(30000+seed)
    start, elapsed = 0, 0.
    if existed:
        checkpoint_path = run/'latest.pt' if (run/'latest.pt').exists() else run/'initial.pt'
        checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        if checkpoint['identity'] != identity:
            raise ValueError('Resume identity differs')
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); restore_rng(checkpoint['rng'], rng)
        draw_generator.set_state(checkpoint['draw_rng'].cpu()); drop_generator.set_state(checkpoint['drop_rng'].cpu())
        start, elapsed = checkpoint['step'], checkpoint['training_seconds']
        if (run/'training.jsonl').exists():
            prior = run/f'training_before_resume_{time.time_ns()}.jsonl'
            shutil.copy2(run/'training.jsonl', prior)
            (run/'training.jsonl').write_text(''.join(line+'\n' for line in prior.read_text().splitlines() if json.loads(line)['step'] <= start))
    if until <= start:
        return {'already_at_step': start, 'identity': identity}
    def save(step, path):
        state = {'identity': identity, 'step': step, 'training_seconds': elapsed,
                 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'rng': rng_state(rng), 'draw_rng': draw_generator.get_state(), 'drop_rng': drop_generator.get_state(),
                 'model_config': model.config_dict(), 'normalizers': motion_norms, 'audio_normalizers': base_norms,
                 'extra_normalizers': norms, 'feature_names': manifest['feature_names']}
        if arm == 'planner':
            state['planner_config'] = asdict(model.config)
        atomic_torch(path, state)
        return state
    if not existed:
        save(0, run/'initial.pt')
    torch.compiler.reset()
    fn = None if arm == 'planner' else model.loss_function(compile_vector_field=runtime.compile == 'default')
    model.train(); torch.cuda.reset_peak_memory_stats()
    wall = interval = time.monotonic()
    stream = None if arm == 'planner' else BatchStream(data, config['batch'], config['frames'], rng, runtime.prefetch)
    try:
        with (run/'training.jsonl').open('a') as log:
            for step in range(start, until):
                optimizer.zero_grad(set_to_none=True)
                if arm == 'planner':
                    x, valid, targets = data.batch(config['planner_batch'], rng, 'cuda')
                    with runtime.autocast():
                        losses = planner_losses(model(x, valid), targets, valid)
                else:
                    x, target, valid = stream.next()
                    noise = model.noise(len(x), x.shape[1], generator=draw_generator)
                    t = torch.rand(len(x), generator=draw_generator, device=x.device)
                    keep = torch.rand((len(x), 4), generator=drop_generator, device=x.device) >= model.dynamic.source_dropout
                    x = drop_named_sources(x, protocol['source_indices'], keep)
                    with runtime.autocast():
                        losses = fn(x, target, valid, noise, t)
                gradient = backward_checked(losses, model); optimizer.step(); scheduler.step()
                if step == start or (step+1) % 100 == 0 or step+1 == until:
                    torch.cuda.synchronize(); now = time.monotonic(); elapsed += now-interval; interval = now
                    row = {'arm': arm, 'seed': seed, 'step': step+1, 'training_seconds': elapsed,
                           'valid_frames': int(valid.sum()), 'gradient': float(gradient),
                           'lr': scheduler.get_last_lr()[0], **{key: float(value.detach()) for key, value in losses.items()}}
                    log.write(json.dumps(row)+'\n'); log.flush(); atomic_json(run/'status.json', row)
                    print(json.dumps(row), flush=True)
                if (step+1) % config['checkpoint_interval'] == 0:
                    require_space(root, gb=20); save(step+1, run/'latest.pt')
            state = save(until, run/'latest.pt')
            if until == total:
                atomic_torch(run/'final.pt', state)
                result = {'identity': identity, 'steps': total, 'training_seconds': elapsed,
                          'parameters': sum(p.numel() for p in model.parameters()),
                          'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
                          'model_config': model.config_dict(), 'test_access': False,
                          'artifacts': {str(path): digest(path) for path in (run/'initial.pt', run/'final.pt', run/'training.jsonl')}}
                authenticated_json(run/'report.json', result)
                return result
            return {'identity': identity, 'step': until, 'partial': True}
    except BaseException as error:
        atomic_json(run/f'failure_{time.time_ns()}.json', {'identity': identity, 'type': type(error).__name__, 'message': str(error),
                    'resume_from': str(run/'latest.pt'), 'artifacts_preserved': True})
        raise
    finally:
        if stream is not None:
            stream.close()
        atomic_json(run/f'segment_{start}_{until}_{time.time_ns()}.json', {'identity': identity, 'requested_start': start,
                    'requested_end': until, 'wall_seconds': time.monotonic()-wall})


def load_learned(root, arm, seed=0, device='cuda'):
    root = Path(root)
    _protocol(root)
    path = root/f'learning/runs/{arm}/seed_{seed}/final.pt'
    report = read_authenticated(path.parent/'report.json')
    verify_files(report['artifacts'])
    state = torch.load(path, map_location='cpu', weights_only=False)
    if arm == 'planner':
        model = EffortPlanner(PlannerConfig(**state['planner_config']))
    else:
        native = state['model_config']
        cfg = MusicConfig(**{name: native[name] for name in ('arm', 'width', 'frames', 'dilations', 'steps')})
        model = DynamicFlow(cfg, state['normalizers'], DynamicFlowConfig(**native['dynamic']))
    model.load_state_dict(state['model'])
    return model.to(device).eval(), state, path


def planner_predict(model, raw_audio, norms, device='cuda'):
    raw_audio = torch.as_tensor(raw_audio, dtype=torch.float32)
    if raw_audio.ndim != 2 or not torch.isfinite(raw_audio).all():
        raise ValueError('Expected finite T,F audio')
    pooled, valid = pool_half_seconds(raw_audio[None])
    mean, std = (torch.tensor(norms[key], dtype=torch.float32) for key in ('mean', 'std'))
    x = ((pooled-mean)/std).to(device)
    with torch.inference_mode():
        predictions = model(x, valid.to(device))
    return {key: value[0].cpu() for key, value in predictions.items()}


def validate_audio_binding(timeline, base92_path):
    """Both learned audio inputs must come from the identical analyzed waveform.

    Hashing unrelated arrays separately cannot establish a paired generation.
    Waveform interventions therefore require BOTH full audio paths to rerun.
    """
    base92_path = Path(base92_path)
    receipt = read_authenticated(base92_path.with_suffix('.json'))
    if receipt.get('kind') != 'Base92' or receipt.get('npz_sha256') != digest(base92_path):
        raise ValueError('Unauthenticated or changed Base92 inference features')
    waveform = timeline.metadata.get('analyzed_waveform_sha256')
    if not isinstance(waveform, str) or len(waveform) != 64 or waveform != receipt.get('analyzed_waveform_sha256'):
        raise ValueError('Base92 and MusicTimeline analyze different or unknown waveforms')
    if timeline.metadata.get('intervention') != receipt.get('intervention'):
        raise ValueError('Base92 and MusicTimeline intervention provenance differs')
    declared = timeline.metadata.get('base92', {}).get('npz_sha256', timeline.metadata.get('base92_sha256'))
    if declared is not None and declared != digest(base92_path):
        raise ValueError('Base92 differs from the timeline paired-feature receipt')
    return receipt


def timeline_provenance(timeline, timeline_path):
    """Persist resolvable music observations and their exact source identity."""
    timeline_path = Path(timeline_path).resolve()
    metadata = timeline.metadata
    return {'timeline_path': str(timeline_path), 'timeline_sha256': digest(timeline_path),
            'timeline_metadata_sha256': digest(timeline_path.with_suffix('.json')),
            'source_key': metadata.get('key', timeline_path.stem),
            'audio_input_path': metadata.get('input_path'),
            'audio_input_sha256': metadata.get('input_sha256'),
            'analyzed_waveform_sha256': metadata.get('analyzed_waveform_sha256'),
            # Clone JSON provenance so later caller mutations cannot change it.
            'audio_intervention': json.loads(json.dumps(metadata.get('intervention')))}


def observed_regional_speed(world, fps=30.):
    """D6 denominator using the same interval alignment as its proxy targets.

    Frame i>0 uses displacement from i-1 to i; frame0 repeats the first
    observed interval. A single pose has no observed speed and returns zero.
    Centered differences are prohibited because alternating motion cancels.
    """
    from .learning import REGION_JOINTS
    world = np.asarray(world, dtype=np.float64)
    if world.ndim != 3 or world.shape[1:] != (22, 3) or len(world) < 1 or not np.isfinite(world).all():
        raise ValueError('Expected finite nonempty [T,22,3] body')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Expected positive finite frame rate')
    intervals = np.linalg.norm(np.diff(world, axis=0)*fps, axis=-1)
    speed = np.concatenate((intervals[:1], intervals), axis=0) if len(world) > 1 else np.zeros((1, 22))
    return np.stack([speed[:, list(indices)].mean(-1) for indices in REGION_JOINTS], -1)


def save_inference_plan_telemetry(path, time, plan, *, arm, source_evidence, coarse_time=None, coarse_plan=None):
    """Write exact already-used conditioning values; never mutate generation state.

    D8/D8_flat carry learned planning probabilities. Other flow arms retain
    unused zero slots only; these are explicitly not planning decisions.
    """
    path = Path(path).resolve()
    metadata_path = path.with_suffix('.json')
    if path.exists() or metadata_path.exists():
        raise FileExistsError('Refuse to overwrite inference-plan telemetry: '+str(path))
    active = arm in ('D8', 'D8_flat')
    if arm not in FLOW_ARMS:
        raise ValueError('Inference-plan telemetry requires a registered flow arm')
    arrays = {}
    def capture(prefix, timestamps, values):
        timestamps, values = np.asarray(timestamps), np.asarray(values)
        if timestamps.ndim != 1 or not len(timestamps) or values.shape != (len(timestamps), PLAN_CHANNELS):
            raise ValueError('Expected aligned plan timestamps and 34 conditioning channels')
        if not np.isfinite(timestamps).all() or not np.isfinite(values).all() or np.any(np.diff(timestamps) <= 0):
            raise ValueError('Inference-plan telemetry must contain finite ordered values')
        # Copies protect the caller's arrays; no activation, interpolation or
        # rescaling is performed on the actual generator conditioning.
        arrays.update({prefix+'time': timestamps.copy(), prefix+'plan': values.copy(),
                       prefix+'log_speed': values[:, :5].copy(), prefix+'support': values[:, 5:9].copy(),
                       prefix+'hold': values[:, 9:14].copy(),
                       prefix+'duration': values[:, 14:].reshape(len(values), 5, 4).copy()})
    capture('', time, plan)
    if active:
        if coarse_time is None or coarse_plan is None:
            raise ValueError('Learned planning telemetry requires the actual coarse plan')
        capture('coarse_', coarse_time, coarse_plan)
    elif coarse_time is not None or coarse_plan is not None or np.any(arrays['plan'] != 0):
        raise ValueError('No-planner arms must expose only their unused zero conditioning slots')
    schema = {'schema_version': 1, 'kind': 'DynamicDanceInferencePlan', 'arm': arm,
              'planner_active': active, 'regions': list(REGION_NAMES),
              'support_joints': ['left_ankle', 'right_ankle', 'left_foot', 'right_foot'],
              'channel_slices': {'log_speed': [0, 5], 'support': [5, 9], 'hold': [9, 14], 'duration': [14, 34]},
              'field_meanings_when_planner_active': {'log_speed': 'log1p regional mean joint speed in m/s',
                  'support': 'predicted contact fraction for each listed foot joint',
                  'hold': 'probability of the regional low-speed proxy',
                  'duration': 'probabilities of remaining proxy-state duration bins'},
              'duration_layout': '[time, region, bin]; contiguous region-major flattening in plan',
              'duration_bin_intervals_seconds': ['remaining <= 0.5', '0.5 < remaining <= 1', '1 < remaining <= 2', 'remaining > 2'],
              'duration_target': 'remaining duration in regional speed <=0.2 m/s state or its complement; right-censored final target runs excluded',
              'hold_target': 'regional speed <=0.05 m/s kinematic proxy, not an annotated semantic hold',
              'values_role': 'actual learned conditioning supplied to generator' if active else 'unused zero conditioning slots; no planning decisions',
              'coarse_values_role': 'actual post-intervention plan used for interpolation' if active else 'absent: no planner',
              'source_evidence': json.loads(json.dumps(source_evidence)),
              'arrays': {name: {'sha256': array_sha256(value), 'shape': list(value.shape), 'dtype': value.dtype.str}
                         for name, value in arrays.items()}}
    _atomic_npz(path, **arrays)
    schema['npz_sha256'] = digest(path)
    authenticated_json(metadata_path, schema)
    return {'planner_active': active,
            'planner_status': 'learned plan recorded' if active else 'no planner; zero slots intentionally unused',
            'effort_role': 'log-speed slice of actual learned plan' if active else 'constant display placeholder, not conditioning or a planning decision',
            'plan_telemetry': {'path': str(path), 'sha256': schema['npz_sha256'],
                               'metadata_path': str(metadata_path), 'metadata_sha256': digest(metadata_path)}}


def infer(root, arm, seed, timeline_path, base92_path, output, seconds=180., draw_seed=7001, plan_override=None, source_body=None):
    """Infer from recomputed full-song audio; generate one native30Hz state."""
    from .contracts import MusicTimeline, DanceResult, DancePlan
    from .audio import model_features
    root, output = Path(root), Path(output)
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Inference output already exists; preserve it or select a new path')
    timeline = MusicTimeline.load(timeline_path)
    binding = None if arm == 'D6' else validate_audio_binding(timeline, base92_path)
    full_audio, names = model_features(timeline, target_time=timeline.time)
    model, state, checkpoint_path = load_learned(root, 'planner' if arm == 'D6' else arm, seed)
    if list(names) != state['feature_names']:
        raise ValueError('Inference feature schema differs from training')
    frames = min(len(timeline.time), int(round(seconds*30)))
    time30 = np.arange(frames)/30.
    extra = np.stack([np.interp(time30, timeline.time, full_audio[:, index]) for index in range(full_audio.shape[1])], -1)
    provenance = {'arm': arm, 'seed': seed, 'draw_seed': draw_seed, 'checkpoint': str(checkpoint_path),
                  'checkpoint_sha256': digest(checkpoint_path), **timeline_provenance(timeline, timeline_path),
                  'seconds': frames/30., 'conditioning': 'new full-song audio analysis', 'plan_override': plan_override,
                  'feature_schema_sha256': json_digest(list(names))}
    if arm == 'D6':
        predictions = planner_predict(model, full_audio, state['extra_normalizers'])
        raw = predictions['local']
        values = unpack_plan(raw)
        # Report raw learned quantities. Retiming chooses physical feasibility,
        # never rescales by the desired number of stops or rendered appearance.
        if source_body is None:
            raise ValueError('D6 needs its own accepted --source-body; no invented source')
        from .retiming import retime_with_effort
        with np.load(source_body, allow_pickle=False) as data:
            world, source_times = data['world'].copy(), data['time'].copy()
        if len(world) != frames or not np.allclose(source_times, time30, atol=1e-7):
            raise ValueError('D6 source must match native30Hz prefix exactly')
        source_speed = observed_regional_speed(world, fps=30.)
        coarse_time = np.arange(len(raw))*.5
        target_speed = torch.expm1(values['log_speed']).numpy()
        hold = values['hold'].numpy()
        # Bayesian mixture of learned moving effort and conservative hold speed.
        target_speed = target_speed*(1-hold) + .025*hold
        desired = np.stack([np.interp(time30, coarse_time, target_speed[:, i]) for i in range(5)], -1)
        duration = (values['duration'].numpy()*np.array([.5, 1., 2., 4.])).sum(-1)
        duration = np.stack([np.interp(time30, coarse_time, duration[:, i]) for i in range(5)], -1)
        # Duration controls the averaging window, never inserts a synthetic pose.
        from scipy.ndimage import gaussian_filter1d
        ratio = desired/np.maximum(source_speed, .05)
        smoothed = np.empty_like(ratio)
        for region in range(5):
            levels = np.stack([gaussian_filter1d(ratio[:, region], scale*30/4, mode='nearest')
                               for scale in (.5, 1., 2., 4.)], -1)
            coarse_probs = values['duration'][:, region].numpy()
            probs = np.stack([np.interp(time30, coarse_time, coarse_probs[:, i]) for i in range(4)], -1)
            smoothed[:, region] = (levels*probs).sum(-1)
        provenance.update(source_body=str(source_body), source_body_sha256=digest(source_body),
                          regions=REGION_NAMES, planner_branch='local', newly_generated_motion=False,
                          source_speed_estimator='incoming adjacent-frame displacement; frame0 repeats first interval; matches kinematic_plan_targets',
                          effort_to_rate='mixture (1-p_hold)*expm1(log_speed)+p_hold*.025m/s divided by source speed floored .05; duration-weighted .5/1/2/4s smoothing; contact-aware bounded endpoint projection')
        result, plan = retime_with_effort(world, source_times, smoothed, timeline, metadata=provenance)
        result.save(output)
        plan.save(output.parent/'plan.npz')
        _atomic_npz(output.parent/'learned_plan.npz', time=coarse_time, raw=raw.numpy(),
                    **{key: value.numpy() for key, value in values.items()})
    else:
        with np.load(base92_path, allow_pickle=False) as arrays:
            if 'x30' in arrays and 'time30' in arrays:
                base, base_time = arrays['x30'].copy(), arrays['time30'].copy()
            elif 'x' in arrays and 'time' in arrays:
                base, base_time = arrays['x'].copy(), arrays['time'].copy()
            else:
                raise ValueError('Base92 inference features require explicit timestamps')
        if len(base_time) < 1 or np.any(np.diff(base_time) <= 0) or base_time[0] > time30[0]+1e-8 or base_time[-1] < time30[-1]-1/30:
            raise ValueError('Base92 timestamps do not cover the generated prefix')
        if base.shape[-1] != 92:
            raise ValueError('Expected existing Base92 features')
        base = np.stack([np.interp(time30, base_time, base[:, index]) for index in range(92)], -1)
        bn, en = state['audio_normalizers'], state['extra_normalizers']
        packed = [((base-np.asarray(bn['mean']))/np.asarray(bn['std'])).astype(np.float32),
                  ((extra-np.asarray(en['mean']))/np.asarray(en['std'])).astype(np.float32)]
        plan = np.zeros((frames, PLAN_CHANNELS), np.float32)
        if arm in ('D8', 'D8_flat'):
            planner, planner_state, planner_path = load_learned(root, 'planner', seed)
            predictions = planner_predict(planner, full_audio, planner_state['extra_normalizers'])
            branch = 'global' if arm == 'D8' else 'local'
            coarse = plan_condition(predictions[branch]).numpy()
            if plan_override == 'shift':
                # Circular plan shift is explicitly a diagnostic plan corruption,
                # not a valid within-song audio shift or a newly generated plan.
                coarse = np.roll(coarse, len(coarse)//2, axis=0)
            elif plan_override is not None:
                raise ValueError('Unknown plan intervention')
            plan = np.stack([np.interp(time30, np.arange(len(coarse))*.5, coarse[:, i]) for i in range(PLAN_CHANNELS)], -1).astype(np.float32)
            provenance['planner_sha256'] = digest(planner_path)
        packed.append(plan)
        x = torch.from_numpy(np.concatenate(packed, -1))[None].cuda()
        with torch.inference_mode():
            native = model.generate(x, seed=draw_seed, chunked=True)[0]
            body = joints_from_features(native)
        if not bool(torch.isfinite(native).all() & torch.isfinite(body).all()):
            raise FloatingPointError('Generated body nonfinite; no substituted output')
        provenance['base92_sha256'] = digest(base92_path)
        provenance['newly_generated_motion'] = True
        provenance['source_clock'] = 'fresh native30Hz flow, one full-sequence initial noise tensor'
        DanceResult(time=time30, world=body.cpu().numpy(), native=native.cpu().numpy(),
                    contacts=native[:, 135:].cpu().numpy(), metadata=provenance).save(output)
        DancePlan(time=time30, source_time=time30, effort=plan[:, :5] if arm in ('D8', 'D8_flat') else np.ones((frames, 5)),
                  metadata={**provenance, 'effort_is_generated_plan': arm in ('D8', 'D8_flat'),
                            'fixed_effort_marker_for_D7': arm not in ('D8', 'D8_flat'),
                            **save_inference_plan_telemetry(output.parent/'plan_telemetry.npz', time30, plan,
                                arm=arm, source_evidence=provenance,
                                coarse_time=np.arange(len(coarse))*.5 if arm in ('D8', 'D8_flat') else None,
                                coarse_plan=coarse if arm in ('D8', 'D8_flat') else None)}).save(output.parent/'plan.npz')
    provenance['output_sha256'] = digest(output)
    authenticated_json(output.parent/'inference_receipt.json', provenance)
    return provenance


def _bootstrap_conditioning(rows, field, comparator, correct='correct'):
    grouped = defaultdict(list)
    for row in rows:
        if comparator in row['conditions'] and correct in row['conditions']:
            a, b = row['conditions'][correct][field], row['conditions'][comparator][field]
            grouped[row['canonical_id']].append((b-a)/max(abs(b), 1e-12))
    values = np.array([np.mean(grouped[key]) for key in sorted(grouped)], np.float64)
    if not len(values):
        return {'canonical_songs': 0, 'relative_improvement': None, 'ci95': None, 'passes_5pct': False}
    rng = np.random.default_rng(20261005)
    samples = values[rng.integers(len(values), size=(2000, len(values)))].mean(-1)
    interval = np.quantile(samples, [.025, .975])
    return {'canonical_songs': len(values), 'relative_improvement': float(values.mean()),
            'ci95': interval.tolist(), 'passes_5pct': bool(values.mean() >= .05 and interval[0] > 0)}


def fit_rms_baseline(manifest, norms):
    """Fixed ridge .001; equal canonical/record/frame TRAIN weighting only."""
    names = norms['feature_names']
    candidates = [i for i, name in enumerate(names) if 'mixture' in name.lower() and 'rms' in name.lower()]
    if len(candidates) != 1:
        raise ValueError('Cannot identify unique measured mixture RMS feature')
    column = candidates[0]
    counts = Counter(e['canonical_id'] for e in manifest['entries'] if e['partition'] == 'train')
    xx, xy = np.zeros((2, 2)), np.zeros((2, 5))
    for entry in manifest['entries']:
        if entry['partition'] != 'train':
            continue
        verify_files({entry['cache_path']: entry['cache_sha256']})
        with np.load(entry['cache_path'], allow_pickle=False) as arrays:
            valid = arrays['target_valid']
            rms = arrays['pooled'][:, column][valid]
            target = arrays['target_log_speed'][valid]
        if not len(rms):
            continue
        x = np.stack((np.ones_like(rms), np.log1p(np.maximum(rms, 0))), -1).astype(np.float64)
        weight = 1/(len(counts)*counts[entry['canonical_id']]*len(x))
        xx += weight*x.T@x; xy += weight*x.T@target
    coefficients = np.linalg.solve(xx + np.diag([0., .001]), xy)
    return {'column': column, 'feature_name': names[column], 'coefficients': coefficients.tolist(),
            'ridge': .001, 'fit': 'TRAIN canonical-equal regression from log1p measured RMS to regional log_speed'}


def array_sha256(value):
    """Hash dtype, shape and contiguous values, independent of NPZ container time."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        raise ValueError('Object arrays cannot be authenticated scientific values')
    digest_value = hashlib.sha256()
    digest_value.update(json.dumps({'dtype': value.dtype.str, 'shape': value.shape}, sort_keys=True).encode())
    digest_value.update(value.tobytes())
    return digest_value.hexdigest()


def temporal_activity(value, scale=None):
    """Descriptive output variation; effective rank is not a semantic gate."""
    value = np.asarray(value)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError('Expected finite nonempty time-by-channel values')
    x = value.astype(np.float64)
    fixed = np.all(value == value[:1], axis=0)
    centered = x-x.mean(axis=0)
    centered[:, fixed] = 0.
    variance = np.square(centered).mean(axis=0)
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float64)
        if scale.shape != (value.shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Expected finite positive channel scale')
        rank_values = centered/scale
    else:
        rank_values = centered
    covariance = rank_values.T@rank_values/len(value)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.)
    total = eigenvalues.sum()
    positive = eigenvalues[eigenvalues > 0]/total if total > 0 else np.empty(0)
    effective_rank = float(np.exp(-np.sum(positive*np.log(positive)))) if len(positive) else 0.
    return {'frames': len(value), 'temporal_observation_available': len(value) > 1,
            'channel_variance': variance.tolist(), 'channel_std': np.sqrt(variance).tolist(),
            'mean_channel_std': float(np.sqrt(variance).mean()),
            'exact_constant': bool(fixed.all()), 'exact_constant_channels': np.flatnonzero(fixed).tolist(),
            'effective_rank': effective_rank,
            'effective_rank_definition': 'entropy rank of centered temporal covariance; TRAIN-scaled channels' if scale is not None else 'entropy rank of centered raw temporal covariance',
            'interpretation': 'output variation only; no threshold or semantic-collapse verdict'}


def response_activity(value, reference):
    """Report input/output changes separately, without qualifying choreography."""
    value, reference = np.asarray(value), np.asarray(reference)
    if value.shape != reference.shape or not np.isfinite(value).all() or not np.isfinite(reference).all():
        raise ValueError('Response comparison requires same-shaped finite arrays')
    delta = value.astype(np.float64)-reference.astype(np.float64)
    return {'rms_difference': float(np.sqrt(np.square(delta).mean())),
            'max_abs_difference': float(np.abs(delta).max()),
            'exact_equal': bool(np.array_equal(value, reference))}


def body_activity(native, world, fps=30., native_scale=None):
    """Separate native variability, world inactivity and relative articulation."""
    native, world = np.asarray(native), np.asarray(world)
    if native.ndim != 2 or native.shape[-1] != 139 or world.shape != (len(native), 22, 3) or not len(native):
        raise ValueError('Expected aligned nonempty native139 and world22 arrays')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Expected finite positive fps')
    result = {'native_finite': bool(np.isfinite(native).all()), 'world_finite': bool(np.isfinite(world).all())}
    result['finite'] = result['native_finite'] and result['world_finite']
    if not result['finite']:
        return {**result, 'metrics_unavailable_reason': 'nonfinite generation retained as failure artifact'}
    world64 = world.astype(np.float64)
    relative = world64-world64[:, :1]
    result.update(native=temporal_activity(native, native_scale),
                  world_exact_constant=bool(np.array_equal(world, np.broadcast_to(world[:1], world.shape))),
                  root_relative_exact_constant=bool(np.array_equal(relative, np.broadcast_to(relative[:1], relative.shape))),
                  interval_count=max(0, len(world)-1), hold_threshold_m_s=.05,
                  hold_interpretation='exploratory speed threshold, not an expressive-dance or semantic-collapse criterion')
    def quantiles(value):
        return {name: float(result) for name, result in zip(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'),
                np.quantile(value, (0., .05, .25, .5, .75, .95, 1.)))} if value.size else None
    speed = np.linalg.norm(np.diff(world64, axis=0)*fps, axis=-1)
    root_speed = speed[:, 0]
    relative_speed = np.linalg.norm(np.diff(relative, axis=0)*fps, axis=-1)[:, 1:]
    body_mean = speed.mean(-1)
    relative_mean = relative_speed.mean(-1)
    result.update(world_joint_speed_quantiles_m_s=quantiles(speed),
                  world_body_mean_speed_quantiles_m_s=quantiles(body_mean),
                  root_speed_quantiles_m_s=quantiles(root_speed),
                  root_relative_joint_speed_quantiles_m_s=quantiles(relative_speed),
                  world_hold_occupancy=float(np.mean(body_mean <= .05)) if len(body_mean) else None,
                  root_relative_hold_occupancy=float(np.mean(relative_mean <= .05)) if len(relative_mean) else None)
    return result


def save_evaluation_artifact(path, arrays, metadata):
    """Keep every evaluated condition; retries may reuse exact authenticated data."""
    path = Path(path)
    arrays = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
              for key, value in arrays.items()}
    array_hashes = {key: array_sha256(value) for key, value in arrays.items()}
    metadata_path = path.with_suffix('.json')
    identity = {'arrays': array_hashes, 'metadata': metadata}
    if path.exists() or metadata_path.exists():
        if not (path.exists() and metadata_path.exists()):
            raise ValueError('Partial evaluation artifact preserved; explicit inspection required: '+str(path))
        receipt = read_authenticated(metadata_path)
        verify_files({str(path): receipt['npz_sha256']})
        if receipt['identity'] != identity:
            raise ValueError('Refusing to replace changed evaluation artifact: '+str(path))
    else:
        _atomic_npz(path, **arrays)
        receipt = {'kind': 'DynamicDanceEvaluation', 'npz_sha256': digest(path), 'identity': identity}
        authenticated_json(metadata_path, receipt)
    return {'path': str(path), 'sha256': digest(path), 'metadata_path': str(metadata_path),
            'metadata_sha256': digest(metadata_path), 'arrays': array_hashes}


def evaluate(root=DEFAULT_ROOT, arm='planner', seed=0):
    """Development diagnostics only; no hidden TEST access or checkpoint selection."""
    from ..motion_study.music_data import motion_summary
    from ..motion_study.music_physical import physical_losses
    root = Path(root)
    protocol = _protocol(root)
    path = root/f'learning/evaluation/{arm}/seed_{seed}.json'
    model, state, checkpoint = load_learned(root, arm, seed)
    identity = {'checkpoint_sha256': digest(checkpoint), 'protocol_sha256': digest(root/'learning/protocol.json')}
    if path.exists():
        result = read_authenticated(path)
        if result['identity'] != identity:
            raise ValueError('Evaluation identity changed')
        verify_files(result['artifacts'])
        return result
    manifest, norms, base_norms, motion_norms = _context(root)
    entries = {entry['key']: entry for entry in manifest['entries'] if entry['partition'] == 'val'}
    windows = protocol['evaluation']['windows']
    rows = []
    artifact_root = path.parent/f'seed_{seed}_artifacts'
    artifacts = {}
    if (artifact_root/'index.json').exists():
        prior_index = read_authenticated(artifact_root/'index.json')
        if prior_index['identity'] != identity:
            raise ValueError('Existing evaluation artifacts have a different checkpoint/protocol identity')
        verify_files(prior_index['artifacts'])
        artifacts.update(prior_index['artifacts'])
    def retain(name, arrays, metadata):
        require_space(root, gb=20)
        artifact = save_evaluation_artifact(artifact_root/(name+'.npz'), arrays, {**identity, **metadata})
        artifacts[artifact['path']] = artifact['sha256']
        artifacts[artifact['metadata_path']] = artifact['metadata_sha256']
        authenticated_json(artifact_root/'index.json', {'identity': identity, 'artifacts': artifacts})
        return artifact
    if arm == 'planner':
        rms = fit_rms_baseline(manifest, norms)
        predictions, targets, pooled, normalized_audio, full_plan_artifacts = {}, {}, {}, {}, {}
        data = PlannerData(manifest, norms, partition='val')
        for record in data.records:
            key = record['entry']['key']
            x = torch.from_numpy(record['audio'])[None].cuda()
            valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            with torch.inference_mode():
                predictions[key] = {branch: value[0].cpu() for branch, value in model(x, valid).items()}
            targets[key] = record['targets']
            normalized_audio[key] = record['audio']
            finite = all(bool(torch.isfinite(value).all()) for value in predictions[key].values())
            full_plan_artifacts[key] = retain(key+'_full_plans',
                {'time': np.arange(len(record['audio']))*.5, **predictions[key],
                 **{'target_'+name: value for name, value in targets[key].items()}},
                {'key': key, 'partition': 'val', 'finite': finite,
                 'normalized_input_sha256': array_sha256(record['audio']),
                 'normalized_input_shape': list(record['audio'].shape),
                 'target_kind': 'kinematic proxy targets, not semantic labels'})
            if not finite:
                raise FloatingPointError('Nonfinite planner outputs retained: '+full_plan_artifacts[key]['path'])
            with np.load(record['entry']['cache_path'], allow_pickle=False) as arrays:
                pooled[key] = arrays['pooled'].copy()
        for window in windows:
            key, n, start = window['key'], (window['frames']+14)//15, window['start']//15
            valid = targets[key]['valid'][start:start+n]
            target = targets[key]['log_speed'][start:start+n]
            if not valid.any():
                rows.append({**window, 'conditions': {}, 'unavailable': 'no observed speed in this one-frame recording'})
                continue
            row = {**window, 'conditions': {}, 'full_plan_artifact': full_plan_artifacts[key]}
            correct_plan = {branch: plan_condition(predictions[key][branch][start:start+n]).numpy() for branch in ('local', 'global')}
            for condition, donor, position in [('correct', key, start), ('donor', window['donor_key'], window['donor_start']//15),
                    ('shift', key, None if window['shift_start'] is None else window['shift_start']//15)]:
                if position is None:
                    continue
                result = {}
                condition_arrays = {'time': np.arange(n)*.5, 'target_valid': valid,
                                    **{'target_'+name: values[start:start+n] for name, values in targets[key].items()}}
                conditioned = normalized_audio[donor][position:position+n]
                input_response = response_activity(conditioned, normalized_audio[key][start:start+n])
                for branch in ('local', 'global'):
                    raw = predictions[donor][branch][position:position+n]
                    value = unpack_plan(raw)
                    actual_plan = plan_condition(raw).numpy()
                    condition_arrays[branch+'_raw'] = raw.numpy()
                    condition_arrays[branch+'_plan'] = actual_plan
                    result[branch+'_activity'] = temporal_activity(actual_plan)
                    result[branch+'_response_from_correct'] = response_activity(actual_plan, correct_plan[branch])
                    result[branch+'_speed_rmse'] = float(np.sqrt(np.mean(np.square(value['log_speed'].numpy()[valid]-target[valid]))))
                    result[branch+'_hold_brier'] = float(np.mean(np.square(value['hold'].numpy()[valid]-targets[key]['hold'][start:start+n][valid])))
                x = np.stack((np.ones(n), np.log1p(np.maximum(pooled[donor][position:position+n, rms['column']], 0))), -1)
                prediction = np.maximum(x@np.asarray(rms['coefficients']), 0)
                result['rms_speed_rmse'] = float(np.sqrt(np.mean(np.square(prediction[valid]-target[valid]))))
                condition_arrays['rms_log_speed_prediction'] = prediction
                result['normalized_input_response_from_correct'] = input_response
                result['normalized_input_sha256'] = array_sha256(conditioned)
                result['full_input_sha256'] = array_sha256(normalized_audio[donor])
                result['artifact'] = retain(key+'_'+condition, condition_arrays,
                    {'key': key, 'condition': condition, 'audio_key': donor, 'audio_start_bin': position,
                     'normalized_input_sha256': result['normalized_input_sha256'],
                     'full_input_sha256': result['full_input_sha256'],
                     'full_input_artifact': full_plan_artifacts[donor], 'finite': True,
                     'activity': {branch: result[branch+'_activity'] for branch in ('local', 'global')},
                     'input_response': input_response})
                row['conditions'][condition] = result
            # Matched shuffled learned effort schedule, not shuffled audio.
            permutation = np.random.default_rng(7001).permutation(n)
            shuffled = unpack_plan(predictions[key]['local'][start:start+n][permutation])['log_speed'].numpy()
            shuffled_raw = predictions[key]['local'][start:start+n][permutation].numpy()
            row['conditions']['shuffled_plan'] = {'local_speed_rmse': float(np.sqrt(np.mean(np.square(shuffled[valid]-target[valid])))),
                'activity': temporal_activity(plan_condition(torch.from_numpy(shuffled_raw)).numpy()),
                'artifact': retain(key+'_shuffled_plan', {'time': np.arange(n)*.5, 'local_raw': shuffled_raw,
                    'target_log_speed': target, 'target_valid': valid, 'permutation': permutation},
                    {'key': key, 'condition': 'shuffled_plan', 'finite': True,
                     'normalized_input_sha256': array_sha256(normalized_audio[key][start:start+n]),
                     'full_input_sha256': array_sha256(normalized_audio[key]), 'intervention': 'plan order only'})}
            row['conditions']['rms_control'] = {'local_speed_rmse': row['conditions']['correct']['rms_speed_rmse'],
                'artifact': row['conditions']['correct']['artifact'],
                'array_name': 'rms_log_speed_prediction'}
            rows.append(row)
        summary = {name: _bootstrap_conditioning(rows, 'local_speed_rmse', name) for name in ('donor', 'shift', 'shuffled_plan', 'rms_control')}
        result = {'identity': identity, 'rows': rows, 'summary': summary, 'rms_baseline': rms,
                  'claim_scope': 'exposed VAL motion-proxy prediction, not semantic or lighting qualification', 'test_access': False}
    else:
        branch = 'global' if arm == 'D8' else 'local'
        plans = None
        if arm in ('D8', 'D8_flat'):
            plans = read_authenticated(root/f'learning/plans/seed_{seed}/{branch}/index.json')['entries']
        def conditioning(entry, start, frames):
            verify_files({entry['audio_path']: entry['audio_sha256'], entry['cache_path']: entry['cache_sha256']})
            with np.load(entry['audio_path'], allow_pickle=False) as arrays:
                base = arrays['x'][start:start+frames].copy()
            with np.load(entry['cache_path'], allow_pickle=False) as arrays:
                extra = arrays['extra'][start:start+frames].astype(np.float32)
            plan = np.zeros((frames, PLAN_CHANNELS), np.float32)
            if plans is not None:
                verify_files({plans[entry['key']]['path']: plans[entry['key']]['sha256']})
                with np.load(plans[entry['key']]['path'], allow_pickle=False) as arrays:
                    plan = np.stack([np.interp((np.arange(frames)+start)/30, arrays['time'], arrays['plan'][:, i]) for i in range(PLAN_CHANNELS)], -1)
            packed = np.concatenate(((base-np.asarray(base_norms['mean']))/np.asarray(base_norms['std']),
                                      (extra-np.asarray(norms['mean']))/np.asarray(norms['std']), plan), -1)
            return torch.from_numpy(packed.astype(np.float32))[None].cuda()
        def retain_flow(window, condition, generated, packed, initial, audio_key, audio_start, correct_packed):
            native = generated[0].detach().cpu().numpy()
            world = joints_from_features(generated)[0].detach().cpu().numpy()
            packed = packed[0].detach().cpu().numpy()
            activity = body_activity(native, world, native_scale=motion_norms['std'])
            metadata = {'key': window['key'], 'condition': condition, 'audio_key': audio_key,
                        'audio_start_frame': audio_start, 'target_start_frame': window['start'],
                        'target_motion_path': entries[window['key']]['path'],
                        'target_motion_sha256': entries[window['key']]['sha256'],
                        'normalized_input_sha256': array_sha256(packed), 'normalized_input_shape': list(packed.shape),
                        'shared_initial_noise_sha256': array_sha256(initial[0]),
                        'shared_initial_noise_shape': list(initial[0].shape), 'draw_seed': window['draw_seed'],
                        'activity': activity, 'input_response_from_correct': response_activity(packed, correct_packed)}
            artifact = retain(window['key']+'_'+condition,
                              {'native': native, 'world': world, 'time': np.arange(len(native))/30.}, metadata)
            if not activity['finite']:
                raise FloatingPointError('Nonfinite development generation retained: '+artifact['path'])
            return {'activity': activity, 'artifact': artifact,
                    'normalized_input_sha256': metadata['normalized_input_sha256'],
                    'shared_initial_noise_sha256': metadata['shared_initial_noise_sha256'],
                    'normalized_input_response_from_correct': metadata['input_response_from_correct']}
        summary_mean = torch.tensor(motion_norms['summary']['mean'], device='cuda')
        summary_std = torch.tensor(motion_norms['summary']['std'], device='cuda')
        for window in windows:
            entry, frames, start = entries[window['key']], window['frames'], window['start']
            verify_files({entry['path']: entry['sha256']})
            with np.load(entry['path'], allow_pickle=False) as arrays:
                reference = torch.from_numpy(arrays['features'][start:start+frames].copy())[None].cuda()
            generator = torch.Generator(device='cuda').manual_seed(window['draw_seed'])
            initial = model.noise(1, frames, generator=generator)
            row = {**window, 'conditions': {}}
            with torch.inference_mode():
                target_summary = (motion_summary(reference)-summary_mean)/summary_std
                correct_native = None
                correct_packed = None
                for condition, key, position in [('correct', window['key'], start),
                        ('donor', window['donor_key'], window['donor_start']), ('shift', window['key'], window['shift_start'])]:
                    if position is None:
                        continue
                    x = conditioning(entries[key], position, frames)
                    flow_errors, temporal_errors, physical_totals = [], [], []
                    for evaluation_time in protocol['evaluation']['flow_loss_times']:
                        loss = model.losses_from_draws(x, reference, torch.ones(reference.shape[:2], dtype=torch.bool, device='cuda'),
                            initial, torch.tensor([evaluation_time], device='cuda'))
                        flow_errors.append(float(loss['native']))
                        temporal_errors.append(float(loss['native']+.05*loss['velocity']))
                        physical_totals.append(float(loss['total']))
                    generated = model.generate(x, initial_noise=initial, chunked=True)
                    if correct_packed is None:
                        correct_packed = x[0].detach().cpu().numpy()
                    output_evidence = retain_flow(window, condition, generated, x, initial, key, position, correct_packed)
                    summary = (motion_summary(generated)-summary_mean)/summary_std
                    score = float(torch.sqrt((summary-target_summary).square().mean()))
                    metrics = physical_losses((generated-model.mean)/model.std, reference, model.mean, model.std,
                                              torch.ones(generated.shape[:2], dtype=torch.bool, device='cuda'))
                    if correct_native is None:
                        correct_native = generated.clone()
                    row['conditions'][condition] = {**output_evidence, 'summary_rmse': score,
                        'flow_loss': float(np.mean(flow_errors)), 'temporal_flow_loss': float(np.mean(temporal_errors)),
                        'physical_total_loss': float(np.mean(physical_totals)),
                        'native_difference_from_correct_rms': float(torch.sqrt((generated-correct_native).square().mean())),
                        'contact_coverage': float((generated[..., 135:] >= .5).float().mean()),
                        **{key: float(value) for key, value in metrics.items()}}
                if arm == 'D8':
                    x = conditioning(entry, start, frames)
                    x[..., -PLAN_CHANNELS:] = torch.roll(x[..., -PLAN_CHANNELS:], frames//2, dims=1)
                    generated = model.generate(x, initial_noise=initial, chunked=True)
                    output_evidence = retain_flow(window, 'shifted_plan', generated, x, initial, entry['key'], start, correct_packed)
                    summary = (motion_summary(generated)-summary_mean)/summary_std
                    row['conditions']['shifted_plan'] = {**output_evidence, 'summary_rmse': float(torch.sqrt((summary-target_summary).square().mean())),
                        'native_difference_from_correct_rms': float(torch.sqrt((generated-correct_native).square().mean()))}
            rows.append(row)
            atomic_json(path.with_suffix('.progress.json'), {'identity': identity, 'completed_recordings': len(rows), 'total_recordings': len(windows)})
        comparisons = ['donor', 'shift'] + (['shifted_plan'] if arm == 'D8' else [])
        result = {'identity': identity, 'rows': rows,
                  'summary': {condition: _bootstrap_conditioning(rows, 'summary_rmse', condition) for condition in comparisons},
                  'primary_flow_loss': {condition: _bootstrap_conditioning(rows, 'flow_loss', condition) for condition in ('donor', 'shift')},
                  'claim_scope': 'exposed VAL single-draw physical-summary diagnostics; not artistic or light-transfer qualification',
                  'noise': 'one identical full-window field for every paired audio/plan condition', 'test_access': False}
    result['artifacts'] = artifacts
    result['artifact_index'] = {'path': str(artifact_root/'index.json'), 'sha256': digest(artifact_root/'index.json')}
    result['activity_scope'] = 'native output variation, physical inactivity and input response reported separately; no automatic semantic-collapse verdict'
    authenticated_json(path, result)
    return result


@contextmanager
def admitted_gpu():
    """Same lock ordering as the audio campaign; no bypass for foreign jobs."""
    with Path('/tmp/lightsem-audio-understanding-gpu.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Audio campaign owns GPU admission lock') from error
        with gpu_lock():
            yield
        fcntl.flock(stream, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'preflight', 'freeze', 'train', 'export-plans', 'infer', 'evaluate'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--guarded', action='store_true')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--arm', choices=('planner', 'D6', *FLOW_ARMS), default='planner')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--until', type=int)
    parser.add_argument('--timeline', type=Path)
    parser.add_argument('--base92', type=Path)
    parser.add_argument('--source-body', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--seconds', type=float, default=180.)
    parser.add_argument('--plan-override', choices=('shift',))
    args = parser.parse_args(argv)
    if args.stage == 'prepare':
        result = prepare_cache(args.root, args.index)
    elif args.stage == 'freeze':
        result = freeze_protocol(args.root)
    else:
        if args.guarded:
            from scripts.motion.dynamic_dance_guard import authenticate_guarded_job
            authenticate_guarded_job(Path(__file__).parents[3]/'scripts/motion/train_dynamic_dance.py')
        with nullcontext() if args.guarded else admitted_gpu():
            if args.stage == 'preflight':
                result = preflight(args.root)
            elif args.stage == 'train':
                result = train(args.root, args.arm, args.seed, args.resume, args.until)
            elif args.stage == 'export-plans':
                result = export_training_plans(args.root, args.seed)
            elif args.stage == 'evaluate':
                result = evaluate(args.root, args.arm, args.seed)
            else:
                if args.timeline is None or args.output is None or (args.arm != 'D6' and args.base92 is None):
                    parser.error('infer requires --timeline --output and --base92 for flow arms')
                result = infer(args.root, args.arm, args.seed, args.timeline, args.base92, args.output,
                               args.seconds, plan_override=args.plan_override, source_body=args.source_body)
    if result is not None:
        print(json.dumps(result, indent=2), flush=True)

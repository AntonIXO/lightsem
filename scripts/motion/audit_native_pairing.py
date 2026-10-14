#!/usr/bin/env python3
"""Re-evaluate saved native checkpoints with the canonical noise/context clock.

No optimization, mapping or rendering. Old unpaired artifacts are read-only.
One protocol binds every checkpoint before any new output is generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch

from lightsem.dance.baseline import load_config, sha
from lightsem.dance.pipeline import accelerator_lock, music_input, native_body, write_json
from lightsem.experiments.dynamic_dance.audio import model_features
from lightsem.experiments.motion_study.geometry import joints_from_features
from lightsem.experiments.motion_study.music_model import MusicConfig
from lightsem.experiments.native_dynamics.dataset import CONTEXT_BINS
from lightsem.experiments.native_dynamics.evaluation import cpu_noise, setup
from lightsem.experiments.native_dynamics.model import Config, NativeDynamicsModel
from lightsem.experiments.native_repair.assessment import raw_metrics
from lightsem.eval.training_health import motion_metrics


def array_sha(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def canonical_inputs(music, base, stereo, state, config, device):
    n = len(base)
    extra, _ = model_features(music, np.arange(n) / 30.)
    bins, pad = (n + 14) // 15, (-n) % 15
    if bins > CONTEXT_BINS:
        raise ValueError('Planner context overflow')
    # Preserve the production arithmetic, including float64 partial-bin divide.
    pooled = np.pad(extra, ((0, pad), (0, 0))).reshape(bins, 15, 1061).sum(1)
    pooled /= np.minimum(15, n - np.arange(bins) * 15)[:, None]
    norms = [tuple(torch.tensor(state[name][k], device=device, dtype=torch.float32)
                   for k in ('mean', 'std'))
             for name in ('audio_normalizers', 'extra_normalizers')]
    fine = torch.cat(((torch.tensor(base, device=device) - norms[0][0]) / norms[0][1],
                      (torch.tensor(extra, device=device) - norms[1][0]) / norms[1][1]), -1)[None]
    coarse = torch.zeros(1, CONTEXT_BINS, 1061, device=device)
    valid = torch.zeros(1, CONTEXT_BINS, device=device, dtype=torch.bool)
    coarse[0, :bins] = (torch.tensor(pooled, device=device) - norms[1][0]) / norms[1][1]
    valid[0, :bins] = True
    if Config(**state['config']).stereo_dim:
        if stereo is None:
            raise ValueError('Stereo checkpoint requires an authenticated context')
        sm = torch.tensor(state['stereo_mean'], device=device, dtype=torch.float32)
        ss = torch.tensor(state['stereo_std'], device=device, dtype=torch.float32)
        stereo_t = (torch.tensor(stereo, device=device) - sm) / ss
        stereo_pad = np.pad(stereo, ((0, pad), (0, 0)))
        stereo_coarse = stereo_pad.reshape(bins, 15, stereo.shape[-1]).sum(1)
        stereo_coarse /= np.minimum(15, n - np.arange(bins) * 15)[:, None]
        cs = torch.zeros(1, CONTEXT_BINS, stereo.shape[-1], device=device)
        cs[0, :bins] = (torch.tensor(stereo_coarse, device=device) - sm) / ss
        return (fine, coarse, valid, torch.arange(n, device=device)[None]), dict(
            stereo_context=stereo_t[None], coarse_stereo=cs)
    if stereo is not None:
        raise ValueError('Non-stereo checkpoint received a stereo context')
    return (fine, coarse, valid, torch.arange(n, device=device)[None]), {}


def acoustic_masks(audio, clock):
    y, sr = sf.read(audio, always_2d=True)
    y = y.mean(1)
    # Exact sample boundaries avoid accumulating half-sample clock errors.
    edges = np.minimum(np.rint(np.arange(len(clock) + 1) * sr / 30).astype(int), len(y))
    cs = np.r_[0., np.cumsum(y.astype(float) ** 2)]
    energy = np.sqrt((cs[edges[1:]] - cs[edges[:-1]]) / np.maximum(np.diff(edges), 1))
    energy = np.convolve(energy, np.ones(30) / 30, mode='same')
    interior = (clock >= 2) & (clock <= clock[-1] - 2)
    low, high = np.quantile(energy[interior], [.2, .8])
    return energy, (energy <= low) & interior, (energy >= high) & interior


def conditional_motion(world, quiet, active):
    q = np.asarray(world, dtype=float) - world[:, :1]
    speed = np.linalg.norm(np.diff(q, axis=0) * 30, axis=-1).mean(1)
    jerk = np.linalg.norm(np.diff(q, n=3, axis=0) * 30**3, axis=-1).mean(1)
    quiet_v, active_v = quiet[1:] & quiet[:-1], active[1:] & active[:-1]
    quiet_j = quiet[:-3] & quiet[1:-2] & quiet[2:-1] & quiet[3:]
    return dict(quiet_speed=float(speed[quiet_v].mean()), active_speed=float(speed[active_v].mean()),
                quiet_jerk_p95=float(np.quantile(jerk[quiet_j], .95)),
                quiet_active_ratio=float(speed[quiet_v].mean() / speed[active_v].mean()),
                quiet_seconds=float(quiet_v.sum() / 30), active_seconds=float(active_v.sum() / 30),
                raw=raw_metrics(world), world=motion_metrics(world, fps=30))


def generate(protocol_path):
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['files'].items():
        if sha(name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    out = Path(protocol['output'])
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    setup()
    config, _ = load_config()
    base_state = torch.load(config['checkpoint']['path'], weights_only=False, map_location='cpu')
    song_inputs = {}
    report = dict(protocol_sha256=sha(protocol_path), status='running', rows=[],
                  scope='Corrected paired descriptive audit; previously inspected showcase songs, no selection or retraining')
    started = time.monotonic()
    with accelerator_lock(), torch.inference_mode():
        for song, audio in protocol['songs'].items():
            music, base, binding, stereo = music_input(Path(audio), out, config)
            body, _, _, _ = native_body(music, base, config, stereo=stereo, device='cuda')
            reference_file = Path(protocol['reference_root']) / f'{song}.npz'
            with np.load(reference_file) as z:
                ref = z['body'].copy()
                if not np.array_equal(z['time'], body.time):
                    raise ValueError('Reference clock differs')
            mismatch = float(np.max(np.abs(body.world - ref)))
            if mismatch > protocol['identity_max_abs_m']:
                raise ValueError(f'Canonical identity test failed: {song} {mismatch}')
            energy, quiet, active = acoustic_masks(audio, body.time)
            folder = out / 'baseline'; folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f'{song}.npz', time=body.time, body=body.world,
                                native=body.native, energy=energy, quiet=quiet, active=active)
            song_inputs[song] = music, base, stereo, body, binding, quiet, active
            report.setdefault('identity', {})[song] = dict(reference_path=str(reference_file),
                reference_sha256=sha(reference_file), maximum_absolute_difference_m=mismatch,
                noise_sha256=body.metadata['noise_sha256'])
        for arm in protocol['arms']:
            state = torch.load(arm['checkpoint'], weights_only=False, map_location='cpu')
            model = NativeDynamicsModel(MusicConfig(**state['base_config']), state['normalizers'], Config(**state['config']))
            model.load_state_dict(state['model'], strict=True)
            model.cuda().eval().requires_grad_(False)
            unchanged = all(torch.equal(v, base_state['model'][k]) for k, v in state['model'].items()
                            if k in base_state['model'])
            for song, (music, base, stereo, baseline, binding, quiet, active) in song_inputs.items():
                arm_stereo = stereo
                if arm.get('context_manifest'):
                    manifest = json.loads(Path(arm['context_manifest']).read_text())
                    row = next(r for r in manifest['rows'] if r['id'] == song)
                    if sha(row['context_path']) != row['context_sha256']:
                        raise ValueError('Changed experimental context')
                    with np.load(row['context_path']) as z:
                        if not np.array_equal(z['time30'], music.time):
                            raise ValueError('Experimental context clock differs')
                        arm_stereo = z['fused30'].astype(np.float32)
                args, kwargs = canonical_inputs(music, base, arm_stereo, state, config, 'cuda')
                noise = cpu_noise(model.flow, max(config['noise_reference_frames'], len(base)), config['draw_seed'])[:, :len(base)].contiguous().cuda()
                if array_sha(noise) != baseline.metadata['noise_sha256']:
                    raise ValueError('Noise differs from paired baseline')
                native, _ = model.generate(*args, noise, **kwargs)
                world = joints_from_features(native)[0].cpu().numpy()
                folder = out / arm['id']; folder.mkdir(exist_ok=True)
                path = folder / f'{song}.npz'
                np.savez_compressed(path, time=baseline.time, body=world, native=native[0].cpu().numpy())
                measured, reference = conditional_motion(world, quiet, active), conditional_motion(baseline.world, quiet, active)
                ratios = {k: measured[k] / max(reference[k], 1e-12)
                          for k in ('quiet_speed', 'active_speed', 'quiet_jerk_p95', 'quiet_active_ratio')}
                safety = dict(speed=measured['world']['speed_p95'] / reference['world']['speed_p95'],
                              acceleration=measured['world']['acceleration_raw']['p95'] / reference['world']['acceleration_raw']['p95'],
                              jerk=measured['world']['jerk_raw']['p95'] / reference['world']['jerk_raw']['p95'])
                passed = measured['raw']['finite'] and all(safety[k] <= v for k, v in protocol['safety_ratios'].items())
                rec = dict(arm=arm['id'], song=song, body_path=str(path), body_sha256=sha(path),
                    checkpoint_path=arm['checkpoint'], checkpoint_sha256=sha(arm['checkpoint']),
                    baseline_body=str(out / 'baseline' / f'{song}.npz'),
                    baseline_body_sha256=sha(out / 'baseline' / f'{song}.npz'),
                    finite=bool(np.isfinite(world).all() and np.isfinite(native.cpu().numpy()).all()),
                    noise_sha256=array_sha(noise), noise_reference_frames=max(config['noise_reference_frames'], len(base)),
                    planner_context_bins=CONTEXT_BINS, incumbent_weights_unchanged=unchanged,
                    inputs={**binding, 'normalized_fine_sha256': array_sha(args[0]),
                            'normalized_coarse_sha256': array_sha(args[1]),
                            'normalized_stereo_sha256': (array_sha(kwargs['stereo_context'])
                                                         if 'stereo_context' in kwargs else None),
                            'arm_context_manifest': arm.get('context_manifest')},
                    candidate=measured, baseline=reference, ratios=ratios, raw_safety=safety,
                    status='diagnostic_only' if passed else 'rejected_raw_safety')
                write_json(folder / f'{song}.json', rec)
                report['rows'].append(rec)
                print(json.dumps({k:rec[k] for k in ('arm','song','status','ratios')}), flush=True)
            del model
            torch.cuda.empty_cache()
    for name, digest in protocol['files'].items():
        if sha(name) != digest:
            raise ValueError('Input drift during audit: ' + name)
    report.update(status='complete', elapsed_s=time.monotonic()-started)
    write_json(out / 'REPORT.json', report)
    write_json(out / 'REPORT.sha256.json', {'sha256': sha(out / 'REPORT.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', required=True, type=Path)
    generate(p.parse_args().protocol)

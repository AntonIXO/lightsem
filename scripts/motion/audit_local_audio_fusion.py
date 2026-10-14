#!/usr/bin/env python3
"""Audit trained local audio fusion arms before mapper or rendering.

The source intervention zeroes only the fine native audio descriptor and holds
the audio-derived planner output, stereo context, noise, mapping, and song
clock fixed. This isolates the local audio-fusion response from planner and
stereo changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from lightsem.dance.baseline import load_config, sha
from lightsem.dance.pipeline import accelerator_lock, music_input, native_body, write_json
from lightsem.experiments.motion_study.geometry import joints_from_features
from lightsem.experiments.motion_study.music_model import MusicConfig
from lightsem.experiments.native_dynamics.evaluation import cpu_noise, setup
from lightsem.experiments.native_dynamics.model import Config, NativeDynamicsModel
from lightsem.experiments.native_repair.assessment import raw_metrics
from lightsem.eval.training_health import motion_metrics
from scripts.motion.audit_native_pairing import acoustic_masks, canonical_inputs, conditional_motion


def array_sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def root_relative(value):
    return value - value[:, :1]


def generate(protocol_path: Path):
    protocol = json.loads(protocol_path.read_text())
    for name, digest in protocol['files'].items():
        if sha(name) != digest:
            raise ValueError('Frozen input changed: ' + name)
    output = Path(protocol['output'])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    setup()
    config, _ = load_config()
    baseline_state = torch.load(config['checkpoint']['path'], weights_only=False, map_location='cpu')
    started = time.monotonic()
    report = {'status': 'running', 'protocol_sha256': sha(protocol_path), 'rows': []}
    songs = {}
    with accelerator_lock(), torch.inference_mode():
        for song, audio in protocol['songs'].items():
            music, base, binding, stereo = music_input(Path(audio), output, config)
            reference, _, _, _ = native_body(music, base, config, stereo=stereo, device='cuda')
            energy, quiet, active = acoustic_masks(audio, reference.time)
            songs[song] = dict(music=music, base=base, stereo=stereo, binding=binding,
                               baseline=reference, quiet=quiet, active=active, energy=energy)
            if not np.isfinite(reference.world).all():
                raise ValueError('Nonfinite baseline body: ' + song)
        for arm in protocol['arms']:
            state = torch.load(arm['checkpoint'], weights_only=False, map_location='cpu')
            model = NativeDynamicsModel(MusicConfig(**state['base_config']), state['normalizers'], Config(**state['config']))
            model.load_state_dict(state['model'], strict=True)
            model.cuda().eval().requires_grad_(False)
            for song, values in songs.items():
                args, kwargs = canonical_inputs(values['music'], values['base'], values['stereo'], state, config, 'cuda')
                noise = cpu_noise(model.flow, max(config['noise_reference_frames'], len(values['base'])), config['draw_seed'])
                noise = noise[:, :len(values['base'])].contiguous().cuda()
                if array_sha(noise) != values['baseline'].metadata['noise_sha256']:
                    raise ValueError('Canonical noise mismatch: ' + song)
                candidate, plan = model.generate(*args, noise, **kwargs)
                world = joints_from_features(candidate)[0].cpu().numpy()
                # Hold planner output and stereo fixed while removing only the
                # descriptor read by the local audio fusion module.
                altered = [x.clone() if torch.is_tensor(x) else x for x in args]
                altered[0][..., 92:92 + state['config']['extra_dim']] = 0.
                intervention, _ = model.generate(*altered, noise,
                    plan_override=plan['conditioning'], **kwargs)
                intervention_world = joints_from_features(intervention)[0].cpu().numpy()
                folder = output / arm['id']; folder.mkdir(exist_ok=True)
                body_path = folder / f'{song}.npz'
                intervention_path = folder / f'{song}_zero_local_audio.npz'
                np.savez_compressed(body_path, time=values['baseline'].time, body=world,
                                    native=candidate[0].cpu().numpy())
                np.savez_compressed(intervention_path, time=values['baseline'].time,
                                    body=intervention_world, native=intervention[0].cpu().numpy())
                measured = conditional_motion(world, values['quiet'], values['active'])
                reference = conditional_motion(values['baseline'].world, values['quiet'], values['active'])
                safety = {
                    'speed': measured['world']['speed_p95'] / max(reference['world']['speed_p95'], 1e-12),
                    'acceleration': measured['world']['acceleration_raw']['p95'] / max(reference['world']['acceleration_raw']['p95'], 1e-12),
                    'jerk': measured['world']['jerk_raw']['p95'] / max(reference['world']['jerk_raw']['p95'], 1e-12),
                }
                rel_response = float(np.sqrt(np.mean((root_relative(intervention_world) - root_relative(world)) ** 2)))
                rec = {
                    'arm': arm['id'], 'song': song,
                    'checkpoint': arm['checkpoint'], 'checkpoint_sha256': sha(arm['checkpoint']),
                    'body_path': str(body_path), 'body_sha256': sha(body_path),
                    'zero_local_audio_path': str(intervention_path), 'zero_local_audio_sha256': sha(intervention_path),
                    'noise_sha256': array_sha(noise), 'noise_reference_frames': max(config['noise_reference_frames'], len(values['base'])),
                    'candidate': measured, 'baseline': reference, 'ratios': {
                        key: measured[key] / max(reference[key], 1e-12)
                        for key in ('quiet_speed', 'active_speed', 'quiet_jerk_p95', 'quiet_active_ratio')
                    },
                    'raw_safety': safety,
                    'local_audio_zero_root_relative_rms_m': rel_response,
                    'finite': bool(np.isfinite(world).all() and np.isfinite(intervention_world).all()),
                    'status': 'diagnostic_only' if bool(np.isfinite(world).all()) and all(safety[k] <= v for k, v in protocol['safety_ratios'].items()) else 'rejected_raw_safety',
                    'inputs': {**values['binding'], 'normalized_fine_sha256': array_sha(args[0]),
                               'local_audio_intervention': 'fine extra audio channels zeroed; planner override held original',
                               'plan_override_sha256': array_sha(plan['conditioning'])},
                }
                write_json(folder / f'{song}.json', rec)
                report['rows'].append(rec)
                print(json.dumps({'arm': arm['id'], 'song': song, 'status': rec['status'],
                                  'local_audio_zero_root_relative_rms_m': rel_response,
                                  'ratios': rec['ratios']}), flush=True)
            del model
            torch.cuda.empty_cache()
    for name, digest in protocol['files'].items():
        if sha(name) != digest:
            raise ValueError('Input drift during audit: ' + name)
    report.update(status='complete', elapsed_s=time.monotonic() - started)
    write_json(output / 'REPORT.json', report)
    write_json(output / 'REPORT.sha256.json', {'sha256': sha(output / 'REPORT.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    generate(parser.parse_args().protocol)

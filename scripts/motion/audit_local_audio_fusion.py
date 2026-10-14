#!/usr/bin/env python3
"""Audit trained local audio fusion arms before mapper or rendering.

The intervention replaces only the cached keys/values of the new local module.
Legacy audio adapters, planner, stereo, gate energy, noise and clock stay fixed.
Earlier audits editing fine_audio also changed legacy adapters and were confounded.
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
from lightsem.experiments.native_dynamics.local_audio_fusion import masked_energy_quantile
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
            reference_file=Path(protocol['reference_root'])/f'{song}.npz'
            with np.load(reference_file) as stored:
                if not np.array_equal(stored['time'],reference.time):
                    raise ValueError('Baseline native clock differs')
                identity_delta=float(np.max(np.abs(stored['body']-reference.world)))
            if identity_delta>protocol['identity_max_abs_m']:
                raise ValueError('Baseline identity failed: '+song)
            folder=output/'baseline';folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'{song}.npz',time=reference.time,body=reference.world,native=reference.native)
            report.setdefault('baseline_identity',{})[song]={'max_absolute_m':identity_delta,'reference_sha256':sha(reference_file)}
            energy, quiet, active = acoustic_masks(audio, reference.time)
            songs[song] = dict(music=music, base=base, stereo=stereo, binding=binding,
                               baseline=reference, quiet=quiet, active=active, energy=energy)
            if not np.isfinite(reference.world).all():
                raise ValueError('Nonfinite baseline body: ' + song)
        for arm in protocol['arms']:
            state = torch.load(arm['checkpoint'], weights_only=False, map_location='cpu')
            model = NativeDynamicsModel(MusicConfig(**state['base_config']), state['normalizers'], Config(**state['config']))
            model.load_state_dict(state['model'], strict=True)
            changed=[k for k,v in baseline_state['model'].items() if k not in state['model'] or not torch.equal(v,state['model'][k])]
            if changed:raise ValueError('Frozen incumbent parameters changed: '+str(changed))
            model.cuda().eval().requires_grad_(False)
            for song, values in songs.items():
                args, kwargs = canonical_inputs(values['music'], values['base'], values['stereo'], state, config, 'cuda')
                noise = cpu_noise(model.flow, max(config['noise_reference_frames'], len(values['base'])), config['draw_seed'])
                noise = noise[:, :len(values['base'])].contiguous().cuda()
                if array_sha(noise) != values['baseline'].metadata['noise_sha256']:
                    raise ValueError('Canonical noise mismatch: ' + song)
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();inference_start=time.monotonic()
                candidate, plan = model.generate(*args, noise, **kwargs)
                torch.cuda.synchronize();inference_s=time.monotonic()-inference_start
                peak_bytes=torch.cuda.max_memory_allocated()
                world = joints_from_features(candidate)[0].cpu().numpy()
                # Only cached local keys/values change. Keep original energy
                # and threshold to isolate content from gate response.
                extra=args[0][...,92:92+model.config.extra_dim]
                reference_energy=None
                if model.config.audio_energy_relative:
                    reference_energy=masked_energy_quantile(extra[...,1040],torch.ones(extra.shape[:2],dtype=torch.bool,device=extra.device),model.config.audio_energy_quantile)
                context=model.flow.local_audio.prepare(extra,energy_reference=reference_energy)
                empty=model.flow.local_audio.prepare(torch.zeros_like(extra),energy_reference=reference_energy)
                local_override=(*empty[:3],*context[3:])
                intervention, _ = model.generate(*args, noise,local_context_override=local_override, **kwargs)
                intervention_world = joints_from_features(intervention)[0].cpu().numpy()
                # Removing the whole newly trained residual must reconstruct
                # the selected incumbent, not just yield similar movement.
                hook=model.flow.local_audio.register_forward_hook(lambda _m,_a,result:torch.zeros_like(result))
                try:
                    bypass,_=model.generate(*args,noise,**kwargs)
                finally:hook.remove()
                bypass_world=joints_from_features(bypass)[0].cpu().numpy()
                bypass_delta=float(np.max(np.abs(bypass_world-values['baseline'].world)))
                if bypass_delta>protocol['identity_max_abs_m']:
                    raise ValueError(f'Local bypass identity failed: {arm["id"]}/{song}: {bypass_delta}')
                folder = output / arm['id']; folder.mkdir(exist_ok=True)
                body_path = folder / f'{song}.npz'
                intervention_path = folder / f'{song}_local_train_mean_gate_fixed.npz'
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
                    'baseline_body':str(output/'baseline'/f'{song}.npz'),
                    'baseline_body_sha256':sha(output/'baseline'/f'{song}.npz'),
                    'bypass_identity_max_absolute_m':bypass_delta,
                    'candidate_minus_baseline_root_relative_rms_m':float(np.sqrt(np.mean((root_relative(world)-root_relative(values['baseline'].world))**2))),
                    'inference_seconds':inference_s,'peak_allocated_gpu_bytes':peak_bytes,
                    'candidate': measured, 'baseline': reference, 'ratios': {
                        key: measured[key] / max(reference[key], 1e-12)
                        for key in ('quiet_speed', 'active_speed', 'quiet_jerk_p95', 'quiet_active_ratio')
                    },
                    'raw_safety': safety,
                    'local_audio_zero_root_relative_rms_m': rel_response,
                    'finite': bool(np.isfinite(world).all() and np.isfinite(intervention_world).all()),
                    'status': 'diagnostic_only' if bool(np.isfinite(world).all()) and all(safety[k] <= v for k, v in protocol['safety_ratios'].items()) else 'rejected_raw_safety',
                    'inputs': {**values['binding'], 'normalized_fine_sha256': array_sha(args[0]),
                               'local_audio_intervention': 'only local cached keys/values use training-mean descriptors; gate/legacy fine/coarse/stereo/noise unchanged',
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

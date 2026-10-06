"""Frozen two-song body-to-N-output engineering demonstration.

Uses the user-accepted old body generator without training. Color is P2.
This pipeline does not import the rhythm companion or use its motion bytes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

from lightsem.conductor.body_transfer import TransferConfig, build_outputs, sample_body, constrained_transfer, raw_transfer, decode


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def read(path):
    obj = json.loads(Path(path).read_text())
    return obj.get('payload', obj)


def check(plan):
    for path, expected in plan['files'].items():
        if sha(path) != expected:
            raise RuntimeError('Frozen input changed: ' + path)
    if min(shutil.disk_usage('/root').free, shutil.disk_usage('/mnt').free) < 2 ** 30:
        raise RuntimeError('Free-space floor')
    root = Path(plan['output'])
    if root.exists() and sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > plan['max_bytes']:
        raise RuntimeError('Artifact budget')


def colors(plan, track):
    """Read or reconstruct the exact accepted P2 RGB policy, never its motion."""
    if track['id'] == 'shpongle':
        with np.load(plan['baseline']['reference_streams']) as data:
            stream = data['candidate'][:plan['prefix_s'] * 40]
    else:
        from lightsem.experiments.visual_next.delivery import color_delivery
        with np.load(track['old_color_streams']) as data:
            old = data['candidate'].copy()
            companion_rgb = data['companion'].copy()
            times = data['time'].copy()
        with np.load(track['color_outputs']) as data:
            ids, cues = data['generated_palette_ids'], data['color_cues']
        transforms = read(plan['color_transforms'])
        stream, report = color_delivery(old, companion_rgb, times, ids, cues,
            transforms['palette_prototypes'], layout=plan['baseline']['color_layout'])
        assert report['layout'] == 'two_groups'
    # mh1 and mh6 are the left/right P2 colors in the canonical reference.
    from lightsem.conductor.rig import canonical_rig
    heads = sorted(canonical_rig().by_kind('moving_head'), key=lambda f: f.position[0])
    slots = [f.address - 1 + np.array([6, 7, 8]) for f in (heads[0], heads[-1])]
    return stream[:, slots[0]].copy(), stream[:, slots[1]].copy()


def generate(plan):
    import torch
    from lightsem.experiments.motion_study.music_model import MusicModel, MusicConfig
    from lightsem.experiments.motion_study.geometry import joints_from_features
    from lightsem.experiments.stage11.stage11_common import gpu_lock
    from lightsem.model.light_data import AUDIO_SCALARS
    check(plan)
    root = Path(plan['output'])
    root.mkdir(exist_ok=False)
    cfg = TransferConfig(**plan['transfer'])
    outputs = build_outputs(**plan['counts'])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    worlds = {}
    with gpu_lock():
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU already in use')
        for track in plan['tracks']:
            out = root / track['id']
            out.mkdir()
            if track['id'] == 'shpongle':
                with np.load(track['body']) as data:
                    world = data['world'][:plan['prefix_s'] * 30].copy()
                provenance = {'source': track['body'], 'sha256': sha(track['body']),
                    'route': 'exact user-reviewed Shpongle body, prefix of full-song inference',
                    'original_draw_seed': plan['draw_seed']}
            else:
                # Motion model's historical Base92 contract rounds cached mel
                # through float16 BEFORE interpolation and TRAIN normalization.
                with np.load(track['audio_bus']) as data:
                    mel = data['mel'].astype(np.float16).astype(np.float32)
                    raw40 = np.concatenate((mel.T, np.stack([data['ch__' + k] for k in AUDIO_SCALARS], 1)), 1)
                    t40 = data['times'].copy()
                t30 = np.arange(plan['prefix_s'] * 30) / 30
                raw30 = np.stack([np.interp(t30, t40, raw40[:, j]) for j in range(92)], 1).astype(np.float32)
                normalizers = read(plan['audio_normalizers'])
                normalized = ((raw30 - np.asarray(normalizers['mean'])) / np.asarray(normalizers['std'])).astype(np.float32)
                np.savez_compressed(out / 'body_audio_input.npz', raw30=raw30, normalized=normalized, time=t30)
                state = torch.load(plan['body_checkpoint'], map_location='cpu', weights_only=False)
                model = MusicModel(MusicConfig(arm='flow'), read(plan['motion_normalizers'])).cuda().eval()
                model.load_state_dict(state['model'])
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                with torch.inference_mode():
                    features = model.generate(torch.from_numpy(normalized[None]).cuda(), seed=plan['draw_seed'])[0].cpu()
                    world = joints_from_features(features).numpy()
                reserved = torch.cuda.max_memory_reserved()
                if reserved > plan['max_reserved_gpu_bytes']:
                    raise RuntimeError('VRAM budget')
                np.savez_compressed(out / 'body_features.npz', features=features.numpy())
                provenance = {'source_audio': track['audio'], 'audio_sha256': sha(track['audio']),
                    'source_bus': track['audio_bus'], 'checkpoint': plan['body_checkpoint'],
                    'checkpoint_sha256': sha(plan['body_checkpoint']), 'draw_seed': plan['draw_seed'],
                    'input_sha256': sha(out / 'body_audio_input.npz'), 'elapsed_s': time.monotonic() - started,
                    'peak_reserved_gpu_bytes': reserved, 'route': 'fresh Cure For Me audio-conditioned body; no training'}
                del model, state, features
                torch.cuda.empty_cache()
            assert world.shape == (plan['prefix_s'] * 30, 22, 3) and np.isfinite(world).all()
            worlds[track['id']] = world
            np.savez_compressed(out / 'body.npz', world=world, time=np.arange(len(world)) / 30)
            save(out / 'BODY_PROVENANCE.json', provenance)
            print('BODY_READY', track['id'], flush=True)
    assert not np.array_equal(worlds['shpongle'], worlds['cure_for_me'])
    reports = []
    for track in plan['tracks']:
        out = root / track['id']
        times, body = sample_body(worlds[track['id']])
        left, right = colors(plan, track)
        assert len(times) == len(left) == plan['prefix_s'] * 40
        dmx, patch, raw = constrained_transfer(body, outputs, left, right, cfg)
        emitted = decode(dmx, patch)
        moving = np.array([p['kind'] != 'bulb' for p in patch])
        angles = emitted['angles_deg'][:, moving]
        speed = np.linalg.norm(np.diff(angles, axis=0) * cfg.fps, axis=-1)
        acceleration = np.linalg.norm(np.diff(angles, n=2, axis=0) * cfg.fps ** 2, axis=-1)
        assert speed.max() <= cfg.maximum_speed_deg_s + 1e-6
        assert acceleration.max() <= cfg.maximum_acceleration_deg_s2 + 1e-6
        # All-output constant-body control, including quantization and dimmer.
        frozen = np.repeat(body[0:1], 80, axis=0)
        control, _, _ = constrained_transfer(frozen, outputs, np.repeat(left[:1], 80, 0), np.repeat(right[:1], 80, 0), cfg)
        assert np.all(control == control[:1])
        np.savez_compressed(out / 'timeline.npz', universes=dmx, time=times,
            body=body, targets=raw['targets'], source_points=raw['source_points'],
            bulb_field_centers=raw['bulb_field_centers'])
        save(out / 'patch.json', patch)
        ranges = np.ptp(angles, axis=0)
        unique = len({hashlib.sha256(angles[:, i].tobytes()).hexdigest() for i in range(angles.shape[1])})
        color_changes = int(np.count_nonzero(np.any(np.diff(left.astype(int), axis=0), axis=1)))
        report = {'track': track['id'], 'count_by_kind': plan['counts'], 'frames40': len(dmx),
            'stationary_body_DMX_exact': True, 'moving_outputs': int(moving.sum()),
            'unique_moving_trajectories': unique, 'angle_ranges_deg': ranges.tolist(),
            'speed_max_deg_s': float(speed.max()), 'acceleration_max_deg_s2': float(acceleration.max()),
            'median_moving_speed_deg_s': float(np.median(speed)),
            'bulb_dimmer_range': np.ptp(emitted['dimmer'][:, ~moving], axis=0).tolist(),
            'RGB_changing_ticks_including_fades': color_changes,
            'timeline_sha256': sha(out / 'timeline.npz'), 'body_sha256': sha(out / 'body.npz'),
            'no_independent_motion_clock': True, 'color_baseline': 'P2',
            'scope': 'Engineering transfer diagnostics; not a claim of artistic success or fresh model qualification'}
        save(out / 'DIAGNOSTICS.json', report)
        reports.append(report)
        check(plan)
        print('N_OUTPUTS_MAPPED', track['id'], unique, 'distinct trajectories', flush=True)
    save(root / 'GENERATION.json', {'status': 'two_song_bodies_and_outputs_ready', 'reports': reports,
        'body_arrays_distinct': True, 'transfer': asdict(cfg), 'inputs_rechecked': True})


def render(plan):
    from scripts.rendering.render_body_transfer import render_one
    from scripts.motion.run_rhythm_companion import audit_movie
    check(plan)
    root = Path(plan['output'])
    reports = read(root / 'GENERATION.json')['reports']
    movies = []
    for track in plan['tracks']:
        out = root / track['id']
        report = next(row for row in reports if row['track'] == track['id'])
        assert sha(out / 'timeline.npz') == report['timeline_sha256']
        movie = render_one(out, track, plan)
        audit = audit_movie(out, movie)
        assert audit['decoded_frames'] == plan['review_duration_s'] * 30
        # Independent AAC encoding of the same source crop verifies soundtrack.
        reference = out / 'audio_reference.m4a'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-ss', str(plan['review_start_s']),
            '-i', track['audio'], '-t', str(plan['review_duration_s']), '-vn', '-c:a', 'aac', '-b:a', '128k', str(reference)], check=True)
        pcm = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(reference),
            '-ac', '2', '-ar', '44100', '-f', 'f32le', '-'])
        assert hashlib.sha256(pcm).hexdigest() == audit['pcm_sha256']
        movies.append({'id': track['id'], 'path': str(movie), 'sha256': sha(movie),
            'bytes': movie.stat().st_size, 'audio_matches_independent_source_crop': True,
            'decoded_frames': audit['decoded_frames'], 'audit': str(out / (movie.stem + '_audit.json'))})
        check(plan)
    save(root / 'DELIVERY.json', {'status': 'two_body_transfer_demos_complete', 'movies': movies,
        'scope': 'Body + decoded 8 beams/5 virtual lasers/50 bulbs. Geometric transfer preview; user evaluates choreography.'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('generate', 'render'))
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = read(args.plan)
    {'generate': generate, 'render': render}[args.stage](plan)


if __name__ == '__main__':
    main()

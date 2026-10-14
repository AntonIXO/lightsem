#!/usr/bin/env python3
"""Render a raw-safe local-audio candidate through the canonical native path.

This is a diagnostic delivery only. The raw-body gate must already be recorded
by the paired audit; mapping adds the independent source-dependence gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lightsem.dance.baseline import load_config
from lightsem.dance.lighting import map_lights
from lightsem.dance.render import render_video
from lightsem.dance.pipeline import write_json
from lightsem.experiments.dynamic_dance.contracts import DanceResult


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    candidate_root = args.candidate_root.expanduser().resolve()
    body_path = candidate_root / f'{args.song}.npz'
    if not body_path.is_file():
        raise FileNotFoundError(body_path)
    output_root = args.out.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    with np.load(body_path, allow_pickle=False) as values:
        time = values['time'].astype(np.float64)
        world = values['body'].astype(np.float32)
        native = values['native'].astype(np.float32)
    if (world.ndim != 3 or world.shape[1:] != (22, 3) or len(time) != len(world)
            or native.shape[0] != len(world) or not np.isfinite(world).all()
            or not np.isfinite(native).all()):
        raise ValueError('Candidate body is nonfinite or has an invalid shape')
    audio = args.audio.expanduser().resolve()
    body = DanceResult(
        time=time, world=world, contacts=np.zeros((len(world), 4), dtype=np.float32), native=native,
        metadata={
            'audio': {'audio_sha256': sha(audio)}, 'd4max_enabled': False, 'd6_enabled': False,
            'newly_generated_motion': True, 'diagnostic_only': True,
            'display_title': f'Fixed C | Local {args.fusion} diagnostic',
        })
    output_root.mkdir(parents=True)
    body.save(output_root / 'body.npz')
    np.savez_compressed(output_root / 'plan.npz', time=time, source_time=time,
                        effort=np.ones((len(time), 1), dtype=np.float32))
    generation = {
        'schema_version': 1, 'status': 'finite_body_ready', 'diagnostic_only': True,
        'candidate': f'local_audio_{args.fusion}', 'song': args.song,
        'audio': str(audio), 'audio_sha256': sha(audio),
        'candidate_source_body': str(body_path), 'candidate_source_body_sha256': sha(body_path),
        'body_sha256': sha(output_root / 'body.npz'), 'body_fps': 30,
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': sha(args.checkpoint),
        'source_body_used': False, 'd4max': False, 'd6': False,
        'raw_safety_gate': 'passed before mapping; see paired raw audit receipt',
        'musical_superiority_claim': False,
    }
    write_json(output_root / 'GENERATION.json', generation)
    config, _ = load_config()
    lighting = map_lights(body, audio, output_root, config)
    source = lighting[3].get('source_dependence_checks', {})
    if source.get('status') != 'pass':
        raise RuntimeError(f'source-dependence gate failed: {source}')
    output = output_root / f'{args.song}_local_{args.fusion}_{args.mode}.mp4'
    render_video(body, output_root, audio, output, lighting, duration=args.duration)
    result = {'status': 'diagnostic_only', 'song': args.song, 'fusion': args.fusion,
              'mode': args.mode, 'video': str(output), 'video_sha256': sha(output),
              'run': str(output_root), 'source_dependence': source.get('status'),
              'duration_s': args.duration, 'lighting_receipt': lighting[3]}
    write_json(output_root / 'DELIVERY.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-root', type=Path, required=True)
    parser.add_argument('--song', choices=('shpongle', 'cure_for_me'), required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fusion', default='hybrid_strict')
    parser.add_argument('--mode', choices=('smoke', 'full'), required=True)
    parser.add_argument('--duration', type=float, required=True)
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError('duration must be positive')
    print(json.dumps(run(args), indent=2, sort_keys=True))

#!/usr/bin/env python
"""Render short previews of the isolated continuous-color field.

The preview consumes actual decoded DMX from color_motion_field and the saved
native body. It is diagnostic-only; Shpongle uses the archived authenticated
reference soundtrack because the original FLAC is absent from this checkout.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from lightsem.dance.baseline import sha
from lightsem.experiments.dynamic_dance.contracts import DanceResult
from lightsem.conductor.body_transfer import decode
from lightsem.dance.render import render_video


COLOR_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/color_motion_field_20261015_v4')
BODY_ROOT = Path('/mnt/lightsem-data/experiments/lightsem/native_full_native_repair_20261018_v2/render_full')
AUDIO = {
    'shpongle': Path('/mnt/lightsem-data/experiments/lightsem/d4max_20261005_v1/audio/shpongle/reference.m4a'),
    'cure_for_me': Path('/root/lightsem/experiments/native_bf16_v7b/demo_audio/Cure_For_Me.wav'),
}
DEFAULT_STARTS = {'shpongle': 106., 'cure_for_me': 23.}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def one(song, experiment_root, out, duration, start):
    source_body_path = BODY_ROOT / song / 'body.npz'
    source = DanceResult.load(source_body_path)
    audio = AUDIO[song]
    if not audio.is_file():
        raise FileNotFoundError(audio)
    field = experiment_root / song / 'color_field.npz'
    patch_path = experiment_root / song / 'color_field_patch.json'
    with np.load(field, allow_pickle=False) as z:
        dmx = z['dmx'].copy()
        time = z['time'].copy()
    patch = json.loads(patch_path.read_text())
    controls = decode(dmx, patch)
    root = out / song
    root.mkdir(parents=True, exist_ok=False)
    if audio.suffix.lower() == '.m4a':
        # libsndfile (used for duration validation) cannot inspect AAC. Keep a
        # deterministic decoded reference beside this diagnostic render.
        decoded = root / 'archived_reference.wav'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                        '-i', str(audio), '-ar', '44100', '-ac', '2', str(decoded)], check=True)
        audio = decoded
    metadata = dict(source.metadata)
    metadata.update({'audio': {'audio_sha256': sha(audio)}, 'diagnostic_only': True,
                     'd4max_enabled': False, 'd6_enabled': False,
                     'display_title': 'Continuous OKLab gradient | 01 distributed readout'})
    body_path = root / 'body.npz'
    body = DanceResult(time=source.time.copy(), world=source.world.copy(), native=source.native.copy(),
                       contacts=source.contacts.copy() if source.contacts is not None else None,
                       metadata=metadata)
    body.save(body_path)
    shutil.copy2(BODY_ROOT / song / 'plan.npz', root / 'plan.npz')
    shutil.copy2(field, root / 'lights.npz')
    shutil.copy2(patch_path, root / 'patch.json')
    write(root / 'GENERATION.json', {
        'schema_version': 1, 'status': 'finite_body_ready', 'diagnostic_only': True,
        'candidate': 'color_field_01_distributed', 'song': song,
        'audio': str(audio), 'audio_sha256': sha(audio), 'body_sha256': sha(body_path),
        'candidate_source_body': str(source_body_path), 'candidate_source_body_sha256': sha(source_body_path),
        'raw_safety_gate': {'status': 'pass', 'receipt': str(COLOR_ROOT / song / 'RAW_SAFETY.json'),
                            'receipt_sha256': sha(COLOR_ROOT / song / 'RAW_SAFETY.json')},
        'motion_origin': 'saved native body; no new generation or temporal fallback',
        'soundtrack_scope': 'archived full-song reference for Shpongle' if song == 'shpongle' else 'own-song source WAV',
    })
    write(root / 'LIGHTING.json', {
        'schema_version': 1, 'status': 'diagnostic_only', 'mapper': '01_distributed_gain',
        'display_label': 'OKLab flowing gradient | source-driven color field',
        'color_experiment': str(experiment_root / song / 'COLOR_FIELD.json'),
        'color_experiment_sha256': sha(experiment_root / song / 'COLOR_FIELD.json'),
        'lights_sha256': sha(root / 'lights.npz'), 'patch_sha256': sha(root / 'patch.json'),
        'body_sha256': sha(body_path), 'fixture_motion_limits': False,
        'actual_dmx_decoded': True, 'source_dependence_status': 'pass',
        'brightness_scope': 'historical LUFS/body dimmer; color and brightness audited separately',
    })
    output = root / f'{song}_continuous_oklab_gradient.mp4'
    render_video(body, root, audio, output, (time, patch, controls, {}), start=start, duration=duration)
    return {'song': song, 'video': str(output), 'video_sha256': sha(output),
            'body_sha256': sha(body_path), 'audio': str(audio), 'start_s': start,
            'duration_s': duration}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=COLOR_ROOT)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--duration', type=float, default=18.)
    ap.add_argument('--start', type=float, help='override the active-song start for both previews')
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    out = (args.out or root / 'demo').expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'preserve existing demo root: {out}')
    out.mkdir(parents=True)
    rows = [one(song, root, out, args.duration,
                DEFAULT_STARTS[song] if args.start is None else args.start)
            for song in ('shpongle', 'cure_for_me')]
    write(out / 'REPORT.json', {'status': 'diagnostic_only', 'rows': rows,
        'scope': '18-second own-song previews of saved actual emitted DMX; no baseline promotion'})
    print(json.dumps({'out': str(out), 'status': 'diagnostic_only', 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()


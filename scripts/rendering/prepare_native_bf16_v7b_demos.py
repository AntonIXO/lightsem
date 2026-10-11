"""Prepare and render the saved native_bf16_v7b body/DMX demo clips.

This is a presentation-only adapter. It never regenerates a body, changes the
mapper, or changes the selected baseline. The source body and decoded DMX are
copied into a new, hash-bound demo root before the existing geometric renderer
is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path('/root/lightsem/experiments/native_bf16_v7b')
OUT = ROOT / 'demo_videos_v1'
SMOKE = ROOT / 'demo_videos_v1_smoke'
SONGS = {
    'shpongle': Path('/root/shpongle.flac'),
    'cure_for_me': ROOT / 'demo_audio/Cure_For_Me.wav',
}
COUNTS = {'beams': 8, 'lasers': 5, 'bulbs': 50}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def patch_for_saved_emission(song: str, source_emission: Path, audio: Path) -> list[dict]:
    """Recompile only the fixed mapper metadata and authenticate its DMX bytes."""
    from lightsem.dance.lighting import _load_p2_colors
    from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
    from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
    from lightsem.conductor.body_transfer_v2 import make_outputs
    combo = json.loads(Path(
        '/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1/PLAN.json'
    ).read_text())
    with np.load(source_emission, allow_pickle=False) as data:
        time40 = data['time'].astype(np.float64, copy=True)
        world40 = data['body'].astype(np.float32, copy=True)
        saved = data['universes'].astype(np.uint8, copy=True)
    frames = len(time40)
    colors = ROOT / 'demo_colors_v2' / song / 'P2.npz'
    loudness, _ = load_verified_loudness(ROOT / 'demo_loudness' / song / 'loudness.json', audio)
    left, right, _, _, _ = _load_p2_colors(colors, audio, frames)
    outputs = make_outputs(**combo['counts'])
    rebuilt, patch, _, _, _ = compile_world(
        world40, time40, outputs, left, right, loudness, combo['transfer'], mapper_id='01'
    )
    # The saved emission is immutable.  Rebuilding only recovers the patch
    # metadata needed by the decoder; it is deliberately never substituted for
    # the authenticated saved universes.
    if rebuilt.shape != saved.shape or rebuilt.dtype != saved.dtype:
        raise ValueError(f'Mapper metadata shape drift: {song}')
    return patch


def prepare(root: Path) -> None:
    if root.exists():
        raise FileExistsError(f'Preserving existing demo root: {root}')
    root.mkdir(parents=True)
    first_song = next(iter(SONGS))
    first_emission = ROOT / 'demo_emission/joint_seed0' / f'{first_song}.npz'
    patch = patch_for_saved_emission(first_song, first_emission, SONGS[first_song])
    patch_path = root / 'patch.json'
    write_json(patch_path, patch)
    rows = {}
    for song, audio in SONGS.items():
        source_body = ROOT / 'demo_bodies/joint_seed0' / f'{song}.npz'
        source_emission = ROOT / 'demo_emission/joint_seed0' / f'{song}.npz'
        if not source_body.is_file() or not source_emission.is_file() or not audio.is_file():
            raise FileNotFoundError(f'Missing source for {song}')
        with np.load(source_body, allow_pickle=False) as body_data:
            body = body_data['body'].astype(np.float32, copy=True)
            native_time = body_data['time'].astype(np.float64, copy=True)
        with np.load(source_emission, allow_pickle=False) as emission_data:
            time40 = emission_data['time'].astype(np.float64, copy=True)
            world40 = emission_data['body'].astype(np.float32, copy=True)
            universes = emission_data['universes'].astype(np.uint8, copy=True)
        if body.ndim != 3 or body.shape[1:] != (22, 3) or not np.isfinite(body).all():
            raise ValueError(f'Invalid native body: {song}')
        if world40.ndim != 3 or world40.shape[1:] != (22, 3) or not np.isfinite(world40).all():
            raise ValueError(f'Invalid mapped body: {song}')
        if universes.ndim != 3 or universes.shape[1:] != (1, 512):
            raise ValueError(f'Invalid DMX shape: {song}')
        if len(time40) != len(world40) or len(time40) != len(universes):
            raise ValueError(f'Clock mismatch: {song}')
        if not np.array_equal(time40, np.arange(len(time40), dtype=np.float64) / 40):
            raise ValueError(f'Noncanonical 40 Hz clock: {song}')
        target = root / song
        target.mkdir()
        (target / 'patch.json').write_text(patch_path.read_text())
        timeline = target / 'timeline.npz'
        # Keep the authenticated (T, universe, 512) DMX schema expected by the
        # decoder and renderer.
        np.savez_compressed(timeline, time=time40, body=world40, universes=universes)
        rows[song] = {
            'source_body': str(source_body),
            'source_body_sha256': sha(source_body),
            'source_native_body_shape': list(body.shape),
            'source_native_time_sha256': hashlib.sha256(native_time.tobytes()).hexdigest(),
            'source_emission': str(source_emission),
            'source_emission_sha256': sha(source_emission),
            'timeline': str(timeline),
            'timeline_sha256': sha(timeline),
            'audio': str(audio),
            'audio_sha256': sha(audio),
            'body_30hz_frames': int(len(body)),
            'lighting_40hz_frames': int(len(time40)),
            'duration_s': float(len(time40) / 40),
            'counts': COUNTS,
        }
    manifest = {
        'schema_version': 1,
        'status': 'prepared',
        'source_root': str(ROOT),
        'checkpoint': str(ROOT / 'learning/runs/joint/seed_0/final.pt'),
        'checkpoint_sha256': sha(ROOT / 'learning/runs/joint/seed_0/final.pt'),
        'patch': str(patch_path),
        'patch_sha256': sha(patch_path),
        'review_start_s': 0.0,
        # New review generations use 90 seconds; existing roots are immutable.
        'review_duration_s': 90.0,
        'songs': rows,
        'scope': 'Presentation adapter over saved native body and decoded mapper-01 DMX; no body or DMX regeneration.',
    }
    write_json(root / 'MANIFEST.json', manifest)


def render(root: Path, smoke: bool, song: str) -> None:
    from scripts.rendering.render_body_transfer import render_one
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    row = manifest['songs'][song]
    out = root / song
    track = {'id': song, 'title': 'Shpongle' if song == 'shpongle' else 'AURORA — Cure For Me',
             'audio': row['audio']}
    plan = {'review_start_s': 0.0, 'review_duration_s': 1.0 if smoke else 90.0}
    render_one(out, track, plan)
    render_record = out / f'{song}_body_to_63_P2_1080p_av1_render.json'
    movie = out / f'{song}_body_to_63_P2_1080p_av1.mp4'
    record = json.loads(render_record.read_text())
    record.update({'demo_manifest_sha256': sha(root / 'MANIFEST.json'),
                   'patch_sha256': sha(root / 'patch.json'),
                   'timeline_sha256': sha(out / 'timeline.npz'),
                   'checkpoint_sha256': manifest['checkpoint_sha256'],
                   'smoke': smoke,
                   'audio_source_sha256': sha(Path(row['audio'])),
                   'movie_sha256': sha(movie)})
    write_json(out / ('SMOKE_RENDER.json' if smoke else 'RENDER.json'), record)


def audit(root: Path, song: str) -> None:
    root_manifest = json.loads((root / 'MANIFEST.json').read_text())
    row = root_manifest['songs'][song]
    out = root / song
    movie = out / f'{song}_body_to_63_P2_1080p_av1.mp4'
    if not movie.is_file():
        raise FileNotFoundError(movie)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(movie)
    ], text=True))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    expected_frames = 90 * 30
    if int(video.get('nb_read_frames', -1)) != expected_frames:
        raise ValueError(f'Frame count mismatch for {song}: {video.get("nb_read_frames")}')
    if video.get('r_frame_rate') != '30/1':
        raise ValueError(f'Video rate mismatch: {video.get("r_frame_rate")}')
    if abs(float(video['duration']) - 90.0) > 1 / 30 or abs(float(audio['duration']) - 90.0) > .1:
        raise ValueError(f'Duration mismatch for {song}')
    frames = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(movie)
    ], text=True))['frames']
    timestamps = np.array([float(frame['best_effort_timestamp_time']) for frame in frames])
    if len(timestamps) != expected_frames or not np.allclose(
            timestamps, np.arange(expected_frames) / 30, rtol=0, atol=1e-5):
        raise ValueError(f'Nonuniform decoded PTS for {song}')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-hwaccel', 'none',
                    '-i', str(movie), '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], check=True)
    report = {
        'status': 'software_decode_valid',
        'song': song,
        'movie': str(movie),
        'movie_sha256': sha(movie),
        'decoded_frames': len(frames),
        'pts_uniform': True,
        'software_decode': True,
        'video_fps': video['r_frame_rate'],
        'video_duration_s': float(video['duration']),
        'audio_duration_s': float(audio['duration']),
        'source_audio': row['audio'],
        'source_audio_sha256': row['audio_sha256'],
        'timeline_sha256': sha(out / 'timeline.npz'),
        'patch_sha256': sha(root / 'patch.json'),
        'checkpoint_sha256': root_manifest['checkpoint_sha256'],
        'note': 'Audio is bound to the authenticated source path; this geometric renderer encodes an AAC delivery stream.',
    }
    write_json(out / 'AUDIT.json', report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'smoke', 'render', 'audit'))
    parser.add_argument('--song', choices=tuple(SONGS))
    parser.add_argument('--root', type=Path, default=OUT)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root)
    else:
        if args.song is None:
            parser.error('--song required')
        if args.action == 'smoke':
            render(args.root, True, args.song)
        elif args.action == 'render':
            render(args.root, False, args.song)
        else:
            audit(args.root, args.song)


if __name__ == '__main__':
    main()

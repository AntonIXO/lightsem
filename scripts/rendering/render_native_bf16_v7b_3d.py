"""Skill-compliant native 3D diagnostic videos for the v7b checkpoint.

This consumes the saved v7b native body and mapper-01 DMX only.  It does not
regenerate motion, change fixture limits, or promote the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/root/lightsem/experiments/native_bf16_v7b')
SONGS = {
    'shpongle': Path('/root/shpongle.flac'),
    'cure_for_me': ROOT / 'demo_audio/Cure_For_Me.wav',
}
FPS = 30
DURATION = 60.0
FRAMES = 1800
WIDTH, HEIGHT = 1280, 1024


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def source_data(song: str):
    body_path = ROOT / 'demo_bodies/joint_seed0' / f'{song}.npz'
    emission_path = ROOT / 'demo_emission/joint_seed0' / f'{song}.npz'
    with np.load(body_path, allow_pickle=False) as z:
        body = z['body'].astype(np.float32, copy=True)
        body_time = z['time'].astype(np.float64, copy=True)
    with np.load(emission_path, allow_pickle=False) as z:
        time40 = z['time'].astype(np.float64, copy=True)
        world40 = z['body'].astype(np.float32, copy=True)
        universes = z['universes'].astype(np.uint8, copy=True)
    if body.shape[1:] != (22, 3) or not np.isfinite(body).all():
        raise ValueError(f'Invalid native body for {song}')
    if time40.shape != (len(world40),) or not np.array_equal(time40, np.arange(len(time40)) / 40):
        raise ValueError(f'Invalid lighting clock for {song}')
    if universes.shape != (len(time40), 1, 512) or universes.dtype != np.uint8:
        raise ValueError(f'Invalid emitted DMX for {song}')
    return body_path, emission_path, body, body_time, time40, world40, universes


def patch_for_saved_emission(song: str, emission_path: Path, audio: Path):
    from lightsem.conductor.body_transfer_v2 import make_outputs
    from lightsem.dance.lighting import _load_p2_colors
    from lightsem.experiments.dynamic_dance_combo.brightness import load_verified_loudness
    from lightsem.experiments.dynamic_dance_combo.campaign import compile_world
    combo = json.loads(Path(
        '/mnt/lightsem-data/experiments/lightsem/dynamic_dance_combo_20261005_v1/PLAN.json'
    ).read_text())
    with np.load(emission_path, allow_pickle=False) as z:
        time40 = z['time'].astype(np.float64, copy=True)
        world40 = z['body'].astype(np.float32, copy=True)
        saved = z['universes'].astype(np.uint8, copy=True)
    colors = ROOT / 'demo_colors_v2' / song / 'P2.npz'
    loudness, _ = load_verified_loudness(ROOT / 'demo_loudness' / song / 'loudness.json', audio)
    left, right, _, _, _ = _load_p2_colors(colors, audio, len(time40))
    rebuilt, patch, intent, _, _ = compile_world(
        world40, time40, make_outputs(**combo['counts']), left, right, loudness,
        combo['transfer'], mapper_id='01')
    if rebuilt.shape != saved.shape or rebuilt.dtype != saved.dtype:
        raise ValueError(f'Mapper patch schema drift for {song}')
    return patch, intent


def pcm_sha(path: Path) -> str:
    proc = subprocess.run([
        'ffmpeg', '-v', 'error', '-nostdin', '-i', str(path), '-map', '0:a:0',
        '-f', 's16le', '-ac', '2', '-ar', '44100', '-'
    ], check=True, stdout=subprocess.PIPE)
    return hashlib.sha256(proc.stdout).hexdigest()


def prepare(root: Path) -> None:
    if root.exists():
        raise FileExistsError(f'Preserving existing root: {root}')
    root.mkdir(parents=True)
    rows = {}
    for song, audio in SONGS.items():
        body_path, emission_path, body, body_time, time40, world40, universes = source_data(song)
        patch, intent = patch_for_saved_emission(song, emission_path, audio)
        target = root / song
        target.mkdir()
        (target / 'patch.json').write_text(json.dumps(patch, indent=2) + '\n')
        np.savez_compressed(target / 'timeline.npz', time=time40, body=world40, universes=universes,
                            targets=np.asarray(intent['targets'], dtype=np.float32))
        rows[song] = {
            'audio': str(audio), 'audio_sha256': sha(audio),
            'body': str(body_path), 'body_sha256': sha(body_path),
            'body_frames': int(len(body)), 'body_time_sha256': hashlib.sha256(body_time.tobytes()).hexdigest(),
            'emission': str(emission_path), 'emission_sha256': sha(emission_path),
            'timeline': str(target / 'timeline.npz'), 'timeline_sha256': sha(target / 'timeline.npz'),
            'patch': str(target / 'patch.json'), 'patch_sha256': sha(target / 'patch.json'),
            'lighting_frames': int(len(time40)), 'duration_s': float(len(time40) / 40),
        }
    write_json(root / 'MANIFEST.json', {
        'schema_version': 1, 'status': 'prepared', 'source_root': str(ROOT), 'songs': rows,
        'checkpoint': str(ROOT / 'learning/runs/joint/seed_0/final.pt'),
        'checkpoint_sha256': sha(ROOT / 'learning/runs/joint/seed_0/final.pt'),
        'review_start_s': 0.0, 'review_duration_s': DURATION,
        'scope': 'Native 3D presentation over saved v7b body and mapper-01 DMX; diagnostic only.',
    })


def controls(song: str, root: Path):
    from lightsem.conductor.body_transfer import decode
    row = json.loads((root / 'MANIFEST.json').read_text())['songs'][song]
    patch = json.loads(Path(row['patch']).read_text())
    with np.load(row['timeline'], allow_pickle=False) as z:
        time40, world40, universes = z['time'].copy(), z['body'].copy(), z['universes'].copy()
        targets = z['targets'].copy() if 'targets' in z.files else None
    decoded = decode(universes, patch)
    return patch, time40, world40, decoded, targets


def traces(body: np.ndarray, body_time: np.ndarray, decoded: dict, time40: np.ndarray) -> dict:
    from scripts.rendering.native_repair_traces import body_traces, motor_speed
    values = body_traces(body, body_time)
    moving = np.array([True] * 13 + [False] * 50)
    values['emitted'] = motor_speed(decoded['angles_deg'][:, moving], time40)
    beams = np.array([p['kind'] == 'beam' for p in json.loads('[]')]) if False else None
    return values


def make_chart(song: str, root: Path, out: Path, duration: float = DURATION) -> dict:
    manifest_row = json.loads((root / 'MANIFEST.json').read_text())['songs'][song]
    cache_receipt = out.with_suffix('.json')
    cache_inputs = {
        'song': song,
        'body_sha256': manifest_row['body_sha256'],
        'timeline_sha256': manifest_row['timeline_sha256'],
        'audio_sha256': manifest_row['audio_sha256'],
        'duration_s': duration,
    }
    if out.with_suffix('.png').is_file() and out.with_suffix('.npz').is_file() and cache_receipt.is_file():
        cached = json.loads(cache_receipt.read_text())
        if cached.get('cache_inputs') == cache_inputs:
            return cached
    body_path, _, body, body_time, time40, _, universes = source_data(song)
    patch, time40, world40, decoded, targets = controls(song, root)
    from scripts.rendering.native_repair_traces import body_traces, motor_speed, pixel_envelope
    values = body_traces(body, body_time)
    moving = np.array([p['kind'] != 'bulb' for p in patch])
    beams = np.array([p['kind'] == 'beam' for p in patch])
    values['emitted'] = motor_speed(decoded['angles_deg'][:, moving], time40)
    positions = np.asarray([p['position'] for p in patch])[moving]
    vectors = targets[:, moving] - positions[None]
    lengths = np.linalg.norm(vectors, axis=-1)
    requested = np.stack((np.rad2deg(np.arctan2(vectors[..., 0], -vectors[..., 1])),
                          np.rad2deg(np.arcsin(np.clip(vectors[..., 2] / lengths, -1, 1)))), -1)
    values['requested'] = motor_speed(requested, time40)
    values['brightness'] = (time40, decoded['dimmer'][:, beams].mean(1))
    import soundfile as sf
    audio = Path(json.loads((root / 'MANIFEST.json').read_text())['songs'][song]['audio'])
    samples, rate = sf.read(audio, dtype='float32', always_2d=True)
    samples = samples[:min(len(samples), int(duration * rate))]
    mono = samples.mean(1); edges = np.rint(np.arange(0, duration + 1/40, 1/40) * rate).astype(int); edges[-1] = len(mono)
    wave_t = (np.arange(len(edges)-1) + .5) / 40
    wave_min = np.array([mono[a:b].min() for a, b in zip(edges[:-1], edges[1:])]); wave_max = np.array([mono[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
    rows = [('waveform', 'Source waveform'), ('left_arm', 'Raw left arm'), ('right_arm', 'Raw right arm'), ('lower', 'Raw lower body'),
            ('body_jerk', 'World RMS jerk'), ('root_y', 'Root Y'), ('requested', 'Requested motors'), ('emitted', 'Emitted motors'),
            ('brightness', 'Beam dimmer')]
    chart = Image.new('RGB', (1280, 380), (12, 18, 27)); draw = ImageDraw.Draw(chart)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    draw.text((10, 4), f'{song} | raw native body and actual emitted DMX | no smoothing', font=font, fill='#d1e1ed')
    arrays = {}
    for i, (key, label) in enumerate(rows):
        if key == 'waveform':
            t, v = wave_t, wave_max
            v_lo = wave_min
        else:
            t, v = values[key]; v_lo = v
        mask = (t >= 0) & (t <= duration)
        t, v = np.asarray(t)[mask], np.asarray(v)[mask]
        lo, hi = float(np.min(v_lo)), float(np.max(v))
        if key == 'waveform':
            lo, hi = min(lo, -1e-8), max(hi, 1e-8)
        elif key == 'brightness': lo, hi = 0., 1.
        elif key not in ('root_y',): lo, hi = 0., max(hi, 1e-8)
        elif hi - lo < 1e-8: lo, hi = lo - .01, hi + .01
        y = 30 + i * 40; draw.text((10, y), label, font=font, fill='#c1d2e2'); draw.text((160, y), f'{lo:.3g}..{hi:.3g}', font=font, fill='#9bb5cc')
        low, high = pixel_envelope(t, v, left=0., right=duration, pixels=1050)
        if key == 'waveform':
            low, _ = pixel_envelope(t, v_lo, left=0., right=duration, pixels=1050)
        for x in np.flatnonzero(np.isfinite(low)):
            draw.line((215 + int(x), y + 26 - (low[x]-lo)/(hi-lo)*22,
                       215 + int(x), y + 26 - (high[x]-lo)/(hi-lo)*22), fill='#4bd6e6')
        arrays[f'{key}_time'] = t; arrays[f'{key}_raw'] = v
        if key == 'waveform': arrays['waveform_minimum'] = v_lo; arrays['waveform_maximum'] = v
    out.parent.mkdir(parents=True, exist_ok=True)
    chart.save(out.with_suffix('.png'))
    np.savez_compressed(out.with_suffix('.npz'), **arrays)
    result = {'chart': str(out.with_suffix('.png')), 'chart_sha256': sha(out.with_suffix('.png')),
            'raw_trace': str(out.with_suffix('.npz')), 'raw_trace_sha256': sha(out.with_suffix('.npz')),
            'rows': [x[0] for x in rows], 'd6_rate': 1.0,
            'scope': 'unfiltered native 30 Hz and decoded 40 Hz DMX; source waveform min/max retained',
            'cache_inputs': cache_inputs}
    write_json(cache_receipt, result)
    return result


def renderers(patch):
    from lightsem.sim.lit_body_renderer import LitBodyStageRenderer
    from lightsem.sim.renderer import RenderConfig
    cams = [dict(cam_pos=(2.8, 2.15, 3.8), cam_target=(0., 1., -3.5), camera_scale=.20),
            dict(cam_pos=(0., 6., 11.), cam_target=(0., 2., -3.), camera_scale=1.02)]
    rs = []
    for camera, fill in zip(cams, (.50, .12)):
        cfg = RenderConfig(width=640, height=600, fps=30, cam_orbit=0., haze=.25,
                           intensity_gain=2.4, exposure_profile='fixed', fixed_exposure_scale=.4,
                           fixed_role_gains=(1., .3, .03, .2), beam_edge_profile='angular',
                           beam_edge_softness_deg=.6, beam_flux_reference_deg=14., gpu_device=0,
                           require_gpu=True, **camera)
        rs.append(LitBodyStageRenderer(patch, cfg, body_offset=(0., 0., -3.5), body_scale=1., neutral_fill=fill))
    return rs


def smoke(root: Path, song: str) -> None:
    body_path, _, body, _, _, _, _ = source_data(song)
    patch, time40, _, decoded, _ = controls(song, root)
    out = root / song; samples = []; files = {}
    rs = renderers(patch)
    try:
        for stamp in (0., 15., 30., 45., 59.):
            index, tick = round(stamp * 30), round(stamp * 40)
            pose = body[index].copy(); pose -= pose[:1] * np.array([1., 0., 1.])
            vals = tuple(decoded[k][tick] for k in ('directions', 'rgb', 'dimmer'))
            views = []
            metrics = []
            for j, renderer in enumerate(rs):
                diagnostic = renderer.render_diagnostics(*vals, pose)
                repeated = renderer.render_controls(*vals, pose)
                if not diagnostic['metrics']['body_visible_pixels'] or not np.array_equal(diagnostic['actual'], repeated):
                    raise ValueError(f'Smoke body/DMX check failed: {song} {stamp}')
                views.append(diagnostic['actual']); metrics.append(diagnostic['metrics'])
            picture = Image.new('RGB', (1280, 600)); picture.paste(Image.fromarray(views[0]), (0, 0)); picture.paste(Image.fromarray(views[1]), (640, 0))
            path = out / f'smoke_{int(stamp):02d}.png'; picture.save(path); files[str(path)] = sha(path)
            samples.append({'source_s': stamp, 'body_frame': index, 'dmx_tick': tick, 'frozen_pixels_exact': True, 'views': metrics})
    finally:
        for renderer in reversed(rs): renderer.close()
    chart = make_chart(song, root, out / 'RAW_TRACE')
    write_json(out / 'SMOKE.json', {'song': song, 'samples': samples, 'files': files, 'chart': chart,
                                    'body_sha256': sha(body_path), 'patch_sha256': sha(out / 'patch.json'),
                                    'timeline_sha256': sha(out / 'timeline.npz')})
    print(json.dumps({'song': song, 'samples': len(samples), 'smoke': str(out / 'SMOKE.json')}, indent=2))


def prepare_audio(root: Path, song: str, duration: float) -> Path:
    row = json.loads((root / 'MANIFEST.json').read_text())['songs'][song]
    out = root / song / 'reference.m4a'; record = root / song / 'REFERENCE.json'
    if out.exists() or record.exists():
        raise FileExistsError(f'Preserving existing audio reference: {out}')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', row['audio'],
                    '-map', '0:a:0', '-vn', '-t', str(duration), '-c:a', 'aac', '-b:a', '128k',
                    '-ac', '2', '-ar', '44100', '-movflags', '+faststart', str(out)], check=True)
    write_json(record, {'source_audio': row['audio'], 'source_audio_sha256': row['audio_sha256'],
                        'reference': str(out), 'reference_sha256': sha(out), 'duration_s': duration,
                        'decoded_pcm_sha256': pcm_sha(out), 'playback_rate': 1.0})
    return out


def full_render(root: Path, song: str, video_encoder: str = 'h264_nvenc', video_preset: str = 'p4', full: bool = False) -> None:
    body_path, _, body, _, _, _, _ = source_data(song)
    patch, time40, _, decoded, _ = controls(song, root)
    manifest_row = json.loads((root / 'MANIFEST.json').read_text())['songs'][song]
    duration = float(manifest_row['duration_s']) if full else DURATION
    frames = int(math.ceil(duration * FPS - 1e-9))
    if frames > len(body) or int(math.ceil(duration * 40 - 1e-9)) > len(time40):
        raise ValueError('Requested render exceeds saved body or lighting coverage')
    out = root / song; audio = prepare_audio(root, song, duration)
    chart = make_chart(song, root, out / 'RAW_TRACE', duration)
    rs = renderers(patch); movie = out / 'preview.mp4'; partial = out / 'preview.partial.mp4'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1280x1024', '-r', '30', '-i', 'pipe:0', '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0',
               '-frames:v', str(frames), '-c:v', video_encoder]
    if video_encoder.endswith('_nvenc'):
        command += ['-gpu', '0', '-preset', video_preset]
    else:
        command += ['-preset', video_preset, '-threads', '2']
    command += ['-b:v', '1100k', '-maxrate', '1200k', '-bufsize', '2000k']
    if video_encoder.endswith('_nvenc'):
        command += ['-rc', 'vbr', '-multipass', 'fullres', '-rc-lookahead', '32',
                    '-spatial-aq', '1', '-temporal-aq', '1']
    command += ['-pix_fmt', 'yuv420p', '-c:a', 'copy', '-t', str(duration),
                '-movflags', '+faststart', str(partial)]
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    large = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    chart_image = Image.open(chart['chart']).convert('RGB').resize((WIDTH, 304), Image.Resampling.BILINEAR)
    static_canvas = Image.new('RGB', (WIDTH, HEIGHT), (7, 10, 16))
    static_draw = ImageDraw.Draw(static_canvas)
    static_draw.text((16, 8), f'RTMPose BF16 / native C candidate | {song}', font=large, fill='white')
    static_draw.text((16, 40), 'P2 | restored brightness | actual emitted DMX | diagnostic only', font=font, fill='#c1d2e2')
    static_draw.text((16, 65), 'OWN-SONG BODY', font=font, fill='#c1d2e2')
    static_draw.text((656, 65), 'ACTUAL EMITTED LIGHTS', font=font, fill='#c1d2e2')
    try:
        with partial.open('wb') as _:
            pass
        partial.unlink()
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for frame in range(frames):
                index, tick = frame, round(frame * 40 / 30)
                pose = body[index].copy(); pose -= pose[:1] * np.array([1., 0., 1.])
                vals = tuple(decoded[k][tick] for k in ('directions', 'rgb', 'dimmer'))
                canvas = static_canvas.copy()
                canvas.paste(Image.fromarray(rs[0].render_controls(*vals, pose)), (0, 84))
                canvas.paste(Image.fromarray(rs[1].render_controls(*vals, pose)), (640, 84))
                canvas.paste(chart_image, (0, 720))
                ImageDraw.Draw(canvas).text((16, 690), f'{frame / 30:.2f}s | original audio speed | neutral fill separate from DMX', font=font, fill='#bacedd')
                encoder.stdin.write(canvas.tobytes())
            encoder.stdin.close()
            stderr = encoder.stderr.read().decode(errors='replace')
            if encoder.wait() != 0: raise RuntimeError(stderr)
        finally:
            if encoder.poll() is None: encoder.kill(); encoder.wait()
    finally:
        for renderer in reversed(rs): renderer.close()
    partial.rename(movie)
    write_json(out / 'RENDER.json', {'status': 'diagnostic_only', 'song': song, 'path': str(movie), 'sha256': sha(movie),
        'frames': frames, 'duration_s': duration, 'full_song': full,
        'audio_reference_sha256': sha(audio), 'source_audio_sha256': sha(Path(row_audio(root, song))),
        'body_sha256': sha(body_path), 'timeline_sha256': sha(out / 'timeline.npz'), 'patch_sha256': sha(out / 'patch.json'), 'chart': chart,
        'renderer': 'LitBodyStageRenderer 2 views, actual decoded mapper-01 DMX',
        'video_encoder': video_encoder, 'video_preset': video_preset, 'command': command})


def row_audio(root: Path, song: str) -> str:
    return json.loads((root / 'MANIFEST.json').read_text())['songs'][song]['audio']


def audit(root: Path, song: str) -> None:
    out = root / song; movie = out / 'preview.mp4'; reference = out / 'reference.m4a'; ref_record = json.loads((out / 'REFERENCE.json').read_text())
    render_record = json.loads((out / 'RENDER.json').read_text())
    expected_frames = int(render_record['frames'])
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams', '-of', 'json', str(movie)], text=True))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video'); audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    if int(video['nb_read_frames']) != expected_frames or video['r_frame_rate'] != '30/1': raise ValueError('Video clock/frame count failed')
    frames = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(movie)], text=True))['frames']
    times = np.array([float(x['best_effort_timestamp_time']) for x in frames])
    if len(times) != expected_frames or not np.allclose(times, np.arange(expected_frames) / 30, rtol=0, atol=1e-5): raise ValueError('PTS failed')
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-xerror', '-hwaccel', 'none', '-i', str(movie), '-f', 'null', '-'], check=True)
    delivered_pcm = pcm_sha(movie)
    if delivered_pcm != ref_record['decoded_pcm_sha256']: raise ValueError('Decoded delivery PCM differs from reference')
    manifest_row = json.loads((root / 'MANIFEST.json').read_text())['songs'][song]
    report = {'status': 'software_decode_valid', 'song': song, 'movie_sha256': sha(movie), 'decoded_frames': expected_frames,
              'pts_uniform': True, 'software_decode': True, 'decoded_pcm_sha256': delivered_pcm,
              'reference_pcm_sha256': ref_record['decoded_pcm_sha256'], 'audio_pcm_identity': True,
              'source_audio': row_audio(root, song), 'source_audio_sha256': sha(Path(row_audio(root, song))),
              'body_sha256': manifest_row['body_sha256'], 'timeline_sha256': sha(out / 'timeline.npz'),
              'patch_sha256': sha(out / 'patch.json'),
              'render_receipt_sha256': sha(out / 'RENDER.json'), 'scope': 'Diagnostic candidate; no baseline promotion.'}
    write_json(out / 'AUDIT.json', report); print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=('prepare', 'smoke', 'render', 'audit')); ap.add_argument('--song', choices=tuple(SONGS)); ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--video-encoder', choices=('av1_nvenc', 'h264_nvenc', 'libx264'), default='av1_nvenc')
    ap.add_argument('--video-preset', default='p7', help='NVENC p1-p7 or libx264 preset; project default is AV1 p7')
    ap.add_argument('--full', action='store_true', help='Render the complete saved song instead of the 60-second review window')
    args = ap.parse_args()
    if args.action == 'prepare': prepare(args.root); return
    if not args.song: ap.error('--song required')
    if args.action == 'smoke': smoke(args.root, args.song)
    elif args.action == 'render': full_render(args.root, args.song, args.video_encoder, args.video_preset, args.full)
    else: audit(args.root, args.song)


if __name__ == '__main__': main()
